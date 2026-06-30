from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.common import free_swa_out_of_window_slots
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    LRURefreshPhase,
    TreeComponent,
    next_component_uuid,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


class SWAComponent(TreeComponent):
    """🪟 Sliding Window Attention 组件 —— SWA KV 的 tombstone/窗口锁/驱逐策略。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧩 对外接口清单（按调用阶段分组）                                                     ║
    ║                                                                                  ║
    ║  🔍 Match 阶段:                                                                   ║
    ║    create_match_validator()  → 状态化闭包, 累计连续 SWA token ≥ sliding_window_size  ║
    ║    finalize_match_result()   → 统计匹配路径上的 swa_host_hit_length                ║
    ║    refresh_lru(MATCH_END/INSERT_END) → 窗口限制 LRU 刷新 (只刷窗口内祖先)             ║
    ║                                                                                  ║
    ║  📝 Insert 阶段:                                                                   ║
    ║    update_component_on_insert_overlap() → 三分支处理 tombstone 重叠 (全/部分/不消费)   ║
    ║    should_skip_leaf_creation()          → 整叶在 SWA 窗口外则跳过创建               ║
    ║    recover_after_unevict()              → Full unevict 后重建窗口内 SWA             ║
    ║    commit_insert_component_data()       → 新叶提交时分配 SWA value + 按窗口 split     ║
    ║                                                                                  ║
    ║  ✂️ Split 阶段:                                                                   ║
    ║    redistribute_on_node_split()  → 节点分裂时切片 SWA value + 复制 UUID/LR           ║
    ║                                                                                  ║
    ║  🗑️ Eviction 阶段:                                                                ║
    ║    evict_component()  → 释放 SWA pool slot, 变为 tombstone (value=None)             ║
    ║    eviction_priority() → leaf=0 (最低), internal=1 (中等, 高于 Mamba)               ║
    ║    drive_eviction()    → 从 SWA LRU 尾遍历: 叶子全删, 内部 tombstone + 级联         ║
    ║    drive_host_eviction() → Host 层 SWA 驱逐 (HiCache 场景)                          ║
    ║                                                                                  ║
    ║  🔒 Lock 阶段:                                                                     ║
    ║    acquire_component_lock(node)  → Window-lock: 向上累计至 sliding_window_size     ║
    ║    release_component_lock(node)  → 反向释放, UUID 边界处停止                        ║
    ║                                                                                  ║
    ║  💾 Cache / Free 阶段:                                                              ║
    ║    prepare_for_caching_req()    → 设置 swa_evicted_seqlen                          ║
    ║    free_out_of_window_slots()   → 调用 free_swa_out_of_window_slots() 回收旧 slot  ║
    ║                                                                                  ║
    ║  💿 HiCache Hooks:                                                                 ║
    ║    build_hicache_transfers()     → 4 阶段构造传输描述符 (BACKUP_HOST/LOAD_BACK/…    ║
    ║    commit_hicache_transfer()     → 4 阶段提交/回填 SWA host/device 索引             ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔗 宏观交互调用链（Framework → UnifiedRadixCache → SWAComponent）                   ║
    ║                                                                                  ║
    ║  ┌─ Scheduler ─┐                                                                 ║
    ║  │ 每轮 match:                                                                    ║
    ║  │   UnifiedRadixCache._match_prefix_helper()                                    ║
    ║  │     ├─ comp.create_match_validator()     ← 创建累计验证闭包 (unified_radix_cache.py:977)║
    ║  │     ├─ (树遍历 + validator(node) 逐节点判)                                       ║
    ║  │     └─ comp.finalize_match_result()      ← 后处理匹配结果 (:1079)               ║
    ║  │                                                                                ║
    ║  │ 每轮 insert (cache_finished_req / cache_unfinished_req):                        ║
    ║  │   comp.prepare_for_caching_req()         ← 赋值 swa_evicted_seqlen (:813)      ║
    ║  │   comp.free_out_of_window_slots()        ← 释放 decoder 旧 SWA slot (:891)     ║
    ║  │   UnifiedRadixCache._insert_helper()                                           ║
    ║  │     ├─ comp.should_skip_leaf_creation()   ← 否决窗口外叶子 (:1239)              ║
    ║  │     ├─ comp.recover_after_unevict()       ← Full unevict 后重建 SWA (:1202)    ║
    ║  │     ├─ comp.update_component_on_insert_overlap() ← 复活重叠 tombstone (:1213)  ║
    ║  │     └─ comp.commit_insert_component_data()  ← 新叶 SWA 分配+窗口 split (:1260) ║
    ║  │                                                                                ║
    ║  │ 驱逐 (alloc 空间不足时):                                                         ║
    ║  │   UnifiedRadixCache.evict()                                                    ║
    ║  │     └─ comp.drive_eviction(params, tracker) ← 遍历 SWA LRU 驱逐 (:715)        ║
    ║  │                                                                                ║
    ║  │ 锁 (每 request 生命周期):                                                        ║
    ║  │   UnifiedRadixCache.inc_lock_ref(node)     ← 保护 SWA 窗口不被驱逐              ║
    ║  │     └─ comp.acquire_component_lock()       ← Window-lock (:738)                ║
    ║  │   UnifiedRadixCache.dec_lock_ref(node)     ← 释放保护                           ║
    ║  │     └─ comp.release_component_lock()       ← 反向递减 (:752)                    ║
    ║  └──────────────────────────────────────────┘                                     ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧬 核心设计要点                                                                     ║
    ║                                                                                  ║
    ║  1️⃣  Value 的语义:                                                                 ║
    ║     value 存储 SWA pool 中的索引 (而非 Full pool 索引)。每个 token 有两份 KV:         ║
    ║       BASE_COMPONENT_TYPE (Full).value  → Full pool 索引                          ║
    ║       SWA component.value               → SWA pool 索引 (经 translate 转换)       ║
    ║     SWA value 在 insert 时通过 _translate_full_to_swa() 从 Full value 转换得到。    ║
    ║                                                                                  ║
    ║  2️⃣  Tombstone 机制:                                                               ║
    ║     内部节点 SWA 被驱逐后, value 设为 None 但节点不删除 (⇒ tombstone)。                 ║
    ║     原因: SWA 窗口是向前滑动且大小固定的, 旧 token 的 SWA KV 必须释放;                  ║
    ║     但树的拓扑结构（用于 prefix match）仍然需要保留, 因为 Full KV 可能还有效。          ║
    ║     tombstone 节点在 insert 时若落在窗口内可以被"复活"(恢复 SWA value)。                ║
    ║                                                                                  ║
    ║  3️⃣  Window-Lock (滑动窗口锁):                                                      ║
    ║     不同于 Full 的 path-lock (沿树向上锁全部祖先), SWA 是累计锁:                        ║
    ║       从 node 向上走, 累计每个非 tombstone 祖先的 SWA value 长度,                       ║
    ║       达到 sliding_window_size 时停止, 在该节点设置 component_uuid 做边界标记。          ║
    ║     release 时反向递减, 遇到同 UUID 的节点就停止 —— 只释放窗口正好覆盖的部分。          ║
    ║                                                                                  ║
    ║  4️⃣  SWA LRU 刷新策略 (窗口限制):                                                    ║
    ║     WALKDOWN 阶段 NO-OP (大部分遍历节点在滑动窗口外, 不应变成 MRU)。                      ║
    ║     MATCH_END / INSERT_END 阶段: 只刷新窗口内祖先为 MRU                               ║
    ║       (sliding_window_size + page_size 范围内, 且 node_has_component_data=True)。   ║
    ║                                                                                  ║
    ║  5️⃣  驱逐优先级 (级联):                                                               ║
    ║     leaf=0: 叶子驱逐 → 级联到所有组件 (整叶删除)                                       ║
    ║     internal=1 (Medium): 高于 Mamba(0) 低于 Full(2)。                                 ║
    ║       原因: SWA 内部节点上的数据是滑动窗口"路径数据", 路径中断则窗口不连贯;                ║
    ║       Mamba 数据只在 match boundary 有意义, 内部节点上的 Mamba 无关紧要。               ║
    ║                                                                                  ║
    ║  6️⃣  Tombstone 重叠复活 (insert overlap):                                           ║
    ║     当新的 insert 路径穿过一个 tombstone 节点时, 若该节点落在 SWA 窗口内,                  ║
    ║     可以"复活"——从新分配的 Full value 重新翻译出 SWA value 并写回。                      ║
    ║     三分支: 全在窗口内(全恢复) / 骑墙(部分恢复+split) / 全在窗口外(不消费)。              ║
    ║                                                                                  ║
    ║  7️⃣  _maybe_split_leaf_for_swa_lock:                                                ║
    ║     新 SWA 叶子可能很长 (如 chunked prefill 几百 token), 直接 lock 会锁住整个叶子的      ║
    ║     SWA pool, 浪费空间。在 commit 时将叶子尾部裁剪到 1 个窗口大小并 split,                ║
    ║     只锁住恰好一个 sliding window 的 SWA pool。                                       ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        """🪟 构造函数 —— 校验 SWATokenToKVPoolAllocator, 记录 sliding_window_size。

        🔗 UnifiedRadixCache.__init__ 通过 COMPONENT_REGISTRY 创建本实例 (:417)。
            COMPONENT_REGISTRY = { ..., ComponentType.SWA: SWAComponent } (:328-332)。

        📥 cache: UnifiedRadixCache 实例 (包含 token_to_kv_pool_allocator)。
        📥 params: CacheInitParams, 必须含 sliding_window_size。
        ⚙️ 断言 cache 的 allocator 是 SWATokenToKVPoolAllocator —— SWA 需要双池 (Full + SWA)。
        ⚠️  _swa_kv_pool_host 默认为 None, HiCache 启用后由 HostPoolGroup 设置。"""
        from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

        assert isinstance(
            cache.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
        ), f"SWAComponent requires SWATokenToKVPoolAllocator, got {type(cache.token_to_kv_pool_allocator)}"
        super().__init__(cache, params)
        self.sliding_window_size = params.sliding_window_size
        # HiCache state: set to host SWA pool when HiCache enabled
        self._swa_kv_pool_host = None

    component_type = ComponentType.SWA

    def _translate_full_to_swa(self, full_indices: torch.Tensor) -> torch.Tensor:
        """🔄 Full pool 索引 → SWA pool 索引。

        🔗 调用场景: SWA 模型有双池 (Full pool + SWA pool), 每个 token 有两份 KV。
            insert 时 Full value 已由 _add_new_node / _unevict_node_on_insert 写入,
            但 SWA value 需要从 Full value 翻译得到。被以下方法调用:
            ├─ update_component_on_insert_overlap()  — tombstone 复活时转换新 Full
            ├─ recover_after_unevict()                — Full unevict 后重建 SWA
            └─ commit_insert_component_data()         — 新叶 SWA value 生成

        📥 full_indices: Full pool 中的 KV slot 索引 (BASE_COMPONENT_TYPE.value 的内容)。
        📤 对应的 SWA pool 索引。
        ⚙️ 委托给 SWATokenToKVPoolAllocator.translate_loc_from_full_to_swa()。
            该内部维护 full↔swa 的 index 映射表 (allocator 分配时建立)。"""
        return self.cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            full_indices
        )

    def refresh_lru(
        self,
        phase: LRURefreshPhase,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
    ) -> None:
        """🕐 SWA 窗口限制的 LRU 刷新策略 —— WALKDOWN 跳过, MATCH/INSERT_END 仅刷窗口内。

        🔗
          调用者 (unified_radix_cache.py):
            ├─ _touch_node()         → phase=WALKDOWN       (:1130)
            ├─ _match_post_processor  → phase=MATCH_END     (:1050)
            └─ _insert_helper()       → phase=INSERT_END    (:1271)

        ⚙️ 行为:
            WALKDOWN (树遍历触碰): NO-OP —— SWA 窗口前移时, 大部分遍历到的祖先节点
              已经滚出滑动窗口, 若将其设为 MRU 会错误地保护不再需要的旧 SWA 数据。
              窗口边界内的刷新推迟到 MATCH_END / INSERT_END 做。

            MATCH_END / INSERT_END: 调用 reset_node_and_window_ancestors_mru(),
              从 node 向上走, 只刷新 sliding_window_size+page_size 范围内的、有 SWA value 的祖先。
              这个范围上限=窗口大小+1个 page 的缓冲 (因为驱逐边界是 page 对齐的)。

        📥 phase: 触发阶段枚举 (WALKDOWN | MATCH_END | INSERT_END)。
        📥 node: 当前处理的树节点。
        📥 root_node: 根节点 (向上遍历的终止条件)。
        ⚠️  基类默认 WALKDOWN 做 reset_node_mru, 这里刻意 override 为空操作。"""
        match phase:
            case LRURefreshPhase.WALKDOWN:
                # Walk-down would refresh every visited ancestor to MRU,
                # but most are outside the active sliding window and must
                # stay evictable. Window-bounded refresh runs at
                # MATCH_END / INSERT_END instead.
                return
            case LRURefreshPhase.MATCH_END | LRURefreshPhase.INSERT_END:
                self.cache.lru_lists[
                    self.component_type
                ].reset_node_and_window_ancestors_mru(
                    node,
                    root_node,
                    self.sliding_window_size + self.cache.page_size,
                    self.node_has_component_data,
                )
            case _:
                raise ValueError(f"Unknown LRURefreshPhase: {phase}")

    def _restore_device_value(self, node: UnifiedTreeNode, value: torch.Tensor) -> None:
        """🔄 Tombstone 复活 —— 写回 SWA value, 从 Host LRU 迁移到 Device LRU, 更新 evictable_size。

        🔗 调用场景: SWA tombstone (value=None) 恢复为有效状态的统一入口, 被以下三种场景调用:
            ① update_component_on_insert_overlap() — insert 路径穿过窗口内 tombstone, 复活 SWA
            ② recover_after_unevict()              — Full unevict 后从新 Full value 重建 SWA
            ③ commit_hicache_transfer(LOAD_BACK)   — HiCache H→D 加载完成后回填 SWA device value

        ⚙️ 三步:
            ① 写回 cd.value ← value (结束 tombstone 状态)。
            ② 若节点在 Host LRU 则移除 (Host 侧已无效, 因为 device 恢复了)。
            ③ 插入 Device LRU 的 MRU 位置 + 增加 evictable_size (使节点重新可被驱逐)。"""
        ct = self.component_type
        # ① 复活: value 从 None 恢复为有效 SWA pool 索引
        node.component_data[ct].value = value
        # ② 若有 host 备份且 host LRU 仍在, 从中移除 (device 已恢复, host 侧不需 LRU 跟踪)
        host_lru = self.cache.host_lru_lists[ct]
        if host_lru.in_list(node):
            host_lru.remove_node(node)
        # ③ 加入 Device LRU (MRU 端) + 增加可驱逐 token 计数
        self.cache.lru_lists[ct].insert_mru(node)
        self.cache.component_evictable_size_[ct] += len(value)

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        """🔍 创建状态化闭包 —— 在树遍历中累计连续非 tombstone token, >= sliding_window_size 才 True。

        🔗 UnifiedRadixCache._match_prefix_helper() 每轮 match 调用一次 (:977)。
            match_device_only=True: HiCache 场景下的"纯设备匹配"判断, host tombstone 不能作为有效边界。

        ⚙️ 闭包行为:
            state["len"] 记录累计连续有效 SWA token 数。
            从根向叶遍历时逐节点调用:
              - 若 cd.value 是 None (tombstone) 且是纯设备匹配 → state["len"] 归零 (窗口断裂)
              - 若有有效 value → 累加 len(node.key)
              - 当累计 >= sliding_window_size 时返回 True (窗口已满, 可以在此处匹配)

            为什么用"累计"而非"单节点"?
              不同节点可能被 SWA LRU 驱逐了部分, 但连续有效的祖先加起来达到窗口大小, 路径仍然可用。

        📥 match_device_only: True 时只有 device value 算有效 (host_value 不算), 用于 HiCache 的 separate_device_match 分支。
        📤 返回闭包 validator(node) → bool。
        ⚠️  初始 state["len"] = inf 使得第一个 tombstone 就归零。
            tombstone 断点之前的节点不能作为匹配边界, 保证匹配路径的 SWA 连贯性。"""
        sliding_window_size = self.sliding_window_size
        ct = self.component_type
        # inf 初始化使遇到第一个 tombstone 就归零, 保证"从根开始累计"语义
        state = {"len": float("inf")}

        def validator(node: UnifiedTreeNode) -> bool:
            cd = node.component_data[ct]
            # HiCache: a host-only tombstone is a valid match boundary too
            # — load_back will restore SWA from host before use.
            # 纯设备匹配时 host_value 不算有效: match_device_only=True 且 cd.value=None
            #   → state["len"] 归零 → 窗口断裂 → 不会在此处匹配
            if cd.value is None and (match_device_only or cd.host_value is None):
                state["len"] = 0
                return False
            # 累加非 tombstone 节点的 token 数
            state["len"] += len(node.key)
            return state["len"] >= sliding_window_size

        return validator

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        """🔍 匹配后处理 —— 沿 best_match_node 向上统计 swa_host_hit_length。

        🔗 _match_post_processor() 在拼接 device_indices 后遍历所有 component 调用 (:1079)。

        ⚙️ 行为: 从 best_match_node 向上走最多 sliding_window_size 个 token,
            对每个祖先:
              - 有 device value: 累加 len (device 已有数据)
              - 有 host_value 但无 device value: 累加到 swa_host_hit 计数器
                (HiCache 场景, 这些 host 数据需要 load_back 回 device)
              - 都无 (tombstone both layers): 直接 break (窗口断裂)

            然后将 swa_host_hit 写入 result.swa_host_hit_length,
            供调度器据此预估 load_back 需要的 device 空间。

        📥 result: 当前匹配结果 (namedtuple MatchResult)。
        📥 params: MatchPrefixParams (含 request 上下文)。
        📥 value_chunks: 匹配路径上各节点 device value 的列表 (SWA 暂不直接使用)。
        📥 best_value_len: 已匹配的 token 总数。
        📤 可能更新 swa_host_hit_length 字段的 MatchResult。"""
        ct = self.component_type
        n_swa = 0      # 累计连续有效 SWA token
        swa_host_hit = 0  # 其中需要从 host load_back 的
        node = result.best_match_node
        root = self.cache.root_node
        while node is not root and n_swa < self.sliding_window_size:
            cd = node.component_data[ct]
            if cd.value is not None:
                # Device 已有 SWA KV, 直接计入连续覆盖
                n_swa += len(cd.value)
            elif cd.host_value is not None:
                # TODO(hzh): load_back may currently restore a full host-tombstone
                # segment whose length exceeds sliding_window_size. Once
                # load_back is constrained to fetch only one sliding window
                # worth of pages, cap swa_host_hit at sliding_window_size
                # here so the scheduler budget matches the actual device-pool
                # consumption.
                swa_host_hit += len(cd.host_value)
                n_swa += len(cd.host_value)
            else:
                # 双重 tombstone: 窗口断裂, 停止向上搜
                break
            node = node.parent
        if swa_host_hit > 0:
            # 更新 result: 取最大值 (可能已有其他源计入 swa_host_hit_length)
            return result._replace(
                swa_host_hit_length=max(result.swa_host_hit_length, swa_host_hit)
            )
        return result

    def update_component_on_insert_overlap(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
    ) -> int:
        """📝 处理 insert 路径与 tombstone 节点的重叠 —— 窗口内复活 SWA, 三分支: 全恢复/部分恢复/不消费。

        🔗 调用场景: cache_finished_req / cache_unfinished_req → insert → _insert_helper 遍历树时,
            若路径上的节点 SWA value=None (tombstone, 曾被 SWA LRU 驱逐), 且该节点现在落在
            滑动窗口内, 则可以"复活"——从新 Full value 重新翻译出 SWA value 并写回。
            这是 SWA 独有的机制: Full 数据在树中保留, SWA 数据可驱逐/复活。

        ⚙️ 三分支逻辑 (基于 swa_evicted_seqlen —— 当前窗口左边界):
            Branch 1: 整节点在窗口内 (swa_evicted_seqlen <= total_prefix_len)
              → 全节点复活: free 旧 Full value, 写新 value_slice, 翻译 SWA, _restore_device_value

            Branch 2: 节点骑跨窗口边界 (swa_evicted_seqlen < total_prefix_len + prefix_len)
              → 部分复活: free 旧尾部, split 节点, 对新尾部写克隆 + 翻译 SWA

            Branch 3: 整节点在窗口外 (swa_evicted_seqlen >= total_prefix_len + prefix_len)
              → 不消费: return prefix_len (节点保持 tombstone)

        📥 node: 当前重叠节点 (tombstone, 即 node.component_data[SWA].value is None)。
        📥 prefix_len: 当前节点的 key 长度。
        📥 total_prefix_len: 从根到当前节点前的累计 token 数。
        📥 value_slice: 新分配的 Full value 中对应本节点的 slices。
        📥 params: InsertParams, 含 swa_evicted_seqlen (SWA 驱逐边界, page 对齐)。
        📤 返回 consumed_from: 本组件"消耗"了多少 token —— 0=全消费, prefix_len=不消费, start_idx=部分消费。
        ⚠️  前置条件: cd.lock_ref == 0 (tombstone 不应被锁)。"""
        # 如果上一次插入已经覆盖了本节点, 跳过
        if params.prev_prefix_len >= total_prefix_len + prefix_len:
            return prefix_len

        is_tombstone = node.component_data[self.component_type].value is None
        if not is_tombstone:
            # 已有 SWA value, 正常重叠无需处理
            return prefix_len

        swa_evicted_seqlen = params.swa_evicted_seqlen
        assert (
            node.component_data[self.component_type].lock_ref == 0
        ), f"tombstone {self.component_type} lock_ref should be 0, node {node.id}"
        assert (
            swa_evicted_seqlen % self.cache.page_size == 0
        ), f"{self.component_type}: swa_evicted_seqlen must be page-aligned, {swa_evicted_seqlen=}"

        if swa_evicted_seqlen <= total_prefix_len:
            # ── Branch 1: 整节点在窗口内 → 全恢复 ──
            # 释放旧的 Full value, 用新的 value_slice 替换
            self.cache.token_to_kv_pool_allocator.free(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            node.component_data[BASE_COMPONENT_TYPE].value = value_slice.clone()
            # 从新 Full value 翻译出 SWA value
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            self._restore_device_value(node, swa_value)
            return 0  # 消费了整个节点的旧数据

        elif swa_evicted_seqlen < total_prefix_len + prefix_len:
            # ── Branch 2: 节点骑跨窗口边界 → 部分恢复 ──
            # 窗口左边界落在节点中间, 只恢复边界右边的部分
            start_idx = swa_evicted_seqlen - total_prefix_len
            # 释放旧节点的 [start_idx:] 部分
            self.cache.token_to_kv_pool_allocator.free(
                node.component_data[BASE_COMPONENT_TYPE].value[start_idx:]
            )
            # 分裂: 左边做 tombstone parent, 右边做有 SWA 的 child (node 变成 child)
            self.cache._split_node(node.key, node, start_idx)
            node.component_data[BASE_COMPONENT_TYPE].value = value_slice[
                start_idx:
            ].clone()
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            self._restore_device_value(node, swa_value)
            return start_idx  # 只消费了 start_idx 之前的旧数据

        else:
            # ── Branch 3: 整节点在窗口外 → 不消费 ──
            # 旧 SWA 数据完全被窗口前移淘汰, 保持 tombstone
            return prefix_len

    def should_skip_leaf_creation(
        self, total_prefix_len: int, key_len: int, params: InsertParams
    ) -> bool:
        """📝 否决新叶创建 —— 当整个新叶的 SWA 数据都在窗口外时 (全 tombstone)。

        🔗 调用场景: _insert_helper 遍历完已有节点后, 若有剩余 key 后缀需要创建新叶子,
            在 _add_new_node 前检查。任意组件返回 True 则放弃叶子创建 (free value 后返回)。
            SWA 场景: 新叶若完全在 swa_evicted_seqlen 之外, 对 SWA 是纯 tombstone,
            创建了也没有 SWA 数据 → 跳过避免无意义的叶子。

        ⚙️ swa_evicted_seqlen 是窗口左边界, >= total_prefix_len + key_len 意味着
            整叶都在左边界之外 → 对 SWA 完全是 tombstone → 不需要创建 SWA 叶子。"""
        return params.swa_evicted_seqlen >= total_prefix_len + key_len

    def recover_after_unevict(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        params: InsertParams,
    ) -> None:
        """📝 Full unevict 后, 从新的 Full value 重建窗口内的 SWA value。

        🔗 调用场景: _insert_helper 遍历树时遇到 evicted 节点 (Full value=None),
            _unevict_node_on_insert 用新 KV 恢复了 Full value 后, 遍历辅组件调用本方法。
            SWA 可能仍是 tombstone (曾随 Full 一起被驱逐), 需要从新 Full value 重建。
            与 update_component_on_insert_overlap 类似但更简单 —— 不需要 free 旧 Full value,
            因为 unevict 的 Full value 已经是新的。

        ⚙️ 三分支 (同 update_component_on_insert_overlap):
            整节点在窗口内 → 翻译 SWA; 骑跨边界 → split 后翻译右半; 整节点在窗口外 → 保持 tombstone。"""

        # _unevict_node_on_insert already wrote the request's fresh KV slice
        # into the base value. We just need to rebuild SWA from that slice for
        # the in-window portion. There is no old SWA slot to free here.
        ct = self.component_type
        if node.component_data[ct].value is not None:
            # 已有 SWA value, 无需重建
            return
        assert (
            node.component_data[ct].lock_ref == 0
        ), f"tombstone {ct} lock_ref should be 0 on unevict, node {node.id}"
        swa_evicted_seqlen = params.swa_evicted_seqlen
        assert (
            swa_evicted_seqlen % self.cache.page_size == 0
        ), f"{ct}: swa_evicted_seqlen must be page-aligned, {swa_evicted_seqlen=}"

        full_value = node.component_data[BASE_COMPONENT_TYPE].value
        if swa_evicted_seqlen <= total_prefix_len:
            # 整节点在窗口内: 从完整 Full value 翻译 SWA
            swa_value = self._translate_full_to_swa(full_value)
        elif swa_evicted_seqlen < total_prefix_len + prefix_len:
            # 骑跨窗口边界: 先 split 再翻译右半
            start_idx = swa_evicted_seqlen - total_prefix_len
            self.cache._split_node(node.key, node, start_idx)
            full_value = node.component_data[BASE_COMPONENT_TYPE].value
            swa_value = self._translate_full_to_swa(full_value)
        else:
            # 整节点在窗口外: 保持 tombstone
            return
        self._restore_device_value(node, swa_value)

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        """📝 Insert 完成后在目标节点上提交 SWA 数据 —— 分配 SWA value 并按窗口边界 split。

        🔗 调用场景: _insert_helper 遍历完所有重叠节点 + 创建新叶子后, 末尾调用各组件 commit。
            Full 的 value 已由 _add_new_node 写入, 但 SWA value 需要在此处从 Full value 翻译并分配。
            每次 insert 仅调用一次。

        ⚙️ 对非新叶节点无操作 (SWA 已在 overlap/unevict 步骤处理)。对新叶节点:
            ① 计算 split_pos = swa_evicted_seqlen - node_start (窗口边界在节点内的位置)
            ② split_pos <= 0: 整叶在窗口内 → 翻译 SWA, 插入 LRU
            ③ 0 < split_pos < len(key): 骑跨边界 → split, child 有 SWA, parent tombstone
            ④ split_pos >= len(key): 整叶在窗口外 → 保持 tombstone
            ⑤ _maybe_split_leaf_for_swa_lock 裁剪尾部到窗口大小"""
        if not is_new_leaf:
            # 非新叶节点: SWA 数据已在之前的 overlap/unevict 步骤处理完
            return

        node_start = result.prefix_len        # 本节点开始的全局 token 位置
        split_pos = params.swa_evicted_seqlen - node_start  # 窗口左边界在节点内的偏移

        if split_pos <= 0:
            # ── 整叶在窗口内 → 分配 SWA value ──
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            node.component_data[self.component_type].value = swa_value
            # 加入 LRU + 增加可驱逐计数
            self.cache.lru_lists[self.component_type].insert_mru(node)
            self.cache.component_evictable_size_[self.component_type] += len(swa_value)
        elif split_pos < len(node.key):
            # ── 骑跨边界: split 为 parent(tombstone) + child(SWA) ──
            # Node straddles the SWA eviction boundary
            # Split into parent (tombstone, no SWA) and child (with SWA)
            # After _split_node, `node` becomes the child
            self.cache._split_node(node.key, node, split_pos)
            swa_value = self._translate_full_to_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            node.component_data[self.component_type].value = swa_value
            self.cache.lru_lists[self.component_type].insert_mru(node)
            self.cache.component_evictable_size_[self.component_type] += len(swa_value)
        else:
            # ── 整叶在窗口外 → 保持 tombstone (无 SWA value) ──
            # Entire leaf is outside the SWA window — left as a tombstone.
            return

        # 裁剪叶子尾部: 长 prefill 叶子可能远大于窗口, lock 时只需锁一个窗口
        self._maybe_split_leaf_for_swa_lock(node)

    def _maybe_split_leaf_for_swa_lock(self, leaf: UnifiedTreeNode) -> None:
        """✂️ 裁剪新 SWA 叶子的尾部 —— lock 时只锁一个窗口的 SWA pool, 不锁整个长叶子。

        🔗 由 commit_insert_component_data() 在末尾调用 (:294)。

        ⚙️ 场景: chunked prefill 可能分配几百 token 的叶子, 若 lock 整个叶子会锁住远超
            窗口大小的 SWA pool, 浪费空间。所以将尾部裁剪到恰好 1 个 sliding window 大小:
              tail_size = ceil(sliding_window_size / page_size) * page_size
              若 leaf_len > tail_size → split 在 leaf_len - tail_size 处

        ⚠️  仅在 leaf 是 root、是 tombstone、已被 lock、或长度 < tail_size 时跳过。
            page_size > 1 时要求分裂点和对齐都满足才 split (避免切断 page)。"""
        # Cap a fresh SWA leaf at one page-aligned window so locking it pins
        # only one window of SWA pool, not the whole (long chunked-prefill) leaf.
        ct = self.component_type
        cd = leaf.component_data[ct]
        # 不需要裁剪的情况:
        #   - 根节点 / tombstone (无 value) / 已被锁 (lock_ref>0 说明已在使用)
        if leaf is self.cache.root_node or cd.value is None or cd.lock_ref > 0:
            return

        page_size = self.cache.page_size
        # Smallest page-aligned size that still covers the sliding window.
        tail_size = (self.sliding_window_size + page_size - 1) // page_size * page_size
        leaf_len = len(leaf.key)
        if leaf_len <= tail_size:
            return
        split_at = leaf_len - tail_size
        # page_size > 1 时确保分裂点和对齐都正确, 避免非 page 对齐的切分
        if page_size > 1 and (split_at % page_size != 0 or leaf_len % page_size != 0):
            return

        self.cache._split_node(leaf.key, leaf, split_at)

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        """✂️ 节点分裂时重分配 SWA 数据 —— 切片 value/host_value + 复制 lock_ref + 搬运 UUID。

        🔗 调用场景: _split_node 在 match/insert 遇部分匹配时分裂节点,
            创建 new_parent 后、重新插入 LRU 前调用每个组件。
            SWA 的 value/host_value/lock_ref/UUID 都需要按 split_len 切分到两个节点。

        ⚙️ 动作:
            ① lock_ref: new_parent 继承 child 的 lock_ref (分裂前的锁保护整段, 分裂后两段都需保护)。
            ② SWA value: 按 key 长度切片 → new_parent[:split_len], child[split_len:]
            ③ SWA host_value: 同样切片语义。
            ④ Host LRU: 若切出 tombstone (device value=None) 但有 host_value → 插入 Host LRU。
            ⑤ UUID: new_parent 继承 child 的 swa_uuid (window-lock 边界标记向上移动)。"""
        # ① lock_ref 继承
        new_parent.component_data[self.component_type].lock_ref = child.component_data[
            self.component_type
        ].lock_ref

        # ② 切片 device SWA value
        child_swa_value = child.component_data[self.component_type].value
        if child_swa_value is not None:
            split_len = len(new_parent.key)  # new_parent 的 token 数即为分界点
            new_parent.component_data[self.component_type].value = child_swa_value[
                :split_len
            ].clone()
            child.component_data[self.component_type].value = child_swa_value[
                split_len:
            ].clone()
        else:
            new_parent.component_data[self.component_type].value = None

        # ③ 切片 host SWA value + Host LRU 管理
        child_swa_host_value = child.component_data[self.component_type].host_value
        if child_swa_host_value is not None:
            split_len = len(new_parent.key)
            new_parent.component_data[self.component_type].host_value = (
                child_swa_host_value[:split_len].clone()
            )
            child.component_data[self.component_type].host_value = child_swa_host_value[
                split_len:
            ].clone()
            host_lru = self.cache.host_lru_lists[self.component_type]
            # 如果切出来的新 parent 是 device tombstone (但有 host_value), 加入 Host LRU
            if new_parent.component_data[self.component_type].value is None:
                host_lru.insert_mru(new_parent)
            # 如果 child 也是 device tombstone 且不在 Host LRU, 加入 Host LRU
            if child.component_data[
                self.component_type
            ].value is None and not host_lru.in_list(child):
                host_lru.insert_mru(child)

        # ④ UUID 向上传递: new_parent 继承 child 的 swa_uuid (windo-lock 边界标记)
        # parent inherits the swa_uuid from child for swa lock ref
        new_parent.component_data[self.component_type].metadata["uuid"] = (
            child.component_data[self.component_type].metadata.get("uuid")
        )
        child.component_data[self.component_type].metadata.pop("uuid", None)

    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        """🗑️ 释放节点的 SWA KV 资源 —— 变 tombstone (value→None), 返回释放的 token 数。

        🔗 调用场景: 驱逐流水线中的原子步骤, 被以下路径调用:
            ① drive_eviction → _evict_component_and_detach_lru
               (SWA LRU 主动驱逐内部节点 → tombstone)
            ② _cascade_evict → _evict_component_and_detach_lru
               (Full 驱逐时级联清理 SWA, 因 Full(2) > SWA(1))
            ③ _evict_device_leaf / _evict_host_leaf → ALL 层
               (整叶删除时清理 SWA)
            ④ _evict_to_host → DEVICE 层
               (HiCache D→H 降级时 tombstone SWA device, 保留 host)

        ⚙️ DEVICE 层: free_swa(full_indices) + value→None (tombstone)。
            ⚠️ 用 full_indices 而非 swa_value: 无 SWA 映射的 slot 指向同一 sentinel,
            直接 free swa_value 会 double-free。
            若有残留 host_value → 插入 Host LRU (HiCache: Host 侧仍可用)。
        HOST 层: free host pool + host_value→None, 从 Host LRU 移除。
        ALL: 同时执行 DEVICE + HOST (整叶删除场景)。"""
        ct = self.component_type
        cd = node.component_data[ct]
        freed = 0
        host_freed = 0

        # ── Device layer ──
        if EvictLayer.DEVICE in target and cd.value is not None:
            # Pass full indices to free_swa so slots with no SWA pair are
            # skipped. Freeing swa_value directly would double free those
            # entries since they all map to the same sentinel slot.
            self.cache.token_to_kv_pool_allocator.free_swa(
                node.component_data[BASE_COMPONENT_TYPE].value
            )
            freed = len(cd.value)
            self.cache.component_evictable_size_[ct] -= freed
            cd.value = None  # → tombstone

        # ── Host layer ──
        host_lru = self.cache.host_lru_lists[ct]
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            if self._swa_kv_pool_host is not None:
                self._swa_kv_pool_host.free(cd.host_value)
            cd.host_value = None
            if host_lru.in_list(node):
                host_lru.remove_node(node)

        # ── 仅 DEVICE 层驱逐后: 若还有 host_value, 移入 Host LRU 管理 ──
        # After device tombstone: if host_value remains, move into host LRU
        if (
            target is EvictLayer.DEVICE
            and cd.value is None
            and cd.host_value is not None
        ):
            if not host_lru.in_list(node):
                host_lru.insert_mru(node)

        return freed, host_freed

    def eviction_priority(self, is_leaf: bool) -> int:
        """🎯 SWA 驱逐优先级: leaf=0 (最低), internal=1 (中等)。

        🔗 由 _evict_component_and_detach_lru / _cascade_evict 在判断级联范围时调用。

        ⚙️ leaf=0: 与 Full/Mamba 齐平 → 驱逐任一叶子组件时级联到所有 (整叶删除)。
           internal=1: 高于 Mamba(0) 低于 Full(2)。
             为什么 SWA > Mamba? SWA 内部节点数据是"窗口路径数据": 路径中断则后面
             节点的 SWA 也无法到达窗口边界。Mamba 内部数据仅 match boundary 有意义。
             因此驱逐 SWA 时级联到 Mamba, 但反之不级联。"""
        return 0 if is_leaf else 1

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        """🗑️ 从 SWA LRU 尾遍历驱逐 —— 直到满足 swa_num_tokens 请求。

        🔗 调用场景: SWA pool 空闲空间不足时, alloc_token_slots → evict_from_tree_cache
            → UnifiedRadixCache.evict() → 遍历组件调 drive_eviction。
            SWA 用 LRU 链表 (与 Full 的叶子集合堆不同), 因为 SWA 可以 tombstone 内部节点。

        ⚙️ 遍历 SWA LRU 从 LRU (最久未用) 端开始:
            对每个节点:
              - 若是 evictable_device_leaves (可驱逐叶子):
                  → _evict_device_leaf(x, tracker)  原子级驱逐所有组件 (整叶删除)
              - 若是内部节点:
                  → _evict_component_and_detach_lru(x, SWA, DEVICE)  tombstone SWA
                  → _cascade_evict(x, SWA, tracker)                  级联驱逐 Mamba

            为什么叶子走整叶删除? 叶子上的 Full 数据也需要释放 KV slot; tombstone
            只适用于内部节点 (树结构保留)。

        📥 params: EvictParams, 含 swa_num_tokens (SWA 需要释放的 token 数)。
        📥 tracker: 跨组件共享的驱逐计数 (ComponentType → 已驱逐 token 数)。
        ⚠️  每个 cycle 前先检查 x_next 是否仍在 LRU (可能被级联驱逐先移除了)。"""
        request = params.swa_num_tokens    # SWA 池需要释放的目标 token 数
        ct = self.component_type
        lru = self.cache.lru_lists[ct]
        x = lru.get_lru_no_lock()          # LRU 尾 (最久未被刷新的节点)
        while tracker[ct] < request and x is not None and lru.in_list(x):
            assert x.component_data[ct].value is not None
            if x in self.cache.evictable_device_leaves:
                # 🌿 可驱逐叶子: 原子驱逐所有组件 (整叶删除, 包括 Full KV)
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_device_leaf(x, tracker)
                # 叶驱逐可能影响 x_next 的 LRU 状态, 安全回退
                if not lru.in_list(x_next):
                    x_next = lru.get_lru_no_lock()
                x = x_next
            else:
                # 🪦 内部节点: tombstone SWA + 级联驱逐 Mamba
                #   SWA priority(1) > Mamba(0) → 驱逐 SWA 时级联清理 Mamba
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_component_and_detach_lru(
                    x, self, target=EvictLayer.DEVICE, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = x_next

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        """🔒 Window-Lock —— 向上累计 SWA value 长度直到 sliding_window_size, 边界设 UUID。

        🔗 UnifiedRadixCache.inc_lock_ref(node) → 遍历 component 调用 (:738)。
            每个 request 在开始处理前获取 lock, 保护其 SWA 窗口不被驱逐。

        ⚙️ 不同于 Full 的 path-lock (沿树向上锁全部祖先), SWA 是"窗口累计锁":
            ① 从 node 向上走, 累计每个非 tombstone 祖先的 SWA value 长度。
            ② 遇到 tombstone (cd.value=None) 跳过并标记到 skip_lock_node_ids。
            ③ 当累计 swa_lock_size >= sliding_window_size 时停止, 在该节点设置 UUID 边界。
            ④ 第一次 lock 时: lock_ref 0→1, 从 evictable_size 移到 protected_size (或从 LRU 移除)。

            为什么累计而不是锁全部? 滑动窗口大小固定, 只需保护恰好一个窗口的 SWA KV,
            多余的祖先即使被驱逐也不影响当前 request 的窗口连续性。

        📥 node: 请求插入/匹配的节点 (锁从此节点开始向上)。
        📥 result: IncLockRefResult, 含 skip_lock_node_ids / swa_uuid_for_lock 等字段。
        📥 lock_host: True 时锁 Host 侧 (HiCache 场景)。
        📤 更新后的 result (含 swa_uuid_for_lock 或 swa_uuid_for_host_lock)。"""
        ct = self.component_type
        root = self.cache.root_node
        sliding_window_size = self.sliding_window_size
        swa_lock_size = 0       # 累计锁定的 SWA token 数
        swa_uuid = None         # 锁边界 UUID
        uuid_key = "host_uuid" if lock_host else "uuid"
        lru = self.cache.host_lru_lists[ct] if lock_host else self.cache.lru_lists[ct]

        # Tombstoned nodes (cd.value is None) have no SWA chunk to protect
        # skip them and keep walking up. This path is hit when HiCache
        # backs up a FULL present internal node whose SWA was already evicted.
        cur = node
        while cur != root and swa_lock_size < sliding_window_size:
            comp = cur.component_data[ct]
            value = comp.host_value if lock_host else comp.value
            # tombstone 节点: 无 SWA 数据需要保护, 跳过并记录
            if value is None:
                result.skip_lock_node_ids.setdefault(ct, set()).add(cur.id)
                cur = cur.parent
                continue

            ref = comp.host_lock_ref if lock_host else comp.lock_ref
            if ref == 0:
                # 首次 lock: 从可驱逐池移除 (移出 LRU 或移动计数)
                if lock_host:
                    if lru.in_list(cur):
                        lru.remove_node(cur)
                else:
                    key_len = len(cur.key)
                    self.cache.component_evictable_size_[ct] -= key_len
                    self.cache.component_protected_size_[ct] += key_len
            # 锁计数 +1
            if lock_host:
                comp.host_lock_ref = ref + 1
            else:
                comp.lock_ref = ref + 1

            # 边界之外不就lock了
            # 累计锁定的 SWA token 数
            swa_lock_size += len(value)
            # 达到窗口大小时: 设置 UUID 边界 (release 时的终止标记)
            if swa_lock_size >= sliding_window_size:
                if comp.metadata.get(uuid_key) is None:
                    comp.metadata[uuid_key] = next_component_uuid()
                swa_uuid = comp.metadata[uuid_key]
            cur = cur.parent

        # 记录锁边界 UUID 到 result (release 时通过此 UUID 找到停止位置)
        if lock_host:
            result.swa_uuid_for_host_lock = swa_uuid
        else:
            result.swa_uuid_for_lock = swa_uuid
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: Optional[DecLockRefParams],
        lock_host: bool = False,
    ) -> None:
        """🔓 反向释放 Window-Lock —— 递减 lock_ref, 遇到 UUID 边界时停止。

        🔗 调用场景:
            ① lock_host=False (device unlock): 请求完成 (cache_finished_req) 或锁交换
               (cache_unfinished_req) 时调 dec_lock_ref, 释放对旧匹配路径的 SWA 窗口保护。
            ② lock_host=True (host unlock): HiCache DMA 完成后 (load_back / prefetch 结束)
               调 dec_host_lock_ref, 释放 Host 锚点保护。

        ⚙️ 与 acquire 反向: 从 node 向上走, lock_ref 递减。
            当遇到 swa_uuid_for_lock 标记的 UUID 节点时 dec_swa=False (该节点及以上不再递减)。
            skip_lock_node_ids: release 时跳过这些 tombstone 节点 (它们在 acquire 时被跳过)。

            lock_ref 从 1→0 时:
              Device: 从 protected_size 移回 evictable_size。
              Host: 若有 host_value 且 device tombstone, 重新插入 Host LRU (使可被 host eviction)。"""
        ct = self.component_type
        root = self.cache.root_node
        swa_uuid_for_lock = (
            (params.swa_uuid_for_host_lock if lock_host else params.swa_uuid_for_lock)
            if params
            else None
        )
        skip_lock_node_ids = params.skip_lock_node_ids.get(ct, ()) if params else ()
        dec_swa = True       # 是否继续向上递减 (遇到 UUID 边界后变 False)
        uuid_key = "host_uuid" if lock_host else "uuid"

        # A node in skip_lock_node_ids was a tombstone when this lock was acquired.
        cur = node
        while cur != root and dec_swa:
            comp = cur.component_data[ct]
            # 跳过 acquire 时就已经是 tombstone 的节点 (它们没有 lock_ref 被增加)
            if cur.id in skip_lock_node_ids:
                cur = cur.parent
                continue
            ref = comp.host_lock_ref if lock_host else comp.lock_ref
            if ref == 0:
                # 没锁可减, 继续向上
                cur = cur.parent
                continue
            if ref == 1:
                # 最后一个锁释放: lock_ref 1→0, 恢复可驱逐状态
                if lock_host:
                    # Host 锁释放: 若 device 是 tombstone 且 host 还有数据, 放回 Host LRU
                    if comp.value is None and comp.host_value is not None:
                        host_lru = self.cache.host_lru_lists[ct]
                        if not host_lru.in_list(cur):
                            host_lru.insert_mru(cur)
                else:
                    # Device 锁释放: 从 protected 移回 evictable
                    key_len = len(comp.value)
                    self.cache.component_evictable_size_[ct] += key_len
                    self.cache.component_protected_size_[ct] -= key_len
            # lock_ref 递减
            if lock_host:
                comp.host_lock_ref = ref - 1
            else:
                comp.lock_ref = ref - 1
            # 遇到 UUID 边界 → 停止 (该节点及以上不属于本窗口, 无需释放)
            if swa_uuid_for_lock and comp.metadata.get(uuid_key) == swa_uuid_for_lock:
                dec_swa = False
            cur = cur.parent

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        """💾 缓存前准备 —— 仅在 is_finished 时写入 swa_evicted_seqlen 到 insert_params。

        🔗 UnifiedRadixCache.cache_finished_req() / cache_unfinished_req() 中调用 (:813, :880)。

        ⚙️ is_finished=True (请求完成, 整段 prefill KV 可缓存):
            把 req.swa_evicted_seqlen 写入 insert_params, 后续 insert 时据此判断 SWA 窗口边界。
            非完成请求只缓存部分, 不需要 SWA tombstone 判断 (窗口边界由 decoder 前移决定)。

            返回 None: 不截断缓存长度 (使用 full length)。

        📥 req: 请求对象 (含 swa_evicted_seqlen 字段)。
        📥 insert_params: 待填充的 InsertParams。
        📥 token_ids_len: token ID 列表长度。
        📥 is_finished: 该请求是否已完成生成。
        📤 返回 None (不截断缓存长度)。"""
        if is_finished:
            # 完整请求: 记录 SWA 窗口驱逐边界 → insert 时判断 tombstone 重叠
            insert_params.swa_evicted_seqlen = req.swa_evicted_seqlen
        return None

    def free_out_of_window_slots(
        self, req: Req, pre_len: int, insert_params: InsertParams
    ) -> None:
        """🗑️ 释放 decoder 过程中滑出窗口的旧 SWA slot —— 回收窗口外 SWA pool 空间。

        🔗 UnifiedRadixCache.cache_finished_req / cache_unfinished_req (:891)。
            在树缓存 insert 之前调用, 把 decoder 前移产生的旧 SWA token 的 KV pool slot 归还。

        ⚙️ 委托给 free_swa_out_of_window_slots() (common.py:62):
            计算 evict_threshold = pre_len - sliding_window_size (- page_size 缓冲),
            将 req.swa_evicted_seqlen 到 evict_threshold 之间的 SWA slot 释放。
            (减去 page_size 是为了保留至少 1 页在窗口内, 避免叶子节点变 tombstone 导致 SWA 泄漏)

            之后把 req.swa_evicted_seqlen 同步到 insert_params (供插入阶段判断边界)。

        📥 req: 请求对象。
        📥 pre_len: 本次缓存前的 token 长度 (prefix 匹配后的总长度)。
        📥 insert_params: 待更新的 InsertParams。"""
        if self.sliding_window_size is not None:
            free_swa_out_of_window_slots(
                req,
                pre_len,
                sliding_window_size=self.sliding_window_size,
                page_size=self.cache.page_size,
                req_to_token_pool=self.cache.req_to_token_pool,
                token_to_kv_pool_allocator=self.cache.token_to_kv_pool_allocator,
            )
        insert_params.swa_evicted_seqlen = req.swa_evicted_seqlen

    # ═════════════════════════ 💿 HiCache Hooks ═════════════════════════

    def build_hicache_transfers(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        *,
        req: Optional[Req] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:
        """💿 构建 HiCache 传输描述符 —— 4 阶段: BACKUP_HOST / LOAD_BACK / BACKUP_STORAGE / PREFETCH。

        🔗 UnifiedRadixCache._backup_to_host() / _load_back_from_host() / _backup_to_storage() / _prefetch_from_storage()
            在各个阶段遍历所有 component 调用 (:1622, :1747)。

        ⚙️ 四阶段行为:
            ┌─ BACKUP_HOST (Device→Host) ─┐
            │ 将节点 SWA value (Device pool 索引) 封装为 PoolTransfer(device_indices=...)  │
            │ 供 HostPoolGroup 执行 D→H 拷贝。                                          │
            │ cd.value 已是 SWA pool 索引 (insert 时已 translate), 转为 int64 给 host。   │
            │ cd.value=None (tombstone) 时返回 None (无需备份)。                        │
            └──────────────────────────────┘

            ┌─ LOAD_BACK (Host→Device) ─┐
            │ `node` = best_match_node; SWA validator 保证窗口内每个有效祖先都有 value 或 host_value.│
            │ 从 node 向上走, 收集仅有 host_value 的 tombstone 节点的 host_indices。               │
            │ 跳到 device 已有 value 的节点时只累加 n_swa 但不收集 (device 已够)。                  │
            │ 拼接所有 host_indices → PoolTransfer(host_indices=cat(...), nodes_to_load=[...])    │
            └─────────────────────────────┘

            ┌─ BACKUP_STORAGE (Host→Storage) ─┐
            │ 将 host_value 尾部配以 hash keys 写入存储 (page 对齐)。                           │
            │ hit_policy=TRAILING_PAGES: 查询时匹配尾页 hash, 命中则返回匹配到的页。              │
            └─────────────────────────────────┘

            ┌─ PREFETCH (Storage→Host) ─┐
            │ 申请 host pool 缓冲区 (一个完整 SWA 窗口大小), 构造 placeholder transfer。         │
            │ 内存不足时先触发 host eviction。                                                │
            │ 返回池 (alloc 失败 → 返回空列表让 caller 放弃 prefetch)。                         │
            │ 实际数据由 storage backend 根据 placeholder keys 异步加载。                       │
            └────────────────────────────┘

        📥 node: 操作目标节点 (BACKUP_HOST/STORAGE: 单一节点; LOAD_BACK: match 节点)。
        📥 phase: 传输阶段 (BACKUP_HOST | LOAD_BACK | BACKUP_STORAGE | PREFETCH)。
        📥 req: 请求对象 (LOAD_BACK 用; PREFETCH 用 token_ids)。
        📥 token_ids: 请求 token 序列 (PREFETCH 阶段 source hash)。
        📥 prefetch_tokens: 请求还未被 prefix match 的剩余 token 数 (PREFETCH 阶段判断窗口是否满)。
        📥 last_hash: 请求最后 page 的 hash (PREFETCH 阶段 range 查询的终点)。
        📤 Optional[list[PoolTransfer]]: None=无传输, [] = 放弃, 或含传输描述符的列表。"""
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            cd = node.component_data[ct]
            if cd.value is None:
                # tombstone: 无 device SWA 数据需要备份
                return None
            # cd.value already holds SWA-pool indices (translated at insert time).
            # Host pool indexing wants int64.
            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    device_indices=cd.value.to(torch.int64),
                )
            ]

        if phase == CacheTransferPhase.LOAD_BACK:
            # `node` is best_match_node; the SWA validator guarantees every
            # ancestor within `sliding_window_size` has value or host_value.
            n_swa = 0          # 累计窗口内 token 数
            backed_up: list[torch.Tensor] = []  # 收集需要 load 的 host_indices
            nodes: list = []   # 对应的节点列表 (load 后回填用)
            cur = node
            while cur is not self.cache.root_node and n_swa < self.sliding_window_size:
                cd = cur.component_data[ct]
                assert cd.host_value is not None or cd.value is not None
                if cd.value is not None:
                    # device 已有数据, 跳过 (只累加计数)
                    n_swa += len(cd.value)
                else:
                    # host only tombstone, 需要 load_back
                    backed_up.append(cd.host_value)
                    nodes.append(cur)
                    n_swa += len(cd.host_value)
                cur = cur.parent

            if not backed_up:
                # 窗口内全在 device, 无需 load
                return None

            # 从最近祖先到最佳匹配节点顺序 (loading 从远端到近端)
            backed_up.reverse()
            nodes.reverse()

            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=torch.cat(backed_up),
                    device_indices=None,           # 运行时由 HostPoolGroup 分配
                    nodes_to_load=nodes,            # commit 时回填 device_indices
                )
            ]

        if phase == CacheTransferPhase.BACKUP_STORAGE:
            cd = node.component_data[ct]
            if cd.host_value is None or not node.hash_value:
                return None
            num_pages = len(cd.host_value) // self.cache.page_size
            if num_pages == 0:
                return None
            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=cd.host_value[-num_pages * self.cache.page_size :],
                    keys=node.hash_value[-num_pages:],      # 尾部 page 的 hash
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,  # 匹配尾页
                )
            ]

        if phase == CacheTransferPhase.PREFETCH:
            # Require a full sliding window.
            sw_pages = (
                self.sliding_window_size + self.cache.page_size - 1
            ) // self.cache.page_size
            # 窗口至少需要 sw_pages 页; 若剩余 token 不够或窗口大小=0, 放弃
            if sw_pages == 0 or prefetch_tokens // self.cache.page_size < sw_pages:
                return None
            num_tokens = sw_pages * self.cache.page_size
            # 在 host pool 中申请一个完整窗口的缓冲区
            host_indices = self._swa_kv_pool_host.alloc(num_tokens)
            if host_indices is None:
                # Host pool 满了, 尝试驱逐 SWA host 资源
                self.cache.evict_host(num_tokens, ComponentType.SWA)
                host_indices = self._swa_kv_pool_host.alloc(num_tokens)
            if host_indices is None:
                # 驱逐后仍失败, 放弃 prefetch
                return []
            return [
                PoolTransfer(
                    name=PoolName.SWA,
                    host_indices=host_indices,
                    keys=["__placeholder__"] * sw_pages,  # 占位 key, 由 storage backend 匹配
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]

        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
        *,
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        """💿 HiCache 传输后处理 —— 4 阶段各自更新 SWA host/device 状态。

        🔗 UnifiedRadixCache._backup_to_host() / _load_back_from_host() / _backup_to_storage() / _prefetch_from_storage()
            在各阶段的数据传输完成后调用 (:1654, :1794)。

        ⚙️ 四阶段行为:
            BACKUP_HOST: 将 D→H 复制后的 host_indices 写入 cd.host_value。
            LOAD_BACK: 遍历 nodes_to_load, 对每个节点:
                         ① 从 device_indices 切片出对应 SWA chunk
                         ② 调用 _restore_device_value 复活 tombstone
                         ③ 调用 set_full_to_swa_mapping 重建 Full↔SWA index 映射
            BACKUP_STORAGE: (当前无 SWA 特定操作, 基类处理)
            PREFETCH: 委托给 _commit_prefetch() (复杂逻辑: 填 tombstone / 释放多余 slice)。"""
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            # D→H 完成后: 记录 host pool 索引
            if transfers and transfers[0].host_indices is not None:
                cd = node.component_data[ct]
                if cd.host_value is None:
                    cd.host_value = transfers[0].host_indices.clone()
            return

        if phase == CacheTransferPhase.LOAD_BACK:
            assert transfers and transfers[0].device_indices is not None
            xfer = transfers[0]
            device_indices = xfer.device_indices  # HostPoolGroup 分配好的 device 索引
            allocator = self.cache.token_to_kv_pool_allocator

            offset = 0
            for n in xfer.nodes_to_load or []:
                cd_n = n.component_data[ct]
                cd_full_n = n.component_data[BASE_COMPONENT_TYPE]
                n_tokens = len(cd_n.host_value)
                # 切片出本节点的 SWA device chunk
                swa_chunk = device_indices[offset : offset + n_tokens].clone()
                self._restore_device_value(n, swa_chunk)
                assert cd_full_n.value is not None and len(cd_full_n.value) == n_tokens
                # rebuild the mapping for the loaded SWA chunk
                # Full↔SWA 映射在 allocator 内部维护 (set_full_to_swa_mapping)
                # 这确保后续 free_swa(full_indices) 能找到对应的 SWA slot
                allocator.set_full_to_swa_mapping(cd_full_n.value, swa_chunk)
                offset += n_tokens
            assert offset == len(xfer.host_indices)  # 所有 host 数据都 load 完了
            return

        if phase == CacheTransferPhase.PREFETCH:
            self._commit_prefetch(
                node,
                transfers,
                insert_result=insert_result,
                pool_storage_result=pool_storage_result,
            )
            return

    def _release_swa_host(self, host_indices: torch.Tensor) -> None:
        """🗑️ 释放 SWA host pool 分配——将未使用的 host_indices 归还给 cache_controller。

        🔗 _commit_prefetch() 中多次调用 (释放多余/未命中的 host buffer 片段)。"""
        if host_indices is not None and host_indices.numel() > 0:
            self.cache.cache_controller.append_host_mem_release(
                extra_pools=[PoolTransfer(name=PoolName.SWA, host_indices=host_indices)]
            )

    def _attach_swa_host_value(
        self, node: UnifiedTreeNode, host_indices: torch.Tensor
    ) -> None:
        """💿 将 host_indices 写入节点的 SWA host_value 并刷新树状态。

        🔗 _commit_prefetch() 中对每个新 tombstone→host 的节点调用。

        ⚙️ ① cd.host_value = host_indices.clone()
            ② 若 device 是 tombstone 且不在 Host LRU → 加入 Host LRU
            ③ _update_evictable_leaf_sets(node) 及 parent (更新可驱逐叶子集合)。"""
        # Write host_indices into node's SWA host_value and refresh tree state.
        ct = self.component_type
        cd = node.component_data[ct]
        cd.host_value = host_indices.clone()
        host_lru = self.cache.host_lru_lists[ct]
        # 若 device 是 tombstone (无 device value) 则加入 Host LRU 以便 host eviction
        if cd.value is None and not host_lru.in_list(node):
            host_lru.insert_mru(node)
        # 更新可驱逐叶子集合 (节点状态从 tombstone→有 host_value 可能改变 evictable 判定)
        self.cache._update_evictable_leaf_sets(node)
        if node.parent:
            self.cache._update_evictable_leaf_sets(node.parent)

    def _commit_prefetch(
        self,
        anchor,
        transfers: list[PoolTransfer],
        *,
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        """💿 Prefetch 提交 —— 将预取的 SWA 窗口沿 leaf→anchor 路径填充 tombstone。

        🔗 commit_hicache_transfer(PREFETCH) → 委托本方法 (:697)。

        ⚙️ 全量or无: loaded_pages 是跨 TP rank MIN, 若不足 window_require_pages 则整窗口放弃
            (保持所有 TP rank 的树一致)。否则填充:
              1. 定位 target (inserted_host_node) 和 anchor (调用方传入的 node)
              2. 计算 loaded_start (buffer 覆盖的 token 范围起点)
              3. 从 leaf(target) 向 anchor 遍历:
                   - tombstone 节点且在 buffer 内: split 若骑跨, 然后 _attach_swa_host_value
                   - 已有 host_value: 释放该 slice (不需要重复填充)
              4. buffer 前缀 (在 leaf→anchor 路径之外的) 释放。

        📥 anchor: 锚节点 (根方向边界, 通常是调用 build 时的 `node`)。
        📥 transfers: PoolTransfer 列表 (含 host_indices buffer)。
        📥 insert_result: InsertResult (含 total_len, inserted_host_node)。
        📥 pool_storage_result: PoolTransferResult (含 extra_pool_hit_pages for SWA)。"""
        # Fill the prefetched SWA window onto the leaf→anchor path.
        #
        # All-or-nothing over one full window: ``loaded_pages`` is the cross-rank
        # MIN, so ``loaded_pages < window_pages`` drops the whole window (keeps the
        # tree identical across TP ranks). Otherwise map the buffer to token range
        # ``[loaded_start, total_len)`` and walk leaf→anchor, filling SWA
        # tombstones and releasing slices that already have host_value.
        if not transfers:
            return
        ct = self.component_type
        page_size = self.cache.page_size
        host_indices = transfers[0].host_indices
        window_require_pages = (
            host_indices.numel() // page_size if host_indices is not None else 0
        )
        loaded_pages = (
            pool_storage_result.extra_pool_hit_pages.get(PoolName.SWA, 0)
            if pool_storage_result
            else 0
        )
        target = insert_result.inserted_host_node if insert_result else None
        # 放弃条件: 无 target / 窗口页数为0 / 命中页数不足 → 释放 host buffer
        if (
            target is None
            or window_require_pages == 0
            or loaded_pages < window_require_pages
        ):
            self._release_swa_host(host_indices)
            return

        # Buffer covers token range [loaded_start, total_len).
        loaded_start = insert_result.total_len - window_require_pages * page_size

        # Walk leaf → anchor; ``pos`` is the right edge of ``cur`` in tokens.
        pos, cur = insert_result.total_len, target
        while cur is not anchor and pos > loaded_start:
            node_start = pos - len(cur.key)  # cur 节点的左侧边界 (token 位置)
            # Intersection of cur's range and the buffer.
            fill_start = max(node_start, loaded_start)
            fill_len = pos - fill_start
            buf_off = fill_start - loaded_start
            slice_ = host_indices[buf_off : buf_off + fill_len]

            cd = cur.component_data[ct]
            if cd.host_value is None and fill_len > 0:
                # 当前节点是 tombstone (无 host_value) 且在 buffer 范围内 → 填充
                # Tombstone: split off the in-buffer tail if needed, then fill.
                if fill_start > node_start:
                    # 节点骑跨 buffer 左边界 → split
                    self.cache._split_node(cur.key, cur, fill_start - node_start)
                self._attach_swa_host_value(cur, slice_)
            else:
                # 已有 SWA (或重叠为空): 释放此 slice
                # Already has SWA (or empty overlap): drop this slice.
                self._release_swa_host(slice_)

            pos = node_start
            cur = cur.parent

        # Buffer prefix that fell outside the anchor→leaf path.
        if pos > loaded_start:
            self._release_swa_host(host_indices[: pos - loaded_start])

    def drive_host_eviction(
        self, num_tokens: int, tracker: dict[ComponentType, int]
    ) -> None:
        """🗑️ SWA Host 层驱逐 —— 遍历 Host LRU, 叶子全删, 内部 tombstone + 级联。

        🔗 HostPoolGroup → cache.evict_host() → 遍历 component 调用 (:715)。

        ⚙️ 类似 drive_eviction 但操作 Host 层:
            叶子 (evictable_host_leaves): _evict_host_leaf(x, tracker) 原子驱逐所有组件
            内部节点: _evict_component_and_detach_lru(x, SWA, HOST) + _cascade_evict

            内部节点 device 可能是 tombstone (仅有 host_value), 此时 host_value 被释放。"""
        # Evict SWA host resources.
        # Internal nodes: private tombstone (free SWA host only).
        # Host leaves: atomic eviction via _evict_host_leaf.
        ct = self.component_type
        host_lru = self.cache.host_lru_lists[ct]
        x = host_lru.get_lru_no_host_lock()
        while tracker[ct] < num_tokens and x is not None and host_lru.in_list(x):
            x_next = host_lru.get_prev_no_host_lock(x)
            cd = x.component_data[ct]
            if x in self.cache.evictable_host_leaves:
                # Host 叶子: 原子驱逐所有组件 (整叶 host 删除)
                self.cache._evict_host_leaf(x, tracker)
            else:
                assert cd.host_value is not None
                # 内部节点: tombstone host 数据 + 级联
                self.cache._evict_component_and_detach_lru(
                    x, self, target=EvictLayer.HOST, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker, target=EvictLayer.HOST)
            x = x_next
