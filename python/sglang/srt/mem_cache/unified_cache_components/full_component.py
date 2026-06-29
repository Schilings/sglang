from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    TreeComponent,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedTreeNode,
    )


class FullComponent(TreeComponent):
    """📦 Full Attention KV Cache 组件 —— 标准全注意力模型的缓存驱逐/锁定策略。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧩 对外接口清单 (按 UnifiedRadixCache 调用阶段分组)                                  ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                  ║
    ║  🔍 Match 阶段 (match_prefix → _match_prefix_helper + _match_post_processor):     ║
    ║    create_match_validator()  → 闭包: value 非 None (或 backuped) 即有效匹配边界     ║
    ║    finalize_match_result()   → 统计 host_hit_length (HiCache: 需 load_back 的量)    ║
    ║    refresh_lru()             → 不 override! Full 用 last_access_time, 不走 LRU      ║
    ║                                                                                  ║
    ║  📝 Insert 阶段 (insert → _insert_helper):                                         ║
    ║    update_component_on_insert_overlap() → 不 override! Full 无 tombstone, 直接复用  ║
    ║    should_skip_leaf_creation()          → 不 override! Full 总是创建叶子            ║
    ║    recover_after_unevict()              → 不 override! Full 由 _unevict_node 直接恢复║
    ║    commit_insert_component_data()       → 不 override! Full value 由 _add_new_node 设║
    ║    redistribute_on_node_split()         → 复制 lock_ref + 切片 value/host_value     ║
    ║                                                                                  ║
    ║  🗑️ Evict 阶段 (evict → drive_eviction → _cascade_evict):                          ║
    ║    drive_eviction()     → 从 evictable_device_leaves 堆驱逐 (last_access_time 排序) ║
    ║    evict_component()    → free_full + 更新 evictable_size (value=None 延迟到级联后)  ║
    ║    eviction_priority()  → leaf=0, internal=2 (最高, 级联驱逐 SWA+Mamba)              ║
    ║    drive_host_eviction()→ 从 evictable_host_leaves 堆驱逐 Host 端 Full KV            ║
    ║                                                                                  ║
    ║  🔒 Lock 阶段 (inc_lock_ref / dec_lock_ref):                                        ║
    ║    acquire_component_lock() → Path-lock: 从 node 沿父链锁到 root (跳过 evicted 段)    ║
    ║    release_component_lock() → Path-unlock: 对称释放 (跳过 inc 时跳过的 evicted 段)    ║
    ║                                                                                  ║
    ║  💿 HiCache Hooks (write_backup / load_back / write_backup_storage / prefetch):     ║
    ║    build_hicache_transfers()  → LOAD_BACK: 收集 evicted 节点的 host_value            ║
    ║    commit_hicache_transfer()  → BACKUP_HOST: 写 host_value; LOAD_BACK: 回填 value    ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔗 宏观交互调用链 (UnifiedRadixCache → FullComponent)                              ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                  ║
    ║  Scheduler → UnifiedRadixCache.match_prefix(key)                                  ║
    ║    └─ _match_prefix_helper()                                                     ║
    ║         ├─ FullComponent.create_match_validator()  ← 判定匹配边界 (value/backuped)  ║
    ║         └─ _match_post_processor()                                                ║
    ║              └─ FullComponent.finalize_match_result() ← 统计 host_hit_length       ║
    ║    (Full 不走 refresh_lru; _match_post_processor 直接更新 last_access_time)         ║
    ║                                                                                  ║
    ║  Scheduler → UnifiedRadixCache.insert(key, value)                                 ║
    ║    └─ _insert_helper()                                                            ║
    ║         ├─ _add_new_node() → 直接写 Full value (不经 component hook)                ║
    ║         ├─ _unevict_node_on_insert() → 直接恢复 Full value (不经 component hook)    ║
    ║         ├─ _split_node()                                                          ║
    ║         │    └─ FullComponent.redistribute_on_node_split() ← 切片 value + 复制 lock ║
    ║         └─ (Full 不 override commit/overlap/skip/unevict — 基类默认即可)             ║
    ║                                                                                  ║
    ║  Scheduler → UnifiedRadixCache.evict(params)                                      ║
    ║    └─ FullComponent.drive_eviction() ← 从 evictable_device_leaves 堆驱逐            ║
    ║         └─ _evict_device_leaf() → _evict_component_and_detach_lru()               ║
    ║              └─ FullComponent.evict_component() ← free_full + 更新 evictable_size  ║
    ║         └─ _cascade_evict() → 级联驱逐 SWA(1)+Mamba(0) (Full priority=2 最高)       ║
    ║                                                                                  ║
    ║  Scheduler → UnifiedRadixCache.inc_lock_ref(node)                                 ║
    ║    └─ FullComponent.acquire_component_lock() ← Path-lock: node→root 逐祖先 +1      ║
    ║  Scheduler → UnifiedRadixCache.dec_lock_ref(node)                                 ║
    ║    └─ FullComponent.release_component_lock() ← Path-unlock: 对称递减                ║
    ║                                                                                  ║
    ║  HiCache: write_backup / load_back / prefetch                                     ║
    ║    ├─ FullComponent.build_hicache_transfers()  ← 构造 KV 传输描述符                 ║
    ║    └─ FullComponent.commit_hicache_transfer()  ← 回填 host_value / device value    ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧬 设计要点                                                                       ║
    ║                                                                                  ║
    ║  1️⃣  Full 是 BASE_COMPONENT_TYPE — 许多操作由 UnifiedRadixCache 直接处理,           ║
    ║     不经 component hook:                                                           ║
    ║       - _add_new_node() 直接写 Full value (不经 commit_insert_component_data)       ║
    ║       - _unevict_node_on_insert() 直接恢复 Full value (不经 recover_after_unevict)  ║
    ║       - _match_post_processor() 直接更新 last_access_time (不经 refresh_lru)        ║
    ║     原因: Full 是"主线"组件, 树的 key/value 操作天然就是 Full 的操作。                 ║
    ║                                                                                  ║
    ║  2️⃣  free 策略 (SWA 共存时):                                                       ║
    ║     若 SWA 同时存在 → _free_full = full_attn_allocator.free (仅 free Full KV)       ║
    ║     否则            → _free_full = allocator.free (释放全部 KV)                    ║
    ║     SWA 的 KV 由级联驱逐触发 SWAComponent.evict_component 释放。                     ║
    ║                                                                                  ║
    ║  3️⃣  驱逐用叶子集合堆 (evictable_device_leaves), 不用 LRU:                           ║
    ║     Full 只驱逐叶子 (内部节点保留以维持树拓扑)。                                       ║
    ║     堆键 = eviction_strategy.get_priority(n) (基于 last_access_time 的 LRU/FIFO)。   ║
    ║     驱逐叶后 parent 可能变叶子 → 推入堆继续驱逐。                                      ║
    ║                                                                                  ║
    ║  4️⃣  Path-lock (不同于 SWA 的 window-lock):                                         ║
    ║     从 node 沿父链锁到 root, 跳过底部 evicted 段 (HiCache 场景)。                     ║
    ║     lock_ref 0→1: evictable_size → protected_size + 从 evictable_device_leaves 移除 ║
    ║     lock_ref 1→0: protected_size → evictable_size + 可能加入 evictable_device_leaves║
    ║                                                                                  ║
    ║  5️⃣  级联优先级: leaf=0, internal=2 (最高)                                          ║
    ║     Full(2) > SWA(1) > Mamba(0): 驱逐 Full 时级联清理 SWA+Mamba。                     ║
    ║     原因: Full 是树拓扑的基础, Full 驱逐意味着节点要删除, 辅组件数据失去依附。          ║
    ║                                                                                  ║
    ║  6️⃣  evict_component 的 value=None 延迟:                                            ║
    ║     evict_component 只 free_full + 更新 size, 不设 value=None。                      ║
    ║     value=None 延迟到 _cascade_evict 末尾 — 因为 SWA 的 free_swa 需要读 Full.value   ║
    ║     做索引映射 (free_swa(full_indices))。                                            ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """
    component_type = ComponentType.FULL

    def __init__(self, cache, params):
        """📦 Full 组件初始化 —— 确定 free 策略和 HiCache Host Pool。

        🔗 UnifiedRadixCache.__init__ 通过 COMPONENT_REGISTRY 实例化本组件。
            仅在服务启动时调用一次。

        ⚙️ free 策略:
            若 SWA 同时存在 → free_full = full_attn_allocator.free (仅 Free Full KV)
            否则          → free_full = allocator.free (释放全部 KV, 无 SWA)
            SWA 的 KV 由级联驱逐触发 SWAComponent.evict_component 释放。"""
        super().__init__(cache, params)
        allocator = cache.token_to_kv_pool_allocator
        # When SWA is present, only free full-attention KV here;
        # SWA KV will be freed by cascade via SWAComponent.evict_component.
        if ComponentType.SWA in cache.tree_components:
            self._free_full = allocator.full_attn_allocator.free
        else:
            self._free_full = allocator.free
        # HiCache state: set to host KV pool when HiCache enabled
        self._full_kv_pool_host = None

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        """🔍 返回闭包判断节点是否有效 Full 匹配边界。

        🔗 调用场景: 每个请求 match_prefix 时, _match_prefix_helper 创建一次闭包,
            在树遍历中对每个节点调用 validator(node) 判定是否推进匹配。
            ├─ match_device_only=True:  用于 best_match_device_node (scheduler 用, 只认 device)
            └─ match_device_only=False: 用于 best_match_node (HiCache 用, device 或 host backup 均可)

        ⚙️ Full 的匹配判定:
            - device 模式: value 非 None → True (有 device KV 才能直接用)
            - host 模式:   value 非 None OR backuped → True (HiCache: evicted+backuped 也可匹配, 后续 load_back 恢复)

        📥 match_device_only: True=纯设备匹配, False=含 Host 匹配 (HiCache 场景)。
        📤 闭包 (node) → bool。"""
        if match_device_only:
            return (
                lambda node: node.component_data[self.component_type].value is not None
            )

        # Full Component默认要匹配device和host的链
        # 丨 ----- 在device上----丨------backuped----丨
        # node0 -> node1 -> node2
        # node0 -> node1 -> node2  -> node3 -> node4
        # HiCache: evicted + backuped nodes are valid match boundaries.
        return lambda node: (                                  # 这里是 OR
            node.component_data[self.component_type].value is not None or node.backuped
        )

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        """🔍 匹配后处理 —— 计算 Full KV 的 Host 命中长度 (host_hit_length)。

        🔗 调用场景: match_prefix → _match_post_processor → 遍历组件调用本方法。
            在 HiCache 场景下, 匹配路径可能跨越 device 和 host 两段:
              best_match_node (含 host) → ... → last_device_node (device 锚点)
            中间的 evicted+backuped 节点的 host_value 需要被 load_back 回 device。
            本函数统计这些 host_value 的总长度, 写入 result.host_hit_length,
            供 Scheduler 据此分配 device 内存并触发 load_back。

        ⚙️ 非 HiCache 场景: best_match_node == last_device_node, 循环不执行, host_hit_length=0。"""
        ct = self.component_type
        kv_host_hit = 0
        node = result.best_match_node
        root_node = self.cache.root_node
        while node is not result.last_device_node and node is not root_node:
            full_host = node.component_data[ct].host_value
            if full_host is not None:
                kv_host_hit += len(full_host)
            node = node.parent
        if kv_host_hit > 0:
            return result._replace(
                host_hit_length=max(result.host_hit_length, kv_host_hit)
            )
        return result

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        """✂️ 分裂时重分配 Full 数据 —— 把 lock_ref 复制到新 parent, 切片 value/host_value。

        🔗 调用场景: _split_node 在以下两种情况被触发时调用本方法:
            ① match_prefix 遍历树时, key 与 child.key 只部分匹配 → split
            ② insert 遍历树时, key 与已有 node.key 只部分匹配 → split
            split 后 new_parent 持有前半段 key, child 持有后半段。
            Full 的 value/host_value/lock_ref 需要按 split_len 切分到两个节点。

        ⚙️ lock_ref 复制 (非切分): new_parent 继承 child 的 lock_ref,
            因为分裂前的锁保护的是整段, 分裂后两段都需要保护。"""
        ct = self.component_type
        new_parent.component_data[ct].lock_ref = child.component_data[ct].lock_ref
        child_cd = child.component_data[ct]
        split_len = len(new_parent.key)
        if child_cd.value is not None:
            new_parent.component_data[ct].value = child_cd.value[:split_len].clone()
            child_cd.value = child_cd.value[split_len:].clone()
        if child_cd.host_value is not None:
            new_parent.component_data[ct].host_value = child_cd.host_value[
                :split_len
            ].clone()
            child_cd.host_value = child_cd.host_value[split_len:].clone()

    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        """🗑️ 驱逐节点上的 Full KV 资源。

        🔗 调用场景: 驱逐流水线中的原子步骤, 被以下路径调用:
            ① drive_eviction → _evict_device_leaf → _evict_component_and_detach_lru
               (Full 主动驱逐叶子时)
            ② _cascade_evict → _evict_component_and_detach_lru
               (其他组件触发级联, Full priority=2 最高, 通常不会被级联)
            ③ _evict_to_host / _evict_host_leaf
               (HiCache D→H 降级或 Host 叶子驱逐)

        ⚙️ DEVICE 层: free_full + 更新 evictable_size。
            ⚠️ value=None 不在此处设! 延迟到 _cascade_evict 末尾。
            原因: SWA 的 free_swa(full_indices) 需要读 Full.value 做索引映射,
            若提前置 None 会导致 SWA 无法正确释放。
        HOST 层: free host pool + host_value=None (无延迟)。"""
        cd = node.component_data[self.component_type]
        freed = 0
        host_freed = 0

        # Device layer
        if EvictLayer.DEVICE in target and cd.value is not None:
            self._free_full(cd.value)
            freed = len(cd.value)
            self.cache.component_evictable_size_[self.component_type] -= freed
            # NOTE: cd.value = None is deferred to _cascade_evict (Full as trigger)
            # because SWA's free_swa still needs to read Full.value.
            # cd.value = None

        # Host layer
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            if self._full_kv_pool_host is not None:
                self._full_kv_pool_host.free(cd.host_value)
            cd.host_value = None
        return freed, host_freed

    def eviction_priority(self, is_leaf: bool) -> int:
        """🎯 Full 驱逐优先级: leaf=0 (最低), internal=2 (最高)。

        🔗 调用场景: _cascade_evict 判断级联范围时调用。
            Full(2) > SWA(1) > Mamba(0) → 驱逐 Full 时级联到所有组件。
            但 Full 通常不会被其他组件级联 (它是最高优先级)。

        ⚙️ leaf=0: 叶子驱逐时所有组件齐平, 任一组件驱逐叶子都级联到全部 (整叶删除)。
           internal=2: 内部节点最高优先级, 最后被驱逐。
           原因: Full 是树拓扑基础, 内部 Full 驱逐意味着路径断裂, 辅组件数据失去依附。"""
        return 0 if is_leaf else 2

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        """🗑️ 从 evictable_device_leaves 堆驱逐直到满足 request token 数。

        🔗 调用场景: KV pool 空闲空间不足时, alloc_token_slots → evict_from_tree_cache
            → UnifiedRadixCache.evict() → 遍历组件调 drive_eviction。
            Full 用叶子集合堆 (last_access_time 排序), 不用 LRU 链表。

        ⚙️ 流程:
            ① 将 evictable_device_leaves 构建为最小堆 (堆键=eviction_strategy.get_priority)
            ② 循环弹出最旧叶子 → _evict_device_leaf (整叶删除 + 级联)
            ③ 叶子删除后 parent 可能变叶子 → 推入堆继续驱逐
            ④ 直到 tracker[FULL] >= request 或堆空

        📥 params: EvictParams (num_tokens=需要释放的 Full token 数)。
        📥 tracker: 跨组件共享的驱逐计数, 级联时 SWA/Mamba 也会累加。"""
        request = params.num_tokens
        heap = [
            (self.cache.eviction_strategy.get_priority(n), n)
            for n in self.cache.evictable_device_leaves
        ]
        heapq.heapify(heap)
        ct = self.component_type
        while tracker[ct] < request and heap:
            _, x = heapq.heappop(heap)
            if x not in self.cache.evictable_device_leaves:
                continue
            self.cache._evict_device_leaf(x, tracker)
            if x.parent is not None and x.parent in self.cache.evictable_device_leaves:
                heapq.heappush(
                    heap,
                    (self.cache.eviction_strategy.get_priority(x.parent), x.parent),
                )

    def drive_host_eviction(
        self, num_tokens: int, tracker: dict[ComponentType, int]
    ) -> None:
        """🗑️ 从 evictable_host_leaves 堆驱逐 Host 端 Full KV。

        🔗 调用场景: HiCache 场景下 Host pool 空间不足时,
            write_backup / prefetch_from_storage → evict_host → drive_host_eviction。
            与 drive_eviction 结构对称, 操作 evictable_host_leaves 而非 device leaves。

        ⚙️ H-leaf = evicted + backuped + 无子节点 + 未锁的节点。
            驱逐 H-leaf 时 _evict_host_leaf 整叶删除 (Host 层 ALL)。"""
        heap = [
            (self.cache.eviction_strategy.get_priority(n), n)
            for n in self.cache.evictable_host_leaves
        ]
        heapq.heapify(heap)
        ct = self.component_type
        while tracker[ct] < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x not in self.cache.evictable_host_leaves:
                continue
            self.cache._evict_host_leaf(x, tracker)
            if x.parent is not None and x.parent in self.cache.evictable_host_leaves:
                heapq.heappush(
                    heap,
                    (self.cache.eviction_strategy.get_priority(x.parent), x.parent),
                )

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        """🔒 Path-lock: 从 node 沿 parent 一路锁到 root。

        🔗 调用场景:
            ① lock_host=False (device lock): 每个请求 match_prefix 后, Scheduler 调
               inc_lock_ref(last_device_node) 锁定匹配路径, 防止 KV 在使用中被驱逐。
               请求完成/锁交换时 dec_lock_ref 释放。
            ② lock_host=True (host lock): HiCache 的 write_backup / load_back / prefetch
               调 inc_host_lock_ref 锁定 Host 锚点, 防止 DMA 过程中 Host 数据被驱逐。

        ⚙️ Device path-lock 流程:
            ① 跳过底部 evicted 段 (HiCache: node 可能 evicted, 记录到 skip_lock_node_ids)
            ② 从首个 device-on 节点开始, 逐祖先 lock_ref += 1
            ③ lock_ref 0→1: evictable_size → protected_size + 从 evictable_device_leaves 移除

        ⚙️ Host lock: 仅锁单节点 (host_lock_ref += 1), 不沿父链 (只需保护 Host 锚点)。"""
        ct = self.component_type

        # Only the last host node needs to be protected.
        if lock_host:
            cd = node.component_data[ct]
            if cd.host_value is None:
                return result
            cd.host_lock_ref += 1
            self.cache._update_evictable_leaf_sets(node)
            return result

        root = self.cache.root_node
        cur = node

        # Skip the bottom evicted segment
        while cur is not root and cur.component_data[ct].value is None:
            result.skip_lock_node_ids.setdefault(ct, set()).add(cur.id)
            cur = cur.parent

        # Lock the device-on segment up to root
        delta = 0
        while cur is not root:
            cd = cur.component_data[ct]
            assert (
                cd.value is not None
            ), f"FULL invariant broken: evicted ancestor {cur.id} above device-on segment"
            if cd.lock_ref == 0:
                key_len = len(cd.value)
                self.cache.component_evictable_size_[ct] -= key_len
                self.cache.component_protected_size_[ct] += key_len
                delta += key_len
            cd.lock_ref += 1
            self.cache.evictable_device_leaves.discard(cur)
            cur = cur.parent
        result.delta = delta
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: Optional[DecLockRefParams],
        lock_host: bool = False,
    ) -> None:
        """🔓 Path-unlock: 从 node 沿 parent 一路解到 root (对称于 acquire)。

        🔗 调用场景:
            ① lock_host=False: 请求完成 (cache_finished_req) 或锁交换 (cache_unfinished_req)
               时调 dec_lock_ref, 释放对旧匹配路径的保护。
            ② lock_host=True: HiCache DMA 完成后 (write_backup / load_back / prefetch 结束)
               调 dec_host_lock_ref, 释放 Host 锚点保护。

        ⚙️ Device path-unlock 流程:
            ① 跳过 skip_lock_node_ids (acquire 时已跳过的 evicted 段, 不需释放)
            ② 逐祖先 lock_ref -= 1
            ③ lock_ref 1→0: protected_size → evictable_size + _update_evictable_leaf_sets (可能重新加入可驱逐集合)"""
        ct = self.component_type
        if lock_host:
            cd = node.component_data[ct]
            if cd.host_value is None or cd.host_lock_ref == 0:
                return
            cd.host_lock_ref -= 1
            self.cache._update_evictable_leaf_sets(node)
            return

        root = self.cache.root_node
        skip_lock_node_ids = params.skip_lock_node_ids.get(ct, ()) if params else ()
        cur = node
        while cur != root:
            if cur.id in skip_lock_node_ids:
                cur = cur.parent
                continue
            cd = cur.component_data[ct]
            assert cd.value is not None
            assert cd.lock_ref > 0

            if cd.lock_ref == 1:
                key_len = len(cd.value)
                self.cache.component_evictable_size_[ct] += key_len
                self.cache.component_protected_size_[ct] -= key_len
            cd.lock_ref -= 1
            if cd.lock_ref == 0:
                self.cache._update_evictable_leaf_sets(cur)
            cur = cur.parent

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
        """💿 构建 Full KV 的 HiCache 传输描述符。

        🔗 调用场景: HiCache D↔H↔Storage 传输时, UnifiedRadixCache 的以下方法调用:
            ① BACKUP_HOST (D→H): write_backup() → build_hicache_transfers(BACKUP_HOST)
               → Full KV 由主流程直接 cache_controller.write(device_value) 操作,
               不需要额外 PoolTransfer → 返回 None。
            ② LOAD_BACK (H→D): load_back() → build_hicache_transfers(LOAD_BACK)
               → 从 best_match_node 向上收集所有 evicted 节点的 host_value,
               拼成 PoolTransfer(KV, host_indices=cat(...), nodes_to_load=[...])。
               这些节点的 device value 已被驱逐, 需要 load 回来。
            ③ BACKUP_STORAGE / PREFETCH: Full 不参与 (返回 None, 由辅组件处理)。

        ⚙️ LOAD_BACK 的向上遍历: Full 只驱逐叶子, 所以一旦遇到 device-on 节点
            (value 非 None), 其上所有祖先也必然 device-on → 停止遍历。"""
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            # Full KV backup is handled by the main flow
            # (write_backup → cache_controller.write on host_value directly).
            # No extra PoolTransfer needed.
            return None

        if phase == CacheTransferPhase.LOAD_BACK:
            # `node` is best_match_node. FULL device evict only from leaves,
            # so once we hit a device-on node, everything above is also device-on
            backed_up: list[torch.Tensor] = []
            nodes: list = []
            cur = node
            while cur.evicted:
                cd = cur.component_data[ct]
                assert cd.host_value is not None
                backed_up.append(cd.host_value)
                nodes.append(cur)
                cur = cur.parent
            backed_up.reverse()
            nodes.reverse()
            return [
                PoolTransfer(
                    name=PoolName.KV,
                    host_indices=(
                        torch.cat(backed_up)
                        if backed_up
                        else torch.empty((0,), dtype=torch.int64, device="cpu")
                    ),
                    device_indices=None,
                    nodes_to_load=nodes,
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
        """💿 完成 HiCache 传输后的状态提交。

        🔗 调用场景: DMA 拷贝完成后, UnifiedRadixCache 的以下方法回调:
            ① BACKUP_HOST: write_backup() → cache_controller.write() 完成 D→H 拷贝后
               → commit_hicache_transfer(BACKUP_HOST)
               → 将返回的 host_indices clone 到 node.component_data[FULL].host_value。
               此后 node.backuped=True, 可被 _evict_to_host 降级。
            ② LOAD_BACK: load_back() → cache_controller.load() 完成 H→D 拷贝后
               → commit_hicache_transfer(LOAD_BACK)
               → 将 device_indices 按 token 段切片写入 nodes_to_load 中各节点的 value,
                 更新 evictable_size + _update_evictable_leaf_sets (节点从 evicted 恢复)。

        ⚙️ LOAD_BACK 切片: 按 nodes_to_load 顺序, 每个节点分得 len(host_value) 个 token,
            从 device_indices[offset:offset+n_len] 切片写入 cd.value。"""
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            if transfers and transfers[0].host_indices is not None:
                node.component_data[ct].host_value = transfers[0].host_indices.clone()

        elif phase == CacheTransferPhase.LOAD_BACK:
            if not transfers or transfers[0].device_indices is None:
                self.cache._update_evictable_leaf_sets(node)
                return

            xfer = transfers[0]
            device_indices = xfer.device_indices
            offset = 0
            for n in xfer.nodes_to_load or []:
                cd = n.component_data[ct]
                n_len = len(cd.host_value)
                cd.value = device_indices[offset : offset + n_len].clone()
                offset += n_len
                # Full uses leaf sets, not LRU
                self.cache.component_evictable_size_[ct] += n_len
                self.cache._update_evictable_leaf_sets(n)

            self.cache._update_evictable_leaf_sets(node)
