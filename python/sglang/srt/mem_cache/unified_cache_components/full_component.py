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
    ║  🔑 关键特征                                                                       ║
    ║    Lock: Full path-lock (从 node 一路锁到 root)                                    ║
    ║    驱逐: 用 evictable_device_leaves 叶子集合堆 (last_access_time 排序)              ║
    ║    split: 复制 lock_ref 到新 parent，切片 value/host_value                         ║
    ║    级联优先级: leaf=0, internal=2 (最高，最后被驱逐)                                 ║
    ║    validator: 要求 Full device data 非 None (或 HiCache 匹配时有 host backup)       ║
    ║                                                                                  ║
    ║  ⚠️ 若 SWA 存在，free_full 指向 full_attn_allocator.free(仅 free Full, SWA 由级联)  ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """
    component_type = ComponentType.FULL

    def __init__(self, cache, params):
        """📦 Full 组件初始化 —— 确定 free 策略和 HiCache Host Pool。

        ⚙️ free 策略:
            若 SWA 同时存在 → free_full = full_attn_allocator.free (仅 Free Full KV)
            否则          → free_full = allocator.free (释放全部 KV, 无 SWA)
            SWA 的 KV 由级联驱逐触发 SWAComponent.evict_component 释放。
        """
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

        🔗 UnifiedRadixCache._match_prefix_helper 每轮 match 只创建一次。
            match_device_only=True: 仅 device data 非 None (用于 scheduler best_match_device_node)
            match_device_only=False: device data 或 host backup 任意存在即为有效边界
        """
        if match_device_only:
            return (
                lambda node: node.component_data[self.component_type].value is not None
            )

        # HiCache: evicted + backuped nodes are valid match boundaries.
        return lambda node: (
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

        🔗 UnifiedRadixCache._match_post_processor 中调用。
            从 best_match_node 向上走到 last_device_node, 累加 evicted 节点的 host_value 长度。
            用于 HiCache: 告知 scheduler 需要从 Host load_back 多少 token。
        """
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

        🔗 UnifiedRadixCache._split_node 中在创建 new_parent 后调用。
            new_parent 得前 split_len 个 token 的 value/host_value clone。
            child 保留后半段。
        """
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

        🔗 _cascade_evict → _evict_component_and_detach_lru 中调用。
            DEVICE: free_full + 更新 evictable_size。⚠️ value=None 延迟到 _cascade_evict (SWA 的 free_swa 还需读 Full.value)。
            HOST:   free host pool + host_value=None。
        """
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
        """🗑️ 叶节点=0, 内部节点=2 (最高, 最后驱逐)。"""
        return 0 if is_leaf else 2

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        """🗑️ 从 evictable_device_leaves 堆驱逐直到满足 request token 数。

        🔗 UnifiedRadixCache.evict() 中遍历组件调用。
            堆键 = eviction_strategy.get_priority(n) (基于 last_access_time 的 LRU/FIFO 策略)。
            驱逐叶节点后, 若 parent 变成叶子则推入堆继续驱逐。
        """
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

        🔗 HostPoolGroup 在 Host 池满时调用。与 drive_eviction 结构对称, 操作 Host 端叶子。"""
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

        🔗 UnifiedRadixCache.inc_lock_ref 中调用。
            lock_host=True: 仅锁单节点的 host_lock_ref (只有最后一个 host node 需要保护)。
            否则: 跳过底部 evicted 段 → 从首个 device-on 节点开始逐祖先 +1。
            lock_ref 0→1 时 token 从 evictable_size 转 protected_size, 同时从 evictable_device_leaves 移除。
        """
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
        """🔓 Path-unlock: 从 node 沿 parent 一路解到 root (跳过 inc_lock 时跳过的 evicted 段)。

        🔗 UnifiedRadixCache.dec_lock_ref 中调用。
            lock_host=True: 仅解单节点的 host_lock_ref。
            否则: 跳过 skip_lock_node_ids 中记录的被 inc_lock 跳过的 evicted 段。
            lock_ref 1→0 时 token 转回 evictable_size, 节点可能加入 evictable_device_leaves。
        """
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

        🔗 HybridCacheController 在 BACKUP_HOST / LOAD_BACK / BACKUP_STORAGE / PREFETCH 阶段回调。
            BACKUP_HOST: Full KV 由主流程直接操作 host_value, 无需额外 PoolTransfer → None。
            LOAD_BACK:   从 best_match_node 向上收集 evicted 节点的 host_value → PoolTransfer(KV)。
        """
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

        🔗 HybridCacheController 在 DMA 完成后回调。
            BACKUP_HOST: 将 host_indices clone 到 node.component_data[FULL].host_value。
            LOAD_BACK:   将 device_indices 按 token 段切片写入各节点的 value, 更新 evictable_size 和叶子集合。
        """
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
