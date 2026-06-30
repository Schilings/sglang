from __future__ import annotations
import logging
import sys
import threading
import time
from array import array
from collections import defaultdict
from functools import partial
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Iterator, Optional, TypeVar

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.events import KVCacheEventMixin
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components import (
    _NUM_COMPONENT_TYPES,
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentData,
    ComponentType,
    EvictLayer,
    FullComponent,
    LRURefreshPhase,
    MambaComponent,
    SWAComponent,
    TreeComponent,
    get_and_increase_time_counter,
)
from sglang.srt.mem_cache.utils import (
    compute_node_hash_values,
    get_eviction_strategy,
    split_node_hash_value,
)
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)
from sglang.srt.session.streaming_session import StreamingSession

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        PrefetchOperation,
    )
    from sglang.srt.server_args import ServerArgs


T = TypeVar("T")


class UnifiedTreeNode:
    """🌳 统一 radix tree 节点 —— 每个节点独立存储各组件的数据。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🏷️ component_data: list[ComponentData] 按 ComponentType 枚举索引                   ║
    ║       component_data[FULL]   → Full 组件的 value/lock_ref/host_value               ║
    ║       component_data[SWA]    → SWA 组件的 value/lock_ref/host_value                ║
    ║       component_data[MAMBA]  → Mamba 组件的 value/lock_ref/host_value              ║
    ║                                                                                  ║
    ║  🗂️ lru_prev / lru_next: list 长度 = _NUM_COMPONENT_TYPES × 2                     ║
    ║       前半段 = 各 component device LRU 指针                                         ║
    ║       后半段(偏移 _NUM_COMPONENT_TYPES) = 各 component host LRU 指针                ║
    ║                                                                                  ║
    ║  🔑 关键属性:                                                                       ║
    ║    priority: int         叶子节点驱逐优先级 (last_access_time 派生)                   ║
    ║    hash_value            每页 SHA256 (HiCache/Storage 层用)                        ║
    ║    last_access_time      最后访问时间戳 (sanity check + 叶子集合堆排序)                ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """
    counter = 0

    def __init__(self, tree_components: tuple[ComponentType, ...], priority: int = 0):
        self.children = defaultdict(partial(UnifiedTreeNode, tree_components))
        self.parent: UnifiedTreeNode | None = None
        self.key: Optional[RadixKey] = None
        self.tree_components = tree_components
        # list indexed by ComponentType (int enum 0..N-1)
        self.component_data: list[ComponentData] = [
            ComponentData() for _ in range(_NUM_COMPONENT_TYPES)
        ]
        self.last_access_time = get_and_increase_time_counter()
        self.creation_time = get_and_increase_time_counter()
        self.hash_value = None
        self.hit_count = 0
        self.priority = priority
        self.lru_prev: list[UnifiedTreeNode | None] = [None] * (
            _NUM_COMPONENT_TYPES * 2
        )
        self.lru_next: list[UnifiedTreeNode | None] = [None] * (
            _NUM_COMPONENT_TYPES * 2
        )
        self.id = UnifiedTreeNode.counter
        UnifiedTreeNode.counter += 1
        self.write_through_pending_id: Optional[int] = None

    def component(self, component_type: ComponentType) -> ComponentData:
        return self.component_data[component_type]

    @property
    def backuped(self) -> bool:
        """Tree-level: Full KV present on host."""
        return self.component_data[ComponentType.FULL].host_value is not None

    @property
    def evicted(self) -> bool:
        """Tree-level: Full KV not on device (non-root with value=None)."""
        return (
            self.parent is not None
            and self.component_data[ComponentType.FULL].value is None
        )

    def __lt__(self, other: UnifiedTreeNode):
        return self.last_access_time < other.last_access_time

    def get_last_hash_value(self) -> Optional[str]:
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    def get_prefix_hash_values(self, node: UnifiedTreeNode) -> list[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value


class UnifiedLRUList:
    """🗂️ 统一 LRU 双向链表 —— 每个 TreeComponent 拥有自己的 device LRU + host LRU。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧩 对外接口清单                                                                   ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  insert_mru / remove_node / reset_node_mru            ✍️ 单节点增删 / 重置        ║
    ║  reset_node_and_parents_mru                          ✍️ 沿父链刷新 (Full/Mamba)  ║
    ║  reset_node_and_window_ancestors_mru                 ✍️ 窗口限制刷新 (SWA)        ║
    ║  in_list                                              📖 查询节点是否在 LRU       ║
    ║  get_prev_no_lock / get_lru_no_lock                  📖 跳过被锁节点取 LRU 端     ║
    ║  get_prev_leaf_no_lock / get_leaf_lru_no_lock        📖 仅取叶子 LRU 端           ║
    ║  get_prev_no_host_lock / get_lru_no_host_lock        📖 Host 层 LRU 端            ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔗 调用链 (谁使用本类)                                                            ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  UnifiedRadixCache.__init__()                                                       ║
    ║    └─ lru_lists[ct]       = UnifiedLRUList(ct, ...)        ← device LRU           ║
    ║    └─ host_lru_lists[ct]  = UnifiedLRUList(ct, ..., host)  ← host LRU             ║
    ║                                                                                   ║
    ║  UnifiedRadixCache._split_node() → _for_each_component_lru(remove_node/insert_mru)║
    ║  UnifiedRadixCache._evict_component_and_detach_lru() → in_list/remove_node        ║
    ║  TreeComponent.refresh_lru()  → reset_node_mru / reset_node_and_parents_mru       ║
    ║                                  / reset_node_and_window_ancestors_mru (SWA)       ║
    ║  TreeComponent.drive_eviction() → get_lru_no_lock / get_prev_no_lock              ║
    ║  TreeComponent.drive_host_eviction() → get_lru_no_host_lock / get_prev_no_host    ║
    ║  TreeComponent.acquire_component_lock() → in_list / remove_node                   ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧬 设计要点                                                                       ║
    ║                                                                                  ║
    ║  🔑 指针槽位设计: lru_prev/lru_next 是长度 N×2 的 list                              ║
    ║     device LRU: slot = component_type              (前半段)                       ║
    ║     host LRU:   slot = component_type + N          (后半段, use_host_ptr=True)    ║
    ║     这样同一节点的 device/host LRU 指针互不冲突, 一个节点可同时在多条 LRU 中。      ║
    ║                                                                                  ║
    ║  📐 head(dummy) ↔ MRU ... ↔ LRU ↔ tail(dummy)                                    ║
    ║     MRU 端在 head 后 (新节点从 head 后插入), LRU 端在 tail 前 (驱逐从 tail 取)。    ║
    ║                                                                                  ║
    ║  🗃️ cache: dict[node.id → node] 是 O(1) 成员查询索引, 与链表同步维护。              ║
    ║                                                                                  ║
    ║  ⚡ O(1) insert/remove/reset_mru | O(L) 驱逐扫描(L=跳过的被锁节点数)                 ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """
    def __init__(
        self,
        component_type: ComponentType,
        tree_components: tuple[ComponentType, ...],
        use_host_ptr: bool = False,
    ):
        """🆕 构造 —— 创建空的双向链表 + 选定本 LRU 在节点的指针槽位。

        📥 component_type: 本 LRU 追踪的组件类型 (FULL/SWA/MAMBA)。
        📥 tree_components: 树启用的组件类型元组 (用于构造 dummy 节点)。
        📥 use_host_ptr: True=Host LRU (用后半段槽位), False=Device LRU (前半段槽位)。"""
        self.component_type = component_type
        # Pointer slot: host LRU uses offset slots so device/host pointers
        # never collide on the same node.
        # 槽位计算: host LRU 用后半段 (offset _NUM_COMPONENT_TYPES), device 用前半段
        self._pt: int = component_type + (_NUM_COMPONENT_TYPES if use_host_ptr else 0)
        # dummy head/tail: 避免边界判空, 真正节点总是 head.next ... tail.prev
        self.head = UnifiedTreeNode(tree_components)
        self.tail = UnifiedTreeNode(tree_components)
        self.head.lru_next[self._pt] = self.tail
        self.tail.lru_prev[self._pt] = self.head
        # O(1) 成员索引: node.id → node (与链表节点同步维护)
        self.cache: dict[int, UnifiedTreeNode] = {}

    def _add_node_after(self, prev_node: UnifiedTreeNode, new_node: UnifiedTreeNode):
        """🔗 在 prev_node 之后插入 new_node —— 双向链表标准插入。

        ⚙️ 四步指针操作: prev ← new ← next; prev.next = new; next.prev = new。"""
        pt = self._pt
        new_node.lru_prev[pt] = prev_node
        new_node.lru_next[pt] = prev_node.lru_next[pt]
        prev_node.lru_next[pt].lru_prev[pt] = new_node
        prev_node.lru_next[pt] = new_node

    def _add_node(self, node: UnifiedTreeNode):
        """➕ 在 head 之后插入 node → 变为 MRU 端。"""
        self._add_node_after(self.head, node)

    def _remove_node(self, node: UnifiedTreeNode):
        """➖ 从链表中移除 node, 并清空其指针 (断引用环)。"""
        pt = self._pt
        node.lru_prev[pt].lru_next[pt] = node.lru_next[pt]
        node.lru_next[pt].lru_prev[pt] = node.lru_prev[pt]
        # Clear self pointers to break reference cycles among evicted nodes.
        # 清空自身指针: 被驱逐节点可能引用环 → 防 GC 泄漏
        node.lru_prev[pt] = None
        node.lru_next[pt] = None

    def insert_mru(self, node: UnifiedTreeNode):
        """✍️ 插入新节点到 MRU 端 (head 后)。

        🔗 _split_node / _for_each_component_lru / acquire_component_lock / evict_component 等调用。
        ⚠️  node 必须不在 cache 中 (assert 防止重复插入)。"""
        assert node.id not in self.cache
        self.cache[node.id] = node  # 同步更新成员索引
        self._add_node(node)

    def remove_node(self, node: UnifiedTreeNode):
        """✍️ 从 LRU 移除节点 —— 用于驱逐 / lock_ref 增到 1 / 节点分裂。

        🔗 _split_node / _evict_component_and_detach_lru / acquire_component_lock 等调用。
        ⚠️  node 必须在 cache 中。"""
        assert node.id in self.cache
        del self.cache[node.id]  # 同步删除成员索引
        self._remove_node(node)

    def reset_node_mru(self, node: UnifiedTreeNode):
        """🔄 将已存在的节点移到 MRU 端 (等价于 remove + insert)。

        🔗 TreeComponent.refresh_lru(WALKDOWN) 在路径遍历到节点时调用。"""
        assert node.id in self.cache
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(
        self,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
        should_include,
    ):
        """🔄 沿父链向上: 每个满足 should_include 的祖先重新置为 MRU (子孙比祖先更"新")。

        🔗 TreeComponent.refresh_lru(MATCH_END) 调用, Full/Mamba 用。

        ⚙️ 子孙的 prev_node 是其父亲, 因此插入顺序天然保证"子孙 MRU > 父亲 MRU"。
            这与 Full 组件用 last_access_time 递减的拓扑序保持一致。

        📥 should_include: 回调 (node) → bool, 决定该祖先是否参与刷新
              (例如只刷有 component value 的节点)。"""
        prev_node = self.head
        while node != root_node:
            if should_include(node):
                assert node.id in self.cache
                self._remove_node(node)
                # 在上一刷新节点 (或 head) 后插入: 子节点总是比父节点更 MRU
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def reset_node_and_window_ancestors_mru(
        self,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
        window_size: int,
        should_include,
    ):
        """🔄 滑动窗口受限的父链刷新 —— 累计 key 长度达到 window_size 即停止。

        🔗 SWAComponent.refresh_lru(MATCH_END/INSERT_END) 专用。
            SWA 只需保护滑动窗口内的祖先, 之外的祖先应保持可驱逐。

        ⚙️ 与 reset_node_and_parents_mru 的唯一差别:
            额外累加 len(node.key), 达到 window_size 时停止向上 (避免刷新窗口外的祖先)。

        📥 window_size: sliding_window_size + page_size (caller 传入, 多 1 页缓冲)。"""
        prev_node = self.head
        accumulated = 0  # 累计刷新路径上的 token 数
        while node != root_node and accumulated < window_size:
            if should_include(node):
                assert node.id in self.cache
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            accumulated += len(node.key)  # 累加 key 长度 (无论是否 include)
            node = node.parent

    def in_list(self, node: Optional[UnifiedTreeNode]):
        """📖 节点是否在本 LRU 中 (O(1) dict 查询)。"""
        return node is not None and node.id in self.cache

    def get_prev_no_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        """📖 从 node 向 LRU 端走, 跳过 lock_ref>0 的节点, 返回第一个可驱逐节点。

        🔗 TreeComponent.drive_eviction() 用于在 LRU 中找下一个驱逐目标。

        ⚙️ 跳过被锁节点: lock_ref>0 的节点不能驱逐 (正在被 request 使用)。
            若遍历到 head (dummy) → 返回 None (无可驱逐节点)。

        📥 node: 起点节点。
        📥 check_id: True=断言 node 在 cache 中 (驱逐循环外部调用可设 False)。"""
        if check_id:
            assert node.id in self.cache
        pt = self._pt
        ct = self.component_type
        x = node.lru_prev[pt]
        # 跳过 lock_ref>0 的节点 (正在被某 request 锁定, 不可驱逐)
        while x.component_data[ct].lock_ref > 0:
            x = x.lru_prev[pt]
        if x == self.head:
            return None  # 全部被锁 → 无可驱逐
        return x

    def get_prev_leaf_no_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        """📖 从 node 向 LRU 端走, 跳过被锁节点和非叶子, 返回第一个可驱逐叶子。

        🔗 Full 组件 drive_eviction() 用 (Full 只驱逐叶子, 内部不 tombstone)。

        ⚙️ 比 get_prev_no_lock 多一个判断: len(x.children) > 0 的非叶子也跳过。"""
        if check_id:
            assert node.id in self.cache
        pt = self._pt
        ct = self.component_type
        x = node.lru_prev[pt]
        # 跳过被锁节点 + 非叶子节点 (Full 只驱逐叶子)
        while x.component_data[ct].lock_ref > 0 or len(x.children) > 0:
            x = x.lru_prev[pt]
        if x == self.head:
            return None
        return x

    def get_prev_no_host_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        """📖 Host-LRU 版的 get_prev_no_lock —— 跳过 host_lock_ref>0 的节点。

        🔗 TreeComponent.drive_host_eviction() 用 (Host 层驱逐)。"""
        # Host-LRU walker: skip nodes whose component host_lock_ref > 0.
        if check_id:
            assert node.id in self.cache
        pt = self._pt
        ct = self.component_type
        x = node.lru_prev[pt]
        while x.component_data[ct].host_lock_ref > 0:
            x = x.lru_prev[pt]
        if x == self.head:
            return None
        return x

    def get_lru_no_lock(self):
        """📖 取 LRU 端第一个可驱逐节点 (从 tail 向 head 走, 跳过 lock_ref>0)。"""
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self):
        """📖 取 LRU 端第一个可驱逐叶子节点。"""
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)

    def get_lru_no_host_lock(self):
        """📖 取 Host LRU 端第一个可驱逐节点 (跳过 host_lock_ref>0)。"""
        return self.get_prev_no_host_lock(self.tail, check_id=False)


COMPONENT_REGISTRY: dict[ComponentType, type[TreeComponent]] = {
    ComponentType.FULL: FullComponent,
    ComponentType.MAMBA: MambaComponent,
    ComponentType.SWA: SWAComponent,
}

logger = logging.getLogger(__name__)


"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  🌳 Unified Radix Cache —— 基于组件的统一前缀缓存框架                                  ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║  将 Full / SWA / Mamba 三种缓存统一到一棵 radix tree 中，通过可插拔的 TreeComponent     ║
║  hook 接口实现各组件独立的锁、驱逐、匹配逻辑。                                           ║
║                                                                                      ║
║  🔗 从 Scheduler 出发的完整调用链                                                       ║
║                                                                                      ║
║  Scheduler.get_next_batch_to_run() → match_prefix(key)                                ║
║    ├─ session.try_match_prefix()  ← 流式会话 shortcut                                ║
║    └─ _match_prefix_helper()      ← 遍历 radix tree + 每个 component 的 validator    ║
║         ├─ 每节点调 all 组件 create_match_validator() 闭包                              ║
║         └─ _match_post_processor() → 刷新 LRU + finalize_match_result()              ║
║                                                                                      ║
║  Scheduler.process_batch_result() → cache_unfinished_req() / cache_finished_req()     ║
║    ├─ prepare_for_caching_req() 每组件                                                  ║
║    ├─ insert() → _insert_helper() → update_on_overlap / commit_insert / split        ║
║    ├─ dec_lock_ref(old) + inc_lock_ref(new)                                           ║
║    └─ cleanup_after_caching_req() 每组件                                                ║
║                                                                                      ║
║  evict() → drive_eviction() 每个组件按自己策略驱逐 → _cascade_evict 级联低优先级组件       ║
║                                                                                      ║
║  组件 hook 接口详见 tree_component.py / README-zh.md                                    ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""
class UnifiedRadixCache(KVCacheEventMixin, BasePrefixCache):
    """🌳 统一 Radix Cache —— 基于可插拔 TreeComponent 的多缓存类型前缀缓存框架。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧩 对外接口清单                                                                   ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                  ║
    ║  match_prefix(key)       → 🔍 前缀匹配，所有组件 validator 都通过才推进              ║
    ║  insert(key, value, ...) → 📝 插入 KV 入树，复用前缀 + tombstone 修复 + 释放重复     ║
    ║  evict(params)           → 🗑️ 各组件按自己策略驱动驱逐，级联低优先级组件               ║
    ║  inc_lock_ref(node)      → 🔒 锁定节点路径 (Full path-lock, SWA window-lock)        ║
    ║  dec_lock_ref(node)      → 🔓 解锁节点路径，对称于 inc_lock_ref                      ║
    ║  cache_finished_req(req) → 💾 请求完成后 KV 入树 + 释放锁                            ║
    ║  cache_unfinished_req(req)→ 🚧 未完成请求 KV 入树 + re-match + 锁交换                  ║
    ║                                                                                  ║
    ║  ════════════ 🔗 从 Scheduler 出发的宏观调用链 ════════════                          ║
    ║                                                                                  ║
    ║  Scheduler.get_next_batch_to_run()                                                ║
    ║    └─ Req.init_batch_info() → match_prefix(key)                                   ║
    ║         ├─ session.try_match_prefix()    ← 流式会话 shortcut                       ║
    ║         └─ _match_prefix_helper(key)     ← 从 root 遍历 tree                       ║
    ║              ├─ 每节点调 all 组件 create_match_validator() 闭包                     ║
    ║              ├─ 遇部分匹配 → _split_node() → redistribute_on_node_split()          ║
    ║              └─ _match_post_processor() → 刷新 LRU + finalize_match_result()      ║
    ║    └─ inc_lock_ref(last_node) → acquire_component_lock() 每组件                     ║
    ║    └─ alloc / evict → evict() → drive_eviction() → _cascade_evict()               ║
    ║                                                                                  ║
    ║  Scheduler.process_batch_result()                                                 ║
    ║    └─ cache_unfinished_req(req)             ← prefill 后若未完成 / chunked 暂存      ║
    ║         ├─ prepare_for_caching_req() 每组件                                        ║
    ║         ├─ insert() → _insert_helper() → update_on_overlap / commit / split       ║
    ║         ├─ re-match prefix → 写回 req_to_token_pool                               ║
    ║         └─ dec_lock_ref(old) + inc_lock_ref(new) → 锁交换                          ║
    ║    └─ cache_finished_req(req)               ← 请求完成                              ║
    ║         └─ insert() + dec_lock_ref() + cleanup_after_caching_req()                 ║
    ║                                                                                  ║
    ║  ════════════ 🧬 核心数据结构 ════════════                                          ║
    ║                                                                                  ║
    ║  UnifiedTreeNode: 每个节点 component_data[ct] = ComponentData(value, lock_ref)     ║
    ║  UnifiedLRUList:  每个 component 独立的 device/host LRU 双向链表                    ║
    ║  TreeComponent:   组件 hook 接口 (create_match_validator, evict_component, ...)    ║
    ║  COMPONENT_REGISTRY: {FULL→FullComp, SWA→SWAComp, MAMBA→MambaComp}               ║
    ║                                                                                  ║
    ║  ════════════ 🔑 关键设计 ════════════                                              ║
    ║                                                                                  ║
    ║  1. 树只操作 key (逻辑层)，所有物理资源管理由组件 hook 完成                            ║
    ║  2. 级联驱逐：Full(2) > SWA(1) > Mamba(0)，驱逐时同步清理低优先级组件                  ║
    ║  3. Full 驱逐用叶子集合堆 (last_access_time)，SWA/Mamba 用各自 LRU                  ║
    ║  4. 流式会话: session.try_* 方法支持流式解码的前缀匹配                                 ║
    ║                                                                                  ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """
    def __init__(
        self,
        params: CacheInitParams,
    ):
        # ── ① 基础 pool / page 配置 ──
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable = params.disable
        self.is_eagle = params.is_eagle
        # KV cache 事件追踪 (disaggregation 场景)
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.kv_event_queue = []
        self.eviction_policy = params.eviction_policy.lower()
        self.eviction_strategy = get_eviction_strategy(self.eviction_policy)

        # ── ② 设备信息 ──
        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        # ── ③ 指标收集 ──
        if params.enable_metrics:
            self.init_metrics_collector()
        self._enable_metrics_flag = params.enable_metrics
        self.enable_storage_metrics = False
        self.storage_metrics_collector: Optional[StorageMetricsCollector] = None
        self.extra_metric_labels = None

        # ── ④ 组件注册: 按 tree_components 从 COMPONENT_REGISTRY 实例化各组件 ──
        assert params.tree_components is not None
        self.tree_components = tuple(params.tree_components)
        component_registry = COMPONENT_REGISTRY
        if params.component_registry_override:
            # 允许用户覆盖默认组件类 (如自定义 SWA 实现)
            component_registry = {
                **COMPONENT_REGISTRY,
                **params.component_registry_override,
            }
        # components[ct] = 组件实例; _components_tuple = 元组形式 (遍历用)
        self.components: dict[ComponentType, TreeComponent] = {
            ct: component_registry[ct](self, params) for ct in self.tree_components
        }
        self._components_tuple: tuple[TreeComponent, ...] = tuple(
            self.components.values()
        )
        self.sidecar_pool_specs: list[SidecarPoolSpec] = []

        # ── ⑤ 流式会话: 始终启用, 非流式请求零开销 (try_* 短路) ──
        # Streaming session: embedded StreamingSession with self as inner.
        # Always on -- zero overhead when no streaming session is open (the
        # try_* entries short-circuit on non-streaming reqs / real TreeNodes).
        # Dispatch methods below pre-check conditions so the session's
        # internal fall-through to self.inner.xxx never fires -- no recursion.
        self.session = StreamingSession(inner=self)

        # ── ⑥ 分布式通信组 (TP / CP / PP) ──
        self.tp_group = params.tp_cache_group
        self.attn_cp_group = params.attn_cp_cache_group
        self.attn_tp_group = params.attn_tp_cache_group
        self.pp_group = params.pp_cache_group
        self.tp_world_size = (
            1
            if self.tp_group is None
            else torch.distributed.get_world_size(group=self.tp_group)
        )
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.work_list: list[torch.distributed.Work] = []  # 异步 send work 列表

        # ── ⑦ HiCache D↔H 默认参数 (init_hicache 覆盖) ──
        self.cache_controller: Optional[HybridCacheController] = None
        self.write_through_threshold = 256  # write-through: 命中 N 次后触发 D→H backup
        self.prefetch_stop_policy = "best_effort"
        self.prefetch_threshold = 256       # prefetch 最小 token 数
        self.prefetch_timeout_base = 1.0
        self.prefetch_timeout_per_page = 0.25
        self.hicache_storage_pass_prefix_keys = False

        # ── ⑧ 初始化树状态 (root_node / LRU / 叶子集合 等) ──
        self.reset()
        logger.info(f"Init Unified RadixTree with components {self.tree_components}")

    def _all_reduce_attn_groups(self, tensor: torch.Tensor, op):
        reduced = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.all_reduce(tensor, op=op, group=group)
                reduced = True
        if not reduced and self.tp_world_size > 1:
            torch.distributed.all_reduce(tensor, op=op, group=self.tp_group)

    def _barrier_attn_groups(self):
        waited = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.barrier(group=group)
                waited = True
        if not waited and self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)

    def _reap_completed_async_work(self):
        """
        Poll outstanding async work and reap completed ones.

        Must be called in the scheduler thread.
        """
        count = 0
        while count < len(self.work_list) and self.work_list[count].is_completed():
            count += 1
        if count > 0:
            logger.debug(f"Reap {count} completed async work")
            self.work_list = self.work_list[count:]

    def _all_reduce(self, data: torch.Tensor, tp_reduce_op: torch.distributed.ReduceOp):
        """
        Synchronize data across all TP and PP ranks.

        In particular, "tp_reduce_op" is performed on all TP ranks of the first PP rank,
        and then the result is propagated to all following PP ranks.

        Must be called in the scheduler thread.
        """
        if self.pp_rank == 0:
            self._all_reduce_attn_groups(data, tp_reduce_op)
        self._pp_sync(data)

    def _pp_sync(self, data: torch.Tensor) -> None:
        """
        Synchronize data across the PP pipeline, where PPn (n>0) will receive PP0's data.
        """
        if self.pp_size <= 1 or self.pp_group is None:
            return
        if self.pp_rank > 0:
            torch.distributed.recv(
                data, group_src=self.pp_rank - 1, group=self.pp_group, tag=2
            )
        if self.pp_rank + 1 < self.pp_size:
            copy_of_data = data.clone()
            send_work = torch.distributed.isend(
                copy_of_data, group_dst=self.pp_rank + 1, group=self.pp_group, tag=2
            )
            self.work_list.append(send_work)

    def reset(self) -> None:
        self._reset_full()

    def _reset_full(self) -> None:
        """Full reset: destroy entire tree and all state."""
        # ── ① 创建 root_node: priority=-maxsize 保证 root 永远不被驱逐 ──
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.root_node.priority = -sys.maxsize
        self.root_node.key = RadixKey(array("q"), None)
        self.root_node.component_data[BASE_COMPONENT_TYPE].value = []
        self.root_node.hash_value = []
        # root 的每个组件 lock_ref=1: 永久锁定, 防止 root 被误驱逐
        for ct in self.tree_components:
            self.root_node.component_data[ct].lock_ref = 1

        # ── ② 组件级 evictable / protected 计数器 ──
        self.component_evictable_size_ = {ct: 0 for ct in self.tree_components}
        self.component_protected_size_ = {ct: 0 for ct in self.tree_components}

        # ── ③ Device LRU 链表 (每个辅组件一条) + 流式会话 slot 清空 ──
        self.lru_lists = {
            ct: UnifiedLRUList(ct, self.tree_components) for ct in self.tree_components
        }
        self.session.slots.clear()

        # ── ④ 可驱逐叶子集合 (Full 用, 基于 last_access_time 堆排序) ──
        self.evictable_device_leaves: set[UnifiedTreeNode] = set()
        self.evictable_host_leaves: set[UnifiedTreeNode] = set()
        # ── ⑤ Host LRU 链表 (HiCache 用, use_host_ptr=True) ──
        self.host_lru_lists = {
            ct: UnifiedLRUList(ct, self.tree_components, use_host_ptr=True)
            for ct in self.tree_components
        }

        # ── ⑥ HiCache 异步操作追踪表 ──
        # ongoing_write_through: {ack_id → (node, lock_params, publish_nodes)}
        self.ongoing_write_through: dict[
            int,
            tuple[
                UnifiedTreeNode,
                Optional[DecLockRefParams],
                list[UnifiedTreeNode],
            ],
        ] = {}
        # ongoing_load_back: {node.id → (node, device_lock_params, host_lock_params)}
        self.ongoing_load_back: dict[
            int,
            tuple[UnifiedTreeNode, DecLockRefParams, DecLockRefParams],
        ] = {}
        self.enable_storage = False
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        # ongoing_prefetch: {req_id → (host_node, key, host_indices, op, lock_params, comp_xfers)}
        self.ongoing_prefetch: dict[
            str,
            tuple[
                UnifiedTreeNode,
                RadixKey,
                torch.Tensor,
                PrefetchOperation,
                DecLockRefParams,
                dict[ComponentType, list[PoolTransfer]],
            ],
        ] = {}
        # ongoing_backup: {operation_id → (node, host_lock_params)}
        self.ongoing_backup: dict[int, tuple[UnifiedTreeNode, DecLockRefParams]] = {}

        # ── ⑦ 重置 cache_controller (HiCache 控制器) ──
        if self.cache_controller is not None:
            self.cache_controller.reset()
            self.cache_controller.mem_pool_host.clear()
            self.enable_storage = self.cache_controller.enable_storage

        # ── ⑧ 空 MatchResult 缓存 (避免每次 match 空结果都创建新 tensor) ──
        self._empty_match_result = MatchResult(
            device_indices=torch.empty(
                (0,),
                dtype=torch.int64,
                device=self.device,
            ),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
            best_match_node=self.root_node,
        )
        self._record_all_cleared_event()

    def init_hicache(self, server_args: ServerArgs, params: CacheInitParams) -> None:
        """Initialize HiCache infrastructure."""
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            attach_hybrid_pool_to_unified_cache,
        )

        # ── ① Direct IO 布局修正: direct IO 不支持 page_first, 自动切 page_first_direct ──
        if server_args.hicache_io_backend == "direct":
            if server_args.hicache_mem_layout == "page_first":
                server_args.hicache_mem_layout = "page_first_direct"
                logger.warning(
                    "Page first layout is not supported with direct IO backend, "
                    "switching to page first direct layout"
                )

        # ── ② HiCache 事件 + sidecar pool + 指标标签 ──
        self.load_cache_event = threading.Event()
        self.sidecar_pool_specs.clear()
        self.extra_metric_labels = server_args.extra_metric_labels

        # ── ③ 解析 storage 配置 (prefetch_threshold / timeout / prefix_keys) ──
        storage_backend = server_args.hicache_storage_backend
        storage_extra_config = None
        storage_prefetch_threshold = 256
        prefetch_timeout_base = 1.0
        prefetch_timeout_per_ki_token = 0.25
        hicache_storage_pass_prefix_keys = False
        if storage_backend is not None:
            (
                storage_extra_config,
                storage_prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = HybridCacheController.parse_storage_backend_extra_config(
                server_args.hicache_storage_backend_extra_config
            )

        # ── ④ 组装 hybrid pool (Host pool + Storage backend) 并挂载到本 cache ──
        attach_hybrid_pool_to_unified_cache(
            self,
            params,
            server_args,
            load_cache_event=self.load_cache_event,
            attn_cp_group=params.attn_cp_cache_group,
            attn_tp_group=params.attn_tp_cache_group,
            storage_backend=storage_backend,
            storage_extra_config=storage_extra_config,
            storage_prefetch_threshold=storage_prefetch_threshold,
        )

        # ── ⑤ HiCache 策略参数 ──
        # write_through: 命中 1 次即 backup; write_back: 命中 2 次后才 backup (驱逐时触发)
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10  # H→D load_back 的最小 token 数
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        # ── ⑥ 若启用 Storage, 应用 runtime 配置 (prefetch 参数 + 指标) ──
        if storage_backend is not None:
            self._apply_storage_runtime_config(
                storage_backend=storage_backend,
                prefetch_threshold=storage_prefetch_threshold,
                prefetch_timeout_base=prefetch_timeout_base,
                prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
                enable_storage=self.cache_controller.enable_storage,
                enable_storage_metrics=self._enable_metrics_flag,
                extra_metric_labels=self.extra_metric_labels,
            )

    def register_sidecar_pool(self, spec: SidecarPoolSpec) -> None:
        self.sidecar_pool_specs.append(spec)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """🔍 前缀匹配 —— 在 radix tree 中查找 key 的最长公共前缀。

        🔗 Scheduler.get_next_batch_to_run() → Req.init_batch_info() → match_prefix(key)。
            这是每个请求最先调用的入口之一。

        ⚙️ 流程:
            ① session.try_match_prefix() — 流式会话 shortcut (正在解码的 session)。
            ② key 预处理: bigram 视图 (Eagle 模式) + page 对齐。
            ③ _match_prefix_helper(key) — 从 root 遍历树, 逐 component validator 判定。
            ④ _match_post_processor() — 刷新 LRU、last_access_time、构造 MatchResult。

        📥 params: MatchPrefixParams (含 key=RadixKey)。
        📤 MatchResult: 含 device_indices (匹配到的 KV slot 索引)、
                          last_device_node (设备端锚点)、last_host_node (Host 端锚点)、
                          swa_host_hit_length (SWA 需 load_back 的 token 数)。"""
        # ① 流式会话 shortcut: 正在解码的 session 直接返回缓存结果
        result = self.session.try_match_prefix(params)
        if result is not None:
            return result

        # ② key 预处理: bigram 视图 (Eagle 投机解码) + page 对齐
        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if self.disable or len(key) == 0:
            return self._empty_match_result
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return self._empty_match_result

        # ③ 树遍历 + 后处理
        (
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
        ) = self._match_prefix_helper(key)
        return self._match_post_processor(
            params,
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        """📝 插入 KV 入树 —— 复用已有前缀, 释放重叠 KV slot, 可选创建新叶子。

        🔗 cache_finished_req() (:837) / cache_unfinished_req() (:915) 调用。

        ⚙️ 流程:
            ① key 预处理: bigram 视图 (Eagle) + page 对齐。
            ② 若 value 为 None, 用 key 的 token_ids 做 value (占位)。
            ③ _insert_helper(root, key, value, params) — 递归遍历+插入。
                内部让各 component 处理 overlap/unevict/commit。

        📥 params: InsertParams (含 key, value, prev_prefix_len, swa_evicted_seqlen 等)。
        📤 InsertResult: prefix_len=插入后匹配的总 token 数。"""
        if self.disable:
            return InsertResult(prefix_len=0)

        # ① key/value 预处理: bigram 视图 + page 对齐
        key = params.key
        value = params.value
        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]  # 截断到 page 对齐长度
        else:
            # value=None: 用 token_ids 做占位 (仅测 key 匹配, 不真正分配 KV)
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        # ② 递归插入: 从 root 开始遍历+匹配+分裂+创建叶子
        result = self._insert_helper(self.root_node, key, value, params)
        return result

    def evict(self, params: EvictParams) -> EvictResult:
        """🗑️ 驱逐 KV 缓存 —— 各组件按自己策略从 LRU 驱逐节点直到满足空间需求。

        🔗 alloc_token_slots() / alloc_paged_token_slots_extend() (:307 / :363 在 common.py)
            当 KV pool 空闲空间不足时调 tree_cache.evict()。

        ⚙️ 流程:
            ① 初始化 tracker {ComponentType → 0}, 跨组件共享驱逐计数。
            ② 遍历每个 component.drive_eviction(params, tracker):
                 - Full: 从叶子集合堆驱逐 (last_access_time 排序)
                 - SWA:  从 SWA LRU 驱逐 (叶子全删 / 内部 tombstone)
                 - Mamba: 从 Full LRU 驱逐
                各组件内部调用 _cascade_evict 级联清理低优先级组件。
            ③ write_back 策略下触发 writing_check (确保驱逐的节点 D→H 已备份)。
            ④ 收集指标。

        📥 params: EvictParams (含 num_tokens, swa_num_tokens, mamba_num)。
        📤 EvictResult: 各组件驱逐的 token 数。"""
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        # tracker: 跨组件共享的驱逐计数 {ComponentType → 已驱逐 token 数}
        tracker = {ct: 0 for ct in self.tree_components}

        # ① 各组件独立驱动驱逐: Full 从叶子集合, SWA/Mamba 从各自 LRU
        for component in self._components_tuple:
            component.drive_eviction(params=params, tracker=tracker)

        # ② write_back 策略: 驱逐后检查是否有 pending D→H 写回需要完成
        if (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        ):
            self.writing_check(write_back=True)

        # ③ 指标收集
        self.update_eviction_metrics(sum(tracker.values()), start_time)
        return EvictResult(
            num_tokens_evicted=tracker[BASE_COMPONENT_TYPE],
            swa_num_tokens_evicted=tracker.get(ComponentType.SWA, 0),
            mamba_num_evicted=tracker.get(ComponentType.MAMBA, 0),
        )

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        """🔒 锁定节点路径 —— 保护匹配路径上的 KV 不被驱逐。

        🔗 Scheduler 在每个 request init 时调用, 锁定 match 返回的 last_node。

        ⚙️ 遍历所有 component.acquire_component_lock(node):
             - Full: path-lock (沿树向上锁全部祖先)
             - SWA:  window-lock (向上累计 sliding_window_size 后停止)
             - Mamba: single-node lock (只锁节点自身)
            后更新 _update_evictable_leaf_sets (locked 节点从可驱逐集合移除)。

        📥 node: 待锁定的节点 (匹配锚点)。
        📤 IncLockRefResult: 含 delta (lock 前后 protected 差值), swa_uuid_for_lock。"""
        # 流式会话 shortcut
        result = self.session.try_inc_lock_ref(node)
        if result is not None:
            return result
        if self.disable:
            return IncLockRefResult()
        # 遍历所有组件: Full path-lock / SWA window-lock / Mamba single-node
        result = IncLockRefResult()
        for component in self._components_tuple:
            result = component.acquire_component_lock(node=node, result=result)

        # locked 节点从可驱逐集合移除
        self._update_evictable_leaf_sets(node)
        return result

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        """🔓 解锁节点路径 —— 对称于 inc_lock_ref。

        🔗 请求完成/中间缓存时调用, 释放对旧匹配路径的保护。

        ⚙️ 遍历 component.release_component_lock(node, params):
             与 inc 相反, lock_ref 递减, 1→0 时恢复可驱逐状态。
             SWA 在 UUID 边界停止 (只释放窗口覆盖的部分)。"""
        # 流式会话 shortcut
        result = self.session.try_dec_lock_ref(node, params)
        if result is not None:
            return result
        if self.disable:
            return DecLockRefResult()
        # 遍历所有组件释放锁
        for component in self._components_tuple:
            component.release_component_lock(node=node, params=params)

        # unlocked 节点重新加入可驱逐集合
        self._update_evictable_leaf_sets(node)
        # TODO: delta is not aggregated from components; no caller uses it yet.
        return DecLockRefResult()

    def inc_host_lock_ref(self, node: Any) -> IncLockRefResult:
        """🔒 Host 层锁定 —— 保护 Host KV 不被 host eviction 驱逐 (HiCache 用)。

        🔗 write_backup / load_back / prefetch_from_storage 内部调用。"""
        if self.disable:
            return IncLockRefResult()
        result = IncLockRefResult()
        # 遍历组件: lock_host=True → 操作 host_lock_ref 而非 device lock_ref
        for component in self._components_tuple:
            result = component.acquire_component_lock(
                node=node, result=result, lock_host=True
            )

        self._update_evictable_leaf_sets(node)
        return result

    def dec_host_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        """🔓 Host 层解锁 —— 对称于 inc_host_lock_ref。"""
        if self.disable:
            return DecLockRefResult()
        for component in self._components_tuple:
            component.release_component_lock(node=node, params=params, lock_host=True)

        self._update_evictable_leaf_sets(node)
        return DecLockRefResult()

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs) -> None:
        """💾 已完成请求的 KV 缓存 —— insert 入树 + 释放锁 + 清理组件。

        🔗 Scheduler.process_batch_result() 对每个完成的请求调用。

        ⚙️ 流程:
            ① session.try_cache_finished_req() — 流式会话 shortcut。
            ② 若 is_insert=True:
                 - 收集 token_ids + kv_indices
                 - 遍历 component.prepare_for_caching_req() 收集 effective_cache_len
                 - 按 page 对齐后 insert() 入树
            ③ dec_lock_ref(last_node) 释放旧锁。
            ④ 遍历 component.cleanup_after_caching_req() 清理各组件资源。

        📥 req: 已完成生成的请求。
        📥 is_insert: 是否将 KV 插入 radix tree (skip_radix_cache_insert 时为 False)。"""
        # ① 流式会话 shortcut
        if self.session.try_cache_finished_req(req, is_insert=is_insert, **kwargs):
            return

        kv_committed_len = req.pop_committed_kv_cache()

        # ② disable 路径: 直接 free KV + cleanup
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(req, is_finished=True)
            return

        # ③ 收集 token_ids 和 KV indices (从 req_to_token_pool 取值)
        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        result = None
        insert_params = None

        if is_insert:
            insert_params = InsertParams(
                prev_prefix_len=req.cache_protected_len,
                priority=getattr(req, "priority", 0) or 0,
            )

            # ④ 各组件 prepare: 返回 effective_cache_len (可能截断)
            effective_cache_len = len(token_ids)
            for comp in self._components_tuple:
                cl = comp.prepare_for_caching_req(
                    req=req,
                    insert_params=insert_params,
                    token_ids_len=len(token_ids),
                    is_finished=True,
                )
                if cl is not None:
                    effective_cache_len = min(effective_cache_len, cl)

            # ⑤ 截断: 若组件缩了缓存长度 → free 多余 KV slot
            if effective_cache_len < len(token_ids):
                free_start = max(effective_cache_len, req.cache_protected_len)
                self.token_to_kv_pool_allocator.free(kv_indices[free_start:])
                token_ids = token_ids[:effective_cache_len]
                kv_indices = kv_indices[:effective_cache_len]

            # ⑥ page 对齐后 insert 入树
            radix_key = RadixKey(
                token_ids, req.extra_key, is_bigram=self.is_eagle
            ).page_aligned(self.page_size)
            page_aligned_len = len(radix_key)
            values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)

            insert_params.key = radix_key
            insert_params.value = values
            result = self.insert(insert_params)

            # ⑦ 释放未对齐的尾部 KV (page 外, 不能入树)
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            # is_insert=False: 直接释放受保护的区间之外的 KV
            self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len :])

        # ⑧ 释放旧匹配路径的锁
        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )

        # ⑨ 各组件清理
        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(
                req, is_finished=True, insert_result=result, insert_params=insert_params
            )

    def cache_unfinished_req(self, req: Req, chunked: bool = False, **kwargs) -> None:
        """🚧 未完成请求的 KV 缓存 —— insert + re-match + 锁交换。

        🔗 两个调用点:
            ① batch_result_processor.process_batch_result_prefill()
               — 任何 prefill 完成后若请求未结束 (req.finished()==False) 就调用。
               decode 阶段不调用 (避免每 decode 1 token 就 insert 的浪费)。
            ② scheduler.stash_chunked_request()
               — chunked prefill 暂存时调用, 传 chunked=True (不增 hit_count,
               防止同一请求在多个 chunk 中虚增)。

        ⚙️ 流程:
            ① 收集 token_ids + kv_indices。
            ② 遍历 component.prepare_for_caching_req() (is_finished=False)。
            ③ free_out_of_window_slots() 释放 SWA 窗口外的旧 slot。
            ④ insert() 入树。
            ⑤ 用相同的 key 做 match_prefix(), 获取新的匹配路径。
            ⑥ 将新 indices 写回 req_to_token_pool。
            ⑦ 锁交换: dec_lock_ref(old_node) + inc_lock_ref(new_node)。
            ⑧ 更新 req.last_node, req.swa_uuid_for_lock 等字段。
            ⑨ component.cleanup_after_caching_req() 清理。

        📥 req: 未完成的请求 (prefill 后但生成未结束)。
        📥 chunked: True=chunked prefill 暂存场景 (不增 hit_count)。"""
        # ① 流式会话 shortcut
        if self.session.try_cache_unfinished_req(req, chunked=chunked, **kwargs):
            return

        token_ids = req.get_fill_ids()

        # ② disable 路径: 直接用 kv_indices 做 prefix_indices
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(token_ids)
            ]
            req.prefix_indices = kv_indices
            return

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # ③ 各组件 prepare + 收集 effective_cache_len (可能截断)
        insert_params = InsertParams(
            prev_prefix_len=req.cache_protected_len,
            chunked=chunked,
            priority=getattr(req, "priority", 0) or 0,
        )
        effective_cache_len = len(token_ids)
        for comp in self._components_tuple:
            # 如果是finished请求，记录 SWA 窗口驱逐边界swa_evicted_seqlen → insert 时判断 tombstone 重叠
            # full/swa正常prefill什么也不做
            cl = comp.prepare_for_caching_req(
                req=req,
                insert_params=insert_params,
                token_ids_len=len(token_ids),
                is_finished=False,
            )
            if cl is not None:
                effective_cache_len = min(effective_cache_len, cl)

        # ④ SWA 窗口释放: decoder 前移产生的旧 SWA token
        if envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get():
            for comp in self._components_tuple:
                comp.free_out_of_window_slots(
                    req, effective_cache_len - 1, insert_params
                )

        # effective_cache_len <= 0 → 无有效缓存, 直接返回
        if effective_cache_len <= 0:
            req.prefix_indices = kv_indices_orig.to(dtype=torch.int64, copy=True)
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(
                    req, is_finished=False, insert_params=insert_params
                )
            return

        kv_indices = kv_indices_orig[:effective_cache_len]

        # ⑤ page 对齐后 insert 入树
        radix_key = RadixKey(
            token_ids[:effective_cache_len],
            req.extra_key,
            is_bigram=self.is_eagle,
        ).page_aligned(self.page_size)
        page_aligned_len = len(radix_key)
        values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)

        insert_params.key = radix_key
        insert_params.value = values
        result = self.insert(insert_params)

        # ⑥ re-match: 用相同 key 查询刚插入的树, 获取新的匹配路径
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        new_prefix_len = result.prefix_len
        # cache_protected_len 不应超出重新匹配的范围
        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ), f"{req.cache_protected_len=}, {len(new_indices)=}, {page_aligned_len=}"
        assert new_prefix_len <= len(
            new_indices
        ), f"{new_prefix_len=}, {len(new_indices)=}"

        # ⑦ 将新的匹配 indices 写回 req_to_token_pool
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # ⑧ 锁交换: 释放旧 last_node 的保护, 锁上新匹配节点
        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )
        lock_result = self.inc_lock_ref(new_last_node)

        # ⑨ 更新 req 字段: indices, last_node, swa_uuid
        if len(new_indices) < len(kv_indices_orig):
            # 新匹配 < 原始 kv: 拼接尾部 (在树外但仍在 req 中)
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices_orig[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.cache_protected_len = len(new_indices)
        req.last_node = new_last_node
        req.swa_uuid_for_lock = lock_result.swa_uuid_for_lock

        # ⑩ 各组件清理
        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(
                req,
                is_finished=False,
                insert_result=result,
                insert_params=insert_params,
            )

    # ---- Internal Helpers ----

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> tuple[list[torch.Tensor], UnifiedTreeNode, UnifiedTreeNode, int]:
        """🔍 从 root 遍历 radix tree 做前缀匹配 —— 核心匹配引擎。

        🔗 match_prefix() 内部调用 (:678)。

        ⚙️ 行为:
            ① 创建 validators: 遍历每个 component.create_match_validator()。
                HiCache 模式 (separate_device_match=True): 创建两套 validator
                  - validators: 允许 host_value 作为有效匹配 (用于 best_match_node)
                  - device_validators: 只认 device value (用于 best_match_device_node)
            ② 从 root 向下遍历, 逐个 child 匹配 prefix:
                  - 部分匹配 → split node → 停止
                  - 完全匹配 → 收集 Full value + _update_best_if_valid
                  - 遇 evicted+not backuped 节点 → 停止 (HiCache 断点)
            ③ 返回 (value_list, best_match_node, best_match_device_node, best_match_device_value_len)。

        📥 key: page-aligned 的 RadixKey。
        📤 四元组: Full value chunks 列表, 最佳匹配节点 (含 Host),
                   最佳设备匹配节点, 设备端匹配长度。"""
        # Non-HiCache mode has only device-resident matches, so the scheduler
        # device anchor follows the best match. In HiCache mode, host-backed
        # nodes can also match, so we separately track the best device-resident
        # match for scheduler prefix indices and locking.
        node = self.root_node
        child_key = key.child_key(self.page_size)  # 取首 page 的 child key, 进入第一层
        value: list[torch.Tensor] = []  # 收集匹配路径上各节点的 Full device value
        best_match_node = node           # 最佳匹配 (含 Host 端, HiCache 用)
        best_match_device_node = node    # 最佳设备匹配 (调度器用, 用于 prefix_indices + lock)
        best_match_device_value_len = 0  # 设备匹配长度 = len(value)
        separate_device_match = self.cache_controller is not None  # HiCache 启用时两套 validator
        if separate_device_match:
            # HiCache: host_value 也算有效 → 用于 best_match_node (含 Host 的匹配节点)
            validators = tuple(
                comp.create_match_validator() for comp in self._components_tuple
            )
            # device_validators: 只认 device value → 用于 best_match_device_node
            device_validators = tuple(
                comp.create_match_validator(match_device_only=True)
                for comp in self._components_tuple
            )
        else:
            # 无 HiCache: 只一套 validator, host/device 合一
            validators = tuple(
                comp.create_match_validator(match_device_only=True)
                for comp in self._components_tuple
            )

        def _all_valid(validators, node):
            """所有组件 validator 都通过 → True"""
            return all([v(node) for v in validators])

        def _update_best_if_valid(node):
            """若所有 validator 通过, 更新 best_match_node / best_match_device_node"""
            nonlocal best_match_node
            nonlocal best_match_device_value_len, best_match_device_node
            matched = _all_valid(validators, node)
            if matched:
                best_match_node = node  # Host/Device 合一或 host 也有效 → best_match_node

            if not separate_device_match:
                # 无 HiCache: host/device 合一, matched 就是 device match
                if matched:
                    # 节点数
                    best_match_device_value_len = len(value)
                    best_match_device_node = node
                return

            # HiCache 模式: 额外检查纯设备端 validator
            # HiCache模式的确实是Host Node链要比Device Node链长！
            if _all_valid(device_validators, node):
                best_match_device_value_len = len(value)
                best_match_device_node = node

        # ② 从 root 向叶遍历 key
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]

            # HiCache: dead node (evicted + not backuped) — stop traversal
            # 设备驱逐了且 Host 也没备份 → 树在此断裂
            if child.evicted and not child.backuped:
                break

            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                # 部分匹配: key 与 child.key 只重合了 prefix_len → split 后停止
                node = self._split_node(child.key, child, prefix_len)
                # value对应的是device上的可用indices，只考虑Full情况，没删就是有
                if not node.evicted:
                    value.append(node.component_data[BASE_COMPONENT_TYPE].value)
                #
                _update_best_if_valid(node)
                break

            # 完全匹配: key 覆盖了 child 的全部
            if not child.evicted:
                value.append(child.component_data[BASE_COMPONENT_TYPE].value)
            node = child
            _update_best_if_valid(node)
            key = key[prefix_len:]  # 截去已匹配的前缀
            if len(key):
                child_key = key.child_key(self.page_size)  # 下一 page 的 child key

        return (
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
        )

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: list[torch.Tensor],
        best_match_node: UnifiedTreeNode,
        best_match_device_node: UnifiedTreeNode,
        best_match_device_value_len: int,
    ) -> MatchResult:
        """🔍 匹配后处理 —— 刷新 LRU / last_access_time / 构造 MatchResult。

        🔗 _match_prefix_helper() 返回值直接传入, match_prefix() 末尾调用 (:684)。

        ⚙️ 流程:
            ① LRU 刷新: 辅组件 (SWA/Mamba) 刷新 MATCH_END; Full 用 last_access_time。
            ② last_access_time: 沿匹配路径向上更新, 每级递减 0.00001 (确保祖先比后代"更旧")。
            ③ last_host_node: HiCache 模式下向上找到最近的 backuped 节点; 否则用 device node。
            ④ 拼接 device_indices = cat(value[:best_match_device_value_len])。
            ⑤ 构造 MatchResult + 遍历 component.finalize_match_result() (如 SWA 统计 host_hit_length)。"""
        node_update = best_match_node
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue  # Full uses last_access_time, not LRU
            comp.refresh_lru(LRURefreshPhase.MATCH_END, node_update, self.root_node)

        # 沿路径向上更新 last_access_time: 子孙比祖先更"新"
        cur_time = get_and_increase_time_counter()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= 0.00001  # 递减保证拓扑序
            node_update = node_update.parent

        # Walk up to find last_host_node for full component.
        if self.cache_controller is None:
            last_host_node = best_match_device_node
        else:
            # HiCache: 向上找到最近已 backup 的节点
            last_host_node = best_match_node
            while last_host_node is not self.root_node and not last_host_node.backuped:
                last_host_node = last_host_node.parent

        # 拼接设备端 value chunks 为连续 tensor
        if best_match_device_value_len > 0:
            device_indices = torch.cat(value[:best_match_device_value_len])
        else:
            device_indices = self._empty_match_result.device_indices
        result = MatchResult(
            device_indices=device_indices,
            last_device_node=best_match_device_node,
            last_host_node=last_host_node,
            best_match_node=best_match_node,
            host_hit_length=0,
        )

        # Full: 标记host_hit_length
        # SWA: 标记 swa_host_hit_length
        for component in self._components_tuple:
            result = component.finalize_match_result(
                result=result,
                params=params,
                value_chunks=value,
                best_value_len=best_match_device_value_len,
            )
        return result

    def _split_node(
        self, key: RadixKey, child: UnifiedTreeNode, split_len: int
    ) -> UnifiedTreeNode:
        """✂️ 在 split_len 处分裂节点 —— 创建 new_parent 并重分配 child。

        🔗 _match_prefix_helper() / _insert_helper() / 各 component 内部调用。

        ⚙️ 分裂操作:
            ① 创建 new_node (new_parent), 继承 child 的 parent/priority/hit_count。
            ② child 变为 new_node 的子节点, key 截断为 [split_len:]。
            ③ 调用各 component.redistribute_on_node_split() 重分配数据。
            ④ 将 new_node 插入原 parent 的 children + 推入各辅组件 LRU。
            ⑤ split hash_value (用于 HiCache Storage)。
            ⑥ 若有 pending write-through, 替换引用 (_replace_pending_write_through_node)。"""
        # ① 创建 new_node (作为 new_parent, 在 split_len 处截断)
        new_node = UnifiedTreeNode(self.tree_components, priority=child.priority)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent  # 继承原 parent
        new_node.key = child.key[:split_len]  # new_node 持有前半段 key
        new_node.hit_count = child.hit_count
        new_node.creation_time = child.creation_time

        # ② 从辅组件 LRU 中移除 child (分裂后需要重新插入)
        self._for_each_component_lru(child, UnifiedLRUList.remove_node)

        # ③ child 变为 new_node 的子节点: key 截断为 [split_len:]
        child.parent = new_node
        child.key = child.key[split_len:]
        # 分裂 hash_value: child 原有 hash 按 split_len 切分给两个节点
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        # ④ 各 component 重分配组件数据 (value/host_value/lock_ref/UUID)
        for component in self._components_tuple:
            component.redistribute_on_node_split(new_parent=new_node, child=child)
        # 将 new_node 接入原 parent 的 children
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        # ⑤ HiCache write-through: 若 child 有 pending backup, 替换引用
        if child.backuped:
            self._replace_pending_write_through_node(child, [new_node, child])

        # ⑥ 将 new_node 和 child 重新插入辅组件 LRU (MRU 位置)
        self._for_each_component_lru(
            new_node, UnifiedLRUList.insert_mru, skip_existing=True
        )
        self._for_each_component_lru(
            child, UnifiedLRUList.insert_mru, skip_existing=True
        )
        child.last_access_time = get_and_increase_time_counter()

        # ⑦ 更新驱逐叶子集合 (new_node/child 可能变成叶子)
        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(child)
        return new_node

    def _touch_node(self, node: UnifiedTreeNode):
        """👆 触碰节点 —— 更新 last_access_time + 刷新辅组件 LRU (WALKDOWN)。

        🔗 _insert_helper() 和 _match_prefix_helper() 遍历树时在每个节点调用。"""
        node.last_access_time = get_and_increase_time_counter()
        if node != self.root_node:
            for comp in self._components_tuple:
                if comp.component_type == BASE_COMPONENT_TYPE:
                    continue  # Full uses last_access_time, not LRU
                comp.refresh_lru(LRURefreshPhase.WALKDOWN, node, self.root_node)

    def _add_new_node(
        self,
        parent: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        priority: int = 0,
    ) -> UnifiedTreeNode:
        """🌿 创建新叶子节点 —— 分配 Full value + 计算 hash + 更新叶子集合。

        🔗 _insert_helper() 在无重叠后缀时调用。"""
        new_node = UnifiedTreeNode(self.tree_components, priority=priority)
        new_node.parent = parent
        new_node.key = key
        new_node.component_data[BASE_COMPONENT_TYPE].value = value.clone()
        parent.children[key.child_key(self.page_size)] = new_node
        self.component_evictable_size_[BASE_COMPONENT_TYPE] += len(value)
        if self.enable_storage:
            new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(parent)
        self._record_store_event(new_node)
        return new_node

    def _unevict_node_on_insert(
        self, node: UnifiedTreeNode, fresh_value: torch.Tensor
    ) -> None:
        """🔄 恢复已驱逐节点的 Full device value —— insert 时复用新 KV indices。

        🔗 _insert_helper() 遇 evicted 节点时调用 (:1195)。"""
        # Restore an evicted node's Full device value from fresh KV indices
        # during insert.
        ct = BASE_COMPONENT_TYPE
        cd = node.component_data[ct]
        assert cd.value is None
        n = len(fresh_value)
        cd.value = fresh_value.clone()
        self.component_evictable_size_[ct] += n
        self._update_evictable_leaf_sets(node)
        if node.parent is not None:
            self._update_evictable_leaf_sets(node.parent)
        self._record_store_event(node, medium=StorageMedium.GPU)

    def _insert_helper(
        self,
        node: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        params: InsertParams,
    ) -> InsertResult:
        """📝 Insert 核心引擎 —— 递归匹配+插入, 处理 overlap/unevict/split/叶子创建。

        🔗 insert() 直接调用 (:705)。

        ⚙️ 三重循环:
            ① 遍历已有节点 (while child_key in children):
                 - partial match → _split_node()
                 - node.evicted → _unevict_node_on_insert() + component.recover_after_unevict()
                 - 否则 → component.update_component_on_insert_overlap() 处理重叠 KV
                 - free 重复的旧 KV slot (dup_start:consumed_from)
                 - _inc_hit_count() 计数
            ② 若有剩余 key:
                 - component.should_skip_leaf_creation() 检查 (SWA 全在窗口外则跳过)
                 - 否则 _add_new_node() 创建叶子
            ③ Finalize:
                 - component.commit_insert_component_data() (SWA: 分配 value + split)
                 - component.refresh_lru(INSERT_END) 刷新辅组件 LRU

        📥 node: 起始节点 (通常 root)。
        📥 key: page-aligned RadixKey。
        📥 value: KV pool 索引。
        📥 params: InsertParams (含 prev_prefix_len, chunked, swa_evicted_seqlen)。
        📤 InsertResult: prefix_len=总匹配长度。"""
        priority = params.priority
        if priority is None:
            priority = 0
        self._touch_node(node)  # 更新当前节点的访问时间
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return InsertResult(prefix_len=0, mamba_exist=True)

        # ① 遍历已有子节点: 匹配 + 处理 overlap
        child_key = key.child_key(self.page_size)
        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            self._touch_node(node)
            prefix_len = node.key.match(key, page_size=self.page_size)
            if prefix_len < len(node.key):
                # 部分匹配: 需要 split 节点 (匹配只覆盖了 child 的前部分)
                node = self._split_node(node.key, node, prefix_len)
            node.priority = max(node.priority, priority)

            if node.evicted:
                # ── 分支 A: 节点曾被驱逐 (evicted=True) ──
                # 用新 KV 恢复 Full device value; 辅组件可能有 tombstone 需重建
                self._unevict_node_on_insert(node, value[:prefix_len])
                for component in self._components_tuple:
                    if component.component_type == BASE_COMPONENT_TYPE:
                        continue
                    component.recover_after_unevict(
                        node=node,
                        prefix_len=prefix_len,
                        total_prefix_len=total_prefix_length,
                        params=params,
                    )
            else:
                # ── 分支 B: 节点存活 (not evicted) ──
                # insert 路径上的已有节点与新 key 重叠: 各组件声明对重叠 KV slot 的所有权。
                #
                # 🔗 解耦: _insert_helper 只做通用树遍历 (匹配/split/创建叶子),
                #    重叠时各组件如何处理自己的数据由组件 hook 决定, 互不干扰。
                #
                # 各组件实现:
                #   Full/Mamba: 不 override, 基类默认 return prefix_len (不消费, 直接复用)
                #   SWA:        override, 三分支:
                #     - 窗口内 tombstone → 复活 (return 0=全消费, 不释放旧 slot)
                #     - 骑跨窗口边界   → 部分复活 (return start_idx=部分消费)
                #     - 窗口外 tombstone → 不消费 (return prefix_len, 保持 tombstone)
                value_slice = value[:prefix_len]
                consumed_from = prefix_len  # 默认: 整段被复用, 无重复 slot 需释放
                for component in self._components_tuple:
                    comp_consumed_from = component.update_component_on_insert_overlap(
                        node=node,
                        prefix_len=prefix_len,
                        total_prefix_len=total_prefix_length,
                        value_slice=value_slice,
                        params=params,
                    )
                    # 取最小值: 任一组件消费了某个前缀位置 → 该位置之前的 slot 已被复用, 不可释放
                    consumed_from = min(consumed_from, comp_consumed_from)

                # 释放重复 KV: [dup_start, consumed_from) 范围的 slot
                #   - 既有节点已有这些 KV, 新 insert 又分配了相同位置的 slot → 重复
                #   - dup_start: 不受 prev_prefix_len 保护的重叠起点
                #     (prev_prefix_len 之前的 slot 已被上次 insert 保护, 不能 free)
                #   - consumed_from 之后的 slot 被某组件消费 (复用), 也不能 free
                #   - 仅 [dup_start, consumed_from) 之间的 slot 是"重复且无人消费" → 可释放
                dup_start = max(0, params.prev_prefix_len - total_prefix_length)
                if dup_start < consumed_from:
                    self.token_to_kv_pool_allocator.free(
                        value_slice[dup_start:consumed_from]
                    )

            self._inc_hit_count(node, params.chunked)  # 命中计数 → 可能触发 write_through backup
            total_prefix_length += prefix_len
            key = key[prefix_len:]      # 截去已匹配前缀
            value = value[prefix_len:]  # 对应截去 value
            if len(key):
                child_key = key.child_key(self.page_size)  # 下一 page

        # ② 处理剩余 key 后缀: 可能创建新叶子
        is_new_leaf = False
        if len(key):
            # 任一组件拒绝叶子创建 → 放弃 (free value 后返回)
            # 实现: 仅 SWAComponent override (Full/Mamba 用基类默认 return False)
            #
            # swa_evicted_seqlen: 请求序列中 SWA KV 已释放的前缀长度 ([0, swa_evicted_seqlen)
            #   的 SWA KV 已因滑出窗口被释放回 pool)。随 decode 单调递增,
            #   由 maybe_evict_swa() 更新: evict_threshold = pre_len - sliding_window_size - page_size
            #
            # SWA 跳过条件: swa_evicted_seqlen >= total_prefix_len + key_len
            #   → 整叶在滑动窗口左边界之外 → 对 SWA 是纯 tombstone → 跳过
            #   正常情况不会触发 (-page_size 缓冲保护尾部), 仅为防御性检查。
            #   可能的边缘场景: disagg PD 分离 / 开启 SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN
            if any(
                comp.should_skip_leaf_creation(
                    total_prefix_len=total_prefix_length,
                    key_len=len(key),
                    params=params,
                )
                for comp in self._components_tuple
            ):
                self.token_to_kv_pool_allocator.free(value)  # 释放已分配但未使用的 KV slot
                return InsertResult(prefix_len=total_prefix_length)
            target_node = self._add_new_node(node, key, value, priority=priority)
            is_new_leaf = True
        else:
            target_node = node  # 完整匹配已有节点, 不需要创建叶子

        # Finalize: let each component attach its data to the target node.
        # e.g. Mamba attaches mamba_value to the leaf node
        result = InsertResult(prefix_len=total_prefix_length)
        for component in self._components_tuple:
            component.commit_insert_component_data(
                node=target_node,
                is_new_leaf=is_new_leaf,
                params=params,
                result=result,
            )

        if target_node is not self.root_node:
            for component in self._components_tuple:
                if component.component_type == BASE_COMPONENT_TYPE:
                    continue
                component.refresh_lru(
                    LRURefreshPhase.INSERT_END, target_node, self.root_node
                )

        if is_new_leaf:
            self._inc_hit_count(target_node, params.chunked)
        return result

    def _insert_helper_host(
        self,
        node: UnifiedTreeNode,
        key: RadixKey,
        host_value: torch.Tensor,
        hash_value: list[str],
    ) -> InsertResult:
        """📝 Host 树插入 —— 将 prefetch 的 host KV 插入 radix tree (仅 Host 层)。

        🔗 check_prefetch_progress() 在 prefetch 完成后调用。

        ⚙️ 与 _insert_helper 类似但只操作 host_value (不涉及 device value):
            ① 遍历已有节点匹配前缀
            ② 创建新叶子 (host_value + hash_value)
            ③ 返回 InsertResult (含 inserted_host_node 供 commit 用)。"""
        total_len = len(key)
        self._touch_node(node)
        if total_len == 0:
            return InsertResult(prefix_len=0, mamba_exist=True)

        # ① 遍历已有子节点匹配前缀
        child_key = key.child_key(self.page_size)
        matched_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            self._touch_node(node)
            prefix_len = node.key.match(key, page_size=self.page_size)

            # 截去已匹配部分 (key / host_value / hash_value 三者同步)
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                # 部分匹配 → split
                node = self._split_node(node.key, node, prefix_len)

            if len(key):
                child_key = key.child_key(self.page_size)

        result = InsertResult(prefix_len=matched_length, total_len=total_len)
        # ② key 全部匹配: 若当前节点已有 host_value → 返回该节点
        if len(key) == 0:
            if (
                node is not self.root_node
                and node.component_data[BASE_COMPONENT_TYPE].host_value is not None
            ):
                result.inserted_host_node = node
            return result

        # ③ 有剩余 key → 创建新 Host 叶子节点
        new_node = UnifiedTreeNode(self.tree_components, priority=node.priority)
        new_node.parent = node
        new_node.key = key
        new_node.hash_value = hash_value
        new_node.component_data[BASE_COMPONENT_TYPE].host_value = host_value.clone()
        node.children[child_key] = new_node
        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(node)
        result.inserted_host_node = new_node
        return result

    # ---- Evict Helpers ----

    def _cascade_evict(
        self,
        node: UnifiedTreeNode,
        trigger: TreeComponent,
        tracker: dict[ComponentType, int],
        target: EvictLayer = EvictLayer.DEVICE,
    ):
        """⛓️ 级联驱逐 —— 驱逐 trigger 组件时同步清理 ≤ 其优先级的其他组件数据。

        🔗 各 component 的 drive_eviction() 在驱逐后调用本方法。

        ⚙️ 优先级体系: Full(2) > SWA(1) > Mamba(0)
            触发时把同节点上优先级 ≤ trigger 的其他组件也驱逐。
            例如: 驱逐 SWA(1) 时级联驱逐 Mamba(0), 但不动 Full(2)。
            叶子节点优先级均为 0, 任一组件的 leaf eviction 都级联到所有组件。

        ⚠️ Full 驱逐的特殊延迟: free_swa() 需要 Full.value 做索引映射,
            所以 Full.value 的 tombstone (→ None) 延迟到级联驱逐之后执行。"""
        # Cascade eviction from trigger to lower-or-equal priority components.

        is_leaf = False
        if target == EvictLayer.DEVICE:
            is_leaf = node in self.evictable_device_leaves
        elif target == EvictLayer.HOST:
            is_leaf = node in self.evictable_host_leaves

        trigger_priority = trigger.eviction_priority(is_leaf)

        # 遍历组件, 驱逐优先级 ≤ trigger 的
        for comp in self._components_tuple:
            if comp.eviction_priority(is_leaf) <= trigger_priority:
                if comp is not trigger and comp.node_has_component_data(node, target):
                    cd = node.component_data[comp.component_type]
                    if EvictLayer.DEVICE in target:
                        assert cd.lock_ref == 0
                    if EvictLayer.HOST in target:
                        assert cd.host_lock_ref == 0
                    self._evict_component_and_detach_lru(
                        node, comp, target=target, tracker=tracker
                    )

        # Now that all components (including SWA which depends on Full.value)
        # have been freed, we can safely tombstone Full.value.
        # This is deferred from evict_component because free_swa needs it.
        if (
            target is EvictLayer.DEVICE
            and trigger.component_type == BASE_COMPONENT_TYPE
        ):
            node.component_data[trigger.component_type].value = None

        self._update_evictable_leaf_sets(node)

    def _remove_leaf_from_parent(self, node: UnifiedTreeNode):
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node

    def _evict_component_and_detach_lru(
        self,
        node: UnifiedTreeNode,
        comp: TreeComponent,
        target: EvictLayer = EvictLayer.DEVICE,
        tracker: Optional[dict[ComponentType, int]] = None,
    ) -> tuple[int, int]:
        """🗑️ 驱逐单组件数据 + 从 LRU 移除 —— 驱逐流水线的原子步骤。

        🔗 _cascade_evict / _evict_device_leaf / _evict_host_leaf 调用。

        ⚙️ ① comp.evict_component() 释放 KV pool slot (返回 freed token 数)。
            ② 累加到 tracker (跨组件共享计数)。
            ③ 从对应 LRU 链表移除 (device LRU 或 host LRU)。"""
        # ① 释放组件 KV 资源 (free pool slot, tombstone value)
        device_freed, host_freed = comp.evict_component(node, target=target)
        # ② 累加到 tracker
        if tracker is not None:
            if EvictLayer.DEVICE in target:
                tracker[comp.component_type] += device_freed
            elif EvictLayer.HOST in target:
                tracker[comp.component_type] += host_freed

        # ③ 从 LRU 链表移除: device LRU 和 host LRU 分别检查
        # Detach from the appropriate LRU list(s)
        ct = comp.component_type
        for layer, lru_lists in (
            (EvictLayer.DEVICE, self.lru_lists),
            (EvictLayer.HOST, self.host_lru_lists),
        ):
            if layer in target:
                lru = lru_lists[ct]
                if lru.in_list(node):
                    lru.remove_node(node)
        return device_freed, host_freed

    def _iteratively_delete_tombstone_leaf(
        self, deleted_node: UnifiedTreeNode, tracker: dict[ComponentType, int]
    ):
        """Walk up from *deleted_node* and cascade-delete childless ancestors.

        Only the Full (base) component decides whether a node survives:
          - Full device present  → keep as D-leaf
          - Full host present    → keep as H-leaf
          - neither              → evict all remaining data, delete, continue up
        """
        ct = BASE_COMPONENT_TYPE
        cur = deleted_node.parent
        # 从被删叶子向上遍历: 只要祖先已经没有子节点了, 就级联检查是否该删
        while cur != self.root_node and len(cur.children) == 0:
            # 有锁 → 不删
            if any(
                cd.lock_ref > 0 or cd.host_lock_ref > 0 for cd in cur.component_data
            ):
                break

            has_device = cur.component_data[ct].value is not None
            has_host = cur.component_data[ct].host_value is not None

            if has_device:
                # Full device 还在 → 保留为 D-leaf, 停止向上
                self._update_evictable_leaf_sets(cur)
                break

            # Full device 已驱逐 → 清理这个祖先上的辅组件 device 残留
            for comp in self.components.values():
                if comp.node_has_component_data(cur):
                    self._evict_component_and_detach_lru(
                        cur, comp, target=EvictLayer.DEVICE, tracker=tracker
                    )

            if has_host:
                # Full host 还在 → 保留为 H-leaf, 停止向上
                self._update_evictable_leaf_sets(cur)
                break

            # Full 两层都无 → 清理 Host 残留 + 删除节点, 继续向上
            for comp in self.components.values():
                if comp.node_has_component_data(cur, target=EvictLayer.HOST):
                    self._evict_component_and_detach_lru(
                        cur, comp, target=EvictLayer.HOST, tracker=tracker
                    )

            self.evictable_host_leaves.discard(cur)
            self._remove_leaf_from_parent(cur)  # 从 parent.children 移除
            parent = cur.parent
            self._update_evictable_leaf_sets(parent)  # parent 可能变成叶子
            cur = parent  # 继续向上

    def _for_each_component_lru(
        self,
        node: UnifiedTreeNode,
        lru_op,
        target: EvictLayer = EvictLayer.DEVICE,
        skip_existing: bool = False,
    ):
        """Apply lru_op to each aux component's LRU that has data on this node.
        If skip_existing=True, skip components already in the target LRU list.

        🔁 对节点上每个"辅助组件"(SWA/Mamba 等) 的 LRU 链表统一应用一个操作。

        🔗 调用链定位:
            ├─ _split_node()    → 分裂前 remove_node; 分裂后 insert_mru (×2, new_node+child)
            └─ _evict_to_host() → device 驱逐后, 辅组件 host_value 入 Host LRU

            本函数不直接驱逐, 仅作为"对节点各辅组件 LRU 批量施法"的统一入口;
            真正的 LRU 操作由调用方通过 lru_op 传入 (UnifiedLRUList.remove_node /
            insert_mru 这类未绑定方法, 签名 lru_op(lru, node))。

        📥 node          : 要操作的树节点。
        📥 lru_op        : 对 LRU 施加的操作 (未绑定方法, 第一参数 lru, 第二参数 node)。
        📥 target        : DEVICE → 用 self.lru_lists 并看 cd.value;
                           HOST   → 用 self.host_lru_lists 并看 cd.host_value。
        📥 skip_existing : True 时跳过已在该 LRU 中的组件 (split 后重插时避免重复入链)。

        ⚙️ 行为:
            ① 选目标层 LRU 字典 (Device / Host)。
            ② 遍历 tree_components, 跳过 Full (Full 用 leaf sets 管驱逐, 不走 LRU)。
            ③ 仅对"该层有数据"的辅组件施法 —— 无数据的组件不在 LRU 中, 无需操作。
            ④ skip_existing 时用 lru.in_list(node) 去重。

        ⚠️ Full 组件必须跳过: Full 的驱逐由 evictable_device_leaves / evictable_host_leaves
            叶子集合驱动, 不维护 LRU 链; 若误对 Full 施法会破坏叶子集合与 LRU 的一致性。"""
        # ① 按 target 选 LRU 字典: Host 层用 host_lru_lists, Device 层用 lru_lists
        lru_dict = self.host_lru_lists if target is EvictLayer.HOST else self.lru_lists
        # ② 遍历树启用的所有组件类型 (FULL / SWA / Mamba ...)
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue  # Full uses leaf sets, not LRU
            cd = node.component_data[ct]
            # ③ 仅处理"该层有数据"的辅组件: 无数据则不在 LRU 中, 施法无意义
            if (cd.host_value if target is EvictLayer.HOST else cd.value) is not None:
                lru = lru_dict[ct]
                # ④ skip_existing: split 后重插场景, 节点可能已在 LRU 中, 跳过避免重复
                if skip_existing and lru.in_list(node):
                    continue
                lru_op(lru, node)

    def evict_host(
        self, num_tokens: int, component_type: ComponentType = BASE_COMPONENT_TYPE
    ) -> int:
        """Evict host resources for a specific component to free host pool space."""
        tracker: dict[ComponentType, int] = {ct: 0 for ct in self.tree_components}
        comp = self.components.get(component_type)
        if comp is not None:
            comp.drive_host_eviction(num_tokens, tracker)
        return tracker[component_type]

    def _is_device_leaf(self, node: UnifiedTreeNode) -> bool:
        """D-leaf: Full device value present, no child with Full KV on device,
        unlocked, not root.

        Only the Full (base) component is required; auxiliary components
        (Mamba, SWA) are not mandatory for D-leaf membership."""
        ct = BASE_COMPONENT_TYPE
        # root 或已驱逐 → 不是 D-leaf
        if node is self.root_node or node.evicted:
            return False
        # 任一组件被锁 → 不可驱逐 → 不是 D-leaf
        if any(cd.lock_ref > 0 for cd in node.component_data):
            return False
        # 有子节点在 device 上有 Full KV → 不是叶子 (还有可驱逐的更深层叶子)
        if any(
            child.component_data[ct].value is not None
            for child in node.children.values()
        ):
            return False
        return True

    def _is_host_leaf(self, node: UnifiedTreeNode) -> bool:
        """H-leaf: evicted, Full host value present, no children, unlocked, not root.

        Only the Full (base) component host_value is required; auxiliary
        components are not mandatory for H-leaf membership."""
        # root 或未驱逐 → 不是 H-leaf (H-leaf 必须是 device 已驱逐的)
        if node is self.root_node or not node.evicted:
            return False
        # 未 backup → 没有 host 数据 → 不是 H-leaf
        if not node.backuped:
            return False
        # 任一组件 host 被锁 → 不可驱逐
        if any(cd.host_lock_ref > 0 for cd in node.component_data):
            return False
        # 有子节点 → 不是叶子
        if len(node.children) > 0:
            return False
        return True

    def _update_evictable_leaf_sets(self, node: UnifiedTreeNode) -> None:
        """Update both device and host leaf sets for a node."""
        if self._is_device_leaf(node):
            self.evictable_device_leaves.add(node)
        else:
            self.evictable_device_leaves.discard(node)

        if self._is_host_leaf(node):
            self.evictable_host_leaves.add(node)
        else:
            self.evictable_host_leaves.discard(node)

    def _evict_to_host(
        self, node: UnifiedTreeNode, tracker: Optional[dict[ComponentType, int]] = None
    ) -> None:
        """GPU→CPU demotion: release all device resources, node stays in tree."""
        # 前提: 节点未驱逐 + 已 backup (Host 有数据)
        assert not node.evicted and node.backuped
        # ① 驱逐 Full device value (Full 是 trigger, 优先级最高)
        trigger = self.components[BASE_COMPONENT_TYPE]
        self._evict_component_and_detach_lru(
            node, trigger, target=EvictLayer.DEVICE, tracker=tracker
        )
        # ② 级联驱逐辅组件 device 数据 (SWA/Mamba)
        self._cascade_evict(node, trigger, tracker)
        self._record_remove_event(node, medium=StorageMedium.GPU)

        # ③ device 驱逐后: 辅组件的 host_value 仍在 → 插入 Host LRU (可被 host eviction)
        # after device eviction, insert aux components into host LRU.
        self._for_each_component_lru(
            node, UnifiedLRUList.insert_mru, target=EvictLayer.HOST, skip_existing=True
        )
        # parent 可能因 node 变成 evicted 而改变叶子状态
        self._update_evictable_leaf_sets(node.parent)

    def _evict_device_leaf(
        self, node: UnifiedTreeNode, tracker: dict[ComponentType, int]
    ) -> None:
        """Evict a device leaf node, choosing the right strategy:

        - backuped: demote to host via _evict_to_host (node stays in tree)
        - not backuped + write_back: write_backup first, then demote
        - not backuped + write_through: Cascade evict all components

        All freed device tokens are accumulated into *tracker*.
        """
        assert self._is_device_leaf(node), f"node {node.id} is not a D-leaf"
        if not node.backuped:
            # ── 分支 1: write_back 策略 —— 先 D→H 备份, 再降级到 Host ──
            if (
                self.cache_controller is not None
                and self.cache_controller.write_policy == "write_back"
            ):
                written = self.write_backup(node, write_back=True)
                if written == 0:
                    return  # 备份失败 → 不驱逐
                self.writing_check(write_back=True)
                self._evict_to_host(node, tracker)  # 降级: 保留 Host 数据, 释放 Device
                return
            else:
                # ── 分支 2: write_through 策略 —— 无备份, 整叶删除 ──
                self._record_remove_event(node, medium=StorageMedium.GPU)
                for comp in self._components_tuple:
                    self._evict_component_and_detach_lru(
                        node, comp, target=EvictLayer.ALL, tracker=tracker
                    )
                self.evictable_device_leaves.discard(node)
                parent = node.parent
                self._remove_leaf_from_parent(node)
                self._update_evictable_leaf_sets(parent)
                self._iteratively_delete_tombstone_leaf(node, tracker)  # 级联清理祖先
                return
        # ── 分支 3: 已 backup → 直接降级到 Host ──
        self._evict_to_host(node, tracker)

    def _evict_host_leaf(
        self, node: UnifiedTreeNode, tracker: dict[ComponentType, int]
    ) -> None:
        """Atomically evict all components on a host leaf.

        All freed tokens are accumulated into *tracker*."""
        assert self._is_host_leaf(node), f"node {node.id} is not an H-leaf"

        self._record_remove_event(node, medium=StorageMedium.CPU)
        # ① 驱逐所有组件的 Host 数据 (ALL = device + host, 但 H-leaf 已无 device)
        for comp in self._components_tuple:
            _, hf = self._evict_component_and_detach_lru(
                node, comp, target=EvictLayer.ALL, tracker=None
            )
            tracker[comp.component_type] += hf
        # ② 从可驱逐 Host 叶子集合移除
        self.evictable_host_leaves.discard(node)
        # ③ 从 parent.children 删除
        self._remove_leaf_from_parent(node)
        # ④ 级联向上清理无子节点的祖先
        self._iteratively_delete_tombstone_leaf(node, tracker)

    # ---- HiCache: Backup / LoadBack ----

    def write_backup(self, node: UnifiedTreeNode, write_back: bool = False) -> int:
        """💿 D→H 备份 —— 将节点的 device KV + 辅组件数据复制到 Host 池。

        🔗 _evict_device_leaf() (write_back 策略), load_back 用, _finish_write_through_ack。

        ⚙️ 流程:
            ① write-through 不变式: parent 必须先备份 (递归 write_backup)。
            ② 构造 PoolTransfer: Full KV + 各 component.build_hicache_transfers(BACKUP_HOST)。
            ③ 若 Host pool 不够, 先 evict_host 腾空间。
            ④ cache_controller.write() 执行 D→H 拷贝。
            ⑤ 各 component.commit_hicache_transfer(BACKUP_HOST) 记录 host_value。
            ⑥ 锁路径: 备份后 inc_lock_ref (write-through 保护)。

        📥 node: 待备份节点。
        📥 write_back: True=write-back 策略 (驱逐触发), False=write-through 策略。
        📤 备份的 host token 数 (0 表示失败)。"""
        if self.cache_controller is None:
            return 0

        # ① write-through 不变式: parent 必须先备份 (递归向上)
        if not write_back and (
            node.parent is not self.root_node and not node.parent.backuped
        ):
            if self.write_backup(node.parent) <= 0:
                return 0  # 父节点备份失败 → 放弃

        # ② 构造 Full KV 传输描述符
        device_value = node.component_data[BASE_COMPONENT_TYPE].value
        kv_xfer = PoolTransfer(name=PoolName.KV, device_indices=device_value)

        # ③ 各辅组件构造各自的 BACKUP_HOST 传输 (如 SWA 的 host_indices)
        comp_xfers: dict[ComponentType, list] = {}
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            t = comp.build_hicache_transfers(node, CacheTransferPhase.BACKUP_HOST)
            if t:
                comp_xfers[comp.component_type] = t
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_HOST, kv_xfer, comp_xfers
        )

        # ④ Host pool 不够 → 先驱逐 Host 腾空间
        kv_tokens = len(device_value)
        host_avail = self.cache_controller.mem_pool_host.available_size()
        if host_avail < kv_tokens:
            needed = kv_tokens - host_avail
            evicted = self.evict_host(needed)
            if evicted < needed:
                return 0  # 驱逐后仍不够 → 放弃

        # ⑤ 合并所有传输, 执行 D→H 拷贝
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        host_indices = self.cache_controller.write(
            device_value, node_id=node.id, extra_pools=aux_xfers or None
        )
        if host_indices is None:
            return 0  # 拷贝失败

        # ⑥ 各组件 commit: 将 host_indices 记录到 component_data.host_value
        kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)
        self.components[BASE_COMPONENT_TYPE].commit_hicache_transfer(
            node,
            CacheTransferPhase.BACKUP_HOST,
            transfers=[kv_xfer],
        )
        for ct, xfers in comp_xfers.items():
            self.components[ct].commit_hicache_transfer(
                node,
                CacheTransferPhase.BACKUP_HOST,
                transfers=xfers,
            )

        # ⑦ write_through 策略: 备份后 lock 路径 (防止被驱逐)
        lock_params = None
        if not write_back:
            lock_params = self.inc_lock_ref(node).to_dec_params()
        self._track_write_through_node(node, lock_params)
        return len(host_indices)

    def _track_write_through_node(
        self,
        node: UnifiedTreeNode,
        lock_params: Optional[DecLockRefParams],
    ) -> None:
        """📋 记录 write-through pending 状态 —— 备份完成后需 dec_lock_ref。"""
        node.write_through_pending_id = node.id
        # ongoing_write_through[ack_id] = (lock_node, lock_params, publish_nodes)
        # publish_nodes 初始为 [node], split 后可能变为 [new_parent, child]
        self.ongoing_write_through[node.id] = (node, lock_params, [node])

    def _replace_pending_write_through_node(
        self, old_node: UnifiedTreeNode, new_nodes: list[UnifiedTreeNode]
    ) -> None:
        """🔄 split 时替换 pending write-through 引用 —— old_node 拆成 new_nodes。"""
        ack_id = old_node.write_through_pending_id
        if ack_id is None:
            return  # 无 pending backup → 无需替换

        pending = self.ongoing_write_through.get(ack_id)
        if pending is None:
            return

        lock_node, lock_params, publish_nodes = pending
        # 在 publish_nodes 中用 new_nodes 替换 old_node
        updated_nodes = []
        replaced = False
        for node in publish_nodes:
            if node is old_node:
                updated_nodes.extend(new_nodes)
                replaced = True
            else:
                updated_nodes.append(node)

        if not replaced:
            return

        # 新节点继承 ack_id (后续 ack 回调时需要清理)
        for node in new_nodes:
            node.write_through_pending_id = ack_id
        self.ongoing_write_through[ack_id] = (
            lock_node,
            lock_params,
            updated_nodes,
        )

    def _finish_write_through_ack(self, ack_id: int) -> None:
        """✅ write-through D→H backup 完成 —— 释放锁 + 触发 H→Storage 备份。"""
        lock_node, lock_params, publish_nodes = self.ongoing_write_through.pop(ack_id)
        # 清理所有相关节点的 pending 标记 + 记录 CPU 存储事件
        for node in publish_nodes:
            if node.write_through_pending_id == ack_id:
                node.write_through_pending_id = None
            self._record_store_event(node, medium=StorageMedium.CPU)
        # 释放 write_through 时获取的 device lock
        if lock_params is not None:
            self.dec_lock_ref(lock_node, lock_params)
        # 若启用 Storage: 每个 fragment 都要 H→Storage 备份
        if self.enable_storage:
            # Back up each fragment: after a split, lock_node only holds the
            # suffix; the prefix fragment must be persisted as well.
            for node in publish_nodes:
                self.write_backup_storage(node)

    def load_back(
        self,
        best_match_node: UnifiedTreeNode,
        mem_quota: Optional[int] = None,
        req=None,
    ) -> bool:
        """💿 H→D 加载 —— 将匹配节点的 Host KV 数据加载回 Device 池。

        🔗 Scheduler 在 match 返回后, 若检测到 host_hit, 分配 token 前调用。

        ⚙️ 流程:
            ① inc_host_lock_ref(best_match_node) 锁定 Host 锚点。
            ② build_hicache_transfers(LOAD_BACK) 构造传输 (仅需 load 的部分)。
            ③ inc_lock_ref(best_match_node) 锁定设备路径 + pre-evict 腾空间。
            ④ cache_controller.load() 执行 H→D 拷贝。
            ⑤ 各 component.commit_hicache_transfer(LOAD_BACK) 回填 device value。
            ⑥ dec_host_lock_ref 释放 Host 锁。
            ⑦ 若 load 的 token 太小 (< load_back_threshold) 或无辅组件, 跳过。

        📥 best_match_node: match 返回的最佳匹配节点。
        📥 mem_quota: 可用设备内存上限。
        📥 req: 请求对象 (传给组件, 用于 SWA LOAD_BACK)。
        📤 True=加载成功, False=跳过 (太小或超配额)。"""
        if self.cache_controller is None:
            return False

        start_time = time.perf_counter()
        # ① 锁定 Host 锚点 (防止 load 过程中 Host 数据被驱逐)
        host_anchor_params = self.inc_host_lock_ref(best_match_node).to_dec_params()
        # ② Build KV transfer: Full 组件列出需要 load 的 host_indices
        kv_xfer = self.components[BASE_COMPONENT_TYPE].build_hicache_transfers(
            best_match_node, CacheTransferPhase.LOAD_BACK
        )[0]

        # ③ 锁定设备路径 + 提前驱逐腾空间
        result = self.inc_lock_ref(best_match_node)
        ancestor_lock_params = result.to_dec_params()
        kv_tokens = len(kv_xfer.host_indices)

        # ④ 各辅组件构造 LOAD_BACK 传输 (如 SWA 的 tombstone host_value)
        comp_xfers: dict[ComponentType, list] = {}
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            t = comp.build_hicache_transfers(
                best_match_node, CacheTransferPhase.LOAD_BACK, req=req
            )
            if t:
                comp_xfers[comp.component_type] = t
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.LOAD_BACK, kv_xfer, comp_xfers
        )

        # ⑤ 跳过条件: 太小 (< threshold) 或超出内存配额
        if (kv_tokens < self.load_back_threshold and not comp_xfers) or (
            mem_quota is not None and kv_tokens > mem_quota + result.delta
        ):
            self.dec_lock_ref(best_match_node, ancestor_lock_params)
            self.dec_host_lock_ref(best_match_node, host_anchor_params)
            return False

        # ⑥ Device pool 不够 → 驱逐腾出空间
        if self.supports_swa():
            avail = self.token_to_kv_pool_allocator.full_available_size()  # SWA: 只检查 Full pool
        else:
            avail = self.token_to_kv_pool_allocator.available_size()
        if avail < kv_tokens:
            needed = kv_tokens - avail
            result = self.evict(EvictParams(num_tokens=needed))
            if result.num_tokens_evicted < needed:
                self.dec_lock_ref(best_match_node, ancestor_lock_params)
                self.dec_host_lock_ref(best_match_node, host_anchor_params)
                return False  # 驱逐不够 → 放弃 load

        # ⑦ 执行 H→D 拷贝 (cache_controller.load 分配 device indices + DMA)
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        device_indices = self.cache_controller.load(
            host_indices=kv_xfer.host_indices,
            node_id=best_match_node.id,
            extra_pools=aux_xfers or None,
        )

        # ⑧ 释放设备路径锁 (load 完成, 数据已在 device)
        self.dec_lock_ref(best_match_node, ancestor_lock_params)
        if device_indices is None:
            self.dec_host_lock_ref(best_match_node, host_anchor_params)
            return False  # 拷贝失败

        # ⑨ 各组件 commit: 将 device_indices 回填到 component_data.value
        kv_xfer.device_indices = device_indices
        self.components[BASE_COMPONENT_TYPE].commit_hicache_transfer(
            best_match_node,
            CacheTransferPhase.LOAD_BACK,
            [kv_xfer],
        )
        for node in kv_xfer.nodes_to_load or ():
            self._record_store_event(node, medium=StorageMedium.GPU)  # 记录 GPU 存储事件
        for ct, xfers in comp_xfers.items():
            self.components[ct].commit_hicache_transfer(
                best_match_node,
                CacheTransferPhase.LOAD_BACK,
                xfers,
            )

        # ⑩ 更新叶子集合 + 记录 ongoing_load_back (供后续确认回调)
        self._update_evictable_leaf_sets(best_match_node)
        self.ongoing_load_back[best_match_node.id] = (
            best_match_node,
            self.inc_lock_ref(best_match_node).to_dec_params(),
            host_anchor_params,
        )

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        return True

    def _build_sidecar_transfers(
        self,
        phase: CacheTransferPhase,
        kv_xfer: PoolTransfer,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> list[PoolTransfer]:
        """🔗 构造 sidecar pool 传输 —— 从 KV/SWA/MAMBA 主 pool 派生 sidecar 传输描述符。

        🔗 write_backup / load_back / write_backup_storage / prefetch_from_storage 调用。

        ⚙️ sidecar pool 是附加的存储池 (如 disagg 场景的传输 buffer):
            根据 spec.indices_from_pool 找到源 pool 的传输 (kv_xfer 或 comp_xfers),
            复用其 keys + hit_policy 构造 sidecar 的 PoolTransfer。"""
        transfers: list[PoolTransfer] = []
        for spec in self.sidecar_pool_specs:
            # 确定索引来源: KV 主 pool 或辅组件 pool (SWA/MAMBA)
            if spec.indices_from_pool == PoolName.KV:
                indices_source = kv_xfer
            else:
                # 从 PoolName 映射到 ComponentType
                source_component = {
                    PoolName.SWA: ComponentType.SWA,
                    PoolName.MAMBA: ComponentType.MAMBA,
                }.get(spec.indices_from_pool)
                if source_component is None:
                    raise AssertionError(
                        f"Unsupported sidecar indices source pool "
                        f"{spec.indices_from_pool}."
                    )
                matching_sources = comp_xfers.get(source_component, ())
                if not matching_sources:
                    continue  # 该辅组件无传输 → 跳过此 sidecar
                indices_source = matching_sources[0]
                if indices_source.name != spec.indices_from_pool:
                    raise AssertionError(
                        f"Sidecar indices source pool {spec.indices_from_pool} "
                        f"resolved to {indices_source.name} during {phase}."
                    )

            # BACKUP_HOST 用 device_indices, 其他阶段用 host_indices
            indices = (
                indices_source.device_indices
                if phase == CacheTransferPhase.BACKUP_HOST
                else indices_source.host_indices
            )
            if indices is None or len(indices) == 0:
                continue  # 无索引 → 跳过
            transfers.append(
                PoolTransfer(
                    name=spec.pool_name,
                    keys=indices_source.keys,  # 复用源 pool 的 keys
                    hit_policy=spec.hit_policy,
                    indices_from_pool=spec.indices_from_pool,
                )
            )
        return transfers

    def _inc_hit_count(self, node: UnifiedTreeNode, chunked: bool = False) -> None:
        """Increment hit count; trigger write_backup when threshold reached."""
        if node.evicted or chunked:
            return
        if (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        ):
            return
        node.hit_count += 1
        if (
            self.cache_controller is not None
            and not node.backuped
            and node.hit_count >= self.write_through_threshold
        ):
            self.write_backup(node)

    def write_backup_storage(self, node: UnifiedTreeNode) -> None:
        """💿 H→Storage 备份 —— 将 Host KV 写入远程存储 (如 shared filesystem / S3)。"""
        # 前提: 启用 storage + 有 controller + 节点已 backup (有 host 数据)
        if (
            not self.enable_storage
            or self.cache_controller is None
            or not node.backuped
        ):
            return

        # 可选: 传递 prefix hash keys (用于 storage 端的范围查询优化)
        prefix_keys = None
        if self.hicache_storage_pass_prefix_keys:
            prefix_keys = node.get_prefix_hash_values(node.parent)

        # ① 各辅组件构造 BACKUP_STORAGE 传输
        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            transfers = comp.build_hicache_transfers(
                node,
                CacheTransferPhase.BACKUP_STORAGE,
            )
            if transfers:
                comp_xfers[comp.component_type] = transfers

        # ② Full KV 传输: host_indices + hash_value (每页一个 hash)
        kv_xfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=node.component_data[BASE_COMPONENT_TYPE].host_value,
            keys=node.hash_value,
        )
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_STORAGE, kv_xfer, comp_xfers
        )
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)

        # ③ 异步写入 Storage, 返回 operation_id
        operation_id = self.cache_controller.write_storage(
            node.component_data[BASE_COMPONENT_TYPE].host_value,
            node.key.token_ids,
            node.hash_value,
            prefix_keys,
            extra_pools=aux_xfers or None,
        )
        # ④ 记录 ongoing_backup + 锁定 Host (防止 backup 过程中 Host 被驱逐)
        self.ongoing_backup[operation_id] = (
            node,
            self.inc_host_lock_ref(node).to_dec_params(),
        )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: UnifiedTreeNode,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
    ) -> None:
        """💿 Storage→Host Prefetch —— 从远程存储预取 KV 到 Host 池。

        🔗 Scheduler 在 match 后, 若有 host hit, 预测后续 token 并触发异步预取。

        ⚙️ 流程:
            ① 构造 prefetch_key (page-aligned), 检查 threshold/rate_limit。
            ② 在 Host pool 申请空间 + 各 component 申请辅组件缓冲区。
            ③ cache_controller.prefetch() 异步从 Storage 加载。
            ④ 记录到 ongoing_prefetch, 后续由 check_prefetch_progress() 轮询完成。"""
        if not self.enable_storage or self.cache_controller is None:
            return

        # ① 构造 prefetch_key: 预测的新 token → page-aligned RadixKey
        extra_key = last_host_node.key.extra_key if last_host_node.key else None
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=extra_key,
            is_bigram=self.is_eagle,
        ).page_aligned(self.page_size)
        prefetch_length = len(prefetch_key)
        # 太小或限流 → 放弃 prefetch
        if (
            prefetch_length < self.prefetch_threshold
            or self.cache_controller.prefetch_rate_limited()
        ):
            return

        # ② 锁定 Host 锚点 + 在 Host pool 申请空间
        anchor_lock_params = self.inc_host_lock_ref(last_host_node).to_dec_params()
        host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            # Host pool 不够 → 先驱逐
            self.evict_host(prefetch_length)
            host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            # 驱逐后仍不够 → 缩减 prefetch_length 到可用空间
            available_size = self.cache_controller.mem_pool_host.available_size()
            prefetch_length = available_size - (available_size % self.page_size)
            if prefetch_length >= self.prefetch_threshold:
                prefetch_key = prefetch_key[:prefetch_length]
                host_indices = self.cache_controller.mem_pool_host.alloc(
                    prefetch_length
                )
            else:
                self.dec_host_lock_ref(last_host_node, anchor_lock_params)
                return
        if host_indices is None:
            self.dec_host_lock_ref(last_host_node, anchor_lock_params)
            return

        # ③ 各辅组件构造 PREFETCH 传输 (如 SWA 申请一个窗口的 host buffer)
        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        alloc_failed = False
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            transfers = comp.build_hicache_transfers(
                last_host_node,
                CacheTransferPhase.PREFETCH,
                token_ids=prefetch_key.token_ids,
                prefetch_tokens=len(prefetch_key),
                last_hash=last_hash,
            )
            if transfers == []:
                # 辅组件 alloc 失败 (返回空列表) → 整体放弃
                alloc_failed = True
                break
            if transfers:
                comp_xfers[comp.component_type] = transfers
        kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.PREFETCH, kv_xfer, comp_xfers
        )
        if alloc_failed:
            # 释放已分配的 host buffer + 解锁
            self.cache_controller.append_host_mem_release(
                host_indices=host_indices,
                extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
            )
            self.dec_host_lock_ref(last_host_node, anchor_lock_params)
            return

        # ④ 异步 prefetch: cache_controller 负责从 Storage 加载到 host_indices
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        operation = self.cache_controller.prefetch(
            req_id,
            host_indices,
            prefetch_key,
            last_hash,
            prefix_keys,
            extra_pools=aux_xfers or None,
        )
        # ⑤ 记录到 ongoing_prefetch (后续 check_prefetch_progress 轮询)
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        )
        self.cache_controller.prefetch_tokens_occupied += len(prefetch_key)

    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation) -> bool:
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    def can_terminate_prefetch(self, operation: PrefetchOperation) -> bool:
        """🔍 判断 prefetch 是否可以终止 —— 按 stop_policy + 跨 TP rank 同步。

        ⚙️ 三种策略:
            best_effort: 立即可终止
            wait_complete: 必须等所有 page 加载完成
            timeout: 完成或超时可终止

        ⚠️ 跨 TP rank all_reduce(MAX): 任一 rank 不可终止 → 全体不可终止。"""
        if self.prefetch_stop_policy == "best_effort":
            return True

        # 检查是否已加载完所有 page
        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        # 按 stop_policy 判断本 rank 是否可终止
        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self._prefetch_timeout_check_linear_func(
                operation
            )
        else:
            return True
        # 即使 completed: 若辅组件 pool_transfers 未完成 → 不可终止
        if (
            completed
            and getattr(operation, "pool_transfers", None)
            and not getattr(operation, "pool_transfers_done", True)
        ):
            can_terminate = False

        # 跨 TP rank 同步: states = [1-can_terminate, operation_terminated]
        operation_terminated = operation.is_terminated()
        states = torch.tensor(
            [1 - int(can_terminate), int(operation_terminated)],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(states, torch.distributed.ReduceOp.MAX)
        # MAX 后: states[0]>0 表示至少一个 rank 不可终止 → can_terminate=False
        can_terminate = states[0].item() == 0
        operation_terminated = states[1].item() == 1
        return can_terminate or operation_terminated

    def check_prefetch_progress(self, req_id: str) -> bool:
        """💿 轮询 Prefetch 进展 —— 检查异步预取是否完成, 完成时 insert 入树。

        🔗 Scheduler 轮询调用。

        ⚙️ 若 operation 未完成 → return False (继续等待)。
            若完成 → terminate_prefetch() + 跨 TP rank all_reduce(MIN) 对齐 →
            _insert_helper_host() 插入 Host 树 + 各 component.commit_hicache_transfer(PREFETCH)。"""
        if req_id not in self.ongoing_prefetch:
            return True  # 无 prefetch 记录 → 视为完成

        (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return True  # 无操作 → 完成
        # ① 检查是否可终止 (跨 TP rank 同步)
        if not self.can_terminate_prefetch(operation):
            return False  # 未完成 → 继续等待

        # ② 终止 prefetch, 获取已完成的 token 数
        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        min_completed_tokens = completed_tokens
        hit_pages = operation.pool_storage_result.extra_pool_hit_pages
        # ③ 跨 TP rank all_reduce(MIN): 取所有 rank 中最小的完成数 (保证一致)
        if self.tp_world_size > 1:
            # Reduce full completed tokens together with the sidecar pools that
            # this prefetch actually transferred, in one all_reduce.
            sidecar_pools = [t.name for xfers in comp_xfers.values() for t in xfers]
            packed = torch.tensor(
                [completed_tokens] + [hit_pages.get(p, 0) for p in sidecar_pools],
                dtype=torch.int,
            )
            self._all_reduce_attn_groups(packed, torch.distributed.ReduceOp.MIN)
            min_completed_tokens = int(packed[0].item())
            for i, p in enumerate(sidecar_pools, start=1):
                hit_pages[p] = int(packed[i].item())

        # ④ 将预取的 KV 插入 Host 树
        fetched_key = prefetch_key[:min_completed_tokens]
        insert_result = self._insert_helper_host(
            last_host_node,
            fetched_key,
            host_indices[:min_completed_tokens],
            hash_value[: min_completed_tokens // self.page_size],
        )

        # ⑤ 各辅组件 commit PREFETCH (如 SWA 填充 tombstone)
        for ct, xfers in comp_xfers.items():
            self.components[ct].commit_hicache_transfer(
                last_host_node,
                CacheTransferPhase.PREFETCH,
                xfers,
                insert_result=insert_result,
                pool_storage_result=operation.pool_storage_result,
            )

        # ⑥ 释放 Host pool: 已插入树的部分 free, 多余部分 release
        self.cache_controller.mem_pool_host.free(
            host_indices[: insert_result.prefix_len]
        )
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )
        # ⑦ 解锁 + 清理 ongoing_prefetch
        self.dec_host_lock_ref(last_host_node, anchor_lock_params)
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)

        # ⑧ 记录指标: 从 storage 新加载的 token 数 = min_completed - 已匹配
        loaded_from_storage = min_completed_tokens - insert_result.prefix_len
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage
        logger.info(
            "HiCache prefetch success req=%s completed_local=%d completed_synced=%d matched=%d loaded=%d tail_release=%d occupied=%d",
            req_id,
            completed_tokens,
            min_completed_tokens,
            insert_result.prefix_len,
            loaded_from_storage,
            completed_tokens - min_completed_tokens,
            self.cache_controller.prefetch_tokens_occupied,
        )
        if self.enable_storage_metrics and self.storage_metrics_collector is not None:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)
        return True

    def terminate_prefetch(self, req_id: str) -> None:
        if req_id not in self.ongoing_prefetch:
            return
        _, _, _, operation, _, _ = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def release_aborted_request(self, rid: str) -> None:
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
        if rid not in self.ongoing_prefetch:
            return

        (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[rid]
        if operation.host_indices is None:
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        self._barrier_attn_groups()
        self.dec_host_lock_ref(last_host_node, anchor_lock_params)
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(
            host_indices=host_indices[:completed_tokens],
            extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
        )
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        extra_release_counts: Optional[dict[PoolName, int]],
        log_metrics: bool,
    ) -> None:
        """🔧 实际执行队列消费 —— revoke / backup ack / release / extra release 四类。"""
        cc = self.cache_controller

        def _drain_queue(q: Queue[T], limit: Optional[int]) -> Iterator[T]:
            """通用队列消费: 最多取 limit 个 (None=全部)"""
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _drain_revoke():
            """处理 prefetch 撤销: 释放 host buffer + 解锁"""
            drained = 0
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                info = self.ongoing_prefetch.pop(req_id, None)
                if info is None:
                    continue
                drained += 1
                (
                    last_host_node,
                    prefetch_key,
                    _host_indices,
                    _operation,
                    anchor_lock_params,
                    comp_xfers,
                ) = info
                cc.append_host_mem_release(
                    extra_pools=[x for xfers in comp_xfers.values() for x in xfers]
                )
                self.dec_host_lock_ref(last_host_node, anchor_lock_params)
                cc.prefetch_tokens_occupied -= len(prefetch_key)
                if cc.prefetch_tokens_occupied < 0:
                    cc.prefetch_tokens_occupied = 0
            return drained

        def _drain_backup():
            """处理 backup ack: 释放 host lock + 记录指标"""
            drained = 0
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                drained += 1
                entry = self.ongoing_backup.pop(operation.id, None)
                if entry is not None:
                    node, lock_params = entry
                    self.dec_host_lock_ref(node, lock_params)
                if (
                    log_metrics
                    and self.enable_storage_metrics
                    and self.storage_metrics_collector is not None
                ):
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )
            return drained

        def _drain_release():
            """处理 host mem release: 批量 free host pool"""
            host_indices_list = []
            released_tokens = 0
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
                released_tokens += len(host_indices)
            if host_indices_list:
                cc.mem_pool_host.free(torch.cat(host_indices_list, dim=0))
            return len(host_indices_list), released_tokens

        def _drain_extra_release():
            """处理辅组件 pool (如 SWA host pool) 的 release"""
            drained: dict[PoolName, tuple[int, int]] = {}
            if not extra_release_counts:
                return drained
            for pool_name, limit in extra_release_counts.items():
                release_queue = cc.extra_host_mem_release_queues.get(pool_name)
                if release_queue is None:
                    continue
                host_indices_list = []
                released_tokens = 0
                for host_indices in _drain_queue(release_queue, limit):
                    host_indices_list.append(host_indices)
                    released_tokens += len(host_indices)
                if host_indices_list:
                    entry = cc.mem_pool_host.entry_map.get(pool_name)
                    if entry is not None:
                        entry.host_pool.free(torch.cat(host_indices_list, dim=0))
                drained[pool_name] = (len(host_indices_list), released_tokens)
            return drained

        # 依次消费四类队列
        _drain_revoke()
        _drain_backup()
        _drain_release()
        _drain_extra_release()

    def drain_storage_control_queues(self) -> None:
        """🔧 消费 Storage 控制队列 —— 跨 TP rank all_reduce(MIN) 对齐后批量处理。

        🔗 Scheduler 每轮调用, 处理 prefetch revoke / backup ack / mem release。"""
        cc = self.cache_controller
        extra_release_queues = getattr(cc, "extra_host_mem_release_queues", {})
        extra_pool_names = list(extra_release_queues)
        # 收集本地队列大小
        local_qsize_list = [
            cc.prefetch_revoke_queue.qsize(),
            cc.ack_backup_queue.qsize(),
            cc.host_mem_release_queue.qsize(),
            *[
                extra_release_queues[pool_name].qsize()
                for pool_name in extra_pool_names
            ],
        ]
        qsizes = torch.tensor(
            local_qsize_list,
            dtype=torch.int,
        )
        # 跨 TP rank all_reduce(MIN): 取最小队列大小 (保证各 rank 处理一致数量)
        self._all_reduce_attn_groups(qsizes, torch.distributed.ReduceOp.MIN)
        qsize_list = list(map(int, qsizes.tolist()))
        n_revoke, n_backup, n_release = qsize_list[:3]
        extra_release_counts = {
            pool_name: count
            for pool_name, count in zip(extra_pool_names, qsize_list[3:])
        }
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_backup=n_backup,
            n_release=n_release,
            extra_release_counts=extra_release_counts,
            log_metrics=True,
        )

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_base: float,
        prefetch_timeout_per_ki_token: float,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[dict[str, str]],
    ) -> None:
        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        self.enable_storage_metrics = enable_storage_metrics

        if self.enable_storage_metrics:
            attn_cp_rank, attn_cp_size = (
                self.cache_controller.get_attn_cp_rank_and_size()
            )
            labels = {
                "storage_backend": storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
                "attn_cp_rank": attn_cp_rank,
                "attn_cp_size": attn_cp_size,
            }
            if extra_metric_labels:
                labels.update(extra_metric_labels)
            existing_collector = self.storage_metrics_collector
            if existing_collector is None:
                from sglang.srt.server_args import get_global_server_args

                storage_cls = resolve_collector_class(
                    get_global_server_args(),
                    STAT_LOGGER_ROLE_STORAGE,
                    StorageMetricsCollector,
                )
                self.storage_metrics_collector = storage_cls(labels=labels)
            elif set(existing_collector.labels.keys()) == set(labels.keys()):
                existing_collector.labels = labels
            else:
                logger.warning(
                    "Storage metrics labels changed (%s -> %s). Keep existing labels to avoid duplicate metric registration.",
                    sorted(existing_collector.labels.keys()),
                    sorted(labels.keys()),
                )
        else:
            self.storage_metrics_collector = None

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        return (
            False,
            "UnifiedRadixCache does not support runtime HiCache storage attach yet. "
            "Configure hicache_storage_backend at startup instead.",
        )

    def detach_storage_backend(self) -> tuple[bool, str]:
        return (
            False,
            "UnifiedRadixCache does not support runtime HiCache storage detach yet. "
            "Restart without hicache_storage_backend to disable it.",
        )

    def clear_storage_backend(self) -> bool:
        try:
            ok = self.cache_controller.clear_storage_backend()
        except Exception as e:
            logger.error("Failed to clear hierarchical cache storage backend: %s", e)
            return False
        if ok:
            logger.info("Hierarchical cache storage backend cleared successfully!")
        return ok

    # ---- HiCache: Async Event Management ----

    def writing_check(self, write_back: bool = False) -> None:
        """Poll write-through completions."""
        cc = self.cache_controller
        if cc is None:
            return

        if write_back:
            # Blocking: wait for all pending write-backs
            while self.ongoing_write_through:
                for _, finish_event, ack_list in cc.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        if ack_id in self.ongoing_write_through:
                            self._finish_write_through_ack(ack_id)
                cc.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # Every rank must enter the all_reduce below; ongoing_write_through can
        # diverge across ranks (e.g. write_backup returning 0 on a subset).
        finish_count = 0
        if self.pp_rank == 0:
            for _, finish_event, ack_list in cc.ack_write_queue:
                if not finish_event.query():
                    break
                finish_count += 1

        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
        finish_count = finish_count_tensor.item()

        # Process completed acks
        while finish_count > 0:
            _, finish_event, ack_list = cc.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                self._finish_write_through_ack(ack_id)
            finish_count -= 1

    def loading_check(self) -> None:
        """Poll load-back completions."""
        cc = self.cache_controller
        if cc is None:
            return
        # Every rank must enter the all_reduce below; ongoing_load_back can
        # diverge across ranks.
        finish_count = 0
        if self.pp_rank == 0:
            for _, finish_event, ack_list in cc.ack_load_queue:
                if not finish_event.query():
                    break
                finish_count += 1
        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
        finish_count = finish_count_tensor.item()

        while finish_count > 0:
            _, finish_event, ack_list = cc.ack_load_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                node, lock_params, host_lock_params = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(node, lock_params)
                self.dec_host_lock_ref(node, host_lock_params)
            finish_count -= 1

    # ---- HiCache: Scheduler Entry Points ----

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> tuple[torch.Tensor, UnifiedTreeNode]:
        """Prepare KV cache loading from host to device.
        Returns (device_indices, last_node) tuple."""
        best_match_node = params.best_match_node
        mem_quota = params.mem_quota
        req = params.req
        assert req is not None
        last_best_match_device_node = req.last_node

        def _collect_new_prefix_indices() -> torch.Tensor:
            prefix_chunks: list[torch.Tensor] = []
            node = best_match_node
            while node is not last_best_match_device_node:
                value = node.component_data[BASE_COMPONENT_TYPE].value
                assert value is not None
                prefix_chunks.append(value)
                node = node.parent
            if not prefix_chunks:
                return self._empty_match_result.device_indices
            prefix_chunks.reverse()
            return torch.cat(prefix_chunks)

        if (
            best_match_node.evicted
            or params.host_hit_length > 0
            or (
                req is not None
                and (req.swa_host_hit_length > 0 or req.mamba_host_hit_length > 0)
            )
        ):
            if self.load_back(best_match_node, mem_quota, req=req):
                new_indices = _collect_new_prefix_indices()
                if new_indices.numel() == 0:
                    return (
                        self._empty_match_result.device_indices,
                        last_best_match_device_node,
                    )

                logger.debug(
                    "init_load_back success: loaded %d tokens for node %d",
                    len(new_indices),
                    best_match_node.id,
                )
                return new_indices, best_match_node

        return (
            self._empty_match_result.device_indices,
            last_best_match_device_node,
        )

    def check_hicache_events(self) -> None:
        """Called per scheduler step to poll async HiCache events."""
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
        self._reap_completed_async_work()
        if self.enable_storage_metrics and self.storage_metrics_collector is not None:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def flush_write_through_acks(self) -> None:
        """Flush pending write-through acknowledgements."""
        self.writing_check()

    def ready_to_load_host_cache(self) -> int:
        """Notify the cache controller to start the KV cache loading."""
        if self.cache_controller is not None:
            return self.cache_controller.start_loading()
        return 0

    # ---- Query / Inspection APIs ----
    # These APIs exist for compatibility with other RadixTree implementations.
    # TODO: simplify and consolidate in a future refactor.

    @property
    def sliding_window_size(self):
        swa = self.components.get(ComponentType.SWA)
        return swa.sliding_window_size if swa else None

    def supports_swa(self) -> bool:
        return ComponentType.SWA in self.components

    def supports_mamba(self) -> bool:
        return ComponentType.MAMBA in self.components

    # ---- Streaming session API (delegates to composed StreamingSession) ----

    def supports_streaming_session(self) -> bool:
        return True

    def release_session(self, session_id: str) -> None:
        self.session.release_session(session_id)

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_tokens(active_pool_idxs)

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_full_tokens(active_pool_idxs)

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_swa_tokens(active_pool_idxs)

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_req_count(active_pool_idxs)

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_mamba_slots(active_pool_idxs)

    def evictable_size(self) -> int:
        return self.component_evictable_size_.get(BASE_COMPONENT_TYPE, 0)

    def protected_size(self) -> int:
        return self.component_protected_size_.get(BASE_COMPONENT_TYPE, 0)

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def full_protected_size(self) -> int:
        return self.protected_size()

    def swa_evictable_size(self) -> int:
        return self.component_evictable_size_.get(ComponentType.SWA, 0)

    def mamba_evictable_size(self) -> int:
        return self.component_evictable_size_.get(ComponentType.MAMBA, 0)

    def swa_protected_size(self) -> int:
        return self.component_protected_size_.get(ComponentType.SWA, 0)

    def mamba_protected_size(self) -> int:
        return self.component_protected_size_.get(ComponentType.MAMBA, 0)

    def total_size(self):
        total_size = 0
        total_aux_size = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            full_value = node.component_data[BASE_COMPONENT_TYPE].value
            if full_value is not None:
                total_size += len(full_value)
            for ct in self.tree_components:
                if ct == BASE_COMPONENT_TYPE:
                    continue
                value = node.component_data[ct].value
                if value is not None:
                    total_aux_size += len(value)
            for child in node.children.values():
                stack.append(child)
        return total_size, total_aux_size

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs(node: UnifiedTreeNode):
            for child in node.children.values():
                v = child.component_data[BASE_COMPONENT_TYPE].value
                if v is not None:
                    values.append(v)
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def _all_component_values_flatten(
        self, component_type: ComponentType
    ) -> torch.Tensor:
        if component_type not in self.components:
            return torch.tensor([], dtype=torch.int64, device=self.device)

        values = []

        def _dfs(node: UnifiedTreeNode):
            value = node.component_data[component_type].value
            if value is not None:
                values.append(value)
            for child in node.children.values():
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self._all_component_values_flatten(ComponentType.MAMBA)

    def all_swa_values_flatten(self) -> torch.Tensor:
        return self._all_component_values_flatten(ComponentType.SWA)

    def available_and_evictable_str(self) -> str:
        if self.supports_swa():
            full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        else:
            full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable = self.component_evictable_size_[BASE_COMPONENT_TYPE]
        lines = [
            f"Available full tokens: {full_available_size + full_evictable} "
            f"(full_available_size={full_available_size} + full_evictable_size_={full_evictable})"
        ]
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue
            if ct.is_swa:
                available_size = self.token_to_kv_pool_allocator.swa_available_size()
            elif ct.is_mamba:
                available_size = self.req_to_token_pool.mamba_allocator.available_size()
            else:
                continue

            lines.append(
                f"Available {ct}: {available_size + self.component_evictable_size_[ct]} "
                f"(available_size={available_size} + component_evictable_size_={self.component_evictable_size_[ct]})"
            )
        return "\n".join(lines) + "\n"

    def _collect_all_nodes(self) -> list[UnifiedTreeNode]:
        nodes = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
        return nodes

    def sanity_check(self):
        """Verify tree invariants.

        TODO(hzh): This method has relatively high latency; simplify the
        check logic once the tree implementation stabilizes.
        """
        # Skip when streaming sessions hold tree locks: the check asserts
        # all nodes are unlocked during idle, which streaming sessions break
        # by design (they hold a first-turn lock across turns).
        if self.session.any_holding_kv():
            return

        write_back = (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        )

        errors: list[str] = []
        E = errors.append
        all_nodes = self._collect_all_nodes()
        all_node_set = set(all_nodes)
        FCT = BASE_COMPONENT_TYPE

        # ── PART 1: Tree Structure ──
        # Root state
        if self.root_node.component_data[FCT].value is None:
            E("[Root] root missing Full device value")
        if self.root_node.component_data[FCT].lock_ref <= 0:
            E(
                f"[Root] root Full lock_ref={self.root_node.component_data[FCT].lock_ref}"
            )
        if self.root_node.parent is not None:
            E("[Root] root has a parent pointer")
        # Parent ↔ child bidirectional consistency
        for node in all_nodes:
            for child in node.children.values():
                if child.parent is not node:
                    pid = child.parent.id if child.parent else None
                    E(f"[Tree] child {child.id} parent={pid}, expected {node.id}")
                if child.key is None:
                    E(f"[Tree] node {child.id} has no key")

        # ── PART 2: Per-node state machine and leaf qualification ──
        expected_dev_leaves: set[UnifiedTreeNode] = set()
        expected_hst_leaves: set[UnifiedTreeNode] = set()

        for node in all_nodes:
            if node is self.root_node:
                continue
            nid = node.id
            full_dev = node.component_data[FCT].value is not None
            full_hst = node.component_data[FCT].host_value is not None

            # Full is the tree backbone, so aux data requires Full data.
            for ct in self.tree_components:
                if ct == FCT:
                    continue
                cd = node.component_data[ct]
                if cd.value is not None and not full_dev:
                    E(f"node {nid} {ct} device present but Full.value=None")
                if cd.host_value is not None and not full_hst:
                    E(f"node {nid} {ct} host present but Full.host_value=None")

            # Every node must keep Full data on at least one layer.
            if not full_dev and not full_hst:
                E(f"node {nid} dead: no Full device and no Full host")

            # Parent prefixes must keep data whenever the child does.
            if node.parent is not None and node.parent is not self.root_node:
                p_dev = node.parent.component_data[FCT].value is not None
                p_hst = node.parent.component_data[FCT].host_value is not None
                if full_dev and not p_dev:
                    E(f"node {nid} device present but parent {node.parent.id} evicted")
                if full_hst and not p_hst and not write_back:
                    E(f"node {nid} backed up but parent {node.parent.id} not backed up")

            # Lock hierarchy and counters must stay sane.
            fl = node.component_data[FCT].lock_ref
            for ct in self.tree_components:
                cd = node.component_data[ct]
                if cd.lock_ref < 0:
                    E(f"node {nid} {ct} lock_ref={cd.lock_ref}")
                if cd.host_lock_ref < 0:
                    E(f"node {nid} {ct} host_lock_ref={cd.host_lock_ref}")
                if ct != FCT and fl < cd.lock_ref:
                    E(f"node {nid} full_lock={fl} < {ct}_lock={cd.lock_ref}")
                if cd.value is None and cd.lock_ref > 0:
                    E(f"node {nid} {ct} evicted but lock_ref={cd.lock_ref}")

            # Collect expected leaf qualification (single pass)
            if self._is_device_leaf(node):
                expected_dev_leaves.add(node)
            if self._is_host_leaf(node):
                expected_hst_leaves.add(node)

        # ── PART 3: Tracking structures ──

        # Device leaf set must match the expected leaves.
        if self.evictable_device_leaves != expected_dev_leaves:
            extra = self.evictable_device_leaves - expected_dev_leaves
            missing = expected_dev_leaves - self.evictable_device_leaves
            if extra:
                E(f"D-leaf extra: {[n.id for n in list(extra)[:5]]}")
            if missing:
                E(f"D-leaf missing: {[n.id for n in list(missing)[:5]]}")

        # Host leaf set must match the expected leaves.
        if self.evictable_host_leaves != expected_hst_leaves:
            extra = self.evictable_host_leaves - expected_hst_leaves
            missing = expected_hst_leaves - self.evictable_host_leaves
            if extra:
                E(f"H-leaf extra: {[n.id for n in list(extra)[:5]]}")
            if missing:
                E(f"H-leaf missing: {[n.id for n in list(missing)[:5]]}")

        # D-leaf ∩ H-leaf = ∅
        overlap = self.evictable_device_leaves & self.evictable_host_leaves
        if overlap:
            E(
                f"[Leaf] {len(overlap)} in both sets: {[n.id for n in list(overlap)[:5]]}"
            )

        # Stale nodes: leaf sets must only contain tree-reachable nodes
        stale = self.evictable_device_leaves - all_node_set
        if stale:
            E(
                f"{len(stale)} stale nodes in device_leaves: {[n.id for n in list(stale)[:5]]}"
            )
        stale = self.evictable_host_leaves - all_node_set
        if stale:
            E(
                f"{len(stale)} stale nodes in host_leaves: {[n.id for n in list(stale)[:5]]}"
            )

        # Per-component LRU tracking
        for ct in self.tree_components:
            lru = self.lru_lists[ct]
            if ct == FCT:
                # Full uses leaf sets, not LRU
                if len(lru.cache) > 0:
                    E(f"Full device LRU not empty: {len(lru.cache)}")
                if len(self.host_lru_lists[ct].cache) > 0:
                    E(f"Full host LRU not empty: {len(self.host_lru_lists[ct].cache)}")
            else:
                # Aux device values must match the device LRU.
                tree_ids = {
                    n.id
                    for n in all_nodes
                    if n is not self.root_node
                    and n.component_data[ct].value is not None
                }
                lru_ids = set(lru.cache.keys())
                if tree_ids != lru_ids:
                    E(
                        f"{ct} device LRU: "
                        f"+tree={tree_ids - lru_ids}, +lru={lru_ids - tree_ids}"
                    )
                # Aux host-only states must match the host LRU.
                host_lru = self.host_lru_lists[ct]
                s3_ids = {
                    n.id
                    for n in all_nodes
                    if n is not self.root_node
                    and n.component_data[ct].value is None
                    and n.component_data[ct].host_value is not None
                }
                host_lru_ids = set(host_lru.cache.keys())
                if s3_ids != host_lru_ids:
                    E(
                        f"{ct} host LRU: "
                        f"+S3={s3_ids - host_lru_ids}, +lru={host_lru_ids - s3_ids}"
                    )
                # The same aux node must not appear in both device and host LRU.
                inv5_overlap = lru_ids & host_lru_ids
                if inv5_overlap:
                    E(f"{ct} in both device and host LRU: {inv5_overlap}")
                # Linked-list integrity
                self._check_lru_linked_list(lru, ct, "device", errors)
                self._check_lru_linked_list(host_lru, ct, "host", errors)

        # ── PART 4: Size Accounting ──
        for ct in self.tree_components:
            evictable = 0
            protected = 0
            for n in all_nodes:
                if n is self.root_node:
                    continue
                cd = n.component_data[ct]
                if cd.value is not None:
                    toks = len(cd.value)
                    if cd.lock_ref > 0:
                        protected += toks
                    else:
                        evictable += toks
            if self.component_evictable_size_[ct] != evictable:
                E(
                    f"[Size] {ct} evictable={self.component_evictable_size_[ct]} "
                    f"!= recomputed={evictable}"
                )
            if self.component_protected_size_[ct] != protected:
                E(
                    f"[Size] {ct} protected={self.component_protected_size_[ct]} "
                    f"!= recomputed={protected}"
                )

        # ── PART 5: Ongoing Operations ──
        for nid, (n, _, _) in self.ongoing_write_through.items():
            if n not in all_node_set:
                E(f"[Ongoing] write_through node {nid} not in tree")
            elif n.component_data[FCT].lock_ref <= 0:
                E(
                    f"[Ongoing] write_through node {nid} lock_ref={n.component_data[FCT].lock_ref}"
                )
        for nid, (n, _, _) in self.ongoing_load_back.items():
            if n not in all_node_set:
                E(f"[Ongoing] load_back node {nid} not in tree")
            elif n.component_data[FCT].lock_ref <= 0:
                E(
                    f"[Ongoing] load_back node {nid} lock_ref={n.component_data[FCT].lock_ref}"
                )

        # ── Result ──
        if errors:
            msg = (
                f"Sanity check FAILED ({len(errors)} violations "
                f"across {len(all_nodes)} nodes):\n"
                + "\n".join(f"  {e}" for e in errors)
            )
            logger.error(msg)
            self.pretty_print()
            raise AssertionError(msg)

    def _check_lru_linked_list(
        self,
        lru: UnifiedLRUList,
        ct: ComponentType,
        label: str,
        errors: list[str],
    ) -> None:
        """Walk a LRU doubly-linked list, collect integrity errors."""
        pt = lru._pt  # use LRU's own pointer slot
        visited: set[int] = set()
        x = lru.head.lru_next[pt]
        prev = lru.head
        while x is not None and x != lru.tail:
            if x.lru_prev[pt] != prev:
                errors.append(f"[{label}][{ct}] broken prev at node {x.id}")
            if x.id not in lru.cache:
                errors.append(f"[{label}][{ct}] node {x.id} in list not cache")
            if x.id in visited:
                errors.append(f"[{label}][{ct}] cycle at node {x.id}")
                break
            visited.add(x.id)
            prev = x
            x = x.lru_next[pt]
        if x is None:
            errors.append(
                f"[{label}][{ct}] broken chain: lru_next is None "
                f"after node {prev.id if hasattr(prev, 'id') else 'head'}"
            )
        if len(visited) != len(lru.cache):
            errors.append(
                f"[{label}][{ct}] list={len(visited)} != cache={len(lru.cache)}"
            )

    def pretty_print(self) -> None:
        stack = [(self.root_node, 0)]
        while stack:
            node, indent = stack.pop()
            component_str = " ".join(
                f"{ct}={'yes' if node.component_data[ct].value is not None else 'no'}"
                for ct in self.tree_components
            )
            print(
                " " * indent,
                f"[{node.id}]",
                len(node.key),
                f"full_lock={node.component_data[BASE_COMPONENT_TYPE].lock_ref}",
                component_str,
            )
            for child in node.children.values():
                stack.append((child, indent + 2))

    def _rebuild_host_leaf_sets(self) -> None:
        """Rebuild evictable_host_leaves after L1-only reset."""
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node is not self.root_node:
                self._update_evictable_leaf_sets(node)
            stack.extend(node.children.values())

    def _rebuild_host_lru_lists(self) -> None:
        """Rebuild host_lru_lists for extra components after L1-only reset.
        Walks the tree and adds nodes with host component data to the
        appropriate host LRU list."""
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node is not self.root_node:
                for ct in self.tree_components:
                    if ct == BASE_COMPONENT_TYPE:
                        continue  # Full uses evictable_host_leaves, not host LRU
                    cd = node.component_data[ct]
                    if cd.host_value is not None:
                        self.host_lru_lists[ct].insert_mru(node)
            stack.extend(node.children.values())
