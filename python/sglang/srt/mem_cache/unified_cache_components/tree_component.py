from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from enum import Enum, IntFlag
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import torch
from numpy import float64

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolTransfer, PoolTransferResult

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


class ComponentType(int, Enum):
    """🏷️ 组件类型枚举 —— 可直接当作 int 索引 component_data[ct] 和 LRU 指针 slot。"""


    FULL = 0
    SWA = 1
    MAMBA = 2

    def __str__(self) -> str:  # keep human-readable logging
        return self.name.lower()

    @property
    def is_full(self) -> bool:
        return self == ComponentType.FULL

    @property
    def is_swa(self) -> bool:
        return self == ComponentType.SWA

    @property
    def is_mamba(self) -> bool:
        return self == ComponentType.MAMBA


BASE_COMPONENT_TYPE = ComponentType.FULL
_NUM_COMPONENT_TYPES = len(ComponentType)

_LAST_ACCESS_TIME_COUNTER_FLOAT = float64(1.0)
_COMPONENT_UUID_COUNTER = 1


# ═════════════════════════ 🏷️ ComponentType & Data ═════════════════════════


@dataclasses.dataclass
class ComponentData:
    """🧬 每节点单组件数据 —— value=设备端 KV 索引, host_value=Host 端备份, lock_ref>0 防驱逐。"""
    value: Optional[torch.Tensor] = None
    lock_ref: int = 0
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    host_value: Optional[torch.Tensor] = None
    host_lock_ref: int = 0


class EvictLayer(IntFlag):
    """Which storage layer(s) to evict.  Combinable via bitwise OR."""

    DEVICE = 1
    HOST = 2
    ALL = DEVICE | HOST


class CacheTransferPhase(str, Enum):

    BACKUP_HOST = "backup_host"  # D→H
    LOAD_BACK = "load_back"  # H→D
    BACKUP_STORAGE = "backup_storage"  # H→Storage
    PREFETCH = "prefetch"  # Storage→H


class LRURefreshPhase(str, Enum):

    WALKDOWN = "walkdown"  # touching a node while walking through the tree
    MATCH_END = "match_end"  # end of a successful prefix match
    INSERT_END = "insert_end"  # after a new/updated leaf is committed


def get_and_increase_time_counter() -> float64:
    global _LAST_ACCESS_TIME_COUNTER_FLOAT
    ret = _LAST_ACCESS_TIME_COUNTER_FLOAT
    _LAST_ACCESS_TIME_COUNTER_FLOAT += 1.0
    return ret


def next_component_uuid() -> int:
    global _COMPONENT_UUID_COUNTER
    _COMPONENT_UUID_COUNTER += 1
    return _COMPONENT_UUID_COUNTER


class TreeComponent(ABC):
    """🧩 树组件抽象基类 —— 所有缓存类型 (Full/SWA/Mamba) 通过 hook 接口与 UnifiedRadixCache 交互。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔗 调用者: UnifiedRadixCache 在 match/insert/evict/lock/cache 各阶段遍历组件调用 hook ║
    ║                                                                                  ║
    ║  🏷️ 子类必须设置 class attr component_type: ComponentType                           ║
    ║                                                                                  ║
    ║  📋 Hook 分组:                                                                     ║
    ║    Match:    create_match_validator / finalize_match_result                      ║
    ║    Insert:   update_component_on_insert_overlap / commit_insert_component_data    ║
    ║    Split:    redistribute_on_node_split                                           ║
    ║    Evict:    evict_component / eviction_priority / drive_eviction                 ║
    ║    Lock:     acquire_component_lock / release_component_lock                      ║
    ║    Cache:    prepare_for_caching_req / cleanup_after_caching_req                  ║
    ║    HiCache:  build_hicache_transfers / commit_hicache_transfer                    ║
    ║                                                                                  ║
    ║  👆 详解见 README-zh.md 钩子参考表。                                                 ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """
    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        self.cache = cache

    # Subclasses MUST set this as a class attribute (not @property)
    component_type: ComponentType

    def node_has_component_data(
        self, node: UnifiedTreeNode, target: EvictLayer = EvictLayer.DEVICE
    ) -> bool:
        """📖 查询节点是否有该组件的 Device 或 Host 数据。

        🔗 被 _cascade_evict、LRU refresh、insert overlap 等内部逻辑调用。"""
        cd = node.component_data[self.component_type]
        if target is EvictLayer.DEVICE:
            return cd.value is not None
        return cd.host_value is not None

    # ---- Utilities: node-level data queries ----

    def value_len(self, node: UnifiedTreeNode) -> int:
        value = node.component_data[self.component_type].value
        return len(value) if value is not None else 0

    def refresh_lru(
        self,
        phase: LRURefreshPhase,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
    ) -> None:
        """🕐 按阶段刷新组件 LRU。

        🔗 UnifiedRadixCache._touch_node (WALKDOWN) / _match_post_processor (MATCH_END) / _insert_helper (INSERT_END) 回调。

        ⚙️ WALKDOWN: 单节点 reset_node_mru (仅当有 value)。
           MATCH_END: 匹配路径全部提升到 MRU (子节点比父节点更 MRU)。
           INSERT_END: WALKDOWN 已刷新过, 跳过。"""
        ct = self.component_type
        match phase:
            case LRURefreshPhase.WALKDOWN:
                if node.component_data[ct].value is None:
                    return
                self.cache.lru_lists[ct].reset_node_mru(node)
            case LRURefreshPhase.MATCH_END:
                self.cache.lru_lists[ct].reset_node_and_parents_mru(
                    node, root_node, self.node_has_component_data
                )
            case LRURefreshPhase.INSERT_END:
                # WALKDOWN already refreshed every node on the insert path
                # (including the new leaf), so there is nothing more to do.
                return
            case _:
                raise ValueError(f"Unknown LRURefreshPhase: {phase}")

    @abstractmethod
    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        """🔍 返回状态化闭包 —— 在 _match_prefix_helper 遍历 tree 时逐节点判断是否有效匹配边界。

        🔗 UnifiedRadixCache._match_prefix_helper 每轮 match 只创建一次。
            match_device_only=True 时 host backup 不能作为有效匹配边界。

        📤 Full: value 非 None; SWA: 累计 >= sliding_window_size; Mamba: value 非 None。"""
        ...

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        """🔍 匹配完成后后处理。

        🔗 _match_post_processor 在拼接 device_indices 后调用。
            Full/SWA: pass through; Mamba: COW 分配新 Mamba slot + 复制 SSM state。"""
        return result

    def update_component_on_insert_overlap(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
    ) -> int:
        """📝 insert 时处理与已有节点的重叠 —— 返回组件"消费"了多少 KV slot。

        🔗 _insert_helper 对每个重叠节点调用。返回的 consumed_from 决定释放范围。
            Full/Mamba: 默认返回 prefix_len; SWA: 窗口内复活 tombstone 时可消费全部/部分。"""
        return prefix_len

    def should_skip_leaf_creation(
        self, total_prefix_len: int, key_len: int, params: InsertParams
    ) -> bool:
        """📝 否决新叶创建 —— 当整个新叶对本组件都是 tombstone 时。

        🔗 _insert_helper 在创建叶子前检查。任意组件返回 True 则跳过叶子创建。"""
        return False

    def recover_after_unevict(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        params: InsertParams,
    ) -> None:
        """📝 unevict 后重建辅组件数据。

        🔗 _unevict_node_on_insert 恢复了 Full value 后回调。SWA 用此 hook 从新 Full value 重建窗口内 SWA。"""
        return None

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        """📝 insert 遍历完成后在目标节点上最终确定组件数据。

        🔗 _insert_helper 末尾调用, 每 insert 仅一次。
            Full: no-op; SWA: 可能按窗口边界再次分裂; Mamba: 设 mamba_value + 插入 LRU。"""
        pass

    @abstractmethod
    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        """✂️ 节点分裂时在新 parent 和 child 间重分配组件数据。

        🔗 _split_node 在创建 new_parent 后、插入 LRU 前调用。
            Full: 复制 lock_ref; SWA: 切片 value + 复制 UUID; Mamba: parent 得 None。"""
        ...

    @abstractmethod
    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        """Free this component's KV resources on a node being evicted.

        *target* controls which layer(s) to evict:
          - DEVICE: free device memory and tombstone (value = None).
                    Host data is untouched.
          - HOST:   free host memory (host_value = None).
                    Device data is untouched.
          - ALL:    free both device and host memory.
                    No tombstone — caller will delete the node.

        Returns (device_freed, host_freed) token counts."""
        ...

    def eviction_priority(self, is_leaf: bool) -> int:
        """Eviction priority on this node type. Higher = evicted later.
        When a component is evicted, all other components with equal or
        lower priority on the same node are also cascade-evicted.

        Leaf: all components equal (0) — evicting any cascades to all,
        because the node will be deleted.

        Internal: full=2 > swa=1 > mamba=0.
        Why swa > mamba: SWA data on internal nodes is *path data* —
        the sliding window needs continuous SWA coverage along the path
        from root to the match boundary. E.g. A->B->C->D->E where C
        and E both have mamba and the window covers C->E: if C's mamba
        is evicted, C's SWA must stay so E remains reachable.
        Mamba data, by contrast, is only meaningful at the match
        boundary node; on internal nodes it
        contributes nothing to the path. So SWA is more valuable to
        keep and should be evicted later.

        Cascade consequences:
        - Mamba evict internal: no cascade.
        - SWA evict internal: cascades to Mamba. SWA gone -> SWA
          validator fails -> mamba data is useless (match requires all
          validators to pass).
        - Full evict internal: cascades to SWA + Mamba."""
        return 0

    @abstractmethod
    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        """Drive eviction from this component's LRU list.
        Each component extracts its own request from params, walks its own
        LRU, evicts, and calls cache._cascade_evict for priority cascade.
        Updates the shared tracker with freed amounts for all components.
        - Full: walks leaf LRU, evicts full then cascades entire leaf.
        - Mamba: walks full LRU; tombstones internal nodes (with cascade
          to equal-priority components like swa), cascades leaves to all."""
        ...

    @abstractmethod
    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        """Increment component lock refs, protecting nodes from
        eviction. Updates evictable → protected size on first lock.
        - Full: path-lock — walks from node up to root, incrementing
          lock_ref on every ancestor.
        - SWA: path-lock — walks upward collecting swa values until the
          sliding window is filled; records a component_uuid at the
          boundary for release_component_lock to know where to stop.
        - Mamba: single-node lock — only increments lock_ref on the
          node itself (mamba state is per-leaf, not per-path).

        When ``lock_host`` is True, the lock applies to host-side state:
        - Full: single-node host lock.
        - SWA: host window-lock with a dedicated host UUID boundary.
        - Mamba: single-node host lock with host LRU detach."""
        ...

    @abstractmethod
    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: Optional[DecLockRefParams],
        lock_host: bool = False,
    ) -> None:
        """Decrement component lock refs, un-protecting nodes.
        Updates protected → evictable size when lock_ref drops to 0.
        - Full: path-unlock — walks from node up to root, decrementing
          lock_ref on every ancestor.
        - SWA: path-unlock — walks upward, stopping at the node whose
          component_uuid matches the one recorded during acquire.
        - Mamba: single-node unlock — only decrements lock_ref on the
          node itself.

        When ``lock_host`` is True, the inverse host-side semantics apply."""
        ...

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        """Prepare component-specific data before insert, fill component
        fields in insert_params, return effective cache_len.
        Return None for no truncation opinion (use full length);
        return int >= 0 for effective cache length.
        - Full: no-op, returns None.
        - SWA: sets insert_params.swa_evicted_seqlen on finished; returns None.
        - Mamba: prepares mamba_value (finished from ping-pong buffer,
          unfinished fork from req); returns mamba_last_track_seqlen."""
        return None

    def cleanup_after_caching_req(
        self,
        req: Req,
        is_finished: bool,
        insert_result: Optional[InsertResult] = None,
        insert_params: Optional[InsertParams] = None,
    ) -> None:
        """Post-cache cleanup for component-specific resources.

        ``is_finished`` — whether the request has finished generation.
        True means the request is complete and its resources can be released;
        ``insert_result`` is None when insert was skipped (cache disabled
        or effective_cache_len <= 0); treat as "no insert happened".
        ``insert_params`` is None only on the disabled path; on early-return
        paths it is still provided so components can free their resources."""
        pass

    def free_out_of_window_slots(
        self, req: Req, pre_len: int, insert_params: InsertParams
    ) -> None:
        pass

    # ════════════════════════ 🔍 Match → 📝 Insert → ✂️ Split → 🗑️ Evict → 🔒 Lock → 💾 Cache → 💿 HiCache ═══

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
        """Build transfer descriptors for this component in the given phase.
        Returns None if the component has nothing to transfer."""
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
        """Post-transfer bookkeeping: store host indices, update LRU, etc."""
        pass

    def drive_host_eviction(
        self, num_tokens: int, tracker: dict[ComponentType, int]
    ) -> None:
        """Evict from this component's host-side resources.
        Called by HostPoolGroup when the host pool is full.
        Default no-op for components without host storage."""
        pass
