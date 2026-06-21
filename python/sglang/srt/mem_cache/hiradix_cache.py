from __future__ import annotations

import atexit
import heapq
import json
import logging
import os
import threading
import time
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import (
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
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PrefetchTimeoutConfig,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    attach_hybrid_dsa_pool_to_hiradix_cache,
)
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MLATokenToKVPoolHost,
    get_mha_host_pool_cls,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
)
from sglang.srt.mem_cache.utils import (
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class HiRadixCache(RadixCache):
    """分层 Radix Cache —— 在 RadixCache 基础上增加 GPU↔Host↔Storage 三级 KV 存储。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  与 RadixCache 的核心区别：节点三态 vs 两态                                           ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                  ║
    ║  RadixCache 节点只有两种状态：                                                      ║
    ║    ┌──────────┐   evict   ┌──────────┐                                           ║
    ║    │ 在 GPU   │ ────────→ │ 删除     │  value=None 就意味着节点没了                  ║
    ║    │ value≠∅  │           │ 节点移除  │                                           ║
    ║    └──────────┘           └──────────┘                                           ║
    ║                                                                                  ║
    ║  HiRadixCache 节点状态流转（四态）：                                                 ║
    ║                                                                                  ║
    ║  GPU→Host 有两种触发路径：                                                               ║
    ║    write_through: 新 token 立即 DMA → GPU+Host 共存（不等 evict）                        ║
    ║    write_back:    等 GPU 内存不足 evict 时才 DMA → GPU 被释放                             ║
    ║                                                                                             ║
    ║                           write_backup                   write_backup_storage               ║
    ║    ┌──────────┐  insert(wt)   ┌──────────┐  evict(GPU)  ┌──────────┐   backup  ┌──────────┐ ║
    ║    │  GPU     │ ────────────→ │ GPU+Host │ ────────────→│  Host    │ ────────→ │ Storage  │ ║
    ║    │ value≠∅  │               │ value≠∅  │              │ value=∅  │           │ host=∅   │ ║
    ║    │ host=∅   │               │ host≠∅   │              │ host≠∅   │           │          │ ║
    ║    │ backuped=F│              │ backuped=T│             │ evicted=T│           │          │ ║
    ║    └──────────┘               └──────────┘              └──────────┘           └──────────┘ ║
    ║         │                           ↑                        ↑  prefetch           ↑        ║
    ║         │      evict(wb): 直接跳到这里                        └──────────────────────────────┘║
    ║         └──────────────────────────────────────────────────┘                                   ║
    ║                           ↑  load_back                                                         ║
    ║                           └───────────────────────────────────────────────────────────────────┘║
    ║                                                                                  ║
    ║  关键区别：                                                                         ║
    ║    write_through: GPU 和 Host 同时持有 KV（value≠∅, host≠∅），threshold=1              ║
    ║    write_back:    evict 后才写 Host（value=∅, host≠∅），threshold=2                   ║
    ║    evict 不是真删除，而是"降级"（GPU→Host），节点仍在树中。                              ║
    ║    被驱逐的 KV 可以通过 load_back 从 Host 恢复，无需重新计算。                           ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  逐方法对比 HiRadixCache vs RadixCache                                             ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                  ║
    ║  match_prefix:  RadixCache 匹配到就返回 device_indices                             ║
    ║                 HiRadixCache 额外返回 host_hit_length / last_host_node             ║
    ║                 为后续 load_back / prefetch 提供起点                                ║
    ║                                                                                  ║
    ║  insert:        RadixCache 遇匹配节点 inc_hit_count                                ║
    ║                 HiRadixCache 遇 evicted 节点要重新赋 value（KV 回到 GPU）             ║
    ║                                                                                  ║
    ║  evict:         RadixCache 直接 free + _delete_leaf（真删除）                       ║
    ║                 HiRadixCache 分两条路：                                           ║
    ║                   write_back 模式：先 write_backup → 再 _evict_backuped（降级）     ║
    ║                   非 write_back：_evict_regular（真删除，同 RadixCache）            ║
    ║                                                                                  ║
    ║  新增方法（RadixCache 完全没有）：                                                   ║
    ║    write_backup           GPU→Host DMA，node.host_value = host_indices           ║
    ║    load_back              Host→GPU DMA，恢复 node.value，清除 evicted 状态          ║
    ║    evict_host             驱逐 Host 端 KV（Host 内存不够时）                         ║
    ║    prefetch_from_storage  Storage→Host 预取                                      ║
    ║    write_backup_storage   Host→Storage 持久化                                    ║
    ║    writing_check          等待异步 write DMA 完成                                 ║
    ║    loading_check          等待异步 load DMA 完成                                  ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  写策略（write_policy）：write_through vs write_back                               ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                  ║
    ║  write_through（默认）—— 命中即备份，主动同步                                       ║
    ║  ═══════════════════════════════════════════════════                              ║
    ║  节点命中时：                                                                      ║
    ║    _inc_hit_count(node)                                                          ║
    ║      ├── hit_count >= write_through_threshold(=1) 且 node 未 backuped             ║
    ║      │     └── write_backup(node)  ← 主动异步写回 Host                             ║
    ║      │            ├── node.host_value = Host端slot索引                             ║
    ║      │            ├── node.backuped → True（@property: host_value is not None）    ║
    ║      │            └── inc_lock_ref(node) 防止 DMA 飞行中被 evict                    ║
    ║      │   DMA 完成后：writing_check → _finish_write_through_ack → dec_lock_ref      ║
    ║      │                                                                             ║
    ║  GPU evict 时（节点已 backuped）：                                                 ║
    ║    node.backuped=True                                                             ║
    ║      → _evict_backuped(node)  只释放 GPU slot，节点降级                             ║
    ║        node.value = None      → evicted 自动=True                                 ║
    ║        节点仍在树中，host_value 保留，可 load_back 恢复                              ║
    ║                                                                                  ║
    ║  write_back —— 驱逐时才写，被动备份                                                 ║
    ║  ═══════════════════════════════════                                              ║
    ║  节点命中时：                                                                      ║
    ║    _inc_hit_count(node)                                                           ║
    ║      └── write_policy=="write_back" → return（跳过，不计数不备份）                   ║
    ║                                                                                  ║
    ║  GPU evict 时（节点没 backuped）：                                                 ║
    ║    node.backuped=False                                                            ║
    ║      → write_policy=="write_back"                                                 ║
    ║        ├── write_backup(x, write_back=True)  ← 紧急阻塞写回 Host                    ║
    ║        ├── writing_check(write_back=True)     ← 阻塞等全部 DMA 完成                 ║
    ║        └── _evict_backuped(x)                  ← 然后释放 GPU slot                  ║
    ║                                                                                  ║
    ║  对比总结：                                                                        ║
    ║  ┌──────────────┬─────────────────────┬─────────────────────┐                    ║
    ║  │              │ write_through       │ write_back          │                    ║
    ║  ├──────────────┼─────────────────────┼─────────────────────┤                    ║
    ║  │ 备份时机      │ 命中时主动写         │ 驱逐时被动写          │                    ║
    ║  │ hit_count    │ 正常递增触发阈值      │ 直接跳过              │                    ║
    ║  │ evict 行为    │ _evict_backuped     │ write_backup →      │                    ║
    ║  │              │ （无需等 DMA）        │ _evict_backuped      │                    ║
    ║  │              │                     │ （必须等 DMA 完成）    │                    ║
    ║  │ 优势          │ 热门节点 Host 有副本  │ 冷门节点不浪费        │                    ║
    ║  │              │ evict 延迟低          │ Host IO 带宽          │                    ║
    ║  └──────────────┴─────────────────────┴─────────────────────┘                    ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  TreeNode 新增字段（RadixCache 的 TreeNode 没有）                                    ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  host_value               Host 端 slot 索引，write_backup 后非空                    ║
    ║  host_ref_counter         防止 Host KV 被驱逐（类似 lock_ref 但针对 Host）            ║
    ║  write_through_pending_id 跟踪异步写回（write_through 模式 DMA 未完成时）              ║
    ║  hash_value               每页的 SHA256，Storage 层去重/查找用                       ║
    ║  evicted 属性             value is None（RadixCache 中 value=None=删除，           ║
    ║                           HiRadixCache 中 value=None=在 Host，节点还在树中）        ║
    ║  backuped 属性            host_value is not None                                 ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self._enable_metrics_flag = params.enable_metrics

        self.page_size = params.page_size
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()

        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = get_mha_host_pool_cls(self.kv_cache)(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, DSATokenToKVPool):
            # Filled by attach_hybrid_dsa_pool_to_hiradix_cache after storage extra_config is parsed.
            self.token_to_kv_pool_host = None
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        else:
            raise ValueError("HiRadixCache only supports MHA, MLA, and DSA models")

        self.tp_group = params.tp_cache_group
        self.attn_cp_group = params.attn_cp_cache_group
        self.attn_tp_group = params.attn_tp_cache_group
        self.pp_group = params.pp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics
        self.extra_metric_labels = server_args.extra_metric_labels

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_config,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        # TODO: support more timeout check functions
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        self.load_cache_event = threading.Event()
        if isinstance(self.kv_cache, DSATokenToKVPool):
            attach_hybrid_dsa_pool_to_hiradix_cache(
                self,
                params,
                server_args,
                extra_config=extra_config,
                prefetch_threshold=prefetch_threshold,
                enable_storage_metrics=self.enable_storage_metrics,
                load_cache_event=self.load_cache_event,
                attn_cp_group=self.attn_cp_group,
                attn_tp_group=self.attn_tp_group,
            )
        else:
            self.cache_controller = HiCacheController(
                params.token_to_kv_pool_allocator,
                self.token_to_kv_pool_host,
                self.page_size,
                self.tp_group,
                load_cache_event=self.load_cache_event,
                attn_cp_group=self.attn_cp_group,
                attn_tp_group=self.attn_tp_group,
                pp_group=self.pp_group,
                write_policy=server_args.hicache_write_policy,
                io_backend=server_args.hicache_io_backend,
                storage_backend=server_args.hicache_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=extra_config,
                enable_storage_metrics=self.enable_storage_metrics,
            )
        self._apply_storage_runtime_config(
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_config=prefetch_timeout_config,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=self.enable_storage,
            enable_storage_metrics=self.enable_storage_metrics,
            extra_metric_labels=self.extra_metric_labels,
        )

        # ═══════════════════════════════════════════════════════════════════
        # 异步数据传输追踪器 — HiCache 的多层数据流是异步(non-blocking DMA)的，
        # 需要追踪进行中的操作以确保事件回调和节点状态正确性。
        # ═══════════════════════════════════════════════════════════════════

        # ---- GPU→Host 写回追踪 ----
        # key:   ack_id (node.id)
        # value: (lock_node, backup_len, publish_nodes)
        #   - lock_node:     持有读锁的节点(DMA期间防止KV被覆盖)
        #   - backup_len:    写回的token长度(用于split后恢复)
        #   - publish_nodes: DMA完成后标记为CPU_READY的节点列表
        # 完整调用链:
        #   创建: write_backup() → _track_write_through_node()
        #   更新: radix tree split → _replace_pending_write_through_node() 替换旧节点
        #   清除: scheduler writing_check() → _finish_write_through_ack() → pop + dec_lock_ref
        self.ongoing_write_through = {}

        # ---- Host→GPU 加载追踪 ----
        # key:   ack_id (last_hit_node.id)
        # value: last_hit_node (加载任务对应的radix tree叶子节点)
        # 完整调用链:
        #   创建: scheduler → load_back() → ongoing_load_back[last_hit_node.id] = ...
        #   清除: scheduler loading_check() → pop + dec_lock_ref(释放读锁)
        self.ongoing_load_back = {}

        # ---- L3存储→Host 预取追踪 ----
        # key:   req_id (请求ID)
        # value: (last_host_node, prefetch_key, host_indices, operation)
        #   - last_host_node: 最后一个已缓存的host节点
        #   - prefetch_key:   待预取的token序列
        #   - host_indices:   预取token在host pool中的位置
        #   - operation:      存储后端返回的异步操作句柄
        # 完整调用链:
        #   创建: scheduler → prefetch_from_storage() → ongoing_prefetch[req_id] = ...
        #   完成: scheduler prefill_check → del + release_host + 标记loaded tokens
        #   取消: scheduler prefill_revoke → del + release_host
        self.ongoing_prefetch = {}

        # ---- Host→L3存储 后备追踪 ----
        # key:   operation_id (存储操作ID)
        # value: node (被后备的radix tree节点)
        # 完整调用链:
        #   创建: write_backup_storage() → cache_controller.write_storage() → ongoing_backup[op_id] = node
        #   清除: cache_controller ack_backup_queue 回调 → pop + entry.release_host()
        self.ongoing_backup = {}

        # ---- 统计: 每个请求从L3存储加载的token数 ----
        # key:   request_id
        # value: 该请求从存储中实际加载的token数(不含host cache命中)
        # 用途:  监控/指标，评估 storage prefetch 的效果
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}

        # ---- NCCL集合通信异步句柄列表 ----
        # 存储 torch.distributed.Work 对象, 用于attention group barrier同步
        # 作用: 在跨TP/DP组同步时，等待所有成员完成通信
        self.work_list: List[torch.distributed.Work] = []

        # ═══════════════════════════════════════════════════════════════════
        # 阈值控制
        # ═══════════════════════════════════════════════════════════════════

        # write_through_threshold: 控制write-through vs write-back策略
        #   =1 (write_through): 每个新token立即DMA到Host → 延迟低但带宽浪费
        #       调用入口: scheduler → radix_cache.insert() → write_backup(node, write_back=False)
        #   =2 (write_back):    累积2次访问后才写回 → 减少DMA次数,适合热token
        #       调用入口: scheduler → radix_cache.evict() → write_backup(node, write_back=True)
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )

        # load_back_threshold: 预取深度阈值
        #   当L2 host cache命中点离序列末尾 < threshold 时, 提前从L3 storage预取后续token
        #   调用入口: scheduler → match_prefix() 返回结果含 last_host_node → prefetch_from_storage()
        self.load_back_threshold = 10

        # Detach storage backend automatically on process shutdown
        atexit.register(self.shutdown)

        self.evictable_host_leaves = set()

        super().__init__(params=params)

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

        The following diagram illustrates the behavior of _pp_sync.

        time  | pp0                     | pp1                     | pp2
        ------|-------------------------|-------------------------|-----------------------------
        0     | _pp_sync(data=1) starts | _pp_sync(data=?) starts | _pp_sync(data=?) starts
        1     | _pp_sync(data=1) ends   |                         |
        2     |                         | _pp_sync(data=1) ends   |
        3     |                         |                         | _pp_sync(data=1) ends

        _pp_sync requires no synchronization point among ranks. The following case may also happen.

        time  | pp0                     | pp1                     | pp2
        ------|-------------------------|-------------------------|-----------------------------
        0     | _pp_sync(data=1) starts |                         |
        1     | _pp_sync(data=1) ends   |                         |
        2     |                         | _pp_sync(data=?) starts |
        3     |                         | _pp_sync(data=1) ends   |
        4     |                         |                         | _pp_sync(data=?) starts
        5     |                         |                         | _pp_sync(data=1) ends
        """
        if self.pp_size <= 1 or self.pp_group is None:
            return
        if self.pp_rank > 0:
            torch.distributed.recv(
                data, group_src=self.pp_rank - 1, group=self.pp_group, tag=2
            )
        if self.pp_rank + 1 < self.pp_size:
            # Make a copy of data, so that the caller is safe to modify `data` after this call.
            # This is cheap, as _pp_sync is not to be used for transmitting large data.
            copy_of_data = data.clone()
            send_work = torch.distributed.isend(
                copy_of_data, group_dst=self.pp_rank + 1, group=self.pp_group, tag=2
            )
            self.work_list.append(send_work)

    def shutdown(self):
        """Best-effort auto-detach of storage backend on process shutdown.

        This keeps startup and runtime behavior consistent: if a backend was attached
        (either via CLI args or via admin API), we attempt to detach it on exit.
        """
        try:
            if self.enable_storage:
                self.detach_storage_backend()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_config: PrefetchTimeoutConfig,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[Dict[str, str]],
    ) -> None:
        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_config = prefetch_timeout_config
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
            existing_collector = getattr(self, "storage_metrics_collector", None)
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
                    "Storage metrics labels changed (%s -> %s). Keep existing labels to "
                    "avoid duplicate metric registration.",
                    sorted(existing_collector.labels.keys()),
                    sorted(labels.keys()),
                )

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Attach (enable) storage backend at runtime.

        This will start storage threads inside `HiCacheController` and enable
        prefetch/backup paths. Caller must ensure there are no running/queued
        requests to avoid races.
        """
        # Validate inputs first (no side effects).
        if hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if hicache_storage_prefetch_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_storage_prefetch_policy: {hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if hicache_write_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        # If already enabled:
        # - backend unchanged: treat as success, update policies only.
        # - backend changed: treat as failure, do NOT update policies.
        if self.enable_storage:
            current_backend = self.cache_controller.storage_backend_type

            if current_backend == storage_backend:
                if hicache_storage_prefetch_policy is not None:
                    self.prefetch_stop_policy = hicache_storage_prefetch_policy
                    logger.info(
                        f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
                    )
                if hicache_write_policy is not None:
                    self.cache_controller.write_policy = hicache_write_policy
                    self.write_through_threshold = (
                        1 if hicache_write_policy == "write_through" else 2
                    )
                    logger.info(f"Set hicache_write_policy to {hicache_write_policy}")
                return (
                    True,
                    "HiCache storage backend already enabled with same backend; policies updated.",
                )

            return (
                False,
                f"HiCache storage backend is already enabled with backend '{current_backend}'. "
                f"Cannot attach different backend '{storage_backend}'. Detach first.",
            )

        # Not enabled: update policies before controller attach so storage threads observe new values.
        if hicache_storage_prefetch_policy is not None:
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
            logger.info(
                f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
            )

        if hicache_write_policy is not None:
            self.cache_controller.write_policy = hicache_write_policy
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )
            logger.info(f"Set hicache_write_policy to {hicache_write_policy}")

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_config,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json '{storage_backend_extra_config_json}': {e}",
            )

        try:
            self.cache_controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
                **self._get_hybrid_storage_attach_kwargs(),
            )
        except Exception as e:
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self._apply_storage_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_config=prefetch_timeout_config,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=True,
            enable_storage_metrics=self._enable_metrics_flag,
            extra_metric_labels=self.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def detach_storage_backend(self) -> tuple[bool, str]:
        """Detach (disable) storage backend at runtime.

        Caller must ensure there are no running/queued requests to avoid races.
        """
        try:
            # Drain any pending control queues before tearing down storage threads/backend.
            # IMPORTANT: this must happen before we clear `ongoing_*`, otherwise acks/releases
            # cannot be matched to nodes and may leak host pages / locks.
            self._drain_storage_control_queues_local()
            # Idempotent detach: always ask controller to best-effort cleanup, even if
            # `self.enable_storage` is already False (may be leftover state from a
            # previous partial detach).
            self.cache_controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            # Do NOT crash the server for admin operations. Return failure with detail.
            return False, f"Failed to detach HiCache storage backend: {e}"

        # Best-effort cleanup of any leftover bookkeeping.
        self._drain_storage_control_queues_local()
        # After controller threads are fully stopped, it's safe to force-release any
        # leftover pending ops (e.g., async prefetch/backup that didn't get a revoke/ack).
        self._force_release_pending_storage_ops()

        self.enable_storage = False
        self.enable_storage_metrics = False
        return True, "Detached HiCache storage backend successfully."

    def _force_release_pending_storage_ops(self):
        """Force release any leftover pending prefetch/backup bookkeeping.

        This is a safety net for detach/shutdown paths. It assumes storage threads
        have been stopped already (via controller.detach), so no concurrent access
        to these structures should happen.
        """
        cc = self.cache_controller

        # Force release leftover prefetch ops: free pre-allocated host pages and
        # drop the host protection on the matched prefix node.
        try:
            for req_id, info in list(self.ongoing_prefetch.items()):
                try:
                    last_host_node, token_ids, host_indices, _operation = info
                except Exception:
                    # Unexpected shape; just drop it.
                    self.ongoing_prefetch.pop(req_id, None)
                    continue

                try:
                    if host_indices is not None:
                        cc.mem_pool_host.free(host_indices)
                except Exception:
                    logger.exception(
                        "Failed to free host indices for prefetch %s", req_id
                    )

                try:
                    last_host_node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for prefetch %s", req_id
                    )

                try:
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0
                except Exception:
                    pass

                self.ongoing_prefetch.pop(req_id, None)
        except Exception:
            logger.exception("Force release pending prefetch ops failed.")

        # Force release leftover backup ops: drop host protection on nodes.
        try:
            for ack_id, node in list(self.ongoing_backup.items()):
                try:
                    node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for backup op %s", ack_id
                    )
                self.ongoing_backup.pop(ack_id, None)
        except Exception:
            logger.exception("Force release pending backup ops failed.")

    def _drain_storage_control_queues_local(self):
        """Drain storage control queues without TP synchronization.

        This is intended for shutdown/detach paths where we want to make best-effort
        cleanup even if queue sizes temporarily differ across ranks.
        """
        self._drain_storage_control_queues_impl(
            n_revoke=None,
            n_backup=None,
            n_release=None,
            log_metrics=False,
        )

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        log_metrics: bool,
    ):
        cc = self.cache_controller

        def _drain_queue(q, limit: Optional[int]):
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _drain_revoke():
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                info = self.ongoing_prefetch.pop(req_id, None)
                if info is not None:
                    last_host_node, token_ids, _, _ = info
                    last_host_node.release_host()
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0

        def _drain_backup():
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                ack_id = operation.id
                entry = self.ongoing_backup.pop(ack_id, None)
                if entry is not None:
                    entry.release_host()
                if log_metrics and self.enable_storage_metrics:
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )

        def _drain_release():
            host_indices_list = []
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
            if host_indices_list:
                host_indices = torch.cat(host_indices_list, dim=0)
                cc.mem_pool_host.free(host_indices)

        _drain_revoke()
        _drain_backup()
        _drain_release()

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        """
        Parse storage backend extra config JSON and extract specific parameters.

        Args:
            storage_backend_extra_config: JSON string containing extra configuration

        Returns:
            tuple: (extra_config_dict, prefetch_threshold, prefetch_timeout_config, hicache_storage_pass_prefix_keys)
        """
        # Parse extra config if provided. Extra config can be a JSON string or a json/toml/yaml file path prefixed with "@".
        extra_config = {}
        if storage_backend_extra_config:
            try:
                if storage_backend_extra_config.startswith("@"):
                    # Read config from a json/toml/yaml file
                    path = storage_backend_extra_config[1:]
                    ext = os.path.splitext(path)[1].lower()
                    with open(path, "rb" if ext == ".toml" else "r") as f:
                        if ext == ".json":
                            extra_config = json.load(f)
                        elif ext == ".toml":
                            import tomllib

                            extra_config = tomllib.load(f)
                        elif ext in (".yaml", ".yml"):
                            import yaml

                            extra_config = yaml.safe_load(f)
                        else:
                            raise ValueError(
                                f"Unsupported config file {path} (config format: {ext})"
                            )
                else:
                    # read config from JSON string
                    extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        defaults = PrefetchTimeoutConfig()
        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)  # tokens
        prefetch_timeout_base = extra_config.pop(
            "prefetch_timeout_base", defaults.base
        )  # seconds
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", defaults.per_ki_token
        )  # seconds per 1024 tokens
        prefetch_timeout_max = extra_config.pop(
            "prefetch_timeout_max", defaults.max
        )  # seconds, upper bound for the linear timeout
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(prefetch_timeout_max, (int, float)):
            raise ValueError(
                f"prefetch_timeout_max must be number, got {type(prefetch_timeout_max).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        prefetch_timeout_config = PrefetchTimeoutConfig(
            base=float(prefetch_timeout_base),
            per_ki_token=float(prefetch_timeout_per_ki_token),
            max=float(prefetch_timeout_max),
        )

        return (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_config,
            hicache_storage_pass_prefix_keys,
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        # Clear per-request tracking dicts
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.evictable_host_leaves.clear()
        super().reset()

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def _get_extra_pools(self) -> dict:
        if not isinstance(self.cache_controller, HybridCacheController):
            return {}
        if isinstance(self.kv_cache, DSATokenToKVPool):
            pool = PoolTransfer(
                name=PoolName.INDEXER,
                hit_policy=PoolHitPolicy.ALL_PAGES,
                indices_from_pool=PoolName.KV,
            )
            return {"extra_pools": [pool]}
        else:
            return {}

    def _get_hybrid_storage_attach_kwargs(self) -> dict:
        """Extra kwargs for attach_storage_backend when controller is HybridCacheController."""
        if isinstance(self.cache_controller, HybridCacheController):
            return {"host_pools": self.cache_controller.mem_pool_host.entries}
        return {}

    def clear_storage_backend(self) -> bool:
        if self.enable_storage:
            try:
                # Check if the storage backend has a clear method (for nixl backends)
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )
                    return True
                else:
                    logger.warning(
                        f"Storage backend {type(self.cache_controller.storage_backend).__name__} does not support clear operation."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    def write_backup(self, node: TreeNode, write_back=False) -> int:
        """将节点的 KV 从 GPU 写入 Host（GPU→Host DMA）。

        ╔════════════════════════════════════════════════════════════════════════════╗
        ║  write_backup 的两种触发场景                                                 ║
        ╠════════════════════════════════════════════════════════════════════════════╣
        ║                                                                              ║
        ║  1. write_through 模式（write_back=False）：                                  ║
        ║     节点命中次数达标时自动触发，将 KV 主动写入 Host                             ║
        ║     调用链：_inc_hit_count → write_backup(node) → inc_lock_ref(node)             ║
        ║            [DMA 飞行中] → writing_check → _finish_write_through_ack              ║
        ║            → dec_lock_ref + write_backup_storage（写三层存储）                    ║
        ║     前置约束：parent 必须 backuped（保证从 root 到当前节点连续备份，无空隙）      ║
        ║                                                                              ║
        ║  2. write_back 模式（write_back=True）：                                     ║
        ║     evict 时调用，先把 KV 备份到 Host，再释放 GPU                             ║
        ║     调用链：evict → write_backup(write_back=True) → _evict_backuped          ║
        ║     无 parent 约束（evict 是被动触发的，不保证连续性）                          ║
        ║     写完不 inc_lock_ref（马上就要 _evict_backuped）                           ║
        ╚════════════════════════════════════════════════════════════════════════════╝

        写入成功后：
          node.host_value = host_indices  ← 记录 Host 端 slot 索引
          node.backuped → True
          后续可通过 load_back 从 Host 恢复到 GPU

        Host 内存不足时：
          先调 evict_host 释放其他节点的 Host KV，再重试一次
        """
        # write_through 模式下的 backup 连续约束：
        #   从 root 到任意 backuped 节点必须是连续的备份链（不允许中间断层）。
        #   如果父节点还没 backup，先跳过本次写回，等父节点先被 backup。
        if not write_back and (
            node.parent != self.root_node and not node.parent.backuped
        ):
            return 0

        # 发起 GPU→Host DMA 写入，返回 Host 端分配的 slot 索引
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
            **self._get_extra_pools(),
        )

        # Host 内存不足 → 先驱逐其他 Host 节点腾空间，再重试一次
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
                **self._get_extra_pools(),
            )

        if host_indices is not None:
            # 记录 Host 端 slot 索引，标记 backuped 状态
            node.host_value = host_indices.clone()
            assert len(node.host_value) > 0

            # 加入 ongoing_write_through 跟踪字典 —
            #   记录当前节点的 DMA 正在飞行中（pending），后继：
            #   • 防止同一节点重复写 Host（node.write_through_pending_id 不为空时跳过）
            #   • 树分裂时通过 _replace_pending_write_through_node 替换为拆分后的节点
            #   • DMA 完成后 writing_check → _finish_write_through_ack → dec_lock_ref 收尾
            self._track_write_through_node(node, len(node.key))

            # write_through 模式：加 lock_ref 防止 DMA 完成前节点被 evict
            # write_back 模式：不加锁（紧随其后就是 _evict_backuped，节点即将被驱逐）
            if not write_back:
                self.inc_lock_ref(node)
        else:
            # Host 驱逐后仍然分配失败（极端 OOM），本次写回放弃
            return 0

        return len(host_indices)

    def _track_write_through_node(self, node: TreeNode, backup_len: int) -> None:
        node.write_through_pending_id = node.id
        self.ongoing_write_through[node.id] = (node, backup_len, [node])

    def _replace_pending_write_through_node(
        self, old_node: TreeNode, new_nodes: List[TreeNode]
    ) -> None:
        """radix tree 节点 split 后，更新 write-through 等待队列中的节点引用。

        背景：
        GPU→Host DMA 写回是异步的（non_blocking），写回完成后通过 ack_id 确认。
        split 操作在 DMA 完成之前可能发生：一个正在写回的节点因为新请求插入
        共享前缀而被 split 成多个新节点。

        此时 onboard_write_through 中的 publish_nodes 仍然持有 old_node 引用，
        但 DMA 实际写回的是 old_node 对应的 KV 数据。如果不更新引用，
        后续 _finish_write_through_ack 会把写入完成状态标到已被废弃的旧节点上。

        本函数做的事：
          将 publish_nodes 中的 old_node 替换为 new_nodes，
          并给 new_nodes 打上相同的 ack_id，确保 DMA 完成时正确标记新节点。

        Args:
            old_node: split 前被替换的旧节点
            new_nodes: split 后产生的新节点列表
        """
        # 1. 检查 old_node 是否在等待 write-through 完成
        #    write_through_pending_id 非空 → 该节点的 KV 数据正在 DMA 写回中
        ack_id = old_node.write_through_pending_id
        if ack_id is None:
            return  # 不在等待中，无需处理

        # 2. 从全局等待队列中取出对应的 pending 记录
        #    pending = (lock_node, backup_len, publish_nodes)
        #    publish_nodes 就是 DMA 完成后需要标记的节点列表
        pending = self.ongoing_write_through.get(ack_id)
        if pending is None:
            return  # ack 已经处理过了（极端情况）

        lock_node, backup_len, publish_nodes = pending

        # 3. 遍历 publish_nodes，把 old_node 替换为 new_nodes
        updated_nodes = []
        replaced = False
        for node in publish_nodes:
            if node is old_node:
                # old_node 已被 split 成 new_nodes（如 [new_inner, child]）
                updated_nodes.extend(new_nodes)
                replaced = True
            else:
                updated_nodes.append(node)

        if not replaced:
            return  # old_node 不在 publish_nodes 中（已被其他路径处理）

        # 4. 新节点继承 ack_id，等待 DMA 完成确认
        for node in new_nodes:
            node.write_through_pending_id = ack_id

        # 5. 用更新后的节点列表替换队列中的记录
        self.ongoing_write_through[ack_id] = (lock_node, backup_len, updated_nodes)

    def _finish_write_through_ack(self, ack_id: int, *, release_lock: bool) -> None:
        """GPU→Host DMA 完成后的收尾回调，做四件事。

        这是 write_through 流程的终点回调，由 scheduler writing_check() 轮询调用。
        writing_check 先通过 all_reduce(MIN) 确保所有 PP rank 的 DMA 都完成，
        再 FIFO 遍历 ack_write_queue 逐个调用本函数。

        📌 四步收尾：
        1. ongoing_write_through.pop(ack_id)   — 清除异步追踪记录
        2. node.write_through_pending_id = None — DMA 确认，Host 已有副本
        3. _record_store_event(CPU)            — 通知下游索引器（如 disaggregation router）
        📌 4. if enable_storage: write_backup_storage(lock_node, backup_len)
             — 将刚写入 Host 的 KV 进一步备份到 L3 持久化存储！
             — 这是 L2→L3 的数据流入口：Host 写完后立即触发 storage 后备

        Args:
            ack_id:       DMA 操作的唯一 ID（等于 node.id）
            release_lock: 是否释放 write_backup 时加的锁
                          write_through: True（write_backup 加了 inc_lock_ref，现在释放）
                          write_back:    False（write_backup 没加锁，evict 紧随其后）
        """
        lock_node, backup_len, publish_nodes = self.ongoing_write_through.pop(ack_id)
        for node in publish_nodes:
            if node.write_through_pending_id == ack_id:
                node.write_through_pending_id = None  # 清除 pending 标记
            # DMA 确认完成 → 节点 KV 数据现在在 CPU Host 上可用
            self._record_store_event(node, medium=StorageMedium.CPU)
        # 异步写入 L3 持久化存储（mooncake/EIC 等后端）
        if self.enable_storage:
            self.write_backup_storage(lock_node, backup_len)
        # write_through 模式下释放 DMA 飞行期间的保护锁
        if release_lock:
            self.dec_lock_ref(lock_node)

    def write_backup_storage(self, node: TreeNode, backup_len: Optional[int] = None):
        # Recover pre-split data via walk-and-concat if node was split.
        # prefix_keys anchored at chain top to avoid double-counting.
        if backup_len is None or len(node.key) == backup_len:
            top, key, hash_value, host_value = (
                node,
                node.key,
                node.hash_value,
                node.host_value,
            )
        else:
            top, key, hash_value, host_value = self._concat_split_chain(
                node, backup_len
            )

        prefix_keys = (
            top.get_prefix_hash_values(top.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )

        operation_id = self.cache_controller.write_storage(
            host_value, key, hash_value, prefix_keys, **self._get_extra_pools()
        )
        self.ongoing_backup[operation_id] = node
        node.protect_host()

    def _concat_split_chain(self, node: TreeNode, backup_len: int):
        """Recover enqueue-time key/hash/host by walking the split chain."""
        chain, accumulated = [], 0
        current = node
        while current is not self.root_node and accumulated < backup_len:
            chain.append(current)
            accumulated += len(current.key)
            current = current.parent
        assert accumulated == backup_len, (
            f"backup chain length mismatch for node {node.id}: "
            f"expected {backup_len}, got {accumulated}"
        )
        chain.reverse()  # parent-first
        top = chain[0]
        if top.key.is_bigram:
            # Bigram segments share boundary tokens; drop overlap after first.
            token_ids = list(chain[0].key.token_ids)
            for n in chain[1:]:
                token_ids.extend(n.key.token_ids[1:])
        else:
            token_ids = []
            for n in chain:
                token_ids.extend(n.key.token_ids)
        key = RadixKey(token_ids, top.key.extra_key, top.key.is_bigram)

        if all(n.hash_value is not None for n in chain):
            hash_value = []
            for n in chain:
                hash_value.extend(n.hash_value)
        else:
            hash_value = None
        host_value = torch.cat([n.host_value for n in chain])
        return top, key, hash_value, host_value

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        """重写父类 RadixCache._inc_hit_count，增加 HiCache 的 write-through 自动触发逻辑。

        父类仅递增 hit_count（chunked 时跳过防止虚增）；
        本方法额外：
        1. write_back 模式下也跳过 —— 此时节点在 evict 时才写 Host，无需靠 hit_count 触发。
        2. 非 backuped 节点命中数达到 write_through_threshold 时，自动触发 write_backup 将 KV 写回 Host。
           这是 write_through 策略的核心：热门节点自动落 Host，后续可直接从 Host 加载无需重算。
        """
        # write_back 模式下不跟踪 hit_count（写 Host 由 evict 驱动）；
        # chunked 请求也跳过，防止同一请求多 chunk 反复命中自己创建的节点导致虚增
        if self.cache_controller.write_policy == "write_back" or chunked:
            return

        # 父类的 hit_count 只是驱逐策略的优先级参考（热门节点推迟驱逐）；
        # 子类把它升级为 write-through 自动触发器——当一个节点足够热门（命中次数达标），
        # 就自动把它异步写回 Host，后续请求可直接从 Host 加载而无需重算。
        node.hit_count += 1

        # hit_count（历史命中次数） —— 热度统计
        # 含义：节点自创建以来，一共被多少个请求匹配到过
        # 只增不减：每次匹配 _inc_hit_count +1，永不减少
        # 软性排序：不阻止驱逐，只在驱逐策略中影响优先级

        # lock_ref（引用计数） —— 即时安全锁
        # 含义：当前有多少个活跃请求正在使用这个节点
        # 有增有减：请求开始时沿路径 inc_lock_ref，完成时 dec_lock_ref
        # 硬性保护：lock_ref > 0 时节点绝对不能被驱逐
        # 沿祖先路径生效：对一个节点加锁，它的所有祖先也会被加锁

        # 节点尚未在 Host 有备份，且命中数达到阈值 → 触发异步写回 Host
        # ⚠️ 注意：_inc_hit_count 可能在没有 inc_lock_ref(node) 的情况下被调用
        # （例如 insert 创建新节点时，line ~2240）。此时节点仅靠祖先节点的 lock_ref
        # 间接保护，未被直接锁定。write_backup(node) 内部会补 inc_lock_ref(node)
        # 来保护 DMA 飞行期间的节点（line ~1009）。
        if not node.backuped:
            if node.hit_count >= self.write_through_threshold:
                self.write_backup(node)

    def writing_check(self, write_back=False):
        """轮询并收割已完成的 GPU→Host DMA 写入。

        两种写入策略（write_policy）对比：
        ┌──────────────┬────────────────────────────────┬──────────────────────────────────┐
        │              │  write_through                 │  write_back                      │
        ├──────────────┼────────────────────────────────┼──────────────────────────────────┤
        │ 触发时机       │ _inc_hit_count 命中达标时        │ evict(GPU) 驱逐时                 │
        │              │   write_through_threshold=1    │                                  │
        │ 写后行为       │ KV 同时存在于 GPU+Host           │ 写完即 _evict_backuped 释放 GPU   │
        │              │   后续 evict 直接降级，无需再写    │   节点降级为 evicted 状态          │
        │ writing_check│ 非阻塞（write_back=False）       │ 阻塞（write_back=True）           │
        │              │   本轮收割已完成的，不等 pending   │   必须等全部 pending 完成           │
        └──────────────┴────────────────────────────────┴──────────────────────────────────┘

        Args:
            write_back: True=阻塞等待全部 DMA 完成（evict 场景），False=非阻塞收割（write_through 场景）
        """
        if write_back:
            # evict 场景：必须等待所有 pending DMA 完成才能安全释放 GPU slot。
            # 阻塞循环直到 ongoing_write_through 清空。
            while len(self.ongoing_write_through) > 0:
                # 遍历 ack_write_queue，同步等待每个 finish_event
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()          # 阻塞等待 GPU→Host DMA 完成
                    for ack_id in ack_list:
                        self._finish_write_through_ack(ack_id, release_lock=False)
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # write_through / 非 evict 场景：非阻塞收割。
        # 所有 rank 的 ongoing_write_through 一致，空时跳过 all_reduce 同步。
        if len(self.ongoing_write_through) == 0:
            return

        # ── 分布式同步收割：多 GPU pipeline parallelism 场景 ──
        # 每个 rank 独立发起 GPU→Host DMA，完成时间可能不同。
        # 不能某个 rank 擅自收割（release lock_ref），否则其他 rank 的 DMA 还在飞。
        # 因此需要 pp_rank=0 先统计 → all_reduce(MIN) 取各 rank 都完成的数量 → FIFO 收割。
        #
        # 示例：rank0 看到 5 个完成，rank1 只看到 3 个 → all_reduce(MIN)=3，安全收割前 3 个。
        # 场景：write_through 模式下，DMA 异步飞行中，周期调用 writing_check 清理已完成的。
        #
        # pp_rank=0 统计已完成的 DMA 数量（按顺序检查，遇到未完成的就停止）
        finish_count = 0
        if self.pp_rank == 0:
            for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                if not finish_event.query():            # query() 非阻塞，true=已完成
                    break                               # 按序检查，遇到未完成就停止
                finish_count += 1

        # all_reduce(MIN)：跨所有 TP+PP rank 取最小完成数，保证一致性。
        # 通信分两步：
        #   1. pp_rank=0 的 TP 组内 all_reduce(MIN) → PP0 上所有注意力 rank 达成一致
        #   2. _pp_sync 广播 PP0 结果到 PP1, PP2, ... → 所有 PP rank 拿到相同值
        # 效果：finish_count = min(所有 TP 和 PP rank 中各自统计的完成数)
        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
        finish_count = finish_count_tensor.item()

        if finish_count > 0:
            logger.debug(f"Process {finish_count} write back operations")

        # 📌按 FIFO 顺序收割已完成的 DMA —
        #   synchronize() 确认 GPU→Host DMA 真正完成（防假完成）
        #   → _finish_write_through_ack 清理 ongoing_write_through + 发 Store 事件
        #   → dec_lock_ref 释放 write_backup 时加的锁（release_lock=True）
        #
        # 📌 ack_write_queue vs ongoing_write_through 的关系：
        #   ack_write_queue 每条记录 = 一次 flush_write DMA 操作
        #   ongoing_write_through 每条记录 = 一个树节点
        #   一次 DMA 可能合并写多个节点（merge），所以 ack_list 有多个 ack_id，
        #   每个 ack_id 对应 ongoing_write_through 中的一条。常见情况是 1:1。
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()                  # 阻塞确认 DMA 物理完成
            for ack_id in ack_list:
                # 📌 _finish_write_through_ack 收尾四件事：
                #   1. ongoing_write_through.pop(ack_id)    ← 清除 pending 记录
                #   2. node.write_through_pending_id = None ← DMA 确认，Host 已有副本
                #   3. _record_store_event(CPU)             ← 通知下游索引器
                #   4. if enable_storage: write_backup_storage ← 写第三层存储
                #
                # 📌 release_lock=True 的原因：
                #   write_through 模式下 write_backup 执行了 inc_lock_ref(node)
                #   防止 DMA 飞行期间节点被 evict，这里 DMA 确认完成后释放锁。
                #   对比 write_back 模式（line ~1113）release_lock=False：
                #   write_back 的 write_backup 没有 inc_lock_ref（紧随其后就 evict），无需释放。
                self._finish_write_through_ack(ack_id, release_lock=True)
            finish_count -= 1

    def loading_check(self):
        """轮询并收割已完成的 Host→GPU DMA 加载（load_back 的异步收尾）。

        writing_check 的对称方法，处理反向数据传输（Host→GPU）。
        同样是多 GPU 分布式场景：各 rank DMA 完成时间不同，
        pp_rank=0 统计 → all_reduce(MIN) → FIFO 收割。

        writing_check 收尾时：_finish_write_through_ack → dec_lock_ref + Storage 持久化
        loading_check 收尾时：ongoing_load_back.pop → dec_lock_ref 即可
        （load_back 不涉及 Host→Storage 持久化）
        """
        # pp_rank=0 统计已完成的 load DMA 数量（按序检查，遇到未完成就停止）
        finish_count = 0
        if self.pp_rank == 0:
            for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
                if not finish_event.query():            # query() 非阻塞，true=已完成
                    break                               # 按序检查，遇到未完成就停止
                finish_count += 1

        # all_reduce(MIN)：取所有 rank 中完成的 min 值，保证跨 rank 一致
        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
        finish_count = finish_count_tensor.item()

        if finish_count > 0:
            logger.debug(f"Process {finish_count} load operations")

        # 按 FIFO 顺序收割：同步等待 → 清除 pending 状态 → 释放 lock_ref
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_load_queue.pop(0)
            finish_event.synchronize()                  # 确认 Host→GPU DMA 完成
            for ack_id in ack_list:
                end_node = self.ongoing_load_back.pop(ack_id)  # 移除 pending 记录
                self.dec_lock_ref(end_node)                      # 释放 load_back 时加的锁
            finish_count -= 1

    def is_load_back_event_done(self, consumer_index: int) -> bool:
        """Return True after the local load-back event is complete."""
        if consumer_index < 0:
            return True

        finish_event = self.cache_controller.layer_done_counter.events[
            consumer_index
        ].finish_event
        if not finish_event.query():
            return False

        self.loading_check()
        return True

    def evictable_size(self):
        return self.evictable_size_

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        """引用计数 +1（沿路径到 root），同时更新 GPU 和 Host 两套可驱逐叶节点集合。

        与 RadixCache.inc_lock_ref 的唯一区别：
        多了 _update_host_leaf_status(node) 调用。
        RadixCache 只需维护 evictable_leaves（GPU 层），
        HiRadixCache 额外维护 evictable_host_leaves（Host 层），
        确保 Host 内存不足时也能正确驱逐 Host 端 KV。
        除此之外，lock_ref 的增减逻辑、evictable_size_/protected_size_ 统计完全一致。
        """
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                # 从可驱逐变为受保护 → 更新计数器
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)        # GPU 层：更新 evictable_leaves
            self._update_host_leaf_status(node)    # Host 层：更新 evictable_host_leaves（HiRadixCache 独有）
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        """引用计数 -1（沿路径到 root），同时更新 GPU 和 Host 两套可驱逐叶节点集合。

        与 RadixCache.dec_lock_ref 的唯一区别：
        多了 _update_host_leaf_status(node) 调用，原因同 inc_lock_ref。

        额外维护 node.parent is None 的断言（RadixCache 也有），
        防止跨树引用（不同 radix tree 实例）。
        """
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                # 从受保护变为可驱逐 → 更新计数器
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)        # GPU 层：更新 evictable_leaves
            self._update_host_leaf_status(node)    # Host 层：更新 evictable_host_leaves（HiRadixCache 独有）
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def _update_host_leaf_status(self, node: TreeNode):
        if not node.evicted or node.lock_ref > 0:
            if node in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(node)
            return

        for child in node.children.values():
            if child.backuped:
                if node in self.evictable_host_leaves:
                    self.evictable_host_leaves.remove(node)
                return

        if node not in self.evictable_host_leaves:
            self.evictable_host_leaves.add(node)

    def evict(self, params: EvictParams) -> EvictResult:
        """驱逐 GPU 端 KV 以释放显存。

        📌 完整调用链（从 scheduler 到 evict）：
        scheduler.run_batch()
          → prepare_for_decode / prepare_for_extend       (schedule_batch.py)
            → alloc_for_decode / alloc_for_extend         (common.py)
              → alloc_token_slots / alloc_paged_token_slots_extend  (common.py:272)
                → evict_from_tree_cache(tree_cache, N)    (common.py:302)
                  → tree_cache.evict(EvictParams(num_tokens=N))   ← 你在这
        scheduler 在每次 alloc KV slot 之前检查显存。若不够，
        先 evict 腾空间，再 alloc。这是 write_back / CPU offload 的核心触发点。

        📌 阶段 0: 收集可驱逐叶子 → 按优先级建最小堆
        📌 阶段 1: 循环驱逐
           ├── lock_ref>0 → 跳过
           ├── 未 backuped + write_back → write_backup(DMA)，记录到 write_back_nodes
           ├── 未 backuped + write_through → _evict_regular(真删除)
           └── 已 backuped → _evict_backuped(降级到 Host)
        📌 阶段 2: write_back 两阶段收尾
           └── writing_check(等 DMA 完成) → _evict_backuped(释放 GPU)

        ╔════════════════════════════════════════════════════════════════════════════╗
        ║  evict 的两条路径                                                             ║
        ╠════════════════════════════════════════════════════════════════════════════╣
        ║                                                                              ║
        ║  RadixCache.evict：                                                          ║
        ║    直接 free(node.value) + _delete_leaf  → 节点从树中删除（真删除）            ║
        ║                                                                              ║
        ║  HiRadixCache.evict：                                                        ║
        ║    ┌──────────────────────────────────────────────────────────────────┐       ║
        ║    │ node.backuped?                                                  │       ║
        ║    │   ├── Yes → _evict_backuped: 只释放 GPU slot                    │       ║
        ║    │   │         node.value = None, evicted=True                     │       ║
        ║    │   │         节点仍在树中，host_value 保留，可 load_back 恢复       │       ║
        ║    │   │                                                              │       ║
        ║    │   └── No  → write_back 模式？                                    │       ║
        ║    │         ├── Yes → write_backup(node) 先备份到 Host              │       ║
        ║    │         │         写完后 _evict_backuped（降级）                  │       ║
        ║    │         │                                                        │       ║
        ║    │         └── No  → _evict_regular: 真删除（同 RadixCache）         │       ║
        ║    │                   free + _delete_leaf，节点从树中移除              │       ║
        ║    └──────────────────────────────────────────────────────────────────┘       ║
        ╚════════════════════════════════════════════════════════════════════════════╝
        """
        start_time = time.perf_counter()
        num_tokens = params.num_tokens

        # ── 阶段 0: 收集可驱逐的叶节点 ──
        # 可驱逐叶子：lock_ref=0 且没有 GPU 上存在的子节点
        leaves = list(self.evictable_leaves)

        # 按驱逐策略优先级构建最小堆（优先驱逐"冷"节点）
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        write_back_nodes = []  # write_back 模式下先写 Host 再 evict 的节点（两阶段）

        # ── 阶段 1: 按优先级循环驱逐 ──
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            # 跳过被锁定的节点（正在被请求使用）
            if x.lock_ref > 0:
                continue

            if not x.backuped:
                if self.cache_controller.write_policy == "write_back":
                    # write_back 模式：GPU→Host DMA，先备份，稍后 evict
                    written = self.write_backup(x, write_back=True)
                    num_evicted += written
                    if written > 0:
                        write_back_nodes.append(x)
                else:
                    # write_through 模式下未 backuped → 不该出现（write_through 会主动备份）
                    # 若出现说明 write_backup 失败后节点仍未 backuped → 真删除
                    num_evicted += self._evict_regular(x)
            else:
                # 已在 Host 有副本 → 直接释放 GPU slot（降级，节点仍在树中）
                num_evicted += self._evict_backuped(x)

            # ── 检查父节点是否变成新的可驱逐叶子 ──
            # 驱逐一个节点后，如果父节点的所有子节点都被驱逐了，
            # 父节点也变成"叶子"（可以被驱逐），加入堆中。
            for child in x.parent.children.values():
                if child in write_back_nodes:
                    continue  # write_back 节点还没真正 evict，不算
                if not child.evicted:
                    break       # 还有子节点在 GPU，父节点不能驱逐
            else:
                # 所有子节点都已被驱逐（或无子节点）→ 父节点加入候选堆
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

        # ── 阶段 2: write_back 模式下的两阶段收尾 ──
        # write_back 的 write_backup 只是发起了 DMA，数据还在飞行。
        # 先等 DMA 完成（writing_check），再 evict GPU slot。
        if self.cache_controller.write_policy == "write_back":
            self.writing_check(write_back=True)    # 等待所有 write_back DMA 完成
            for node in write_back_nodes:
                assert node.backuped               # DMA 成功后 host_value 非空
                self._evict_backuped(node)          # 释放 GPU slot，节点降级到 Host

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _evict_backuped(self, node: TreeNode):
        """驱逐已备份到 Host 的节点：只释放 GPU slot，节点降级为 evicted 状态。
        ╔══════════════════════════════════════════════════════════════╗
        ║  驱逐前后节点状态变化                                          ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  驱逐前：value=[gpu_slots], host_value=[host_slots]           ║
        ║  驱逐后：value=None,        host_value=[host_slots]  不变     ║
        ║          evicted=False → evicted=True                        ║
        ║          backuped=True  → backuped=True   不变               ║
        ║                                                                ║
        ║  对比 _evict_regular（真删除）：                                ║
        ║    free + _delete_leaf，节点从树中移除，host_value 也没了      ║
        ╚══════════════════════════════════════════════════════════════╝
        """
        self._record_remove_event(node, medium=StorageMedium.GPU)
        num_evicted = self.cache_controller.evict_device(node.value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        self._update_leaf_status(node)
        self._update_host_leaf_status(node)
        # update leaf status for the parent because the node is evicted
        self._update_leaf_status(node.parent)
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        """真删除：节点既不在 Host 有备份，直接从树中移除。

        只有在非 write_back 模式下，且节点没有 backuped 时才会走这条路。
        效果等同于 RadixCache 的 evict：free GPU slot + _delete_leaf。
        """
        # evict a node not initiated write to host -- emit BlockRemoved
        assert len(node.children) == 0, f"non-leaf, {node.id=}"

        self._record_remove_event(node)
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def evict_host(self, num_tokens: int):
        """驱逐 Host 端的 KV 缓存（Host→无），当 Host 内存不足时触发。

        ╔════════════════════════════════════════════════════════════════════════════╗
        ║  evict_host 只驱逐"GPU 已驱逐 + Host 还有副本"的节点                          ║
        ╠════════════════════════════════════════════════════════════════════════════╣
        ║                                                                              ║
        ║  三级驱逐链：                                                                ║
        ║    evict(GPU)   → 节点在 Host，evicted=True，可 load_back 恢复              ║
        ║    evict_host   → 节点从树中删除（真删除），Host KV 也丢了                    ║
        ║                                                                              ║
        ║  前提：只有 evicted=True 的节点才能被 evict_host                               ║
        ║        （如果 GPU 还有 value，说明还在用，不能删 Host 副本）                   ║
        ║        host_ref_counter > 0 的也不能删（有 prefetch 在用）                    ║
        ╚════════════════════════════════════════════════════════════════════════════╝
        """
        leaves = list(self.evictable_host_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            if not x.evicted:
                continue

            if x.host_ref_counter > 0:
                continue

            # Block deleted entirely (GPU already evicted, now CPU freed) --
            # emit remove(CPU) so the router drops the host-tier entry.
            self._record_remove_event(x, medium=StorageMedium.CPU)
            num_evicted += self.cache_controller.evict_host(x.host_value)

            key = x.key.child_key(self.page_size)
            v = x.parent.children.pop(key, None)
            assert v == x, f"parent does not have child key, {key}"
            if x in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(x)
            self._update_host_leaf_status(x.parent)

            if len(x.parent.children) == 0 and x.parent.evicted:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """将节点的 KV 从 Host 加载回 GPU（Host→GPU DMA），恢复 evicted 状态。

        ╔════════════════════════════════════════════════════════════════════════════╗
        ║  load_back 工作原理                                                          ║
        ╠════════════════════════════════════════════════════════════════════════════╣
        ║                                                                              ║
        ║  场景：match_prefix 匹配到一个 evicted 节点，需要把 KV 从 Host 搬回 GPU        ║
        ║                                                                              ║
        ║  树结构示例：A(root) → B → C → D                                             ║
        ║    B: value=[10,11], host_value=None, evicted=False                          ║
        ║    C: value=None,    host_value=[20,21], evicted=True                        ║
        ║    D: value=None,    host_value=[30,31], evicted=True                        ║
        ║                                                                              ║
        ║  load_back(D) 时：                                                            ║
        ║    1. 从 D 往上走，收集所有连续 evicted 节点 → nodes_to_load = [C, D]         ║
        ║    2. 找到最近的非 evicted 祖先 → ancestor_node = B                           ║
        ║    3. inc_lock_ref(B)：防止 B 在 DMA 期间被 evict                             ║
        ║    4. cat C.host_value + D.host_value → 一笔 DMA 传完                         ║
        ║    5. cache_controller.load(host_indices) → device_indices                   ║
        ║    6. C.value = device_indices[0:2], D.value = device_indices[2:4]            ║
        ║    7. C.evicted=False, D.evicted=False                                       ║
        ║    8. inc_lock_ref(D)：防止刚加载回来就被 evict                               ║
        ╚════════════════════════════════════════════════════════════════════════════╝

        GPU 内存不足时：
          先调 evict 释放其他节点的 GPU KV，再重试一次
        过短或超 mem_quota 时：
          跳过加载（不值得花 DMA 开销）
        """

        start_time = time.perf_counter()
        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        result = self.inc_lock_ref(ancester_node)
        delta = result.delta

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices,
            node_id=last_hit_node.id,
            **self._get_extra_pools(),
        )
        if device_indices is None:
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load(
                host_indices=host_indices,
                node_id=last_hit_node.id,
                **self._get_extra_pools(),
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            logger.warning(
                "load_back: FAILED to load %d tokens for node %d "
                "even after eviction (evictable_size=%d)",
                len(host_indices),
                last_hit_node.id,
                self.evictable_size_,
            )
            return None

        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)].clone()
            offset += len(node.host_value)
            # Block promoted from host to GPU -- emit store(GPU) so downstream
            # indexers see it as device-local again.
            self._record_store_event(node, medium=StorageMedium.GPU)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        return device_indices

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ):
        last_node = params.best_match_node
        mem_quota = params.mem_quota
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            self._empty_match_result.device_indices,
            last_node,
        )

    def query_storage_hit_length(
        self,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> int:
        if not self.enable_storage or self.cache_controller.prefetch_rate_limited():
            return 0

        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=last_host_node.key.extra_key,
            is_bigram=self.is_eagle,
        ).page_aligned(self.page_size)
        if len(prefetch_key) < self.prefetch_threshold:
            return 0

        operation = PrefetchOperation(
            "__storage_hit_query__",
            self.cache_controller.mem_pool_host.get_dummy_flat_data_page()[:0],
            prefetch_key,
            last_hash,
            prefix_keys,
        )
        hash_values, storage_hit_count = self.cache_controller._storage_hit_query(
            operation
        )
        storage_hit_count_tensor = torch.tensor(storage_hit_count, dtype=torch.int)
        self._all_reduce_attn_groups(
            storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
        )
        storage_hit_count = storage_hit_count_tensor.item()
        storage_hit_count = storage_hit_count - (storage_hit_count % self.page_size)
        return storage_hit_count

    def ready_to_load_host_cache(self) -> int:
        """
        Notify the cache controller to start the KV cache loading.
        Return the consumer index for the schedule batch manager to track.
        """
        return self.cache_controller.start_loading()

    def flush_write_through_acks(self) -> None:
        self.writing_check()

    def check_hicache_events(self):
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
        self._reap_completed_async_work()
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def drain_storage_control_queues(self):
        """
        Combine prefetch revoke, backup ack, and host mem release checks
        to minimize TP synchronization and Python overhead.
        """
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                cc.prefetch_revoke_queue.qsize(),
                cc.ack_backup_queue.qsize(),
                cc.host_mem_release_queue.qsize(),
            ],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(qsizes, torch.distributed.ReduceOp.MIN)

        n_revoke, n_backup, n_release = map(int, qsizes.tolist())
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_backup=n_backup,
            n_release=n_release,
            log_metrics=True,
        )

    # Timeout is linearly increasing with the number of pages
    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation):
        cfg = self.prefetch_timeout_config
        num_tokens = len(operation.hash_value) * self.page_size
        timeout = min(cfg.max, cfg.base + cfg.per_ki_token * num_tokens / 1024)
        return time.monotonic() - operation.start_time > timeout

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        can_terminate = True

        if self.prefetch_stop_policy == "best_effort":
            return can_terminate

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self.is_prefetch_timeout(operation)
        else:
            # unknown prefetch stop policy, just return True
            return True

        if (
            completed
            and getattr(operation, "pool_transfers", None)
            and not getattr(operation, "pool_transfers_done", True)
        ):
            can_terminate = False

        operation_terminated = operation.is_terminated()
        states = torch.tensor(
            [1 - int(can_terminate), int(operation_terminated)],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(states, torch.distributed.ReduceOp.MAX)
        can_terminate = states[0].item() == 0
        operation_terminated = states[1].item() == 1
        # the operation should be terminated if it is already terminated on any TP worker
        # or it meets the termination condition on all TP workers
        can_terminate = can_terminate or operation_terminated
        return can_terminate

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            # there is no ongoing prefetch for this request or it has been revoked
            return True

        # todo: more policies for prefetch progress such as timeout
        # the current policy is to prefetch with best effort and terminate when queuing is over
        last_host_node, prefetch_key, host_indices, operation = self.ongoing_prefetch[
            req_id
        ]

        if operation.host_indices is None:
            # prefetch has not been issued due to insufficient host memory
            return True

        if not self.can_terminate_prefetch(operation):
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        logger.debug(f"Prefetch {req_id} completed with {completed_tokens} tokens")

        min_completed_tokens = completed_tokens
        # Synchronize workers before mutating host cache tree state.
        completed_tokens_tensor = torch.tensor(min_completed_tokens, dtype=torch.int)
        self._all_reduce_attn_groups(
            completed_tokens_tensor, torch.distributed.ReduceOp.MIN
        )
        min_completed_tokens = completed_tokens_tensor.item()
        fetched_key = prefetch_key[:min_completed_tokens]
        written_indices = host_indices[:min_completed_tokens]
        matched_length = self._insert_helper_host(
            last_host_node,
            fetched_key,
            written_indices,
            hash_value[: min_completed_tokens // self.page_size],
        )

        self.cache_controller.mem_pool_host.free(host_indices[:matched_length])
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )
        last_host_node.release_host()
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)

        # Track tokens actually loaded from storage for this request (L3 hits)
        loaded_from_storage = min_completed_tokens - matched_length
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage

        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)

        return True

    def terminate_prefetch(self, req_id: str):
        if req_id not in self.ongoing_prefetch:
            return

        _, _, _, operation = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        """
        Pop and return the number of tokens loaded from storage for a request.
        Returns 0 if no prefetch was done or was revoked.
        This should be called after check_prefetch_progress() returns True.
        """
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def match_prefix(self, params: MatchPrefixParams):
        """在 radix tree 中查找与 key 匹配的最长前缀。

        ╔════════════════════════════════════════════════════════════════════════════╗
        ║  与 RadixCache.match_prefix 的区别                                            ║
        ╠════════════════════════════════════════════════════════════════════════════╣
        ║                                                                              ║
        ║  RadixCache 返回：                                                            ║
        ║    device_indices, last_device_node                                           ║
        ║    （匹配到多长就返回多长，简单直接）                                            ║
        ║                                                                              ║
        ║  HiRadixCache 返回：                                                          ║
        ║    device_indices   ← 匹配链上非 evicted 节点的 value 拼接                    ║
        ║    last_device_node ← 最深的非 evicted 节点                                   ║
        ║    last_host_node   ← 最深的 backuped 节点（为 load_back/prefetch 提供起点）   ║
        ║    host_hit_length  ← 被驱逐到 Host 的 token 数                               ║
        ║                                                                              ║
        ║  示例树：A(root) → B → C → D                                                  ║
        ║    B: evicted=False, backuped=False                                           ║
        ║    C: evicted=True,  backuped=True                                            ║
        ║    D: evicted=True,  backuped=True                                            ║
        ║                                                                              ║
        ║  匹配到 D 时：                                                                ║
        ║    device_indices = B.value  （C、D 被 evicted，value=None，不包含）           ║
        ║    last_device_node = B      （最深的非 evicted）                              ║
        ║    last_host_node = D        （最深的 backuped，load_back 从这里开始）          ║
        ║    host_hit_length = len(C) + len(D) （C、D 在 Host 上）                      ║
        ║                                                                              ║
        ║  后续调度器根据 host_hit_length 决定是否 load_back                             ║
        ╚════════════════════════════════════════════════════════════════════════════╝
        """
        if self.disable:
            return self._empty_match_result

        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)  # Eagle 模式下转 bigram 视图
        key = key.page_aligned(self.page_size)              # 对齐到 page 边界（截断尾部不足一页的部分）
        if len(key) == 0:
            return self._empty_match_result

        # 沿 radix tree 匹配前缀，收集匹配节点的 value（GPU slot 索引）
        # ⚠️ value只含有device上存在的indices
        # ⚠️ last node是完整的匹配链的尾端node
        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)                        # 拼接各匹配节点的 device_indices
        else:
            value = self._empty_match_result.device_indices

        # 从 last_node 向上回溯，累计被 evicted（KV 在 Host 不在 GPU）的 token 数
        host_hit_length = 0
        last_host_node = last_node                          # 先记住最深的匹配节点
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)    # 累加 Host 端的 slot 数
            last_node = last_node.parent                    # 继续往上找，直到遇到非 evicted 节点
        # ⚠️ 排除末端的evicted节点，last_node 现在是最深的仍在 GPU 上的匹配节点（last_device_node）

        while not last_host_node.backuped:
            last_host_node = last_host_node.parent          # 找到最近的有 Host 副本的祖先
        # ⚠️ 排除末端的非backuped节点，last_host_node是最近的有 Host 副本的祖先

        return MatchResult(
            # ⚠️ value只含有device上存在的indices
            device_indices=value,
            # ⚠️ 排除末端的evicted节点，last_node 现在是最深的仍在 GPU 上的匹配节点（last_device_node）
            last_device_node=last_node,                     # 最深仍在 GPU 上的匹配节点
            # ⚠️ 排除末端的非backuped节点，last_host_node是最近的有 Host 副本的祖先
            last_host_node=last_host_node,                  # 最近有 Host 副本的祖先，load_back 起点
            # TODO(ispoblock): use best_match_node as start node for load_back
            best_match_node=last_host_node,
            # evicted不代表backuped吧？
            host_hit_length=host_hit_length,                # Host 端命中的 token 数（需 load_back 的量）
        )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=last_host_node.key.extra_key,
            is_bigram=self.is_eagle,
        )
        # align the number of fetching tokens to the page size
        prefetch_key = prefetch_key.page_aligned(self.page_size)
        prefetch_length = len(prefetch_key)
        if (
            not self.enable_storage
            or prefetch_length < self.prefetch_threshold
            or self.cache_controller.prefetch_rate_limited()
        ):
            return

        last_host_node.protect_host()
        host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            self.evict_host(prefetch_length)
            host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            available_size = self.cache_controller.mem_pool_host.available_size()
            prefetch_length = available_size - (available_size % self.page_size)
            if prefetch_length >= self.prefetch_threshold:
                prefetch_key = prefetch_key[:prefetch_length]
                host_indices = self.cache_controller.mem_pool_host.alloc(
                    prefetch_length
                )
                if host_indices is None:
                    last_host_node.release_host()
                    return
            else:
                last_host_node.release_host()
                # no sufficient host memory for prefetch
                return
        operation = self.cache_controller.prefetch(
            req_id,
            host_indices,
            prefetch_key,
            last_hash,
            prefix_keys,
            **self._get_extra_pools(),
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
        )
        self.cache_controller.prefetch_tokens_occupied += len(prefetch_key)

    def _insert_helper_host(
        self, node: TreeNode, key: RadixKey, host_value, hash_value
    ):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)

        matched_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = node.key.match(key, page_size=self.page_size)
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=node.priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = None
            new_node.host_value = host_value.clone()
            new_node.hash_value = hash_value
            node.children[child_key] = new_node
            self._update_host_leaf_status(new_node)
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            # Publish the newly materialized host suffix immediately so downstream
            # cache indexers can resolve descendants that extend this L2-only prefix.
            self._record_store_event(new_node, medium=StorageMedium.CPU)

        return matched_length

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()
        child_key = key.child_key(self.page_size)  # 取 key 的第一页作为子节点查找键
        value = []                                  # 收集匹配节点的 GPU slot 索引

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = child.key.match(key, page_size=self.page_size)  # 计算当前 child 与 key 的公共前缀长度

            if prefix_len < len(child.key):
                # 部分匹配：child 的 key 比 key 的前缀长，需要分裂 child
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)     # 分裂后的新节点持有公共前缀部分
                node = new_node
                break                                # 前缀匹配到此结束
            else:
                # 完全匹配：child 的 key 是 key 的前缀，继续往下走
                if not child.evicted:
                    value.append(child.value)         # 只收集 KV 在 GPU 上的节点
                node = child
                key = key[prefix_len:]                # 剥离已匹配部分，继续匹配剩余 key

                if len(key):
                    child_key = key.child_key(self.page_size)  # 剩余 key 的第一页作为下一步查找键

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        """重写父类 RadixCache._split_node，增加 HiCache 三层存储（GPU/Host）的分裂逻辑。

        父类仅处理 GPU value 的拆分；本方法额外处理：
        1. evicted 状态：子节点 KV 被驱逐到 Host 时，共享前缀 new_node 的 GPU value 置 None。
        2. backuped 状态：分裂 host_value，保持 Host 层 KV 与树结构一致。
        3. write-through pending 队列：替换待写队列中的 old child 为拆分后的两个节点。
        """
        # 创建共享前缀节点（new_node），作为原 child 的父节点
        new_node = TreeNode(priority=child.priority)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

        # GPU 层 value 分裂 -
        #   child.evicted → GPU 上没有 KV，共享前缀节点 value 为 None
        #   child 未 evicted → 正常拆分 value
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len].clone()
            child.value = child.value[split_len:].clone()

        # Host 层 host_value 分裂 -
        #   child 在 Host 有备份时，同步拆分 host_value
        if child.backuped:
            new_node.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()

        # hash_value 拆分（用于 KV cache 事件路由/去重）
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        # 重定向父子关系：new_node 插入 child 与 grandparent 之间
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        # 📌 可能该节点还处于 GPU → CPU的DMA搬运中
        # 更新 write-through 待写队列：旧的 child 被替换为 [new_node, child]
        if child.backuped:
            self._replace_pending_write_through_node(child, [new_node, child])

        return new_node

    def insert(self, params: InsertParams) -> InsertResult:
        """将 KV 缓存写入 radix tree。

        ╔════════════════════════════════════════════════════════════════════════════╗
        ║  与 RadixCache.insert 的区别：evicted 节点的复活                              ║
        ╠════════════════════════════════════════════════════════════════════════════╣
        ║                                                                              ║
        ║  RadixCache：节点匹配到就 inc_hit_count，不修改 value                          ║
        ║                                                                              ║
        ║  HiRadixCache：遇到 evicted 节点时，需要"复活"：                               ║
        ║    node.value = value[:prefix_len].clone()   ← KV 重新回到 GPU               ║
        ║    evictable_size_ += len(node.value)         ← GPU 可驱逐量增加              ║
        ║    node.evicted → False                       ← 节点恢复为 GPU 状态           ║
        ║                                                                              ║
        ║  为什么需要复活？                                                              ║
        ║    场景：请求 R1 的 KV 被驱逐到 Host（evicted=True），                         ║
        ║    新请求 R2 与 R1 共享前缀，insert 时匹配到 evicted 节点，                     ║
        ║    R2 的 value 已经在 GPU 上了（调度时分配了新 slot），                         ║
        ║    所以直接用 R2 的 value 恢复 node.value，无需 load_back。                    ║
        ║    这比 load_back（Host→GPU DMA）更快，因为 R2 的 KV 已经在 GPU 了。           ║
        ╚════════════════════════════════════════════════════════════════════════════╝
        """
        key = params.key
        value = params.value
        chunked = params.chunked
        priority = params.priority

        if priority is None:
            priority = 0

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)  # Eagle 模式下转 bigram 视图
        key = key.page_aligned(self.page_size)                        # 对齐到 page 边界
        if value is not None:
            value = value[: len(key)]                                 # 截断 value 与 key 等长

        if len(key) == 0:
            return InsertResult(prefix_len=0)

        node = self.root_node
        child_key = key.child_key(self.page_size)                     # 取 key 的第一页作为子节点查找键
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            node.priority = max(node.priority, priority)              # 继承最高优先级
            prefix_len = node.key.match(key, page_size=self.page_size)

            if prefix_len == len(node.key):
                # 完全匹配：node.key 是 key 的前缀
                if node.evicted:
                    # node 的 KV 之前被驱逐到 Host，现在新请求重新计算了 KV（已在 GPU），
                    # 直接用新 value 恢复 node.value，无需 load_back DMA。
                    # 注意：evicted 是 @property，等价于 self.value is None，
                    # 所以 node.value = ... 赋值后 node.evicted 自动变为 False，无需显式设置。
                    node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(node.value)           # 恢复后重新计入可驱逐大小
                    # 节点 KV 从 Host 恢复到 GPU，更新其 GPU 可驱逐叶子状态
                    self._update_leaf_status(node)
                    # 同步更新 Host 层可驱逐叶子状态（节点回到 GPU 后不再是 Host 候选叶子）
                    self._update_host_leaf_status(node)
                    # 父节点原本可能被子节点全部 evicted → 被标记为叶子；
                    # 现在子节点恢复回 GPU，父节点不再是叶子，需要重新计算其状态
                    self._update_leaf_status(node.parent)
                else:
                    # node 的 KV 还在 GPU，只需增加引用计数
                    self._inc_hit_count(node, chunked)
                    total_prefix_length += prefix_len
            else:
                # 部分匹配：node.key 比 key 的公共前缀长，需要分裂
                new_node = self._split_node(node.key, node, prefix_len)
                # shared-prefix node should also reflect max priority
                new_node.priority = max(new_node.priority, priority)
                if new_node.evicted:
                    # 分裂后的新节点也是 evicted 状态，用新 value 恢复
                    new_node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(new_node.value)
                    self._update_leaf_status(new_node)
                    self._update_host_leaf_status(new_node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(new_node.parent)
                else:
                    self._inc_hit_count(new_node, chunked)
                    total_prefix_length += prefix_len
                node = new_node                                        # 继续从分裂后的新节点往下

            key = key[prefix_len:]                                     # 剥离已匹配部分
            value = value[prefix_len:]

            if len(key):
                child_key = key.child_key(self.page_size)

        # 循环结束后 key 仍有剩余 → 树中没有匹配，创建新叶子节点
        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._update_leaf_status(node)                             # 父节点不再是叶子
            self._update_leaf_status(new_node)                         # 新节点是叶子

            # Compute hash_value if storage or kv events are enabled
            if self.enable_storage or self.enable_kv_cache_events:
                new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

            # Emit BlockStored so the router indexes this block.
            self._record_store_event(new_node)

            # write_through 模式下，新节点也算"命中"（会触发 write_backup）
            # write_back 模式下不 inc_hit_count，等 evict 时才写 Host
            if self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(new_node, chunked)
        return InsertResult(prefix_len=total_prefix_length)

    def release_aborted_request(self, rid: str):
        # Clean up storage hit tracking for aborted request
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)

        if rid not in self.ongoing_prefetch:
            return

        last_host_node, prefetch_key, host_indices, operation = self.ongoing_prefetch[
            rid
        ]
        if operation.host_indices is None:
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        self._barrier_attn_groups()
        last_host_node.release_host()
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(host_indices[:completed_tokens])
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)
