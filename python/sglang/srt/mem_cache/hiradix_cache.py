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

"""
    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  与 Scheduler 的完整交互流程                                                       ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                  ║
    ║  scheduler 每次迭代 = get_next_batch → run_batch → process_batch_result          ║
    ║  与 HiRadixCache 的交互贯穿在 get_next_batch 和 process_batch_result 中：           ║
    ║                                                                                  ║
    ║  ════════════ get_next_batch_to_run() 阶段 ════════════                            ║
    ║                                                                                  ║
    ║  ① check_hicache_events()        — 每轮调度前：                                    ║
    ║      ├─ writing_check()           非阻塞收割 write_through DMA（GPU→Host）         ║
    ║      └─ loading_check()           非阻塞收割 load_back DMA（Host→GPU）             ║
    ║                                                                                  ║
    ║  ② match_prefix(key)              — 查 radix tree，匹配最长前缀：                  ║
    ║      └─ 返回 MatchResult(device_indices, last_device_node, last_host_node, ...)   ║
    ║                                                                                  ║
    ║  ③ init_load_back(params)         — 若 best_match_node 在 Host 不在 GPU：         ║
    ║      └─ load_back(node) → cache_controller.load() 入队 → start_loading 逐层 DMA  ║
    ║                                                                                  ║
    ║  ④ inc_lock_ref(path)             — 锁定匹配路径，防驱逐                            ║
    ║                                                                                  ║
    ║  ⑤ prepare_for_extend / decode    — 分配 GPU KV slot：                            ║
    ║      └─ alloc → 不够 → evict_from_tree_cache → tree_cache.evict()                 ║
    ║           write_back: write_backup(DMA) → writing_check(⏳阻塞) → _evict_backuped  ║
    ║           write_through: 直接 _evict_backuped（insert 时已 write_backup）          ║
    ║                                                                                  ║
    ║  ════════════ process_batch_result() 阶段 ════════════                             ║
    ║                                                                                  ║
    ║  ⑥ cache_unfinished_req(req)       — prefill 后未完成的请求立即入树：              ║
    ║      └─ insert(key, value) → _inc_hit_count → write_through: write_backup(DMA)   ║
    ║         → dec_lock_ref(旧) + inc_lock_ref(新)  ← 锁交换                            ║
    ║                                                                                  ║
    ║  ⑦ cache_finished_req(req)         — 请求完成时入树 + 释放 slot：                  ║
    ║      └─ insert + dec_lock_ref + free overallocated                               ║
    ║                                                                                  ║
    ║  ════════════ 典型 Extend + Decode 时序 ════════════                               ║
    ║                                                                                  ║
    ║  Iter N (prefill):                                                                ║
    ║    ① check → ② match_prefix → ③ init_load_back → ④ inc_lock_ref                   ║
    ║    → ⑤ evict+alloc → run_batch → ⑥ cache_unfinished_req (insert+write_backup)    ║
    ║                                                                                  ║
    ║  Iter N+1 (decode):                                                               ║
    ║    ① check (收割上次 write_backup DMA) → ⑤ evict+alloc → run_batch                 ║
    ║    → (decode 不 cache_unfinished_req)                                              ║
    ║                                                                                  ║
    ║  Iter N+K (finished):                                                             ║
    ║    → ⑦ cache_finished_req (insert + free)                                        ║
    ║                                                                                  ║
    ║  🔑 关键区别：                                                                     ║
    ║    write_through: insert 时异步 DMA，evict 时不等待，writing_check 每轮非阻塞收割   ║
    ║    write_back:    insert 时不 DMA，evict 时才紧急 write_backup + 阻塞等待          ║
    ║    decode 每步不调 cache_unfinished_req（每步仅 1 token，page_size>1 时浪费）      ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
"""
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
        # 🔄 异步数据传输追踪器
        # HiCache 的多层数据流是异步(non-blocking DMA)的，
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
        # ⚙️ 阈值控制
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
        """🔧 Storage 控制队列排空的核心实现。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        ① 正常路径（TP 同步）:
          check_hicache_events → drain_storage_control_queues
            → all_reduce(MIN) → _drain_storage_control_queues_impl(n_revoke, n_backup, n_release, log_metrics=True)

        ② 本地清理路径（无 TP 同步，shutdown / detach 场景）:
          detach_storage_backend → _drain_storage_control_queues_local
            → _drain_storage_control_queues_impl(n_revoke=None, n_backup=None, n_release=None, log_metrics=False)
          shutdown → detach_storage_backend → ...

        ━━━━━━━━━━━━━━ 2️⃣ limit 参数语义 ━━━━━━━━━━━━━━
        limit=None（本地清理）: 排空队列直到 Empty，不做跨 rank 协调
        limit=N  （TP 同步）  : 只处理 N 条，由 all_reduce(MIN) 保证跨 rank 一致

        ━━━━━━━━━━━━━━ 3️⃣ 📌 核心：_drain_backup —— ongoing_backup 的取出时机 ━━━━━━━━━━━━━━
        _drain_backup() 遍历 ack_backup_queue（Storage 后端写入完成的确认队列）:
          ① ack_id = operation.id
          ② entry = ongoing_backup.pop(ack_id)   ← 从追踪器移除
          ③ entry.release_host()                  ← 解除 Host KV 保护，允许后续 evict_host 驱逐
          ④ storage_metrics_collector.log_backuped_tokens()  ← 记录指标

        这就是 ongoing_backup 注册（write_backup_storage）后，被取出的完整路径。

        ━━━━━━━━━━━━━━ 4️⃣ 🔄 三个子 drain 的处理顺序 ━━━━━━━━━━━━━━
        _drain_revoke → _drain_backup → _drain_release
        ① 先 revoke：取消进行中的 prefetch，释放 ongoing_prefetch 占用的 Host KV
        ② 再 backup：确认 Storage 写入完成，释放 ongoing_backup 保护的 Host KV
        ③ 最后 release：批量回收已释放的 Host 内存页（①和②可能产生新的待释放页）
        """
        cc = self.cache_controller

        def _drain_queue(q, limit: Optional[int]):
            """从队列中取出最多 limit 条（limit=None 则排空）。"""
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _drain_revoke():
            """🗑️ 撤销进行中的 prefetch：释放 Host KV 保护 + 回收 token 配额。

            prefetch_revoke_queue 中的 req_id 表示该请求的 prefetch 被调度器取消
            （如请求已完成 prefill、超时等），需要清理 ongoing_prefetch 中的记录。
            """
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                info = self.ongoing_prefetch.pop(req_id, None)
                if info is not None:
                    last_host_node, token_ids, _, _ = info
                    last_host_node.release_host()          # 解除 Host KV 保护
                    cc.prefetch_tokens_occupied -= len(token_ids)  # 回收 token 配额
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0

        def _drain_backup():
            """✅ 收割 Storage 写入完成的 ack：ongoing_backup.pop + release_host()。

            📌 这就是 ongoing_backup 的取出时机：
            Storage 后端（mooncake/EIC）完成 Host→L3 写入后，将 ack 放入 ack_backup_queue。
            _drain_backup 取出 ack → pop 追踪器 → release_host 解除保护。
            """
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                ack_id = operation.id
                # 📌 ongoing_backup 正常路径的取出点
                entry = self.ongoing_backup.pop(ack_id, None)
                if entry is not None:
                    entry.release_host()                   # Storage 写入完成，解除 Host KV 保护
                if log_metrics and self.enable_storage_metrics:
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )

        def _drain_release():
            """♻️ 批量释放 Host 内存页。

            host_mem_release_queue 中积累的是之前 revoke/backup 操作释放后
            产生的待回收 Host KV 页索引，这里批量 free 归还给 mem_pool_host。
            """
            host_indices_list = []
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
            if host_indices_list:
                host_indices = torch.cat(host_indices_list, dim=0)
                cc.mem_pool_host.free(host_indices)        # 批量归还 Host 内存页

        # ── 按顺序执行三个子 drain ──
        _drain_revoke()   # ① 先取消 prefetch，释放 Host KV 保护
        _drain_backup()   # ② 再确认 backup，ongoing_backup.pop + release_host
        _drain_release()  # ③ 最后批量释放 Host 内存页

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
        """💾 Host→L3 Storage 持久化 —— 将 Host 上的 KV 备份到 mooncake/EIC 等存储后端。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链：发生在 GPU→Host DMA 完成后 ━━━━━━━━━━━━━━
        GPU→Host DMA 完成 → writing_check() → _finish_write_through_ack()
          → ① ongoing_write_through.pop(ack_id)     — 清除 DMA 追踪
          → ② node.write_through_pending_id = None  — 清除 pending 标记
          → ③ _record_store_event(CPU)              — 通知下游索引器
          → ④ if enable_storage: write_backup_storage(lock_node, backup_len)  ← 你在这
                 ↓
              cache_controller.write_storage(...) → mooncake/EIC 异步写入
              ongoing_backup[op_id] = node → protect_host() 防止 Host KV 被驱逐

        💡 write_backup_storage 是 L2→L3 数据流入口：
          L1(GPU) → write_backup → L2(Host) → write_backup_storage → L3(Storage)

        ━━━━━━━━━━━━━━ 2️⃣ backup_len：处理节点 split 的竞态 ━━━━━━━━━━━━━━
        DMA 写回 Host 和 write_backup_storage 不是原子的：
          T0: write_backup(node, backup_len=len(node.key)) → GPU→Host DMA 飞行中
          T1: node 被 split（新请求插入共享前缀）→ node.key 变短
          T2: DMA 完成 → _finish_write_through_ack → write_backup_storage(node, backup_len)
        
        此时 node.key 长度 ≠ backup_len，需要回溯拼出原始数据：
          _concat_split_chain(node, backup_len) → 沿孩子链拼接完整 KV 数据
        """
        # 节点未被 split：直接使用 node 的属性
        if backup_len is None or len(node.key) == backup_len:
            top, key, hash_value, host_value = (
                node,
                node.key,
                node.hash_value,
                node.host_value,
            )
        else:
            # 节点已被 split：沿 child 链拼接原始完整数据
            top, key, hash_value, host_value = self._concat_split_chain(
                node, backup_len
            )

        # prefix_keys: 从 root 到当前节点的前缀 hash 链（Storage 去重用）
        prefix_keys = (
            top.get_prefix_hash_values(top.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )

        # 异步写入 L3 存储后端
        operation_id = self.cache_controller.write_storage(
            host_value, key, hash_value, prefix_keys, **self._get_extra_pools()
        )
        # 📌 ongoing_backup 追踪器：注册 → 等待 Storage 确认 → 解除 Host KV 保护
        #
        # ━━━━━━━━━━━━━━ 注册（此处）━━━━━━━━━━━━━━
        # write_backup_storage() → cache_controller.write_storage() → ongoing_backup[op_id] = node
        #   → node.protect_host() 防止 Storage 写入期间 Host KV 被 evict_host 驱逐
        #
        # ━━━━━━━━━━━━━━ 取出时机 ━━━━━━━━━━━━━━
        # ① 正常路径：Storage 写入完成回调（scheduler 主循环每步调用）
        #   scheduler event_loop_normal() / event_loop_overlap()
        #     → check_hicache_events()
        #       → drain_storage_control_queues()
        #         → all_reduce_attn_groups(MIN)           — 跨 TP rank 同步队列长度
        #         → _drain_storage_control_queues_impl(n_backup, ...)
        #           → _drain_backup() 遍历 ack_backup_queue
        #             → op_id = operation.id
        #             → entry = ongoing_backup.pop(op_id) — 从追踪器移除
        #             → entry.release_host()              — 解除 Host KV 保护，允许后续驱逐
        #
        # ② 清理路径：shutdown / detach 时强制释放
        #   路径 A（先排空队列再强制清理）:
        #     shutdown → detach_storage_backend
        #       → _drain_storage_control_queues_local()
        #         → _drain_storage_control_queues_impl(n_backup=None, ...)
        #           → _drain_backup() → pop + release_host()
        #       → _force_release_pending_storage_ops()    — 兜底：清理残留项
        #         → for ack_id, node in ongoing_backup.items():
        #             → node.release_host()               — 强制解除保护
        #             → ongoing_backup.pop(ack_id)        — 清理追踪器
        #
        # 💡 为什么需要 protect / release_host 机制？
        #   Storage 异步写入是 non-blocking 的，期间 Host KV 可能被其他请求的 evict_host 驱逐。
        #   protect_host() 锁定节点，禁止驱逐；等 Storage 确认写入完成后 release_host() 解锁。
        self.ongoing_backup[operation_id] = node
        # 保护 Host KV：Storage 写入期间防止 Host 端被 evict_host 驱逐
        node.protect_host()

    def _concat_split_chain(self, node: TreeNode, backup_len: int):
        """通过遍历 split 链恢复写入时的完整 key / hash / host_value。

        📌 为什么需要这个？
        写 Storage 的时刻（write_backup_storage）晚于发起 DMA 的时刻（write_backup）。
        期间节点可能被 split（新请求插入共享前缀），原始 node 被拆成多个小子节点。
        在树中现在的样子：
            原 node(key="ABCD", host=[0,1,2,3])
                   ↓ split
            父(key="AB") → 子(key="CD"), 子(key="CD") → 孙...
        树结构变成了父子链 [父, 子, 孙, ...]，需要沿链拼接还原原始数据。

        📌 拼接逻辑：
        ① 从 node 向 root 方向爬，累计 key 长度直到 == backup_len
        ② reverse 得到父→子→孙的正序
        ③ 拼接 key:     父.key + 子.key + ... （bigram 边界去重）
        ④ 拼接 hash:    父.hash + 子.hash + ...
        ⑤ 拼接 host:    torch.cat([父.host, 子.host, ...])

        📌 示例：
        backup_len=4, node 被 split 为:
          父(key="AB", host=[0,1]) → 子(key="CD", host=[2,3])
        拼接结果: key="ABCD", host=[0,1,2,3]
        """
        # ① 从当前 node 向 root 爬，累计 key 长度
        chain, accumulated = [], 0
        current = node
        while current is not self.root_node and accumulated < backup_len:
            chain.append(current)                    # 子→父顺序（反序）
            accumulated += len(current.key)
            current = current.parent
        assert accumulated == backup_len, (
            f"backup chain length mismatch for node {node.id}: "
            f"expected {backup_len}, got {accumulated}"
        )
        chain.reverse()  # ② 反转得到父→子→孙正序
        top = chain[0]   # 链顶端 = 最老的父节点

        # ③ 拼接 token_ids（bigram 模式下相邻节点共享一个边界 token，需去重）
        if top.key.is_bigram:
            token_ids = list(chain[0].key.token_ids)
            for n in chain[1:]:
                token_ids.extend(n.key.token_ids[1:])  # 跳过第一个（与前一个尾部重复）
        else:
            token_ids = []
            for n in chain:
                token_ids.extend(n.key.token_ids)
        key = RadixKey(token_ids, top.key.extra_key, top.key.is_bigram)

        # ④ 拼接 hash_value（每个节点每页的 SHA256 hash 序列）
        if all(n.hash_value is not None for n in chain):
            hash_value = []
            for n in chain:
                hash_value.extend(n.hash_value)
        else:
            hash_value = None

        # ⑤ 拼接 host_value（Host KV slot 索引张量）
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
        """🔍 轮询并收割已完成的 GPU→Host DMA 写入。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
          write_through 模式: scheduler 主循环 → writing_check(write_back=False)
          write_back 模式:    evict() 阶段2 → writing_check(write_back=True)

        ━━━━━━━━━━━━━━ 2️⃣ write_back vs write_through 核心区别：query() vs synchronize() ━━━━━━━━━━━━━━

        write_back=True（evict 场景，阻塞）：
          直接对所有 DMA 事件 finish_event.synchronize() → 强制等全部完成
          原因：evict 必须确保 GPU 已释放才能安全回收 slot

        write_through=False（write_through 场景，非阻塞）：
          先用 finish_event.query() 非阻塞问"完成了没" → 只统计已完成的
          all_reduce(MIN) 跨 rank 同步 → 只收割所有 rank 都确认完成的
          对已完成的才 synchronize()（此时立即返回，仅二次确认）
          原因：让异步 DMA 继续飞，下次 poll 再来收；不阻塞调度主循环

        write_through_threshold 控制两种模式：
          =1 (write_through): 新 token 立即 DMA，writing_check 非阻塞收割
          =2 (write_back):    等 evict 才 DMA，writing_check 阻塞等全部完成
        """
        if write_back:
            # write_back 模式：阻塞等全部 DMA 完成。
            # 直接对所有事件 synchronize() → 强制等待 GPU→Host 传输结束。
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()          # 阻塞等待
                    for ack_id in ack_list:
                        self._finish_write_through_ack(ack_id, release_lock=False)
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # write_through 模式：非阻塞收割，只收已完成的。
        if len(self.ongoing_write_through) == 0:
            return

        # 📌 query() vs synchronize() 的区别：
        #   query()  = 非阻塞，返回 True/False（完成没？）  ← write_through 用这个，不阻塞
        #   synchronize() = 阻塞，等完成                     ← write_back 用这个，必须等
        # pp_rank=0 统计已完成的 DMA 数量（query() 非阻塞检查，遇到未完成就停止）
        finish_count = 0
        if self.pp_rank == 0:
            for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                if not finish_event.query():            # query() 非阻塞，true=已完成
                    break                               # 按序检查，遇到未完成就停止
                finish_count += 1

        # all_reduce(MIN)：跨 TP+PP rank 取各 rank 都已完成的 DMA 数量，保证一致性
        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
        finish_count = finish_count_tensor.item()

        if finish_count > 0:
            logger.debug(f"Process {finish_count} write back operations")

        # FIFO 收割已完成的 DMA（synchronize 此时立即返回，因为是 query 确认过的）
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()                  # 二次确认（query 过了，立即返回）
            for ack_id in ack_list:
                self._finish_write_through_ack(ack_id, release_lock=True)
            finish_count -= 1

    def loading_check(self):
        """🔍 轮询并收割已完成的 Host→GPU DMA 加载（load_back 的异步收尾）。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        scheduler 主循环 → check_hicache_events() → loading_check()
        load_back → cache_controller.load() → 入 load_queue
        cache_controller.start_loading() → 逐层 DMA（load_to_device_per_layer × N）
          → 每层完成后 record finish_event → 入 ack_load_queue
        loading_check 轮询 ack_load_queue → 收割每层已完成的事件 → dec_lock_ref

        ━━━━━━━━━━━━━━ 2️⃣ 为什么 load 是逐层而 write 是全层？━━━━━━━━━━━━━━
          📤 start_writing: backup_from_device_all_layer 全层一次 → 无需等计算流
          📥 start_loading: load_to_device_per_layer × N 逐层 → prefill 计算流逐层等 KV
          LayerDoneCounter 管理逐层同步，计算在第 0 层 DMA 完成就可开始，不等全层

        ━━━━━━━━━━━━━━ 3️⃣ vs writing_check ━━━━━━━━━━━━━━
          writing_check: GPU→Host, _finish_write_through_ack → dec_lock_ref + 💾 Storage 持久化
          loading_check: Host→GPU, ongoing_load_back.pop → dec_lock_ref 即可
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
        """🗑️ 驱逐 GPU 端 KV 以释放显存 —— write_back / CPU offload 的核心触发点。

        ━━━━━━━━━━━━━━ 1️⃣ 完整调用链（从 scheduler 到 evict）━━━━━━━━━━━━━━
        scheduler.run_batch()
          → prepare_for_decode / prepare_for_extend       (schedule_batch.py)
            → alloc_for_decode / alloc_for_extend         (common.py)
              → alloc_token_slots / alloc_paged_token_slots_extend  (common.py:272)
                → evict_from_tree_cache(tree_cache, N)    (common.py:302)
                  → tree_cache.evict(EvictParams(num_tokens=N))   ← 你在这
        💡 scheduler 在每次 alloc KV slot 之前检查显存。若不够，
        先 evict 腾空间，再 alloc。这是 write_back / CPU offload 的核心触发点。

        ━━━━━━━━━━━━━━ 2️⃣ 三阶段驱逐流程 ━━━━━━━━━━━━━━
        0️⃣ 收集可驱逐叶子 → 按优先级建最小堆
        1️⃣ 循环驱逐（按优先级）
           ├── 🔒 lock_ref>0 → 跳过（被其他请求锁定）
           ├── 🔴 未 backuped + write_back → write_backup(DMA) → 入 write_back_nodes
           ├── 🟡 未 backuped + write_through(命中未达阈值/写失败) → _evict_regular(真删除)
           └── 🟢 已 backuped → _evict_backuped(降级到 Host，可 load_back 恢复)
        2️⃣ write_back 收尾（仅 write_back 模式）
           └── writing_check(⏳ 阻塞等 DMA) → _evict_backuped(释放 GPU)

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

            # ⚠️ TOCTOU 竞态：快照时 lock_ref=0（在 evictable_leaves 中），
            # 但遍历到它时可能已被其他请求 inc_lock_ref 锁定。
            # _update_leaf_status 会从 evictable_leaves 移除锁定的节点，
            # 但我们持有的是旧快照（list(self.evictable_leaves)），
            # 因此需要在此二次确认。
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
                    # write_through 模式：insert 时已尝试 write_backup（threshold=1，即首次 hit 就触发），
                    # 正常情况此时节点已 backuped。但若 Host 内存不足 write_backup 失败，
                    # 节点仍是未 backuped 状态 → 只能真删除（无 Host 副本可用）
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
        # 释放 GPU KV slot，本质是 cache_controller.evict_device → allocator.free(node.value)
        # 😅无语，不就是 self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = self.cache_controller.evict_device(node.value)
        assert num_evicted > 0, f"Expected to evict > 0 tokens, got {num_evicted}"
        # 更新可驱逐 token 计数：释放了 GPU slot，可驱逐空间减少
        self.evictable_size_ -= num_evicted
        # 降级标记：GPU 已释放，但 Host 副本保留 → 可 load_back 恢复
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
        """🗑️ 驱逐 Host 端的 KV 缓存（Host→无），当 Host 内存不足时触发。

        ━━━━━━━━━━━━━━ 1️⃣ evict(GPU) vs evict_host(Host) 区别 ━━━━━━━━━━━━━━

        evict(GPU):
          触发时机: 每次 alloc GPU KV slot 前，显存不够
          调用链:   scheduler → alloc_token_slots → evict_from_tree_cache → evict()
          驱逐对象: GPU 上的节点（降级到 Host 或真删除）
          触发频率: 极高，每次 batch 的 extend/decode 都可能触发

        evict_host(Host):
          触发时机: 分配 Host slot 失败时（Host 内存不足）
          调用链:
            ① write_backup（GPU→Host 写回）
               → host alloc 失败 → evict_host(len(node.value)) → 重试 alloc
            ② prefetch_from_storage（L3→Host 预取）
               → host alloc 失败 → evict_host(prefetch_length) → 重试 alloc
          驱逐对象: 已 evicted=True（GPU 已释放）且 host_ref_counter=0 的节点
          触发频率: 较低，Host 内存池通常比 GPU 大很多

        📌 两级驱逐链：
          evict(GPU)   → 降级：value=None, host_value 保留，可 load_back 恢复
          evict_host   → 真删除：从树中移除，Host KV 也释放，无法恢复
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
            # 只驱逐 GPU 已释放且 host 副本仍在的节点
            if not x.evicted:
                continue

            # 有 prefetch 正在使用该节点的 Host KV，不能驱逐
            if x.host_ref_counter > 0:
                continue

            # 真删除：Host KV 没了，从树中彻底移除
            self._record_remove_event(x, medium=StorageMedium.CPU)
            num_evicted += self.cache_controller.evict_host(x.host_value)

            # 从父节点 children 中移除
            key = x.key.child_key(self.page_size)
            v = x.parent.children.pop(key, None)
            assert v == x, f"parent does not have child key, {key}"
            if x in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(x)
            self._update_host_leaf_status(x.parent)

            # 父节点是否变成新的可驱逐叶子？
            if len(x.parent.children) == 0 and x.parent.evicted:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """🔄 将节点的 KV 从 Host 加载回 GPU（Host→GPU DMA），恢复 evicted 状态。

        ━━━━━━━━━━━━━━ 1️⃣ 完整调用链（谁调用了 load_back）━━━━━━━━━━━━━━
        📥 schedule_policy.py（prefill 调度阶段）
          → tree_cache.init_load_back(InitLoadBackParams(best_match_node=req.best_match_node, ...))
          → HiRadixCache.init_load_back():
              若 best_match_node.evicted 🔴 → load_back(node, mem_quota)  ← 你在这
            （不是 evicted 🟢 → 说明 KV 还在 GPU，无需 load_back）
        📥 decode_hicache_mixin.py（disaggregation decode 场景）
          → tree_cache.init_load_back(...)（同上路径）

        ━━━━━━━━━━━━━━ 2️⃣ load_back 只负责入队，异步 DMA 完整流程 ━━━━━━━━━━
          load_back → cache_controller.load()  ← 🔑 只分配 GPU slot + 入 load_queue
          scheduler → cache_controller.start_loading()  ← 🚀 真正逐层 DMA
            → load_to_device_per_layer × N → finish_event → ack_load_queue
          scheduler 主循环 → loading_check()  ← 🔍 轮询收割每层完成事件
            → ongoing_load_back.pop → dec_lock_ref

        ━━━━━━━━━━━━━━ 3️⃣ load_back 工作原理 ━━━━━━━━━━━━━━

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

        # ── 阶段 1: 收集被驱逐的连续节点链 ──
        # 从 node 向上溯，收集所有连续的 evicted 节点。
        # 例如：A → B → C → D，C 和 D evicted → nodes_to_load = [C, D]（正序）
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)  # 头插保证正序
            node = node.parent
        else:
            ancester_node = node  # 最近的非 evicted 祖先

        # 锁定祖先节点，防止 DMA 期间被驱逐
        result = self.inc_lock_ref(ancester_node)
        delta = result.delta

        # 拼接所有待加载节点的 host_value → 一笔 DMA 传完
        host_indices = torch.cat([n.host_value for n in nodes_to_load])

        # ── 阶段 2: 判断是否值得加载 ──
        # 太小（< threshold）：不值得花 DMA 开销
        # 太大（> mem_quota + 本次 evict 腾出的空间）：GPU 放不下
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            self.dec_lock_ref(ancester_node)
            return None

        # ── 阶段 3: 入队 Host→GPU DMA（此时只是分配 GPU slot + 入 load_queue，尚未开始传输）──
        # cache_controller.load() = alloc device slot + CacheOperation 入 load_queue
        # 真正的逐层 DMA 由 scheduler 调用 cache_controller.start_loading() 触发
        device_indices = self.cache_controller.load(
            host_indices=host_indices,
            node_id=last_hit_node.id,
            **self._get_extra_pools(),
        )
        if device_indices is None:
            # GPU 显存不够 → 先 evict 腾空间，再重试入队
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load(
                host_indices=host_indices,
                node_id=last_hit_node.id,
                **self._get_extra_pools(),
            )
        self.dec_lock_ref(ancester_node)  # 释放祖先锁：DMA 已完成，不再需要保护路径
        if device_indices is None:
            logger.warning(
                "load_back: FAILED to load %d tokens for node %d "
                "even after eviction (evictable_size=%d)",
                len(host_indices),
                last_hit_node.id,
                self.evictable_size_,
            )
            return None

        # ── 阶段 4: 恢复节点状态 ──
        # 注册到 ongoing_load_back，loading_check 会异步收割
        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        for node in nodes_to_load:
            # 分配 device_indices 的对应段给每个节点
            node.value = device_indices[offset : offset + len(node.host_value)].clone()
            offset += len(node.host_value)
            # 恢复 evicted 状态 → 节点重新在 GPU 上可用
            self._record_store_event(node, medium=StorageMedium.GPU)
        self.evictable_size_ += len(device_indices)
        # 锁交换：释放祖先锁(不再需要)，改为锁定最深节点（防止刚加载回来就被 evict）。
        # 刚恢复的叶子节点没有 GPU 子节点保护，极易被下一轮 evict 选中，
        # inc_lock_ref 确保它至少撑到当前 prefill 批次结束。
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
        """📥 Host→GPU 加载入口 —— 由 scheduler 在 prefill 调度时调用。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        schedule_policy.py（add_one_req 准入判断, line ~973）
          → req.needs_host_load_back() 检查 host_hit_length > 0
          → tree_cache.init_load_back(InitLoadBackParams(
                best_match_node=req.best_match_node,
                host_hit_length=req.host_hit_length,
                mem_quota=...))

        ━━━━━━━━━━━━━━ 2️⃣ 三种返回值语义 ━━━━━━━━━━━━━━
        🟢 load_back 成功:
            return (new_device_indices, last_node)
            last_node 是 load_back 恢复后的最深 GPU 节点（用于 inc_lock_ref）
        🟡 load_back 失败 (GPU OOM, 即使 evict 后仍不够):
            return (empty_indices, last_node)
            last_node 回退到最近的非 evicted 祖先
        🔵 无需 load_back (last_node 不在 Host, 即 evicted=False):
            return (empty_indices, last_node)
            last_node = best_match_node（本来就在 GPU 上）

        ━━━━━━━━━━━━━━ 3️⃣ 三步流程 ━━━━━━━━━━━━━━
        ① 若 best_match_node 在 Host 不在 GPU → load_back(node)
           load_back 内部: 收集连续 evicted 节点链 → 分配 GPU slot → 入 load_queue
        ② 若 load_back 成功: 返回新的 device_indices + 最深恢复节点
           调度器侧会用返回的 indices 拼接 prefix_indices
        ③ 若 load_back 失败: 逐级向上溯找最近非 evicted 祖先，仅返回祖先的 GPU indices
        """
        last_node = params.best_match_node
        mem_quota = params.mem_quota

        # ① 节点在 Host 不在 GPU → 需要 load_back 恢复
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node  # 🟢 成功：返回新 GPU indices

            # 🟡 失败：向上找到最近的非 evicted 祖先，回退到祖先的 GPU indices
            while last_node.evicted:
                last_node = last_node.parent

        # 🔵 节点已在 GPU 或回退到祖先 → 返回空（不需要额外加载）
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
        """🔍 Scheduler 主循环每步调用的统一事件轮询入口。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        scheduler event_loop_normal() / event_loop_overlap()
          → check_hicache_events()                               ← 你在这
            → ① writing_check()                                 — GPU→Host DMA 收割
            → ② loading_check()                                 — Host→GPU DMA 收割
            → ③ drain_storage_control_queues()                  — L3 Storage 控制队列排空（enable_storage=True 时）
            → ④ _reap_completed_async_work()                    — NCCL 异步 work 清理
            → ⑤ storage_metrics_collector.log_storage_metrics() — Storage 指标上报

        ━━━━━━━━━━━━━━ 2️⃣ 执行顺序设计意图 ━━━━━━━━━━━━━━
        ①→② 先收割 DMA：GPU↔Host 双向传输收尾，释放锁引用
        ③    再处理 Storage：必须等 DMA 完成后（writing_check 已触发 write_backup_storage
              并注册 ongoing_backup），此时 Storage 写入的回调才会出现在 ack_backup_queue 中
        ④    最后清理异步 work：NCCL barrier 等通信句柄释放
        ⑤    指标上报：无副作用的纯观测

        ━━━━━━━━━━━━━━ 3️⃣ 关键：此函数中 ongoing_backup 的完整生命周期 ━━━━━━━━━━━━━━
        写入注册（上一步/前几轮）:
          writing_check → _finish_write_through_ack → write_backup_storage
            → cache_controller.write_storage(...) → ongoing_backup[op_id] = node
            → node.protect_host()  防止 Host KV 被驱逐

        取出解除（本轮）:
          check_hicache_events → drain_storage_control_queues
            → _drain_storage_control_queues_impl → _drain_backup()
              → 遍历 ack_backup_queue → ongoing_backup.pop(op_id) → entry.release_host()
        """
        # ① GPU→Host：收割已完成的 write DMA，触发 write_through 收尾 + Storage 备份启动
        self.writing_check()
        # ② Host→GPU：收割已完成的 load DMA（逐层），释放 lock_ref
        self.loading_check()
        # ③ Storage 控制面：排空 prefetch 撤销 / backup 确认 / host 内存释放三个队列
        if self.enable_storage:
            self.drain_storage_control_queues()
        # ④ 清理已完成的 NCCL 异步通信句柄
        self._reap_completed_async_work()
        # ⑤ Storage 指标上报（不影响调度逻辑）
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def drain_storage_control_queues(self):
        """📬 排空 Storage 控制队列 — 用一次 all_reduce(MIN) 同步三个队列，减少 TP 同步开销。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        check_hicache_events() → drain_storage_control_queues()   ← 你在这
          → all_reduce_attn_groups(MIN)                          — 跨 TP rank 同步队列长度
          → _drain_storage_control_queues_impl(n_revoke, n_backup, n_release, log_metrics=True)
            → _drain_revoke()   — 取消进行中的 prefetch，释放 Host KV + 回收 token 配额
            → _drain_backup()   — 收割 Storage 写入完成的 ack，pop ongoing_backup + release_host()
            → _drain_release()  — 批量释放 Host 内存页

        ━━━━━━━━━━━━━━ 2️⃣ 为什么需要 all_reduce(MIN)？━━━━━━━━━━━━━━
        三个控制队列由 Storage 后端线程写入，各 TP rank 的写入进度可能不同。
        用 MIN 取所有 rank 中最少的待处理数，保证所有 rank 处理同样数量的条目，
        避免跨 rank 的 ongoing_backup / ongoing_prefetch 状态不一致。
        例如 rank0 有 3 个 ack，rank1 有 2 个，则两边都只处理 2 个。

        ━━━━━━━━━━━━━━ 3️⃣ 三个子 drain 的作用 ━━━━━━━━━━━━━━
        📌 _drain_revoke:  prefetch_revoke_queue → ongoing_prefetch.pop → release_host() → 回收 token 配额
        📌 _drain_backup:  ack_backup_queue → ongoing_backup.pop → release_host() → 📊 记录 backup 指标
        📌 _drain_release: host_mem_release_queue → mem_pool_host.free()  → 批量回收 Host KV 页
        """
        cc = self.cache_controller

        # 收集三个控制队列的当前长度
        qsizes = torch.tensor(
            [
                cc.prefetch_revoke_queue.qsize(),  # 待撤销的 prefetch 操作数
                cc.ack_backup_queue.qsize(),       # 待确认的 backup 操作数
                cc.host_mem_release_queue.qsize(), # 待释放的 Host 内存块数
            ],
            dtype=torch.int,
        )
        # all_reduce(MIN)：跨 attention group 取最小值，保证所有 rank 一致
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
