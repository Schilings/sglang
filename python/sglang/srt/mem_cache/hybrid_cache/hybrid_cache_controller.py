from __future__ import annotations

import json
import logging
import os
import threading
import time
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch

from sglang.srt.managers.cache_controller import CacheOperation as BaseCacheOperation
from sglang.srt.managers.cache_controller import (
    HiCacheAck,
)
from sglang.srt.managers.cache_controller import (
    HiCacheController as BaseHiCacheController,
)
from sglang.srt.managers.cache_controller import (
    LayerDoneCounter,
)
from sglang.srt.managers.cache_controller import (
    StorageOperation as BaseStorageOperation,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.memory_pool_host import PoolEntry
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)
device_module = get_device_module()


class CacheOperation(BaseCacheOperation):
    """🚚 混合架构下的 GPU↔Host 搬运操作描述符 —— 在基类单池操作上扩展多池搬运。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔑 与基类 BaseCacheOperation 的关键区别                                            ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  BaseCacheOperation (cache_controller.py):  仅描述 KV 主池的 host/device indices    ║
    ║  CacheOperation (本类):                     额外携带 pool_transfers 描述辅池搬运    ║
    ║    └─ pool_transfers: List[PoolTransfer]                                           ║
    ║         每项描述一个辅池 (SWA / Mamba / Indexer / DSA ...) 的 host/device indices   ║
    ║         及其命中策略 (ALL_PAGES / TRAILING_PAGES)                                    ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  🎯 适用场景：混合模型架构 (KV+SWA / KV+Mamba / DSA / DeepSeek V4) 开启             ║
    ║     hierarchical cache 时，一次 GPU↔Host DMA 需同时搬运多个物理池的数据。            ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        """
        📥 参数：
          host_indices   : KV 主池在 Host 端的 slot 索引 (由 mem_pool_host.alloc 分配)
          device_indices : KV 主池在 GPU 端的 slot 索引 (radix tree node.value)
          node_id        : radix tree 节点 id, DMA 完成后凭此找到 node 做收尾
          priority       : 优先级 (数值越小越高), merge_ops 取 min 用于排序
          pool_transfers : 辅池搬运描述列表; None 表示纯 KV 搬运 (退化为基类行为)
        """
        super().__init__(host_indices, device_indices, node_id, priority)
        # 辅池搬运描述: 每个 PoolTransfer 对应一个非 KV 主池的 host/device indices
        self.pool_transfers = pool_transfers

    @staticmethod
    def merge_pool_transfers(
        ops: List[CacheOperation],
    ) -> Optional[list[PoolTransfer]]:
        """🔗 合并多个 CacheOperation 的 pool_transfers —— 按 (池名, 来源池) 分组 cat。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        start_writing / start_loading → CacheOperation.merge_ops(ops)
          → merge_pool_transfers(ops)   ← 你在这 (合并辅池 indices)

        ⚙️ 行为：
          ① 按 (transfer.name, transfer.indices_from_pool) 分组 —— 同池同来源的才可合并
          ② 每组 cat host_indices / device_indices (跳过 None)
          ③ keys 拍平拼接; hit_policy / indices_from_pool 取组内首项 (同组必然一致)
        📤 返回：合并后的 PoolTransfer 列表, 无辅池时返回 None
        """
        # 分组键 = (池名, 来源池): indices_from_pool 标识该 pool 是"派生池"(复用源池 indices)
        grouped: dict[tuple[PoolName, Optional[PoolName]], list[PoolTransfer]] = {}
        for op in ops:
            for t in op.pool_transfers or []:
                grouped.setdefault((t.name, t.indices_from_pool), []).append(t)
        if not grouped:
            return None

        def cat_or_none(tensors):
            # 拼接非 None 的 tensor; 全为 None 则返回 None
            parts = [x for x in tensors if x is not None]
            return torch.cat(parts) if parts else None

        # 每组重建一个 PoolTransfer, 字段取组内首项 (同组语义一致)
        return [
            PoolTransfer(
                name=ts[0].name,
                host_indices=cat_or_none(t.host_indices for t in ts),
                device_indices=cat_or_none(t.device_indices for t in ts),
                keys=[k for t in ts if t.keys for k in t.keys] or None,
                hit_policy=ts[0].hit_policy,
                indices_from_pool=ts[0].indices_from_pool,
            )
            for ts in grouped.values()
        ]

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        """🔗 合并多个 CacheOperation 为一次大 DMA —— cat 主池 indices + 合并辅池。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        write() / load() → start_writing() / start_loading()
          → CacheOperation.merge_ops(write_queue / load_queue)   ← 你在这

        ⚙️ 行为：
          ① 单个 op 直接返回 (no-op 快速路径)
          ② cat 所有 op 的 host_indices / device_indices
          ③ priority 取 min (最高优先级), node_ids 拼接 (保留全部 DMA 回调凭证)
          ④ node_id = -1: 合并后不再对应单个 node, 由 node_ids 列表承载
          ⑤ 辅池通过 merge_pool_transfers 合并
        """
        if len(ops) == 1:
            return ops[0]
        # 主池 indices 直接 cat 成一个大 tensor
        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        # priority 取最小值: 合并后的操作继承最高优先级
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged = CacheOperation(
            host_indices,
            device_indices,
            -1,
            priority,
            pool_transfers=CacheOperation.merge_pool_transfers(ops),
        )
        # 合并后的 node_ids 携带全部原始节点 id, 供 writing_check 收割时逐个回调
        merged.node_ids = node_ids
        return merged


class StorageOperation(BaseStorageOperation):
    """🚚 混合架构下的 Host↔Storage 搬运操作描述符 —— 在基类单池操作上扩展多池搬运。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔑 与基类 BaseStorageOperation 的关键区别                                          ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  BaseStorageOperation: 仅描述 KV 主池的 host_indices + token_ids + hash             ║
    ║  StorageOperation:     额外携带 pool_transfers (辅池) + pool_storage_result (命中)  ║
    ║    └─ pool_storage_result: PoolTransferResult, 记录 KV/辅池各自实际命中的页数         ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  🎯 适用场景：混合模型架构开启 L3 存储时, 一次 Host↔Storage IO 需同时读写            ║
    ║     KV 主池与各辅池 (SWA state / Mamba state / Indexer ...) 的数据。               ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        """
        📥 参数：
          host_indices   : KV 主池在 Host 端的 slot 索引
          token_ids      : 本操作覆盖的 token id 序列
          last_hash      : 上一页的 hash (链式 hash 起点)
          hash_value     : 各 page 的 hash 列表 (未给则 _storage_hit_query 时计算)
          prefix_keys    : 祖先节点的 hash, 供存储后端做层级索引
          pool_transfers : 辅池搬运描述列表
        """
        super().__init__(host_indices, token_ids, last_hash, hash_value, prefix_keys)
        # 辅池搬运描述 (与 CacheOperation.pool_transfers 同构)
        self.pool_transfers = pool_transfers
        # 多池命中结果: 记录 KV 主池 + 各辅池实际成功读/写的页数
        #   创建: __init__ 初始化为空 (kv_hit_pages=0)
        #   更新: _page_transfer / _page_backup → update_kv_hit_pages / update_extra_pool_hit_pages
        #   读取: unified_radix_cache / hiradix_cache 的 prefetch/backup 收尾逻辑据此判断实际命中量
        self.pool_storage_result = PoolTransferResult.empty()


class PrefetchOperation(StorageOperation):
    """⬇️ L3→Host 预取操作 —— 带线程安全终止控制的 StorageOperation。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧩 对外接口（框架通过这些方法使用本类）                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  increment(num_tokens)  ✍️ 原子累加已完成的 token 数, 终止后返回 False              ║
    ║  mark_terminate()       🛑 标记终止 (超时 / TP 不一致 / 控制器主动取消)              ║
    ║  is_terminated()        🔍 查询是否已终止                                            ║
    ║  pool_transfers_done    🏁 辅池 IO 是否全部完成 (无辅池时构造即 True)                ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  🔗 调用链                                                                          ║
    ║  scheduler → prefetch_from_storage() → controller.prefetch() → PrefetchOperation   ║
    ║    → 入 prefetch_queue → prefetch_thread → prefetch_buffer → prefetch_io_aux_thread║
    ║    → _page_transfer() 逐页读 L3 → increment(page_size) / mark_terminate()           ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    🧬 设计要点：
       _lock + _terminated_flag 保证 increment 与 mark_terminate 的原子性 ——
       prefetch_io_aux_thread (IO 线程) 调 increment, scheduler 主线程可能调
       mark_terminate (超时取消), 二者并发时 increment 必须能看到终止标记并停止累加。
    """

    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        """
        📥 参数：
          request_id : 唯一请求标识, 用于 ongoing_prefetch 字典索引
          host_indices / token_ids / last_hash / prefix_keys / pool_transfers: 同 StorageOperation
        """
        # 预取请求标识: scheduler 用它在 ongoing_prefetch 中追踪本次预取状态
        self.request_id = request_id
        # 线程安全锁: 保护 _terminated_flag 与 completed_tokens 的原子读写
        self._lock = threading.Lock()
        # 终止标记: True 后 increment 不再累加, _page_transfer 据此提前 break
        self._terminated_flag = False
        # 预取开始时刻: 用于 prefetch 超时判定 (linear timeout 策略)
        self.start_time = time.monotonic()
        super().__init__(
            host_indices,
            token_ids,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=pool_transfers,
        )
        # 辅池 IO 完成标记: 无辅池时构造即 True; 有辅池时 _page_transfer 结束后置 True
        self.pool_transfers_done = not bool(pool_transfers)

    def increment(self, num_tokens: int):
        """✍️ 原子累加已完成 token 数 —— IO 线程每读完一页调用一次。

        ⚙️ 行为：加锁后检查终止标记, 已终止则返回 False (调用方据此 break 逐页循环)。
        📤 返回：True=成功累加 / False=已终止 (调用方应停止后续 IO)
        """
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        """🛑 标记预取终止 —— scheduler 超时取消或 TP 不一致时调用。

        ⚠️ 仅设置标记, 不中断正在执行的 IO; IO 线程下次 increment 时感知并停止。
        """
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        """🔍 查询是否已终止 (无锁读, 用于非关键的快速检查)。"""
        return self._terminated_flag


class HybridCacheController(BaseHiCacheController):
    """🚚 混合架构分层 KV cache 控制器 —— 多池版 HiCacheController。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔑 与基类 BaseHiCacheController 的关键区别                                          ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  BaseHiCacheController (cache_controller.py):  单 KV 池, 仅搬运 host/device indices ║
    ║  HybridCacheController (本类):                 多池, 携带 pool_transfers 描述辅池    ║
    ║                                                                                  ║
    ║  ┌────────────────────────┬────────────────────────┬────────────────────────────┐  ║
    ║  │ 维度                    │ BaseHiCacheController  │ HybridCacheController      │  ║
    ║  ├────────────────────────┼────────────────────────┼────────────────────────────┤  ║
    ║  │ CacheOperation          │ 仅 host/device indices │ + pool_transfers (辅池)     │  ║
    ║  │ write/load              │ 单池 alloc + DMA       │ 多池 alloc + 多池 DMA       │  ║
    ║  │ move_indices            │ 搬运 KV indices        │ move_hybrid_indices 搬全部  │  ║
    ║  │ _page_transfer/_backup  │ 单池 L3 IO             │ KV 先行 + 辅池跟进          │  ║
    ║  │ host_mem_release        │ 单 release_queue       │ + extra_host_mem_release_   │  ║
    ║  │                         │                        │   queues (每辅池一条)       │  ║
    ║  │ transfer_layer_num      │ = full_kv_pool 层数    │ = 全模型层数 (KV+Mamba等)   │  ║
    ║  └────────────────────────┴────────────────────────┴────────────────────────────┘  ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  🎯 适用场景：混合模型架构 (KV+SWA / KV+Mamba / DSA / DeepSeek V4) 开启              ║
    ║     --enable-hierarchical-cache 时, 由 hybrid_pool_assembler.build_*_stack() 构造,  ║
    ║     注入 cache.cache_controller。基类只懂单 KV 池, 本类让多池共享同一套 DMA/L3 流水线。 ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧩 对外接口（框架通过这些方法使用本类, 均继承或 override 基类）                       ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  write(device_indices, extra_pools)         ✍️ GPU→Host: 分配多池 slot + DMA       ║
    ║  load(host_indices, extra_pools)            📥 Host→GPU: 分配多池 slot + 入队       ║
    ║  start_writing() / start_loading()          🚚 合并队列 + 提交异步 DMA              ║
    ║  prefetch(req_id, host_indices, ...)        ⬇️ L3→Host 预取 (返回 PrefetchOperation) ║
    ║  write_storage(host_indices, ..., extra_pools) 💿 Host→L3 备份                      ║
    ║  append_host_mem_release(host_indices, extra_pools) 🗑️ 回收多池 Host 内存           ║
    ║  reset() / clear_storage_backend() / attach_storage_backend(...)  🔧 生命周期管理   ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  🔗 框架如何与本类交互 (宏观调用链)                                                   ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  ════════ ① 创建链 (启动期) ════════                                                ║
    ║  hybrid_pool_assembler.attach_hybrid_pool_to_unified_cache()                       ║
    ║    └─ build_*_stack() → HybridCacheController(...)  ← 本类                         ║
    ║    └─ cache.cache_controller = cache_controller   ← 注入回 UnifiedRadixCache 等     ║
    ║                                                                                  ║
    ║  ════════ ② GPU→Host 写入 (evict / write_through 备份) ════════                    ║
    ║  scheduler → HiRadixCache/UnifiedRadixCache.write_backup(node)                     ║
    ║    └─ controller.write(device_indices, extra_pools=[PoolTransfer(SWA/Mamba/...)])  ║
    ║         └─ start_writing() → backup_from_device_all_layer(pool_transfers=...)      ║
    ║    └─ scheduler → writing_check() → 收割 ack → dec_lock_ref                        ║
    ║                                                                                  ║
    ║  ════════ ③ Host→GPU 加载 (prefill 恢复 KV) ════════                               ║
    ║  scheduler → cache.load_back(node) → controller.load(host_indices, extra_pools)    ║
    ║    └─ cache.ready_to_load_host_cache() → controller.start_loading()                ║
    ║         └─ 逐层 load_to_device_per_layer(pool_transfers=...) + LayerDoneCounter    ║
    ║                                                                                  ║
    ║  ════════ ④ Host↔Storage (L2↔L3) ════════                                         ║
    ║  scheduler → prefetch_from_storage() → controller.prefetch() → prefetch_queue      ║
    ║    → prefetch_thread → _storage_hit_query() → _page_transfer() (KV 先 + 辅池跟进)   ║
    ║  scheduler → write_backup_storage() → controller.write_storage() → backup_queue     ║
    ║    → backup_thread → _page_backup() (辅池先写 + KV 后写)                            ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    🧬 设计要点：
       ① 多池共享同一条 DMA 流 (write_stream / load_stream): 一次 DMA 同时搬 KV + 辅池,
          避免多次 stream 提交的开销, indices 通过 move_hybrid_indices 统一搬到正确设备。
       ② transfer_layer_num override: 混合模型 (如 KV+Mamba) 的总传输层数 ≠ full_kv_pool
          报告的全注意力层数, 需用 transfer_layer_num 覆盖, 保证逐层 DMA 覆盖所有层。
       ③ 辅池"派生"机制 (indices_from_pool): SWA/Indexer 等辅池可复用 KV 主池的 indices,
          避免重复分配; _resolve_sidecar_derived_pool_transfers 负责在 L3 IO 前填充。
       ④ L3 IO 顺序: prefetch 时 KV 先行 (决定命中页数), 辅池仅在 KV 完整完成时跟进,
          避免 KV 提前终止导致辅池数据错位; backup 时辅池先写, KV 后写。
    """

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: Any,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        transfer_layer_num: Optional[int] = None,
        enable_storage_metrics: bool = False,
    ):
        """
        🚚 HybridCacheController 初始化 —— 在基类单池控制器上扩展多池能力。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        hybrid_pool_assembler.build_*_stack() (build_kv_only / build_hybrid_swa / ...)
          → HybridCacheController(...)   ← 你在这
          → 返回 (HostPoolGroup, HybridCacheController) 注入 cache.cache_controller

        📥 参数：
          token_to_kv_pool_allocator : GPU KV pool 分配器 (可能含 full_attn_allocator 子分配器)
          mem_pool_host              : HostPoolGroup, 多池聚合 (entries / entry_map / anchor_entry)
          page_size .. write_policy .. enable_storage_metrics : 同基类
          transfer_layer_num         : ⚠️ 混合模型总传输层数 (KV+Mamba 等), 覆盖基类的 full_kv 层数
          storage_backend            : L3 后端类型, None 则不启用 L3

        ⚙️ 行为：
          ① 先以 storage_backend=None 调 super().__init__ (基类不感知多池)
          ② 用 transfer_layer_num 覆盖 layer_num + 重建 layer_done_counter
          ③ 若指定 storage_backend, 调 attach_storage_backend 注册多池到 L3 后端
        """
        # 暂存 startup storage_backend: 基类 __init__ 先传 None (不启动 L3),
        # 等 transfer_layer_num 覆盖后再 attach, 确保 layer_num 正确后再注册多池
        startup_storage_backend = storage_backend
        # 辅池 Host 内存回收队列: 每个非锚定辅池一条独立 Queue (在 _init_extra_host_mem_release_queues 填充)
        self.extra_host_mem_release_queues: dict[PoolName, Queue[torch.Tensor]] = {}
        super().__init__(
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            mem_pool_host=mem_pool_host,
            page_size=page_size,
            tp_group=tp_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=pp_group,
            write_policy=write_policy,
            io_backend=io_backend,
            storage_backend=None,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        # Override layer_num: hybrid models transfer all layers (For example, Linear Model (KV + Mamba)),
        # not just the full attention layers reported by full_kv_pool.
        if transfer_layer_num is not None and transfer_layer_num != self.layer_num:
            self.layer_num = transfer_layer_num
            self.layer_done_counter = LayerDoneCounter(self.layer_num)

        if startup_storage_backend is not None:
            self.attach_storage_backend(
                storage_backend=startup_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=model_name,
                storage_backend_extra_config=storage_backend_extra_config,
                host_pools=getattr(mem_pool_host, "entries", None),
            )

    def _start_storage_threads(self):
        """🔧 启动 L3 存储线程 + 初始化多池 Host 内存回收队列。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        基类 attach_storage_backend() → super()._start_storage_threads()
          → _init_extra_host_mem_release_queues()   ← 你在这 (多池扩展)
        """
        super()._start_storage_threads()
        self._init_extra_host_mem_release_queues()

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        host_pools: Optional[list[PoolEntry]] = None,
    ):
        """🔧 挂载 L3 存储后端 —— 在基类基础上注册多池到后端。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        __init__ / 运行时热加载 → attach_storage_backend()
          → super().attach_storage_backend()  (基类: 创建 storage_backend 实例 + 启动线程)
          → 逐 entry register_mem_host_pool_v2  (本类扩展: 注册每个辅池的 host_pool)

        📥 参数：
          host_pools : PoolEntry 列表 (来自 HostPoolGroup.entries), 每项含一个 host_pool
        """
        super().attach_storage_backend(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
        )

        # 向 L3 后端注册每个辅池的 host_pool: batch_get_v2/batch_set_v2 据此找到对应物理内存
        for entry in host_pools or []:
            self.storage_backend.register_mem_host_pool_v2(entry.host_pool, entry.name)

    @staticmethod
    def parse_storage_backend_extra_config(
        storage_backend_extra_config: Optional[str],
    ) -> tuple[dict, int, float, float, bool]:
        """⚙️ 解析 L3 存储后端的额外配置 (JSON / @file / 内联 JSON)。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        hybrid_pool_assembler → HybridCacheController (构造前预解析)
          → parse_storage_backend_extra_config()   ← 你在这

        ⚙️ 行为：
          ① @前缀 → 从文件读取 (.json / .toml / .yaml / .yml)
          ② 否则 → 当作内联 JSON 字符串解析
          ③ pop 出 4 个已知键 (prefetch_threshold / prefetch_timeout_base /
             prefetch_timeout_per_ki_token / hicache_storage_pass_prefix_keys),
             剩余键原样返回给后端
          ④ 类型校验: 上述 4 键必须分别是 int / number / number / bool

        📤 返回：(extra_config, prefetch_threshold, timeout_base, timeout_per_ki_token, pass_prefix_keys)
        """
        extra_config = {}
        if storage_backend_extra_config:
            if storage_backend_extra_config.startswith("@"):
                # @path 形式: 从文件加载, 支持多格式
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
                # 内联 JSON 字符串
                extra_config = json.loads(storage_backend_extra_config)

        # pop 出本控制器关心的 4 个键, 剩余键留给 storage_backend 自身使用
        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        # ── 类型校验: 防止配置错误导致运行时崩溃 ──
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
                "prefetch_timeout_per_ki_token must be number, got "
                f"{type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def clear_storage_backend(self) -> bool:
        """🗑️ 清空 L3 存储后端全部数据。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        scheduler → cache.clear_hicache_storage() → controller.clear_storage_backend()

        📤 返回：True=已清空 / False=未启用 L3 或后端不支持 clear
        """
        if not self.enable_storage:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False
        if not hasattr(self.storage_backend, "clear"):
            logger.warning(
                "Storage backend %s does not support clear operation.",
                type(self.storage_backend).__name__,
            )
            return False
        self.storage_backend.clear()
        return True

    def _init_extra_host_mem_release_queues(self) -> None:
        """🔧 为每个非锚定辅池创建独立的 Host 内存回收队列。

        ⚙️ 行为：
          遍历 HostPoolGroup.entries, 跳过 anchor_entry 和 is_primary_index_anchor 池
          (锚定池的释放走基类的 host_mem_release_queue), 其余辅池各建一条 Queue。
        💡 为什么辅池要独立队列？锚定池的 page_size/layout 决定整个 group 的属性,
           辅池 page_size 可能不同, 释放时必须按各自 page_size 切分, 不能混用一条队列。
        """
        self.extra_host_mem_release_queues = {}
        entries = getattr(self.mem_pool_host, "entries", None) or []
        anchor_entry = getattr(self.mem_pool_host, "anchor_entry", None)
        for entry in entries:
            # 跳过锚定池: 其释放由基类 host_mem_release_queue 处理
            if entry is anchor_entry or entry.is_primary_index_anchor:
                continue
            self.extra_host_mem_release_queues[entry.name] = Queue()

    def _append_host_mem_release_pages(
        self, release_queue: Queue, host_indices: torch.Tensor, page_size: int
    ) -> None:
        """🗑️ 将 host_indices 按 page_size 切分后逐页放入回收队列。

        ⚙️ 行为：空 tensor 直接返回; 否则 split(page_size) 后逐页 put。
        💡 按页 put 而非整块 put: 回收侧 (storage 线程) 按页消费, 整块会导致
           一次释放过多而其他操作拿不到内存。
        """
        if host_indices.numel() == 0:
            return
        for page in host_indices.split(page_size):
            release_queue.put(page)

    def append_host_mem_release(
        self,
        host_indices: Optional[torch.Tensor] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ):
        """🗑️ 回收多池 Host 内存 —— KV 主池走基类队列, 辅池走各自独立队列。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        ╔══════════════════════════════════════════════════════════════════╗
        ║  触发场景 (均来自 scheduler → cache 收尾路径)                        ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║  1. prefetch 完成/中止: 回收未使用的预分配 Host 内存                  ║
        ║     cache._finish_prefetch / _abort_prefetch / _abort_prefetch_batch ║
        ║       → controller.append_host_mem_release(host_indices, extra_pools)║
        ║                                                                      ║
        ║  2. prefetch alloc 失败: 释放已分配的 host_indices                    ║
        ║     cache.prefetch_from_storage (alloc 失败分支)                      ║
        ║       → controller.append_host_mem_release(...)                      ║
        ╚══════════════════════════════════════════════════════════════════╝

        ⚙️ 行为：
          ① host_indices (KV 主池) → 基类 host_mem_release_queue (按 anchor page_size 切分)
          ② extra_pools 中每个辅池 → 查 entry_map, 非锚定且非派生池 → 各自独立队列
          ⚠️ 跳过 is_primary_index_anchor (锚定池) 和 indices_from_pool (派生池复用源池 indices, 不独立释放)
        """
        # ── KV 主池: 走基类回收队列 ──
        if host_indices is not None:
            self._append_host_mem_release_pages(
                self.host_mem_release_queue,
                host_indices,
                self.mem_pool_host.page_size,
            )
        # ── 辅池: 各自走独立回收队列 ──
        for transfer in extra_pools or []:
            if transfer.host_indices is None or transfer.host_indices.numel() == 0:
                continue
            entry = self.mem_pool_host.entry_map.get(transfer.name)
            # 跳过锚定池 (已由 host_indices 处理) 和派生池 (indices 复用源池, 不独立释放)
            if (
                entry is None
                or entry.is_primary_index_anchor
                or transfer.indices_from_pool is not None
            ):
                continue
            release_queue = self.extra_host_mem_release_queues.get(transfer.name)
            if release_queue is None:
                continue
            self._append_host_mem_release_pages(
                release_queue, transfer.host_indices, entry.host_pool.page_size
            )

    def reset(self):
        """🔄 重置控制器 —— 在基类基础上额外清空辅池回收队列和预取占用计数。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        scheduler → cache.reset() → controller.reset()
          → super().reset()  (基类: 停线程 + 清队列 + 重启线程)
          → 清空 extra_host_mem_release_queues + prefetch_tokens_occupied

        ⚙️ 行为：基类已清 host_mem_release_queue; 本类追加清辅池队列。
        """
        super().reset()
        if self.enable_storage:
            self.host_mem_release_queue.queue.clear()
            for release_queue in self.extra_host_mem_release_queues.values():
                release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        """✍️ GPU→Host 多池备份 —— 分配 KV + 辅池 Host slot, 入队并立即 DMA。
        📥 参数：
          device_indices : KV 主池 GPU slot 索引 (node.value)
          extra_pools    : 辅池搬运描述; None=纯 KV (退化为基类)
        📤 返回：host_indices (KV 主池 Host slot); None=Host 内存不足或辅池 alloc 失败
        ⚠️ 辅池 alloc 失败时会回滚已分配的 KV host slot (原子性保证)

        ━━━━━━━━━━━━━━  调用链 ━━━━━━━━━━━━━━
        scheduler → HiRadixCache / UnifiedRadixCache / HiMambaRadixCache.write_backup(node)
          → controller.write(device_indices=node.value, node_id=node.id,
                             extra_pools=[PoolTransfer(SWA/Mamba/...)])
            → start_writing()  ← 立即合并 + DMA
               → ack_write_queue
          → scheduler → writing_check() → 收割 ack → dec_lock_ref
        """
        # ① KV 主池: 在锚定 Host 池分配 slot
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        # ② 辅池: 为 pool_transfers 中 host_indices=None 的项自动分配 Host slot
        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools,
            alloc_host=True,
            kv_device_indices=device_indices,
            kv_host_indices=host_indices,
        )
        # 辅池 alloc 失败: 回滚 KV host slot (原子性, 避免泄漏)
        if pool_transfers is None and extra_pools:
            self.mem_pool_host.free(host_indices)
            return None

        # ③ 入 write_queue 并立即提交 DMA (基类同款行为)
        self.write_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        """📤 合并 write_queue, 一次性 GPU→Host 多池异步 DMA。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        write() → start_writing()   ← 你在这
          → merge_ops → move_hybrid_indices → write_stream DMA → ack_write_queue

        ⚙️ 行为 (在基类基础上增加 pool_transfers 处理)：
          ① merge_ops(write_queue): 合并 KV + 辅池 indices
          ② kernel + page_first 特例: indices 保留 CPU (write-back kernel 接受 CPU dst indices)
             其他情况: move_hybrid_indices 把 KV + 辅池 indices 搬到正确设备
          ③ write_stream: backup_from_device_all_layer(pool_transfers=...) 一次搬多池
          ④ draft pool: best-effort 搭载搬运 (仅 host_indices>0 时)
          ⑤ record_stream: 防止 CUDA allocator 提前回收 indices tensor
        """
        if not self.write_queue:
            return
        op = CacheOperation.merge_ops(self.write_queue)
        # For now, kernel write-back keeps host indices on CPU only for page_first.
        # More layouts can use this path once their write-back kernels accept CPU
        # destination indices.
        if self.io_backend == "kernel" and self.mem_pool_host.layout == "page_first":
            # page_first + kernel: write-back kernel 直接接受 CPU host_indices, 无需搬移
            host_indices = op.host_indices
            device_indices = op.device_indices
            resolved_pool_transfers = op.pool_transfers
        else:
            # 其他后端: move_hybrid_indices 把 KV + 辅池 indices 搬到正确设备 (GPU/CPU)
            host_indices, device_indices, resolved_pool_transfers = (
                self.move_hybrid_indices(op)
            )
        self.write_queue.clear()
        # ── 提交 write_stream 异步 DMA ──
        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            # 全层一次性拷贝 KV + 辅池 (pool_transfers 传给底层 kernel)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                self.io_backend,
                pool_transfers=resolved_pool_transfers,
            )
            # draft pool best-effort: 投机解码的 draft KV 搭载搬运 (无辅池扩展, 纯 KV)
            if self.has_draft and host_indices.numel() > 0:
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )
            finish_event.record()
            # 防止 CUDA caching allocator 在 stream 完成前回收 indices tensor
            self._record_transfer_indices_on_stream(
                self.write_stream,
                host_indices,
                device_indices,
                resolved_pool_transfers,
            )
        # ack 携带 node_ids, 供 writing_check 收割时逐个回调
        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        """📥 Host→GPU 多池加载 —— 分配 KV + 辅池 Device slot, 入 load_queue (延迟执行)。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        scheduler → cache.load_back(node) / _restore_mamba_state(node)
          → controller.load(host_indices=node.host_value, node_id=node.id,
                            extra_pools=[PoolTransfer(...)])
            → 入 load_queue (暂存, 不立即执行)
          → cache.ready_to_load_host_cache() → controller.start_loading()  ← 真正 DMA

        📥 参数：
          host_indices : KV 主池 Host slot 索引 (node.host_value)
          extra_pools  : 辅池搬运描述; None=纯 KV
        📤 返回：device_indices (KV 主池 GPU slot); None=Device 内存不足或辅池 alloc 失败
        ⚠️ 与 write 不同: load 入队后不立即执行, 需等 start_loading() 统一提交
           (因为 load 需逐层 DMA + LayerDoneCounter 三缓冲, 必须批量提交)
        """
        # 是否需要加载 KV 主池 (host_indices 为空时跳过 KV, 仅加载辅池)
        need_load_kv = host_indices.numel() > 0

        # 取 full_attn_allocator (混合 pool 时取子分配器; 纯 KV 时就是 allocator 本身)
        full_allocator = getattr(
            self.mem_pool_device_allocator,
            "full_attn_allocator",
            self.mem_pool_device_allocator,
        )
        if not need_load_kv:
            # 无 KV 需加载: 分配空 tensor 占位 (辅池仍可能需要 device slot)
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            # KV 主池: 在 GPU 分配 slot
            device_indices = full_allocator.alloc(len(host_indices))
            if device_indices is None:
                return None

        # 辅池: 为 pool_transfers 中 device_indices=None 的项自动分配 GPU slot
        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools,
            alloc_host=False,
            kv_device_indices=device_indices,
            kv_host_indices=host_indices,
        )
        # 辅池 alloc 失败: 回滚 KV device slot
        if pool_transfers is None and extra_pools:
            if need_load_kv:
                full_allocator.free(device_indices)
            return None

        # 入 load_queue: 等 start_loading() 批量合并 + 逐层 DMA
        self.load_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        return device_indices

    def start_loading(self) -> int:
        """📥 合并 load_queue, 逐层 Host→GPU 多池 DMA + LayerDoneCounter 三缓冲。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        scheduler → cache.ready_to_load_host_cache()
          → controller.start_loading()   ← 你在这
            → merge_ops → move_hybrid_indices → load_stream 逐层 DMA
            → return producer_id
          → scheduler: batch.hicache_consumer_index = producer_id
          → TpWorker: layer_done_counter.set_consumer(producer_id)
          → ModelRunner forward: 每层 wait_until(layer_id) 等该层 KV 就绪

        ⚙️ 行为 (在基类基础上增加 pool_transfers 传给逐层 DMA)：
          ① update_producer(): 三缓冲轮转, 获取下一个 event slot
          ② merge_ops(load_queue): 合并 KV + 辅池
          ③ move_hybrid_indices: indices 搬到正确设备
          ④ load_stream 逐层: load_to_device_per_layer(i, pool_transfers=...)
             + producer_event.complete(i) 在 load_stream record event
          ⑤ draft pool: best-effort 逐层搭载 (i < draft layer_num 时)
          ⑥ return producer_id (供 forward 流水线同步)
        """
        if not self.load_queue:
            return -1
        # ① 三缓冲轮转: 获取下一个 event slot (断言上一个该 slot 的 DMA 已完成)
        producer_id = self.layer_done_counter.update_producer()
        # ② 合并 load_queue (KV + 辅池 indices cat 到一起)
        op = CacheOperation.merge_ops(self.load_queue)
        # ③ indices 搬到正确设备 (kernel→GPU / direct→CPU)
        host_indices, device_indices, resolved_pool_transfers = (
            self.move_hybrid_indices(op)
        )
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        # ── load_stream 上逐层 DMA + 逐层 record event ──
        producer_event.start_event.record()
        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.layer_num):
                # 逐层拷贝 KV + 辅池 (pool_transfers 传给底层 kernel)
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                    pool_transfers=resolved_pool_transfers,
                )
                # draft pool best-effort: 仅当 draft 有该层时搭载
                if (
                    self.has_draft
                    and host_indices.numel() > 0
                    and i < self.mem_pool_host_draft.layer_num
                ):
                    self.mem_pool_host_draft.load_to_device_per_layer(
                        self.mem_pool_device_draft,
                        host_indices,
                        device_indices,
                        i,
                        self.io_backend,
                    )
                # 在 load_stream record 第 i 层完成 event: forward 流水线据此逐层就绪
                producer_event.complete(i)
            # 防止 CUDA caching allocator 提前回收 indices tensor
            self._record_transfer_indices_on_stream(
                self.load_stream,
                host_indices,
                device_indices,
                resolved_pool_transfers,
            )
        self.ack_load_queue.append(
            HiCacheAck(
                producer_event.start_event,
                producer_event.finish_event,
                op.node_ids,
            )
        )
        return producer_id

    def _record_transfer_indices_on_stream(
        self,
        stream: torch.Stream,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ) -> None:
        """🔒 在指定 stream 上登记所有 indices tensor, 防止 CUDA allocator 提前回收。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        start_writing / start_loading → _record_transfer_indices_on_stream()

        ⚙️ 行为：对 KV + 辅池的 host/device indices 逐一调 record_stream(stream)。
        💡 为什么需要？indices tensor 可能来自 CPU/GPU, DMA 在异步 stream 上执行,
           若主线程提前释放 tensor, CUDA allocator 可能回收显存导致 DMA 读到脏数据。
           record_stream 让 allocator 知道该 tensor 正被该 stream 使用, 延后回收。
        """
        if host_indices.is_cuda:
            host_indices.record_stream(stream)
        if device_indices.is_cuda:
            device_indices.record_stream(stream)
        for transfer in pool_transfers or []:
            if transfer.host_indices is not None and transfer.host_indices.is_cuda:
                transfer.host_indices.record_stream(stream)
            if transfer.device_indices is not None and transfer.device_indices.is_cuda:
                transfer.device_indices.record_stream(stream)

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> PrefetchOperation:
        """⬇️ 发起 L3→Host 预取 —— 构造 PrefetchOperation 入 prefetch_queue。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        scheduler → cache.prefetch_from_storage(request_id, ...)
          → controller.prefetch(...)   ← 你在这
            → PrefetchOperation 入 prefetch_queue
              → prefetch_thread: _storage_hit_query → all_reduce(MIN) → 命中决策
              → prefetch_io_aux_thread: _page_transfer 逐页读 L3 → increment/mark_terminate

        📥 参数：
          request_id      : 唯一请求标识
          host_indices    : 预分配的 Host slot (已 alloc 好, 预取数据将写入此处)
          new_input_tokens: 待预取的 token id 序列
          extra_pools     : 辅池预取描述 (含 keys / hit_policy)
        📤 返回：PrefetchOperation (调用方据此轮询 is_terminated / completed_tokens)
        """
        operation = PrefetchOperation(
            request_id,
            host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.prefetch_queue.put(operation)
        return operation

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> int:
        """💿 Host→L3 多池备份 —— 构造 StorageOperation 入 backup_queue。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        scheduler → writing_check() → _finish_write_through_ack(node)
          → cache.write_backup_storage(node, backup_len)
            → controller.write_storage(host_indices, token_ids, hash_value,
                                       prefix_keys, extra_pools)   ← 你在这
              → StorageOperation 入 backup_queue
                → backup_thread: _page_backup (辅池先写 + KV 后写) → ack_backup_queue
          → scheduler → drain_storage_control_queues → _drain_backup → release_host

        📥 参数：
          host_indices : KV 主池 Host slot
          token_ids    : token id 序列 (供 hash 计算, 若 hash_value 未给)
          hash_value   : 各 page 的 hash (未给则 _page_backup 时用 token_ids 计算)
          extra_pools  : 辅池备份描述 (含 keys)
        📤 返回：operation.id (供 ongoing_backup 字典索引, ack 时据此找到 node)
        """
        operation = StorageOperation(
            host_indices,
            token_ids,
            hash_value=hash_value,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.backup_queue.put(operation)
        return operation.id

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        """🔍 查询 L3 存储命中 —— 计算 hash + 调 batch_exists_v2 确定可用前缀长度。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        prefetch_thread: prefetch_queue → _storage_hit_query(operation)   ← 你在这
          → batch_exists_v2 (多池) / batch_exists (单 KV)
          → all_reduce(MIN) 多卡对齐命中长度 (在调用方完成)

        ⚙️ 行为：
          ① 按 page_size 逐页计算链式 hash (last_hash 滚动)
          ② 有辅池 → batch_exists_v2 (多池联合查询, 取各池命中 min)
             无辅池 → batch_exists (仅 KV) + 包装成 PoolTransferResult
          ③ update_kv_hit_pages: 记录 KV 命中页数到 operation.pool_storage_result
          ④ 截断: hash_value[:kv_hit_pages] (仅返回命中的 hash), token 数 = 命中页 × page_size

        📤 返回：(命中页的 hash 列表, 命中 token 数)
        """
        last_hash = operation.last_hash
        hash_value = []
        # 逐页计算链式 hash: 每页的 hash 依赖前一页 (prefix tree 语义)
        for start in range(0, len(operation.token_ids), self.page_size):
            last_hash = self.get_hash_str(
                operation.token_ids[start : start + self.page_size], last_hash
            )
            hash_value.append(last_hash)

        extra_info = HiCacheStorageExtraInfo(
            prefix_keys=operation.prefix_keys.copy() if operation.prefix_keys else None
        )
        # 有辅池 → 多池联合查询 (各池按 hit_policy 取 min); 无辅池 → 仅查 KV
        if operation.pool_transfers:
            hit_result = self.storage_backend.batch_exists_v2(
                hash_value, operation.pool_transfers, extra_info
            )
        else:
            kv_hit_count = self.storage_backend.batch_exists(hash_value, extra_info)
            hit_result = PoolTransferResult(
                kv_hit_pages=kv_hit_count, extra_pool_hit_pages={}
            )

        kv_hit_pages = hit_result.kv_hit_pages
        # 记录 KV 命中页数 (max 语义: 跨 batch 取最大, 因为多 batch 时后续 batch 可能命中更多)
        operation.pool_storage_result.update_kv_hit_pages(kv_hit_pages)

        # 截断到命中部分: hash_value[:kv_hit_pages], token 数 = 页数 × page_size
        return (
            hash_value[:kv_hit_pages],
            kv_hit_pages * self.page_size,
        )

    def move_hybrid_indices(
        self, operation: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]:
        """🚚 把 KV + 辅池的 indices 搬到正确设备 (kernel→GPU / direct→CPU)。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        start_writing / start_loading → move_hybrid_indices(op)   ← 你在这
          → move_indices (基类: 搬 KV indices)
          → 逐辅池 move_indices (搬辅池 indices)

        ⚙️ 行为：
          ① KV 主池: 调基类 move_indices 搬 host/device indices
          ② 辅池: 逐个调 move_indices, 重建 PoolTransfer (不改原对象)
          ③ resolved_pool_transfers 是执行时副本, 原 PoolTransfer 不变

        💡 为什么不修改原 PoolTransfer？树拥有的 transfers 可能仍引用 radix-tree
           的 host 状态, 控制器只需一份归一化的执行时副本, 避免污染树状态。
        📤 返回：(host_indices, device_indices, resolved_pool_transfers)
        """
        # ① KV 主池 indices 搬到正确设备
        host_indices, device_indices = self.move_indices(
            operation.host_indices, operation.device_indices
        )
        resolved_pool_transfers = None
        if operation.pool_transfers:
            resolved_pool_transfers = []
            for transfer in operation.pool_transfers:
                # ② 逐辅池搬 indices (与 KV 同款 move_indices 逻辑)
                transfer_host_indices, transfer_device_indices = self.move_indices(
                    transfer.host_indices, transfer.device_indices
                )
                # Keep the original PoolTransfer unchanged because tree-owned
                # transfers may still reference radix-tree host state. The
                # controller only needs a normalized execution-time copy.
                resolved_pool_transfers.append(
                    PoolTransfer(
                        name=transfer.name,
                        host_indices=transfer_host_indices,
                        device_indices=transfer_device_indices,
                        keys=transfer.keys,
                        hit_policy=transfer.hit_policy,
                        indices_from_pool=transfer.indices_from_pool,
                    )
                )
        return host_indices, device_indices, resolved_pool_transfers

    def _page_transfer(self, operation):
        """⬇️ L3→Host 逐页多池预取 —— KV 先行决定命中页数, 辅池仅在 KV 完整时跟进。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        prefetch_io_aux_thread: prefetch_buffer → _page_transfer(operation)   ← 你在这
          → super()._page_transfer (KV 逐页读 L3)
          → KV 完整完成? → _sync_trailing_keys + batch_get_v2 (辅池读 L3)

        ╔══════════════════════════════════════════════════════════════════╗
        ║  ⚠️ IO 顺序约束: KV 先行, 辅池跟进                                    ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║  ① KV pools 先读 (super()._page_transfer): 决定实际完成的页数        ║
        ║  ② 仅当 KV 完整完成 (kv_completed_pages == len(hash_value)) 才读辅池  ║
        ║     → 若 KV 提前终止 (IO 失败/超时/TP 不一致), 跳过辅池 IO 整体       ║
        ║     → 避免辅池数据与截断后的 KV 错位                                 ║
        ║  ③ 辅池读前: _sync_trailing_keys 重对齐 trailing-page keys           ║
        ║  ④ 辅池读前: _resolve_sidecar_derived_pool_transfers 填充派生池 indices ║
        ║  ⑤ 辅池读: batch_get_v2 → update_extra_pool_hit_pages               ║
        ╚══════════════════════════════════════════════════════════════════╝
        """
        # KV pools first — determines actual completed page count
        super()._page_transfer(operation)

        # Extra pools only after KV fully completes. If KV terminated early
        # (IO failure, timeout, TP mismatch), skip extra IO entirely to avoid
        # data misalignment.
        kv_completed_pages = operation.completed_tokens // self.page_size
        # KV 完整完成 (页数 == hash_value 长度) 才读辅池, 否则跳过避免错位
        if operation.pool_transfers and kv_completed_pages == len(operation.hash_value):
            # ① 重对齐 trailing-page keys: KV 命中截断后, 辅池 keys 需取命中范围的尾部 N 页
            self._sync_trailing_keys(
                operation.pool_transfers, operation.hash_value, kv_completed_pages
            )
            # ② 填充派生池 (indices_from_pool) 的 host_indices/keys: 从源池复制
            self._resolve_sidecar_derived_pool_transfers(operation)
            # ③ 逐辅池读 L3 → 记录各辅池命中页数
            results = self.storage_backend.batch_get_v2(operation.pool_transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)
        # 标记辅池 IO 完成 (无论是否执行, 后续不再尝试)
        operation.pool_transfers_done = True

    def _page_backup(self, operation):
        """💿 Host→L3 逐页多池备份 —— 辅池先写, KV 后写。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        backup_thread: backup_queue → _page_backup(operation)   ← 你在这
          → 辅池 batch_set_v2 (先写)
          → super()._page_backup (KV 后写)
          → ack_backup_queue

        ⚠️ 与 _page_transfer 的顺序相反: backup 时辅池先写, KV 后写。
        💡 为什么 backup 辅池先写？backup 不存在"截断"问题 (写入是全量的),
           先写辅池可让 KV 的 completed_tokens 直接反映 KV 写入进度,
           与基类 _page_backup 的 completed_tokens 语义保持一致。
        """
        # Backup extra pools
        # ① 辅池先写: 填充派生池 indices → batch_set_v2 → 记录各辅池成功页数
        if operation.pool_transfers:
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_set_v2(operation.pool_transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)

        # Backup kv pools
        # ② KV 后写: 基类逐 batch 写 L3, completed_tokens 反映 KV 写入进度
        super()._page_backup(operation)

    def _resolve_sidecar_derived_pool_transfers(self, operation):
        """🔧 填充派生池 (indices_from_pool) 的 host_indices / keys —— 从源池复制。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        _page_transfer / _page_backup → _resolve_sidecar_derived_pool_transfers(op)

        ⚙️ 行为：
          遍历 operation.pool_transfers, 对 indices_from_pool != None 的派生池:
          ① indices_from_pool == KV: 从 operation 本体复制 host_indices + hash_value
          ② indices_from_pool == 其他辅池: 从同 operation 中找到同名源池复制
          ③ keys 为 None 时也同步填充

        💡 什么是派生池？SWA/Indexer 等辅池可能复用 KV 主池的 indices (不独立分配),
           indices_from_pool 标记其源池。L3 IO 前需把源池的 indices 填到派生池上。
        ⚠️ 源池缺失会抛 AssertionError (配置错误, 不应发生)
        """
        for transfer in operation.pool_transfers:
            if transfer.indices_from_pool is None:
                # 非派生池: 已有自己的 indices, 跳过
                continue
            if transfer.indices_from_pool != PoolName.KV:
                # 派生自其他辅池: 在同 operation 中找同名源池 (indices_from_pool=None 即源池)
                source = next(
                    (
                        t
                        for t in operation.pool_transfers
                        if t.indices_from_pool is None
                        and t.name == transfer.indices_from_pool
                    ),
                    None,
                )
                if source is None:
                    raise AssertionError(
                        "Storage sidecar derived pool source missing: "
                        f"{transfer.name} from {transfer.indices_from_pool}."
                    )
                # 从源池复制 host_indices
                transfer.host_indices = source.host_indices
                if transfer.keys is None:
                    transfer.keys = source.keys
            else:
                # 派生自 KV 主池: 从 operation 本体复制
                transfer.host_indices = operation.host_indices
                if transfer.keys is None:
                    transfer.keys = operation.hash_value

    def _sync_trailing_keys(
        self,
        pool_transfers: list[PoolTransfer],
        all_hashes: list[str],
        kv_hit_pages: int,
    ) -> None:
        """🔧 KV 命中截断后, 重对齐 trailing-page 辅池的 keys。

        ━━━━━━━━━━━━━━ 1️⃣ 调用链 ━━━━━━━━━━━━━━
        _page_transfer → _sync_trailing_keys(pool_transfers, hash_value, kv_hit_pages)

        ⚙️ 行为：
          对 hit_policy == TRAILING_PAGES 的辅池 (如 Mamba/SWA state):
            trailing_n = 原始 keys 长度 (Mamba N=1, SWA N>1)
            transfer.keys = all_hashes[max(0, kv_hit_pages - trailing_n) : kv_hit_pages]
          即取命中范围的"最后 N 页" hash, 而非原始目标范围的最后 N 页。

        💡 为什么需要？storage 命中可能短于原始目标前缀, trailing-page 辅池只需
           覆盖命中范围的尾部 N 页 (Mamba 只需最后 1 页 state, SWA 需最后 N 页)。
           若不重对齐, keys 会指向未命中的 page, 导致 batch_get_v2 读到空数据。
        """
        for transfer in pool_transfers:
            if transfer.hit_policy != PoolHitPolicy.TRAILING_PAGES:
                # 仅处理 trailing-page 策略的辅池 (ALL_PAGES 不需重对齐)
                continue
            # trailing_n: 该辅池需要的尾部页数 (Mamba=1, SWA>1); 无 keys 时默认 1
            trailing_n = len(transfer.keys) if transfer.keys else 1
            # 取命中范围 [0, kv_hit_pages) 的最后 trailing_n 页 hash
            transfer.keys = all_hashes[max(0, kv_hit_pages - trailing_n) : kv_hit_pages]

    def _resolve_pool_transfers_allocation(
        self,
        extra_pools: Optional[list[PoolTransfer]],
        alloc_host: bool,
        kv_device_indices: Optional[torch.Tensor] = None,
        kv_host_indices: Optional[torch.Tensor] = None,
    ) -> Optional[list[PoolTransfer]]:
        """🔧 为 pool_transfers 中 indices=None 的项自动分配 host 或 device slot。
        📥 参数：
          alloc_host        : True=分配 Host slot (write 路径) / False=分配 Device slot (load 路径)
          kv_device_indices : KV 主池 device indices (派生自 KV 的辅池复用)
          kv_host_indices   : KV 主池 host indices (同上)
        📤 返回：填充好 indices 的 extra_pools; None=分配失败 (调用方应回滚 KV slot)
        ⚠️ 原子性: 任一辅池 alloc 失败, 已成功分配的全部回滚, 保证不泄漏

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        write() → _resolve_pool_transfers_allocation(alloc_host=True)   ← 分配 Host slot
        load()  → _resolve_pool_transfers_allocation(alloc_host=False)  ← 分配 Device slot
        """
        if not extra_pools:
            return None
        # (pool, free_fn, indices) for atomic rollback on failure.
        # 记录本次新分配的 (池, 释放函数, indices), 失败时逐个 free
        newly_allocated: list[tuple[PoolTransfer, Callable, torch.Tensor]] = []
        derived_transfers: list[PoolTransfer] = []

        def rollback_allocated() -> None:
            # 原子回滚: 释放所有已分配的 indices, 并把 pool 上的 indices 置 None
            for prev_pool, prev_free_fn, prev_indices in newly_allocated:
                prev_free_fn(prev_indices)
                if alloc_host:
                    prev_pool.host_indices = None
                else:
                    prev_pool.device_indices = None

        # ① 遍历 extra_pools, 跳过已有 indices 的项和派生池 (indices_from_pool)
        for pool in extra_pools:
            # 派生池延后处理 (等源池 indices 填好再复制)
            if pool.indices_from_pool is not None:
                derived_transfers.append(pool)
                continue

            # ② 查 entry_map 取该池的 alloc/free/evict 函数
            entry = self.mem_pool_host.entry_map.get(pool.name)
            if entry is None:
                continue
            if alloc_host:
                # write 路径: 需分配 Host slot (前提: device_indices 已有, host_indices 缺)
                if pool.host_indices is not None or pool.device_indices is None:
                    continue
                alloc_fn = entry.host_pool.alloc
                free_fn = entry.host_pool.free
                evict_fn = entry.host_evict_fn
                size = len(pool.device_indices)
            else:
                # load 路径: 需分配 Device slot (前提: host_indices 已有, device_indices 缺)
                if pool.device_indices is not None or pool.host_indices is None:
                    continue
                # device_alloc_fn / device_free_fn override entry.device_pool's
                # methods for pools whose device_pool is a raw KV pool (layout)
                # rather than an allocator (e.g. SWA).
                alloc_fn = entry.device_alloc_fn or entry.device_pool.alloc
                free_fn = entry.device_free_fn or entry.device_pool.free
                evict_fn = entry.device_evict_fn
                size = len(pool.host_indices)
            # 尝试分配
            indices = alloc_fn(size)
            # ③ alloc 失败 → 调 evict_fn 腾空间后重试
            if indices is None and evict_fn:
                evict_fn(size)
                indices = alloc_fn(size)
            # ④ 仍失败 → 原子回滚 (free 所有已分配的) + 返回 None
            if indices is None:
                rollback_allocated()
                return None

            if alloc_host:
                pool.host_indices = indices
            else:
                pool.device_indices = indices
            newly_allocated.append((pool, free_fn, indices))

        # Assign indices to deferred pools from their source.
        # ⑤ 派生池: 从源池 (KV 或其他辅池) 复制 indices
        # 派生池: 从源池复制 indices (源池已在上面分配好)
        for pool in derived_transfers:
            if pool.indices_from_pool == PoolName.KV:
                # 派生自 KV 主池: 直接用 KV 的 indices
                pool.host_indices = kv_host_indices
                pool.device_indices = kv_device_indices
                continue

            # 派生自其他辅池: 找到同名源池 (indices_from_pool=None) 复制
            source = next(
                (
                    transfer
                    for transfer in extra_pools
                    if transfer.indices_from_pool is None
                    and transfer.name == pool.indices_from_pool
                ),
                None,
            )
            if source is None:
                # 源池缺失: 回滚 + 返回 None
                rollback_allocated()
                return None
            pool.host_indices = source.host_indices
            pool.device_indices = source.device_indices
        return extra_pools
