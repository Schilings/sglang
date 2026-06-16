from __future__ import annotations

"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import threading
import time
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

from sglang.srt.distributed import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)

device_module = get_device_module()


class LayerLoadingEvent:
    """单次 KV cache 加载操作的逐层 CUDA event 追踪器。

    每一层的 Host→Device 拷贝完成后，在 load_stream 上 record 一个 CUDA event，
    使得 compute stream 可以通过 wait_event() 按层粒度等待，实现传输与计算的重叠。
    """
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        # 第 i 层拷贝完成后，load_events[i] 会被 record
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        # start event on controller stream，用于 load_stream 等待调度器就绪
        self.start_event = device_module.Event()

    def complete(self, layer_index: int):
        """在 load_stream 上 record 第 layer_index 层拷贝完成的 event。"""
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int):
        """在当前 stream（compute stream）上等待第 layer_index 层的拷贝完成。"""
        device_module.current_stream().wait_event(self.load_events[layer_index])

    @property
    def finish_event(self):
        """最后一层的 event，代表整个加载操作完成。"""
        return self.load_events[-1]


class LayerDoneCounter:
    """逐层加载完成追踪器，实现 KV cache Host→GPU 异步传输与注意力计算的逐层同步。

    核心目的：让计算不必等全部层拷贝完毕，而是每层 KV 就绪即可开始计算，从而 overlap 传输延迟。

    三层结构：
      - LayerLoadingEvent: 单次加载操作的逐层 CUDA event，第 i 层拷贝完成后 complete(i) 记录 event
      - LayerDoneCounter: 管理 3 个 LayerLoadingEvent 的循环缓冲，支持 producer-consumer 模式

    工作流程（以 start_loading 为例）：
      1. update_producer()  → 获取下一个可用的 event slot（producer_index 轮转）
      2. 在 load_stream 上逐层拷贝 KV：
           for i in range(layer_num):
               host→device 拷贝第 i 层
               producer_event.complete(i)   ← 在 load_stream 上记录第 i 层完成的 CUDA event
      3. 返回 producer_id 给调度器
      4. 调度器 set_consumer(producer_id) → 绑定当前等待的 event slot
      5. 计算 prefill 时，wait_until(layer_threshold) →
         在 compute stream 上等待第 layer_threshold 层的 CUDA event
         → 该层 KV 就绪后计算立刻开始，不必等全部层拷贝完成

    为什么是 3 个 counter（三缓冲）：
      - slot A: 当前批次正在被 consumer 等待
      - slot B: 下一批次的 load 正在执行
      - slot C: 确保上一批次完成且 event 已 query 就绪可复用
      update_producer 中的断言 finish_event.query() 保证了复用前该 slot 的所有层拷贝确实已完成。

    上层使用接口（LayerLoadingEvent 是内部实现，上层无需直接操作）：
      - HiCacheController.start_loading()：
          调用 update_producer() 获取 slot → 在 load_stream 上逐层拷贝并 complete(i)
          → 返回 producer_id
      - TpWorker.register_hicache_layer_transfer_counter(counter)：
          注册 counter 引用到 model runner，使 model runner 可调用 set_consumer / wait_until
      - ModelRunner 前向计算（各 memory_pool 的 get_key/value_buffer）：
          每层计算前调用 wait_until(layer_id) → compute stream 等该层 KV 拷贝完成
      - HiRadixCache.is_load_back_event_done(consumer_index)：
          通过 events[consumer_index].finish_event.query() 检查整个加载是否完成
      - HiRadixCache.ready_to_load_host_cache()：
          封装 start_loading()，返回 consumer_index 给调度器
      - Scheduler：
          调度时 new_batch.hicache_consumer_index = ready_to_load_host_cache()
          → batch 运行时通过 set_consumer 绑定 → 各层前向时 wait_until
    """
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # 三缓冲：slot A 被 consumer 等待 / slot B 正在 load / slot C 已完成可复用
        self.num_counters = 3
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        # producer: 当前 load 操作使用的 event slot 索引，-1 表示尚未开始
        self.producer_index = -1
        # consumer: 当前 compute 等待的 event slot 索引，-1 表示无需等待
        self.consumer_index = -1

    def update_producer(self):
        """轮转到下一个 event slot 供 load 操作使用。

        断言保证该 slot 的上一次加载已全部完成（finish_event.query()），
        否则复用会导致 consumer 等到错误的事件。
        """
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ].finish_event.query(), (
            "Producer finish event should be ready before being reused."
        )
        return self.producer_index

    def set_consumer(self, index: int):
        """绑定 consumer 等待的 event slot，通常为 start_loading() 返回的 producer_id。"""
        self.consumer_index = index

    def wait_until(self, threshold: int):
        """在 compute stream 上等待 consumer slot 的第 threshold 层拷贝完成。

        compute stream 会阻塞直到 load stream 上第 threshold 层的 KV 已就绪，
        从而实现逐层 overlap：该层 KV 到位后计算立刻开始，不必等全部层拷贝完成。
        """
        if self.consumer_index < 0:
            return
        self.events[self.consumer_index].wait(threshold)

    def reset(self):
        """重置 producer/consumer 索引，用于 cache reset 时清理状态。"""
        self.producer_index = -1
        self.consumer_index = -1


class CacheOperation:
    """GPU↔Host 之间一次 KV cache 搬运操作的描述符。

    无论是 write（GPU→Host）还是 load（Host→GPU），都用 CacheOperation 描述，
    主要记录：搬运哪些 slot（host_indices / device_indices）、属于哪个树节点、优先级。

    生命周期：
      1. HiCacheController.write()/load() 将操作 append 到 write_queue / load_queue
      2. flush_write()/start_loading() 调用 merge_ops() 将队列中多个操作合并为一个
      3. 合并后在 load_stream/write_stream 上执行 DMA 拷贝
      4. 拷贝完成后生成 HiCacheAck 入 ack_write_queue / ack_load_queue

    merge_ops 设计：
      同一批次可能有多个小操作（每个对应一个树节点），merge_ops 将它们
      cat 为一次大 DMA，减少 kernel launch 开销。合并后 node_ids 保留
      所有原始节点 ID，用于 ack 时逐节点回调。
    """

    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
    ):
        # Host 端 KV pool 的 slot 索引
        self.host_indices = host_indices
        # Device 端 KV pool 的 slot 索引
        self.device_indices = device_indices
        # 操作涉及的树节点 ID 列表（merge 后可能包含多个）
        self.node_ids = [node_id]
        # 传输数据缓冲区（可选，用于 storage backend）
        self.data = None

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        self.priority = priority if priority is not None else self.id

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        """将多个 CacheOperation 合并为一个，cat host/device indices 以减少 DMA 次数。"""
        assert len(ops) > 0
        if len(ops) == 1:
            return ops[0]

        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        # 取最高优先级（数值最小），确保高优先级操作不会被低优先级拖慢
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged_op = CacheOperation(host_indices, device_indices, -1, priority)
        merged_op.node_ids = node_ids
        return merged_op

    def __lt__(self, other: CacheOperation):
        """按 priority 排序，数值越小优先级越高。"""
        return self.priority < other.priority


class HiCacheAck(NamedTuple):
    """DMA 搬运完成的确认凭据，由 flush_write / start_loading 生成，由 HiRadixCache 消费。

    生产端（HiCacheController）：
      - flush_write() 完成后生成 ack 入 ack_write_queue
      - start_loading() 完成后生成 ack 入 ack_load_queue

    消费端（HiRadixCache）：
      - writing_check() 遍历 ack_write_queue，对每个 ack：
          1. finish_event.query() / synchronize() 等待 DMA 完成
          2. 遍历 node_ids，从 ongoing_write_through 中 pop 出对应树节点
          3. 确认 KV 已到 host → 触发 write_backup_storage（如有 storage 层）→ dec_lock_ref
      - loading_check() 遍历 ack_load_queue，对每个 ack：
          1. finish_event.query() 检查 DMA 是否完成
          2. 遍历 node_ids，从 ongoing_load_back 中 pop 出对应树节点
          3. dec_lock_ref 释放保护

    字段说明：
      - start_event:  DMA 拷贝的起始 CUDA event（可用于 stream 同步）
      - finish_event: DMA 拷贝的结束 CUDA event（query/synchronize 判断拷贝是否完成）
      - node_ids:     本次 DMA 涉及的树节点 ID 列表（merge 后可能包含多个节点）
    """
    start_event: device_module.Event
    finish_event: device_module.Event
    node_ids: List[int]


class TransferBuffer:
    """基于有界 Queue 的线程安全缓冲区，设计用于 GPU↔Host 传输操作的生产者-消费者解耦。

    核心设计：
      - 有界队列（maxsize=buffer_count）：当队列满时 put() 阻塞，自动反压生产者
      - stop_event：外部线程安全停止信号。put() 在队列满时轮询检查 stop_event，
        收到停止信号后立即退出，避免 shutdown 时死锁
      - 1 秒轮询超时：put/get 都用 timeout=1 轮询，而非无限阻塞，以便及时响应 stop_event

    put() 的阻塞语义：
      正常模式（block=True）：队列满时反复重试（每次等 1s），直到成功入队或 stop_event 被设置
      非阻塞模式（block=False）：队列满时立即返回（丢弃），不重试

    get() 的语义：
      队列空时返回 None，不阻塞等待 stop_event（因为消费者通常在循环中调用，空队列是正常情况）

    当前使用状况：
      HiCacheController 中创建了 write_buffer / load_buffer 两个实例，但当前 write/load
      操作实际走的是 write_queue / load_queue（List[CacheOperation]）+ merge_ops 的同步路径，
      TransferBuffer 的 put/get 尚未被调用，仅 clear() 在 reset() 中使用。
      该类可能是为未来的异步双缓冲流水线预留的。

    预期的异步流水线用法（当前未启用）：
      1. 生产者线程：HiRadixCache.evict() → controller.write_buffer.put(CacheOperation(...))
      2. 消费者线程：后台 DMA 线程循环 → op = controller.write_buffer.get() → 执行拷贝
      3. 好处：拷贝准备（alloc/merge）和 DMA 传输可以在不同线程上重叠执行
      4. stop_event：controller.reset() 或 shutdown 时设置，让 put/get 安全退出
    """

    def __init__(self, stop_event, buffer_count: int = 3) -> None:
        self.stop_event = stop_event
        # 有界队列，buffer_count 控制流水线深度（默认 3 = 三缓冲）
        self.buffers = Queue(maxsize=buffer_count)

    def full(self) -> bool:
        return self.buffers.full()

    def empty(self) -> bool:
        return self.buffers.empty()

    def put(self, item, block=True, timeout=1) -> None:
        """将操作入队。队列满时轮询重试，直到成功或 stop_event 被设置。"""
        while not self.stop_event.is_set():
            try:
                self.buffers.put(item, block=block, timeout=timeout)
                break
            except Full:
                if not block:
                    break
                continue
            except Exception as e:
                logger.error(e)

    def get(self, block=True, timeout=1) -> Optional[CacheOperation]:
        """从队列取操作。队列空时返回 None，不等待 stop_event。"""
        try:
            return self.buffers.get(block=block, timeout=timeout)
        except Empty:
            return None
        except Exception as e:
            logger.error(e)

    def clear(self):
        self.buffers.queue.clear()


class StorageOperation:
    """Host↔Storage 之间一次 KV cache 搬运操作的描述符（基类）。

    StorageOperation 描述的是第三级存储（磁盘/远端）与 Host 内存之间的 KV 搬运，
    对应 CacheOperation 描述的是 GPU↔Host 之间的搬运。

    两种子类使用场景：
      - StorageOperation（本类）：write_storage() 中描述 Host→Storage 的备份操作
      - PrefetchOperation（子类）：prefetch() 中描述 Storage→Host 的预取操作

    生命周期（以 write_storage 为例）：
      1. write_storage() 创建 StorageOperation → 放入 backup_queue
      2. backup_thread 从 backup_queue 取出 → _page_backup() 逐页写入 storage
      3. 写入完成后 ack_backup_queue 通知上层

    生命周期（以 prefetch 为例）：
      1. prefetch() 创建 PrefetchOperation → 放入 prefetch_queue
      2. prefetch_thread 从 prefetch_queue 取出 → 查询 storage hit → 放入 prefetch_buffer
      3. prefetch_io_aux_thread 从 prefetch_buffer 取出 → _page_transfer() 逐页读回 host
      4. 调度器通过 is_terminated() / completed_tokens 判断预取进度

    字段说明：
      - host_indices:  Host 端 KV pool 的 slot 索引（读/写的目标位置）
      - token_ids:     本次操作涉及的 token ID 列表
      - last_hash:     前缀路径上最后一个节点的 hash（用于 storage 查询）
      - hash_value:    本次操作涉及的各页 hash 值列表（逐页检索/写入的 key）
      - completed_tokens: 已完成的 token 数（逐页递增，用于进度追踪）
      - prefix_keys:   前缀 hash 列表（用于 storage 的层级索引，逐批次累积）
    """

    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.host_indices = host_indices
        self.token_ids = token_ids
        self.last_hash = last_hash
        self.completed_tokens = 0
        self.hash_value = hash_value if hash_value is not None else []
        self.prefix_keys = prefix_keys

        self.id = StorageOperation.counter
        StorageOperation.counter += 1

    def __lt__(self, other: StorageOperation):
        return self.id < other.id


class PrefetchOperation(StorageOperation):
    """Storage→Host 预取操作的描述符，增加了请求级生命周期管理。

    与 StorageOperation（用于备份）的区别：
      - 绑定了 request_id：标识哪个请求触发了此次预取
      - 线程安全的终止机制：调度器可在任意时刻 mark_terminate()，IO 线程通过
        increment() 返回值或 is_terminated() 感知后立即停止传输
      - 进度追踪：completed_tokens 被 IO 线程 increment()，调度器可查询判断预取进度

    典型交互流程：
      1. 调度器：operation = controller.prefetch(request_id, ...)
         → 放入 prefetch_queue，返回 operation 引用
      2. prefetch_thread：查询 storage 命中情况 → 决定是否值得预取
         → 值得则放入 prefetch_buffer，否则 revoke
      3. prefetch_io_aux_thread：从 prefetch_buffer 取出 → _page_transfer() 逐页读回
         → 每页成功后 operation.increment(page_size)，失败则 mark_terminate()
      4. 调度器：can_terminate_prefetch(operation) 检查是否可以结束预取
         → 超时或完成则 terminate_prefetch(operation) → mark_terminate()
      5. IO 线程下次 increment() 返回 False → 停止传输
    """

    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.request_id = request_id

        # 线程安全的终止标志：调度器设置，IO 线程读取
        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()

        super().__init__(host_indices, token_ids, last_hash, prefix_keys=prefix_keys)

    def increment(self, num_tokens: int):
        """IO 线程调用：增加 completed_tokens。返回 False 表示已被终止，调用方应停止传输。"""
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        """调度器调用：标记操作为终止。IO 线程下次 increment() 或 is_terminated() 检查时停止。"""
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        return self._terminated_flag


class HiCacheController:

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
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
        enable_storage_metrics: bool = False,
    ):
        self.tp_group = tp_group
        self.attn_cp_group = attn_cp_group
        self.attn_tp_group = attn_tp_group
        self.pp_group = pp_group
        self.prefetch_sync_groups: List[torch.distributed.ProcessGroup] = []
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        if isinstance(mem_pool_device, HybridLinearKVPool):
            mem_pool_device = mem_pool_device.full_kv_pool
        self.mem_pool_device = mem_pool_device
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size
        self.io_backend = io_backend
        self.enable_storage = False
        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage_metrics = enable_storage_metrics

        # Draft KV pool support (best-effort piggyback on target L2/L3 ops).
        self.has_draft = False
        self.mem_pool_device_draft = None
        self.mem_pool_host_draft = None
        self.draft_page_get_func = None
        self.draft_page_set_func = None

        # Default storage page IO functions (may be overridden by attach).
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set

        # Dedicated stop event for storage background threads (prefetch/backup).
        # NOTE: Do NOT reuse `self.stop_event` here since it also guards core HiCache
        # transfer buffers (CPU<->GPU). We want to allow runtime attach/detach of
        # storage without stopping the whole controller.
        self.storage_stop_event = threading.Event()

        self.device = self.mem_pool_device.device
        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        # self.write_queue = PriorityQueue[CacheOperation]()
        self.load_queue: List[CacheOperation] = []
        self.write_queue: List[CacheOperation] = []
        self.ack_load_queue: List[HiCacheAck] = []
        self.ack_write_queue: List[HiCacheAck] = []

        self.stop_event = threading.Event()
        self.write_buffer = TransferBuffer(self.stop_event)
        self.load_buffer = TransferBuffer(self.stop_event, buffer_count=10)

        self.write_stream = device_module.Stream()
        self.load_stream = device_module.Stream()

        # If a storage backend is provided at startup, treat it as an implicit attach,
        # so init/runtime share the same lifecycle semantics and code paths.
        if storage_backend is not None:
            try:
                self.attach_storage_backend(
                    storage_backend=storage_backend,
                    prefetch_threshold=prefetch_threshold,
                    model_name=model_name,
                    storage_backend_extra_config=storage_backend_extra_config,
                )
            except ValueError as e:
                # Preserve the historical error shape on init for unknown backends.
                raise ValueError(f"Failed to create storage backend: {e}") from e

    def get_attn_cp_rank_and_size(self) -> tuple[int, int]:
        """Derive CP rank/size from the attn_cp process group."""
        if self.attn_cp_group is not None:
            return (
                torch.distributed.get_rank(group=self.attn_cp_group),
                torch.distributed.get_world_size(group=self.attn_cp_group),
            )
        return 0, 1

    def _create_prefetch_sync_groups(self) -> None:
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        self.prefetch_sync_groups = []
        seen_rank_sets = set()

        if self.attn_cp_group is not None or self.attn_tp_group is not None:
            base_groups = [self.attn_cp_group, self.attn_tp_group]
        else:
            base_groups = [self.tp_group]

        for group in base_groups:
            if group is None or torch.distributed.get_world_size(group=group) == 1:
                continue
            group_ranks = tuple(torch.distributed.get_process_group_ranks(group))
            if group_ranks in seen_rank_sets:
                continue
            seen_rank_sets.add(group_ranks)
            self.prefetch_sync_groups.append(
                create_custom_parallel_group(
                    group_ranks=list(group_ranks), backend="gloo"
                )
            )

    def _destroy_prefetch_sync_groups(self) -> None:
        for group in self.prefetch_sync_groups:
            try:
                torch.distributed.destroy_process_group(group)
            except Exception:
                pass
        self.prefetch_sync_groups = []

    def _all_reduce_prefetch_groups(self, tensor: torch.Tensor, op) -> None:
        for group in self.prefetch_sync_groups:
            torch.distributed.all_reduce(tensor, op=op, group=group)

    def _start_storage_threads(self):
        """Start storage prefetch/backup threads and their queues.

        This is used by runtime attach, and also by reset when storage is enabled.
        """
        assert self.enable_storage
        assert not self.storage_stop_event.is_set()

        self.prefetch_thread = threading.Thread(
            target=self.prefetch_thread_func, daemon=True
        )
        self.backup_thread = threading.Thread(
            target=self.backup_thread_func, daemon=True
        )
        self.prefetch_queue = Queue()
        self.backup_queue = Queue()

        self.prefetch_revoke_queue: Queue[str] = Queue()
        self.ack_backup_queue: Queue[StorageOperation] = Queue()
        self.host_mem_release_queue: Queue[torch.Tensor] = Queue()

        self.prefetch_thread.start()
        self.backup_thread.start()

    def _stop_storage_threads(self):
        """Stop storage prefetch/backup threads and drain internal queues.

        Caller should ensure no in-flight requests.
        """
        # Always request stop. This is safe even when storage is already disabled,
        # and makes detach truly idempotent (previous partial detach may have left
        # threads alive).
        # NOTE: do NOT clear stop_event unless threads have fully stopped; otherwise
        # a still-alive thread may resume and touch released state.
        self.storage_stop_event.set()

        # Best-effort wakeups so threads exit promptly even if blocked on queues.
        try:
            if hasattr(self, "prefetch_queue"):
                self.prefetch_queue.put_nowait(None)
            if hasattr(self, "backup_queue"):
                self.backup_queue.put_nowait(None)
            if hasattr(self, "prefetch_buffer"):
                self.prefetch_buffer.put_nowait(None)
        except Exception:
            pass

        # Best-effort joins (threads are daemon, but join keeps state clean).
        threads = []
        if hasattr(self, "prefetch_thread"):
            threads.append(self.prefetch_thread)
        if hasattr(self, "backup_thread"):
            threads.append(self.backup_thread)
        if hasattr(self, "prefetch_io_aux_thread"):
            threads.append(self.prefetch_io_aux_thread)

        for t in threads:
            try:
                t.join(timeout=10)
            except Exception:
                pass

        alive = [t for t in threads if getattr(t, "is_alive", lambda: False)()]
        if alive:
            logger.error(
                "Failed to stop HiCache storage threads cleanly: %s",
                [getattr(t, "name", repr(t)) for t in alive],
            )
            raise RuntimeError("Failed to stop HiCache storage threads cleanly.")

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        """Attach (enable) storage backend at runtime.

        Requirement: no in-flight requests. This call is expected to run on the scheduler
        thread (control path), not concurrently with prefetch/backup.
        """
        if self.enable_storage:
            raise RuntimeError("Storage backend already attached.")

        # Defensive: a previous partial detach may have flipped `enable_storage` but
        # left background threads alive. Attaching on top of them is unsafe.
        try:
            self._stop_storage_threads()
        except Exception as e:
            raise RuntimeError(
                "Cannot attach storage backend: previous detach did not stop storage threads cleanly."
            ) from e

        # Rollback-safe init: if creation fails, keep controller state consistent
        # for future attach attempts.
        self.storage_backend_type = storage_backend
        from sglang.srt.mem_cache.utils import get_hash_str

        self.get_hash_str = get_hash_str
        self.storage_config = self._generate_storage_config(
            model_name, storage_backend_extra_config
        )
        # for MLA models, only one rank needs to backup the KV cache
        self.backup_skip = (
            self.storage_config.is_mla_model
            # todo: load balancing
            and self.storage_config.tp_rank != 0
        )

        # Use storage backend factory for dynamic backend creation
        from sglang.srt.mem_cache.storage import StorageBackendFactory

        try:
            self.storage_backend = StorageBackendFactory.create_backend(
                storage_backend, self.storage_config, self.mem_pool_host
            )
            self.storage_backend.register_mem_pool_host(self.mem_pool_host)

            self.enable_storage = True
            # todo: threshold policy for prefetching
            self.prefetch_threshold = max(prefetch_threshold, self.page_size)
            self.prefetch_capacity_limit = max(
                0, int(0.8 * (self.mem_pool_host.size - self.mem_pool_device.size))
            )
            # tracking the number of tokens locked in prefetching, updated by the main scheduler thread
            self.prefetch_tokens_occupied = 0

            # Use dedicated gloo groups so storage prefetch sync is isolated
            # from other collectives and consistent across CPxTP participants.
            self._create_prefetch_sync_groups()

            # Select the get and set functions
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set

            if (
                self.storage_backend_type
                in ["hf3fs", "mooncake", "eic", "nixl", "simm"]
            ) or (
                self.storage_backend_type == "dynamic"
                and bool(self.storage_config.extra_config.get("interface_v1", 0))
            ):
                self.page_get_func = self._page_get_zero_copy
                self.page_set_func = self._page_set_zero_copy

            self._maybe_register_draft_with_storage()

            # Ensure stop_event is clear before starting threads.
            self.storage_stop_event.clear()
            self._start_storage_threads()
        except Exception:
            # Best-effort cleanup for partial init.
            try:
                self._stop_storage_threads()
            except Exception:
                pass
            self._destroy_prefetch_sync_groups()
            try:
                if (
                    hasattr(self, "storage_backend")
                    and self.storage_backend is not None
                ):
                    if hasattr(self.storage_backend, "close"):
                        self.storage_backend.close()
            except Exception:
                pass
            self.storage_backend = None
            self.storage_backend_type = None
            self.enable_storage = False
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set
            self.draft_page_get_func = None
            self.draft_page_set_func = None
            raise

    def detach_storage_backend(self):
        """Detach (disable) storage backend at runtime.

        Requirement: no in-flight requests. This will stop storage threads and release
        the backend instance (best-effort close).
        """
        # Idempotent cleanup: even if `enable_storage` is already False,
        # we may still have leftover resources (threads/backend/process group) from a
        # previous partial detach. We attempt cleanup whenever possible.
        try:
            self._stop_storage_threads()
        except Exception as e:
            # Do not proceed tearing down backend/process group if threads are not
            # fully stopped; otherwise still-alive threads may touch released state.
            # Caller can retry detach.
            logger.exception("Stop storage threads failed: %s", e)
            # IMPORTANT: Do not silently succeed. Upper layers rely on exceptions here
            # to avoid flipping `enable_storage` flags while threads are still alive.
            raise RuntimeError("Stop storage threads failed; detach aborted.") from e

        # Best-effort destroy process groups created for storage ops.
        self._destroy_prefetch_sync_groups()

        # Best-effort close (some backends rely on GC/destructor).
        try:
            if (
                hasattr(self, "storage_backend")
                and self.storage_backend is not None
                and hasattr(self.storage_backend, "close")
            ):
                self.storage_backend.close()
        except Exception:
            logger.exception("Failed to close storage backend cleanly.")

        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage = False
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        # Now it's safe to clear the stop event for future re-attach.
        self.storage_stop_event.clear()

    def _generate_storage_config(
        self,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        if storage_backend_extra_config is None:
            storage_backend_extra_config = {}

        if is_dp_attention_enabled():
            self.tp_rank = get_attention_tp_rank()
            self.tp_size = get_attention_tp_size()
            self.dp_rank = get_attention_dp_rank()
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.dp_rank = 0

        self.pp_rank = get_pipeline_model_parallel_rank()
        self.pp_size = get_pipeline_model_parallel_world_size()

        # Currently, NPUMLATokenToKVPool is the subclass of MLATokenToKVPool.
        # DeepSeekV4TokenToKVPool has compressed MLA-style rank-replicated cache
        # data. storage only needs rank 0 to write it back.
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        is_mla_model = isinstance(self.mem_pool_device, MLATokenToKVPool)
        is_compressed_mla_model = isinstance(
            self.mem_pool_device, DeepSeekV4TokenToKVPool
        )
        is_rank_replicated = is_mla_model or is_compressed_mla_model
        # Least Common Multiple among heterogeneous tp size
        tp_lcm_size = storage_backend_extra_config.pop("tp_lcm_size", None)
        should_split_heads = False

        if tp_lcm_size:
            assert (
                tp_lcm_size % self.tp_size == 0
            ), "tp_lcm_size must be divisible by tp_size."
            should_split_heads = (
                not is_rank_replicated
                and self.mem_pool_host.layout == "page_head"
                and tp_lcm_size > self.tp_size
            )

        attn_cp_rank, attn_cp_size = self.get_attn_cp_rank_and_size()

        return HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            # TODO(hzh): Rename is_mla_model to is_rank_replicated.
            is_mla_model=is_rank_replicated,
            enable_storage_metrics=self.enable_storage_metrics,
            is_page_first_layout=self.mem_pool_host.layout == "page_first",
            model_name=model_name,
            tp_lcm_size=tp_lcm_size,
            should_split_heads=should_split_heads,
            extra_config=storage_backend_extra_config,
        )

    def reset(self):
        self.stop_event.set()
        self.storage_stop_event.set()

        self.write_queue.clear()
        self.load_queue.clear()
        self.write_buffer.clear()
        self.load_buffer.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        if self.enable_storage:
            self.prefetch_thread.join()
            self.backup_thread.join()
            self.prefetch_queue.queue.clear()
            self.backup_queue.queue.clear()
            self.prefetch_revoke_queue.queue.clear()
            self.ack_backup_queue.queue.clear()
            self.host_mem_release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

        self.stop_event.clear()
        self.storage_stop_event.clear()

        if self.enable_storage:
            self.prefetch_thread = threading.Thread(
                target=self.prefetch_thread_func, daemon=True
            )
            self.backup_thread = threading.Thread(
                target=self.backup_thread_func, daemon=True
            )
            self.prefetch_thread.start()
            self.backup_thread.start()

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        # For now, kernel write-back keeps host indices on CPU only for page_first.
        # More layouts can use this path once their write-back kernels accept CPU
        # destination indices.
        if self.io_backend == "kernel" and self.mem_pool_host.layout == "page_first":
            host_indices, device_indices = op.host_indices, op.device_indices
        else:
            host_indices, device_indices = self.move_indices(
                op.host_indices, op.device_indices
            )
        self.write_queue.clear()

        start_event = device_module.Event()
        finish_event = device_module.Event()

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, self.io_backend
            )
            if self.has_draft:
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )
            finish_event.record()
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the write stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)

        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        self.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        return device_indices

    def move_indices(self, host_indices: torch.Tensor, device_indices: torch.Tensor):
        # move indices to GPU if using kernels, to host if using direct indexing
        if self.io_backend == "kernel":
            if not host_indices.is_cuda:
                host_indices = host_indices.to(self.device, non_blocking=True)
            return host_indices, device_indices
        elif self.io_backend == "direct":
            if self.mem_pool_host.layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, idx = host_indices.sort()
                return host_indices, device_indices.index_select(0, idx)
            elif self.mem_pool_host.layout == "page_first_direct":
                return host_indices, device_indices.cpu()
            else:
                raise ValueError(
                    f"Unsupported layout {self.mem_pool_host.layout!r} for io backend 'direct'"
                )
        elif self.io_backend == "kernel_ascend":
            return host_indices, device_indices.cpu()
        else:
            raise ValueError(f"Unsupported io backend")

    def start_loading(self) -> int:
        if len(self.load_queue) == 0:
            return -1

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                )
                if self.has_draft and i < self.mem_pool_host_draft.layer_num:
                    self.mem_pool_host_draft.load_to_device_per_layer(
                        self.mem_pool_device_draft,
                        host_indices,
                        device_indices,
                        i,
                        self.io_backend,
                    )
                producer_event.complete(i)
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the load stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.load_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.load_stream)

        self.ack_load_queue.append(
            HiCacheAck(
                start_event=producer_event.start_event,
                finish_event=producer_event.finish_event,
                node_ids=op.node_ids,
            )
        )
        return producer_id

    def evict_device(self, device_indices: torch.Tensor) -> int:
        self.mem_pool_device_allocator.free(device_indices)
        return len(device_indices)

    def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")

        self.mem_pool_host.free(host_indices)
        return len(host_indices)

    def set_draft_kv_pool(self, draft_device_pool, draft_host_pool) -> None:
        """Register draft KV pools so L2/L3 ops piggyback draft transfers."""
        self.has_draft = True
        self.mem_pool_device_draft = draft_device_pool
        self.mem_pool_host_draft = draft_host_pool
        logger.info(
            "HiCache draft KV registered: %s (host %d slots)",
            type(draft_device_pool).__name__,
            draft_host_pool.size,
        )

        # If storage is already attached, wire up the draft I/O path now.
        # Otherwise this will be deferred until attach_storage_backend().
        self._maybe_register_draft_with_storage()

    def _maybe_register_draft_with_storage(self) -> None:
        """Pick the draft L3 IO implementation."""
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        if not self.has_draft or not self.enable_storage:
            return

        backend = self.storage_backend_type

        # Multi-pool zero-copy backends.
        if backend == "mooncake":
            if self.storage_config.should_split_heads:
                logger.warning(
                    "HiCache draft L3 disabled: should_split_heads not yet "
                    "supported on the mooncake v2 path."
                )
                return
            self.storage_backend.register_mem_host_pool_v2(
                self.mem_pool_host_draft, PoolName.DRAFT
            )
            self.draft_page_get_func = self._draft_page_get_v2
            self.draft_page_set_func = self._draft_page_set_v2
            return

        # TODO: support "hf3fs", "eic", "nixl", "simm"
        if backend in {"hf3fs", "eic", "nixl", "simm"}:
            logger.warning(
                "HiCache draft L3 disabled: backend %s does not yet support "
                "draft pool registration.",
                backend,
            )
            return

        # Generic backends.
        self.draft_page_get_func = self._draft_page_get_generic
        self.draft_page_set_func = self._draft_page_set_generic

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> PrefetchOperation:
        """
        Prefetch KV caches from storage backend to host memory.
        """
        operation = PrefetchOperation(
            request_id, host_indices, new_input_tokens, last_hash, prefix_keys
        )
        self.prefetch_queue.put(operation)
        return operation

    def terminate_prefetch(self, operation):
        operation.mark_terminate()
        return operation.completed_tokens, operation.hash_value

    def append_host_mem_release(self, host_indices: torch.Tensor):
        if host_indices.numel() == 0:
            return
        pages = host_indices.split(self.mem_pool_host.page_size)
        for page in pages:
            self.host_mem_release_queue.put(page)

    def _page_get_zero_copy(
        self, operation, hash_values, host_indices, extra_info=None
    ):
        results = self.storage_backend.batch_get_v1(
            hash_values, host_indices, extra_info
        )
        inc = 0
        for i in range(len(hash_values)):
            if not results[i]:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            inc += self.page_size
        operation.increment(inc)

    # todo: deprecate
    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):
        dummy_page_dst = [
            self.mem_pool_host.get_dummy_flat_data_page() for _ in hash_values
        ]
        page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)
        if page_data is None:
            return
        for i in range(len(hash_values)):
            if page_data[i] is None:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            # Must set the data before increasing the completed tokens.
            # Otherwise this page may be read before being set.
            self.mem_pool_host.set_from_flat_data_page(
                host_indices[i * self.page_size],
                page_data[i],
            )
            if not operation.increment(self.page_size):
                break  # Operation terminated by controller

    def _page_transfer(self, operation):
        # Transfer batch by batch
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]

            # Best-effort draft L3 read before publishing target completion.
            # Otherwise wait_complete can race and load back target KV before
            # draft KV reaches host memory.
            if self.has_draft:
                self._draft_page_get(batch_hashes, batch_host_indices)

            prev_completed_tokens = operation.completed_tokens
            # Get one batch token, and update the completed_tokens if succeed
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            self.page_get_func(operation, batch_hashes, batch_host_indices, extra_info)
            # Check termination
            if (
                operation.completed_tokens
                != prev_completed_tokens + len(batch_hashes) * self.page_size
            ):
                operation.mark_terminate()
                break  # Some operations fail or operation terminated by controller

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

    def prefetch_io_aux_func(self):
        """
        Auxiliary function conducting IO operations for prefetching.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.prefetch_buffer.get(block=True, timeout=1)
                if operation is None:
                    continue
                self._page_transfer(operation)
                # operation terminated by controller, release pre-allocated memory
                self.append_host_mem_release(
                    operation.host_indices[operation.completed_tokens :]
                )
            except Empty:
                continue

    def prefetch_rate_limited(self) -> bool:
        """
        Rate limit the prefetching operations to avoid overwhelming the storage backend.
        """
        # cancel prefetch if too much memory is occupied
        if self.prefetch_tokens_occupied >= self.prefetch_capacity_limit:
            return True
        # todo: more sophisticated rate limiting based on storage backend performance
        return False

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        last_hash = operation.last_hash
        tokens_to_fetch = operation.token_ids
        prefix_keys = operation.prefix_keys.copy() if operation.prefix_keys else None

        storage_query_count = 0
        hash_value = []

        for start in range(
            0, len(tokens_to_fetch), self.page_size * STORAGE_BATCH_SIZE
        ):
            end = min(start + self.page_size * STORAGE_BATCH_SIZE, len(tokens_to_fetch))
            batch_tokens = tokens_to_fetch[start:end]
            batch_hashes = []
            for i in range(0, len(batch_tokens), self.page_size):
                last_hash = self.get_hash_str(
                    batch_tokens[i : i + self.page_size], last_hash
                )
                batch_hashes.append(last_hash)
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            hit_page_num = self.storage_backend.batch_exists(batch_hashes, extra_info)
            hash_value.extend(batch_hashes[:hit_page_num])
            storage_query_count += hit_page_num * self.page_size
            if hit_page_num < len(batch_hashes):
                break
            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

        return hash_value, storage_query_count

    def prefetch_thread_func(self):
        """
        Manage prefetching operations from storage backend to host memory.
        """
        self.prefetch_buffer = Queue()
        self.prefetch_io_aux_thread = threading.Thread(
            target=self.prefetch_io_aux_func, daemon=True
        )
        self.prefetch_io_aux_thread.start()
        while (not self.storage_stop_event.is_set()) or not self.prefetch_queue.empty():
            try:
                operation = self.prefetch_queue.get(block=True, timeout=1)
                if operation is None:
                    continue
                hash_value, storage_hit_count = self._storage_hit_query(operation)
                storage_hit_count_tensor = torch.tensor(
                    storage_hit_count, dtype=torch.int
                )
                self._all_reduce_prefetch_groups(
                    storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
                )
                storage_hit_count = storage_hit_count_tensor.item()

                if storage_hit_count < self.prefetch_threshold:
                    # not to prefetch if not enough benefits
                    self.prefetch_revoke_queue.put(operation.request_id)
                    self.append_host_mem_release(operation.host_indices)
                    logger.debug(
                        f"Revoking prefetch for request {operation.request_id} due to insufficient hits ({storage_hit_count})."
                    )
                else:
                    operation.hash_value = hash_value[
                        : (storage_hit_count // self.page_size)
                    ]
                    # free the pre-allocated memory for pages that are not hit
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )
                    operation.host_indices = operation.host_indices[:storage_hit_count]
                    logger.debug(
                        f"Prefetching {len(operation.hash_value)} pages for request {operation.request_id}."
                    )
                    self.prefetch_buffer.put(operation)

            except Empty:
                continue

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> int:
        """
        Write KV caches from host memory to storage backend.
        """
        operation = StorageOperation(
            host_indices, token_ids, hash_value=hash_value, prefix_keys=prefix_keys
        )
        self.backup_queue.put(operation)
        return operation.id

    # todo: deprecate
    def _generic_page_set(self, hash_values, host_indices, extra_info=None) -> bool:
        data = [
            self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
            for i in range(len(hash_values))
        ]
        return self.storage_backend.batch_set(hash_values, data)

    def _page_set_zero_copy(self, hash_values, host_indices, extra_info=None) -> bool:
        return all(
            self.storage_backend.batch_set_v1(hash_values, host_indices, extra_info)
        )

    def _draft_page_set(self, hash_values, host_indices) -> None:
        """Best-effort write draft KV pages to L3 alongside the target backup."""
        if self.draft_page_set_func is None:
            return
        try:
            self.draft_page_set_func(hash_values, host_indices)
        except Exception:
            logger.debug(
                "Draft L3 write failed (best-effort), skipping.", exc_info=True
            )

    def _draft_page_get(self, hash_values, host_indices) -> None:
        """Best-effort read draft KV pages from L3 (mirrors `_draft_page_set`)."""
        if self.draft_page_get_func is None:
            return
        try:
            self.draft_page_get_func(hash_values, host_indices)
        except Exception:
            logger.debug("Draft L3 read failed (best-effort), skipping.", exc_info=True)

    def _draft_page_set_v2(self, hash_values, host_indices) -> None:
        self.storage_backend.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=host_indices,
                    keys=list(hash_values),
                )
            ]
        )

    def _draft_page_get_v2(self, hash_values, host_indices) -> None:
        self.storage_backend.batch_get_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=host_indices,
                    keys=list(hash_values),
                )
            ]
        )

    def _draft_page_set_generic(self, hash_values, host_indices) -> None:
        # `{hash}.draft` mirrors HiCacheStorage._get_component_key's
        # `{key}.{pool_name}` convention so target/draft pages never collide.
        draft_keys = [f"{h}.{PoolName.DRAFT}" for h in hash_values]
        draft_data = [
            self.mem_pool_host_draft.get_data_page(host_indices[i * self.page_size])
            for i in range(len(draft_keys))
        ]
        self.storage_backend.batch_set(draft_keys, draft_data)

    def _draft_page_get_generic(self, hash_values, host_indices) -> None:
        draft_keys = [f"{h}.{PoolName.DRAFT}" for h in hash_values]
        draft_dummy = [
            self.mem_pool_host_draft.get_dummy_flat_data_page() for _ in draft_keys
        ]
        draft_pages = self.storage_backend.batch_get(draft_keys, draft_dummy)
        if draft_pages is None:
            return
        for i, p in enumerate(draft_pages):
            if p is not None:
                self.mem_pool_host_draft.set_from_flat_data_page(
                    host_indices[i * self.page_size], p
                )

    # Backup batch by batch
    def _page_backup(self, operation):
        # Backup batch by batch
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]
            # Set one batch token, and record if success.
            # todo: allow partial success
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            success = self.page_set_func(batch_hashes, batch_host_indices, extra_info)
            if not success:
                logger.warning(
                    f"Write page to storage: {len(batch_hashes)} pages failed."
                )
                break

            # Best-effort draft L3 write alongside target.
            if self.has_draft:
                self._draft_page_set(batch_hashes, batch_host_indices)

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes
            operation.completed_tokens += self.page_size * len(batch_hashes)

    def backup_thread_func(self):
        """
        Manage backup operations from host memory to storage backend.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.backup_queue.get(block=True, timeout=1)
                if operation is None:
                    continue

                if not self.backup_skip:
                    self._page_backup(operation)
                self.ack_backup_queue.put(operation)

            except Empty:
                continue
