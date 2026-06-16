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

    ──────── node_id 完整链路: DMA 完成后的回调凭证 ────────

    node_id 不是给 Host/GPU 端用的, 而是 DMA 异步完成后,
    让 HiRadixCache 能找到对应 radix tree 节点做后续处理的"收据编号"。

    ┌─────────────────────────────────────────────────────────────────┐
    │  Write 方向 (GPU → Host)                                        │
    │                                                                 │
    │  ① HiRadixCache.evict()                                         │
    │     → controller.write(node_id=node.id)                         │
    │     → CacheOperation.node_ids = [node.id]                       │
    │                                                                 │
    │  ② start_writing() + merge_ops()                                │
    │     → 多个 write 合并: node_ids = [id1, id2, ...]               │
    │     → 一次性 DMA 减少 kernel launch                              │
    │                                                                 │
    │  ③ DMA 完成 → HiCacheAck(node_ids=[id1, id2, ...])             │
    │     → 入 ack_write_queue                                        │
    │                                                                 │
    │  ④ HiRadixCache.writing_check() 消费 ack                        │
    │     for ack_id in ack.node_ids:                                 │
    │       _finish_write_through_ack(ack_id)                         │
    │         ├─ ongoing_write_through.pop(ack_id) → 找到 TreeNode    │
    │         ├─ write_through_pending_id = None   → 清除 DMA 等待    │
    │         ├─ _record_store_event(CPU)          → 记录已到 Host    │
    │         ├─ write_backup_storage()            → 触发 L3 写入     │
    │         └─ dec_lock_ref()                    → 释放 evict 锁    │
    │                                                                 │
    ├─────────────────────────────────────────────────────────────────┤
    │  Load 方向 (Host → GPU)                                          │
    │                                                                 │
    │  ① HiRadixCache._try_load_from_host()                           │
    │     → controller.load(node_id=last_hit_node.id)                 │
    │     → CacheOperation.node_ids = [last_hit_node.id]              │
    │                                                                 │
    │  ② start_loading() + merge_ops()                                │
    │     → 合并后 DMA                                                │
    │                                                                 │
    │  ③ DMA 完成 → HiCacheAck(node_ids=[...])                       │
    │                                                                 │
    │  ④ HiRadixCache.loading_check() 消费 ack                        │
    │     for ack_id in ack.node_ids:                                 │
    │       node = ongoing_load_back.pop(ack_id)                      │
    │       dec_lock_ref(node)  → 释放 evict 保护锁                   │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
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
    """分层 KV cache 控制器，管理 GPU↔Host↔Storage 三级存储之间的数据搬运。

        ╔════════════════════════════════════════════════════════════════════════════════════════════════╗
        ║  整体架构：三级存储层次                                                                            ║
        ║                                                                                                ║
        ║   GPU (device)  ←──write/load──→  Host (CPU DRAM)  ←──backup/prefetch──→  Storage (磁盘/远端)    ║
        ║   mem_pool_device                mem_pool_host                      storage_backend            ║
        ╚════════════════════════════════════════════════════════════════════════════════════════════════╝

        ─── 宏观：上层如何使用 HiCacheController ───

        HiCacheController 的唯一上层是 HiRadixCache（及 HybridCacheController 子类），
        调度器/worker 不直接操作 HiCacheController，而是通过 HiRadixCache 间接使用。

        上层调用按场景分为 5 组：

        【场景 1：GPU→Host 写入（evict 时备份 KV）】
          HiRadixCache.write_backup(node)
            → controller.write(device_indices=node.value, node_id=node.id)
              → 分配 host slot → 入 write_queue → start_writing() 合并+DMA
            → controller.ack_write_queue            ← 写入完成后检查 ack

        【场景 2：Host→GPU 加载（prefill 时恢复 KV）】
          HiRadixCache.load_back(host_indices, ...)
            → controller.load(host_indices=...)      ← 分配 device slot + 入 load_queue
            → HiRadixCache.ready_to_load_host_cache()
              → controller.start_loading()           ← 合并+逐层DMA → 返回 consumer_index
            → Scheduler: batch.hicache_consumer_index = consumer_index
            → TpWorker: set_hicache_consumer(consumer_index)
              → layer_done_counter.set_consumer(...)
            → ModelRunner: 每层前向时 wait_until(layer_id)

        【场景 3：Host→Storage 备份（三级持久化）】
          HiRadixCache.write_backup_storage(node, ...)
            → controller.write_storage(host_indices, token_ids, hash_value, ...)
              → 创建 StorageOperation → 入 backup_queue
            → backup_thread → _page_backup() → storage_backend.batch_set
            → controller.ack_backup_queue            ← 备份完成后检查 ack

        【场景 4：Storage→Host 预取（请求到达时提前拉取）】
          HiRadixCache.prefetch_from_storage(request_id, ...)
            → controller.prefetch(request_id, ...)
              → 创建 PrefetchOperation → 入 prefetch_queue
            → prefetch_thread: _storage_hit_query() → 判断是否值得预取
              → 值得：放入 prefetch_buffer
              → 不值得：revoke + 释放 host 内存
            → prefetch_io_aux_thread: _page_transfer() → storage_backend.batch_get
            → HiRadixCache.can_terminate_prefetch() / terminate_prefetch()
              → controller.terminate_prefetch(operation) → mark_terminate()

        【场景 5：设备/主机内存释放】
          controller.evict_device(device_indices)   ← 释放 GPU KV slot
          controller.evict_host(host_indices)        ← 释放 Host KV slot

        ─── 微观：内部实现拆解 ───

        【L2 层（GPU↔Host）核心数据流】
          write_queue / load_queue  ─→  merge_ops()  ─→  write_stream/load_stream DMA  ─→  ack_write_queue / ack_load_queue
          (List[CacheOperation])       (合并为一次大DMA)     (异步CUDA stream)              (List[HiCacheAck])

          - write(): 分配 host slot → CacheOperation 入 write_queue → start_writing() 立即执行
          - load():  分配 device slot → CacheOperation 入 load_queue（不立即执行，等 start_loading）
          - start_loading(): merge load_queue → 在 load_stream 逐层拷贝 + LayerDoneCounter 记录 event
          - writing_check() / loading_check(): 遍历 ack 队列，query/synchronize event → 回调 HiRadixCache

        【L3 层（Host↔Storage）核心数据流】
          prefetch_queue  ─→  prefetch_thread  ─→  prefetch_buffer  ─→  prefetch_io_aux_thread  ─→  storage read
          backup_queue    ─→  backup_thread     ─→  storage write

          三个后台线程（daemon）：
            - prefetch_thread:      查询 storage 命中 → 决定是否预取 → 放入 prefetch_buffer
            - prefetch_io_aux_thread: 从 prefetch_buffer 取出 → 逐页 DMA 读回 host
            - backup_thread:         从 backup_queue 取出 → 逐页写入 storage

        【写策略】
          write_through:          每次命中立刻写 Host（hit_count >= 1 触发 write_backup）
          write_through_selective: 命中 2 次才写（hit_count >= write_through_threshold）
          write_back:             仅在 evict 时才写（延迟写入，减少不必要传输）

        【IO 后端】
          kernel:  使用自定义 CUDA kernel 做 DMA（indices 移到 GPU）
          direct:  使用 PyTorch 原生索引（indices 移到 CPU）
          kernel_ascend: NPU 专用

        【线程安全】
          - stop_event: 控制 write_buffer/load_buffer 的停止（当前未启用异步路径）
          - storage_stop_event: 独立控制 storage 线程的启停，支持运行时 attach/detach
          - L2 层（write/load/ack 队列）由调度器主线程单线程访问，无需锁
          - L3 层（prefetch/backup 队列）由 Queue 自带锁保证线程安全
        """
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
        """
        HiCacheController 初始化 —— 构建三层 KV cache 搬运器的全部运行时状态。

        ╔══════════════════════════════════════════════════════════════════════╗
        ║  成员变量全景图 (按功能分组)                                              ║
        ╠══════════════════════════════════════════════════════════════════════╣
        ║                                                                      ║
        ║  ┌─ ① 分布式通信组 (TP/CP/PP) ────────────────────────────────────┐    ║
        ║  │  tp_group          TP 通信组 (必须)                            │    ║
        ║  │  attn_cp_group     Context Parallel 通信组                    │    ║
        ║  │  attn_tp_group     Attention TP 通信组                        │    ║
        ║  │  pp_group          Pipeline Parallel 通信组                   │    ║
        ║  │  prefetch_sync_groups  基于 tp/cp 组派生的 gloo all_reduce 组   │    ║
        ║  └──────────────────────────────────────────────────────────────┘    ║
        ║                                                                      ║
        ║  ┌─ ② 内存池 (L1 GPU / L2 Host / L3 Storage) ────────────────────┐    ║
        ║  │  mem_pool_device_allocator  GPU KV pool 分配器                 │    ║
        ║  │  mem_pool_device            GPU KV pool (实际 tensor)         │    ║
        ║  │  mem_pool_host              Host KV pool (CPU pinned)        │    ║
        ║  │  storage_backend            L3 存储后端 (disk/remote)          │    ║
        ║  └──────────────────────────────────────────────────────────────┘    ║
        ║                                                                      ║
        ║  ┌─ ③ 策略与配置 ─────────────────────────────────────────────────┐    ║
        ║  │  write_policy       写回策略 (write_through/selective/back)    │    ║
        ║  │  page_size          每个 page 的 token 数 (通常 16)             │    ║
        ║  │  io_backend         L1↔L2 IO 后端 (kernel/direct)             │    ║
        ║  └──────────────────────────────────────────────────────────────┘    ║
        ║                                                                      ║
        ║  ┌─ ④ GPU DMA 引擎 ─────────────────────────────────────────────┐    ║
        ║  │  write_stream       GPU→Host DMA 流                          │    ║
        ║  │  load_stream        Host→GPU DMA 流                          │    ║
        ║  │  write_queue/load_queue  待执行的 CacheOperation 列表          │    ║
        ║  │  ack_write_queue/ack_load_queue  DMA 完成 ack 列表            │    ║
        ║  │  layer_done_counter   load 时逐层同步计数器 (三缓冲)             │    ║
        ║  │  write_buffer/load_buffer  预留的异步双缓冲 (当前未启用)          │    ║
        ║  └──────────────────────────────────────────────────────────────┘    ║
        ║                                                                      ║
        ║  ┌─ ⑤ L3 后台线程 ───────────────────────────────────────────────┐    ║
        ║  │  storage_stop_event  L3 线程停止信号 (独立于 stop_event)         │    ║
        ║  │  stop_event          L1↔L2 停止信号                            │    ║
        ║  │  page_get_func / page_set_func  L3 页读写函数指针               │    ║
        ║  └──────────────────────────────────────────────────────────────┘    ║
        ║                                                                      ║
        ║  ┌─ ⑥ Draft KV (投机解码, best-effort) ──────────────────────────┐    ║
        ║  │  has_draft           是否启用 draft pool                       │    ║
        ║  │  mem_pool_device_draft  GPU draft KV pool                    │    ║
        ║  │  mem_pool_host_draft    Host draft KV pool                   │    ║
        ║  │  draft_page_get/set_func  L3 draft 读写函数指针                │    ║
        ║  └──────────────────────────────────────────────────────────────┘    ║
        ║                                                                      ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """

        # ────────────── ① 分布式通信组 ──────────────
        # tp_group: Tensor Parallel 通信组, 用于 all_reduce(MIN) 同步 prefetch 命中长度
        self.tp_group = tp_group
        # attn_cp_group: Context Parallel 通信组, 用于长上下文场景下的 TP 同步
        self.attn_cp_group = attn_cp_group
        # attn_tp_group: Attention TP 通信组, 某些模型架构中 attention 层独立做 TP
        self.attn_tp_group = attn_tp_group
        # pp_group: Pipeline Parallel 通信组 (当前未使用, 预留)
        self.pp_group = pp_group
        # prefetch_sync_groups: 由 tp_group/attn_cp_group/attn_tp_group 派生的
        #   gloo backend ProcessGroup 列表, 用于 CPU tensor 上做 all_reduce(MIN)
        #   保证 TP 多卡 prefetch 命中长度一致 (详见 prefetch_thread_func 注释)
        self.prefetch_sync_groups: List[torch.distributed.ProcessGroup] = []

        # ────────────── ② 内存池 ──────────────
        # GPU KV pool 分配器: 管理 GPU 端 slot 分配/释放, start_writing 从此取目标地址
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        # GPU KV pool 实际 tensor: 存放 L1 层 KV cache 数据
        #   如果是 HybridLinearKVPool (SWA+full 混合), 取其 full_kv_pool
        mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        if isinstance(mem_pool_device, HybridLinearKVPool):
            mem_pool_device = mem_pool_device.full_kv_pool
        self.mem_pool_device = mem_pool_device
        # Host KV pool: CPU pinned memory, 存放 L2 层 KV cache 数据
        self.mem_pool_host = mem_pool_host

        # ────────────── ③ 策略与配置 ──────────────
        # write_policy: GPU→Host 写回策略
        #   write_through           → 每个 write 操作都同时触发 L3 备份
        #   write_through_selective → 仅对首次出现的 prefix 触发 L3 备份
        #   write_back              → 仅写回 Host, 不主动备份 L3
        self.write_policy = write_policy
        # page_size: KV cache page 粒度 (通常 16 tokens)
        #   影响: hash 计算粒度、L3 存储单元、host_indices 按 page_size 分组
        self.page_size = page_size
        # io_backend: L1↔L2 DMA 后端
        #   "kernel"  → 自定义 CUDA kernel (indices 在 GPU 上, 更快)
        #   "direct"  → PyTorch 原生 copy (indices 在 CPU 上, 通用)
        #   "kernel_ascend" → NPU 后端
        self.io_backend = io_backend

        # ────────────── L3 存储相关 (初始状态: 未启用) ──────────────
        # enable_storage: L3 存储是否已启用 (通过 attach_storage_backend 开启)
        self.enable_storage = False
        # storage_backend: L3 存储后端实例 (mooncake/simm/nixl/hf3fs 等)
        self.storage_backend = None
        # storage_backend_type: L3 存储后端类型字符串
        self.storage_backend_type = None
        # enable_storage_metrics: 是否采集 L3 存储指标 (延迟、命中率等)
        self.enable_storage_metrics = enable_storage_metrics

        # ────────────── ⑥ Draft KV (投机解码) ──────────────
        # Draft KV pool 搭载在 target 的 L2/L3 操作上 best-effort 读写,
        # 不单独占用 IO 线程, 失败也不影响 target 正确性
        self.has_draft = False
        self.mem_pool_device_draft = None
        self.mem_pool_host_draft = None
        self.draft_page_get_func = None
        self.draft_page_set_func = None

        # ────────────── ⑤ L3 页 IO 函数指针 ──────────────
        # page_set_func: L3 写入函数 (host_indices → storage_backend)
        # page_get_func: L3 读取函数 (storage_backend → host_indices)
        # 默认使用 _generic_page_set/get (逐页拷贝), attach 时可能替换为零拷贝版本
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set

        # ────────────── ⑤ 停止信号 ──────────────
        # storage_stop_event: 仅控制 L3 后台线程 (prefetch_thread / backup_thread)
        #   独立于 stop_event, 支持 runtime attach/detach L3 而不影响 L1↔L2 传输
        self.storage_stop_event = threading.Event()

        # ────────────── ④ GPU DMA 引擎 ──────────────
        # device: "cuda" / "npu" 等, 决定 CUDA/MemoryPool backend
        self.device = self.mem_pool_device.device
        # layer_num: 模型层数, 决定 layer_done_counter 的大小
        #   write 时一次性搬全部层, load 时逐层搬+逐层同步
        self.layer_num = self.mem_pool_device.layer_num
        # layer_done_counter: load 时三缓冲层同步计数器
        #   compute_stream wait 每一层 load 完成, 而非等全部层搬完
        #   详见 start_loading() 注释中的时序图
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        # write_queue / load_queue: 待执行的 GPU↔Host 传输操作列表
        #   由 start_writing/start_loading 填入, 由 write()/load() 消费
        #   注意: 不是线程安全队列! 在 scheduler 线程中单线程访问
        self.load_queue: List[CacheOperation] = []
        self.write_queue: List[CacheOperation] = []
        # ack_write_queue / ack_load_queue: DMA 完成 ack 列表
        #   由 write()/load() 填入, 由上层 evict 或 drain_queues 消费
        self.ack_load_queue: List[HiCacheAck] = []
        self.ack_write_queue: List[HiCacheAck] = []

        # stop_event: L1↔L2 传输停止信号 (不同于 storage_stop_event)
        #   设置后 write_buffer/load_buffer 的 put/get 不再阻塞
        self.stop_event = threading.Event()
        # write_buffer / load_buffer: TransferBuffer (有界阻塞队列)
        #   当前仅 clear() 在 reset() 中被调用, put/get 未启用
        #   预留用于未来的异步双缓冲流水线
        #   load_buffer 的 buffer_count=10 > write_buffer 的 1, 因为
        #   load 是逐层传输, 需要更多缓冲槽位
        self.write_buffer = TransferBuffer(self.stop_event)
        self.load_buffer = TransferBuffer(self.stop_event, buffer_count=10)

        # write_stream: GPU→Host DMA 专用 CUDA 流
        #   start_writing() 中: wait(start_event) → DMA → record(finish_event)
        # load_stream:  Host→GPU DMA 专用 CUDA 流
        #   start_loading() 中: 逐层 DMA + layer_done_counter 通知 compute_stream
        self.write_stream = device_module.Stream()
        self.load_stream = device_module.Stream()

        # 如果启动时指定了 storage_backend, 视为隐式 attach, 走同一套生命周期
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
        """Back up KV caches from device memory to host memory.

        GPU (device_indices) ──DMA──→ Host (host_indices)
                                       ↑ alloc

        调用后立即触发 start_writing()，将本操作与队列中其他操作合并后一次性 DMA。
        返回 host_indices 供上层（HiRadixCache）记录到 node.host_value。

        ──────── node_id 的作用: DMA 完成后的回调凭证 ────────

        node_id 不是给 Host 端用的，而是 DMA 异步完成后的"收据编号"。
        GPU→Host DMA 是异步的，发起 write 时 node 正在被搬运，
        搬完才能做后续动作 (L3 备份、释放 host 锁等)。

        完整链路:
          ① write(node_id=node.id)
             → CacheOperation.node_ids = [node_id]
          ② merge_ops() 合并多个 write
             → node_ids = [id1, id2, ...]  (一次 DMA 搬多个 node)
          ③ DMA 完成 → HiCacheAck(node_ids=[id1, id2, ...])
          ④ HiRadixCache.writing_check() 遍历 ack.node_ids:
             for ack_id in ack.node_ids:
                 _finish_write_through_ack(ack_id)
                   ├─ ongoing_write_through.pop(ack_id)  → 找到 TreeNode
                   ├─ write_through_pending_id = None    → 清除 DMA 等待标记
                   ├─ _record_store_event(CPU)           → 记录 KV 已到 Host
                   ├─ write_backup_storage()             → 触发 L3 写入
                   └─ dec_lock_ref()                     → 释放 evict 保护锁

        为什么是 node_ids (复数)?
          merge_ops() 会把多个小 write 合并成一次大 DMA 以减少 kernel launch 开销,
          合并后 node_ids = [id1, id2, id3], DMA 一次性搬完, ack 时逐个回调。
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
        """将 write_queue 中积攒的所有 GPU→Host 写入操作合并后一次性提交 DMA。

        ┌─────────────── 数据流全景 ──────────────────┐
        │                                           │
        │  GPU (device)  ──── DMA ────→  Host (CPU) │
        │  mem_pool_device           mem_pool_host  │
        │       │                        │          │
        │   device_indices          host_indices    │
        └───────────────────────────────────────────┘

        ┌─────────────── 执行流程 ────────────────────┐
        │                                           │
        │  write() × N                              │
        │    │  每次: alloc host slot                │
        │    │  → CacheOperation 入 write_queue      │
        │    ▼                                      │
        │  merge_ops()                              │
        │    │  cat 所有 host/device indices         │
        │    │  → 减少多次小 DMA 为一次大 DMA           │
        │    ▼                                      │
        │  move_indices()                           │
        │    │  kernel → indices 放 GPU             │
        │    │  direct → indices 放 CPU             │
        │    ▼                                      │
        │  write_stream 上异步执行:                   │
        │    ├─ wait(start_event)  ← 等源数据就绪      │
        │    ├─ backup_from_device_all_layer()      │
        │    │    全层一次性拷贝 (L0,L1,...,Ln)        │
        │    ├─ [draft: 同样拷贝 draft KV]            │
        │    └─ finish_event.record()               │
        │    ▼                                      │
        │  HiCacheAck 入 ack_write_queue            │
        │    │                                      │
        │    ▼                                      │
        │  HiRadixCache.writing_check()             │
        │    finish_event.query()/synchronize()     │
        │    → 确认 DMA 完成 → 回调树节点               │
        └───────────────────────────────────────────┘

        ┌─────────── CUDA 流同步时序 ─────────────────────────────┐
        │                                                       │
        │  compute stream:  ──● start_event                     │
        │                            │                          │
        │  write stream:    ──wait(start)──DMA──● finish_event  │
        │                                                       │
        │  ● = event.record()                                   │
        │  wait = stream.wait_event()                           │
        └───────────────────────────────────────────────────────┘

        与 start_loading() 的关键区别：
          ┌──────────────┬──────────────────┬──────────────────────┐
          │              │ start_writing    │ start_loading        │
          ├──────────────┼──────────────────┼──────────────────────┤
          │ 拷贝方式       │ 全层一次性         │ 逐层拷贝              │
          │ 逐层同步       │ 不需要            │ LayerDoneCounter     │
          │ 原因          │ 无计算流在等       │ prefill 计算流在等 KV  │
          │ ack 消费者    │ writing_check()  │ loading_check()      │
          └──────────────┴──────────────────┴──────────────────────┘

        异步安全性：
          - start_event: 在 compute stream 上 record → write_stream.wait → 确保源数据已就绪
          - record_stream(): 防止 CUDA caching allocator 在异步 DMA 完成前回收 indices 张量
          - finish_event: HiRadixCache.writing_check() 轮询或 synchronize，
            确认 DMA 完成后才执行后续逻辑（如 dec_lock_ref、write_backup_storage）
        """
        if len(self.write_queue) == 0:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        # indices 搬运策略: 两条路径, 区别在于 host_indices 和 device_indices 放在 GPU 还是 CPU
        #
        # ┌──────────────────────────────────────────────────────────────────────────┐
        # │  路径选择                                                                │
        # │                                                                          │
        # │  条件: io_backend=="kernel" AND layout=="page_first"                      │
        # │  ┌────────────────────────────────────────────────────────────────────┐  │
        # │  │ 路径 A: 直接用原始 indices (不走 move_indices)                      │  │
        # │  │                                                                    │  │
        # │  │   host_indices   → 保持在 CPU  (原始分配就在 CPU)                  │  │
        # │  │   device_indices → 保持在 GPU  (原始分配就在 GPU)                  │  │
        # │  │                                                                    │  │
        # │  │   原因: page_first 的 kernel 使用 staging buffer 方式:             │  │
        # │  │     1) GPU kernel 用 device_indices 从 GPU 读取源数据              │  │
        # │  │     2) 写入 GPU 上的 staging buffer (不直接写 Host)                │  │
        # │  │     3) staging buffer → Host 的最终 scatter 用 host_indices (CPU)  │  │
        # │  │     所以 host_indices 不需要搬到 GPU                                │  │
        # │  └────────────────────────────────────────────────────────────────────┘  │
        # │                                                                          │
        # │  其他所有情况 (kernel+layer_first, direct, kernel_ascend, ...)           │
        # │  ┌────────────────────────────────────────────────────────────────────┐  │
        # │  │ 路径 B: 走 move_indices()                                         │  │
        # │  │                                                                    │  │
        # │  │   kernel + layer_first:                                            │  │
        # │  │     host_indices   → CPU 搬到 GPU  (kernel 从 GPU 读目标地址)      │  │
        # │  │     device_indices → 已在 GPU, 不动                                │  │
        # │  │                                                                    │  │
        # │  │   direct + layer_first:                                            │  │
        # │  │     host_indices   → CPU 上 sort 排序 (内存连续性优化)             │  │
        # │  │     device_indices → GPU 搬到 CPU, 按相同顺序重排                  │  │
        # │  │                                                                    │  │
        # │  │   direct + page_first_direct:                                      │  │
        # │  │     host_indices   → CPU, 不动                                     │  │
        # │  │     device_indices → GPU 搬到 CPU                                  │  │
        # │  │                                                                    │  │
        # │  │   kernel_ascend:                                                   │  │
        # │  │     host_indices   → 不动                                          │  │
        # │  │     device_indices → GPU 搬到 CPU                                  │  │
        # │  └────────────────────────────────────────────────────────────────────┘  │
        # └──────────────────────────────────────────────────────────────────────────┘
        if self.io_backend == "kernel" and self.mem_pool_host.layout == "page_first":
            host_indices, device_indices = op.host_indices, op.device_indices
        else:
            host_indices, device_indices = self.move_indices(
                op.host_indices, op.device_indices
            )
        self.write_queue.clear()

        start_event = device_module.Event()
        finish_event = device_module.Event()

        # 在默认流（compute stream）上记录起始点，确保源数据已就绪
        start_event.record()
        with device_module.stream(self.write_stream):
            # write_stream 等待 compute stream 上的数据产出完成
            start_event.wait(self.write_stream)
            # 全层一次性 GPU→Host DMA 拷贝
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, self.io_backend
            )
            # 如果有 draft KV pool，同样拷贝（best-effort 搭便车）
            if self.has_draft:
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )
            # 在 write_stream 上记录完成 event
            finish_event.record()
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the write stream is executing.
            # 防止 CUDA caching allocator 在异步 DMA 未完成时回收这些张量的 GPU 内存
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)

        # HiCacheAck 入 ack_write_queue, 由 HiRadixCache.writing_check() 消费:
        #   for _, finish_event, ack_list in ack_write_queue:
        #       finish_event.synchronize()           # 等 DMA 完成
        #       for ack_id in ack_list:              # ack_list = node_ids
        #           _finish_write_through_ack(ack_id)
        #             ├─ ongoing_write_through.pop(ack_id) → 找到 TreeNode
        #             ├─ write_backup_storage()            → 触发 L3 写入
        #             └─ dec_lock_ref()                    → 释放 evict 锁
        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """Load KV caches from host memory to device memory.

        Host (host_indices) ──DMA──→ GPU (device_indices)
                                       ↑ alloc

        仅入 load_queue，不立即执行 DMA。等 start_loading() 被调度器触发时
        才合并所有 load 操作 + 逐层拷贝 + LayerDoneCounter 同步。
        返回 device_indices 供上层分配 GPU slot。

        node_id 作用与 write() 相同: DMA 完成后的回调凭证。
        loading_check() 拿 node_id 去 ongoing_load_back pop 出节点做 dec_lock_ref。
        """
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        self.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        return device_indices

    def move_indices(self, host_indices: torch.Tensor, device_indices: torch.Tensor):
        """根据 io_backend 和 host layout 将 indices 搬到正确的设备 (GPU/CPU)。

        ┌────────────────────────────────────────────────────────────────────┐
        │  io_backend + layout         │ host_indices     │ device_indices  │
        │──────────────────────────────│──────────────────│─────────────────│
        │  kernel (layer_first)        │ CPU → GPU        │ 已在 GPU, 不动  │
        │  kernel (page_first)         │ 不走此函数, 直接用原始 CPU indices │
        │  direct + layer_first        │ sort 排序(CPU)   │ CPU + 重排对齐  │
        │  direct + page_first_direct  │ 不动 (CPU)       │ → CPU           │
        │  kernel_ascend               │ 不动             │ → CPU           │
        └────────────────────────────────────────────────────────────────────┘

        layer_first + direct 需要 sort 的原因:
          layer_first layout 下 Host 内存按 [layer][token] 排列,
          为保证 DMA 写入时地址连续 (减少 page fault), 需要先 sort host_indices,
          然后 device_indices 也要按相同顺序重排, 保证 src/dst 索引一一对应。
        """
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
        """将 load_queue 中积攒的所有 Host→GPU 加载操作合并后提交逐层 DMA，返回 consumer_index。

        ┌─────────────── 数据流全景 ──────────────────┐
        │                                           │
        │  Host (CPU)  ──── DMA ────→  GPU (device) │
        │  mem_pool_host             mem_pool_device│
        │       │                        │          │
        │   host_indices            device_indices  │
        └───────────────────────────────────────────┘

        ┌─────────────── 执行流程 ────────────────────┐
        │                                           │
        │  load() × N                               │
        │    │  每次: alloc device slot              │
        │    │  → CacheOperation 入 load_queue      │
        │    ▼                                      │
        │  update_producer()                        │
        │    │  获取三缓冲中的下一个 event slot         │
        │    ▼                                      │
        │  merge_ops()                              │
        │    │  cat 所有 host/device indices         │
        │    ▼                                      │
        │  move_indices()                           │
        │    │  kernel → indices 放 GPU             │
        │    │  direct → indices 放 CPU             │
        │    ▼                                      │
        │  load_stream 上逐层拷贝:                    │
        │    for i in range(layer_num):             │
        │      load_to_device_per_layer(i)          │
        │      producer_event.complete(i) ← 记录     │
        │    ▼                                      │
        │  HiCacheAck 入 ack_load_queue             │
        │    │                                      │
        │    └→ 返回 producer_id                     │
        │       ↓                                   │
        │  Scheduler: batch.hicache_consumer_index  │
        │       ↓                                   │
        │  TpWorker: set_consumer(producer_id)      │
        │       ↓                                   │
        │  ModelRunner: wait_until(layer_id) 逐层等  │
        └───────────────────────────────────────────┘

        ┌──────── 逐层 overlap 时序（核心价值）─────────────────┐
        │                                                   │
        │  load_stream:   ──L0──L1──L2──L3──L4──●           │
        │                        │   │   │   │              │
        │  compute stream: ──────w───w───w───w──→           │
        │                       ↓   ↓   ↓   ↓               │
        │                  attention 计算开始                 │
        │                                                   │
        │  L_i = load_to_device_per_layer(i) + complete(i)  │
        │  w = wait_until(i) → compute stream 等该层就绪      │
        │  ● = finish_event（最后一层完成 = 整体完成）           │
        │                                                   │
        │  效果：计算不必等全部层拷贝完，每层就绪即可开始            │
        └───────────────────────────────────────────────────┘

        ┌────── 三缓冲（LayerDoneCounter）────────────┐
        │                                           │
        │  slot A: consumer 正在等待（当前 batch）     │
        │  slot B: 正在 load（下一 batch 预取）        │
        │  slot C: 上一 batch 已完成，可安全复用        │
        │                                           │
        │  update_producer() 断言 slot C 的          │
        │  finish_event.query() == True             │
        └───────────────────────────────────────────┘

        Returns:
            producer_id: 传给 set_consumer() 绑定 consumer 等待的 event slot，
                         -1 表示 load_queue 为空无需加载
        """
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

        # HiCacheAck 入 ack_load_queue, 由 HiRadixCache.loading_check() 消费:
        #   for _, finish_event, ack_list in ack_load_queue:
        #       finish_event.synchronize()
        #       for ack_id in ack_list:              # ack_list = node_ids
        #           node = ongoing_load_back.pop(ack_id)
        #           dec_lock_ref(node)                → 释放 evict 保护锁
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
        """
        查询 L3 存储中实际命中了多少 token 的 KV cache，返回 (hash列表, 命中token数)。

        ┌──────────────────────────────────────────────────────────────┐
        │  逐 page 查询流程                                           │
        │                                                              │
        │  token_ids: [t0, t1, ..., tN]                               │
        │       │                                                      │
        │       ▼                                                      │
        │  ┌──────────────────────────────────────┐                   │
        │  │ 按 STORAGE_BATCH_SIZE 分批:           │                   │
        │  │   batch0: page0..pageK               │                   │
        │  │   batch1: pageK+1..page2K            │                   │
        │  │   ...                                │                   │
        │  └──────────────────────────────────────┘                   │
        │       │  每个 batch 内:                                       │
        │       │  1) 逐 page 计算 hash (滚动哈希, last_hash 链式)     │
        │       │  2) batch_exists(batch_hashes) → hit_page_num       │
        │       │  3) hit_page_num < len(batch_hashes)?               │
        │       │     YES → 命中到此为止, break (prefix 必须连续)       │
        │       │     NO  → 继续下一个 batch                            │
        │       ▼                                                      │
        │  返回: (hash_value[:hit_pages], hit_token_count)             │
        │                                                              │
        │  关键: L3 的 KV cache 以 prefix tree 组织，                 │
        │       命中必须是连续前缀，一旦中间断开就停止                  │
        └──────────────────────────────────────────────────────────────┘
        """
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
        L3 预取调度线程 —— 从 prefetch_queue 消费请求，决策是否真正执行 IO。

        ╔══════════════════════════════════════════════════════════════════════╗
        ║  整体流水线架构 (三线程两队列)                                      ║
        ╠══════════════════════════════════════════════════════════════════════╣
        ║                                                                      ║
        ║  scheduler                                                          ║
        ║    │  prefetch(req_id, host_indices, tokens, ...)                   ║
        ║    ▼                                                                ║
        ║  ┌─────────────────┐                                               ║
        ║  │ prefetch_queue  │  ← PrefetchOperation 入队                     ║
        ║  └────────┬────────┘                                               ║
        ║           │  get()                                                  ║
        ║           ▼                                                        ║
        ║  ┌─────────────────────────────────────────────────┐                ║
        ║  │  prefetch_thread_func  (本函数 — 调度决策层)      │                ║
        ║  │                                                   │                ║
        ║  │  1) _storage_hit_query()  → 查 L3 实际命中长度    │                ║
        ║  │  2) all_reduce(MIN)      → TP 多卡取最小命中     │                ║
        ║  │  3) 判定: hit >= threshold ?                      │                ║
        ║  │     ├─ YES → 截断 host_indices, 放入 prefetch_buffer              ║
        ║  │     └─ NO  → 撤销: 释放 host_mem + 通知 revoke_queue             ║
        ║  └──────────────────────┬──────────────────────────┘                ║
        ║                          │  put()                                  ║
        ║                          ▼                                         ║
        ║  ┌─────────────────┐                                               ║
        ║  │ prefetch_buffer │  ← 仅存放通过决策的操作                         ║
        ║  └────────┬────────┘                                               ║
        ║           │  get()                                                  ║
        ║           ▼                                                        ║
        ║  ┌─────────────────────────────────────────┐                      ║
        ║  │  prefetch_io_aux_func (IO 执行线程)       │                      ║
        ║  │    → _page_transfer() 逐 batch 读写 L3    │                      ║
        ║  │    → 支持 mark_terminate() 提前终止       │                      ║
        ║  └─────────────────────────────────────────┘                      ║
        ║                                                                      ║
        ╚══════════════════════════════════════════════════════════════════════╝

        ──────── 关键机制: all_reduce(MIN) 保证 TP 多卡一致性 ────────

        TP 多卡场景下，每个 rank 独立查询本地 L3，命中长度可能不同:

          rank 0:  L3 命中 1024 tokens (64 pages)
          rank 1:  L3 命中  960 tokens (60 pages)   ← 丢失了 4 pages
          rank 2:  L3 命中 1024 tokens (64 pages)
          rank 3:  L3 命中 1024 tokens (64 pages)

        如果各 rank 各自取各自的命中长度:
          → rank0/2/3 分配了 64 pages 的 host_indices
          → rank1 只分配了 60 pages
          → 同一个 request 在不同 rank 的 page table 不一致 → 后续 load 时错位!

        解决: all_reduce(MIN):
          min(1024, 960, 1024, 1024) = 960
          → 所有 rank 统一按 960 tokens (60 pages) 取
          → page table 在所有 rank 上保持一致

        ┌─────────────────────────────────────────────────────────────┐
        │  TP 多卡一致性流程                                          │
        │                                                             │
        │  rank0  ─┐                                                  │
        │  rank1  ─┤  all_reduce(MIN)  ──→  统一 hit_count = 960     │
        │  rank2  ─┤                                                  │
        │  rank3  ─┘                                                  │
        │                                                             │
        │  同步组构建: _create_prefetch_sync_groups()                  │
        │    → 基于 tp_group / attn_cp_group / attn_tp_group          │
        │    → 使用 gloo backend (CPU tensor 上的 all_reduce)         │
        └─────────────────────────────────────────────────────────────┘

        ──────── 阈值判定: prefetch_threshold ────────

        if storage_hit_count < prefetch_threshold:
            → 撤销: 命中太少不值得 IO 开销
            → 释放预分配的 host memory
            → 通知 scheduler 通过 prefetch_revoke_queue
        else:
            → 放行: 截断 hash_value / host_indices 到命中部分
            → 多分配的部分释放回 host mem pool
            → 操作进入 prefetch_buffer 等待 IO 线程执行

        ┌──────────────────────────────────────────────────────────┐
        │  阈值判定流程                                            │
        │                                                          │
        │  storage_hit_count (all_reduce 后)                      │
        │       │                                                  │
        │       ├──── < prefetch_threshold ──→  ❌ REVOKE          │
        │       │     ├ 释放 host_indices 全部                      │
        │       │     └ 通知 prefetch_revoke_queue                 │
        │       │                                                  │
        │       └──── >= prefetch_threshold ──→  ✅ PROCEED         │
        │             ├ 截断 hash_value[:hit_pages]                 │
        │             ├ 截断 host_indices[:hit_count]               │
        │             ├ 释放多余的 host_indices[hit_count:]         │
        │             └ 放入 prefetch_buffer                       │
        └──────────────────────────────────────────────────────────┘
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
        L3 写入入口 —— 将 Host 上的 KV page 备份到 L3 存储后端。

        ╔══════════════════════════════════════════════════════════════════════╗
        ║  完整写入链路 (GPU → Host → L3)                                     ║
        ╠══════════════════════════════════════════════════════════════════════╣
        ║                                                                      ║
        ║  ① GPU→Host (DMA)                                                   ║
        ║     start_writing() → write_stream DMA → finish_event              ║
        ║                          │                                           ║
        ║                          ▼                                           ║
        ║  ② Host 确认 (HiRadixCache 层)                                       ║
        ║     _finish_write_through_ack()                                     ║
        ║       ├─ node.write_through_pending_id = None  (清除 DMA 等待标记)    ║
        ║       ├─ _record_store_event(medium=CPU)        (记录存储事件)        ║
        ║       └─ write_backup_storage(node)             ←── 触发 L3 写入    ║
        ║                          │                                           ║
        ║                          ▼                                           ║
        ║  ③ L3 写入 (本函数)                                                  ║
        ║     write_storage(host_indices, token_ids, hash_value, prefix_keys) ║
        ║       → 创建 StorageOperation → 放入 backup_queue                  ║
        ║                          │                                           ║
        ║                          ▼                                           ║
        ║  ④ backup_thread 消费                                                ║
        ║     backup_thread_func()                                            ║
        ║       → _page_backup() 逐 batch 写 L3                               ║
        ║       → ack_backup_queue 通知完成                                    ║
        ║                          │                                           ║
        ║                          ▼                                           ║
        ║  ⑤ 上层确认 (HiRadixCache 层)                                        ║
        ║     _drain_backup():                                                ║
        ║       → ongoing_backup.pop(ack_id)                                  ║
        ║       → node.release_host()  (L3 备份完成, host mem 可释放)          ║
        ║                                                                      ║
        ╚══════════════════════════════════════════════════════════════════════╝

        ┌─────────────────────────────────────────────────────────────────────┐
        │  调用方: HiRadixCache.write_backup_storage()                       │
        │                                                                     │
        │  write_backup_storage 做了两件事:                                    │
        │                                                                     │
        │  1) 处理 split: 如果 node 在 DMA 期间被 radix tree split 了，         │
        │     需要通过 _concat_split_chain() 沿父节点回溯拼接,                   │
        │     恢复入队时刻的完整 (key, hash_value, host_value)                  │
        │                                                                     │
        │     原始 node:  [A|B|C|D]                                           │
        │     DMA 期间 split:  [A|B]  +  [C|D]                               │
        │     → concat: 遍历 parent 拼回 [A|B|C|D]                            │
        │                                                                     │
        │  2) 构建 prefix_keys:                                               │
        │     如果 hicache_storage_pass_prefix_keys=True,                     │
        │     从 root 沿路收集祖先节点的 hash_value 作为前缀,                  │
        │     用于 L3 后端(如 mooncake)做层级索引                              │
        │                                                                     │
        │  3) 调用本函数:                                                     │
        │     write_storage(host_value, key, hash_value, prefix_keys)         │
        │     + node.protect_host()  (防止 L3 写完前 host mem 被回收)          │
        │                                                                     │
        └─────────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────────────┐
        │  Host Memory 生命周期 (protect_host / release_host)                  │
        │                                                                     │
        │  write_storage() 时刻:                                              │
        │    → node.protect_host()   host ref+1, 防止 L3 IO 期间被 evict       │
        │                                                                     │
        │  ack_backup_queue 确认时刻:                                         │
        │    → node.release_host()   host ref-1, L3 已有副本, 可安全释放       │
        │                                                                     │
        │  如果 L3 写入失败 (backup_skip / batch_set 失败):                    │
        │    → 仍会 ack, 但 completed_tokens < 预期                           │
        │    → release_host 仍会执行, 只是该 node 的 L3 副本不完整             │
        │                                                                     │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            host_indices: Host memory pool 中的 page 索引 (每个 page_size 个 slot 一组)
            token_ids:   该 node 包含的 token ID 序列
            hash_value:  每个 page 的 hash 值列表 (None 则由 backup_thread 计算)
            prefix_keys:  祖先节点的 hash 值列表, 用于 L3 后端层级索引

        Returns:
            operation.id —— 用于在 ongoing_backup 中追踪, ack 时释放 host ref
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
        """
        逐 batch 将 Host KV page 写入 L3 存储后端 (由 backup_thread 调用)。

        ┌──────────────────────────────────────────────────────────────┐
        │  逐 batch 写入流程                                           │
        │                                                              │
        │  hash_value: [h0, h1, h2, ..., hN]                          │
        │                  │                                           │
        │                  ▼  按 STORAGE_BATCH_SIZE 分批               │
        │  ┌─────────────────────────────────────┐                    │
        │  │ batch 0:  h0..hK                    │                    │
        │  │   → page_set_func(batch_hashes,      │                    │
        │  │       batch_host_indices, extra_info) │                    │
        │  │   → 成功?                              │                    │
        │  │     ├─ draft_page_set (best-effort)   │                    │
        │  │     ├─ prefix_keys += batch_hashes    │                    │
        │  │     ├─ completed_tokens += pages      │                    │
        │  │     └─ 继续下一 batch                  │                    │
        │  │   → 失败?                              │                    │
        │  │     └─ break (后续 batch 跳过)          │                    │
        │  ├─────────────────────────────────────┤                    │
        │  │ batch 1:  hK+1..h2K                  │                    │
        │  │   ...                                  │                    │
        │  └─────────────────────────────────────┘                    │
        │                                                              │
        │  关键: prefix_keys 滚动累加, 供后端做层级索引                  │
        │        batch0 完成后 prefix_keys += [h0..hK]                 │
        │        batch1 写入时 extra_info.prefix_keys 已包含 batch0     │
        └──────────────────────────────────────────────────────────────┘
        """
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
        L3 备份后台线程 —— 消费 backup_queue, 逐 operation 调用 _page_backup 写入 L3。

        ┌────────────────────────────────────────────────────────────────┐
        │  backup_thread 生命周期                                         │
        │                                                                │
        │  backup_queue ──→  _page_backup(op)  ──→  ack_backup_queue    │
        │     │                   │                       │               │
        │     │                   │  逐 batch 写 L3       │  通知上层     │
        │     │                   │  completed_tokens++   │  release_host │
        │     │                   │                       │               │
        │     │              backup_skip?                  │               │
        │     │              ├─ YES: 跳过写入               │               │
        │     │              └─ NO:  正常执行               │               │
        │                                                                │
        │  注意: 即使 backup_skip 或写入失败, 仍会 ack,                  │
        │        上层通过 completed_tokens 判断实际写入量                  │
        └────────────────────────────────────────────────────────────────┘
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
