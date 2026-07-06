# to be combined with the sparse coordinator class and sparse algorithm family

import logging
from typing import List, NamedTuple, Union

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseDSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    MLATokenToKVPoolHost,
)
from sglang.srt.utils import get_device_module

device_module = get_device_module()

from sglang.jit_kernel.hisparse import (
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


class HiSparseAct(NamedTuple):
    """📨 staging DMA 异步跟踪记录 —— 一个 prefill 请求的 D→H backup 飞行票据。

    存入 ``ack_staging_queue``，由 ``collect_ready_reqs`` 轮询 ``finish_event``
    收割；完成后弹出并调用 ``alloc_device_buffer`` 把请求转为 ready 状态。
    """

    start_event: device_module.Event   # DMA 入队前的 cuda Event（调试用）
    finish_event: device_module.Event  # DMA 完成 Event — 轮询点
    req: Req                           # 关联的请求


class HiSparseTokenStats(NamedTuple):
    """📊 HiSparse Device/Host 双层 token 占用统计 —— 喂给 pool_stats_observer。"""

    device_tokens: int          # Device buffer 已占用 token 数
    device_token_usage: float   # 占用率 [0, 1]
    host_tokens: int            # Host pool 已占用 token 数
    host_token_usage: float     # 占用率 [0, 1]


class HiSparseCoordinator:
    """🧩 HiSparse 稀疏注意力协调器 —— 调度 Device buffer / Host pool 之间的 KV 搬运。

    DSv4 / DSA 等稀疏 attention 模型只对 top-k 相关 token 做精确 attend，因此
    **完整 KV 不必常驻 GPU**：Device 只保留一个小的 hot buffer（按 page 分配），
    其余历史 KV 卸载到 Host pool，decode 时按 top-k 结果把命中的页 swap-in 回 Device。
    本协调器就是这个分层调度的"大脑"，与 Scheduler / AttentionBackend / CUDA graph
    紧密协作。

    ╔════════════════════════════════════════════════════════════════════════╗
    ║  🧩 对外接口清单（按运行时序）                                            ║
    ╠════════════════════════════════════════════════════════════════════════╣
    ║  入场阶段（一个 req 的 KV 进入分层缓存）:                                ║
    ║    • admit_request_into_staging — prefill 完成后异步 D→H backup prefill KV║
    ║    • admit_request_direct       — disagg/PP 场景 KV 已 RDMA 到 Host, 跳过 DMA║
    ║  轮询阶段（每轮 scheduler 主循环）:                                      ║
    ║    • collect_ready_reqs   — 收割 staging DMA 完成 → 转 ready, 进入 decode batch║
    ║    • has_ongoing_staging — 判断是否还有未完成 staging, 影响 idle 判定    ║
    ║  Decode 前准备（ForwardBatch.prepare_attn_batch, schedule_batch.py:2768）: ║
    ║    • map_last_loc_to_buffer — 注册最新 token 的 Device buffer 映射       ║
    ║      ├─ _eager_backup_previous_token — 增量 backup 新产生的 compressed token║
    ║      ├─ _grow_device_buffers (DSA)    — seq_len 增长时扩容 buffer        ║
    ║      └─ full_to_hisparse_device_index_mapping 更新                     ║
    ║  Forward 前 sync（model_runner.py:3620）:                                ║
    ║    • wait_for_pending_backup — 阻塞当前流等待上一步 backup DMA 完成      ║
    ║  Attention 执行中（dsa_backend / dsv4/indexer）:                         ║
    ║    • swap_in_selected_pages — 生产路径：JIT kernel 把 top-k 选中页搬回 Device║
    ║    • naive_load_topk         — 调试用 Python 循环 oracle, 不上生产        ║
    ║  生命周期结束:                                                          ║
    ║    • request_finished — req 完成时释放 Device buffer + Host pool 资源     ║
    ║    • abort_staging_request — 中途 abort 且还在 staging 队列 → 释放资源     ║
    ║    • retract_req       — 统一入口, 按状态分发到上面两个                   ║
    ║  其他:                                                                  ║
    ║    • get_token_stats / set_decode_producer_stream / alloc_device_buffer   ║
    ╚════════════════════════════════════════════════════════════════════════╝

    ╔════════════════════════════════════════════════════════════════════════╗
    ║  🔗 宏观调用链（谁创建 / 谁调用本协调器）                                  ║
    ╠════════════════════════════════════════════════════════════════════════╣
    ║  创建:                                                                  ║
    ║    ModelRunner.initialize() (model_runner.py:865)                       ║
    ║      → new HiSparseCoordinator(...)                                     ║
    ║    Scheduler.get_model_runner() 后                                     ║
    ║      → scheduler.hisparse_coordinator = runner.hisparse_coordinator    ║
    ║      → set_decode_producer_stream(forward_stream) (scheduler.py:545)    ║
    ║                                                                          ║
    ║  运行时（每个 prefill req 完成后）:                                      ║
    ║    BatchResultProcessor.process_batch_result (batch_result_processor:261)║
    ║      → admit_request_into_staging(req)        ← 入队 staging DMA        ║
    ║                                                                          ║
    ║  运行时（scheduler 每轮主循环）:                                          ║
    ║    scheduler.event_loop_overlap() (scheduler.py:2624)                   ║
    ║      → collect_ready_reqs()                   ← 收割完成的 staging       ║
    ║      → _build_hisparse_decode_batch(ready)    ← 构造 decode batch        ║
    ║    scheduler.is_idle() (scheduler.py:3576)                             ║
    ║      → has_ongoing_staging()                  ← 影响 idle 判定          ║
    ║                                                                          ║
    ║  运行时（每个 decode step）:                                              ║
    ║    ForwardBatch.prepare_attn_batch (schedule_batch.py:2768)             ║
    ║      → map_last_loc_to_buffer()               ← 注册最新 token 映射     ║
    ║    ModelRunner.forward() (model_runner.py:3620)                         ║
    ║      → wait_for_pending_backup()              ← 阻塞 sync 上一步 DMA    ║
    ║    AttentionBackend (dsa_backend:1641 / dsv4/indexer:619)               ║
    ║      → swap_in_selected_pages()               ← top-k swap-in           ║
    ╚════════════════════════════════════════════════════════════════════════╝

    🧬 设计要点:
    - **两套 KV 池**: ``mem_pool_device`` (GPU hot buffer, 容量小) + ``mem_pool_host``
      (Host paged pool, 容量大)。dsv4 走 ``DeepSeekV4PagedHostPool`` (c4 压缩 KV),
      普通 DSA 走 ``MLATokenToKVPoolHost`` (未压缩 KV)。
    - **三种索引表**: ``req_to_device_buffer`` (req → Device slot),
      ``req_to_host_pool`` (req → Host slot), ``req_device_buffer_tokens`` /
      ``req_device_buffer_token_locs`` (swap-in kernel 用的 layer-major 映射)。
    - **双流异步**: ``write_staging_stream`` 跑 prefill backup,
      ``decode_backup_stream`` 跑 decode 增量 backup；用 ``_backup_done_event`` 同步。
    - **CUDA graph 友好**: 所有 buffer 预分配, ``num_real_reqs`` 标量在 graph
      replay 前填入实际 batch_size, 让 padding 位早退。
    - **两条入场路径**: ``admit_request_into_staging`` (普通 prefill, 走异步 DMA) vs
      ``admit_request_direct`` (disagg/PP, KV 已 RDMA 到 Host, 跳过 DMA)。
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: Union[
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ],
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
        host_to_device_ratio: int = 2,
    ):
        """🏗️ 协调器初始化 —— 选配 KV 池 + 预分配所有 req 级索引表与 swap-in buffer。

        所有张量在此一次性分配, 之后运行期不再扩容, 保证 CUDA graph 可重放。
        详见类 docstring 的"对外接口清单"理解每个字段被谁读写。
        """
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size  # Device hot buffer 容量 (token 数)
        self.device = device
        # compress_ratio: c4=4 / c128=128, 仅 dsv4 用；DSA 路径恒为 1
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

        # ── ① 按后端类型 (DSV4 vs DSA) 选配 Device / Host 双层 KV 池 ──
        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        if self.is_dsv4_hisparse:
            # dsv4 路径: 压缩 KV (c4/c128), Host 池按压缩后页数分配
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
            page_size = self.mem_pool_device.page_size
            # 压缩后总 token 数向上取整到 page 边界 → Host 页数
            num_host_pages = (
                self.token_to_kv_pool_allocator.size_full // self.compress_ratio
                + page_size
                - 1
            ) // page_size
            self.mem_pool_host = DeepSeekV4PagedHostPool(
                pool_name="dsv4_hisparse_c4",
                device_buffers=self.mem_pool_device.kv_buffer,
                item_bytes=self.mem_pool_device.bytes_per_page_padded,
                num_host_pages=num_host_pages,
                slot_page_size=page_size,
                layout="layer_first",  # [layer, page, token_dim] 便于逐层 DMA
            )
            # 单 token 字节数 = kv_cache_total_dim × dtype.itemsize
            self.item_size_bytes = (
                self.mem_pool_device.kv_cache_total_dim
                * self.mem_pool_device.store_dtype.itemsize
            )
        else:
            assert isinstance(
                self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
            )
            # DSA 路径: 未压缩 MLA KV, Host 池按 host_to_device_ratio 等比扩容
            self.mem_pool_device: HiSparseDSATokenToKVPool = (
                self.token_to_kv_pool_allocator.get_kvcache()
            )
            self.mem_pool_host = MLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,  # 0 = 由 device_pool 推算
                page_size=self.mem_pool_device.page_size,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
            self.item_size_bytes = self.mem_pool_host.token_stride_size
        self.page_size = self.mem_pool_device.page_size

        # ── ② 维度推算: req 槽位 / 上下文长度 / 压缩后上下文长度 ──
        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        # 压缩后最大长度 (向上取整): 用于 req_to_host_pool 第二维
        max_compressed_context_len = (
            max_context_len + self.compress_ratio - 1
        ) // self.compress_ratio

        # padded_buffer_size: 多保留一页, 给新 token 写入留缓冲 (避免边界扩容)
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        # ── ③ req 级索引表 (req_pool_idx → buffer/host slot 映射) ──
        # req → Device buffer slot 列表 (长度 padded_buffer_size)
        self.req_to_device_buffer = torch.zeros(
            (max_num_req_slots, self.padded_buffer_size),
            dtype=torch.int64,
            device=device,
        )
        # req → 当前已分配的 Device buffer 大小 (CPU 标量, 供 _grow_device_buffers 判断)
        self.req_device_buffer_size = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        # req → Host pool slot 列表 (长度 compressed_context + page, -1=未分配)
        self.req_to_host_pool = torch.full(
            (max_num_req_slots, max_compressed_context_len + self.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        )
        # req → Host pool 已分配长度 (CPU 标量, 用于增量 alloc)
        self.req_to_host_pool_allocated_len = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )

        # ── ④ 异步 DMA 流 + 同步 Event ──
        self.write_staging_stream = device_module.Stream()  # prefill D→H backup 流
        self.decode_backup_stream = device_module.Stream()  # decode 增量 backup 流
        self.ack_staging_queue: List[HiSparseAct] = []     # 飞行中的 staging 票据
        self.decode_producer_stream = None                  # forward 主流, 由 set_decode_producer_stream 注入
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False                   # decode backup 是否在飞行中

        # ── ⑤ TP 同步组 ──
        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # ── ⑥ swap-in kernel 用的 layer-major 映射表 (CUDA graph 友好) ──
        # [layer, req, slot] 三个张量同步维护:
        #   req_device_buffer_tokens      - 每个 slot 当前 cache 的 token 序号 (-1=空)
        #   req_device_buffer_token_locs  - 每个 token 序号对应的 Device KV 索引
        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        # LRU 初始化: 每个 req 的 buffer slot 按顺序 [0,1,2,...,buf_size-1]
        # swap_in kernel 据此做 LRU 替换决策
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_req_slots, 1)
            .contiguous()
        )
        # 复用的 arange, alloc_device_buffer 时填 tokens=[0,1,...,size-1]
        self._device_buffer_arange_i32 = torch.arange(
            self.device_buffer_size, dtype=torch.int32, device=device
        )

        # ── ⑦ swap_in_selected_pages 的输出 buffer (预分配, graph 友好) ──
        # [req, top_k] 输出每个 req 选中的 top-k token 的 Device KV 索引
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        # 备用 buffer: dsv4 indexer 走 raw_indices 路径时复用
        self.raw_indices_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        # 标量张量: graph replay 前填入真实 batch_size, 让 padding 位早退
        # (避免无效 swap-in 写脏真实 slot)
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # ── ⑧ skip_first_backup 标记位 ──
        # staging 完成后第一个 decode step 不需要 backup (prefill KV 已 backup 过)
        # 用后即清, 仅生效一次
        self._skip_first_backup = [False] * max_num_req_slots

    def set_decode_producer_stream(self, stream) -> None:
        """🔗 注册 forward 主流 —— 让 decode_backup_stream 知道该等谁。

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        Scheduler.__init__ → tp_worker 拉起后 (scheduler.py:545)
          → hisparse_coordinator.set_decode_producer_stream(self.forward_stream)

        ⚙️ 行为:
            之后 _eager_backup_previous_token 在 decode_backup_stream 上 launch
            DMA 前, 会先 wait_stream(decode_producer_stream), 确保主流已经把
            新 token 的 KV 写完, 再开始 D→H backup。
        """
        self.decode_producer_stream = stream

    def get_token_stats(self) -> HiSparseTokenStats:
        """📊 拉取 Device/Host 双层 token 占用率 —— 喂给 pool_stats_observer。

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        Scheduler.event_loop → PoolStatsObserver._get_hisparse_token_info
          (pool_stats_observer.py:230)
          → hisparse_coordinator.get_token_stats()  ← 你在这
          → 替换 PoolStats 的 device/host 字段后上报

        📤 返回: HiSparseTokenStats(device/host 各两项: 已占 token 数 + 占用率)
        """
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        host_capacity = self.mem_pool_host.size
        host_tokens = host_capacity - self.mem_pool_host.available_size()
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

    def admit_request_into_staging(self, req: Req) -> None:
        """🚚 入场 staging 路径 —— 异步把 prefill KV 从 Device backup 到 Host。

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        BatchResultProcessor.process_batch_result (batch_result_processor.py:261)
          — prefill 完成后逐 req 处理
          → admit_request_into_staging(req)            ← 你在这
            [异步] write_staging_stream.backup_from_device_all_layer
          → ack_staging_queue.append(HiSparseAct(...))  ← 入队等收割

        后续:
          scheduler.event_loop_overlap (scheduler.py:2624)
            → collect_ready_reqs()  — 轮询 finish_event, 完成后调 alloc_device_buffer
            → _build_hisparse_decode_batch(ready_reqs)  — 进入 decode batch

        ⚙️ 行为 (4 步):
          ① req.hisparse_staging=True — 标记 req 处于 staging 中 (影响 retract 路径)
          ② 取 prefill 写入的 full_kv_indices → translate 到 Device buffer 索引
          ③ alloc_paged_token_slots — 在 Host pool 给 prefill_len 个 token 分配 slot
          ④ write_staging_stream 上 launch D→H backup, 记录 finish_event, 入队

        ⚠️ 注意:
          - DMA 在 write_staging_stream 异步执行, 本函数立即返回, scheduler 不阻塞
          - host/device_indices 都 record_stream(write_staging_stream),
            防止主流提前 free tensor 导致 UAF
        """
        # ① 标记 staging 中 — retract_req 据此走 abort_staging_request 分支
        req.hisparse_staging = True

        # ② 取 req 在 req_to_token_pool 中已写入的 full indices → 转 Device buffer 索引
        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.fill_len
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        # ③ 在 Host pool 给 prefill 长度个 token 分配 slot (写入 req_to_host_pool)
        prefill_len = len(device_indices)
        host_indices = self.mem_pool_host.alloc_paged_token_slots(
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            0,
            prefill_len,
        )

        # ④ 异步 D→H backup: 全 layer 一次性 DMA
        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            finish_event.record()  # collect_ready_reqs 轮询此事件
            # 让索引张量归属 write_staging_stream, 防止主流提前释放
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        # 入队等 scheduler 收割
        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def admit_request_direct(self, req: Req) -> None:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging DMA entirely. Only allocates a small device buffer
        (4KB) for decode-time swap-in, then marks the request as ready.
        Host indices were already written to req_to_host_pool.

        Metadata fixups after alloc_device_buffer():
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        ① PP 场景 (scheduler_pp_mixin.py:1397)
            PP 上游 rank 完成后 → KV 通过 RDMA 直接写到本 rank 的 Host pool
            → admit_request_direct(req)  ← 你在这
        ② Disagg decode 场景 (disaggregation/decode.py:1968)
            prefill 节点的 KV 通过 RDMA 推到 decode 节点 Host pool
            → admit_request_direct(req)

        ⚙️ 与 staging 路径的关键差异:
            staging:  本函数内部启动 D→H DMA (写完后 collect_ready_reqs 转 ready)
            direct :  RDMA 已经把 KV 写到 Host, 跳过 DMA, 直接转 ready
                      仅在 Device 上分配 swap-in 用的 buffer, 然后立即标 staging=False
        """
        self.alloc_device_buffer(req)

        # 短序列: 全量预加载到 Device buffer, 后续 kernel fast-path 直接命中
        host_len = self.host_token_len(req.kv_allocated_len)
        if host_len <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            self._preload_to_device_buffer(req)
        else:
            # 长序列: 把 device_buffer_tokens 复位为 -1
            #   → swap-in kernel 看到所有 slot 都是空 → 每个 top-k 查询都 miss → 走 host load
            # Long sequence: reset device_buffer_tokens to -1 so the kernel
            # sees all slots as empty -> every top-k lookup is a miss -> host load.
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1

        # 立即转 ready: 不进 ack_staging_queue, 下一轮直接可调度 decode
        req.hisparse_staging = False
        # 第一个 decode step 跳过 backup: direct 路径 prefill KV 已在 Host (RDMA 写入)
        self._skip_first_backup[req.req_pool_idx] = True
        logger.debug("HiSparse: admitting request %s directly", req.rid)

    def host_token_len(self, kv_allocated_len: int) -> int:
        """📐 全 KV token 数 → Host 压缩后 token 数。

        dsv4 路径: 压缩 KV (compress_ratio=4 或 128), host_token_len = alloc_len / ratio
        DSA 路径:  未压缩, host_token_len = alloc_len (compress_ratio=1)
        """
        if self.is_dsv4_hisparse:
            return kv_allocated_len // self.compress_ratio
        return kv_allocated_len

    def _preload_to_device_buffer(self, req: Req) -> None:
        """Preload all tokens from host pool into the device buffer.

        ⚙️ 用途: admit_request_direct 短序列分支调用 —— 因为 swap-in kernel
        短序列走 fast-path (不查 host), 必须先把所有 host KV 拉到 device buffer,
        否则 top-k 会 miss。逐层调用 load_to_device_per_layer 完成 H→D 拷贝。
        """
        n = self.host_token_len(req.kv_allocated_len)
        host_indices = self.req_to_host_pool[req.req_pool_idx, :n]
        device_locs = self.req_to_device_buffer[req.req_pool_idx, :n]

        # 逐层 H→D 拷贝 (与 swap_in_selected_pages 单层调用不同, 这里需要预热全部层)
        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def alloc_device_buffer(self, req: Req) -> None:
        """🎁 给 req 在 Device 上分配 hot buffer slot, 并填好 swap-in 映射表。

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        ① staging 路径: collect_ready_reqs (本文件) — staging DMA 完成后调用
        ② direct 路径: admit_request_direct (本文件) — RDMA 完成后立即调用
          → alloc_device_buffer(req)  ← 你在这
            → token_to_kv_pool_allocator.alloc_device_buffer(...)  ← 真正分配 slot

        ⚙️ 行为 (4 步):
          ① 算 alloc_size: dsv4 直接用 padded_buffer_size; DSA 按 page 对齐 + 上界裁剪
          ② translate full_kv → compressed 逻辑索引 (压缩空间)
          ③ 调 allocator.alloc_device_buffer 拿到 buffer_indices (None=OOM)
          ④ 把 indices 填进三张表:
              req_to_device_buffer          (req 级 — 给 host backup / preload 用)
              req_device_buffer_tokens      (layer-major — 告诉 kernel slot 存的是 token #N)
              req_device_buffer_token_locs  (layer-major — token #N → Device KV 索引)

        ⚠️ 副作用: alloc 失败抛 RuntimeError (调度器层面会重试或 abort)
        """
        # ① 算 alloc_size —— dsv4 / DSA 路径差异
        if self.is_dsv4_hisparse:
            # dsv4: prefill 的 compressed KV 一次性铺满 padded buffer
            allocated_len = req.fill_len
            alloc_size = self.padded_buffer_size
        else:
            # DSA: 按 page 对齐到当前 token 数, 但不超过 device_buffer_size
            allocated_len = req.kv_allocated_len
            page_size = self.mem_pool_device.page_size
            # Allocate only enough for current tokens (page-aligned).
            # When prefill already fills device_buffer_size, include the reserved page.
            alloc_size = min(
                ((allocated_len + page_size - 1) // page_size) * page_size,
                self.device_buffer_size,
            )
            # 已经接近满 → 多带一页缓冲 (避免下一步立即扩容)
            if alloc_size == self.device_buffer_size:
                alloc_size = self.padded_buffer_size

        # ② full_kv → compressed 逻辑索引 (compression space)
        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.req_pool_idx, :allocated_len]
            )
        )
        compressed_len = len(compressed_logical_indices)

        # ③ 真正向 allocator 申请 hot buffer slot
        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            compressed_logical_indices, alloc_size
        )
        if buffer_indices is None:
            # OOM — 调度器上层应 evict 或重试, 这里直接抛
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(compressed_len=%d, alloc_size=%d)",
                req.rid,
                compressed_len,
                alloc_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

        # ④ 填三张表 —— 之后 swap-in kernel 据此决策
        buffer_indices = buffer_indices.to(torch.int32)
        self.req_to_device_buffer[req.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.req_pool_idx] = alloc_size
        # tokens 表填 [0,1,...,buf_size-1]: 表示 slot i 当前 cache 第 i 个 token
        # (swap-in kernel 用此判断 LRU 与命中)
        self.req_device_buffer_tokens[
            :, req.req_pool_idx, : self.device_buffer_size
        ] = self._device_buffer_arange_i32
        # token_locs 表: slot i 的真实 Device KV 索引
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )

    def _grow_device_buffers(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """📈 DSA 路径专用: decode 时若 seq_len 超过当前 buffer 容量, 批量扩容。

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        ForwardBatch.prepare_attn_batch (schedule_batch.py:2768)
          → map_last_loc_to_buffer
            → _grow_device_buffers(...)  ← 你这 (仅 DSA, 非 dsv4)

        ⚙️ 行为:
          ① 筛选需要扩容的 req (短序列 & seq_len > current_cap)
          ② CPU 上批量算每个 req 的新 cap + grow_size
          ③ 一次 hisparse_attn_allocator.alloc(total_grow) 拿到大块索引
          ④ 切片分给每个 req, 同步更新 req_to_device_buffer / token_locs / size
          ⑤ 最后返回所有 req 的"最新 token 对应的 buffer slot" (供 map_last_loc_to_buffer 后续填映射表)

        ⚠️ 设计: 一次性大块 alloc 比逐 req alloc 更高效 (减少锁/同步开销)
        """
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]
        # 短序列分支: seq_len ≤ device_buffer_size 才有"在 buffer 内扩容"语义
        # (长序列已经走 host swap-in 路径, 不需要 grow)
        short_reqs_cpu = seq_lens_cpu <= self.device_buffer_size
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)

        if torch.any(needs_grow_cpu):
            page_size = self.mem_pool_device.page_size
            grow_indices = torch.where(needs_grow_cpu)[0]

            # Compute all grow sizes on CPU, then do a single bulk allocation
            # ② CPU 端算每个 req 的新 cap + grow_size
            req_idxs = []
            old_caps = []
            new_caps = []
            grow_sizes = []
            total_grow = 0
            for i in grow_indices.tolist():
                req_idx = int(req_pool_indices_cpu[i])
                current_cap = int(current_caps[i])
                seq_len = int(seq_lens_cpu[i])

                # page 对齐 → 不超过 device_buffer_size; 满则带 extra page
                new_cap = min(
                    ((seq_len + page_size - 1) // page_size) * page_size,
                    self.device_buffer_size,
                )
                if new_cap == self.device_buffer_size:
                    new_cap = self.padded_buffer_size
                grow_size = new_cap - current_cap
                if grow_size <= 0:
                    continue
                req_idxs.append(req_idx)
                old_caps.append(current_cap)
                new_caps.append(new_cap)
                grow_sizes.append(grow_size)
                total_grow += grow_size

            # ③ 一次性向 allocator 申请所有 grow 索引 (bulk alloc)
            if total_grow > 0:
                all_new_indices = (
                    self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                        total_grow
                    )
                )
                if all_new_indices is None:
                    logger.error(
                        "HiSparse: _grow_device_buffers bulk alloc failed "
                        "(total_grow=%d)",
                        total_grow,
                    )
                    raise RuntimeError(
                        f"HiSparse _grow_device_buffers failed (total_grow={total_grow})"
                    )

                # ④ 切片分配 + 更新三张表 (与 alloc_device_buffer 一致)
                offset = 0
                for req_idx, current_cap, new_cap, grow_size in zip(
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]
                    offset += grow_size
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk
                    self.req_device_buffer_size[req_idx] = new_cap

        # ⑤ 返回每个 req 最新 token 的 buffer slot (clamp 防 long 序列越界)
        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]

    def has_ongoing_staging(self) -> bool:
        """❓ 是否还有 staging DMA 在飞行中 —— 影响 scheduler 的 idle 判定。

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        Scheduler.is_idle (scheduler.py:3576)
          → has_ongoing_staging()  ← 你在这
            → idle &= not has_ongoing_staging
            (有 staging 时 scheduler 不能进 idle, 需要继续轮询 collect_ready_reqs)
        """
        return len(self.ack_staging_queue) > 0

    def collect_ready_reqs(self) -> List[Req]:
        """🔍 轮询 staging 队列, 收割已完成 DMA 的 req, 转为 ready。

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        Scheduler.event_loop_overlap (scheduler.py:2624)
          → collect_ready_reqs()                       ← 你在这
            → (TP all_reduce MIN 同步所有 rank 的完成数)
            → pop FIFO 的前 finish_count 个 req
            → alloc_device_buffer(req)                 ← 给 ready req 分配 device buffer
            → _skip_first_backup=True                  ← 标记下一步 decode 不 backup
            → req.hisparse_staging=False
          → _build_hisparse_decode_batch(ready_reqs)   ← 进入 decode batch

        ⚙️ 关键: TP 同步
            每个 rank 的 finish_count 可能不同 (各 rank DMA 速度不同),
            用 all_reduce(MIN) 取最小值, 保证所有 rank 同步收割相同的 req。
            否则不同 rank 的 ready_reqs 不一致 → decode batch 各 rank 视图分裂。

        📤 返回: 已 ready 的 req 列表 (可能为空)
        """
        ready_reqs: List[Req] = []
        if len(self.ack_staging_queue) == 0:
            return ready_reqs

        # ① 数已完成事件个数 (FIFO 队列, 遇到未完成即停 — 保证顺序)
        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
            if not finish_event.query():
                break
            finish_count += 1
        # ② TP 同步: 取所有 rank 的最小完成数, 保证各 rank 收割同一批 req
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        # ③ pop + 转 ready: 每个 req 分配 device buffer 并清标记位
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            # prepare device buffer and update req
            self.alloc_device_buffer(req)
            # staging 已 backup 了 prefill KV, 第一个 decode step 跳过 backup
            self._skip_first_backup[req.req_pool_idx] = True
            req.hisparse_staging = False
            finish_count -= 1
            ready_reqs.append(req)
        return ready_reqs

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """🗺️ 注册"最新 token 的 Device buffer 映射" —— decode 前的关键准备步骤。

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        ForwardBatch.prepare_attn_batch (schedule_batch.py:2768)
          → map_last_loc_to_buffer(...)              ← 你在这
            ├─ _eager_backup_previous_token(...)     ← 增量 backup 上一步产生的 compressed token
            └─ (DSA) _grow_device_buffers(...)        ← 短序列扩容 + 填 full_to_hisparse_device_index_mapping
            └─ (dsv4) 仅对齐的 req 处理 (compress_ratio 整数倍才产生新 compressed token)

        ⚙️ 做两件事:
          ① _eager_backup_previous_token — 把上一步新产生 (或刚刚) 的 compressed token
             从 Device buffer backup 到 Host (decode 增量 backup)
          ② 把最新 token 的 out_cache_loc ↔ Device buffer slot 的映射写入两张表:
              - req_device_buffer_token_locs[:, req, device_buffer_size]  ← reserved slot
              - mem_pool_device.full_to_hisparse_device_index_mapping    ← 全局反查表
             swap-in kernel 据此快速找到最新 token 的 Device KV 索引。
        """
        # ① 增量 backup 上一步的 compressed token (内部会跳过首个 decode step)
        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        if not self.is_dsv4_hisparse:
            # ② DSA 路径: 每个 decode step 都产生新 token → 都要扩容 + 映射
            # Grow device buffers if needed and resolve the latest-token slot.
            reserved_buffer_loc = self._grow_device_buffers(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            # reserved slot 位置 (device_buffer_size) 存最新 token 的 buffer slot
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)

            # No need to clear prior mappings: the only consumer of the mapping
            # for past tokens is the swap-in kernel, and it goes through
            # top_k_device_locs returned by swap_in_selected_pages -- not via
            # mapping[old_out_cache_loc] -- so stale entries are harmless.
            # 全局反查表: out_cache_loc → Device buffer slot (供 swap-in kernel 复用)
            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
                out_cache_loc
            )
            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = reserved_buffer_loc
            return

        # ② dsv4 路径: 只有 seq_len % compress_ratio == 0 才产生新 compressed token
        active_reqs = seq_lens % self.compress_ratio == 0
        if not torch.any(active_reqs):
            return

        active_seq_lens = seq_lens[active_reqs]
        active_out_cache_loc = out_cache_loc[active_reqs]
        active_req_pool_indices = req_pool_indices[active_reqs]

        # 压缩空间下的 seq_len (整数除)
        compressed_seq_lens = active_seq_lens // self.compress_ratio
        # reserved slot 位置 (clamp 防长序列越界)
        reserved_positions = (compressed_seq_lens - 1).clamp(
            max=self.device_buffer_size
        )
        reserved_buffer_loc = self.req_to_device_buffer[
            active_req_pool_indices, reserved_positions
        ]

        # 填两张表 (与 DSA 路径同样的语义, 只是仅作用于 active_reqs)
        self.req_device_buffer_token_locs[
            :, active_req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)

        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            active_out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            reserved_buffer_loc
        )

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Back up the previous compressed token to host memory.

        Each newly produced compressed token (one per `compress_ratio` decode
        steps) must be backed up to host so the swap-in kernel can later
        recover it.

        Two cases are skipped:
        - The first decode step right after staging: all prefill tokens were
          already backed up during staging, so there is nothing new to save.
        - Steps where `(seq_len - 1) % compress_ratio != 0`: no new compressed
          token was produced this step.

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        map_last_loc_to_buffer (本文件)
          → _eager_backup_previous_token(...)  ← 你在这
            → mem_pool_host.alloc_paged_token_slots   ← 在 Host 分配 1 个 slot
            → wait_for_pending_backup()              ← 等上一步 decode_backup DMA 完成
            → decode_backup_stream.launch backup_from_device_all_layer  ← 异步 H→D backup

        ⚙️ 工作原理:
          一个 compressed token = compress_ratio 个 decode step 累积产生。
          本函数在"刚好产生新 compressed token"的 step 把上一个 compressed token
          backup 到 Host。设计成增量 backup 而非全量, 是因为全量 backup 每步都做
          开销过大 —— 只在恰好产生新 token 的 step 做一次即可。

        ⚠️ 双流异步:
          - 主流 (forward) 写 KV → decode_backup_stream 读 KV
          - 用 _backup_done_event 跨流同步: 下一次 forward 前 wait_for_pending_backup
            阻塞主流, 直到 backup DMA 完成
        """
        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up),
        # and skip non-aligned steps that did not produce a new compressed token.
        # ① 筛选需要 backup 的 req:
        #    - 跳过 staging 后首个 decode step (prefill 已 backup)
        #    - 跳过未对齐的 step ((seq_len-1) % ratio != 0 表示未产生新 compressed token)
        backup_indices = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                # staging 刚完成的第一个 step: prefill KV 已 backup, 跳过并清标记
                self._skip_first_backup[req_idx] = False
                continue
            if (int(seq_lens_cpu[i]) - 1) % self.compress_ratio == 0:
                backup_indices.append(i)

        if not backup_indices:
            return

        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device=self.device
        )
        backup_req_indices = req_pool_indices[backup_indices_gpu]

        # The previous compressed token's position and its device buffer slot:
        #  compressed_pos = (seq_len - 1) // compress_ratio - 1
        #  - short: slot = compressed_pos          (within the regular buffer)
        #  - long:  slot = device_buffer_size      (the reserved slot)
        # ② 算"上一个 compressed token"的 buffer slot 位置
        #    short 序列: slot = compressed_pos (常规 buffer 区)
        #    long 序列:  slot = device_buffer_size (reserved slot, 最新 token)
        prev_seq_lens = seq_lens[backup_indices_gpu] - 1
        compressed_prev_seq_lens = prev_seq_lens // self.compress_ratio
        actual_compressed_pos = compressed_prev_seq_lens - 1

        # clamp: long 序列时统一指向 reserved slot
        buffer_slot = actual_compressed_pos.clamp(max=self.device_buffer_size)

        # 取出每个 req 的 Device buffer slot (KV 在 Device 上的位置)
        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]

        # ③ 在 Host pool 给每个 req 分配 1 个 slot (增量 alloc, 写入 req_to_host_pool)
        host_locs_list = []
        for i in backup_indices:
            req_idx = int(req_pool_indices_cpu[i])
            start_pos = (int(seq_lens_cpu[i]) - 1) // self.compress_ratio - 1
            host_locs = self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                req_idx,
                start_pos,
                1,
            )
            host_locs_list.append(host_locs)
        host_locs = torch.cat(host_locs_list)

        # ④ 异步 DMA: 先等上一步 backup 完成, 再等主流 (forward) 写完 KV, 然后 launch
        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            # 跨流同步: backup_stream 等 schedule_stream (主流) 把新 KV 写完
            self.decode_backup_stream.wait_stream(schedule_stream)
            # decode_producer_stream 是 forward 主流 (set_decode_producer_stream 注入)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            # 真正的 D→H backup, 全 layer 一次性传
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            # 记录完成事件, 下一次 wait_for_pending_backup 用它同步
            self._backup_done_event.record()
            # 索引张量归属 backup_stream, 防止主流提前释放
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
            if actual_compressed_pos.is_cuda:
                actual_compressed_pos.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        # 标记有飞行中的 backup, 下次 wait_for_pending_backup 时阻塞等待
        self._has_pending_backup = True

    def wait_for_pending_backup(self) -> None:
        """⏳ 阻塞当前流, 等待上一步 decode backup DMA 完成。

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        ① ModelRunner.forward (model_runner.py:3620) — decode 前调
            → wait_for_pending_backup()  ← 你在这
            (保证 forward 读 KV 前, 上一步 backup DMA 已经写完 Host)
        ② _eager_backup_previous_token (本文件) — launch 新 backup 前调
            → wait_for_pending_backup()
            (保证不与上一步 backup DMA 在 decode_backup_stream 上冲突)

        ⚙️ 行为:
            若 _has_pending_backup=True, 让当前流 wait _backup_done_event
            (event 由 decode_backup_stream record, 当前流 wait 后会同步)
            完成后清标记位。
        """
        if not self._has_pending_backup:
            return
        self._backup_done_event.wait(device_module.current_stream())
        self._has_pending_backup = False

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA kernel) instead.

        Note: dsv4 hisparse is not supported — DeepSeekV4SingleKVPoolHost has no
        load_to_device_per_layer and indices live in compressed space. Currently
        only used as a kernel oracle in test_hisparse_unit.py (non-dsv4 path).

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        仅 test_hisparse_unit.py 调用 (作 swap_in_selected_pages 的 oracle 对照)
          → naive_load_topk(...)              ← 你在这
          → 逐 req + 逐 token 串行 load_to_device_per_layer (慢但可靠)

        ⚙️ 与 swap_in_selected_pages 的差异:
            本函数: Python 双重循环 + per-layer API, 串行执行
            生产路径: JIT fused kernel, 一次 batch 处理所有 req + 所有 top_k token
            语义等价 → 可作 oracle 验证 JIT kernel 正确性

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
        """
        # 仅 DSA 路径可用 — dsv4 的 Host pool 没有 load_to_device_per_layer
        assert (
            not self.is_dsv4_hisparse
        ), "naive_load_topk is not implemented for dsv4 hisparse"
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        # 逐 req 处理 (与 JIT kernel 的 batch 化路径形成对比)
        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)

            # 越界检查: top-k 选中位置不能为负 / 不能 ≥ seq_len
            assert torch.all(
                selected_tokens >= 0
            ), f"Req {req_idx}: selected tokens contain negative positions"
            assert torch.all(selected_tokens < seq_len), (
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:
                # 短序列 fast-path: token 已在 device_buffer 中, 直接索引
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                # 长序列: 需要从 Host 拉到 Device (swap-in)
                device_indices = torch.empty(
                    top_n, dtype=torch.int64, device=self.device
                )

                # 最新 token: 已在 reserved slot (map_last_loc_to_buffer 写入)
                is_latest_token = selected_tokens == (seq_len - 1)
                needs_host_load = ~is_latest_token

                # 最新 token 直接从 reserved slot 取
                device_indices[is_latest_token] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]

                # 非最新 token: 需要从 Host load 到 device buffer
                num_to_load = int(needs_host_load.sum().item())
                if num_to_load > 0:
                    tokens_to_load = selected_tokens[needs_host_load]
                    # req_to_host_pool 给出每个 token 在 Host 的 slot
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]

                    # 健全性: 每个 token 必须有 host backup, 否则 swap-in 无源可拉
                    invalid_mask = host_locs < 0
                    if torch.any(invalid_mask):
                        bad_positions = tokens_to_load[invalid_mask].tolist()
                        raise AssertionError(
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    # 借用 device buffer 的前 num_to_load 个 slot 作 swap-in 落点
                    # (JIT kernel 用 LRU 选 slot, 这里简化为顺序借用)
                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]
                    device_indices[needs_host_load] = buffer_locs

                    # 逐层 load (生产路径的 swap_in_selected_pages 也是逐层调用,
                    # 但通过 JIT kernel 一次处理整个 batch 的所有 token)
                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)

        return top_k_indices

    def abort_staging_request(self, req: Req) -> None:
        """Remove a request from the staging queue and free its host + device resources.

        Must be called when aborting a request that has been admitted into staging
        but has not yet completed (i.e. req.hisparse_staging is True).

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        release_kv_cache (schedule_batch.py:1684)
          → retract_req(req)                    ← 上层统一入口
            → if req.hisparse_staging:
                → abort_staging_request(req)   ← 你在这

        ⚙️ 步骤:
          ① 从 ack_staging_queue 移除该 req 的票据
          ② write_staging_stream.synchronize() — 等 staging DMA 真正停止 (防 UAF)
          ③ free_hisparse(prefill_locs) — 释放 Device 上已分配的 compressed KV
          ④ mem_pool_host.free(host_indices) — 释放 Host 上已分配的 slot
          ⑤ 清空 req_to_host_pool / allocated_len / _skip_first_backup
          ⑥ req.hisparse_staging = False
        """
        # ① 移除票据
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        # ② 等 staging 流停止 — 否则 free 后 DMA 仍在写, 触发 UAF
        self.write_staging_stream.synchronize()

        # ③ 释放 Device 上已分配的 compressed KV (prefill_len 个 token)
        prefill_len = req.fill_len
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefill_len
        ]
        self.token_to_kv_pool_allocator.free_hisparse(allocated_locs)

        # Free host memory that was allocated during admit_request_into_staging
        # ④ 释放 Host 上已分配的 slot
        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        # ⑤ 清空 req 级表 (复位到初始状态)
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        self._skip_first_backup[req.req_pool_idx] = False
        # ⑥ 清标记位
        req.hisparse_staging = False

    def retract_req(self, req: Req) -> None:
        """🚪 req 中途退场统一入口 —— 按状态分发到 abort / finished。

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        release_kv_cache (schedule_batch.py:1684)
          → retract_req(req)       ← 你在这
            ├─ hisparse_staging=True → abort_staging_request(req)
            └─ hisparse_staging=False → request_finished(req)

        ⚙️ 分发依据: req.hisparse_staging
            True  — 还在 staging 队列, DMA 可能未完成 → abort (等流 + 释放)
            False — 已 ready 或已在 decode → request_finished (走常规清理)
        """
        if req.hisparse_staging:
            self.abort_staging_request(req)
        else:
            self.request_finished(req)

    def request_finished(self, req: Req):
        """🧹 req 完成时的资源清理 —— 释放 Device buffer + Host pool + 清所有映射表。

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        ① BatchResultProcessor (batch_result_processor.py:97 / 882)
            — req 正常完成或 abort 完成后调
            → request_finished(req)         ← 你在这
        ② disagg/decode.py (1673 / 1697)
            — disagg decode req 结束时调

        ⚙️ 步骤:
          ① 等待 decode_producer_stream + pending backup 完成
             (deocde 可能还有飞行中的 DMA, 必须先 sync 再 free)
          ② 释放 Device buffer slot (取 req_to_device_buffer 中已分配的部分)
          ③ 复位 full_to_hisparse_device_index_mapping (清除全局反查)
          ④ 释放 Host pool slot (allocated_host_indices → free)
          ⑤ 清空所有 req 级表 + lru_slots 复位

        ⚠️ 用 kv_allocated_len 而非 seq_len:
            speculative decoding 下 allocator 会过分配, 这些 slot 可能带着 stale
            mapping 指向 buffer, 若不清会被随后的 release_kv_cache 二次 free
            (double-free into page allocator's free list)。
        """
        # release resources only after the execution of a potential overlapped batch
        # ① 等待飞行中的 DMA 完成
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()

        # Use kv_allocated_len (not seqlen): under speculative decoding the
        # allocator can over-allocate beyond the committed seqlen, and those
        # extra slots may carry stale mapping entries pointing at buffer slots
        # we just freed via free_hisparse_indices(all_hi). If left set, the
        # subsequent release_kv_cache -> allocator.free -> free_hisparse path
        # re-frees them (double-free into the page allocator's free list).
        allocated_len = req.kv_allocated_len

        # release memory -- only free actually-allocated buffer indices
        # ② 释放 Device buffer slot: 取 req_to_device_buffer 中已分配的部分 (>0 的)
        # 用 unique 去重 (避免 speculative 多次分配同一 slot 导致重复)
        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        if current_cap > 0:
            side_buf_hi = self.req_to_device_buffer[req.req_pool_idx, :current_cap]
            all_hi = torch.unique(side_buf_hi[side_buf_hi > 0])
            if all_hi.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(all_hi)

        # ③ 复位全局反查表: full_kv_idx → Device buffer slot 映射清零
        #    (不清零的话, swap-in kernel 可能命中已释放的 slot)
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ]
        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            allocated_locs
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = 0

        # ④ 释放 Host pool slot
        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        # clear req info
        # ⑤ 清空所有 req 级表 (复位到初始状态, 供下一次复用 req_pool_idx)
        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        # lru_slots 复位为 [0,1,...,buf_size-1] 初始序
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req.req_pool_idx] = False

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices.

        ━━━━━━━━━━━━━━ 调用链 ━━━━━━━━━━━━━━
        Attention 执行中 (decode step):
          ① DSA 路径: dsa_backend.py:1641
             → swap_in_selected_pages(...)    ← 你在这
          ② dsv4 路径: dsv4/indexer.py:619
             → swap_in_selected_pages(...)
          → JIT kernel: load_cache_to_device_buffer_dsv4_mla / _mla
            (sgl_kernel, 一次 batch 处理所有 req + 所有 top-k token)

        ⚙️ 行为:
          ① 复用预分配的 top_k_device_locs_buffer, fill -1
          ② 选 JIT kernel 入口 (按 is_dsv4_hisparse 选 dsv4_mla / mla)
          ③ 调用 JIT kernel: 输入 top_k_result (token 位置) + 三张映射表 +
              host/device KV buffer, 输出每个 req 的 top-k Device KV 索引
          ④ 返回 top_k_indices (切片到 num_reqs, 含 padding 时填 -1)

        📤 返回: (num_reqs, top_k) 的 Device KV 索引张量 (-1=无效)
        """
        num_reqs = req_pool_indices.size(0)

        # ① 复用预分配 buffer, 避免每次 forward 分配 (CUDA graph 友好)
        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]
        top_k_indices.fill_(-1)

        # todo, adjustable for performance
        block_size = 1024
        # ② 按 is_dsv4_hisparse 选 JIT kernel 入口
        #    dsv4: 压缩 KV + 不同 indexing 语义
        #    DSA:  未压缩 MLA KV
        swap_in_fn = (
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
        )
        # ③ 调用 JIT fused kernel —— 一次处理整个 batch
        #    关键输入:
        #      top_k_tokens           - 每个 req 选中的 token 序号
        #      device_buffer_tokens   - 每个 slot cache 的 token 序号 (查命中)
        #      host_cache_locs        - req_to_host_pool, miss 时从这里查 Host slot
        #      device_buffer_locs    - token 序号 → Device KV 索引 (命中时直取)
        #      host_cache/device_buffer - 实际 KV 数据 buffer (按 layer_id 取一层)
        #      lru_slots              - LRU 替换决策
        #      num_real_reqs          - CUDA graph 下让 padding 位早退
        #    输出: top_k_indices (req × top_k, 每个 token 的 Device KV 索引)
        swap_in_fn(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=1,
            block_size=block_size,
            num_real_reqs=self.num_real_reqs,
        )
        return top_k_indices
