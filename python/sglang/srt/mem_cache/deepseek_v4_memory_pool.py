from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Literal, NamedTuple, Optional, Tuple

import torch

from sglang.jit_kernel.dsv4 import fused_k_norm_rope_flashmla, fused_store_cache
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import index_buf_accessor
from sglang.srt.layers.attention.dsv4 import (
    index_buf_accessor as dsv4_index_buf_accessor,
)
from sglang.srt.layers.attention.dsv4.index_buf_accessor import NopeFp8RopeBf16Pack
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import ceil_div, is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()

ONLINE_C128 = not _is_hip and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()


def get_compress_state_ring_size(
    compress_ratio: int, is_speculative: bool = False
) -> int:
    assert compress_ratio in [4, 128], f"Unsupported {compress_ratio = }"
    # Online c128 keeps a single (max, sum, kv) state per index instead of a
    # 128-slot ring buffer of raw tokens, so ring_size collapses to 1. Online
    # is incompatible with speculative decode for now.
    if compress_ratio == 128 and ONLINE_C128:
        if is_speculative and not envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
            raise AssertionError("online c128 does not support MTP")
        return 1
    if is_speculative:
        return 16 if compress_ratio == 4 else 256
    else:
        return 8 if compress_ratio == 4 else 128


class DeepSeekV4SingleKVPool(KVCache):
    """💾 DeepSeek-V4 单层 KV 池 —— SWA / C4 / C128 压缩层共用的"按页 packed 字节"KV 缓存。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧬 DSV4 单 token 字节布局（get_bytes_per_token = 584，固定）                            ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  ┌── nope 段(FP8, 1B/值) ──┬── rope 段(BF16, 2B/值)──┬─ nope scales ─┬ pad ─┐    ║
    ║  │ qk_nope_head_dim = 448B │ qk_rope_head_dim*2=128B │ 448//64 = 7B  │  1B │    ║
    ║  │ (压缩潜变量 c_kv，FP8)  │ (解耦 RoPE 位置，BF16)  │ (UE8M0 指数) │     │    ║
    ║  └─────────────────────────┴─────────────────────────┴───────────────┴─────┘    ║
    ║   合计 = 448 + 128 + 7 + 1 = 584 字节/token                                              ║
    ║  • scales：每 64 个 nope 值共享 1 个 UE8M0 指数 scale(quantize_block_size=64)            ║
    ║  • scale_pad=1：把 scale 段(7B)补到 8B 边界，便于内核向量化读取                           ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧬 buffer 形状：按"页"分配，而非按 token（与 MLA/DSA 的 [tokens, dim] 不同）            ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  kv_buffer[layer] : [num_pages, bytes_per_page_padded] uint8                            ║
    ║    • num_pages = (size + page_size + 1) // page_size  (+1 容纳哨兵页)                    ║
    ║    • bytes_per_page_padded = ceil(page_size * 584, 576) * 576  (按 576 对齐，flashmla 要求)║
    ║  store_dtype = uint8(dtype 为 FP8 时由基类推断)；整块当裸字节解释，nope/rope/scale       ║
    ║  按 offset 切片定位，故写入/读取都靠专用内核按字节散点搬运，而非 PyTorch 索引。               ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔑 与 MLA / DSA KV 池的关键区别                                                       ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  ┌────────────┬──────────────────────────┬────────────────────────────────────┐   ║
    ║  │ 维度        │ MLA/DSA 池                │ DeepSeekV4SingleKVPool(本类)       │   ║
    ║  ├────────────┼──────────────────────────┼────────────────────────────────────┤   ║
    ║  │ buffer 形状 │ [tokens, head, dim]      │ [num_pages, bytes_per_page] uint8 │   ║
    ║  │ 索引粒度     │ 按 token(slot)           │ 按 page(页粒度，page_size token/页) │   ║
    ║  │ 单 token   │ 单一 dtype 等宽          │ 混合：nope_fp8 + rope_bf16 + scale │   ║
    ║  │ V 缓冲      │ 有(或与 K 共享 latent)   │ 无独立 V(latent 风格，V 折叠进 K)  │   ║
    ║  │ 写入接口    │ set_mla_kv_buffer/set_kv │ set_key_buffer / set_key_buffer_fused│  ║
    ║  │ 读写内核    │ triton mla_buffer.py     │ dsv4 SetKAndS.triton / JIT fused_store│  ║
    ║  └────────────┴──────────────────────────┴────────────────────────────────────┘   ║
    ║  🎯 适用场景：DeepSeek-V4 模型；每层按压缩比(ratio 0/4/128)归属 SWA/C4/C128 之一，       ║
    ║     由 DeepSeekV4TokenToKVPool 持有 3 个本类实例(swa/c4/c128)，见 :568/:582/:593。        ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧩 对外接口（框架通过这些方法使用本类）                                                  ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  get_bytes_per_token()              🧬 单 token 字节数(584)，供 create_buffer 算页宽  ║
    ║  create_buffer(num_pages)            💾 按 576 对齐分配一层 uint8 buffer             ║
    ║  set_key_buffer(layer,loc,pack)      ✍️ 分段写：FP8 nope+BF16 rope+scale 散点写页    ║
    ║  set_key_buffer_fused(layer,loc,k)   ✍️ 融合写：预 packed cache_k 整页写入           ║
    ║  get_key_buffer(layer_id)           📖 读整层 buffer(fp8 时零拷贝 view(dtype))       ║
    ║  set_kv_buffer/get_value_buffer/get_kv_buffer  🚫 NotImplementedError(无独立 V)       ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔗 调用链（写 KV：两条路径，由不同 attention 路径触发）                                  ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  路径 A（分段写，compressor 已分别量化好 FP8/BF16/scale）：                              ║
    ║    dsv4/compressor.py:192 / compressor_v2.py:663 (DSV4 attention 压缩器)               ║
    ║      └─ DeepSeekV4TokenToKVPool.set_extra_key_buffer(:882) / set_swa_key_buffer(:861)  ║
    ║         └─ DeepSeekV4SingleKVPool.set_key_buffer  ← 当前类✍️                              ║
    ║            └─ dsv4_index_buf_accessor.SetKAndS.triton  (按字节 offset 散点写页)         ║
    ║  路径 B（融合写，上游已把整页 cache_k 预 packed）：                                     ║
    ║    deepseek_v4_backend.py:1299 / deepseek_v4_backend_hip_radix.py:1292 (radix/MQA 后端)║
    ║      └─ ...set_swa_key_buffer_radix_fused(:955) / set_extra_key_buffer_fused(:986)     ║
    ║         └─ DeepSeekV4SingleKVPool.set_key_buffer_fused  ← 当前类✍️                     ║
    ║            └─ fused_store_cache(type="flashmla")  (JIT CUDA 内核 / HIP 走 triton)      ║
    ║  读 KV 📖：attention backend                                                              ║
    ║    └─ DeepSeekV4TokenToKVPool.get_swa_key_buffer(:857) / get_extra_key_buffer(:876)     ║
    ║       └─ DeepSeekV4SingleKVPool.get_key_buffer  ← 当前类📖                              ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ⚠️ 注意（仅注释，未改代码）：
      1. get_key_buffer 的 fp8 分支用 [layer_id - start_layer]，而 set_key_buffer /
         set_key_buffer_fused / 以及 get_key_buffer 的非 fp8 分支都用裸 [layer_id]。
         实际不冲突：DeepSeekV4TokenToKVPool 创建本类时不传 start_layer(默认 0)，且调用前
         已用 _swa_local_layer_id / layer_mapping 换成本地层号，故 start_layer 恒为 0。
      2. set_kv_buffer / get_value_buffer / get_kv_buffer 显式 NotImplementedError：
         DSV4 走 MLA latent 路径，V 折叠进 K（无独立 V buffer），统一用 get_key_buffer。
      3. bytes_per_token 固定 584(nope=448/rope=128/scale=7/pad=1)，create_buffer 中有
         assert 锁死该布局；若改 head_dim / block_size 需同步更新 assert 与布局说明。
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        """🧱 构造单层 KV 池：记录 nope/rope 维度与量化布局参数，再委托 _create_buffers 申请显存。

        🔗 调用链定位(① 创建链)：
            DeepSeekV4TokenToKVPool.__init__  →  本类(size, page_size, dtype, ...)
            (见 :568/:582/:593 的 3 处实例化：swa / c4 / c128)

        📥 参数：
            size, page_size : 容量(token 数) 与页大小；buffer 实际按页分配(见 _create_buffers)
            dtype           : FP8(float8_e4m3fn)——使基类把 store_dtype 置为 uint8(裸字节存储)
            qk_nope_head_dim: nope 压缩潜变量维度(=448，FP8 存储)
            qk_rope_head_dim: rope 解耦位置维度(=64，BF16 存储)
            layer_num       : 本池负责的层数(swa/c4/c128 各自的层数，按 stage_ratios 统计)
        ⚙️ 布局参数(scale_pad/quantize_block_size/rope_storage_dtype)详见类 docstring 字节布局图。
        """
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        # nope / rope 两段维度：nope 走 FP8，rope 走 BF16(见 get_bytes_per_token 的字节计算)。
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim

        # scale_pad=1：把 nope 的 scale 段(7B)补到 8B，便于内核向量化读取。
        self.scale_pad = 1
        # quantize_block_size=64：每 64 个 nope 值共享 1 个 UE8M0 指数 scale(448//64=7 个 scale)。
        self.quantize_block_size = 64
        # rope 以 BF16 存储(itemsize=2)，故 rope 段字节 = 64*2 = 128。
        self.rope_storage_dtype = torch.bfloat16
        # k_with_scale_buffer_dtype=int8：标记"数据+scale"打包 buffer 的语义 dtype(供内核识别)。
        self.k_with_scale_buffer_dtype = torch.int8
        self._create_buffers()

    def _create_buffers(self):
        """💾 为每层分配一块按页对齐的 uint8 buffer（委托 create_buffer 计算页宽）。

        🔗 调用链定位(① 创建链)：__init__ → _create_buffers → create_buffer。
        ⚙️ 行为：每层一个 [num_pages, bytes_per_page_padded] uint8 张量；
            num_pages = (size + page_size + 1)//page_size，+1 容纳哨兵页(承接 padding/越界)。
        """
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                # custom_mem_pool：PD 分离 + NVLink 场景下用专用内存池，使 buffer 落在
                # 可被 RDMA/NVLink 直接访问的地址空间(详见基类 _create_buffers 注释)。
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.kv_buffer = [
                    self.create_buffer(
                        num_pages=(self.size + self.page_size + 1) // self.page_size,
                    )
                    for _ in range(self.layer_num)
                ]

    def get_bytes_per_token(self) -> int:
        """🧬 计算单个 token 在 page buffer 中占用的字节数（固定 584）。

        🔗 调用链定位：create_buffer → get_bytes_per_token(决定页宽)。
        📤 返回：nope(FP8) + rope(BF16) + nope 的 UE8M0 scale + scale_pad。
        ⚙️ 拆解：448(nope,1B/值) + 64*2(rope,2B/值) + 448//64(7 个 scale) + 1(pad) = 584。
        """
        dim_per_token = (
            # nope 段：FP8，每值 1 字节 → qk_nope_head_dim 字节(448)。
            self.qk_nope_head_dim
            # rope 段：BF16，每值 2 字节 → qk_rope_head_dim * 2(128)。
            + self.qk_rope_head_dim * self.rope_storage_dtype.itemsize
            # nope 的逐块 scale：每 quantize_block_size(64) 个值 1 个 UE8M0(1B) → 448//64 = 7。
            + self.qk_nope_head_dim // self.quantize_block_size
            # + scale_pad(1)：把 7B scale 补到 8B 边界。
            + self.scale_pad
        )
        return dim_per_token

    def create_buffer(self, *, num_pages: int):
        """💾 分配一层 page buffer：按 576 字节对齐页宽后建 uint8 张量。

        🔗 调用链定位：_create_buffers → create_buffer(逐层)。
        📥 参数：num_pages : 页数(已含哨兵页)。
        📤 返回：[num_pages, bytes_per_page_padded] uint8 张量(device 上)。
        ⚙️ 行为：页宽 = page_size * 584，再向上对齐到 576 的倍数(flashmla 内核对页宽对齐要求)。
        ⚠️ assert 锁死 584 布局与 store_dtype==uint8(dtype 必须是 FP8)。
        """
        bytes_per_token = self.get_bytes_per_token()
        # kv_cache_total_dim：对外暴露"单 token 字节数"，供 backend / 日志引用。
        self.kv_cache_total_dim = bytes_per_token
        bytes_per_page_non_padded = self.page_size * bytes_per_token
        # 按 576 对齐(576=64*9，flashmla paged-KV 内核的页宽对齐要求)，尾部 padding 字节不存数据。
        self.bytes_per_page_padded = ceil_div(bytes_per_page_non_padded, 576) * 576

        assert bytes_per_token == 448 + 64 * 2 + 8, (
            "DSV4 KV layout: qk_nope_head_dim FP8 (448) + qk_rope_head_dim BF16 "
            "(64*2) + nope FP8 scales + scale_pad = 584 bytes/token"
        )
        # dtype 必须是 FP8 → 基类已把 store_dtype 置为 uint8；buffer 整块当裸字节解释。
        assert self.store_dtype == torch.uint8

        return torch.zeros(
            num_pages,
            self.bytes_per_page_padded,
            dtype=self.store_dtype,
            device=self.device,
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ):
        """✍️ 分段写路径——上游已分别量化好 FP8 nope / BF16 rope / UE8M0 scale，按字节散点写页。

        🔗 调用链定位(③ 前向写 KV，路径 A：compressor 分段量化)：
            dsv4/compressor.py:192 / compressor_v2.py:663
              └─ DeepSeekV4TokenToKVPool.set_extra_key_buffer(:882) / set_swa_key_buffer(:861)
                 └─ set_key_buffer(layer_id, loc, pack)  ← 当前函数
                    └─ dsv4_index_buf_accessor.SetKAndS.triton  (按字节 offset 写页)

        📥 参数：
            layer_id        : 本地层号(调用方已用 _swa_local_layer_id / layer_mapping 换算)
            loc             : 写入目标页索引(= forward_batch 的 out_cache_loc，页粒度)
            cache_nope_fp8_rope_bf16_pack : 已量化打包的 (k_nope_fp8, k_rope_bf16, scale_ue8m0)
        📤 返回：无(原地写入 self.kv_buffer[layer_id])。
        ⚙️ 行为：交给 triton 内核，按 page_size 把三段写入 page buffer 的对应字节区间。
        ⚠️ 这里用裸 [layer_id] 而非 [layer_id - start_layer]；因本池构造时 start_layer=0(见类⚠️1)。
        """
        dsv4_index_buf_accessor.SetKAndS.execute(
            pool=self,
            buf=self.kv_buffer[layer_id],
            loc=loc,
            nope_fp8_rope_bf16_pack=cache_nope_fp8_rope_bf16_pack,
        )

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        """✍️ 融合写路径——上游已把整页 cache_k 预 packed 成 flashmla 布局，直接整页写入。

        🔗 调用链定位(③ 前向写 KV，路径 B：radix/MQA 后端融合算子)：
            deepseek_v4_backend.py:1299 / deepseek_v4_backend_hip_radix.py:1292
              └─ ...set_swa_key_buffer_radix_fused(:955) / set_extra_key_buffer_fused(:986)
                 └─ set_key_buffer_fused(layer_id, loc, cache_k)  ← 当前函数
                    └─ fused_store_cache(type="flashmla")  (JIT CUDA 内核 / HIP 走 triton)

        📥 参数：
            layer_id : 本地层号(同 set_key_buffer)
            loc      : 写入目标页索引(页粒度)
            cache_k  : 预 packed 的整页 KV(flashmla 布局，含 nope+rope+scale)
        📤 返回：无(原地写入 self.kv_buffer[layer_id])。
        ⚙️ 行为：fused_store_cache 做 paged 散点存储；CUDA 用 JIT 编译内核，HIP 回退 triton。
        ⚠️ 与 set_key_buffer 的区别：本路径上游已 packed，省去内核内分段拼装；type="flashmla"
           区别于 indexer 的 "indexer" 类型(见 DeepSeekV4IndexerPool.set_index_fused)。
        """
        return fused_store_cache(
            input=cache_k,
            cache=self.kv_buffer[layer_id],
            indices=loc,
            page_size=self.page_size,
            type="flashmla",
        )

    def get_key_buffer(self, layer_id: int):
        """📖 读取整层 page buffer（FP8 存储时零拷贝 view 回 dtype）。

        🔗 调用链定位(④ 前向读 KV)：
            attention backend → DeepSeekV4TokenToKVPool.get_swa_key_buffer(:857)
              / get_extra_key_buffer(:876) → get_key_buffer(layer_id)  ← 当前函数

        📥 参数：layer_id : 本地层号。
        📤 返回：整层 [num_pages, bytes_per_page_padded] 的 buffer(fp8 时 view 成 dtype)。
        ⚠️ fp8 分支用 [layer_id - start_layer]，非 fp8 分支用裸 [layer_id]——因 start_layer=0
           而不冲突(见类⚠️1)；非 fp8 路径实际未用(DSV4 dtype 恒为 FP8)。
        """
        if self.store_dtype != self.dtype:
            # FP8 主路径：buffer 实际 uint8 存储，按真实 dtype 零拷贝重解释后返回。
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.kv_buffer[layer_id]

    def set_kv_buffer(self, *args, **kwargs) -> None:
        """🚫 未实现——DSV4 用 set_key_buffer / set_key_buffer_fused 写 KV，不走通用 K+V 接口。"""
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """🚫 未实现——DSV4 无独立 V(MLA latent 风格，V 折叠进 K)，请用 get_key_buffer。"""
        raise NotImplementedError("Use get_key_buffer instead.")

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """🚫 未实现——同 get_value_buffer，DSV4 无独立 V，请用 get_key_buffer。"""
        raise NotImplementedError("Use get_key_buffer instead.")


class HiSparseC4DevicePool(DeepSeekV4SingleKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.compress_ratio = 4

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):
        mask = (full_indices + 1) % self.compress_ratio == 0
        compressed_indices = full_indices[mask] // self.compress_ratio
        return compressed_indices

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to(
            torch.int32
        )

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self._translate_loc_to_hisparse_device(
            self.translate_loc_from_full_to_compressed(full_indices)
        )

    def set_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)

    def set_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().set_key_buffer_fused(layer_id, loc, cache_k)

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseC4DevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseC4DevicePool does not support load_cpu_copy")


class DeepSeekV4IndexerPool(KVCache):
    """🔍 DeepSeek-V4 Indexer KV 池 —— DSV4 稀疏注意力中"选 token"的 MQA 索引键缓存。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧬 Indexer 是什么：稀疏注意力的"预筛选"键                                            ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  DSV4 稀疏注意力两阶段：                                                                ║
    ║    ① Indexer(MQA, 1 head, head_dim=128) 用低维 K 给所有历史 token 打分 → 选 topk       ║
    ║    ② 主 MLA attention 只对 topk 个 token 算 → 复杂度 O(seq) → O(topk)                 ║
    ║  本池专门存 ① 的 indexer K：低维(128)、单头、按 token 配 1 个 fp32 scale。              ║
    ║  ⚠️ 仅 C4 压缩层(ratio=4)有 indexer（见调用方 assert compress_ratio == 4）。            ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧬 单 token 字节布局（quant_block_size=128=index_head_dim → 每 token 恰 1 个 scale） ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  ┌── index K 数据 ──┬─ scale(fp32, 4B)─┐    use_fp4_indexer=False: 128 + 4 = 132 B  ║
    ║  │ FP8: index_head_dim│ 1 个/token     │     use_fp4_indexer=True : 128//2 + 4 = 68 B║
    ║  │  =128B(1B/值)     │                 │     (FP4: 2 值/字节 → 64B 数据)             ║
    ║  └───────────────────┴─────────────────┘                                               ║
    ║  buffer: index_k_with_scale_buffer[layer] = [num_pages, page_bytes] uint8               ║
    ║    • num_pages = (size + page_size + 1) // page_size  (+1 哨兵页)                        ║
    ║    • page_bytes = page_size * bytes_per_token  (注意：未按 576 对齐，与 SingleKVPool 不同)║
    ║  store_dtype = uint8(dtype 为 FP8 时基类推断)；index_k_with_scale_buffer_dtype = uint8。 ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧩 对外接口（框架通过这些方法使用本类）                                                  ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  get_bytes_per_token()               🧬 单 token 字节数(132 或 68)                      ║
    ║  get_index_k_with_scale_buffer(id)   📖 取整层 indexer buffer(自行 gather)              ║
    ║  get_index_k_scale_buffer(id,seq,pi) 📖 融合读 (k, scale)：按 page_indices paged gather   ║
    ║  set_index_k_scale_buffer(id,loc,k,s) ✍️ 分段写：data+scale 分开，经 SetKAndS.triton      ║
    ║  set_index_fused(id,loc,k)           ✍️ 融合写：预 packed cache_k 经 fused_store_cache    ║
    ║  set_index_fp4(id,loc,k)            ✍️ FP4 写：量化成 fp4+scale 经专用内核               ║
    ║  get_kv_buffer/get_key_buffer/get_value_buffer/set_kv_buffer  🚫 NotImplementedError    ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔗 调用链（经 DeepSeekV4TokenToKVPool 中转，仅 c4 层可达）                              ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  ③ 写 indexer K（三选一，由 compressor 按算子可用性择优）：                             ║
    ║    dsv4/compressor.py:217 / compressor_v2.py:641,655                                    ║
    ║      └─ DeepSeekV4TokenToKVPool.set_index_k_fused(:1173) → set_index_fused ✍️(融合)    ║
    ║         └─ fused_store_cache(type="indexer")  (JIT CUDA / HIP triton)                  ║
    ║      └─ ...set_index_k_scale_buffer(:1093) → set_index_k_scale_buffer ✍️(分段, 回退)   ║
    ║         └─ index_buf_accessor.SetKAndS.triton                                          ║
    ║      └─ ...set_index_k_fp4(:1183) → set_index_fp4 ✍️(FP4, use_fp4_indexer 时)         ║
    ║         └─ store_fp4_index_k_cache  (fp4 量化 + 专用内核)                              ║
    ║  ④ 读 indexer K（两选一）：                                                              ║
    ║    dsv4/indexer.py:385,428 / compressor_v2.py:506                                       ║
    ║      └─ ...get_index_k_with_scale_buffer(:1074) → get_index_k_with_scale_buffer 📖(整层)║
    ║      └─ ...get_index_k_scale_buffer(:1080) → get_index_k_scale_buffer 📖(融合 gather)   ║
    ║         └─ index_buf_accessor.GetKAndS.execute  (aiter preshuffle / triton)           ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ⚠️ 注意（仅注释，未改代码）：
      1. start_layer 索引不一致：读路径 get_index_k_with_scale_buffer / get_index_k_scale_buffer
         用裸 [layer_id]，写路径 set_index_k_scale_buffer / set_index_fused / set_index_fp4 用
         [layer_id - start_layer]。因 DeepSeekV4TokenToKVPool 创建本池时不传 start_layer(默认 0)、
         且调用前已用 layer_mapping 换成本地层号，故 start_layer 恒为 0，实际不冲突。
      2. 仅 C4 层(ratio=4)有 indexer：调用方(DeepSeekV4TokenToKVPool)对每个方法都 assert
         compress_ratio == 4，C128/SWA 层不会路由到本池。
      3. use_fp4_indexer 由 server_args.enable_deepseek_v4_fp4_indexer 全局开关控制，
         决定 get_bytes_per_token 与写路径(set_index_fp4 vs set_index_fused)的选择。
    """
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        """🧱 构造 indexer KV 池：记录 head_dim 与 FP4 开关，再委托 _create_buffer 申请显存。

        🔗 调用链定位(① 创建链)：
            DeepSeekV4TokenToKVPool.__init__  →  本类(:786, 作为 c4_indexer_kv_pool)

        📥 参数：
            size, page_size : 容量(token 数) 与页大小(buffer 按页分配)
            dtype           : FP8(float8_e4m3fn)——使基类把 store_dtype 置为 uint8
            index_head_dim  : indexer K 的单头维度(=128，= quant_block_size → 每 token 1 个 scale)
            layer_num       : 本池负责的 c4 层数
        ⚙️ use_fp4_indexer：读全局开关，决定字节布局(132 vs 68)与写路径。
        """
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        # indexer K 的单头维度(128)；与 quant_block_size 相等 → 每 token 恰 1 个 fp32 scale。
        self.index_head_dim = index_head_dim
        # FP4 indexer 开关：enable_deepseek_v4_fp4_indexer。开则数据按 FP4(2值/字节)存。
        self.use_fp4_indexer = get_global_server_args().enable_deepseek_v4_fp4_indexer

        self._create_buffer()

    def get_bytes_per_token(self) -> int:
        """🧬 单 token 字节数：index K 数据 + 1 个 fp32 scale(4B)。

        📤 返回：
            FP4 关：index_head_dim // 2 + 4  (FP4 packed: 2值/字节 → 64B + 4B = 68)
            FP8 关：index_head_dim + 4        (FP8: 1B/值 → 128B + 4B = 132)
        ⚙️ +4 = 1 个 fp32 scale(每 token 1 个，因 quant_block_size=128=index_head_dim)。
        """
        if self.use_fp4_indexer:
            # FP4：2 值 packed 进 1 字节 → 数据 = index_head_dim // 2。
            return self.index_head_dim // 2 + 4
        # FP8：每值 1 字节 → 数据 = index_head_dim。
        return self.index_head_dim + 4

    def _create_buffer(self):
        """💾 为每层分配 indexer 的 [num_pages, page_bytes] uint8 buffer。

        🔗 调用链定位(① 创建链)：__init__ → _create_buffer。
        ⚙️ 行为：page_bytes = page_size * bytes_per_token(不对齐 576，与 SingleKVPool 不同)；
            num_pages = (size + page_size + 1)//page_size(+1 哨兵页)。
        """
        page_bytes = self.page_size * self.get_bytes_per_token()
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                # custom_mem_pool：PD 分离 + NVLink 场景下用专用内存池(详见基类 _create_buffers 注释)。
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # 每层一个 [num_pages, page_bytes] uint8 张量；data+scale 按字节打包在同一页。
                self.index_k_with_scale_buffer = [
                    torch.zeros(
                        (self.size + self.page_size + 1) // self.page_size,
                        page_bytes,
                        dtype=self.index_k_with_scale_buffer_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """🚫 未实现——indexer 无独立 K/V 概念，用 get_index_k_with_scale_buffer / get_index_k_scale_buffer。"""
        raise NotImplementedError()

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """🚫 未实现——同 get_kv_buffer，indexer 走专属 get_index_* 接口。"""
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """🚫 未实现——同 get_kv_buffer，indexer 走专属 get_index_* 接口。"""
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        """🚫 未实现——indexer 走 set_index_k_scale_buffer / set_index_fused / set_index_fp4。"""
        raise NotImplementedError()

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        """📖 取整层 indexer buffer（data+scale 打包，由调用方自行 paged gather）。

        🔗 调用链定位(④ 读)：
            dsv4/indexer.py:385,428 / compressor_v2.py:506
              └─ DeepSeekV4TokenToKVPool.get_index_k_with_scale_buffer(:1074) → 当前函数

        📥 参数：layer_id : 本地 c4 层号。
        📤 返回：整层 [num_pages, page_bytes] uint8 buffer(data+scale 打包)。
        ⚠️ 用裸 [layer_id]（见类⚠️1：start_layer=0 不冲突）。
        """
        return self.index_k_with_scale_buffer[layer_id]

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """📖 融合读——按 page_indices 把 indexer 的 (K, scale) paged gather 出来。

        🔗 调用链定位(④ 读，融合 gather)：
            dsa/dsa_indexer.py:826 等
              └─ DeepSeekV4TokenToKVPool.get_index_k_scale_buffer(:1080) → 当前函数
                 └─ index_buf_accessor.GetKAndS.execute  (aiter preshuffle / triton)

        📥 参数：
            layer_id     : 本地 c4 层号
            seq_len      : 序列长度(决定 gather 的 token 数)
            page_indices : 页索引(按页定位 buffer 行)
        📤 返回：(index_k, index_k_scale) —— K 与 scale 分开返回，供 indexer 打分。
        ⚙️ 行为：复用 DSA 的 GetKAndS(布局与 DSA indexer 一致)；aiter preshuffle 路径需
            page_size=64 preshuffle 布局，否则回退 triton。
        """
        buf = self.index_k_with_scale_buffer[layer_id]
        return index_buf_accessor.GetKAndS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        """✍️ 分段写——index_k 与 index_k_scale 分开传入，经 triton 散点写页（融合内核不可用时的回退）。

        🔗 调用链定位(③ 写，回退路径)：
            dsv4/compressor.py:226 / compressor_v2.py:655
              └─ DeepSeekV4TokenToKVPool.set_index_k_scale_buffer(:1093) → 当前函数
                 └─ index_buf_accessor.SetKAndS.triton  (data+scale 融合写入同一页)

        📥 参数：
            layer_id      : 本地 c4 层号
            loc           : 写入目标页索引
            index_k       : indexer K 数据(FP8/FP4)
            index_k_scale : 每 token 的 fp32 scale
        📤 返回：无(原地写入 index_k_with_scale_buffer)。
        ⚠️ 这里用 [layer_id - start_layer]（与读路径的裸 [layer_id] 不同，见类⚠️1）。
        """
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def set_index_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        """✍️ 融合写——上游已把 indexer 的 (data+scale) 预 packed，直接整页写入。

        🔗 调用链定位(③ 写，主路径)：
            dsv4/compressor.py:217 / compressor_v2.py:641
              └─ DeepSeekV4TokenToKVPool.set_index_k_fused(:1173) → 当前函数
                 └─ fused_store_cache(type="indexer")  (JIT CUDA / HIP triton)

        📥 参数：
            layer_id : 本地 c4 层号
            loc      : 写入目标页索引
            cache_k  : 预 packed 的整页 indexer KV(data+scale 布局)
        📤 返回：无(原地写入 index_k_with_scale_buffer)。
        ⚙️ type="indexer" 区别于 SingleKVPool 的 "flashmla"（页宽/布局不同，用不同内核）。
        """
        return fused_store_cache(
            input=cache_k,
            cache=self.index_k_with_scale_buffer[layer_id - self.start_layer],
            indices=loc,
            page_size=self.page_size,
            type="indexer",
        )

    def set_index_fp4(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        """✍️ FP4 写——把 indexer K 量化成 FP4(2值/字节)+scale 后写入（use_fp4_indexer 开启时）。

        🔗 调用链定位(③ 写，FP4 路径)：
            DeepSeekV4TokenToKVPool.set_index_k_fp4(:1183) → 当前函数
               └─ store_fp4_index_k_cache  (fp4 量化 + 专用 triton 内核)

        📥 参数：
            layer_id : 本地 c4 层号
            loc      : 写入目标页索引
            cache_k  : 待量化的 indexer K(BF16, last_dim=128)
        📤 返回：无(原地写入 index_k_with_scale_buffer)。
        ⚙️ 行为：先 quantize_fp4_indexer_tensor → (k_fp4[64], k_sf)，再按页散点写；
            cache 页宽须 == page_size * (64 + 4) = page_size * 68。
        ⚠️ 仅 use_fp4_indexer=True 时由上层调用；否则走 set_index_fused。
        """
        from sglang.srt.layers.attention.dsv4.fp4_indexer import (
            store_fp4_index_k_cache,
        )

        return store_fp4_index_k_cache(
            input=cache_k,
            cache=self.index_k_with_scale_buffer[layer_id - self.start_layer],
            loc=loc,
            page_size=self.page_size,
        )


class DeepSeekV4LayerItem(NamedTuple):
    compress_ratio: Literal[0, 4, 128]
    compress_layer_id: int
    compress_kv_pool: Optional[DeepSeekV4SingleKVPool] = None


class DeepSeekV4UnifiedKVPool:
    """
    Layout:
    unified_kv[L]: ``[swa_pages + compress_pages, head_dim]`` bf16
    - rows ``[0, swa_pages)``   = SWA ring (``req_pool_indices * swa_window + pos % swa_window``)
    - rows ``[swa_pages, ...)`` = compressed (``swa_pages + page_index``)
    """

    K_PER_BLOCK = {0: 0, 4: 32, 128: 1}

    def __init__(
        self,
        *,
        stage_ratios: List[int],
        num_slots: int,
        num_blocks: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        device: str,
        memory_saver_adapter,
        custom_mem_pool,
        swa_ring_size: int,
    ):
        self.swa_ring_size = swa_ring_size
        self.head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.num_slots = num_slots
        self.swa_pages = num_slots * self.swa_ring_size
        self.num_blocks = num_blocks
        self.k_per_block = dict(self.K_PER_BLOCK)

        bufs = []
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(custom_mem_pool)
                if custom_mem_pool
                else nullcontext()
            ):
                for ratio in stage_ratios:
                    compress_pages = self.num_blocks * self.k_per_block[ratio]
                    bufs.append(
                        torch.zeros(
                            self.swa_pages + compress_pages,
                            self.head_dim,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                    )
        self.kv_buffer = bufs

    def get_unified_kv(self, local_layer_id: int) -> torch.Tensor:
        return self.kv_buffer[local_layer_id]

    def get_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs = [b.data_ptr() for b in self.kv_buffer]
        data_lens = [b.nbytes for b in self.kv_buffer]
        item_lens = [b[0].nbytes for b in self.kv_buffer]
        return data_ptrs, data_lens, item_lens


class DeepSeekV4TokenToKVPool(BaseSWAKVPool):
    """🗂️ DeepSeek-V4 顶层 KV 池（门面/组合）—— DSV4 部署时唯一的 token_to_kv_pool 入口。

    由 ModelRunnerKVCacheMixin 在 is_deepseek_v4(hf_config) 为真时创建
    (model_runner_kv_cache_mixin.py:410)；模型侧 deepseek_v4.py:505 断言
    isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)。

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧬 组合结构：一个门面持有 5 类子池，按"层压缩比"路由                                    ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  DSV4 把每层按 compression_ratio 归类，不同类用不同子池：                                ║
    ║    ratio=0   → SWA 层      → swa_kv_pool        (DeepSeekV4SingleKVPool)              ║
    ║    ratio=4   → C4 压缩层   → c4_kv_pool          (SingleKVPool / HiSparseC4DevicePool)║
    ║    ratio=128 → C128 压缩层 → c128_kv_pool        (DeepSeekV4SingleKVPool)              ║
    ║    ratio=4 层额外有 indexer → c4_indexer_kv_pool (DeepSeekV4IndexerPool)              ║
    ║    ratio=4/128 层有压缩状态 → compress_state_pools[] / indexer_compress_state_pools[] ║
    ║       (CompressStatePool，存 kv_score 等聚合状态)                                       ║
    ║  另有"统一路径"开关 _unified_kv(is_unified_kv_triton)：开启时 swa+c4+c128 合进一块       ║
    ║  unified_kv_pool(DeepSeekV4UnifiedKVPool)，上述三池置 None。                              ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧩 layer_mapping：全局层号 → (ratio, 本地层号, 子池) 的路由表                          ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  _init_compressed_layer_mapping 按 stage_ratios 顺序为每层建 DeepSeekV4LayerItem：      ║
    ║    (compress_ratio, compress_layer_id, compress_kv_pool)                                ║
    ║  所有 *_extra_key_buffer* / *_index_k_* 方法都先 layer_mapping[layer_id] 拿到            ║
    ║  (ratio, 本地层号, 子池)，再转发给对应子池。SWA 方法用 _swa_local_layer_id 换成本地索引。    ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🧩 对外接口（attention backend / compressor 经这些门面方法访问子池）                    ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  SWA 读写：get_swa_key_buffer / set_swa_key_buffer(+radix/fused 变体)                   ║
    ║  压缩层读写：get_extra_key_buffer / set_extra_key_buffer(+fused)                        ║
    ║  Indexer 读写：get_index_k_* / set_index_k_*(scale/fused/fp4)                           ║
    ║  状态：get_attention_compress_states / get_indexer_compress_states                      ║
    ║  PD 分离传输：get_contiguous_buf_infos / get_unified_swa_ring_buf_infos / get_state_buf_infos ║
    ║  统一 buffer：get_unified_kv(_unified_kv 开启时)                                        ║
    ║  get_key_buffer/get_value_buffer/get_kv_buffer/set_kv_buffer  🚫 NotImplementedError     ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║  🔗 调用链（创建 + 前向读写，attention backend / compressor 为调用方）                     ║
    ╠══════════════════════════════════════════════════════════════════════════════════╣
    ║  ① 创建：ModelRunnerKVCacheMixin → DeepSeekV4TokenToKVPool(...) (:410)                  ║
    ║       └─ 内部建 swa/c4/c128 + c4_indexer + compress_state_pools + layer_mapping         ║
    ║  ③ 写 KV：deepseek_v4_backend / compressor(v1/v2)                                       ║
    ║       └─ set_swa_key_buffer(_radix_fused) → swa_kv_pool.set_key_buffer(_fused)          ║
    ║       └─ set_extra_key_buffer(_fused)     → c4/c128_kv_pool.set_key_buffer(_fused)       ║
    ║       └─ set_index_k_(scale/fused/fp4)    → c4_indexer_kv_pool.set_index_*              ║
    ║  ④ 读 KV：deepseek_v4_backend / indexer                                                   ║
    ║       └─ get_swa_key_buffer(_radix) / get_extra_key_buffer → 子池 get_key_buffer         ║
    ║       └─ get_index_k_with_scale_buffer / get_index_k_scale_buffer → indexer 子池         ║
    ║  ⑥ PD 分离：get_contiguous_buf_infos 暴露各子池 ptr/len/item 给传输引擎注册                ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝

    ⚠️ 注意：本类几乎都是薄转发（全局层号→本地层号→子池）；详细字节布局/量化在子池
       DeepSeekV4SingleKVPool / DeepSeekV4IndexerPool 的 docstring 里。layer_id 路由约定：
       SWA 用 _swa_local_layer_id(= layer_id - _stage_start)；压缩层/indexer 用 layer_mapping。
    """

    def __init__(
        self,
        max_num_reqs: int,
        swa_size: int,
        c4_size: int,
        c128_size: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        page_size: int,
        swa_page_size: int,
        dtype: torch.dtype,
        state_dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        indexer_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        compression_ratios: List[int],
        sliding_window: int = 128,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_hisparse: bool = False,
        online_mtp_max_draft_tokens: int = 0,
        num_req_slots: Optional[int] = None,
    ):
        """🧱 构造 DSV4 顶层 KV 池：建 5 类子池 + 压缩状态池 + 层路由表。

        🔗 调用链定位(① 创建链)：
            ModelRunnerKVCacheMixin(:410, is_deepseek_v4 时) → 本类
              ├─ 建 swa/c4/c128 子池(或 unified_kv_pool，二选一)
              ├─ 建 c4_indexer_kv_pool
              ├─ _init_compressed_layer_mapping(层路由表)
              └─ _init_paged_compress_states(每层压缩状态)

        📥 关键参数：
            compression_ratios : 每层的压缩比(0/4/128)，决定该层走哪个子池
            swa/c4/c128_size   : 三类子池各自的容量(token 数)
            *_state_pool_size   : 压缩状态池容量(CompressStatePool)
            enable_hisparse     : c4 层是否用 HiSparseC4DevicePool(分层稀疏)
            online_mtp_max_draft_tokens : C128 在线压缩 + MTP 的草稿 token 数
        ⚙️ 两条路径：_unified_kv=True 走 unified_kv_pool(三合一)；False 走三个独立 SingleKVPool。
        """
        super().__init__(
            swa_size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        # c4 的"逻辑容量"= c128_size * 32：C4 每 32 token 压成 1 个 C128 token 的对应关系，
        # 用于 indexer 容量等推导（indexer_size 默认取 c4_logical_size，见下方）。
        c4_logical_size = c128_size * 32

        logger.info(
            "Initialize DeepSeekV4TokenToKVPool with "
            f"{max_num_reqs=} {swa_size=} {c4_size=} "
            f"{c4_logical_size=} {c128_size=} "
            f"{c4_state_pool_size=} {c128_state_pool_size=}"
        )

        self.max_num_reqs = max_num_reqs
        # SWA ring needs one slot per addressable req_pool_idx. PD decode inflates
        # req_to_token past max_num_reqs (pre-alloc), so the caller passes the real
        # capacity; sizing as max_num_reqs+1 overflows ("length out of range").
        # SWA ring 按 req_pool_idx 寻址，故槽位数 = 真实可寻址请求数(PD decode 会预分配超额)。
        self.num_req_slots = (
            num_req_slots if num_req_slots is not None else max_num_reqs + 1
        )
        # 以下 size 均由 pool_configurator 运行时按 GPU 显存预算算出（非固定默认），
        # 公式见 pool_configurator.py:456-464（full_token = available_bytes / bytes_per_full_token）：
        #   swa_size              = swa_tokens = full_token * swa_ratio
        #   c4_size               = full_token // (4 * c4_shrink_factor)   ← c4_shrink_factor 默认 1(HiSparse 才 >1)
        #   c128_size             = full_token // 128
        #   c4_state_pool_size    = swa_tokens // swa_page_size * c4_ring_size    (c4_ring_size: 非spec 8 / spec 16)
        #   c128_state_pool_size  = swa_tokens // swa_page_size * c128_ring_size  (c128_ring_size: 非spec 128 / spec 256 / online 1)
        self.c4_size = c4_size
        self.c4_logical_size = c4_logical_size  # = c128_size * 32
        self.c128_size = c128_size
        self.c4_state_pool_size = c4_state_pool_size
        self.c128_state_pool_size = c128_state_pool_size
        # state_dtype：DSV4 架构常量 = torch.float32（c4/c128 状态 buffer 用，由 mixin:959 设置）。
        self.state_dtype = state_dtype
        self.compression_ratios = compression_ratios
        self.online_mtp_max_draft_tokens = online_mtp_max_draft_tokens
        self.online_c128_mtp_pending_seq_lens: Optional[torch.Tensor] = None
        if ONLINE_C128 and envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get():
            # 在线 C128 + MTP：预分配"待处理 seq_len"槽，供 MTP 多草稿合并压缩。
            self.online_c128_mtp_pending_seq_lens = torch.empty(
                max_num_reqs, dtype=torch.int64, device=device
            )

        # Determine this PP stage's absolute layer range
        # PP 切分：本 rank 持有的绝对层区间 [_stage_start, _stage_end)，用于层号本地化。
        if (
            start_layer is not None
            and end_layer is not None
            and len(compression_ratios) >= end_layer
        ):
            self._stage_start = start_layer
            self._stage_end = end_layer
        else:
            self._stage_start = 0
            self._stage_end = len(compression_ratios)
        stage_ratios = compression_ratios[self._stage_start : self._stage_end]

        # page_size：DSV4 硬要求 = 256（mixin:398 assert swa_page_size==256）；swa_page_size = page_size。
        assert page_size % swa_page_size == 0
        # sliding_window：来自 model_config.window_size，默认 128（DeepSeekV4Config.window_size=128）。
        self.sliding_window = sliding_window

        self.swa_size = swa_size
        self.swa_window_size = swa_page_size  # = swa_page_size = 256
        self.swa_page_size = swa_page_size    # = page_size = 256
        # scale_pad：写死 1（把 nope 的 scale 段补到 8B 边界，见 SingleKVPool 字节布局）。
        self.scale_pad = 1

        # 以下 dim 为模型架构常量（DeepSeekV4Config 默认值），固定不可改：
        #   qk_nope_head_dim=448(FP8 nope 段) / qk_rope_head_dim=64(BF16 rope 段) / indexer_head_dim=128(indexer MQA)
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.indexer_head_dim = indexer_head_dim

        # 本 stage 各类层数：按 ratio 统计，决定各子池 layer_num。
        c4_layer_num = sum(1 for r in stage_ratios if r == 4)
        c128_layer_num = sum(1 for r in stage_ratios if r == 128)
        # 压缩层页大小 = page_size // ratio（C4 每 4 token 一页、C128 每 128 token 一页）。
        c4_page_size = page_size // 4
        c128_page_size = page_size // 128

        from sglang.srt.layers.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        # _unified_kv：是否走"三合一"统一 buffer(triton 内核路径)。
        self._unified_kv = is_unified_kv_triton()

        if self._unified_kv:
            # 统一路径：swa+c4+c128 合进一块 unified_kv_pool，三池置 None。
            self.swa_kv_pool = None
            self.c4_kv_pool = None
            self.c128_kv_pool = None
            server_args = get_global_server_args()
            # 投机解码多草稿：SWA ring 需额外 (num_draft_tokens - 1) 槽承接草稿。
            spec_extra = (
                (server_args.speculative_num_draft_tokens - 1)
                if server_args.speculative_algorithm is not None
                else 0
            )
            self.unified_kv_pool = DeepSeekV4UnifiedKVPool(
                stage_ratios=stage_ratios,
                num_slots=self.num_req_slots,
                num_blocks=self.c128_size,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                device=device,
                memory_saver_adapter=self.memory_saver_adapter,
                custom_mem_pool=self.custom_mem_pool,
                swa_ring_size=self.sliding_window + spec_extra,
            )

            self.unified_swa_window = self.sliding_window
            self.unified_swa_ring_size = self.sliding_window + spec_extra
            self.unified_swa_pages = self.unified_kv_pool.swa_pages
        else:
            # 分离路径：三个独立 DeepSeekV4SingleKVPool。
            self.unified_kv_pool = None
            self.swa_kv_pool = DeepSeekV4SingleKVPool(
                swa_size,
                swa_page_size,
                dtype,
                qk_nope_head_dim,
                qk_rope_head_dim,
                layer_num,
                device,
                enable_memory_saver,
            )

            # c4 层：默认 SingleKVPool；开 HiSparse 则用 HiSparseC4DevicePool(分层稀疏)。
            c4_kv_pool_type = DeepSeekV4SingleKVPool
            if enable_hisparse:
                c4_kv_pool_type = HiSparseC4DevicePool
            self.c4_kv_pool = c4_kv_pool_type(
                c4_size,
                c4_page_size,
                dtype,
                qk_nope_head_dim,
                qk_rope_head_dim,
                c4_layer_num,
                device,
                enable_memory_saver,
            )

            self.c128_kv_pool = DeepSeekV4SingleKVPool(
                c128_size,
                c128_page_size,
                dtype,
                qk_nope_head_dim,
                qk_rope_head_dim,
                c128_layer_num,
                device,
                enable_memory_saver,
            )

        # indexer 容量：非 HIP 或开 compressor_v2 时用 c4_logical_size，否则用 c4_size。
        indexer_size = (
            self.c4_logical_size
            if (not _is_hip or envs.SGLANG_OPT_USE_COMPRESSOR_V2.get())
            else c4_size
        )
        # indexer 子池：仅 C4 层有(ratio=4)，复用 DeepSeekV4IndexerPool。
        self.c4_indexer_kv_pool = DeepSeekV4IndexerPool(
            indexer_size,
            c4_page_size,
            dtype,
            indexer_head_dim,
            c4_layer_num,
            device,
            enable_memory_saver,
        )

        # 建层路由表 + 压缩状态池。
        self._init_compressed_layer_mapping()

        if _is_hip:
            # HIP 上压缩状态不走 memory_saver。
            self._init_paged_compress_states(False)
        else:
            self._init_paged_compress_states(enable_memory_saver)

    def get_unified_kv(self, layer_id: int) -> torch.Tensor:
        """📖 取统一 buffer(_unified_kv 路径)的一层 KV，层号换成本地索引后转发。"""
        return self.unified_kv_pool.get_unified_kv(layer_id - self._stage_start)

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        """🔗 登记全局 token 索引 → SWA 索引的映射表（由 allocator/调度器在分配后注入）。"""
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def get_ring_size(self, compress_ratio: int) -> int:
        """🧬 查压缩状态 ring 大小：按 ratio 与是否投机解码决定(c4:8/16, c128:128/256, online=1)。"""
        server_args = get_global_server_args()
        is_speculative = server_args.speculative_algorithm is not None
        return get_compress_state_ring_size(compress_ratio, is_speculative)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        """🔀 把全局 token 索引翻译成 SWA ring 内的索引（查 full_to_swa_index_mapping）。"""
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices]

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """🌐 PD 分离：把各子池 buffer 的 (ptr, len, item_len) 暴露给传输引擎注册。

        🔗 调用链定位(⑥ PD 分离传输)：BootstrapManager → 本函数 → KV 传输引擎(RDMA/NVLink)。
        📤 返回：三组并行 list——data_ptrs / data_lens / item_lens，每个元素对应一块 buffer。
        ⚙️ 行为分两条路径：
            • _unified_kv：unified_kv_pool 每层的"压缩区"[swa_pages:] + c4_indexer 各自一段；
              SWA ring 区单独由 get_unified_swa_ring_buf_infos 暴露。
            • 分离路径：c4 / c4_indexer / c128 三组子池 buffer 顺序拼接。
          item_len = 一页 KV 的字节数（传输按页对齐）。
        ⚠️ 顺序 [c4, c4_indexer, c128] 须与 PP ptr-slicing 约定一致，勿随意调换。
        """
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        if self._unified_kv:
            # Unified buffer per layer: [swa_pages + compress_pages, head_dim].
            # Compressed region [swa_pages:] is page-contiguous (row swa_pages +
            # loc//ratio), so reuse the page-block PD transfer by offsetting the ptr
            # past the SWA ring and setting item_len = one page of rows. The SWA ring
            # ships separately as StateType.SWA_RING. Order [c4, c4_indexer, c128]
            # mirrors the non-unified kv_data layout (keeps PP ptr-slicing valid).
            # 统一 buffer 每层 [swa_pages + compress_pages, head_dim]：压缩区页连续，
            # 跳过 SWA ring 区(ptr 偏移 swa_pages)，按页块传输；SWA ring 另走 SWA_RING。
            stage_ratios = self.compression_ratios[self._stage_start : self._stage_end]
            swa_pages = self.unified_kv_pool.swa_pages

            def _append_compressed_entry(local_layer_id: int, ratio: int) -> None:
                buf = self.unified_kv_pool.kv_buffer[local_layer_id]
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                row_bytes = buf[0].nbytes
                rows_per_page = self.page_size // ratio
                compress_rows = buf.shape[0] - swa_pages
                data_ptrs.append(buf.data_ptr() + swa_pages * row_bytes)
                data_lens.append(compress_rows * row_bytes)
                item_lens.append(rows_per_page * row_bytes)

            c4_locals = [i for i, r in enumerate(stage_ratios) if r == 4]
            c128_locals = [i for i, r in enumerate(stage_ratios) if r == 128]

            for i in c4_locals:
                _append_compressed_entry(i, 4)
            for buf in self.c4_indexer_kv_pool.index_k_with_scale_buffer:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)
            for i in c128_locals:
                _append_compressed_entry(i, 128)

            return data_ptrs, data_lens, item_lens

        # 分离路径：c4 / c4_indexer / c128 三组子池 buffer 顺序拼接。
        buf_groups = [
            self.c4_kv_pool.kv_buffer,
            self.c4_indexer_kv_pool.index_k_with_scale_buffer,
            self.c128_kv_pool.kv_buffer,
        ]

        for bufs in buf_groups:
            for buf in bufs:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)

        return data_ptrs, data_lens, item_lens

    def get_unified_swa_ring_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """🌐 PD 分离：unified_kv 每层的 SWA-ring 区 [0, swa_pages)，按 ring slot 逐行寻址。

        🔗 调用链定位(⑥ PD 分离)：作为 StateType.SWA_RING 组件单独传输（与压缩区分开）。
        📤 返回：unified_kv_pool 每层 SWA ring 区的 (ptr, len, item_len)；非 unified 路径返回空。
        """
        # TODO(billishyahao): validate PP layer-slicing for SWA_RING.
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        if not self._unified_kv:
            return data_ptrs, data_lens, item_lens
        swa_pages = self.unified_kv_pool.swa_pages
        for buf in self.unified_kv_pool.kv_buffer:
            assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
            row_bytes = buf[0].nbytes
            data_ptrs.append(buf.data_ptr())
            data_lens.append(swa_pages * row_bytes)
            item_lens.append(row_bytes)
        return data_ptrs, data_lens, item_lens

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """🌐 PD 分离：暴露压缩状态(swa buffer + 各 CompressStatePool 的 kv_score)给传输引擎。

        📤 返回：分离路径下 swa_kv_pool 各层 buffer + compress_state_pools /
            indexer_compress_state_pools 的 kv_score buffer 的 (ptr, len, item_len)。
        ⚙️ item_len = 一页状态 × ring_size（状态按 ring 分块，传输按 ring 对齐）。
        """
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []

        if not self._unified_kv:
            # 分离路径：SWA 状态即 swa_kv_pool 各层 buffer（统一路径下 SWA 在 unified 里）。
            for buf in self.swa_kv_pool.kv_buffer:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)

        # 压缩状态：主 attention 状态 + indexer 状态，按 ring 分块。
        for pools in [
            self.compress_state_pools,
            self.indexer_compress_state_pools,
        ]:
            for pool in pools:
                if pool is None:
                    continue
                t = pool.kv_score_buffer.kv_score
                assert t.ndim == 2, f"expected 2D buffer, got {t.ndim}D"
                data_ptrs.append(t.data_ptr())
                data_lens.append(t.nbytes)
                item_lens.append(t[0].nbytes * pool.ring_size)

        return data_ptrs, data_lens, item_lens

    def _init_paged_compress_states(self, enable_memory_saver: bool):
        """🧱 为每个压缩层(c4/c128)建 CompressStatePool（主状态 + indexer 状态）。

        🔗 调用链定位(① 创建链)：__init__ → 本函数。
        ⚙️ 行为：按 ratio 逐层建状态池；ratio=0(SWA)跳过；ratio=4 额外建 indexer 状态池。
            每个 CompressStatePool 存 kv_score 等聚合状态，ring_size 由 get_ring_size 决定。
        """
        c4_state_pool_size = self.c4_state_pool_size
        c128_state_pool_size = self.c128_state_pool_size
        total_L = len(self.compression_ratios)
        # 主压缩状态池 + indexer 压缩状态池，按全局层号索引（SWA 层为 None）。
        self.compress_state_pools: List[Optional[CompressStatePool]] = [None] * total_L
        self.indexer_compress_state_pools: List[Optional[CompressStatePool]] = [
            None
        ] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 0:
                # SWA 层无压缩状态。
                continue
            overlap = ratio == 4
            size = c4_state_pool_size if ratio == 4 else c128_state_pool_size
            ring_size = self.get_ring_size(ratio)

            # 主 attention 压缩状态：head_dim = nope + rope。
            self.compress_state_pools[idx] = CompressStatePool(
                size=size,
                ring_size=ring_size,
                overlap=overlap,
                head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
                dtype=self.state_dtype,
                device=self.device,
                enable_memory_saver=enable_memory_saver,
                ratio=ratio,
                online=(ratio == 128 and ONLINE_C128),
                swa_page_size=self.swa_page_size,
                online_mtp_max_draft_tokens=(
                    self.online_mtp_max_draft_tokens if ratio == 128 else 0
                ),
            )

            if ratio == 4:
                # 仅 C4 层有 indexer 状态：head_dim = indexer_head_dim。
                self.indexer_compress_state_pools[idx] = CompressStatePool(
                    size=size,
                    ring_size=ring_size,
                    overlap=overlap,
                    head_dim=self.indexer_head_dim,
                    device=self.device,
                    dtype=self.state_dtype,
                    enable_memory_saver=enable_memory_saver,
                    ratio=ratio,
                    swa_page_size=self.swa_page_size,
                )

    def _init_compressed_layer_mapping(self):
        """🧱 建层路由表 layer_mapping：全局层号 → DeepSeekV4LayerItem(ratio, 本地层号, 子池)。

        🔗 调用链定位(① 创建链)：__init__ → 本函数（在子池建好后调用）。
        ⚙️ 行为：按 stage_ratios 顺序，ratio=0/4/128 分别用各自计数器(c1/c4/c128_cnt)作本地层号，
            并绑定对应子池（c4→c4_kv_pool、c128→c128_kv_pool；ratio=0 无子池）。
        ⚠️ layer_mapping 是所有 *_extra_key_buffer* / *_index_k_* 方法的路由依据。
        """
        c1_cnt = c4_cnt = c128_cnt = 0
        total_L = len(self.compression_ratios)
        self.layer_mapping: List[Optional[DeepSeekV4LayerItem]] = [None] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 0:
                # SWA 层：无压缩子池，仅记 ratio 与本地层号。
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=0,
                    compress_layer_id=c1_cnt,
                )
                c1_cnt += 1
            elif ratio == 4:
                # C4 层：绑 c4_kv_pool。
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=4,
                    compress_layer_id=c4_cnt,
                    compress_kv_pool=self.c4_kv_pool,
                )
                c4_cnt += 1
            elif ratio == 128:
                # C128 层：绑 c128_kv_pool。
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=128,
                    compress_layer_id=c128_cnt,
                    compress_kv_pool=self.c128_kv_pool,
                )
                c128_cnt += 1
            else:
                raise ValueError(f"Unsupported compression ratio: {ratio}")

    def wait_layer_transfer(self, layer_id: int) -> None:
        """⏳ 分层流水：若开启了 layer_transfer_counter，阻塞等本层 KV 搬运完成（避免读到半成品）。"""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_attention_compress_states(self, layer_id: int) -> CompressStatePool:
        """📖 取主 attention 压缩状态池（仅 c4/c128 层有，SWA 层会 assert 失败）。"""
        self.wait_layer_transfer(layer_id)
        compress_state_pool = self.compress_state_pools[layer_id]
        assert (
            compress_state_pool is not None
        ), "Only c4/c128 layers have attention states."
        return compress_state_pool

    def get_online_c128_mtp_state_slot_offset(self) -> int:
        """🧬 在线 C128 + MTP：取 c128 状态池的 mtp state slot offset（无则 0）。"""
        for pool in self.compress_state_pools:
            if pool is not None and pool.ratio == 128:
                return int(pool.online_mtp_state_slot_offset)
        return 0

    def get_online_c128_mtp_max_draft_tokens(self) -> int:
        """🧬 在线 C128 + MTP：取 c128 状态池的最大草稿 token 数（无则 0）。"""
        for pool in self.compress_state_pools:
            if pool is not None and pool.ratio == 128:
                return int(pool.online_mtp_max_draft_tokens)
        return 0

    def get_online_c128_mtp_pending_seq_lens(self) -> torch.Tensor:
        """🧬 在线 C128 + MTP：取"待处理 seq_len"张量（MTP 多草稿合并压缩用）。"""
        assert self.online_c128_mtp_pending_seq_lens is not None
        return self.online_c128_mtp_pending_seq_lens

    def get_indexer_compress_states(self, layer_id: int) -> CompressStatePool:
        """📖 取 indexer 压缩状态池（仅 c4 层有，其它层会 assert 失败）。"""
        self.wait_layer_transfer(layer_id)
        indexer_compress_state_pool = self.indexer_compress_state_pools[layer_id]
        assert (
            indexer_compress_state_pool is not None
        ), "Only c4 layers have indexer states."
        return indexer_compress_state_pool

    def _swa_local_layer_id(self, layer_id: int) -> int:
        """🔀 全局层号 → SWA 池本地索引(PP-stage-local)。

        layer_id - _stage_start：把绝对层号换算成本 rank 持有的本地索引；
        SWA 池跨所有层(swa_kv_pool 用 layer_num=总层数)，故只减 _stage_start。
        """
        return layer_id - self._stage_start

    def get_swa_key_buffer(self, layer_id: int) -> torch.Tensor:
        """📖 读 SWA 层 KV（转发 swa_kv_pool.get_key_buffer，层号本地化）。"""
        self.wait_layer_transfer(layer_id)
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))

    def set_swa_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        """✍️ 写 SWA 层 KV（分段写，转发 swa_kv_pool.set_key_buffer）。"""
        self.swa_kv_pool.set_key_buffer(
            self._swa_local_layer_id(layer_id), loc, cache_nope_fp8_rope_bf16_pack
        )

    def get_extra_key_page_size(self, layer_id: int) -> int:
        """📖 取压缩层页大小（经 layer_mapping 路由到 c4/c128 子池的 page_size）。"""
        _, _, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.page_size

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor | None:
        """📖 读压缩层 KV（经 layer_mapping 路由到 c4/c128 子池的 get_key_buffer）。"""
        self.wait_layer_transfer(layer_id)
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.get_key_buffer(compress_layer_id)

    def set_extra_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        """✍️ 写压缩层 KV（分段写，经 layer_mapping 路由到 c4/c128 子池的 set_key_buffer）。"""
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        compress_kv_pool.set_key_buffer(
            compress_layer_id, loc, cache_nope_fp8_rope_bf16_pack
        )

    def get_index_k_page_size(self) -> int:
        """📖 取 indexer 页大小（c4_indexer_kv_pool.page_size）。"""
        return self.c4_indexer_kv_pool.page_size

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        """📖 取整层 indexer buffer（仅 c4 层，转发 c4_indexer_kv_pool）。"""
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_with_scale_buffer(compress_layer_id)

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """📖 融合读 indexer (k, scale)（仅 c4 层，转发 c4_indexer_kv_pool）。"""
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.get_index_k_scale_buffer(
            compress_layer_id, seq_len, page_indices
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        """✍️ 分段写 indexer K+scale（仅 c4 层，转发 c4_indexer_kv_pool）。"""
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        self.c4_indexer_kv_pool.set_index_k_scale_buffer(
            compress_layer_id, loc, index_k, index_k_scale
        )

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """🚫 未实现——DSV4 用 get_swa_key_buffer / get_extra_key_buffer 按层类型分流。"""
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """🚫 未实现——DSV4 无独立 V(latent 风格)，见 get_*_key_buffer。"""
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """🚫 未实现——同 get_value_buffer。"""
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        """🚫 未实现——DSV4 用 set_swa_key_buffer / set_extra_key_buffer 按层类型分流。"""
        raise NotImplementedError()

    def set_swa_key_buffer_radix(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        """✍️ 写 SWA 层 KV（radix 后端入口，转发 swa_kv_pool.set_key_buffer）。"""
        self.swa_kv_pool.set_key_buffer(
            self._swa_local_layer_id(layer_id), swa_loc, cache_nope_fp8_rope_bf16_pack
        )

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        """📖 读 SWA 层 KV（radix 后端入口，转发 swa_kv_pool.get_key_buffer）。"""
        self.wait_layer_transfer(layer_id)
        return self.swa_kv_pool.get_key_buffer(self._swa_local_layer_id(layer_id))

    def set_swa_key_buffer_radix_fused(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        """✍️ 融合写 SWA 层 KV（转发 swa_kv_pool.set_key_buffer_fused，cache_k 已预 packed）。"""
        return self.swa_kv_pool.set_key_buffer_fused(
            self._swa_local_layer_id(layer_id), swa_loc, cache_k
        )

    def set_swa_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """✍️ 融合写 SWA 层 KV：把 norm + RoPE + 写缓存融合进一个 fused_k_norm_rope_flashmla 内核。

        🔗 调用链定位(③ 写 KV，radix 主路径)：
            deepseek_v4_backend.py / deepseek_v4_backend_hip_radix.py
              └─ set_swa_key_buffer_radix_fused_norm_rope → fused_k_norm_rope_flashmla
        ⚙️ 行为：kv 经 RMSNorm(kv_weight) → RoPE(freqs_cis, positions) → 直接散点写 swa buffer，
            省去中间临时张量；page_size 取 swa_kv_pool.page_size。
        """
        fused_k_norm_rope_flashmla(
            kv=kv,
            kv_weight=kv_weight,
            eps=eps,
            freqs_cis=freqs_cis,
            positions=positions,
            out_loc=swa_loc,
            kvcache=self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)],
            page_size=self.swa_kv_pool.page_size,
        )

    def set_extra_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        """✍️ 融合写压缩层 KV（经 layer_mapping 路由到 c4/c128 子池的 set_key_buffer_fused）。"""
        _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
        assert compress_kv_pool is not None
        return compress_kv_pool.set_key_buffer_fused(compress_layer_id, loc, cache_k)

    def set_index_k_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        """✍️ 融合写 indexer K（仅 c4 层，转发 c4_indexer_kv_pool.set_index_fused）。"""
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.set_index_fused(compress_layer_id, loc, cache_k)

    def set_index_k_fp4(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        """✍️ FP4 写 indexer K（仅 c4 层，转发 c4_indexer_kv_pool.set_index_fp4）。"""
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        return self.c4_indexer_kv_pool.set_index_fp4(compress_layer_id, loc, cache_k)
