# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⚡ HiCache JIT Kernels —— GPU↔Host DMA 的自定义 CUDA/Triton kernel                   ║
# ║  transfer_hicache_all_layer / transfer_hicache_one_layer / ...                       ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module

# 每个 SM 最多同时运行的 block 数，控制 DMA kernel 对 SM 的占用率，防止挤占计算资源
DEFAULT_BLOCK_QUOTA = 2


# ┌─────────────────────────────────────────────────────────────────────┐
# _jit_hicache_module: JIT 编译 HiCache CUDA kernel 并缓存          │
#                                                                     │
# 整体流程:                                                           │
#   Python 参数 ──→ C++ 模板参数 ──→ JIT 编译 .cuh ──→ 导出 Python 函数│
#                                                                     │
#   element_size ─┐                                                   │
#   unroll        ├─→ make_cpp_args ──→ "element_size, unroll,        │
#   block_quota   │      block_quota, num_threads"     block_quota,   │
#   num_threads  ─┘                            num_threads"           │
#                                    │                                │
#                                    ▼                                │
#   C++ 模板实例化:  HiCacheKernel<8192, 4, 2, 1024>                  │
#                                    │                                │
#                                    ▼                                │
#   load_jit 编译 .cuh → 生成 .so → 包装为 Python 可调用的 module     │
#                                                                     │
# @cache_once: 同一组参数只编译一次，后续调用直接返回缓存的 Module     │
# └─────────────────────────────────────────────────────────────────────┘
@cache_once
def _jit_hicache_module(*, element_size: int, unroll: int, block_quota: int) -> Module:
    # 将 Python 参数转为 C++ 模板参数字符串，如 "8192, 4, 2, 1024"
    args = make_cpp_args(
        element_size,       # 每个 token 每层的 KV 字节数 = head_num × head_dim × dtype_size
        unroll,             # 循环展开次数，小 element_size 展开更多（减少循环开销）
        block_quota,        # 每 SM 并发 block 上限，限制 DMA kernel 占用率
        1024,               # num_threads, 每个 block 的线程数，可调优
    )
    return load_jit(
        "hicache",          # kernel 名称标识
        *args,
        # 需要编译的 CUDA 头文件列表
        cuda_files=[
            "kvcacheio/hicache.cuh",              # 主 kernel: HiCacheKernel (单层/全层 DMA)
            "kvcacheio/relayout.cuh",             # 布局重排辅助
            "kvcacheio/staged_write_back.cuh",     # 分阶段写回 kernel: HiCacheStagedWriteBackKernel
        ],
        # Python 名称 → C++ 函数符号 的映射表
        # 每个元组: (Python 调用名, C++ mangled symbol)
        # f"&HiCacheKernel<{args}>::run_one" 会被解析为 C++ 模板实例化的静态方法指针
        cuda_wrappers=[
            # ── 标准 MHA (K/V 分开存储) ──
            ("launch_one",     f"&HiCacheKernel<{args}>::run_one"),       # 单层 DMA (load_one_layer 调用)
            ("launch_all",     f"&HiCacheKernel<{args}>::run_all"),       # 全层 DMA (load_all_layer 调用)
            # ── MLA (K/V 合并存储，只有一个 cache tensor) ──
            ("launch_one_mla", f"&HiCacheKernel<{args}>::run_one_mla"),   # MLA 单层
            ("launch_all_mla", f"&HiCacheKernel<{args}>::run_all_mla"),   # MLA 全层
            # ── 分阶段写回 (Staged Write-Back: 先写 staging buffer, 再异步拷贝到目标) ──
            # lf = layer_first 布局, pf = page_first 布局, staged = 分阶段
            (
                "launch_all_lf_pf_staged",
                f"&HiCacheStagedWriteBackKernel<{args}>::run_all_lf_pf_staged",
            ),
            (
                "launch_all_mla_lf_pf_staged",
                f"&HiCacheStagedWriteBackKernel<{args}>::run_all_mla_lf_pf_staged",
            ),
        ],
    )


def can_use_hicache_jit_kernel(
    *,
    element_size: int,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> bool:
    """检查当前环境能否使用 JIT 编译的 HiCache DMA kernel。

    两个条件必须同时满足:
      1. element_size 必须是 128 的倍数 (TMA 对齐要求, 对应 128B cache line)
      2. JIT 编译不抛异常 (CUDA 版本兼容、驱动正常等)

    主流模型 (head_dim=128, fp16/bf16) 几乎都满足条件 1。
    """
    logger = logging.getLogger(__name__)
    if element_size % 128 != 0:
        logger.warning(f"Unsupported {element_size = } for JIT HiCache kernel")
        return False
    try:
        unroll = unroll or _default_unroll(element_size)
        block_quota = block_quota or DEFAULT_BLOCK_QUOTA
        _jit_hicache_module(
            element_size=element_size,
            unroll=unroll,
            block_quota=block_quota,
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT HiCache kernel: {e}")
        return False


def _default_unroll(element_size: int) -> int:
    """根据 element_size 选择循环展开次数。

    element_size 小时多展开 (减少循环开销), 大时少展开 (避免寄存器溢出)。
    - ≤ 512B: unroll=4 (每个 warp 处理 4 个 token 的数据)
    - ≤ 1024B: unroll=2
    - > 1024B: unroll=1 (不展开)
    """
    if element_size <= 512:
        return 4

    if element_size <= 1024:
        return 2

    # fallback: no unroll
    return 1


# ═══════════════════════════════════════════════════════════════════════
#  高层 Python API: 按 "单层/全层 × MHA/MLA × 直传/分阶段" 共 6 个函数
#
#  调用链: Python 函数 → _jit_hicache_module() → module.launch_xxx()
#           → C++ HiCacheKernel::run_xxx() (参数校验+组装)
#           → CUDA hicache_transfer_per_layer / hicache_transfer_all_layer (实际 GPU kernel)
#
#  CUDA kernel 核心逻辑 (非缓存 DMA, 不污染 L1 cache):
#    for token i (并行):
#        pos_src = indices_src[i]       # 源 page 编号
#        pos_dst = indices_dst[i]       # 目标 page 编号
#        load_vec<kElementSize>(src + pos_src * stride)   # PTX ld.global.L1::no_allocate
#        store_vec<kElementSize>(dst + pos_dst * stride)  # PTX st.global.L1::no_allocate
# ═══════════════════════════════════════════════════════════════════════

@debug_kernel_api
def transfer_hicache_one_layer(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    element_dim: int | None = None,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> None:
    """单层 K/V DMA 拷贝 (MHA 模式, K 和 V 分开存储)。

    调用链:
      module.launch_one → HiCacheKernel::run_one → hicache_transfer_per_layer (CUDA)
    用于: load_one_layer 场景, 只搬运某一层的 KV cache。
    """
    element_dim = element_dim or k_cache_dst.size(-1)
    k_cache_src = k_cache_src.view(-1, element_dim)
    v_cache_src = v_cache_src.view(-1, element_dim)
    k_cache_dst = k_cache_dst.view(-1, element_dim)
    v_cache_dst = v_cache_dst.view(-1, element_dim)
    element_size = element_dim * k_cache_dst.element_size()
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_one(
        k_cache_dst,
        v_cache_dst,
        indices_dst,
        k_cache_src,
        v_cache_src,
        indices_src,
    )


@debug_kernel_api
def transfer_hicache_all_layer(
    k_ptr_dst: torch.Tensor,
    v_ptr_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_ptr_src: torch.Tensor,
    v_ptr_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    kv_cache_src_stride_bytes: int,
    kv_cache_dst_stride_bytes: int,
    element_size: int | None = None,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> None:
    """全层 K/V DMA 拷贝 (MHA 模式, K 和 V 分开存储)。

    调用链:
      module.launch_all → HiCacheKernel::run_all → hicache_transfer_all_layer (CUDA)
    用于: load_all_layer 场景, 一次 kernel 调用搬运所有层的 KV cache。

    与 one_layer 的区别:
      - one_layer: k_cache_src 是 2D tensor [num_pages, element_dim]
      - all_layer: k_ptr_src 是 1D 指针数组 [num_layers], 每个元素指向该层的 cache base
      - all_layer 多一层内循环: for layer in range(num_layers)
    """
    if element_size is None:  # assume both contiguous
        assert kv_cache_dst_stride_bytes == kv_cache_src_stride_bytes
        element_size = kv_cache_dst_stride_bytes

    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_all(
        k_ptr_dst,
        v_ptr_dst,
        indices_dst,
        k_ptr_src,
        v_ptr_src,
        indices_src,
        kv_cache_src_stride_bytes,
        kv_cache_dst_stride_bytes,
    )


def transfer_hicache_one_layer_mla(
    cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    element_dim: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    """单层 KV DMA 拷贝 (MLA 模式, K/V 合并存储为单个 cache tensor)。

    调用链:
      module.launch_one_mla → HiCacheKernel::run_one_mla
        → hicache_transfer_per_layer<..., kIsMLA=true> (CUDA, 跳过 V 拷贝)
    """
    element_dim = element_dim or cache_dst.size(-1)
    cache_src = cache_src.view(-1, element_dim)
    cache_dst = cache_dst.view(-1, element_dim)
    element_size = element_dim * cache_dst.element_size()
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_one_mla(
        cache_dst,
        indices_dst,
        cache_src,
        indices_src,
    )


def transfer_hicache_all_layer_mla(
    ptr_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    ptr_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    cache_src_stride_bytes: int,
    cache_dst_stride_bytes: int,
    element_size: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    """全层 KV DMA 拷贝 (MLA 模式, K/V 合并存储, 指针数组)。

    调用链:
      module.launch_all_mla → HiCacheKernel::run_all_mla
        → hicache_transfer_all_layer<..., kIsMLA=true> (CUDA, 跳过 V 拷贝)
    """
    if element_size is None:
        assert cache_dst_stride_bytes == cache_src_stride_bytes
        element_size = cache_dst_stride_bytes

    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_all_mla(
        ptr_dst,
        indices_dst,
        ptr_src,
        indices_src,
        cache_src_stride_bytes,
        cache_dst_stride_bytes,
    )


@debug_kernel_api
def transfer_hicache_all_layer_staged_lf_pf(
    k_ptr_src: torch.Tensor,
    v_ptr_src: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    staging_k: torch.Tensor,
    staging_v: torch.Tensor,
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    *,
    page_size: int,
    element_size: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    """分阶段写回: Host → staging buffer (GPU) → 目标 KV cache (GPU)。

    调用链:
      module.launch_all_lf_pf_staged → HiCacheStagedWriteBackKernel::run_all_lf_pf_staged

    与直传 (transfer_hicache_all_layer) 的区别:
      直传:   src → dst (一步到位)
      staged: src → staging buffer → dst (两步, 中间经过一个小的 staging buffer)

    为什么需要 staging:
      Host pinned memory → GPU 时, 数据先落入 staging buffer,
      再由第二个 kernel 从 staging 搬到最终目标位置, 避免 Host→GPU 直接写时的
      对齐/跨页问题。

    分批处理:
      staging buffer 大小有限 (staging_page_capacity 页),
      超出时按 staging_page_capacity 分批调用 kernel。
    """
    element_dim = staging_k[0, 0].numel()
    element_size = element_size or (element_dim * staging_k.element_size())
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    # src_indices 是 per-token 索引 (同一 page 内所有 token 索引相同)，
    # [::page_size] 降采样为 per-page 索引: 取每个 page 第一个 token 的索引作为代表，
    # 结果长度 = num_pages (不是只取 1 页!)。
    # 例: page_size=4, src_indices=[10,10,10,10, 20,20,20,20] → [10, 20]
    src_page_indices = src_indices[::page_size].contiguous()
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    # staging buffer 能容纳多少个 page (远大于 1)
    # 例: staging_k 有 4096 token, page_size=16 → staging_page_capacity=256
    staging_page_capacity = staging_k.shape[0] // page_size
    staging_k = staging_k.view(staging_k.shape[0], staging_k.shape[1], -1)
    staging_v = staging_v.view(staging_v.shape[0], staging_v.shape[1], -1)
    dst_k = dst_k.view(dst_k.shape[0], dst_k.shape[1], -1)
    dst_v = dst_v.view(dst_v.shape[0], dst_v.shape[1], -1)
    # 按 staging buffer 容量分批搬运, 步长 = staging_page_capacity (非逐页!)
    # 例: 总 1000 页, staging 容量 256 页 → 分 4 批: [0,256), [256,512), [512,768), [768,1000)
    for page_begin in range(0, src_page_indices.numel(), staging_page_capacity):
        chunk_pages = min(staging_page_capacity, src_page_indices.numel() - page_begin)
        chunk_tokens = chunk_pages * page_size
        module.launch_all_lf_pf_staged(
            dst_k,
            dst_v,
            dst_indices[
                page_begin * page_size : (page_begin + chunk_pages) * page_size
            ],
            staging_k[:chunk_tokens],
            staging_v[:chunk_tokens],
            src_page_indices[page_begin : page_begin + chunk_pages],
            k_ptr_src,
            v_ptr_src,
            page_size,
        )


@debug_kernel_api
def transfer_hicache_all_layer_mla_staged_lf_pf(
    ptr_src: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    staging: torch.Tensor,
    dst: torch.Tensor,
    *,
    page_size: int,
    element_size: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    """分阶段写回 (MLA 模式): Host → staging buffer → 目标 KV cache。

    与 transfer_hicache_all_layer_staged_lf_pf 的区别:
      - MLA 模式下 K/V 合并存储, 只有单个 cache tensor 和单个 staging tensor
    其余逻辑 (分批、staging buffer 容量限制) 完全一致。
    """
    element_dim = staging[0, 0].numel()
    element_size = element_size or (element_dim * staging.element_size())
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    src_page_indices = src_indices[::page_size].contiguous()
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    staging_page_capacity = staging.shape[0] // page_size
    staging = staging.view(staging.shape[0], staging.shape[1], -1)
    dst = dst.view(dst.shape[0], dst.shape[1], -1)
    for page_begin in range(0, src_page_indices.numel(), staging_page_capacity):
        chunk_pages = min(staging_page_capacity, src_page_indices.numel() - page_begin)
        chunk_tokens = chunk_pages * page_size
        module.launch_all_mla_lf_pf_staged(
            dst,
            dst_indices[
                page_begin * page_size : (page_begin + chunk_pages) * page_size
            ],
            staging[:chunk_tokens],
            src_page_indices[page_begin : page_begin + chunk_pages],
            ptr_src,
            page_size,
        )
