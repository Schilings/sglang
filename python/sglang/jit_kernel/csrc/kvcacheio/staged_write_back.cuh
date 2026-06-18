#pragma once

// ============================================================================
// staged_write_back.cuh — Page-first 布局下的两阶段写回管线
// ============================================================================
//
// 背景: 当 Host 端 KV cache 是 page_first 布局时，GPU 无法像 layer_first 那样
//       直接用 store_vec 一步写入 Host（目标地址不连续，无法用 UVA 直写）。
//       因此需要两阶段管线:
//
//   第一阶段 (GPU kernel — relayout):
//     Host pinned (page_first, 地址不连续)
//       → GPU staging buffer (layer_first, 地址连续)
//     由 launch_hicache_relayout_kernel 完成，利用 UVA 从 Host 读取数据，
//     在 GPU 端重排为连续布局写入 staging buffer。
//
//   第二阶段 (DMA — cudaMemcpyBatchAsync):
//     GPU staging buffer (连续)
//       → GPU dst KV cache (按 dst_indices scatter)
//     由 try_copy_page_first_pages_batch 完成:
//       - CUDA 12.8+: 用 cudaMemcpyBatchAsync 一次批量提交所有 page 的 DMA
//       - 低版本/不支持: 回退为逐页 cudaMemcpyAsync (copy_page_first_pages_fallback)
//
// 数据流示意:
//   Host pinned    GPU staging      GPU dst
//   (page_first)   (layer_first)    (layer_first, scatter by dst_indices)
//   ┌─────────┐    ┌──────────┐    ┌──────────┐
//   │ page 0  │───▶│ contig 0 │───▶│ dst[idx] │
//   │ page 1  │───▶│ contig 1 │───▶│ dst[idx] │
//   │ ...     │───▶│ ...      │───▶│ ...      │
//   └─────────┘    └──────────┘    └──────────┘
//    relayout        cudaMemcpyBatchAsync
//    (GPU kernel)    (driver DMA, 非 kernel)
//
// ============================================================================

#include "hicache.cuh"
#include "relayout.cuh"
#include <dlfcn.h>
#include <limits>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// cudaMemcpyBatchAsync 函数签名与动态加载
// ---------------------------------------------------------------------------
// CUDA 12.8 引入 cudaMemcpyBatchAsync, 可一次提交多对 (src, dst, size) 的拷贝,
// 驱动在内部做批量 DMA 调度, 比逐页 cudaMemcpyAsync 效率高得多。
// CUDA 13.0 移除了 failIdx 参数, 所以签名有 8 参数 / 9 参数两个版本。
// 通过 dlsym 运行时加载, 以兼容不同 CUDA 版本。
// ---------------------------------------------------------------------------
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
// CUDA >= 13.0: 8 参数版本 (无 failIdx)
#if CUDA_VERSION >= 13000
using CudaMemcpyBatchPtr = const void*;
using CudaMemcpyBatchAsyncFn = cudaError_t (*)(
    CudaMemcpyBatchPtr*,
    CudaMemcpyBatchPtr*,
    const size_t*,
    size_t,
    cudaMemcpyAttributes*,
    size_t*,
    size_t,
    cudaStream_t);
// CUDA 12.8~12.x: 9 参数版本 (有 failIdx)
#else
using CudaMemcpyBatchPtr = void*;
using CudaMemcpyBatchAsyncFn = cudaError_t (*)(
    CudaMemcpyBatchPtr*,
    CudaMemcpyBatchPtr*,
    size_t*,
    size_t,
    cudaMemcpyAttributes*,
    size_t*,
    size_t,
    size_t*,
    cudaStream_t);
#endif

// dlsym 动态加载 cudaMemcpyBatchAsync, 避免编译时硬依赖 CUDA 12.8+ 运行时
inline auto get_cuda_memcpy_batch_async() -> CudaMemcpyBatchAsyncFn {
  static CudaMemcpyBatchAsyncFn cuda_memcpy_batch_async = []() {
    void* symbol = dlsym(RTLD_DEFAULT, "cudaMemcpyBatchAsync");
    return reinterpret_cast<CudaMemcpyBatchAsyncFn>(symbol);
  }();
  return cuda_memcpy_batch_async;
}

// 统一调用入口: 根据 CUDA 版本选择 8 参数或 9 参数签名
inline auto call_cuda_memcpy_batch_async(
    CudaMemcpyBatchAsyncFn copy_fn,
    CudaMemcpyBatchPtr* dsts,
    CudaMemcpyBatchPtr* srcs,
    size_t* sizes,
    size_t count,
    cudaMemcpyAttributes* attrs,
    size_t* attrs_idxs,
    size_t num_attrs,
    cudaStream_t stream) -> cudaError_t {
#if CUDA_VERSION >= 13000
  return copy_fn(dsts, srcs, sizes, count, attrs, attrs_idxs, num_attrs, stream);
#else
  size_t fail_idx = std::numeric_limits<size_t>::max();
  return copy_fn(dsts, srcs, sizes, count, attrs, attrs_idxs, num_attrs, &fail_idx, stream);
#endif
}
#endif

// ---------------------------------------------------------------------------
// Fallback: 逐页 cudaMemcpyAsync (CUDA < 12.8 或 cudaMemcpyBatchAsync 不可用时)
// 每个 (tensor, page) 对独立发起一次 DMA, 效率低于批量版本。
// ---------------------------------------------------------------------------
inline void copy_page_first_pages_fallback(
    const std::vector<tvm::ffi::TensorView>& src_ptrs,
    std::vector<tvm::ffi::TensorView> dst_ptrs,
    const int64_t* dst_indices_ptr,
    int64_t num_pages,
    int64_t page_size,
    cudaStream_t stream) {
  using namespace host;

  RuntimeCheck(src_ptrs.size() == dst_ptrs.size(), "Source and destination tensors must have the same count");
  for (const auto tensor_id : irange(src_ptrs.size())) {
    RuntimeCheck(
        src_ptrs[tensor_id].dtype() == dst_ptrs[tensor_id].dtype(),
        "Source and destination tensors must have the same dtype");
    const int64_t elem_size = host::dtype_bytes(src_ptrs[tensor_id].dtype());
    const int64_t src_stride0 = src_ptrs[tensor_id].stride(0);
    const int64_t dst_stride0 = dst_ptrs[tensor_id].stride(0);
    const size_t src_page_bytes = static_cast<size_t>(page_size * src_stride0 * elem_size);
    const size_t dst_page_bytes = static_cast<size_t>(page_size * dst_stride0 * elem_size);
    RuntimeCheck(src_page_bytes == dst_page_bytes, "Source and destination page spans must match");
    for (const auto page_offset : irange(num_pages)) {
      // src: staging buffer 中第 page_offset 页的起始地址 (连续排列, stride=page_size)
      const char* src_ptr = static_cast<const char*>(src_ptrs[tensor_id].data_ptr()) +
                            static_cast<size_t>(page_offset * page_size * src_stride0 * elem_size);
      // dst: Host pinned memory 中 dst_indices 指定的目标位置 (可能不连续)
      char* dst_ptr = static_cast<char*>(dst_ptrs[tensor_id].data_ptr()) +
                      static_cast<size_t>(dst_indices_ptr[page_offset * page_size] * dst_stride0 * elem_size);
      RuntimeDeviceCheck(cudaMemcpyAsync(dst_ptr, src_ptr, src_page_bytes, cudaMemcpyDeviceToHost, stream));
    }
  }
}

// ---------------------------------------------------------------------------
// 批量 DMA: 用 cudaMemcpyBatchAsync 一次提交所有 page 的拷贝请求
// 返回 true 表示成功, false 表示不支持 (需 fallback)
// 前置条件: CUDA >= 12.8, driver >= 12.8, 单页 >= 128KB (kLargeCopyThresholdBytes)
// ---------------------------------------------------------------------------
inline bool try_copy_page_first_pages_batch(
    const std::vector<tvm::ffi::TensorView>& src_ptrs,
    std::vector<tvm::ffi::TensorView> dst_ptrs,
    const int64_t* dst_indices_ptr,
    int64_t num_pages,
    int64_t page_size,
    int device_id,
    cudaStream_t stream) {
#if defined(USE_ROCM) || !defined(CUDA_VERSION) || (CUDA_VERSION < 12080)
  return false;
#else
  host::RuntimeCheck(src_ptrs.size() == dst_ptrs.size(), "Source and destination tensors must have the same count");
  // 小于 128KB 的 page 用 cudaMemcpyBatchAsync 不划算, 走 fallback
  constexpr size_t kLargeCopyThresholdBytes = 128 * 1024;
  // thread_local 避免每次调用都分配内存
  thread_local std::vector<CudaMemcpyBatchPtr> batch_srcs;
  thread_local std::vector<CudaMemcpyBatchPtr> batch_dsts;
  thread_local std::vector<size_t> batch_sizes;

  // 检查 1: driver 版本 >= 12.8
  int driver_version = 0;
  cudaError_t driver_version_err = cudaDriverGetVersion(&driver_version);
  if (driver_version_err != cudaSuccess || driver_version < 12080) {
    return false;
  }

  // 检查 2: 运行时是否暴露了 cudaMemcpyBatchAsync 符号
  auto copy_fn = get_cuda_memcpy_batch_async();
  if (copy_fn == nullptr) {
    return false;
  }

  // 构造所有 page 的 (src, dst, size) 列表
  // num_copies = num_tensors * num_pages, 例: 2(K,V) * 256 = 512 次拷贝
  const size_t num_copies = static_cast<size_t>(src_ptrs.size()) * static_cast<size_t>(num_pages);
  batch_srcs.clear();
  batch_dsts.clear();
  batch_sizes.clear();
  batch_srcs.reserve(num_copies);
  batch_dsts.reserve(num_copies);
  batch_sizes.reserve(num_copies);

  size_t first_page_bytes = 0;
  for (const auto tensor_id : host::irange(src_ptrs.size())) {
    host::RuntimeCheck(
        src_ptrs[tensor_id].dtype() == dst_ptrs[tensor_id].dtype(),
        "Source and destination tensors must have the same dtype");
    const int64_t elem_size = host::dtype_bytes(src_ptrs[tensor_id].dtype());
    const int64_t src_stride0 = src_ptrs[tensor_id].stride(0);
    const int64_t dst_stride0 = dst_ptrs[tensor_id].stride(0);
    const size_t src_page_bytes = static_cast<size_t>(page_size * src_stride0 * elem_size);
    const size_t dst_page_bytes = static_cast<size_t>(page_size * dst_stride0 * elem_size);
    host::RuntimeCheck(src_page_bytes == dst_page_bytes, "Source and destination page spans must match");
    if (tensor_id == 0) {
      first_page_bytes = src_page_bytes;
    }
    for (const auto page_offset : host::irange(num_pages)) {
      // src: staging buffer 中第 page_offset 页 (连续排列)
      char* src_ptr = static_cast<char*>(src_ptrs[tensor_id].data_ptr()) +
                      static_cast<size_t>(page_offset * page_size * src_stride0 * elem_size);
      // dst: Host pinned memory 中 dst_indices 指定的位置 (scatter)
      char* dst_ptr = static_cast<char*>(dst_ptrs[tensor_id].data_ptr()) +
                      static_cast<size_t>(dst_indices_ptr[page_offset * page_size] * dst_stride0 * elem_size);
      batch_srcs.push_back(src_ptr);
      batch_dsts.push_back(dst_ptr);
      batch_sizes.push_back(src_page_bytes);
    }
  }
  // 检查 3: 单页太小 (<128KB) 则不使用批量 API
  if (first_page_bytes < kLargeCopyThresholdBytes) {
    return false;
  }

  // 设置 DMA 属性: Device → Host, 按流顺序访问源
  std::vector<size_t> attrs_idxs(1, 0);
  cudaMemcpyAttributes attrs{};
  attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;  // 保证与前面 relayout kernel 的顺序
  attrs.srcLocHint.type = cudaMemLocationTypeDevice;  // 源在 GPU 上
  attrs.srcLocHint.id = device_id;
  attrs.dstLocHint.type = cudaMemLocationTypeHost;    // 目标在 Host 上
  attrs.dstLocHint.id = 0;
  attrs.flags = 0;

  // 一次性提交所有 page 的 DMA 请求, 驱动内部做批量调度
  cudaError_t err = call_cuda_memcpy_batch_async(
      copy_fn,
      batch_dsts.data(),
      batch_srcs.data(),
      batch_sizes.data(),
      num_copies,
      &attrs,
      attrs_idxs.data(),
      1,
      stream);
  if (err == cudaErrorNotSupported || err == cudaErrorCallRequiresNewerDriver || err == cudaErrorInvalidValue) {
    (void)cudaGetLastError();
    return false;
  }
  host::RuntimeCheck(err == cudaSuccess, "cudaMemcpyBatchAsync failed. error=", cudaGetErrorString(err));
  return true;
#endif
}

// ---------------------------------------------------------------------------
// HiCacheStagedWriteBackKernel — 两阶段写回管线的入口
// ---------------------------------------------------------------------------
// 模板参数:
//   kElementSize: 单个 (head_num * head_dim) 的字节数
//   kUnroll: 每个 warp 拆成多少个 worker
//   kBlockQuota / kBlockSize: block 资源配置
// ---------------------------------------------------------------------------
template <int64_t kElementSize, uint32_t kUnroll, uint32_t kBlockQuota, uint32_t kBlockSize>
struct HiCacheStagedWriteBackKernel {
 private:
  // =======================================================================
  // run_staged_impl — 两阶段管线的核心实现
  // =======================================================================
  // 第一阶段: relayout kernel (GPU)
  //   从 Host pinned 读取 page_first 数据, 重排后写入 GPU staging buffer (连续)
  //   ★ 这是 "读 Host" 方向, 用 UVA 的 load_vec 实现
  //
  // 第二阶段: cudaMemcpyBatchAsync (Driver DMA)
  //   从 GPU staging buffer 批量 DMA 到 GPU dst KV cache (按 dst_indices scatter)
  //   ★ 这是 "GPU→GPU" 方向, 用 driver 的批量 DMA 实现
  //   ★ 如果不支持, fallback 为逐页 cudaMemcpyAsync
  //
  // 为什么不直接 Host→GPU dst?
  //   因为 dst 是按 dst_indices scatter 的, 地址不连续,
  //   cudaMemcpyBatchAsync 可以处理不连续目标, 但需要 GPU 端的连续 staging
  //   作为中转, 保证 DMA 效率。
  // =======================================================================
  template <bool kIsMLA>
  static void run_staged_impl(
      const tvm::ffi::TensorView k_cache_dst,
      const tvm::ffi::TensorView v_cache_dst,
      const tvm::ffi::TensorView dst_indices_cpu,
      const tvm::ffi::TensorView staging_k,
      const tvm::ffi::TensorView staging_v,
      const tvm::ffi::TensorView page_indices_src,
      const tvm::ffi::TensorView k_ptr_src,
      const tvm::ffi::TensorView v_ptr_src,
      const int64_t page_size) {
    using namespace host;

    auto T = SymbolicSize{"num_tokens"};
    auto N = SymbolicSize{"num_layers"};
    auto D = SymbolicSize{"element_dim"};
    auto P = SymbolicSize{"num_pages"};
    auto cache_dtype = SymbolicDType{};
    auto indices_dtype = SymbolicDType{};
    auto dst_indices_dtype = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({T, N, D})  //
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA>(device_)
        .verify(staging_k);
    if constexpr (!kIsMLA) {
      TensorMatcher({T, N, D})  //
          .with_dtype(cache_dtype)
          .with_device<kDLCUDA>(device_)
          .verify(staging_v);
    }
    TensorMatcher({-1, N, D})  //
        .with_dtype(cache_dtype)
        .with_device<kDLCPU, kDLCUDAHost>()
        .verify(k_cache_dst);
    if constexpr (!kIsMLA) {
      TensorMatcher({-1, N, D})  //
          .with_dtype(cache_dtype)
          .with_device<kDLCPU, kDLCUDAHost>()
          .verify(v_cache_dst);
    }
    TensorMatcher({N})  //
        .with_dtype<uint64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(k_ptr_src);
    if constexpr (!kIsMLA) {
      TensorMatcher({N})  //
          .with_dtype<uint64_t>()
          .with_device<kDLCUDA>(device_)
          .verify(v_ptr_src);
    }
    TensorMatcher({P})  //
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLCUDA>(device_)
        .verify(page_indices_src);
    TensorMatcher({T})  //
        .with_dtype<int64_t>(dst_indices_dtype)
        .with_device<kDLCPU, kDLCUDAHost>()
        .verify(dst_indices_cpu);

    RuntimeCheck(page_size > 0, "HiCache staged relayout: page_size must be positive");
    RuntimeCheck(T.unwrap() == P.unwrap() * page_size, "HiCache staged relayout: staging token count mismatch");
    RuntimeCheck(
        kElementSize == D.unwrap() * dtype_bytes(cache_dtype.unwrap()),
        "HiCache staged relayout: element size mismatch");
    RuntimeCheck(kElementSize % 16 == 0, "HiCache staged relayout: element size must be 16-byte aligned");

    // =====================================================================
    // 第一阶段: relayout kernel — Host pinned → GPU staging buffer
    // =====================================================================
    // 从 Host pinned (page_first, 不连续) 读取 KV 数据,
    // 重排为 layer_first (连续) 写入 GPU staging buffer。
    // k_cache_dst / v_cache_dst 在此处是 Host 端的源, 通过 UVA 读取。
    const auto params = HicacheRelayoutParams{
        .k_cache_dst = staging_k.data_ptr(),          // 写入目标: GPU staging
        .v_cache_dst = kIsMLA ? nullptr : staging_v.data_ptr(),
        .indices_src = page_indices_src.data_ptr(),    // 每页在 Host 中的索引
        .k_ptr_src = k_ptr_src.data_ptr(),            // 每层的 Host base address
        .v_ptr_src = kIsMLA ? nullptr : v_ptr_src.data_ptr(),
        .num_pages = static_cast<uint32_t>(P.unwrap()),
        .num_layers = static_cast<uint32_t>(N.unwrap()),
        .page_size = static_cast<uint32_t>(page_size),
    };
    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype.unwrap().bits == 32;
    launch_hicache_relayout_kernel<kElementSize, kIsMLA>(params, P.unwrap(), N.unwrap(), page_size, use_int32, device);

    // =====================================================================
    // 第二阶段: 批量 DMA — GPU staging buffer → GPU dst KV cache
    // =====================================================================
    // staging buffer 中的数据已是连续排列, 按页批量拷贝到 dst 的 scatter 位置。
    // 优先使用 cudaMemcpyBatchAsync (一次提交所有页), 不支持则逐页 fallback。
    auto stream = LaunchKernel::resolve_device(device);
    const int64_t* dst_indices_ptr = static_cast<const int64_t*>(dst_indices_cpu.data_ptr());
    if constexpr (kIsMLA) {
      // MLA: 只有 K (即合并的 K=V), 无单独 V
      if (!try_copy_page_first_pages_batch(
              {staging_k}, {k_cache_dst}, dst_indices_ptr, P.unwrap(), page_size, device.device_id, stream)) {
        copy_page_first_pages_fallback({staging_k}, {k_cache_dst}, dst_indices_ptr, P.unwrap(), page_size, stream);
      }
    } else {
      // MHA: K 和 V 分别搬运
      if (!try_copy_page_first_pages_batch(
              {staging_k, staging_v},
              {k_cache_dst, v_cache_dst},
              dst_indices_ptr,
              P.unwrap(),
              page_size,
              device.device_id,
              stream)) {
        copy_page_first_pages_fallback(
            {staging_k, staging_v}, {k_cache_dst, v_cache_dst}, dst_indices_ptr, P.unwrap(), page_size, stream);
      }
    }
  }

 public:
  // MHA staged 写回入口 (K, V 分离)
  static void run_all_lf_pf_staged(
      const tvm::ffi::TensorView k_cache_dst,
      const tvm::ffi::TensorView v_cache_dst,
      const tvm::ffi::TensorView dst_indices_cpu,
      const tvm::ffi::TensorView staging_k,
      const tvm::ffi::TensorView staging_v,
      const tvm::ffi::TensorView page_indices_src,
      const tvm::ffi::TensorView k_ptr_src,
      const tvm::ffi::TensorView v_ptr_src,
      const int64_t page_size) {
    run_staged_impl<false>(
        k_cache_dst,
        v_cache_dst,
        dst_indices_cpu,
        staging_k,
        staging_v,
        page_indices_src,
        k_ptr_src,
        v_ptr_src,
        page_size);
  }

  // MLA staged 写回入口 (K=V 合并, 只有单个 cache tensor)
  static void run_all_mla_lf_pf_staged(
      const tvm::ffi::TensorView cache_dst,
      const tvm::ffi::TensorView dst_indices_cpu,
      const tvm::ffi::TensorView staging,
      const tvm::ffi::TensorView page_indices_src,
      const tvm::ffi::TensorView ptr_src,
      const int64_t page_size) {
    run_staged_impl<true>(
        cache_dst, cache_dst, dst_indices_cpu, staging, staging, page_indices_src, ptr_src, ptr_src, page_size);
  }
};

}  // namespace
