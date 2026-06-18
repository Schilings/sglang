#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace device {

namespace details {

template <typename T, uint32_t N>
struct LocalStorage {
  T data[N];
};

template <int kUnit>
inline constexpr auto get_mem_package() {
  if constexpr (kUnit == 16) {
    return uint4{};
  } else if constexpr (kUnit == 8) {
    return uint2{};
  } else if constexpr (kUnit == 4) {
    return uint1{};
  } else {
    static_assert(kUnit == 16 || kUnit == 8 || kUnit == 4, "Unsupported memory package size");
  }
}

template <int kUnit>
using PackageType = decltype(get_mem_package<kUnit>());

SGL_DEVICE uint1 load_nc(const uint1* __restrict__ src) {
  uint32_t tmp;
  asm volatile("ld.global.L1::no_allocate.b32 %0,[%1];" : "=r"(tmp) : "l"(src));
  return uint1{tmp};
}

SGL_DEVICE uint2 load_nc(const uint2* __restrict__ src) {
  uint32_t tmp0, tmp1;
  asm volatile("ld.global.L1::no_allocate.v2.b32 {%0,%1},[%2];" : "=r"(tmp0), "=r"(tmp1) : "l"(src));
  return uint2{tmp0, tmp1};
}

SGL_DEVICE uint4 load_nc(const uint4* __restrict__ src) {
  uint32_t tmp0, tmp1, tmp2, tmp3;
  asm volatile("ld.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3},[%4];"
               : "=r"(tmp0), "=r"(tmp1), "=r"(tmp2), "=r"(tmp3)
               : "l"(src));
  return uint4{tmp0, tmp1, tmp2, tmp3};
}

SGL_DEVICE void store_nc(uint1* __restrict__ dst, const uint1& value) {
  uint32_t tmp = value.x;
  asm volatile("st.global.L1::no_allocate.b32 [%0],%1;" ::"l"(dst), "r"(tmp));
}

SGL_DEVICE void store_nc(uint2* __restrict__ dst, const uint2& value) {
  uint32_t tmp0 = value.x;
  uint32_t tmp1 = value.y;
  asm volatile("st.global.L1::no_allocate.v2.b32 [%0],{%1,%2};" ::"l"(dst), "r"(tmp0), "r"(tmp1));
}

SGL_DEVICE void store_nc(uint4* __restrict__ dst, const uint4& value) {
  uint32_t tmp0 = value.x;
  uint32_t tmp1 = value.y;
  uint32_t tmp2 = value.z;
  uint32_t tmp3 = value.w;
  asm volatile(
      "st.global.L1::no_allocate.v4.b32 [%0],{%1,%2,%3,%4};" ::"l"(dst), "r"(tmp0), "r"(tmp1), "r"(tmp2), "r"(tmp3));
}

}  // namespace details

// ══════════════════════════════════════════════════════════════════════════
// HiCache DMA Kernel: GPU ↔ Host pinned memory 的零拷贝直传机制
// ══════════════════════════════════════════════════════════════════════════
//
// 核心原理: CUDA 统一虚拟寻址 (Unified Virtual Addressing, UVA)
//   当 Host 端使用 cudaMallocHost / cudaHostAlloc 分配 pinned memory 时,
//   GPU kernel 可以直接通过 st.global 写入该地址, 驱动自动转为 PCIe DMA 传输,
//   无需额外的 cudaMemcpy 调用。
//
// ┌─────────────────────────────────────────────────────────────┐
// │  GPU Kernel 内部数据流 (以 GPU→Host 写回为例):                  │
// │                                                             │
// │  GPU VRAM                    寄存器              Host Pinned │
// │  ┌──────────┐   load_vec    ┌───┐   store_vec  ┌──────────┐ │
// │  │ k_cache  │ ──────────→   │vec│ ──────────→  │ k_buffer │ │
// │  │ (src)    │  ld.global    │   │  st.global   │ (dst)    │ │
// │  └──────────┘               └───┘              └──────────┘ │
// │       GPU                    GPU               CPU pinned   │
// │                                                 (UVA可见)    │
// │  一次 kernel launch 完成 scatter DMA, 无需中间 cudaMemcpy      │
// └─────────────────────────────────────────────────────────────┘
//
// 不同布局的搬运策略:
//   layer_first: GPU→Host 直传 (store_vec 直接写 Host pinned memory)
//     原因: 每层 cache 在内存中连续, st.global 写入对齐良好, PCIe 效率高
//   page_first:  GPU→staging buffer→Host (两阶段)
//     原因: 同一 page 数据跨层不连续, 先 gather 到连续 staging buffer,
//           再整块拷贝, 避免大量零散 PCIe 小传输
//
// 反向 (Host→GPU 加载) 原理相同: ld.global 从 Host pinned memory 直读
// ══════════════════════════════════════════════════════════════════════════

// ── 向量化加载/存储: kNumThreads 个线程协作搬运 kBytes 字节 ──
//
// 以 kBytes=8192 (1 token K/V), kNumThreads=8 为例:
//   kLoopCount = 8192 / 128 = 64 次迭代
//   Package = 128 / 8 = 16B = uint4 (每线程每次搬运 16B)
//   每次迭代: 8 线程 × 16B = 128B = 1 个 cache line
//   64 次迭代: 64 × 128B = 8192B = 1 token 完整数据
//
// 使用 ld.global.L1::no_allocate (非缓存加载), 不污染 L1 cache,
// 因为 DMA 搬运的数据不需要被后续计算复用。
//
// store_vec 的 st.global.L1::no_allocate 同理:
//   - 目标在 GPU VRAM → 普通 global store
//   - 目标在 Host pinned memory → 硬件自动走 PCIe DMA 通道 (UVA)
//   - 两种情况对 kernel 代码完全透明, 由 CUDA 驱动处理
template <int64_t kBytes, uint32_t kNumThreads>
SGL_DEVICE auto load_vec(const void* __restrict__ src) {
  static_assert(kBytes % 128 == 0, "kBytes must be multiple of 128 bytes");
  static_assert(128 % kNumThreads == 0, "kNumThreads must divide 128 bytes");
  constexpr uint32_t kLoopCount = kBytes / 128;        // 迭代次数 = 总字节数 / 每次迭代128B
  using Package = details::PackageType<128 / kNumThreads>;  // 每线程每次搬运的字节类型
  using Storage = details::LocalStorage<Package, kLoopCount>; // 寄存器存储空间

  const auto src_packed = static_cast<const Package*>(src);
  const auto lane_id = threadIdx.x % kNumThreads;  // 线程在 Worker 内的 ID
  Storage vec;

#pragma unroll kLoopCount
  for (uint32_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kNumThreads + lane_id;  // 当前线程负责的数据偏移
    vec.data[i] = details::load_nc(&src_packed[j]);  // 非缓存加载
  }

  return vec;
}

// store_vec: 将 load_vec 读入的寄存器数据写回目标地址, 逻辑对称
template <int64_t kBytes, uint32_t kNumThreads, typename Storage>
SGL_DEVICE void store_vec(void* __restrict__ dst, const Storage& vec) {
  using Package = std::decay_t<decltype(vec.data[0])>;
  constexpr uint32_t kBytesPerLoop = sizeof(Package) * kNumThreads;
  constexpr uint32_t kLoopCount = kBytes / kBytesPerLoop;
  static_assert(kBytes % kBytesPerLoop == 0, "Invalid Storage configuration");

  const auto dst_packed = static_cast<Package*>(dst);
  const auto lane_id = threadIdx.x % kNumThreads;

#pragma unroll kLoopCount
  for (uint32_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kNumThreads + lane_id;
    details::store_nc(&dst_packed[j], vec.data[i]);  // 非缓存存储
  }
}

}  // namespace device

namespace {

#define SGL_HICACHE_KERNEL __global__ __launch_bounds__(kBlockSize, 1)

struct HicacheKernelParams {
  void* __restrict__ k_cache_dst;
  void* __restrict__ v_cache_dst;
  const void* __restrict__ indices_dst;
  void* __restrict__ k_cache_src;
  void* __restrict__ v_cache_src;
  const void* __restrict__ indices_src;
  int64_t kv_cache_src_stride;
  int64_t kv_cache_dst_stride;
  uint32_t length;
  uint32_t num_layers = 0;  // only used in all_layer transfer
};

// 单层 KV cache DMA 拷贝 kernel
// 与 all_layer 版本的分工模型完全相同, 区别仅在于:
//   - per_layer: k_cache_src/v_cache_src 是直接的 cache base 指针, 只拷 1 层
//   - all_layer: k_ptr_src/v_ptr_src 是指针数组 [num_layers], 需要内循环遍历所有层
template <
    typename T,
    int64_t kElementSize,
    uint32_t kUnroll,
    uint32_t kBlockQuota,
    uint32_t kBlockSize,
    bool kIsMLA = false>
SGL_HICACHE_KERNEL void hicache_transfer_per_layer(const __grid_constant__ HicacheKernelParams params) {
  using namespace device;
  static_assert(kBlockSize % kWarpThreads == 0);
  static_assert(kWarpThreads % kUnroll == 0);

  constexpr uint32_t kNumThreads = kWarpThreads / kUnroll;
  constexpr uint32_t kWorkersPerBlock = kBlockSize / kNumThreads;
  constexpr uint32_t kNumWorkers = kWorkersPerBlock * kBlockQuota;

  const auto& [
    k_cache_dst, v_cache_dst, indices_dst, // dst
    k_cache_src, v_cache_src, indices_src, // src
    kv_cache_src_stride, kv_cache_dst_stride, length, _ // metadata (num_layers 未使用)
  ] = params;

  // Worker ID + stride 遍历 (与 all_layer 相同的分工模型)
  const uint32_t work_id = blockIdx.x * kWorkersPerBlock + threadIdx.x / kNumThreads;
  for (uint32_t i = work_id; i < length; i += kNumWorkers) {
    const auto pos_src = static_cast<const T*>(indices_src)[i];
    const auto pos_dst = static_cast<const T*>(indices_dst)[i];
    // 直接用 base + offset 计算地址 (只有 1 层, 无内循环)
    const auto src_k = pointer::offset(k_cache_src, pos_src * kv_cache_src_stride);
    const auto dst_k = pointer::offset(k_cache_dst, pos_dst * kv_cache_dst_stride);
    const auto vec_k = load_vec<kElementSize, kNumThreads>(src_k);
    store_vec<kElementSize, kNumThreads>(dst_k, vec_k);
    if constexpr (!kIsMLA) {
      const auto src_v = pointer::offset(v_cache_src, pos_src * kv_cache_src_stride);
      const auto dst_v = pointer::offset(v_cache_dst, pos_dst * kv_cache_dst_stride);
      const auto vec_v = load_vec<kElementSize, kNumThreads>(src_v);
      store_vec<kElementSize, kNumThreads>(dst_v, vec_v);
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════
// 并行分工模型 (以 kElementSize=8192, kUnroll=4, kBlockQuota=2, kBlockSize=1024 为例)
//
// 层级: GPU → Block → Worker → Thread
//
// ┌─────────────────────────────────────────────────────────────────────┐
// │  GPU: 最多启动 kBlockQuota=2 个 Block                                │
// │  ┌──────────────────────────┐  ┌──────────────────────────┐        │
// │  │ Block 0 (1024 threads)   │  │ Block 1 (1024 threads)   │        │
// │  │ ┌───────┐ ┌───────┐      │  │ ┌───────┐ ┌───────┐     │        │
// │  │ │Worker0│ │Worker1│ ...  │  │ │Worker0│ │Worker1│ ... │        │
// │  │ │8 thr  │ │8 thr  │      │  │ │8 thr  │ │8 thr  │     │        │
// │  │ └───────┘ └───────┘      │  │ └───────┘ └───────┘     │        │
// │  │  共 128 Workers/Block     │  │  共 128 Workers/Block   │        │
// │  └──────────────────────────┘  └──────────────────────────┘        │
// │                                                                    │
// │  总 Worker 数 = kWorkersPerBlock × kBlockQuota = 128 × 2 = 256    │
// │  每个 Worker 由 kNumThreads=8 个线程协作搬运 1 个 token 的数据         │
// │  Worker i 负责 token: i, i+256, i+512, ... (stride=256)           │
// └─────────────────────────────────────────────────────────────────────┘
//
// Worker 内部: 8 线程协作搬运 8192B
//   每次迭代 128B (一个 cache line), 8192/128 = 64 次迭代
//   每个线程每次迭代搬运 128/8 = 16B (一个 uint4)
//   使用 ld.global.L1::no_allocate (非缓存读, 不污染 L1)
//
// 为什么 kUnroll 存在:
//   kNumThreads = kWarpThreads(32) / kUnroll
//   unroll=4 → kNumThreads=8 → 1 个 warp 分裂为 4 个 worker 并行处理 4 个 token
//   unroll=1 → kNumThreads=32 → 1 个 warp 只有 1 个 worker, 处理 1 个 token
//   element_size 大时减少并行度, 避免 register spillover
// ══════════════════════════════════════════════════════════════════════════
//
// Q: 外循环是 token、内循环是 layer, 单个 Worker 跳层写不是访存不连续吗?
// A: 不会! 因为所有 Worker 并行执行, 从内存子系统的视角看:
//
//    单 Worker 视角 (看起来不连续):
//      Worker 0: store Layer0_page0 → store Layer1_page0 → store Layer2_page0
//                地址跳跃大, 但这只是指令顺序
//
//    内存子系统视角 (实际是连续的):
//      同一时刻, 256 个 Worker 同时发射 store:
//        Worker 0   → Layer0 + page0
//        Worker 1   → Layer0 + page1    ← 写 Layer0 的请求, page 连续!
//        Worker 2   → Layer0 + page2
//        ...
//        Worker 255 → Layer0 + page255
//      下一轮:
//        Worker 0   → Layer1 + page0    ← 写 Layer1 的请求, 也是连续!
//        ...
//      GPU memory controller 和 PCIe 写合并缓冲区自动合并 scatter write,
//      实际每个 Layer 收到的写入都是连续 page 序列, PCIe 效率与按层搬一致。
//
// 为什么不用 "按层搬" (per_layer × L 次 kernel launch)?
//    按层搬: launch L 个 kernel, 每个 kernel 只搬 1 层, 单 Worker 内连续
//    all_layer: 1 次 kernel launch, 内循环遍历所有层
//    区别: kernel launch overhead ~10-50μs, L=32 层时多花 300-1500μs
//          实际 PCIe 传输效率两者几乎一样 (memory controller 会合并)
//    结论: all_layer 赢在省掉 L-1 次 launch overhead, 且不牺牲带宽
// ══════════════════════════════════════════════════════════════════════════

template <
    typename T,             // indices 的数据类型 (int32_t 或 int64_t)
    int64_t kElementSize,   // 每个 token 每层的 KV 字节数 = head_num × head_dim × dtype_size
    uint32_t kUnroll,       // 循环展开次数, 控制 1 个 warp 分裂为几个 worker
    uint32_t kBlockQuota,   // 最多启动的 block 数, 限制 SM 占用率
    uint32_t kBlockSize,    // 每个 block 的线程数 (通常 1024)
    bool kIsMLA = false>    // MLA 模式下 K/V 合并, 跳过 V 的拷贝
SGL_HICACHE_KERNEL void hicache_transfer_all_layer(const __grid_constant__ HicacheKernelParams params) {
  using namespace device;
  using src_ptr_t = const void*;
  using dst_ptr_t = void*;

  static_assert(kBlockSize % kWarpThreads == 0);
  static_assert(kWarpThreads % kUnroll == 0);

  // ── 并行维度计算 ──
  // kNumThreads: 每个 Worker 的线程数
  //   1 个 warp(32线程) 被 kUnroll 个 Worker 平分, 每个 Worker 分到 32/kUnroll 个线程
  constexpr uint32_t kNumThreads = kWarpThreads / kUnroll;
  // kWorkersPerBlock: 每个 Block 的 Worker 数
  //   1 个 Block 有 kBlockSize 个线程, 每 kNumThreads 个线程组成 1 个 Worker
  constexpr uint32_t kWorkersPerBlock = kBlockSize / kNumThreads;
  // kNumWorkers: 整个 GPU 的总 Worker 数
  //   所有 Block 的 Worker 加起来, 也就是处理 token 时的并行度
  constexpr uint32_t kNumWorkers = kWorkersPerBlock * kBlockQuota;

  const auto& [
    k_ptr_dst, v_ptr_dst, indices_dst, // dst
    k_ptr_src, v_ptr_src, indices_src, // src
    kv_cache_src_stride, kv_cache_dst_stride, length, num_layers // metadata
  ] = params;

  // ── Worker ID 计算 ──
  // work_id: 当前线程所属 Worker 的全局 ID
  //   blockIdx.x * kWorkersPerBlock = 当前 Block 的起始 Worker ID
  //   threadIdx.x / kNumThreads = 当前线程在 Block 内属于第几个 Worker
  const uint32_t work_id = blockIdx.x * kWorkersPerBlock + threadIdx.x / kNumThreads;

  // ── 每个 Worker 以 stride=kNumWorkers 遍历 token ──
  // Worker 0 处理 token 0, 256, 512, ...
  // Worker 1 处理 token 1, 257, 513, ...
  // ... 确保所有 Worker 均匀分担 length 个 token
  for (uint32_t i = work_id; i < length; i += kNumWorkers) {
    // 从 indices 数组中读取源/目标 page 编号 (scatter/gather 地址映射)
    const auto pos_src = static_cast<const T*>(indices_src)[i];
    const auto pos_dst = static_cast<const T*>(indices_dst)[i];
    // ── 遍历所有层 ──
    // 与 per_layer 的区别: 这里 k_ptr_src/k_ptr_dst 是指针数组 [num_layers]
    // 每个 layer 一个指针, 指向该层的 cache base address
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
      // 取出当前层的 K cache 源/目标基地址
      const auto k_cache_src = static_cast<const src_ptr_t*>(k_ptr_src)[layer];
      const auto k_cache_dst = static_cast<const dst_ptr_t*>(k_ptr_dst)[layer];
      // 计算当前 token 在该层 cache 中的实际地址 = base + page_index × stride
      const auto src_k = pointer::offset(k_cache_src, pos_src * kv_cache_src_stride);
      const auto dst_k = pointer::offset(k_cache_dst, pos_dst * kv_cache_dst_stride);
      // kNumThreads 个线程协作: load_vec 读 kElementSize 字节, store_vec 写回
      // 内部使用 ld.global.L1::no_allocate (非缓存读, 不污染 L1 cache)
      const auto vec_k = load_vec<kElementSize, kNumThreads>(src_k);
      store_vec<kElementSize, kNumThreads>(dst_k, vec_k);
      // ── 非 MLA: 额外拷贝 V cache ──
      // MLA 模式下 K/V 合并存储, 只需拷贝一份, 编译期跳过
      if constexpr (!kIsMLA) {
        const auto v_cache_src = static_cast<const src_ptr_t*>(v_ptr_src)[layer];
        const auto v_cache_dst = static_cast<const dst_ptr_t*>(v_ptr_dst)[layer];
        const auto src_v = pointer::offset(v_cache_src, pos_src * kv_cache_src_stride);
        const auto dst_v = pointer::offset(v_cache_dst, pos_dst * kv_cache_dst_stride);
        const auto vec_v = load_vec<kElementSize, kNumThreads>(src_v);
        store_vec<kElementSize, kNumThreads>(dst_v, vec_v);
      }
    }
  }
}

template <int64_t kElementSize, uint32_t kUnroll, uint32_t kBlockQuota, uint32_t kBlockSize>
struct HiCacheKernel {
  template <typename T>
  static constexpr auto kernel_one = hicache_transfer_per_layer<T, kElementSize, kUnroll, kBlockQuota, kBlockSize>;
  template <typename T>
  static constexpr auto kernel_all = hicache_transfer_all_layer<T, kElementSize, kUnroll, kBlockQuota, kBlockSize>;
  template <typename T>
  static constexpr auto kernel_one_mla =
      hicache_transfer_per_layer<T, kElementSize, kUnroll, kBlockQuota, kBlockSize, true>;
  template <typename T>
  static constexpr auto kernel_all_mla =
      hicache_transfer_all_layer<T, kElementSize, kUnroll, kBlockQuota, kBlockSize, true>;

  static void run_one(
      const tvm::ffi::TensorView k_cache_dst,
      const tvm::ffi::TensorView v_cache_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView k_cache_src,
      const tvm::ffi::TensorView v_cache_src,
      const tvm::ffi::TensorView indices_src) {
    using namespace host;

    auto D = SymbolicSize{"head dimension"};
    auto N = SymbolicSize{"src kv stride"};
    auto M = SymbolicSize{"dst kv stride"};
    auto L = SymbolicSize{"indices length"};
    auto cache_dtype = SymbolicDType{};
    auto indices_dtype = SymbolicDType{};
    auto indices_device = SymbolicDevice{};

    TensorMatcher({-1, D})  //
        .with_strides({N, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(k_cache_src)
        .verify(v_cache_src);
    TensorMatcher({-1, D})  //
        .with_strides({M, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(k_cache_dst)
        .verify(v_cache_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLCUDA>(indices_device)
        .verify(indices_src)
        .verify(indices_dst);

    // verify dimension match
    const auto dtype_size = dtype_bytes(cache_dtype.unwrap());
    const auto element_bytes = D.unwrap() * dtype_size;
    RuntimeCheck(kElementSize == element_bytes, "HicacheKernel: cache dimension mismatch.");

    const auto k_cache_dst_ptr = k_cache_dst.data_ptr();
    const auto v_cache_dst_ptr = v_cache_dst.data_ptr();
    const auto k_cache_src_ptr = k_cache_src.data_ptr();
    const auto v_cache_src_ptr = v_cache_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto kv_cache_src_stride = static_cast<int64_t>(N.unwrap() * dtype_size);
    const auto kv_cache_dst_stride = static_cast<int64_t>(M.unwrap() * dtype_size);
    const auto use_int32 = indices_dtype.unwrap().bits == 32;
    const auto device = indices_device.unwrap();

    constexpr auto kWorkersPerBlock = kBlockSize / (device::kWarpThreads / kUnroll);
    const auto num_blocks = std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = k_cache_dst_ptr,
        .v_cache_dst = v_cache_dst_ptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = k_cache_src_ptr,
        .v_cache_src = v_cache_src_ptr,
        .indices_src = indices_src_ptr,
        .kv_cache_src_stride = kv_cache_src_stride,
        .kv_cache_dst_stride = kv_cache_dst_stride,
        .length = length,
    };
    const auto kernel = use_int32 ? kernel_one<int32_t> : kernel_one<int64_t>;
    LaunchKernel(num_blocks, kBlockSize, device)(kernel, params);
  }

  static void run_all(
      const tvm::ffi::TensorView k_ptr_dst,
      const tvm::ffi::TensorView v_ptr_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView k_ptr_src,
      const tvm::ffi::TensorView v_ptr_src,
      const tvm::ffi::TensorView indices_src,
      const int64_t kv_src_stride_bytes,
      const int64_t kv_dst_stride_bytes) {
    using namespace host;

    auto N = SymbolicSize{"num_layers"};
    auto L = SymbolicSize{"indices length"};
    auto dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({N})  //
        .with_dtype<uint64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(k_ptr_src)
        .verify(v_ptr_src)
        .verify(k_ptr_dst)
        .verify(v_ptr_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(indices_src)
        .verify(indices_dst);

    // verify dimension match
    const auto k_cache_dst_ptr = k_ptr_dst.data_ptr();
    const auto v_cache_dst_ptr = v_ptr_dst.data_ptr();
    const auto k_cache_src_ptr = k_ptr_src.data_ptr();
    const auto v_cache_src_ptr = v_ptr_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto use_int32 = dtype_.unwrap().bits == 32;
    const auto device = device_.unwrap();

    constexpr auto kWorkersPerBlock = kBlockSize / (device::kWarpThreads / kUnroll);
    const auto num_blocks = std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = k_cache_dst_ptr,
        .v_cache_dst = v_cache_dst_ptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = k_cache_src_ptr,
        .v_cache_src = v_cache_src_ptr,
        .indices_src = indices_src_ptr,
        .kv_cache_src_stride = kv_src_stride_bytes,
        .kv_cache_dst_stride = kv_dst_stride_bytes,
        .length = length,
        .num_layers = static_cast<uint32_t>(N.unwrap()),
    };
    const auto kernel = use_int32 ? kernel_all<int32_t> : kernel_all<int64_t>;
    LaunchKernel(num_blocks, kBlockSize, device)(kernel, params);
  }

  static void run_one_mla(
      const tvm::ffi::TensorView cache_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView cache_src,
      const tvm::ffi::TensorView indices_src) {
    using namespace host;

    auto D = SymbolicSize{"head dimension"};
    auto N = SymbolicSize{"src stride"};
    auto M = SymbolicSize{"dst stride"};
    auto L = SymbolicSize{"indices length"};
    auto cache_dtype = SymbolicDType{};
    auto indices_dtype = SymbolicDType{};
    auto indices_device = SymbolicDevice{};

    TensorMatcher({-1, D})  //
        .with_strides({N, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(cache_src);
    TensorMatcher({-1, D})  //
        .with_strides({M, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(cache_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLCUDA>(indices_device)
        .verify(indices_src)
        .verify(indices_dst);

    const auto dtype_size = dtype_bytes(cache_dtype.unwrap());
    const auto element_bytes = D.unwrap() * dtype_size;
    RuntimeCheck(kElementSize == element_bytes, "HicacheKernel MLA: cache dimension mismatch.");

    const auto cache_dst_ptr = cache_dst.data_ptr();
    const auto cache_src_ptr = cache_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto cache_src_stride = static_cast<int64_t>(N.unwrap() * dtype_size);
    const auto cache_dst_stride = static_cast<int64_t>(M.unwrap() * dtype_size);
    const auto use_int32 = indices_dtype.unwrap().bits == 32;
    const auto device = indices_device.unwrap();

    constexpr auto kWorkersPerBlock = kBlockSize / (device::kWarpThreads / kUnroll);
    const auto num_blocks = std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = cache_dst_ptr,
        .v_cache_dst = nullptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = cache_src_ptr,
        .v_cache_src = nullptr,
        .indices_src = indices_src_ptr,
        .kv_cache_src_stride = cache_src_stride,
        .kv_cache_dst_stride = cache_dst_stride,
        .length = length,
    };
    const auto kernel = use_int32 ? kernel_one_mla<int32_t> : kernel_one_mla<int64_t>;
    LaunchKernel(num_blocks, kBlockSize, device)(kernel, params);
  }

  static void run_all_mla(
      const tvm::ffi::TensorView ptr_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView ptr_src,
      const tvm::ffi::TensorView indices_src,
      const int64_t src_stride_bytes,
      const int64_t dst_stride_bytes) {
    using namespace host;

    auto N = SymbolicSize{"num_layers"};
    auto L = SymbolicSize{"indices length"};
    auto dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({N})  //
        .with_dtype<uint64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(ptr_src)
        .verify(ptr_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(indices_src)
        .verify(indices_dst);

    const auto cache_dst_ptr = ptr_dst.data_ptr();
    const auto cache_src_ptr = ptr_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto use_int32 = dtype_.unwrap().bits == 32;
    const auto device = device_.unwrap();

    constexpr auto kWorkersPerBlock = kBlockSize / (device::kWarpThreads / kUnroll);
    const auto num_blocks = std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = cache_dst_ptr,
        .v_cache_dst = nullptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = cache_src_ptr,
        .v_cache_src = nullptr,
        .indices_src = indices_src_ptr,
        .kv_cache_src_stride = src_stride_bytes,
        .kv_cache_dst_stride = dst_stride_bytes,
        .length = length,
        .num_layers = static_cast<uint32_t>(N.unwrap()),
    };
    const auto kernel = use_int32 ? kernel_all_mla<int32_t> : kernel_all_mla<int64_t>;
    LaunchKernel(num_blocks, kBlockSize, device)(kernel, params);
  }
};

#undef SGL_HICACHE_KERNEL

}  // namespace
