/**
 * \brief Here's some dimension info for the main buffer used in C128 prefill and decode.
 *
 * kv_buffer: [num_indices, 128, head_dim * 2]
 * - last dimension layout: | kv | score |
 * kv_input: [batch_size, head_dim * 2]
 * kv_output: [batch_size, head_dim]
 * score_bias (ape): [128, head_dim]
 * plan_c/plan_w: [variable length]
 *
 * For prefill, batch_size = num_q_tokens
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/compress_v2.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/object.h>

#include <cstdint>

namespace {

using PlanD = device::compress::DecodePlan;
using PlanC = device::compress::CompressPlan;
using PlanW = device::compress::WritePlan;

// 每个线程沿 head_dim 维度处理的元素数，用于向量化访存（一次读 2 个元素）
constexpr int32_t kTileElements = 2;
// 每个 warp 沿 128（token 序列）维度处理的位置数，16 个 warp × 8 = 128 完整覆盖
constexpr int32_t kElementsPerWarp = 8;
// 需要的 warp 数，128 个 token / 每个 warp 处理 8 个 = 16 个 warp
constexpr uint32_t kNumWarps = 128 / kElementsPerWarp;
// 压缩 kernel 的 CTA 大小，32 线程/warp × 16 warp = 512 线程
constexpr uint32_t kBlockSize = device::kWarpThreads * kNumWarps;
// 写入 kernel 的 CTA 大小，128 线程（= 4 warp），每个 warp 处理一个 token 的写入
constexpr uint32_t kWriteBlockSize = 128;

/// \brief Need to reduce register usage to increase occupancy
#define C128_KERNEL __global__ __launch_bounds__(kBlockSize, 2)
#define WRITE_KERNEL __global__ __launch_bounds__(kWriteBlockSize, 16)

struct Compress128DecodeParams {
  void* __restrict__ kv_buffer;       // [num_indices, 128, head_dim*2], 环形缓冲区，存 [kv|score]
  const void* __restrict__ kv_input;  // [batch_size, head_dim*2], 当前步新 token 的 [kv|score]
  void* __restrict__ kv_output;       // [batch_size, head_dim], 压缩后的 KV 输出
  const void* __restrict__ score_bias;// [128, head_dim], APE 绝对位置编码偏置
  const PlanD* __restrict__ plan_d;   // [batch_size], 每个请求的 decode 计划 (read_page, write_loc)
  uint32_t batch_size;                // decode 请求数
};

struct Compress128PrefillParams {
  void* __restrict__ kv_buffer;       // [num_indices, 128, head_dim*2], 环形缓冲区，存 [kv|score]
  const void* __restrict__ kv_input;  // [num_q_tokens, head_dim*2], 新 token 的 [kv|score]（ragged）
  void* __restrict__ kv_output;       // [num_compress, head_dim], 紧凑排列的压缩 KV 输出
  const void* __restrict__ score_bias;// [128, head_dim], APE 绝对位置编码偏置
  const PlanC* __restrict__ plan_c;   // [num_compress], 完整 128-chunk 的压缩计划
  const PlanW* __restrict__ plan_w;   // [num_write], 新 token 写入 ring buffer 的写入计划
  uint32_t num_compress;              // 需要压缩的完整 chunk 数
  uint32_t num_write;                 // 需要写入 ring buffer 的新 token 数
};

struct Compress128SharedBuffer {
  using Storage = device::AlignedVector<float, kTileElements>;
  Storage data[kNumWarps][device::kWarpThreads + 1];  // padding to avoid bank conflict
  SGL_DEVICE Storage& operator()(uint32_t warp_id, uint32_t lane_id) {
    return data[warp_id][lane_id];
  }
  SGL_DEVICE float& operator()(uint32_t warp_id, uint32_t lane_id, uint32_t tile_id) {
    return data[warp_id][lane_id][tile_id];
  }
};

template <int64_t kHeadDim_>
struct C128Trait {
  // 每个 tile 处理的 head_dim 维度，2元素/线程 × 32线程/warp = 64
  static constexpr int64_t kTileDim = kTileElements * device::kWarpThreads;  // 64
  // 注意力头维度（DeepSeek V4 为 512）
  static constexpr int64_t kHeadDim = kHeadDim_;
  // kv_score buffer 中 score 的起始偏移，布局为 | kv [0, head_dim) | score [head_dim, 2*head_dim) |
  static constexpr int64_t kScoreOffset = kHeadDim;
  // 每个 token 在 kv_score buffer 中占的元素数 = kv + score = 2 * head_dim
  static constexpr int64_t kElementSize = kHeadDim * 2;
  // 一页（128 个 token）在 kv_score buffer 中占的元素数，用于 read_page_1 索引
  static constexpr int64_t kPageElementSize = 128 * kElementSize;  // page size = 128
  // head_dim 拆分成多少个 tile 并行处理，512/64 = 8 个 split
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static_assert(kHeadDim % kTileDim == 0);
};

template <typename Trait, bool kUsePDL, typename InFloat, typename OutFloat>
SGL_DEVICE void c128_forward(
    const InFloat* kv_buf,  // ring buffer 中该 chunk 的起始地址，[128, head_dim*2]
    const InFloat* kv_src,  // 当前新 token 的 [kv|score] 指针，用于填充 buffer_len < 128 的情况
    OutFloat* kv_out,       // 压缩后 KV 输出，[head_dim]
    const InFloat* score_bias, // APE 位置偏置，[128, head_dim]
    const int32_t buffer_len) { // ring buffer 中有效 token 数（≤128，最后不完整 chunk 可能 < 128）
  using namespace device;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;

  /// NOTE: part 1: 每个 warp 加载自己负责的 8 个位置的 kv + score + bias
  /// warp 0 → 位置 [0,8), warp 1 → [8,16), ..., warp 15 → [120,128)
  /// 每个 warp 处理 8 位置 × kTileElements(2) 元素/线程 × 32 线程 = 8 × 64 维
  using StorageIn = AlignedVector<InFloat, kTileElements>;
  const auto gmem_in = tile::Memory<StorageIn>{lane_id, kWarpThreads};
  StorageIn kv[kElementsPerWarp];    // 每个 warp 的 8 个位置的 kv
  StorageIn score[kElementsPerWarp]; // 每个 warp 的 8 个位置的 score
  StorageIn bias[kElementsPerWarp];  // 每个 warp 的 8 个位置的 bias
  const int32_t warp_offset = warp_id * kElementsPerWarp; // 该 warp 在 128 中的起始偏移

  // 加载 bias：每个 warp 加载 8 个位置的 APE 偏置
#pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    const int32_t j = i + warp_offset;
    bias[i] = gmem_in.load(score_bias + j * Trait::kHeadDim);
  }

  const auto kv_start = kv_src - 127 * Trait::kElementSize;  // 回退到 chunk 起始位置

  // 加载 kv + score：有效位置从 kv_buf 读，超出的用 kv_src（当前 token）填充
#pragma unroll
  for (int32_t i = 0; i < kElementsPerWarp; ++i) {
    const int32_t j = i + warp_offset;
    __builtin_assume(j < 128);
    const auto src = j < buffer_len ? kv_buf : kv_start;
    kv[i] = gmem_in.load(src + j * Trait::kElementSize);
    score[i] = gmem_in.load(src + j * Trait::kElementSize + Trait::kScoreOffset);
  }

  /// NOTE: part 2: 每个 warp 内做局部 online softmax + 加权求和
  /// 对 8 个位置的 score 做 softmax，得到该 warp 局部的 (max, sum_exp, weighted_kv)
  using TmpStorage = typename Compress128SharedBuffer::Storage;
  // shared memory 存每个 warp 的局部规约结果，供 part 3 跨 warp 全局规约
  __shared__ Compress128SharedBuffer s_local_val_max;   // 每个 warp 局部最大 score
  __shared__ Compress128SharedBuffer s_local_exp_sum;   // 每个 warp 局部 exp(score-max) 之和
  __shared__ Compress128SharedBuffer s_local_product;   // 每个 warp 局部加权 KV 之和

  TmpStorage tmp_val_max;
  TmpStorage tmp_exp_sum;
  TmpStorage tmp_product;

  float score_fp32[kTileElements][kElementsPerWarp];

  // 转 fp32 并加上位置偏置 bias
#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    for (int32_t j = 0; j < kElementsPerWarp; ++j) {
      score_fp32[i][j] = cast<float>(score[j][i]) + cast<float>(bias[j][i]);
    }
  }

  // 对 8 个位置做局部 online softmax：找 max → 算 exp → 加权聚合 KV
#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    const auto& score = score_fp32[i];
    float max_value = score[0];
    float sum_exp_value = 0.0f;

    // 找 8 个 score 中的最大值（用于数值稳定）
#pragma unroll
    for (int32_t j = 1; j < kElementsPerWarp; ++j) {
      const auto fp32_score = score[j];
      max_value = fmaxf(max_value, fp32_score);
    }

    // 计算 exp(score - max) 和 加权 KV
    float sum_product = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      const auto fp32_score = score[j];
      const auto exp_score = expf(fp32_score - max_value);
      sum_product += cast<float>(kv[j][i]) * exp_score;  // kv × softmax_weight
      sum_exp_value += exp_score;
    }

    tmp_val_max[i] = max_value;
    tmp_exp_sum[i] = sum_exp_value;
    tmp_product[i] = sum_product;
  }

  // 局部结果写入 shared memory（自然对齐，无 bank conflict）
  s_local_val_max(warp_id, lane_id) = tmp_val_max;
  s_local_exp_sum(warp_id, lane_id) = tmp_exp_sum;
  s_local_product(warp_id, lane_id) = tmp_product;

  __syncthreads();

  /// NOTE: part 3: 跨 warp 全局 online softmax 规约
  /// 共有 kTileElements × kWarpThreads × kNumWarps = 2 × 32 × 16 = 1024 个值需要规约
  /// 每个 thread 处理 kIteration = 1024 / 512 = 2 个值的规约
  /// 规约使用 partial warp reduction：每 kNumWarps=16 个 thread 做一次 warp reduce
  constexpr uint32_t kReductionCount = kTileElements * kWarpThreads * kNumWarps;
  constexpr uint32_t kIteration = kReductionCount / kBlockSize;

  PDLTriggerSecondary<kUsePDL>();

#pragma unroll
  for (uint32_t i = 0; i < kIteration; ++i) {
    // j ∈ [0, kTileElements * kWarpThreads * kNumWarps) — 遍历所有需要规约的值
    const uint32_t j = i * kBlockSize + warp_id * kWarpThreads + lane_id;
    // local_warp_id ∈ [0, kNumWarps) — 对应 part 2 中哪个 warp 的局部结果
    const uint32_t local_warp_id = j % kNumWarps;
    // local_elem_id ∈ [0, kTileElements * kWarpThreads) — 输出中的元素索引
    const uint32_t local_elem_id = j / kNumWarps;
    // local_tile_id ∈ [0, kTileElements) — 该元素在 tile 内的偏移
    const uint32_t local_tile_id = local_elem_id % kTileElements;
    // local_lane_id ∈ [0, kWarpThreads) — 对应 part 2 中的 lane
    const uint32_t local_lane_id = local_elem_id / kTileElements;
    // 不同 lane 只在 local_warp_id 上不同，所以 shared memory 访问无 bank conflict
    static_assert(kTileElements * kNumWarps == kWarpThreads, "TODO: support other configs");
    const auto local_val_max = s_local_val_max(local_warp_id, local_lane_id, local_tile_id);
    const auto local_exp_sum = s_local_exp_sum(local_warp_id, local_lane_id, local_tile_id);
    const auto local_product = s_local_product(local_warp_id, local_lane_id, local_tile_id);
    // 跨 kNumWarps=16 个 warp 做 warp 级规约
    const auto global_val_max = warp::reduce_max<kNumWarps>(local_val_max);
    // 校正：旧 sum 乘以 exp(旧max - 新max) 以适配新的全局 max
    const auto rescale = expf(local_val_max - global_val_max);
    const auto global_exp_sum = warp::reduce_sum<kNumWarps>(local_exp_sum * rescale);
    // 最终缩放：校正后的 product / 全局 sum = softmax 加权聚合 KV
    const auto final_scale = rescale / global_exp_sum;
    const auto global_product = warp::reduce_sum<kNumWarps>(local_product * final_scale);
    kv_out[local_elem_id] = cast<OutFloat>(global_product);
  }
}

template <typename Trait, typename InFloat>
SGL_DEVICE void c128_write_decode(InFloat* kv_buf, const InFloat* kv_src) {
  using namespace device;

  using Storage = AlignedVector<InFloat, kTileElements>;
  const auto gmem = tile::Memory<Storage>::warp();

  Storage data[2];
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    data[i] = gmem.load(kv_src + Trait::kHeadDim * i);
  }
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    gmem.store(kv_buf + Trait::kHeadDim * i, data[i]);
  }
}

/// Decode kernel: 每个 decode 请求写 1 个新 token 到 ring buffer，chunk 满时触发压缩
///
/// Block 布局: gridDim = batch_size × kNumSplit (8)
///   global_bid = blockIdx.x / kNumSplit → 请求 id
///   global_sid = blockIdx.x % kNumSplit → head_dim 分片 id (每片 64 维)
///
/// Warp 布局: 每_block 16 warp
///   warp 0~14: 参与 c128_forward 压缩
///   warp 15:   负责写入新 token 到 ring buffer (c128_write_decode)
///
/// 执行逻辑:
///   1. warp 15 将新 token 的 [kv|score] 写入 ring buffer[write_loc]
///   2. 如果 write_loc % 128 == 127 (chunk 满):
///      所有 16 个 warp 执行 c128_forward → softmax 聚合 → 产出 1 个压缩 KV
///   3. 否则: 只写 buffer，不触发压缩
template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
C128_KERNEL void flash_c128_decode(const __grid_constant__ Compress128DecodeParams params) {
  using namespace device;
  using Trait = C128Trait<kHeadDim>;

  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t global_bid = blockIdx.x / Trait::kNumSplit;  // batch id
  const uint32_t global_sid = blockIdx.x % Trait::kNumSplit;  // split id
  const int64_t split_offset = global_sid * Trait::kTileDim;
  if (global_bid >= params.batch_size) return;

  const auto plan = params.plan_d[global_bid];
  const auto kv_input = static_cast<const InFloat*>(params.kv_input) + split_offset;
  const auto kv_output = static_cast<OutFloat*>(params.kv_output) + split_offset;
  const auto kv_buffer = static_cast<InFloat*>(params.kv_buffer) + split_offset;
  const auto score_bias = static_cast<const InFloat*>(params.score_bias) + split_offset;

  const auto kv_src = kv_input + global_bid * Trait::kElementSize;
  const auto kv_out = kv_output + global_bid * Trait::kHeadDim;
  const auto kv_buf = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  const auto kv_dst = kv_buffer + plan.write_loc * Trait::kElementSize;

  PDLWaitPrimary<kUsePDL>();
  // the write warp must match the load warp in the following `c128_forward`
  if (warp_id == kNumWarps - 1) {
    c128_write_decode<Trait>(kv_dst, kv_src);
  }
  if (plan.write_loc % 128 == 127) {
    c128_forward<Trait, kUsePDL>(kv_buf, kv_src, kv_out, score_bias, 128);
  }
}

/// Prefill 压缩 kernel: 对已满的 128-chunk 做 softmax 聚合，产出压缩 KV
///
/// Block 布局: gridDim = num_compress × kNumSplit (8)
///   global_pid = blockIdx.x / kNumSplit → 压缩计划 id
///   global_sid = blockIdx.x % kNumSplit → head_dim 分片 id (每片 64 维)
///
/// Warp 布局: 每_block 16 warp，每个 warp 负责 8 个位置（与 decode 的 c128_forward 相同）
///
/// 执行逻辑:
///   所有 16 个 warp 执行 c128_forward → softmax 聚合 → 产出 1 个压缩 KV
///   buffer_len 可能 < 128（最后一个不完整 chunk）
// compress kernel
template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
C128_KERNEL void flash_c128_prefill(const __grid_constant__ Compress128PrefillParams params) {

  // kv_buffer [num_indices, 128, head_dim*2], 环形缓冲区，存 [kv|score]
  // kv_input [num_q_tokens, head_dim*2], 新 token 的 [kv|score]（ragged）
  // kv_output [num_compress, head_dim], 紧凑排列的压缩 KV 输出
  // score_bias [128, head_dim], APE 绝对位置编码偏置
  // plan_c [num_compress], 完整 128-chunk 的压缩计划
  // plan_w [num_write], 新 token 写入 ring buffer 的写入计划
  // num_compress 需要压缩的完整 chunk 数
  // num_write 需要写入 ring buffer 的新 token 数
  using namespace device;
  using Trait = C128Trait<kHeadDim>;

  const uint32_t global_pid = blockIdx.x / Trait::kNumSplit;  // plan id
  const uint32_t global_sid = blockIdx.x % Trait::kNumSplit;  // split id
  const int64_t split_offset = global_sid * Trait::kTileDim;
  if (global_pid >= params.num_compress) return;

  const auto plan = params.plan_c[global_pid];
  const auto kv_input = static_cast<const InFloat*>(params.kv_input) + split_offset;
  const auto kv_output = static_cast<OutFloat*>(params.kv_output) + split_offset;
  const auto kv_buffer = static_cast<InFloat*>(params.kv_buffer) + split_offset;
  const auto score_bias = static_cast<const InFloat*>(params.score_bias) + split_offset;
  if (plan.is_invalid()) return;

  const auto kv_src = kv_input + plan.ragged_id * Trait::kElementSize;
  // Compact output: one row per compress plan, indexed by `global_pid`.
  const auto kv_out = kv_output + global_pid * Trait::kHeadDim;
  const auto kv_buf = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  PDLWaitPrimary<kUsePDL>();
  c128_forward<Trait, kUsePDL>(kv_buf, kv_src, kv_out, score_bias, plan.buffer_len);
}

/// Prefill 写入 kernel: 将新 token 的 [kv|score] 写入 ring buffer（不触发压缩）
///
/// Block 布局: gridDim = div_ceil(num_write × kNumSplit, kWarpsPerWriteBlock)
///   = div_ceil(num_write × 8, 4)
///
/// Warp 布局: 每_block 4 warp (128 线程)
///   global_wid = global_tid / 32
///   global_pid = global_wid / kNumSplit → 写入计划 id
///   global_sid = global_wid % kNumSplit → head_dim 分片 id
///
/// 执行逻辑:
///   每个 warp 从 kv_input 读 1 个 token 的 [kv|score]，写入 kv_buffer[write_loc]
template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
WRITE_KERNEL void write_c128_prefill(const __grid_constant__ Compress128PrefillParams params) {
  using namespace device;
  using Trait = C128Trait<kHeadDim>;
  using StorageIn = AlignedVector<InFloat, kTileElements>;

  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_wid = global_tid / kWarpThreads;      // warp id
  const uint32_t global_pid = global_wid / Trait::kNumSplit;  // plan id
  const uint32_t global_sid = global_wid % Trait::kNumSplit;  // split id
  // split the contiguous `kHeadDim * 2` into `kNumSplit` tiles
  // each warp handles 1 contiguous tile (in contrast, decode handle the strided head_dim)
  const int64_t split_offset = global_sid * (Trait::kTileDim * 2);
  if (global_pid >= params.num_write) return;

  const auto plan = params.plan_w[global_pid];
  const auto kv_input = static_cast<const InFloat*>(params.kv_input) + split_offset;
  const auto kv_buffer = static_cast<InFloat*>(params.kv_buffer) + split_offset;
  if (plan.is_invalid()) return;

  // each warp will handle a contiguous region
  const auto kv_src = kv_input + plan.ragged_id * Trait::kElementSize;
  const auto kv_buf = kv_buffer + plan.write_loc * Trait::kElementSize;
  const auto gmem = tile::Memory<StorageIn>::warp();

  PDLWaitPrimary<kUsePDL>();
  StorageIn data[2];
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    data[i] = gmem.load(kv_src, i);
  }
  PDLTriggerSecondary<kUsePDL>();
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    gmem.store(kv_buf, data[i], i);
  }
}

/// Host 端入口：校验 shape → 构造参数 → 启动 CUDA kernel
template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
struct FlashCompress128Kernel {
  static constexpr auto decode_kernel = flash_c128_decode<kHeadDim, InFloat, OutFloat, kUsePDL>;
  static constexpr auto prefill_c_kernel = flash_c128_prefill<kHeadDim, InFloat, OutFloat, kUsePDL>;
  static constexpr auto prefill_w_kernel = write_c128_prefill<kHeadDim, InFloat, OutFloat, kUsePDL>;
  static constexpr int64_t kTileDim = kTileElements * device::kWarpThreads;  // 64
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  using Trait = C128Trait<kHeadDim>;

  /// Decode 入口：启动 1 个 kernel
  /// grid: batch_size × kNumSplit 个 block，每 block 512 线程 (16 warp)
  static void run_decode(
      const tvm::ffi::TensorView kv_buffer,
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView kv_output,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView plan_d_) {
    using namespace host;

    auto N = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({-1, 128, Trait::kElementSize})  // kv score
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_buffer);
    TensorMatcher({N, Trait::kElementSize})  // kv score input
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_input);
    TensorMatcher({N, kHeadDim})  // kv compressed output
        .with_dtype<OutFloat>()
        .with_device(device_)
        .verify(kv_output);
    TensorMatcher({128, kHeadDim})  // ape
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(ape);

    const auto plan_d = compress::verify_plan_d(plan_d_, N, device_);
    const auto batch_size = static_cast<uint32_t>(N.unwrap());
    const auto params = Compress128DecodeParams{
        .kv_buffer = kv_buffer.data_ptr(),
        .kv_input = kv_input.data_ptr(),
        .kv_output = kv_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .plan_d = plan_d,
        .batch_size = batch_size,
    };
    const uint32_t num_blocks = batch_size * kNumSplit;  // batch_size × 8
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(decode_kernel, params);
  }

  /// Prefill 入口：启动 2 个 kernel（可并行）
  /// kernel 1 (压缩): num_c × kNumSplit 个 block，每 block 512 线程 (16 warp)
  /// kernel 2 (写入): div_ceil(num_w × kNumSplit, 4) 个 block，每 block 128 线程 (4 warp)
  static void run_prefill(
      const tvm::ffi::TensorView kv_buffer,
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView kv_output,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView plan_c_,
      const tvm::ffi::TensorView plan_w_) {
    using namespace host;

    auto N = SymbolicSize{"num_q_tokens"};
    auto C = SymbolicSize{"num_c_plans"};
    auto W = SymbolicSize{"num_w_plans"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({-1, 128, Trait::kElementSize})  // kv score
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_buffer);
    TensorMatcher({N, Trait::kElementSize})  // kv score input (ragged)
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_input);
    TensorMatcher({C, kHeadDim})  // kv compressed output (compact)
        .with_dtype<OutFloat>()
        .with_device(device_)
        .verify(kv_output);
    TensorMatcher({128, kHeadDim})  // ape
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(ape);

    const auto plan_c = compress::verify_plan_c(plan_c_, C, device_);
    const auto plan_w = compress::verify_plan_w(plan_w_, W, device_);
    const auto device = device_.unwrap();
    const auto num_q_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_c = static_cast<uint32_t>(C.unwrap());
    const auto num_w = static_cast<uint32_t>(W.unwrap());
    const auto params = Compress128PrefillParams{
        .kv_buffer = kv_buffer.data_ptr(),
        .kv_input = kv_input.data_ptr(),
        .kv_output = kv_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .plan_c = plan_c,
        .plan_w = plan_w,
        .num_compress = num_c,
        .num_write = num_w,
    };
    RuntimeCheck(num_q_tokens >= num_w, "invalid prefill plan: num_q < num_w");
    // kernel 1: 压缩已满的 128-chunk
    if (const auto num_c_blocks = num_c * kNumSplit) {  // num_c × 8 个 block
      constexpr auto kBlockSize_C = kBlockSize;         // 512 线程 = 16 warp
      LaunchKernel(num_c_blocks, kBlockSize_C, device)  //
          .enable_pdl(kUsePDL)(prefill_c_kernel, params);
    }
    // kernel 2: 将新 token 写入 ring buffer（不触发压缩）
    constexpr uint32_t kWarpsPerWriteBlock = kWriteBlockSize / device::kWarpThreads;  // 128/32 = 4
    if (const auto num_w_blocks = div_ceil(num_w * kNumSplit, kWarpsPerWriteBlock)) {
      constexpr auto kBlockSize_W = kWriteBlockSize;
      LaunchKernel(num_w_blocks, kBlockSize_W, device)  //
          .enable_pdl(kUsePDL)(prefill_w_kernel, params);
    }
  }
};

}  // namespace
