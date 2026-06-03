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

/// Decode kernel 参数结构体
struct Compress128DecodeParams {
  void* __restrict__ kv_buffer;       // [num_indices, 128, head_dim*2], 环形缓冲区，存 [kv|score]
  const void* __restrict__ kv_input;  // [batch_size, head_dim*2], 当前步新 token 的 [kv|score]
  void* __restrict__ kv_output;       // [batch_size, head_dim], 压缩后的 KV 输出
  const void* __restrict__ score_bias;// [128, head_dim], APE 绝对位置编码偏置
  const PlanD* __restrict__ plan_d;   // [batch_size], 每个请求的 decode 计划
                                       //   .read_page_1: ring buffer 起始页 (压缩时读取)
                                       //   .write_loc:   ring buffer 写入位置 (写入新 token)
  uint32_t batch_size;                // decode 请求数
};

/// Prefill kernel 参数结构体 (两个 kernel 共享)
struct Compress128PrefillParams {
  void* __restrict__ kv_buffer;       // [num_indices, 128, head_dim*2], 环形缓冲区，存 [kv|score]
  const void* __restrict__ kv_input;  // [num_q_tokens, head_dim*2], 新 token 的 [kv|score]（ragged）
  void* __restrict__ kv_output;       // [num_compress, head_dim], 紧凑排列的压缩 KV 输出
  const void* __restrict__ score_bias;// [128, head_dim], APE 绝对位置编码偏置
  const PlanC* __restrict__ plan_c;   // [num_compress], 压缩计划数组
                                       //   .ragged_id:    输入 token 行号 (补充不完整 chunk)
                                       //   .read_page_1:  ring buffer 起始页
                                       //   .buffer_len:   ring buffer 有效 token 数
  const PlanW* __restrict__ plan_w;   // [num_write], 写入计划数组
                                       //   .ragged_id:  输入 token 行号
                                       //   .write_loc:  ring buffer 写入位置
  uint32_t num_compress;              // 需要压缩的完整 chunk 数 (= PlanC 条数)
  uint32_t num_write;                 // 需要写入 ring buffer 的新 token 数 (= PlanW 条数)
};

/// Shared memory 缓冲区: 存储 16 个 warp 的局部 online softmax 结果
/// 布局: [kNumWarps=16][kWarpThreads+1=33]，+1 消除 bank conflict
/// 每个元素是 AlignedVector<float, 2>，即 2 个 float (对应 kTileElements)
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

/// C128 压缩的编译期常量集合
/// 核心维度关系:
///   head_dim = 512 (DeepSeek V4)
///   kTileDim = 64  (每个 split 处理的 head_dim 维度)
///   kNumSplit = 8  (head_dim 拆分成 8 个 split 并行处理)
///   kElementSize = 1024 (每个 token 在 buffer 中占的元素数: kv 512 + score 512)
///   kPageElementSize = 128 * 1024 = 131072 (一页 = 128 个 token 在 buffer 中的元素数)
template <int64_t kHeadDim_>
struct C128Trait {
  // 每个 tile 处理的 head_dim 维度: 2元素/线程 × 32线程/warp = 64
  static constexpr int64_t kTileDim = kTileElements * device::kWarpThreads;  // 64
  static constexpr int64_t kHeadDim = kHeadDim_;                             // 512
  // kv_score buffer 布局: | kv [0, head_dim) | score [head_dim, 2*head_dim) |
  static constexpr int64_t kScoreOffset = kHeadDim;                          // 512
  // 每个 token 在 kv_score buffer 中占的元素数 = kv + score = 2 * head_dim
  static constexpr int64_t kElementSize = kHeadDim * 2;                     // 1024
  // 一页（128 个 token）在 kv_score buffer 中占的元素数，用于 read_page 索引
  static constexpr int64_t kPageElementSize = 128 * kElementSize;           // 131072
  // head_dim 拆分成多少个 split 并行处理，512/64 = 8
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;
  static_assert(kHeadDim % kTileDim == 0);
};

/// ===========================================================================
/// c128_forward: C128 压缩的核心计算函数
///
/// 将 128 个 token 的 (kv, score) 通过 online softmax 加权聚合成 1 个压缩 KV。
/// 这是 prefill 和 decode 共享的压缩逻辑。
///
/// 计算公式: output[d] = Σ_{i=0}^{127} kv[i][d] * softmax(score[i] + bias[i])[i]
///
/// 数据流:
///   ring buffer (128个历史token) + 输入token (补充不完整chunk)
///       ↓ 加载 kv, score, bias
///       ↓ Part 1: 每个 warp 加载 8 个位置
///       ↓ Part 2: 每个 warp 内做局部 online softmax (8个位置)
///       ↓ Part 3: 跨 16 个 warp 做全局 online softmax 规约
///       ↓ 输出 1 个压缩 KV [head_dim]
/// ===========================================================================
template <typename Trait, bool kUsePDL, typename InFloat, typename OutFloat>
SGL_DEVICE void c128_forward(
    const InFloat* kv_buf,     // ring buffer 中该 chunk 的起始地址，[128, head_dim*2]
    const InFloat* kv_src,     // 当前新 token 的 [kv|score] 指针（回退127步可得到 chunk 起始）
    OutFloat* kv_out,          // 压缩后 KV 输出，[head_dim]
    const InFloat* score_bias, // APE 位置偏置，[128, head_dim]
    const int32_t buffer_len)  // ring buffer 中有效 token 数；<128 时尾部从 kv_src 补充
{
  using namespace device;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;

  /// ---- Part 1: 加载数据 ----
  /// 128 个 token 位置分配给 16 个 warp:
  ///   warp 0 → 位置 [0,8), warp 1 → [8,16), ..., warp 15 → [120,128)
  /// 每个 warp 处理 8 位置 × 2元素/线程 × 32线程 = 8 × 64 维 (head_dim 的 1/8)
  using StorageIn = AlignedVector<InFloat, kTileElements>;  // 一次读 2 个 float
  const auto gmem_in = tile::Memory<StorageIn>{lane_id, kWarpThreads};
  StorageIn kv[kElementsPerWarp];    // 8 个位置的 kv (每个位置 2 个元素)
  StorageIn score[kElementsPerWarp]; // 8 个位置的 score
  StorageIn bias[kElementsPerWarp];  // 8 个位置的 APE 偏置
  const int32_t warp_offset = warp_id * kElementsPerWarp;  // 该 warp 在 128 中的起始偏移

  // 加载 APE 偏置: bias[j] 对应位置 j 的绝对位置编码偏置
#pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    const int32_t j = i + warp_offset;
    bias[i] = gmem_in.load(score_bias + j * Trait::kHeadDim);
  }

  // kv_src 指向 chunk 最后一个 token，回退 127 步得到 chunk 起始位置
  // 当 buffer_len < 128 时，位置 [buffer_len, 128) 的 token 还在 kv_src（输入）中，
  // 不在 ring buffer 中，所以需要从 kv_src 回退读取
  const auto kv_start = kv_src - 127 * Trait::kElementSize;

  // 加载 kv + score:
  //   位置 [0, buffer_len) → 从 kv_buf (ring buffer) 读
  //   位置 [buffer_len, 128) → 从 kv_start (输入回退) 读
#pragma unroll
  for (int32_t i = 0; i < kElementsPerWarp; ++i) {
    const int32_t j = i + warp_offset;
    __builtin_assume(j < 128);
    const auto src = j < buffer_len ? kv_buf : kv_start;
    kv[i] = gmem_in.load(src + j * Trait::kElementSize);                        // kv 部分
    score[i] = gmem_in.load(src + j * Trait::kElementSize + Trait::kScoreOffset); // score 部分
  }

  /// ---- Part 2: 每个 warp 内做局部 online softmax + 加权求和 ----
  /// 对 8 个位置的 score 做 softmax，得到该 warp 局部的 (max, sum_exp, weighted_kv)
  /// 后续 Part 3 会跨 warp 做全局规约
  using TmpStorage = typename Compress128SharedBuffer::Storage;
  __shared__ Compress128SharedBuffer s_local_val_max;   // 每个 warp 局部最大 score
  __shared__ Compress128SharedBuffer s_local_exp_sum;   // 每个 warp 局部 exp(score-max) 之和
  __shared__ Compress128SharedBuffer s_local_product;   // 每个 warp 局部加权 KV 之和

  TmpStorage tmp_val_max;
  TmpStorage tmp_exp_sum;
  TmpStorage tmp_product;

  float score_fp32[kTileElements][kElementsPerWarp];

  // 将 score 从 InFloat 转 fp32，并加上 APE 偏置
  // 同时完成转置: score[j][i] → score_fp32[i][j] (tile_id × position)
#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    for (int32_t j = 0; j < kElementsPerWarp; ++j) {
      score_fp32[i][j] = cast<float>(score[j][i]) + cast<float>(bias[j][i]);
    }
  }

  // 对每个 tile 元素 (2个)，在 8 个位置上做局部 online softmax
#pragma unroll
  for (int32_t i = 0; i < kTileElements; ++i) {
    const auto& score = score_fp32[i];
    float max_value = score[0];
    float sum_exp_value = 0.0f;

    // Step 2a: 找 8 个 score 中的最大值（数值稳定: softmax 减 max 防溢出）
#pragma unroll
    for (int32_t j = 1; j < kElementsPerWarp; ++j) {
      const auto fp32_score = score[j];
      max_value = fmaxf(max_value, fp32_score);
    }

    // Step 2b: 计算 exp(score - max) 和 加权 KV 聚合
    //   sum_product = Σ kv[j] * exp(score[j] - max)  (加权 KV)
    //   sum_exp_value = Σ exp(score[j] - max)         (归一化因子)
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      const auto fp32_score = score[j];
      const auto exp_score = expf(fp32_score - max_value);
      sum_product += cast<float>(kv[j][i]) * exp_score;
      sum_exp_value += exp_score;
    }

    tmp_val_max[i] = max_value;
    tmp_exp_sum[i] = sum_exp_value;
    tmp_product[i] = sum_product;
  }

  // 局部结果写入 shared memory，供 Part 3 跨 warp 全局规约
  s_local_val_max(warp_id, lane_id) = tmp_val_max;
  s_local_exp_sum(warp_id, lane_id) = tmp_exp_sum;
  s_local_product(warp_id, lane_id) = tmp_product;

  __syncthreads();  // 等待所有 warp 写完 shared memory

  /// ---- Part 3: 跨 warp 全局 online softmax 规约 ----
  ///
  /// 16 个 warp 各自产出了局部 (max, sum_exp, product)，
  /// 现在需要把它们合并成全局的 softmax 结果。
  ///
  /// Online softmax 合并公式:
  ///   global_max = max(warp0_max, warp1_max, ..., warp15_max)
  ///   rescale_w = exp(warp_w_max - global_max)  // 将每个 warp 的 sum/product 缩放到统一尺度
  ///   global_exp_sum = Σ (warp_w_exp_sum * rescale_w)
  ///   global_product = Σ (warp_w_product * rescale_w)
  ///   output = global_product / global_exp_sum
  ///
  /// 数据总量: kTileElements(2) × kWarpThreads(32) × kNumWarps(16) = 1024 个值
  /// 每个 thread 处理 kIteration = 1024 / 512 = 2 个值
  /// 规约方式: partial warp reduction — 每 kNumWarps=16 个 thread 做一次 warp reduce
  constexpr uint32_t kReductionCount = kTileElements * kWarpThreads * kNumWarps;
  constexpr uint32_t kIteration = kReductionCount / kBlockSize;

  PDLTriggerSecondary<kUsePDL>();

#pragma unroll
  for (uint32_t i = 0; i < kIteration; ++i) {
    // j ∈ [0, 1024) — 遍历所有需要规约的值
    const uint32_t j = i * kBlockSize + warp_id * kWarpThreads + lane_id;
    // local_warp_id ∈ [0, 16) — 对应 Part 2 中哪个 warp 的局部结果
    const uint32_t local_warp_id = j % kNumWarps;
    // local_elem_id ∈ [0, 64) — 输出中的元素索引 (head_dim 的 1/8 内)
    const uint32_t local_elem_id = j / kNumWarps;
    // local_tile_id ∈ [0, 2) — 该元素在 tile 内的偏移
    const uint32_t local_tile_id = local_elem_id % kTileElements;
    // local_lane_id ∈ [0, 32) — 对应 Part 2 中的 lane
    const uint32_t local_lane_id = local_elem_id / kTileElements;
    static_assert(kTileElements * kNumWarps == kWarpThreads, "TODO: support other configs");

    // 从 shared memory 读出 Part 2 写入的局部结果
    const auto local_val_max = s_local_val_max(local_warp_id, local_lane_id, local_tile_id);
    const auto local_exp_sum = s_local_exp_sum(local_warp_id, local_lane_id, local_tile_id);
    const auto local_product = s_local_product(local_warp_id, local_lane_id, local_tile_id);

    // Step 3a: 跨 16 个 warp 求全局最大 score
    const auto global_val_max = warp::reduce_max<kNumWarps>(local_val_max);

    // Step 3b: 计算缩放因子 rescale = exp(局部max - 全局max)
    //   将每个 warp 的 sum/product 统一到以 global_max 为基准的尺度
    const auto rescale = expf(local_val_max - global_val_max);

    // Step 3c: 跨 warp 求全局 exp_sum 和全局加权 product
    const auto global_exp_sum = warp::reduce_sum<kNumWarps>(local_exp_sum * rescale);

    // Step 3d: 最终归一化: product / exp_sum = softmax 加权聚合 KV
    const auto final_scale = rescale / global_exp_sum;
    const auto global_product = warp::reduce_sum<kNumWarps>(local_product * final_scale);

    // 写出结果
    kv_out[local_elem_id] = cast<OutFloat>(global_product);
  }
}

/// c128_write_decode: 将 1 个新 token 的 [kv|score] 写入 ring buffer
/// 由 decode kernel 的 warp 15 调用
/// 写入 head_dim*2 = 1024 个元素 (kv: [0, head_dim), score: [head_dim, 2*head_dim))
template <typename Trait, typename InFloat>
SGL_DEVICE void c128_write_decode(InFloat* kv_buf, const InFloat* kv_src) {
  using namespace device;

  using Storage = AlignedVector<InFloat, kTileElements>;  // 一次读/写 2 个 float
  const auto gmem = tile::Memory<Storage>::warp();        // warp 级访存

  // 每个 warp 写入 head_dim*2 元素 = 32线程 × 2元素/线程 × 2次迭代 = 128 元素/次
  // 第 i 次迭代写入 [i*head_dim, (i+1)*head_dim) 区域
  // i=0: kv 部分, i=1: score 部分
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

/// ===========================================================================
/// flash_c128_decode: Decode 阶段 kernel
///
/// 每个 decode 请求写 1 个新 token 到 ring buffer，当 chunk 满(128个token)时触发压缩。
///
/// Block 布局: gridDim = batch_size × kNumSplit (8)
///   global_bid = blockIdx.x / kNumSplit → 请求 id
///   global_sid = blockIdx.x % kNumSplit → head_dim 分片 id (每片 64 维)
///
/// Warp 布局: 每 block 16 warp (512 线程)
///   warp 0~14: 参与 c128_forward 压缩
///   warp 15:   负责写入新 token 到 ring buffer (c128_write_decode)
///
/// 执行逻辑:
///   1. warp 15 将新 token 的 [kv|score] 写入 ring buffer[write_loc]
///   2. 如果 write_loc % 128 == 127 (chunk 满):
///      所有 16 个 warp 执行 c128_forward → softmax 聚合 → 产出 1 个压缩 KV
///   3. 否则: 只写 buffer，不触发压缩
///
/// PlanD (DecodePlan) 各字段使用:
///   plan_d[global_bid].read_page_1 → ring buffer 起始页 (用于压缩时读取历史 token)
///   plan_d[global_bid].write_loc   → ring buffer 写入位置 (写入新 token)
/// ===========================================================================
template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
C128_KERNEL void flash_c128_decode(const __grid_constant__ Compress128DecodeParams params) {
  using namespace device;
  using Trait = C128Trait<kHeadDim>;

  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t global_bid = blockIdx.x / Trait::kNumSplit;  // 请求 id
  const uint32_t global_sid = blockIdx.x % Trait::kNumSplit;  // head_dim 分片 id
  const int64_t split_offset = global_sid * Trait::kTileDim;   // 该分片在 head_dim 中的偏移
  if (global_bid >= params.batch_size) return;

  // 读取 decode 计划
  const auto plan = params.plan_d[global_bid];

  // 基地址 + split_offset: 每个 split 独立处理 head_dim 的 64 维
  const auto kv_input = static_cast<const InFloat*>(params.kv_input) + split_offset;
  const auto kv_output = static_cast<OutFloat*>(params.kv_output) + split_offset;
  const auto kv_buffer = static_cast<InFloat*>(params.kv_buffer) + split_offset;
  const auto score_bias = static_cast<const InFloat*>(params.score_bias) + split_offset;

  // kv_src: 输入新 token 的 [kv|score] 地址
  const auto kv_src = kv_input + global_bid * Trait::kElementSize;
  // kv_out: 压缩后 KV 输出地址
  const auto kv_out = kv_output + global_bid * Trait::kHeadDim;
  // kv_buf: ring buffer 中该 chunk 的起始页 (读取历史 128 个 token)
  const auto kv_buf = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;
  // kv_dst: ring buffer 中新 token 的写入位置
  const auto kv_dst = kv_buffer + plan.write_loc * Trait::kElementSize;

  PDLWaitPrimary<kUsePDL>();
  // Step 1: warp 15 负责将新 token 写入 ring buffer
  // the write warp must match the load warp in the following `c128_forward`
  if (warp_id == kNumWarps - 1) {
    c128_write_decode<Trait>(kv_dst, kv_src);
  }
  // Step 2: 如果 write_loc 是 chunk 的最后一个位置 (127)，说明 chunk 已满，触发压缩
  if (plan.write_loc % 128 == 127) {
    c128_forward<Trait, kUsePDL>(kv_buf, kv_src, kv_out, score_bias, 128);
  }
}

/// ===========================================================================
/// flash_c128_prefill: Prefill 压缩 kernel
///
/// 对已满的 128-chunk 做 online softmax 聚合，产出 1 个压缩 KV。
///
/// Block 布局: gridDim = num_compress × kNumSplit (8)
///   global_pid = blockIdx.x / kNumSplit → PlanC 索引 (第几个压缩任务)
///   global_sid = blockIdx.x % kNumSplit → head_dim 分片 id (每片 64 维)
///
/// Warp 布局: 每 block 16 warp，每个 warp 负责 8 个位置
///
/// PlanC (CompressPlan) 各字段使用:
///   plan.ragged_id    → 在 kv_input 中的行号 (定位新 token，用于补充不完整 chunk)
///   plan.read_page_1  → ring buffer 起始页 (读取历史 token 的起始位置)
///   plan.buffer_len   → ring buffer 中有效 token 数 (<128 时从 kv_src 补充)
///   plan.is_invalid() → 跳过 padding 填充的无效条目
/// ===========================================================================
template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
C128_KERNEL void flash_c128_prefill(const __grid_constant__ Compress128PrefillParams params) {
  using namespace device;
  using Trait = C128Trait<kHeadDim>;

  const uint32_t global_pid = blockIdx.x / Trait::kNumSplit;  // PlanC 索引
  const uint32_t global_sid = blockIdx.x % Trait::kNumSplit;  // head_dim 分片 id
  const int64_t split_offset = global_sid * Trait::kTileDim;
  if (global_pid >= params.num_compress) return;

  // 读取压缩计划
  const auto plan = params.plan_c[global_pid];

  // 基地址 + split_offset
  const auto kv_input = static_cast<const InFloat*>(params.kv_input) + split_offset;
  const auto kv_output = static_cast<OutFloat*>(params.kv_output) + split_offset;
  const auto kv_buffer = static_cast<InFloat*>(params.kv_buffer) + split_offset;
  const auto score_bias = static_cast<const InFloat*>(params.score_bias) + split_offset;

  // 跳过 padding 的无效计划 (cuda graph 模式下 num_c 会 pad 到 num_q_tokens)
  if (plan.is_invalid()) return;

  // kv_src: 定位到 kv_input 中该 chunk 最后一个新 token
  //   c128_forward 内部会回退 127 步得到 chunk 起始，用于补充 buffer_len < 128 的部分
  const auto kv_src = kv_input + plan.ragged_id * Trait::kElementSize;
  // kv_out: 紧凑排列的输出，每条 PlanC 对应输出一行
  const auto kv_out = kv_output + global_pid * Trait::kHeadDim;
  // kv_buf: ring buffer 中该 chunk 的起始页 (历史 token 存储)
  const auto kv_buf = kv_buffer + plan.read_page_1 * Trait::kPageElementSize;

  PDLWaitPrimary<kUsePDL>();
  // 核心压缩: 128 个 token 的 online softmax 加权聚合
  c128_forward<Trait, kUsePDL>(kv_buf, kv_src, kv_out, score_bias, plan.buffer_len);
}

/// ===========================================================================
/// write_c128_prefill: Prefill 写入 kernel
///
/// 将新 token 的 [kv|score] 写入 ring buffer，不触发压缩。
///
/// Block 布局: gridDim = div_ceil(num_write × kNumSplit, kWarpsPerWriteBlock)
///   = div_ceil(num_write × 8, 4)
///
/// Warp 布局: 每 block 4 warp (128 线程)
///   global_wid = global_tid / 32
///   global_pid = global_wid / kNumSplit → PlanW 索引 (第几个写入任务)
///   global_sid = global_wid % kNumSplit → head_dim 分片 id
///
/// PlanW (WritePlan) 各字段使用:
///   plan.ragged_id   → 在 kv_input 中的行号 (定位要写入的 token)
///   plan.write_loc   → ring buffer 写入位置
///   plan.is_invalid() → 跳过 padding 填充的无效条目
/// ===========================================================================
template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
WRITE_KERNEL void write_c128_prefill(const __grid_constant__ Compress128PrefillParams params) {
  using namespace device;
  using Trait = C128Trait<kHeadDim>;
  using StorageIn = AlignedVector<InFloat, kTileElements>;

  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_wid = global_tid / kWarpThreads;      // 全局 warp id
  const uint32_t global_pid = global_wid / Trait::kNumSplit;  // PlanW 索引
  const uint32_t global_sid = global_wid % Trait::kNumSplit;  // head_dim 分片 id
  // 注意: 写入 kernel 的 split_offset 覆盖的是 head_dim*2 (kv+score 连续布局)
  // 与压缩 kernel 不同 (压缩 kernel 的 split_offset 只覆盖 head_dim)
  const int64_t split_offset = global_sid * (Trait::kTileDim * 2);
  if (global_pid >= params.num_write) return;

  // 读取写入计划
  const auto plan = params.plan_w[global_pid];

  const auto kv_input = static_cast<const InFloat*>(params.kv_input) + split_offset;
  const auto kv_buffer = static_cast<InFloat*>(params.kv_buffer) + split_offset;

  // 跳过 padding 的无效计划
  if (plan.is_invalid()) return;

  // kv_src: 从 kv_input 读取新 token 的 [kv|score]
  const auto kv_src = kv_input + plan.ragged_id * Trait::kElementSize;
  // kv_buf: 写入 ring buffer 的目标位置
  const auto kv_buf = kv_buffer + plan.write_loc * Trait::kElementSize;
  const auto gmem = tile::Memory<StorageIn>::warp();

  PDLWaitPrimary<kUsePDL>();
  // 读取 head_dim*2 元素 (kv + score)，分 2 次迭代
  StorageIn data[2];
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    data[i] = gmem.load(kv_src, i);
  }
  PDLTriggerSecondary<kUsePDL>();
  // 写入 ring buffer
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    gmem.store(kv_buf, data[i], i);
  }
}

/// ===========================================================================
/// Host 端入口: 校验 tensor shape → 构造参数 → 启动 CUDA kernel
/// ===========================================================================
template <int64_t kHeadDim, typename InFloat, typename OutFloat, bool kUsePDL>
struct FlashCompress128Kernel {
  static constexpr auto decode_kernel = flash_c128_decode<kHeadDim, InFloat, OutFloat, kUsePDL>;
  static constexpr auto prefill_c_kernel = flash_c128_prefill<kHeadDim, InFloat, OutFloat, kUsePDL>;
  static constexpr auto prefill_w_kernel = write_c128_prefill<kHeadDim, InFloat, OutFloat, kUsePDL>;
  static constexpr int64_t kTileDim = kTileElements * device::kWarpThreads;  // 64
  static constexpr uint32_t kNumSplit = kHeadDim / kTileDim;                 // 512/64 = 8
  using Trait = C128Trait<kHeadDim>;

  /// -----------------------------------------------------------------------
  /// run_decode: Decode 入口，启动 1 个 kernel
  ///
  /// grid: batch_size × kNumSplit 个 block，每 block 512 线程 (16 warp)
  ///
  /// 每个请求: 写入新 token → 若 chunk 满(128个) 则压缩
  /// -----------------------------------------------------------------------
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

    // 校验 tensor shape
    TensorMatcher({-1, 128, Trait::kElementSize})  // kv_buffer: [num_indices, 128, head_dim*2]
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_buffer);
    TensorMatcher({N, Trait::kElementSize})  // kv_input: [batch_size, head_dim*2]
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_input);
    TensorMatcher({N, kHeadDim})  // kv_output: [batch_size, head_dim]
        .with_dtype<OutFloat>()
        .with_device(device_)
        .verify(kv_output);
    TensorMatcher({128, kHeadDim})  // ape: [128, head_dim]
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(ape);

    // 解码 PlanD 并构造 kernel 参数
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

  /// -----------------------------------------------------------------------
  /// run_prefill: Prefill 入口，启动 2 个 kernel（可并行，无依赖）
  ///
  /// kernel 1 (压缩): num_c × kNumSplit 个 block，每 block 512 线程 (16 warp)
  ///   → 对已满的 128-chunk 做 online softmax 聚合，产出压缩 KV
  ///
  /// kernel 2 (写入): div_ceil(num_w × kNumSplit, 4) 个 block，每 block 128 线程 (4 warp)
  ///   → 将新 token 的 [kv|score] 写入 ring buffer（不触发压缩）
  ///
  /// 两个 kernel 可以并行执行，因为:
  ///   - kernel 1 只读 ring buffer 的已满 chunk (read_page_1)
  ///   - kernel 2 只写 ring buffer 的尾部空位 (write_loc)
  ///   - 两者操作的 ring buffer 区域不重叠
  /// -----------------------------------------------------------------------
  static void run_prefill(
      const tvm::ffi::TensorView kv_buffer,
      const tvm::ffi::TensorView kv_input,
      const tvm::ffi::TensorView kv_output,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView plan_c_,
      const tvm::ffi::TensorView plan_w_) {
    using namespace host;

    auto N = SymbolicSize{"num_q_tokens"};
    auto C = SymbolicSize{"num_c_plans"};  // PlanC 条数 = 需要压缩的 chunk 数
    auto W = SymbolicSize{"num_w_plans"};  // PlanW 条数 = 需要写入的 token 数
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    // 校验 tensor shape
    TensorMatcher({-1, 128, Trait::kElementSize})  // kv_buffer: [num_indices, 128, head_dim*2]
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_buffer);
    TensorMatcher({N, Trait::kElementSize})  // kv_input: [num_q_tokens, head_dim*2] (ragged)
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(kv_input);
    TensorMatcher({C, kHeadDim})  // kv_output: [num_compress, head_dim] (紧凑排列)
        .with_dtype<OutFloat>()
        .with_device(device_)
        .verify(kv_output);
    TensorMatcher({128, kHeadDim})  // ape: [128, head_dim]
        .with_dtype<InFloat>()
        .with_device(device_)
        .verify(ape);

    // 解析 PlanC 和 PlanW
    const auto plan_c = compress::verify_plan_c(plan_c_, C, device_);
    const auto plan_w = compress::verify_plan_w(plan_w_, W, device_);
    const auto device = device_.unwrap();
    const auto num_q_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_c = static_cast<uint32_t>(C.unwrap());  // 需要压缩的 chunk 数
    const auto num_w = static_cast<uint32_t>(W.unwrap());  // 需要写入的 token 数
    const auto params = Compress128PrefillParams{
        .kv_buffer = kv_buffer.data_ptr(),
        .kv_input = kv_input.data_ptr(),
        .kv_output = kv_output.data_ptr(),
        .score_bias = ape.data_ptr(),
        .plan_c = plan_c,     // 压缩计划数组
        .plan_w = plan_w,     // 写入计划数组
        .num_compress = num_c,
        .num_write = num_w,
    };
    RuntimeCheck(num_q_tokens >= num_w, "invalid prefill plan: num_q < num_w");

    // kernel 1: 压缩已满的 128-chunk (读取 ring buffer → softmax 聚合 → 输出压缩 KV)
    if (const auto num_c_blocks = num_c * kNumSplit) {  // num_c × 8 个 block
      constexpr auto kBlockSize_C = kBlockSize;         // 512 线程 = 16 warp
      // 本质：每个压缩任务需要 8 个 block 并行处理 8 个 head_dim 分片，所以总 block 数 = 任务数 × 8
      LaunchKernel(num_c_blocks, kBlockSize_C, device)  //
          .enable_pdl(kUsePDL)(prefill_c_kernel, params); 
    }

    // kernel 2: 将新 token 写入 ring buffer（不触发压缩，简单拷贝）
    constexpr uint32_t kWarpsPerWriteBlock = kWriteBlockSize / device::kWarpThreads;  // 128/32 = 4
    if (const auto num_w_blocks = div_ceil(num_w * kNumSplit, kWarpsPerWriteBlock)) {
      constexpr auto kBlockSize_W = kWriteBlockSize; // 128 线程 = 4 warp
      //总共需要 num_w × 8 个 warp，每 block 有 4 个 warp，所以 block 数 = ceil(num_w × 8 / 4) = ceil(num_w × 2)
      // 
      LaunchKernel(num_w_blocks, kBlockSize_W, device)  //
          .enable_pdl(kUsePDL)(prefill_w_kernel, params);
    }
  }
};

}  // namespace
