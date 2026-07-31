/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// CK FMHA Split-KV pipeline includes
// NOTE: include order matters. tile_fmha_traits.hpp pulls in
// block_attention_quant_scale_enum.hpp, which the splitkv pipeline headers
// below require but do not include themselves. Do not alphabetize this block.
// clang-format off
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/block/block_position_encoding.hpp"
#include "ck_tile/ops/fmha/block/page_block_navigator.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
// clang-format on

#ifndef CK_TILE_FMHA_FWD_FAST_EXP2
#define CK_TILE_FMHA_FWD_FAST_EXP2 1
#endif

namespace kernel {

// ============================================================================
// Head-dim-dependent tile configuration
// From CK codegen (gfx9): hdim 64 → <64,64,32,64,32,64>, hdim 128 →
// <64,128,32,128,32,128>
// ============================================================================
using Gemm0BlockWarps = ck_tile::sequence<4, 1, 1>;
using Gemm0WarpTile = ck_tile::sequence<16, 16, 16>;
using Gemm1BlockWarps = ck_tile::sequence<4, 1, 1>;
using Gemm1WarpTile = ck_tile::sequence<16, 16, 16>;

using FmhaVariant = ck_tile::ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>;

// float32 epilogue for split-KV with merge (NUM_KV_CHUNKS > 1)
using FmhaEpilogueProblem =
    ck_tile::Default2DEpilogueProblem<float, float, true, false, false>;
using FmhaEpilogue = ck_tile::Default2DEpilogue<FmhaEpilogueProblem>;

// bf16 epilogue for direct output (NUM_KV_CHUNKS == 1, no merge needed)
using FmhaEpilogueBf16Problem = ck_tile::
    Default2DEpilogueProblem<float, ck_tile::bf16_t, true, false, false>;
using FmhaEpilogueBf16 = ck_tile::Default2DEpilogue<FmhaEpilogueBf16Problem>;

struct BlockIndices {
  ck_tile::index_t batch_idx;
  ck_tile::index_t qo_head_idx;
  ck_tile::index_t kv_head_idx;
};

// ============================================================================
// Decode/Prefill traits and masks (head-dim independent)
// ============================================================================
using DecodeTraits =
    ck_tile::TileFmhaFwdSplitKVTraits<true,
                                      true,
                                      false,
                                      false,
                                      false,
                                      ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                      false,
                                      true,
                                      false,
                                      true,
                                      true,
                                      true, // kMergeNumHeadGroupsSeqLenQ
                                      -1,
                                      false>;
using DecodeMask = ck_tile::SimplifiedGenericAttentionMask<false>;

using PrefillTraits =
    ck_tile::TileFmhaFwdSplitKVTraits<true,
                                      true,
                                      false,
                                      false,
                                      false,
                                      ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                      false,
                                      true,
                                      false,
                                      true,
                                      true,
                                      false, // kMergeNumHeadGroupsSeqLenQ
                                      -1,
                                      false>;
using PrefillMask = ck_tile::SimplifiedGenericAttentionMask<true>;

// ============================================================================
// Template struct to select tile config based on HEAD_DIM
// ============================================================================
template <int HEAD_DIM_T>
struct FmhaTileConfig;

template <>
struct FmhaTileConfig<64> {
  using BlockTile = ck_tile::sequence<64, 64, 32, 64, 32, 64>;
  using Shape = ck_tile::TileFmhaShape<BlockTile,
                                       Gemm0BlockWarps,
                                       Gemm0WarpTile,
                                       Gemm1BlockWarps,
                                       Gemm1WarpTile,
                                       true>;
  using DecodePipelineProblem =
      ck_tile::BlockFmhaFwdSplitKVPipelineProblem<ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  float,
                                                  float,
                                                  ck_tile::bf16_t,
                                                  float,
                                                  ck_tile::bf16_t,
                                                  float,
                                                  float,
                                                  Shape,
                                                  true,
                                                  FmhaVariant,
                                                  DecodeMask,
                                                  DecodeTraits>;
  using DecodePipeline =
      ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVS<DecodePipelineProblem>;
  using PrefillPipelineProblem =
      ck_tile::BlockFmhaFwdSplitKVPipelineProblem<ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  float,
                                                  float,
                                                  ck_tile::bf16_t,
                                                  float,
                                                  ck_tile::bf16_t,
                                                  float,
                                                  float,
                                                  Shape,
                                                  true,
                                                  FmhaVariant,
                                                  PrefillMask,
                                                  PrefillTraits>;
  using PrefillPipeline =
      ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVS<PrefillPipelineProblem>;
};

template <>
struct FmhaTileConfig<128> {
  using BlockTile = ck_tile::sequence<64, 128, 32, 128, 32, 128>;
  using Shape = ck_tile::TileFmhaShape<BlockTile,
                                       Gemm0BlockWarps,
                                       Gemm0WarpTile,
                                       Gemm1BlockWarps,
                                       Gemm1WarpTile,
                                       true>;
  using DecodePipelineProblem =
      ck_tile::BlockFmhaFwdSplitKVPipelineProblem<ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  float,
                                                  float,
                                                  ck_tile::bf16_t,
                                                  float,
                                                  ck_tile::bf16_t,
                                                  float,
                                                  float,
                                                  Shape,
                                                  true,
                                                  FmhaVariant,
                                                  DecodeMask,
                                                  DecodeTraits>;
  using DecodePipeline =
      ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVS<DecodePipelineProblem>;
  using PrefillPipelineProblem =
      ck_tile::BlockFmhaFwdSplitKVPipelineProblem<ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  ck_tile::bf16_t,
                                                  float,
                                                  float,
                                                  ck_tile::bf16_t,
                                                  float,
                                                  ck_tile::bf16_t,
                                                  float,
                                                  float,
                                                  Shape,
                                                  true,
                                                  FmhaVariant,
                                                  PrefillMask,
                                                  PrefillTraits>;
  using PrefillPipeline =
      ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVS<PrefillPipelineProblem>;
};

// ============================================================================
// Decode: merge all qo_heads into M, single pipeline call, no mask
// __noinline__ isolates register allocation from prefill path
// ============================================================================
template <typename T,
          int NUM_QO_PER_KV,
          int HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE_T,
          int NUM_KV_HEADS_T>
__device__ __noinline__ void
    paged_attention_ck_fmha_decode(void const *q_workspace_ptr,
                                   void *paged_k_cache_ptr,
                                   void *paged_v_cache_ptr,
                                   void *o_acc_ptr,
                                   void *lse_acc_ptr,
                                   int const *qo_indptr_buffer_ptr,
                                   int const *paged_kv_indptr_buffer_ptr,
                                   int const *paged_kv_indices_buffer_ptr,
                                   int const *paged_kv_last_page_len_buffer_ptr,
                                   int16_t request_id,
                                   int kv_head_idx,
                                   int kv_chunk_idx,
                                   float scale_s,
                                   int sliding_window = 0) {
  // Simple scalar attention for decode (BS=1).
  // Each workgroup handles NUM_QO_PER_KV Q heads against 1 KV head.
  // 256 threads, assign (256 / NUM_QO_PER_KV) = 32 threads per Q head.
  // Each thread loops over seqlen_k positions, computes Q·K^T, online softmax,
  // accumulates V.

  using bf16 = ck_tile::bf16_t;
  constexpr int kv_cache_stride = KV_CACHE_STRIDE_T;
  constexpr int THREADS_PER_HEAD = 256 / NUM_QO_PER_KV; // 32 for 8 heads

  int const tid = threadIdx.x;
  int const q_head_local =
      tid / THREADS_PER_HEAD; // which Q head within this KV group (0..7)
  int const lane = tid % THREADS_PER_HEAD; // lane within head (0..31)

  int const req = request_id;
  int const query_start = qo_indptr_buffer_ptr[req];
  int const first_page = paged_kv_indptr_buffer_ptr[req];
  int const last_page = paged_kv_indptr_buffer_ptr[req + 1];
  int const num_pages = last_page - first_page;
  int const seqlen_k =
      (num_pages - 1) * PAGE_SIZE + paged_kv_last_page_len_buffer_ptr[req];

  // Q pointer: [num_tokens, num_q_heads * HEAD_DIM]
  bf16 const *q_base = reinterpret_cast<bf16 const *>(q_workspace_ptr) +
                       query_start * Q_WORKSPACE_STRIDE +
                       (kv_head_idx * NUM_QO_PER_KV + q_head_local) * HEAD_DIM;

  // Load Q into registers (HEAD_DIM / THREADS_PER_HEAD elements per thread)
  // For HEAD_DIM=64, THREADS_PER_HEAD=32: 2 elements per thread
  constexpr int Q_PER_THREAD =
      (HEAD_DIM + THREADS_PER_HEAD - 1) / THREADS_PER_HEAD;
  float q_reg[Q_PER_THREAD];
#pragma unroll
  for (int i = 0; i < Q_PER_THREAD; i++) {
    int d = lane * Q_PER_THREAD + i;
    q_reg[i] = (d < HEAD_DIM) ? static_cast<float>(q_base[d]) : 0.0f;
  }

  // Online softmax accumulators
  float m_val = -1e30f; // running max
  float d_val = 0.0f;   // running sum of exp
  float o_acc[Q_PER_THREAD];
#pragma unroll
  for (int i = 0; i < Q_PER_THREAD; i++) {
    o_acc[i] = 0.0f;
  }

  // K/V cache base pointers
  bf16 const *k_cache = reinterpret_cast<bf16 const *>(paged_k_cache_ptr);
  bf16 const *v_cache = reinterpret_cast<bf16 const *>(paged_v_cache_ptr);

  // Sliding window: skip KV positions before the window
  int kv_start = 0;
  if (sliding_window > 0 && seqlen_k > sliding_window) {
    kv_start = seqlen_k - sliding_window;
  }

  // Iterate over KV positions (starting from kv_start for sliding window)
  for (int kv_pos = kv_start; kv_pos < seqlen_k; kv_pos++) {
    int page_idx = kv_pos / PAGE_SIZE;
    int page_offset = kv_pos % PAGE_SIZE;
    int phys_page = paged_kv_indices_buffer_ptr[first_page + page_idx];

    // K element address: cache[phys_page * PAGE_SIZE * kv_cache_stride +
    // page_offset * kv_cache_stride + d] Note: kv_head_idx offset is already
    // applied by the framework's tiling system
    bf16 const *k_ptr = k_cache +
                        (long long)phys_page * PAGE_SIZE * kv_cache_stride +
                        page_offset * kv_cache_stride;

    // Compute Q·K^T partial dot product (each thread does Q_PER_THREAD
    // elements)
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < Q_PER_THREAD; i++) {
      int d = lane * Q_PER_THREAD + i;
      if (d < HEAD_DIM) {
        dot += q_reg[i] * static_cast<float>(k_ptr[d]);
      }
    }

// Warp reduction to get full dot product (across THREADS_PER_HEAD=32 threads =
// 1 wave for WAVE_SIZE=64, need sub-wave reduction) Use __shfl_xor for
// reduction within 32 threads
#pragma unroll
    for (int offset = THREADS_PER_HEAD / 2; offset > 0; offset >>= 1) {
      dot += __shfl_xor(dot, offset, THREADS_PER_HEAD);
    }

    // Scale
    float score = dot * scale_s;

    // Online softmax update (exp2f matches CK_TILE_FMHA_FWD_FAST_EXP2=1 scale)
    float m_new = fmaxf(m_val, score);
    float exp_diff = exp2f(m_val - m_new);
    float exp_score = exp2f(score - m_new);
    d_val = d_val * exp_diff + exp_score;

// Rescale old output accumulator
#pragma unroll
    for (int i = 0; i < Q_PER_THREAD; i++) {
      o_acc[i] *= exp_diff;
    }

    // Add V contribution
    // Note: kv_head_idx offset is already applied by the framework's tiling
    // system
    bf16 const *v_ptr = v_cache +
                        (long long)phys_page * PAGE_SIZE * kv_cache_stride +
                        page_offset * kv_cache_stride;
#pragma unroll
    for (int i = 0; i < Q_PER_THREAD; i++) {
      int d = lane * Q_PER_THREAD + i;
      if (d < HEAD_DIM) {
        o_acc[i] += exp_score * static_cast<float>(v_ptr[d]);
      }
    }

    m_val = m_new;
  }

  // Normalize: o_acc /= d_val
  float inv_d = (d_val > 0.0f) ? (1.0f / d_val) : 0.0f;

  // Write output (bf16)
  bf16 *o_ptr = reinterpret_cast<bf16 *>(o_acc_ptr) +
                query_start * Q_WORKSPACE_STRIDE +
                (kv_head_idx * NUM_QO_PER_KV + q_head_local) * HEAD_DIM;
#pragma unroll
  for (int i = 0; i < Q_PER_THREAD; i++) {
    int d = lane * Q_PER_THREAD + i;
    if (d < HEAD_DIM) {
      o_ptr[d] = bf16(o_acc[i] * inv_d);
    }
  }

#ifdef SCALAR_ATTN_DEBUG
  if (kv_head_idx == 0 && q_head_local == 0 && lane == 0) {
    printf("[SCALAR_ATTN] req=%d query_start=%d seqlen_k=%d m_val=%.4f "
           "d_val=%.4f inv_d=%.6f\n",
           req,
           query_start,
           seqlen_k,
           m_val,
           d_val,
           inv_d);
    printf("[SCALAR_ATTN] q[0..3]=%.4f %.4f %.4f %.4f\n",
           q_reg[0],
           q_reg[1],
           0.0f,
           0.0f);
    printf("[SCALAR_ATTN] o[0..3]=%.4f %.4f %.4f %.4f\n",
           o_acc[0] * inv_d,
           o_acc[1] * inv_d,
           0.0f,
           0.0f);
    // Print first K values at pos 0
    bf16 const *k0_ptr =
        k_cache + (long long)paged_kv_indices_buffer_ptr[first_page] *
                      PAGE_SIZE * kv_cache_stride;
    printf("[SCALAR_ATTN] k0[0..3]=%.4f %.4f %.4f %.4f\n",
           (float)k0_ptr[0],
           (float)k0_ptr[1],
           (float)k0_ptr[2],
           (float)k0_ptr[3]);
  }
#endif

  // Write LSE (log-sum-exp) for attention sink correction
  // LSE = m_val + log(d_val)
  // Layout: lse_acc[token, q_head] where q_head is global across all KV heads
  if (lane == 0) {
    constexpr int LSE_STRIDE = NUM_KV_HEADS_T * NUM_KV_CHUNKS * NUM_QO_PER_KV;
    float *lse_out = reinterpret_cast<float *>(lse_acc_ptr) +
                     query_start * LSE_STRIDE + kv_head_idx * NUM_QO_PER_KV +
                     q_head_local;
    float lse_val = (d_val > 0.0f) ? (m_val + logf(d_val)) : -1e30f;
    *lse_out = lse_val;
  }
}

// ============================================================================
// Prefill: loop over qo_heads with causal mask
// __noinline__ isolates register allocation from decode path
// ============================================================================
template <typename T,
          int NUM_QO_PER_KV,
          int HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE_T,
          int NUM_KV_HEADS_T>
__device__ __noinline__ void paged_attention_ck_fmha_prefill(
    void const *q_workspace_ptr,
    void *paged_k_cache_ptr,
    void *paged_v_cache_ptr,
    void *o_acc_ptr,
    void *lse_acc_ptr,
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int16_t request_id,
    int kv_head_idx,
    int kv_chunk_idx,
    float scale_s,
    ck_tile::index_t seqlen_q,
    int sliding_window = 0,
    void const *sinks_ptr = nullptr) {
  using namespace ck_tile;
  using TileConfig = FmhaTileConfig<HEAD_DIM>;
  using MyPrefillPipeline = typename TileConfig::PrefillPipeline;

  extern __shared__ char smem_ptr[];

  constexpr int kv_cache_stride = KV_CACHE_STRIDE_T;
  constexpr index_t kM0 = MyPrefillPipeline::kM0;
  constexpr index_t kN0 = MyPrefillPipeline::kN0;
  constexpr index_t kK0 = MyPrefillPipeline::kK0;
  constexpr index_t kN1 = MyPrefillPipeline::kN1;
  constexpr index_t kK1 = MyPrefillPipeline::kK1;
  constexpr index_t kSubQKHeaddim = MyPrefillPipeline::kSubQKHeaddim;

  constexpr index_t i_m0 = 0;
  constexpr index_t i_n1 = 0;
  const index_t i_split = kv_chunk_idx;

  constexpr int LSE_STRIDE = NUM_KV_HEADS_T * NUM_KV_CHUNKS * NUM_QO_PER_KV;
  constexpr int O_ACC_STRIDE = LSE_STRIDE * HEAD_DIM;

  int const req = request_id;
  const index_t query_start = qo_indptr_buffer_ptr[req];

  const index_t first_page = paged_kv_indptr_buffer_ptr[req];
  const index_t last_page = paged_kv_indptr_buffer_ptr[req + 1];
  const index_t num_pages = last_page - first_page;
  index_t seqlen_k =
      (num_pages - 1) * PAGE_SIZE + paged_kv_last_page_len_buffer_ptr[req];

  long_index_t batch_offset_q = query_start * Q_WORKSPACE_STRIDE;
  bf16_t const *q_ptr =
      reinterpret_cast<bf16_t const *>(q_workspace_ptr) + batch_offset_q;

  // K/V view factories
  auto make_k_dram_fn = [&](bf16_t const *data, index_t height) {
    const auto k = make_naive_tensor_view<address_space_enum::global>(
        data,
        make_tuple(height, (index_t)HEAD_DIM),
        make_tuple((index_t)kv_cache_stride, (index_t)1),
        number<MyPrefillPipeline::kAlignmentK>{},
        number<1>{});
    return pad_tensor_view(
        k, make_tuple(number<kN0>{}, number<kK0>{}), sequence<false, false>{});
  };

  auto make_v_dram_fn = [&](bf16_t const *data, index_t length) {
    const auto v = make_naive_tensor_view<address_space_enum::global>(
        data,
        make_tuple(length, (index_t)HEAD_DIM),
        make_tuple((index_t)kv_cache_stride, (index_t)1),
        number<MyPrefillPipeline::kAlignmentV>{},
        number<1>{});
    const auto vt = transform_tensor_view(
        v,
        make_tuple(make_pass_through_transform((index_t)HEAD_DIM),
                   make_pass_through_transform(length)),
        make_tuple(sequence<1>{}, sequence<0>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));
    return pad_tensor_view(
        vt, make_tuple(number<kN1>{}, number<kK1>{}), sequence<false, true>{});
  };

  // Page block navigators
  int32_t const *block_indices =
      reinterpret_cast<int32_t const *>(paged_kv_indices_buffer_ptr) +
      first_page;
  const index_t num_blocks = integer_divide_ceil(seqlen_k, (index_t)PAGE_SIZE);

  auto k_nav = make_page_block_navigator<const bf16_t, 0>(
      reinterpret_cast<bf16_t const *>(paged_k_cache_ptr),
      (long_index_t)(PAGE_SIZE * kv_cache_stride),
      (long_index_t)0,
      block_indices,
      num_blocks,
      (index_t)PAGE_SIZE,
      make_k_dram_fn(nullptr, (index_t)PAGE_SIZE),
      make_k_dram_fn(nullptr, seqlen_k - (num_blocks - 1) * PAGE_SIZE));

  auto v_nav = make_page_block_navigator<const bf16_t, 1>(
      reinterpret_cast<bf16_t const *>(paged_v_cache_ptr),
      (long_index_t)(PAGE_SIZE * kv_cache_stride),
      (long_index_t)0,
      block_indices,
      num_blocks,
      (index_t)PAGE_SIZE,
      make_v_dram_fn(nullptr, (index_t)PAGE_SIZE),
      make_v_dram_fn(nullptr, seqlen_k - (num_blocks - 1) * PAGE_SIZE));

  constexpr auto bias_dram_window_lengths =
      make_tuple(number<kM0>{}, number<kN0>{});
  auto null_bias_window = make_null_tile_window(bias_dram_window_lengths);
  auto position_encoding = EmptyPositionEncoding<float>{};
  FmhaVariant variant;

  // Bottom-left causal mask (with optional sliding window)
  // left_size=-1 means full causal; sliding_window>0 limits to last N positions
  index_t left_size =
      (sliding_window > 0) ? (index_t)(sliding_window - 1) : (index_t)-1;
  PrefillMask mask =
      ck_tile::make_generic_attention_mask_from_lr_window<PrefillMask>(
          left_size, (index_t)0, seqlen_q, seqlen_k, false);

  for (int qo_h = 0; qo_h < NUM_QO_PER_KV; qo_h++) {
    index_t i_nhead = kv_head_idx * NUM_QO_PER_KV + qo_h;

    // Q: (seqlen_q, HEAD_DIM) stride (Q_WORKSPACE_STRIDE, 1)
    bf16_t const *q_head_ptr =
        q_ptr + static_cast<long_index_t>(i_nhead) * HEAD_DIM;

    auto const q_dram_naive =
        make_naive_tensor_view<address_space_enum::global>(
            q_head_ptr,
            make_tuple(seqlen_q, (index_t)HEAD_DIM),
            make_tuple((index_t)Q_WORKSPACE_STRIDE, (index_t)1),
            number<MyPrefillPipeline::kAlignmentQ>{},
            number<1>{});

    auto const q_dram = [&]() {
      if constexpr (MyPrefillPipeline::kQLoadOnce) {
        return pad_tensor_view(
            q_dram_naive,
            make_tuple(number<kM0>{}, number<kSubQKHeaddim>{}),
            sequence<true, false>{});
      } else {
        return pad_tensor_view(q_dram_naive,
                               make_tuple(number<kM0>{}, number<kK0>{}),
                               sequence<true, false>{});
      }
    }();

    auto q_dram_window = make_tile_window(
        q_dram,
        [&]() {
          if constexpr (MyPrefillPipeline::kQLoadOnce) {
            return make_tuple(number<kM0>{}, number<kSubQKHeaddim>{});
          } else {
            return make_tuple(number<kM0>{}, number<kK0>{});
          }
        }(),
        {i_m0, 0});

    // LSE: (seqlen_q,) stride LSE_STRIDE
    float *lse_for_head = reinterpret_cast<float *>(lse_acc_ptr) +
                          query_start * LSE_STRIDE +
                          kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV +
                          kv_chunk_idx * NUM_QO_PER_KV + qo_h;

    auto const lse_dram =
        pad_tensor_view(make_naive_tensor_view<address_space_enum::global>(
                            lse_for_head,
                            make_tuple(seqlen_q),
                            make_tuple((index_t)LSE_STRIDE),
                            number<1>{},
                            number<1>{}),
                        make_tuple(number<kM0>{}),
                        sequence<true>{});
    auto lse_window =
        make_tile_window(lse_dram, make_tuple(number<kM0>{}), {i_m0});

    auto variant_params =
        ck_tile::StandardAttentionParams<PrefillMask>{mask, scale_s};

    auto o_acc_tile = MyPrefillPipeline{}(
        q_dram_window,
        make_tuple(number<kN0>{}, number<kK0>{}),
        k_nav,
        make_tuple(number<kN1>{}, number<kK1>{}),
        v_nav,
        null_bias_window,
        lse_window,
        (index_t)NUM_KV_CHUNKS,
        (index_t)i_split,
        mask,
        position_encoding,
        scale_s,
        variant,
        variant_params,
        BlockIndices{(index_t)req, i_nhead, (index_t)kv_head_idx},
        (index_t)0,
        smem_ptr,
        -ck_tile::numeric<float>::infinity());

    if constexpr (NUM_KV_CHUNKS == 1) {
      // Direct bf16 output — no merge needed
      // Prefill: seqlen_q tokens, each with stride Q_WORKSPACE_STRIDE between
      // rows
      bf16_t *o_for_head = reinterpret_cast<bf16_t *>(o_acc_ptr) +
                           query_start * Q_WORKSPACE_STRIDE +
                           (kv_head_idx * NUM_QO_PER_KV + qo_h) * HEAD_DIM;

      auto const o_dram = pad_tensor_view(
          make_naive_tensor_view<address_space_enum::global>(
              o_for_head,
              make_tuple(seqlen_q, (index_t)HEAD_DIM),
              make_tuple((index_t)Q_WORKSPACE_STRIDE, (index_t)1),
              number<MyPrefillPipeline::kAlignmentOacc>{},
              number<1>{}),
          make_tuple(number<kM0>{}, number<kN1>{}),
          sequence<true, false>{});
      auto o_window = make_tile_window(
          o_dram, make_tuple(number<kM0>{}, number<kN1>{}), {i_m0, i_n1});

      FmhaEpilogueBf16{}(o_window, o_acc_tile, nullptr);

      // Fuse per-head attention sink correction for prefill (NUM_KV_CHUNKS==1).
      // After the bf16 output is written, multiply by 1/(1+exp(sink-LSE)) per
      // token. LSE was just written by the pipeline so it's hot in cache. This
      // eliminates the standalone attention_sink_layer task during prefill.
      if (sinks_ptr != nullptr) {
        __syncthreads();
        bf16_t const *d_sinks = reinterpret_cast<bf16_t const *>(sinks_ptr);
        float sink_val = type_convert<float>(d_sinks[i_nhead]);
        int const total_threads = blockDim.x;
        int const tid = threadIdx.x;
        for (index_t s = 0; s < seqlen_q; s++) {
          float lse_val = lse_for_head[s * LSE_STRIDE];
          float correction = 1.0f / (1.0f + expf(sink_val - lse_val));
          bf16_t *o_row = o_for_head + s * Q_WORKSPACE_STRIDE;
          for (int d = tid; d < HEAD_DIM; d += total_threads) {
            float v = type_convert<float>(o_row[d]);
            o_row[d] = type_convert<bf16_t>(v * correction);
          }
        }
      }
    } else {
      // Float32 accumulator for split-KV merge
      float *o_for_head =
          reinterpret_cast<float *>(o_acc_ptr) + query_start * O_ACC_STRIDE +
          kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV * HEAD_DIM +
          kv_chunk_idx * NUM_QO_PER_KV * HEAD_DIM + qo_h * HEAD_DIM;

      auto const o_dram =
          pad_tensor_view(make_naive_tensor_view<address_space_enum::global>(
                              o_for_head,
                              make_tuple(seqlen_q, (index_t)HEAD_DIM),
                              make_tuple((index_t)O_ACC_STRIDE, (index_t)1),
                              number<MyPrefillPipeline::kAlignmentOacc>{},
                              number<1>{}),
                          make_tuple(number<kM0>{}, number<kN1>{}),
                          sequence<true, false>{});
      auto o_window = make_tile_window(
          o_dram, make_tuple(number<kM0>{}, number<kN1>{}), {i_m0, i_n1});

      FmhaEpilogue{}(o_window, o_acc_tile, nullptr);
    }
  }
}

// ============================================================================
// Dispatch: thin wrapper that checks seqlen_q and routes to the right path
// ============================================================================
template <typename T,
          int NUM_QO_PER_KV,
          int HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE_T,
          int NUM_KV_HEADS_T,
          bool DECODE_ONLY = false>
__device__ __forceinline__ void paged_attention_ck_fmha_split_kv_impl(
    void const *q_workspace_ptr,
    void *paged_k_cache_ptr,
    void *paged_v_cache_ptr,
    void *o_acc_ptr,
    void *lse_acc_ptr,
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int16_t request_id,
    int kv_head_idx,
    int kv_chunk_idx,
    float scale_s,
    int sliding_window = 0,
    void const *sinks_ptr = nullptr) {
  int const req = request_id;
  const ck_tile::index_t query_start = qo_indptr_buffer_ptr[req];
  const ck_tile::index_t query_end = qo_indptr_buffer_ptr[req + 1];
  if (query_start == query_end) {
    return;
  }

  ck_tile::index_t seqlen_q = query_end - query_start;

  if (seqlen_q == 1) {
    if constexpr (HEAD_DIM == 64) {
      // Fast MFMA decode path for HD=64 (GPT-OSS 120B).
      paged_attention_minimal_decode_hd64<T,
                                          NUM_QO_PER_KV,
                                          HEAD_DIM,
                                          PAGE_SIZE,
                                          MAX_SEQ_LEN,
                                          NUM_KV_CHUNKS,
                                          Q_WORKSPACE_STRIDE,
                                          KV_CACHE_STRIDE_T,
                                          NUM_KV_HEADS_T>(
          q_workspace_ptr,
          paged_k_cache_ptr,
          paged_v_cache_ptr,
          o_acc_ptr,
          lse_acc_ptr,
          qo_indptr_buffer_ptr,
          paged_kv_indptr_buffer_ptr,
          paged_kv_indices_buffer_ptr,
          paged_kv_last_page_len_buffer_ptr,
          request_id,
          kv_head_idx,
          kv_chunk_idx,
          scale_s,
          sliding_window,
          sinks_ptr);
    } else {
      paged_attention_ck_fmha_decode<T,
                                     NUM_QO_PER_KV,
                                     HEAD_DIM,
                                     PAGE_SIZE,
                                     MAX_SEQ_LEN,
                                     NUM_KV_CHUNKS,
                                     Q_WORKSPACE_STRIDE,
                                     KV_CACHE_STRIDE_T,
                                     NUM_KV_HEADS_T>(
          q_workspace_ptr,
          paged_k_cache_ptr,
          paged_v_cache_ptr,
          o_acc_ptr,
          lse_acc_ptr,
          qo_indptr_buffer_ptr,
          paged_kv_indptr_buffer_ptr,
          paged_kv_indices_buffer_ptr,
          paged_kv_last_page_len_buffer_ptr,
          request_id,
          kv_head_idx,
          kv_chunk_idx,
          scale_s,
          sliding_window);
    }
  } else {
    if constexpr (!DECODE_ONLY) {
      paged_attention_ck_fmha_prefill<T,
                                      NUM_QO_PER_KV,
                                      HEAD_DIM,
                                      PAGE_SIZE,
                                      MAX_SEQ_LEN,
                                      NUM_KV_CHUNKS,
                                      Q_WORKSPACE_STRIDE,
                                      KV_CACHE_STRIDE_T,
                                      NUM_KV_HEADS_T>(
          q_workspace_ptr,
          paged_k_cache_ptr,
          paged_v_cache_ptr,
          o_acc_ptr,
          lse_acc_ptr,
          qo_indptr_buffer_ptr,
          paged_kv_indptr_buffer_ptr,
          paged_kv_indices_buffer_ptr,
          paged_kv_last_page_len_buffer_ptr,
          request_id,
          kv_head_idx,
          kv_chunk_idx,
          scale_s,
          seqlen_q,
          sliding_window,
          sinks_ptr);
    }
  }
}

} // namespace kernel
