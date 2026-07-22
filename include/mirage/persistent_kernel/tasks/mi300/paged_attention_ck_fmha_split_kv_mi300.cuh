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
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_nwarp_sshuffle_qr_ks_vs.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/block/block_position_encoding.hpp"
#include "ck_tile/ops/fmha/block/page_block_navigator.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"

#ifndef CK_TILE_FMHA_FWD_FAST_EXP2
#define CK_TILE_FMHA_FWD_FAST_EXP2 1
#endif

namespace kernel {

// ============================================================================
// Shared tile configuration
// ============================================================================
using FmhaBlockTile = ck_tile::sequence<64, 128, 32, 128, 32, 128>;
using Gemm0BlockWarps = ck_tile::sequence<4, 1, 1>;
using Gemm0WarpTile = ck_tile::sequence<16, 16, 16>;
using Gemm1BlockWarps = ck_tile::sequence<4, 1, 1>;
using Gemm1WarpTile = ck_tile::sequence<16, 16, 16>;

using FmhaShape = ck_tile::TileFmhaShape<
    FmhaBlockTile, Gemm0BlockWarps, Gemm0WarpTile,
    Gemm1BlockWarps, Gemm1WarpTile, true /* VRowMajor */>;

using FmhaVariant = ck_tile::ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>;

// float32 epilogue for split-KV with merge (NUM_KV_CHUNKS > 1)
using FmhaEpilogueProblem = ck_tile::Default2DEpilogueProblem<float, float, true, false, false>;
using FmhaEpilogue = ck_tile::Default2DEpilogue<FmhaEpilogueProblem>;

// bf16 epilogue for direct output (NUM_KV_CHUNKS == 1, no merge needed)
using FmhaEpilogueBf16Problem = ck_tile::Default2DEpilogueProblem<float, ck_tile::bf16_t, true, false, false>;
using FmhaEpilogueBf16 = ck_tile::Default2DEpilogue<FmhaEpilogueBf16Problem>;

struct BlockIndices {
  ck_tile::index_t batch_idx;
  ck_tile::index_t qo_head_idx;
  ck_tile::index_t kv_head_idx;
};

// Decode tile: kM0=16 with <1,4,1> warps for nwarp sshuffle pipeline.
// 1 warp in M (kM0=16), 4 warps split KV sequence in parallel.
// 4 warps * 64 = 256 threads = persistent kernel block size.
// kM0=16 reduces M-tile padding (75% vs 93.75% with kM0=64).
// The nwarp pipeline handles warp-level KV split and merge via LDS shuffle.
using FmhaBlockTileDecode = ck_tile::sequence<16, 128, 32, 128, 32, 128>;
using Gemm0BlockWarpsDecode = ck_tile::sequence<1, 4, 1>;
using Gemm0WarpTileDecode = ck_tile::sequence<16, 16, 16>;
using Gemm1BlockWarpsDecode = ck_tile::sequence<1, 4, 1>;
using Gemm1WarpTileDecode = ck_tile::sequence<16, 16, 16>;

using FmhaShapeDecode = ck_tile::TileFmhaShape<
    FmhaBlockTileDecode, Gemm0BlockWarpsDecode, Gemm0WarpTileDecode,
    Gemm1BlockWarpsDecode, Gemm1WarpTileDecode, true /* VRowMajor */>;

// ============================================================================
// Decode path types: merge GQA heads into M dimension, no mask
// ============================================================================
using DecodeTraits = ck_tile::TileFmhaFwdSplitKVTraits<
    true, true, false, false, false,
    ck_tile::BlockAttentionBiasEnum::NO_BIAS,
    false, true, false, true, true,
    true,   // kMergeNumHeadGroupsSeqLenQ
    -1, false>;
using DecodeMask = ck_tile::SimplifiedGenericAttentionMask<false>;
using DecodePipelineProblem = ck_tile::BlockFmhaFwdSplitKVPipelineProblem<
    ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t,
    float, float, ck_tile::bf16_t, float, ck_tile::bf16_t,
    float, float,
    FmhaShapeDecode, true, FmhaVariant, DecodeMask, DecodeTraits>;
// Use nwarp sshuffle pipeline: splits KV across warps within the workgroup
// Each warp processes a different KV range, then results merged via LDS shuffle
using DecodePipeline = ck_tile::BlockFmhaFwdSplitKVPipelineNWarpSShuffleQRKSVS<DecodePipelineProblem>;

// ============================================================================
// Prefill path types: per-head loop with causal mask
// ============================================================================
using PrefillTraits = ck_tile::TileFmhaFwdSplitKVTraits<
    true, true, false, false, false,
    ck_tile::BlockAttentionBiasEnum::NO_BIAS,
    false, true, false, true, true,
    false,  // kMergeNumHeadGroupsSeqLenQ
    -1, false>;
using PrefillMask = ck_tile::SimplifiedGenericAttentionMask<true>;
using PrefillPipelineProblem = ck_tile::BlockFmhaFwdSplitKVPipelineProblem<
    ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t,
    float, float, ck_tile::bf16_t, float, ck_tile::bf16_t,
    float, float,
    FmhaShape, true, FmhaVariant, PrefillMask, PrefillTraits>;
using PrefillPipeline = ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVS<PrefillPipelineProblem>;

// Compile-time LDS check
static_assert(DecodePipeline::GetSmemSize() <= mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
              "CK FMHA decode pipeline LDS exceeds available shared memory");
static_assert(PrefillPipeline::GetSmemSize() <= mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
              "CK FMHA prefill pipeline LDS exceeds available shared memory");

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
__device__ __noinline__ void paged_attention_ck_fmha_decode(
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
    float scale_s)
{
  using namespace ck_tile;

  extern __shared__ char smem_ptr[];

  constexpr int kv_cache_stride = KV_CACHE_STRIDE_T;
  constexpr index_t kM0 = DecodePipeline::kM0;
  constexpr index_t kN0 = DecodePipeline::kN0;
  constexpr index_t kK0 = DecodePipeline::kK0;
  constexpr index_t kN1 = DecodePipeline::kN1;
  constexpr index_t kK1 = DecodePipeline::kK1;

  constexpr index_t i_m0 = 0;
  constexpr index_t i_n1 = 0;
  const index_t i_split = kv_chunk_idx;

  constexpr int LSE_STRIDE = NUM_KV_HEADS_T * NUM_KV_CHUNKS * NUM_QO_PER_KV;
  constexpr int O_ACC_STRIDE = LSE_STRIDE * HEAD_DIM;
  constexpr index_t effective_sq = NUM_QO_PER_KV;

  const int req = request_id;
  const index_t query_start = qo_indptr_buffer_ptr[req];

  const index_t first_page = paged_kv_indptr_buffer_ptr[req];
  const index_t last_page = paged_kv_indptr_buffer_ptr[req + 1];
  const index_t num_pages = last_page - first_page;
  index_t seqlen_k = (num_pages - 1) * PAGE_SIZE +
                      paged_kv_last_page_len_buffer_ptr[req];

  long_index_t batch_offset_q = query_start * Q_WORKSPACE_STRIDE;
  const bf16_t* q_ptr = reinterpret_cast<const bf16_t*>(q_workspace_ptr) + batch_offset_q;

  // K/V view factories
  auto make_k_dram_fn = [&](const bf16_t* data, index_t height) {
    const auto k = make_naive_tensor_view<address_space_enum::global>(
        data, make_tuple(height, (index_t)HEAD_DIM),
        make_tuple((index_t)kv_cache_stride, (index_t)1),
        number<DecodePipeline::kAlignmentK>{}, number<1>{});
    return pad_tensor_view(k, make_tuple(number<kN0>{}, number<kK0>{}),
                           sequence<false, false>{});
  };

  auto make_v_dram_fn = [&](const bf16_t* data, index_t length) {
    const auto v = make_naive_tensor_view<address_space_enum::global>(
        data, make_tuple(length, (index_t)HEAD_DIM),
        make_tuple((index_t)kv_cache_stride, (index_t)1),
        number<DecodePipeline::kAlignmentV>{}, number<1>{});
    const auto vt = transform_tensor_view(v,
        make_tuple(make_pass_through_transform((index_t)HEAD_DIM),
                   make_pass_through_transform(length)),
        make_tuple(sequence<1>{}, sequence<0>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));
    return pad_tensor_view(vt, make_tuple(number<kN1>{}, number<kK1>{}),
                           sequence<false, true>{});
  };

  // Page block navigators
  const int32_t* block_indices =
      reinterpret_cast<const int32_t*>(paged_kv_indices_buffer_ptr) + first_page;
  const index_t num_blocks = integer_divide_ceil(seqlen_k, (index_t)PAGE_SIZE);

  auto k_nav = make_page_block_navigator<const bf16_t, 0>(
      reinterpret_cast<const bf16_t*>(paged_k_cache_ptr),
      (long_index_t)(PAGE_SIZE * kv_cache_stride), (long_index_t)0,
      block_indices, num_blocks, (index_t)PAGE_SIZE,
      make_k_dram_fn(nullptr, (index_t)PAGE_SIZE),
      make_k_dram_fn(nullptr, seqlen_k - (num_blocks - 1) * PAGE_SIZE));

  auto v_nav = make_page_block_navigator<const bf16_t, 1>(
      reinterpret_cast<const bf16_t*>(paged_v_cache_ptr),
      (long_index_t)(PAGE_SIZE * kv_cache_stride), (long_index_t)0,
      block_indices, num_blocks, (index_t)PAGE_SIZE,
      make_v_dram_fn(nullptr, (index_t)PAGE_SIZE),
      make_v_dram_fn(nullptr, seqlen_k - (num_blocks - 1) * PAGE_SIZE));

  // Q: (NUM_QO_PER_KV, HEAD_DIM) stride (HEAD_DIM, 1) — merged qo_heads
  const bf16_t* q_merge_ptr = q_ptr +
      static_cast<long_index_t>(kv_head_idx) * NUM_QO_PER_KV * HEAD_DIM;

  const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
      q_merge_ptr,
      make_tuple((index_t)effective_sq, (index_t)HEAD_DIM),
      make_tuple((index_t)HEAD_DIM, (index_t)1),
      number<DecodePipeline::kAlignmentQ>{}, number<1>{});

  const auto q_dram = [&]() {
    if constexpr (DecodePipeline::kQLoadOnce) {
      return pad_tensor_view(q_dram_naive,
          make_tuple(number<kM0>{}, number<DecodePipeline::kSubQKHeaddim>{}),
          sequence<true, false>{});
    } else {
      return pad_tensor_view(q_dram_naive,
          make_tuple(number<kM0>{}, number<kK0>{}),
          sequence<true, false>{});
    }
  }();

  auto q_dram_window = make_tile_window(q_dram,
      [&]() {
        if constexpr (DecodePipeline::kQLoadOnce)
          return make_tuple(number<kM0>{}, number<DecodePipeline::kSubQKHeaddim>{});
        else
          return make_tuple(number<kM0>{}, number<kK0>{});
      }(), {i_m0, 0});

  // LSE: (NUM_QO_PER_KV,) stride (1,)
  float* lse_ptr = reinterpret_cast<float*>(lse_acc_ptr) +
      query_start * LSE_STRIDE +
      kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV +
      kv_chunk_idx * NUM_QO_PER_KV;

  const auto lse_dram = pad_tensor_view(
      make_naive_tensor_view<address_space_enum::global>(
          lse_ptr, make_tuple((index_t)effective_sq), make_tuple((index_t)1),
          number<1>{}, number<1>{}),
      make_tuple(number<kM0>{}), sequence<true>{});
  auto lse_window = make_tile_window(lse_dram, make_tuple(number<kM0>{}), {i_m0});

  constexpr auto bias_dram_window_lengths = make_tuple(number<kM0>{}, number<kN0>{});
  auto null_bias_window = make_null_tile_window(bias_dram_window_lengths);
  auto position_encoding = EmptyPositionEncoding<float>{};
  FmhaVariant variant;

  DecodeMask decode_mask = ck_tile::make_generic_attention_mask_from_lr_window<DecodeMask>(
      (index_t)-1, (index_t)0, (index_t)effective_sq, seqlen_k, false);
  auto variant_params = ck_tile::StandardAttentionParams<DecodeMask>{decode_mask, scale_s};

  auto o_acc_tile = DecodePipeline{}(
      q_dram_window,
      make_tuple(number<kN0>{}, number<kK0>{}),
      k_nav,
      make_tuple(number<kN1>{}, number<kK1>{}),
      v_nav,
      null_bias_window,
      lse_window,
      (index_t)NUM_KV_CHUNKS,
      (index_t)i_split,
      decode_mask,
      position_encoding,
      scale_s,
      variant,
      variant_params,
      BlockIndices{(index_t)req, (index_t)(kv_head_idx * NUM_QO_PER_KV), (index_t)kv_head_idx},
      (index_t)0,
      smem_ptr,
      -ck_tile::numeric<float>::infinity());

  if constexpr (NUM_KV_CHUNKS == 1) {
    // Direct bf16 output — no merge needed
    // Decode: effective_sq = NUM_QO_PER_KV qo heads packed contiguously
    // Output layout: [num_tokens, num_q_heads * HEAD_DIM]
    // Within a token, heads are contiguous: stride between rows = HEAD_DIM
    // Token stride = Q_WORKSPACE_STRIDE (= num_q_heads * HEAD_DIM)
    bf16_t* o_ptr = reinterpret_cast<bf16_t*>(o_acc_ptr) +
        query_start * Q_WORKSPACE_STRIDE +
        kv_head_idx * NUM_QO_PER_KV * HEAD_DIM;

    const auto o_dram = pad_tensor_view(
        make_naive_tensor_view<address_space_enum::global>(
            o_ptr, make_tuple((index_t)effective_sq, (index_t)HEAD_DIM),
            make_tuple((index_t)HEAD_DIM, (index_t)1),
            number<DecodePipeline::kAlignmentOacc>{}, number<1>{}),
        make_tuple(number<kM0>{}, number<kN1>{}),
        sequence<true, false>{});
    auto o_window = make_tile_window(o_dram, make_tuple(number<kM0>{}, number<kN1>{}), {i_m0, i_n1});

    FmhaEpilogueBf16{}(o_window, o_acc_tile, nullptr);
  } else {
    // Float32 accumulator for split-KV merge
    // O_acc: (NUM_QO_PER_KV, HEAD_DIM) stride (HEAD_DIM, 1)
    float* o_ptr = reinterpret_cast<float*>(o_acc_ptr) +
        query_start * O_ACC_STRIDE +
        kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV * HEAD_DIM +
        kv_chunk_idx * NUM_QO_PER_KV * HEAD_DIM;

    const auto o_dram = pad_tensor_view(
        make_naive_tensor_view<address_space_enum::global>(
            o_ptr, make_tuple((index_t)effective_sq, (index_t)HEAD_DIM),
            make_tuple((index_t)HEAD_DIM, (index_t)1),
            number<DecodePipeline::kAlignmentOacc>{}, number<1>{}),
        make_tuple(number<kM0>{}, number<kN1>{}),
        sequence<true, false>{});
    auto o_window = make_tile_window(o_dram, make_tuple(number<kM0>{}, number<kN1>{}), {i_m0, i_n1});

    FmhaEpilogue{}(o_window, o_acc_tile, nullptr);
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
    ck_tile::index_t seqlen_q)
{
  using namespace ck_tile;

  extern __shared__ char smem_ptr[];

  constexpr int kv_cache_stride = KV_CACHE_STRIDE_T;
  constexpr index_t kM0 = PrefillPipeline::kM0;
  constexpr index_t kN0 = PrefillPipeline::kN0;
  constexpr index_t kK0 = PrefillPipeline::kK0;
  constexpr index_t kN1 = PrefillPipeline::kN1;
  constexpr index_t kK1 = PrefillPipeline::kK1;
  constexpr index_t kSubQKHeaddim = PrefillPipeline::kSubQKHeaddim;

  constexpr index_t i_m0 = 0;
  constexpr index_t i_n1 = 0;
  const index_t i_split = kv_chunk_idx;

  constexpr int LSE_STRIDE = NUM_KV_HEADS_T * NUM_KV_CHUNKS * NUM_QO_PER_KV;
  constexpr int O_ACC_STRIDE = LSE_STRIDE * HEAD_DIM;

  const int req = request_id;
  const index_t query_start = qo_indptr_buffer_ptr[req];

  const index_t first_page = paged_kv_indptr_buffer_ptr[req];
  const index_t last_page = paged_kv_indptr_buffer_ptr[req + 1];
  const index_t num_pages = last_page - first_page;
  index_t seqlen_k = (num_pages - 1) * PAGE_SIZE +
                      paged_kv_last_page_len_buffer_ptr[req];

  long_index_t batch_offset_q = query_start * Q_WORKSPACE_STRIDE;
  const bf16_t* q_ptr = reinterpret_cast<const bf16_t*>(q_workspace_ptr) + batch_offset_q;

  // K/V view factories
  auto make_k_dram_fn = [&](const bf16_t* data, index_t height) {
    const auto k = make_naive_tensor_view<address_space_enum::global>(
        data, make_tuple(height, (index_t)HEAD_DIM),
        make_tuple((index_t)kv_cache_stride, (index_t)1),
        number<PrefillPipeline::kAlignmentK>{}, number<1>{});
    return pad_tensor_view(k, make_tuple(number<kN0>{}, number<kK0>{}),
                           sequence<false, false>{});
  };

  auto make_v_dram_fn = [&](const bf16_t* data, index_t length) {
    const auto v = make_naive_tensor_view<address_space_enum::global>(
        data, make_tuple(length, (index_t)HEAD_DIM),
        make_tuple((index_t)kv_cache_stride, (index_t)1),
        number<PrefillPipeline::kAlignmentV>{}, number<1>{});
    const auto vt = transform_tensor_view(v,
        make_tuple(make_pass_through_transform((index_t)HEAD_DIM),
                   make_pass_through_transform(length)),
        make_tuple(sequence<1>{}, sequence<0>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));
    return pad_tensor_view(vt, make_tuple(number<kN1>{}, number<kK1>{}),
                           sequence<false, true>{});
  };

  // Page block navigators
  const int32_t* block_indices =
      reinterpret_cast<const int32_t*>(paged_kv_indices_buffer_ptr) + first_page;
  const index_t num_blocks = integer_divide_ceil(seqlen_k, (index_t)PAGE_SIZE);

  auto k_nav = make_page_block_navigator<const bf16_t, 0>(
      reinterpret_cast<const bf16_t*>(paged_k_cache_ptr),
      (long_index_t)(PAGE_SIZE * kv_cache_stride), (long_index_t)0,
      block_indices, num_blocks, (index_t)PAGE_SIZE,
      make_k_dram_fn(nullptr, (index_t)PAGE_SIZE),
      make_k_dram_fn(nullptr, seqlen_k - (num_blocks - 1) * PAGE_SIZE));

  auto v_nav = make_page_block_navigator<const bf16_t, 1>(
      reinterpret_cast<const bf16_t*>(paged_v_cache_ptr),
      (long_index_t)(PAGE_SIZE * kv_cache_stride), (long_index_t)0,
      block_indices, num_blocks, (index_t)PAGE_SIZE,
      make_v_dram_fn(nullptr, (index_t)PAGE_SIZE),
      make_v_dram_fn(nullptr, seqlen_k - (num_blocks - 1) * PAGE_SIZE));

  constexpr auto bias_dram_window_lengths = make_tuple(number<kM0>{}, number<kN0>{});
  auto null_bias_window = make_null_tile_window(bias_dram_window_lengths);
  auto position_encoding = EmptyPositionEncoding<float>{};
  FmhaVariant variant;

  // Bottom-left causal mask
  PrefillMask mask = ck_tile::make_generic_attention_mask_from_lr_window<PrefillMask>(
      (index_t)-1, (index_t)0, seqlen_q, seqlen_k, false);

  for (int qo_h = 0; qo_h < NUM_QO_PER_KV; qo_h++) {
    index_t i_nhead = kv_head_idx * NUM_QO_PER_KV + qo_h;

    // Q: (seqlen_q, HEAD_DIM) stride (Q_WORKSPACE_STRIDE, 1)
    const bf16_t* q_head_ptr = q_ptr +
        static_cast<long_index_t>(i_nhead) * HEAD_DIM;

    const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
        q_head_ptr,
        make_tuple(seqlen_q, (index_t)HEAD_DIM),
        make_tuple((index_t)Q_WORKSPACE_STRIDE, (index_t)1),
        number<PrefillPipeline::kAlignmentQ>{}, number<1>{});

    const auto q_dram = [&]() {
      if constexpr (PrefillPipeline::kQLoadOnce) {
        return pad_tensor_view(q_dram_naive,
            make_tuple(number<kM0>{}, number<kSubQKHeaddim>{}),
            sequence<true, false>{});
      } else {
        return pad_tensor_view(q_dram_naive,
            make_tuple(number<kM0>{}, number<kK0>{}),
            sequence<true, false>{});
      }
    }();

    auto q_dram_window = make_tile_window(q_dram,
        [&]() {
          if constexpr (PrefillPipeline::kQLoadOnce)
            return make_tuple(number<kM0>{}, number<kSubQKHeaddim>{});
          else
            return make_tuple(number<kM0>{}, number<kK0>{});
        }(), {i_m0, 0});

    // LSE: (seqlen_q,) stride LSE_STRIDE
    float* lse_for_head = reinterpret_cast<float*>(lse_acc_ptr) +
        query_start * LSE_STRIDE +
        kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV +
        kv_chunk_idx * NUM_QO_PER_KV + qo_h;

    const auto lse_dram = pad_tensor_view(
        make_naive_tensor_view<address_space_enum::global>(
            lse_for_head, make_tuple(seqlen_q), make_tuple((index_t)LSE_STRIDE),
            number<1>{}, number<1>{}),
        make_tuple(number<kM0>{}), sequence<true>{});
    auto lse_window = make_tile_window(lse_dram, make_tuple(number<kM0>{}), {i_m0});

    auto variant_params = ck_tile::StandardAttentionParams<PrefillMask>{mask, scale_s};

    auto o_acc_tile = PrefillPipeline{}(
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
      // Prefill: seqlen_q tokens, each with stride Q_WORKSPACE_STRIDE between rows
      bf16_t* o_for_head = reinterpret_cast<bf16_t*>(o_acc_ptr) +
          query_start * Q_WORKSPACE_STRIDE +
          (kv_head_idx * NUM_QO_PER_KV + qo_h) * HEAD_DIM;

      const auto o_dram = pad_tensor_view(
          make_naive_tensor_view<address_space_enum::global>(
              o_for_head,
              make_tuple(seqlen_q, (index_t)HEAD_DIM),
              make_tuple((index_t)Q_WORKSPACE_STRIDE, (index_t)1),
              number<PrefillPipeline::kAlignmentOacc>{}, number<1>{}),
          make_tuple(number<kM0>{}, number<kN1>{}),
          sequence<true, false>{});
      auto o_window = make_tile_window(o_dram, make_tuple(number<kM0>{}, number<kN1>{}), {i_m0, i_n1});

      FmhaEpilogueBf16{}(o_window, o_acc_tile, nullptr);
    } else {
      // Float32 accumulator for split-KV merge
      float* o_for_head = reinterpret_cast<float*>(o_acc_ptr) +
          query_start * O_ACC_STRIDE +
          kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV * HEAD_DIM +
          kv_chunk_idx * NUM_QO_PER_KV * HEAD_DIM +
          qo_h * HEAD_DIM;

      const auto o_dram = pad_tensor_view(
          make_naive_tensor_view<address_space_enum::global>(
              o_for_head,
              make_tuple(seqlen_q, (index_t)HEAD_DIM),
              make_tuple((index_t)O_ACC_STRIDE, (index_t)1),
              number<PrefillPipeline::kAlignmentOacc>{}, number<1>{}),
          make_tuple(number<kM0>{}, number<kN1>{}),
          sequence<true, false>{});
      auto o_window = make_tile_window(o_dram, make_tuple(number<kM0>{}, number<kN1>{}), {i_m0, i_n1});

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
          int NUM_KV_HEADS_T>
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
    float scale_s)
{
  const int req = request_id;
  const ck_tile::index_t query_start = qo_indptr_buffer_ptr[req];
  const ck_tile::index_t query_end = qo_indptr_buffer_ptr[req + 1];
  if (query_start == query_end) return;

  ck_tile::index_t seqlen_q = query_end - query_start;

  if (seqlen_q == 1) {
    paged_attention_minimal_decode<T, NUM_QO_PER_KV, HEAD_DIM, PAGE_SIZE,
        MAX_SEQ_LEN, NUM_KV_CHUNKS, Q_WORKSPACE_STRIDE, KV_CACHE_STRIDE_T,
        NUM_KV_HEADS_T>(
        q_workspace_ptr, paged_k_cache_ptr, paged_v_cache_ptr,
        o_acc_ptr, lse_acc_ptr,
        qo_indptr_buffer_ptr, paged_kv_indptr_buffer_ptr,
        paged_kv_indices_buffer_ptr, paged_kv_last_page_len_buffer_ptr,
        request_id, kv_head_idx, kv_chunk_idx, scale_s);
  } else {
    paged_attention_ck_fmha_prefill<T, NUM_QO_PER_KV, HEAD_DIM, PAGE_SIZE,
        MAX_SEQ_LEN, NUM_KV_CHUNKS, Q_WORKSPACE_STRIDE, KV_CACHE_STRIDE_T,
        NUM_KV_HEADS_T>(
        q_workspace_ptr, paged_k_cache_ptr, paged_v_cache_ptr,
        o_acc_ptr, lse_acc_ptr,
        qo_indptr_buffer_ptr, paged_kv_indptr_buffer_ptr,
        paged_kv_indices_buffer_ptr, paged_kv_last_page_len_buffer_ptr,
        request_id, kv_head_idx, kv_chunk_idx, scale_s, seqlen_q);
  }

}

} // namespace kernel
