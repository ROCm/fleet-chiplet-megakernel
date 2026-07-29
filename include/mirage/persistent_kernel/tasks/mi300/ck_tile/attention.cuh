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

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace mirage_ck {

using namespace ck_tile;

// ============================================================================
// Type definitions for attention
// ============================================================================
using bf16 = ck_tile::bf16_t;
using fp32 = float;

// Warp GEMM types for attention computation
// Q*K^T: uses 16x16x16 MFMA (Q_ROWS x KV_TILE x HEAD_DIM)
using WarpGemmQK = ck_tile::WarpGemmMfmaBf16Bf16F32M16N16K16;

// P*V: uses 16x16x16 MFMA with transposed C distribution for better output
// layout
using WarpGemmPV =
    ck_tile::WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution;

// ============================================================================
// MFMA vector types
// ============================================================================
typedef short __attribute__((ext_vector_type(4))) attn_bf16x4;
typedef float __attribute__((ext_vector_type(4))) attn_float4;

union attn_bf16x4_union {
  unsigned short u16[4];
  uint64_t u64;
  attn_bf16x4 vec;
};

union attn_bf16x8_union {
  unsigned short u16[8];
  __uint128_t u128;
};

// ============================================================================
// Attention configuration
// ============================================================================
template <int NUM_QO_HEADS_,
          int NUM_KV_HEADS_,
          int HEAD_DIM_,
          int MAX_TOKENS_,
          int KV_TILE_SIZE_ = 32>
struct AttentionConfig {
  static constexpr int NUM_QO_HEADS = NUM_QO_HEADS_;
  static constexpr int NUM_KV_HEADS = NUM_KV_HEADS_;
  static constexpr int HEAD_DIM = HEAD_DIM_;
  static constexpr int MAX_TOKENS = MAX_TOKENS_;
  static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;

  static constexpr int NUM_QO_PER_KV = NUM_QO_HEADS / NUM_KV_HEADS;
  static constexpr int Q_ROWS = MAX_TOKENS * NUM_QO_PER_KV;

  // MFMA tile sizes
  static constexpr int MFMA_M = 16;
  static constexpr int MFMA_N = 16;
  static constexpr int MFMA_K = 16;

  // Number of tiles
  static constexpr int Q_TILES_M = (Q_ROWS + MFMA_M - 1) / MFMA_M;
  static constexpr int KV_TILES_N = (KV_TILE_SIZE + MFMA_N - 1) / MFMA_N;
  static constexpr int HEAD_TILES_K = HEAD_DIM / MFMA_K;

  // Thread configuration (256 threads, 4 warps on AMD)
  static constexpr int NUM_THREADS = 256;
  static constexpr int NUM_WARPS = NUM_THREADS / 64;
};

// ============================================================================
// Compute Q * K^T using MFMA
// Computes scores[Q_ROWS, KV_TILE_SIZE] = Q[Q_ROWS, HEAD_DIM] @ K^T[HEAD_DIM,
// KV_TILE_SIZE]
// ============================================================================
template <typename Config>
__device__ __forceinline__ void
    compute_qk_mfma(bf16 const *__restrict__ s_q, // [Q_ROWS, HEAD_DIM]
                    bf16 const *__restrict__ s_k, // [KV_TILE_SIZE, HEAD_DIM]
                    float *__restrict__ s_scores, // [Q_ROWS, KV_TILE_SIZE]
                    int q_rows,                   // actual number of Q rows
                    int curr_kv_len,  // actual number of KV positions
                    float sm_scale,   // softmax scale
                    int tile_start,   // KV tile start position
                    int seq_len,      // total sequence length
                    int num_tokens) { // number of new tokens

  constexpr int MFMA_M = Config::MFMA_M;
  constexpr int MFMA_N = Config::MFMA_N;
  constexpr int MFMA_K = Config::MFMA_K;
  constexpr int HEAD_DIM = Config::HEAD_DIM;
  constexpr int KV_TILE_SIZE = Config::KV_TILE_SIZE;
  constexpr int HEAD_TILES_K = Config::HEAD_TILES_K;
  constexpr int Q_TILES_M = Config::Q_TILES_M;
  constexpr int KV_TILES_N = Config::KV_TILES_N;
  constexpr int NUM_QO_PER_KV = Config::NUM_QO_PER_KV;

  int lane = threadIdx.x % 64;
  int warp_id = threadIdx.x / 64;

  // Each warp handles one KV tile column
  if (warp_id < KV_TILES_N) {
    int tile_n = warp_id;

    for (int tile_m = 0; tile_m < Q_TILES_M; tile_m++) {
      attn_float4 accum = {0.0f, 0.0f, 0.0f, 0.0f};

// Iterate over HEAD_DIM in MFMA_K chunks
#pragma unroll
      for (int tile_k = 0; tile_k < HEAD_TILES_K; tile_k++) {
        int k_base = tile_k * MFMA_K;

        // Load Q fragment
        int q_row = tile_m * MFMA_M + (lane % 16);
        int q_col = k_base + (lane / 16) * 4;

        attn_bf16x4_union reg_q;
        if (q_row < q_rows) {
          reg_q.u64 = *reinterpret_cast<uint64_t const *>(
              &s_q[q_row * HEAD_DIM + q_col]);
        } else {
          reg_q.u64 = 0;
        }

        // Load K^T fragment (K is stored as [KV_TILE_SIZE, HEAD_DIM])
        int k_row = tile_n * MFMA_N + (lane % 16);
        int k_col = k_base + (lane / 16) * 4;

        attn_bf16x4_union reg_k;
        if (k_row < curr_kv_len) {
          reg_k.u64 = *reinterpret_cast<uint64_t const *>(
              &s_k[k_row * HEAD_DIM + k_col]);
        } else {
          reg_k.u64 = 0;
        }

        // MFMA: C += A * B
        accum = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
            reg_q.vec, reg_k.vec, accum, 0, 0, 0);
      }

      // Store scores and apply scale + causal mask
      int out_row_base = tile_m * MFMA_M + (lane / 16) * 4;
      int out_col = tile_n * MFMA_N + (lane % 16);

      if (out_col < curr_kv_len) {
        int kv_pos = tile_start + out_col;
#pragma unroll
        for (int r = 0; r < 4; r++) {
          int out_row = out_row_base + r;
          if (out_row < q_rows) {
            int token = out_row / NUM_QO_PER_KV;
            int q_pos = seq_len - num_tokens + token;
            float score = accum[r] * sm_scale;
            // Causal mask
            if (kv_pos > q_pos) {
              score = -INFINITY;
            }
            s_scores[out_row * KV_TILE_SIZE + out_col] = score;
          }
        }
      }
    }
  }
}

// ============================================================================
// Compute P * V using MFMA
// Computes O[Q_ROWS, HEAD_DIM] += P[Q_ROWS, KV_TILE_SIZE] @ V[KV_TILE_SIZE,
// HEAD_DIM]
// ============================================================================
template <typename Config>
__device__ __forceinline__ void compute_pv_mfma(
    float const *__restrict__ s_p, // [Q_ROWS, KV_TILE_SIZE] attention probs
    bf16 const *__restrict__ s_v,  // [KV_TILE_SIZE, HEAD_DIM]
    float *__restrict__ s_o,       // [Q_ROWS, HEAD_DIM] output accumulator
    int q_rows,
    int curr_kv_len) {

  constexpr int MFMA_M = Config::MFMA_M;
  constexpr int MFMA_N = Config::MFMA_N;
  constexpr int MFMA_K = Config::MFMA_K;
  constexpr int HEAD_DIM = Config::HEAD_DIM;
  constexpr int KV_TILE_SIZE = Config::KV_TILE_SIZE;
  constexpr int Q_TILES_M = Config::Q_TILES_M;

  // Number of output tiles in HEAD_DIM
  constexpr int HEAD_TILES_N = HEAD_DIM / MFMA_N;
  // Number of KV tiles for reduction
  constexpr int KV_TILES_K = (KV_TILE_SIZE + MFMA_K - 1) / MFMA_K;

  int lane = threadIdx.x % 64;
  int warp_id = threadIdx.x / 64;

  // Each warp handles one output HEAD_DIM tile
  if (warp_id < HEAD_TILES_N) {
    int tile_n = warp_id;

    for (int tile_m = 0; tile_m < Q_TILES_M; tile_m++) {
      attn_float4 accum = {0.0f, 0.0f, 0.0f, 0.0f};

      // Iterate over KV_TILE in MFMA_K chunks
      for (int tile_k = 0; tile_k < KV_TILES_K; tile_k++) {
        int k_base = tile_k * MFMA_K;
        if (k_base >= curr_kv_len) {
          break;
        }

        // Load P fragment (fp32 -> bf16 conversion needed)
        int p_row = tile_m * MFMA_M + (lane % 16);
        int p_col = k_base + (lane / 16) * 4;

        attn_bf16x4_union reg_p;
        if (p_row < q_rows && p_col + 3 < KV_TILE_SIZE) {
#pragma unroll
          for (int i = 0; i < 4; i++) {
            float val = (p_col + i < curr_kv_len)
                            ? s_p[p_row * KV_TILE_SIZE + p_col + i]
                            : 0.0f;
            reg_p.u16[i] = __bfloat16_as_ushort(__float2bfloat16(val));
          }
        } else {
          reg_p.u64 = 0;
        }

        // Load V fragment
        int v_row = k_base + (lane / 16) * 4;
        int v_col = tile_n * MFMA_N + (lane % 16);

        attn_bf16x4_union reg_v;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          int vr = v_row + i;
          if (vr < curr_kv_len) {
            reg_v.u16[i] = reinterpret_cast<unsigned short const *>(
                &s_v[vr * HEAD_DIM + v_col])[0];
          } else {
            reg_v.u16[i] = 0;
          }
        }

        // MFMA
        accum = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
            reg_p.vec, reg_v.vec, accum, 0, 0, 0);
      }

      // Accumulate to output
      int out_row_base = tile_m * MFMA_M + (lane / 16) * 4;
      int out_col = tile_n * MFMA_N + (lane % 16);

#pragma unroll
      for (int r = 0; r < 4; r++) {
        int out_row = out_row_base + r;
        if (out_row < q_rows && out_col < HEAD_DIM) {
          atomicAdd(&s_o[out_row * HEAD_DIM + out_col], accum[r]);
        }
      }
    }
  }
}

} // namespace mirage_ck
