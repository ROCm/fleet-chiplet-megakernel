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
// Type definitions for linear
// ============================================================================
using bf16 = ck_tile::bf16_t;
using fp32 = float;

// Warp GEMM types for linear (GEMM) computation
// Uses 16x16x16 MFMA for bf16 inputs with fp32 accumulation
using WarpGemmLinear = ck_tile::WarpGemmMfmaBf16Bf16F32M16N16K16;

// Larger 32x32x8 MFMA for higher throughput
using WarpGemmLinear32 = ck_tile::WarpGemmMfmaBf16Bf16F32M32N32K8;

// ============================================================================
// MFMA vector types
// ============================================================================
typedef short __attribute__((ext_vector_type(4))) linear_bf16x4;
typedef float __attribute__((ext_vector_type(4))) linear_float4;

union linear_bf16x4_union {
  unsigned short u16[4];
  uint64_t u64;
  linear_bf16x4 vec;
};

union linear_bf16x8_union {
  unsigned short u16[8];
  __uint128_t u128;
};

// ============================================================================
// Linear configuration
// ============================================================================
template <int BATCH_SIZE_,
          int OUTPUT_SIZE_,
          int REDUCTION_SIZE_,
          int TILE_K_ = 64,
          int OUTPUT_ATOM_SIZE_ = 64>
struct LinearConfig {
  static constexpr int BATCH_SIZE = BATCH_SIZE_;
  static constexpr int OUTPUT_SIZE = OUTPUT_SIZE_;
  static constexpr int REDUCTION_SIZE = REDUCTION_SIZE_;
  static constexpr int TILE_K = TILE_K_;
  static constexpr int OUTPUT_ATOM_SIZE = OUTPUT_ATOM_SIZE_ <= OUTPUT_SIZE ? OUTPUT_ATOM_SIZE_ : OUTPUT_SIZE;

  // MFMA tile sizes
  static constexpr int MFMA_M = 16;
  static constexpr int MFMA_N = 16;
  static constexpr int MFMA_K = 16;

  // Number of tiles
  static constexpr int TILES_M = (BATCH_SIZE + MFMA_M - 1) / MFMA_M;
  static constexpr int TILES_N = (OUTPUT_ATOM_SIZE + MFMA_N - 1) / MFMA_N;
  static constexpr int NUM_K_TILES = REDUCTION_SIZE / TILE_K;
  static constexpr int MFMA_K_ITERS = TILE_K / MFMA_K;

  // Thread configuration
  static constexpr int NUM_THREADS = 256;
  static constexpr int NUM_WARPS = NUM_THREADS / 64;

  // Vectorized load sizes
  static constexpr int VEC_SIZE = 8;  // 8 bf16 = 128 bits
  static constexpr int INPUT_ELEMS = BATCH_SIZE * TILE_K;
  static constexpr int WEIGHT_ELEMS = TILE_K * OUTPUT_ATOM_SIZE;
  static constexpr int INPUT_VEC_LOADS = INPUT_ELEMS / VEC_SIZE;
  static constexpr int WEIGHT_VEC_LOADS = WEIGHT_ELEMS / VEC_SIZE;
};

// ============================================================================
// Load input tile to shared memory with vectorized loads
// ============================================================================
template <typename T, typename Config>
__device__ __forceinline__ void load_input_tile(
    T* __restrict__ s_input,
    const T* __restrict__ d_input,
    int k_offset,
    int tid) {

  constexpr int TILE_K = Config::TILE_K;
  constexpr int BATCH_SIZE = Config::BATCH_SIZE;
  constexpr int REDUCTION_SIZE = Config::REDUCTION_SIZE;
  constexpr int NUM_THREADS = Config::NUM_THREADS;
  constexpr int VEC_SIZE = Config::VEC_SIZE;
  constexpr int INPUT_VEC_LOADS = Config::INPUT_VEC_LOADS;

  for (int idx = tid; idx < INPUT_VEC_LOADS; idx += NUM_THREADS) {
    int elem_idx = idx * VEC_SIZE;
    int row = elem_idx / TILE_K;
    int col = elem_idx % TILE_K;

    linear_bf16x8_union vec_data;
    if (row < BATCH_SIZE && col + 7 < TILE_K) {
      vec_data.u128 = *reinterpret_cast<const __uint128_t*>(
          &d_input[row * REDUCTION_SIZE + k_offset + col]);
    } else {
      vec_data.u128 = 0;
    }
    *reinterpret_cast<__uint128_t*>(&s_input[row * TILE_K + col]) = vec_data.u128;
  }
}

// ============================================================================
// Load weight tile to shared memory
// Weight is stored as [OUTPUT_SIZE, REDUCTION_SIZE]
// ============================================================================
template <typename T, typename Config>
__device__ __forceinline__ void load_weight_tile(
    T* __restrict__ s_weight,
    const T* __restrict__ d_weight,
    int k_offset,
    int n_offset,
    int tid) {

  constexpr int TILE_K = Config::TILE_K;
  constexpr int OUTPUT_ATOM_SIZE = Config::OUTPUT_ATOM_SIZE;
  constexpr int OUTPUT_SIZE = Config::OUTPUT_SIZE;
  constexpr int REDUCTION_SIZE = Config::REDUCTION_SIZE;
  constexpr int NUM_THREADS = Config::NUM_THREADS;
  constexpr int VEC_SIZE = Config::VEC_SIZE;
  constexpr int WEIGHT_VEC_LOADS = Config::WEIGHT_VEC_LOADS;

  for (int idx = tid; idx < WEIGHT_VEC_LOADS; idx += NUM_THREADS) {
    int elem_idx = idx * VEC_SIZE;
    int k = elem_idx / OUTPUT_ATOM_SIZE;
    int out_col = elem_idx % OUTPUT_ATOM_SIZE;
    int global_col = n_offset + out_col;

    linear_bf16x8_union vec_data;
    // Weight layout: d_weight[out_col][k]
    if (k < TILE_K && global_col + 7 < OUTPUT_SIZE) {
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        vec_data.u16[i] = reinterpret_cast<const unsigned short*>(
            &d_weight[(global_col + i) * REDUCTION_SIZE + k_offset + k])[0];
      }
    } else {
      vec_data.u128 = 0;
    }
    *reinterpret_cast<__uint128_t*>(&s_weight[k * OUTPUT_ATOM_SIZE + out_col]) = vec_data.u128;
  }
}

// ============================================================================
// MFMA GEMM computation for one K-tile
// ============================================================================
template <typename Config>
__device__ __forceinline__ void compute_gemm_mfma(
    const bf16* __restrict__ s_input,   // [BATCH_SIZE, TILE_K]
    const bf16* __restrict__ s_weight,  // [TILE_K, OUTPUT_ATOM_SIZE]
    float* __restrict__ s_output,       // [BATCH_SIZE, OUTPUT_ATOM_SIZE]
    int num_active_tokens,
    int lane,
    int warp_id) {

  constexpr int MFMA_M = Config::MFMA_M;
  constexpr int MFMA_N = Config::MFMA_N;
  constexpr int MFMA_K = Config::MFMA_K;
  constexpr int TILE_K = Config::TILE_K;
  constexpr int BATCH_SIZE = Config::BATCH_SIZE;
  constexpr int OUTPUT_ATOM_SIZE = Config::OUTPUT_ATOM_SIZE;
  constexpr int TILES_M = Config::TILES_M;
  constexpr int TILES_N = Config::TILES_N;
  constexpr int MFMA_K_ITERS = Config::MFMA_K_ITERS;

  // Each warp handles one output column tile
  if (warp_id < TILES_N) {
    int tile_n = warp_id;

    for (int tile_m = 0; tile_m < TILES_M; tile_m++) {
      linear_float4 accum = {0.0f, 0.0f, 0.0f, 0.0f};

      // K-dimension loop within tile
      #pragma unroll
      for (int k_iter = 0; k_iter < MFMA_K_ITERS; k_iter++) {
        int k_base = k_iter * MFMA_K;

        // Load A fragment (input)
        int a_row = tile_m * MFMA_M + (lane % 16);
        int a_col = k_base + (lane / 16) * 4;

        linear_bf16x4_union reg_a;
        if (a_row < BATCH_SIZE) {
          reg_a.u64 = *reinterpret_cast<const uint64_t*>(
              &s_input[a_row * TILE_K + a_col]);
        } else {
          reg_a.u64 = 0;
        }

        // Load B fragment (weight)
        int b_row = k_base + (lane / 16) * 4;
        int b_col = tile_n * MFMA_N + (lane % 16);

        linear_bf16x4_union reg_b;
        if (b_col < OUTPUT_ATOM_SIZE) {
          #pragma unroll
          for (int i = 0; i < 4; i++) {
            reg_b.u16[i] = reinterpret_cast<const unsigned short*>(
                &s_weight[(b_row + i) * OUTPUT_ATOM_SIZE + b_col])[0];
          }
        } else {
          reg_b.u64 = 0;
        }

        // MFMA
        accum = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
            reg_a.vec, reg_b.vec, accum, 0, 0, 0);
      }

      // Accumulate results to shared memory
      int out_row_base = tile_m * MFMA_M + (lane / 16) * 4;
      int out_col = tile_n * MFMA_N + (lane % 16);

      if (out_col < OUTPUT_ATOM_SIZE) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
          int out_row = out_row_base + r;
          if (out_row < BATCH_SIZE && out_row < num_active_tokens) {
            s_output[out_row * OUTPUT_ATOM_SIZE + out_col] += accum[r];
          }
        }
      }
    }
  }
}

} // namespace mirage_ck
