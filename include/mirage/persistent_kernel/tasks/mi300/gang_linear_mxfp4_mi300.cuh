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

// Gang dense linear kernel using MXFP4 weights with FP4×FP8 MFMA (gfx950).
//
// Reuses all FP4×FP8 helpers from gang_moe_linear_mxfp4_mi300.cuh:
//   _gang_mfma_f4xf8, _gang_load_fp8_mfma_b, _gang_quant_bf16_block_fp8
//
// Weight format: MXFP4 packed per workgroup (same as MoE path):
//   [1, expert_wgs, wg_bytes] where wg_bytes = OPW*(K/2) + OPW*(K/32)
//
// Unlike the MoE kernel, this has no expert routing — just direct 2D tiling.
//
// Optimization: Depth-4 software-pipelined MFMA loop (matching MoE path).
// Pre-loads 4 future K-tiles while computing MFMA on the current tile.
// Each MFMA takes ~32 cycles; with 4 in flight, load-to-use distance is
// ~128 cycles, covering ~32% of ~400-cycle HBM latency.

#pragma once
#include "tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh"

namespace kernel {

// Dense gang linear with MXFP4 weights + HipKittens Algorithm 1 windowed
// traversal.
//
// 256 threads / 4 waves. Each wave handles 16 output rows (N-parallel).
// K=128 per MFMA instruction via
// __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4.
//
// Weight layout (per workgroup, OPW rows):
//   [FP4 data: OPW * K/2 bytes][E8M0 scales: OPW * K/32 bytes]
//
// Input is quantized on-the-fly from bf16 to FP8 E4M3 in shared memory.
template <int BATCH_SIZE,     // = m_per_tile (rows per M-tile)
          int REDUCTION_SIZE> // K dimension
__device__ __noinline__ void gang_linear_mxfp4_kernel(
    void const *input_ptr,  // [batch, REDUCTION_SIZE] bf16
    void const *weight_ptr, // [1, n_wgs, wg_bytes] MXFP4 packed
    void *output_ptr,       // [batch, output_stride] bf16
    int num_active_tokens,
    int tile_n,        // = output_per_wg (64)
    int output_stride, // full output row stride
    int output_size,   // actual output dim (may be < output_stride)
    int m_tiles,       // total M-tiles
    int n_tiles,       // N-tiles per XCD
    int wgm,           // window height W (Algorithm 1)
    int tile_idx,
    void const *bias_ptr) // [1, output_stride] bf16, optional
{
  static_assert(REDUCTION_SIZE % 128 == 0,
                "K must be multiple of 128 for FP4 MFMA");

  assert(tile_idx >= 0);

  // HipKittens Algorithm 1, Step 2: windowed traversal
  int W = (wgm > 0 && wgm < m_tiles) ? wgm : m_tiles;
  int tid_per_group = W * n_tiles;
  int group_id = tile_idx / tid_per_group;
  int first_row = group_id * W;
  int win_h = m_tiles - first_row;
  if (win_h > W) {
    win_h = W;
  }
  int local = tile_idx % tid_per_group;
  if (local >= win_h * n_tiles) {
    return;
  }
  int m_tile = first_row + (local % win_h);
  int n_tile = local / win_h;

  // MXFP4 weight layout constants
  constexpr int OUTPUT_PER_WG = 64; // 4 waves × 16 rows
  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;

  // MFMA constants
  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  constexpr int NUM_WAVES = 4;
  constexpr int N_TILES_PER_WG = OUTPUT_PER_WG / 16;         // = 4
  constexpr int TILES_PER_WAVE = N_TILES_PER_WG / NUM_WAVES; // = 1
  static_assert(MFMA_ITERS >= 4,
                "Depth-4 pipeline requires REDUCTION_SIZE >= 512");

  // FP8 token storage in shared memory
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;
  constexpr int FP8_TOK_SCALES = MFMA_ITERS;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W_data = (uint8_t const *)weight_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;

  extern __shared__ char _gang_lin_mxfp4_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_gang_lin_mxfp4_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15; // output row within 16×16 MFMA tile
  int const g = lane_id >> 4;   // K-group (0..3), each handles 32 FP8 bytes

  // Input pointer for this M-tile
  unsigned short const *input_base =
      A + static_cast<size_t>(m_tile) * BATCH_SIZE * REDUCTION_SIZE;

  // Weight pointer for this N-tile (workgroup)
  uint8_t const *wg_data = W_data + static_cast<size_t>(n_tile) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  // Output pointer
  unsigned short *out_base =
      d_output + static_cast<size_t>(m_tile) * BATCH_SIZE * output_stride +
      static_cast<size_t>(n_tile) * OUTPUT_PER_WG;

  // ── Phase 1: Quantize bf16 input → FP8 E4M3 in shared memory ──────────
  _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
      input_base, s_tok_fp8, s_tok_scales);

  // ── Phase 2: Depth-4 pipelined FP4(weight) × FP8(token) MFMA ─────────
  // TILES_PER_WAVE=1, so only one tile_iter; wave_tile=warp_id.
  {
    int wave_tile = warp_id;
    int w_row = wave_tile * 16 + col;
    int const row_data_base = w_row * (REDUCTION_SIZE / 2);
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // Pre-fill: load k-tiles 0..3 into pipeline slots
    i32x8_t a0 = *(i32x8_t const *)(wg_data + row_data_base + 0 * 64 + g * 16);
    int sa0 = (int)wg_scales[row_scale_base + 0 * 4 + g];
    i32x8_t a1 = *(i32x8_t const *)(wg_data + row_data_base + 1 * 64 + g * 16);
    int sa1 = (int)wg_scales[row_scale_base + 1 * 4 + g];
    i32x8_t a2 = *(i32x8_t const *)(wg_data + row_data_base + 2 * 64 + g * 16);
    int sa2 = (int)wg_scales[row_scale_base + 2 * 4 + g];
    i32x8_t a3 = *(i32x8_t const *)(wg_data + row_data_base + 3 * 64 + g * 16);
    int sa3 = (int)wg_scales[row_scale_base + 3 * 4 + g];

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation.
#pragma unroll 1
    for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
      // Slot 0: compute k-tile ki, prefetch ki+4
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki];
        acc = _gang_mfma_f4xf8(a0, b, acc, sa0, sb);
      }
      if (ki + 4 < MFMA_ITERS) {
        int kt4 = (ki + 4) * K_PER_MFMA;
        a0 = *(i32x8_t const *)(wg_data + row_data_base + kt4 / 2 + g * 16);
        sa0 = (int)wg_scales[row_scale_base + kt4 / 32 + g];
      }

      // Slot 1: compute k-tile ki+1, prefetch ki+5
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 1];
        acc = _gang_mfma_f4xf8(a1, b, acc, sa1, sb);
      }
      if (ki + 5 < MFMA_ITERS) {
        int kt5 = (ki + 5) * K_PER_MFMA;
        a1 = *(i32x8_t const *)(wg_data + row_data_base + kt5 / 2 + g * 16);
        sa1 = (int)wg_scales[row_scale_base + kt5 / 32 + g];
      }

      // Slot 2: compute k-tile ki+2, prefetch ki+6
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 2];
        acc = _gang_mfma_f4xf8(a2, b, acc, sa2, sb);
      }
      if (ki + 6 < MFMA_ITERS) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = *(i32x8_t const *)(wg_data + row_data_base + kt6 / 2 + g * 16);
        sa2 = (int)wg_scales[row_scale_base + kt6 / 32 + g];
      }

      // Slot 3: compute k-tile ki+3, prefetch ki+7
      if (ki + 3 < MFMA_ITERS) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 3];
        acc = _gang_mfma_f4xf8(a3, b, acc, sa3, sb);
      }
      if (ki + 7 < MFMA_ITERS) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = *(i32x8_t const *)(wg_data + row_data_base + kt7 / 2 + g * 16);
        sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
      }
    }

    // ── Epilogue: write result with optional bias ──────────────────────
    if (col == 0) {
      for (int i = 0; i < 4; i++) {
        int out_n = n_tile * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
        if (out_n < output_size) {
          float sum = acc[i];

          // Add bias if provided
          if (d_bias) {
            unsigned bt = (unsigned)d_bias[out_n] << 16;
            float bv;
            __builtin_memcpy(&bv, &bt, 4);
            sum += bv;
          }

          int out_idx = wave_tile * 16 + g * 4 + i;
          out_base[out_idx] = _gang_float_to_bf16(sum);
        }
      }
    }
  }

  __syncthreads();
}

// Dense gang linear with MXFP4 weights + residual add.
// Same as gang_linear_mxfp4_kernel but epilogue adds residual.
// Uses depth-4 pipelined MFMA.
template <int BATCH_SIZE,
          int REDUCTION_SIZE>
__device__ __noinline__ void gang_linear_res_mxfp4_kernel(
    void const *input_ptr,    // [batch, REDUCTION_SIZE] bf16
    void const *weight_ptr,   // [1, n_wgs, wg_bytes] MXFP4 packed
    void const *residual_ptr, // [batch, output_stride] bf16
    void *output_ptr,         // [batch, output_stride] bf16
    int num_active_tokens,
    int tile_n,
    int output_stride,
    int output_size,
    int m_tiles,
    int n_tiles,
    int wgm,
    int tile_idx,
    void const *bias_ptr) {
  static_assert(REDUCTION_SIZE % 128 == 0,
                "K must be multiple of 128 for FP4 MFMA");

  assert(tile_idx >= 0);

  // Windowed traversal
  int W = (wgm > 0 && wgm < m_tiles) ? wgm : m_tiles;
  int tid_per_group = W * n_tiles;
  int group_id = tile_idx / tid_per_group;
  int first_row = group_id * W;
  int win_h = m_tiles - first_row;
  if (win_h > W) {
    win_h = W;
  }
  int local = tile_idx % tid_per_group;
  if (local >= win_h * n_tiles) {
    return;
  }
  int m_tile = first_row + (local % win_h);
  int n_tile = local / win_h;

  constexpr int OUTPUT_PER_WG = 64;
  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;

  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  constexpr int NUM_WAVES = 4;
  constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;
  static_assert(MFMA_ITERS >= 4,
                "Depth-4 pipeline requires REDUCTION_SIZE >= 512");

  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;
  constexpr int FP8_TOK_SCALES = MFMA_ITERS;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W_data = (uint8_t const *)weight_ptr;
  unsigned short const *d_residual = (unsigned short const *)residual_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;

  extern __shared__ char _gang_lin_mxfp4_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_gang_lin_mxfp4_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

  unsigned short const *input_base =
      A + static_cast<size_t>(m_tile) * BATCH_SIZE * REDUCTION_SIZE;

  uint8_t const *wg_data = W_data + static_cast<size_t>(n_tile) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  size_t out_row_off =
      static_cast<size_t>(m_tile) * BATCH_SIZE * output_stride +
      static_cast<size_t>(n_tile) * OUTPUT_PER_WG;
  unsigned short *out_base = d_output + out_row_off;
  unsigned short const *res_base = d_residual + out_row_off;

  // Phase 1: Quantize input to FP8
  _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
      input_base, s_tok_fp8, s_tok_scales);

  // Phase 2: Depth-4 pipelined FP4×FP8 MFMA
  {
    int wave_tile = warp_id;
    int w_row = wave_tile * 16 + col;
    int const row_data_base = w_row * (REDUCTION_SIZE / 2);
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // Pre-fill: load k-tiles 0..3 into pipeline slots
    i32x8_t a0 = *(i32x8_t const *)(wg_data + row_data_base + 0 * 64 + g * 16);
    int sa0 = (int)wg_scales[row_scale_base + 0 * 4 + g];
    i32x8_t a1 = *(i32x8_t const *)(wg_data + row_data_base + 1 * 64 + g * 16);
    int sa1 = (int)wg_scales[row_scale_base + 1 * 4 + g];
    i32x8_t a2 = *(i32x8_t const *)(wg_data + row_data_base + 2 * 64 + g * 16);
    int sa2 = (int)wg_scales[row_scale_base + 2 * 4 + g];
    i32x8_t a3 = *(i32x8_t const *)(wg_data + row_data_base + 3 * 64 + g * 16);
    int sa3 = (int)wg_scales[row_scale_base + 3 * 4 + g];

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation.
#pragma unroll 1
    for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
      // Slot 0: compute k-tile ki, prefetch ki+4
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki];
        acc = _gang_mfma_f4xf8(a0, b, acc, sa0, sb);
      }
      if (ki + 4 < MFMA_ITERS) {
        int kt4 = (ki + 4) * K_PER_MFMA;
        a0 = *(i32x8_t const *)(wg_data + row_data_base + kt4 / 2 + g * 16);
        sa0 = (int)wg_scales[row_scale_base + kt4 / 32 + g];
      }

      // Slot 1: compute k-tile ki+1, prefetch ki+5
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 1];
        acc = _gang_mfma_f4xf8(a1, b, acc, sa1, sb);
      }
      if (ki + 5 < MFMA_ITERS) {
        int kt5 = (ki + 5) * K_PER_MFMA;
        a1 = *(i32x8_t const *)(wg_data + row_data_base + kt5 / 2 + g * 16);
        sa1 = (int)wg_scales[row_scale_base + kt5 / 32 + g];
      }

      // Slot 2: compute k-tile ki+2, prefetch ki+6
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 2];
        acc = _gang_mfma_f4xf8(a2, b, acc, sa2, sb);
      }
      if (ki + 6 < MFMA_ITERS) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = *(i32x8_t const *)(wg_data + row_data_base + kt6 / 2 + g * 16);
        sa2 = (int)wg_scales[row_scale_base + kt6 / 32 + g];
      }

      // Slot 3: compute k-tile ki+3, prefetch ki+7
      if (ki + 3 < MFMA_ITERS) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 3];
        acc = _gang_mfma_f4xf8(a3, b, acc, sa3, sb);
      }
      if (ki + 7 < MFMA_ITERS) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = *(i32x8_t const *)(wg_data + row_data_base + kt7 / 2 + g * 16);
        sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
      }
    }

    // Epilogue: acc + bias + residual → output
    if (col == 0) {
      for (int i = 0; i < 4; i++) {
        int out_n = n_tile * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
        if (out_n < output_size) {
          float sum = acc[i];

          // Add bias
          if (d_bias) {
            unsigned bt = (unsigned)d_bias[out_n] << 16;
            float bv;
            __builtin_memcpy(&bv, &bt, 4);
            sum += bv;
          }

          // Add residual
          int out_idx = wave_tile * 16 + g * 4 + i;
          unsigned rt = (unsigned)res_base[out_idx] << 16;
          float rv;
          __builtin_memcpy(&rv, &rt, 4);
          sum += rv;

          out_base[out_idx] = _gang_float_to_bf16(sum);
        }
      }
    }
  }

  __syncthreads();
}

} // namespace kernel
