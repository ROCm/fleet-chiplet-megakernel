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

// Fused O-PROJ + RMSNorm + Router Linear + TopK Softmax for MI300/MI350.
//
// Combines two gang tasks into one:
//   1. gang_linear_mxfp4_res_bias_kernel (O-projection with residual)
//   2. gang_rmsnorm_linear_bias_topk_kernel (RMSNorm + router + TopK)
//
// Eliminates one inter-task event barrier per layer. Over 36 layers
// of transitions at ~14.5us each: ~522us saved.
//
// Pipeline:
//   Phase 1: O-PROJ MXFP4 GEMM + bias + residual -> write-through to HBM
//   Phase 2: Atomic barrier (oproj_counter), buffer_inv
//   Phase 3: RMSNorm (redundant across all workers) + Router GEMV ->
//   write-through logit Phase 4: Atomic barrier (topk_counter), last worker
//   runs TopK softmax
//
// Pointer layout (10 inputs, 4 outputs):
//   input_ptrs[0]: attn_out          [batch, REDUCTION_SIZE] bf16
//   input_ptrs[1]: mxfp4_weight      [n_wgs_per_xcd, wg_bytes] packed MXFP4
//   input_ptrs[2]: residual          [batch, output_stride] bf16
//   input_ptrs[3]: oproj_bias        [1, output_size_per_xcd] bf16
//   input_ptrs[4]: norm_weight       [ACTUAL_HIDDEN_DIM] bf16
//   input_ptrs[5]: norm_output       [batch, ACTUAL_HIDDEN_DIM] bf16 scratch
//   input_ptrs[6]: router_weight     [chunk_N, ACTUAL_HIDDEN_DIM] bf16
//   input_ptrs[7]: router_bias       [1, NUM_EXPERTS] bf16
//   input_ptrs[8]: logits_scratch    XCD-partitioned [batch, chunk_N] bf16
//   input_ptrs[9]: counters          int32[2]: [0]=oproj_counter
//   [1]=topk_counter output_ptrs[0]: attn_proj_out    [batch, output_stride]
//   bf16 output_ptrs[1]: topk_weight      [batch, K] float output_ptrs[2]:
//   routing_indices  [NUM_EXPERTS, batch] int32 output_ptrs[3]:
//   active_expert_ids [NUM_EXPERTS+1] int32

#pragma once
#include "tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh"    // FP4xFP8 helpers
#include "tasks/mi300/gang_rmsnorm_linear_bias_mi300.cuh" // topk_noinline

namespace kernel {

template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM,
          int NUM_EXPERTS,
          int K>
__device__ __attribute__((noinline)) void
    gang_linear_mxfp4_res_bias_rmsnorm_topk_kernel(
        // O-PROJ inputs
        void const *input_ptr,    // input_ptrs[0]
        void const *weight_ptr,   // input_ptrs[1]
        void const *residual_ptr, // input_ptrs[2]
        void const *bias_ptr,     // input_ptrs[3]
        // TopK inputs
        void const *norm_weight_ptr,   // input_ptrs[4]
        void *norm_output_ptr,         // input_ptrs[5]
        void const *router_weight_ptr, // input_ptrs[6]
        void const *router_bias_ptr,   // input_ptrs[7]
        void *logits_scratch_ptr,      // input_ptrs[8]
        void *counters_ptr,            // input_ptrs[9]
        // Outputs
        void *output_ptr,            // output_ptrs[0]
        void *topk_weight_ptr,       // output_ptrs[1]
        void *routing_indices_ptr,   // output_ptrs[2]
        void *active_expert_ids_ptr, // output_ptrs[3]
        // Parameters
        int num_active_tokens,
        int n_wgs_per_xcd,
        int output_stride,
        int router_tile_n,
        int total_oproj_tiles,
        int total_topk_tiles,
        int tiles_per_xcd,
        int tile_idx,
        // Optional: when non-null, the TopK-completing worker writes 1 here
        // via st_wt_u32 so the fused wrapper can poll before MoE.
        int *routing_ready_ptr = nullptr,
        // Optional: per-worker timestamp ring buffer pointer (g_fused_ts).
        // When non-null, writes slots 9 (oproj_done), 10 (barrier_done),
        // 11 (rmsnorm_router_done) for sub-phase breakdown.
        unsigned long long *ts_base = nullptr) {

  static_assert(OUTPUT_PER_WG % 16 == 0,
                "OUTPUT_PER_WG must be multiple of 16");
  static_assert(REDUCTION_SIZE % 128 == 0,
                "REDUCTION_SIZE must be multiple of 128 for FP4 MFMA");

  // ── Weight layout constants ─────────────────────────────────────────────
  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;

  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;

  // LDS weight layout (populated by fused kernel Phase 6 DMA)
  constexpr int OPROJ_LDS_N16_DATA = WG_DATA_BYTES / 16;
  constexpr int OPROJ_LDS_LPT = (OPROJ_LDS_N16_DATA + 255) / 256;
  constexpr int OPROJ_LDS_DATA_PAD = OPROJ_LDS_LPT * 256 * 16;
  constexpr int OPROJ_LDS_N16_SCALE = (WG_SCALE_BYTES + 15) / 16;
  constexpr int OPROJ_LDS_SLPT = (OPROJ_LDS_N16_SCALE + 255) / 256;
  constexpr int OPROJ_LDS_SCALE_PAD = OPROJ_LDS_SLPT * 256 * 16;
  static_assert(MFMA_ITERS >= 4,
                "Depth-4 pipeline requires REDUCTION_SIZE >= 512");
  constexpr int BF16_MFMA_ITERS = REDUCTION_SIZE / 32;

  constexpr int NUM_WAVES = 4;

  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_residual = (unsigned short const *)residual_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;
  // Mechanism C barrier layout (HIER_STRIDE=16 int32 = 64 bytes per slot):
  //   [xcd*16]:  per-XCD release flag (written by last global arrival)
  //   [8*16]:    global_arrive — all-worker arrival count (polls >=
  //   total_oproj_tiles) [9*16]:    topk_counter
  constexpr int HIER_STRIDE = 16;
  int *hier_barrier = (int *)counters_ptr;
  int *topk_counter = hier_barrier + 9 * HIER_STRIDE;

  extern __shared__ char _lm_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_lm_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _sp_t0 = __builtin_amdgcn_s_memrealtime();
#endif

  int batch_count =
      (num_active_tokens < BATCH_SIZE) ? num_active_tokens : BATCH_SIZE;

  // Decode XCD index from globally-unique tile_idx
  // tile_idx = xcd_id * tiles_per_xcd + local_tile
  int xcd_id = tile_idx / tiles_per_xcd;
  int local_tile = tile_idx % tiles_per_xcd;
  int xcd_output_col_offset = xcd_id * n_wgs_per_xcd * OUTPUT_PER_WG;

  // ════════════════════════════════════════════════════════════════════════
  // PHASE 1: O-PROJ (MXFP4 linear + bias + residual)
  // ════════════════════════════════════════════════════════════════════════

  int tok_idx = local_tile / n_wgs_per_xcd;
  int wg_idx = local_tile % n_wgs_per_xcd;

  if (tok_idx >= batch_count) {
    goto oproj_barrier;
  }

  {
    uint8_t const *wg_data = W + static_cast<int64_t>(wg_idx) * WG_BYTES;
    uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

    unsigned short const *input_row = A + tok_idx * REDUCTION_SIZE;

    // FP8 quant only — weights already in LDS via Phase 6 buffer_load_lds
    if constexpr (OUTPUT_PER_WG < 64) {
      constexpr int SUB_BLOCK = 32;
      constexpr int NSUBBLOCKS = REDUCTION_SIZE / SUB_BLOCK;
      int const lane_id_q = tid & 63;

      if (tid < NSUBBLOCKS) {
        int const sb = tid;
        int const base = sb * SUB_BLOCK;
        int const super_blk = sb / 4;
        int const sub_idx = sb & 3;
        uint32_t const *base_ptr = (uint32_t const *)(input_row + base);

        uint32_t dw[16];
        asm volatile("global_load_dwordx4 %0, %4, off\n"
                     "global_load_dwordx4 %1, %5, off\n"
                     "global_load_dwordx4 %2, %6, off\n"
                     "global_load_dwordx4 %3, %7, off"
                     : "=&v"(*(i32x4_t *)&dw[0]),
                       "=&v"(*(i32x4_t *)&dw[4]),
                       "=&v"(*(i32x4_t *)&dw[8]),
                       "=&v"(*(i32x4_t *)&dw[12])
                     : "v"(base_ptr),
                       "v"(base_ptr + 4),
                       "v"(base_ptr + 8),
                       "v"(base_ptr + 12)
                     : "memory");
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

        float vals[32];
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; j++) {
          float lo = _gang_bf16_to_float((unsigned short)(dw[j] & 0xFFFF));
          float hi = _gang_bf16_to_float((unsigned short)(dw[j] >> 16));
          vals[j * 2] = lo;
          vals[j * 2 + 1] = hi;
          amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
        }

        int base_lane = lane_id_q & ~3;
        float a0 = __shfl(amax, base_lane);
        float a1 = __shfl(amax, base_lane + 1);
        float a2 = __shfl(amax, base_lane + 2);
        float a3 = __shfl(amax, base_lane + 3);
        float block_amax = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));

        uint8_t se = _gang_compute_e8m0_fp8(block_amax);
        float scale_f;
        if (se == 0) {
          scale_f = 1.0f;
        } else {
          union {
            float f;
            uint32_t u;
          } sv;
          sv.u = (uint32_t)se << 23;
          scale_f = sv.f;
        }

#pragma unroll
        for (int j = 0; j < 32; j += 4) {
          fp8x4_t pk = {};
          pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
              pk, vals[j], vals[j + 1], scale_f, false);
          pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
              pk, vals[j + 2], vals[j + 3], scale_f, true);
          *(int *)(s_tok_fp8 + base + j) = *(int const *)&pk;
        }

        if (sub_idx == 0) {
          s_tok_scales[super_blk] = se;
        }
      }
      __syncthreads();
    } else {
      _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
          input_row, s_tok_fp8, s_tok_scales);
    }

    if constexpr (OUTPUT_PER_WG >= 64) {
      // N-parallel: 4 waves handle different output rows
      constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;

      for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
        int wave_tile = warp_id + tile_iter * NUM_WAVES;
        int w_row = wave_tile * 16 + col;

        int const row_data_base = w_row * (REDUCTION_SIZE / 2);
        int const row_scale_base = w_row * NUM_BLOCKS_32;

        f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

        int sa0, sa1, sa2, sa3;
        i32x8_t a0, a1, a2, a3;
        {
          i32x4_t _w0t, _w1t, _w2t, _w3t;
          asm volatile("global_load_dwordx4 %0, %1, off"
                       : "=v"(_w0t)
                       : "v"(wg_data + row_data_base + 0 * 64 + g * 16)
                       : "memory");
          asm volatile("global_load_dwordx4 %0, %1, off"
                       : "=v"(_w1t)
                       : "v"(wg_data + row_data_base + 1 * 64 + g * 16)
                       : "memory");
          asm volatile("global_load_dwordx4 %0, %1, off"
                       : "=v"(_w2t)
                       : "v"(wg_data + row_data_base + 2 * 64 + g * 16)
                       : "memory");
          asm volatile("global_load_dwordx4 %0, %1, off"
                       : "=v"(_w3t)
                       : "v"(wg_data + row_data_base + 3 * 64 + g * 16)
                       : "memory");
          asm volatile("global_load_ubyte %0, %1, off"
                       : "=v"(sa0)
                       : "v"(wg_scales + row_scale_base + 0 * 4 + g)
                       : "memory");
          asm volatile("global_load_ubyte %0, %1, off"
                       : "=v"(sa1)
                       : "v"(wg_scales + row_scale_base + 1 * 4 + g)
                       : "memory");
          asm volatile("global_load_ubyte %0, %1, off"
                       : "=v"(sa2)
                       : "v"(wg_scales + row_scale_base + 2 * 4 + g)
                       : "memory");
          asm volatile("global_load_ubyte %0, %1, off"
                       : "=v"(sa3)
                       : "v"(wg_scales + row_scale_base + 3 * 4 + g)
                       : "memory");
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          a0[0] = _w0t[0];
          a0[1] = _w0t[1];
          a0[2] = _w0t[2];
          a0[3] = _w0t[3];
          a0[4] = 0;
          a0[5] = 0;
          a0[6] = 0;
          a0[7] = 0;
          a1[0] = _w1t[0];
          a1[1] = _w1t[1];
          a1[2] = _w1t[2];
          a1[3] = _w1t[3];
          a1[4] = 0;
          a1[5] = 0;
          a1[6] = 0;
          a1[7] = 0;
          a2[0] = _w2t[0];
          a2[1] = _w2t[1];
          a2[2] = _w2t[2];
          a2[3] = _w2t[3];
          a2[4] = 0;
          a2[5] = 0;
          a2[6] = 0;
          a2[7] = 0;
          a3[0] = _w3t[0];
          a3[1] = _w3t[1];
          a3[2] = _w3t[2];
          a3[3] = _w3t[3];
          a3[4] = 0;
          a3[5] = 0;
          a3[6] = 0;
          a3[7] = 0;
        }

#pragma unroll 1
        for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
          {
            i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
            int sb = (int)s_tok_scales[ki];
            acc = _gang_mfma_f4xf8(a0, b, acc, sa0, sb);
          }
          if (ki + 4 < MFMA_ITERS) {
            int kt4 = (ki + 4) * K_PER_MFMA;
            i32x4_t _wt;
            asm volatile("global_load_dwordx4 %0, %1, off"
                         : "=v"(_wt)
                         : "v"(wg_data + row_data_base + kt4 / 2 + g * 16)
                         : "memory");
            asm volatile("global_load_ubyte %0, %1, off"
                         : "=v"(sa0)
                         : "v"(wg_scales + row_scale_base + kt4 / 32 + g)
                         : "memory");
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            a0[0] = _wt[0];
            a0[1] = _wt[1];
            a0[2] = _wt[2];
            a0[3] = _wt[3];
            a0[4] = 0;
            a0[5] = 0;
            a0[6] = 0;
            a0[7] = 0;
          }
          {
            i32x8_t b =
                _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
            int sb = (int)s_tok_scales[ki + 1];
            acc = _gang_mfma_f4xf8(a1, b, acc, sa1, sb);
          }
          if (ki + 5 < MFMA_ITERS) {
            int kt5 = (ki + 5) * K_PER_MFMA;
            i32x4_t _wt;
            asm volatile("global_load_dwordx4 %0, %1, off"
                         : "=v"(_wt)
                         : "v"(wg_data + row_data_base + kt5 / 2 + g * 16)
                         : "memory");
            asm volatile("global_load_ubyte %0, %1, off"
                         : "=v"(sa1)
                         : "v"(wg_scales + row_scale_base + kt5 / 32 + g)
                         : "memory");
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            a1[0] = _wt[0];
            a1[1] = _wt[1];
            a1[2] = _wt[2];
            a1[3] = _wt[3];
            a1[4] = 0;
            a1[5] = 0;
            a1[6] = 0;
            a1[7] = 0;
          }
          {
            i32x8_t b =
                _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
            int sb = (int)s_tok_scales[ki + 2];
            acc = _gang_mfma_f4xf8(a2, b, acc, sa2, sb);
          }
          if (ki + 6 < MFMA_ITERS) {
            int kt6 = (ki + 6) * K_PER_MFMA;
            i32x4_t _wt;
            asm volatile("global_load_dwordx4 %0, %1, off"
                         : "=v"(_wt)
                         : "v"(wg_data + row_data_base + kt6 / 2 + g * 16)
                         : "memory");
            asm volatile("global_load_ubyte %0, %1, off"
                         : "=v"(sa2)
                         : "v"(wg_scales + row_scale_base + kt6 / 32 + g)
                         : "memory");
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            a2[0] = _wt[0];
            a2[1] = _wt[1];
            a2[2] = _wt[2];
            a2[3] = _wt[3];
            a2[4] = 0;
            a2[5] = 0;
            a2[6] = 0;
            a2[7] = 0;
          }
          if (ki + 3 < MFMA_ITERS) {
            i32x8_t b =
                _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
            int sb = (int)s_tok_scales[ki + 3];
            acc = _gang_mfma_f4xf8(a3, b, acc, sa3, sb);
          }
          if (ki + 7 < MFMA_ITERS) {
            int kt7 = (ki + 7) * K_PER_MFMA;
            i32x4_t _wt;
            asm volatile("global_load_dwordx4 %0, %1, off"
                         : "=v"(_wt)
                         : "v"(wg_data + row_data_base + kt7 / 2 + g * 16)
                         : "memory");
            asm volatile("global_load_ubyte %0, %1, off"
                         : "=v"(sa3)
                         : "v"(wg_scales + row_scale_base + kt7 / 32 + g)
                         : "memory");
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            a3[0] = _wt[0];
            a3[1] = _wt[1];
            a3[2] = _wt[2];
            a3[3] = _wt[3];
            a3[4] = 0;
            a3[5] = 0;
            a3[6] = 0;
            a3[7] = 0;
          }
        }

        // Epilogue: acc + bias + residual -> bf16, write-through store
        // bias/residual are XCD-partitioned -> use local out_n_base
        // output is replicated -> add xcd_output_col_offset for writes
        if (col == 0) {
          int out_n_local = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4;
          int out_n_global = xcd_output_col_offset + out_n_local;
          int res_idx_base = tok_idx * output_stride + out_n_local;
          int out_idx_base = tok_idx * output_stride + out_n_global;

          uint2 bias_packed;
          __builtin_memcpy(&bias_packed, &d_bias[out_n_local], 8);
          uint2 res_packed;
          __builtin_memcpy(&res_packed, &d_residual[res_idx_base], 8);

          unsigned bt0 = (bias_packed.x & 0xFFFFu) << 16;
          unsigned bt1 = bias_packed.x & 0xFFFF0000u;
          unsigned bt2 = (bias_packed.y & 0xFFFFu) << 16;
          unsigned bt3 = bias_packed.y & 0xFFFF0000u;
          float bv0, bv1, bv2, bv3;
          __builtin_memcpy(&bv0, &bt0, 4);
          __builtin_memcpy(&bv1, &bt1, 4);
          __builtin_memcpy(&bv2, &bt2, 4);
          __builtin_memcpy(&bv3, &bt3, 4);

          unsigned rt0 = (res_packed.x & 0xFFFFu) << 16;
          unsigned rt1 = res_packed.x & 0xFFFF0000u;
          unsigned rt2 = (res_packed.y & 0xFFFFu) << 16;
          unsigned rt3 = res_packed.y & 0xFFFF0000u;
          float rv0, rv1, rv2, rv3;
          __builtin_memcpy(&rv0, &rt0, 4);
          __builtin_memcpy(&rv1, &rt1, 4);
          __builtin_memcpy(&rv2, &rt2, 4);
          __builtin_memcpy(&rv3, &rt3, 4);

          unsigned short o0 = _gang_float_to_bf16(acc[0] + bv0 + rv0);
          unsigned short o1 = _gang_float_to_bf16(acc[1] + bv1 + rv1);
          unsigned short o2 = _gang_float_to_bf16(acc[2] + bv2 + rv2);
          unsigned short o3 = _gang_float_to_bf16(acc[3] + bv3 + rv3);
          unsigned long long out64 =
              (unsigned long long)o0 | ((unsigned long long)o1 << 16) |
              ((unsigned long long)o2 << 32) | ((unsigned long long)o3 << 48);
          st_wt_u64(&d_output[out_idx_base], out64);
        }
      }
    } else {
      // K-parallel: 4 waves split K, reduce via LDS
      // Load-balanced: waves 0..extra-1 get base+1 iters, rest get base.
      // Weight loads were prefetched BEFORE FP8 quant (see above).
      constexpr int KP_TOTAL = MFMA_ITERS;
      constexpr int KP_BASE = KP_TOTAL / NUM_WAVES;
      constexpr int KP_EXTRA = KP_TOTAL % NUM_WAVES;
      static_assert(KP_BASE >= 4,
                    "K-parallel depth-4 requires >= 4 iters per wave");

      int const kp_my_iters = KP_BASE + (warp_id < KP_EXTRA ? 1 : 0);
      int const kp_ki_start =
          (warp_id < KP_EXTRA)
              ? warp_id * (KP_BASE + 1)
              : KP_EXTRA * (KP_BASE + 1) + (warp_id - KP_EXTRA) * KP_BASE;
      int const kp_ki_end = kp_ki_start + kp_my_iters;

      f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

      // Prefetch bias + residual from HBM now — they'll be needed after
      // the MFMA loop + K-parallel reduction (~800 ns away).  Addresses
      // depend only on tile indices, not on MFMA results.
      uint2 pf_bias = {0, 0}, pf_res = {0, 0};
      if (warp_id == 0 && col == 0) {
        int out_n_local_pf = wg_idx * OUTPUT_PER_WG + g * 4;
        int res_idx_pf = tok_idx * output_stride + out_n_local_pf;
        unsigned short const *bias_addr = &d_bias[out_n_local_pf];
        unsigned short const *res_addr = &d_residual[res_idx_pf];
        asm volatile("global_load_dwordx2 %0, %2, off\n"
                     "global_load_dwordx2 %1, %3, off"
                     : "=v"(pf_bias), "=v"(pf_res)
                     : "v"(bias_addr), "v"(res_addr)
                     : "memory");
      }

      // Read weights from LDS (populated by Phase 6 buffer_load_lds DMA).
      // LDS layout mirrors HBM tile layout: data at [0..WG_DATA_BYTES),
      // scales at [OPROJ_LDS_DATA_PAD..OPROJ_LDS_DATA_PAD + WG_SCALE_BYTES).
      // Must match Phase 6 OPROJ_LDS_W_OFF (uses NUM_BLOCKS_32, not MFMA_ITERS)
      constexpr int OPROJ_LDS_OFF =
          ((FP8_TOK_DATA + NUM_BLOCKS_32 + 15) / 16) * 16;
      uint8_t const *lds_w_data = (uint8_t const *)_lm_smem + OPROJ_LDS_OFF;
      uint8_t const *lds_w_scales = lds_w_data + OPROJ_LDS_DATA_PAD;

      int w_row_lds = col;
      int const lds_row_data_base = w_row_lds * (REDUCTION_SIZE / 2);
      int const lds_row_scale_base = w_row_lds * NUM_BLOCKS_32;

#define DO_MFMA_LDS_FP8(KI)                                                    \
  do {                                                                         \
    i32x4_t _wt;                                                               \
    __builtin_memcpy(&_wt,                                                     \
                     lds_w_data + lds_row_data_base +                          \
                         (KI) * (K_PER_MFMA / 2) + g * 16,                     \
                     16);                                                      \
    int sa = (int)lds_w_scales[lds_row_scale_base + (KI)*4 + g];               \
    i32x8_t a;                                                                 \
    a[0] = _wt[0];                                                             \
    a[1] = _wt[1];                                                             \
    a[2] = _wt[2];                                                             \
    a[3] = _wt[3];                                                             \
    a[4] = 0;                                                                  \
    a[5] = 0;                                                                  \
    a[6] = 0;                                                                  \
    a[7] = 0;                                                                  \
    i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (KI)*K_PER_MFMA, g);          \
    int sb = (int)s_tok_scales[KI];                                            \
    acc = _gang_mfma_f4xf8(a, b, acc, sa, sb);                                 \
  } while (0)
#define DO_MFMA_LDS_FP4(KI)                                                    \
  do {                                                                         \
    i32x4_t _wt;                                                               \
    __builtin_memcpy(&_wt,                                                     \
                     lds_w_data + lds_row_data_base +                          \
                         (KI) * (K_PER_MFMA / 2) + g * 16,                     \
                     16);                                                      \
    int sa = (int)lds_w_scales[lds_row_scale_base + (KI)*4 + g];               \
    i32x8_t a;                                                                 \
    a[0] = _wt[0];                                                             \
    a[1] = _wt[1];                                                             \
    a[2] = _wt[2];                                                             \
    a[3] = _wt[3];                                                             \
    a[4] = 0;                                                                  \
    a[5] = 0;                                                                  \
    a[6] = 0;                                                                  \
    a[7] = 0;                                                                  \
    i32x8_t b = _gang_load_fp4_mfma_b(s_tok_fp4, (KI)*K_PER_MFMA, g);          \
    int sb = (int)s_tok_scales[(KI)*4 + g];                                    \
    acc = _gang_mfma_f4xf4(a, b, acc, sa, sb);                                 \
  } while (0)

      DO_MFMA_LDS_FP8(kp_ki_start + 0);
      DO_MFMA_LDS_FP8(kp_ki_start + 1);
      DO_MFMA_LDS_FP8(kp_ki_start + 2);
      DO_MFMA_LDS_FP8(kp_ki_start + 3);
      DO_MFMA_LDS_FP8(kp_ki_start + 4);
      if (kp_ki_start + 5 < kp_ki_end) {
        DO_MFMA_LDS_FP8(kp_ki_start + 5);
      }
      if (kp_ki_start + 6 < kp_ki_end) {
        DO_MFMA_LDS_FP8(kp_ki_start + 6);
      }
      if (kp_ki_start + 7 < kp_ki_end) {
        DO_MFMA_LDS_FP8(kp_ki_start + 7);
      }
#undef DO_MFMA_LDS_FP8
#undef DO_MFMA_LDS_FP4

      // K-parallel reduce via LDS
      // CRITICAL: All lanes must write acc to LDS unconditionally.
      // MFMA is a wave-level op that reads B operands from ALL 64 lanes.
      // If the compiler can skip MFMAs for col!=0 lanes (because only
      // col==0 uses acc), it hoists the exec mask before ds_read_b128
      // for token B, causing 60/64 lanes to have stale B data and
      // producing wrong MFMA results for ALL lanes including col==0.
      //
      // Fix: every lane writes to a unique LDS slot.
      // Layout: [warp_id][lane_id][4_accum_values]
      // Total: NUM_WAVES * 64 * 4 = 1024 floats = 4096 bytes
      // The reduction reads only col==0 lanes' data (lane_id = g*16+0).
      float *lds_reduce = (float *)_lm_smem;
      for (int i = 0; i < 4; i++) {
        lds_reduce[(warp_id * 64 + lane_id) * 4 + i] = acc[i];
      }
      __syncthreads();

      if (warp_id == 0 && col == 0) {
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        for (int w = 0; w < NUM_WAVES; w++) {
          // Read from col==0 lanes: lane_id = g*16 + 0 = g*16
          int src_lane = g * 16; // col==0 lane for this g
          v0 += lds_reduce[(w * 64 + src_lane) * 4 + 0];
          v1 += lds_reduce[(w * 64 + src_lane) * 4 + 1];
          v2 += lds_reduce[(w * 64 + src_lane) * 4 + 2];
          v3 += lds_reduce[(w * 64 + src_lane) * 4 + 3];
        }

        // bias/residual are XCD-partitioned -> use local offset
        // output is replicated -> add xcd_output_col_offset
        int out_n_local = wg_idx * OUTPUT_PER_WG + g * 4;
        int out_n_global = xcd_output_col_offset + out_n_local;
        int out_idx_base = tok_idx * output_stride + out_n_global;

        // Wait for bias+residual prefetched before MFMA loop.
        // Early-clobber outputs prevent the compiler from aliasing
        // an output register with a not-yet-read input register.
        uint2 bias_packed, res_packed;
        asm volatile(
            "s_waitcnt vmcnt(0)\n"
            "v_mov_b32_e32 %0, %4\n"
            "v_mov_b32_e32 %1, %5\n"
            "v_mov_b32_e32 %2, %6\n"
            "v_mov_b32_e32 %3, %7"
            : "=&v"(bias_packed.x),
              "=&v"(bias_packed.y),
              "=&v"(res_packed.x),
              "=&v"(res_packed.y)
            : "v"(pf_bias.x), "v"(pf_bias.y), "v"(pf_res.x), "v"(pf_res.y)
            : "memory");

        unsigned bt0 = (bias_packed.x & 0xFFFFu) << 16;
        unsigned bt1 = bias_packed.x & 0xFFFF0000u;
        unsigned bt2 = (bias_packed.y & 0xFFFFu) << 16;
        unsigned bt3 = bias_packed.y & 0xFFFF0000u;
        float bv0, bv1, bv2, bv3;
        __builtin_memcpy(&bv0, &bt0, 4);
        __builtin_memcpy(&bv1, &bt1, 4);
        __builtin_memcpy(&bv2, &bt2, 4);
        __builtin_memcpy(&bv3, &bt3, 4);

        unsigned rt0 = (res_packed.x & 0xFFFFu) << 16;
        unsigned rt1 = res_packed.x & 0xFFFF0000u;
        unsigned rt2 = (res_packed.y & 0xFFFFu) << 16;
        unsigned rt3 = res_packed.y & 0xFFFF0000u;
        float rv0, rv1, rv2, rv3;
        __builtin_memcpy(&rv0, &rt0, 4);
        __builtin_memcpy(&rv1, &rt1, 4);
        __builtin_memcpy(&rv2, &rt2, 4);
        __builtin_memcpy(&rv3, &rt3, 4);

        unsigned short o0 = _gang_float_to_bf16(v0 + bv0 + rv0);
        unsigned short o1 = _gang_float_to_bf16(v1 + bv1 + rv1);
        unsigned short o2 = _gang_float_to_bf16(v2 + bv2 + rv2);
        unsigned short o3 = _gang_float_to_bf16(v3 + bv3 + rv3);
        unsigned long long out64 =
            (unsigned long long)o0 | ((unsigned long long)o1 << 16) |
            ((unsigned long long)o2 << 32) | ((unsigned long long)o3 << 48);
        st_wt_u64(&d_output[out_idx_base], out64);
      }
    }
  }

oproj_barrier :
  // ════════════════════════════════════════════════════════════════════════
  // PHASE 2: O-PROJ hierarchical barrier
  // ════════════════════════════════════════════════════════════════════════
  // Level 1: per-XCD arrival (24 intra-XCD atomics, each on own cache line)
  // Level 2: last tile per XCD → leader increments global (8 cross-XCD atomics)
  // All workers poll global_arrive >= 8 via ld_nt
#ifdef MPK_ENABLE_SUBPHASE_TIMING
{
  unsigned long long _sp_t1 = __builtin_amdgcn_s_memrealtime();
  if (tid == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[3][0], (_sp_t1 - _sp_t0) * 10); // OProjCompute
  }
  _sp_t0 = _sp_t1;
}
#endif
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  if (tid == 0 && ts_base) {
    ts_base[9] = __builtin_amdgcn_s_memrealtime(); // slot 9: oproj_mfma_done
  }
#endif
  __syncthreads();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

  // ── Prefetch gamma + router weights during barrier wait ──────────
  // These are data-independent of O-proj output.
  // Same pattern as W2 LDS prefetch: issue loads before barrier,
  // drain after barrier + buffer_inv.
  typedef int __attribute__((ext_vector_type(2))) i32x2_pf_t;
  constexpr int H4_PF = ACTUAL_HIDDEN_DIM >> 2;
  constexpr int MAX_ITERS_PF = (H4_PF + 255) / 256;

  i32x2_pf_t g_pf_buf[MAX_ITERS_PF];
  i32x2_pf_t w_pf_buf[MAX_ITERS_PF];

  {
    // Mechanism C: single global arrive + per-XCD release flags
    // Uses monotonically increasing expected values (no reset needed)
    // All threads read release_expected independently (ld_nt is uniform),
    // eliminating shared variables and both __syncthreads.
    int oproj_release_expected =
        ld_nt_s32(&hier_barrier[xcd_id * HIER_STRIDE]) + 1;

    if (tid == 0) {
      // Single global arrival (all workers increment one counter)
      int prev_global =
          atom_add_release_gpu_s32(&hier_barrier[8 * HIER_STRIDE], 1);
      if ((prev_global % total_oproj_tiles) == total_oproj_tiles - 1) {
        // Last arrival: write 8 per-XCD release flags (write-through)
        for (int x = 0; x < 8; x++) {
          st_wt_u32((void *)&hier_barrier[x * HIER_STRIDE],
                    (unsigned)oproj_release_expected);
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
    }

    // Issue prefetch loads AFTER barrier atomics but BEFORE poll loop.
    if (local_tile < router_tile_n) {
      char const *g_base_pf = (char const *)norm_weight_ptr;
      char const *w_base_pf = (char const *)router_weight_ptr +
                              (int64_t)local_tile * output_stride * 2;

#pragma unroll
      for (int iter = 0; iter < MAX_ITERS_PF; iter++) {
        int i_cur = tid + iter * 256;
        if (i_cur >= H4_PF) {
          break;
        }
        int byte_off = i_cur * 8;
        asm volatile("global_load_dwordx2 %0, %1, off sc0 nt"
                     : "=v"(g_pf_buf[iter])
                     : "v"(g_base_pf + byte_off)
                     : "memory");
        asm volatile("global_load_dwordx2 %0, %1, off sc0 nt"
                     : "=v"(w_pf_buf[iter])
                     : "v"(w_base_pf + byte_off)
                     : "memory");
      }
    }

    // All threads poll per-XCD release flag independently.
    // Eliminates __syncthreads — each thread confirms the barrier itself.
    // ld_nt coalesces across waves, so no extra HBM traffic.
    while (ld_nt_s32(&hier_barrier[xcd_id * HIER_STRIDE]) <
           oproj_release_expected) {
      __builtin_amdgcn_s_sleep(1);
    }
  }

  // Invalidate L2 so we read fresh O-PROJ output from HBM
  asm volatile("buffer_inv" ::: "memory");
  // Drain prefetched gamma + router weight loads (issued before barrier).
  // NT loads bypass L2, unaffected by buffer_inv.
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _sp_t2 = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[3][1], (_sp_t2 - _sp_t0) * 10); // BarrierWait
    }
    _sp_t0 = _sp_t2;
  }
#endif
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  if (tid == 0 && ts_base) {
    ts_base[10] =
        __builtin_amdgcn_s_memrealtime(); // slot 10: oproj_barrier_done
  }
#endif

  // ════════════════════════════════════════════════════════════════════════
  // PHASE 3: RMSNorm + Router GEMV
  // ════════════════════════════════════════════════════════════════════════
  // All workers redundantly compute RMSNorm on d_output (O-PROJ result).
  // Each worker computes one router expert's logit.

  if (local_tile >= router_tile_n) {
    // This worker has no TopK tile — skip TopK entirely.
    // Do NOT participate in topk_counter atomicAdd.
    goto done;
  }

  {
    using bf16 = __hip_bfloat16;
    bf16 const *__restrict__ d_hidden = static_cast<bf16 const *>(output_ptr);
    bf16 const *__restrict__ d_gamma =
        static_cast<bf16 const *>(norm_weight_ptr);
    bf16 *__restrict__ d_normed = static_cast<bf16 *>(norm_output_ptr);
    bf16 const *__restrict__ d_gate_w =
        static_cast<bf16 const *>(router_weight_ptr);
    bf16 const *__restrict__ d_rbias =
        static_cast<bf16 const *>(router_bias_ptr);
    bf16 *__restrict__ d_logits = static_cast<bf16 *>(logits_scratch_ptr);

    int const lane = tid & 63;
    int const wave = tid >> 6;

    // ── Single-pass RMSNorm + Router GEMV ──────────────────────────────
    // Fused approach: load hidden/gamma/gate_weight once, cache in
    // registers, compute ssq, then reuse cached values for norm+GEMV.
    // Eliminates redundant HBM re-read of hidden state and prefetches
    // next iteration's loads to overlap memory latency with compute.

    typedef int __attribute__((ext_vector_type(2))) i32x2_t;
    constexpr int H4 = ACTUAL_HIDDEN_DIM >> 2;
    // Max iterations per thread: ceil(H4 / 256)
    constexpr int MAX_ITERS = (H4 + 255) / 256;

    float ssq = 0.0f;
    float dp = 0.0f;
    bf16 const *my_gate = d_gate_w + local_tile * output_stride;

    char const *h_base = (char const *)d_hidden;
    char const *g_base = (char const *)d_gamma;
    char const *w_base = (char const *)my_gate;
    char *n_base = (char *)d_normed;

    // Register cache for hidden, gamma, and gate_weight raw bf16 values.
    // 3 arrays × MAX_ITERS entries × 2 VGPRs = 18 VGPRs (MAX_ITERS=3).
    i32x2_t h_cache[MAX_ITERS];
    i32x2_t g_cache[MAX_ITERS];
    i32x2_t w_cache[MAX_ITERS];
    int n_cached = 0;

    // ── Pass 1: Pipelined hidden load + ssq accumulation ────────────
    // Gamma and router weights already prefetched before O-proj barrier
    // (g_pf_buf / w_pf_buf). Only hidden state needs fresh loads here.

    // Prefetch first hidden iteration
    i32x2_t h_pf;
    if (tid < H4) {
      int byte_off = tid * 8;
      asm volatile("global_load_dwordx2 %0, %1, off"
                   : "=v"(h_pf)
                   : "v"(h_base + byte_off)
                   : "memory");
    }

#pragma unroll
    for (int iter = 0; iter < MAX_ITERS; iter++) {
      int i_cur = tid + iter * 256;
      if (i_cur >= H4) {
        break;
      }

      // Wait for hidden load (gamma+router already in VGPRs from pre-barrier)
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      i32x2_t h_v = h_pf;

      // Prefetch next hidden iteration
      int i_next = i_cur + 256;
      if (i_next < H4) {
        int byte_off_next = i_next * 8;
        asm volatile("global_load_dwordx2 %0, %1, off"
                     : "=v"(h_pf)
                     : "v"(h_base + byte_off_next)
                     : "memory");
      }

      // Cache: hidden from fresh load, gamma+router from pre-barrier prefetch
      h_cache[iter] = h_v;
      __builtin_memcpy(&g_cache[iter], &g_pf_buf[iter], 8);
      __builtin_memcpy(&w_cache[iter], &w_pf_buf[iter], 8);
      n_cached = iter + 1;

      // Compute ssq from hidden values
      float v0, v1, v2, v3;
      asm volatile("v_cvt_f32_bf16 %0, %4\n"
                   "v_cvt_f32_bf16 %1, %4 src0_sel:WORD_1\n"
                   "v_cvt_f32_bf16 %2, %5\n"
                   "v_cvt_f32_bf16 %3, %5 src0_sel:WORD_1"
                   : "=&v"(v0), "=&v"(v1), "=&v"(v2), "=&v"(v3)
                   : "v"(h_v[0]), "v"(h_v[1]));
      ssq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }

// Wave-level reduction for ssq
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
      ssq += __shfl_xor(ssq, off);
    }

    // Cross-wave reduction via LDS
    __shared__ float red[16];
    if (lane == 0) {
      red[wave] = ssq;
    }
    __syncthreads();

    float irms;
    if (tid == 0) {
      float tot = 0.0f;
      for (int w = 0; w < NUM_WAVES; w++) {
        tot += red[w];
      }
      red[0] = rsqrtf(tot / (float)ACTUAL_HIDDEN_DIM + 1e-5f);
    }
    __syncthreads();
    irms = red[0];

// ── Pass 2: Register-only norm + GEMV (no HBM re-reads) ─────────
#pragma unroll
    for (int iter = 0; iter < MAX_ITERS; iter++) {
      if (iter >= n_cached) {
        break;
      }
      int byte_off = (tid + iter * 256) * 8;

      // Unpack cached hidden
      float h0, h1, h2, h3;
      asm volatile("v_cvt_f32_bf16 %0, %4\n"
                   "v_cvt_f32_bf16 %1, %4 src0_sel:WORD_1\n"
                   "v_cvt_f32_bf16 %2, %5\n"
                   "v_cvt_f32_bf16 %3, %5 src0_sel:WORD_1"
                   : "=&v"(h0), "=&v"(h1), "=&v"(h2), "=&v"(h3)
                   : "v"(h_cache[iter][0]), "v"(h_cache[iter][1]));

      // Unpack cached gamma
      float g0, g1, g2, g3;
      asm volatile("v_cvt_f32_bf16 %0, %4\n"
                   "v_cvt_f32_bf16 %1, %4 src0_sel:WORD_1\n"
                   "v_cvt_f32_bf16 %2, %5\n"
                   "v_cvt_f32_bf16 %3, %5 src0_sel:WORD_1"
                   : "=&v"(g0), "=&v"(g1), "=&v"(g2), "=&v"(g3)
                   : "v"(g_cache[iter][0]), "v"(g_cache[iter][1]));

      // Compute norm = hidden * irms * gamma
      float n0 = h0 * irms * g0;
      float n1 = h1 * irms * g1;
      float n2 = h2 * irms * g2;
      float n3 = h3 * irms * g3;

      // Pack and store normed output
      uint32_t pk_lo, pk_hi;
      asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
                   : "=v"(pk_lo)
                   : "v"(n0), "v"(n1));
      asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
                   : "=v"(pk_hi)
                   : "v"(n2), "v"(n3));
      i32x2_t n_packed;
      n_packed[0] = (int)pk_lo;
      n_packed[1] = (int)pk_hi;
      asm volatile("global_store_dwordx2 %0, %1, off" ::"v"(n_base + byte_off),
                   "v"(n_packed)
                   : "memory");

      // Unpack cached gate_weight and accumulate dot product
      float w0, w1, w2, w3;
      asm volatile("v_cvt_f32_bf16 %0, %4\n"
                   "v_cvt_f32_bf16 %1, %4 src0_sel:WORD_1\n"
                   "v_cvt_f32_bf16 %2, %5\n"
                   "v_cvt_f32_bf16 %3, %5 src0_sel:WORD_1"
                   : "=&v"(w0), "=&v"(w1), "=&v"(w2), "=&v"(w3)
                   : "v"(w_cache[iter][0]), "v"(w_cache[iter][1]));
      dp += w0 * n0 + w1 * n1 + w2 * n2 + w3 * n3;
    }

// Wave-level reduction for dp
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
      dp += __shfl_xor(dp, off);
    }

    // Cross-wave LDS reduce
    if (lane == 0) {
      red[wave] = dp;
    }
    __syncthreads();

    // tid==0 writes logit + bias via write-through store
    if (tid == 0) {
      float s = 0.0f;
      for (int w = 0; w < NUM_WAVES; w++) {
        s += red[w];
      }
      if (d_rbias) {
        s += __bfloat162float(d_rbias[local_tile]);
      }
      bf16 bval = __float2bfloat16(s);
      st_wt_u16(&d_logits[local_tile],
                *reinterpret_cast<unsigned short *>(&bval));
    }
  }

topk_barrier :
  // ════════════════════════════════════════════════════════════════════════
  // PHASE 4: TopK barrier + softmax
  // ════════════════════════════════════════════════════════════════════════
#ifdef MPK_ENABLE_SUBPHASE_TIMING
{
  unsigned long long _sp_t3 = __builtin_amdgcn_s_memrealtime();
  if (tid == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[3][2], (_sp_t3 - _sp_t0) * 10); // RMSNormRouter
  }
  _sp_t0 = _sp_t3;
}
#endif
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  if (tid == 0 && ts_base) {
    ts_base[11] =
        __builtin_amdgcn_s_memrealtime(); // slot 11: rmsnorm_router_done
  }
#endif
  __syncthreads();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

  __shared__ int s_topk_done;
  if (tid == 0) {
    s_topk_done = atomicAdd(topk_counter, 1) + 1;
  }
  __syncthreads();

  if (s_topk_done == total_topk_tiles) {
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
    if (tid == 0 && ts_base) {
      ts_base[14] =
          __builtin_amdgcn_s_memrealtime(); // slot 14: topk_compute_start
    }
#endif
    gang_rmsnorm_topk_detail::topk_noinline<__hip_bfloat16, NUM_EXPERTS, K>(
        logits_scratch_ptr,
        topk_weight_ptr,
        routing_indices_ptr,
        active_expert_ids_ptr,
        topk_counter,
        num_active_tokens);
    // topk_noinline resets topk_counter internally
    // topk_noinline's return has per-thread s_waitcnt, but thread 0 could reach
    // threadfence_gpu before other threads finish their stores. syncthreads
    // ensures all threads complete TopK stores before thread 0 flushes L2→HBM.
    __syncthreads();
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
    if (tid == 0 && ts_base) {
      ts_base[15] =
          __builtin_amdgcn_s_memrealtime(); // slot 15: topk_compute_done
    }
#endif

    if (tid == 0) {
      // Signal routing release for fused wrapper (if present).
      //
      // The fence is required, not an optimization. The consumers of this
      // release are MoE workers on *other* XCDs (see the Phase 7b poll in
      // gang_full_layer_fused_mi300.cuh), and what they read after it are
      // active_expert_ids and routing_indices -- written just above by all
      // 256 threads of this block with ordinary st_wt stores.
      //
      // __syncthreads alone only orders those stores within this block; it
      // says nothing about when they become visible to another XCD's L2. The
      // release flags below go out via st_wt (write-through, bypassing L2),
      // so without a GPU-scope fence the flag can land in HBM ahead of the
      // routing data it advertises. A remote MoE worker then passes the
      // barrier and reads a stale active_expert_ids -- including a stale
      // count at [NUM_EXPERTS] -- and derives its expert weight base and
      // barrier slot from it. That is exactly the wild-address shape of the
      // nil-address fault.
      //
      // The sibling path already documents this contract: see
      // gang_oproj_topk_moe_fused_mi300.cuh, "TopK worker wrote per-XCD flags
      // via st_wt after threadfence_gpu". The comment above was written for a
      // fence that was never actually here.
      threadfence_gpu();
      if (routing_ready_ptr) {
        int epoch = ld_nt_s32(routing_ready_ptr) + 1;
        st_wt_u32((void *)routing_ready_ptr, (unsigned)epoch);
        for (int x = 0; x < 8; x++) {
          st_wt_u32((void *)&routing_ready_ptr[(1 + x) * 16], (unsigned)epoch);
        }
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
  }

done :
#ifdef MPK_ENABLE_SUBPHASE_TIMING
{
  unsigned long long _sp_t4 = __builtin_amdgcn_s_memrealtime();
  if (tid == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[3][3], (_sp_t4 - _sp_t0) * 10); // TopK
    atomicAdd(&g_subphase_cnt[3], 1ULL);
  }
}
#endif
  (void)0;
}

} // namespace kernel
