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

// ── O-proj LDS layout, shared with the Phase-6 weight DMA ─────────────────
// gang_full_layer_fused_mi300.cuh issues the buffer_load_lds that fills the
// weight region, and it derives the base independently -- different file,
// different template parameters. If the two formulas drift, the DMA writes
// where one says and the MFMA reads where the other says: wrong numerics, no
// crash, nothing out of bounds. So both sites call these instead.
constexpr int oproj_lds_tok_rows(int batch_size) {
  return batch_size < 16 ? batch_size : 16;
}

// End of the token staging region (FP8 activations + E8M0 block scales), which
// is also the base of the K-parallel cross-wave reduction buffer. That buffer
// used to alias the staging region at offset 0, writing into it before the
// __syncthreads while other waves could still be reading their B operands. At
// one token row that survived only because wave 2's write window just missed
// wave 3's read window; with 16 rows staged it is a clean overlap, so it gets
// its own space.
constexpr int oproj_lds_red_off(int batch_size, int reduction_size) {
  int const rows = oproj_lds_tok_rows(batch_size);
  // +16 pad per row: at ds_read_b128 granularity lane `col` lands in bank
  // group (col * (stride/16)) % 8, and 2960/16 == 185 is odd, so the 16 lanes
  // spread over all 8 groups instead of piling into one.
  int const tok_region = rows * (reduction_size + 16);
  int const sc_region = rows * (((reduction_size / 128 + 3) / 4) * 4);
  return ((tok_region + sc_region + 15) / 16) * 16;
}

// Base of the MXFP4 weight region: past the reduction buffer, which is
// NUM_WAVES(4) * 64 lanes * 4 floats = 4096 B.
constexpr int oproj_lds_w_off(int batch_size, int reduction_size) {
  return ((oproj_lds_red_off(batch_size, reduction_size) + 4096 + 15) / 16) *
         16;
}

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
        // Per-layer epoch for the Phase 2 barrier release target. Must be a
        // value that is monotonic across the whole run, bumps exactly once per
        // invocation of this barrier, and is computable by every worker without
        // reading shared state -- the fused callers pass layer_counter + 1.
        // 0 means "no layer loop", which selects the snapshot form; see the
        // comment at oproj_release_expected for why that is only safe there.
        int layer_epoch = 0,
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

  // ── Token staging layout (N-axis MFMA packing) ──────────────────────────
  // Token `col` lives at LDS row `col` and feeds N column `col` of the
  // 16x16x128 MFMA, so up to 16 batch rows cost what 1 row used to.
  constexpr int MFMA_N = 16;
  constexpr int TOK_ROWS = BATCH_SIZE < MFMA_N ? BATCH_SIZE : MFMA_N;
  constexpr int TOK_ROW_STRIDE = REDUCTION_SIZE + 16;
  constexpr int SC_STRIDE = ((MFMA_ITERS + 3) / 4) * 4;
  constexpr int TOK_REGION = TOK_ROWS * TOK_ROW_STRIDE;
  constexpr int SC_REGION = TOK_ROWS * SC_STRIDE;

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
  uint8_t *s_tok_scales = s_tok_fp8 + TOK_REGION;

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

  // Tile space is (column block, weight group), not (token, weight group):
  // the token axis moved into the MFMA's N dimension, so the host emits
  // n_bblk * n_wgs_per_xcd tiles rather than batch_size * n_wgs_per_xcd.
  int bblk = local_tile / n_wgs_per_xcd;
  int wg_idx = local_tile % n_wgs_per_xcd;
  // At TOK_ROWS == 1 there is exactly one column block, so every tile that
  // survives the guard below has bblk == 0 and the base is a literal zero.
  // The compiler cannot derive that from `batch_count - bblk*16 > 0`, and
  // leaving it as a runtime value makes every downstream address non-uniform
  // -- worth 32 bytes of scratch in the epilogue.
  int tok_row_base = TOK_ROWS == 1 ? 0 : bblk * MFMA_N;
  // Whole-block early-out only. An individual inactive lane inside a live
  // block must NOT leave -- it still owes its ds_reads and its share of the
  // MFMA, which is a wave-level op reading B from all 64 lanes.
  int n_valid_tok = batch_count - tok_row_base;
  // The token this lane owns. At TOK_ROWS == 1 only col 0 is live, so adding
  // col is a no-op -- but writing it costs the uniform token index its
  // scalar-ness, and every address derived from it becomes per-lane. Fold.
  int my_tok = TOK_ROWS == 1 ? 0 : tok_row_base + col;
  // At TOK_ROWS == 1 the surviving block has n_valid_tok == 1, so the general
  // form is exactly `col == 0` -- but only the constant form is compile-time
  // known, and handing the allocator a runtime predicate here costs 16 bytes
  // of scratch in the epilogue. Spell out the fold.
  bool tok_active = TOK_ROWS == 1 ? (col == 0) : (col < n_valid_tok);

  if (n_valid_tok <= 0) {
    goto oproj_barrier;
  }

  {
    uint8_t const *wg_data = W + static_cast<int64_t>(wg_idx) * WG_BYTES;
    uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

    // Stage up to 16 token rows. The bespoke inline quantizer that used to
    // live here was a second copy of the E8M0 logic that only ever handled
    // row 0; the shared helper covers both OUTPUT_PER_WG shapes.
    _gang_multirow_fp8_quant<REDUCTION_SIZE, TOK_ROWS, BATCH_SIZE,
                             TOK_ROW_STRIDE, SC_STRIDE>(
        A, REDUCTION_SIZE, tok_row_base, n_valid_tok, s_tok_fp8, s_tok_scales);

    // B operand base for this lane: token row `col`. Inactive lanes clamp to
    // row 0 rather than skipping, so the exec mask cannot sink above the
    // ds_read_b128 (see the CRITICAL note at the K-parallel reduce).
    uint8_t const *b_tok = s_tok_fp8 + (tok_active ? col : 0) * TOK_ROW_STRIDE;
    uint8_t const *b_scl =
        s_tok_scales + (tok_active ? col : 0) * SC_STRIDE;

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
            i32x8_t b = _gang_load_fp8_mfma_b(b_tok, ki * K_PER_MFMA, g);
            int sb = (int)b_scl[ki];
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
                _gang_load_fp8_mfma_b(b_tok, (ki + 1) * K_PER_MFMA, g);
            int sb = (int)b_scl[ki + 1];
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
                _gang_load_fp8_mfma_b(b_tok, (ki + 2) * K_PER_MFMA, g);
            int sb = (int)b_scl[ki + 2];
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
                _gang_load_fp8_mfma_b(b_tok, (ki + 3) * K_PER_MFMA, g);
            int sb = (int)b_scl[ki + 3];
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
        //
        // Lane (g, col) holds D[m = wave_tile*16 + g*4 + i][n = col], i.e. four
        // output columns of token `col`. The guard is on the token, not on
        // col == 0: every lane now has live results. Bias depends only on the
        // output column, so the 16 lanes load the same 8 bytes and coalesce;
        // residual becomes a 16-row gather -- exactly the loads the 16
        // separate per-token tiles used to issue.
        if (tok_active) {
          int out_n_local = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4;
          int out_n_global = xcd_output_col_offset + out_n_local;
          int res_idx_base = my_tok * output_stride + out_n_local;
          int out_idx_base = my_tok * output_stride + out_n_global;

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
      if (warp_id == 0 && tok_active) {
        int out_n_local_pf = wg_idx * OUTPUT_PER_WG + g * 4;
        int res_idx_pf = my_tok * output_stride + out_n_local_pf;
        unsigned short const *bias_addr = &d_bias[out_n_local_pf];
        unsigned short const *res_addr = &d_residual[res_idx_pf];
        asm volatile("global_load_dwordx2 %0, %2, off\n"
                     "global_load_dwordx2 %1, %3, off"
                     : "=&v"(pf_bias), "=&v"(pf_res)
                     : "v"(bias_addr), "v"(res_addr)
                     : "memory");
      }

      // Read weights from LDS (populated by Phase 6 buffer_load_lds DMA).
      // LDS layout mirrors HBM tile layout: data at [0..WG_DATA_BYTES),
      // scales at [OPROJ_LDS_DATA_PAD..OPROJ_LDS_DATA_PAD + WG_SCALE_BYTES).
      // Must match the Phase 6 DMA base byte-for-byte -- both call
      // oproj_lds_w_off() so there is only one formula to keep right.
      constexpr int OPROJ_LDS_OFF = oproj_lds_w_off(BATCH_SIZE, REDUCTION_SIZE);
      static_assert(OPROJ_LDS_OFF >= TOK_REGION + SC_REGION + 4096,
                    "weight region must clear the token staging + reduce "
                    "buffers");
      static_assert(OPROJ_LDS_OFF + OPROJ_LDS_DATA_PAD + OPROJ_LDS_SCALE_PAD <=
                        mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
                            mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END,
                    "O-proj LDS exceeds the MI350X budget");
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
    i32x8_t b = _gang_load_fp8_mfma_b(b_tok, (KI)*K_PER_MFMA, g);          \
    int sb = (int)b_scl[KI];                                            \
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
      //
      // Since lane_id == g*16 + col, that layout already *is*
      // [warp][g][col][4] == [warp][n-column][token][4]. Packing the token
      // axis therefore costs zero extra LDS: every lane simply keeps its own
      // slot instead of only the col==0 lanes' being read back.
      //
      // The buffer sits past the token staging region rather than aliasing it
      // at offset 0. Aliasing was safe at one token row only by accident of
      // timing; with 16 rows staged, these writes land on B operands other
      // waves are still reading.
      float *lds_reduce =
          (float *)((uint8_t *)_lm_smem +
                    oproj_lds_red_off(BATCH_SIZE, REDUCTION_SIZE));
      for (int i = 0; i < 4; i++) {
        lds_reduce[(warp_id * 64 + lane_id) * 4 + i] = acc[i];
      }
      __syncthreads();

      if (warp_id == 0 && tok_active) {
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        // Each lane reduces its own (g, col) slot: output columns g*4..g*4+3
        // of token col. At TOK_ROWS == 1 that slot is always g*16, and saying
        // so keeps the address scalar in g -- reading lane_id here instead
        // costs scratch even though the two are equal under tok_active.
        int const src_lane = TOK_ROWS == 1 ? g * 16 : lane_id;
        for (int w = 0; w < NUM_WAVES; w++) {
          v0 += lds_reduce[(w * 64 + src_lane) * 4 + 0];
          v1 += lds_reduce[(w * 64 + src_lane) * 4 + 1];
          v2 += lds_reduce[(w * 64 + src_lane) * 4 + 2];
          v3 += lds_reduce[(w * 64 + src_lane) * 4 + 3];
        }

        // bias/residual are XCD-partitioned -> use local offset
        // output is replicated -> add xcd_output_col_offset
        int out_n_local = wg_idx * OUTPUT_PER_WG + g * 4;
        int out_n_global = xcd_output_col_offset + out_n_local;
        int out_idx_base = my_tok * output_stride + out_n_global;

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
  // Drain BEFORE the rendezvous, not after. `s_waitcnt` is a per-wave
  // guarantee: run after __syncthreads it only retires wave 0's stores, and
  // tid 0 then publishes an arrival advertising output that waves 1..3 may
  // still have in flight. Draining first makes every wave's stores retire,
  // and the barrier then makes that true block-wide.
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();

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
    // Mechanism C: single global arrive + per-XCD release flags.
    //
    // The release target is derived from the layer counter, not snapshotted.
    // It used to be `ld_nt_s32(&hier_barrier[xcd_id*16]) + 1` -- "whatever is
    // there now, plus one" -- which is only correct if every one of the 184
    // workers performs that read before *any* XCD's releaser overwrites the
    // slot. Nothing orders those two events: the read sits after this block's
    // own __syncthreads, but the releaser is a different workgroup on a
    // different XCD, and it can fan out the eight release flags while a worker
    // here has not yet taken its snapshot.
    //
    // A worker that reads *after* the update computes a target one lower than
    // the value already published, so its poll is satisfied on entry: it falls
    // straight through and reads attn_proj_out while the other XCDs' O-proj
    // stores are still in flight. O-proj writes column-partitioned and the
    // RMSNorm below reads the full row, so seven eighths of what it consumes
    // may not be written yet. The observable signature is exact -- rmsnorm_out
    // differing run to run while attn_proj_out, the settled HBM content, is
    // bit-identical.
    //
    // This is the same defect the fused kernel already fixed for the other
    // three counters (see the layer_counter block in
    // gang_full_layer_fused_mi300.cuh: "These used to be snapshots of current
    // value + 1"). This barrier was the one holdout.
    //
    // `layer_epoch` is that counter, passed in by the caller rather than read
    // from a shared location: (iterations * num_layers + layer) + 1, monotonic,
    // never reset, and identical for every worker with nothing to order. The
    // two fused callers both have it in hand. The standalone task type
    // (register_gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300_task) has no
    // layer loop -- it is one task per dispatch -- so it passes 0 and keeps the
    // snapshot, which is safe there precisely because there is no second layer
    // to race with.
    int const oproj_release_expected =
        layer_epoch > 0 ? layer_epoch
                        : ld_nt_s32(&hier_barrier[xcd_id * HIER_STRIDE]) + 1;

    if (tid == 0) {
      // GPU-scope release fence before the arrival.
      //
      // atom_add_release_gpu_s32 is not a release on AMD -- its own definition
      // (mpk_atoms.cuh) emits only `flat_atomic_add ... sc0 sc1; s_waitcnt`,
      // with the comment "Ordering provided by explicit threadfence_gpu()
      // before this call." That call was never made here.
      //
      // The drain above (`s_waitcnt vmcnt(0)` + __syncthreads) retires this
      // block's O-proj stores, which is what a *same-XCD* consumer needs. But
      // this barrier is global: the workers released by it read
      // attn_proj_out columns produced on all eight XCDs, and MI300/MI350 L2
      // is not coherent across XCDs. Retiring a store means it reached this
      // XCD's L2, not that a remote XCD can see it.
      //
      // threadfence_gpu lowers to `buffer_wbl2 sc1; s_waitcnt vmcnt(0)` on
      // gfx950 -- the L2->HBM writeback that actually makes those columns
      // visible to the other seven XCDs. Without it the release flag (st_wt,
      // straight to HBM) can overtake the data it advertises.
      //
      // Contrast Phase 4's chunk barrier in gang_full_layer_fused_mi300.cuh,
      // which deliberately does *not* do this: that barrier is per-XCD, so
      // its producers and consumer share one L2 and the writeback would be
      // pure cost. Here the consumer is remote and the writeback is the whole
      // point. Only the arriving thread of each block pays it.
      threadfence_gpu();
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
    // ld_nt coalesces across waves, so no extra HBM traffic.
    while (ld_nt_s32(&hier_barrier[xcd_id * HIER_STRIDE]) <
           oproj_release_expected) {
      __builtin_amdgcn_s_sleep(1);
    }
  }

  // Rendezvous before the acquire. The per-thread poll above establishes, for
  // each wave independently, that the barrier has been released -- and that
  // used to be the whole argument for dropping this __syncthreads ("each
  // thread confirms the barrier itself"). It is not sufficient, because the
  // acquire that follows is `buffer_inv`, which is a *per-wave* instruction
  // acting on caches the whole CU shares.
  //
  // Without the rendezvous: wave 0 clears its poll, invalidates, and begins
  // reading attn_proj_out through vL1/L2 -- repopulating those lines. Wave 3
  // has not cleared its poll yet, so some of the lines wave 0 just pulled in
  // are pre-barrier values from XCDs that had not finished storing. Wave 3
  // then runs its own buffer_inv, but that invalidate happens *before* it
  // reads, and the stale lines were already re-cached by wave 0 after it. The
  // sc1 invalidate cannot undo a fill that a sibling wave performs behind it.
  //
  // Draining, rendezvousing, then invalidating makes the invalidate the first
  // memory event any wave performs after the barrier is known released
  // block-wide, which is what the acquire has to mean. This is the read side of
  // the same drain-then-rendezvous discipline the release sides in this file
  // already follow.
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();

  // Cross-XCD ACQUIRE for the O-proj output. Must be `sc1` (vL1 + L2).
  //
  // The barrier above is global, not per-XCD: every O-proj worker on all eight
  // XCDs increments hier_barrier[8*16], and the last arrival releases all of
  // them. It has to be, because O-proj *writes* column-partitioned -- worker
  // (xcd, wg) owns columns xcd*368 + wg*16 .. +16 -- while the RMSNorm below
  // reads all ACTUAL_HIDDEN_DIM columns of its token row. Seven eighths of
  // what each worker reads here was produced on a different XCD, and
  // MI300/MI350 L2 is not coherent across XCDs.
  //
  // The producer is `st_wt_u64` (sc0 sc1), so the data bypasses L2 and lands
  // in HBM. Plain `buffer_inv` drops vL1 only. This same worker read these
  // same addresses one layer ago, so its L2 still holds those lines and the
  // load is served from L2 -- returning the *previous* layer's O-proj output.
  // Which lines survive depends on L2 eviction timing, hence run to run
  // variation. The observable signature is exact: rows of `rmsnorm_out`
  // differing between two runs whose `attn_proj_out` (the settled HBM content)
  // is bit-identical -- the norm read something that is not what is in memory.
  //
  // The comment this replaces already said "Invalidate L2"; the instruction
  // simply never encoded it.
  asm volatile("buffer_inv sc1" ::: "memory");
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

    bf16 const *my_gate = d_gate_w + local_tile * output_stride;

    char const *g_base = (char const *)d_gamma;
    char const *w_base = (char const *)my_gate;

    // Register cache for hidden, gamma, and gate_weight raw bf16 values.
    // 3 arrays × MAX_ITERS entries × 2 VGPRs = 18 VGPRs (MAX_ITERS=3).
    i32x2_t h_cache[MAX_ITERS];
    i32x2_t g_cache[MAX_ITERS];
    i32x2_t w_cache[MAX_ITERS];
    int n_cached = 0;

    // Two independent cross-wave reductions run in this phase, and they must
    // not share storage. `red` carries the ssq reduction and then, at red[0],
    // the *broadcast* of irms, which every one of the 256 threads reads. The
    // dp reduction that follows used to write red[wave] -- so wave 0's
    // `red[0] = dp` lands on the very slot waves 1..3 are still reading irms
    // from.
    //
    // Nothing separates the two: the `__syncthreads()` after the irms store is
    // the *last* barrier before pass 2, and pass 2 both reads irms and writes
    // dp. Worse, irms is consumed inside pass 2's unrolled loop (n = h*irms*g),
    // so the compiler is free to keep re-reading the LDS slot rather than
    // hoisting it into a VGPR -- which widens the window from a few
    // instructions to the whole loop.
    //
    // A wave that reads a clobbered irms computes a wrong norm for its quarter
    // of the row. That value is *stored* to norm_output (the MoE's input) and
    // folded into the router dot product, so the damage lands in
    // rmsnorm_out_moe and in the routing decision at once -- and it is timing
    // dependent, hence different every run. Which quarter of the row is hit
    // depends on which wave loses the race, so the corruption is spread evenly
    // across the row rather than confined to one workgroup's columns.
    //
    // Giving dp its own slots removes the aliasing outright; 64 bytes of LDS
    // is cheaper than a third barrier on the critical path.
    __shared__ float red[16];
    __shared__ float red_dp[16];

    // This worker owns router expert `local_tile` for *every* token. The
    // pipeline below used to run once with no token offset anywhere, so at
    // batch > 1 all rows were routed by token 0's logits and the normed
    // buffer the MoE reads held token 0 in every row. It is a GEMV, not a
    // GEMM: ~184 extra FMA per thread per token, so a plain loop is the right
    // shape -- restructuring it into MFMA would cost more than it saves.
    //
    // At BATCH_SIZE == 1 the bound is a compile-time 1 and the loop vanishes.
    int const n_tok_router = BATCH_SIZE == 1 ? 1 : batch_count;

    for (int b = 0; b < n_tok_router; b++) {
      char const *h_base =
          (char const *)(d_hidden + (int64_t)b * output_stride);
      char *n_base = (char *)(d_normed + (int64_t)b * output_stride);

      float ssq = 0.0f;
      float dp = 0.0f;

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

        // Cache: hidden from the fresh load, gamma+router from the
        // pre-barrier prefetch. Both of the latter are token-invariant, but
        // the copy stays inside the loop rather than being hoisted: hoisting
        // extends g_pf_buf/w_pf_buf's live range across the whole token
        // sweep and the allocator answers with 16 more bytes of scratch even
        // at BATCH_SIZE == 1. Re-copying the same VGPRs is free.
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
        asm volatile("global_store_dwordx2 %0, %1, off" ::"v"(n_base +
                                                              byte_off),
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

      // Cross-wave LDS reduce. Into red_dp, not red: red[0] is still serving
      // as the irms broadcast that pass 2 above reads.
      if (lane == 0) {
        red_dp[wave] = dp;
      }
      __syncthreads();

      // tid==0 writes logit + bias via write-through store.
      // logits_scratch is [batch, NUM_EXPERTS], XCD-partitioned along the
      // expert axis: the pointer is pre-offset by xcd_id*CHUNK_N (which
      // topk_noinline undoes) while the row stride stays NUM_EXPERTS.
      if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < NUM_WAVES; w++) {
          s += red_dp[w];
        }
        if (d_rbias) {
          s += __bfloat162float(d_rbias[local_tile]);
        }
        bf16 bval = __float2bfloat16(s);
        st_wt_u16(&d_logits[(int64_t)b * NUM_EXPERTS + local_tile],
                  *reinterpret_cast<unsigned short *>(&bval));
      }

      // Keep the next token's `red[wave] = ssq` from racing tid 0's read of
      // this token's dp slots. Compiled out at BATCH_SIZE == 1, where the
      // loop body runs once.
      if constexpr (BATCH_SIZE > 1) {
        __syncthreads();
      }
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
  // Drain BEFORE the rendezvous, not after. `s_waitcnt` is a per-wave
  // guarantee: run after __syncthreads it only retires wave 0's stores, and
  // tid 0 then publishes an arrival advertising output that waves 1..3 may
  // still have in flight. Draining first makes every wave's stores retire,
  // and the barrier then makes that true block-wide.
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();

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
    gang_rmsnorm_topk_detail::topk_noinline<__hip_bfloat16, NUM_EXPERTS, K, BATCH_SIZE>(
        logits_scratch_ptr,
        topk_weight_ptr,
        routing_indices_ptr,
        active_expert_ids_ptr,
        topk_counter,
        num_active_tokens);
    // topk_noinline resets topk_counter internally.
    //
    // Drain then rendezvous. __syncthreads alone is NOT enough here: it makes
    // every wave *reach* this point, but `s_waitcnt` is per-wave, so a wave
    // can arrive at the barrier with its routing_indices / active_expert_ids
    // stores still in flight. tid 0 then fences and publishes the release,
    // advertising data that has not landed. Each wave must retire its own
    // stores first; the barrier then makes that true block-wide.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
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
      // barrier and reads a stale active_expert_ids / routing_indices.
      //
      // Impact is silent numerical corruption, not a crash: every stale value
      // is still in range (the count at [NUM_EXPERTS] is the compile-time
      // constant k=4 on every layer, expert ids stay in [0,128), route_val in
      // [0,4]). A token gets routed through a previous layer's expert.
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
