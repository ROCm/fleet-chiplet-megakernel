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

// Fused W13+SwiGLU+W2 MoE gang kernel for MI350 (gfx950).
//
// Single gang task replaces separate W13+SwiGLU and W2 tasks.
// Uses in-kernel atomicAdd barrier between phases (no extra event dispatch).
//
// Phase-ordered tile encoding (all W13 before all W2):
//   global_tile ∈ [0, num_activated * W13_TILES):  W13+SwiGLU phase
//   global_tile ∈ [num_activated * W13_TILES, total):  W2 phase
//
// This ensures workers exhaust all W13 tiles before reaching W2 tiles,
// avoiding spin-wait blocking while W13 work remains.
// Per-expert atomicAdd barrier: W2 for expert E starts once all W13
// tiles for expert E complete across all XCDs.
//
// Supports different OUTPUT_PER_WG for W13 and W2 phases.

#pragma once
#include "tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh" // reuse type defs + helpers
#include "tasks/mi300/swigluoai_mi300.cuh"             // fast_swigluoai()

#define MOE_DBG_SUBPHASE(code) ((void)0)

namespace kernel {

template <int BATCH_SIZE,
          int INTERMEDIATE_SIZE,
          int HIDDEN_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int W13_OUTPUT_PER_WG,
          int W2_OUTPUT_PER_WG>
__device__ __noinline__ void gang_moe_fused_mxfp4_kernel_mi300(
    void const *input_ptr,          // [batch, hidden] BF16
    void const *gate_up_weight_ptr, // [E, W13_WGS, wg_bytes] MXFP4 (interleaved
                                    // gate/up)
    void const *down_weight_ptr,    // [E, W2_WGS, wg_bytes] MXFP4
    void const *routing_ptr,        // [E, batch] int32
    void const *mask_ptr,           // [E+1] int32
    void const
        *w13_bias_ptr, // [E, 2*INTERMEDIATE_SIZE] BF16 (interleaved gate/up)
    void const *w2_bias_ptr,        // [E, HIDDEN_SIZE] BF16
    void const *routing_weight_ptr, // [batch, NUM_TOPK] float32
    void *swiglu_out_ptr,           // [batch, topk, INTERMEDIATE_SIZE] BF16
    void *workspace_f32_ptr, // [batch, HIDDEN_SIZE] float32 (atomicAdd target)
    void *barrier_ptr,       // [2*NUM_EXPERTS] int32
    int tile_idx) {

  // ── W13 constants (gate+up interleaved, MFMA reduction over hidden_size) ──
  constexpr int W13_OUTPUT_SIZE = 2 * INTERMEDIATE_SIZE; // 6144
  constexpr int W13_K = HIDDEN_SIZE;                     // 3072
  constexpr int W13_NUM_BLK32 = W13_K / 32;
  constexpr int W13_WG_DATA = W13_OUTPUT_PER_WG * (W13_K / 2);
  constexpr int W13_WG_SCALE = W13_OUTPUT_PER_WG * W13_NUM_BLK32;
  constexpr int W13_WG_BYTES = W13_WG_DATA + W13_WG_SCALE;
  constexpr int W13_WGS = W13_OUTPUT_SIZE / W13_OUTPUT_PER_WG;
  constexpr int64_t W13_EXPERT_BYTES =
      static_cast<int64_t>(W13_WGS) * W13_WG_BYTES;
  constexpr int W13_MFMA_ITERS = W13_K / 128;
  constexpr int W13_TILES = BATCH_SIZE * W13_WGS;

  // ── W2 constants (down projection, MFMA reduction over intermediate_size) ─
  constexpr int W2_OUTPUT_SIZE = HIDDEN_SIZE; // 3072
  constexpr int W2_K = INTERMEDIATE_SIZE;     // 3072
  constexpr int W2_NUM_BLK32 = W2_K / 32;
  constexpr int W2_WG_DATA = W2_OUTPUT_PER_WG * (W2_K / 2);
  constexpr int W2_WG_SCALE = W2_OUTPUT_PER_WG * W2_NUM_BLK32;
  constexpr int W2_WG_BYTES = W2_WG_DATA + W2_WG_SCALE;
  constexpr int W2_WGS = W2_OUTPUT_SIZE / W2_OUTPUT_PER_WG;
  constexpr int64_t W2_EXPERT_BYTES =
      static_cast<int64_t>(W2_WGS) * W2_WG_BYTES;
  constexpr int W2_MFMA_ITERS = W2_K / 128;
  constexpr int W2_TILES = BATCH_SIZE * W2_WGS;

  // Common constants
  constexpr int K_PER_MFMA = 128; // FP4/FP8 MFMA: 16x16x128
  constexpr int NUM_WAVES = 4;
  constexpr int W13_TILES_PER_WAVE = W13_OUTPUT_PER_WG / 16 / NUM_WAVES;
  constexpr int W2_TILES_PER_WAVE = W2_OUTPUT_PER_WG / 16 / NUM_WAVES;

  // ── Pointer setup ─────────────────────────────────────────────────────────
  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W_gate_up = (uint8_t const *)gate_up_weight_ptr;
  uint8_t const *W_down = (uint8_t const *)down_weight_ptr;
  int const *d_routing = (int const *)routing_ptr;
  int const *d_mask = (int const *)mask_ptr;
  unsigned short const *d_w13_bias = (unsigned short const *)w13_bias_ptr;
  unsigned short const *d_w2_bias = (unsigned short const *)w2_bias_ptr;
  float const *d_routing_weight = (float const *)routing_weight_ptr;
  // SwiGLU intermediate is always BF16 (avoids broken FP8 intermediate from
  // commit 89c4f70)
  unsigned short *d_swiglu_out = (unsigned short *)swiglu_out_ptr;
  float *d_workspace_f32 = (float *)workspace_f32_ptr;
  int *d_barrier = (int *)barrier_ptr;

  extern __shared__ char _fused_smem[];

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

  // ── Tile decode (phase-ordered: padded W13 then all W2) ─────────────────
  // Pad total_w13 to next multiple of 240 (30 workers × 8 XCDs) so every
  // worker's first tile is a W13 tile. This eliminates compute imbalance
  // where W2 workers would otherwise start polling the barrier immediately
  // while W13 is still running (~7.8 us wasted per layer).
  // Padding tiles (expert_idx >= num_activated) early-return in ~0 cycles.
  constexpr int PAD_MULTIPLE =
      240; // Required: see PAD_MULTIPLE investigation in memory

  int xcd_id = _gang_moe_get_xcd_id();
  int num_activated_experts = d_mask[NUM_EXPERTS];
#ifdef MPK_MOE_SINGLE_EXPERT
  num_activated_experts = min(num_activated_experts, 1);
#endif

  int global_tile = tile_idx * 8 + xcd_id;
  int total_w13_real = num_activated_experts * W13_TILES;
  int total_w13 =
      ((total_w13_real + PAD_MULTIPLE - 1) / PAD_MULTIPLE) * PAD_MULTIPLE;
  int total_w2 = num_activated_experts * W2_TILES;
  int total_tiles = total_w13 + total_w2;
  if (global_tile >= total_tiles) {
    return;
  }

  bool is_w2 = (global_tile >= total_w13);
  int expert_idx, phase_tile;
  if (!is_w2) {
    expert_idx = global_tile / W13_TILES;
    phase_tile = global_tile % W13_TILES;
    // Padding tile: expert_idx beyond activated range → skip
    if (expert_idx >= num_activated_experts) {
      return;
    }
  } else {
    int w2_tile = global_tile - total_w13;
    expert_idx = w2_tile / W2_TILES;
    phase_tile = w2_tile % W2_TILES;
  }

  int n_wgs = is_w2 ? W2_WGS : W13_WGS;
  int tok_idx = phase_tile / n_wgs;
  int wg_idx = phase_tile % n_wgs;

  int expert_id = d_mask[expert_idx];
  int const *expert_routing = d_routing + expert_id * BATCH_SIZE;

  if (tok_idx >= BATCH_SIZE) {
    return;
  }

  int route_val = expert_routing[tok_idx];
  if (route_val == 0) {
    return;
  }
  int topk_slot = route_val - 1;

#ifdef MPK_ENABLE_MOE_SUBPHASE
  g_subphase_scratch[0] = __builtin_amdgcn_s_memrealtime();
#endif

  // ══════════════════════════════════════════════════════════════════════════
  // PHASE 0: W13 + SwiGLU → write BF16 to swiglu_out
  // ══════════════════════════════════════════════════════════════════════════
  if (!is_w2) {
    MOE_DBG_SUBPHASE(2000);
    // Shared memory layout: FP8 quantized tokens + scales
    uint8_t *s_tok_fp8 = (uint8_t *)_fused_smem;
    uint8_t *s_tok_scales = s_tok_fp8 + W13_K;

    // Weight pointers
    uint8_t const *expert_weight =
        W_gate_up + static_cast<int64_t>(expert_id) * W13_EXPERT_BYTES;
    uint8_t const *wg_data =
        expert_weight + static_cast<int64_t>(wg_idx) * W13_WG_BYTES;
    uint8_t const *wg_scales = wg_data + W13_WG_DATA;

    unsigned short const *input_base = A + tok_idx * W13_K;

#ifdef MPK_W13_LDS_PREFETCH
    // ── Phase A: Issue tile_iter=0 HBM weight loads BEFORE quant ──────────
    // Loads fly during FP4 quant (microseconds of ALU+LDS work).
    constexpr int W13_TILE_ROWS = 16;
    constexpr int W13_TILE_DATA = W13_TILE_ROWS * (W13_K / 2);
    constexpr int W13_TILE_SCALE = W13_TILE_ROWS * W13_NUM_BLK32;
    constexpr int w13_n16_data = W13_TILE_DATA / 16;
    constexpr int W13_LPT = (w13_n16_data + 255) / 256;
    constexpr int W13_TILE_DATA_PADDED = W13_LPT * 256 * 16;
    constexpr int W13_TILE_BYTES = W13_TILE_DATA_PADDED + W13_TILE_SCALE;

    // Compute LDS base offset (hoisted before loads for direct HBM→LDS path)
    constexpr int LDS_W13_OFF = ((W13_K + W13_MFMA_ITERS + 15) / 16) * 16;
    static_assert(LDS_W13_OFF + W13_TILE_BYTES * NUM_WAVES <= 155 * 1024,
                  "W13 LDS weight tiles exceed MI350X LDS budget");
    uint8_t *lds_w13_base = (uint8_t *)_fused_smem + LDS_W13_OFF;
    i32x4_t w13_rsrc = make_w_buffer_rsrc(
        expert_weight, static_cast<uint32_t>(W13_EXPERT_BYTES));
    uint32_t w13_wg_voff_base = static_cast<uint32_t>(wg_idx) * W13_WG_BYTES;

    // W13 T0: direct HBM→LDS via buffer_load_dwordx4 lds:1
    // Single inline asm block to prevent compiler vmcnt serialization.
    // Without this, compiler inserts s_waitcnt vmcnt(0) between each
    // __llvm_amdgcn_raw_buffer_load_lds call, serializing 24 loads
    // (24 × ~35ns = 840ns instead of ~75ns concurrent).
    {
      unsigned lds_base_off =
          (unsigned)(uintptr_t)(lds_w13_base + warp_id * 1024);
      unsigned t0v[24], t0m[24];
#pragma unroll
      for (int t = 0; t < NUM_WAVES; t++) {
#pragma unroll
        for (int j = 0; j < W13_LPT; j++) {
          int idx = tid + j * 256;
          int clamped = idx < w13_n16_data ? idx : w13_n16_data - 1;
          t0v[t * W13_LPT + j] =
              w13_wg_voff_base +
              static_cast<uint32_t>(t * W13_TILE_ROWS * (W13_K / 2)) +
              static_cast<uint32_t>(clamped * 16);
          t0m[t * W13_LPT + j] = __builtin_amdgcn_readfirstlane(
              lds_base_off + t * W13_TILE_BYTES + j * 4096);
        }
      }
      asm volatile("s_mov_b32 m0, %[m0]\n  buffer_load_dwordx4 %[v0],  "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m1]\n  buffer_load_dwordx4 %[v1],  "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m2]\n  buffer_load_dwordx4 %[v2],  "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m3]\n  buffer_load_dwordx4 %[v3],  "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m4]\n  buffer_load_dwordx4 %[v4],  "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m5]\n  buffer_load_dwordx4 %[v5],  "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m6]\n  buffer_load_dwordx4 %[v6],  "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m7]\n  buffer_load_dwordx4 %[v7],  "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m8]\n  buffer_load_dwordx4 %[v8],  "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m9]\n  buffer_load_dwordx4 %[v9],  "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m10]\n buffer_load_dwordx4 %[v10], "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m11]\n buffer_load_dwordx4 %[v11], "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m12]\n buffer_load_dwordx4 %[v12], "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m13]\n buffer_load_dwordx4 %[v13], "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m14]\n buffer_load_dwordx4 %[v14], "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m15]\n buffer_load_dwordx4 %[v15], "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m16]\n buffer_load_dwordx4 %[v16], "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m17]\n buffer_load_dwordx4 %[v17], "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m18]\n buffer_load_dwordx4 %[v18], "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m19]\n buffer_load_dwordx4 %[v19], "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m20]\n buffer_load_dwordx4 %[v20], "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m21]\n buffer_load_dwordx4 %[v21], "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m22]\n buffer_load_dwordx4 %[v22], "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   "s_mov_b32 m0, %[m23]\n buffer_load_dwordx4 %[v23], "
                   "%[rsrc], 0 offen sc0 nt lds\n"
                   :
                   : [rsrc] "s"(w13_rsrc),
                     [v0] "v"(t0v[0]),
                     [v1] "v"(t0v[1]),
                     [v2] "v"(t0v[2]),
                     [v3] "v"(t0v[3]),
                     [v4] "v"(t0v[4]),
                     [v5] "v"(t0v[5]),
                     [v6] "v"(t0v[6]),
                     [v7] "v"(t0v[7]),
                     [v8] "v"(t0v[8]),
                     [v9] "v"(t0v[9]),
                     [v10] "v"(t0v[10]),
                     [v11] "v"(t0v[11]),
                     [v12] "v"(t0v[12]),
                     [v13] "v"(t0v[13]),
                     [v14] "v"(t0v[14]),
                     [v15] "v"(t0v[15]),
                     [v16] "v"(t0v[16]),
                     [v17] "v"(t0v[17]),
                     [v18] "v"(t0v[18]),
                     [v19] "v"(t0v[19]),
                     [v20] "v"(t0v[20]),
                     [v21] "v"(t0v[21]),
                     [v22] "v"(t0v[22]),
                     [v23] "v"(t0v[23]),
                     [m0] "s"(t0m[0]),
                     [m1] "s"(t0m[1]),
                     [m2] "s"(t0m[2]),
                     [m3] "s"(t0m[3]),
                     [m4] "s"(t0m[4]),
                     [m5] "s"(t0m[5]),
                     [m6] "s"(t0m[6]),
                     [m7] "s"(t0m[7]),
                     [m8] "s"(t0m[8]),
                     [m9] "s"(t0m[9]),
                     [m10] "s"(t0m[10]),
                     [m11] "s"(t0m[11]),
                     [m12] "s"(t0m[12]),
                     [m13] "s"(t0m[13]),
                     [m14] "s"(t0m[14]),
                     [m15] "s"(t0m[15]),
                     [m16] "s"(t0m[16]),
                     [m17] "s"(t0m[17]),
                     [m18] "s"(t0m[18]),
                     [m19] "s"(t0m[19]),
                     [m20] "s"(t0m[20]),
                     [m21] "s"(t0m[21]),
                     [m22] "s"(t0m[22]),
                     [m23] "s"(t0m[23])
                   : "memory", "m0");
    }
#endif // MPK_W13_LDS_PREFETCH — 24 dwordx4 loads in flight

    _gang_wave_parallel_fp8_quant<W13_K>(input_base, s_tok_fp8, s_tok_scales);

#ifdef MPK_ENABLE_MOE_SUBPHASE
    g_subphase_scratch[1] = __builtin_amdgcn_s_memrealtime();
#endif

#ifdef MPK_W13_LDS_PREFETCH
    // ── Phase B: Drain tile_iter=0 HBM loads + scales concurrently ──────────
    {
      constexpr int W13_SC_DW4_PER_TILE = W13_TILE_SCALE / 16; // 96

      // Issue scale loads BEFORE draining buffer_load_lds — both HBM reads
      // fly in parallel. Phase A loads are likely done (flew during quant),
      // but overlapping scale loads guarantees no unnecessary serialization.
      constexpr int W13_TOTAL_SC_DW4 = (W13_TILE_SCALE * NUM_WAVES) / 16; // 384
      constexpr int W13_SC_LPT = (W13_TOTAL_SC_DW4 + 255) / 256;          // 2
      i32x4_t w13_sc_buf[W13_SC_LPT];
      {
        i32x4_t const *sc_src = (i32x4_t const *)wg_scales;
#pragma unroll
        for (int j = 0; j < W13_SC_LPT; j++) {
          int idx = tid + j * 256;
          if (idx < W13_TOTAL_SC_DW4) {
            w13_sc_buf[j] = sc_src[idx];
          }
        }
      }

      // Drain ALL: buffer_load_lds (Phase A) + scale loads
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      {
#pragma unroll
        for (int j = 0; j < W13_SC_LPT; j++) {
          int idx = tid + j * 256;
          if (idx < W13_TOTAL_SC_DW4) {
            int tile = idx / W13_SC_DW4_PER_TILE;
            int off = idx % W13_SC_DW4_PER_TILE;
            i32x4_t *dst_sc = (i32x4_t *)(lds_w13_base + tile * W13_TILE_BYTES +
                                          W13_TILE_DATA_PADDED);
            dst_sc[off] = w13_sc_buf[j];
          }
        }
      }

      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      __syncthreads();

      // (timestamp [2] moved inside asm block below)
      // ── Phase C: MFMA from LDS ───────────────────────────────────────
      {
        uint8_t *lds_w13_data = lds_w13_base + warp_id * W13_TILE_BYTES;
        uint8_t *lds_w13_sc = lds_w13_data + W13_TILE_DATA_PADDED;
        int w_row_local = col;
        int const row_data_base = w_row_local * (W13_K / 2);
        int const row_scale_base = w_row_local * W13_NUM_BLK32;

        int wave_tile_0 = warp_id;
        f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

        // Depth-2 pipelined FP8 MFMA loop (full asm, single-buffer).
        //
        // Pipeline: after s_waitcnt, regs hold iter N data. Issue ds_reads for
        // iter N+1 into the SAME data regs (v[22:25], v7, v[8:15]), then MFMA.
        // MFMA reads source VGPRs at issue time — before ds_reads complete
        // (~20 cycles later) — so it gets iter N data correctly.
        //
        // Token B scale uses a separate register (v17) to avoid collision
        // with MFMA's v16 read. Copy v17→v16 after MFMA while data reads fly.
        //
        // Baseline: ~53 cycles/iter (20 wait + 32 MFMA + 1 overhead)
        // Pipelined: ~36 cycles/iter (0 wait + 32 MFMA + 4 overhead)
        // Saves 17 × 24 = 408 cycles per loop invocation.
        {
          unsigned w_addr =
              (unsigned)(uintptr_t)(lds_w13_data + row_data_base + g * 16);
          unsigned ws_addr =
              (unsigned)(uintptr_t)(lds_w13_sc + row_scale_base + g);
          unsigned t_addr = (unsigned)(uintptr_t)(s_tok_fp8 + g * 16);
          unsigned ts_addr = (unsigned)(uintptr_t)(s_tok_scales);
          asm volatile(
              // Zero accumulator
              "v_accvgpr_write_b32 a0, 0\n"
              "v_accvgpr_write_b32 a1, 0\n"
              "v_accvgpr_write_b32 a2, 0\n"
              "v_accvgpr_write_b32 a3, 0\n"

              // Pre-issue 5 reads for iteration 0
              "ds_read_b128 v[22:25], %[wa]\n"           // weight A (16B FP4)
              "ds_read_u8   v7, %[wsa]\n"                // weight A scale
              "ds_read_b128 v[8:11], %[ta]\n"            // token B lo (16B FP8)
              "ds_read_b128 v[12:15], %[ta] offset:64\n" // token B hi (16B FP8)
              "ds_read_u8   v16, %[tsa]\n"               // token B scale
              "s_mov_b32 s13, 0\n"                       // loop counter

              // ── Iterations 0..22: prefetch next, MFMA current ──
              "PIPELINED_W13_T0_%=:\n"
              "s_waitcnt lgkmcnt(0)\n" // iter N reads complete

              // Advance addresses for iter N+1
              "v_add_u32_e32 %[wa], 64, %[wa]\n"
              "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
              "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
              "s_add_i32 s13, s13, 1\n"

              // Token B scale → v17 FIRST (oldest in queue, completes first)
              "v_add_u32_e32 v17, s13, %[tsa]\n"
              "ds_read_u8   v17, v17\n" // [lgkmcnt +1] oldest
              // Prefetch iter N+1 data into SAME regs (MFMA reads old values)
              "ds_read_b128 v[22:25], %[wa]\n"           // [lgkmcnt +2]
              "ds_read_u8   v7, %[wsa]\n"                // [lgkmcnt +3]
              "ds_read_b128 v[8:11], %[ta]\n"            // [lgkmcnt +4]
              "ds_read_b128 v[12:15], %[ta] offset:64\n" // [lgkmcnt +5]

              // MFMA from iter N data (32 cycles, reads v[22:25] v[8:15] v7
              // v16)
              "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
              "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"

              // Copy next token scale during MFMA execution.
              // lgkmcnt(4): wait for token scale (oldest, issued first), leave
              // 4 data reads flying
              "s_waitcnt lgkmcnt(4)\n"
              "v_mov_b32_e32 v16, v17\n"

              "s_cmpk_lt_i32 s13, %[iters_m1]\n"
              "s_cbranch_scc1 PIPELINED_W13_T0_%=\n"

              // ── Final iteration: no more prefetch needed ──
              "s_waitcnt lgkmcnt(0)\n"
              "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
              "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"

              // Read accumulator into output
              "s_nop 7\n"
              "s_nop 0\n"
              "v_accvgpr_read_b32 %[acc0], a0\n"
              "v_accvgpr_read_b32 %[acc1], a1\n"
              "v_accvgpr_read_b32 %[acc2], a2\n"
              "v_accvgpr_read_b32 %[acc3], a3\n"
              : [acc0] "=v"(acc[0]),
                [acc1] "=v"(acc[1]),
                [acc2] "=v"(acc[2]),
                [acc3] "=v"(acc[3]),
                [wa] "+v"(w_addr),
                [wsa] "+v"(ws_addr),
                [ta] "+v"(t_addr)
              : [tsa] "v"(ts_addr), [iters_m1] "n"(W13_MFMA_ITERS - 1)
              : "memory",
                "s13",
                "v7",
                "v8",
                "v9",
                "v10",
                "v11",
                "v12",
                "v13",
                "v14",
                "v15",
                "v16",
                "v17",
                "v22",
                "v23",
                "v24",
                "v25",
                "a0",
                "a1",
                "a2",
                "a3");
        }

        // ── Issue tile_iter=1 per-wave HBM→LDS loads BEFORE SwiGLU ──
        // Each wave loads only its own tile slot. The buffer_load_lds
        // writes go to [warp_id * W13_TILE_BYTES + s*1024 + j*4096]
        // replicating the cooperative layout without cross-wave deps.
        // No __syncthreads needed: each wave's MFMA is done reading LDS
        // before the loads overwrite that same tile slot.
        // Loads fly during SwiGLU epilogue (overlap HBM latency).
        if (W13_TILES_PER_WAVE > 1) {
          unsigned lds_t1_base =
              (unsigned)(uintptr_t)(lds_w13_base + warp_id * W13_TILE_BYTES);
          uint32_t t1_hbm_base =
              w13_wg_voff_base +
              static_cast<uint32_t>((warp_id + NUM_WAVES) * W13_TILE_ROWS *
                                    (W13_K / 2));
          unsigned t1v[24], t1m[24];
#pragma unroll
          for (int s = 0; s < NUM_WAVES; s++) {
#pragma unroll
            for (int j = 0; j < W13_LPT; j++) {
              int element = s * 64 + lane_id + j * 256;
              int clamped = element < w13_n16_data ? element : w13_n16_data - 1;
              t1v[s * W13_LPT + j] =
                  t1_hbm_base + static_cast<uint32_t>(clamped * 16);
              t1m[s * W13_LPT + j] = __builtin_amdgcn_readfirstlane(
                  lds_t1_base + s * 1024 + j * 4096);
            }
          }
          asm volatile("s_mov_b32 m0, %[m0]\n  buffer_load_dwordx4 %[v0],  "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m1]\n  buffer_load_dwordx4 %[v1],  "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m2]\n  buffer_load_dwordx4 %[v2],  "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m3]\n  buffer_load_dwordx4 %[v3],  "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m4]\n  buffer_load_dwordx4 %[v4],  "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m5]\n  buffer_load_dwordx4 %[v5],  "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m6]\n  buffer_load_dwordx4 %[v6],  "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m7]\n  buffer_load_dwordx4 %[v7],  "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m8]\n  buffer_load_dwordx4 %[v8],  "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m9]\n  buffer_load_dwordx4 %[v9],  "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m10]\n buffer_load_dwordx4 %[v10], "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m11]\n buffer_load_dwordx4 %[v11], "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m12]\n buffer_load_dwordx4 %[v12], "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m13]\n buffer_load_dwordx4 %[v13], "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m14]\n buffer_load_dwordx4 %[v14], "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m15]\n buffer_load_dwordx4 %[v15], "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m16]\n buffer_load_dwordx4 %[v16], "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m17]\n buffer_load_dwordx4 %[v17], "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m18]\n buffer_load_dwordx4 %[v18], "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m19]\n buffer_load_dwordx4 %[v19], "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m20]\n buffer_load_dwordx4 %[v20], "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m21]\n buffer_load_dwordx4 %[v21], "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m22]\n buffer_load_dwordx4 %[v22], "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       "s_mov_b32 m0, %[m23]\n buffer_load_dwordx4 %[v23], "
                       "%[rsrc], 0 offen sc0 nt lds\n"
                       :
                       : [rsrc] "s"(w13_rsrc),
                         [v0] "v"(t1v[0]),
                         [v1] "v"(t1v[1]),
                         [v2] "v"(t1v[2]),
                         [v3] "v"(t1v[3]),
                         [v4] "v"(t1v[4]),
                         [v5] "v"(t1v[5]),
                         [v6] "v"(t1v[6]),
                         [v7] "v"(t1v[7]),
                         [v8] "v"(t1v[8]),
                         [v9] "v"(t1v[9]),
                         [v10] "v"(t1v[10]),
                         [v11] "v"(t1v[11]),
                         [v12] "v"(t1v[12]),
                         [v13] "v"(t1v[13]),
                         [v14] "v"(t1v[14]),
                         [v15] "v"(t1v[15]),
                         [v16] "v"(t1v[16]),
                         [v17] "v"(t1v[17]),
                         [v18] "v"(t1v[18]),
                         [v19] "v"(t1v[19]),
                         [v20] "v"(t1v[20]),
                         [v21] "v"(t1v[21]),
                         [v22] "v"(t1v[22]),
                         [v23] "v"(t1v[23]),
                         [m0] "s"(t1m[0]),
                         [m1] "s"(t1m[1]),
                         [m2] "s"(t1m[2]),
                         [m3] "s"(t1m[3]),
                         [m4] "s"(t1m[4]),
                         [m5] "s"(t1m[5]),
                         [m6] "s"(t1m[6]),
                         [m7] "s"(t1m[7]),
                         [m8] "s"(t1m[8]),
                         [m9] "s"(t1m[9]),
                         [m10] "s"(t1m[10]),
                         [m11] "s"(t1m[11]),
                         [m12] "s"(t1m[12]),
                         [m13] "s"(t1m[13]),
                         [m14] "s"(t1m[14]),
                         [m15] "s"(t1m[15]),
                         [m16] "s"(t1m[16]),
                         [m17] "s"(t1m[17]),
                         [m18] "s"(t1m[18]),
                         [m19] "s"(t1m[19]),
                         [m20] "s"(t1m[20]),
                         [m21] "s"(t1m[21]),
                         [m22] "s"(t1m[22]),
                         [m23] "s"(t1m[23])
                       : "memory", "m0");
        }

        // tile_iter=0 SwiGLU epilogue (tile_iter=1 HBM loads fly in background)
        if (col == 0) {
          constexpr int ACT_STRIDE = W13_OUTPUT_SIZE / 2;
          for (int i = 0; i < 4; i += 2) {
            int out_n =
                wg_idx * W13_OUTPUT_PER_WG + wave_tile_0 * 16 + g * 4 + i;
            if (out_n + 1 < W13_OUTPUT_SIZE) {
              unsigned bt_g =
                  (unsigned)d_w13_bias[expert_id * W13_OUTPUT_SIZE + out_n]
                  << 16;
              unsigned bt_u =
                  (unsigned)d_w13_bias[expert_id * W13_OUTPUT_SIZE + out_n + 1]
                  << 16;
              float bias_g;
              __builtin_memcpy(&bias_g, &bt_g, 4);
              float bias_u;
              __builtin_memcpy(&bias_u, &bt_u, 4);
              float activated =
                  fast_swigluoai(acc[i] + bias_g, acc[i + 1] + bias_u);
              int act_n = out_n / 2;
              int out_idx = tok_idx * (NUM_TOPK * ACT_STRIDE) +
                            topk_slot * ACT_STRIDE + act_n;
              st_wt_u16(&d_swiglu_out[out_idx], _gang_float_to_bf16(activated));
            }
          }
        }
      }

      // ── tile_iter=1: drain per-wave loads + scales → MFMA ──────────────
      if (W13_TILES_PER_WAVE > 1) {
        // Drain per-wave buffer_load_lds writes (issued before SwiGLU above)
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

        // Per-wave scale load + scatter (each wave loads only its own tile's
        // scales)
        {
          constexpr int W13_SC_DW4_PER_TILE = W13_TILE_SCALE / 16;
          constexpr int W13_SC_LPT_WAVE = (W13_SC_DW4_PER_TILE + 63) / 64;
          i32x4_t const *sc_src =
              (i32x4_t const *)(wg_scales + (warp_id + NUM_WAVES) *
                                                W13_TILE_ROWS * W13_NUM_BLK32);
          i32x4_t w13_sc_wave[W13_SC_LPT_WAVE];
#pragma unroll
          for (int j = 0; j < W13_SC_LPT_WAVE; j++) {
            int idx = lane_id + j * 64;
            if (idx < W13_SC_DW4_PER_TILE) {
              w13_sc_wave[j] = sc_src[idx];
            }
          }
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          i32x4_t *dst_sc =
              (i32x4_t *)(lds_w13_base + warp_id * W13_TILE_BYTES +
                          W13_TILE_DATA_PADDED);
#pragma unroll
          for (int j = 0; j < W13_SC_LPT_WAVE; j++) {
            int idx = lane_id + j * 64;
            if (idx < W13_SC_DW4_PER_TILE) {
              dst_sc[idx] = w13_sc_wave[j];
            }
          }
        }
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        // No __syncthreads needed — each wave only reads from its own LDS tile

        // tile_iter=1 MFMA loop — reads weights from LDS
        {
          uint8_t *lds_w13_data = lds_w13_base + warp_id * W13_TILE_BYTES;
          uint8_t *lds_w13_sc = lds_w13_data + W13_TILE_DATA_PADDED;
          int w_row_local = col;
          int const row_data_base = w_row_local * (W13_K / 2);
          int const row_scale_base = w_row_local * W13_NUM_BLK32;

          int wave_tile_1 = warp_id + NUM_WAVES; // tile_iter=1
          f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

          // Depth-2 pipelined FP8 MFMA loop (tile_iter=1, full asm)
          {
            unsigned w_addr =
                (unsigned)(uintptr_t)(lds_w13_data + row_data_base + g * 16);
            unsigned ws_addr =
                (unsigned)(uintptr_t)(lds_w13_sc + row_scale_base + g);
            unsigned t_addr = (unsigned)(uintptr_t)(s_tok_fp8 + g * 16);
            unsigned ts_addr = (unsigned)(uintptr_t)(s_tok_scales);

            asm volatile(
                "v_accvgpr_write_b32 a0, 0\n"
                "v_accvgpr_write_b32 a1, 0\n"
                "v_accvgpr_write_b32 a2, 0\n"
                "v_accvgpr_write_b32 a3, 0\n"
                "ds_read_b128 v[22:25], %[wa]\n"
                "ds_read_u8   v7, %[wsa]\n"
                "ds_read_b128 v[8:11], %[ta]\n"
                "ds_read_b128 v[12:15], %[ta] offset:64\n"
                "ds_read_u8   v16, %[tsa]\n"
                "s_mov_b32 s13, 0\n"
                "PIPELINED_W13_T1_%=:\n"
                "s_waitcnt lgkmcnt(0)\n"
                "v_add_u32_e32 %[wa], 64, %[wa]\n"
                "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
                "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
                "s_add_i32 s13, s13, 1\n"
                "v_add_u32_e32 v17, s13, %[tsa]\n"
                "ds_read_u8   v17, v17\n"
                "ds_read_b128 v[22:25], %[wa]\n"
                "ds_read_u8   v7, %[wsa]\n"
                "ds_read_b128 v[8:11], %[ta]\n"
                "ds_read_b128 v[12:15], %[ta] offset:64\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
                "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
                "s_waitcnt lgkmcnt(4)\n"
                "v_mov_b32_e32 v16, v17\n"
                "s_cmpk_lt_i32 s13, %[iters_m1]\n"
                "s_cbranch_scc1 PIPELINED_W13_T1_%=\n"
                "s_waitcnt lgkmcnt(0)\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
                "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
                "s_nop 7\n"
                "s_nop 0\n"
                "v_accvgpr_read_b32 %[acc0], a0\n"
                "v_accvgpr_read_b32 %[acc1], a1\n"
                "v_accvgpr_read_b32 %[acc2], a2\n"
                "v_accvgpr_read_b32 %[acc3], a3\n"
                : [acc0] "=v"(acc[0]),
                  [acc1] "=v"(acc[1]),
                  [acc2] "=v"(acc[2]),
                  [acc3] "=v"(acc[3]),
                  [wa] "+v"(w_addr),
                  [wsa] "+v"(ws_addr),
                  [ta] "+v"(t_addr)
                : [tsa] "v"(ts_addr), [iters_m1] "n"(W13_MFMA_ITERS - 1)
                : "memory",
                  "s13",
                  "v7",
                  "v8",
                  "v9",
                  "v10",
                  "v11",
                  "v12",
                  "v13",
                  "v14",
                  "v15",
                  "v16",
                  "v17",
                  "v22",
                  "v23",
                  "v24",
                  "v25",
                  "a0",
                  "a1",
                  "a2",
                  "a3");
          }
          // tile_iter=1 SwiGLU epilogue
          if (col == 0) {
            constexpr int ACT_STRIDE = W13_OUTPUT_SIZE / 2;
            for (int i = 0; i < 4; i += 2) {
              int out_n =
                  wg_idx * W13_OUTPUT_PER_WG + wave_tile_1 * 16 + g * 4 + i;
              if (out_n + 1 < W13_OUTPUT_SIZE) {
                unsigned bt_g =
                    (unsigned)d_w13_bias[expert_id * W13_OUTPUT_SIZE + out_n]
                    << 16;
                unsigned bt_u =
                    (unsigned)
                        d_w13_bias[expert_id * W13_OUTPUT_SIZE + out_n + 1]
                    << 16;
                float bias_g;
                __builtin_memcpy(&bias_g, &bt_g, 4);
                float bias_u;
                __builtin_memcpy(&bias_u, &bt_u, 4);
                float activated =
                    fast_swigluoai(acc[i] + bias_g, acc[i + 1] + bias_u);
                int act_n = out_n / 2;
                int out_idx = tok_idx * (NUM_TOPK * ACT_STRIDE) +
                              topk_slot * ACT_STRIDE + act_n;
                st_wt_u16(&d_swiglu_out[out_idx],
                          _gang_float_to_bf16(activated));
              }
            }
          }
        }
      }
    }
#else // !MPK_W13_LDS_PREFETCH — original HBM-direct code
    // Depth-8 pipelined MFMA loop for W13.
    // 8 slots × 32 cycles = 256-cycle prefetch distance (~64% of HBM latency).
    // 24/8=3 loop iterations — compiler keeps the loop (verified in assembly).
    for (int tile_iter = 0; tile_iter < W13_TILES_PER_WAVE; tile_iter++) {
      int wave_tile = warp_id + tile_iter * NUM_WAVES;
      int w_row = wave_tile * 16 + col;
      int const row_data_base = w_row * (W13_K / 2);
      int const row_scale_base = w_row * W13_NUM_BLK32;

      f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

#ifdef MPK_W13_LDS_WEIGHTS
      i32x8_t a0 = {0, 0, 0, 0, 0, 0, 0, 0};
      int sa0 = 127;
      i32x8_t a1 = a0;
      int sa1 = 127;
      i32x8_t a2 = a0;
      int sa2 = 127;
      i32x8_t a3 = a0;
      int sa3 = 127;
      i32x8_t a4 = a0;
      int sa4 = 127;
      i32x8_t a5 = a0;
      int sa5 = 127;
      i32x8_t a6 = a0;
      int sa6 = 127;
      i32x8_t a7 = a0;
      int sa7 = 127;
#else
      // Pre-fill: load k-tiles 0..7 into 8 pipeline slots
      i32x8_t a0 =
          *(i32x8_t const *)(wg_data + row_data_base + 0 * 64 + g * 16);
      int sa0 = (int)wg_scales[row_scale_base + 0 * 4 + g];
      i32x8_t a1 =
          *(i32x8_t const *)(wg_data + row_data_base + 1 * 64 + g * 16);
      int sa1 = (int)wg_scales[row_scale_base + 1 * 4 + g];
      i32x8_t a2 =
          *(i32x8_t const *)(wg_data + row_data_base + 2 * 64 + g * 16);
      int sa2 = (int)wg_scales[row_scale_base + 2 * 4 + g];
      i32x8_t a3 =
          *(i32x8_t const *)(wg_data + row_data_base + 3 * 64 + g * 16);
      int sa3 = (int)wg_scales[row_scale_base + 3 * 4 + g];
      i32x8_t a4 =
          *(i32x8_t const *)(wg_data + row_data_base + 4 * 64 + g * 16);
      int sa4 = (int)wg_scales[row_scale_base + 4 * 4 + g];
      i32x8_t a5 =
          *(i32x8_t const *)(wg_data + row_data_base + 5 * 64 + g * 16);
      int sa5 = (int)wg_scales[row_scale_base + 5 * 4 + g];
      i32x8_t a6 =
          *(i32x8_t const *)(wg_data + row_data_base + 6 * 64 + g * 16);
      int sa6 = (int)wg_scales[row_scale_base + 6 * 4 + g];
      i32x8_t a7 =
          *(i32x8_t const *)(wg_data + row_data_base + 7 * 64 + g * 16);
      int sa7 = (int)wg_scales[row_scale_base + 7 * 4 + g];
#endif

#pragma unroll 1
      for (int ki = 0; ki < W13_MFMA_ITERS; ki += 8) {
        {
          i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
          int sb = (int)s_tok_scales[ki];
          acc = _gang_mfma_f4xf8(a0, b, acc, sa0, sb);
        }
        if (ki + 8 < W13_MFMA_ITERS) {
#ifdef MPK_W13_LDS_WEIGHTS
          a0 = {0, 0, 0, 0, 0, 0, 0, 0};
          sa0 = 127;
#else
          a0 = *(i32x8_t const *)(wg_data + row_data_base + (ki + 8) * 64 +
                                  g * 16);
          sa0 = (int)wg_scales[row_scale_base + (ki + 8) * 4 + g];
#endif
        }
        {
          i32x8_t b =
              _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
          int sb = (int)s_tok_scales[ki + 1];
          acc = _gang_mfma_f4xf8(a1, b, acc, sa1, sb);
        }
        if (ki + 9 < W13_MFMA_ITERS) {
#ifdef MPK_W13_LDS_WEIGHTS
          a1 = {0, 0, 0, 0, 0, 0, 0, 0};
          sa1 = 127;
#else
          a1 = *(i32x8_t const *)(wg_data + row_data_base + (ki + 9) * 64 +
                                  g * 16);
          sa1 = (int)wg_scales[row_scale_base + (ki + 9) * 4 + g];
#endif
        }
        {
          i32x8_t b =
              _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
          int sb = (int)s_tok_scales[ki + 2];
          acc = _gang_mfma_f4xf8(a2, b, acc, sa2, sb);
        }
        if (ki + 10 < W13_MFMA_ITERS) {
#ifdef MPK_W13_LDS_WEIGHTS
          a2 = {0, 0, 0, 0, 0, 0, 0, 0};
          sa2 = 127;
#else
          a2 = *(i32x8_t const *)(wg_data + row_data_base + (ki + 10) * 64 +
                                  g * 16);
          sa2 = (int)wg_scales[row_scale_base + (ki + 10) * 4 + g];
#endif
        }
        {
          i32x8_t b =
              _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
          int sb = (int)s_tok_scales[ki + 3];
          acc = _gang_mfma_f4xf8(a3, b, acc, sa3, sb);
        }
        if (ki + 11 < W13_MFMA_ITERS) {
#ifdef MPK_W13_LDS_WEIGHTS
          a3 = {0, 0, 0, 0, 0, 0, 0, 0};
          sa3 = 127;
#else
          a3 = *(i32x8_t const *)(wg_data + row_data_base + (ki + 11) * 64 +
                                  g * 16);
          sa3 = (int)wg_scales[row_scale_base + (ki + 11) * 4 + g];
#endif
        }
        {
          i32x8_t b =
              _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 4) * K_PER_MFMA, g);
          int sb = (int)s_tok_scales[ki + 4];
          acc = _gang_mfma_f4xf8(a4, b, acc, sa4, sb);
        }
        if (ki + 12 < W13_MFMA_ITERS) {
#ifdef MPK_W13_LDS_WEIGHTS
          a4 = {0, 0, 0, 0, 0, 0, 0, 0};
          sa4 = 127;
#else
          a4 = *(i32x8_t const *)(wg_data + row_data_base + (ki + 12) * 64 +
                                  g * 16);
          sa4 = (int)wg_scales[row_scale_base + (ki + 12) * 4 + g];
#endif
        }
        {
          i32x8_t b =
              _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 5) * K_PER_MFMA, g);
          int sb = (int)s_tok_scales[ki + 5];
          acc = _gang_mfma_f4xf8(a5, b, acc, sa5, sb);
        }
        if (ki + 13 < W13_MFMA_ITERS) {
#ifdef MPK_W13_LDS_WEIGHTS
          a5 = {0, 0, 0, 0, 0, 0, 0, 0};
          sa5 = 127;
#else
          a5 = *(i32x8_t const *)(wg_data + row_data_base + (ki + 13) * 64 +
                                  g * 16);
          sa5 = (int)wg_scales[row_scale_base + (ki + 13) * 4 + g];
#endif
        }
        {
          i32x8_t b =
              _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 6) * K_PER_MFMA, g);
          int sb = (int)s_tok_scales[ki + 6];
          acc = _gang_mfma_f4xf8(a6, b, acc, sa6, sb);
        }
        if (ki + 14 < W13_MFMA_ITERS) {
#ifdef MPK_W13_LDS_WEIGHTS
          a6 = {0, 0, 0, 0, 0, 0, 0, 0};
          sa6 = 127;
#else
          a6 = *(i32x8_t const *)(wg_data + row_data_base + (ki + 14) * 64 +
                                  g * 16);
          sa6 = (int)wg_scales[row_scale_base + (ki + 14) * 4 + g];
#endif
        }
        if (ki + 7 < W13_MFMA_ITERS) {
          i32x8_t b =
              _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 7) * K_PER_MFMA, g);
          int sb = (int)s_tok_scales[ki + 7];
          acc = _gang_mfma_f4xf8(a7, b, acc, sa7, sb);
        }
        if (ki + 15 < W13_MFMA_ITERS) {
#ifdef MPK_W13_LDS_WEIGHTS
          a7 = {0, 0, 0, 0, 0, 0, 0, 0};
          sa7 = 127;
#else
          a7 = *(i32x8_t const *)(wg_data + row_data_base + (ki + 15) * 64 +
                                  g * 16);
          sa7 = (int)wg_scales[row_scale_base + (ki + 15) * 4 + g];
#endif
        }
      }

      // Fused SwiGLU epilogue (identical to gang_moe_linear_mxfp4 FUSE_SWIGLU
      // path)
      if (col == 0) {
        constexpr int ACT_STRIDE = W13_OUTPUT_SIZE / 2; // = INTERMEDIATE_SIZE
        for (int i = 0; i < 4; i += 2) {
          int out_n = wg_idx * W13_OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
          if (out_n + 1 < W13_OUTPUT_SIZE) {
            unsigned bt_g =
                (unsigned)d_w13_bias[expert_id * W13_OUTPUT_SIZE + out_n] << 16;
            unsigned bt_u =
                (unsigned)d_w13_bias[expert_id * W13_OUTPUT_SIZE + out_n + 1]
                << 16;
            float bias_g;
            __builtin_memcpy(&bias_g, &bt_g, 4);
            float bias_u;
            __builtin_memcpy(&bias_u, &bt_u, 4);

            float activated =
                fast_swigluoai(acc[i] + bias_g, acc[i + 1] + bias_u);

            int act_n = out_n / 2;
            int out_idx = tok_idx * (NUM_TOPK * ACT_STRIDE) +
                          topk_slot * ACT_STRIDE + act_n;
            st_wt_u16(&d_swiglu_out[out_idx], _gang_float_to_bf16(activated));
          }
        }
      }
    }
#endif // MPK_W13_LDS_PREFETCH

#ifdef MPK_ENABLE_MOE_SUBPHASE
    g_subphase_scratch[4] = __builtin_amdgcn_s_memrealtime();
#endif

    __asm__ __volatile__("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();

    // ── Mechanism C W13 signal (producer side)
    // ──────────────────────────────── Uses layer index from shared memory for
    // monotonically increasing release
    MOE_DBG_SUBPHASE(2001);
    constexpr int HIER_STRIDE = 16;
    if (tid == 0) {
      int base = expert_id * HIER_STRIDE;
      // Single global arrival (all W13 tiles increment one counter)
      int prev_global = atom_add_release_gpu_s32(&d_barrier[base + 8], 1);
      if ((prev_global % W13_TILES) == W13_TILES - 1) {
        // Last W13 arrival: write per-XCD release = layer_idx + 1
        constexpr int LAYER_IDX_SMEM_OFF =
            mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
            mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END;
        int layer_idx =
            *reinterpret_cast<int *>(&_fused_smem[LAYER_IDX_SMEM_OFF]);
        int release_val = layer_idx + 1;
        for (int x = 0; x < 8; x++) {
          st_wt_u32((void *)&d_barrier[base + x], (unsigned)release_val);
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
    }

#ifdef MPK_ENABLE_MOE_SUBPHASE
    g_subphase_scratch[2] = __builtin_amdgcn_s_memrealtime();
    // Raw timestamps in scratch[0..4] — deltas computed by scheduler
#endif

    return;
  }

  // ══════════════════════════════════════════════════════════════════════════
  // PHASE 1: W2 (down projection) → write BF16 to mlp_out
  // ══════════════════════════════════════════════════════════════════════════

  // Shared memory layout: FP8 tokens + per-MFMA-tile scales
  uint8_t *s_tok_fp8 = (uint8_t *)_fused_smem;
  constexpr int W2_TOTAL_MFMA = W2_K / K_PER_MFMA;
  uint8_t *s_tok_scales = s_tok_fp8 + W2_K;

  MOE_DBG_SUBPHASE(3000);
  // Weight pointers — depend only on expert_id/wg_idx, available before barrier
  uint8_t const *expert_weight =
      W_down + static_cast<int64_t>(expert_id) * W2_EXPERT_BYTES;
  uint8_t const *wg_data =
      expert_weight + static_cast<int64_t>(wg_idx) * W2_WG_BYTES;
  uint8_t const *wg_scales = wg_data + W2_WG_DATA;

  constexpr int W2_TILE_ROWS = 16;
  constexpr int W2_TILE_DATA = W2_TILE_ROWS * (W2_K / 2);
  constexpr int W2_TILE_SCALE = W2_TILE_ROWS * W2_NUM_BLK32;
  constexpr int w2_n16_data = W2_TILE_DATA / 16;
  constexpr int W2_LPT = (w2_n16_data + 255) / 256;
  constexpr int W2_TILE_DATA_PADDED = W2_LPT * 256 * 16;
  constexpr int W2_TILE_BYTES = W2_TILE_DATA_PADDED + W2_TILE_SCALE;

  // ── W2 weight prefetch + barrier wait overlap ────────────────────────────
  // Strategy: issue buffer_load_lds for W2 weights BEFORE barrier poll so
  // HBM latency (~3us) overlaps with barrier wait instead of serializing.
  // Token quant writes to LDS[0..W2_K+scales], weights write to LDS[W2_OFF..],
  // so they don't conflict — both can be in flight simultaneously.

  // W2 resource descriptor + voff_base (data-independent, compute before
  // barrier)
  i32x4_t w2_rsrc =
      make_w_buffer_rsrc(expert_weight, static_cast<uint32_t>(W2_EXPERT_BYTES));
  uint32_t w2_wg_voff_base = static_cast<uint32_t>(wg_idx) * W2_WG_BYTES;

  constexpr int LDS_W2_OFF = ((W2_K + W2_MFMA_ITERS + 15) / 16) * 16;
  static_assert(LDS_W2_OFF + W2_TILE_BYTES * NUM_WAVES <= 155 * 1024,
                "W2 LDS weight tiles exceed MI350X LDS budget");
  uint8_t *lds_w2_base = (uint8_t *)_fused_smem + LDS_W2_OFF;

  MOE_DBG_SUBPHASE(3001);
  // All threads independently read layer_idx from LDS (uniform value).
  // Eliminates shared variable and __syncthreads broadcast.
  constexpr int HIER_STRIDE = 16;
  int base = expert_id * HIER_STRIDE;
  int w2_expected;
  {
    constexpr int LAYER_IDX_SMEM_OFF =
        mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
        mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END;
    int layer_idx = *reinterpret_cast<int *>(&_fused_smem[LAYER_IDX_SMEM_OFF]);
    w2_expected = layer_idx + 1;
  }

  // Issue W2 weight buffer_load_lds BEFORE barrier poll — HBM loads fly
  // during barrier wait (~3us overlap instead of serial).
  // Single inline asm block to prevent compiler vmcnt serialization.
  {
    unsigned lds_w2_off = (unsigned)(uintptr_t)(lds_w2_base + warp_id * 1024);
    unsigned w2v[24], w2m[24];
#pragma unroll
    for (int t = 0; t < NUM_WAVES; t++) {
#pragma unroll
      for (int j = 0; j < W2_LPT; j++) {
        int idx = tid + j * 256;
        int clamped = idx < w2_n16_data ? idx : w2_n16_data - 1;
        w2v[t * W2_LPT + j] =
            w2_wg_voff_base +
            static_cast<uint32_t>(t * W2_TILE_ROWS * (W2_K / 2)) +
            static_cast<uint32_t>(clamped * 16);
        w2m[t * W2_LPT + j] = __builtin_amdgcn_readfirstlane(
            lds_w2_off + t * W2_TILE_BYTES + j * 4096);
      }
    }
    asm volatile("s_mov_b32 m0, %[m0]\n  buffer_load_dwordx4 %[v0],  %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m1]\n  buffer_load_dwordx4 %[v1],  %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m2]\n  buffer_load_dwordx4 %[v2],  %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m3]\n  buffer_load_dwordx4 %[v3],  %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m4]\n  buffer_load_dwordx4 %[v4],  %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m5]\n  buffer_load_dwordx4 %[v5],  %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m6]\n  buffer_load_dwordx4 %[v6],  %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m7]\n  buffer_load_dwordx4 %[v7],  %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m8]\n  buffer_load_dwordx4 %[v8],  %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m9]\n  buffer_load_dwordx4 %[v9],  %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m10]\n buffer_load_dwordx4 %[v10], %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m11]\n buffer_load_dwordx4 %[v11], %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m12]\n buffer_load_dwordx4 %[v12], %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m13]\n buffer_load_dwordx4 %[v13], %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m14]\n buffer_load_dwordx4 %[v14], %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m15]\n buffer_load_dwordx4 %[v15], %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m16]\n buffer_load_dwordx4 %[v16], %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m17]\n buffer_load_dwordx4 %[v17], %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m18]\n buffer_load_dwordx4 %[v18], %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m19]\n buffer_load_dwordx4 %[v19], %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m20]\n buffer_load_dwordx4 %[v20], %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m21]\n buffer_load_dwordx4 %[v21], %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m22]\n buffer_load_dwordx4 %[v22], %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 "s_mov_b32 m0, %[m23]\n buffer_load_dwordx4 %[v23], %[rsrc], "
                 "0 offen sc0 nt lds\n"
                 :
                 : [rsrc] "s"(w2_rsrc),
                   [v0] "v"(w2v[0]),
                   [v1] "v"(w2v[1]),
                   [v2] "v"(w2v[2]),
                   [v3] "v"(w2v[3]),
                   [v4] "v"(w2v[4]),
                   [v5] "v"(w2v[5]),
                   [v6] "v"(w2v[6]),
                   [v7] "v"(w2v[7]),
                   [v8] "v"(w2v[8]),
                   [v9] "v"(w2v[9]),
                   [v10] "v"(w2v[10]),
                   [v11] "v"(w2v[11]),
                   [v12] "v"(w2v[12]),
                   [v13] "v"(w2v[13]),
                   [v14] "v"(w2v[14]),
                   [v15] "v"(w2v[15]),
                   [v16] "v"(w2v[16]),
                   [v17] "v"(w2v[17]),
                   [v18] "v"(w2v[18]),
                   [v19] "v"(w2v[19]),
                   [v20] "v"(w2v[20]),
                   [v21] "v"(w2v[21]),
                   [v22] "v"(w2v[22]),
                   [v23] "v"(w2v[23]),
                   [m0] "s"(w2m[0]),
                   [m1] "s"(w2m[1]),
                   [m2] "s"(w2m[2]),
                   [m3] "s"(w2m[3]),
                   [m4] "s"(w2m[4]),
                   [m5] "s"(w2m[5]),
                   [m6] "s"(w2m[6]),
                   [m7] "s"(w2m[7]),
                   [m8] "s"(w2m[8]),
                   [m9] "s"(w2m[9]),
                   [m10] "s"(w2m[10]),
                   [m11] "s"(w2m[11]),
                   [m12] "s"(w2m[12]),
                   [m13] "s"(w2m[13]),
                   [m14] "s"(w2m[14]),
                   [m15] "s"(w2m[15]),
                   [m16] "s"(w2m[16]),
                   [m17] "s"(w2m[17]),
                   [m18] "s"(w2m[18]),
                   [m19] "s"(w2m[19]),
                   [m20] "s"(w2m[20]),
                   [m21] "s"(w2m[21]),
                   [m22] "s"(w2m[22]),
                   [m23] "s"(w2m[23])
                 : "memory", "m0");
  }

  // Issue scale loads concurrently with buffer_load_lds
  constexpr int W2_TOTAL_SC_DW4 = (W2_TILE_SCALE * NUM_WAVES) / 16;
  constexpr int W2_SC_LPT = (W2_TOTAL_SC_DW4 + 255) / 256;
  i32x4_t w2_sc_buf[W2_SC_LPT];
  {
    i32x4_t const *sc_src = (i32x4_t const *)wg_scales;
#pragma unroll
    for (int j = 0; j < W2_SC_LPT; j++) {
      int idx = tid + j * 256;
      if (idx < W2_TOTAL_SC_DW4) {
        w2_sc_buf[j] = sc_src[idx];
      }
    }
  }

  // All threads poll per-XCD release flag independently.
  // Eliminates tid==0 + __syncthreads — each thread confirms barrier itself.
  {
    int expected = w2_expected;
    while (ld_nt_s32(&d_barrier[base + xcd_id]) < expected) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  MOE_DBG_SUBPHASE(3002);

  // No buffer_inv needed — NT loads bypass L2 entirely.

  // FP8 quant of SwiGLU output — writes to LDS[0..W2_K+scales]
  // buffer_load_lds writes to LDS[W2_OFF..] — no conflict, both in flight.
  MOE_DBG_SUBPHASE(3003);
  {
    unsigned short const *w2_input_base =
        d_swiglu_out + tok_idx * (NUM_TOPK * INTERMEDIATE_SIZE) +
        topk_slot * INTERMEDIATE_SIZE;
    _gang_wave_parallel_fp8_quant_nt<W2_K>(
        w2_input_base, s_tok_fp8, s_tok_scales);
  }

  // Drain ALL pending HBM loads: buffer_load_lds (weight) + scale loads
  // Weight loads were issued before barrier poll, should be done by now.
  MOE_DBG_SUBPHASE(3004);
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  {
    constexpr int W2_SC_DW4_PER_TILE = W2_TILE_SCALE / 16;
#pragma unroll
    for (int j = 0; j < W2_SC_LPT; j++) {
      int idx = tid + j * 256;
      if (idx < W2_TOTAL_SC_DW4) {
        int tile = idx / W2_SC_DW4_PER_TILE;
        int off = idx % W2_SC_DW4_PER_TILE;
        i32x4_t *dst_sc = (i32x4_t *)(lds_w2_base + tile * W2_TILE_BYTES +
                                      W2_TILE_DATA_PADDED);
        dst_sc[off] = w2_sc_buf[j];
      }
    }
  }
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __syncthreads();

#if 0 // W2 timestamp disabled — near asm block
    g_subphase_scratch[6] = __builtin_amdgcn_s_memrealtime();
#endif
  MOE_DBG_SUBPHASE(3005);
  // LDS-based MFMA loop: weights already in LDS, compiler pipelines ds_reads.
  // Assembly shows lgkmcnt(7)/lgkmcnt(1) interleaving — much better than
  // the HBM path's vmcnt(0) stalls before every MFMA group.
  //
  // tile_iter=0 uses weights pre-loaded during FP8 quant. tile_iter=1
  // (when W2_TILES_PER_WAVE > 1, i.e. OPW=128) reloads weights into the
  // same LDS slots before its MFMA loop — matching the W13 dual-tile pattern.
  {
    constexpr int W2_TILE_ROWS_L = 16;
    constexpr int W2_TILE_DATA_L = W2_TILE_ROWS_L * (W2_K / 2);
    constexpr int W2_TILE_SCALE_L = W2_TILE_ROWS_L * W2_NUM_BLK32;
    constexpr int w2_n16_L = W2_TILE_DATA_L / 16;
    constexpr int W2_LPT_L = (w2_n16_L + 255) / 256;
    constexpr int W2_TILE_DATA_PADDED_L = W2_LPT_L * 256 * 16;
    constexpr int W2_TILE_BYTES_L = W2_TILE_DATA_PADDED_L + W2_TILE_SCALE_L;
    constexpr int LDS_W2_OFF_L = ((W2_K + W2_MFMA_ITERS + 15) / 16) * 16;
    uint8_t *lds_w2_base_l = (uint8_t *)_fused_smem + LDS_W2_OFF_L;

    // ── tile_iter=0: weights already in LDS from pre-load ──────────────
    {
      int wave_tile_0 = warp_id;

      uint8_t *lds_w2_data = lds_w2_base_l + warp_id * W2_TILE_BYTES_L;
      uint8_t *lds_w2_scales = lds_w2_data + W2_TILE_DATA_PADDED_L;

      int w_row_local = col;
      int const row_data_base = w_row_local * (W2_K / 2);
      int const row_scale_base = w_row_local * W2_NUM_BLK32;

      // Prefetch epilogue data before MFMA loop so loads fly during compute.
      int out_n_base = wg_idx * W2_OUTPUT_PER_WG + wave_tile_0 * 16 + g * 4;
      float pf_rw = 0.0f;
      uint2 pf_bias = {0, 0};
      if (col == 0 && out_n_base < W2_OUTPUT_SIZE) {
        float const *rw_ptr = &d_routing_weight[tok_idx * NUM_TOPK + topk_slot];
        unsigned short const *bias_ptr =
            &d_w2_bias[expert_id * W2_OUTPUT_SIZE + out_n_base];
        asm volatile("global_load_dword %0, %2, off\n"
                     "global_load_dwordx2 %1, %3, off"
                     : "=v"(pf_rw), "=v"(pf_bias)
                     : "v"(rw_ptr), "v"(bias_ptr)
                     : "memory");
      }
      asm volatile("" ::: "memory");

      f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

      // Pipelined W2 MFMA loop: overlap ds_reads with MFMA compute.
      // Same technique as W13 pipelined loop: issue next iteration's reads
      // into same registers before MFMA (MFMA reads old values at issue time).
      // Baseline: ~53 cycles/iter (20 wait + 32 MFMA + 1 overhead)
      // Pipelined: ~36 cycles/iter (0 wait + 32 MFMA + 4 overhead)
      {
        unsigned w2_w_addr =
            (unsigned)(uintptr_t)(lds_w2_data + row_data_base + g * 16);
        unsigned w2_ws_addr =
            (unsigned)(uintptr_t)(lds_w2_scales + row_scale_base + g);
        unsigned w2_t_addr = (unsigned)(uintptr_t)(s_tok_fp8 + g * 16);
        unsigned w2_ts_addr = (unsigned)(uintptr_t)(s_tok_scales);
        asm volatile(
            // Zero accumulator
            "v_accvgpr_write_b32 a0, 0\n"
            "v_accvgpr_write_b32 a1, 0\n"
            "v_accvgpr_write_b32 a2, 0\n"
            "v_accvgpr_write_b32 a3, 0\n"

            // Pre-issue 5 reads for iteration 0
            "ds_read_b128 v[22:25], %[wa]\n"           // weight A (16B FP4)
            "ds_read_u8   v7, %[wsa]\n"                // weight A scale
            "ds_read_b128 v[8:11], %[ta]\n"            // token B lo (16B FP8)
            "ds_read_b128 v[12:15], %[ta] offset:64\n" // token B hi (16B FP8)
            "ds_read_u8   v16, %[tsa]\n"               // token B scale
            "s_mov_b32 s13, 0\n"                       // loop counter

            // ── Iterations 0..W2_MFMA_ITERS-2: prefetch next, MFMA current ──
            "PIPELINED_W2_T0_%=:\n"
            "s_waitcnt lgkmcnt(0)\n" // iter N reads complete

            // Advance addresses for iter N+1
            "v_add_u32_e32 %[wa], 64, %[wa]\n"
            "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
            "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
            "s_add_i32 s13, s13, 1\n"

            // Token B scale → v17 FIRST (oldest in queue, completes first)
            "v_add_u32_e32 v17, s13, %[tsa]\n"
            "ds_read_u8   v17, v17\n" // [lgkmcnt +1] oldest
            // Prefetch iter N+1 data into SAME regs (MFMA reads old values)
            "ds_read_b128 v[22:25], %[wa]\n"           // [lgkmcnt +2]
            "ds_read_u8   v7, %[wsa]\n"                // [lgkmcnt +3]
            "ds_read_b128 v[8:11], %[ta]\n"            // [lgkmcnt +4]
            "ds_read_b128 v[12:15], %[ta] offset:64\n" // [lgkmcnt +5]

            // MFMA from iter N data (32 cycles, reads v[22:25] v[8:15] v7 v16)
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
            "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"

            // Copy next token scale during MFMA execution.
            // lgkmcnt(4): wait for token scale (oldest, issued first), leave 4
            // data reads flying
            "s_waitcnt lgkmcnt(4)\n"
            "v_mov_b32_e32 v16, v17\n"

            "s_cmpk_lt_i32 s13, %[iters_m1]\n"
            "s_cbranch_scc1 PIPELINED_W2_T0_%=\n"

            // ── Final iteration: no more prefetch needed ──
            "s_waitcnt lgkmcnt(0)\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
            "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"

            // Read accumulator into output
            "s_nop 7\n"
            "s_nop 0\n"
            "v_accvgpr_read_b32 %[acc0], a0\n"
            "v_accvgpr_read_b32 %[acc1], a1\n"
            "v_accvgpr_read_b32 %[acc2], a2\n"
            "v_accvgpr_read_b32 %[acc3], a3\n"
            : [acc0] "=v"(acc[0]),
              [acc1] "=v"(acc[1]),
              [acc2] "=v"(acc[2]),
              [acc3] "=v"(acc[3]),
              [wa] "+v"(w2_w_addr),
              [wsa] "+v"(w2_ws_addr),
              [ta] "+v"(w2_t_addr)
            : [tsa] "v"(w2_ts_addr), [iters_m1] "n"(W2_MFMA_ITERS - 1)
            : "memory",
              "s13",
              "v7",
              "v8",
              "v9",
              "v10",
              "v11",
              "v12",
              "v13",
              "v14",
              "v15",
              "v16",
              "v17",
              "v22",
              "v23",
              "v24",
              "v25",
              "a0",
              "a1",
              "a2",
              "a3");
      }

      MOE_DBG_SUBPHASE(3006);
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      if (col == 0 && out_n_base < W2_OUTPUT_SIZE) {
        unsigned bt0 = (pf_bias.x & 0xFFFFu) << 16;
        unsigned bt1 = pf_bias.x & 0xFFFF0000u;
        unsigned bt2 = (pf_bias.y & 0xFFFFu) << 16;
        unsigned bt3 = pf_bias.y & 0xFFFF0000u;
        float bv0, bv1, bv2, bv3;
        __builtin_memcpy(&bv0, &bt0, 4);
        __builtin_memcpy(&bv1, &bt1, 4);
        __builtin_memcpy(&bv2, &bt2, 4);
        __builtin_memcpy(&bv3, &bt3, 4);
        int ws_base = tok_idx * HIDDEN_SIZE + out_n_base;
        atomicAdd(&d_workspace_f32[ws_base + 0], (acc[0] + bv0) * pf_rw);
        atomicAdd(&d_workspace_f32[ws_base + 1], (acc[1] + bv1) * pf_rw);
        atomicAdd(&d_workspace_f32[ws_base + 2], (acc[2] + bv2) * pf_rw);
        if (out_n_base + 3 < W2_OUTPUT_SIZE) {
          atomicAdd(&d_workspace_f32[ws_base + 3], (acc[3] + bv3) * pf_rw);
        }
      }
    }

    // ── tile_iter=1: reload weights for tiles [NUM_WAVES..2*NUM_WAVES) ──
    if (W2_TILES_PER_WAVE > 1) {
      __syncthreads();
      // Reload W2 weights for second set of 4 tiles (same LDS slots, different
      // HBM offsets) Single inline asm block to prevent compiler vmcnt
      // serialization.
      {
        unsigned lds_w2t1_off =
            (unsigned)(uintptr_t)(lds_w2_base_l + warp_id * 1024);
        uint32_t w2t1_hbm_base =
            w2_wg_voff_base +
            static_cast<uint32_t>(NUM_WAVES * W2_TILE_ROWS * (W2_K / 2));
        unsigned w2t1v[24], w2t1m[24];
#pragma unroll
        for (int t = 0; t < NUM_WAVES; t++) {
#pragma unroll
          for (int j = 0; j < W2_LPT; j++) {
            int idx = tid + j * 256;
            int clamped = idx < w2_n16_data ? idx : w2_n16_data - 1;
            w2t1v[t * W2_LPT + j] =
                w2t1_hbm_base +
                static_cast<uint32_t>(t * W2_TILE_ROWS * (W2_K / 2)) +
                static_cast<uint32_t>(clamped * 16);
            w2t1m[t * W2_LPT + j] = __builtin_amdgcn_readfirstlane(
                lds_w2t1_off + t * W2_TILE_BYTES + j * 4096);
          }
        }
        asm volatile("s_mov_b32 m0, %[m0]\n  buffer_load_dwordx4 %[v0],  "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m1]\n  buffer_load_dwordx4 %[v1],  "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m2]\n  buffer_load_dwordx4 %[v2],  "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m3]\n  buffer_load_dwordx4 %[v3],  "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m4]\n  buffer_load_dwordx4 %[v4],  "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m5]\n  buffer_load_dwordx4 %[v5],  "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m6]\n  buffer_load_dwordx4 %[v6],  "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m7]\n  buffer_load_dwordx4 %[v7],  "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m8]\n  buffer_load_dwordx4 %[v8],  "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m9]\n  buffer_load_dwordx4 %[v9],  "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m10]\n buffer_load_dwordx4 %[v10], "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m11]\n buffer_load_dwordx4 %[v11], "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m12]\n buffer_load_dwordx4 %[v12], "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m13]\n buffer_load_dwordx4 %[v13], "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m14]\n buffer_load_dwordx4 %[v14], "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m15]\n buffer_load_dwordx4 %[v15], "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m16]\n buffer_load_dwordx4 %[v16], "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m17]\n buffer_load_dwordx4 %[v17], "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m18]\n buffer_load_dwordx4 %[v18], "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m19]\n buffer_load_dwordx4 %[v19], "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m20]\n buffer_load_dwordx4 %[v20], "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m21]\n buffer_load_dwordx4 %[v21], "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m22]\n buffer_load_dwordx4 %[v22], "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     "s_mov_b32 m0, %[m23]\n buffer_load_dwordx4 %[v23], "
                     "%[rsrc], 0 offen sc0 nt lds\n"
                     :
                     : [rsrc] "s"(w2_rsrc),
                       [v0] "v"(w2t1v[0]),
                       [v1] "v"(w2t1v[1]),
                       [v2] "v"(w2t1v[2]),
                       [v3] "v"(w2t1v[3]),
                       [v4] "v"(w2t1v[4]),
                       [v5] "v"(w2t1v[5]),
                       [v6] "v"(w2t1v[6]),
                       [v7] "v"(w2t1v[7]),
                       [v8] "v"(w2t1v[8]),
                       [v9] "v"(w2t1v[9]),
                       [v10] "v"(w2t1v[10]),
                       [v11] "v"(w2t1v[11]),
                       [v12] "v"(w2t1v[12]),
                       [v13] "v"(w2t1v[13]),
                       [v14] "v"(w2t1v[14]),
                       [v15] "v"(w2t1v[15]),
                       [v16] "v"(w2t1v[16]),
                       [v17] "v"(w2t1v[17]),
                       [v18] "v"(w2t1v[18]),
                       [v19] "v"(w2t1v[19]),
                       [v20] "v"(w2t1v[20]),
                       [v21] "v"(w2t1v[21]),
                       [v22] "v"(w2t1v[22]),
                       [v23] "v"(w2t1v[23]),
                       [m0] "s"(w2t1m[0]),
                       [m1] "s"(w2t1m[1]),
                       [m2] "s"(w2t1m[2]),
                       [m3] "s"(w2t1m[3]),
                       [m4] "s"(w2t1m[4]),
                       [m5] "s"(w2t1m[5]),
                       [m6] "s"(w2t1m[6]),
                       [m7] "s"(w2t1m[7]),
                       [m8] "s"(w2t1m[8]),
                       [m9] "s"(w2t1m[9]),
                       [m10] "s"(w2t1m[10]),
                       [m11] "s"(w2t1m[11]),
                       [m12] "s"(w2t1m[12]),
                       [m13] "s"(w2t1m[13]),
                       [m14] "s"(w2t1m[14]),
                       [m15] "s"(w2t1m[15]),
                       [m16] "s"(w2t1m[16]),
                       [m17] "s"(w2t1m[17]),
                       [m18] "s"(w2t1m[18]),
                       [m19] "s"(w2t1m[19]),
                       [m20] "s"(w2t1m[20]),
                       [m21] "s"(w2t1m[21]),
                       [m22] "s"(w2t1m[22]),
                       [m23] "s"(w2t1m[23])
                     : "memory", "m0");
      }

      // Drain buffer_load_lds writes
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

      // Issue scale loads for tile_iter=1
      constexpr int W2_SC_DW4_PER_TILE_L = W2_TILE_SCALE / 16;
      constexpr int W2_TOTAL_SC_DW4_L = (W2_TILE_SCALE * NUM_WAVES) / 16;
      constexpr int W2_SC_LPT_L = (W2_TOTAL_SC_DW4_L + 255) / 256;
      i32x4_t w2_sc_buf2[W2_SC_LPT_L];
      {
        i32x4_t const *sc_src =
            (i32x4_t const *)(wg_scales +
                              NUM_WAVES * W2_TILE_ROWS * W2_NUM_BLK32);
#pragma unroll
        for (int j = 0; j < W2_SC_LPT_L; j++) {
          int idx = tid + j * 256;
          if (idx < W2_TOTAL_SC_DW4_L) {
            w2_sc_buf2[j] = sc_src[idx];
          }
        }
      }

      // Drain scales, scatter to per-tile slots
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      {
#pragma unroll
        for (int j = 0; j < W2_SC_LPT_L; j++) {
          int idx = tid + j * 256;
          if (idx < W2_TOTAL_SC_DW4_L) {
            int tile = idx / W2_SC_DW4_PER_TILE_L;
            int off = idx % W2_SC_DW4_PER_TILE_L;
            i32x4_t *dst_sc =
                (i32x4_t *)(lds_w2_base_l + tile * W2_TILE_BYTES_L +
                            W2_TILE_DATA_PADDED_L);
            dst_sc[off] = w2_sc_buf2[j];
          }
        }
      }
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      __syncthreads();

      // tile_iter=1 MFMA loop — reads reloaded weights from LDS
      {
        int wave_tile_1 = warp_id + NUM_WAVES;

        uint8_t *lds_w2_data = lds_w2_base_l + warp_id * W2_TILE_BYTES_L;
        uint8_t *lds_w2_scales = lds_w2_data + W2_TILE_DATA_PADDED_L;

        int w_row_local = col;
        int const row_data_base = w_row_local * (W2_K / 2);
        int const row_scale_base = w_row_local * W2_NUM_BLK32;

        int out_n_base = wg_idx * W2_OUTPUT_PER_WG + wave_tile_1 * 16 + g * 4;
        float pf_rw = 0.0f;
        uint2 pf_bias = {0, 0};
        if (col == 0 && out_n_base < W2_OUTPUT_SIZE) {
          float const *rw_ptr =
              &d_routing_weight[tok_idx * NUM_TOPK + topk_slot];
          unsigned short const *bias_ptr =
              &d_w2_bias[expert_id * W2_OUTPUT_SIZE + out_n_base];
          asm volatile("global_load_dword %0, %2, off\n"
                       "global_load_dwordx2 %1, %3, off"
                       : "=v"(pf_rw), "=v"(pf_bias)
                       : "v"(rw_ptr), "v"(bias_ptr)
                       : "memory");
        }
        asm volatile("" ::: "memory");

        f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
        for (int ki = 0; ki < W2_MFMA_ITERS; ki++) {
          int kt = ki * K_PER_MFMA;
          i32x4_t a_lo =
              *(i32x4_t const *)(lds_w2_data + row_data_base + kt / 2 + g * 16);
          i32x8_t a = {};
          a[0] = a_lo[0];
          a[1] = a_lo[1];
          a[2] = a_lo[2];
          a[3] = a_lo[3];
          int sa = (int)lds_w2_scales[row_scale_base + kt / 32 + g];
          i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, kt, g);
          int sb = (int)s_tok_scales[ki];
          acc = _gang_mfma_f4xf8(a, b, acc, sa, sb);
        }

        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        if (col == 0 && out_n_base < W2_OUTPUT_SIZE) {
          unsigned bt0 = (pf_bias.x & 0xFFFFu) << 16;
          unsigned bt1 = pf_bias.x & 0xFFFF0000u;
          unsigned bt2 = (pf_bias.y & 0xFFFFu) << 16;
          unsigned bt3 = pf_bias.y & 0xFFFF0000u;
          float bv0, bv1, bv2, bv3;
          __builtin_memcpy(&bv0, &bt0, 4);
          __builtin_memcpy(&bv1, &bt1, 4);
          __builtin_memcpy(&bv2, &bt2, 4);
          __builtin_memcpy(&bv3, &bt3, 4);
          int ws_base = tok_idx * HIDDEN_SIZE + out_n_base;
          atomicAdd(&d_workspace_f32[ws_base + 0], (acc[0] + bv0) * pf_rw);
          atomicAdd(&d_workspace_f32[ws_base + 1], (acc[1] + bv1) * pf_rw);
          atomicAdd(&d_workspace_f32[ws_base + 2], (acc[2] + bv2) * pf_rw);
          if (out_n_base + 3 < W2_OUTPUT_SIZE) {
            atomicAdd(&d_workspace_f32[ws_base + 3], (acc[3] + bv3) * pf_rw);
          }
        }
      }
    }
  }

  __syncthreads();

#if 0 // W2 reporting disabled — timestamps not captured
    {
      g_subphase_scratch[7] = __builtin_amdgcn_s_memrealtime();
      if (is_w2 && tid == 0 && g_subphase_active) {
        atomicAdd(&g_subphase_ns[5][0], (g_subphase_scratch[1] - g_subphase_scratch[0]) * 10);
        atomicAdd(&g_subphase_ns[5][1], (g_subphase_scratch[6] - g_subphase_scratch[1]) * 10);
        atomicAdd(&g_subphase_ns[5][2], (g_subphase_scratch[7] - g_subphase_scratch[6]) * 10);
        atomicAdd(&g_subphase_cnt[5], 1ULL);
      }
    }
#endif

  // No barrier reset needed — all counters use monotonically increasing
  // expected values (per-XCD release = layer_idx + 1, global_arrive uses
  // modular check). Eliminates stale L2 issues across layers.
}

} // namespace kernel
