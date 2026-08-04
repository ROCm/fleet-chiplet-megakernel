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
// Barrier layout, per expert (see MOE_BAR_* below).
//
// Every slot gets its own 64-byte cache line. That is not padding for
// performance -- it is required for correctness. The release fan-out uses
// st_wt (sc0 sc1), which bypasses L2 and lands in HBM, while the arrival
// counter is an ordinary L2-resident atomic read-modify-write. If the two
// share a line, the L2 copy still holding the *old* release values can be
// written back over the fresh write-through data, silently reverting slots
// that were already released, and the W2 workers for that expert then wait
// forever on a release that did happen. The sibling barrier in
// gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh has always spaced its
// slots this way; this one packed all eight releases plus the counter into a
// single line, which is the deadlock captured at 32k.
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

#if defined(MPK_NIL_TRIPWIRE) && defined(MPK_TW_SUB)
// Resolve *where inside* the MoE kernel a worker was when it died.
//
// The fused layer's MPK_TW_SUB(80, moe_t) only resolves to "somewhere in the
// MoE tile", which spans W13, the per-expert W13->W2 barrier, and W2. The
// nil-address fault lands somewhere in there, and at that resolution a worker
// that faulted is indistinguishable from one merely parked at the barrier.
//
// aux carries the decoded tile identity, because every address this kernel
// computes is derived from it -- expert_id indexes the weight bases and the
// barrier, and the w13/w2 split decides which pointer set is live:
//   [15:0] global_tile  [23:16] expert_idx  [31:24] expert_id
//   [39:32] num_activated_experts  [40] is_w2
#define MOE_TW_AUX()                                                           \
  (((unsigned long long)(unsigned short)global_tile) |                         \
   (((unsigned long long)(unsigned char)expert_idx) << 16) |                   \
   (((unsigned long long)(unsigned char)expert_id) << 24) |                    \
   (((unsigned long long)(unsigned char)num_activated_experts) << 32) |        \
   (((unsigned long long)(is_w2 ? 1 : 0)) << 40))
#define MOE_DBG_SUBPHASE(code) MPK_TW_SUB((code), MOE_TW_AUX())
// Pre-decode marker: expert_id/is_w2 do not exist yet, so pass aux explicitly.
// Distinguishes a fault in the routing-mask read itself from one in the
// compute that follows it.
#define MOE_DBG_ENTRY(code, aux) MPK_TW_SUB((code), (aux))
#else
#define MOE_DBG_SUBPHASE(code) ((void)0)
#define MOE_DBG_ENTRY(code, aux) ((void)0)
#endif

namespace kernel {

// Per-expert MoE barrier geometry. One 64-byte line per slot (see the layout
// note at the top of this file): 8 per-XCD release flags then the arrival
// counter, so 9 lines used out of 10 reserved per expert.
//   [xcd * MOE_BAR_LINE]         per-XCD release flag  (st_wt, HBM)
//   [MOE_BAR_COUNTER_SLOT * ..]  global arrival count  (atomic, L2)
constexpr int MOE_BAR_LINE = 16;         // int32 per cache line
constexpr int MOE_BAR_COUNTER_SLOT = 8;  // line index of the arrival counter
constexpr int MOE_BAR_SLOTS = 10;        // lines reserved per expert
constexpr int MOE_BAR_STRIDE = MOE_BAR_SLOTS * MOE_BAR_LINE; // ints per expert

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
  // Marker 1000: about to read the routing mask. Everything downstream --
  // expert_id, the weight base pointers, the barrier slot -- derives from it.
  MOE_DBG_ENTRY(1000, (unsigned long long)tile_idx);
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
    MPK_WS_MARK(8100, global_tile); // exit: past end of tile range
    return;
  }

  bool is_w2 = (global_tile >= total_w13);
  int expert_idx, phase_tile;
  if (!is_w2) {
    expert_idx = global_tile / W13_TILES;
    phase_tile = global_tile % W13_TILES;
    // Padding tile: expert_idx beyond activated range → skip
    if (expert_idx >= num_activated_experts) {
      MPK_WS_MARK(8101, global_tile); // exit: W13 padding tile
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

  // Marker 1001: tile decoded, expert_id read. If num_activated_experts or
  // expert_id is out of range here, every pointer built below is wild --
  // this is the marker that separates "bad routing input" from "bad compute".
  MOE_DBG_ENTRY(1001,
                ((unsigned long long)(unsigned short)global_tile) |
                    (((unsigned long long)(unsigned char)expert_idx) << 16) |
                    (((unsigned long long)(unsigned char)
                          num_activated_experts)
                     << 32) |
                    (((unsigned long long)(is_w2 ? 1 : 0)) << 40));
  int expert_id = d_mask[expert_idx];
  int const *expert_routing = d_routing + expert_id * BATCH_SIZE;

  if (tok_idx >= BATCH_SIZE) {
    MPK_WS_MARK(8102, global_tile); // exit: token out of batch
    return;
  }

  int route_val = expert_routing[tok_idx];
  if (route_val == 0) {
    MPK_WS_MARK(8103, global_tile); // exit: token not routed here
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
    MPK_WS_MARK(8200, global_tile); // W13 compute
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
              // ── Two disjoint operand banks ──
              //   Bank 0: A v[22:25], A scale v7,  B v[8:15],  B scale v16
              //   Bank 1: A v[26:29], A scale v18, B v[32:39], B scale v19
              //   Address scratch v17, accumulator a[0:3].
              //
              // Prefetching into the registers the current MFMA reads is a WAR
              // race: lgkmcnt tracks when LDS data lands in the VGPR, not when
              // the MFMA finished sampling its operands, and a 16x16x128 MFMA
              // streams them over the op rather than latching at issue. When LDS
              // returns fast the write-back lands mid-MFMA and the op sees
              // mixed-iteration operands (~17-22% of launches before banking).
              // Ping-pong: while the MFMA consumes bank X, prefetch writes bank
              // 1-X, so no register is ever both a live source and an in-flight
              // LDS destination.
              //
              // Verified by tests/standalone/test_mfma_pipeline_hazards.hip.
              "v_accvgpr_write_b32 a0, 0\n"
              "v_accvgpr_write_b32 a1, 0\n"
              "v_accvgpr_write_b32 a2, 0\n"
              "v_accvgpr_write_b32 a3, 0\n"

              // Pre-issue 5 reads for iteration 0 into bank 0
              "ds_read_b128 v[22:25], %[wa]\n"
              "ds_read_u8   v7, %[wsa]\n"
              "ds_read_b128 v[8:11], %[ta]\n"
              "ds_read_b128 v[12:15], %[ta] offset:64\n"
              "ds_read_u8   v16, %[tsa]\n"
              "s_mov_b32 s13, 0\n"

              "PIPELINED_W13_T0_%=:\n"
              // ---- consume bank 0, prefetch into bank 1 ----
              "s_waitcnt lgkmcnt(0)\n"
              "v_add_u32_e32 %[wa], 64, %[wa]\n"
              "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
              "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
              "s_add_i32 s13, s13, 1\n"
              "v_add_u32_e32 v17, s13, %[tsa]\n"
              "ds_read_u8   v19, v17\n"
              "ds_read_b128 v[26:29], %[wa]\n"
              "ds_read_u8   v18, %[wsa]\n"
              "ds_read_b128 v[32:35], %[ta]\n"
              "ds_read_b128 v[36:39], %[ta] offset:64\n"
              "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
              "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
              "s_cmpk_lt_i32 s13, %[iters_m1]\n"
              "s_cbranch_scc0 W13_T0_TAIL_B1_%=\n"

              // ---- consume bank 1, prefetch into bank 0 ----
              "s_waitcnt lgkmcnt(0)\n"
              "v_add_u32_e32 %[wa], 64, %[wa]\n"
              "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
              "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
              "s_add_i32 s13, s13, 1\n"
              "v_add_u32_e32 v17, s13, %[tsa]\n"
              "ds_read_u8   v16, v17\n"
              "ds_read_b128 v[22:25], %[wa]\n"
              "ds_read_u8   v7, %[wsa]\n"
              "ds_read_b128 v[8:11], %[ta]\n"
              "ds_read_b128 v[12:15], %[ta] offset:64\n"
              "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], "
              "a[0:3], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
              "s_cmpk_lt_i32 s13, %[iters_m1]\n"
              "s_cbranch_scc1 PIPELINED_W13_T0_%=\n"

              // ── Final MFMA ──
              // Both tails are emitted because exit parity decides which bank
              // holds the final operands. Live MFMA_ITERS is 23 (odd), so the
              // loop falls out of the bank 1 half with the last operands in
              // BANK 0 -- this path. An even count exits via W13_T0_TAIL_B1
              // with them in bank 1. One tail alone would silently use the
              // wrong bank for one parity.
              "s_waitcnt lgkmcnt(0)\n"
              "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
              "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
              "s_branch W13_T0_ACC_%=\n"

              "W13_T0_TAIL_B1_%=:\n"
              "s_waitcnt lgkmcnt(0)\n"
              "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], "
              "a[0:3], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"

              "W13_T0_ACC_%=:\n"
              // 32 clocks: the scaled MFMA is a 32-cycle op on CDNA4. The old
              // "s_nop 7; s_nop 0" was 9 clocks (correct only for a 4-pass
              // MFMA) and returned a partially-retired accumulator every time.
              "s_nop 15\n"
              "s_nop 15\n"
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
                "v18",
                "v19",
                "v22",
                "v23",
                "v24",
                "v25",
                "v26",
                "v27",
                "v28",
                "v29",
                "v32",
                "v33",
                "v34",
                "v35",
                "v36",
                "v37",
                "v38",
                "v39",
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
                // ── Two disjoint operand banks ──
                //   Bank 0: A v[22:25], A scale v7,  B v[8:15],  B scale v16
                //   Bank 1: A v[26:29], A scale v18, B v[32:39], B scale v19
                //   Address scratch v17, accumulator a[0:3].
                //
                // Prefetching into the registers the current MFMA reads is a WAR
                // race: lgkmcnt tracks when LDS data lands in the VGPR, not when
                // the MFMA finished sampling its operands, and a 16x16x128 MFMA
                // streams them over the op rather than latching at issue. When LDS
                // returns fast the write-back lands mid-MFMA and the op sees
                // mixed-iteration operands (~17-22% of launches before banking).
                // Ping-pong: while the MFMA consumes bank X, prefetch writes bank
                // 1-X, so no register is ever both a live source and an in-flight
                // LDS destination.
                //
                // Verified by tests/standalone/test_mfma_pipeline_hazards.hip.
                "v_accvgpr_write_b32 a0, 0\n"
                "v_accvgpr_write_b32 a1, 0\n"
                "v_accvgpr_write_b32 a2, 0\n"
                "v_accvgpr_write_b32 a3, 0\n"

                // Pre-issue 5 reads for iteration 0 into bank 0
                "ds_read_b128 v[22:25], %[wa]\n"
                "ds_read_u8   v7, %[wsa]\n"
                "ds_read_b128 v[8:11], %[ta]\n"
                "ds_read_b128 v[12:15], %[ta] offset:64\n"
                "ds_read_u8   v16, %[tsa]\n"
                "s_mov_b32 s13, 0\n"

                "PIPELINED_W13_T1_%=:\n"
                // ---- consume bank 0, prefetch into bank 1 ----
                "s_waitcnt lgkmcnt(0)\n"
                "v_add_u32_e32 %[wa], 64, %[wa]\n"
                "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
                "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
                "s_add_i32 s13, s13, 1\n"
                "v_add_u32_e32 v17, s13, %[tsa]\n"
                "ds_read_u8   v19, v17\n"
                "ds_read_b128 v[26:29], %[wa]\n"
                "ds_read_u8   v18, %[wsa]\n"
                "ds_read_b128 v[32:35], %[ta]\n"
                "ds_read_b128 v[36:39], %[ta] offset:64\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
                "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
                "s_cmpk_lt_i32 s13, %[iters_m1]\n"
                "s_cbranch_scc0 W13_T1_TAIL_B1_%=\n"

                // ---- consume bank 1, prefetch into bank 0 ----
                "s_waitcnt lgkmcnt(0)\n"
                "v_add_u32_e32 %[wa], 64, %[wa]\n"
                "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
                "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
                "s_add_i32 s13, s13, 1\n"
                "v_add_u32_e32 v17, s13, %[tsa]\n"
                "ds_read_u8   v16, v17\n"
                "ds_read_b128 v[22:25], %[wa]\n"
                "ds_read_u8   v7, %[wsa]\n"
                "ds_read_b128 v[8:11], %[ta]\n"
                "ds_read_b128 v[12:15], %[ta] offset:64\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], "
                "a[0:3], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
                "s_cmpk_lt_i32 s13, %[iters_m1]\n"
                "s_cbranch_scc1 PIPELINED_W13_T1_%=\n"

                // ── Final MFMA ──
                // Both tails are emitted because exit parity decides which bank
                // holds the final operands. Live MFMA_ITERS is 23 (odd), so the
                // loop falls out of the bank 1 half with the last operands in
                // BANK 0 -- this path. An even count exits via W13_T1_TAIL_B1
                // with them in bank 1. One tail alone would silently use the
                // wrong bank for one parity.
                "s_waitcnt lgkmcnt(0)\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
                "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
                "s_branch W13_T1_ACC_%=\n"

                "W13_T1_TAIL_B1_%=:\n"
                "s_waitcnt lgkmcnt(0)\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], "
                "a[0:3], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"

                "W13_T1_ACC_%=:\n"
                // 32 clocks: the scaled MFMA is a 32-cycle op on CDNA4. The old
                // "s_nop 7; s_nop 0" was 9 clocks (correct only for a 4-pass
                // MFMA) and returned a partially-retired accumulator every time.
                "s_nop 15\n"
                "s_nop 15\n"
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
                  "v18",
                  "v19",
                  "v22",
                  "v23",
                  "v24",
                  "v25",
                  "v26",
                  "v27",
                  "v28",
                  "v29",
                  "v32",
                  "v33",
                  "v34",
                  "v35",
                  "v36",
                  "v37",
                  "v38",
                  "v39",
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
    MPK_WS_MARK(8201, global_tile); // W13 done, arriving at barrier
    if (tid == 0) {
      int base = expert_id * MOE_BAR_STRIDE;
      // Single global arrival (all W13 tiles increment one counter).
      //
      // The counter must NOT share a cache line with the release slots below.
      // The release fan-out uses st_wt (sc0 sc1), which bypasses L2 and writes
      // straight to HBM, while this atomic is an L2-resident read-modify-write
      // on the same 64-byte line. When both are in flight on the same line the
      // L2 copy -- still holding the *old* release values -- is written back
      // over the fresh write-through data, silently reverting slots that were
      // already released. The captured deadlock is exactly that: every producer
      // had arrived (arrivals % W13_TILES == 0, so the release did fire) yet
      // the per-XCD slots of one expert held *different* epochs, which is
      // impossible if the eight stores from one producer all survived.
      // COUNTER_OFF puts the counter on the next line.
      int prev_global = atom_add_release_gpu_s32(
          &d_barrier[base + MOE_BAR_COUNTER_SLOT * MOE_BAR_LINE], 1);
      if ((prev_global % W13_TILES) == W13_TILES - 1) {
        // Last W13 arrival: write per-XCD release = layer_idx + 1
        constexpr int LAYER_IDX_SMEM_OFF =
            mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
            mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END;
        int layer_idx =
            *reinterpret_cast<int *>(&_fused_smem[LAYER_IDX_SMEM_OFF]);
        int release_val = layer_idx + 1;
        for (int x = 0; x < 8; x++) {
          st_wt_u32((void *)&d_barrier[base + x * MOE_BAR_LINE],
                    (unsigned)release_val);
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
  MPK_WS_MARK(8300, global_tile); // W2 entry
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
  int base = expert_id * MOE_BAR_STRIDE;
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
    // Barrier id encodes the expert so the dump can tell which of the 4
    // activated experts never got its W13 release (see MPK_WS_WAIT_BEGIN).
    MPK_WS_WAIT_BEGIN(800 + expert_idx, expected);
    // Each wave clears its own bits in both masks before it starts spinning,
    // so what the dump reads describes this poll and not an earlier one. No
    // __syncthreads here on purpose -- this poll is deliberately divergent.
    MPK_WS_WAVE_CLEAR(warp_id);
    int _obs;
    int _spins = 0;
    while ((_obs = ld_nt_s32(&d_barrier[base + xcd_id * MOE_BAR_LINE])) <
           expected) {
      MPK_WS_WAIT_TICK(_obs, _spins);
      // Refresh the discriminating values on the same cadence as the tick:
      // the raw arrival counter (whether it sits on a multiple of W13_TILES
      // separates "release fired but was lost" from "arrivals never landed"),
      // and how many of the 8 per-XCD slots agree. All 8 are written by one
      // producer in one loop, so any spread means releases are being lost.
      if ((_spins & (MPK_WS_WAIT_REFRESH - 1)) == 0) {
        int _n_ok = 0, _mn = 0x7fffffff, _mx = -0x7fffffff;
        for (int _x = 0; _x < 8; _x++) {
          int _v = ld_nt_s32(&d_barrier[base + _x * MOE_BAR_LINE]);
          if (_v >= expected) {
            _n_ok++;
          }
          if (_v < _mn) {
            _mn = _v;
          }
          if (_v > _mx) {
            _mx = _v;
          }
        }
        // a3 is now the per-wave exit mask (MPK_WS_WAVE_EXIT), so fold _mn
        // into a2 instead of overwriting it.
        MPK_WS_WAIT_AUX(
            ld_nt_s32(&d_barrier[base + MOE_BAR_COUNTER_SLOT * MOE_BAR_LINE]),
            expert_id,
            _n_ok * 1000000 + (_mx - _mn),
            -1);
      }
      _spins++;
      __builtin_amdgcn_s_sleep(1);
    }
    // This wave's threads all cleared the release. Record it: the poll is
    // per-thread with no __syncthreads, so waves leave independently and a
    // block can be split across the barrier.
    MPK_WS_WAVE_EXIT(warp_id);
  }
  MOE_DBG_SUBPHASE(3002);
  MPK_WS_MARK(8302, global_tile); // W2: cleared W13->W2 barrier

  // No buffer_inv needed — NT loads bypass L2 entirely.

  // FP8 quant of SwiGLU output — writes to LDS[0..W2_K+scales]
  // buffer_load_lds writes to LDS[W2_OFF..] — no conflict, both in flight.
  MOE_DBG_SUBPHASE(3003);
  MPK_WS_MARK(8303, global_tile); // W2: FP8 quant of SwiGLU output
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
  MPK_WS_MARK(8304, global_tile); // W2: drain HBM loads
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
  MPK_WS_MARK(8305, global_tile); // W2: MFMA loop
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
                     : "=&v"(pf_rw), "=&v"(pf_bias)
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
            // ── Two disjoint operand banks ──
            //   Bank 0: A v[22:25], A scale v7,  B v[8:15],  B scale v16
            //   Bank 1: A v[26:29], A scale v18, B v[32:39], B scale v19
            //   Address scratch v17, accumulator a[0:3].
            //
            // Prefetching into the registers the current MFMA reads is a WAR
            // race: lgkmcnt tracks when LDS data lands in the VGPR, not when
            // the MFMA finished sampling its operands, and a 16x16x128 MFMA
            // streams them over the op rather than latching at issue. When LDS
            // returns fast the write-back lands mid-MFMA and the op sees
            // mixed-iteration operands (~17-22% of launches before banking).
            // Ping-pong: while the MFMA consumes bank X, prefetch writes bank
            // 1-X, so no register is ever both a live source and an in-flight
            // LDS destination.
            //
            // Verified by tests/standalone/test_mfma_pipeline_hazards.hip.
            "v_accvgpr_write_b32 a0, 0\n"
            "v_accvgpr_write_b32 a1, 0\n"
            "v_accvgpr_write_b32 a2, 0\n"
            "v_accvgpr_write_b32 a3, 0\n"

            // Pre-issue 5 reads for iteration 0 into bank 0
            "ds_read_b128 v[22:25], %[wa]\n"
            "ds_read_u8   v7, %[wsa]\n"
            "ds_read_b128 v[8:11], %[ta]\n"
            "ds_read_b128 v[12:15], %[ta] offset:64\n"
            "ds_read_u8   v16, %[tsa]\n"
            "s_mov_b32 s13, 0\n"

            "PIPELINED_W2_T0_%=:\n"
            // ---- consume bank 0, prefetch into bank 1 ----
            "s_waitcnt lgkmcnt(0)\n"
            "v_add_u32_e32 %[wa], 64, %[wa]\n"
            "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
            "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
            "s_add_i32 s13, s13, 1\n"
            "v_add_u32_e32 v17, s13, %[tsa]\n"
            "ds_read_u8   v19, v17\n"
            "ds_read_b128 v[26:29], %[wa]\n"
            "ds_read_u8   v18, %[wsa]\n"
            "ds_read_b128 v[32:35], %[ta]\n"
            "ds_read_b128 v[36:39], %[ta] offset:64\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
            "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
            "s_cmpk_lt_i32 s13, %[iters_m1]\n"
            "s_cbranch_scc0 W2_T0_TAIL_B1_%=\n"

            // ---- consume bank 1, prefetch into bank 0 ----
            "s_waitcnt lgkmcnt(0)\n"
            "v_add_u32_e32 %[wa], 64, %[wa]\n"
            "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
            "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
            "s_add_i32 s13, s13, 1\n"
            "v_add_u32_e32 v17, s13, %[tsa]\n"
            "ds_read_u8   v16, v17\n"
            "ds_read_b128 v[22:25], %[wa]\n"
            "ds_read_u8   v7, %[wsa]\n"
            "ds_read_b128 v[8:11], %[ta]\n"
            "ds_read_b128 v[12:15], %[ta] offset:64\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], "
            "a[0:3], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
            "s_cmpk_lt_i32 s13, %[iters_m1]\n"
            "s_cbranch_scc1 PIPELINED_W2_T0_%=\n"

            // ── Final MFMA ──
            // Both tails are emitted because exit parity decides which bank
            // holds the final operands. Live MFMA_ITERS is 23 (odd), so the
            // loop falls out of the bank 1 half with the last operands in
            // BANK 0 -- this path. An even count exits via W2_T0_TAIL_B1
            // with them in bank 1. One tail alone would silently use the
            // wrong bank for one parity.
            "s_waitcnt lgkmcnt(0)\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
            "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
            "s_branch W2_T0_ACC_%=\n"

            "W2_T0_TAIL_B1_%=:\n"
            "s_waitcnt lgkmcnt(0)\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], "
            "a[0:3], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"

            "W2_T0_ACC_%=:\n"
            // 32 clocks: the scaled MFMA is a 32-cycle op on CDNA4. The old
            // "s_nop 7; s_nop 0" was 9 clocks (correct only for a 4-pass
            // MFMA) and returned a partially-retired accumulator every time.
            "s_nop 15\n"
            "s_nop 15\n"
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
              "v18",
              "v19",
              "v22",
              "v23",
              "v24",
              "v25",
              "v26",
              "v27",
              "v28",
              "v29",
              "v32",
              "v33",
              "v34",
              "v35",
              "v36",
              "v37",
              "v38",
              "v39",
              "a0",
              "a1",
              "a2",
              "a3");
      }

      MOE_DBG_SUBPHASE(3006);
      MPK_WS_MARK(8306, global_tile); // W2: epilogue
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
                       : "=&v"(pf_rw), "=&v"(pf_bias)
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
