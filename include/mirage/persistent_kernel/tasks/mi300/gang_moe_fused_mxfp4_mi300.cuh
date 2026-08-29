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
#include "tasks/mi300/moe_ws_layout.cuh" // MOE_WS_SLOTS, moe_ws_offset()
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
constexpr int MOE_BAR_LINE = 16;        // int32 per cache line
constexpr int MOE_BAR_COUNTER_SLOT = 8; // line index of the arrival counter
constexpr int MOE_BAR_SLOTS = 10;       // lines reserved per expert
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

  // ── N-axis packing geometry ───────────────────────────────────────────────
  // Token `c` lives at LDS row `c` and feeds N column `c` of the 16x16x128
  // MFMA, so an expert's whole routed token set costs one tile sweep instead
  // of one per token.
  //
  // Batches wider than one MFMA N-tile keep the old token-per-tile decode. A
  // NUM_BBLK-blocked compaction would leave some (expert, block) tiles with no
  // live token, and an empty tile still has to arrive at the W13->W2 barrier
  // or the `% W13_TILES` modulus below stops being exact -- which is the bs>1
  // hang this packing exists to fix.
  constexpr int MFMA_N = 16;
  constexpr bool PACK_N = BATCH_SIZE <= MFMA_N;
  constexpr int TOK_ROWS = PACK_N ? BATCH_SIZE : 1;
  // Distinguishes "one staged row because the batch is one" (fold token index
  // to a literal 0) from "one staged row because we fell back to the legacy
  // per-token decode" (token index comes from the tile).
  constexpr bool SINGLE_TOK = PACK_N && BATCH_SIZE == 1;

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
  // Tile space is weight groups only under packing: the token axis moved into
  // the MFMA's N dimension. This is also what makes the barrier modulus at the
  // bottom of Phase 0 exact again -- see the note there.
  constexpr int W13_TILES = PACK_N ? W13_WGS : BATCH_SIZE * W13_WGS;

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
  constexpr int W2_TILES = PACK_N ? W2_WGS : BATCH_SIZE * W2_WGS;

#ifdef MPK_MFMA_PINGPONG_SCHED
  // The scheduled MFMA loops emit MFMAs in pairs plus one tail, so they can
  // only express an odd trip count. Both are 23 (2944/128) for GPT-OSS 120B.
  // An even count would silently compute one extra MFMA -- reading past the
  // end of the K reduction -- so fail the build instead.
  static_assert(W13_MFMA_ITERS % 2 == 1,
                "MPK_MFMA_PINGPONG_SCHED requires an odd W13_MFMA_ITERS: the "
                "scheduled loop emits 2*trips+1 MFMAs and has a single "
                "bank-0 tail.");
  static_assert(W2_MFMA_ITERS % 2 == 1,
                "MPK_MFMA_PINGPONG_SCHED requires an odd W2_MFMA_ITERS: the "
                "scheduled loop emits 2*trips+1 MFMAs and has a single "
                "bank-0 tail.");
#endif

// ── Assembler-unrolled T0 MFMA loops ────────────────────────────────────────
//
// These unroll the *unscheduled* ping-pong arm -- the one that runs when
// MPK_MFMA_PINGPONG_SCHED is off, which is the live configuration. The
// scheduled arm has a different body (hazard padding, a top-of-loop test and a
// single tail), so a flag set on top of it would expand nothing and read as a
// free no-op in a measurement. Fail the build instead of measuring a lie.
#if defined(MPK_MFMA_PINGPONG_SCHED) &&                                        \
    (defined(MPK_W13_T0_MFMA_UNROLLED) || defined(MPK_W2_T0_MFMA_UNROLLED))
#error "MPK_W13_T0_MFMA_UNROLLED / MPK_W2_T0_MFMA_UNROLLED unroll the "        \
       "unscheduled ping-pong arm and have no effect under "                   \
       "MPK_MFMA_PINGPONG_SCHED. Drop one of the two."
#endif
#ifdef MPK_W13_T0_MFMA_UNROLLED
  // .rept emits (W13_MFMA_ITERS-1)/2 bank pairs and exactly one bank-0 tail,
  // which only covers W13_MFMA_ITERS when the count is odd. At an even count
  // the last K block's operands land in bank 1 and the emitted tail would
  // re-run block W13_MFMA_ITERS-2 out of bank 0 -- silently wrong rather than a
  // build error, so catch it here. The branching arm handles both parities.
  static_assert(W13_MFMA_ITERS % 2 == 1,
                "MPK_W13_T0_MFMA_UNROLLED assumes an odd W13_MFMA_ITERS "
                "(W13_K / 128); at an even count use the default branching "
                "loop, which emits both tails");
#endif
#ifdef MPK_W2_T0_MFMA_UNROLLED
  static_assert(W2_MFMA_ITERS % 2 == 1,
                "MPK_W2_T0_MFMA_UNROLLED assumes an odd W2_MFMA_ITERS "
                "(W2_K / 128); at an even count use the default branching "
                "loop, which emits both tails");
#endif

// ── Multi-chain MFMA accumulation ───────────────────────────────────────────
//
// A scaled MFMA is 32 cycles on CDNA4, and consecutive ops accumulating into
// the same a[0:3] carry a SrcC RAW dependency that forces 32 states of
// separation. MPK_MFMA_PINGPONG_SCHED *fills* that gap with `s_nop 15;
// s_nop 10`. These flags *remove* it: the K reduction is split across
// independent accumulator chains, so adjacent MFMAs target disjoint SrcC and
// the cadence drops to 8 states. Each chain is then reused after 16 states,
// which same-opcode SrcC forwarding permits.
//
// That 16-state reuse is not architecturally documented. It is verified on
// hardware by tests/standalone/test_mfma_pipeline_hazards.hip, which checks
// both schedules bit-exactly against order-matched references (the chains
// reassociate the FP32 sum, so a single-chain reference would report
// rounding as a hazard).
//
// The unscheduled arm -- the live one -- has no `s_nop` to delete, but it does
// not escape the dependency: its two MFMAs sit about eleven issue slots apart
// (one wait, four address updates, five ds_reads), so the hardware supplies the
// remaining ~21 states itself. Splitting the chains removes that stall too, and
// the only thing the flag needs from the schedule is that consecutive MFMAs
// alternate banks -- which both arms already do. So this is gated on the W13
// body existing in *some* two-bank form, not on the padding being present.
#if defined(MPK_MOE_DUAL_ACCUMULATOR)
#define MPK_W13_DUAL_ACC 1
  // Unchanged from the single-chain loop: pairs plus one chain-0 tail.
  static_assert(W13_MFMA_ITERS % 2 == 1,
                "MPK_MOE_DUAL_ACCUMULATOR requires an odd W13_MFMA_ITERS.");
#endif

#if defined(MPK_MFMA_PINGPONG_SCHED) && defined(MPK_MOE_QUAD_ACCUMULATOR)
#define MPK_W2_QUAD_ACC 1
// One ping-pong pair of the quad-chain W2 body: CH_A consumes bank 0, CH_B
// bank 1. Both edges are between disjoint chains, so each is 8 states (5
// issued address updates + s_nop 2, then s_nop 3 + the 4 updates that open
// the next pair) instead of the 32 a shared accumulator would force.
#define MPK_W2_QUAD_PAIR(CH_A, CH_B)                                           \
  "v_add_u32_e32 %[wa], 64, %[wa]\n"                                           \
  "v_add_u32_e32 %[wsa], 4, %[wsa]\n"                                          \
  "v_add_u32_e32 %[ta], 0x80, %[ta]\n"                                         \
  "s_add_i32 s13, s13, 1\n"                                                    \
  "v_add_u32_e32 v17, s13, %[tsa]\n"                                           \
  "ds_read_u8   v19, v17\n"                                                    \
  "ds_read_b128 v[26:29], %[wa]\n"                                             \
  "ds_read_u8   v18, %[wsa]\n"                                                 \
  "ds_read_b128 v[32:35], %[ta]\n"                                             \
  "ds_read_b128 v[36:39], %[ta] offset:64\n"                                   \
  "v_mfma_scale_f32_16x16x128_f8f6f4 " CH_A ", v[22:25], v[8:15], " CH_A       \
  ", v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"                                       \
  "v_add_u32_e32 %[wa], 64, %[wa]\n"                                           \
  "v_add_u32_e32 %[wsa], 4, %[wsa]\n"                                          \
  "v_add_u32_e32 %[ta], 0x80, %[ta]\n"                                         \
  "s_add_i32 s13, s13, 1\n"                                                    \
  "v_add_u32_e32 v17, s13, %[tsa]\n"                                           \
  "s_nop 2\n"                                                                  \
  "s_waitcnt lgkmcnt(0)\n"                                                     \
  "v_mfma_scale_f32_16x16x128_f8f6f4 " CH_B ", v[26:29], v[32:39], " CH_B      \
  ", v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"                                      \
  "ds_read_u8   v16, v17\n"                                                    \
  "ds_read_b128 v[22:25], %[wa]\n"                                             \
  "ds_read_u8   v7, %[wsa]\n"                                                  \
  "ds_read_b128 v[8:11], %[ta]\n"                                              \
  "ds_read_b128 v[12:15], %[ta] offset:64\n"                                   \
  "s_nop 3\n"                                                                  \
  "s_waitcnt lgkmcnt(0)\n"
  // The quad body consumes four K blocks per trip and leaves a three-MFMA
  // tail on chains 0/1/2, so K block k lands on chain k % 4 throughout.
  static_assert(W2_MFMA_ITERS % 4 == 3,
                "MPK_MOE_QUAD_ACCUMULATOR requires W2_MFMA_ITERS % 4 == 3: "
                "the scheduled loop emits 4*trips + 3 MFMAs.");
#endif

// ── W2 epilogue overlap ─────────────────────────────────────────────────────
//
// The `s_nop 15; s_nop 15` after the last W2 MFMA exists for Hazard 2: an
// AccVGPR read must not beat the eight-pass MFMA to retirement. Under the quad
// schedule three of the four chains matured several MFMAs earlier, so that
// wait can be *spent* instead of idled -- read and merge the mature chains
// first, and touch the final MFMA's chain last.
//
// The final MFMA of the tail writes a[8:11]. Reading a12..a15, a0..a3, a4..a7
// and issuing the eight merge adds puts 20 instructions between that MFMA and
// the first read of a8 -- above the 11-clock retirement threshold measured on
// gfx950 (fleet pads to 32 to stay clear of it). The three chains read early
// are each well past their own last write: a[4:7] by ~18 states, a[0:3] by
// ~24, a[12:15] by an entire loop trip.
//
// The merge order is unchanged from the padded arm (c0, c1, c3, c2), so this
// flag is a pure scheduling change -- it must not move a single ULP.
#if defined(MPK_W2_QUAD_ACC) && defined(MPK_MOE_W2_EPILOGUE_OVERLAP)
#define MPK_W2_EPI_OVERLAP 1
#elif defined(MPK_MOE_W2_EPILOGUE_OVERLAP)
#error                                                                         \
    "MPK_MOE_W2_EPILOGUE_OVERLAP needs MPK_MOE_QUAD_ACCUMULATOR (+ MPK_MFMA_PINGPONG_SCHED): with one accumulator chain there is no mature chain to read during the wait."
#endif

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

#ifdef MPK_MOE_INNER_TIMING
  // Entry stamp, before the tile decode. The decode is not free and is not
  // covered by _mt0: it is two *dependent* loads that both miss (d_mask, then
  // d_routing indexed by what d_mask returned), and every worker pays it on
  // every tile in both arms. Reported as `dec` on each arm's line.
  unsigned long long _mtE = __builtin_amdgcn_s_memrealtime();
#endif

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

  // ── Tile decode (phase-ordered: padded W13 then all W2) ─────────────────
  // Pad total_w13 to the next multiple of 240 so every worker's first tile is
  // a W13 tile. This eliminates compute imbalance where W2 workers would
  // otherwise start polling the barrier immediately while W13 is still
  // running (~7.8 us wasted per layer).
  // Padding tiles (expert_idx >= num_activated) early-return in ~0 cycles.
  // A "round" is workers_per_xcd * 8, so 240 is exact only at 30 workers/XCD.
  // The shipped default is 31, where the round is 248 and a 240-tile W13 space
  // leaves 8 slots of the first round to be filled by W2 tiles -- precisely
  // the imbalance the padding was added to remove: those 8 workers start
  // polling the W13->W2 barrier before W13 has run. Correcting it to 248
  // (MPK_MOE_PAD_ROUND=1) is worth ~0.01 ms and flakes the generated text one
  // run in four, so 240 stays; see the host-side note for why.
  // MPK_MOE_PAD_MULTIPLE is set from the same env var on the host, so the two
  // sides cannot drift (a drift here does not merely mistune -- the `%
  // W13_TILES` release stops firing and the W2 workers spin forever).
#ifdef MPK_MOE_PAD_MULTIPLE
  constexpr int PAD_MULTIPLE = MPK_MOE_PAD_MULTIPLE;
#else
  constexpr int PAD_MULTIPLE =
      240; // Required: see PAD_MULTIPLE investigation in memory
#endif

  int xcd_id = _gang_moe_get_xcd_id();
  // Marker 1000: about to read the routing mask. Everything downstream --
  // expert_id, the weight base pointers, the barrier slot -- derives from it.
  MOE_DBG_ENTRY(1000, (unsigned long long)tile_idx);
  // Read for diagnostics only. Nothing that decides tile partitioning or
  // padding may derive from this word -- see the padding note below.
  int num_activated_experts = d_mask[NUM_EXPERTS];
#ifdef MPK_MOE_SINGLE_EXPERT
  constexpr bool MPK_MOE_SINGLE_EXPERT_ACTIVE = true;
  num_activated_experts = min(num_activated_experts, 1);
#else
  constexpr bool MPK_MOE_SINGLE_EXPERT_ACTIVE = false;
#endif

  // ── Tile space is compile-time, not routing-dependent ─────────────────────
  // MAX_ACTIVATED must equal the host's `max_activated` in
  // persistent_kernel.py, which is what sizes moe_total_tiles_per_xcd.
  //
  // The partitioning below (total_w13, the is_w2 split, expert_idx) MUST NOT
  // depend on num_activated_experts. There is one moe_mask buffer shared by
  // all 36 layers and no barrier at the layer boundary, so workers straddle
  // layers -- a worker still finishing layer L can read layer L+1's mask.
  // While the count was the compile-time constant k on every layer that was
  // harmless: a stale read gave the same partitioning and only shifted which
  // expert ids were used. Once the count became a true per-layer union over
  // routed tokens (needed for bs>1 dedup) it varies, two workers computing
  // this arithmetic from different counts disagree about which tiles are W13,
  // and the `% W13_TILES` release below never fires -- the W2 workers then
  // spin forever. Deriving it from a constant makes every worker agree by
  // construction, whichever layer's mask it happened to read.
  //
  // The runtime count is still used, but only to skip padding tiles, where
  // reading a stale value costs at worst a wasted or skipped expert-slot on
  // one layer -- never a partitioning disagreement.
  constexpr int MAX_ACTIVATED =
      (NUM_TOPK * BATCH_SIZE < NUM_EXPERTS) ? NUM_TOPK * BATCH_SIZE
                                            : NUM_EXPERTS;

  // The output workspace is indexed by topk slot, so its slot count must be
  // this model's experts-per-token. If they disagree, slot writes alias across
  // tokens (too few) or the consumer sums uninitialized slabs (too many) --
  // both silent. Consumers derive the same stride from MOE_WS_SLOTS.
  static_assert(NUM_TOPK == MOE_WS_SLOTS,
                "MOE_WS_SLOTS in moe_ws_layout.cuh must equal NUM_TOPK");

  int global_tile = tile_idx * 8 + xcd_id;
  constexpr int TOTAL_W13_REAL = MAX_ACTIVATED * W13_TILES;
  constexpr int TOTAL_W13 =
      ((TOTAL_W13_REAL + PAD_MULTIPLE - 1) / PAD_MULTIPLE) * PAD_MULTIPLE;
  constexpr int TOTAL_W2 = MAX_ACTIVATED * W2_TILES;
  constexpr int TOTAL_TILES = TOTAL_W13 + TOTAL_W2;
  if (global_tile >= TOTAL_TILES) {
    MPK_WS_MARK(8100, global_tile); // exit: past end of tile range
    return;
  }

  // ── AID-invariant workgroup map (opt-in) ──────────────────────────────────
  // Which AID reads weight workgroup w is, by default, a function of the
  // expert's *position* in the activated list, not of w. Tile e*W13_TILES + p
  // runs on XCD (e*W13_TILES + p) % 8, and W13_TILES is 46 here, so each
  // expert's block is rotated by a further 46 % 8 == 6 XCDs. A workgroup
  // therefore changes AID from token to token and there is no AID its pages
  // could usefully be homed on.
  //
  // Re-deal the same tiles so that the AID is fixed instead. Both phase
  // regions are already whole multiples of 8 tiles (W13 is padded up to
  // PAD_MULTIPLE, W2 is MAX_ACTIVATED * W2_TILES), so each splits exactly
  // half-and-half across the two AIDs, and each half has room for
  // MAX_ACTIVATED * (TILES/2) workgroups -- W2 to the tile, W13 inside the
  // padding it already carries. Enumerate one AID's slots as s, hand them out
  // in runs of TILES/2, and workgroup w < TILES/2 sits on AID 0 for every
  // expert, always.
  //
  // This is a pure re-interpretation of the tile index. The tile -> XCD
  // mapping, the tile count, and the arrival count per slot (TILES/2 per AID,
  // TILES in total) are all identical to the default, so scheduling, load
  // balance and the `% W13_TILES` release below are untouched. On its own it
  // moves no byte to a different HBM stack -- it is what makes homing the
  // weight pages meaningful.
  constexpr bool AID_MAP_OK =
      PACK_N && (W13_TILES % 2 == 0) && (W2_TILES % 2 == 0) &&
      (TOTAL_W13 % 8 == 0) && (TOTAL_W2 % 8 == 0) &&
      (TOTAL_W13 / 2 >= MAX_ACTIVATED * (W13_TILES / 2)) &&
      (TOTAL_W2 / 2 >= MAX_ACTIVATED * (W2_TILES / 2));
#ifdef MPK_MOE_AID_INVARIANT_MAP
  static_assert(AID_MAP_OK,
                "MPK_MOE_AID_INVARIANT_MAP needs N-packed tiles, an even "
                "workgroup count in both phases, both phase regions a whole "
                "multiple of 8 tiles, and room for MAX_ACTIVATED*(TILES/2) "
                "workgroups in each AID half.");
  constexpr bool USE_AID_MAP = true;
#else
  constexpr bool USE_AID_MAP = false;
#endif

  bool is_w2 = (global_tile >= TOTAL_W13);
  int expert_idx, phase_tile;
  if constexpr (USE_AID_MAP) {
    // Offset within the phase region. TOTAL_W13 % 8 == 0 is what keeps
    // t % 8 equal to the XCD id, so the AID bit survives the rebase.
    int t = is_w2 ? (global_tile - TOTAL_W13) : global_tile;
    int aid = (t >> 2) & 1;
    int s = (t >> 3) * 4 + (t & 3); // slot index within this AID
    if (!is_w2) {
      constexpr int HALF = W13_TILES / 2;
      expert_idx = s / HALF;
      phase_tile = aid * HALF + (s % HALF);
    } else {
      constexpr int HALF = W2_TILES / 2;
      expert_idx = s / HALF;
      phase_tile = aid * HALF + (s % HALF);
    }
  } else if (!is_w2) {
    expert_idx = global_tile / W13_TILES;
    phase_tile = global_tile % W13_TILES;
  } else {
    int w2_tile = global_tile - TOTAL_W13;
    expert_idx = w2_tile / W2_TILES;
    phase_tile = w2_tile % W2_TILES;
  }

  // ── Nothing read from the mask may steer control flow ─────────────────────
  // There is one moe_mask buffer for all 36 layers and no layer-boundary
  // barrier, so workers straddle layers and two of them can read *different
  // layers' masks* for the same slot. That rules out deciding anything
  // structural from the mask -- not the tile partitioning, not whether a slot
  // is padding, and not the barrier address:
  //
  //   * count-derived padding (`expert_idx >= num_activated_experts`): the two
  //     sides disagree about which tiles are W13, one returns without arriving,
  //     the counter stops short, the `% W13_TILES` release never fires.
  //     Observed as bid=805 with arrivals%46=41 -- exactly the 5 tiles skipped.
  //   * slot-derived padding (a -1 sentinel at [count, NUM_EXPERTS)): better,
  //     but slot s still reads live in a layer with a larger count and padding
  //     in one with a smaller count, so the same disagreement returns.
  //   * `base = expert_id * MOE_BAR_STRIDE`: the address itself came from mask
  //     *contents*, so slot s resolving to expert 110 for one worker and
  //     another id for its partner split the arrivals across two counters.
  //     Observed as obs=0 with arrivals=43 -- 3 tiles landed elsewhere.
  //
  // So: the tile space is compile-time (above), the barrier is indexed by
  // *slot* (below), and padding tiles do not return -- they fall through with
  // no active token, run their MFMA over the clamped row, store nothing, and
  // arrive like every other tile. Every slot in [0, MAX_ACTIVATED) therefore
  // arrives exactly W13_TILES times every layer no matter what the mask says,
  // which is what makes the release unconditional.
  //
  // The mask is still read, for expert_id -- but only to pick weights. A stale
  // read there is the benign numeric drift the mask always had (a token gets a
  // neighbouring layer's expert), not a deadlock.
  int expert_id_raw = d_mask[expert_idx];
  bool const is_padding_slot =
      (expert_id_raw < 0) || (expert_id_raw >= NUM_EXPERTS) ||
      (MPK_MOE_SINGLE_EXPERT_ACTIVE && expert_idx >= 1);
  // Clamp before it reaches any pointer arithmetic: the sentinel is -1, and the
  // weight/routing base pointers are built unconditionally below.
  int expert_id = is_padding_slot ? 0 : expert_id_raw;
  if (is_padding_slot) {
    MPK_WS_MARK(is_w2 ? 8105 : 8101, global_tile); // padding tile (runs empty)
  }

  int n_wgs = is_w2 ? W2_WGS : W13_WGS;
  // Under packing the tile space *is* the weight-group space, so the divide
  // folds away and every tile covers all of the expert's routed tokens.
  int tok_idx = PACK_N ? 0 : phase_tile / n_wgs;
  int wg_idx = PACK_N ? phase_tile : phase_tile % n_wgs;

  // Marker 1001: tile decoded, expert_id read. If num_activated_experts or
  // expert_id is out of range here, every pointer built below is wild --
  // this is the marker that separates "bad routing input" from "bad compute".
  MOE_DBG_ENTRY(
      1001,
      ((unsigned long long)(unsigned short)global_tile) |
          (((unsigned long long)(unsigned char)expert_idx) << 16) |
          (((unsigned long long)(unsigned char)num_activated_experts) << 32) |
          (((unsigned long long)(is_w2 ? 1 : 0)) << 40));
  int const *expert_routing = d_routing + expert_id * BATCH_SIZE;

  if (!PACK_N && tok_idx >= BATCH_SIZE) {
    MPK_WS_MARK(8102, global_tile); // exit: token out of batch
    return;
  }

  // ── Per-expert token compaction ───────────────────────────────────────────
  // Which of the batch's tokens routed to this expert, packed into MFMA N
  // columns 0..n_tok-1. This is what replaces the 16 separate tile launches
  // (one per token, 15 of which used to early-return) with 16 scalar loads.
  //
  // The early return those launches took is exactly the bs>1 hang: a tile that
  // returns above never reaches the atom_add_release_gpu_s32 at the end of
  // Phase 0, so `prev_global % W13_TILES == W13_TILES - 1` never fires and the
  // W2 workers spin forever. Nothing below this point may return early.
  __shared__ int s_tok_of_col[PACK_N ? MFMA_N : 1];
  __shared__ int s_slot_of_col[PACK_N ? MFMA_N : 1];
  __shared__ int s_row_off[PACK_N ? MFMA_N : 1];
  __shared__ int s_n_tok;

  int my_tok, topk_slot, n_tok;
  bool tok_active;

  if constexpr (SINGLE_TOK) {
    // BATCH_SIZE == 1: there is one token, it is token 0, and an expert only
    // appears in the activated list because that token routed to it. Keep the
    // original scalar path -- no table, no extra barrier, and the token index
    // stays a compile-time literal so every address derived from it is
    // uniform. (Packing a one-row batch would cost a __syncthreads for
    // nothing; that class of drift is what the ISA gate exists to catch.)
    int route_val = expert_routing[0];
    if (route_val == 0) {
      MPK_WS_MARK(8103, global_tile); // exit: token not routed here
      return;
    }
    my_tok = 0;
    topk_slot = route_val - 1;
    n_tok = 1;
    tok_active = (col == 0);
  } else if constexpr (PACK_N) {
    if (warp_id == 0) {
      // Clear then compact, both from wave 0: LDS ops from one wave retire in
      // program order, so the pad entries are overwritten by the compaction
      // and never read stale. (They are also never read -- inactive lanes and
      // the quantizer's row clamp both land on slot 0 -- but leaving the table
      // undefined would make that an invariant nobody states.)
      bool const in_range = lane_id < MFMA_N;
      if (in_range) {
        s_tok_of_col[lane_id] = 0;
        s_slot_of_col[lane_id] = 0;
        s_row_off[lane_id] = 0;
      }

      // The ballot runs on the whole wave, not inside the lane_id < 16 branch:
      // it reports only currently-active lanes, so issuing it under divergence
      // would make the prefix count depend on which lanes happen to be on.
      // A padding slot contributes no tokens. Forced to 0 rather than skipped:
      // expert_id was clamped to 0 for padding, so expert_routing points at a
      // real expert's row and would otherwise compact that expert's tokens into
      // a tile that must produce nothing. The ballot below still runs on the
      // whole wave, so n_tok comes out 0 and every lane is inactive -- the tile
      // does its MFMA over the clamped row, stores nothing, and arrives.
      int rv = (in_range && lane_id < BATCH_SIZE && !is_padding_slot)
                   ? expert_routing[lane_id]
                   : 0;
      unsigned long long hit = __ballot(rv != 0);
      int dst = __popcll(hit & ((1ull << lane_id) - 1));
      if (rv != 0) {
        s_tok_of_col[dst] = lane_id;
        s_slot_of_col[dst] = rv - 1;
        // W13 reads the token straight out of the norm buffer; W2 reads the
        // SwiGLU output, which is indexed by (token, topk slot). `is_w2` is
        // uniform across the block, so one table serves whichever phase this
        // tile is in.
        s_row_off[dst] =
            is_w2 ? lane_id * (NUM_TOPK * INTERMEDIATE_SIZE) +
                        (rv - 1) * INTERMEDIATE_SIZE
                  : lane_id * W13_K;
      }
      if (lane_id == 0) {
        s_n_tok = (int)__popcll(hit);
      }
    }
    __syncthreads();
    n_tok = s_n_tok;
    if (n_tok == 0) {
      // Should be unreachable: an expert is in active_expert_ids only because
      // some token routed to it. Mark rather than return -- returning is what
      // breaks the barrier modulus. Every lane is inactive, so the tile runs
      // its MFMA over token 0's clamped row and writes nothing.
      MPK_WS_MARK(8104, global_tile);
    }
    tok_active = col < n_tok;
    int const src_col = tok_active ? col : 0;
    my_tok = s_tok_of_col[src_col];
    topk_slot = s_slot_of_col[src_col];
  } else {
    // Legacy per-token tile decode for batches wider than one MFMA N-tile.
    int route_val = expert_routing[tok_idx];
    if (route_val == 0) {
      MPK_WS_MARK(8103, global_tile); // exit: token not routed here
      return;
    }
    my_tok = tok_idx;
    topk_slot = route_val - 1;
    n_tok = 1;
    tok_active = (col == 0);
  }
  (void)n_tok;

#ifdef MPK_ENABLE_MOE_SUBPHASE
  g_subphase_scratch[0] = __builtin_amdgcn_s_memrealtime();
#endif

#ifdef MPK_MOE_INNER_TIMING
  // Inner split of Phase 8, on the same pattern as MPK_OPROJ_INNER_TIMING.
  // Not MPK_ENABLE_MOE_SUBPHASE: that mechanism writes g_subphase_scratch,
  // one global slot shared by all 240 workers, so its deltas come out
  // negative and are unusable. These are per-thread locals.
  //
  // A worker is either a W13 tile or a W2 tile, never both, so the two arms
  // print separately -- there is no single timeline covering W13 then W2.
  unsigned long long _mt0 = __builtin_amdgcn_s_memrealtime();
#endif

  // ══════════════════════════════════════════════════════════════════════════
  // PHASE 0: W13 + SwiGLU → write BF16 to swiglu_out
  // ══════════════════════════════════════════════════════════════════════════
  if (!is_w2) {
    MOE_DBG_SUBPHASE(2000);
    MPK_WS_MARK(8200, global_tile); // W13 compute

    // A tile with no routed token has nothing to compute, but it still has to
    // arrive -- the release fires on `% W13_TILES`, which counts every tile in
    // the compile-time space whether or not the mask happened to fill its slot.
    // Skipping only the compute is what keeps the barrier exact *and* the cost
    // proportional to the tokens actually routed: the GEMM below streams the
    // full expert weight tile from HBM, so running it for an empty slot would
    // make latency scale with MAX_ACTIVATED (4/8/16/32/64 at bs=1/2/4/8/16)
    // rather than with the handful of experts a batch really touches.
    //
    // Jumping rather than nesting the whole phase in an `if`: the arrival block
    // is ~1000 lines below, and every path between here and there must reach it.
    if (n_tok == 0) {
      goto w13_arrive;
    }
    // Braced so the `goto w13_arrive` above does not jump across these
    // initializations -- the label must sit outside their scope.
    {
    // Shared memory layout: FP8 quantized tokens + scales, TOK_ROWS rows.
    // The +16 pad per row is load-bearing: at ds_read_b128 granularity lane
    // `col` lands in bank group (col * (stride/16)) % 8, and 2960/16 == 185 is
    // odd, so the 16 lanes spread over all 8 groups instead of piling into one.
    constexpr int W13_TOK_ROW_STRIDE = W13_K + 16;
    constexpr int W13_SC_STRIDE = ((W13_MFMA_ITERS + 3) / 4) * 4;
    constexpr int W13_TOK_REGION = TOK_ROWS * W13_TOK_ROW_STRIDE;
    constexpr int W13_SC_REGION = TOK_ROWS * W13_SC_STRIDE;
    uint8_t *s_tok_fp8 = (uint8_t *)_fused_smem;
    uint8_t *s_tok_scales = s_tok_fp8 + W13_TOK_REGION;

    // B operand base for this lane: token row `col`. Inactive lanes clamp to
    // row 0 rather than skipping -- they still owe their ds_reads and their
    // share of the MFMA, which is a wave-level op reading B from all 64 lanes.
    // Folded to a literal at TOK_ROWS == 1 so the address stays uniform.
    uint8_t *b_tok =
        s_tok_fp8 +
        (TOK_ROWS == 1 ? 0 : (tok_active ? col : 0) * W13_TOK_ROW_STRIDE);
    uint8_t *b_scl =
        s_tok_scales +
        (TOK_ROWS == 1 ? 0 : (tok_active ? col : 0) * W13_SC_STRIDE);

    // Weight pointers
    uint8_t const *expert_weight =
        W_gate_up + static_cast<int64_t>(expert_id) * W13_EXPERT_BYTES;
    uint8_t const *wg_data =
        expert_weight + static_cast<int64_t>(wg_idx) * W13_WG_BYTES;
    uint8_t const *wg_scales = wg_data + W13_WG_DATA;

    // Single-row paths quantize one contiguous row at `tok_idx`; the packed
    // multi-row path gathers via s_row_off and `tok_idx` is a literal 0 there,
    // so this base is right for both. `my_tok` is per-lane under packing and
    // must not be used to form a block-wide base.
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
    constexpr int LDS_W13_OFF =
        ((W13_TOK_REGION + W13_SC_REGION + 15) / 16) * 16;
    // AMD reserves 3 KB of the 155 KB (runtime_header.h), and the layer index
    // sits in the last 4 bytes. The old bound was 155*1024, wrong by 3 KB in
    // the permissive direction.
    static_assert(LDS_W13_OFF + W13_TILE_BYTES * NUM_WAVES <=
                      mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
                          mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END,
                  "W13 LDS weight tiles exceed MI350X LDS budget");
    uint8_t *lds_w13_base = (uint8_t *)_fused_smem + LDS_W13_OFF;
#ifdef MPK_W13_T1_SPLIT_LDS_STAGE
    // Split staging area for W13 tile 1.
    //
    // The default path issues all 23 KiB of tile 1 in one burst after tile 0's
    // MFMA has finished, then drains it with `vmcnt(0)` before the tile-1
    // MFMA. Both the issue and the drain sit on the critical path.
    //
    // Split form: 11 chunks go early, into storage *disjoint* from tile 0, so
    // they can fly under the entire tile-0 MFMA chain without a WAR hazard on
    // the tile it is still reading. The remaining 13 are issued from inside
    // the final MFMA's result window (where the default schedule has nothing
    // but `s_nop` hazard padding), and the drain becomes a counted
    // `vmcnt(13)` -- gfx950 retires VMEM in issue order, so that number waits
    // for everything older while leaving exactly the 13 known-younger suffix
    // requests in flight under the SwiGLU epilogue.
    //
    // 11 chunks is set by the row boundary, not by the byte count: the staged
    // prefix must end on a whole row so every lane reads its row from exactly
    // one of the two buffers and no per-iteration pointer switch is needed.
    // 7 rows * (W13_K/2) = 10,080 bytes, which rounds up to 11 KiB.
#ifdef MPK_W13_T1_STAGE_CHUNKS_OVERRIDE
    // Debug: shrink the staged prefix. Used to test whether the fault tracks
    // the *address* the stage occupies rather than anything about the split
    // itself -- nothing else in this kernel has ever addressed LDS above
    // ~110 KB, so a capacity cliff there would never have been exercised.
    constexpr int W13_T1_STAGE_CHUNKS = MPK_W13_T1_STAGE_CHUNKS_OVERRIDE;
#else
    constexpr int W13_T1_STAGE_CHUNKS = 11;
#endif
    constexpr int W13_T1_STAGE_BYTES = W13_T1_STAGE_CHUNKS * 1024;
    // The prefix always *writes* whole rows, whatever the reader consumes:
    // as many as fit in the chunks reserved for it.
    constexpr int W13_T1_STAGED_PREFIX_ROWS =
        (W13_T1_STAGE_CHUNKS * 1024) / (W13_K / 2);
#ifdef MPK_W13_T1_STAGED_ROWS_OVERRIDE
    // Debug: shrink the set of rows served from the stage buffer. Fewer rows
    // read strictly less of a region that is already fully written, so any
    // value in [0, W13_T1_STAGED_PREFIX_ROWS] must be numerically equivalent.
    // A value that is not localizes the fault to reading the stage rather
    // than filling it. Clamped, because reading a row the prefix never wrote
    // is a bug in the experiment rather than a result.
    constexpr int W13_T1_STAGED_ROWS =
        MPK_W13_T1_STAGED_ROWS_OVERRIDE < W13_T1_STAGED_PREFIX_ROWS
            ? MPK_W13_T1_STAGED_ROWS_OVERRIDE
            : W13_T1_STAGED_PREFIX_ROWS;
#else
    // Serving every row the prefix wrote is the whole point; deriving it from
    // the chunk count keeps the reader and the writer from drifting apart.
    constexpr int W13_T1_STAGED_ROWS = W13_T1_STAGED_PREFIX_ROWS;
#endif
    constexpr int W13_T1_STAGED_DATA_BYTES =
        W13_T1_STAGED_PREFIX_ROWS * (W13_K / 2);
    static_assert(W13_T1_STAGED_DATA_BYTES <= W13_T1_STAGE_BYTES,
                  "the staged prefix must fit the chunks reserved for it");
#ifndef MPK_W13_T1_STAGE_CHUNKS_OVERRIDE
    static_assert((W13_TILE_DATA - W13_T1_STAGED_DATA_BYTES + 1023) / 1024 ==
                      13,
                  "the suffix sequence below hardcodes .rept 12; W13_K "
                  "changed, so re-derive the chunk count and the vmcnt");
#endif
    constexpr int LDS_W13_T1_STAGE_OFF =
        LDS_W13_OFF + W13_TILE_BYTES * NUM_WAVES;
    static_assert(LDS_W13_T1_STAGE_OFF + W13_T1_STAGE_BYTES * NUM_WAVES <=
                      mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
                          mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END,
                  "split W13 tile-1 stage exceeds the worker's LDS budget");
    uint8_t *lds_w13_t1_stage = (uint8_t *)_fused_smem + LDS_W13_T1_STAGE_OFF;
#endif
    i32x4_t w13_rsrc = make_w_buffer_rsrc(
        expert_weight, static_cast<uint32_t>(W13_EXPERT_BYTES));
    uint32_t w13_wg_voff_base = static_cast<uint32_t>(wg_idx) * W13_WG_BYTES;

#ifdef MPK_W13_T0_COUNTED_HANDOFF
#ifndef MPK_W13_PREQUANT
#error                                                                         \
    "MPK_W13_T0_COUNTED_HANDOFF reorders the prequantized-activation handoff ahead of the weight burst. Without MPK_W13_PREQUANT there is no handoff -- the activation is quantized in place from BF16 -- and nothing to hoist."
#endif
#ifndef MPK_W13_LINEAR_LOAD
#error                                                                         \
    "MPK_W13_T0_COUNTED_HANDOFF moves the tile-0 weight drain past the Phase B barrier, which is only sound when each wave issues every request filling the tile it reads. That is MPK_W13_LINEAR_LOAD; the striped load interleaves all four waves across all four tiles."
#endif
    static_assert(SINGLE_TOK,
                  "MPK_W13_T0_COUNTED_HANDOFF sweeps one contiguous activation "
                  "row. The packed multi-row path gathers per token and has no "
                  "single payload to issue ahead of the weights.");
    // ── Handoff payload issued ahead of the tile-0 weight burst ─────────────
    //
    // Two reads have to happen before the MFMA can start: the prequantized
    // activation row the router published, and this workgroup's W13 block
    // scales. The default schedule issues both *after* the 23 KiB tile-0
    // weight burst, and gfx950 retires VMEM strictly in issue order -- so
    // waiting for either of them means `vmcnt(0)`, which waits for all 23 KiB
    // of weights the MFMA has not asked for yet. The activation copy and the
    // scale scatter, neither of which touches a weight byte, end up gated on
    // the whole weight fetch.
    //
    // Issued here instead, they are the *oldest* requests in the queue, so a
    // counted wait can retire exactly them: `vmcnt(25)` leaves the 2 scale and
    // 23 weight requests untouched, and `vmcnt(23)` later leaves the 23 weight
    // requests untouched. The weights keep flying underneath both rendezvous.
    //
    // Every count below is per *wave*, because vmcnt is a per-wave counter, so
    // the arithmetic is only sound if all four waves issue the same number of
    // requests. The two work splits are therefore chosen to be wave-uniform
    // rather than thread-uniform -- see each one. That is the whole reason
    // this is not simply the existing code moved up.
    constexpr int PQ_DW4 = W13_K / 16;               // 184 activation
    constexpr int PQ_SC = W13_K / 128;               // 23 scale bytes
    constexpr int PQ_SC_DW4 = (PQ_SC + 15) / 16;     // 2
    constexpr int PQ_TOTAL_DW4 = PQ_DW4 + PQ_SC_DW4; // 186
    // Ceiling, so wave 3 covers the remainder. 47*4 == 188 >= 186, and the
    // two surplus units fall off the end under the `< PQ_TOTAL_DW4` test --
    // wave 3 still has 45 live lanes, so it issues the load like everyone
    // else. A floor here would leave the tail unfetched.
    constexpr int PQ_DW4_PER_WAVE =
        (PQ_TOTAL_DW4 + NUM_WAVES - 1) / NUM_WAVES; // 47
    static_assert(PQ_DW4_PER_WAVE <= 64,
                  "one dwordx4 per lane per wave is what makes the activation "
                  "cost exactly one VMEM request per wave");
    static_assert((NUM_WAVES - 1) * PQ_DW4_PER_WAVE < PQ_TOTAL_DW4,
                  "the last wave must have at least one live lane, or it "
                  "issues no request and its vmcnt below is off by one");
    // The scale bytes ride the same sweep as the data rather than taking a
    // second request. They are contiguous with the data in the source (the
    // router wrote FP8 then E8M0 into one allocation) but not in LDS, where
    // the token region is padded to W13_TOK_ROW_STRIDE. Both bases are
    // 16-aligned and the gap is exactly one dwordx4, so the destination is the
    // source unit shifted by one past the data -- no repacking, no second
    // load.
    static_assert(W13_TOK_ROW_STRIDE - W13_K == 16,
                  "the LDS scale base is assumed to sit exactly one dwordx4 "
                  "past the end of the token data");
    static_assert(LDS_W13_OFF >= W13_TOK_ROW_STRIDE + PQ_SC_DW4 * 16,
                  "the scale sweep rounds up to a whole dwordx4 and would "
                  "spill past the token staging region into the W13 tiles");
    // Source over-read: `input_base` is sized for the BF16 row this path
    // replaced (2 bytes/elem), and the payload is 1 byte/elem plus one byte
    // per 128, so the sweep ends far inside the allocation.
    static_assert(PQ_TOTAL_DW4 * 16 <= W13_K * 2,
                  "the rounded-up sweep reads past the handoff allocation");

    i32x4_t pq_chunk;
    int const pq_unit = warp_id * PQ_DW4_PER_WAVE + lane_id;
    bool const pq_active = lane_id < PQ_DW4_PER_WAVE && pq_unit < PQ_TOTAL_DW4;
    if (pq_active) {
        // Inline asm, not `((i32x4_t const *)pq)[unit]`: the point is to keep
        // the load opaque to LLVM's waitcnt pass so it does not "helpfully"
        // insert its own `vmcnt(0)` in front of the consume below and undo the
        // whole reordering.
        uint8_t const *pq_src = (uint8_t const *)input_base + pq_unit * 16;
        asm volatile("global_load_dwordx4 %0, %1, off"
                     : "=v"(pq_chunk)
                     : "v"(pq_src)
                     : "memory");
    }

    // W13 block scales, one tile per wave. The default split is by `tid` over
    // all 368 dwordx4, which gives waves 0-1 two requests and waves 2-3 one --
    // a per-wave asymmetry that would otherwise force three different vmcnt
    // values. Splitting by tile instead makes it 92 units per wave, hence two
    // requests for every wave, and one vmcnt for all of them. It is also the
    // same wave-owns-tile-w mapping the linear weight load already uses.
    constexpr int W13_SC_DW4_PER_TILE = W13_TILE_SCALE / 16;         // 92
    constexpr int W13_SC_LPT_WAVE = (W13_SC_DW4_PER_TILE + 63) / 64; // 2
    static_assert(W13_SC_LPT_WAVE == 2,
                  "the vmcnt values below count two scale requests per wave");
    static_assert((W13_SC_LPT_WAVE - 1) * 64 < W13_SC_DW4_PER_TILE,
                  "every scale request must have a live lane in every wave");
    i32x4_t w13_sc_buf[W13_SC_LPT_WAVE];
    {
        i32x4_t const *sc_src =
            (i32x4_t const *)wg_scales + warp_id * W13_SC_DW4_PER_TILE;
#pragma unroll
        for (int j = 0; j < W13_SC_LPT_WAVE; j++) {
          int idx = lane_id + j * 64;
          if (idx < W13_SC_DW4_PER_TILE) {
            asm volatile("flat_load_dwordx4 %0, %1"
                         : "=v"(w13_sc_buf[j])
                         : "v"(sc_src + idx)
                         : "memory");
          }
        }
    }

    // Younger-request counts consumed below, spelled out once here so the two
    // wait sites cannot drift from the issue order above.
    constexpr int W13_T0_WEIGHT_REQS = (W13_TILE_DATA + 1023) / 1024; // 23
    constexpr int PQ_VMCNT = W13_SC_LPT_WAVE + W13_T0_WEIGHT_REQS;    // 25
    constexpr int SC_VMCNT = W13_T0_WEIGHT_REQS;                      // 23
#endif

    // W13 T0: direct HBM→LDS via buffer_load_dwordx4 lds:1
    // Single inline asm block to prevent compiler vmcnt serialization.
    // Without this, compiler inserts s_waitcnt vmcnt(0) between each
    // __llvm_amdgcn_raw_buffer_load_lds call, serializing 24 loads
    // (24 × ~35ns = 840ns instead of ~75ns concurrent).
#ifdef MPK_W13_LINEAR_LOAD
    // ── Linear per-wave tile load ────────────────────────────────────────
    // The block-cooperative form below has all four waves stripe every tile:
    // wave w takes the 1 KiB slots at (w*1024 + j*4096), so covering one tile
    // costs W13_LPT = ceil(1440/256) = 6 loads per wave, i.e. 6*4 KiB = 24 KiB
    // fetched for 22.5 KiB of real data. The 256-element granularity of
    // W13_LPT is what forces the round-up: 1536 bytes per tile are re-reads of
    // the last element (`clamped`), pure HBM traffic.
    //
    // Here each wave instead owns one whole tile and walks it linearly in
    // 1 KiB steps, so the cover is ceil(23040/1024) = 23 loads and only 512
    // bytes overshoot. Per workgroup that is 92 KiB fetched instead of 96 --
    // 4.3% of all W13 weight traffic, which is 35.25 MB of the 52.9 MB the
    // MoE moves per layer.
    //
    // The LDS result is bit-identical either way. Both forms use the identity
    // map from tile-relative global offset to tile-relative LDS offset: the
    // striped form lands byte (w*1024 + j*4096 + lane*16) of tile t at the
    // same offset in tile t's LDS slot, and the linear form lands byte
    // (c*1024 + lane*16) of tile w at that offset in tile w's slot. The MFMA
    // reader at `lds_w13_base + warp_id * W13_TILE_BYTES` is unchanged.
    //
    // NUM_WAVES == 4 tiles and 4 waves, so warp_id indexes the tile directly.
    // gfx950 does not interlock an SALU write to m0 against the following
    // load-to-LDS, hence the leading s_nop pair; inside the loop the
    // v_add_u32 between s_addk_i32 and the load supplies that separation.
    // The MUBUF immediate offset would apply to *both* the global address and
    // the LDS destination on a load-to-LDS, so it stays 0 and m0/VADDR advance
    // together instead.
    {
      constexpr int W13_T0_CHUNKS = (W13_TILE_DATA + 1023) / 1024;
      static_assert(W13_T0_CHUNKS == 23,
                    "the linear W13 T0 sequence hardcodes .rept 22 below; "
                    "W13_K changed, so re-derive the chunk count");
      static_assert(NUM_WAVES == 4,
                    "linear load assigns one W13 tile per wave");
      unsigned const lds_t0_base = __builtin_amdgcn_readfirstlane(
          (unsigned)(uintptr_t)(lds_w13_base + warp_id * W13_TILE_BYTES));
      uint32_t t0_voff =
          w13_wg_voff_base +
          static_cast<uint32_t>(warp_id * W13_TILE_ROWS * (W13_K / 2)) +
          static_cast<uint32_t>(lane_id * 16);
      asm volatile(
          "s_nop 4\n"
          "s_mov_b32 m0, %[lds_base]\n"
          "s_nop 0\n"
          "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
          ".rept 22\n"
          "s_addk_i32 m0, 0x400\n"
          "v_add_u32_e32 %[voff], 0x400, %[voff]\n"
          "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
          ".endr\n"
          : [voff] "+v"(t0_voff)
          : [rsrc] "s"(w13_rsrc), [lds_base] "s"(lds_t0_base)
          : "memory", "m0");
    }
#else
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
#endif // MPK_W13_LINEAR_LOAD
#endif // MPK_W13_LDS_PREFETCH — 24 dwordx4 loads in flight

    if constexpr (PACK_N && !SINGLE_TOK) {
#ifdef MPK_W13_PREQUANT
      // The prequantized handoff publishes one row, not a gathered set: the
      // router's writer election is per XCD and per expert-tile, and it has no
      // notion of which tokens this expert drew. Extending it to the packed
      // multi-row path means publishing per token and gathering FP8 rows here,
      // which is a separate change. Fail at compile time rather than quietly
      // reading FP8 bytes as BF16.
      static_assert(!(PACK_N && !SINGLE_TOK),
                    "MPK_W13_PREQUANT is single-token only (BATCH_SIZE == 1)");
#endif
      // Gather: only the tokens routed to this expert, in N-column order.
      // s_row_off was published by the compaction's __syncthreads above.
      //
      // NT_LOAD is deliberately NOT passed here, unlike the W2 gather below.
      // It was tried as a cheap fix for the bs>1 W13 corruption and does not
      // work: `__builtin_nontemporal_load` emits the `nt` cache-*replacement*
      // hint, not a coherence scope, so it does not force a stale L2 line to
      // be re-fetched. Measured 30 bad weight groups against 32 without it,
      // which is inside the run-to-run spread at this geometry. The acquire in
      // gang_full_layer_fused_mi300.cuh is what actually fixes it.
      _gang_multirow_fp8_quant_gather<W13_K, TOK_ROWS, W13_TOK_ROW_STRIDE,
                                      W13_SC_STRIDE>(
          A, s_row_off, n_tok, s_tok_fp8, s_tok_scales);
    } else {
#if defined(MPK_ABLATE_W13_QUANT)
      // PERF PROBE ONLY -- produces garbage tokens. Skips the W13 activation
      // quant to measure its cost; the MFMA below still runs on whatever is in
      // LDS, so the timing is a valid upper bound on what hoisting can save.
      __syncthreads();
#elif defined(MPK_W13_PREQUANT)
      // ── The row arrives already quantized; copy it ─────────────────────
      //
      // The router's single norm writer per XCD published FP8 + E8M0 in the
      // exact layout this LDS region wants (see the MPK_W13_PREQUANT note in
      // gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh for the bit-identity
      // argument). What is left here is a copy, and it moves half the bytes
      // the BF16 read did.
      //
      // The scale bytes trail the data in the same allocation. That is safe
      // because the buffer is sized for BF16 (2 bytes/elem) and this payload is
      // 1 byte/elem plus one byte per 128, so it occupies just over half of it.
      //
      // Ordering: these are plain global loads of lines this XCD's own router
      // workgroup dirtied in this XCD's L2, which is the same visibility the
      // BF16 path relied on. The O-proj -> MoE handoff barrier and its
      // `buffer_inv` sit between the store and this read, unchanged.
      //
      // dwordx4, not dword: the row is 184 transactions in one issue rather
      // than 736 in a loop. That width is what makes the copy cheaper than
      // the quant it replaces -- the first attempt at dword-per-lane measured
      // *slower* than the quant (W13 compute 11.0 -> 12.9 us).
      //
      // The load stays here and drains into its own __syncthreads rather than
      // being issued next to Phase A's weight `buffer_load_lds` and landed
      // after that phase's `vmcnt(0)`. Deferring was tried and is 0.030 ms
      // worse: the chunk then lives in VGPRs across all of Phase A, and the
      // activation load competes for the same vmcnt queue as the 23 KiB of
      // weights the MFMA actually waits on.
#ifdef MPK_W13_T0_COUNTED_HANDOFF
      // The read already happened, above the weight burst. All that is left is
      // to land it, and `vmcnt(25)` retires exactly the one request that
      // carried it -- the 2 scale and 23 weight requests issued after it are
      // still in flight and stay that way.
      asm volatile("s_waitcnt vmcnt(%c[n])" ::[n] "n"(PQ_VMCNT) : "memory");
      if (pq_active) {
        // Units past the data carry the E8M0 scale bytes, which live one
        // dwordx4 further along in LDS than in the source; see the sweep.
        int const dst_unit = pq_unit + (pq_unit >= PQ_DW4 ? 1 : 0);
        ((i32x4_t *)s_tok_fp8)[dst_unit] = pq_chunk;
      }
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      __syncthreads();
#else
      {
        uint8_t const *pq = (uint8_t const *)input_base;
        constexpr int PQ_DW4 = W13_K / 16;
        static_assert(PQ_DW4 <= 256, "prequant copy assumes one dwordx4/lane");
        if (tid < PQ_DW4) {
          ((i32x4_t *)s_tok_fp8)[tid] = ((i32x4_t const *)pq)[tid];
        }
        constexpr int PQ_SC = W13_K / 128;
        if (tid < PQ_SC) {
          s_tok_scales[tid] = pq[W13_K + tid];
        }
      }
      __syncthreads();
#endif
#elif defined(MPK_WIDE_FP8_QUANT)
      _gang_wave_parallel_fp8_quant_wide<W13_K>(input_base, s_tok_fp8,
                                                s_tok_scales);
#else
      _gang_wave_parallel_fp8_quant<W13_K>(input_base, s_tok_fp8, s_tok_scales);
#endif
    }

#ifdef MPK_ENABLE_MOE_SUBPHASE
    g_subphase_scratch[1] = __builtin_amdgcn_s_memrealtime();
#endif

#ifdef MPK_W13_LDS_PREFETCH
    // ── Phase B: Drain tile_iter=0 HBM loads + scales concurrently ──────────
    {
#ifdef MPK_W13_T0_COUNTED_HANDOFF
      // Scales were issued above the weight burst, so retiring them is
      // `vmcnt(23)` -- the 23 tile-0 weight requests are the only thing
      // younger, and they keep flying through this scatter and its barrier.
      // The old `vmcnt(0)` here is the drain this flag exists to remove: it
      // made the scale scatter, which touches no weight byte, wait for all
      // 23 KiB of weights.
      //
      // flat_load bumps lgkmcnt as well as vmcnt, hence the second wait.
      asm volatile("s_waitcnt vmcnt(%c[n])" ::[n] "n"(SC_VMCNT) : "memory");
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      {
        i32x4_t *dst_sc = (i32x4_t *)(lds_w13_base + warp_id * W13_TILE_BYTES +
                                      W13_TILE_DATA_PADDED);
#pragma unroll
        for (int j = 0; j < W13_SC_LPT_WAVE; j++) {
          int idx = lane_id + j * 64;
          if (idx < W13_SC_DW4_PER_TILE) {
            dst_sc[idx] = w13_sc_buf[j];
          }
        }
      }

      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      __syncthreads();
#else
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
#endif

      // (timestamp [2] moved inside asm block below)
#ifdef MPK_W13_T1_EARLY_SCALE_LOAD
      // Tile-1 block scales, held in VGPRs across the tile-0 epilogue.
      //
      // The default path issues these global reads *after* the tile-1 data
      // drain, so their HBM latency is exposed: `vmcnt(0)` (data) -> load
      // scales -> `vmcnt(0)` (scales) -> LDS scatter. Two full round trips
      // back to back, with the second one serialized behind the first.
      //
      // Hoisting the reads next to the tile-1 data loads -- i.e. before the
      // tile-0 SwiGLU epilogue -- puts them in flight alongside data, so the
      // single drain covers both and only the LDS scatter is left afterwards.
      // The declaration lives here, outside the `W13_TILES_PER_WAVE > 1`
      // blocks, because the issue site and the scatter site are two different
      // scopes.
      constexpr int W13_T1_SC_DW4_PER_TILE = W13_TILE_SCALE / 16;
      constexpr int W13_T1_SC_LPT_WAVE = (W13_T1_SC_DW4_PER_TILE + 63) / 64;
      i32x4_t w13_t1_sc_wave[W13_T1_SC_LPT_WAVE];
#endif
      // ── Phase C: MFMA from LDS ───────────────────────────────────────
      {
#ifdef MPK_W13_T1_SPLIT_LDS_STAGE
        // Stage the tile-1 prefix now. The destination is the separate stage
        // buffer, disjoint from the tile-0 storage the MFMA below is about to
        // read, so all 11 chunks can fly under the whole tile-0 chain.
        if (W13_TILES_PER_WAVE > 1) {
          // The LDS byte offset is formed as an integer and only then turned
          // into an address. Casting a `__shared__`-derived *pointer* to
          // unsigned makes clang emit the generic-addrspace null check
          //     v_cmp_ne_u32 -1, v0 ; v_cndmask v0, 0, v0
          // which it is then free to CSE against any other predicate in
          // flight -- here the reader's `col < W13_T1_STAGED_ROWS` test. When
          // that happens the lanes failing the row test get a zeroed base,
          // and `readfirstlane` can broadcast one of *those* as the wave's
          // m0, aiming the whole DMA at LDS offset 0 and overwriting the
          // token buffer. Staying in the integer domain never materializes
          // the check.
          unsigned const lds_t1_stage_base = __builtin_amdgcn_readfirstlane(
              (unsigned)((uintptr_t)_fused_smem) + LDS_W13_T1_STAGE_OFF +
              (unsigned)(warp_id * W13_T1_STAGE_BYTES));
          uint32_t t1_stage_voff =
              w13_wg_voff_base +
              static_cast<uint32_t>((warp_id + NUM_WAVES) * W13_TILE_ROWS *
                                    (W13_K / 2)) +
              static_cast<uint32_t>(lane_id * 16);
          asm volatile(
              // Same m0 hazard rules as the T0 linear load: gfx950 does not
              // interlock an SALU write to m0 against a following load-to-LDS,
              // and the MUBUF immediate offset would advance both the global
              // and the LDS address, so it stays 0 and both walk together.
              "s_nop 4\n"
              "s_mov_b32 m0, %[lds_base]\n"
              "s_nop 0\n"
              "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
              ".rept %c[reps]\n"
              "s_addk_i32 m0, 0x400\n"
              "v_add_u32_e32 %[voff], 0x400, %[voff]\n"
              "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
              ".endr\n"
              : [voff] "+v"(t1_stage_voff)
              : [rsrc] "s"(w13_rsrc), [lds_base] "s"(lds_t1_stage_base),
                [reps] "n"(W13_T1_STAGE_CHUNKS - 1)
              : "memory", "m0");
#ifdef MPK_W13_T1_SPLIT_PROBE_DRAIN
          // Diagnostic: drain the prefix immediately. This throws away the
          // entire point of the split (the loads no longer fly under the
          // tile-0 MFMA), so it is only useful for deciding *why* the split
          // miscomputes -- if numerics come back with this in place, the
          // fault is that the 11 requests are still outstanding across code
          // the compiler believes it has already waited for, not that the
          // stage buffer overlaps live LDS.
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
        }
#endif
#ifdef MPK_W13_T0_COUNTED_HANDOFF
        // The tile-0 weight burst has still not been waited on. That is the
        // point of the flag: both handoff rendezvous above used counted waits
        // and left all 23 requests flying, so the weights have been in the air
        // across the activation store, the scale scatter and two barriers.
        // Here is the first instruction that actually reads a weight byte, so
        // here is where they are drained.
        //
        // A per-wave drain is sufficient only because MPK_W13_LINEAR_LOAD
        // gives wave w sole ownership of tile w -- it issues every request
        // that fills the region it is about to read. Under the striped load
        // the four waves interleave across all four tiles, and moving the
        // drain to after the barrier would let a wave read bytes a sibling
        // has not landed. Hence the #error pairing the two flags.
#if defined(MPK_W13_T1_SPLIT_LDS_STAGE)
        // The tile-1 prefix was issued just above and must keep flying under
        // the tile-0 MFMA; retiring the older tile-0 requests is a counted
        // wait, not a drain.
        constexpr int W13_T0_DRAIN_KEEP =
            (W13_TILES_PER_WAVE > 1) ? W13_T1_STAGE_CHUNKS : 0;
#else
        constexpr int W13_T0_DRAIN_KEEP = 0;
#endif
        asm volatile("s_waitcnt vmcnt(%c[n])" ::[n] "n"(W13_T0_DRAIN_KEEP)
                     : "memory");
#endif
        uint8_t *lds_w13_data = lds_w13_base + warp_id * W13_TILE_BYTES;
        uint8_t *lds_w13_sc = lds_w13_data + W13_TILE_DATA_PADDED;
        int w_row_local = col;
        int const row_data_base = w_row_local * (W13_K / 2);
        int const row_scale_base = w_row_local * W13_NUM_BLK32;

        int wave_tile_0 = warp_id;
        f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

#ifdef MPK_W13_BIAS_PREFETCH
        // Hoist the tile-0 SwiGLU bias read above the MFMA block.
        //
        // The epilogue below needs four consecutive bf16 biases at
        // [expert_id * W13_OUTPUT_SIZE + out_n]. Left where it is used, clang
        // emits it as two scalar `flat_load_dword`s waited on *inside* the
        // epilogue -- ATT puts 3.7% of all cycles on those two `s_waitcnt`s.
        // Two problems, both fixed here: `flat` has to probe the LDS aperture
        // where `global` does not, and the load sits after the MFMA block that
        // could have hidden its latency. One dwordx2 covers all four halves.
        //
        // Deliberately NOT inline asm. A raw `global_load_dwordx2` here would
        // only be safe if a counted `vmcnt(N)` covered it before the consume,
        // and the tile-1 drain is `vmcnt(0)` sitting *after* this epilogue, so
        // an asm load here has nothing guaranteeing it has landed -- which
        // is exactly the nondeterminism the first version of this produced
        // (seven runs, seven different outputs, all plausible text). Going
        // through the compiler keeps the same single dwordx2 while letting it
        // place the s_waitcnt.
        //
        // The addrspace(1) cast is the half that actually matters: the bias
        // pointer arrives generic, which is why the default lowering is
        // `flat_load` rather than `global_load`.
        uint2 w13_t0_bias_pf = {0u, 0u};
        int const t0_out_n_pf = wg_idx * W13_OUTPUT_PER_WG + wave_tile_0 * 16 +
                                g * 4;
        if (tok_active && t0_out_n_pf + 3 < W13_OUTPUT_SIZE) {
          // 4-aligned out_n over a bf16 array is an 8-byte-aligned address,
          // so the pair load is legal for every lane that reaches here.
          typedef unsigned int u32x2_t __attribute__((ext_vector_type(2)));
          auto const *bias_address =
              (u32x2_t const __attribute__((address_space(1))) *)&d_w13_bias
                  [expert_id * W13_OUTPUT_SIZE + t0_out_n_pf];
          u32x2_t const v = *bias_address;
          w13_t0_bias_pf.x = v.x;
          w13_t0_bias_pf.y = v.y;
        }
#endif

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
#ifdef MPK_W13_T1_SPLIT_LDS_STAGE
          static_assert(W13_TILES_PER_WAVE > 1,
                        "the split W13 suffix requires a second tile");
          // The final tile-0 MFMA samples its operands from VGPRs that were
          // already filled by ds_reads retired under `lgkmcnt(0)`, so the
          // tile-0 LDS slot is dead from that point on and the suffix may
          // overwrite it. Both addresses are formed *outside* the asm block:
          // the block is opaque to the scheduler, so anything computed inside
          // it would serialize against the MFMA it is meant to hide under.
          // Integer-domain, for the same reason as the prefix base above: a
          // pointer cast here would give clang another null check to CSE
          // against a lane predicate.
          unsigned const lds_t1_suffix_base = __builtin_amdgcn_readfirstlane(
              (unsigned)((uintptr_t)_fused_smem) + LDS_W13_OFF +
              (unsigned)(warp_id * W13_TILE_BYTES) +
              (unsigned)W13_T1_STAGED_DATA_BYTES);
          uint32_t t1_suffix_voff =
              w13_wg_voff_base +
              static_cast<uint32_t>((warp_id + NUM_WAVES) * W13_TILE_ROWS *
                                    (W13_K / 2)) +
              static_cast<uint32_t>(W13_T1_STAGED_DATA_BYTES) +
              static_cast<uint32_t>(lane_id * 16);
#endif
          unsigned w_addr =
              (unsigned)(uintptr_t)(lds_w13_data + row_data_base + g * 16);
          unsigned ws_addr =
              (unsigned)(uintptr_t)(lds_w13_sc + row_scale_base + g);
          unsigned t_addr = (unsigned)(uintptr_t)(b_tok + g * 16);
          unsigned ts_addr = (unsigned)(uintptr_t)(b_scl);
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
              // streams them over the op rather than latching at issue. When
              // LDS returns fast the write-back lands mid-MFMA and the op sees
              // mixed-iteration operands (~17-22% of launches before banking).
              // Ping-pong: while the MFMA consumes bank X, prefetch writes bank
              // 1-X, so no register is ever both a live source and an in-flight
              // LDS destination.
              //
              // Verified by tests/standalone/test_mfma_pipeline_hazards.hip.
              //
              // ── MPK_MFMA_PINGPONG_SCHED ──────────────────────────────────
              //
              // The banking above is necessary but not sufficient. It removes
              // the WAR *race*; it does not schedule the wait. This loop puts
              // `s_waitcnt lgkmcnt(0)` at the TOP of each half, two
              // instructions after the previous MFMA was issued, so the wave
              // pays the LDS latency and the MFMA's SrcC interlock back to
              // back instead of overlapping them.
              //
              // v_mfma_scale_f32_16x16x128_f8f6f4 is a 32-cycle op on CDNA4,
              // and consecutive MFMAs accumulating into the same a[0:3] have a
              // SrcC RAW dependency that requires 32 states of separation. The
              // hardware will insert that gap whether or not we fill it. This
              // flag fills it: the address updates and `s_nop`s pad to exactly
              // 32 states, and the `s_waitcnt` moves to the bottom of the half
              // -- immediately before the MFMA that consumes the data -- so
              // the LDS latency is spent inside an interval that had to elapse
              // anyway. The wait becomes free rather than additive.
              //
              // State accounting, both edges credited to exactly 32:
              //   MFMA_A -> MFMA_B : 5 VALU + s_nop 15 (16) + s_nop 10 (11)
              //                       = 5 + 27 = 32
              //   MFMA_B -> MFMA_A : s_nop 15 (16) + s_nop 11 (12) = 28, plus
              //                       the 4 VALU address updates at the top of
              //                       the next trip = 32
              // Only VALU/s_nop are credited. The ds_reads and scalar loop
              // control in each half are uncredited margin, so the true
              // separation is >= 32 on both edges. `s_nop N` waits N+1 states.
              //
              // It also closes a WAR window the default schedule leaves open.
              // The default refills bank 0 *before* MFMA(bank 1), i.e. ~25
              // states after MFMA(bank 0) was issued -- inside the 32 states
              // that MFMA needs to finish sampling bank 0. This ordering
              // refills bank 0 *after* MFMA(bank 1), a guaranteed 32+ states
              // later. So the scheduled form is the safer of the two.
              //
              // Precondition (static_assert below): the loop emits MFMAs in
              // pairs and exits only via the top-of-loop test, so it always
              // emits an odd 2*trips+1 with the last operands in bank 0. That
              // makes the second (bank-1) tail dead, which is why this arm has
              // one tail where the default has two. Valid ONLY for an odd
              // W13_MFMA_ITERS/W2_MFMA_ITERS.
              "v_accvgpr_write_b32 a0, 0\n"
              "v_accvgpr_write_b32 a1, 0\n"
              "v_accvgpr_write_b32 a2, 0\n"
              "v_accvgpr_write_b32 a3, 0\n"
#ifdef MPK_W13_DUAL_ACC
              // Odd K blocks accumulate into a disjoint second chain, merged
              // back into a[0:3] after the tail.
              "v_accvgpr_write_b32 a4, 0\n"
              "v_accvgpr_write_b32 a5, 0\n"
              "v_accvgpr_write_b32 a6, 0\n"
              "v_accvgpr_write_b32 a7, 0\n"
#endif

              // Pre-issue 5 reads for iteration 0 into bank 0
              "ds_read_b128 v[22:25], %[wa]\n"
              "ds_read_u8   v7, %[wsa]\n"
              "ds_read_b128 v[8:11], %[ta]\n"
              "ds_read_b128 v[12:15], %[ta] offset:64\n"
              "ds_read_u8   v16, %[tsa]\n"
#ifndef MPK_W13_T0_MFMA_UNROLLED
              "s_mov_b32 s13, 0\n"
#endif
#ifdef MPK_MFMA_PINGPONG_SCHED
              // See the MPK_MFMA_PINGPONG_SCHED block comment above the W13
              // T0 loop for the state accounting. The wait for iteration 0's
              // bank-0 reads moves out of the loop body to here; from then on
              // every wait sits at the bottom of a half, covered by the MFMA
              // interval that has to elapse anyway.
              "s_waitcnt lgkmcnt(0)\n"

              "PIPELINED_W13_T0_%=:\n"
              "s_cmpk_lt_i32 s13, %[iters_m1]\n"
              "s_cbranch_scc0 W13_T0_FINAL_%=\n"

              // ---- prefetch bank 1, then consume bank 0 ----
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
              // MFMA_A -> MFMA_B: 5 issued instructions + s_nop 15 (16 states)
              // + s_nop 10 (11 states) = 5 + 27 = 32. The bank-1 wait is the
              // last thing before MFMA_B, so its latency is spent inside the
              // interval rather than after it.
              "v_add_u32_e32 %[wa], 64, %[wa]\n"
              "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
              "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
              "s_add_i32 s13, s13, 1\n"
              "v_add_u32_e32 v17, s13, %[tsa]\n"
#ifdef MPK_W13_DUAL_ACC
              // MFMA_B targets a disjoint chain, so there is no SrcC RAW edge
              // to cover here: 5 issued address updates + s_nop 2 (3 states)
              // = the 8-state adjacent cadence.
              "s_nop 2\n"
#else
              "s_nop 15\n"
              "s_nop 10\n"
#endif
              "s_waitcnt lgkmcnt(0)\n"

          // ---- consume bank 1, then refill bank 0 ----
          // Refilling bank 0 *after* MFMA_B (not before it, as the
          // unscheduled loop does) is what keeps the WAR window closed.
          //
          // Single-chain: MFMA_A was issued a full 32 states earlier, so
          // it has finished sampling bank 0 before these writes land.
          //
          // Dual-chain: the issue distance shrinks to ~11 states, so the
          // margin is no longer the padding -- it is the `s_waitcnt
          // lgkmcnt(0)` above, which stalls on bank 1's reads (issued
          // before MFMA_A) for the balance of the LDS round trip. The
          // refill therefore still lands well past MFMA_A's 32 cycles.
          // Confirmed bit-exact on gfx950 by
          // tests/standalone/test_mfma_pipeline_hazards.hip, whose
          // uncontended LDS is the worst case for this hazard.
#ifdef MPK_W13_DUAL_ACC
              "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], v[32:39], "
              "a[4:7], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
#else
              "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], "
              "a[0:3], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
#endif
              "ds_read_u8   v16, v17\n"
              "ds_read_b128 v[22:25], %[wa]\n"
              "ds_read_u8   v7, %[wsa]\n"
              "ds_read_b128 v[8:11], %[ta]\n"
              "ds_read_b128 v[12:15], %[ta] offset:64\n"
          // MFMA_B -> next MFMA_A: s_nop 15 (16) + s_nop 11 (12) = 28,
          // plus the 4 VALU address updates at the top of the next trip =
          // 32. Only those 4 VALU are credited; the 5 ds_reads and the
          // scalar loop control are uncredited margin.
#ifdef MPK_W13_DUAL_ACC
              // Disjoint chains again: s_nop 3 (4 states) + the 4 address
              // updates at the top of the next trip = 8. Chain 0 itself was
              // last written 16 states ago, which same-opcode SrcC forwarding
              // covers.
              "s_nop 3\n"
#else
              "s_nop 15\n"
              "s_nop 11\n"
#endif
              "s_waitcnt lgkmcnt(0)\n"
              "s_branch PIPELINED_W13_T0_%=\n"

              // Single tail. The loop emits MFMAs in pairs and exits only via
              // the top-of-loop test, so the emitted count is always
              // 2*trips + 1 -- odd, with the final operands always in bank 0.
              // W13_MFMA_ITERS is 23 (static_assert above), so this is exact.
              "W13_T0_FINAL_%=:\n"
              "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
              "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
#else

// ── MPK_W13_T0_MFMA_UNROLLED ──
//
// W13_MFMA_ITERS is a compile-time constant, so the scalar loop control is
// runtime work spent recomputing something the assembler already knows.
// `.rept` emits the same bank-0/bank-1 pair (W13_MFMA_ITERS-1)/2 times and an
// assembler symbol carries the token-scale offset. Per pair that removes two
// s_add_i32, two s_cmpk_lt_i32, one s_cbranch_scc0, one s_cbranch_scc1 and one
// s_branch -- and, more to the point, takes the taken-branch bubble off the
// end of every pair, which sits directly between two MFMAs.
//
// It also drops the dual-tail hedge below: with a constant trip count the exit
// parity is known here, so only the bank-0 final MFMA is reachable. That is
// what the static_assert pins.
#ifdef MPK_W13_T0_MFMA_UNROLLED
              ".set MPK_W13_T0_SC_%=, 1\n"
              ".rept %c[unroll_pairs]\n"
#else
              "PIPELINED_W13_T0_%=:\n"
#endif
              // ---- consume bank 0, prefetch into bank 1 ----
              "s_waitcnt lgkmcnt(0)\n"
              "v_add_u32_e32 %[wa], 64, %[wa]\n"
              "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
              "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
#ifdef MPK_W13_T0_MFMA_UNROLLED
              "v_add_u32_e32 v17, MPK_W13_T0_SC_%=, %[tsa]\n"
#else
              "s_add_i32 s13, s13, 1\n"
              "v_add_u32_e32 v17, s13, %[tsa]\n"
#endif
              "ds_read_u8   v19, v17\n"
              "ds_read_b128 v[26:29], %[wa]\n"
              "ds_read_u8   v18, %[wsa]\n"
              "ds_read_b128 v[32:35], %[ta]\n"
              "ds_read_b128 v[36:39], %[ta] offset:64\n"
              "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
              "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
#ifndef MPK_W13_T0_MFMA_UNROLLED
              "s_cmpk_lt_i32 s13, %[iters_m1]\n"
              "s_cbranch_scc0 W13_T0_TAIL_B1_%=\n"
#endif

              // ---- consume bank 1, prefetch into bank 0 ----
              "s_waitcnt lgkmcnt(0)\n"
              "v_add_u32_e32 %[wa], 64, %[wa]\n"
              "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
              "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
#ifdef MPK_W13_T0_MFMA_UNROLLED
              "v_add_u32_e32 v17, MPK_W13_T0_SC_%= + 1, %[tsa]\n"
#else
              "s_add_i32 s13, s13, 1\n"
              "v_add_u32_e32 v17, s13, %[tsa]\n"
#endif
              "ds_read_u8   v16, v17\n"
              "ds_read_b128 v[22:25], %[wa]\n"
              "ds_read_u8   v7, %[wsa]\n"
              "ds_read_b128 v[8:11], %[ta]\n"
              "ds_read_b128 v[12:15], %[ta] offset:64\n"
#ifdef MPK_W13_DUAL_ACC
              // Odd K blocks onto the second chain. Nothing else in this half
              // moves: the ~11 slots back to the bank-0 MFMA were never issue
              // work the schedule chose, they were the hardware holding the
              // shared SrcC for its remaining ~21 states. Disjoint SrcC drops
              // that to the 8-state adjacent cadence.
              "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], v[32:39], "
              "a[4:7], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
#else
              "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], "
              "a[0:3], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
#endif
#ifdef MPK_W13_T0_MFMA_UNROLLED
              ".set MPK_W13_T0_SC_%=, MPK_W13_T0_SC_%= + 2\n"
              ".endr\n"

              // ── Final MFMA (bank 0) ──
              // (W13_MFMA_ITERS-1)/2 pairs consumed an even number of K
              // blocks, so the odd one out is always block W13_MFMA_ITERS-1,
              // whose operands the last pair prefetched into bank 0.
              "s_waitcnt lgkmcnt(0)\n"
#ifdef MPK_W13_DUAL_ACC
              // Chain 0 last ran one half ago -- about eleven issued slots,
              // under the 16 states same-opcode SrcC forwarding needs. Inside
              // the loop the *next* pair's four address updates make up the
              // difference; the tail has no next pair, so pad it explicitly.
              // Five states once per tile against 704 removed.
              "s_nop 4\n"
#endif
              "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
              "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
#else
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
#ifdef MPK_W13_DUAL_ACC
              // Same short chain-0 reuse as the unrolled tail; see there.
              "s_nop 4\n"
#endif
              "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
              "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
              "s_branch W13_T0_ACC_%=\n"

              "W13_T0_TAIL_B1_%=:\n"
              "s_waitcnt lgkmcnt(0)\n"
#ifdef MPK_W13_DUAL_ACC
              // Dead at the live odd W13_MFMA_ITERS, but it is the bank-1
              // final, so it belongs to the odd chain like every other bank-1
              // MFMA -- keeping it on a[0:3] would double-count into chain 0
              // at an even count.
              "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], v[32:39], "
              "a[4:7], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
#else
              "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], "
              "a[0:3], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
#endif
#endif // MPK_W13_T0_MFMA_UNROLLED
#endif

              "W13_T0_ACC_%=:\n"
#if defined(MPK_W13_T1_SPLIT_LDS_STAGE) &&                                     \
    !defined(MPK_W13_T1_SPLIT_NOSUFFIX) && !defined(MPK_W13_T1_SPLIT_STAGE_PROBE)
              // The 13 suffix requests take the place of the hazard padding.
              // The issue train is 1 + 12*3 = 37 instruction slots, more than
              // the 32 states the final MFMA needs before its accumulator can
              // be read, so the wait is still satisfied -- it is now doing
              // work. The leading s_nop preserves the gfx950 m0-to-LDS MUBUF
              // hazard workaround, and the immediate offset stays 0 because it
              // would advance the LDS address as well as the global one.
              "s_nop 4\n"
              "s_mov_b32 m0, %[t1_lds_base]\n"
              "s_nop 0\n"
              "buffer_load_dwordx4 %[t1_voff], %[rsrc], 0 offen sc0 nt lds\n"
              ".rept %c[sfx_reps]\n"
              "s_addk_i32 m0, 0x400\n"
              "v_add_u32_e32 %[t1_voff], 0x400, %[t1_voff]\n"
              "buffer_load_dwordx4 %[t1_voff], %[rsrc], 0 offen sc0 nt lds\n"
              ".endr\n"
#else
              // 32 clocks: the scaled MFMA is a 32-cycle op on CDNA4. The old
              // "s_nop 7; s_nop 0" was 9 clocks (correct only for a 4-pass
              // MFMA) and returned a partially-retired accumulator every time.
              "s_nop 15\n"
              "s_nop 15\n"
#endif
#ifdef MPK_W13_DUAL_ACC
              // Chain 1's last MFMA retired well before chain 0's tail, and
              // the bank-0 input registers are dead here, so stage the odd
              // partials there and merge in FP32.
              "v_accvgpr_read_b32 v22, a4\n"
              "v_accvgpr_read_b32 v23, a5\n"
              "v_accvgpr_read_b32 v24, a6\n"
              "v_accvgpr_read_b32 v25, a7\n"
#endif
              "v_accvgpr_read_b32 %[acc0], a0\n"
              "v_accvgpr_read_b32 %[acc1], a1\n"
              "v_accvgpr_read_b32 %[acc2], a2\n"
              "v_accvgpr_read_b32 %[acc3], a3\n"
#ifdef MPK_W13_DUAL_ACC
              "v_add_f32_e32 %[acc0], v22, %[acc0]\n"
              "v_add_f32_e32 %[acc1], v23, %[acc1]\n"
              "v_add_f32_e32 %[acc2], v24, %[acc2]\n"
              "v_add_f32_e32 %[acc3], v25, %[acc3]\n"
#endif
              : [acc0] "=v"(acc[0]),
                [acc1] "=v"(acc[1]),
                [acc2] "=v"(acc[2]),
                [acc3] "=v"(acc[3]),
                [wa] "+v"(w_addr),
                [wsa] "+v"(ws_addr),
                [ta] "+v"(t_addr)
#if defined(MPK_W13_T1_SPLIT_LDS_STAGE) &&                                     \
    !defined(MPK_W13_T1_SPLIT_NOSUFFIX) && !defined(MPK_W13_T1_SPLIT_STAGE_PROBE)
                    ,
                [t1_voff] "+v"(t1_suffix_voff)
#endif
              : [tsa] "v"(ts_addr),
                [iters_m1] "n"(W13_MFMA_ITERS - 1),
                [unroll_pairs] "n"((W13_MFMA_ITERS - 1) / 2)
#if defined(MPK_W13_T1_SPLIT_LDS_STAGE) &&                                     \
    !defined(MPK_W13_T1_SPLIT_NOSUFFIX) && !defined(MPK_W13_T1_SPLIT_STAGE_PROBE)
                    ,
                [rsrc] "s"(w13_rsrc),
                [t1_lds_base] "s"(lds_t1_suffix_base),
                [sfx_reps] "n"(
                    (W13_TILE_DATA - W13_T1_STAGED_DATA_BYTES + 1023) / 1024 -
                    1)
#endif
              : "memory",
                "m0",
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
                "a3"
#ifdef MPK_W13_DUAL_ACC
                ,
                "a4",
                "a5",
                "a6",
                "a7"
#endif
          );
        }

        // ── Issue tile_iter=1 per-wave HBM→LDS loads BEFORE SwiGLU ──
        // Each wave loads only its own tile slot. The buffer_load_lds
        // writes go to [warp_id * W13_TILE_BYTES + s*1024 + j*4096]
        // replicating the cooperative layout without cross-wave deps.
        // No __syncthreads needed: each wave's MFMA is done reading LDS
        // before the loads overwrite that same tile slot.
        // Loads fly during SwiGLU epilogue (overlap HBM latency).
        if (W13_TILES_PER_WAVE > 1) {
#if defined(MPK_W13_T1_SPLIT_STAGE_PROBE)
          // Probe: the prefix above wrote the stage buffer, but nothing reads
          // it and the ordinary 23-chunk T1 load still runs below, so numerics
          // must be bit-identical to the baseline. If they are not, the stage
          // region is landing on LDS that is still live -- which is a layout
          // question, entirely separate from the split's read path.
          {
            unsigned const lds_t1_base = __builtin_amdgcn_readfirstlane(
                (unsigned)(uintptr_t)(lds_w13_base + warp_id * W13_TILE_BYTES));
            uint32_t t1_voff =
                w13_wg_voff_base +
                static_cast<uint32_t>((warp_id + NUM_WAVES) * W13_TILE_ROWS *
                                      (W13_K / 2)) +
                static_cast<uint32_t>(lane_id * 16);
            asm volatile(
                "s_nop 4\n"
                "s_mov_b32 m0, %[lds_base]\n"
                "s_nop 0\n"
                "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
                ".rept 22\n"
                "s_addk_i32 m0, 0x400\n"
                "v_add_u32_e32 %[voff], 0x400, %[voff]\n"
                "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
                ".endr\n"
                : [voff] "+v"(t1_voff)
                : [rsrc] "s"(w13_rsrc), [lds_base] "s"(lds_t1_base)
                : "memory", "m0");
          }
#elif defined(MPK_W13_T1_SPLIT_LDS_STAGE)
          // The 11-chunk prefix went out before the tile-0 MFMA. The suffix
          // normally goes out from inside that MFMA's result window, leaving
          // nothing to issue here; MPK_W13_T1_SPLIT_NOSUFFIX keeps the split
          // *layout* but issues the suffix from this site, which isolates the
          // layout change from the in-MFMA placement when bisecting.
#ifdef MPK_W13_T1_SPLIT_NOSUFFIX
          {
            unsigned const lds_t1_sfx = __builtin_amdgcn_readfirstlane(
                (unsigned)(uintptr_t)(lds_w13_base + warp_id * W13_TILE_BYTES +
                                      W13_T1_STAGED_DATA_BYTES));
            uint32_t t1_sfx_voff =
                w13_wg_voff_base +
                static_cast<uint32_t>((warp_id + NUM_WAVES) * W13_TILE_ROWS *
                                      (W13_K / 2)) +
                static_cast<uint32_t>(W13_T1_STAGED_DATA_BYTES) +
                static_cast<uint32_t>(lane_id * 16);
            asm volatile(
                "s_nop 4\n"
                "s_mov_b32 m0, %[lds_base]\n"
                "s_nop 0\n"
                "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
                ".rept %c[sfx_reps]\n"
                "s_addk_i32 m0, 0x400\n"
                "v_add_u32_e32 %[voff], 0x400, %[voff]\n"
                "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
                ".endr\n"
                : [voff] "+v"(t1_sfx_voff)
                : [rsrc] "s"(w13_rsrc), [lds_base] "s"(lds_t1_sfx),
                  [sfx_reps] "n"((W13_TILE_DATA - W13_T1_STAGED_DATA_BYTES +
                                  1023) / 1024 - 1)
                : "memory", "m0");
          }
#endif
#elif defined(MPK_W13_T1_LINEAR_LOAD)
          // Same striped->linear transform as the T0 site above; see the long
          // note there for the byte accounting and the m0 hazard rules. The
          // only difference is the source tile: T1 reads wave tile
          // (warp_id + NUM_WAVES) from HBM but lands it back in tile
          // warp_id's LDS slot, which tile 0's MFMA has finished reading.
          {
            constexpr int W13_T1_CHUNKS = (W13_TILE_DATA + 1023) / 1024;
            static_assert(W13_T1_CHUNKS == 23,
                          "the linear W13 T1 sequence hardcodes .rept 22 "
                          "below; W13_K changed, so re-derive the chunk count");
            unsigned const lds_t1_base = __builtin_amdgcn_readfirstlane(
                (unsigned)(uintptr_t)(lds_w13_base + warp_id * W13_TILE_BYTES));
            uint32_t t1_voff =
                w13_wg_voff_base +
                static_cast<uint32_t>((warp_id + NUM_WAVES) * W13_TILE_ROWS *
                                      (W13_K / 2)) +
                static_cast<uint32_t>(lane_id * 16);
            asm volatile(
                "s_nop 4\n"
                "s_mov_b32 m0, %[lds_base]\n"
                "s_nop 0\n"
                "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
                ".rept 22\n"
                "s_addk_i32 m0, 0x400\n"
                "v_add_u32_e32 %[voff], 0x400, %[voff]\n"
                "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
                ".endr\n"
                : [voff] "+v"(t1_voff)
                : [rsrc] "s"(w13_rsrc), [lds_base] "s"(lds_t1_base)
                : "memory", "m0");
          }
#else
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
#endif // MPK_W13_T1_LINEAR_LOAD
#ifdef MPK_W13_T1_EARLY_SCALE_LOAD
          // Issue the tile-1 scale reads beside the tile-1 data loads, so both
          // are in flight over the tile-0 epilogue below and the drain at the
          // top of the tile-1 block covers them together.
          {
            i32x4_t const *t1_sc_src =
                (i32x4_t const *)(wg_scales + (warp_id + NUM_WAVES) *
                                                  W13_TILE_ROWS *
                                                  W13_NUM_BLK32);
#pragma unroll
            for (int j = 0; j < W13_T1_SC_LPT_WAVE; j++) {
              int idx = lane_id + j * 64;
              if (idx < W13_T1_SC_DW4_PER_TILE) {
                w13_t1_sc_wave[j] = t1_sc_src[idx];
              }
            }
            // Keep the global reads above the independent tile-0 epilogue:
            // without this the scheduler is free to sink them back down to
            // their eventual LDS scatter, which is the schedule we are
            // trying to leave behind.
            asm volatile("" ::: "memory");
          }
#endif
        }

        // tile_iter=0 SwiGLU epilogue (tile_iter=1 HBM loads fly in background)
        //
        // Lane (g, col) holds D[m = wave_tile*16 + g*4 + i][n = col]: four
        // output columns of token `col`. The guard is on the token, not on
        // col == 0 -- every lane now has live results. The gate/up pairing
        // survives untouched: acc[i] and acc[i+1] are adjacent along m, both
        // in this lane.
        if (tok_active) {
          constexpr int ACT_STRIDE = W13_OUTPUT_SIZE / 2;
#ifdef MPK_W13_BIAS_PREFETCH
          // The two iterations of the loop below write act_n = out_n/2 for
          // out_n = base and base+2, i.e. two *adjacent* bf16 lanes of the
          // activation row -- so they are one aligned dword, not two shorts.
          // Both halves of the prefetched dwordx2 are already here, so the
          // whole epilogue is register-resident: no load, no wait, one store.
          if (t0_out_n_pf + 3 < W13_OUTPUT_SIZE) {
            unsigned const bt_g0 = (w13_t0_bias_pf.x & 0xFFFFu) << 16;
            unsigned const bt_u0 = w13_t0_bias_pf.x & 0xFFFF0000u;
            unsigned const bt_g1 = (w13_t0_bias_pf.y & 0xFFFFu) << 16;
            unsigned const bt_u1 = w13_t0_bias_pf.y & 0xFFFF0000u;
            float bias_g0, bias_u0, bias_g1, bias_u1;
            __builtin_memcpy(&bias_g0, &bt_g0, 4);
            __builtin_memcpy(&bias_u0, &bt_u0, 4);
            __builtin_memcpy(&bias_g1, &bt_g1, 4);
            __builtin_memcpy(&bias_u1, &bt_u1, 4);
            unsigned short const a0 =
                _gang_float_to_bf16(fast_swigluoai(acc[0] + bias_g0,
                                                   acc[1] + bias_u0));
            unsigned short const a1 =
                _gang_float_to_bf16(fast_swigluoai(acc[2] + bias_g1,
                                                   acc[3] + bias_u1));
            int const act_n = t0_out_n_pf / 2;
            int const out_idx = my_tok * (NUM_TOPK * ACT_STRIDE) +
                                topk_slot * ACT_STRIDE + act_n;
            st_wt_u32((void *)&d_swiglu_out[out_idx],
                      (unsigned)a0 | ((unsigned)a1 << 16));
          } else {
            // Tail: the packed store would run past W13_OUTPUT_SIZE, so fall
            // back to the per-element path for the one or two lanes that fit.
            for (int i = 0; i < 4; i += 2) {
              int out_n = t0_out_n_pf + i;
              if (out_n + 1 < W13_OUTPUT_SIZE) {
                unsigned bt_g =
                    (unsigned)d_w13_bias[expert_id * W13_OUTPUT_SIZE + out_n]
                    << 16;
                unsigned bt_u = (unsigned)d_w13_bias[expert_id *
                                                         W13_OUTPUT_SIZE +
                                                     out_n + 1]
                                << 16;
                float bias_g, bias_u;
                __builtin_memcpy(&bias_g, &bt_g, 4);
                __builtin_memcpy(&bias_u, &bt_u, 4);
                float activated =
                    fast_swigluoai(acc[i] + bias_g, acc[i + 1] + bias_u);
                int out_idx = my_tok * (NUM_TOPK * ACT_STRIDE) +
                              topk_slot * ACT_STRIDE + out_n / 2;
                st_wt_u16(&d_swiglu_out[out_idx],
                          _gang_float_to_bf16(activated));
              }
            }
          }
#else
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
              int out_idx = my_tok * (NUM_TOPK * ACT_STRIDE) +
                            topk_slot * ACT_STRIDE + act_n;
              st_wt_u16(&d_swiglu_out[out_idx], _gang_float_to_bf16(activated));
            }
          }
#endif
        }
      }

      // ── tile_iter=1: drain per-wave loads + scales → MFMA ──────────────
      if (W13_TILES_PER_WAVE > 1) {
        // Drain per-wave buffer_load_lds writes (issued before SwiGLU above).
        // Under MPK_W13_T1_SPLIT_LDS_STAGE this drains the 11-chunk prefix and
        // the 13-chunk suffix together. Splitting this into a counted
        // `vmcnt(13)` placed *before* the epilogue only pays when the epilogue
        // consumes an asm-prefetched bias whose wait would otherwise be
        // unbounded; the bias is read through the compiler here, so there is
        // no earlier wait to make cheaper -- see the note at the epilogue
        // about the wait the compiler inserts for it.
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

        // Per-wave scale load + scatter (each wave loads only its own tile's
        // scales)
#ifdef MPK_W13_T1_EARLY_SCALE_LOAD
        // The reads already happened next to the tile-1 data loads and landed
        // under the `vmcnt(0)` above, so only the LDS scatter is left.
        {
          i32x4_t *dst_sc = (i32x4_t *)(lds_w13_base +
                                        warp_id * W13_TILE_BYTES +
                                        W13_TILE_DATA_PADDED);
#pragma unroll
          for (int j = 0; j < W13_T1_SC_LPT_WAVE; j++) {
            int idx = lane_id + j * 64;
            if (idx < W13_T1_SC_DW4_PER_TILE) {
              dst_sc[idx] = w13_t1_sc_wave[j];
            }
          }
        }
#else
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
#endif // MPK_W13_T1_EARLY_SCALE_LOAD
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#ifdef MPK_W13_T1_SPLIT_CMP
        // Used with MPK_W13_T1_SPLIT_STAGE_PROBE, where the ordinary 23-chunk
        // load has filled the tile slot and the early prefix has filled the
        // stage. The two must agree byte-for-byte over the staged rows; this
        // reports the first row that does not, which says whether the staged
        // prefix lands where its reader expects. Debug only -- the printf
        // serializes the wave.
        // Verifies the assembled tile-1 row set against HBM directly, so it
        // covers the *suffix* too. MPK_W13_T1_SPLIT_CMP alone runs under
        // STAGE_PROBE, where the suffix is suppressed and the ordinary
        // 23-chunk load supplies the tile slot -- so it has only ever
        // validated the prefix. Rows come from whichever buffer the MFMA
        // reader would pick, which is the property that actually matters.
        if (lane_id == 0) {
          uint8_t const *src = wg_data + (warp_id + NUM_WAVES) * W13_TILE_ROWS *
                                             (W13_K / 2);
          int bad_row = -1, bad_byte = -1;
          for (int r = 0; r < W13_TILE_ROWS && bad_row < 0; r++) {
            uint8_t const *dst =
                (r < W13_T1_STAGED_ROWS
                     ? lds_w13_t1_stage + warp_id * W13_T1_STAGE_BYTES
                     : lds_w13_base + warp_id * W13_TILE_BYTES) +
                r * (W13_K / 2);
            uint8_t const *ref = src + r * (W13_K / 2);
            for (int b = 0; b < (W13_K / 2) && bad_row < 0; b++) {
              if (dst[b] != ref[b]) {
                bad_row = r;
                bad_byte = b;
              }
            }
          }
          if (bad_row >= 0) {
            printf("[W13_SPLIT_HBM] tile=%d warp=%d bad_row=%d bad_byte=%d "
                   "staged_rows=%d staged_bytes=%d\n",
                   global_tile, warp_id, bad_row, bad_byte,
                   (int)W13_T1_STAGED_ROWS, (int)W13_T1_STAGED_DATA_BYTES);
          }
        }
        // Only meaningful under STAGE_PROBE, where the ordinary 23-chunk
        // load fills the tile slot alongside the staged prefix. In a real
        // split build the tile slot holds only the suffix, so comparing it
        // against the prefix is a guaranteed false positive.
#ifdef MPK_W13_T1_SPLIT_STAGE_PROBE
        if (lane_id == 0) {
          uint8_t const *tile_p = lds_w13_base + warp_id * W13_TILE_BYTES;
          uint8_t const *stage_p =
              lds_w13_t1_stage + warp_id * W13_T1_STAGE_BYTES;
          int bad = -1;
          for (int b = 0; b + 15 < W13_T1_STAGED_DATA_BYTES && bad < 0;
               b += 16) {
            uint4 t = *(uint4 const *)(tile_p + b);
            uint4 s = *(uint4 const *)(stage_p + b);
            if (t.x != s.x || t.y != s.y || t.z != s.z || t.w != s.w) {
              bad = b;
            }
          }
          if (bad >= 0) {
            printf("[W13_SPLIT_CMP] tile=%d warp=%d first_mismatch=%d "
                   "staged=%d tile_off=%u stage_off=%u\n",
                   global_tile, warp_id, bad, (int)W13_T1_STAGED_DATA_BYTES,
                   (unsigned)(uintptr_t)tile_p, (unsigned)(uintptr_t)stage_p);
          }
        }
#endif
#endif
        // No __syncthreads needed — each wave only reads from its own LDS tile

        // tile_iter=1 MFMA loop — reads weights from LDS
        {
          uint8_t *lds_w13_tile = lds_w13_base + warp_id * W13_TILE_BYTES;
          uint8_t *lds_w13_data = lds_w13_tile;
#if defined(MPK_W13_T1_SPLIT_LDS_STAGE) &&                                     \
    (!defined(MPK_W13_T1_SPLIT_STAGE_PROBE) ||                                 \
     defined(MPK_W13_T1_SPLIT_READ_PROBE))
          // The staged prefix ends on a whole row, so a lane's row lives
          // entirely in one of the two buffers and the choice is made once,
          // outside the loop -- no per-iteration pointer switch. Rows
          // [0, W13_T1_STAGED_ROWS) are in the stage buffer at their natural
          // offset; the rest stayed in the tile slot, also at their natural
          // offset, because the suffix landed at W13_T1_STAGED_DATA_BYTES.
          if (col < W13_T1_STAGED_ROWS) {
            lds_w13_data = lds_w13_t1_stage + warp_id * W13_T1_STAGE_BYTES;
          }
#endif
          uint8_t *lds_w13_sc = lds_w13_tile + W13_TILE_DATA_PADDED;
          int w_row_local = col;
          int const row_data_base = w_row_local * (W13_K / 2);
          int const row_scale_base = w_row_local * W13_NUM_BLK32;

          int wave_tile_1 = warp_id + NUM_WAVES; // tile_iter=1
          f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

#if defined(MPK_W13_BIAS_PREFETCH) && !defined(MPK_W13_BIAS_PF_T0_ONLY)
#define MPK_W13_BIAS_PF_T1 1
#endif
#ifdef MPK_W13_BIAS_PF_T1
          // Tile-1's bias, hoisted above its MFMA block for the same reason as
          // tile-0's; see there for why this goes through the compiler rather
          // than inline asm.
          uint2 w13_t1_bias_pf = {0u, 0u};
          int const t1_out_n_pf =
              wg_idx * W13_OUTPUT_PER_WG + wave_tile_1 * 16 + g * 4;
          if (tok_active && t1_out_n_pf + 3 < W13_OUTPUT_SIZE) {
            typedef unsigned int u32x2_t __attribute__((ext_vector_type(2)));
            auto const *bias_address =
                (u32x2_t const __attribute__((address_space(1))) *)&d_w13_bias
                    [expert_id * W13_OUTPUT_SIZE + t1_out_n_pf];
            u32x2_t const v = *bias_address;
            w13_t1_bias_pf.x = v.x;
            w13_t1_bias_pf.y = v.y;
          }
#endif

          // Depth-2 pipelined FP8 MFMA loop (tile_iter=1, full asm)
          {
            unsigned w_addr =
                (unsigned)(uintptr_t)(lds_w13_data + row_data_base + g * 16);
            unsigned ws_addr =
                (unsigned)(uintptr_t)(lds_w13_sc + row_scale_base + g);
            unsigned t_addr = (unsigned)(uintptr_t)(b_tok + g * 16);
            unsigned ts_addr = (unsigned)(uintptr_t)(b_scl);

            asm volatile(
                // ── Two disjoint operand banks ──
                //   Bank 0: A v[22:25], A scale v7,  B v[8:15],  B scale v16
                //   Bank 1: A v[26:29], A scale v18, B v[32:39], B scale v19
                //   Address scratch v17, accumulator a[0:3].
                //
                // Prefetching into the registers the current MFMA reads is a
                // WAR race: lgkmcnt tracks when LDS data lands in the VGPR, not
                // when the MFMA finished sampling its operands, and a 16x16x128
                // MFMA streams them over the op rather than latching at issue.
                // When LDS returns fast the write-back lands mid-MFMA and the
                // op sees mixed-iteration operands (~17-22% of launches before
                // banking). Ping-pong: while the MFMA consumes bank X, prefetch
                // writes bank 1-X, so no register is ever both a live source
                // and an in-flight LDS destination.
                //
                // Verified by tests/standalone/test_mfma_pipeline_hazards.hip.
                "v_accvgpr_write_b32 a0, 0\n"
                "v_accvgpr_write_b32 a1, 0\n"
                "v_accvgpr_write_b32 a2, 0\n"
                "v_accvgpr_write_b32 a3, 0\n"
#ifdef MPK_W13_DUAL_ACC
                // Odd K blocks accumulate into a disjoint second chain.
                "v_accvgpr_write_b32 a4, 0\n"
                "v_accvgpr_write_b32 a5, 0\n"
                "v_accvgpr_write_b32 a6, 0\n"
                "v_accvgpr_write_b32 a7, 0\n"
#endif

                // Pre-issue 5 reads for iteration 0 into bank 0
                "ds_read_b128 v[22:25], %[wa]\n"
                "ds_read_u8   v7, %[wsa]\n"
                "ds_read_b128 v[8:11], %[ta]\n"
                "ds_read_b128 v[12:15], %[ta] offset:64\n"
                "ds_read_u8   v16, %[tsa]\n"
                "s_mov_b32 s13, 0\n"
#ifdef MPK_MFMA_PINGPONG_SCHED
                // Identical schedule to the T0 loop above; see the state
                // accounting in the block comment there.
                "s_waitcnt lgkmcnt(0)\n"

                "PIPELINED_W13_T1_%=:\n"
                "s_cmpk_lt_i32 s13, %[iters_m1]\n"
                "s_cbranch_scc0 W13_T1_FINAL_%=\n"

                // ---- prefetch bank 1, then consume bank 0 ----
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
                // 5 VALU + 16 + 11 = 32 states before MFMA_B.
                "v_add_u32_e32 %[wa], 64, %[wa]\n"
                "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
                "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
                "s_add_i32 s13, s13, 1\n"
                "v_add_u32_e32 v17, s13, %[tsa]\n"
#ifdef MPK_W13_DUAL_ACC
                // Disjoint chains: 5 issued updates + s_nop 2 = 8 states.
                "s_nop 2\n"
#else
                "s_nop 15\n"
                "s_nop 10\n"
#endif
                "s_waitcnt lgkmcnt(0)\n"

            // ---- consume bank 1, then refill bank 0 ----
#ifdef MPK_W13_DUAL_ACC
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], "
                "v[32:39], a[4:7], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
#else
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], "
                "v[32:39], a[0:3], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
#endif
                "ds_read_u8   v16, v17\n"
                "ds_read_b128 v[22:25], %[wa]\n"
                "ds_read_u8   v7, %[wsa]\n"
                "ds_read_b128 v[8:11], %[ta]\n"
                "ds_read_b128 v[12:15], %[ta] offset:64\n"
            // 16 + 12 = 28, plus 4 VALU next trip = 32.
#ifdef MPK_W13_DUAL_ACC
                // Disjoint chains: s_nop 3 + 4 updates next trip = 8 states.
                "s_nop 3\n"
#else
                "s_nop 15\n"
                "s_nop 11\n"
#endif
                "s_waitcnt lgkmcnt(0)\n"
                "s_branch PIPELINED_W13_T1_%=\n"

                "W13_T1_FINAL_%=:\n"
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
                "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
#else

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
#ifdef MPK_W13_DUAL_ACC
                // Odd K blocks onto the second chain; see the T0 body.
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], "
                "v[32:39], a[4:7], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
#else
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], "
                "a[0:3], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
#endif
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
#ifdef MPK_W13_DUAL_ACC
                // Short chain-0 reuse out of the loop; see the T0 tail.
                "s_nop 4\n"
#endif
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
                "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
                "s_branch W13_T1_ACC_%=\n"

                "W13_T1_TAIL_B1_%=:\n"
                "s_waitcnt lgkmcnt(0)\n"
#ifdef MPK_W13_DUAL_ACC
                // Bank-1 final belongs to the odd chain; see the T0 tail.
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], "
                "v[32:39], a[4:7], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
#else
                "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], "
                "a[0:3], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
#endif
#endif

                "W13_T1_ACC_%=:\n"
                // 32 clocks: the scaled MFMA is a 32-cycle op on CDNA4. The old
                // "s_nop 7; s_nop 0" was 9 clocks (correct only for a 4-pass
                // MFMA) and returned a partially-retired accumulator every
                // time.
                "s_nop 15\n"
                "s_nop 15\n"
#ifdef MPK_W13_DUAL_ACC
                // Bank-0 inputs are dead here; stage the odd partials there.
                "v_accvgpr_read_b32 v22, a4\n"
                "v_accvgpr_read_b32 v23, a5\n"
                "v_accvgpr_read_b32 v24, a6\n"
                "v_accvgpr_read_b32 v25, a7\n"
#endif
                "v_accvgpr_read_b32 %[acc0], a0\n"
                "v_accvgpr_read_b32 %[acc1], a1\n"
                "v_accvgpr_read_b32 %[acc2], a2\n"
                "v_accvgpr_read_b32 %[acc3], a3\n"
#ifdef MPK_W13_DUAL_ACC
                "v_add_f32_e32 %[acc0], v22, %[acc0]\n"
                "v_add_f32_e32 %[acc1], v23, %[acc1]\n"
                "v_add_f32_e32 %[acc2], v24, %[acc2]\n"
                "v_add_f32_e32 %[acc3], v25, %[acc3]\n"
#endif
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
                  "a3"
#ifdef MPK_W13_DUAL_ACC
                  ,
                  "a4",
                  "a5",
                  "a6",
                  "a7"
#endif
            );
          }
          // tile_iter=1 SwiGLU epilogue
          if (tok_active) {
            constexpr int ACT_STRIDE = W13_OUTPUT_SIZE / 2;
#ifdef MPK_W13_BIAS_PF_T1
            // Same collapse as tile-0: the loop's two iterations write adjacent
            // bf16 lanes of one activation row, so they are a single aligned
            // dword, and both biases are already in registers.
            if (t1_out_n_pf + 3 < W13_OUTPUT_SIZE) {
              unsigned const bt_g0 = (w13_t1_bias_pf.x & 0xFFFFu) << 16;
              unsigned const bt_u0 = w13_t1_bias_pf.x & 0xFFFF0000u;
              unsigned const bt_g1 = (w13_t1_bias_pf.y & 0xFFFFu) << 16;
              unsigned const bt_u1 = w13_t1_bias_pf.y & 0xFFFF0000u;
              float bias_g0, bias_u0, bias_g1, bias_u1;
              __builtin_memcpy(&bias_g0, &bt_g0, 4);
              __builtin_memcpy(&bias_u0, &bt_u0, 4);
              __builtin_memcpy(&bias_g1, &bt_g1, 4);
              __builtin_memcpy(&bias_u1, &bt_u1, 4);
              unsigned short const a0 = _gang_float_to_bf16(
                  fast_swigluoai(acc[0] + bias_g0, acc[1] + bias_u0));
              unsigned short const a1 = _gang_float_to_bf16(
                  fast_swigluoai(acc[2] + bias_g1, acc[3] + bias_u1));
              int const act_n = t1_out_n_pf / 2;
              int const out_idx = my_tok * (NUM_TOPK * ACT_STRIDE) +
                                  topk_slot * ACT_STRIDE + act_n;
              st_wt_u32((void *)&d_swiglu_out[out_idx],
                        (unsigned)a0 | ((unsigned)a1 << 16));
            } else
#endif
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
                int out_idx = my_tok * (NUM_TOPK * ACT_STRIDE) +
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
          i32x8_t b = _gang_load_fp8_mfma_b(b_tok, ki * K_PER_MFMA, g);
          int sb = (int)b_scl[ki];
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
              _gang_load_fp8_mfma_b(b_tok, (ki + 1) * K_PER_MFMA, g);
          int sb = (int)b_scl[ki + 1];
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
              _gang_load_fp8_mfma_b(b_tok, (ki + 2) * K_PER_MFMA, g);
          int sb = (int)b_scl[ki + 2];
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
              _gang_load_fp8_mfma_b(b_tok, (ki + 3) * K_PER_MFMA, g);
          int sb = (int)b_scl[ki + 3];
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
              _gang_load_fp8_mfma_b(b_tok, (ki + 4) * K_PER_MFMA, g);
          int sb = (int)b_scl[ki + 4];
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
              _gang_load_fp8_mfma_b(b_tok, (ki + 5) * K_PER_MFMA, g);
          int sb = (int)b_scl[ki + 5];
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
              _gang_load_fp8_mfma_b(b_tok, (ki + 6) * K_PER_MFMA, g);
          int sb = (int)b_scl[ki + 6];
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
              _gang_load_fp8_mfma_b(b_tok, (ki + 7) * K_PER_MFMA, g);
          int sb = (int)b_scl[ki + 7];
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
      if (tok_active) {
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
            int out_idx = my_tok * (NUM_TOPK * ACT_STRIDE) +
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

    } // end W13 compute region
  w13_arrive:
#ifdef MPK_MOE_INNER_TIMING
    // Declared at the label, not before the goto: a jump may not cross an
    // initialization, and the empty-slot path jumps straight here.
    unsigned long long _mt1 = __builtin_amdgcn_s_memrealtime();
#endif
    // ── Mechanism C W13 signal (producer side)
    // ──────────────────────────────── Uses layer index from shared memory for
    // monotonically increasing release
    MOE_DBG_SUBPHASE(2001);
    MPK_WS_MARK(8201, global_tile); // W13 done, arriving at barrier
    if (tid == 0) {
      // Indexed by *slot*, not expert_id: the id comes from the shared mask and
      // can differ between two workers straddling a layer boundary, which would
      // split one slot's arrivals across two counters. The slot is derived from
      // the compile-time tile space, so every worker agrees on it.
      int base = expert_idx * MOE_BAR_STRIDE;
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
      // Exact only if every tile counted by W13_TILES arrives here. Under
      // packing W13_TILES == W13_WGS and no tile returns between the decode
      // and this line, so it is. The pre-packing decode had a token axis and
      // a `route_val == 0` early return, which at bs>1 made the count fall
      // short by the number of unrouted (expert, token) pairs -- and the W2
      // workers below spun forever waiting for a release that never fired.
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

#ifdef MPK_MOE_INNER_TIMING
    // W13 arm. `arrive` is the fence + arrival atomic + (on the last arrival)
    // the eight-slot release fan-out, which is why it is worth separating from
    // the GEMM: it is the producer half of the same barrier the W2 arm waits on.
    // MPK_MOE_INNER_WIDE widens the sample from tile 0 to every 37th tile.
    // Tile 0 is the *earliest* worker, so its barrier wait is an upper bound
    // on skew rather than a population figure; 37 is coprime with both arms'
    // tile counts so the stride walks all residues.
#ifdef MPK_MOE_INNER_WIDE
    if (tid == 0 && (global_tile % 37) == 0) {
#else
    if (tid == 0 && global_tile == 0) {
#endif
      unsigned long long _mt2 = __builtin_amdgcn_s_memrealtime();
      printf("[MOE_INNER] arm=w13 dec=%.2f compute=%.2f arrive=%.2f total=%.2f\n",
             (double)(_mt0 - _mtE) * 10.0 / 1000.0,
             (double)(_mt1 - _mt0) * 10.0 / 1000.0,
             (double)(_mt2 - _mt1) * 10.0 / 1000.0,
             (double)(_mt2 - _mtE) * 10.0 / 1000.0);
    }
#endif

    return;
  }

  // ══════════════════════════════════════════════════════════════════════════
  // PHASE 1: W2 (down projection) → write BF16 to mlp_out
  // ══════════════════════════════════════════════════════════════════════════

  // No routed token: nothing to reduce, and unlike W13 there is no arrival to
  // preserve (the W2 epilogue is an atomicAdd into the workspace, not a
  // barrier), so this can return outright instead of jumping past the compute.
  // It must return *before* the W13->W2 poll below: an empty slot's W13 tiles
  // reach the arrival but write no output, so waiting on that release would be
  // waiting to read nothing.
  if (n_tok == 0) {
    MPK_WS_MARK(8106, global_tile); // exit: W2 tile with no routed token
    return;
  }

  // Shared memory layout: FP8 tokens + per-MFMA-tile scales, TOK_ROWS rows.
  // Same +16 row pad as W13 -- see the note there for why it is load-bearing.
  constexpr int W2_TOTAL_MFMA = W2_K / K_PER_MFMA;
  constexpr int W2_TOK_ROW_STRIDE = W2_K + 16;
  constexpr int W2_SC_STRIDE = ((W2_MFMA_ITERS + 3) / 4) * 4;
  constexpr int W2_TOK_REGION = TOK_ROWS * W2_TOK_ROW_STRIDE;
  constexpr int W2_SC_REGION = TOK_ROWS * W2_SC_STRIDE;
  uint8_t *s_tok_fp8 = (uint8_t *)_fused_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + W2_TOK_REGION;

  // Per-lane B operand base; inactive lanes clamp to row 0 rather than
  // skipping. See the W13 note.
  uint8_t *b_tok =
      s_tok_fp8 +
      (TOK_ROWS == 1 ? 0 : (tok_active ? col : 0) * W2_TOK_ROW_STRIDE);
  uint8_t *b_scl =
      s_tok_scales +
      (TOK_ROWS == 1 ? 0 : (tok_active ? col : 0) * W2_SC_STRIDE);

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

  constexpr int LDS_W2_OFF = ((W2_TOK_REGION + W2_SC_REGION + 15) / 16) * 16;
  // 155 KB minus AMD's 3 KB reservation minus the layer-index word. The old
  // bound was 155*1024, wrong by 3 KB in the permissive direction.
  static_assert(LDS_W2_OFF + W2_TILE_BYTES * NUM_WAVES <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
                        mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END,
                "W2 LDS weight tiles exceed MI350X LDS budget");
  uint8_t *lds_w2_base = (uint8_t *)_fused_smem + LDS_W2_OFF;

  MOE_DBG_SUBPHASE(3001);
  // All threads independently read layer_idx from LDS (uniform value).
  // Eliminates shared variable and __syncthreads broadcast.
  // Indexed by *slot*, not expert_id: the id comes from the shared mask and
  // can differ between two workers straddling a layer boundary, which would
  // split one slot's arrivals across two counters. The slot is derived from
  // the compile-time tile space, so every worker agrees on it.
  int base = expert_idx * MOE_BAR_STRIDE;
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
#ifdef MPK_W2_LINEAR_LOAD
  // Linear per-wave tile load, same transform as the W13 T0 site above and
  // for the same reason: W2's geometry is identical (W2_K == W13_K == 2880,
  // so W2_LPT is also 6 and the striped cover also fetches 24 KiB for 22.5
  // KiB of data). One tile per wave in 23 1-KiB chunks fetches 23 KiB and
  // needs 2 address registers instead of 48. LDS contents are unchanged over
  // the real region -- only the clamped padding tail differs, and the MFMA
  // never reads it.
  {
    constexpr int W2_T0_CHUNKS = (W2_TILE_DATA + 1023) / 1024;
    static_assert(W2_T0_CHUNKS == 23,
                  "the linear W2 T0 sequence hardcodes .rept 22 below; "
                  "W2_K changed, so re-derive the chunk count");
    static_assert(NUM_WAVES == 4, "linear load assigns one W2 tile per wave");
    unsigned const lds_w2_t0_base = __builtin_amdgcn_readfirstlane(
        (unsigned)(uintptr_t)(lds_w2_base + warp_id * W2_TILE_BYTES));
    uint32_t w2_voff =
        w2_wg_voff_base +
        static_cast<uint32_t>(warp_id * W2_TILE_ROWS * (W2_K / 2)) +
        static_cast<uint32_t>(lane_id * 16);
    asm volatile(
        "s_nop 4\n"
        "s_mov_b32 m0, %[lds_base]\n"
        "s_nop 0\n"
        "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
        ".rept 22\n"
        "s_addk_i32 m0, 0x400\n"
        "v_add_u32_e32 %[voff], 0x400, %[voff]\n"
        "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
        ".endr\n"
        : [voff] "+v"(w2_voff)
        : [rsrc] "s"(w2_rsrc), [lds_base] "s"(lds_w2_t0_base)
        : "memory", "m0");
  }
#else
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
#endif // MPK_W2_LINEAR_LOAD

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

#ifdef MPK_MOE_INNER_TIMING
  // Taken after the weight prefetch is issued and before the poll, so `prefetch`
  // below is issue cost only -- the HBM latency it hides lands in `barrier`.
  unsigned long long _mt1 = __builtin_amdgcn_s_memrealtime();
#endif

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
    while ((_obs = MPK_LD_GATE2(&d_barrier[base + xcd_id * MOE_BAR_LINE])) <
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
#ifdef MPK_MOE_INNER_TIMING
  unsigned long long _mt2 = __builtin_amdgcn_s_memrealtime();
#endif

  // No buffer_inv needed — NT loads bypass L2 entirely.

  // FP8 quant of SwiGLU output — writes to LDS[0..W2_K+scales]
  // buffer_load_lds writes to LDS[W2_OFF..] — no conflict, both in flight.
  MOE_DBG_SUBPHASE(3003);
  MPK_WS_MARK(8303, global_tile); // W2: FP8 quant of SwiGLU output
  {
    if constexpr (PACK_N && !SINGLE_TOK) {
      // s_row_off already holds tok*(NUM_TOPK*INTERMEDIATE) +
      // slot*INTERMEDIATE for this phase (`is_w2` picked the formula at
      // compaction time). NT loads: the SwiGLU output was just written by
      // another XCD and will not be reused.
      _gang_multirow_fp8_quant_gather<W2_K, TOK_ROWS, W2_TOK_ROW_STRIDE,
                                      W2_SC_STRIDE, /*NT_LOAD=*/true>(
          d_swiglu_out, s_row_off, n_tok, s_tok_fp8, s_tok_scales);
    } else {
      unsigned short const *w2_input_base =
          d_swiglu_out + my_tok * (NUM_TOPK * INTERMEDIATE_SIZE) +
          topk_slot * INTERMEDIATE_SIZE;
#ifdef MPK_WIDE_FP8_QUANT
      _gang_wave_parallel_fp8_quant_nt_wide<W2_K>(
          w2_input_base, s_tok_fp8, s_tok_scales);
#else
      _gang_wave_parallel_fp8_quant_nt<W2_K>(
          w2_input_base, s_tok_fp8, s_tok_scales);
#endif
    }
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
    // Must match LDS_W2_OFF above -- the DMA writes where that says and the
    // MFMA reads where this says.
    constexpr int LDS_W2_OFF_L = LDS_W2_OFF;
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
      // Guard on the token, not col == 0: every lane holds live results now.
      // The routing weight becomes a 16-lane gather (exactly the loads the 16
      // separate per-token tiles used to issue); the bias depends only on the
      // output column, so the 16 lanes load the same bytes and coalesce.
      if (tok_active && out_n_base < W2_OUTPUT_SIZE) {
        float const *rw_ptr = &d_routing_weight[my_tok * NUM_TOPK + topk_slot];
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
        unsigned w2_t_addr = (unsigned)(uintptr_t)(b_tok + g * 16);
        unsigned w2_ts_addr = (unsigned)(uintptr_t)(b_scl);
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
#ifdef MPK_W2_QUAD_ACC
            // K block k accumulates into chain k % 4.
            "v_accvgpr_write_b32 a4, 0\n"
            "v_accvgpr_write_b32 a5, 0\n"
            "v_accvgpr_write_b32 a6, 0\n"
            "v_accvgpr_write_b32 a7, 0\n"
            "v_accvgpr_write_b32 a8, 0\n"
            "v_accvgpr_write_b32 a9, 0\n"
            "v_accvgpr_write_b32 a10, 0\n"
            "v_accvgpr_write_b32 a11, 0\n"
            "v_accvgpr_write_b32 a12, 0\n"
            "v_accvgpr_write_b32 a13, 0\n"
            "v_accvgpr_write_b32 a14, 0\n"
            "v_accvgpr_write_b32 a15, 0\n"
#endif

            // Pre-issue 5 reads for iteration 0 into bank 0
            "ds_read_b128 v[22:25], %[wa]\n"
            "ds_read_u8   v7, %[wsa]\n"
            "ds_read_b128 v[8:11], %[ta]\n"
            "ds_read_b128 v[12:15], %[ta] offset:64\n"
            "ds_read_u8   v16, %[tsa]\n"
#ifndef MPK_W2_T0_MFMA_UNROLLED
            "s_mov_b32 s13, 0\n"
#endif
#ifdef MPK_MFMA_PINGPONG_SCHED
#ifdef MPK_W2_QUAD_ACC
            // Four independent chains, four K blocks per trip. No two
            // adjacent MFMAs share SrcC and no chain is reused within a trip,
            // so every edge is the 8-state cadence. 5 trips cover K blocks
            // 0..19 and the tail covers 20..22 on chains 0/1/2 -- chain k % 4
            // throughout.
            "s_waitcnt lgkmcnt(0)\n"

            "PIPELINED_W2_T0_%=:\n"
            "s_cmpk_lt_i32 s13, %[iters_m3]\n"
            "s_cbranch_scc0 W2_T0_FINAL_%=\n" MPK_W2_QUAD_PAIR("a[0:3]",
                                                               "a[4:7]")
                MPK_W2_QUAD_PAIR(
                    "a[8:11]",
                    "a[12:15]") "s_branch PIPELINED_W2_T0_%=\n"

                                "W2_T0_FINAL_%=:\n" MPK_W2_QUAD_PAIR(
                                    "a[0:3]",
                                    "a[4:7]") "v_mfma_scale_f32_16x16x128_"
                                              "f8f6f4 a[8:11], v[22:25], "
                                              "v[8:15], "
                                              "a[8:11], v7, v16 "
                                              "op_sel_hi:[0,0,0] cbsz:4\n"
#else
            // Identical schedule to the W13 T0 loop; see the state accounting
            // in the block comment there.
            "s_waitcnt lgkmcnt(0)\n"

            "PIPELINED_W2_T0_%=:\n"
            "s_cmpk_lt_i32 s13, %[iters_m1]\n"
            "s_cbranch_scc0 W2_T0_FINAL_%=\n"

            // ---- prefetch bank 1, then consume bank 0 ----
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
            // 5 VALU + 16 + 11 = 32 states before MFMA_B.
            "v_add_u32_e32 %[wa], 64, %[wa]\n"
            "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
            "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
            "s_add_i32 s13, s13, 1\n"
            "v_add_u32_e32 v17, s13, %[tsa]\n"
            "s_nop 15\n"
            "s_nop 10\n"
            "s_waitcnt lgkmcnt(0)\n"

            // ---- consume bank 1, then refill bank 0 ----
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], "
            "a[0:3], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
            "ds_read_u8   v16, v17\n"
            "ds_read_b128 v[22:25], %[wa]\n"
            "ds_read_u8   v7, %[wsa]\n"
            "ds_read_b128 v[8:11], %[ta]\n"
            "ds_read_b128 v[12:15], %[ta] offset:64\n"
            // 16 + 12 = 28, plus 4 VALU next trip = 32.
            "s_nop 15\n"
            "s_nop 11\n"
            "s_waitcnt lgkmcnt(0)\n"
            "s_branch PIPELINED_W2_T0_%=\n"

            "W2_T0_FINAL_%=:\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
            "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
#endif // MPK_W2_QUAD_ACC
#else

// ── MPK_W2_T0_MFMA_UNROLLED ──
// Same transform as the W13 T0 loop above; see the rationale there.
#ifdef MPK_W2_T0_MFMA_UNROLLED
            ".set MPK_W2_T0_SC_%=, 1\n"
            ".rept %c[unroll_pairs]\n"
#else
            "PIPELINED_W2_T0_%=:\n"
#endif
            // ---- consume bank 0, prefetch into bank 1 ----
            "s_waitcnt lgkmcnt(0)\n"
            "v_add_u32_e32 %[wa], 64, %[wa]\n"
            "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
            "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
#ifdef MPK_W2_T0_MFMA_UNROLLED
            "v_add_u32_e32 v17, MPK_W2_T0_SC_%=, %[tsa]\n"
#else
            "s_add_i32 s13, s13, 1\n"
            "v_add_u32_e32 v17, s13, %[tsa]\n"
#endif
            "ds_read_u8   v19, v17\n"
            "ds_read_b128 v[26:29], %[wa]\n"
            "ds_read_u8   v18, %[wsa]\n"
            "ds_read_b128 v[32:35], %[ta]\n"
            "ds_read_b128 v[36:39], %[ta] offset:64\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
            "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
#ifndef MPK_W2_T0_MFMA_UNROLLED
            "s_cmpk_lt_i32 s13, %[iters_m1]\n"
            "s_cbranch_scc0 W2_T0_TAIL_B1_%=\n"
#endif

            // ---- consume bank 1, prefetch into bank 0 ----
            "s_waitcnt lgkmcnt(0)\n"
            "v_add_u32_e32 %[wa], 64, %[wa]\n"
            "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
            "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
#ifdef MPK_W2_T0_MFMA_UNROLLED
            "v_add_u32_e32 v17, MPK_W2_T0_SC_%= + 1, %[tsa]\n"
#else
            "s_add_i32 s13, s13, 1\n"
            "v_add_u32_e32 v17, s13, %[tsa]\n"
#endif
            "ds_read_u8   v16, v17\n"
            "ds_read_b128 v[22:25], %[wa]\n"
            "ds_read_u8   v7, %[wsa]\n"
            "ds_read_b128 v[8:11], %[ta]\n"
            "ds_read_b128 v[12:15], %[ta] offset:64\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], "
            "a[0:3], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
#ifdef MPK_W2_T0_MFMA_UNROLLED
            ".set MPK_W2_T0_SC_%=, MPK_W2_T0_SC_%= + 2\n"
            ".endr\n"

            // ── Final MFMA (bank 0) ──
            // (W2_MFMA_ITERS-1)/2 pairs consumed an even number of K blocks,
            // so the odd one out is always block W2_MFMA_ITERS-1, whose
            // operands the last pair prefetched into bank 0.
            "s_waitcnt lgkmcnt(0)\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
            "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
#else
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
#endif // MPK_W2_T0_MFMA_UNROLLED
#endif

                                              "W2_T0_ACC_%=:\n"
#ifndef MPK_W2_EPI_OVERLAP
                                              // 32 clocks: the scaled MFMA is a
                                              // 32-cycle op on CDNA4. The old
                                              // "s_nop 7; s_nop 0" was 9 clocks
                                              // (correct only for a 4-pass
                                              // MFMA) and returned a
                                              // partially-retired accumulator
                                              // every time.
                                              //
                                              // MPK_MOE_W2_EPILOGUE_OVERLAP
                                              // fills this wait with the
                                              // reduction of the three chains
                                              // that already retired; see the
                                              // note by MPK_W2_EPI_OVERLAP for
                                              // the state accounting that makes
                                              // dropping the pad safe.
                                              "s_nop 15\n"
                                              "s_nop 15\n"
#endif
#ifdef MPK_W2_QUAD_ACC
                                              // Merge order c0, c1, c3, c2 --
                                              // the sequence the
                                              // epilogue-overlap arm produces,
                                              // kept here too so enabling that
                                              // flag is a pure scheduling
                                              // change and not a numerical one.
                                              "v_accvgpr_read_b32 v32, a12\n"
                                              "v_accvgpr_read_b32 v33, a13\n"
                                              "v_accvgpr_read_b32 v34, a14\n"
                                              "v_accvgpr_read_b32 v35, a15\n"
#endif
                                              "v_accvgpr_read_b32 %[acc0], a0\n"
                                              "v_accvgpr_read_b32 %[acc1], a1\n"
                                              "v_accvgpr_read_b32 %[acc2], a2\n"
                                              "v_accvgpr_read_b32 %[acc3], a3\n"
#ifdef MPK_W2_QUAD_ACC
                                              "v_accvgpr_read_b32 v22, a4\n"
                                              "v_accvgpr_read_b32 v23, a5\n"
                                              "v_accvgpr_read_b32 v24, a6\n"
                                              "v_accvgpr_read_b32 v25, a7\n"
                                              "v_add_f32_e32 %[acc0], v22, "
                                              "%[acc0]\n"
                                              "v_add_f32_e32 %[acc1], v23, "
                                              "%[acc1]\n"
                                              "v_add_f32_e32 %[acc2], v24, "
                                              "%[acc2]\n"
                                              "v_add_f32_e32 %[acc3], v25, "
                                              "%[acc3]\n"
                                              "v_add_f32_e32 %[acc0], v32, "
                                              "%[acc0]\n"
                                              "v_add_f32_e32 %[acc1], v33, "
                                              "%[acc1]\n"
                                              "v_add_f32_e32 %[acc2], v34, "
                                              "%[acc2]\n"
                                              "v_add_f32_e32 %[acc3], v35, "
                                              "%[acc3]\n"
                                              "v_accvgpr_read_b32 v26, a8\n"
                                              "v_accvgpr_read_b32 v27, a9\n"
                                              "v_accvgpr_read_b32 v28, a10\n"
                                              "v_accvgpr_read_b32 v29, a11\n"
                                              "v_add_f32_e32 %[acc0], v26, "
                                              "%[acc0]\n"
                                              "v_add_f32_e32 %[acc1], v27, "
                                              "%[acc1]\n"
                                              "v_add_f32_e32 %[acc2], v28, "
                                              "%[acc2]\n"
                                              "v_add_f32_e32 %[acc3], v29, "
                                              "%[acc3]\n"
#endif
            : [acc0] "=v"(acc[0]),
              [acc1] "=v"(acc[1]),
              [acc2] "=v"(acc[2]),
              [acc3] "=v"(acc[3]),
              [wa] "+v"(w2_w_addr),
              [wsa] "+v"(w2_ws_addr),
              [ta] "+v"(w2_t_addr)
            : [tsa] "v"(w2_ts_addr),
              [iters_m1] "n"(W2_MFMA_ITERS - 1),
              // Trip bound for the quad body: 5 trips of 4, then a 3-MFMA tail.
              [iters_m3] "n"(W2_MFMA_ITERS - 3),
              [unroll_pairs] "n"((W2_MFMA_ITERS - 1) / 2)
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
              "a3"
#ifdef MPK_W2_QUAD_ACC
              ,
              "a4",
              "a5",
              "a6",
              "a7",
              "a8",
              "a9",
              "a10",
              "a11",
              "a12",
              "a13",
              "a14",
              "a15"
#endif
        );
      }

      MOE_DBG_SUBPHASE(3006);
      MPK_WS_MARK(8306, global_tile); // W2: epilogue
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      if (tok_active && out_n_base < W2_OUTPUT_SIZE) {
        unsigned bt0 = (pf_bias.x & 0xFFFFu) << 16;
        unsigned bt1 = pf_bias.x & 0xFFFF0000u;
        unsigned bt2 = (pf_bias.y & 0xFFFFu) << 16;
        unsigned bt3 = pf_bias.y & 0xFFFF0000u;
        float bv0, bv1, bv2, bv3;
        __builtin_memcpy(&bv0, &bt0, 4);
        __builtin_memcpy(&bv1, &bt1, 4);
        __builtin_memcpy(&bv2, &bt2, 4);
        __builtin_memcpy(&bv3, &bt3, 4);
        // Stores, not atomicAdd: this tile is the ONLY writer of
        // (my_tok, topk_slot, out_n_base..+3). A token routes to a given
        // expert at most once, so topk_slot is unique per (token, expert), and
        // wg_idx partitions the hidden axis across tiles. See moe_ws_layout.cuh
        // for why the shared per-token accumulator broke row symmetry at B>1.
        //
        // Write-through (sc0 sc1) is mandatory, not an optimization: the
        // consumer runs on a different XCD and MI300/MI350 L2 is not coherent
        // across XCDs, so a plain store would be invisible to it. The atomicAdd
        // this replaced went to the coherent point implicitly. `ws_base` is
        // 16B-aligned -- out_n_base steps by 4 floats and both HIDDEN_SIZE and
        // W2_OUTPUT_SIZE are multiples of 4 -- so all four lanes always store
        // and one dwordx4 covers them.
        int ws_base =
            moe_ws_offset(my_tok, topk_slot, HIDDEN_SIZE) + out_n_base;
        float4 wv = {(acc[0] + bv0) * pf_rw,
                     (acc[1] + bv1) * pf_rw,
                     (acc[2] + bv2) * pf_rw,
                     (acc[3] + bv3) * pf_rw};
        st_wt_f32x4(&d_workspace_f32[ws_base], wv);
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
        if (tok_active && out_n_base < W2_OUTPUT_SIZE) {
          float const *rw_ptr =
              &d_routing_weight[my_tok * NUM_TOPK + topk_slot];
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
          i32x8_t b = _gang_load_fp8_mfma_b(b_tok, kt, g);
          int sb = (int)b_scl[ki];
          acc = _gang_mfma_f4xf8(a, b, acc, sa, sb);
        }

        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        if (tok_active && out_n_base < W2_OUTPUT_SIZE) {
          unsigned bt0 = (pf_bias.x & 0xFFFFu) << 16;
          unsigned bt1 = pf_bias.x & 0xFFFF0000u;
          unsigned bt2 = (pf_bias.y & 0xFFFFu) << 16;
          unsigned bt3 = pf_bias.y & 0xFFFF0000u;
          float bv0, bv1, bv2, bv3;
          __builtin_memcpy(&bv0, &bt0, 4);
          __builtin_memcpy(&bv1, &bt1, 4);
          __builtin_memcpy(&bv2, &bt2, 4);
          __builtin_memcpy(&bv3, &bt3, 4);
          // Write-through store -- sole writer of this (token, slot, hidden)
          // range, and the consumer is on another XCD. See the tile_iter=0
          // epilogue above and moe_ws_layout.cuh.
          int ws_base =
              moe_ws_offset(my_tok, topk_slot, HIDDEN_SIZE) + out_n_base;
          float4 wv = {(acc[0] + bv0) * pf_rw,
                       (acc[1] + bv1) * pf_rw,
                       (acc[2] + bv2) * pf_rw,
                       (acc[3] + bv3) * pf_rw};
          st_wt_f32x4(&d_workspace_f32[ws_base], wv);
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

#ifdef MPK_MOE_INNER_TIMING
  // W2 arm. `prep` covers the decode, the weight-prefetch issue and the scale
  // staging; `barrier` is the wait on the W13 release, which is where the
  // prefetched weights' HBM latency is meant to be hidden; `compute` is the
  // quant + MFMA + epilogue.
  // TOTAL_W13, not W13_TILES: `is_w2` is `global_tile >= TOTAL_W13`, so this is
  // the first W2 tile in the same numbering the arm split uses.
#ifdef MPK_MOE_INNER_WIDE
  if (tid == 0 && ((global_tile - TOTAL_W13) % 37) == 0) {
#else
  if (tid == 0 && global_tile == TOTAL_W13) {
#endif
    unsigned long long _mt3 = __builtin_amdgcn_s_memrealtime();
    printf("[MOE_INNER] arm=w2 dec=%.2f prep=%.2f barrier=%.2f compute=%.2f "
           "total=%.2f\n",
           (double)(_mt0 - _mtE) * 10.0 / 1000.0,
           (double)(_mt1 - _mt0) * 10.0 / 1000.0,
           (double)(_mt2 - _mt1) * 10.0 / 1000.0,
           (double)(_mt3 - _mt2) * 10.0 / 1000.0,
           (double)(_mt3 - _mtE) * 10.0 / 1000.0);
  }
#endif

  // No barrier reset needed — all counters use monotonically increasing
  // expected values (per-XCD release = layer_idx + 1, global_arrive uses
  // modular check). Eliminates stale L2 issues across layers.
}

} // namespace kernel
