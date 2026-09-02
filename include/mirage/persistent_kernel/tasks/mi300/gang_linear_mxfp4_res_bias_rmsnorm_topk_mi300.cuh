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

#ifdef MPK_OPROJ_AMAX_DPP
// 8-lane amax butterfly (xor 1/2/4) via DPP, matching
// gang_moe_linear_mxfp4_mi300.cuh. __shfl_xor here is ds_bpermute.
__device__ __forceinline__ float _oproj_amax8_dpp(float amax) {
  float peer;
  asm volatile("s_nop 1\n"
               "v_mov_b32_dpp %1, %0 quad_perm:[1,0,3,2] row_mask:0xf "
               "bank_mask:0xf\n"
               "v_max_f32 %0, %0, %1\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %1, %0 quad_perm:[2,3,0,1] row_mask:0xf "
               "bank_mask:0xf\n"
               "v_max_f32 %0, %0, %1\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %1, %0 row_half_mirror row_mask:0xf "
               "bank_mask:0xf\n"
               "v_max_f32 %0, %0, %1"
               : "+v"(amax), "=&v"(peer));
  return amax;
}
#endif

#ifdef MPK_OPROJ_PIPE_SLICE_MFMA
#ifndef MPK_ATTN_SLICE_RELEASE
#error "MPK_OPROJ_PIPE_SLICE_MFMA requires MPK_ATTN_SLICE_RELEASE"
#endif
#if defined(MPK_OPROJ_SPLIT_SLICE_WAIT) || defined(MPK_SLICE_DUAL_POLL) ||      \
    defined(MPK_OPROJ_POLL_BEFORE_DRAIN) || defined(MPK_OPROJ_PTR_WALK) ||      \
    defined(MPK_OPROJ_NEXT_WT)
#error "MPK_OPROJ_PIPE_SLICE_MFMA is the default two-slice K-parallel path only"
#endif
// Wait for one XCD's attn_release, acquire, and quantize that 512-element
// slice into LDS. pair_idx 0 uses lanes 0-31 of the wave, 1 uses 32-63, so
// the 8-lane amax xor stays inside a half-wave. sl is the XCD / slice index.
__device__ __forceinline__ void
    oproj_wait_convert_one_slice(unsigned short const *A,
                                   uint8_t *s_tok_fp8,
                                   uint8_t *s_tok_scales,
                                   int *rel,
                                   int sl,
                                   int pair_idx,
                                   int tid,
                                   int layer_epoch) {
  constexpr int kElem = 16;
  constexpr int kScale = 128;
  constexpr int kSlice = 512;
  constexpr int kTps = kScale / kElem;
  while (ld_sys_s32(rel) < layer_epoch) {
    __builtin_amdgcn_s_sleep(1);
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  asm volatile("buffer_inv" ::: "memory");
  int const lane = tid & 63;
  if ((pair_idx == 0 && lane < 32) || (pair_idx == 1 && lane >= 32)) {
    int const sl_lane = lane & 31;
    int const scale_block = sl * (kSlice / kScale) + sl_lane / kTps;
    int const lane_in_scale = sl_lane & (kTps - 1);
    int const base = scale_block * kScale + lane_in_scale * kElem;
    typedef int __attribute__((ext_vector_type(4))) i32x4_t;
    i32x4_t const *src = (i32x4_t const *)(A + base);
    i32x4_t v0 = src[0];
    i32x4_t v1 = src[1];
    float vals[kElem];
    float amax = 0.0f;
    unsigned short const *vs0 = (unsigned short const *)&v0;
    unsigned short const *vs1 = (unsigned short const *)&v1;
#pragma unroll
    for (int j = 0; j < kElem / 2; j++) {
      vals[j] = _gang_bf16_to_float(vs0[j]);
      vals[j + kElem / 2] = _gang_bf16_to_float(vs1[j]);
      amax = fmaxf(amax,
                   fmaxf(fabsf(vals[j]), fabsf(vals[j + kElem / 2])));
    }
    amax = fmaxf(amax, __shfl_xor(amax, 1));
    amax = fmaxf(amax, __shfl_xor(amax, 2));
    amax = fmaxf(amax, __shfl_xor(amax, 4));
    uint8_t const se = _gang_compute_e8m0_fp8(amax);
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
    for (int j = 0; j < kElem; j += 4) {
      fp8x4_t pk = {};
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[j], vals[j + 1], scale_f, false);
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[j + 2], vals[j + 3], scale_f, true);
      *(int *)(s_tok_fp8 + base + j) = *(int const *)&pk;
    }
    if (lane_in_scale == 0) {
      s_tok_scales[scale_block] = se;
    }
  }
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_wave_barrier();
}
#endif

#ifdef MPK_ROUTER_DUAL_REDUCE
#ifndef MPK_ROUTER_FUSED_DP
#error                                                                         \
    "MPK_ROUTER_DUAL_REDUCE interleaves the ssq and router-dp wave reductions, but without MPK_ROUTER_FUSED_DP the dot product is not finished until pass 2 and there is no second reduction here to interleave with. Enable MPK_ROUTER_FUSED_DP or drop this flag."
#endif
// ── Router's two wave reductions, interleaved into one lane-crossing chain ──
//
// Under MPK_ROUTER_FUSED_DP the router finishes pass 1 holding two independent
// per-lane partials -- the RMSNorm sum of squares and the (unscaled) router dot
// product -- and reduces each with its own six-step `__shfl_xor` butterfly.
// That is twelve lane-crossing ops on the LDS data path: `__shfl_xor` lowers to
// ds_bpermute/ds_swizzle, which costs an LDS round trip and an lgkmcnt wait per
// step even though nothing here is in LDS.
//
// Every step of that butterfly has a VALU-native equivalent on gfx950:
// permlane32_swap for the xor-32 stage, permlane16_swap for xor-16, and DPP
// row_shl for the four intra-row stages. None of them touch LDS.
//
// The swaps and the DPP moves both have a hazard requiring a gap before the
// value is consumed -- alone, that gap is an `s_nop`. Running the two
// reductions together fills it with real work instead: ssq's swap issues, dp's
// swap issues, then ssq's add. So the pair costs barely more than one would.
//
// Result semantics differ from the butterfly: `__shfl_xor` leaves the total in
// every lane, whereas row_shl accumulates toward lane 0 of each row of 16 and
// leaves the other lanes holding partial sums. Lane 0 is correct, which is all
// the caller reads (`if (lane == 0) red[wave] = ssq`). Hence the name.
//
// The summation order is the same xor-32, xor-16, 8/4/2/1 tree the shuffles
// walked, so this is bit-identical to the reduction it replaces -- unlike
// MPK_ROUTER_FUSED_DP itself, which does reassociate.
__device__ __forceinline__ void router_dual_wave_sum_to_lane_zero(float &ssq,
                                                                  float &dp) {
  // The swap instructions exchange one operand's low half with the other's
  // high half, so seeding each peer with a copy of its own value turns the
  // swap plus one add into a lane-uniform butterfly. The DPP stages are
  // destructive in the other direction, so the peer is re-derived each time
  // rather than carried.
  float ssq_peer = ssq;
  float dp_peer = dp;
  asm volatile("s_nop 1\n"
               "v_permlane32_swap_b32 %[ssq], %[ssq_peer]\n"
               "v_permlane32_swap_b32 %[dp], %[dp_peer]\n"
               "v_add_f32 %[ssq], %[ssq], %[ssq_peer]\n"
               "v_add_f32 %[dp], %[dp], %[dp_peer]\n"
               "v_mov_b32_e32 %[ssq_peer], %[ssq]\n"
               "v_mov_b32_e32 %[dp_peer], %[dp]\n"
               "s_nop 1\n"
               "v_permlane16_swap_b32 %[ssq], %[ssq_peer]\n"
               "v_permlane16_swap_b32 %[dp], %[dp_peer]\n"
               "v_add_f32 %[ssq], %[ssq], %[ssq_peer]\n"
               "v_add_f32 %[dp], %[dp], %[dp_peer]\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %[ssq_peer], %[ssq] row_shl:8 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "v_mov_b32_dpp %[dp_peer], %[dp] row_shl:8 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32 %[ssq], %[ssq], %[ssq_peer]\n"
               "v_add_f32 %[dp], %[dp], %[dp_peer]\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %[ssq_peer], %[ssq] row_shl:4 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "v_mov_b32_dpp %[dp_peer], %[dp] row_shl:4 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32 %[ssq], %[ssq], %[ssq_peer]\n"
               "v_add_f32 %[dp], %[dp], %[dp_peer]\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %[ssq_peer], %[ssq] row_shl:2 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "v_mov_b32_dpp %[dp_peer], %[dp] row_shl:2 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32 %[ssq], %[ssq], %[ssq_peer]\n"
               "v_add_f32 %[dp], %[dp], %[dp_peer]\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %[ssq_peer], %[ssq] row_shl:1 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "v_mov_b32_dpp %[dp_peer], %[dp] row_shl:1 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32 %[ssq], %[ssq], %[ssq_peer]\n"
               "v_add_f32 %[dp], %[dp], %[dp_peer]"
               : [ssq] "+v"(ssq),
                 [dp] "+v"(dp),
                 [ssq_peer] "+v"(ssq_peer),
                 [dp_peer] "+v"(dp_peer));
}
#endif

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
        unsigned long long *ts_base = nullptr,
        // Optional: base of the eight per-XCD attention release flags, strided
        // by 16 ints, that the Phase 5 merge publishes under
        // MPK_ATTN_SLICE_RELEASE. When supplied, this kernel does its own
        // per-wave wait on the two slices each wave reads and the caller does
        // not wait for attention at all. Passed rather than derived from
        // `counters_ptr` so the slot number stays owned by the one file that
        // lays that buffer out.
        int const *attn_slice_release = nullptr) {

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

  // ── Attention slice release ─────────────────────────────────────────────
  // See the SLICE_RELEASE arm below. The flag only selects the arm when the
  // caller actually supplies the release line and the shape is the one the
  // arm is derived for; every other instantiation of this kernel (the
  // standalone O-proj wrapper, the N-parallel shape) keeps the shared
  // block-wide quantizer, so this needs no second implementation.
  constexpr int ATTN_SLICE = REDUCTION_SIZE / 8;
  constexpr int ELEMENTS_PER_THREAD = REDUCTION_SIZE / 256;
  constexpr int ELEMENTS_PER_SCALE = 128;
#if defined(MPK_ATTN_SLICE_RELEASE)
  constexpr bool SLICE_RELEASE =
      OUTPUT_PER_WG < 64 && BATCH_SIZE == 1 && ELEMENTS_PER_THREAD == 16;
#else
  constexpr bool SLICE_RELEASE = false;
#endif

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
  // MPK_OPROJ_TREE_BARRIER: eight per-XCD arrival lines for the two-level
  // form of the Phase 2 barrier. Slots 28..35 are dead space -- the chunk
  // barrier used to live at 28*16 and moved to 48*16 (see
  // FULL_LAYER_CHUNK_BARRIER_SLOT), and the tail counters this file's fused
  // caller pins are at 44..47. The buffer is sized well past 36 lines by
  // demo.py's counter_size, so this needs no allocation change.
  int *hier_local = hier_barrier + 28 * HIER_STRIDE;
#ifdef MPK_ROUTER_XCD_FOLD
#ifndef MPK_OPROJ_TREE_BARRIER
#error "MPK_ROUTER_XCD_FOLD publishes from the tree-barrier local-last"
#endif
#ifndef MPK_ROUTER_FUSED_DP
#error "MPK_ROUTER_XCD_FOLD reuses the fused-dp ssq/raw reduction"
#endif
  // 8 lines after the fused layer-barrier region. Matches
  // FULL_LAYER_OPROJ_XCD_READY_SLOT in gang_full_layer_fused_mi300.cuh and
  // demo.py counter_size (+128).
  int *oproj_xcd_ready = hier_barrier + (48 * 16 + 128 * MPK_MAX_NUM_BATCHED_REQUESTS + 272);
#endif

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

#ifdef MPK_OPROJ_INNER_TIMING
  // Self-contained inner split of Phase 7. The ts_base slots below are the
  // older mechanism and every caller passes nullptr for that pointer, so they
  // never fire; these four locals plus the printf at the end of the kernel are
  // what actually produce a breakdown. Timestamps are s_memrealtime ticks
  // (10 ns), taken on tid 0 only.
  //
  // Deliberately a *separate* flag from MPK_DEVICE_TIMING rather than nested
  // inside it: this printf runs once per XCD per layer, which is frequent
  // enough to inflate the [FUSED_PHASE] numbers measured by that flag (oproj
  // read 17.2 us without it and 599 us with it). Enable one or the other.
  unsigned long long _op_t0 = __builtin_amdgcn_s_memrealtime();
  unsigned long long _op_t1 = _op_t0, _op_t2 = _op_t0, _op_t3 = _op_t0;
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

    if constexpr (SLICE_RELEASE) {
      // ── Per-wave attention slice wait + quantize ──────────────────────
      //
      // Replaces the block-wide staging below for the one shape that can use
      // it. The caller (MPK_ATTN_SLICE_RELEASE in the fused full-layer
      // kernel) no longer waits for attention at all before entering here;
      // instead each XCD publishes attn_release[x] as soon as its own merge
      // finishes, and each wave here waits for only the two slices it reads.
      //
      // The mapping is forced, not chosen. This is the K-parallel arm, so
      // wave w owns MFMA iterations [8w, 8w+8) -- K elements
      // [1024w, 1024w+1024). The merge gives XCD x elements
      // [512x, 512x+512). So wave w reads exactly slices 2w and 2w+1 and is
      // independent of the other six. The static_asserts below pin every
      // number that argument uses.
      //
      // The payoff is the tail: under the default rendezvous *every* wave on
      // *every* XCD starts when the slowest of eight merges lands. Here wave
      // 0 starts when XCDs 0 and 1 land.
      static_assert(TOK_ROWS == 1 && BATCH_SIZE == 1,
                    "the slice arm stages one token row and drops the "
                    "block-wide barrier that multi-row staging needs");
      static_assert(REDUCTION_SIZE == 256 * ELEMENTS_PER_THREAD,
                    "one contiguous 16-element run per thread is what makes "
                    "wave w's staged range equal to its K range");
      static_assert(K_PER_MFMA == ELEMENTS_PER_SCALE,
                    "one E8M0 block per MFMA iteration, so b_scl[ki] indexes "
                    "the same blocks this wave wrote");
      // MFMA_ITERS % NUM_WAVES == 0 is doing double duty: it is what makes
      // wave w's K range [8w, 8w+8), and it is also what makes the
      // load-balanced split further down (KP_EXTRA waves get one extra iter)
      // degenerate to the same even split. If it ever fails, the staging
      // range and the MFMA range stop agreeing and the arm is wrong.
      static_assert(MFMA_ITERS % NUM_WAVES == 0 &&
                        (MFMA_ITERS / NUM_WAVES) * K_PER_MFMA == 2 * ATTN_SLICE,
                    "wave w must cover exactly slices 2w and 2w+1");

      // The weight DMA is the one thing here that still needs the whole
      // block: the caller's buffer_load_lds slices are issued per wave and
      // interleave across the region every wave then reads. Drain and
      // rendezvous once, before any wave goes off on its own.
#ifndef MPK_OPROJ_POLL_BEFORE_DRAIN
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
#endif

#ifdef MPK_OPROJ_PIPE_SLICE_MFMA
      // Default wave→slice map (XCD 2w, 2w+1). Convert slice 0, then the
      // K-parallel loop MFMAs those 4 iters before waiting for slice 1.
      // Rotating by xcd_id so wave 0 was local-first was +39 / +36 / +42 µs
      // and changed the text hash (fp32 K order); do not combine them.
      int const pipe_sl0 = warp_id * 2;
      int *pipe_rel0 =
          const_cast<int *>(attn_slice_release) + pipe_sl0 * 16;
      oproj_wait_convert_one_slice(A,
                                   s_tok_fp8,
                                   s_tok_scales,
                                   pipe_rel0,
                                   pipe_sl0,
                                   /*pair_idx=*/0,
                                   tid,
                                   layer_epoch);
#else
      int const first_xcd = warp_id * 2;
      // ld_sys_s32 takes a mutable pointer only because every other caller
      // hands it one; the load itself is read-only.
      int *rel0 = const_cast<int *>(attn_slice_release) + first_xcd * 16;
      int *rel1 = const_cast<int *>(attn_slice_release) + (first_xcd + 1) * 16;
      constexpr int THREADS_PER_SCALE =
          ELEMENTS_PER_SCALE / ELEMENTS_PER_THREAD;
      int const scale_block = tid / THREADS_PER_SCALE;
      int const lane_in_scale = tid & (THREADS_PER_SCALE - 1);
      int const base = scale_block * ELEMENTS_PER_SCALE +
                       lane_in_scale * ELEMENTS_PER_THREAD;
#if defined(MPK_OPROJ_SPLIT_SLICE_WAIT) && defined(MPK_SLICE_DUAL_POLL)
#error "MPK_OPROJ_SPLIT_SLICE_WAIT cannot combine with MPK_SLICE_DUAL_POLL"
#endif
#if defined(MPK_OPROJ_POLL_BEFORE_DRAIN) && defined(MPK_OPROJ_SPLIT_SLICE_WAIT)
#error "MPK_OPROJ_POLL_BEFORE_DRAIN is the default two-slice path only"
#endif
#ifdef MPK_OPROJ_SPLIT_SLICE_WAIT
      // Overlap first-slice quantize with the second XCD's merge. Wave w's
      // 1024 K elements are two 512-element XCD slices; 32 lanes x 16 elems
      // each. Wait, acquire, and convert one slice at a time.
      static_assert(ELEMENTS_PER_THREAD == 16,
                    "split-slice wait assumes 32 lanes cover one 512-elem XCD");
      for (int sl = 0; sl < 2; sl++) {
        int *rel = sl == 0 ? rel0 : rel1;
        while (ld_sys_s32(rel) < layer_epoch) {
          __builtin_amdgcn_s_sleep(1);
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        asm volatile("buffer_inv" ::: "memory");
        if (((tid & 63) < 32) == (sl == 0)) {
          i32x4_t const *src = (i32x4_t const *)(A + base);
          i32x4_t v0 = src[0];
          i32x4_t v1 = src[1];

          float vals[ELEMENTS_PER_THREAD];
          float amax = 0.0f;
          unsigned short const *vs0 = (unsigned short const *)&v0;
          unsigned short const *vs1 = (unsigned short const *)&v1;
#pragma unroll
          for (int j = 0; j < ELEMENTS_PER_THREAD / 2; j++) {
            vals[j] = _gang_bf16_to_float(vs0[j]);
            vals[j + ELEMENTS_PER_THREAD / 2] = _gang_bf16_to_float(vs1[j]);
            amax = fmaxf(
                amax,
                fmaxf(fabsf(vals[j]),
                      fabsf(vals[j + ELEMENTS_PER_THREAD / 2])));
          }
#ifdef MPK_OPROJ_AMAX_DPP
          amax = _oproj_amax8_dpp(amax);
#else
          amax = fmaxf(amax, __shfl_xor(amax, 1));
          amax = fmaxf(amax, __shfl_xor(amax, 2));
          amax = fmaxf(amax, __shfl_xor(amax, 4));
#endif

          uint8_t const se = _gang_compute_e8m0_fp8(amax);
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
          for (int j = 0; j < ELEMENTS_PER_THREAD; j += 4) {
            fp8x4_t pk = {};
            pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
                pk, vals[j], vals[j + 1], scale_f, false);
            pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
                pk, vals[j + 2], vals[j + 3], scale_f, true);
            *(int *)(s_tok_fp8 + base + j) = *(int const *)&pk;
          }
          if (lane_in_scale == 0) {
            s_tok_scales[scale_block] = se;
          }
        }
      }
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      __builtin_amdgcn_wave_barrier();
#else
#ifdef MPK_SLICE_DUAL_POLL
      // Wave w waits on XCDs 2w and 2w+1. The serial form issues load-wait
      // twice, so a miss always pays two coherency round trips even though
      // the flags are independent. Overlap them.
      int s0, s1;
      do {
        ld_sys_s32x2(rel0, rel1, s0, s1);
        if (s0 >= layer_epoch && s1 >= layer_epoch) {
          break;
        }
        __builtin_amdgcn_s_sleep(1);
      } while (true);
#else
      while (ld_sys_s32(rel0) < layer_epoch || ld_sys_s32(rel1) < layer_epoch) {
#ifndef MPK_SLICE_BUSY_POLL
        __builtin_amdgcn_s_sleep(1);
#endif
      }
#endif
      // Cross-XCD acquire, per wave, placed at this wave's observation. The
      // caller's Phase 6 `buffer_inv` runs before any flag has been seen on
      // this path, so it cannot be the acquire for attn_out: a line cached
      // here while reading the *previous* layer's attn_out would outlive it.
      // buffer_inv is a per-wave instruction, so four of them is the correct
      // shape, not a redundancy -- each wave invalidates for itself.
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      asm volatile("buffer_inv" ::: "memory");
#ifdef MPK_OPROJ_POLL_BEFORE_DRAIN
      __syncthreads();
#endif

      i32x4_t const *src = (i32x4_t const *)(A + base);
      i32x4_t v0 = src[0];
      i32x4_t v1 = src[1];

      float vals[ELEMENTS_PER_THREAD];
      float amax = 0.0f;
      unsigned short const *vs0 = (unsigned short const *)&v0;
      unsigned short const *vs1 = (unsigned short const *)&v1;
#pragma unroll
      for (int j = 0; j < ELEMENTS_PER_THREAD / 2; j++) {
        vals[j] = _gang_bf16_to_float(vs0[j]);
        vals[j + ELEMENTS_PER_THREAD / 2] = _gang_bf16_to_float(vs1[j]);
        amax = fmaxf(
            amax,
            fmaxf(fabsf(vals[j]), fabsf(vals[j + ELEMENTS_PER_THREAD / 2])));
      }

      // All eight threads of a scale block sit in consecutive lanes of one
      // wave (lane_in_scale is the low 3 bits of tid), so an xor butterfly
      // over 1/2/4 stays inside the wave and leaves every lane holding the
      // block max -- no leader broadcast needed.
#ifdef MPK_OPROJ_AMAX_DPP
      amax = _oproj_amax8_dpp(amax);
#else
      amax = fmaxf(amax, __shfl_xor(amax, 1));
      amax = fmaxf(amax, __shfl_xor(amax, 2));
      amax = fmaxf(amax, __shfl_xor(amax, 4));
#endif

      uint8_t const se = _gang_compute_e8m0_fp8(amax);
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
      for (int j = 0; j < ELEMENTS_PER_THREAD; j += 4) {
        fp8x4_t pk = {};
        pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
            pk, vals[j], vals[j + 1], scale_f, false);
        pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
            pk, vals[j + 2], vals[j + 3], scale_f, true);
        *(int *)(s_tok_fp8 + base + j) = *(int const *)&pk;
      }
      if (lane_in_scale == 0) {
        s_tok_scales[scale_block] = se;
      }
      // Deliberately NOT __syncthreads. Every byte this wave just wrote is
      // read back by this same wave and by no other, so a wave-scope drain is
      // the whole requirement -- and keeping it wave-scope is the point of
      // the arm: wave 0 enters its MFMA while wave 3 is still spinning on
      // XCDs 6 and 7.
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      __builtin_amdgcn_wave_barrier();
#endif
#endif
    } else {
      // Stage up to 16 token rows. The bespoke inline quantizer that used to
      // live here was a second copy of the E8M0 logic that only ever handled
      // row 0; the shared helper covers both OUTPUT_PER_WG shapes.
      _gang_multirow_fp8_quant<REDUCTION_SIZE,
                               TOK_ROWS,
                               BATCH_SIZE,
                               TOK_ROW_STRIDE,
                               SC_STRIDE>(A,
                                          REDUCTION_SIZE,
                                          tok_row_base,
                                          n_valid_tok,
                                          s_tok_fp8,
                                          s_tok_scales);
    }

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
      int const lds_row_scale_base = w_row_lds * NUM_BLOCKS_32;

      // ── Weight fragment addressing ────────────────────────────────────────
      //
      // Row-major (default): lane (g, col) reads
      //   col * (REDUCTION_SIZE / 2) + KI * 64 + g * 16.
      // At REDUCTION_SIZE 4096 the row stride is 2048 B = 512 dwords, and
      // 512 % 32 == 0, so all 16 values of `col` land on the same LDS bank.
      // The 64 lanes of the wave therefore start at only four distinct banks
      // (g * 16 B = g * 4 dwords), sixteen lanes deep on each -- a 16-way
      // conflict on every ds_read_b128 in the K loop.
      //
      // K-major (MPK_OPROJ_KMAJOR): the tile is repacked offline from
      // [row][k128][quarter][16B] to [k128][quarter][row][16B], so the lane's
      // fragment sits at lane_id * 16 within its K block. Since
      // lane_id == g * 16 + col, that is exactly quarter-major-then-row, and
      // the wave reads 64 consecutive 16-byte fragments -- the conflict-free
      // pattern ds_read_b128 is built for. Only the address changes; the same
      // bytes reach the same lane, so this is bit-exact.
      //
      // The scale suffix stays row-major in both layouts (one byte per
      // 32-element block, read as a scalar, not a b128), so
      // lds_row_scale_base is shared.
#ifdef MPK_OPROJ_KMAJOR
      static_assert(OUTPUT_PER_WG == 16,
                    "MPK_OPROJ_KMAJOR assumes a 16-row tile: lane_id spans "
                    "exactly 16 rows x 4 K-quarters, which is what makes "
                    "lane_id * 16 the fragment address.");
      // One K128 block holds all 16 rows x 4 quarters x 16 B.
      constexpr int LDS_DATA_K_STRIDE = 16 * (K_PER_MFMA / 2);
      int const lds_data_lane_offset = lane_id * 16;
#define MPK_OPROJ_W_ADDR(KI)                                                   \
  (lds_w_data + lds_data_lane_offset + (KI)*LDS_DATA_K_STRIDE)
#else
      int const lds_row_data_base = w_row_lds * (REDUCTION_SIZE / 2);
#define MPK_OPROJ_W_ADDR(KI)                                                   \
  (lds_w_data + lds_row_data_base + (KI) * (K_PER_MFMA / 2) + g * 16)
#endif

#ifdef MPK_OPROJ_A_PAD_HOIST
#if defined(MPK_OPROJ_PTR_WALK) || defined(MPK_OPROJ_NEXT_WT)
#error "MPK_OPROJ_A_PAD_HOIST is the live address-form K-parallel loop only"
#endif
      i32x8_t a;
      a[4] = 0;
      a[5] = 0;
      a[6] = 0;
      a[7] = 0;
#endif
#ifdef MPK_OPROJ_PTR_WALK
      // The eight K-parallel iterations are consecutive. Keep one cursor per
      // operand stream instead of rebuilding four thread-varying LDS
      // addresses from kp_ki_start before every MFMA.
      uint8_t const *op_w_data = MPK_OPROJ_W_ADDR(kp_ki_start);
      uint8_t const *op_w_scale =
          lds_w_scales + lds_row_scale_base + kp_ki_start * 4 + g;
      uint8_t const *op_b_data =
          b_tok + kp_ki_start * K_PER_MFMA;
      uint8_t const *op_b_scale = b_scl + kp_ki_start;
#ifdef MPK_OPROJ_KMAJOR
      constexpr int OP_W_DATA_STEP = LDS_DATA_K_STRIDE;
#else
      constexpr int OP_W_DATA_STEP = K_PER_MFMA / 2;
#endif
#define DO_MFMA_LDS_FP8(KI)                                                    \
  do {                                                                         \
    i32x4_t _wt;                                                               \
    __builtin_memcpy(&_wt, op_w_data, 16);                                     \
    int sa = (int)*op_w_scale;                                                 \
    i32x8_t a;                                                                 \
    a[0] = _wt[0];                                                             \
    a[1] = _wt[1];                                                             \
    a[2] = _wt[2];                                                             \
    a[3] = _wt[3];                                                             \
    a[4] = 0;                                                                  \
    a[5] = 0;                                                                  \
    a[6] = 0;                                                                  \
    a[7] = 0;                                                                  \
    i32x8_t b = _gang_load_fp8_mfma_b(op_b_data, 0, g);                        \
    int sb = (int)*op_b_scale;                                                 \
    acc = _gang_mfma_f4xf8(a, b, acc, sa, sb);                                 \
    op_w_data += OP_W_DATA_STEP;                                               \
    op_w_scale += 4;                                                           \
    op_b_data += K_PER_MFMA;                                                   \
    op_b_scale += 1;                                                           \
  } while (0)
#else
#ifdef MPK_OPROJ_A_PAD_HOIST
#define DO_MFMA_LDS_FP8(KI)                                                    \
  do {                                                                         \
    i32x4_t _wt;                                                               \
    __builtin_memcpy(&_wt, MPK_OPROJ_W_ADDR(KI), 16);                          \
    int sa = (int)lds_w_scales[lds_row_scale_base + (KI)*4 + g];               \
    a[0] = _wt[0];                                                             \
    a[1] = _wt[1];                                                             \
    a[2] = _wt[2];                                                             \
    a[3] = _wt[3];                                                             \
    i32x8_t b = _gang_load_fp8_mfma_b(b_tok, (KI)*K_PER_MFMA, g);              \
    int sb = (int)b_scl[KI];                                                   \
    acc = _gang_mfma_f4xf8(a, b, acc, sa, sb);                                 \
  } while (0)
#else
#define DO_MFMA_LDS_FP8(KI)                                                    \
  do {                                                                         \
    i32x4_t _wt;                                                               \
    __builtin_memcpy(&_wt, MPK_OPROJ_W_ADDR(KI), 16);                          \
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
    i32x8_t b = _gang_load_fp8_mfma_b(b_tok, (KI)*K_PER_MFMA, g);              \
    int sb = (int)b_scl[KI];                                                   \
    acc = _gang_mfma_f4xf8(a, b, acc, sa, sb);                                 \
  } while (0)
#endif
#endif
#ifdef MPK_OPROJ_NEXT_WT
#if defined(MPK_OPROJ_PTR_WALK)
#error "MPK_OPROJ_NEXT_WT is the address-form pipeline; not combined with PTR_WALK"
#endif
#undef DO_MFMA_LDS_FP8
      i32x4_t _wt_pipe;
      __builtin_memcpy(&_wt_pipe, MPK_OPROJ_W_ADDR(kp_ki_start), 16);
#define DO_MFMA_LDS_FP8(KI)                                                    \
  do {                                                                         \
    i32x4_t _wt = _wt_pipe;                                                    \
    if ((KI) + 1 < kp_ki_end) {                                                \
      __builtin_memcpy(&_wt_pipe, MPK_OPROJ_W_ADDR((KI) + 1), 16);             \
    }                                                                          \
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
    i32x8_t b = _gang_load_fp8_mfma_b(b_tok, (KI)*K_PER_MFMA, g);              \
    int sb = (int)b_scl[KI];                                                   \
    acc = _gang_mfma_f4xf8(a, b, acc, sa, sb);                                 \
  } while (0)
#endif
#define DO_MFMA_LDS_FP4(KI)                                                    \
  do {                                                                         \
    i32x4_t _wt;                                                               \
    __builtin_memcpy(&_wt, MPK_OPROJ_W_ADDR(KI), 16);                          \
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

#ifdef MPK_OPROJ_PIPE_SLICE_MFMA
      {
        int const pipe_sl0 = warp_id * 2;
        int const pipe_sl1 = pipe_sl0 + 1;
        int const ki0 = pipe_sl0 * 4;
        int const ki1 = pipe_sl1 * 4;
        DO_MFMA_LDS_FP8(ki0 + 0);
        DO_MFMA_LDS_FP8(ki0 + 1);
        DO_MFMA_LDS_FP8(ki0 + 2);
        DO_MFMA_LDS_FP8(ki0 + 3);
        int *pipe_rel1 =
            const_cast<int *>(attn_slice_release) + pipe_sl1 * 16;
        oproj_wait_convert_one_slice(A,
                                     s_tok_fp8,
                                     s_tok_scales,
                                     pipe_rel1,
                                     pipe_sl1,
                                     /*pair_idx=*/1,
                                     tid,
                                     layer_epoch);
        DO_MFMA_LDS_FP8(ki1 + 0);
        DO_MFMA_LDS_FP8(ki1 + 1);
        DO_MFMA_LDS_FP8(ki1 + 2);
        DO_MFMA_LDS_FP8(ki1 + 3);
      }
#else
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
#endif
#undef DO_MFMA_LDS_FP8
#undef DO_MFMA_LDS_FP4
#undef MPK_OPROJ_W_ADDR

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
#ifdef MPK_OPROJ_RED_VEC
      {
        f32x4_t packed = acc;
        __builtin_memcpy(&lds_reduce[(warp_id * 64 + lane_id) * 4], &packed,
                         sizeof(packed));
      }
#else
      for (int i = 0; i < 4; i++) {
        lds_reduce[(warp_id * 64 + lane_id) * 4 + i] = acc[i];
      }
#endif
      __syncthreads();

      if (warp_id == 0 && tok_active) {
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        // Each lane reduces its own (g, col) slot: output columns g*4..g*4+3
        // of token col. At TOK_ROWS == 1 that slot is always g*16, and saying
        // so keeps the address scalar in g -- reading lane_id here instead
        // costs scratch even though the two are equal under tok_active.
        int const src_lane = TOK_ROWS == 1 ? g * 16 : lane_id;
        for (int w = 0; w < NUM_WAVES; w++) {
#ifdef MPK_OPROJ_RED_VEC
          f32x4_t t;
          __builtin_memcpy(&t, &lds_reduce[(w * 64 + src_lane) * 4],
                           sizeof(t));
          v0 += t[0];
          v1 += t[1];
          v2 += t[2];
          v3 += t[3];
#else
          v0 += lds_reduce[(w * 64 + src_lane) * 4 + 0];
          v1 += lds_reduce[(w * 64 + src_lane) * 4 + 1];
          v2 += lds_reduce[(w * 64 + src_lane) * 4 + 2];
          v3 += lds_reduce[(w * 64 + src_lane) * 4 + 3];
#endif
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
#ifdef MPK_OPROJ_BIAS_NO_WAIT
            // __syncthreads above already drained vmcnt for this wave on
            // gfx950 (s_waitcnt vmcnt(0) lgkmcnt(0); s_barrier). The extra
            // wait here was for bias+residual issued before the MFMA loop.
            "v_mov_b32_e32 %0, %4\n"
            "v_mov_b32_e32 %1, %5\n"
            "v_mov_b32_e32 %2, %6\n"
            "v_mov_b32_e32 %3, %7"
#else
            "s_waitcnt vmcnt(0)\n"
            "v_mov_b32_e32 %0, %4\n"
            "v_mov_b32_e32 %1, %5\n"
            "v_mov_b32_e32 %2, %6\n"
            "v_mov_b32_e32 %3, %7"
#endif
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
#ifdef MPK_OPROJ_INNER_TIMING
  _op_t1 = __builtin_amdgcn_s_memrealtime();
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

    // ── Release fan-out: one wave instruction, not eight serial stores ────
    //
    // The eight per-XCD release flags live on eight separate cache lines
    // (HIER_STRIDE apart, deliberately -- see the MoE barrier's note on
    // write-through vs L2 aliasing). Writing them from tid 0 is therefore
    // eight independent `sc0 sc1` stores issued back to back by a single
    // lane, and this lane is the single most critical thread on the GPU: 239
    // workgroups are spinning on the flags it has not written yet, so the
    // eighth XCD is released seven store-issues later than the first.
    //
    // Spreading them over lanes 0..7 of the same wave makes it one
    // instruction -- the addresses differ only by lane, so it is a single
    // `global_store_dword` with a per-lane offset -- and all eight XCDs are
    // released together. Both hierarchical releases are published this way.
    //
    // The decision reaches the other seven lanes by readfirstlane, not LDS:
    // tid 0..7 are the same wave, the arrival below runs with only lane 0
    // active, and readfirstlane reads the first *active* lane -- so once the
    // `if` closes and all lanes are live again, every lane holds lane 0's
    // value. Zero means "not the releaser", which is unambiguous because a
    // real epoch is >= 1.
    int oproj_rel_epoch = 0;
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
      //
      // MPK_OPROJ_NO_WB: the paragraph above is the argument for a *writeback*,
      // and it does not survive contact with what this phase actually stores.
      // The only global publication between the top of the kernel and this
      // arrival is `st_wt_u64` into d_output, and that lowers to
      // `global_store_dwordx2 ... sc0 sc1` -- write-through, straight past L2
      // to HBM. The `s_waitcnt vmcnt(0)` two lines up retires those stores at
      // HBM, not at a local L2, so by the time the arrival is issued the
      // columns are already remotely visible. buffer_wbl2 then walks the whole
      // L2 to write back lines that by construction do not hold any of the
      // guarded data.
      //
      // What it costs: 184 workers each issue a whole-L2 writeback and then
      // wait on it, and they all do so within ~0.14 us of each other (measured
      // arrival spread), so the writebacks serialize against each other rather
      // than overlapping with anything. That matches the measurement -- first
      // and last arriver both wait ~5.6 us, i.e. the wait is protocol latency,
      // not skew.
      //
      // This is the same argument, and the same fix, as commit 57422b1 on the
      // EP fold. Kept behind a flag because the reasoning it overturns is
      // load-bearing if any store on this path is ever changed away from
      // write-through: the failure signature to watch for is rows of
      // rmsnorm_out differing run to run while attn_proj_out is bit-identical.
#ifndef MPK_OPROJ_NO_WB
      threadfence_gpu();
#else
      // Still order the arrival after this block's own stores; drop only the
      // L2 writeback. The stores are write-through, so retiring them is the
      // whole of what the release needs.
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
#ifdef MPK_OPROJ_TREE_BARRIER
      // ── Two-level arrival ────────────────────────────────────────────────
      //
      // The flat form below puts all `total_oproj_tiles` (184 at bs=1)
      // arrivals on one cache line at [8*16]. Every one of those is a
      // cross-die read-modify-write -- `sc0 sc1` carries the atomic past this
      // XCD's private L2 to the device coherency point -- and they serialize
      // there, because a line can only be owned in one place at a time. That
      // is the same cost Phase 9's comment already prices for its own flat
      // predecessor ("~1.0 ms/iter ... every one of those 240 atomics is a
      // cross-die read-modify-write serialising on a single cache line") and
      // the same fix: aggregate per-XCD first so the shared line sees 8
      // arrivals instead of 184.
      //
      // Level 1 is XCD-private -- written and read only at [xcd_id] -- so it
      // takes the `sc0`-only atomic that resolves in the local L2 and never
      // leaves the die. Level 2 keeps `sc0 sc1`: it is the level that is
      // genuinely cross-XCD.
      //
      // The release value and its fan-out are untouched, so a worker's poll
      // below is unchanged and sees exactly the same value at the same point
      // in the layer.
      //
      // Expected counts. tiles_per_xcd is `oproj_topk_tiles_per_xcd` from the
      // fused caller = max(oproj_tiles_per_xcd, router_tile_n), which is what
      // gates `xcd_rank < oproj_topk_tiles_per_xcd` there -- so exactly
      // tiles_per_xcd workers per XCD reach this arrival. total_oproj_tiles is
      // 8 * that, which is what the flat form counts. The two agree by
      // construction; the static relation is asserted at the caller rather
      // than here because tiles_per_xcd is a runtime argument.
      //
      // The `%` form is a monotonic-counter release, identical in shape to
      // the flat one: counters are never reset, so "this layer's last
      // arrival" is `prev % N == N - 1` rather than a compare against N.
      int const local_prev =
          atom_add_xcd_local_s32(&hier_local[xcd_id * HIER_STRIDE], 1);
      if ((local_prev % tiles_per_xcd) == tiles_per_xcd - 1) {
        // Last worker on this XCD. Its own stores are drained (above) and so
        // are every other arriving worker's on this XCD -- each drained
        // before its own arrival, and all of those arrivals precede this one
        // on the same line. Publish one arrival for the whole die.
#ifdef MPK_ROUTER_XCD_FOLD
        // This XCD's 368-col attn_proj_out slice is in HBM. Router workers
        // on every die poll this flag and FMA the slice without waiting
        // for the other seven XCDs.
        st_wt_u32((void *)&oproj_xcd_ready[xcd_id * HIER_STRIDE],
                  (unsigned)oproj_release_expected);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
        int const prev_global =
            atom_add_release_gpu_s32(&hier_barrier[8 * HIER_STRIDE], 1);
        if ((prev_global % 8) == 7) {
          oproj_rel_epoch = oproj_release_expected;
        }
      }
#else
      // Single global arrival (all workers increment one counter)
      int prev_global =
          atom_add_release_gpu_s32(&hier_barrier[8 * HIER_STRIDE], 1);
      if ((prev_global % total_oproj_tiles) == total_oproj_tiles - 1) {
        oproj_rel_epoch = oproj_release_expected;
      }
#endif
    }

    // Lanes 0..7 of wave 0 publish the eight flags in one instruction. See
    // the note above the arrival for why readfirstlane carries the epoch and
    // why 0 is a safe "not the releaser" sentinel.
    oproj_rel_epoch = __builtin_amdgcn_readfirstlane(oproj_rel_epoch);
    if (oproj_rel_epoch != 0) {
      if (tid < 8) {
        st_wt_u32((void *)&hier_barrier[tid * HIER_STRIDE],
                  (unsigned)oproj_rel_epoch);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
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
#ifdef MPK_OPROJ_ARRIVE_ONLY
    // ── TESTED AND REJECTED: incorrect. Kept for the reasoning. ──────────
    //
    // Measured 2.015 mean over 6 reps against 2.019 for the control -- and
    // rep 3 generated *different text* (hash 1304e716e7db against
    // 88861a4763c0 on the other five). It is a race, not a numerics change.
    //
    // The read-side argument below is right as far as it goes: the release
    // says "every XCD's O-proj columns are in HBM", and a worker that skips
    // the RMSNorm never reads them. What it misses is the *write* side. This
    // poll is also the only thing standing between such a worker and the
    // next layer's O-proj store into the same buffer, and it is load-bearing
    // precisely because of which workers skip it. `local_tile >=
    // router_tile_n` means `xcd_rank >= 16`, and MPK_W2_CONSUMER_GATE's
    // Phase 9 joiner test is `xcd_rank < total_qkv_tiles_per_xcd` (10) -- so
    // every worker this flag releases early is also a worker that does not
    // join the layer barrier. It therefore has no gate at all until Phase 6
    // of layer N+1, and can reach layer N+1's Phase 7 and store its
    // attn_proj_out columns while a straggling XCD's layer-N router worker
    // is still reading that row. WAR, cross-XCD, silent.
    //
    // That also explains the shape of the failure: one run in six, and one
    // that produces coherent-but-different text rather than garbage, because
    // the overwritten columns are a valid activation from the wrong layer.
    //
    // Fixing it means giving these workers some later gate, and the only one
    // available is Phase 9's -- which is the wait this flag was removing. So
    // the 2.5 us is not recoverable at this barrier; it would have to come
    // from making the barrier itself cheaper.
    //
    // ── Original rationale, retained ─────────────────────────────────────
    //
    // This barrier is a *producer* barrier: what it guarantees to the worker
    // that clears it is that every XCD's O-proj columns are in HBM, so the
    // RMSNorm below can read the whole row. A worker with
    // `local_tile >= router_tile_n` does not run that RMSNorm -- it falls
    // straight to `goto done` a few lines below -- so the release tells it
    // nothing it uses. It still has to *arrive*, because its own O-proj
    // columns are part of what the barrier publishes, and the arrival above
    // is already drained and issued before this poll.
    //
    // At the shipped geometry that is 7 of the 23 O-proj workers per XCD
    // released ~2.5 us early, and they are the ones that go on to draw MoE
    // tiles in Phase 8 -- so the time comes off the front of the phase that
    // is the critical path, not off an idle worker.
    //
    // Only the poll is skipped. `local_tile` is workgroup-uniform, so this
    // branch is uniform and the __syncthreads below is still executed by all
    // 256 threads of every block -- no divergent barrier. The skippers do run
    // that rendezvous and the `buffer_inv` after it before reaching `goto
    // done`; both are a few cycles against the ~2.5 us poll, and leaving them
    // in keeps the acquire argument below untouched.
    if (local_tile < router_tile_n)
#endif
#ifdef MPK_ROUTER_XCD_FOLD
#if defined(MPK_OPROJ_ARRIVE_ONLY)
#error "MPK_ROUTER_XCD_FOLD skips the hier poll on router tiles; incompatible with ARRIVE_ONLY"
#endif
    // Router workers wait per-XCD in pass 1 instead of for the global last.
    if (local_tile >= router_tile_n)
#endif
#ifdef MPK_NARROW_OPROJ_HIER
#ifdef MPK_OPROJ_LEAN_ACQUIRE
#error "MPK_NARROW_OPROJ_HIER needs the acquire __syncthreads that LEAN_ACQUIRE removes"
#endif
    // One poller. The acquire rendezvous below is already block-wide, so the
    // other 255 threads do not need their own sc0 sc1 reads of this line.
    if (tid == 0)
#endif
      while (MPK_LD_GATE2(&hier_barrier[xcd_id * HIER_STRIDE]) <
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
#ifdef MPK_OPROJ_LEAN_ACQUIRE
  // ── The rendezvous and the invalidate go together, or not at all ─────────
  //
  // The argument above is sound *given* that a `buffer_inv` follows: it is
  // entirely about one wave refilling vL1 behind another wave's invalidate.
  // Remove the invalidate and there is nothing for the rendezvous to order, so
  // this arm removes both. What has to hold instead is that no wave can be
  // holding a stale line for this data in the first place.
  //
  // It does hold, for the same reason MPK_OPROJ_NO_WB holds on the write side.
  // The producer is `st_wt_u64` -> `global_store_dwordx2 ... sc0 sc1`:
  // write-through past vL1 and L2 to HBM, and `sc0` on a store also
  // *invalidates* the writing CU's vL1 line rather than leaving a stale copy.
  // The only other reader of attn_proj_out in the layer is this same phase one
  // layer earlier, and between the two sits Phase 9's layer barrier, whose
  // consumer gate already issues `buffer_inv sc1` under MPK_W2_CONSUMER_GATE.
  //
  // The remaining hole the default build closes is a reader that pulled a line
  // into vL1 *this* layer, before the release -- but nothing in this phase
  // reads attn_proj_out before the poll, and the poll itself is a device-scope
  // load of a different line.
  //
  // Failure signature if this is wrong, and it is worth restating because it is
  // silent: rows of rmsnorm_out differing between two runs whose
  // attn_proj_out is bit-identical. That is why this arm is verified over 8+
  // reps of generated-text hash, not over timing.
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#else
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
  // in HBM. This was `buffer_inv sc1` to invalidate L2 as well as vL1; it is
  // now plain `buffer_inv` (vL1 only). See the layer-boundary acquire at the
  // top of gang_full_layer_fused_mi300.cuh for why that is sufficient given
  // the Phase 9 layer barrier, and for the ablation that established it.
  //
  // The signature to watch for if this ever regresses is exact: rows of
  // `rmsnorm_out` differing between two runs whose `attn_proj_out` (the
  // settled HBM content) is bit-identical -- the norm read something that is
  // not what is in memory.
  asm volatile("buffer_inv" ::: "memory");
  // Drain prefetched gamma + router weight loads (issued before barrier).
  // NT loads bypass L2, unaffected by buffer_inv.
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif

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
#ifdef MPK_OPROJ_INNER_TIMING
  _op_t2 = __builtin_amdgcn_s_memrealtime();
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
      //
      // All MAX_ITERS loads are issued up front rather than one iteration
      // ahead. The depth-1 version that used to be here drained with
      // `s_waitcnt vmcnt(0)` at the top of every trip, and vmcnt(0) is *all*
      // outstanding loads, not just the one being consumed -- so the load
      // issued for iteration i+1 was waited on at iteration i, and the three
      // round trips serialized end to end instead of overlapping. Issuing the
      // whole set first lets each trip wait on `vmcnt(MAX_ITERS-1-iter)`,
      // which retires exactly the one load it needs and leaves the rest in
      // flight.
      //
      // Costs nothing in registers: every value already had to be live in
      // h_cache[] for pass 2, so this only moves where the load is issued,
      // not how long the result lives. MAX_ITERS is 3 at hidden 2880.
      //
      // This sits on the strictly serial part of the layer -- TopK cannot
      // start until the router logits are written, and no MoE worker can
      // start until TopK publishes -- so latency here is layer latency.
      // The counted waits below name a nonzero vmcnt, which is only sound if
      // nothing this thread issued *earlier* is still outstanding. At b == 0
      // the O-proj barrier's drain already guarantees that, but at b > 0 the
      // previous token's pass-2 normed-row stores are vector memory ops too
      // and sit in the same counter -- an undrained store would make
      // vmcnt(2) retire the store instead of the load, and this loop would
      // consume an unwritten register. One drain per token closes that; it is
      // dead code at BATCH_SIZE == 1, where n_tok_router is a compile-time 1.
      if (BATCH_SIZE != 1) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }

      // The wait-count switch below enumerates 0, 1 and 2 outstanding loads.
      // vmcnt takes an immediate, so the arms cannot be generated from a
      // runtime expression -- a deeper pipeline needs more arms, and getting
      // that wrong reads a register the load has not written yet, which is
      // silent wrong numerics in the routing logits rather than a fault.
      static_assert(MAX_ITERS <= 3,
                    "pass-1 hidden prefetch: the s_waitcnt vmcnt switch below "
                    "only covers up to 3 in-flight loads; add arms before "
                    "raising ACTUAL_HIDDEN_DIM past 3*256*4");
#ifdef MPK_ROUTER_XCD_FOLD
      // FMA each XCD's 368-col slice as that die's O-proj local-last
      // publishes. Per-thread iter order is unchanged (1024-dim stride >
      // 368-col slice ⇒ at most one iter per XCD), so ssq/dp association
      // matches the unfolder.
      int const cols_per_xcd = OUTPUT_PER_WG * n_wgs_per_xcd;
      int const slice_epoch = layer_epoch > 0 ? layer_epoch : 1;
      for (int x = 0; x < 8; x++) {
        // Adjacent XCDs' 368-col slices share a 128 B line (736 % 128 != 0).
        // Waiting for x+1 before reading x keeps that line from tearing.
        // All 256 threads poll so there is no per-slice syncthreads.
        while (MPK_LD_GATE(&oproj_xcd_ready[x * 16]) < slice_epoch) {
          __builtin_amdgcn_s_sleep(1);
        }
        if (x < 7) {
          while (MPK_LD_GATE(&oproj_xcd_ready[(x + 1) * 16]) < slice_epoch) {
            __builtin_amdgcn_s_sleep(1);
          }
        }
        asm volatile("buffer_inv" ::: "memory");
        int const d0 = x * cols_per_xcd;
        int const d1 = d0 + cols_per_xcd;
#pragma unroll
        for (int iter = 0; iter < MAX_ITERS; iter++) {
          int i_cur = tid + iter * 256;
          if (i_cur >= H4) {
            break;
          }
          int const dim = i_cur * 4;
          if (dim < d0 || dim >= d1 || dim >= ACTUAL_HIDDEN_DIM) {
            continue;
          }
          i32x2_t h_v;
          asm volatile("global_load_dwordx2 %0, %1, off"
                       : "=v"(h_v)
                       : "v"(h_base + i_cur * 8)
                       : "memory");
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          h_cache[iter] = h_v;
          __builtin_memcpy(&g_cache[iter], &g_pf_buf[iter], 8);
          n_cached = iter + 1;
          float v0, v1, v2, v3;
          asm volatile("v_cvt_f32_bf16 %0, %4\n"
                       "v_cvt_f32_bf16 %1, %4 src0_sel:WORD_1\n"
                       "v_cvt_f32_bf16 %2, %5\n"
                       "v_cvt_f32_bf16 %3, %5 src0_sel:WORD_1"
                       : "=&v"(v0), "=&v"(v1), "=&v"(v2), "=&v"(v3)
                       : "v"(h_v[0]), "v"(h_v[1]));
          ssq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
          float gg0, gg1, gg2, gg3;
          asm volatile("v_cvt_f32_bf16 %0, %4\n"
                       "v_cvt_f32_bf16 %1, %4 src0_sel:WORD_1\n"
                       "v_cvt_f32_bf16 %2, %5\n"
                       "v_cvt_f32_bf16 %3, %5 src0_sel:WORD_1"
                       : "=&v"(gg0), "=&v"(gg1), "=&v"(gg2), "=&v"(gg3)
                       : "v"(g_pf_buf[iter][0]), "v"(g_pf_buf[iter][1]));
          float ww0, ww1, ww2, ww3;
          asm volatile("v_cvt_f32_bf16 %0, %4\n"
                       "v_cvt_f32_bf16 %1, %4 src0_sel:WORD_1\n"
                       "v_cvt_f32_bf16 %2, %5\n"
                       "v_cvt_f32_bf16 %3, %5 src0_sel:WORD_1"
                       : "=&v"(ww0), "=&v"(ww1), "=&v"(ww2), "=&v"(ww3)
                       : "v"(w_pf_buf[iter][0]), "v"(w_pf_buf[iter][1]));
          dp += ww0 * (v0 * gg0) + ww1 * (v1 * gg1) + ww2 * (v2 * gg2) +
                ww3 * (v3 * gg3);
        }
      }
#else
      i32x2_t h_pf[MAX_ITERS];
      int n_issued = 0;
#pragma unroll
      for (int iter = 0; iter < MAX_ITERS; iter++) {
        int i_cur = tid + iter * 256;
        if (i_cur >= H4) {
          break;
        }
        asm volatile("global_load_dwordx2 %0, %1, off"
                     : "=v"(h_pf[iter])
                     : "v"(h_base + i_cur * 8)
                     : "memory");
        n_issued = iter + 1;
      }

#pragma unroll
      for (int iter = 0; iter < MAX_ITERS; iter++) {
        int i_cur = tid + iter * 256;
        if (i_cur >= H4) {
          break;
        }

        // Retire this iteration's load and leave the later ones outstanding.
        // The count is `n_issued - 1 - iter` -- how many of *these* loads are
        // still in flight behind the one being consumed. It is safe to name a
        // nonzero count only because nothing else this thread issued can be
        // outstanding here: gamma and the router weights were drained by the
        // `s_waitcnt vmcnt(0)` that follows the O-proj barrier's buffer_inv,
        // which every path into this block has already executed.
        //
        // vmcnt is an immediate, so the count has to be a compile-time
        // constant -- hence the switch rather than an expression. Both bounds
        // are constexpr, so exactly one arm survives per unrolled trip.
        if (n_issued - 1 - iter == 0) {
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        } else if (n_issued - 1 - iter == 1) {
          asm volatile("s_waitcnt vmcnt(1)" ::: "memory");
        } else {
          asm volatile("s_waitcnt vmcnt(2)" ::: "memory");
        }
        i32x2_t h_v = h_pf[iter];

        // Cache: hidden from the fresh load, gamma+router from the
        // pre-barrier prefetch. Both of the latter are token-invariant, but
        // the copy stays inside the loop rather than being hoisted: hoisting
        // extends g_pf_buf/w_pf_buf's live range across the whole token
        // sweep and the allocator answers with 16 more bytes of scratch even
        // at BATCH_SIZE == 1. Re-copying the same VGPRs is free.
        h_cache[iter] = h_v;
        __builtin_memcpy(&g_cache[iter], &g_pf_buf[iter], 8);
#ifndef MPK_ROUTER_FUSED_DP
        __builtin_memcpy(&w_cache[iter], &w_pf_buf[iter], 8);
#endif
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

#ifdef MPK_ROUTER_FUSED_DP
        // ── The router dot product, folded into this pass ────────────────
        // irms is a single scalar, uniform over the whole row, so
        //   sum_i w_i * (h_i * irms * g_i)  ==  irms * sum_i w_i * h_i * g_i
        // and the scale can be applied once after the reduction instead of
        // once per element. That is the whole trick: the dp no longer needs
        // the normalized values, so it no longer needs to wait for the ssq
        // reduction, so it can consume the hidden payload while it is still
        // in registers from the load above.
        //
        // What that buys is not the arithmetic -- it is pass 2. Pass 2 exists
        // only because dp needed post-irms values; with dp gone from it, the
        // only thing left in it is the normed-row store, which
        // MPK_ONE_NORM_WRITER already restricts to one workgroup per XCD. So
        // 15 of every 16 router workgroups now skip the second unpack of
        // hidden+gamma+weight, its stores, its shuffle reduction and one of
        // its two __syncthreads outright.
        //
        // Not bit-identical to the unfused order: (h*irms)*g*w rounds
        // differently from (h*g*w)*irms. The logit feeds a top-8 argmax over
        // 128 experts whose gaps are far wider than a single ulp, so the
        // routing decision is unchanged -- but this is exactly why the flag
        // is verified by hashing the generated text and not by timing alone.
        float gg0, gg1, gg2, gg3;
        asm volatile("v_cvt_f32_bf16 %0, %4\n"
                     "v_cvt_f32_bf16 %1, %4 src0_sel:WORD_1\n"
                     "v_cvt_f32_bf16 %2, %5\n"
                     "v_cvt_f32_bf16 %3, %5 src0_sel:WORD_1"
                     : "=&v"(gg0), "=&v"(gg1), "=&v"(gg2), "=&v"(gg3)
                     : "v"(g_pf_buf[iter][0]), "v"(g_pf_buf[iter][1]));
        float ww0, ww1, ww2, ww3;
        asm volatile("v_cvt_f32_bf16 %0, %4\n"
                     "v_cvt_f32_bf16 %1, %4 src0_sel:WORD_1\n"
                     "v_cvt_f32_bf16 %2, %5\n"
                     "v_cvt_f32_bf16 %3, %5 src0_sel:WORD_1"
                     : "=&v"(ww0), "=&v"(ww1), "=&v"(ww2), "=&v"(ww3)
                     : "v"(w_pf_buf[iter][0]), "v"(w_pf_buf[iter][1]));
        dp += ww0 * (v0 * gg0) + ww1 * (v1 * gg1) + ww2 * (v2 * gg2) +
              ww3 * (v3 * gg3);
#endif
      }
#endif

#if defined(MPK_ROUTER_FUSED_DP) && defined(MPK_ROUTER_DUAL_REDUCE)
      // Both partials are complete here, so they reduce together through one
      // interleaved permlane/DPP chain rather than two shuffle butterflies.
      // Only lane 0 is left holding the totals, which is what the cross-wave
      // publish below reads. See router_dual_wave_sum_to_lane_zero.
      router_dual_wave_sum_to_lane_zero(ssq, dp);
#else
// Wave-level reduction for ssq
#pragma unroll
      for (int off = 32; off > 0; off >>= 1) {
        ssq += __shfl_xor(ssq, off);
      }

#ifdef MPK_ROUTER_FUSED_DP
      // dp is already complete (unscaled) at this point, so it rides the same
      // cross-wave reduction as ssq instead of paying for its own barrier
      // after pass 2. Two shuffle chains, one LDS round trip, one
      // __syncthreads -- and the pair of slots stays split (red / red_dp)
      // because red[0] is still overwritten with the irms broadcast below.
#pragma unroll
      for (int off = 32; off > 0; off >>= 1) {
        dp += __shfl_xor(dp, off);
      }
#endif
#endif

      // Cross-wave reduction via LDS
      if (lane == 0) {
        red[wave] = ssq;
#ifdef MPK_ROUTER_FUSED_DP
        red_dp[wave] = dp;
#endif
      }
      __syncthreads();

      float irms;
      if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < NUM_WAVES; w++) {
          tot += red[w];
        }
        float const irms_t0 =
            rsqrtf(tot / (float)ACTUAL_HIDDEN_DIM + 1e-5f);
#ifdef MPK_ROUTER_FUSED_DP
        // The logit is finished here rather than after pass 2: irms is the
        // uniform scale factored out of every term, so applying it once to the
        // reduced dot product is the same quantity the per-element version
        // computed. See the fold in pass 1 for why that is legal.
        float s = 0.0f;
        for (int w = 0; w < NUM_WAVES; w++) {
          s += red_dp[w];
        }
        s *= irms_t0;
        if (d_rbias) {
          s += __bfloat162float(d_rbias[local_tile]);
        }
        bf16 bval = __float2bfloat16(s);
        st_wt_u16(&d_logits[(int64_t)b * NUM_EXPERTS + local_tile],
                  *reinterpret_cast<unsigned short *>(&bval));
#endif
        red[0] = irms_t0;
      }
#ifndef MPK_ROUTER_FUSED_DP
      __syncthreads();
      irms = red[0];
#endif

      // ── Pass 2: Register-only norm + GEMV (no HBM re-reads) ─────────
#ifdef MPK_ROUTER_FUSED_DP
      // With dp already reduced above, the only thing pass 2 still produces is
      // the normed row -- and only one workgroup per XCD stores it (see the
      // MPK_ONE_NORM_WRITER note below for why the election is per XCD and not
      // device-wide). So the other 15 skip the pass entirely rather than
      // computing a row they will throw away.
      //
      // Per XCD and not device-wide *also* makes this safe to gate: eliding
      // the pass elides no store that any other worker was relying on this
      // workgroup to make.
      //
      // Without MPK_ONE_NORM_WRITER every worker is a writer, so there is
      // nothing to skip and the pass runs for all of them -- the fold still
      // pays for itself there by dropping the weight unpack, the second
      // shuffle chain and one barrier. Keeping the two flags independent is
      // what lets the batch below attribute the delta to one of them.
#if defined(MPK_W13_PREQUANT) && defined(MPK_ONE_NORM_WRITER)
      // ── Split the publication across MAX_ITERS workgroups ────────────────
      //
      // Under MPK_W13_PREQUANT the elected writer no longer just stores a row;
      // it also quantizes it, and that landed on the critical path
      // (rmsnorm_router 1.88 -> 2.78 us) because one workgroup was doing all
      // 23 scale blocks while fifteen idled at the barrier below.
      //
      // The split is free of data movement. Every router workgroup ran the
      // whole of pass 1, so every one of them holds the entire row in
      // h_cache/g_cache -- the election was never about who *has* the data.
      // And a scale domain is 32 lanes of one iteration (scale_block =
      // i_cur/32 with i_cur = tid + iter*256), so iteration `k` owns blocks
      // [8k, 8k+8) exactly: no domain straddles the cut, and no cross-
      // workgroup reduction is needed.
      //
      // Workgroup `k` publishes iteration `k`. The row is 2944 elements =
      // MAX_ITERS(3) iterations, so three workgroups share it and the other
      // thirteen still skip pass 2 entirely. `router_tile_n` is 16 here; the
      // fallback keeps the single-writer form if a configuration ever has
      // fewer router tiles than iterations.
      bool const pq_split = (router_tile_n >= MAX_ITERS);
      bool const router_norm_writer =
          pq_split ? (local_tile < MAX_ITERS) : (local_tile == 0);
      int const pq_my_iter = pq_split ? local_tile : -1;
#elif defined(MPK_ONE_NORM_WRITER)
      bool const router_norm_writer = (local_tile == 0);
#ifdef MPK_W13_PREQUANT
      int const pq_my_iter = -1;
#endif
#else
      bool const router_norm_writer = true;
#ifdef MPK_W13_PREQUANT
      int const pq_my_iter = -1;
#endif
#endif
      // The irms broadcast barrier moves in here with its only consumer. Under
      // this flag the logit no longer needs irms per element -- tid 0 applied
      // it to the reduced dp above -- so the 15 non-writer workgroups have no
      // reason to read red[0] at all, and making them rendezvous for it is a
      // block-wide barrier bought for one workgroup in sixteen, so the
      // __syncthreads belongs inside the producer branch.
      if (router_norm_writer) {
        __syncthreads();
        irms = red[0];
      }
      if (router_norm_writer)
#endif
#pragma unroll
      for (int iter = 0; iter < MAX_ITERS; iter++) {
        if (iter >= n_cached) {
          break;
        }
#ifdef MPK_W13_PREQUANT
        // The split elects MAX_ITERS writers, one per iteration, but each of
        // them was still running the *whole* loop and throwing away every
        // iteration but its own -- the store was predicated, the unpack /
        // norm / quant above it were not. pq_my_iter is workgroup-uniform, so
        // this branch is uniform and skips the two dead iterations entirely.
        // Under the BF16 path the wasted work was a few v_cvt; under prequant
        // it is the whole ~10-op-per-element quant plus a 5-step shfl_xor
        // tree, which is why it showed up as rmsnorm_router 1.88 -> 2.78 us.
        if (pq_my_iter >= 0 && iter != pq_my_iter) {
          continue;
        }
#endif
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
#ifndef MPK_W13_PREQUANT
        i32x2_t n_packed;
        n_packed[0] = (int)pk_lo;
        n_packed[1] = (int)pk_hi;
#endif
#ifdef MPK_W13_PREQUANT
#ifndef MPK_ROUTER_FUSED_DP
#error "MPK_W13_PREQUANT requires MPK_ROUTER_FUSED_DP (it publishes from the elected norm writer, which that flag defines)"
#endif
        // ── Publish FP8 + E8M0 instead of BF16 ──────────────────────────────
        //
        // Every W13 tile re-derives the same thing from this row: an FP8 E4M3
        // payload with one E8M0 exponent per 128 elements. At bs=1 that is 184
        // workgroups each quantizing an identical 2944-element row, and the
        // ablation prices the whole of it at 0.047 ms. The row is already in
        // registers here, one workgroup per XCD already stores it, and the
        // quant is ~10 ALU ops per element -- so the work moves here and the
        // consumer becomes an LDS fill.
        //
        // It also halves the bytes: FP8 + scales is 2967 bytes against 5888
        // for BF16, read 184 times per layer per XCD.
        //
        // Bit-identity with the BF16 path is the design constraint, not a hope,
        // and it is what lets the generated-text hash verify this flag:
        //
        //   values  -- the consumer today reads back exactly the BF16 this
        //              loop packs, so the quant input must be the *round-
        //              tripped* value, not n0..n3. Hence the unpack of
        //              pk_lo/pk_hi below rather than using n0..n3 directly.
        //   blocks  -- a 128-element scale domain is i_cur/32, and i_cur is
        //              tid + iter*256, so a domain is 32 consecutive lanes of
        //              one iteration: lanes 0-31 or 32-63 of a wave. The
        //              consumer's domains are lanes 8b..8b+7 of a 16-per-lane
        //              mapping, i.e. the same 128 elements. fmax is exact and
        //              commutative, so a different reduction tree gives the
        //              same bits.
        //   scale   -- _gang_compute_e8m0_fp8, the consumer's own function.
        //   packing -- cvt_scalef32_pk_fp8_f32 over (q0,q1) then (q2,q3) with
        //              opsel, one dword per lane at byte offset 4*i_cur. The
        //              consumer writes the identical dword at the identical
        //              LDS offset.
        //
        // The last domain is short. ACTUAL_HIDDEN_DIM is 2880 and the row is
        // padded to output_stride (2944), so domain 22 holds 64 real elements
        // in 16 lanes plus 64 padding elements. Lanes 16..31 of that group
        // already broke out of this loop, so the cross-row shuffle is skipped
        // for it -- __shfl from an inactive lane returns nothing meaningful.
        // The padding itself is zeroed below, which is what the BF16 path got
        // for free from the buffer's zero-init.
        {
          float q0 = _gang_bf16_to_float((unsigned short)(pk_lo & 0xFFFF));
          float q1 = _gang_bf16_to_float((unsigned short)(pk_lo >> 16));
          float q2 = _gang_bf16_to_float((unsigned short)(pk_hi & 0xFFFF));
          float q3 = _gang_bf16_to_float((unsigned short)(pk_hi >> 16));

          // Four elements per lane, so the 128-element scale domain is 32
          // consecutive lanes: (i_cur_q * 4) / 128.
          int const i_cur_q = tid + iter * 256;
          int const scale_block = i_cur_q >> 5;
          int const last_block = (output_stride / 128) - 1;

          float amax = fmaxf(fmaxf(fabsf(q0), fabsf(q1)),
                             fmaxf(fabsf(q2), fabsf(q3)));
          // Butterfly over the 16-lane DPP row, then across the two rows of
          // the half-wave. Every lane in the domain ends up holding the full
          // amax, so no separate broadcast pass is needed.
          amax = fmaxf(amax, __shfl_xor(amax, 8));
          amax = fmaxf(amax, __shfl_xor(amax, 4));
          amax = fmaxf(amax, __shfl_xor(amax, 2));
          amax = fmaxf(amax, __shfl_xor(amax, 1));
          if (scale_block != last_block) {
            amax = fmaxf(amax, __shfl_xor(amax, 16));
          }

          uint8_t const se = _gang_compute_e8m0_fp8(amax);
          float scale_f = 1.0f;
          if (se != 0) {
            uint32_t const sbits = (uint32_t)se << 23;
            __builtin_memcpy(&scale_f, &sbits, sizeof(scale_f));
          }
          fp8x4_t pk = {};
          pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, q0, q1, scale_f,
                                                        false);
          pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, q2, q3, scale_f,
                                                        true);
          uint32_t pk_bits;
          __builtin_memcpy(&pk_bits, &pk, sizeof(pk_bits));

          // Same store flavour and the same per-XCD writer election as the
          // BF16 path below: a plain global_store into this XCD's L2, read
          // back by MoE workers on this XCD. See the MPK_ONE_NORM_WRITER note
          // for why the election is per XCD and not device-wide.
          bool const pq_store =
#if defined(MPK_ONE_NORM_WRITER)
              (pq_my_iter < 0) ? router_norm_writer : (iter == pq_my_iter);
#else
              router_norm_writer;
#endif
          if (pq_store) {
            asm volatile("global_store_dword %0, %1, off" ::"v"(n_base +
                                                                i_cur_q * 4),
                         "v"(pk_bits)
                         : "memory");
            if ((lane & 31) == 0) {
              asm volatile("global_store_byte %0, %1, off" ::"v"(
                               n_base + output_stride + scale_block),
                           "v"((unsigned)se)
                           : "memory");
            }
          }
        }
#elif defined(MPK_ONE_NORM_WRITER)
        // All `router_tile_n` router workers on this XCD compute the identical
        // normed row and store it to the identical global address, so 15 of
        // every 16 stores per XCD are pure redundant HBM traffic.
        //
        // Gated per XCD, not globally. This is a plain `global_store` into a
        // per-XCD L2 that is not coherent with the other seven, so the reason
        // a MoE worker on XCD k sees this row today is that XCD k's own router
        // workers dirtied k's L2 with it. Electing a single *device-wide*
        // producer would leave the other seven XCDs reading a line no one on
        // their die ever wrote. One writer per XCD keeps that locality intact
        // and still drops 128 writers to 8.
        if (local_tile == 0) {
          asm volatile("global_store_dwordx2 %0, %1, off" ::"v"(n_base +
                                                                byte_off),
                       "v"(n_packed)
                       : "memory");
        }
#else
        asm volatile("global_store_dwordx2 %0, %1, off" ::"v"(n_base +
                                                              byte_off),
                     "v"(n_packed)
                     : "memory");
#endif

#ifndef MPK_ROUTER_FUSED_DP
        // Unpack cached gate_weight and accumulate dot product
        float w0, w1, w2, w3;
        asm volatile("v_cvt_f32_bf16 %0, %4\n"
                     "v_cvt_f32_bf16 %1, %4 src0_sel:WORD_1\n"
                     "v_cvt_f32_bf16 %2, %5\n"
                     "v_cvt_f32_bf16 %3, %5 src0_sel:WORD_1"
                     : "=&v"(w0), "=&v"(w1), "=&v"(w2), "=&v"(w3)
                     : "v"(w_cache[iter][0]), "v"(w_cache[iter][1]));
        dp += w0 * n0 + w1 * n1 + w2 * n2 + w3 * n3;
#endif
      }

#ifdef MPK_W13_PREQUANT
      // Zero the padded tail, elements ACTUAL_HIDDEN_DIM..output_stride.
      //
      // The BF16 path got this for free: it never wrote the tail, and the
      // buffer is zero-initialised, so the consumer's quant read zeros there.
      // FP8 halves the addresses, so the tail now lands where BF16 kept *real*
      // data -- byte 2880 is element 1440's low half under BF16 -- and leaving
      // it unwritten would feed the MFMA whatever was there.
      //
      // The scale byte for the last block needs no adjustment: its amax was
      // taken over the real elements only, and fmax against the 64 zeros the
      // consumer used to see cannot change a non-negative maximum. Same byte,
      // same payload for the real half, zeros for the pad.
      if (router_norm_writer && (pq_my_iter <= 0)) {
        int const pad_dw = (output_stride - ACTUAL_HIDDEN_DIM) / 4;
        if ((int)tid < pad_dw) {
          asm volatile("global_store_dword %0, %1, off" ::"v"(
                           n_base + ACTUAL_HIDDEN_DIM + (int)tid * 4),
                       "v"(0u)
                       : "memory");
        }
      }
#endif

#ifndef MPK_ROUTER_FUSED_DP
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
#endif // !MPK_ROUTER_FUSED_DP

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
#ifdef MPK_OPROJ_INNER_TIMING
  _op_t3 = __builtin_amdgcn_s_memrealtime();
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
#ifdef MPK_ROUTING_LANE_RELEASE
    // Carries the release epoch from tid 0 to lanes 1..7 of wave 0 by
    // readfirstlane; see the fan-out at the end of this block. Declared here
    // rather than at the top of the phase because `goto done` above jumps past
    // that point and may not cross an initialization. Zero means "not the
    // releaser".
    int rr_epoch = 0;
#endif
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
        num_active_tokens
#ifdef MPK_EARLY_ROUTING
        ,
        routing_ready_ptr,
        (unsigned)layer_epoch
#endif
    );
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
      //
      // MPK_OPROJ_NO_WB: the argument above says "ordinary st_wt stores" and
      // then concludes a GPU-scope fence is needed. Those two halves don't fit
      // together. st_wt IS the write-through store -- st_wt_u32/st_wt_u16 emit
      // `global_store_dword{,x2} ... sc0 sc1`, bypassing L2 for HBM (see every
      // publication in moe_topk_softmax_mi300.cuh: routing_indices,
      // active_expert_ids, topk output, the count at [NUM_EXPERTS]). The
      // `s_waitcnt vmcnt(0)` immediately above, run per-wave before the
      // __syncthreads, already retires all of them at HBM. There is no L2
      // residency for buffer_wbl2 to write back.
      //
      // What remains true is the *ordering* requirement -- the release flags
      // must not land before the routing data -- and the drain provides it.
      // The whole-L2 writeback does not add ordering the drain lacks.
      //
      // Only one worker per layer runs this, but it is the worker every other
      // XCD's Phase 7b poll is waiting on, so its latency is on the critical
      // path rather than amortized. Same reasoning as the Phase 2 arrival
      // fence above; both are under the one flag because they stand or fall on
      // the same fact about st_wt.
#ifndef MPK_OPROJ_NO_WB
      threadfence_gpu();
#endif
      if (routing_ready_ptr) {
        // ── The epoch is derived, not read back ───────────────────────────
        //
        // `ld_nt_s32(routing_ready_ptr) + 1` is a dependent, cache-bypassing
        // HBM load sitting in front of the nine stores that release all 240
        // workgroups -- the single most-awaited publication in the layer, and
        // the one thread on the GPU whose latency nothing hides. The consumer
        // side never pays it: the Phase 7b poll in
        // gang_full_layer_fused_mi300.cuh compares against `layer_counter + 1`
        // computed locally.
        //
        // `layer_epoch` is that same counter -- (iterations * num_layers +
        // layer) + 1, monotonic, never reset, identical for every worker --
        // already passed in for the O-proj hierarchical barrier a few hundred
        // lines up, which stopped snapshotting for exactly this reason (see
        // `oproj_release_expected`). Using it here makes producer and consumer
        // derive the release value the same way instead of one of them reading
        // what the other wrote.
        //
        // The standalone task type passes layer_epoch == 0 (it has no layer
        // loop), so that path keeps the read-back, which is safe there because
        // there is no second layer to race with.
#ifdef MPK_ROUTING_DERIVED_EPOCH
        int epoch = layer_epoch > 0 ? layer_epoch
                                    : ld_nt_s32(routing_ready_ptr) + 1;
#else
        int epoch = ld_nt_s32(routing_ready_ptr) + 1;
#endif
        st_wt_u32((void *)routing_ready_ptr, (unsigned)epoch);
#ifdef MPK_ROUTING_LANE_RELEASE
        rr_epoch = epoch;
#else
        for (int x = 0; x < 8; x++) {
          st_wt_u32((void *)&routing_ready_ptr[(1 + x) * 16], (unsigned)epoch);
        }
#endif
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
#ifdef MPK_ROUTING_LANE_RELEASE
    // ── Fan the eight per-XCD flags out over lanes, not over time ─────────
    //
    // TESTED AND NOT ADOPTED, twice. Interleaved A/B, 5 rounds each: this
    // readfirstlane form measures 2.022 against 2.025 for the serial loop --
    // inside the noise. An earlier version that carried the epoch through LDS
    // measured 2.042 against 2.017, i.e. clearly worse, because its
    // __syncthreads lands at the exact point 240 workgroups are waiting on
    // this block. Correct on both arms (hash 88861a4763c0 throughout).
    //
    // Why the transform pays elsewhere and not here: the seven store-issues it
    // removes are ~100 ns against the ~5.8 us this phase's poll actually
    // costs, and that 5.8 us is the TopK compute the release is waiting for,
    // not the release's own latency. Kept behind the flag rather than deleted
    // because the reasoning below is the same one that did pay at the two
    // other barriers, and this is the measurement that bounds it.
    //
    // This is the single most-awaited store in the layer: every worker on all
    // eight XCDs is parked in the Phase 7b poll on one of these lines, and
    // until now one lane issued all eight back to back, so XCD 7 was released
    // seven `sc0 sc1` store-issues after XCD 0. The lines are 16 ints apart
    // by design, so lanes 0..7 differ only in address and the eight stores
    // collapse into one `global_store_dword`.
    //
    // Same transform, same argument, as the O-proj hierarchical release a few
    // hundred lines up and Phase 9's layer release -- both already do this and
    // both measured for it. It does *not* apply at Phase 5's attention release
    // (measured 2.047 vs 2.038 there), because those consumers arrive spread
    // out rather than pre-parked; here all 240 are already spinning.
    //
    // The epoch reaches lanes 1..7 by readfirstlane, NOT through LDS. An
    // earlier version put it in a __shared__ int and rendezvoused; that
    // measured 2.042 against 2.017 for the serial loop, because the
    // __syncthreads lands at the one point in the layer where 240 workgroups
    // are waiting on this block and it costs more than the seven store-issues
    // it removes. readfirstlane needs no barrier: tid 0..7 are the same wave,
    // the `if (tid == 0)` above has closed so all lanes are live, and
    // readfirstlane reads the first active lane. Zero is a safe "no release"
    // sentinel -- a real epoch is >= 1 -- so the `routing_ready_ptr ==
    // nullptr` path needs no separate test.
    //
    // Ordering is unchanged from the serial form: the `s_waitcnt vmcnt(0)`
    // inside tid 0's block retires both the routing data and its aggregate
    // store to routing_ready_ptr[0] before any of these eight flags issue.
    rr_epoch = __builtin_amdgcn_readfirstlane(rr_epoch);
    if (rr_epoch != 0 && tid < 8) {
      st_wt_u32((void *)&routing_ready_ptr[(1 + tid) * 16], (unsigned)rr_epoch);
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
#endif
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
#ifdef MPK_OPROJ_INNER_TIMING
  // Inner split of Phase 7, printed by one worker per XCD so the four numbers
  // come from a single consistent timeline. `_op_t3` is taken before the TopK
  // barrier, so `topk` here folds in that wait -- the interesting comparison is
  // mfma vs the rest, since mfma is the only part that is bandwidth-bound.
  // Every tile prints, not just one per XCD: the question this is here to
  // answer is whether `bar` is release latency or arrival skew, and skew is
  // only visible as a spread across tiles. `t1` is the raw arrival tick, kept
  // so the spread of arrivals within one layer can be measured directly.
  // One tile per XCD. Printing all 184 floods the printf buffer badly enough
  // to stall the run; it was done once, off this flag, to establish that the
  // barrier wait is protocol latency and not arrival skew (spread 0.14 us,
  // first and last arriver both waiting ~5.6 us).
  if (tid == 0 && (tile_idx % tiles_per_xcd) == 0) {
    unsigned long long _op_t4 = __builtin_amdgcn_s_memrealtime();
    printf("[OPROJ_INNER] mfma=%.2f bar=%.2f "
           "rmsnorm_router=%.2f topk=%.2f total=%.2f\n",
           (double)(_op_t1 - _op_t0) * 10.0 / 1000.0,
           (double)(_op_t2 - _op_t1) * 10.0 / 1000.0,
           (double)(_op_t3 - _op_t2) * 10.0 / 1000.0,
           (double)(_op_t4 - _op_t3) * 10.0 / 1000.0,
           (double)(_op_t4 - _op_t0) * 10.0 / 1000.0);
  }
#endif
  (void)0;
}

} // namespace kernel
