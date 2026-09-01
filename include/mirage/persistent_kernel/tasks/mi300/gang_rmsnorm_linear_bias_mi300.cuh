/* Fused RMSNorm + Gang Linear + Bias for MI300/MI350.
 *
 * Eliminates the dispatch barrier between RMSNorm and the QKV/router gang
 * linear by having every gang-linear worker compute the RMSNorm prologue
 * locally before its MFMA. All workers compute the same value and write to
 * the same output buffer; concurrent writes are safe (idempotent).
 *
 * Saves ~4-5us per fused op × 2 RMSNorm tasks per layer × 36 layers per
 * iteration (~300us total) by removing 72 task dispatch barriers per token.
 *
 * NOTE: this differs from gang_rmsnorm_linear_mi300.cuh in two ways:
 *   1. Correct AMD wavefront=64 cross-wave reduction (the existing file uses
 *      >>5 which is wrong on AMD and would over-count by 2x).
 *   2. Supports ACTUAL_HIDDEN_DIM != STORAGE_DIM for GPT-OSS (which pads
 *      hidden=2880 to STORAGE_DIM=3072 for tile alignment).
 */
#pragma once
#include "gang_linear_mi300.cuh"
#include "moe_topk_softmax_mi300.cuh"
#include <hip/hip_bf16.h>

namespace kernel {

#ifdef MPK_RMSNORM_DPP_REDUCE
// ── The RMSNorm sum-of-squares wave reduction, off the LDS data path ────────
//
// Every RMSNorm prologue in this pipeline finishes its accumulation loop with
// the same six-step butterfly -- `ssq += __shfl_xor(ssq, off)` for off in
// 32/16/8/4/2/1 -- and then reads lane 0 and nothing else. `__shfl_xor`
// lowers to ds_bpermute, so that is six LDS round trips and six lgkmcnt waits
// for data that never left the register file.
//
// All six steps have a VALU-native equivalent on gfx950:
// `v_permlane32_swap_b32` is xor-32, `v_permlane16_swap_b32` is xor-16, and
// DPP `row_shl` covers the four intra-row steps. Nothing here touches LDS.
//
// Two details are worth stating because both are easy to get wrong:
//
// The `s_nop 1` before each stage is required, not defensive. Table 11 of the
// CDNA4 ISA ("Required Software-inserted Wait States") lists two wait states
// for `VALU writes VGPR` -> `VALU DPP reads that VGPR`, and the same two for
// `VALU writes vdst` -> `V_PERMLANE* reads vdst`. Each stage reads a register
// the previous stage's add just wrote, so each needs the gap. There is no
// listed hazard in the other direction, which is why the adds follow their
// swap immediately.
//
// The four row_shl stages fold the move into the add: `v_add_f32` takes the
// DPP swizzle on src0, so one instruction does what a `v_mov_b32_dpp` plus a
// `v_add_f32` would. Out-of-range lanes read zero under `bound_ctrl:1` and
// adding zero is a no-op, exactly as it was when the move materialised it.
//
// ── Contract: the result is valid in lane 0 ────────────────────────────────
//
// This is where it differs from the butterfly it replaces. `__shfl_xor`
// leaves the total in all 64 lanes; row_shl accumulates toward lane 0 of each
// row of 16 and leaves the rest holding partial sums. Every caller here reads
// `if (lane == 0) red[wave] = ssq` and nothing else, so that is all that is
// promised. Hence the name.
//
// Bit-identical at lane 0: the association tree is the same 32/16/8/4/2/1 the
// shuffles walked, so this does not reassociate the sum. Verified over 8192
// cases including mixed-magnitude inputs, where a different tree rounds away
// a different set of addends. See tests/standalone/test_rmsnorm_dpp_reduce.hip.
__device__ __forceinline__ void rmsnorm_wave_sum_to_lane_zero(float &ssq) {
  // The swap exchanges one operand's low half with the other's high half, so
  // seeding the peer with a copy of ssq turns swap-plus-add into the butterfly
  // step. The DPP stages need no peer at all.
  float peer = ssq;
  asm volatile("s_nop 1\n"
               "v_permlane32_swap_b32 %[s], %[p]\n"
               "v_add_f32 %[s], %[s], %[p]\n"
               "v_mov_b32_e32 %[p], %[s]\n"
               "s_nop 1\n"
               "v_permlane16_swap_b32 %[s], %[p]\n"
               "v_add_f32 %[s], %[s], %[p]\n"
               "s_nop 1\n"
               "v_add_f32_dpp %[s], %[s], %[s] row_shl:8 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "s_nop 1\n"
               "v_add_f32_dpp %[s], %[s], %[s] row_shl:4 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "s_nop 1\n"
               "v_add_f32_dpp %[s], %[s], %[s] row_shl:2 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "s_nop 1\n"
               "v_add_f32_dpp %[s], %[s], %[s] row_shl:1 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1"
               : [s] "+v"(ssq), [p] "+v"(peer));
}
#endif

namespace gang_rmsnorm_detail {
using bf16 = __hip_bfloat16;

// AMD-correct in-place RMSNorm prologue. Computed redundantly by all workers
// in the workgroup; result written to global memory at output_ptr.
//
// STORAGE_DIM: number of bf16 elements stored along the hidden axis (padded)
// ACTUAL_HIDDEN_DIM: divisor for the RMS mean (unpadded hidden size)
//
// The norm weight is assumed zero-padded past ACTUAL_HIDDEN_DIM, so the
// padded tail multiplies by zero. Sum-of-squares is taken over STORAGE_DIM
// (the padded tail is also zero in input, since the producer pads with 0),
// but the divisor uses ACTUAL_HIDDEN_DIM.
template <int STORAGE_DIM, int ACTUAL_HIDDEN_DIM>
__device__ __forceinline__ void rmsnorm_inline_amd(void const *input_ptr,
                                                   void const *weight_ptr,
                                                   void *output_ptr,
                                                   float eps = 1e-5f) {
  bf16 const *__restrict__ d_input = static_cast<bf16 const *>(input_ptr);
  bf16 const *__restrict__ d_weight = static_cast<bf16 const *>(weight_ptr);
  bf16 *__restrict__ d_output = static_cast<bf16 *>(output_ptr);

  constexpr int VEC_SIZE = 8;   // 8 bf16 per 128-bit load
  constexpr int NTHREADS = 256; // block size for gang RMSNorm
  int const tid = threadIdx.x;
  int const nthreads = blockDim.x;
  constexpr int VEC_ITERS = STORAGE_DIM / (NTHREADS * VEC_SIZE);

  // ── Phase 1: sum of squares + cache input in registers ──
  // Cache d_input values to avoid re-reading from HBM in Phase 4.
  // GPT-OSS 120B: STORAGE_DIM=3072, 256 threads, VEC_SIZE=8 → VEC_ITERS=1,
  // tail=1024 elems → 4/thread. Total cache: 8 + 4 = 12 floats/thread.
  constexpr int _VEC_END = VEC_ITERS * NTHREADS * VEC_SIZE;
  constexpr int _TAIL_ELEMS = STORAGE_DIM - _VEC_END;
  constexpr int _MAX_TAIL = (_TAIL_ELEMS + NTHREADS - 1) / NTHREADS;
  float sum = 0.0f;

  // Vectorized cache: raw uint64_t pairs (2 per iter = lo + hi)
  uint64_t in_cache_lo[VEC_ITERS > 0 ? VEC_ITERS : 1];
  uint64_t in_cache_hi[VEC_ITERS > 0 ? VEC_ITERS : 1];
#pragma unroll 1
  for (int v = 0; v < VEC_ITERS; v++) {
    int offset = (v * nthreads + tid) * VEC_SIZE;
    uint64_t in_lo = *reinterpret_cast<uint64_t const *>(&d_input[offset]);
    uint64_t in_hi = *reinterpret_cast<uint64_t const *>(&d_input[offset + 4]);
    in_cache_lo[v] = in_lo;
    in_cache_hi[v] = in_hi;
    bf16 const *lo = reinterpret_cast<bf16 const *>(&in_lo);
    bf16 const *hi = reinterpret_cast<bf16 const *>(&in_hi);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float vlo = __bfloat162float(lo[i]);
      sum += vlo * vlo;
      float vhi = __bfloat162float(hi[i]);
      sum += vhi * vhi;
    }
  }
  // Scalar tail cache
  float tail_cache[_MAX_TAIL > 0 ? _MAX_TAIL : 1];
  int n_tail = 0;
  int const VEC_END = _VEC_END;
  for (int i = VEC_END + tid; i < STORAGE_DIM; i += nthreads) {
    float val = __bfloat162float(d_input[i]);
    tail_cache[n_tail++] = val;
    sum += val * val;
  }

// ── Phase 2: wavefront reduction (AMD wavefront = 64 lanes) ──
  #ifdef MPK_RMSNORM_DPP_REDUCE
  // Same 32/16/8/4/2/1 tree, no LDS. Valid in lane 0 only, which is
  // all the cross-wave publish below reads -- see the contract note at
  // rmsnorm_wave_sum_to_lane_zero.
  rmsnorm_wave_sum_to_lane_zero(sum);
  #else
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }
  #endif

  // ── Phase 3: cross-wavefront reduction via shared memory ──
  // Use a static __shared__ array — independent of CK pipeline's dynamic LDS.
  __shared__ float red[16]; // up to 16 wavefronts (1024 threads)
  int wave_id = tid >> 6;
  int lane_id = tid & 63;
  int num_waves = nthreads >> 6;

  if (lane_id == 0) {
    red[wave_id] = sum;
  }
  __syncthreads();

  if (wave_id == 0) {
    sum = (lane_id < num_waves) ? red[lane_id] : 0.0f;
    // Reduce across num_waves (typically 4 for blockDim=256)
    for (int offset = num_waves >> 1; offset > 0; offset >>= 1) {
      sum += __shfl_xor(sum, offset);
    }
    if (lane_id == 0) {
      red[0] = sum;
    }
  }
  __syncthreads();

  float rms_rcp = rsqrtf(red[0] / float(ACTUAL_HIDDEN_DIM) + eps);

// ── Phase 4: apply normalization using cached input (no HBM re-read) ──
#pragma unroll 1
  for (int v = 0; v < VEC_ITERS; v++) {
    int offset = (v * nthreads + tid) * VEC_SIZE;
    // Reuse cached input from Phase 1
    uint64_t in_lo = in_cache_lo[v];
    uint64_t in_hi = in_cache_hi[v];
    uint64_t w_lo = *reinterpret_cast<uint64_t const *>(&d_weight[offset]);
    uint64_t w_hi = *reinterpret_cast<uint64_t const *>(&d_weight[offset + 4]);
    bf16 const *in_lo_a = reinterpret_cast<bf16 const *>(&in_lo);
    bf16 const *in_hi_a = reinterpret_cast<bf16 const *>(&in_hi);
    bf16 const *w_lo_a = reinterpret_cast<bf16 const *>(&w_lo);
    bf16 const *w_hi_a = reinterpret_cast<bf16 const *>(&w_hi);

    bf16 out[VEC_SIZE];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      out[i] = __float2bfloat16(__bfloat162float(in_lo_a[i]) * rms_rcp *
                                __bfloat162float(w_lo_a[i]));
      out[4 + i] = __float2bfloat16(__bfloat162float(in_hi_a[i]) * rms_rcp *
                                    __bfloat162float(w_hi_a[i]));
    }
    *reinterpret_cast<uint64_t *>(&d_output[offset]) =
        *reinterpret_cast<uint64_t *>(&out[0]);
    *reinterpret_cast<uint64_t *>(&d_output[offset + 4]) =
        *reinterpret_cast<uint64_t *>(&out[4]);
  }
  // Scalar tail: use cached float values
  int ti = 0;
  for (int i = VEC_END + tid; i < STORAGE_DIM; i += nthreads) {
    float val = tail_cache[ti++];
    float w = __bfloat162float(d_weight[i]);
    d_output[i] = __float2bfloat16(val * rms_rcp * w);
  }

  // Workgroup-scope fence: ensure stores visible to the linear loads below
  // within this workgroup. (Cross-XCD coherence not needed: each XCD
  // computes the same value redundantly and reads its own writes.)
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __syncthreads();
}

} // namespace gang_rmsnorm_detail

// Fused RMSNorm + Gang Linear + Bias.
//
// Step 1: every worker on every XCD computes the same normalized output to
//         norm_output_ptr (idempotent concurrent writes).
// Step 2: this workgroup's gang-linear tile reads from norm_output_ptr,
//         applies bias in the epilogue.
template <typename T,
          int BATCH_SIZE,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM = REDUCTION_SIZE>
__device__ __forceinline__ void gang_rmsnorm_linear_bias_kernel(
    void const *norm_input_ptr,  // [batch, REDUCTION_SIZE]
    void const *norm_weight_ptr, // [REDUCTION_SIZE]  (zero-padded past ACTUAL)
    void *norm_output_ptr,       // [batch, REDUCTION_SIZE]
    void const *linear_weight_ptr, // [chunk_N, REDUCTION_SIZE]
    void const *bias_ptr,          // [1, full_N]
    void *linear_output_ptr,       // [batch, o_stride]
    int num_active_tokens,
    int tile_n,
    int o_stride,
    int m_tiles,
    int n_tiles,
    int wgm,
    int tile_idx) {
  // Step 1: redundant RMSNorm.
  gang_rmsnorm_detail::rmsnorm_inline_amd<REDUCTION_SIZE, ACTUAL_HIDDEN_DIM>(
      norm_input_ptr, norm_weight_ptr, norm_output_ptr);

  // Step 2: gang linear with bias, reading from norm_output_ptr.
  gang_linear_kernel<T, BATCH_SIZE, REDUCTION_SIZE>(norm_output_ptr,
                                                    linear_weight_ptr,
                                                    linear_output_ptr,
                                                    num_active_tokens,
                                                    tile_n,
                                                    o_stride,
                                                    m_tiles,
                                                    n_tiles,
                                                    wgm,
                                                    tile_idx,
                                                    bias_ptr);
}

// Fused gate + TopK: 128 workers (one per expert).
//
// Replaces the 8-worker small_router_linear approach with 128-way parallel
// dot products:
//   - Each worker computes 1 expert's gate logit (not 16 serially)
//   - RMSNorm is fused into the GEMV (no intermediate BF16 round-trip)
//   - 128 workers = 16/XCD, vs old 1/XCD = 8x more parallelism
//   - Expected: ~3-5 us/layer (down from 25.3 us/layer)
//
// Cross-XCD synchronization: same atomic counter barrier as before.
// The LAST worker (count == 128) computes TopK softmax inline.
//
// Memory ordering: write-through stores (sc0 sc1) for logits bypass L2.
// buffer_inv on the TopK reader invalidates stale L2 entries.
namespace gang_rmsnorm_topk_detail {
using bf16_t = __hip_bfloat16;

__device__ __forceinline__ int get_xcd_id() {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
#else
  return 0;
#endif
}

// Noinline wrapper for TopK to prevent code bloat in the megakernel's hot path.
// Without this, inlining TopK into the persistent_kernel function causes the
// compiler's instruction scheduler to pessimize the CK GEMM pipeline, making
// ALL task types ~5-12% slower.
// ROUTING_ROW_STRIDE is the *allocated* row stride of routing_indices
// (BATCH_SIZE), which is not num_active_tokens -- see the note on
// topk_softmax_mi300_task_impl's routing_row_stride parameter.
template <typename T, int NUM_EXPERTS, int K, int ROUTING_ROW_STRIDE>
__device__ __attribute__((noinline)) void
    topk_noinline(void *logits_scratch_ptr,
                  void *topk_weight_ptr,
                  void *routing_indices_ptr,
                  void *active_expert_ids_ptr,
                  void *gang_counter_ptr,
                  int num_active_tokens) {
  constexpr int CHUNK_N = NUM_EXPERTS / 8;
  int xcd_id = get_xcd_id();
  void *logits_base = static_cast<T *>(logits_scratch_ptr) -
                      static_cast<int64_t>(xcd_id) * CHUNK_N;

  // Invalidate before reading logits. The 128 producing workers wrote their
  // logit with st_wt_u16 (sc0 sc1), which bypasses L2 and lands in HBM, and
  // they sit on all 8 XCDs while this reader is on one -- MI300/MI350 L2 is
  // not coherent across XCDs.
  //
  // This was `buffer_inv sc1` (vL1 + L2) on that reasoning; it is now plain
  // `buffer_inv` (vL1 only). See the layer-boundary acquire at the top of
  // gang_full_layer_fused_mi300.cuh for why vL1-only is sufficient under the
  // Phase 9 layer barrier, and for the ablation that established it.
  //
  // A stale logit does not crash and does not look like garbage: it is a
  // real logit, so TopK returns real experts and routing stays in range. The
  // token is simply routed through the wrong experts. The signature to watch
  // for is moe_routing_indices / moe_mask differing between runs while
  // attention is bit-identical.
  asm volatile("buffer_inv" ::: "memory");

  topk_softmax_mi300_task_impl<T,
                               /*VPT=*/8,
                               NUM_EXPERTS,
                               /*WARPS_PER_CTA=*/4,
                               /*BYTES_PER_LDG=*/16>(logits_base,
                                                     topk_weight_ptr,
                                                     num_active_tokens,
                                                     ROUTING_ROW_STRIDE,
                                                     K,
                                                     routing_indices_ptr,
                                                     active_expert_ids_ptr,
                                                     0,
                                                     NUM_EXPERTS,
                                                     true);

  // Reset counter for the next layer's use.
  if (threadIdx.x == 0) {
    *static_cast<int *>(gang_counter_ptr) = 0;
  }
}
} // namespace gang_rmsnorm_topk_detail

// Fused RMSNorm + Gate GEMV + TopK.
//
// 128 workers (16/XCD), one expert per worker. Each worker:
//   1. Computes RMSNorm (irms) from hidden state — redundant, same result
//   2. Fused GEMV: dp += gate_w[expert, i] * (hidden[i] * irms * gamma[i])
//      Also writes norm_output (for downstream MoE FP8 quant) as side-effect
//   3. Writes 1 logit via write-through store
//   4. atomicAdd barrier; last worker runs TopK softmax
template <typename T,
          int BATCH_SIZE,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM,
          int NUM_EXPERTS,
          int K>
__device__ __attribute__((noinline)) void gang_rmsnorm_linear_bias_topk_kernel(
    void const *norm_input_ptr,  // input_ptrs[0]: [batch, REDUCTION_SIZE]
    void const *norm_weight_ptr, // input_ptrs[1]: [REDUCTION_SIZE]
    void *norm_output_ptr, // input_ptrs[2]: [batch, REDUCTION_SIZE] scratch
    void const
        *gate_weight_ptr,     // input_ptrs[3]: [chunk_N, REDUCTION_SIZE] bf16
    void const *bias_ptr,     // input_ptrs[4]: [chunk_N] bf16 (XCD-partitioned)
    void *logits_scratch_ptr, // input_ptrs[5]: XCD-partitioned [batch, chunk_N]
    void *gang_counter_ptr,   // input_ptrs[6]: [1] int32 atomic counter
    void *topk_weight_ptr,    // output_ptrs[0]: [batch, K] float
    void *routing_indices_ptr,   // output_ptrs[1]: [NUM_EXPERTS, batch] int32
    void *active_expert_ids_ptr, // output_ptrs[2]: [NUM_EXPERTS+1] int32
    int num_active_tokens,
    int tile_n,
    int o_stride,
    int m_tiles,
    int n_tiles,
    int wgm,
    int tile_idx,
    int total_gang_tiles) {

  using bf16 = __hip_bfloat16;
  bf16 const *__restrict__ d_hidden = static_cast<bf16 const *>(norm_input_ptr);
  bf16 const *__restrict__ d_gamma = static_cast<bf16 const *>(norm_weight_ptr);
  bf16 *__restrict__ d_normed = static_cast<bf16 *>(norm_output_ptr);
  bf16 const *__restrict__ d_gate_w =
      static_cast<bf16 const *>(gate_weight_ptr);
  bf16 const *__restrict__ d_bias = static_cast<bf16 const *>(bias_ptr);
  bf16 *__restrict__ d_logits = static_cast<bf16 *>(logits_scratch_ptr);

  int const tid = threadIdx.x;
  int const lane = tid & 63;
  int const wave = tid >> 6;
  constexpr int NUM_WAVES = 4; // 256 threads / 64 lanes

  // ═══ Step 1: RMSNorm — compute irms from hidden state ═══
  // All 256 threads collaborate. Vectorized 4-wide bf16 loads.
  float ssq = 0.0f;
  {
    int const h4 = REDUCTION_SIZE >> 2;
    for (int i = tid; i < h4; i += (int)blockDim.x) {
      int base = i * 4;
      float v0 = __bfloat162float(d_hidden[base]);
      float v1 = __bfloat162float(d_hidden[base + 1]);
      float v2 = __bfloat162float(d_hidden[base + 2]);
      float v3 = __bfloat162float(d_hidden[base + 3]);
      ssq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }
    // Scalar tail
    for (int i = (h4 << 2) + tid; i < REDUCTION_SIZE; i += (int)blockDim.x) {
      float v = __bfloat162float(d_hidden[i]);
      ssq += v * v;
    }
  }

// Wave-level reduction (64 lanes)
  #ifdef MPK_RMSNORM_DPP_REDUCE
  // Same 32/16/8/4/2/1 tree, no LDS. Valid in lane 0 only, which is
  // all the cross-wave publish below reads -- see the contract note at
  // rmsnorm_wave_sum_to_lane_zero.
  rmsnorm_wave_sum_to_lane_zero(ssq);
  #else
#pragma unroll
  for (int off = 32; off > 0; off >>= 1) {
    ssq += __shfl_xor(ssq, off);
  }
  #endif

  // Cross-wave reduction via LDS.
  //
  // `red` carries ssq and then, at red[0], the irms *broadcast* that all 256
  // threads read in step 2. The dp reduction at the end of step 2 therefore
  // gets its own slots: writing red[wave] there would put wave 0's dp on top
  // of the irms every other wave is still reading, and irms is consumed inside
  // step 2's loop so the read is not necessarily hoisted. See the same fix and
  // the full failure analysis in
  // gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh, which is where this
  // pipeline was copied from.
  __shared__ float red[16];
  __shared__ float red_dp[16];
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

  // ═══ Step 2: Fused Gate GEMV + norm write ═══
  // One expert per worker. Each thread handles REDUCTION_SIZE/blockDim.x
  // elements in a single pass: compute normed, write to d_normed (for
  // downstream MoE FP8 quant), accumulate gate dot product.
  //
  // Gate weight layout: [chunk_N, REDUCTION_SIZE] bf16, row-major.
  // tile_idx = expert index within this XCD (0..chunk_N-1).
  float dp = 0.0f;
  bf16 const *my_gate = d_gate_w + tile_idx * REDUCTION_SIZE;

  {
    int const h4 = REDUCTION_SIZE >> 2;
    for (int i = tid; i < h4; i += (int)blockDim.x) {
      int base = i * 4;
      float h0 = __bfloat162float(d_hidden[base]);
      float h1 = __bfloat162float(d_hidden[base + 1]);
      float h2 = __bfloat162float(d_hidden[base + 2]);
      float h3 = __bfloat162float(d_hidden[base + 3]);

      float g0 = __bfloat162float(d_gamma[base]);
      float g1 = __bfloat162float(d_gamma[base + 1]);
      float g2 = __bfloat162float(d_gamma[base + 2]);
      float g3 = __bfloat162float(d_gamma[base + 3]);

      float n0 = h0 * irms * g0;
      float n1 = h1 * irms * g1;
      float n2 = h2 * irms * g2;
      float n3 = h3 * irms * g3;

      // Write normed output (redundant across 128 workers, idempotent).
      // Needed by downstream MoE FP8 quant task.
      d_normed[base] = __float2bfloat16(n0);
      d_normed[base + 1] = __float2bfloat16(n1);
      d_normed[base + 2] = __float2bfloat16(n2);
      d_normed[base + 3] = __float2bfloat16(n3);

      // Gate GEMV: accumulate dot product
      float w0 = __bfloat162float(my_gate[base]);
      float w1 = __bfloat162float(my_gate[base + 1]);
      float w2 = __bfloat162float(my_gate[base + 2]);
      float w3 = __bfloat162float(my_gate[base + 3]);
      dp += w0 * n0 + w1 * n1 + w2 * n2 + w3 * n3;
    }
    // Scalar tail
    for (int i = (h4 << 2) + tid; i < REDUCTION_SIZE; i += (int)blockDim.x) {
      float h = __bfloat162float(d_hidden[i]);
      float g = __bfloat162float(d_gamma[i]);
      float n = h * irms * g;
      d_normed[i] = __float2bfloat16(n);
      dp += __bfloat162float(my_gate[i]) * n;
    }
  }

// Wave-level reduction for dp
  #ifdef MPK_RMSNORM_DPP_REDUCE
  // Same 32/16/8/4/2/1 tree, no LDS. Valid in lane 0 only, which is
  // all the cross-wave publish below reads -- see the contract note at
  // rmsnorm_wave_sum_to_lane_zero.
  rmsnorm_wave_sum_to_lane_zero(dp);
  #else
#pragma unroll
  for (int off = 32; off > 0; off >>= 1) {
    dp += __shfl_xor(dp, off);
  }
  #endif

  // Cross-wave LDS reduce. Into red_dp, not red: red[0] is still the irms
  // broadcast that step 2 above reads.
  if (lane == 0) {
    red_dp[wave] = dp;
  }
  __syncthreads();

  // tid==0 writes logit + bias via write-through store
  if (tid == 0) {
    float s = 0.0f;
    for (int w = 0; w < NUM_WAVES; w++) {
      s += red_dp[w];
    }
    if (d_bias) {
      s += __bfloat162float(d_bias[tile_idx]);
    }
    bf16 bval = __float2bfloat16(s);
    st_wt_u16(&d_logits[tile_idx], *reinterpret_cast<unsigned short *>(&bval));
  }

  // ═══ Step 3: Cross-XCD barrier via atomic counter ═══
  // Write-through stores (sc0 sc1) bypass L2 → HBM. s_waitcnt ensures
  // stores are globally visible before incrementing counter.
  __syncthreads();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

  __shared__ int s_completed;
  if (tid == 0) {
    s_completed = atomicAdd(static_cast<int *>(gang_counter_ptr), 1) + 1;
  }
  __syncthreads();
  int completed = s_completed;

  // ═══ Step 4: Last worker runs TopK softmax ═══
  if (completed == total_gang_tiles) {
    gang_rmsnorm_topk_detail::topk_noinline<T, NUM_EXPERTS, K, BATCH_SIZE>(
        logits_scratch_ptr,
        topk_weight_ptr,
        routing_indices_ptr,
        active_expert_ids_ptr,
        gang_counter_ptr,
        num_active_tokens);
  }
}

} // namespace kernel
