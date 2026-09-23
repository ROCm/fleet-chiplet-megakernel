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

// MoE TopK Softmax routing kernel for MI300/MI350.
// Fused softmax + top-k selection + routing index generation.
// Adapted from topk_softmax_sm100.cuh for HIP/AMD.

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include "mpk_atoms.cuh" // st_wt_u16, st_wt_u32

namespace kernel {

#ifdef MPK_TOPK_DPP_REDUCE
// Reduce a 16-lane row to lane zero with DPP instead of ds_bpermute.
//
// __shfl_xor lowers to ds_bpermute_b32, which increments LGKM, so every
// dependent consumer has to drain lgkmcnt before it can use the result. DPP
// moves travel between lanes through the VALU and touch neither LGKM nor
// VMCNT, which takes the reduction off that counter entirely.
//
// `s_nop 1` before every DPP move covers the two-wait-state VALU-to-DPP source
// hazard. `bound_ctrl:1` returns 0 for out-of-range lanes, which is why ONLY
// lane 0 holds a valid result: its sources (lanes 8, 4, 2, 1) are all in-range,
// while high lanes read past the row. Every caller must broadcast from lane 0.
__device__ __forceinline__ float topk_dpp_row_max_to_lane_zero(float value) {
  float peer;
  asm volatile("s_nop 1\n"
               "v_mov_b32_dpp %1, %0 row_shl:8 row_mask:0xf bank_mask:0xf "
               "bound_ctrl:1\n"
               "v_max_f32 %0, %0, %1\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %1, %0 row_shl:4 row_mask:0xf bank_mask:0xf "
               "bound_ctrl:1\n"
               "v_max_f32 %0, %0, %1\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %1, %0 row_shl:2 row_mask:0xf bank_mask:0xf "
               "bound_ctrl:1\n"
               "v_max_f32 %0, %0, %1\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %1, %0 row_shl:1 row_mask:0xf bank_mask:0xf "
               "bound_ctrl:1\n"
               "v_max_f32 %0, %0, %1"
               : "+v"(value), "=&v"(peer));
  return value;
}

__device__ __forceinline__ float topk_dpp_row_sum_to_lane_zero(float value) {
  float peer;
  asm volatile("s_nop 1\n"
               "v_mov_b32_dpp %1, %0 row_shl:8 row_mask:0xf bank_mask:0xf "
               "bound_ctrl:1\n"
               "v_add_f32 %0, %0, %1\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %1, %0 row_shl:4 row_mask:0xf bank_mask:0xf "
               "bound_ctrl:1\n"
               "v_add_f32 %0, %0, %1\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %1, %0 row_shl:2 row_mask:0xf bank_mask:0xf "
               "bound_ctrl:1\n"
               "v_add_f32 %0, %0, %1\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %1, %0 row_shl:1 row_mask:0xf bank_mask:0xf "
               "bound_ctrl:1\n"
               "v_add_f32 %0, %0, %1"
               : "+v"(value), "=&v"(peer));
  return value;
}
#endif


static constexpr int MI300_WARP_SIZE = 64; // AMD wavefront size

// Fused TopK softmax kernel for AMD MI300/MI350.
// Uses wave-level reductions with __shfl_xor for max/sum across experts.
//
// Block size: 256 threads (4 wavefronts of 64).
// Each wavefront processes multiple rows in parallel.
//
// Template parameters:
//   T           - data type (bf16)
//   VPT         - values per thread (power of 2)
//   NUM_EXPERTS - total number of experts (power of 2)
//   WARPS_PER_CTA - number of wavefronts per CTA (typically 4)
//   BYTES_PER_LDG - bytes per vectorized load (8 or 16)
template <typename T,
          int VPT,
          int NUM_EXPERTS,
          int WARPS_PER_CTA,
          int BYTES_PER_LDG>
__device__ __forceinline__ void topk_softmax_mi300_task_impl(
    void *__restrict__ input_ptr,  // [num_rows, NUM_EXPERTS]
    void *__restrict__ output_ptr, // [num_rows, k] (float weights)
    int const num_rows,
    // Allocated row stride of routing_indices, which is BATCH_SIZE -- not
    // num_rows. The two differ whenever fewer tokens are active than the batch
    // was sized for (the fused router passes num_active_tokens as num_rows).
    // The MoE tile decode indexes this buffer as `d_routing + expert*BATCH_SIZE`
    // and reads lanes [0, BATCH_SIZE), so writing or clearing at a narrower
    // stride puts every expert's row at the wrong offset and leaves the tail
    // holding a previous layer's routing values.
    int const routing_row_stride,
    int const k,
    void *__restrict__ routing_indices_ptr,   // [NUM_EXPERTS, stride] int32
    void *__restrict__ active_expert_ids_ptr, // [NUM_EXPERTS + 1] int32
    int const start_expert,
    int const end_expert,
    bool const renormalize,
    int *early_routing_ready = nullptr,
    unsigned int early_routing_epoch = 0) {
  T *input = static_cast<T *>(input_ptr);
  float *output = static_cast<float *>(output_ptr);
  int *routing_indices = static_cast<int *>(routing_indices_ptr);
  int *active_expert_ids = static_cast<int *>(active_expert_ids_ptr);

  // Union of experts activated by *any* row, marked during TopK and compacted
  // at the bottom. One byte per local expert; the write is idempotent (every
  // writer stores 1), so no atomics are needed.
  __shared__ unsigned char s_hit[NUM_EXPERTS];

  // Initialize routing indices to 0.
  // active_expert_ids initialization is NOT needed: the compaction pass at the
  // bottom writes [0..count-1] densely and then the count at [NUM_EXPERTS], and
  // no consumer reads past the count.
  for (int expert = start_expert + threadIdx.x; expert < end_expert;
       expert += blockDim.x) {
    if (routing_indices != nullptr) {
      // Clear the whole allocated row, not just the active prefix: the MoE
      // decode ballots over all BATCH_SIZE lanes, so a stale nonzero in
      // [num_rows, stride) would compact a token that does not exist into an
      // MFMA column.
      //
      // Write-through for the same reason as the logits reset below: the
      // winners are published with `st_wt_u32` straight to HBM, so a plain
      // store leaves a dirty zero line in this XCD's L2 whose writeback can
      // land after a later layer has written the same address. Here the
      // consequence is worse than a perturbed weight -- a dropped routing
      // index silently removes a token from an expert's MFMA column set.
#ifdef MPK_TOPK_NO_ROUTING_CLEAR
            // unreachable. The winner store and `s_hit[local_expert] = 1` sit in one
      // block below, and the compaction keys on s_hit, so an expert reaches
      // active_expert_ids IFF its row was written this layer -- MoE never reads
      // a cleared entry.
      //
      // Only sound at stride 1. Above that a hit expert may be selected by one
      // row and not another, and the MoE decode ballots over all BATCH_SIZE
      // lanes, so a stale nonzero row would compact a token that does not
      // exist into an MFMA column. Batch > 1 keeps the clear.
      if (routing_row_stride != 1) {
        for (int row = 0; row < routing_row_stride; ++row) {
          st_wt_u32((void *)&routing_indices[expert * routing_row_stride + row],
                    0u);
        }
      }
#else
      for (int row = 0; row < routing_row_stride; ++row) {
        st_wt_u32((void *)&routing_indices[expert * routing_row_stride + row],
                  0u);
      }
#endif
    }
  }
  for (int e = threadIdx.x; e < NUM_EXPERTS; e += blockDim.x) {
    s_hit[e] = 0;
  }
  __syncthreads();

  // Compile-time constants
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(MI300_WARP_SIZE % THREADS_PER_ROW == 0,
                "THREADS_PER_ROW must divide warp size");

  static constexpr int ELTS_PER_WARP = MI300_WARP_SIZE * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;

  int const warp_idx = threadIdx.x / MI300_WARP_SIZE;
  int const lane_idx = threadIdx.x % MI300_WARP_SIZE;
  int const warp_base_row = warp_idx * ROWS_PER_WARP;
  int const thread_row_in_warp = lane_idx / THREADS_PER_ROW;
  int const thread_row = warp_base_row + thread_row_in_warp;
  int const thread_group_idx = lane_idx % THREADS_PER_ROW;

  if (thread_row < num_rows) {
    // Load row data
    T *thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    int const first_elt = thread_group_idx * ELTS_PER_LDG;
    T *thread_read_ptr = thread_row_ptr + first_elt;

    float row_chunk[VPT];
    // Vectorized load
    for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
      int base = ldg * ELTS_PER_LDG;
      int src_offset = ldg * THREADS_PER_ROW * ELTS_PER_LDG;
      for (int e = 0; e < ELTS_PER_LDG; ++e) {
        row_chunk[base + e] =
            static_cast<float>(thread_read_ptr[src_offset + e]);
      }
    }

    // Reset input buffer to 0 (for split-k gate linear compatibility).
    //
    // WRITE-THROUGH, to match the producer. The 128 router workers publish
    // each logit with `st_wt_u16` (sc0 sc1), which bypasses L2 and lands in
    // HBM, and they sit on all 8 XCDs while this reader is on one. A plain
    // store here leaves a *dirty* copy of the line in this XCD's L2. That line
    // is not merely stale-on-read -- the acquire in topk_noinline drops it
    // before the next read -- it is a pending writeback of zeros. Whenever it
    // is evicted after the next layer's router has already written that
    // address in HBM, the eviction overwrites a live logit with 0.
    //
    // A zeroed logit does not crash and does not change the ranking: 0 is far
    // below the winners, so TopK returns the same experts in the same order
    // and `routing_indices` / `active_expert_ids` come out bit-identical. But
    // the softmax denominator is a sum over *all* NUM_EXPERTS logits, so the
    // renormalized weights all shift in the low bits.
    //
    // Do NOT read the paragraph above as an explanation of any observed
    // moe_topk_weight nondeterminism. It was, and it was the wrong suspect: the
    // real cause of run-to-run moe_topk_weight divergence at batch_size >= 8
    // was a dim-0 `input_map` on the output tensor, which made the host offset
    // the base pointer by whole rows. See the NOTE at the `topk_weight`
    // new_input() sites in persistent_kernel.py. The write-through reset here
    // is still correct and still required for the L2 reason given above; it
    // just shifts low bits, not whole rows. Define MPK_NO_LOGIT_RESET to keep
    // the logits readable when dumping tensors for a golden reference.
#ifndef MPK_NO_LOGIT_RESET
    for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
      int src_offset = ldg * THREADS_PER_ROW * ELTS_PER_LDG;
      for (int e = 0; e < ELTS_PER_LDG; ++e) {
        T zero = static_cast<T>(0);
        st_wt_u16(&thread_read_ptr[src_offset + e],
                  *reinterpret_cast<unsigned short *>(&zero));
      }
    }
#endif

    // Max reduction within subgroup
    float thread_max = row_chunk[0];
    for (int ii = 1; ii < VPT; ++ii) {
      thread_max = fmaxf(thread_max, row_chunk[ii]);
    }
#ifdef MPK_TOPK_DPP_REDUCE
    // A DPP row is exactly 16 lanes, so the ladder row_shl 8/4/2/1 is the whole
    // reduction only at a 16-lane row. This ships ON, so any other width falls
    // back to the butterfly rather than failing the build.
    if constexpr (THREADS_PER_ROW == 16) {
      thread_max = topk_dpp_row_max_to_lane_zero(thread_max);
      // Valid in lane 0 only (bound_ctrl:1) -- broadcast before any lane reads it.
      thread_max = __shfl(thread_max, 0, THREADS_PER_ROW);
    } else {
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other = __shfl_xor(thread_max, mask, THREADS_PER_ROW);
        thread_max = fmaxf(thread_max, other);
      }
    }
#else
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other = __shfl_xor(thread_max, mask, THREADS_PER_ROW);
      thread_max = fmaxf(thread_max, other);
    }
#endif

    // Softmax
    float row_sum = 0.f;
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = __builtin_amdgcn_exp2f((row_chunk[ii] - thread_max) *
                                             1.4426950408889634f);
      row_sum += row_chunk[ii];
    }
#ifdef MPK_TOPK_DPP_REDUCE
    if constexpr (THREADS_PER_ROW == 16) {
      row_sum = topk_dpp_row_sum_to_lane_zero(row_sum);
      row_sum = __shfl(row_sum, 0, THREADS_PER_ROW);
    } else {
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        row_sum += __shfl_xor(row_sum, mask, THREADS_PER_ROW);
      }
    }
#else
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      row_sum += __shfl_xor(row_sum, mask, THREADS_PER_ROW);
    }
#endif
    float const inv_sum = 1.f / row_sum;
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] *= inv_sum;
    }

    // Fused Top-K selection — Step 1 uses inline asm for branchless local
    // argmax.
    int const start_col = first_elt;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;
    float row_sum_for_renorm = 0.f;
    float topk_vals[8];

    // Precompute expert column indices for branchless local argmax.
    int col[VPT];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      col[i] = start_col + i;
    }

    for (int k_idx = 0; k_idx < k; ++k_idx) {
      // ── Step 1: Branchless local argmax over VPT=8 elements ──
      float max_val;
      int expert;
#ifdef MPK_TOPK_LOCAL_MAX3
      // The serial scan below is 23 VALU ops on a ~21-deep dependency chain:
      // each of the seven steps must know the running max before it can
      // compare, and the running index rides the same chain. Both are
      // avoidable.
      //
      // v_max3_f32 takes three sources, so the value reduces as a tree --
      // max3(r0,r1,r2) and max3(r3,r4,r5) issue back to back, then one more
      // max3 folds in r6 and a v_max_f32 folds in r7. Three levels, not seven.
      //
      // Recovering the index then costs seven independent v_cmp_eq against
      // that one max, and a v_cndmask chain to pick the winner. Only the
      // cndmask chain is serial, and it no longer gates the compares.
      //
      // Tie-break: the serial version uses strict `>`, so on equal values the
      // LOWEST column wins. Visiting the equality matches in REVERSE order
      // (c7 first, c0 last) reproduces that exactly -- the last cndmask to
      // fire is the lowest matching column. This matters: BF16 logits collide
      // often enough that a different tie-break reroutes tokens.
      static_assert(VPT == 8,
                    "MPK_TOPK_LOCAL_MAX3 hard-codes an 8-wide reduction tree");
      float _lmax0, _lmax1;
      asm volatile("v_max3_f32 %[m0], %[r0], %[r1], %[r2]\n"
                   "v_max3_f32 %[m1], %[r3], %[r4], %[r5]\n"
                   "v_max3_f32 %[mv], %[m0], %[m1], %[r6]\n"
                   "v_max_f32 %[mv], %[mv], %[r7]\n"
                   "v_mov_b32 %[ex], %[c7]\n"
                   "v_cmp_eq_f32 vcc, %[r6], %[mv]\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c6], vcc\n"
                   "v_cmp_eq_f32 vcc, %[r5], %[mv]\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c5], vcc\n"
                   "v_cmp_eq_f32 vcc, %[r4], %[mv]\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c4], vcc\n"
                   "v_cmp_eq_f32 vcc, %[r3], %[mv]\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c3], vcc\n"
                   "v_cmp_eq_f32 vcc, %[r2], %[mv]\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c2], vcc\n"
                   "v_cmp_eq_f32 vcc, %[r1], %[mv]\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c1], vcc\n"
                   "v_cmp_eq_f32 vcc, %[r0], %[mv]\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c0], vcc\n"
                   : [mv] "=&v"(max_val),
                     [ex] "=&v"(expert),
                     [m0] "=&v"(_lmax0),
                     [m1] "=&v"(_lmax1)
                   : [r0] "v"(row_chunk[0]),
                     [r1] "v"(row_chunk[1]),
                     [r2] "v"(row_chunk[2]),
                     [r3] "v"(row_chunk[3]),
                     [r4] "v"(row_chunk[4]),
                     [r5] "v"(row_chunk[5]),
                     [r6] "v"(row_chunk[6]),
                     [r7] "v"(row_chunk[7]),
                     [c0] "v"(col[0]),
                     [c1] "v"(col[1]),
                     [c2] "v"(col[2]),
                     [c3] "v"(col[3]),
                     [c4] "v"(col[4]),
                     [c5] "v"(col[5]),
                     [c6] "v"(col[6]),
                     [c7] "v"(col[7])
                   : "vcc");
#else
      asm volatile("v_mov_b32 %[mv], %[r0]\n"
                   "v_mov_b32 %[ex], %[c0]\n"
                   "v_cmp_gt_f32 vcc, %[r1], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r1], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c1], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r2], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r2], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c2], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r3], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r3], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c3], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r4], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r4], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c4], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r5], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r5], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c5], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r6], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r6], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c6], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r7], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r7], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c7], vcc\n"
                   : [mv] "=&v"(max_val), [ex] "=&v"(expert)
                   : [r0] "v"(row_chunk[0]),
                     [r1] "v"(row_chunk[1]),
                     [r2] "v"(row_chunk[2]),
                     [r3] "v"(row_chunk[3]),
                     [r4] "v"(row_chunk[4]),
                     [r5] "v"(row_chunk[5]),
                     [r6] "v"(row_chunk[6]),
                     [r7] "v"(row_chunk[7]),
                     [c0] "v"(col[0]),
                     [c1] "v"(col[1]),
                     [c2] "v"(col[2]),
                     [c3] "v"(col[3]),
                     [c4] "v"(col[4]),
                     [c5] "v"(col[5]),
                     [c6] "v"(col[6]),
                     [c7] "v"(col[7])
                   : "vcc");
#endif

      // ── Step 2: Branchless argmax reduce across subgroup ──
      // Uses __shfl_xor for cross-lane communication, inline asm for
      // branchless compare+select (eliminates s_and_saveexec divergence).
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other_max = __shfl_xor(max_val, mask, THREADS_PER_ROW);
        int other_expert = __shfl_xor(expert, mask, THREADS_PER_ROW);
        asm volatile("v_cmp_gt_f32 vcc, %[om], %[mv]\n"
                     "v_cndmask_b32 %[mv], %[mv], %[om], vcc\n"
                     "v_cndmask_b32 %[ex], %[ex], %[oe], vcc\n"
                     : [mv] "+v"(max_val), [ex] "+v"(expert)
                     : [om] "v"(other_max), [oe] "v"(other_expert)
                     : "vcc");
      }

      // ── Step 3: Write top-k result ──
      if (thread_group_idx == 0) {
        bool const node_uses = (expert >= start_expert && expert < end_expert);
        int const out_idx = k * thread_row + k_idx;
        st_wt_u32((void *)&output[out_idx], __float_as_uint(max_val));
        topk_vals[k_idx] = max_val;
        row_sum_for_renorm += max_val;

        if (node_uses && routing_indices != nullptr) {
          int const local_expert = expert - start_expert;
          st_wt_u32((void *)&routing_indices[local_expert * routing_row_stride +
                                             thread_row],
                    (unsigned)(k_idx + 1));
          // Mark, do not write: at num_rows > 1 several rows reach this line
          // with different `expert` for the same k_idx, so indexing the output
          // by k_idx made rows overwrite each other. Marking a per-expert byte
          // is idempotent and deduplicates for free -- which the MoE tile
          // decode requires, since it maps tile -> active_expert_ids[i] and
          // would otherwise run the same expert twice and drop another.
          s_hit[local_expert] = 1;
        }
#ifdef MPK_MOE_XCD_PAIR
        // One selected expert per XCD pair. Publish (epoch, expert) to
        // XCDs 2*s and 2*s+1 as soon as selection s lands, and write
        // d_mask[s] in selection order so W2 can use the same slot. Drain
        // the routing_indices store so the u64 cannot overtake it.
        if (thread_row == 0 && early_routing_ready != nullptr) {
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          unsigned long long const record =
              (unsigned long long)early_routing_epoch |
              ((unsigned long long)(unsigned)expert << 32);
          st_wt_u64((void *)&early_routing_ready[(1 + k_idx * 2) * 16 + 2],
                    record);
          st_wt_u64((void *)&early_routing_ready[(1 + k_idx * 2 + 1) * 16 + 2],
                    record);
          if (active_expert_ids != nullptr) {
            st_wt_u32((void *)&active_expert_ids[k_idx], (unsigned)expert);
          }
        }
#elif defined(MPK_EARLY_ROUTING)
        // Early-routing probe: publish (epoch, first expert) as soon as TopK
        // selection 0 lands, before the remaining 3 picks, softmax, and
        // compact. MoE workers poll this u64 and start expert-0 W13 while
        // this block finishes TopK. routing_indices[expert] was just
        // st_wt'd; drain it so the u64 cannot overtake that payload.
        if (k_idx == 0 && thread_row == 0 && early_routing_ready != nullptr) {
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          unsigned long long const record =
              (unsigned long long)early_routing_epoch |
              ((unsigned long long)(unsigned)expert << 32);
          for (int x = 0; x < 8; x++) {
            st_wt_u64((void *)&early_routing_ready[(1 + x) * 16 + 2], record);
          }
        }
#endif
      }

      // ── Step 4: Branchless blanking of winner ──
      // expert == col[i] matches exactly one thread + one element.
      if (k_idx + 1 < k) {
        float const neg_inf = -10000.f;
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
          asm volatile("v_cmp_eq_u32 vcc, %[ex], %[ci]\n"
                       "v_cndmask_b32 %[rc], %[rc], %[ni], vcc\n"
                       : [rc] "+v"(row_chunk[i])
                       : [ex] "v"(expert), [ci] "v"(col[i]), [ni] "v"(neg_inf)
                       : "vcc");
        }
      }
    }

    // Optional renormalization (write-through stores, using cached values)
    if (renormalize && thread_group_idx == 0) {
      float inv = 1.f / row_sum_for_renorm;
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        int const out_idx = k * thread_row + k_idx;
        st_wt_u32((void *)&output[out_idx],
                  __float_as_uint(topk_vals[k_idx] * inv));
      }
    }
  }
  __syncthreads();

#ifdef MPK_MOE_XCD_PAIR
  // Selection-order ids are already at [0, k). Compact would overwrite them
  // into ascending expert-id order, which is the wrong slot for the XCD-pair
  // map. Still sentinel-fill and publish the count.
  if (active_expert_ids != nullptr && threadIdx.x < MI300_WARP_SIZE) {
    int const lane = threadIdx.x;
    int const count = k;
    for (int e = count + lane; e < NUM_EXPERTS; e += MI300_WARP_SIZE) {
      st_wt_u32((void *)&active_expert_ids[e], (unsigned)-1);
    }
    if (lane == 0) {
      st_wt_u32((void *)&active_expert_ids[NUM_EXPERTS], (unsigned)count);
    }
  }
#else
  // ── Compact the hit set into a dense, ascending, deduplicated list ──
  // Wavefront 0 alone, so the running `count` needs no atomics: it is a plain
  // register carried across the chunk loop, uniform across the wave because
  // every lane derives it from the same ballot masks.
  //
  // At num_rows == 1 this reproduces the old behaviour up to ordering: exactly
  // k distinct experts win, so count == k, and the ids come out sorted instead
  // of in topk order. No consumer depends on that order -- they either scan
  // [0, count) or index by tile.
  if (active_expert_ids != nullptr && threadIdx.x < MI300_WARP_SIZE) {
    int const lane = threadIdx.x;
    unsigned long long const lane_mask = (1ULL << lane) - 1ULL;
    int count = 0;
    for (int base = 0; base < NUM_EXPERTS; base += MI300_WARP_SIZE) {
      int const e = base + lane;
      bool const hit = (e < NUM_EXPERTS) && (s_hit[e] != 0);
      unsigned long long const ballot = __ballot(hit);
      if (hit) {
        int const pos = count + __popcll(ballot & lane_mask);
        st_wt_u32((void *)&active_expert_ids[pos], (unsigned)(start_expert + e));
      }
      count += __popcll(ballot);
    }
    // Sentinel-fill the unused slots. The MoE tile decode must decide "is this
    // tile padding?" from a value that is *per-slot*, never from the count:
    // the count is one word shared by all 36 layers, so two workers that read
    // it at different moments can disagree about how many slots are live, and
    // then disagree about which tiles are W13. That disagreement is a deadlock,
    // not drift -- the skipped tiles never reach the per-expert arrival, the
    // `% W13_TILES` release never fires, and the W2 workers spin forever.
    //
    // A slot read is not immune to the same cross-layer skew, but it degrades
    // the way the mask already did before packing: a worker gets some layer's
    // expert id and produces slightly wrong numbers. Every worker that looks at
    // slot s reads the same address, so they cannot disagree about whether s is
    // padding while agreeing on everything else.
    for (int e = count + lane; e < NUM_EXPERTS; e += MI300_WARP_SIZE) {
      st_wt_u32((void *)&active_expert_ids[e], (unsigned)-1);
    }
    if (lane == 0) {
      st_wt_u32((void *)&active_expert_ids[NUM_EXPERTS], (unsigned)count);
    }
  }
#endif
}


#ifdef MPK_LOCAL_TOPK
// Every workgroup rebuilds this layer's routing from the 128 epoch-tagged
// router logits. The row arithmetic is topk_softmax_mi300_task_impl's for one
// row, line for line -- 16 lanes x 8 contiguous columns, the same max / exp2 /
// sum reductions, the same asm argmax and tie-break, the same
// renormalization -- so every workgroup reproduces the completer's picks and
// weights bit for bit.
template <int NUM_EXPERTS, int K>
__device__ __forceinline__ void mpk_local_topk(int const *tags, int expected,
                                               int const *norm_flags = nullptr,
                                               int n_norm = 0) {
  static_assert(NUM_EXPERTS == 128, "MPK_LOCAL_TOPK: s_ltk_* sized for 128");
  static_assert(K <= 8, "MPK_LOCAL_TOPK: topk_vals holds 8");
  constexpr int VPT = 8;
  constexpr int THREADS_PER_ROW = NUM_EXPERTS / VPT;
  static_assert(THREADS_PER_ROW == 16, "one 16-lane row");
  int const tid = threadIdx.x;
  unsigned const want = mpk_ltk_tag(expected);
  if (tid < 64) {
    // One wave reads all 128 words, two per lane, at the same system scope as
    // the routing_ready poll this replaces. Each word is a single
    // write-through store, so a matching tag carries its own logit.
    unsigned long long v;
#ifdef MPK_ILSUB
    unsigned long long ilsub_miss = 0, ilsub_iters = 0;
#endif
    while (true) {
#ifdef MPK_LTK_POLL_SC1
      asm volatile("global_load_dwordx2 %0, %1, off sc1\n\ts_waitcnt vmcnt(0)"
                   : "=v"(v)
                   : "v"(tags + 2 * tid)
                   : "memory");
#else
      v = ld_sys_u64((void *)(tags + 2 * tid));
#endif
      bool const ok = (((unsigned)(v >> 16)) & 0xFFFFu) == want &&
                      (((unsigned)(v >> 48)) & 0xFFFFu) == want;
#ifdef MPK_ILSUB
      unsigned long long const ilsub_m = __ballot(!ok);
      ilsub_iters++;
      if (ilsub_m) {
        ilsub_miss = ilsub_m;
      }
#endif
      if (__all(ok)) {
        break;
      }
      __builtin_amdgcn_s_sleep(1);
    }
    MPK_ILSTAMP(18, s_ilsub_ml == MPK_ILSUB_L0 + 1);
#ifdef MPK_ILSUB
    // 30: lanes (tag pairs 2l, 2l+1) still missing on the last failed poll;
    // 31: poll iterations.
    if (tid == 0 && s_ilsub_ml == MPK_ILSUB_L0 + 1) {
      g_ilsub[blockIdx.x * 32 + 30] = ilsub_miss;
      g_ilsub[blockIdx.x * 32 + 31] = ilsub_iters;
    }
#endif
    s_ltk_logit[2 * tid] = (unsigned short)(v & 0xFFFFu);
    s_ltk_logit[2 * tid + 1] = (unsigned short)((v >> 32) & 0xFFFFu);
  }
  __syncthreads();
#ifdef MPK_LTK_EARLY_TAG
  if (n_norm > 0 && tid >= 64 && tid < 128) {
    // Wave 1 waits for this XCD's normed row while wave 0 runs the TopK.
    int const w = tid - 64 < n_norm ? tid - 64 : n_norm - 1;
    unsigned f;
    while (true) {
      asm volatile("global_load_dword %0, %1, off sc1\n\ts_waitcnt vmcnt(0)"
                   : "=v"(f)
                   : "v"(norm_flags + w)
                   : "memory");
      if (__all(f == want)) {
        break;
      }
      __builtin_amdgcn_s_sleep(1);
    }
#ifdef MPK_ILSUB
    if (tid == 64 && s_ilsub_ml == MPK_ILSUB_L0 + 1) {
      g_ilsub[blockIdx.x * 32 + 29] = __builtin_amdgcn_s_memrealtime();
    }
#endif
  }
#endif
  MPK_ILSTAMP(24, s_ilsub_ml == MPK_ILSUB_L0 + 1);
  if (tid < THREADS_PER_ROW) {
    float row_chunk[VPT];
    for (int e = 0; e < VPT; ++e) {
      // bf16 -> f32 is exact: the bf16 bits are the float's top half.
      row_chunk[e] =
          __uint_as_float(((unsigned)s_ltk_logit[tid * VPT + e]) << 16);
    }
    float thread_max = row_chunk[0];
    for (int ii = 1; ii < VPT; ++ii) {
      thread_max = fmaxf(thread_max, row_chunk[ii]);
    }
#ifdef MPK_TOPK_DPP_REDUCE
    thread_max = topk_dpp_row_max_to_lane_zero(thread_max);
#ifdef MPK_LTK_DPP
    thread_max = __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(thread_max)));
#else
    thread_max = __shfl(thread_max, 0, THREADS_PER_ROW);
#endif
#else
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other = __shfl_xor(thread_max, mask, THREADS_PER_ROW);
      thread_max = fmaxf(thread_max, other);
    }
#endif
    float row_sum = 0.f;
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = __builtin_amdgcn_exp2f((row_chunk[ii] - thread_max) *
                                             1.4426950408889634f);
      row_sum += row_chunk[ii];
    }
#ifdef MPK_TOPK_DPP_REDUCE
    row_sum = topk_dpp_row_sum_to_lane_zero(row_sum);
#ifdef MPK_LTK_DPP
    row_sum = __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(row_sum)));
#else
    row_sum = __shfl(row_sum, 0, THREADS_PER_ROW);
#endif
#else
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      row_sum += __shfl_xor(row_sum, mask, THREADS_PER_ROW);
    }
#endif
    float const inv_sum = 1.f / row_sum;
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] *= inv_sum;
    }
    MPK_ILSTAMP(25, s_ilsub_ml == MPK_ILSUB_L0 + 1);
    int const start_col = tid * VPT;
    float row_sum_for_renorm = 0.f;
    float topk_vals[8];
    int col[VPT];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      col[i] = start_col + i;
    }
    for (int k_idx = 0; k_idx < K; ++k_idx) {
      float max_val;
      int expert;
#ifdef MPK_TOPK_LOCAL_MAX3
      float _lmax0, _lmax1;
      asm volatile("v_max3_f32 %[m0], %[r0], %[r1], %[r2]\n"
                   "v_max3_f32 %[m1], %[r3], %[r4], %[r5]\n"
                   "v_max3_f32 %[mv], %[m0], %[m1], %[r6]\n"
                   "v_max_f32 %[mv], %[mv], %[r7]\n"
                   "v_mov_b32 %[ex], %[c7]\n"
                   "v_cmp_eq_f32 vcc, %[r6], %[mv]\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c6], vcc\n"
                   "v_cmp_eq_f32 vcc, %[r5], %[mv]\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c5], vcc\n"
                   "v_cmp_eq_f32 vcc, %[r4], %[mv]\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c4], vcc\n"
                   "v_cmp_eq_f32 vcc, %[r3], %[mv]\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c3], vcc\n"
                   "v_cmp_eq_f32 vcc, %[r2], %[mv]\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c2], vcc\n"
                   "v_cmp_eq_f32 vcc, %[r1], %[mv]\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c1], vcc\n"
                   "v_cmp_eq_f32 vcc, %[r0], %[mv]\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c0], vcc\n"
                   : [mv] "=&v"(max_val),
                     [ex] "=&v"(expert),
                     [m0] "=&v"(_lmax0),
                     [m1] "=&v"(_lmax1)
                   : [r0] "v"(row_chunk[0]), [r1] "v"(row_chunk[1]),
                     [r2] "v"(row_chunk[2]), [r3] "v"(row_chunk[3]),
                     [r4] "v"(row_chunk[4]), [r5] "v"(row_chunk[5]),
                     [r6] "v"(row_chunk[6]), [r7] "v"(row_chunk[7]),
                     [c0] "v"(col[0]), [c1] "v"(col[1]), [c2] "v"(col[2]),
                     [c3] "v"(col[3]), [c4] "v"(col[4]), [c5] "v"(col[5]),
                     [c6] "v"(col[6]), [c7] "v"(col[7])
                   : "vcc");
#else
      asm volatile("v_mov_b32 %[mv], %[r0]\n"
                   "v_mov_b32 %[ex], %[c0]\n"
                   "v_cmp_gt_f32 vcc, %[r1], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r1], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c1], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r2], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r2], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c2], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r3], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r3], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c3], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r4], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r4], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c4], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r5], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r5], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c5], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r6], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r6], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c6], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r7], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r7], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c7], vcc\n"
                   : [mv] "=&v"(max_val), [ex] "=&v"(expert)
                   : [r0] "v"(row_chunk[0]), [r1] "v"(row_chunk[1]),
                     [r2] "v"(row_chunk[2]), [r3] "v"(row_chunk[3]),
                     [r4] "v"(row_chunk[4]), [r5] "v"(row_chunk[5]),
                     [r6] "v"(row_chunk[6]), [r7] "v"(row_chunk[7]),
                     [c0] "v"(col[0]), [c1] "v"(col[1]), [c2] "v"(col[2]),
                     [c3] "v"(col[3]), [c4] "v"(col[4]), [c5] "v"(col[5]),
                     [c6] "v"(col[6]), [c7] "v"(col[7])
                   : "vcc");
#endif
#ifdef MPK_LTK_DPP
      {
        // xor 8 = row_ror:8; xor 4 = row_shl:4 for lanes with bit 2 clear, row_shr:4
        // for the rest; xor 2 / xor 1 = quad_perm [2,3,0,1] / [1,0,3,2].
        bool const ltk_hi4 = (tid & 4) != 0;
#define MPK_LTK_STEP(OM, OE)                                              \
        do {                                                                    \
          float const other_max = __int_as_float(OM);                           \
          int const other_expert = (OE);                                        \
          asm volatile("v_cmp_gt_f32 vcc, %[om], %[mv]\n"                        \
                       "v_cndmask_b32 %[mv], %[mv], %[om], vcc\n"                \
                       "v_cndmask_b32 %[ex], %[ex], %[oe], vcc\n"                \
                       : [mv] "+v"(max_val), [ex] "+v"(expert)                  \
                       : [om] "v"(other_max), [oe] "v"(other_expert)            \
                       : "vcc");                                                 \
        } while (0)
        MPK_LTK_STEP(__builtin_amdgcn_update_dpp(0, __float_as_int(max_val), 0x128, 0xF, 0xF, false),
                     __builtin_amdgcn_update_dpp(0, expert, 0x128, 0xF, 0xF, false));
        {
          int const mv_lo = __builtin_amdgcn_update_dpp(0, __float_as_int(max_val), 0x104, 0xF, 0xF, false);
          int const mv_hi = __builtin_amdgcn_update_dpp(0, __float_as_int(max_val), 0x114, 0xF, 0xF, false);
          int const ex_lo = __builtin_amdgcn_update_dpp(0, expert, 0x104, 0xF, 0xF, false);
          int const ex_hi = __builtin_amdgcn_update_dpp(0, expert, 0x114, 0xF, 0xF, false);
          MPK_LTK_STEP(ltk_hi4 ? mv_hi : mv_lo, ltk_hi4 ? ex_hi : ex_lo);
        }
        MPK_LTK_STEP(__builtin_amdgcn_update_dpp(0, __float_as_int(max_val), 0x4E, 0xF, 0xF, false),
                     __builtin_amdgcn_update_dpp(0, expert, 0x4E, 0xF, 0xF, false));
        MPK_LTK_STEP(__builtin_amdgcn_update_dpp(0, __float_as_int(max_val), 0xB1, 0xF, 0xF, false),
                     __builtin_amdgcn_update_dpp(0, expert, 0xB1, 0xF, 0xF, false));
#undef MPK_LTK_STEP
      }
#else
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other_max = __shfl_xor(max_val, mask, THREADS_PER_ROW);
        int other_expert = __shfl_xor(expert, mask, THREADS_PER_ROW);
        asm volatile("v_cmp_gt_f32 vcc, %[om], %[mv]\n"
                     "v_cndmask_b32 %[mv], %[mv], %[om], vcc\n"
                     "v_cndmask_b32 %[ex], %[ex], %[oe], vcc\n"
                     : [mv] "+v"(max_val), [ex] "+v"(expert)
                     : [om] "v"(other_max), [oe] "v"(other_expert)
                     : "vcc");
      }
#endif
      if (tid == 0) {
        topk_vals[k_idx] = max_val;
        row_sum_for_renorm += max_val;
        s_ltk_sel[k_idx] = expert;
      }
      if (k_idx + 1 < K) {
        float const neg_inf = -10000.f;
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
          asm volatile("v_cmp_eq_u32 vcc, %[ex], %[ci]\n"
                       "v_cndmask_b32 %[rc], %[rc], %[ni], vcc\n"
                       : [rc] "+v"(row_chunk[i])
                       : [ex] "v"(expert), [ci] "v"(col[i]), [ni] "v"(neg_inf)
                       : "vcc");
        }
      }
    }
    if (tid == 0) {
      float inv = 1.f / row_sum_for_renorm;
      for (int k_idx = 0; k_idx < K; ++k_idx) {
        s_ltk_w[k_idx] = topk_vals[k_idx] * inv;
      }
    }
  }
  MPK_ILSTAMP(19, s_ilsub_ml == MPK_ILSUB_L0 + 1);
  __syncthreads();
}
#endif

} // namespace kernel
