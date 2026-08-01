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

// Fused O-PROJ+TopK+MoE gang task for MI300/MI350.
//
// Eliminates one inter-task event barrier per layer by combining
// tasks 213 (O-PROJ+RMSNorm+Router+TopK) and 187 (W13+SwiGLU+W2)
// into a single gang task.
//
// Pipeline (all within one gang task dispatch):
//   Phase 1: Workers 0..(oproj_tiles_per_xcd-1) run O-PROJ tiles via
//            gang_linear_mxfp4_res_bias_rmsnorm_topk_kernel.
//            Workers >= oproj_tiles_per_xcd skip Phase 1 entirely.
//            The last TopK worker signals routing_ready.
//   Phase 2: ALL workers (including MoE-only workers) poll routing_ready,
//            then run MoE tiles via gang_moe_fused_mxfp4_kernel_mi300
//            with stride = workers_per_xcd (same as standalone MoE).
//
// Dispatch: gang_task_tiles_per_xcd = workers_per_xcd (30), NOT
//   oproj_tiles_per_xcd (24). This matches standalone MoE worker count.
//   tile_idx = xcd_id * workers_per_xcd + rank.

#pragma once
#include "tasks/mi300/gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh"
#include "tasks/mi300/gang_moe_fused_mxfp4_mi300.cuh"

namespace kernel {

template <int BATCH_SIZE,
          int OPROJ_OUTPUT_PER_WG,
          int OPROJ_REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM,
          int NUM_EXPERTS,
          int TOPK_K,
          int MOE_INTERMEDIATE_SIZE,
          int MOE_HIDDEN_SIZE,
          int MOE_W13_OUTPUT_PER_WG,
          int MOE_W2_OUTPUT_PER_WG>
__device__ __attribute__((always_inline)) void
    gang_oproj_topk_moe_fused_kernel_mi300(
        // ── O-PROJ+TopK inputs (same as task 213) ──
        void const *oproj_input_ptr,    // input_ptrs[0]  attn_out
        void const *oproj_weight_ptr,   // input_ptrs[1]  O-proj MXFP4 weight
        void const *oproj_residual_ptr, // input_ptrs[2]  residual
        void const *oproj_bias_ptr,     // input_ptrs[3]  O-proj bias
        void const *norm_weight_ptr, // input_ptrs[4]  post-attn RMSNorm weight
        void *norm_output_ptr,       // input_ptrs[5]  RMSNorm output scratch
        void const *router_weight_ptr, // input_ptrs[6]  router linear weight
        void const *router_bias_ptr,   // input_ptrs[7]  router bias
        void *logits_scratch_ptr,      // input_ptrs[8]  router logits scratch
        void *oproj_counters_ptr,      // input_ptrs[9]  hierarchical barrier
        // ── MoE inputs (same as task 187) ──
        void const *moe_gate_up_weight_ptr, // input_ptrs[10] W13 MXFP4 weight
        void const *moe_down_weight_ptr,    // input_ptrs[11] W2 MXFP4 weight
        void const *moe_w13_bias_ptr,       // input_ptrs[12] W13 bias
        void const *moe_w2_bias_ptr,        // input_ptrs[13] W2 bias
        void *moe_barrier_ptr,              // input_ptrs[14] per-expert barrier
        void *moe_swiglu_out_ptr, // input_ptrs[15] SwiGLU intermediate
        // ── Outputs ──
        void *oproj_output_ptr, // output_ptrs[0] O-proj output (residual-added)
        void *topk_weight_ptr,  // output_ptrs[1] routing weights
        void *routing_indices_ptr,    // output_ptrs[2] expert assignment
        void *active_expert_ids_ptr,  // output_ptrs[3] activated expert mask
        void *moe_routing_weight_ptr, // output_ptrs[4] routing weight (for MoE
                                      // W2)
        void *moe_workspace_f32_ptr, // output_ptrs[5] MoE atomicAdd accumulator
        // ── O-PROJ parameters ──
        int num_active_tokens,
        int oproj_n_wgs_per_xcd,
        int oproj_output_stride,
        int router_tile_n,
        int total_oproj_tiles,
        int total_topk_tiles,
        int oproj_tiles_per_xcd,
        // ── MoE parameters ──
        int moe_total_tiles_per_xcd,
        int workers_per_xcd,
        // ── Fused dispatch: tile_idx encodes XCD rank ──
        int tile_idx) {

  // tile_idx = xcd_id * workers_per_xcd + rank
  int xcd_rank = tile_idx % workers_per_xcd;

  // CROC hierarchical release pattern for routing synchronization.
  // routing_ready[0] = epoch counter (incremented by TopK worker).
  // routing_ready[(1+xcd_id)*16] = per-XCD release flag (written by TopK via
  // st_wt). Each MoE worker polls only its XCD-local release flag — no
  // cross-XCD contention.
  int *routing_ready = static_cast<int *>(oproj_counters_ptr) + 10 * 16;
  int tid = threadIdx.x;
  int xcd_id = tile_idx / workers_per_xcd;

#ifdef MPK_FUSED_PHASE_TIMING
  unsigned long long _ft0 = 0, _ft1 = 0, _ft2 = 0, _ft3 = 0;
  if (tid == 0) {
    _ft0 = __builtin_amdgcn_s_memrealtime();
  }
#endif

  // Read current epoch BEFORE Phase 1 to know what to poll for.
  __shared__ int s_routing_expected;
  if (tid == 0) {
    s_routing_expected = ld_nt_s32(routing_ready) + 1;
  }
  __syncthreads();
  int expected = s_routing_expected;

  // Publish the layer counter for the MoE W13->W2 barrier, which reads it from
  // this fixed LDS offset and releases with layer_idx + 1. Its d_barrier is
  // monotonic and never reset, so this must advance once per layer for the
  // whole run or the barrier stops gating after the first layer.
  //
  // The routing epoch is that counter: bumped exactly once per layer by the
  // TopK completer, never reset, and identical for every worker here because
  // it is read before Phase 1 signals it.
  {
    constexpr int LAYER_IDX_SMEM_OFF =
        mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
        mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END;
    extern __shared__ char _oproj_moe_smem[];
    if (tid == 0) {
      *reinterpret_cast<int *>(&_oproj_moe_smem[LAYER_IDX_SMEM_OFF]) = expected;
    }
  }
  __syncthreads();

  // ── Phase 1: O-PROJ + RMSNorm + Router + TopK ──
  // Need max(oproj_tiles_per_xcd, router_tile_n) workers: O-proj may need
  // fewer tiles than TopK experts when OUTPUT_PER_WG is large.
  int oproj_topk_tiles_per_xcd =
      oproj_tiles_per_xcd > router_tile_n ? oproj_tiles_per_xcd : router_tile_n;
  if (xcd_rank < oproj_topk_tiles_per_xcd) {
    int oproj_tile_idx = xcd_id * oproj_topk_tiles_per_xcd + xcd_rank;
    gang_linear_mxfp4_res_bias_rmsnorm_topk_kernel<BATCH_SIZE,
                                                   OPROJ_OUTPUT_PER_WG,
                                                   OPROJ_REDUCTION_SIZE,
                                                   ACTUAL_HIDDEN_DIM,
                                                   NUM_EXPERTS,
                                                   TOPK_K>(
        oproj_input_ptr,
        oproj_weight_ptr,
        oproj_residual_ptr,
        oproj_bias_ptr,
        norm_weight_ptr,
        norm_output_ptr,
        router_weight_ptr,
        router_bias_ptr,
        logits_scratch_ptr,
        oproj_counters_ptr,
        oproj_output_ptr,
        topk_weight_ptr,
        routing_indices_ptr,
        active_expert_ids_ptr,
        num_active_tokens,
        oproj_n_wgs_per_xcd,
        oproj_output_stride,
        router_tile_n,
        total_oproj_tiles,
        total_topk_tiles,
        oproj_topk_tiles_per_xcd,
        oproj_tile_idx,
        routing_ready);
  }

#ifdef MPK_FUSED_PHASE_TIMING
  if (tid == 0) {
    _ft1 = __builtin_amdgcn_s_memrealtime();
  }
#endif

  // ── Wait for TopK to finish ──
  // CROC hierarchical release: each worker polls its XCD-local release flag.
  // TopK worker wrote per-XCD flags via st_wt after threadfence_gpu,
  // so polling is XCD-local (hot in L2, no cross-XCD contention).
  if (tid == 0) {
    int *my_release = &routing_ready[(1 + xcd_id) * 16];
    while (ld_nt_s32(my_release) < expected) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  // Invalidate L2 so MoE reads fresh TopK routing data from HBM.
  // Same as baseline O-proj barrier (buffer_inv without sc0 sc1).
  asm volatile("buffer_inv" ::: "memory");

#ifdef MPK_FUSED_PHASE_TIMING
  if (tid == 0) {
    _ft2 = __builtin_amdgcn_s_memrealtime();
  }
#endif

  // ── Phase 2: MoE (W13+SwiGLU+W2) ──
  // Stride = workers_per_xcd (30), matching standalone MoE dispatch.
  for (int moe_t = xcd_rank; moe_t < moe_total_tiles_per_xcd;
       moe_t += workers_per_xcd) {
    gang_moe_fused_mxfp4_kernel_mi300<BATCH_SIZE,
                                      MOE_INTERMEDIATE_SIZE,
                                      MOE_HIDDEN_SIZE,
                                      NUM_EXPERTS,
                                      TOPK_K,
                                      MOE_W13_OUTPUT_PER_WG,
                                      MOE_W2_OUTPUT_PER_WG>(
        norm_output_ptr,
        moe_gate_up_weight_ptr,
        moe_down_weight_ptr,
        routing_indices_ptr,
        active_expert_ids_ptr,
        moe_w13_bias_ptr,
        moe_w2_bias_ptr,
        moe_routing_weight_ptr,
        moe_swiglu_out_ptr,
        moe_workspace_f32_ptr,
        moe_barrier_ptr,
        moe_t);
  }

#ifdef MPK_FUSED_PHASE_TIMING
  if (tid == 0) {
    _ft3 = __builtin_amdgcn_s_memrealtime();
    // Accumulate timing (10ns resolution) into global counters.
    // g_fused_phase_ns[0]=oproj, [1]=poll+inv, [2]=moe, [3]=count
    atomicAdd(&g_fused_phase_ns[0], (_ft1 - _ft0) * 10);
    atomicAdd(&g_fused_phase_ns[1], (_ft2 - _ft1) * 10);
    atomicAdd(&g_fused_phase_ns[2], (_ft3 - _ft2) * 10);
    atomicAdd(&g_fused_phase_ns[3], 1ULL);
  }
#endif
}

} // namespace kernel
