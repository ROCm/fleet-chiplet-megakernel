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

// Fused QKV + Attention gang task for MI300/MI350.
//
// Combines the ResAddF32+RMSNorm+QKV+KVUpdate gang task (211) with
// the paged attention task (141) into a single gang task, eliminating
// the scheduler transition between them (~12µs × 36 layers = 432µs).
//
// CROC-style last-worker-does-attention pattern:
//   Every worker executes its QKV tile, then atomicAdd to per-XCD counter.
//   The LAST worker to arrive (atomicAdd returns prev == total-1) runs
//   attention inline — zero polling overhead, no separate attention tile.
//
// Key: each XCD produces Q/K/V for exactly one kv_head group (kv_head =
// xcd_id), so attention on XCD N only reads data produced by XCD N's own QKV
// tiles.
//
// Pointer layout (9 inputs, 5 outputs):
//   input_ptrs[0] = workspace_f32     input_ptrs[5] = bias
//   input_ptrs[1] = residual          input_ptrs[6] = sinks
//   input_ptrs[2] = norm_weight       input_ptrs[7] = barrier
//   input_ptrs[3] = norm_scratch      input_ptrs[8] = lse_acc
//   input_ptrs[4] = weight
//
//   output_ptrs[0] = x_output         output_ptrs[3] = q_workspace
//   output_ptrs[1] = k_cache          output_ptrs[4] = o_acc
//   output_ptrs[2] = v_cache

#pragma once

// Dependencies: gang_rmsnorm_linear_mxfp4_bias_mi300.cuh (QKV kernel),
//               paged_attention_ck_fmha_split_kv_mi300.cuh (attention kernel),
//               mpk_atoms.cuh (atomic helpers)
// All included via task_header.cuh before this file.

namespace kernel {

template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM,
          int HEAD_DIM,
          int NUM_Q_PER_KV,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE,
          int NUM_KV_HEADS,
          int SLIDING_WINDOW,
          int HAS_SINKS>
__device__ __noinline__ void gang_qkv_attn_fused_kernel_mi300(
    void *workspace_f32_ptr,     // input_ptrs[0]
    void const *residual_ptr,    // input_ptrs[1]
    void const *norm_weight_ptr, // input_ptrs[2]
    void *norm_scratch_ptr,      // input_ptrs[3]
    void const *weight_ptr,      // input_ptrs[4]
    void const *bias_ptr,        // input_ptrs[5]
    void const *sinks_ptr,       // input_ptrs[6] (nullable)
    void *barrier_ptr,           // input_ptrs[7]
    void *lse_acc_ptr,           // input_ptrs[8]
    void *x_output_ptr,          // output_ptrs[0]
    void *k_cache_ptr,           // output_ptrs[1]
    void *v_cache_ptr,           // output_ptrs[2]
    void *q_workspace_ptr,       // output_ptrs[3]
    void *o_acc_ptr,             // output_ptrs[4]
    void const *cos_ptr,
    void const *sin_ptr,
    int const *qo_indptr,
    int const *kv_indptr,
    int const *kv_indices,
    int const *kv_last_page_len,
    int num_active_tokens,
    int n_wgs_per_xcd,
    int kv_stride,
    int q_ws_stride,
    float attn_scale,
    int total_qkv_tiles_per_xcd,
    int tile_idx) {
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));

  int *d_barrier = static_cast<int *>(barrier_ptr);

  // ══════════════════════════════════════════════════════════════════
  // Phase 1: QKV GEMM — every worker executes its assigned tile
  // ══════════════════════════════════════════════════════════════════
  gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_kernel<BATCH_SIZE,
                                                        OUTPUT_PER_WG,
                                                        REDUCTION_SIZE,
                                                        ACTUAL_HIDDEN_DIM,
                                                        HEAD_DIM,
                                                        NUM_Q_PER_KV,
                                                        PAGE_SIZE>(
      workspace_f32_ptr,
      residual_ptr,
      norm_weight_ptr,
      norm_scratch_ptr,
      weight_ptr,
      bias_ptr,
      x_output_ptr,
      k_cache_ptr,
      v_cache_ptr,
      q_workspace_ptr,
      cos_ptr,
      sin_ptr,
      qo_indptr,
      kv_indptr,
      kv_indices,
      kv_last_page_len,
      num_active_tokens,
      n_wgs_per_xcd,
      kv_stride,
      q_ws_stride,
      tile_idx);

  // ══════════════════════════════════════════════════════════════════
  // Per-XCD arrival: atomicAdd with release semantics.
  // The LAST worker to arrive (prev == total-1) runs attention.
  // Release semantics ensure all QKV stores are visible to that worker.
  // ══════════════════════════════════════════════════════════════════
  __shared__ int s_prev;
  if (threadIdx.x == 0) {
    s_prev = atom_add_release_gpu_s32(&d_barrier[xcd_id], 1);
  }
  __syncthreads();

  if (s_prev < total_qkv_tiles_per_xcd - 1) {
    // Not the last worker — done
    return;
  }

  // ══════════════════════════════════════════════════════════════════
  // Phase 2: Attention — only the LAST worker on this XCD reaches here
  // All QKV tiles on this XCD are complete (we were the last to arrive).
  // No polling needed — zero barrier overhead.
  // ══════════════════════════════════════════════════════════════════

  // Pre-offset K/V cache pointers by kv_head_idx (= xcd_id).
  using bf16_t = __hip_bfloat16;
  void const *offset_k = reinterpret_cast<bf16_t const *>(k_cache_ptr) +
                         static_cast<size_t>(xcd_id) * HEAD_DIM;
  void const *offset_v = reinterpret_cast<bf16_t const *>(v_cache_ptr) +
                         static_cast<size_t>(xcd_id) * HEAD_DIM;

  // Run attention: kv_head_idx = xcd_id, request_id = 0, chunk_idx = 0
  paged_attention_ck_fmha_split_kv_impl<bfloat16,
                                        NUM_Q_PER_KV,
                                        HEAD_DIM,
                                        PAGE_SIZE,
                                        MAX_SEQ_LEN,
                                        NUM_KV_CHUNKS,
                                        Q_WORKSPACE_STRIDE,
                                        KV_CACHE_STRIDE,
                                        NUM_KV_HEADS>(
      q_workspace_ptr,
      const_cast<void *>(offset_k),
      const_cast<void *>(offset_v),
      o_acc_ptr,
      lse_acc_ptr,
      qo_indptr,
      kv_indptr,
      kv_indices,
      kv_last_page_len,
      /*request_id=*/0,
      /*kv_head_idx=*/xcd_id,
      /*kv_chunk_idx=*/0,
      attn_scale,
      SLIDING_WINDOW,
      HAS_SINKS ? sinks_ptr : nullptr);

  // ── Reset barrier for next iteration ──
  __syncthreads();
  if (threadIdx.x == 0) {
    st_wt_u32((void *)&d_barrier[xcd_id], 0u);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  }
}

} // namespace kernel
