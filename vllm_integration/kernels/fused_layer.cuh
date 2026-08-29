/* Fleet MK: Fused transformer layer kernel
 *
 * This file re-exports mirage's sub-kernel functions with Fleet MK-compatible
 * names and documents the interface. The actual implementations live in
 * mirage's include/ tree and are accessed via include path.
 *
 * Dependency chain:
 *   fused_layer.cuh
 *     -> common.cuh (Fleet MK's own atomic/barrier primitives)
 *     -> barriers.cuh (Fleet MK's barrier functions)
 *     -> mirage/persistent_kernel/tasks/mi300/gang_full_layer_fused_mi300.cuh
 *          -> gang_rmsnorm_linear_mxfp4_bias_mi300.cuh (QKV)
 *          -> paged_attention_ck_fmha_split_kv_mi300.cuh (Attention)
 *          -> gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh (O-proj+TopK)
 *          -> gang_moe_fused_mxfp4_mi300.cuh (MoE)
 *
 * Build: hipcc needs -I<mirage>/include/mirage/persistent_kernel
 *        to resolve the transitive includes.
 */
#pragma once

#include "common.cuh"
#include "barriers.cuh"

// Include mirage's sub-kernel implementations via include path.
// The build script sets -I${MIRAGE_DIR}/include/mirage/persistent_kernel
// so these resolve correctly.
//
// Include order matters: gang_moe_linear_mi300 defines _gang_moe_get_xcd_id()
// which is used by gang_moe_linear_mxfp4 (pulled in transitively).
// merge_splitkv must come before gang_full_layer_fused.
#include "tasks/mi300/gang_rmsnorm_linear_mxfp4_bias_mi300.cuh"
#include "tasks/mi300/paged_attention_decode_minimal_hd64_mi300.cuh"
#include "tasks/mi300/paged_attention_ck_fmha_split_kv_mi300.cuh"
#include "tasks/mi300/gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh"
#include "tasks/mi300/gang_moe_fused_mxfp4_mi300.cuh"
#include "tasks/ampere/merge_splitkv.cuh"
#include "tasks/mi300/gang_full_layer_fused_mi300.cuh"

namespace fleet_mk {

// ============================================================================
// Sub-kernel interface documentation
// ============================================================================
//
// Phase 1 - QKV GEMM:
//   kernel::gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_kernel<
//       BATCH_SIZE, OUTPUT_PER_WG, REDUCTION_SIZE,
//       ACTUAL_HIDDEN_DIM, HEAD_DIM, NUM_Q_PER_KV, PAGE_SIZE>(
//     workspace_f32, residual, norm_weight, norm_scratch,
//     weight, bias, x_output, k_cache, v_cache, q_workspace,
//     cos_ptr, sin_ptr,
//     qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
//     num_active_tokens, n_wgs_per_xcd, kv_stride, q_ws_stride,
//     tile_idx);
//
// Phase 3 - Split-KV Attention:
//   paged_attention_ck_fmha_split_kv_impl<
//       bfloat16, NUM_Q_PER_KV, HEAD_DIM, PAGE_SIZE,
//       MAX_SEQ_LEN, NUM_KV_CHUNKS, Q_WORKSPACE_STRIDE,
//       KV_CACHE_STRIDE, NUM_KV_HEADS>(
//     q_workspace, k_cache, v_cache, o_acc_f32, lse_acc,
//     qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
//     request_id, kv_head_idx, kv_chunk_idx,
//     attn_scale, sliding_window, sinks_ptr);
//
// Phase 5 - Merge:
//   merge_splitkv_ck_fmha<
//       bf16_t, NUM_Q_PER_KV, NUM_KV_HEADS, HEAD_DIM, NUM_KV_CHUNKS>(
//     lse_acc, o_acc_f32,
//     qo_indptr, kv_indptr, kv_last_page_len,
//     request_id, attn_out, kv_head_idx, sinks_ptr);
//
// Phase 7 - O-proj + RMSNorm + Router + TopK:
//   kernel::gang_linear_mxfp4_res_bias_rmsnorm_topk_kernel<
//       BATCH_SIZE, OPROJ_OUTPUT_PER_WG, OPROJ_REDUCTION_SIZE,
//       ACTUAL_HIDDEN_DIM, NUM_EXPERTS, TOPK_K>(
//     attn_out, oproj_weight, residual, oproj_bias,
//     norm_weight, norm_scratch, router_weight, router_bias,
//     logits_scratch, oproj_counters,
//     attn_proj_out, topk_weight, routing_indices, active_expert_ids,
//     num_active_tokens, oproj_n_wgs_per_xcd, oproj_output_stride,
//     router_tile_n, total_oproj_tiles, total_topk_tiles,
//     oproj_tiles_per_xcd, tile_idx, routing_ready);
//
// Phase 8 - MoE (W13+SwiGLU+W2):
//   kernel::gang_moe_fused_mxfp4_kernel_mi300<
//       BATCH_SIZE, MOE_INTERMEDIATE_SIZE, MOE_HIDDEN_SIZE,
//       NUM_EXPERTS, TOPK_K,
//       MOE_W13_OUTPUT_PER_WG, MOE_W2_OUTPUT_PER_WG>(
//     norm_scratch, gate_up_weight, down_weight,
//     routing_indices, active_expert_ids,
//     w13_bias, w2_bias, routing_weight, swiglu_out,
//     moe_workspace_f32, moe_barrier, tile_idx);

// ============================================================================
// Counter buffer slot indices (matching mirage's full-layer layout)
// ============================================================================
// These are byte offsets into the counter buffer (oproj_counters_base).
// Each slot is 16 bytes apart (cache-line aligned for atomics).

using fleet_mk::ATTN_GLOBAL_COUNTER_SLOT;
using fleet_mk::QKV_EPOCH_SLOT;
using fleet_mk::CHUNK_BARRIER_SLOT;
using fleet_mk::ROUTING_READY_SLOT;

} // namespace fleet_mk
