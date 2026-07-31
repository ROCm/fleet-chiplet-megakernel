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

// MI300 task implementations - clean AMD-only code, no NVIDIA ifdefs
//
// NOTE: the include order below is load-bearing. Several task headers rely on
// types and helpers defined by earlier ones rather than including them
// directly (e.g. attention_sink_mi300.cuh needs bf16/__hip_bfloat16,
// kv_cache_update_mi300.cuh needs vec_load_8/vec_store_8). Do not alphabetize.

// clang-format off
// Generic tasks reused from ampere/ (platform-independent)
#include "tasks/ampere/embedding.cuh"
#include "tasks/ampere/identity.cuh"
#include "tasks/ampere/reduction.cuh"

// MI300-specific task implementations
#include "tasks/mi300/rmsnorm_mi300.cuh"
#include "tasks/mi300/argmax_mi300.cuh"
#include "tasks/mi300/rotary_embedding_mi300.cuh"
#include "tasks/mi300/silu_mul_mi300.cuh"
#include "tasks/mi300/silu_mul_linear_mi300.cuh"
#include "tasks/mi300/linear_mi300.cuh"
#include "tasks/mi300/gang_rmsnorm_linear_bias_mi300.cuh"
#include "tasks/mi300/multitoken_paged_attention_mi300.cuh"
#include "tasks/mi300/multitoken_paged_attention_split_kv_mi300.cuh"
#include "tasks/mi300/kv_cache_update_mi300.cuh"
#include "tasks/mi300/attention_sink_mi300.cuh"
#ifdef MPK_USE_CK_FMHA
#include "tasks/mi300/paged_attention_decode_minimal_hd64_mi300.cuh"
#include "tasks/mi300/paged_attention_ck_fmha_split_kv_mi300.cuh"
#include "tasks/mi300/gang_attention_mi300.cuh"
#endif
// MoE task implementations for MI300/MI350
#include "tasks/mi300/moe_linear_mi300.cuh"
#include "tasks/mi300/moe_linear_mxfp4_mi300.cuh"
#include "tasks/mi300/moe_linear_mxfp4_ck_mi300.cuh"
#include "tasks/mi300/gang_moe_linear_mi300.cuh"
#include "tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh"
#include "tasks/mi300/gang_moe_pipelined_mxfp4_mi300.cuh"
#include "tasks/mi300/gang_rmsnorm_linear_mxfp4_bias_mi300.cuh"
#include "tasks/mi300/gang_rmsnorm_linear_mxfp4_bias_argmax_mi300.cuh"
#ifdef MPK_USE_CK_FMHA
#include "tasks/mi300/gang_qkv_attn_fused_mi300.cuh"
#endif
#include "tasks/mi300/gang_linear_mxfp4_res_bias_mi300.cuh"
#include "tasks/mi300/gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh"
#include "tasks/mi300/gang_moe_fused_mxfp4_mi300.cuh"
#include "tasks/mi300/gang_oproj_topk_moe_fused_mi300.cuh"
// Merge kernel needed by gang_full_layer_fused before it's included
#include "tasks/ampere/merge_splitkv.cuh"
#include "tasks/mi300/gang_full_layer_fused_mi300.cuh"
#include "tasks/mi300/gang_full_layer_with_lmhead_fused_mi300.cuh"
#include "tasks/mi300/gang_moe_swiglu_w2_mxfp4_mi300.cuh"
#include "tasks/mi300/moe_topk_softmax_mi300.cuh"
#include "tasks/mi300/moe_mul_sum_add_mi300.cuh"
#include "tasks/mi300/moe_residual_add_f32_mi300.cuh"
#include "tasks/mi300/swigluoai_mi300.cuh"
#include "tasks/mi300/bias_add_mi300.cuh"
// clang-format on
// Merge kernel moved above gang_full_layer_fused_mi300.cuh (needs it at
// template definition time)
