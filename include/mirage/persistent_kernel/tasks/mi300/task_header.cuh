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
#include "tasks/mi300/multitoken_paged_attention_mi300.cuh"
#include "tasks/mi300/multitoken_paged_attention_split_kv_mi300.cuh"
#include "tasks/mi300/kv_cache_update_mi300.cuh"
#ifdef MPK_USE_CK_FMHA
#include "tasks/mi300/paged_attention_decode_minimal_mi300.cuh"
#include "tasks/mi300/paged_attention_ck_fmha_split_kv_mi300.cuh"
#include "tasks/mi300/gang_attention_mi300.cuh"
#endif
// MoE task implementations for MI300/MI350
#include "tasks/mi300/moe_linear_mi300.cuh"
#include "tasks/mi300/gang_moe_linear_mi300.cuh"
#include "tasks/mi300/moe_topk_softmax_mi300.cuh"
#include "tasks/mi300/moe_mul_sum_add_mi300.cuh"
// Merge kernel is portable scalar code, reuse from ampere/
#include "tasks/ampere/merge_splitkv.cuh"
// Gang merge wrapper for split-KV CK FMHA (depends on merge_splitkv.cuh)
#include "tasks/mi300/gang_attention_merge_mi300.cuh"
