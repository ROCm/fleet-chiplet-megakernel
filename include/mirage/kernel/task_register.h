/* Copyright 2023-2025 CMU
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

#pragma once

#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/threadblock/graph.h"

namespace mirage {
namespace runtime {

class TaskRegister {
public:
  static TaskRegister *singleton;
  TaskRegister();

public:
  static TaskRegister *get_instance();
  int register_embedding_task(threadblock::Graph const &bgraph,
                              std::vector<int> const &params);
  int register_rmsnorm_task(threadblock::Graph const &bgraph,
                            std::vector<int> const &params);
  int register_rmsnorm_linear_task(threadblock::Graph const &bgraph,
                                   std::vector<int> const &params);
  int register_attention_task(threadblock::Graph const &bgraph,
                              std::vector<int> const &params);
  int register_paged_attention_task(threadblock::Graph const &bgraph,
                                    std::vector<int> const &params);
  int register_single_batch_extend_attention_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_linear_task(threadblock::Graph const &bgraph,
                           std::vector<int> const &params,
                           bool with_residual);
  int register_silu_mul_task(threadblock::Graph const &bgraph,
                             std::vector<int> const &params);
  int register_identity_task(threadblock::Graph const &bgraph,
                             std::vector<int> const &params);
  int register_silu_mul_linear_with_residual_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_argmax_partial_task(threadblock::Graph const &bgraph,
                                   std::vector<int> const &params);
  int register_argmax_reduce_task(threadblock::Graph const &bgraph,
                                  std::vector<int> const &params);
  int register_reduce_task(threadblock::Graph const &bgraph,
                           std::vector<int> const &params);
  int register_find_ngram_partial_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params);
  int register_find_ngram_global_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_target_verify_greedy_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);
  // Hopper tasks
  int register_linear_hopper_task(threadblock::Graph const &bgraph,
                                  std::vector<int> const &params,
                                  bool with_residual);
  int register_paged_attention_hopper_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  int register_rmsnorm_hopper_task(threadblock::Graph const &bgraph,
                                   std::vector<int> const &params);
  int register_linear_swapAB_hopper_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params,
                                         bool with_residual);
  int register_linear_cutlass_hopper_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params,
                                          bool with_residual);
  int register_silu_mul_hopper_task(threadblock::Graph const &bgraph,
                                    std::vector<int> const &params);
  int register_embedding_hopper_task(threadblock::Graph const &bgraph,
                                     std::vector<int> const &params);
  int register_moe_linear_sm90_task(threadblock::Graph const &bgraph,
                                    std::vector<int> const &params,
                                    bool w13_linear);
  int register_splitk_linear_swapAB_hopper_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params,
      bool with_residual);
  int register_paged_attention_split_kv_hopper_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  // MI300 split-K linear
  int register_splitk_linear_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  int register_splitk_reduce_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  int register_splitk_linear_res_atomic_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  // MI300 cross-XCD K-split GEMM + finalize
  int register_gang_ksplit_gemm_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  int register_gang_ksplit_finalize_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  // MI300 gang split-K linear with residual
  int register_gang_splitk_linear_res_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  // MI300 gang RMSNorm (redundant per XCD for XCD-local event path)
  int register_gang_rmsnorm_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  // MI300 gang linear (horizontal fusion)
  int register_gang_linear_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  int register_gang_linear_res_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  // MI300 gang linear with N-fast tile iteration (sister to gang_linear; for
  // isolating L2 weight-reuse benefit of M-tiling)
  int register_gang_linear_n_tiling_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  int register_gang_linear_res_n_tiling_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  int register_gang_linear_msplit_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  int register_gang_linear_res_msplit_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  int register_gang_linear_silu_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  int register_linear_silu_mi300_task(threadblock::Graph const &bgraph, std::vector<int> const &params);
  // MI300 gang MoE linear
  int register_gang_moe_w13_linear_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  int register_gang_moe_w2_linear_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  // MI300 gang attention (horizontal fusion)
  int register_gang_attn_split_kv_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  int register_gang_attn_merge_mi300_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params);
  // MI300 split-KV tasks
  int register_paged_attention_split_kv_mi300_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_paged_attention_split_kv_merge_mi300_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  // MI300 CK FMHA batch-independent tasks
  int register_kv_cache_update_mi300_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_paged_attention_ck_fmha_split_kv_mi300_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_paged_attention_ck_fmha_merge_mi300_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  // MI300/MI350 MoE tasks
  int register_moe_topk_softmax_mi300_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  int register_moe_linear_mi300_task(threadblock::Graph const &bgraph,
                                     std::vector<int> const &params,
                                     bool w13_linear);
  int register_moe_mul_sum_add_mi300_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params);
  // SM100 tasks
  int register_splitk_linear_sm100_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params,
                                        bool with_residual);
  int register_linear_sm100_task(threadblock::Graph const &bgraph,
                                 std::vector<int> const &params,
                                 bool with_residual);
  int register_paged_attention_sm100_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params);
  int register_argmax_partial_sm100_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);
  int register_argmax_reduce_sm100_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params);
  int register_sampling_sm100_task(threadblock::Graph const &bgraph,
                                   std::vector<int> const &params);
  int register_tensor_init_task(threadblock::Graph const &bgraph,
                                std::vector<int> const &params);
  int register_moe_topk_softmax_sm100_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  int register_moe_linear_sm100_task(threadblock::Graph const &bgraph,
                                     std::vector<int> const &params,
                                     bool w13_linear);
  int register_moe_silu_mul_task(threadblock::Graph const &bgraph,
                                 std::vector<int> const &params);
  int register_moe_mul_sum_add_sm100_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params);
  int register_paged_attention_split_kv_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_paged_attention_split_kv_merge_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  // SM100 tasks end
  int register_task_variant(TaskType type, std::string const &code);

public:
  std::map<TaskType, std::vector<std::string>> all_task_variants;
};

} // namespace runtime
} // namespace mirage
