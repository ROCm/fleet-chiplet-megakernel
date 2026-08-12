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
#include "mirage/kernel/task_register.h"
#include "mirage/kernel/operator.h"
#include "mirage/transpiler/utils.h"

#ifdef MIRAGE_BACKEND_USE_ROCM
// Forward declaration of mirage::utils::get_max_shared_mem(); the full
// header (rocm_helper.h) drags in device-only templates that don't compile
// in this host-only translation unit.
namespace mirage {
namespace utils {
size_t get_max_shared_mem();
}
} // namespace mirage
#endif

namespace mirage {
namespace runtime {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

TaskRegister *TaskRegister::singleton = nullptr;

TaskRegister::TaskRegister() {}

TaskRegister *TaskRegister::get_instance() {
  if (singleton == nullptr) {
    singleton = new TaskRegister();
  }
  return singleton;
}

int TaskRegister::register_task_variant(runtime::TaskType type,
                                        std::string const &code) {
  std::vector<std::string> &variants = all_task_variants[type];
  for (size_t i = 0; i < variants.size(); i++) {
    if (variants[i] == code) {
      return (int)(i);
    }
  }
  // Add a new variant
  variants.push_back(code);
  return (int)(variants.size() - 1);
}

int TaskRegister::register_embedding_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params) {
  assert(params.size() == 1);
  // params[0]: input source (0: tokens, 1: input_token)
  int batch_size = 0, output_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::embedding_kernel<bfloat16, $, $, $>(",
         batch_size,
         output_size,
         output_stride);
  if (params[0] == 0) {
    code.e("    runtime_config.tokens + runtime_config.step[0], ");
  } else if (params[0] == 1) {
    code.e("    task_desc->input_ptrs[0],");
  }
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_EMBEDDING, code.to_string());
}

int TaskRegister::register_rmsnorm_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params) {
  assert(params.size() <= 1);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = output_ops[0]->output_tensors[0].dim[0];
  int hidden_dim = output_ops[0]->output_tensors[0].dim[1];
  // actual_hidden_dim: if provided and > 0, use for RMS mean computation
  // (avoids bf16 rounding errors when padding hidden dim)
  int actual_hidden_dim =
      (params.size() > 0 && params[0] > 0) ? params[0] : hidden_dim;
  // Currently assume that each rmsnorm task processes one token
  assert(batch_size == 1);
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(output_ops[0]->dtensor.dim[0] == input_ops[0]->dtensor.dim[0]);
  assert(output_ops[0]->dtensor.dim[1] == input_ops[0]->dtensor.dim[1]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::rms_norm_impl<bfloat16, $, $, $>(",
         batch_size,
         hidden_dim,
         actual_hidden_dim);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    1e-5f);");
  return register_task_variant(TASK_RMS_NORM, code.to_string());
}

int TaskRegister::register_rmsnorm_linear_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 3;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::norm_linear_task_impl<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         reduction_size,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    1e-6f,");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_RMS_NORM_LINEAR, code.to_string());
}

int TaskRegister::register_attention_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  assert(params.size() == 4);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = output_size / num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::single_batch_decoding_kernel<bfloat16, $, $, $, $>(",
         num_q_heads / num_kv_heads,
         1,
         head_dim,
         kv_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.step[0] + 1,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f);");
  return register_task_variant(TASK_ATTENTION_1, code.to_string());
}

int TaskRegister::register_paged_attention_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: max_seq_len
  // params[5]: page_size
  assert(params.size() == 6);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = output_size / num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);
  int max_tokens = input_ops[0]->dtensor.dim[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::multitoken_paged_attention_task_impl<bfloat16, $, $, $, $, "
         "$, $, $, $, $>(",
         num_q_heads / num_kv_heads,
         1,
         kv_stride,
         qkv_stride,
         output_size,
         head_dim,
         max_seq_len,
         page_size,
         max_tokens);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f);");
  return register_task_variant(TASK_PAGED_ATTENTION_1, code.to_string());
}

int TaskRegister::register_single_batch_extend_attention_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: extend_num
  // params[5]: output_stride
  assert(params.size() == 6);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int extend_num = params[4];
  int head_dim = output_size / num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int output_stride = params[5];
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::single_batch_extend_kernel<bfloat16, $, $, $, $, $, $>(",
         num_q_heads / num_kv_heads,
         1,
         head_dim,
         kv_stride,
         output_stride,
         extend_num);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.step[0] + 1,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f);");
  return register_task_variant(TASK_SINGLE_BATCH_EXTEND_ATTENTION,
                               code.to_string());
}

int TaskRegister::register_silu_mul_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, input_stride, output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(input_ops[0]->output_tensors[0].dim[1] == output_size * 2);
  // get input stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  input_stride = input_ops[0]->dtensor.dim[1];
  assert(input_stride == static_cast<int>(kn_input_op->input_strides[0]));
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::silu_mul_task_impl<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         input_stride,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_SILU_MUL, code.to_string());
}

int TaskRegister::register_identity_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params) {
  assert(params.size() == 0);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Both input and output tensors should be row major
  assert(input_ops[0]->dtensor.layout == layout::DmemRowMajor);
  assert(output_ops[0]->dtensor.layout == layout::DmemRowMajor);
  // Both input and output tensors should be INPUT OP
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  // Shape should be guranteed by higher-level APIs

  int outer_dim_size = 1, inner_dim_size, outer_dim_stride, output_size;
  for (int i = 0; i < input_ops[0]->dtensor.num_dims - 1; i++) {
    outer_dim_size *= input_ops[0]->dtensor.dim[i];
  }
  inner_dim_size =
      input_ops[0]->dtensor.dim[input_ops[0]->dtensor.num_dims - 1];
  outer_dim_stride = inner_dim_size;
  output_size = output_ops[0]
                    ->output_tensors[0]
                    .dim[output_ops[0]->output_tensors[0].num_dims - 1];
  // assert(output_size >= bgraph.block_dim.x);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::identity_task_impl<bfloat16, $, $, $, $>(",
         outer_dim_size,
         inner_dim_size,
         outer_dim_stride,
         output_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_IDENTITY, code.to_string());
}

int TaskRegister::register_silu_mul_linear_with_residual_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 3;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1] / 2;
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::silu_mul_linear_task_impl<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         reduction_size,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.my_gpu_id == 0);");
  return register_task_variant(TASK_SILU_MUL_LINEAR_WITH_RESIDUAL,
                               code.to_string());
}

int TaskRegister::register_linear_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params,
                                       bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  // When batch_size > 16 and output_size ≤ 64, use small tile (16x64x256)
  // to avoid 50% N-dimension waste from NPerBlock=128 > output_size=64.
  bool force_small_tile = (batch_size > 16 && output_size <= 64);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  if (force_small_tile) {
    code.e("kernel::linear_kernel<bfloat16, $, $, true>(",
           batch_size,
           reduction_size);
  } else {
    code.e(
        "kernel::linear_kernel<bfloat16, $, $>(", batch_size, reduction_size);
  }
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  if (with_residual) {
    code.e("    task_desc->input_ptrs[2],");
  } else {
    code.e("    nullptr,");
  }
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  if (with_residual) {
    code.e("    runtime_config.my_gpu_id == 0,");
  } else {
    code.e("    false/*residual*/,");
  }
  code.e("    $, $);", output_size, output_stride);
  if (with_residual) {
    return register_task_variant(TASK_LINEAR_WITH_RESIDUAL, code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR, code.to_string());
  }
}

int TaskRegister::register_splitk_linear_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2; // input + weight (no separate residual)
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::splitk_linear_kernel<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         reduction_size,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_SPLITK_LINEAR_MI300, code.to_string());
}

// Gang linear with HipKittens Algorithm 1 windowed traversal.
// params: [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
//          n_tiles_per_xcd, wgm]
// Cross-XCD K-split GEMM: each XCD handles K/8, ALL N-tiles
int TaskRegister::register_gang_ksplit_gemm_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int ws_stride = params[0];
  int tile_n = params[1];
  int n_tiles = params[2];
  int k_splits = params[3];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < 2) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  int reduction_size = input_ops[0]->dtensor.dim[1];
  int batch_size = input_ops[0]->dtensor.dim[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_ksplit_gemm_kernel<bfloat16, $, $, $>(",
         batch_size,
         reduction_size,
         k_splits);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", ws_stride);
  code.e("    $,", n_tiles);
  code.e("    (int)task_desc->task_metadata.n_tile_start,");
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_KSPLIT_GEMM_MI300, code.to_string());
}

// Cross-XCD K-split finalize: add residual + convert bf16
int TaskRegister::register_gang_ksplit_finalize_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 3);
  int o_stride = params[0];
  int n_cols_per_xcd = params[1];
  int finalize_tiles = params[2];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < 2) { // 2 inputs: workspace, residual
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops.size() == 1);
  int batch_size = output_ops[0]->output_tensors[0].dim[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_ksplit_finalize_kernel<bfloat16, $>(", batch_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    $,", o_stride);
  code.e("    $,", o_stride);
  code.e("    $,", n_cols_per_xcd);
  code.e("    (int)task_desc->task_metadata.n_tile_start,");
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_KSPLIT_FINALIZE_MI300,
                               code.to_string());
}

// Gang split-K linear with residual: splits K dimension within XCD for
// better worker utilization. Uses XCD-local atomics for merge.
int TaskRegister::register_gang_splitk_linear_res_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int output_stride = params[0];
  int tile_n = params[1];
  int n_tiles_per_xcd = params[2];
  int k_splits = params[3];

  int reduction_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 4; // input, weight, residual, workspace+done_counter
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  int m_per_tile = input_ops[0]->dtensor.dim[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // Done counters stored after the float32 workspace data.
  // Workspace is [batch, chunk_n] float32 partitioned per XCD.
  // Done counters: n_tiles_per_xcd int32 values after the float data.
  int ws_floats = m_per_tile * n_tiles_per_xcd * tile_n;
  code.e("kernel::gang_splitk_linear_res_kernel<bfloat16, $, $, $>(",
         m_per_tile,
         reduction_size,
         k_splits);
  code.e("    task_desc->input_ptrs[0],");  // input
  code.e("    task_desc->input_ptrs[1],");  // weight
  code.e("    task_desc->input_ptrs[2],");  // residual
  code.e("    task_desc->input_ptrs[3],");  // workspace (float32)
  code.e("    task_desc->output_ptrs[0],"); // output
  code.e("    (int*)((float*)task_desc->input_ptrs[3] + $),", ws_floats);
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_SPLITK_LINEAR_RES_MI300,
                               code.to_string());
}

// Gang RMSNorm: 8 tasks (1 per XCD), each computes the same RMSNorm
// redundantly. Enables XCD-local event counting to avoid cross-XCD barrier
// overhead.
int TaskRegister::register_gang_rmsnorm_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;  // input, weight
  int num_outputs = 1; // output

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  int hidden_dim = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // Gang RMSNorm: only rank 0 (tile_idx == 0) on each XCD computes.
  // Other workers skip (they see the result via XCD-local L2).
  code.e("if (tile_idx == 0) {");
  code.e("  kernel::rms_norm_impl<bfloat16, 1, $>(", hidden_dim);
  code.e("      task_desc->input_ptrs[0],");  // input
  code.e("      task_desc->input_ptrs[1],");  // weight
  code.e("      task_desc->output_ptrs[0],"); // output
  code.e("      1e-6f);");
  code.e("}");
  return register_task_variant(TASK_GANG_RMS_NORM_MI300, code.to_string());
}

int TaskRegister::register_gang_linear_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 7);
  int output_stride = params[0];
  int tile_n = params[1];
  int m_tiles = params[2];
  int m_per_tile = params[3];
  int total_tiles_per_xcd = params[4];
  int n_tiles_per_xcd = params[5];
  int wgm = params[6];

  int reduction_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // BATCH_SIZE = m_per_tile (small M-tile, not full batch)
  code.e("kernel::gang_linear_kernel<bfloat16, $, $>(",
         m_per_tile,
         reduction_size);
  code.e("    task_desc->input_ptrs[0],");  // full activation
  code.e("    task_desc->input_ptrs[1],");  // XCD's weight chunk
  code.e("    task_desc->output_ptrs[0],"); // XCD's output columns
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", m_tiles);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    $,", wgm);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_LINEAR_MI300, code.to_string());
}

// Gang linear with residual + HipKittens Algorithm 1 windowed traversal.
// params: [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
//          n_tiles_per_xcd, wgm]
int TaskRegister::register_gang_linear_res_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 7);
  int output_stride = params[0];
  int tile_n = params[1];
  int m_tiles = params[2];
  int m_per_tile = params[3];
  int total_tiles_per_xcd = params[4];
  int n_tiles_per_xcd = params[5];
  int wgm = params[6];

  int reduction_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  // 3 inputs: input, weight, residual; 1 output
  assert(bgraph.operators.size() == 4);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < 3) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_linear_residual_kernel<bfloat16, $, $>(",
         m_per_tile,
         reduction_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", m_tiles);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    $,", wgm);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_LINEAR_RES_MI300, code.to_string());
}

// Gang linear with fused bias_add into epilogue.
// params: [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
//          n_tiles_per_xcd, wgm]
// Inputs: [activation(replicate), weight(partition dim 0), bias(replicate)]
// Outputs: [output(partition dim 1)]
int TaskRegister::register_gang_linear_bias_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 7);
  int output_stride = params[0];
  int tile_n = params[1];
  int m_tiles = params[2];
  int m_per_tile = params[3];
  int total_tiles_per_xcd = params[4];
  int n_tiles_per_xcd = params[5];
  int wgm = params[6];

  int reduction_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 3; // input, weight, bias
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_linear_kernel<bfloat16, $, $>(",
         m_per_tile,
         reduction_size);
  code.e("    task_desc->input_ptrs[0],");  // activation
  code.e("    task_desc->input_ptrs[1],");  // weight
  code.e("    task_desc->output_ptrs[0],"); // output
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", m_tiles);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    $,", wgm);
  code.e("    tile_idx,");
  code.e("    task_desc->input_ptrs[2]);"); // bias
  return register_task_variant(TASK_GANG_LINEAR_BIAS_MI300, code.to_string());
}

// Gang split-K linear with residual + fused bias_add into epilogue.
// params: [output_stride, tile_n, n_tiles_per_xcd, k_splits]
// Inputs: [input, weight, residual, workspace, bias]
// Outputs: [output]
int TaskRegister::register_gang_splitk_linear_res_bias_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int output_stride = params[0];
  int tile_n = params[1];
  int n_tiles_per_xcd = params[2];
  int k_splits = params[3];

  int reduction_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5; // input, weight, residual, workspace, bias
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  int m_per_tile = input_ops[0]->dtensor.dim[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  int ws_floats = m_per_tile * n_tiles_per_xcd * tile_n;
  code.e("kernel::gang_splitk_linear_res_kernel<bfloat16, $, $, $>(",
         m_per_tile,
         reduction_size,
         k_splits);
  code.e("    task_desc->input_ptrs[0],");  // input
  code.e("    task_desc->input_ptrs[1],");  // weight
  code.e("    task_desc->input_ptrs[2],");  // residual
  code.e("    task_desc->input_ptrs[3],");  // workspace (float32)
  code.e("    task_desc->output_ptrs[0],"); // output
  code.e("    (int*)((float*)task_desc->input_ptrs[3] + $),", ws_floats);
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    tile_idx,");
  code.e("    task_desc->input_ptrs[4]);"); // bias
  return register_task_variant(TASK_GANG_SPLITK_LINEAR_RES_BIAS_MI300,
                               code.to_string());
}

// Fused RMSNorm + Gang Linear + Bias.
// params: [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
//          n_tiles_per_xcd, wgm, actual_hidden_dim]
// Inputs: [norm_input, norm_weight, norm_output_scratch, linear_weight, bias]
// Outputs: [linear_output]
int TaskRegister::register_gang_rmsnorm_linear_bias_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 8);
  int output_stride = params[0];
  int tile_n = params[1];
  int m_tiles = params[2];
  int m_per_tile = params[3];
  int total_tiles_per_xcd = params[4];
  int n_tiles_per_xcd = params[5];
  int wgm = params[6];
  int actual_hidden_dim = params[7];
  (void)total_tiles_per_xcd;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs =
      5; // norm_input, norm_weight, norm_output, linear_weight, bias
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[0] is norm_input [batch, reduction_size]
  assert(input_ops[0]->dtensor.num_dims == 2);
  int reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_rmsnorm_linear_bias_kernel<bfloat16, $, $, $>(",
         m_per_tile,
         reduction_size,
         actual_hidden_dim);
  code.e("    task_desc->input_ptrs[0],");  // norm_input
  code.e("    task_desc->input_ptrs[1],");  // norm_weight
  code.e("    task_desc->input_ptrs[2],");  // norm_output scratch (writable)
  code.e("    task_desc->input_ptrs[3],");  // linear_weight
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // linear_output
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", m_tiles);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    $,", wgm);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_RMSNORM_LINEAR_BIAS_MI300,
                               code.to_string());
}

// Fused RMSNorm + Gang Linear + Bias + TopK Softmax.
// params: [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
//          n_tiles_per_xcd, wgm, actual_hidden_dim, num_experts,
//          num_experts_per_tok, total_gang_tiles]
int TaskRegister::register_gang_rmsnorm_linear_bias_topk_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 11);
  int output_stride = params[0];
  int tile_n = params[1];
  int m_tiles = params[2];
  int m_per_tile = params[3];
  int total_tiles_per_xcd = params[4];
  int n_tiles_per_xcd = params[5];
  int wgm = params[6];
  int actual_hidden_dim = params[7];
  int num_experts = params[8];
  int num_experts_per_tok = params[9];
  int total_gang_tiles = params[10];
  (void)total_tiles_per_xcd;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;  // norm_input, norm_weight, norm_output, linear_weight,
                       // bias, logits_scratch, gang_counter
  int num_outputs = 3; // topk_weight, routing_indices, active_expert_ids

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[0] is norm_input [batch, reduction_size]
  assert(input_ops[0]->dtensor.num_dims == 2);
  int reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e(
      "kernel::gang_rmsnorm_linear_bias_topk_kernel<bfloat16, $, $, $, $, $>(",
      m_per_tile,
      reduction_size,
      actual_hidden_dim,
      num_experts,
      num_experts_per_tok);
  code.e("    task_desc->input_ptrs[0],");  // norm_input
  code.e("    task_desc->input_ptrs[1],");  // norm_weight
  code.e("    task_desc->input_ptrs[2],");  // norm_output scratch
  code.e("    task_desc->input_ptrs[3],");  // linear_weight
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->input_ptrs[5],");  // logits_scratch
  code.e("    task_desc->input_ptrs[6],");  // gang_counter
  code.e("    task_desc->output_ptrs[0],"); // topk_weight
  code.e("    task_desc->output_ptrs[1],"); // routing_indices
  code.e("    task_desc->output_ptrs[2],"); // active_expert_ids
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", m_tiles);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    $,", wgm);
  code.e("    tile_idx,");
  code.e("    $);", total_gang_tiles);
  return register_task_variant(TASK_GANG_RMSNORM_LINEAR_BIAS_TOPK_MI300,
                               code.to_string());
}

// Fused RMSNorm + MXFP4 Gang Linear + Bias.
// params: [output_stride, output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
//          actual_hidden_dim]
// Inputs: [norm_input, norm_weight, norm_output_scratch, mxfp4_weight, bias]
// Outputs: [linear_output]
int TaskRegister::register_gang_rmsnorm_linear_mxfp4_bias_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 5);
  int output_stride = params[0];
  int output_per_wg = params[1];
  int n_wgs_per_xcd = params[2];
  int total_tiles_per_xcd = params[3];
  int actual_hidden_dim = params[4];
  (void)total_tiles_per_xcd;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs =
      5; // norm_input, norm_weight, norm_output, mxfp4_weight, bias
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[0] is norm_input [batch, reduction_size]
  assert(input_ops[0]->dtensor.num_dims == 2);
  int batch_size = input_ops[0]->dtensor.dim[0];
  int reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_rmsnorm_linear_mxfp4_bias_kernel<$, $, $, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim);
  code.e("    task_desc->input_ptrs[0],");  // norm_input
  code.e("    task_desc->input_ptrs[1],");  // norm_weight
  code.e("    task_desc->input_ptrs[2],");  // norm_output scratch
  code.e("    task_desc->input_ptrs[3],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // linear_output
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", output_stride);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_MI300,
                               code.to_string());
}

// Fused RMSNorm + MXFP4 Gang Linear + Bias + Argmax (norm-once, internal tile
// loop). params: [output_stride, output_per_wg, n_wgs_per_xcd, workers_per_xcd,
//          actual_hidden_dim]
// total_tiles_per_xcd = workers_per_xcd (each worker enters once, loops
// internally). Inputs: [norm_input, norm_weight, norm_output, mxfp4_weight,
// bias] Outputs: [argmax_part_value (bf16), argmax_part_index (int64)]
//   plus an OPTIONAL third output [ppl_logits (f32)] used only by perplexity
//   mode. When absent the kernel gets a nullptr and skips the HBM logits
//   write entirely, so the serving path is byte-for-byte the previous code.
int TaskRegister::register_gang_rmsnorm_linear_mxfp4_bias_argmax_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 5);
  int output_stride = params[0];
  int output_per_wg = params[1];
  int n_wgs_per_xcd = params[2];
  int workers_per_xcd = params[3];
  int actual_hidden_dim = params[4];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  // 2 outputs normally, 3 when a perplexity logits sink is attached.
  int num_outputs = (int)bgraph.operators.size() - num_inputs;
  assert(num_outputs == 2 || num_outputs == 3);
  bool emit_logits = (num_outputs == 3);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int reduction_size = input_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_rmsnorm_linear_mxfp4_bias_argmax_kernel<$, $, $, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim);
  code.e("    task_desc->input_ptrs[0],");  // norm_input
  code.e("    task_desc->input_ptrs[1],");  // norm_weight
  code.e("    task_desc->input_ptrs[2],");  // norm_output scratch
  code.e("    task_desc->input_ptrs[3],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // argmax_part_value (bf16)
  code.e("    task_desc->output_ptrs[1],"); // argmax_part_index (int64)
  if (emit_logits) {
    code.e("    task_desc->output_ptrs[2],"); // ppl_logits (f32)
  } else {
    code.e("    nullptr,"); // no logits sink
  }
  // Row to write in the logits buffer: the position being scored. step[0] is
  // the last consumed position, so the token produced here lands at step+1.
  // NOTE: single-request only. With MPK_MAX_NUM_BATCHED_REQUESTS > 1 every
  // request would write this same row and race. The host gates PPL_MODE on
  // max_num_batched_requests == 1 (demo/gpt_oss/demo.py).
  code.e("    runtime_config.step[0] + 1,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", workers_per_xcd);
  code.e("    $,", output_stride);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_ARGMAX_MI300,
                               code.to_string());
}

// Fused MulSumAdd + RMSNorm + MXFP4 Gang Linear + Bias.
// params: [output_stride, output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
//          actual_hidden_dim, num_topk, input_stride]
// Inputs: [mlp_out, routing_weight, residual, norm_weight, norm_scratch,
//          mxfp4_weight, bias]
// Outputs: [x_output, qkv_output]
int TaskRegister::register_gang_mulsumradd_rmsnorm_linear_mxfp4_bias_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 7);
  int output_stride = params[0];
  int output_per_wg = params[1];
  int n_wgs_per_xcd = params[2];
  int total_tiles_per_xcd = params[3];
  int actual_hidden_dim = params[4];
  int num_topk = params[5];
  int input_stride = params[6];
  (void)total_tiles_per_xcd;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;  // mlp_out, routing_weight, residual, norm_weight,
                       // norm_scratch, mxfp4_weight, bias
  int num_outputs = 2; // x_output, qkv_output

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[2] is residual [batch, reduction_size]
  assert(input_ops[2]->dtensor.num_dims == 2);
  int batch_size = input_ops[2]->dtensor.dim[0];
  int reduction_size = input_ops[2]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_mulsumradd_rmsnorm_linear_mxfp4_bias_kernel<$, $, $, $, "
         "$, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim,
         num_topk,
         input_stride);
  code.e("    task_desc->input_ptrs[0],");  // mlp_out
  code.e("    task_desc->input_ptrs[1],");  // routing_weight
  code.e("    task_desc->input_ptrs[2],");  // residual
  code.e("    task_desc->input_ptrs[3],");  // norm_weight
  code.e("    task_desc->input_ptrs[4],");  // norm_scratch
  code.e("    task_desc->input_ptrs[5],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[6],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // x_output (MulSumAdd result)
  code.e("    task_desc->output_ptrs[1],"); // qkv_output
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", output_stride);
  code.e("    tile_idx);");
  return register_task_variant(
      TASK_GANG_MULSUMRADD_RMSNORM_LINEAR_MXFP4_BIAS_MI300, code.to_string());
}

// Fused RMSNorm + MXFP4 Gang Linear + KV Cache Update (layer 0).
// params: [output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
//          actual_hidden_dim, head_dim, num_q_per_kv, page_size,
//          kv_stride, q_ws_stride]
// Inputs: [norm_input, norm_weight, norm_output, mxfp4_weight, bias]
// Outputs: [k_cache, v_cache, q_workspace]
int TaskRegister::register_gang_rmsnorm_linear_mxfp4_bias_kvupd_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 9);
  int output_per_wg = params[0];
  int n_wgs_per_xcd = params[1];
  int actual_hidden_dim = params[3];
  int head_dim = params[4];
  int num_q_per_kv = params[5];
  int page_size = params[6];
  int kv_stride = params[7];
  int q_ws_stride = params[8];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 3;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  int batch_size = input_ops[0]->dtensor.dim[0];
  int reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_rmsnorm_linear_mxfp4_bias_kvupd_kernel<$, $, $, $, $, "
         "$, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim,
         head_dim,
         num_q_per_kv,
         page_size);
  code.e("    task_desc->input_ptrs[0],");  // norm_input
  code.e("    task_desc->input_ptrs[1],");  // norm_weight
  code.e("    task_desc->input_ptrs[2],");  // norm_output scratch
  code.e("    task_desc->input_ptrs[3],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // k_cache
  code.e("    task_desc->output_ptrs[1],"); // v_cache
  code.e("    task_desc->output_ptrs[2],"); // q_workspace
  code.e("    runtime_config.rope_cos_ptr,");
  code.e("    runtime_config.rope_sin_ptr,");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", kv_stride);
  code.e("    $,", q_ws_stride);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300,
                               code.to_string());
}

// Fused MulSumAdd + RMSNorm + MXFP4 Gang Linear + KV Cache Update (layers 1+).
// params: [output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
//          actual_hidden_dim, num_topk, input_stride,
//          head_dim, num_q_per_kv, page_size,
//          kv_stride, q_ws_stride]
// Inputs: [mlp_out, routing_weight, residual, norm_weight, norm_scratch,
//          mxfp4_weight, bias]
// Outputs: [x_output, k_cache, v_cache, q_workspace]
int TaskRegister::
    register_gang_mulsumradd_rmsnorm_linear_mxfp4_bias_kvupd_mi300_task(
        threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 11);
  int output_per_wg = params[0];
  int n_wgs_per_xcd = params[1];
  int actual_hidden_dim = params[3];
  int num_topk = params[4];
  int input_stride = params[5];
  int head_dim = params[6];
  int num_q_per_kv = params[7];
  int page_size = params[8];
  int kv_stride = params[9];
  int q_ws_stride = params[10];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 4;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  int batch_size = input_ops[2]->dtensor.dim[0];
  int reduction_size = input_ops[2]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_mulsumradd_rmsnorm_linear_mxfp4_bias_kvupd_kernel<$, $, "
         "$, $, $, $, $, $, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim,
         head_dim,
         num_q_per_kv,
         page_size,
         num_topk,
         input_stride);
  code.e("    task_desc->input_ptrs[0],");  // mlp_out
  code.e("    task_desc->input_ptrs[1],");  // routing_weight
  code.e("    task_desc->input_ptrs[2],");  // residual
  code.e("    task_desc->input_ptrs[3],");  // norm_weight
  code.e("    task_desc->input_ptrs[4],");  // norm_scratch
  code.e("    task_desc->input_ptrs[5],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[6],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // x_output
  code.e("    task_desc->output_ptrs[1],"); // k_cache
  code.e("    task_desc->output_ptrs[2],"); // v_cache
  code.e("    task_desc->output_ptrs[3],"); // q_workspace
  code.e("    runtime_config.rope_cos_ptr,");
  code.e("    runtime_config.rope_sin_ptr,");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", kv_stride);
  code.e("    $,", q_ws_stride);
  code.e("    tile_idx);");
  return register_task_variant(
      TASK_GANG_MULSUMRADD_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300,
      code.to_string());
}

// Fused ResAddF32 + RMSNorm + MXFP4 Gang Linear + Bias (layers 1+).
// Reads from f32 workspace (pre-accumulated by W2 atomicAdd) instead of
// MulSumAdd. params: [output_stride, output_per_wg, n_wgs_per_xcd,
// total_tiles_per_xcd, actual_hidden_dim] Inputs: [workspace_f32, residual,
// norm_weight, norm_scratch, mxfp4_weight, bias] Outputs: [x_output,
// qkv_output]
int TaskRegister::register_gang_resaddf32_rmsnorm_linear_mxfp4_bias_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 5);
  int output_stride = params[0];
  int output_per_wg = params[1];
  int n_wgs_per_xcd = params[2];
  int total_tiles_per_xcd = params[3];
  int actual_hidden_dim = params[4];
  (void)total_tiles_per_xcd;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 6;  // workspace_f32, residual, norm_weight, norm_scratch,
                       // mxfp4_weight, bias
  int num_outputs = 2; // x_output, qkv_output

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[1] is residual [batch, reduction_size]
  assert(input_ops[1]->dtensor.num_dims == 2);
  int batch_size = input_ops[1]->dtensor.dim[0];
  int reduction_size = input_ops[1]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_resaddf32_rmsnorm_linear_mxfp4_bias_kernel<$, $, $, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim);
  code.e("    task_desc->input_ptrs[0],");  // workspace_f32
  code.e("    task_desc->input_ptrs[1],");  // residual
  code.e("    task_desc->input_ptrs[2],");  // norm_weight
  code.e("    task_desc->input_ptrs[3],");  // norm_scratch
  code.e("    task_desc->input_ptrs[4],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[5],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // x_output
  code.e("    task_desc->output_ptrs[1],"); // qkv_output
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", output_stride);
  code.e("    tile_idx);");
  return register_task_variant(
      TASK_GANG_RESADDF32_RMSNORM_LINEAR_MXFP4_BIAS_MI300, code.to_string());
}

// Fused ResAddF32 + RMSNorm + MXFP4 Gang Linear + KV Cache Update (layers 1+).
// params: [output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
//          actual_hidden_dim, head_dim, num_q_per_kv, page_size,
//          kv_stride, q_ws_stride]
// Inputs: [workspace_f32, residual, norm_weight, norm_scratch, mxfp4_weight,
// bias] Outputs: [x_output, k_cache, v_cache, q_workspace]
int TaskRegister::
    register_gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_mi300_task(
        threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 9);
  int output_per_wg = params[0];
  int n_wgs_per_xcd = params[1];
  int actual_hidden_dim = params[3];
  int head_dim = params[4];
  int num_q_per_kv = params[5];
  int page_size = params[6];
  int kv_stride = params[7];
  int q_ws_stride = params[8];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 6;
  int num_outputs = 4;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  int batch_size = input_ops[1]->dtensor.dim[0];
  int reduction_size = input_ops[1]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_kernel<$, $, "
         "$, $, $, $, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim,
         head_dim,
         num_q_per_kv,
         page_size);
  code.e("    task_desc->input_ptrs[0],");  // workspace_f32
  code.e("    task_desc->input_ptrs[1],");  // residual
  code.e("    task_desc->input_ptrs[2],");  // norm_weight
  code.e("    task_desc->input_ptrs[3],");  // norm_scratch
  code.e("    task_desc->input_ptrs[4],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[5],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // x_output
  code.e("    task_desc->output_ptrs[1],"); // k_cache
  code.e("    task_desc->output_ptrs[2],"); // v_cache
  code.e("    task_desc->output_ptrs[3],"); // q_workspace
  code.e("    runtime_config.rope_cos_ptr,");
  code.e("    runtime_config.rope_sin_ptr,");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", kv_stride);
  code.e("    $,", q_ws_stride);
  code.e("    tile_idx);");
  return register_task_variant(
      TASK_GANG_RESADDF32_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300,
      code.to_string());
}

// Fused QKV + Attention gang task.
// Phase 1: ResAddF32+RMSNorm+QKV+KVUpdate (all workers, gang tiles)
// Phase 2: CK FMHA attention (1 worker per XCD, after hierarchical barrier)
// params: [output_per_wg, n_wgs_per_xcd, total_qkv_tiles_per_xcd,
//          actual_hidden_dim, head_dim, num_q_per_kv, page_size,
//          kv_stride, q_ws_stride,
//          max_seq_len, num_kv_chunks, q_workspace_stride, kv_cache_stride,
//          num_kv_heads, sliding_window, has_sinks, total_tiles_per_xcd]
// Inputs (9): workspace_f32, residual, norm_weight, norm_scratch, weight, bias,
//             sinks, barrier, lse_acc
// Outputs (5): x_output, k_cache, v_cache, q_workspace, o_acc
int TaskRegister::register_gang_qkv_attn_fused_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 16);
  int output_per_wg = params[0];
  int n_wgs_per_xcd = params[1];
  int total_qkv_tiles_per_xcd = params[2];
  int actual_hidden_dim = params[3];
  int head_dim = params[4];
  int num_q_per_kv = params[5];
  int page_size = params[6];
  int kv_stride = params[7];
  int q_ws_stride = params[8];
  int max_seq_len = params[9];
  int num_kv_chunks = params[10];
  int q_workspace_stride = params[11];
  int kv_cache_stride = params[12];
  int num_kv_heads = params[13];
  int sliding_window = params[14];
  int has_sinks = params[15];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 9;
  int num_outputs = 5;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // batch_size from residual tensor dim[0], reduction_size from weight tensor
  // dim[1]
  int batch_size = input_ops[1]->dtensor.dim[0];
  int reduction_size = input_ops[1]->dtensor.dim[1];
  float scale_s = 1.0f / sqrtf((float)head_dim) * 1.44269504088896340736f;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_qkv_attn_fused_kernel_mi300<$, $, $, $, $, $, $, $, $, "
         "$, $, $, $, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim,
         head_dim,
         num_q_per_kv,
         page_size,
         max_seq_len,
         num_kv_chunks,
         q_workspace_stride,
         kv_cache_stride,
         num_kv_heads,
         sliding_window,
         has_sinks);
  code.e("    task_desc->input_ptrs[0],");  // workspace_f32
  code.e("    task_desc->input_ptrs[1],");  // residual
  code.e("    task_desc->input_ptrs[2],");  // norm_weight
  code.e("    task_desc->input_ptrs[3],");  // norm_scratch
  code.e("    task_desc->input_ptrs[4],");  // weight
  code.e("    task_desc->input_ptrs[5],");  // bias
  code.e("    task_desc->input_ptrs[6],");  // sinks (nullable)
  code.e("    task_desc->input_ptrs[7],");  // barrier
  code.e("    task_desc->input_ptrs[8],");  // lse_acc
  code.e("    task_desc->output_ptrs[0],"); // x_output
  code.e("    task_desc->output_ptrs[1],"); // k_cache
  code.e("    task_desc->output_ptrs[2],"); // v_cache
  code.e("    task_desc->output_ptrs[3],"); // q_workspace
  code.e("    task_desc->output_ptrs[4],"); // o_acc
  code.e("    runtime_config.rope_cos_ptr,");
  code.e("    runtime_config.rope_sin_ptr,");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", kv_stride);
  code.e("    $,", q_ws_stride);
  code.e("    $f,", scale_s);
  code.e("    $,", total_qkv_tiles_per_xcd);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_QKV_ATTN_FUSED_MI300,
                               code.to_string());
}

// MoE residual add from f32 workspace (last layer).
// params: [output_stride]
// Inputs: [workspace_f32, residual]
// Outputs: [output]
int TaskRegister::register_moe_residual_add_f32_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 1);
  int output_stride = params[0];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[1] is residual [batch, hidden_size] bf16
  assert(input_ops[1]->dtensor.num_dims == 2);
  int batch_size = input_ops[1]->dtensor.dim[0];
  int hidden_size = input_ops[1]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::moe_residual_add_f32_mi300_impl<$, $, $>(",
         batch_size,
         hidden_size,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");   // workspace_f32
  code.e("    task_desc->input_ptrs[1],");   // residual
  code.e("    task_desc->output_ptrs[0]);"); // output
  return register_task_variant(TASK_MOE_RESIDUAL_ADD_F32_MI300,
                               code.to_string());
}

// MXFP4 Gang Linear with Residual + Bias.
// params: [output_stride, output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd]
// Inputs: [input, mxfp4_weight, residual, bias]
// Outputs: [output]
int TaskRegister::register_gang_linear_mxfp4_res_bias_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int output_stride = params[0];
  int output_per_wg = params[1];
  int n_wgs_per_xcd = params[2];
  int total_tiles_per_xcd = params[3];
  (void)total_tiles_per_xcd;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 4; // input, mxfp4_weight, residual, bias
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[0] is input [batch, reduction_size]
  assert(input_ops[0]->dtensor.num_dims == 2);
  int batch_size = input_ops[0]->dtensor.dim[0];
  int reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_linear_mxfp4_res_bias_kernel<$, $, $>(",
         batch_size,
         output_per_wg,
         reduction_size);
  code.e("    task_desc->input_ptrs[0],");  // input
  code.e("    task_desc->input_ptrs[1],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[2],");  // residual
  code.e("    task_desc->input_ptrs[3],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // output
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", output_stride);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_LINEAR_MXFP4_RES_BIAS_MI300,
                               code.to_string());
}

// Fused O-PROJ + RMSNorm + Router + TopK.
// params: [output_stride, output_per_wg, n_wgs_per_xcd, total_oproj_tiles,
//          actual_hidden_dim, num_experts, topk_k, router_tile_n,
//          total_topk_tiles, total_tiles_per_xcd]
// Inputs: [input, mxfp4_weight, residual, bias, norm_weight, norm_output,
//          router_weight, router_bias, logits_scratch, counters]
// Outputs: [output, topk_weight, routing_indices, active_expert_ids]
int TaskRegister::register_gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 10);
  int output_stride = params[0];
  int output_per_wg = params[1];
  int n_wgs_per_xcd = params[2];
  int total_oproj_tiles = params[3];
  int actual_hidden_dim = params[4];
  int num_experts = params[5];
  int topk_k = params[6];
  int router_tile_n = params[7];
  int total_topk_tiles = params[8];
  int total_tiles_per_xcd = params[9];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 10;
  int num_outputs = 4;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[0] is input [batch, reduction_size]
  assert(input_ops[0]->dtensor.num_dims == 2);
  int batch_size = input_ops[0]->dtensor.dim[0];
  int reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_linear_mxfp4_res_bias_rmsnorm_topk_kernel<$, $, $, $, "
         "$, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim,
         num_experts,
         topk_k);
  code.e("    task_desc->input_ptrs[0],");  // input (attn_out)
  code.e("    task_desc->input_ptrs[1],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[2],");  // residual
  code.e("    task_desc->input_ptrs[3],");  // oproj_bias
  code.e("    task_desc->input_ptrs[4],");  // norm_weight
  code.e("    task_desc->input_ptrs[5],");  // norm_output scratch
  code.e("    task_desc->input_ptrs[6],");  // router_weight
  code.e("    task_desc->input_ptrs[7],");  // router_bias
  code.e("    task_desc->input_ptrs[8],");  // logits_scratch
  code.e("    task_desc->input_ptrs[9],");  // counters (int32[2])
  code.e("    task_desc->output_ptrs[0],"); // output (attn_proj_out)
  code.e("    task_desc->output_ptrs[1],"); // topk_weight
  code.e("    task_desc->output_ptrs[2],"); // routing_indices
  code.e("    task_desc->output_ptrs[3],"); // active_expert_ids
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", output_stride);
  code.e("    $,", router_tile_n);
  code.e("    $,", total_oproj_tiles);
  code.e("    $,", total_topk_tiles);
  code.e("    $,", total_tiles_per_xcd);
  code.e("    tile_idx);");
  return register_task_variant(
      TASK_GANG_LINEAR_MXFP4_RES_BIAS_RMSNORM_TOPK_MI300, code.to_string());
}

// Fused O-PROJ+TopK+MoE (combines tasks 213 and 187 into one gang task).
// params: [output_stride, output_per_wg, n_wgs_per_xcd, total_oproj_tiles,
//          actual_hidden_dim, num_experts, topk_k, router_tile_n,
//          total_topk_tiles, oproj_tiles_per_xcd,
//          moe_intermediate_size, moe_hidden_size,
//          w13_output_per_wg, w2_output_per_wg,
//          moe_total_tiles_per_xcd]
// Inputs (16): [attn_out, oproj_weight, residual, oproj_bias, norm_weight,
//               norm_output, router_weight, router_bias, logits_scratch,
//               hier_counters, gate_up_weight, down_weight, w13_bias,
//               w2_bias, moe_barrier, swiglu_out]
// Outputs (6): [oproj_output, topk_weight, routing_indices,
//               active_expert_ids, routing_weight_moe, workspace_f32]
int TaskRegister::register_gang_oproj_topk_moe_fused_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 16);
  int output_stride = params[0];
  int output_per_wg = params[1];
  int n_wgs_per_xcd = params[2];
  int total_oproj_tiles = params[3];
  int actual_hidden_dim = params[4];
  int num_experts = params[5];
  int topk_k = params[6];
  int router_tile_n = params[7];
  int total_topk_tiles = params[8];
  int oproj_tiles_per_xcd = params[9];
  int moe_intermediate_size = params[10];
  int moe_hidden_size = params[11];
  int w13_output_per_wg = params[12];
  int w2_output_per_wg = params[13];
  int moe_total_tiles_per_xcd = params[14];
  int workers_per_xcd = params[15];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 16;
  int num_outputs = 6;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[0] is attn_out [batch, reduction_size]
  assert(input_ops[0]->dtensor.num_dims == 2);
  int batch_size = input_ops[0]->dtensor.dim[0];
  int reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_oproj_topk_moe_fused_kernel_mi300<$, $, $, $, $, $, $, "
         "$, $, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim,
         num_experts,
         topk_k,
         moe_intermediate_size,
         moe_hidden_size,
         w13_output_per_wg,
         w2_output_per_wg);
  // O-PROJ+TopK inputs (same as task 213)
  code.e("    task_desc->input_ptrs[0],"); // attn_out
  code.e("    task_desc->input_ptrs[1],"); // oproj_weight
  code.e("    task_desc->input_ptrs[2],"); // residual
  code.e("    task_desc->input_ptrs[3],"); // oproj_bias
  code.e("    task_desc->input_ptrs[4],"); // norm_weight
  code.e("    task_desc->input_ptrs[5],"); // norm_output scratch
  code.e("    task_desc->input_ptrs[6],"); // router_weight
  code.e("    task_desc->input_ptrs[7],"); // router_bias
  code.e("    task_desc->input_ptrs[8],"); // logits_scratch
  code.e("    task_desc->input_ptrs[9],"); // hier_counters
  // MoE inputs (same as task 187)
  code.e("    task_desc->input_ptrs[10],"); // gate_up_weight
  code.e("    task_desc->input_ptrs[11],"); // down_weight
  code.e("    task_desc->input_ptrs[12],"); // w13_bias
  code.e("    task_desc->input_ptrs[13],"); // w2_bias
  code.e("    task_desc->input_ptrs[14],"); // moe_barrier
  code.e("    task_desc->input_ptrs[15],"); // swiglu_out
  // Outputs
  code.e("    task_desc->output_ptrs[0],"); // oproj_output
  code.e("    task_desc->output_ptrs[1],"); // topk_weight
  code.e("    task_desc->output_ptrs[2],"); // routing_indices
  code.e("    task_desc->output_ptrs[3],"); // active_expert_ids
  code.e("    task_desc->output_ptrs[4],"); // routing_weight (for MoE W2)
  code.e("    task_desc->output_ptrs[5],"); // workspace_f32
  // O-PROJ parameters
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", output_stride);
  code.e("    $,", router_tile_n);
  code.e("    $,", total_oproj_tiles);
  code.e("    $,", total_topk_tiles);
  code.e("    $,", oproj_tiles_per_xcd);
  // MoE parameters
  code.e("    $,", moe_total_tiles_per_xcd);
  code.e("    $,", workers_per_xcd);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_OPROJ_TOPK_MOE_FUSED_MI300,
                               code.to_string());
}

// Full-layer fused gang task (QKV+Attn+O-proj+TopK+MoE in one dispatch).
// params: [qkv_output_per_wg, qkv_n_wgs_per_xcd, total_qkv_tiles_per_xcd,
//          actual_hidden_dim, head_dim, num_q_per_kv, page_size,
//          kv_stride, q_ws_stride, max_seq_len, num_kv_chunks,
//          q_workspace_stride, kv_cache_stride, num_kv_heads,
//          sliding_window, has_sinks,
//          oproj_output_per_wg, oproj_output_stride, total_oproj_tiles,
//          num_experts, topk_k, router_tile_n, total_topk_tiles,
//          oproj_tiles_per_xcd, moe_total_tiles_per_xcd,
//          w13_output_per_wg, w2_output_per_wg,
//          moe_intermediate_size, workers_per_xcd]
// Inputs (23): [workspace_f32, residual, norm_weight_pre, norm_scratch_pre,
//               qkv_weight, qkv_bias, attn_sinks, qkv_barrier, lse_acc,
//               oproj_weight, oproj_bias, norm_weight_post, norm_scratch_post,
//               router_weight, router_bias, logits_scratch, oproj_counters,
//               moe_gate_up_weight, moe_down_weight, moe_w13_bias, moe_w2_bias,
//               moe_barrier, moe_swiglu_out]
// Outputs (11): [x_output, k_cache, v_cache, q_workspace, o_acc,
//                attn_proj_out, topk_weight, routing_indices,
//                active_expert_ids, moe_routing_weight, moe_workspace_f32]
int TaskRegister::register_gang_full_layer_fused_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 29);
  int qkv_output_per_wg = params[0];
  int qkv_n_wgs_per_xcd = params[1];
  int total_qkv_tiles_per_xcd = params[2];
  int actual_hidden_dim = params[3];
  int head_dim = params[4];
  int num_q_per_kv = params[5];
  int page_size = params[6];
  int kv_stride = params[7];
  int q_ws_stride = params[8];
  int max_seq_len = params[9];
  int num_kv_chunks = params[10];
  int q_workspace_stride = params[11];
  int kv_cache_stride = params[12];
  int num_kv_heads = params[13];
  int sliding_window = params[14];
  int has_sinks = params[15];
  int oproj_output_per_wg = params[16];
  int oproj_output_stride = params[17];
  int total_oproj_tiles = params[18];
  int num_experts = params[19];
  int topk_k = params[20];
  int router_tile_n = params[21];
  int total_topk_tiles = params[22];
  int oproj_tiles_per_xcd = params[23];
  int moe_total_tiles_per_xcd = params[24];
  int w13_output_per_wg = params[25];
  int w2_output_per_wg = params[26];
  int moe_intermediate_size = params[27];
  int workers_per_xcd = params[28];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 24;
  int num_outputs = 11;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // input[1] = residual [batch, hidden_dim]
  int batch_size = input_ops[1]->dtensor.dim[0];
  // input[1] = residual dim[1] = QKV reduction_size (padded hidden)
  int qkv_reduction_size = input_ops[1]->dtensor.dim[1];
  // output[4] = o_acc [batch, attn_dim] — O-proj reduction size
  int oproj_reduction_size = output_ops[4]->dtensor.dim[1];
  // MoE hidden size: same as intermediate
  int moe_hidden_size = moe_intermediate_size;

  float scale_s = 1.0f / sqrtf((float)head_dim) * 1.44269504088896340736f;
  // oproj_tiles_per_xcd is now n_bblk * n_wgs_per_xcd (persistent_kernel.py):
  // tokens ride the MFMA N axis, so the host emits one tile block per 16
  // tokens, not one per token. Dividing by batch_size here would recover a
  // fractional -- and at bs>16 simply wrong -- weight-group count.
  int n_bblk = (batch_size + 15) / 16;
  int oproj_n_wgs_per_xcd = oproj_tiles_per_xcd / n_bblk;

  // DECODE_ONLY compiles out paged_attention_ck_fmha_prefill, which is only
  // reachable when a request contributes seqlen_q > 1 in one iteration. That
  // cannot happen while batch_size == 1, and hardcoding `true` there saved
  // real LDS/VGPR. At batch_size > 1 a single request supplies multiple query
  // rows per iteration, so seqlen_q > 1 and the branch IS taken -- baking in
  // `true` would silently produce no attention output for that request.
  bool decode_only = (batch_size == 1);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // 22 template parameters + DECODE_ONLY + NUM_REQS
  code.e("kernel::gang_full_layer_fused_kernel_mi300<$, $, $, $, $, $, $, $, "
         "$, $, $, $, $, $, $, $, $, $, $, $, $, $, $, "
         "MPK_MAX_NUM_BATCHED_REQUESTS>(",
         batch_size,
         qkv_output_per_wg,
         qkv_reduction_size,
         actual_hidden_dim,
         head_dim,
         num_q_per_kv,
         page_size,
         max_seq_len,
         num_kv_chunks,
         q_workspace_stride,
         kv_cache_stride,
         num_kv_heads,
         sliding_window,
         has_sinks,
         oproj_output_per_wg,
         oproj_reduction_size,
         num_experts,
         topk_k,
         moe_intermediate_size,
         moe_hidden_size,
         w13_output_per_wg,
         w2_output_per_wg,
         decode_only);
  // Pass input/output pointer arrays directly (2 params instead of 34)
  code.e("    task_desc->input_ptrs,");
  code.e("    task_desc->output_ptrs,");
  // 6 runtime config pointers
  code.e("    runtime_config.rope_cos_ptr,");
  code.e("    runtime_config.rope_sin_ptr,");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  // 14 runtime parameters
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", qkv_n_wgs_per_xcd);
  code.e("    $,", kv_stride);
  code.e("    $,", q_ws_stride);
  code.e("    $f,", scale_s);
  code.e("    $,", total_qkv_tiles_per_xcd);
  code.e("    $,", oproj_n_wgs_per_xcd);
  code.e("    $,", oproj_output_stride);
  code.e("    $,", router_tile_n);
  code.e("    $,", total_oproj_tiles);
  code.e("    $,", total_topk_tiles);
  code.e("    $,", oproj_tiles_per_xcd);
  code.e("    $,", moe_total_tiles_per_xcd);
  code.e("    $,", workers_per_xcd);
  code.e("    tile_idx,");
  // Deterministic layer counter, published by the ml loop into the free int32
  // of the n_tile union member. The task derives its barrier release values
  // from this instead of snapshotting a shared counter -- see the
  // layer_counter comment in gang_full_layer_fused_mi300.cuh.
  code.e("    (int)task_desc->task_metadata._linear_reserved);");
  return register_task_variant(TASK_GANG_FULL_LAYER_FUSED_MI300,
                               code.to_string());
}

int TaskRegister::register_gang_full_layer_with_lmhead_fused_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 33);
  int qkv_output_per_wg = params[0];
  int qkv_n_wgs_per_xcd = params[1];
  int total_qkv_tiles_per_xcd = params[2];
  int actual_hidden_dim = params[3];
  int head_dim = params[4];
  int num_q_per_kv = params[5];
  int page_size = params[6];
  int kv_stride = params[7];
  int q_ws_stride = params[8];
  int max_seq_len = params[9];
  int num_kv_chunks = params[10];
  int q_workspace_stride = params[11];
  int kv_cache_stride = params[12];
  int num_kv_heads = params[13];
  int sliding_window = params[14];
  int has_sinks = params[15];
  int oproj_output_per_wg = params[16];
  int oproj_output_stride = params[17];
  int total_oproj_tiles = params[18];
  int num_experts = params[19];
  int topk_k = params[20];
  int router_tile_n = params[21];
  int total_topk_tiles = params[22];
  int oproj_tiles_per_xcd = params[23];
  int moe_total_tiles_per_xcd = params[24];
  int w13_output_per_wg = params[25];
  int w2_output_per_wg = params[26];
  int moe_intermediate_size = params[27];
  int workers_per_xcd = params[28];
  int lm_output_per_wg = params[29];
  int lm_n_wgs_per_xcd = params[30];
  int lm_output_stride = params[31];
  int lm_actual_hidden_dim = params[32];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 28;
  int num_outputs = 13;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  int batch_size = input_ops[1]->dtensor.dim[0];
  int qkv_reduction_size = input_ops[1]->dtensor.dim[1];
  int oproj_reduction_size = output_ops[4]->dtensor.dim[1];
  // MoE hidden size: same as intermediate
  int moe_hidden_size = moe_intermediate_size;
  int lm_reduction_size = input_ops[24]->dtensor.dim[0];

  float scale_s = 1.0f / sqrtf((float)head_dim) * 1.44269504088896340736f;
  // oproj_tiles_per_xcd is now n_bblk * n_wgs_per_xcd (persistent_kernel.py):
  // tokens ride the MFMA N axis, so the host emits one tile block per 16
  // tokens, not one per token. Dividing by batch_size here would recover a
  // fractional -- and at bs>16 simply wrong -- weight-group count.
  int n_bblk = (batch_size + 15) / 16;
  int oproj_n_wgs_per_xcd = oproj_tiles_per_xcd / n_bblk;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_full_layer_with_lmhead_fused_kernel_mi300<$, $, $, $, "
         "$, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $>(",
         batch_size,
         qkv_output_per_wg,
         qkv_reduction_size,
         actual_hidden_dim,
         head_dim,
         num_q_per_kv,
         page_size,
         max_seq_len,
         num_kv_chunks,
         q_workspace_stride,
         kv_cache_stride,
         num_kv_heads,
         sliding_window,
         has_sinks,
         oproj_output_per_wg,
         oproj_reduction_size,
         num_experts,
         topk_k,
         moe_intermediate_size,
         moe_hidden_size,
         w13_output_per_wg,
         w2_output_per_wg,
         lm_output_per_wg,
         lm_reduction_size);
  code.e("    task_desc->input_ptrs,");
  code.e("    task_desc->output_ptrs,");
  code.e("    runtime_config.rope_cos_ptr,");
  code.e("    runtime_config.rope_sin_ptr,");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", qkv_n_wgs_per_xcd);
  code.e("    $,", kv_stride);
  code.e("    $,", q_ws_stride);
  code.e("    $f,", scale_s);
  code.e("    $,", total_qkv_tiles_per_xcd);
  code.e("    $,", oproj_n_wgs_per_xcd);
  code.e("    $,", oproj_output_stride);
  code.e("    $,", router_tile_n);
  code.e("    $,", total_oproj_tiles);
  code.e("    $,", total_topk_tiles);
  code.e("    $,", oproj_tiles_per_xcd);
  code.e("    $,", moe_total_tiles_per_xcd);
  code.e("    $,", workers_per_xcd);
  code.e("    $,", lm_n_wgs_per_xcd);
  code.e("    $,", lm_output_stride);
  code.e("    $,", lm_actual_hidden_dim);
  code.e("    tile_idx,");
  // See the matching comment in the non-LM-head variant above.
  code.e("    (int)task_desc->task_metadata._linear_reserved);");
  return register_task_variant(TASK_GANG_FULL_LAYER_WITH_LMHEAD_FUSED_MI300,
                               code.to_string());
}

// Gang linear with fused SiLU+mul.
// params: [output_stride, tile_n, m_tiles, m_per_tile,
// total_tile_pairs_per_xcd,
//          n_tile_pairs_per_xcd, wgm]
// Inputs: [activation(replicate), gate_up_weight(partition dim 0)]
// Outputs: [silu_mul_output(partition dim 1)]
int TaskRegister::register_gang_linear_silu_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 7);
  int output_stride = params[0];
  int tile_n = params[1];
  int m_tiles = params[2];
  int m_per_tile = params[3];
  int total_tiles_per_xcd = params[4];
  int n_tiles_per_xcd = params[5];
  int wgm = params[6];

  int reduction_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_linear_silu_kernel<bfloat16, $, $>(",
         m_per_tile,
         reduction_size);
  code.e("    task_desc->input_ptrs[0],");  // full activation
  code.e("    task_desc->input_ptrs[1],");  // XCD's interleaved gate+up weight
  code.e("    task_desc->output_ptrs[0],"); // XCD's output columns
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", m_tiles);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    $,", wgm);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_LINEAR_SILU_MI300, code.to_string());
}

int TaskRegister::register_linear_silu_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 7);
  int output_stride = params[0], tile_n = params[1], m_tiles = params[2];
  int m_per_tile = params[3], n_tiles = params[5], wgm = params[6];
  int reduction_size = 0;
  std::vector<tb::TBInputOp *> input_ops, output_ops;
  int num_inputs = 2, num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  reduction_size = input_ops[0]->dtensor.dim[1];
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_linear_silu_kernel<bfloat16, $, $>(",
         m_per_tile,
         reduction_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", m_tiles);
  code.e("    $,", n_tiles);
  code.e("    $,", wgm);
  code.e("    task_desc->task_metadata.expert_offset);");
  return register_task_variant(TASK_LINEAR_SILU_MI300, code.to_string());
}

// Gang MoE W13 linear: 8 tasks (1 per XCD), tile_idx = expert_local *
// tiles_per_expert + tile params: [tiles_per_expert, max_experts_per_xcd,
// total_tiles_per_xcd]
int TaskRegister::register_gang_moe_w13_linear_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 3);
  int tiles_per_expert = params[0];
  int max_experts_per_xcd = params[1];
  int total_tiles_per_xcd = params[2];
  (void)max_experts_per_xcd;
  (void)total_tiles_per_xcd;

  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Output: [batch, topk, output_size]
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];
  // Input: [batch, reduction_size]
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  reduction_size = input_ops[0]->output_tensors[0].dim[1];
  // Weight: [num_experts, output_size, reduction_size]
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];
  // Bias: [num_experts, output_stride]
  assert(input_ops[4]->output_tensors[0].num_dims == 2);
  // Output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);

  int n_tiles = output_size / 64;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e(
      "kernel::gang_moe_w13_linear_kernel<bfloat16, $, $, $, $, $, $, $, $>(",
      batch_size,
      output_size,
      output_stride,
      reduction_size,
      num_experts,
      num_experts_per_tok,
      tiles_per_expert,
      n_tiles);
  code.e("    task_desc->input_ptrs[0],");  // input activation
  code.e("    task_desc->input_ptrs[1],");  // expert weights
  code.e("    task_desc->input_ptrs[2],");  // routing indices
  code.e("    task_desc->input_ptrs[3],");  // mask
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // output
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_MOE_W13_LINEAR_MI300,
                               code.to_string());
}

// Gang MoE W2 linear: 8 tasks (1 per XCD), tile_idx = expert_local *
// tiles_per_expert + tile params: [tiles_per_expert, max_experts_per_xcd,
// total_tiles_per_xcd]
int TaskRegister::register_gang_moe_w2_linear_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 3);
  int tiles_per_expert = params[0];
  int max_experts_per_xcd = params[1];
  int total_tiles_per_xcd = params[2];
  (void)max_experts_per_xcd;
  (void)total_tiles_per_xcd;

  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Output: [batch, topk, output_size]
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];
  // Input: [batch, topk, reduction_size]
  assert(input_ops[0]->output_tensors[0].num_dims == 3);
  reduction_size = input_ops[0]->output_tensors[0].dim[2];
  // Weight: [num_experts, output_size, reduction_size]
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];
  // Bias: [num_experts, output_stride]
  assert(input_ops[4]->output_tensors[0].num_dims == 2);
  // Output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);

  int n_tiles = output_size / 64;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_moe_w2_linear_kernel<bfloat16, $, $, $, $, $, $, $, $>(",
         batch_size,
         output_size,
         output_stride,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         tiles_per_expert,
         n_tiles);
  code.e("    task_desc->input_ptrs[0],");  // input activation
  code.e("    task_desc->input_ptrs[1],");  // expert weights
  code.e("    task_desc->input_ptrs[2],");  // routing indices
  code.e("    task_desc->input_ptrs[3],");  // mask
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // output
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_MOE_W2_LINEAR_MI300, code.to_string());
}

// Gang MoE MXFP4 linear: 8 tasks (1 per XCD), MXFP4 weight dequant + MFMA
// params: [tiles_per_expert, max_experts_per_xcd, total_tiles_per_xcd,
// output_per_wg]
int TaskRegister::register_gang_moe_linear_mxfp4_mi300_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool w13_linear) {
  assert(params.size() == 4);
  int tiles_per_expert = params[0];
  int max_experts_per_xcd = params[1];
  int total_tiles_per_xcd = params[2];
  int output_per_wg = params[3];
  (void)max_experts_per_xcd;
  (void)total_tiles_per_xcd;

  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Output: [batch, topk, output_size]
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];
  // Input
  if (w13_linear) {
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    reduction_size = input_ops[0]->output_tensors[0].dim[1];
  } else {
    assert(input_ops[0]->output_tensors[0].num_dims == 3);
    reduction_size = input_ops[0]->output_tensors[0].dim[2];
  }
  // Weight: [num_experts, expert_wgs, wg_bytes]
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];
  // Bias: [num_experts, output_stride]
  assert(input_ops[4]->output_tensors[0].num_dims == 2);
  // Output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_moe_pipelined_mxfp4_kernel_mi300<$, $, $, $, $, $, $, "
         "$, $>(",
         batch_size,
         output_size,
         output_stride,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         tiles_per_expert,
         output_per_wg,
         w13_linear ? "true" : "false");
  code.e("    task_desc->input_ptrs[0],");  // input activation
  code.e("    task_desc->input_ptrs[1],");  // expert weights (MXFP4 packed)
  code.e("    task_desc->input_ptrs[2],");  // routing indices
  code.e("    task_desc->input_ptrs[3],");  // mask
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // output
  code.e("    tile_idx);");
  if (w13_linear) {
    return register_task_variant(TASK_GANG_MOE_W13_LINEAR_MXFP4_MI300,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_GANG_MOE_W2_LINEAR_MXFP4_MI300,
                                 code.to_string());
  }
}

// Gang fused W13+SwiGLU+W2 MXFP4 with per-expert pipelining.
// params: [tiles_per_expert, w13_output_per_wg, total_tiles_per_xcd,
// w2_output_per_wg]
int TaskRegister::register_gang_moe_fused_mxfp4_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int tiles_per_expert = params[0];
  int w13_output_per_wg = params[1];
  int total_tiles_per_xcd = params[2];
  int w2_output_per_wg = params[3];
  (void)tiles_per_expert;
  (void)total_tiles_per_xcd;

  // 8 inputs + 3 outputs
  // inputs: input, gate_up_weight, down_weight, routing, mask, w13_bias,
  // w2_bias, routing_weight outputs: swiglu_out, workspace_f32, barrier
  int num_inputs = 8;
  int num_outputs = 3;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // input[0]: [batch, hidden_size] BF16
  assert(input_ops[0]->dtensor.num_dims == 2);
  int batch_size = input_ops[0]->dtensor.dim[0];
  int hidden_size = input_ops[0]->dtensor.dim[1];

  // input[1]: gate_up weights [E, expert_wgs, wg_bytes]
  assert(input_ops[1]->dtensor.num_dims == 3);
  int num_experts = input_ops[1]->dtensor.dim[0];

  // output[0]: swiglu_out [batch, topk, intermediate_size]
  assert(output_ops[0]->dtensor.num_dims == 3);
  int num_topk = output_ops[0]->dtensor.dim[1];
  int intermediate_size = output_ops[0]->dtensor.dim[2];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_moe_fused_mxfp4_kernel_mi300<$, $, $, $, $, $, $>(",
         batch_size,
         intermediate_size,
         hidden_size,
         num_experts,
         num_topk,
         w13_output_per_wg,
         w2_output_per_wg);
  code.e("    task_desc->input_ptrs[0],"); // input [batch, hidden]
  code.e("    task_desc->input_ptrs[1],"); // gate_up weights [E, W13_WGS,
                                           // wg_bytes]
  code.e("    task_desc->input_ptrs[2],"); // down weights [E, W2_WGS, wg_bytes]
  code.e("    task_desc->input_ptrs[3],"); // routing [E, batch]
  code.e("    task_desc->input_ptrs[4],"); // mask [E+1]
  code.e("    task_desc->input_ptrs[5],"); // w13_bias [E, 2*intermediate]
  code.e("    task_desc->input_ptrs[6],"); // w2_bias [E, hidden]
  code.e("    task_desc->input_ptrs[7],"); // routing_weight [batch, topk] f32
  code.e("    task_desc->output_ptrs[0],"); // swiglu_out [batch, topk,
                                            // intermediate]
  code.e("    task_desc->output_ptrs[1],"); // workspace_f32 [batch, hidden] f32
  code.e("    task_desc->output_ptrs[2],"); // barrier [2*E]
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_MOE_FUSED_MXFP4_MI300,
                               code.to_string());
}

// Gang fused SwiGLU+W2 MXFP4 (no barrier, fused activation)
// Reads interleaved gate/up from W13 output, applies SwiGLU during FP8 quant,
// feeds into W2 MFMA. Same 5+1 signature as W2 but different kernel call.
int TaskRegister::register_gang_moe_swiglu_w2_mxfp4_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int tiles_per_expert = params[0];
  int max_experts_per_xcd = params[1];
  int total_tiles_per_xcd = params[2];
  int output_per_wg = params[3];
  (void)max_experts_per_xcd;
  (void)total_tiles_per_xcd;

  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // Output: [batch, topk, hidden_size]
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  int hidden_size = output_ops[0]->output_tensors[0].dim[2];

  // Input: [batch, topk, 2*intermediate] (interleaved gate/up from W13)
  assert(input_ops[0]->output_tensors[0].num_dims == 3);
  int w13_out_dim = input_ops[0]->output_tensors[0].dim[2];
  int intermediate_size = w13_out_dim / 2;

  // Weight: [num_experts, expert_wgs, wg_bytes] (W2 down weights)
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_moe_swiglu_w2_mxfp4_kernel_mi300<$, $, $, $, $, $, $>(",
         batch_size,
         intermediate_size,
         hidden_size,
         num_experts,
         num_experts_per_tok,
         tiles_per_expert,
         output_per_wg);
  code.e("    task_desc->input_ptrs[0],");  // w13 output (interleaved gate/up)
  code.e("    task_desc->input_ptrs[1],");  // W2 weights (MXFP4)
  code.e("    task_desc->input_ptrs[2],");  // routing indices
  code.e("    task_desc->input_ptrs[3],");  // mask
  code.e("    task_desc->input_ptrs[4],");  // W2 bias
  code.e("    task_desc->output_ptrs[0],"); // output
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_MOE_SWIGLU_W2_MXFP4_MI300,
                               code.to_string());
}

// Gang MoE W13 MXFP4 with SwiGLU fused into epilogue.
// Same MFMA as W13 but epilogue pairs gate/up outputs, applies SwiGLU,
// and writes half-sized output: [batch, topk, intermediate] instead of
// [batch, topk, 2*intermediate].
// params: [tiles_per_expert, max_experts_per_xcd, total_tiles_per_xcd,
// output_per_wg]
int TaskRegister::register_gang_moe_w13_swiglu_mxfp4_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int tiles_per_expert = params[0];
  int max_experts_per_xcd = params[1];
  int total_tiles_per_xcd = params[2];
  int output_per_wg = params[3];
  (void)max_experts_per_xcd;
  (void)total_tiles_per_xcd;

  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // Output: [batch, topk, intermediate] (half of W13's 2*intermediate)
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  int intermediate_size = output_ops[0]->output_tensors[0].dim[2];

  // Input: [batch, hidden] (W13_LINEAR=true, 2D input)
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int reduction_size = input_ops[0]->output_tensors[0].dim[1];

  // Weight: [num_experts, expert_wgs, wg_bytes] — still the full interleaved
  // gate_up weights
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];

  // The MFMA output size is 2*intermediate (full interleaved gate/up)
  int output_size = 2 * intermediate_size;
  int output_stride = output_size;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_moe_pipelined_mxfp4_kernel_mi300<$, $, $, $, $, $, $, "
         "$, true, true>(",
         batch_size,
         output_size,
         output_stride,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         tiles_per_expert,
         output_per_wg);
  code.e("    task_desc->input_ptrs[0],"); // input activation [batch, hidden]
  code.e("    task_desc->input_ptrs[1],"); // expert weights (MXFP4 packed,
                                           // interleaved gate/up)
  code.e("    task_desc->input_ptrs[2],"); // routing indices
  code.e("    task_desc->input_ptrs[3],"); // mask
  code.e("    task_desc->input_ptrs[4],"); // bias (interleaved gate/up bias)
  code.e(
      "    task_desc->output_ptrs[0],"); // output [batch, topk, intermediate]
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_MOE_W13_SWIGLU_MXFP4_MI300,
                               code.to_string());
}

// Gang CK FMHA attention: 8 tasks (1 per XCD), tile_idx → (request_id, kv_head)
// Fuses KV cache update + CK FMHA attention into one gang task.
// params: [num_q_heads, num_kv_heads, qk_norm, rotary_embed, max_seq_len,
//          page_size, num_kv_chunks, total_work_items_per_xcd,
//          total_work_items, q_workspace_stride]
int TaskRegister::register_gang_attn_split_kv_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 10);
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int max_seq_len = params[4];
  int page_size = params[5];
  int num_kv_chunks = params[6];
  int total_work_items = params[8];
  int q_workspace_stride = params[9];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 3;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int num_qo_per_kv = num_q_heads / num_kv_heads;
  int head_dim = input_ops[1]->output_tensors[0].dim[3];
  int kv_stride = head_dim * num_kv_heads;
  int max_tokens = input_ops[0]->dtensor.dim[0];

  // Cap MAX_TOKENS to fit in LDS (same as kv_cache_update)
  {
    constexpr int LDS_LIMIT = 58368; // 60KB - 3KB reserved
    int per_token_bytes = (num_qo_per_kv + 1) * head_dim * 2; // bf16
    int overhead = 256;
    int max_tokens_lds = (LDS_LIMIT - overhead) / per_token_bytes;
    if (max_tokens_lds < 1) {
      max_tokens_lds = 1;
    }
    if (max_tokens > max_tokens_lds) {
      max_tokens = max_tokens_lds;
    }
  }

  // Per-kv-head pointer offsets
  int qkv_head_offset = qkv_stride / num_kv_heads;
  int kv_cache_head_offset = head_dim;

  float scale_s = 1.0f / sqrtf((float)head_dim) * 1.44269504088896340736f;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_attention_split_kv_kernel<bfloat16,");
  code.e("    $, $, $, $, $, $, $, $, $, $, $, $>(",
         num_qo_per_kv,                     // NUM_QO_HEADS_PER_KV
         num_kv_heads,                      // NUM_KV_HEADS_ACTUAL
         kv_stride,                         // KV_CACHE_STRIDE
         qkv_stride,                        // QKV_STRIDE
         head_dim,                          // HEAD_DIM
         max_seq_len,                       // MAX_SEQ_LEN
         page_size,                         // PAGE_SIZE
         max_tokens,                        // MAX_TOKENS
         num_kv_chunks,                     // NUM_KV_CHUNKS
         qkv_head_offset,                   // QKV_HEAD_OFFSET
         kv_cache_head_offset,              // KV_CACHE_HEAD_OFFSET
         q_workspace_stride);               // Q_WORKSPACE_STRIDE
  code.e("    task_desc->input_ptrs[0],");  // qkv (full, un-partitioned)
  code.e("    task_desc->input_ptrs[1],");  // k_cache (full)
  code.e("    task_desc->input_ptrs[2],");  // v_cache (full)
  code.e("    task_desc->output_ptrs[1],"); // output (full)
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    $,", params[2] > 0);         // qk_norm
  code.e("    $,", params[3] > 0);         // rope
  code.e("    task_desc->input_ptrs[3],"); // q_norm
  code.e("    task_desc->input_ptrs[4],"); // k_norm
  code.e("    task_desc->input_ptrs[5],"); // cos
  code.e("    task_desc->input_ptrs[6],"); // sin
  code.e("    1e-6f,");
  code.e("    1e-6f,");
  code.e("    task_desc->output_ptrs[0],"); // lse (full)
  code.e("    task_desc->output_ptrs[2],"); // q_workspace (full)
  code.e("    $,", num_kv_heads);
  code.e("    $,", total_work_items);
  code.e("    tile_idx,");
  code.e("    $f);", scale_s);
  return register_task_variant(TASK_GANG_ATTN_SPLIT_KV_MI300, code.to_string());
}

// Gang merge split-KV: 8 tasks (1 per XCD), tile_idx → (request_id, kv_head)
// params: [num_qo_heads_per_kv, head_dim, max_seq_len, page_size,
//          num_kv_heads, total_work_items_per_xcd, total_work_items]
int TaskRegister::register_gang_attn_merge_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 7);
  int num_qo_heads_per_kv = params[0];
  int head_dim = params[1];
  int max_seq_len = params[2];
  int page_size = params[3];
  int num_kv_heads = params[4];
  int total_work_items = params[6];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  int max_tokens = input_ops[0]->dtensor.dim[0];
  constexpr int SEQ_LEN_PER_BLOCK = 128;
  int merge_output_head_offset = num_qo_heads_per_kv * head_dim;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_attention_merge_kernel<bfloat16,");
  code.e("    $, $, $, $, $, $, $, $>(",
         num_qo_heads_per_kv,
         num_kv_heads,
         head_dim,
         max_tokens,
         ((max_seq_len + SEQ_LEN_PER_BLOCK - 1) /
          SEQ_LEN_PER_BLOCK), // NUM_KV_CHUNKS
         SEQ_LEN_PER_BLOCK,   // KV_CHUNK_SIZE
         page_size,
         merge_output_head_offset);
  code.e("    task_desc->input_ptrs[0],"); // lse (full)
  code.e("    task_desc->input_ptrs[1],"); // output_tmp (full)
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->output_ptrs[0],"); // output (full)
  code.e("    $,", num_kv_heads);
  code.e("    $,", total_work_items);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_ATTN_MERGE_MI300, code.to_string());
}

int TaskRegister::register_splitk_reduce_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 1); // params[0] = K_SPLITS
  int k_splits = params[0];
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;  // workspace (float32) + residual (bf16)
  int num_outputs = 1; // output (bf16)

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Output: [BS, N_per_task]
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = output_ops[0]->output_tensors[0].dim[0];
  int output_size = output_ops[0]->output_tensors[0].dim[1];

  // Output stride (bf16)
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_output_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  int output_stride = static_cast<int>(kn_output_op->input_strides[0]);

  // Workspace stride (float32) - row stride in float elements
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_ws_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  int ws_stride = static_cast<int>(kn_ws_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::splitk_reduce<bfloat16, $, $, $, $, $>(",
         batch_size,
         output_size,
         k_splits,
         ws_stride,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_SPLITK_REDUCE_MI300, code.to_string());
}

int TaskRegister::register_splitk_linear_res_atomic_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0] = K_SPLITS
  assert(params.size() == 1);
  int k_splits = params[0];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  // Inputs: 0=input, 1=weight, 2=residual, 3=workspace(float32),
  // 4=done_counter(int32) Outputs: 5=output(bf16)
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // input[0]: [batch, K/K_SPLITS] — reduction_size per K-split
  assert(input_ops[0]->dtensor.num_dims == 2);
  int reduction_size = input_ops[0]->dtensor.dim[1];

  // output: [batch, NPerBlock]
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = output_ops[0]->output_tensors[0].dim[0];
  int n_per_block = output_ops[0]->output_tensors[0].dim[1];

  // Output stride (bf16)
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_output_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  int output_stride = static_cast<int>(kn_output_op->input_strides[0]);

  // Workspace stride (float32)
  assert(input_ops[3]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_ws_op =
      static_cast<kn::KNInputOp *>(input_ops[3]->dtensor.owner_op);
  int ws_stride = static_cast<int>(kn_ws_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::splitk_linear_res_atomic<bfloat16, $, $, $, $>(",
         batch_size,
         n_per_block,
         reduction_size,
         k_splits);
  code.e("    task_desc->input_ptrs[0],");       // input
  code.e("    task_desc->input_ptrs[1],");       // weight
  code.e("    task_desc->input_ptrs[2],");       // residual
  code.e("    task_desc->input_ptrs[3],");       // workspace (float32)
  code.e("    task_desc->output_ptrs[0],");      // output (bf16)
  code.e("    (int*)task_desc->input_ptrs[4],"); // done_counter (int32)
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $, $);", ws_stride, output_stride);
  return register_task_variant(TASK_SPLITK_LINEAR_RES_ATOMIC_MI300,
                               code.to_string());
}

int TaskRegister::register_argmax_partial_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
  // params[0]: num_partial_tasks
  assert(params.size() == 1);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 2;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int num_elements = input_ops[0]->output_tensors[0].dim[1];
  int num_partial_tasks = params[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::argmax_partial_kernel<bfloat16, $, $, $>(",
         batch_size,
         num_elements,
         num_partial_tasks);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->output_ptrs[1],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_ARGMAX_PARTIAL, code.to_string());
}

int TaskRegister::register_argmax_reduce_task(threadblock::Graph const &bgraph,
                                              std::vector<int> const &params) {
  // params[0]: output size
  assert(params.size() == 1);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int num_parts = input_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::argmax_reduce_kernel<bfloat16, $, $, $>(",
         batch_size,
         params[0],
         num_parts);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_ARGMAX_REDUCE, code.to_string());
}

int TaskRegister::register_reduce_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params) {
  // Currently, allreduce task is split to two sub-tasks: allgather + reduce
  // params[0]: num_gpus
  // params[1]: my_gpu_id
  assert(params.size() == 2);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // For now, the memory partition of the input[0] results in a strided
  // 2D tensor, which cannot be directly transferred by a single nvshmem
  // memput. So we use for loop to iterate over the first dim and transfer each
  // row. If the upperlayer changes this layout, this "for-loop" method can
  // fail. So we assert it here just in case.
  assert(input_ops[0]->input_map.x == 1 && input_ops[0]->input_map.y == -1 &&
         input_ops[0]->input_map.z == -1);
  // Currently support 2D reduction, buffer has an extra world_size dim
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int output_size = input_ops[0]->output_tensors[0].dim[1];
  // get output stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  int input_stride = static_cast<int>(kn_input_op->input_strides[0]);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  int output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  assert(input_stride == output_stride);
  // Register nvshmem copy task (allgather)
  mirage::transpiler::CodeKeeper c;
  c.inc_indent();
  c.e("size_t event_index = "
      "get_event_position_index(task_desc->trigger_event);");
  c.inc_indent();
  c.e("int gpu_id = "
      "static_cast<int>(get_event_gpu_id(task_desc->trigger_event));");
  c.e("assert(gpu_id < runtime_config.num_gpus);");
  c.e("assert(gpu_id != runtime_config.my_gpu_id);");
  c.e("for (int i = 0; i < $; i++) {", batch_size);
  c.e("  nvshmemx_putmem_signal_block(");
  c.e("      reinterpret_cast<char*>(task_desc->output_ptrs[0]) + i * $ * "
      "sizeof(bfloat16),",
      input_stride);
  c.e("      reinterpret_cast<char*>(task_desc->input_ptrs[0]) + i * $ * "
      "sizeof(bfloat16),",
      output_stride);
  c.e("      task_desc->xfer_size_in_bytes / $,", batch_size);
  c.e("      reinterpret_cast<uint64_t "
      "*>(&runtime_config.all_event_counters[event_index]),");
  c.e("      1 /*signal*/,");
  c.e("      NVSHMEM_SIGNAL_ADD,");
  c.e("      gpu_id);");
  c.e("}");
  register_task_variant(TASK_NVSHMEM_COPY, c.to_string());
  // Register reduction kernel
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::reduction_kernel<bfloat16, $, $, $, $, $>(",
         params[0],
         params[1],
         batch_size,
         output_size,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_REDUCE, code.to_string());
}

int TaskRegister::register_find_ngram_partial_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: ngram size
  assert(params.size() == 1);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int num_parts = output_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::find_ngram_partial_kernel<$, $>(", params[0], num_parts);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.step[0] + 1);");

  return register_task_variant(TASK_FIND_NGRAM_PARTIAL, code.to_string());
}

int TaskRegister::register_find_ngram_global_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: ngram size
  // params[1]: spec length
  assert(params.size() == 2);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int num_parts = input_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::find_ngram_global_kernel<$, $, $>(",
         params[0],
         params[1],
         num_parts);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.step[0]);");
  return register_task_variant(TASK_FIND_NGRAM_GLOBAL, code.to_string());
}

int TaskRegister::register_target_verify_greedy_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int num_spec_tokens = input_ops[0]->output_tensors[0].dim[1] - 1;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::target_verify_greedy_kernel<$>(", num_spec_tokens);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    (void*)(runtime_config.new_token_nums),"); // int pointer
  code.e("    (void*)(runtime_config.tokens + runtime_config.step[0] + 1));");
  return register_task_variant(TASK_TARGET_VERIFY_GREEDY, code.to_string());
}

int TaskRegister::register_linear_hopper_task(threadblock::Graph const &bgraph,
                                              std::vector<int> const &params,
                                              bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 128;
  int const Kstages = output_size >= 256 ? 3 : 6;
  int const SMEM_M_SIZE = batch_size;
  // int const SMEM_M_SIZE = 64;
  int const output_tma_cp_size = output_size < 64 ? output_size : 64;
  int const output_atom_size = (output_size >= 256)   ? 256
                               : (output_size >= 128) ? 128
                               : (output_size >= 64)  ? 64
                               : (output_size >= 32)  ? 32
                                                      : 16;
  code.e("using TMA_A = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         batch_size,        /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,          /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_B = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         output_atom_size,  /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,               /*SMEM_REPEAT_COL_*/
         output_atom_size * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  if (with_residual) {
    code.e(
        "using TMA_RESIDUAL = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, "
        "$, $, $, $, $, $, true>;",
        B,
        M,
        S,
        batch_size,         /*GMEM_ROW_*/
        output_size,        /*GMEM_COL_*/
        batch_size,         /*SMEM_ROW_*/
        output_tma_cp_size, /*SMEM_COL_*/
        output_stride,      /*GMEM_STRIDE_ROW_*/
        1,                  /*GMEM_STRIDE_COL_*/
        1,                  /*SMEM_REPEAT_ROW_*/
        (output_atom_size + output_tma_cp_size - 1) /
            output_tma_cp_size,         /*SMEM_REPEAT_COL_*/
        SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
    );
  }

  code.e("using TMA_OUT = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, "
         "$, $, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,         /*GMEM_ROW_*/
         output_size,        /*GMEM_COL_*/
         batch_size,         /*SMEM_ROW_*/
         output_tma_cp_size, /*SMEM_COL_*/
         output_stride,      /*GMEM_STRIDE_ROW_*/
         1,                  /*GMEM_STRIDE_COL_*/
         1,                  /*SMEM_REPEAT_ROW_*/
         (output_atom_size + output_tma_cp_size - 1) /
             output_tma_cp_size,         /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );
  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  if (with_residual) {
    code.e("TMA_RESIDUAL "
           "tma_residual(static_cast<CUtensorMap*>(task_desc->input_tma_desc_"
           "ptrs[2][0]));");
  }
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");
  // code.e("printf(\"linear_kernel_hopper start\");");

  code.e("kernel::linear_kernel_hopper<bfloat16, $, $, $, $, TMA_A, TMA_B, "
         "TMA_OUT, $, $>(",
         batch_size,
         output_size,
         reduction_size,
         Kstages,
         with_residual ? "TMA_RESIDUAL" : "void",
         output_stride);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    tma_out, ");
  if (with_residual) {
    code.e("    &tma_residual");
  } else {
    code.e("    nullptr");
  }
  code.e(");");

  if (with_residual) {
    return register_task_variant(TASK_LINEAR_WITH_RESIDUAL_HOPPER,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR_HOPPER, code.to_string());
  }
}
int TaskRegister::register_paged_attention_hopper_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: max_seq_len
  // params[5]: page_size
  assert(params.size() == 6);

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if ((int)input_ops.size() < num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // Shapes/strides
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int num_q_heads_per_kv = num_q_heads / num_kv_heads;
  int head_dim = output_size / num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  int max_tokens = input_ops[0]->dtensor.dim[0];

  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();

  constexpr int B = 3, M = 3, S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int KV_TILE_SIZE = 64;
  int const qkv_rows = num_q_heads_per_kv + 2;
  int const smem_repeat_col =
      (head_dim + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  int const q_smem_stride = max_tokens * num_q_heads_per_kv * TMA_CP_ASYNC_SIZE;
  int const kv_smem_stride = KV_TILE_SIZE * TMA_CP_ASYNC_SIZE;
  int const non_cached_kv_smem_stride = max_tokens * TMA_CP_ASYNC_SIZE;
  int const num_pages = (max_seq_len + page_size - 1) / page_size;
  int const num_head_group = qkv_stride / head_dim / (num_q_heads_per_kv + 2);

  // code.e("using TMA_Q = kernel::tma::tma_3d<bfloat16, $, $, $, $, $, $, $, $,
  // "
  //        "$, $, $, $, $, $, $, true>;",
  //        B,
  //        M,
  //        S,
  //        max_tokens,         /* GMEM_DEPTH */
  //        qkv_rows,           /* GMEM_ROW   */
  //        head_dim,           /* GMEM_COL   */
  //        max_tokens,         /* SMEM_DEPTH */
  //        num_q_heads_per_kv, /* SMEM_ROW   */
  //        TMA_CP_ASYNC_SIZE,  /* SMEM_COL   */
  //        qkv_stride,         /* GMEM_STRIDE_DEPTH */
  //        head_dim,           /* GMEM_STRIDE_ROW   */
  //        1,                  /* GMEM_STRIDE_COL   */
  //        1,                  /* SMEM_REPEAT_ROW   */
  //        smem_repeat_col,    /* SMEM_REPEAT_COL   */
  //        q_smem_stride       /* SMEM_STRIDE       */
  // );

  // code.e("using TMA_KV = kernel::tma::tma_3d<bfloat16, $, $, $, $, $, $, $,
  // $, "
  //        "$, $, $, $, $, $, $, true>;",
  //        B,
  //        M,
  //        S,
  //        max_tokens,               /* GMEM_DEPTH */
  //        qkv_rows,                 /* GMEM_ROW   */
  //        head_dim,                 /* GMEM_COL   */
  //        max_tokens,               /* SMEM_DEPTH */
  //        1,                        /* SMEM_ROW   */
  //        TMA_CP_ASYNC_SIZE,        /* SMEM_COL   */
  //        qkv_stride,               /* GMEM_STRIDE_DEPTH */
  //        head_dim,                 /* GMEM_STRIDE_ROW   */
  //        1,                        /* GMEM_STRIDE_COL   */
  //        1,                        /* SMEM_REPEAT_ROW   */
  //        smem_repeat_col,          /* SMEM_REPEAT_COL   */
  //        non_cached_kv_smem_stride /* SMEM_STRIDE       */
  // );

  // code.e("using TMA_PAGED_KV_CACHE = kernel::tma::tma_4d<bfloat16, $, $, $,
  // $, "
  //        "$, $, $, $, $, $, $, $, $, $, $, $, $, $, true>;",
  //        B,
  //        M,
  //        S,
  //        num_pages,                             /* GMEM_OUTERMOST_ */
  //        page_size,                             /* GMEM_DEPTH   */
  //        num_head_group,                        /* GMEM_ROW   */
  //        head_dim,                              /* GMEM_COL   */
  //        1,                                     /* SMEM_OUTERMOST_ */
  //        KV_TILE_SIZE,                          /* SMEM_DEPTH   */
  //        num_q_heads_per_kv,                    /* SMEM_ROW   */
  //        TMA_CP_ASYNC_SIZE,                     /* SMEM_COL   */
  //        page_size * head_dim * num_head_group, /* GMEM_STRIDE_OUTERMOST_ */
  //        page_size * head_dim,                  /* GMEM_STRIDE_DEPTH */
  //        head_dim,                              /* GMEM_STRIDE_ROW   */
  //        1,                                     /* GMEM_STRIDE_COL   */
  //        1,                                     /* SMEM_REPEAT_ROW   */
  //        smem_repeat_col,                       /* SMEM_REPEAT_COL   */
  //        kv_smem_stride                         /* SMEM_STRIDE       */
  // );

  // code.e("using TMA_OUTPUT = kernel::tma::tma_3d<bfloat16, $, $, $, $, $, $,
  // "
  //        "$, $, $, $, $, $, $, $, $, true>;",
  //        B,
  //        M,
  //        S,
  //        max_tokens,
  //        num_q_heads_per_kv * num_head_group,
  //        head_dim,
  //        max_tokens,
  //        num_q_heads_per_kv,
  //        TMA_CP_ASYNC_SIZE,
  //        head_dim * num_head_group * num_head_group,
  //        head_dim,
  //        1,
  //        1,
  //        smem_repeat_col,
  //        max_tokens * num_q_heads_per_kv * TMA_CP_ASYNC_SIZE);

  // code.e("TMA_Q  tma_q "
  //        "(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));");
  // code.e("TMA_KV tma_k "
  //        "(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][1]));");
  // code.e("TMA_KV tma_v "
  //        "(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][2]));");

  // code.e("TMA_PAGED_KV_CACHE "
  //        "tma_paged_k_cache(static_cast<CUtensorMap*>(task_desc->input_tma_"
  //        "desc_ptrs[1][0]));");
  // code.e("TMA_PAGED_KV_CACHE "
  //        "tma_paged_v_cache(static_cast<CUtensorMap*>(task_desc->input_tma_"
  //        "desc_ptrs[2][0]));");

  // code.e("TMA_OUTPUT "
  //        "tma_output(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs["
  //        "0][0]));");

  code.e("kernel::multitoken_paged_attention_hopper_impl<bfloat16, $, $, $, $, "
         "$, $, $, $, $, "
         "$, $, $, $>(",
         num_q_heads_per_kv, /* NUM_QO_HEADS               */
         1,                  /* NUM_KV_HEADS               */
         num_kv_heads,       /* NUM_QO_GROUPS              */
         kv_stride,          /* KV_CACHE_STRIDE            */
         qkv_stride,         /* QKV_STRIDE                 */
         output_size,        /* O_STRIDE (= num_q_heads*head_dim) */
         head_dim,           /* HEAD_DIM                   */
         -1,          /* SEQ_LEN (not used for non-split KV tasks)          */
         max_seq_len, /* MAX_SEQ_LEN                */
         page_size,   /* PAGE_SIZE                  */
         max_tokens,  /* MAX_TOKENS                 */
         "false",     /* PARTITION_KV               */
         1            /* NUM_KV_CHUNKS              */
  );
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0); // qk_norm
  code.e("    $,", params[3] > 0); // rope
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f,");
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    nullptr,"); // lse, not used for non-split KV tasks
  code.e("    0);");      // kv_idx, not used for non-split KV tasks

  return register_task_variant(TASK_PAGED_ATTENTION_HOPPER, code.to_string());
}

int TaskRegister::register_rmsnorm_hopper_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
  assert(params.size() == 0);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = output_ops[0]->output_tensors[0].dim[0];
  int hidden_dim = output_ops[0]->output_tensors[0].dim[1];

  // Currently assume that each rmsnorm task processes one token
  // assert(batch_size == 1);
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(output_ops[0]->dtensor.dim[0] == input_ops[0]->dtensor.dim[0]);
  assert(output_ops[0]->dtensor.dim[1] == input_ops[0]->dtensor.dim[1]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e(
      "kernel::rms_norm_hopper_impl<bfloat16, $, $>(", batch_size, hidden_dim);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    1e-6f);");
  return register_task_variant(TASK_RMS_NORM_HOPPER, code.to_string());
}

int TaskRegister::register_linear_swapAB_hopper_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 128;
  constexpr int Kstages = 5;
  assert(batch_size <= 16);
  int const SMEM_M_SIZE = batch_size <= 8 ? 8 : 16;
  // int const SMEM_M_SIZE = 16;
  int const output_tma_cp_size = output_size < 64 ? output_size : 64;
  int const output_atom_size = 64;
  code.e("using TMA_B = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         batch_size,        /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,          /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_A = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         output_atom_size,  /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,               /*SMEM_REPEAT_COL_*/
         output_atom_size * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  if (with_residual) {
    code.e(
        "using TMA_RESIDUAL = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, "
        "$, $, $, $, $, $, true>;",
        0,
        0,
        0,
        batch_size,                      /*GMEM_ROW_*/
        output_size,                     /*GMEM_COL_*/
        batch_size,                      /*SMEM_ROW_*/
        output_tma_cp_size,              /*SMEM_COL_*/
        output_stride,                   /*GMEM_STRIDE_ROW_*/
        1,                               /*GMEM_STRIDE_COL_*/
        1,                               /*SMEM_REPEAT_ROW_*/
        1,                               /*SMEM_REPEAT_COL_*/
        SMEM_M_SIZE * output_tma_cp_size /*SMEM_STRIDE_*/
    );
  }

  code.e("using TMA_OUT = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, "
         "$, $, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,                      /*GMEM_ROW_*/
         output_size,                     /*GMEM_COL_*/
         batch_size,                      /*SMEM_ROW_*/
         output_tma_cp_size,              /*SMEM_COL_*/
         output_stride,                   /*GMEM_STRIDE_ROW_*/
         1,                               /*GMEM_STRIDE_COL_*/
         1,                               /*SMEM_REPEAT_ROW_*/
         1,                               /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * output_tma_cp_size /*SMEM_STRIDE_*/
  );
  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");
  if (with_residual) {
    code.e("TMA_RESIDUAL "
           "tma_residual(static_cast<CUtensorMap*>(task_desc->input_tma_desc_"
           "ptrs[2][0]));");
  }
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");

  code.e(
      "kernel::linear_swapAB_kernel_hopper<bfloat16, $, $, $, $, TMA_A, TMA_B, "
      "TMA_OUT, $, $, $>(",
      batch_size,
      output_size,
      reduction_size,
      Kstages,
      with_residual ? "TMA_RESIDUAL" : "void",
      output_stride,
      "false" /*SplitK*/);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    tma_out, ");
  if (with_residual) {
    code.e("    &tma_residual,");
    code.e("    runtime_config.my_gpu_id == 0");
  } else {
    code.e("    nullptr,");
    code.e("    false/*residual*/");
  }

  code.e(");");

  if (with_residual) {
    return register_task_variant(TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR_SWAPAB_HOPPER, code.to_string());
  }
}

int TaskRegister::register_linear_cutlass_hopper_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
  int num_outputs = 1;
  constexpr int KSTAGES = 4;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  constexpr int TILE_SIZE = 128;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // NOTE: output_size and batch_size are swapped here
  code.e("auto problem_shape = cute::Shape<cute::Int<$>, cute::Int<$>, "
         "cute::Int<$>>{};",
         output_size,
         batch_size,
         reduction_size);
  // NOTE: output_size and batch_size are swapped here
  code.e("using KernelTraits = kernel::MMAKernelTraits<cutlass::bfloat16_t, $, "
         "$, $, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, "
         "cutlass::layout::RowMajor, cutlass::layout::RowMajor, $, $, $, $, "
         "decltype(problem_shape), $, $>;",
         output_size,
         batch_size,
         reduction_size,
         8,
         64,
         batch_size,
         TILE_SIZE,
         batch_size,
         KSTAGES);
  code.e("using Mainloop = kernel::CollectiveMainloop<KernelTraits>;");
  code.e("using Epilogue = kernel::CollectiveEpilogue<KernelTraits>;");
  // code.e("using StrideA = typename KernelTraits::StrideA;");
  // code.e("using StrideB = typename KernelTraits::StrideB;");
  // code.e("using StrideC = typename KernelTraits::StrideC;");
  // code.e("using StrideD = typename KernelTraits::StrideD;");
  // code.e("StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, "
  //        "{KernelTraits::OUTPUT_SIZE, KernelTraits::REDUCTION_SIZE, 1});");
  // code.e("StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, "
  //        "{KernelTraits::BATCH_SIZE, KernelTraits::REDUCTION_SIZE, 1});");
  // code.e("StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, "
  //        "{KernelTraits::BATCH_SIZE, KernelTraits::OUTPUT_SIZE, 1});");
  // code.e("StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, "
  //        "{KernelTraits::BATCH_SIZE, KernelTraits::OUTPUT_SIZE, 1});");
  // code.e("typename Mainloop::Arguments mainloop_args{");
  // code.e("    static_cast<cutlass::bfloat16_t const "
  //        "*>(task_desc.inputs[1].base_ptr),");
  // code.e("    stride_A,");
  // code.e("    static_cast<cutlass::bfloat16_t const "
  //        "*>(task_desc.inputs[0].base_ptr),");
  // code.e("    stride_B,");
  // code.e("};");
  // code.e("typename Epilogue::Arguments epilogue_args{");
  // code.e("    static_cast<cutlass::bfloat16_t const "
  //        "*>(task_desc.inputs[2].base_ptr),");
  // code.e("    stride_C,");
  // code.e(
  //     "    static_cast<cutlass::bfloat16_t
  //     *>(task_desc.outputs[0].base_ptr),");
  // code.e("    stride_C,");
  // code.e("    {1.0f, 1.0f},");
  // code.e("};");
  // code.e("using MainloopParamsDevice = typename Mainloop::template "
  //        "Params<false>;");
  // code.e("MainloopParamsDevice mainloop_params = "
  //        "Mainloop::to_underlying_arguments<false>(problem_shape, "
  //        "mainloop_args);");
  // code.e("typename Epilogue::Params epilogue_params = "
  //        "Epilogue::to_underlying_arguments(problem_shape, epilogue_args);");

  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int Kstages = 5;
  assert(batch_size <= 16);
  int const SMEM_M_SIZE = batch_size;
  int const output_tma_cp_size = output_size < 64 ? output_size : 64;
  int const output_atom_size = 64;

  code.e("using TMA_B = kernel::tma::tma_2d<cutlass::bfloat16_t, $, $, $, $, "
         "$, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         batch_size,        /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,          /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_A = kernel::tma::tma_2d<cutlass::bfloat16_t, $, $, $, $, "
         "$, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         output_atom_size,  /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,               /*SMEM_REPEAT_COL_*/
         output_atom_size * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");

  code.e("kernel::linear_cutlass_ws_hopper<Mainloop, Epilogue, false, "
         "cutlass::bfloat16_t, $, $, $, TMA_A, TMA_B, "
         "$, $>(",
         batch_size,
         output_size,
         reduction_size,
         output_stride,
         with_residual);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->input_ptrs[2]");
  code.e(");");

  if (with_residual) {
    return register_task_variant(TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR_CUTLASS_HOPPER, code.to_string());
  }
}

int TaskRegister::register_silu_mul_hopper_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, input_stride, output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(input_ops[0]->output_tensors[0].dim[1] == output_size * 2);
  // get input stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  input_stride = input_ops[0]->dtensor.dim[1];
  assert(input_stride == static_cast<int>(kn_input_op->input_strides[0]));
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::silu_mul_task_impl_hopper<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         input_stride,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_SILU_MUL_HOPPER, code.to_string());
}

int TaskRegister::register_embedding_hopper_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 1);
  // params[0]: input source (0: tokens, 1: input_token)
  int batch_size = 0, output_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::embedding_kernel_hopper<bfloat16, $, $, $>(",
         batch_size,
         output_size,
         output_stride);
  if (params[0] == 0) {
    code.e("    runtime_config.tokens + runtime_config.step[0], ");
  } else if (params[0] == 1) {
    code.e("    task_desc->input_ptrs[0],");
  }
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_EMBEDDING_HOPPER, code.to_string());
}

// SM100 Tasks
int TaskRegister::register_linear_sm100_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params,
                                             bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // define MMA
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;
  constexpr int bM = 128;
  constexpr int bN = MMA_N;
  constexpr int bK = 64;
  constexpr int num_ab_stages = 8;
  constexpr int num_acc_stages = 2;
  constexpr int num_c_stages = 4;
  constexpr int num_tmem_columns = bN * num_acc_stages;
  assert(num_tmem_columns <= 512);
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  int const output_tma_cp_size = 128;
  int const output_atom_size = 128;
  code.e("using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         MMA_M,             /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_M * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         MMA_N,             /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_N * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, "
         "$, $, $, $, $, true>;",
         0,
         M,
         S,
         batch_size,    /*GMEM_ROW_*/
         output_size,   /*GMEM_COL_*/
         MMA_N,         /*SMEM_ROW_*/
         MMA_M,         /*SMEM_COL_*/
         output_stride, /*GMEM_STRIDE_ROW_*/
         1,             /*GMEM_STRIDE_COL_*/
         1,             /*SMEM_REPEAT_ROW_*/
         (output_atom_size + output_tma_cp_size - 1) /
             output_tma_cp_size, /*SMEM_REPEAT_COL_*/
         MMA_N * MMA_M           /*SMEM_STRIDE_*/
  );
  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");
  // Bias Tensor setup
  code.e("cute::Layout layout_Bias = cute::make_layout(cute::make_shape($, $), "
         "cute::make_stride($, cute::Int<1>{}));",
         batch_size,
         output_size,
         output_stride);
  code.e("cute::Tensor mBias = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "$)), layout_Bias);",
         with_residual ? "task_desc->input_ptrs[2]" : "nullptr");
  code.e("kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, "
         "decltype(mBias), TMA_OUT, "
         "$, $, $, $, $, $, $, "
         "$, $, $>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size,
         reduction_size,
         with_residual ? "false" : "true",
         /*SplitK=*/"false",
         num_ab_stages,
         num_acc_stages,
         num_c_stages);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    mBias,");
  code.e("    tma_out); ");

  if (with_residual) {
    return register_task_variant(TASK_LINEAR_WITH_RESIDUAL_SM100,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR_SM100, code.to_string());
  }
}

int TaskRegister::register_splitk_linear_sm100_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0,
      reduction_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->output_tensors[0].dim[1];
  reduction_stride = input_ops[0]->dtensor.dim[1];
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // define MMA
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;
  constexpr int bM = 128;
  constexpr int bN = MMA_N;
  constexpr int bK = 64;
  constexpr int num_ab_stages = 8;
  constexpr int num_acc_stages = 2;
  constexpr int num_c_stages = 4;
  constexpr int num_tmem_columns = bN * num_acc_stages;
  assert(num_tmem_columns <= 512);
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  int const output_tma_cp_size = 128;
  int const output_atom_size = 128;
  code.e("using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         MMA_M,             /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_stride,  /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_M * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         MMA_N,             /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_stride,  /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_N * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, "
         "$, $, $, $, $, true>;",
         0,
         M,
         S,
         batch_size,    /*GMEM_ROW_*/
         output_size,   /*GMEM_COL_*/
         MMA_N,         /*SMEM_ROW_*/
         MMA_M,         /*SMEM_COL_*/
         output_stride, /*GMEM_STRIDE_ROW_*/
         1,             /*GMEM_STRIDE_COL_*/
         1,             /*SMEM_REPEAT_ROW_*/
         (output_atom_size + output_tma_cp_size - 1) /
             output_tma_cp_size, /*SMEM_REPEAT_COL_*/
         MMA_N * MMA_M           /*SMEM_STRIDE_*/
  );
  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");
  // Bias Tensor setup
  code.e("cute::Layout layout_Bias = cute::make_layout(cute::make_shape($, $), "
         "cute::make_stride($, cute::Int<1>{}));",
         batch_size,
         output_size,
         output_stride);
  code.e("cute::Tensor mBias = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "$)), layout_Bias);",
         with_residual ? "task_desc->input_ptrs[2]" : "nullptr");
  code.e("kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, "
         "decltype(mBias), TMA_OUT, "
         "$, $, $, $, $, $, $, "
         "$, $, $>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size,
         reduction_size,
         with_residual ? "false" : "true",
         /*SplitK=*/"true",
         num_ab_stages,
         num_acc_stages,
         num_c_stages);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    mBias,");
  code.e("    tma_out); ");

  return register_task_variant(TASK_SPLITK_LINEAR_SM100, code.to_string());
}

int TaskRegister::register_paged_attention_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: max_seq_len
  // params[5]: page_size
  assert(params.size() == 6);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = output_size / num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::multitoken_paged_attention_sm100_task_impl<bfloat16, $, $, "
         "$, $, "
         "$, $, $, $>(",
         num_q_heads / num_kv_heads,
         1,
         kv_stride,
         qkv_stride,
         output_size,
         head_dim,
         max_seq_len,
         page_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f);");
  return register_task_variant(TASK_ATTN_SM100, code.to_string());
}

int TaskRegister::register_argmax_partial_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_partial_tasks
  assert(params.size() == 1);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 2;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int num_elements = input_ops[0]->output_tensors[0].dim[1];
  int num_partial_tasks = params[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::argmax_partial_sm100_kernel<bfloat16, $, $, $>(",
         batch_size,
         num_elements,
         num_partial_tasks);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->output_ptrs[1],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_ARGMAX_PARTIAL_SM100, code.to_string());
}

int TaskRegister::register_argmax_reduce_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: output size
  assert(params.size() == 1);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int num_parts = input_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::argmax_reduce_sm100_kernel<bfloat16, $, $, $>(",
         batch_size,
         params[0],
         num_parts);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_ARGMAX_REDUCE_SM100, code.to_string());
}

int TaskRegister::register_sampling_sm100_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
  // params[0]: seed
  assert(params.size() == 1);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int vocab_size = input_ops[0]->output_tensors[0].dim[1];
  int seed = params[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::sampling_from_logits_kernel<256, 4, bfloat16, int>(");
  code.e("    static_cast<bfloat16*>(task_desc->input_ptrs[0]),");
  code.e("    static_cast<int*>(task_desc->output_ptrs[0]),");
  code.e("    $,", vocab_size);
  code.e("    $,", seed);
  code.e("    0,  // philox_offset");
  code.e("    $);", batch_size);
  return register_task_variant(TASK_SAMPLING_SM100, code.to_string());
}

int TaskRegister::register_tensor_init_task(threadblock::Graph const &bgraph,
                                            std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size, output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->dtensor.num_dims == 2);
  batch_size = input_ops[0]->output_tensors[0].dim[0];
  output_size = input_ops[0]->output_tensors[0].dim[1];
  // get input stride
  output_stride = input_ops[0]->dtensor.dim[1];
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::tensor_init_sm100_task_impl<cute::bfloat16_t, $, $, $>(",
         /*BATCH_SIZE=*/batch_size,
         /*OUTPUT_SIZE=*/output_size,
         /*OUTPUT_STRIDE=*/output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    0);");
  return register_task_variant(TASK_TENSOR_INIT, code.to_string());
}

int TaskRegister::register_moe_topk_softmax_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, num_experts = 0, num_experts_per_tok = 0, input_stride,
      output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 3;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  assert(output_ops[1]->output_tensors[0].num_dims == 2);
  assert(output_ops[2]->output_tensors[0].num_dims == 1);
  num_experts = output_ops[1]->output_tensors[0].dim[0];
  batch_size = output_ops[1]->output_tensors[0].dim[1];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  assert(output_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(output_ops[2]->output_tensors[0].dim[0] == num_experts + 1);
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[0]->output_tensors[0].dim[1] == num_experts);
  // get input stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  input_stride = input_ops[0]->dtensor.dim[1];
  assert(input_stride == static_cast<int>(kn_input_op->input_strides[0]));
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::topk_softmax_task_impl<cute::bfloat16_t, $, $, $, $>(",
         /*VPT=*/8,
         /*EXPERTS=*/num_experts,
         /*WARPS_PER_TB=*/8,
         /*BYTES_PER_LDG=*/16);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    nullptr,");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    $,", batch_size);
  code.e("    $,", num_experts_per_tok);
  code.e("    task_desc->output_ptrs[1],");
  code.e("    task_desc->output_ptrs[2],");
  code.e("    0,");
  code.e("    $,", num_experts);
  code.e("    true);");
  return register_task_variant(TASK_MOE_TOPK_SOFTMAX_SM100, code.to_string());
}

int TaskRegister::register_moe_linear_sm100_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool w13_linear) {
  assert(params.size() == 0);
  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      orig_output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 4;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];
  if (w13_linear) {
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    reduction_size = input_ops[0]->output_tensors[0].dim[1];
  } else {
    assert(input_ops[0]->output_tensors[0].num_dims == 3);
    reduction_size = input_ops[0]->output_tensors[0].dim[2];
    assert(input_ops[0]->output_tensors[0].dim[1] == num_experts_per_tok);
  }
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[1]->output_tensors[0].dim[1] == output_size);
  assert(input_ops[1]->output_tensors[0].dim[2] == reduction_size);
  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[2]->output_tensors[0].dim[1] == batch_size);
  assert(input_ops[3]->output_tensors[0].num_dims == 1);
  assert(input_ops[3]->output_tensors[0].dim[0] == num_experts + 1);
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);
  orig_output_size = input_ops[1]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // MoE constant:
  int expert_stride = (w13_linear) ? 10 : 8;
  // define MMA
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;
  constexpr int bM = 128;
  constexpr int bN = MMA_N;
  constexpr int bK = 64;
  constexpr int num_ab_stages = 8;
  constexpr int num_acc_stages = 2;
  constexpr int num_c_stages = 4;
  constexpr int num_tmem_columns = bN * num_acc_stages;
  assert(num_tmem_columns <= 512);
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  int const output_tma_cp_size = 128;
  int const output_atom_size = 128;
  // TMA_B for expert weights
  code.e("using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         (num_experts - 1) * orig_output_size + output_size, /*GMEM_ROW_*/
         reduction_size,                                     /*GMEM_COL_*/
         MMA_M,                                              /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE,                                  /*SMEM_COL_*/
         reduction_size, /*GMEM_STRIDE_ROW_*/
         1,              /*GMEM_STRIDE_COL_*/
         1,              /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_M * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  // Bias Tensor setup
  code.e(
      "cute::Layout layout_Bias = cute::make_layout(cute::make_shape($, $, $), "
      "cute::make_stride($, cute::Int<1>{}, $));",
      batch_size,
      output_size,
      num_experts,
      output_stride,
      output_stride * batch_size);
  code.e("cute::Tensor mBias = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "$)), layout_Bias);",
         "nullptr");
  // Topk_indices Tensor setup
  code.e("cute::Layout layout_routing_indices = "
         "cute::make_layout(cute::make_shape($, $), "
         "cute::make_stride($, cute::Int<1>{}));",
         num_experts,
         batch_size,
         batch_size);
  code.e("cute::Tensor mRoutingIndices = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::int32_t*>("
         "task_desc->input_ptrs[2])), layout_routing_indices);");
  // Topk_mask Tensor setup
  code.e("cute::Layout layout_expert_mask = "
         "cute::make_layout(cute::make_shape($), "
         "cute::make_stride(cute::Int<1>{}));",
         num_experts + 1);
  code.e("cute::Tensor mMask = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::int32_t*>("
         "task_desc->input_ptrs[3])), layout_expert_mask);");
  // Output Tensor setup
  code.e("cute::Layout layout_output = cute::make_layout(cute::make_shape($, "
         "$, $), "
         "cute::make_stride($, cute::Int<1>{}, $));",
         batch_size,
         output_size,
         num_experts_per_tok,
         num_experts_per_tok * output_stride,
         output_stride);
  code.e("cute::Tensor mOutput = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "task_desc->output_ptrs[0])), layout_output);");
  // Input Tensor setup
  if (w13_linear) {
    code.e(
        "cute::Layout layout_input = cute::make_layout(cute::make_shape($, $), "
        "cute::make_stride($, cute::Int<1>{}));",
        batch_size,
        reduction_size,
        reduction_size);
  } else {
    code.e("cute::Layout layout_input = cute::make_layout(cute::make_shape($, "
           "$, $), "
           "cute::make_stride($, cute::Int<1>{}, $));",
           batch_size,
           reduction_size,
           num_experts_per_tok,
           num_experts_per_tok * reduction_size,
           reduction_size);
  }
  code.e("cute::Tensor mInput = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "task_desc->input_ptrs[0])), layout_input);");

  code.e("kernel::moe_linear_sm100_task_impl<cute::bfloat16_t, TMA_A, "
         "decltype(mInput), decltype(mBias), decltype(mRoutingIndices), "
         "decltype(mMask), decltype(mOutput), "
         "$, $, $, $, $, $, $, $, $, $, $, "
         "$, $, $>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size,
         orig_output_size,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         expert_stride,
         w13_linear ? "true" : "false",
         /*no_bias*/ "true",
         num_ab_stages,
         num_acc_stages,
         num_c_stages);
  code.e("    tma_a,");
  code.e("    mInput,");
  code.e("    mBias,");
  code.e("    mRoutingIndices,");
  code.e("    mMask,");
  code.e("    mOutput,");
  code.e("    task_desc->task_metadata.expert_offset);");
  if (w13_linear) {
    return register_task_variant(TASK_MOE_W13_LINEAR_SM100, code.to_string());
  } else {
    return register_task_variant(TASK_MOE_W2_LINEAR_SM100, code.to_string());
  }
}

int TaskRegister::register_moe_silu_mul_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, num_experts_per_tok = 0, output_size = 0, input_stride,
      output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];
  assert(input_ops[0]->output_tensors[0].num_dims == 3);
  assert(input_ops[0]->output_tensors[0].dim[2] == output_size * 2);
  // get input stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  input_stride = input_ops[0]->dtensor.dim[2];
  assert(input_stride == static_cast<int>(kn_input_op->input_strides[1]));
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::silu_mul_task_impl<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         input_stride,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    $);", num_experts_per_tok * batch_size);
  return register_task_variant(TASK_SILU_MUL, code.to_string());
}

int TaskRegister::register_moe_swigluoai_task(threadblock::Graph const &bgraph,
                                              std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, num_experts_per_tok = 0, output_size = 0, input_stride,
      output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];
  assert(input_ops[0]->output_tensors[0].num_dims == 3);
  assert(input_ops[0]->output_tensors[0].dim[2] == output_size * 2);
  // get input stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  input_stride = input_ops[0]->dtensor.dim[2];
  assert(input_stride == static_cast<int>(kn_input_op->input_strides[1]));
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::swigluoai_task_impl<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         input_stride,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    $);", num_experts_per_tok * batch_size);
  return register_task_variant(TASK_SWIGLUOAI_MI300, code.to_string());
}

int TaskRegister::register_bias_add_mi300_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, size = 0, input_stride = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;  // input tensor + bias
  int num_outputs = 1; // output tensor
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Output: [batch_size, size]
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  size = output_ops[0]->output_tensors[0].dim[1];
  // Input tensor: [batch_size, size]
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[0]->output_tensors[0].dim[1] == size);
  // Bias: [1, size] (broadcast across batch)
  assert(input_ops[1]->output_tensors[0].num_dims == 2);
  assert(input_ops[1]->output_tensors[0].dim[1] == size);
  // Get input stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  input_stride = input_ops[0]->dtensor.dim[1];
  assert(input_stride == static_cast<int>(kn_input_op->input_strides[0]));
  // Get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::bias_add_task_impl<$, $, $, $>(",
         batch_size,
         size,
         input_stride,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_BIAS_ADD_MI300, code.to_string());
}

int TaskRegister::register_attention_sink_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: head_dim
  assert(params.size() == 2);
  int num_q_heads = params[0];
  int head_dim = params[1];
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e(
      "kernel::attention_sink_correction_impl<$, $>(", num_q_heads, head_dim);
  code.e("    task_desc->input_ptrs[0],"); // attn_out (in-place)
  code.e("    task_desc->input_ptrs[1],"); // lse_acc
  code.e("    task_desc->input_ptrs[2],"); // sinks
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    runtime_config.qo_indptr_buffer);");
  return register_task_variant(TASK_ATTENTION_SINK_MI300, code.to_string());
}

int TaskRegister::register_moe_mul_sum_add_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, num_experts_per_tok = 0, output_size = 0, input_stride,
      output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 3;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->output_tensors[0].num_dims == 3);
  assert(input_ops[1]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  num_experts_per_tok = input_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[0]->output_tensors[0].dim[2] ==
             input_ops[2]->output_tensors[0].dim[1] &&
         input_ops[0]->output_tensors[0].dim[2] == output_size);
  // get input stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  input_stride = input_ops[0]->dtensor.dim[2];
  assert(input_stride == static_cast<int>(kn_input_op->input_strides[1]));
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::mul_sum_add_sm100_task_impl<cute::bfloat16_t, $, $, $, $>(",
         /*BATCH_SIZE=*/batch_size,
         /*OUTPUT_SIZE=*/output_size,
         /*NUM_TOPK=*/num_experts_per_tok,
         /*OUTPUT_STRIDE=*/output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_MOE_MUL_SUM_ADD_SM100, code.to_string());
}

int TaskRegister::register_moe_linear_sm90_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool w13_linear) {
  assert(params.size() == 0);
  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      orig_output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 4;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];
  if (w13_linear) {
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    reduction_size = input_ops[0]->output_tensors[0].dim[1];
  } else {
    assert(input_ops[0]->output_tensors[0].num_dims == 3);
    reduction_size = input_ops[0]->output_tensors[0].dim[2];
    assert(input_ops[0]->output_tensors[0].dim[1] == num_experts_per_tok);
  }
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[1]->output_tensors[0].dim[1] == output_size);
  assert(input_ops[1]->output_tensors[0].dim[2] == reduction_size);
  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[2]->output_tensors[0].dim[1] == batch_size);
  assert(input_ops[3]->output_tensors[0].num_dims == 1);
  assert(input_ops[3]->output_tensors[0].dim[0] == num_experts + 1);
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);
  orig_output_size = input_ops[1]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // MoE constant:
  int const expert_stride = w13_linear ? 5 : 4;
  // define MMA
  constexpr int MMA_M = 64;
  constexpr int MMA_N = 16;
  constexpr int num_ab_stages = 8;
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  // int const output_tma_cp_size = 128;
  // int const output_atom_size = 128;
  // TMA_B for expert weights
  code.e("using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         //  (num_experts-1) * orig_output_size + output_size, /*GMEM_ROW_*/
         (num_experts)*orig_output_size, /*GMEM_ROW_*/
         reduction_size,                 /*GMEM_COL_*/
         MMA_M,                          /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE,              /*SMEM_COL_*/
         reduction_size,                 /*GMEM_STRIDE_ROW_*/
         1,                              /*GMEM_STRIDE_COL_*/
         1,                              /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_M * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  // Bias Tensor setup
  code.e(
      "cute::Layout layout_Bias = cute::make_layout(cute::make_shape($, $, $), "
      "cute::make_stride($, cute::Int<1>{}, $));",
      batch_size,
      output_size,
      num_experts,
      output_stride,
      output_stride * batch_size);
  code.e("cute::Tensor mBias = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "$)), layout_Bias);",
         "nullptr");
  // Topk_indices Tensor setup
  code.e("cute::Layout layout_routing_indices = "
         "cute::make_layout(cute::make_shape($, $), "
         "cute::make_stride($, cute::Int<1>{}));",
         num_experts,
         batch_size,
         batch_size);
  code.e("cute::Tensor mRoutingIndices = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::int32_t*>("
         "task_desc->input_ptrs[2])), layout_routing_indices);");
  // Topk_mask Tensor setup
  code.e("cute::Layout layout_expert_mask = "
         "cute::make_layout(cute::make_shape($), "
         "cute::make_stride(cute::Int<1>{}));",
         num_experts);
  code.e("cute::Tensor mMask = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::int32_t*>("
         "task_desc->input_ptrs[3])), layout_expert_mask);");
  // Output Tensor setup
  code.e("cute::Layout layout_output = cute::make_layout(cute::make_shape($, "
         "$, $), "
         "cute::make_stride($, cute::Int<1>{}, $));",
         batch_size,
         output_size,
         num_experts_per_tok,
         num_experts_per_tok * output_stride,
         output_stride);
  code.e("cute::Tensor mOutput = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "task_desc->output_ptrs[0])), layout_output);");
  // Input Tensor setup
  if (w13_linear) {
    code.e(
        "cute::Layout layout_input = cute::make_layout(cute::make_shape($, $), "
        "cute::make_stride($, cute::Int<1>{}));",
        batch_size,
        reduction_size,
        reduction_size);
  } else {
    code.e("cute::Layout layout_input = cute::make_layout(cute::make_shape($, "
           "$, $), "
           "cute::make_stride($, cute::Int<1>{}, $));",
           batch_size,
           reduction_size,
           num_experts_per_tok,
           num_experts_per_tok * reduction_size,
           reduction_size);
  }
  code.e("cute::Tensor mInput = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "task_desc->input_ptrs[0])), layout_input);");

  code.e("kernel::moe_linear_sm90_task_impl<cute::bfloat16_t, TMA_A, "
         "decltype(mInput), decltype(mBias), decltype(mRoutingIndices), "
         "decltype(mMask), decltype(mOutput), "
         "$, $, $, $, $, $, $, $, $, $, $, "
         "$>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size,
         orig_output_size,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         expert_stride,
         w13_linear ? "true" : "false",
         /*no_bias*/ "true",
         num_ab_stages);
  code.e("    tma_a,");
  code.e("    mInput,");
  code.e("    mBias,");
  code.e("    mRoutingIndices,");
  code.e("    mMask,");
  code.e("    mOutput,");
  code.e("    task_desc->task_metadata.expert_offset);");
  if (w13_linear) {
    return register_task_variant(TASK_MOE_W13_LINEAR_SM90, code.to_string());
  } else {
    return register_task_variant(TASK_MOE_W2_LINEAR_SM90, code.to_string());
  }
}

// ── MI300/MI350 MoE task registration ──────────────────────────────────

int TaskRegister::register_moe_topk_softmax_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, num_experts = 0, num_experts_per_tok = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 3;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  assert(output_ops[1]->output_tensors[0].num_dims == 2);
  assert(output_ops[2]->output_tensors[0].num_dims == 1);
  num_experts = output_ops[1]->output_tensors[0].dim[0];
  batch_size = output_ops[1]->output_tensors[0].dim[1];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  assert(output_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(output_ops[2]->output_tensors[0].dim[0] == num_experts + 1);
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[0]->output_tensors[0].dim[1] == num_experts);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::topk_softmax_mi300_task_impl<hip_bfloat16, $, $, $, $>(",
         /*VPT=*/8,
         /*EXPERTS=*/num_experts,
         /*WARPS_PER_CTA=*/4,
         /*BYTES_PER_LDG=*/16);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    $,", batch_size);
  // routing_row_stride: routing_indices is [num_experts, batch_size], and
  // this standalone variant runs all batch_size rows, so the number of rows
  // and the allocated stride coincide.
  code.e("    $,", batch_size);
  code.e("    $,", num_experts_per_tok);
  code.e("    task_desc->output_ptrs[1],");
  code.e("    task_desc->output_ptrs[2],");
  code.e("    0,");
  code.e("    $,", num_experts);
  code.e("    true);");
  return register_task_variant(TASK_MOE_TOPK_SOFTMAX_MI300, code.to_string());
}

int TaskRegister::register_moe_linear_mi300_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool w13_linear) {
  assert(params.size() == 0);
  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];
  if (w13_linear) {
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    reduction_size = input_ops[0]->output_tensors[0].dim[1];
  } else {
    assert(input_ops[0]->output_tensors[0].num_dims == 3);
    reduction_size = input_ops[0]->output_tensors[0].dim[2];
    assert(input_ops[0]->output_tensors[0].dim[1] == num_experts_per_tok);
  }
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[1]->output_tensors[0].dim[1] == output_size);
  assert(input_ops[1]->output_tensors[0].dim[2] == reduction_size);
  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[2]->output_tensors[0].dim[1] == batch_size);
  assert(input_ops[3]->output_tensors[0].num_dims == 1);
  assert(input_ops[3]->output_tensors[0].dim[0] == num_experts + 1);
  // input_ops[4] is bias: [num_experts, output_stride]
  assert(input_ops[4]->output_tensors[0].num_dims == 2);
  assert(input_ops[4]->output_tensors[0].dim[0] == num_experts);
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);

  // MI300 expert stride: must match grid_dim.x from the Python API
  int expert_stride = bgraph.grid_dim.x;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::moe_linear_kernel_mi300<bfloat16, $, $, $, $, $, $, $, $>(",
         batch_size,
         output_size,
         output_stride,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         expert_stride,
         w13_linear ? "true" : "false");
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->task_metadata.expert_offset);");
  if (w13_linear) {
    return register_task_variant(TASK_MOE_W13_LINEAR_MI300, code.to_string());
  } else {
    return register_task_variant(TASK_MOE_W2_LINEAR_MI300, code.to_string());
  }
}

int TaskRegister::register_moe_linear_mxfp4_mi300_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool w13_linear) {
  // params[0] = output_per_wg (typically 16)
  assert(params.size() == 1);
  int output_per_wg = params[0];
  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Output: [batch_size, num_experts_per_tok, output_size]
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];

  // Input activation: [batch_size, reduction_size] for W13, [batch_size, topk,
  // reduction_size] for W2
  if (w13_linear) {
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    reduction_size = input_ops[0]->output_tensors[0].dim[1];
  } else {
    assert(input_ops[0]->output_tensors[0].num_dims == 3);
    reduction_size = input_ops[0]->output_tensors[0].dim[2];
    assert(input_ops[0]->output_tensors[0].dim[1] == num_experts_per_tok);
  }

  // Weight: [num_experts, expert_wgs, wg_bytes] as uint8
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];

  // Routing indices: [num_experts, batch_size]
  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[2]->output_tensors[0].dim[1] == batch_size);
  // Mask: [num_experts + 1]
  assert(input_ops[3]->output_tensors[0].num_dims == 1);
  assert(input_ops[3]->output_tensors[0].dim[0] == num_experts + 1);
  // Bias: [num_experts, 1 (tiled), output_per_wg] bf16
  assert(input_ops[4]->output_tensors[0].num_dims == 3);
  assert(input_ops[4]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[4]->output_tensors[0].dim[2] == output_per_wg);

  // Get output stride from the KN-level input op
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);

  int expert_stride = bgraph.grid_dim.x;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::moe_linear_mxfp4_kernel_mi300<$, $, $, $, $, $, $, $, $>(",
         batch_size,
         output_size,
         output_stride,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         expert_stride,
         output_per_wg,
         w13_linear ? "true" : "false");
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->task_metadata.expert_offset);");
  if (w13_linear) {
    return register_task_variant(TASK_MOE_W13_LINEAR_MXFP4_MI300,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_MOE_W2_LINEAR_MXFP4_MI300,
                                 code.to_string());
  }
}

int TaskRegister::register_moe_linear_mxfp4_ck_mi300_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool w13_linear) {
  assert(params.size() == 1);
  int output_per_wg = params[0];
  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];

  if (w13_linear) {
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    reduction_size = input_ops[0]->output_tensors[0].dim[1];
  } else {
    assert(input_ops[0]->output_tensors[0].num_dims == 3);
    reduction_size = input_ops[0]->output_tensors[0].dim[2];
    assert(input_ops[0]->output_tensors[0].dim[1] == num_experts_per_tok);
  }

  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];

  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[2]->output_tensors[0].dim[1] == batch_size);
  assert(input_ops[3]->output_tensors[0].num_dims == 1);
  assert(input_ops[3]->output_tensors[0].dim[0] == num_experts + 1);
  assert(input_ops[4]->output_tensors[0].num_dims == 3);
  assert(input_ops[4]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[4]->output_tensors[0].dim[2] == output_per_wg);

  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);

  int expert_stride = bgraph.grid_dim.x;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::moe_linear_mxfp4_ck_kernel_mi300<$, $, $, $, $, $, $, $, $>(",
         batch_size,
         output_size,
         output_stride,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         expert_stride,
         output_per_wg,
         w13_linear ? "true" : "false");
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->task_metadata.expert_offset);");
  if (w13_linear) {
    return register_task_variant(TASK_MOE_W13_LINEAR_MXFP4_CK_MI300,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_MOE_W2_LINEAR_MXFP4_CK_MI300,
                                 code.to_string());
  }
}

int TaskRegister::register_moe_mul_sum_add_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, num_experts_per_tok = 0, output_size = 0, output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 3;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->output_tensors[0].num_dims == 3);
  assert(input_ops[1]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  num_experts_per_tok = input_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[0]->output_tensors[0].dim[2] ==
             input_ops[2]->output_tensors[0].dim[1] &&
         input_ops[0]->output_tensors[0].dim[2] == output_size);
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::mul_sum_add_mi300_task_impl<hip_bfloat16, $, $, $, $>(",
         /*BATCH_SIZE=*/batch_size,
         /*OUTPUT_SIZE=*/output_size,
         /*NUM_TOPK=*/num_experts_per_tok,
         /*OUTPUT_STRIDE=*/output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_MOE_MUL_SUM_ADD_MI300, code.to_string());
}

int TaskRegister::register_splitk_linear_swapAB_hopper_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool with_residual) {
  assert(params.size() == 0);
  assert(with_residual == false);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  constexpr int Kstages = 5;
  assert(batch_size <= 16);
  int const SMEM_M_SIZE = batch_size <= 8 ? 8 : 16;
  // int const SMEM_M_SIZE = 16;
  int const output_tma_cp_size = output_size < 64 ? output_size : 64;
  int const output_atom_size = 64;
  code.e("using TMA_B = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         batch_size,        /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,          /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_A = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         output_atom_size,  /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,               /*SMEM_REPEAT_COL_*/
         output_atom_size * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  if (with_residual) {
    code.e(
        "using TMA_RESIDUAL = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, "
        "$, $, $, $, $, $, true>;",
        0,
        0,
        0,
        batch_size,                      /*GMEM_ROW_*/
        output_size,                     /*GMEM_COL_*/
        batch_size,                      /*SMEM_ROW_*/
        output_tma_cp_size,              /*SMEM_COL_*/
        output_stride,                   /*GMEM_STRIDE_ROW_*/
        1,                               /*GMEM_STRIDE_COL_*/
        1,                               /*SMEM_REPEAT_ROW_*/
        1,                               /*SMEM_REPEAT_COL_*/
        SMEM_M_SIZE * output_tma_cp_size /*SMEM_STRIDE_*/
    );
  }

  code.e("using TMA_OUT = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, "
         "$, $, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,                      /*GMEM_ROW_*/
         output_size,                     /*GMEM_COL_*/
         batch_size,                      /*SMEM_ROW_*/
         output_tma_cp_size,              /*SMEM_COL_*/
         output_stride,                   /*GMEM_STRIDE_ROW_*/
         1,                               /*GMEM_STRIDE_COL_*/
         1,                               /*SMEM_REPEAT_ROW_*/
         1,                               /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * output_tma_cp_size /*SMEM_STRIDE_*/
  );
  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");
  if (with_residual) {
    code.e("TMA_RESIDUAL "
           "tma_residual(static_cast<CUtensorMap*>(task_desc->input_tma_desc_"
           "ptrs[2][0]));");
  }
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");

  code.e(
      "kernel::linear_swapAB_kernel_hopper<bfloat16, $, $, $, $, TMA_A, TMA_B, "
      "TMA_OUT, $, $, $>(",
      batch_size,
      output_size,
      reduction_size,
      Kstages,
      with_residual ? "TMA_RESIDUAL" : "void",
      output_stride,
      "true" /*SplitK*/);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    tma_out, ");
  if (with_residual) {
    code.e("    &tma_residual");
  } else {
    code.e("    nullptr");
  }
  code.e(");");

  return register_task_variant(TASK_SPLITK_LINEAR_SWAPAB_HOPPER,
                               code.to_string());
}

int TaskRegister::register_paged_attention_split_kv_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: max_seq_len
  // params[5]: page_size
  // params[6]: num_kv_chunks
  assert(params.size() == 7);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 2;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3); // lse
  assert(output_ops[1]->output_tensors[0].num_dims == 3); // output_tmp

  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = input_ops[1]->output_tensors[0].dim[3];
  int output_size = head_dim * num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  int num_kv_chunks = params[6];
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);
  int max_tokens = input_ops[0]->dtensor.dim[0];
  constexpr int SEQ_LEN_PER_BLOCK = 256;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::multitoken_paged_attention_split_kv_task_impl<bfloat16, $, "
         "$, $, $, $, $, "
         "$, $, $, $, $, $, $>(",
         num_q_heads / num_kv_heads,
         1,
         num_kv_heads,
         kv_stride,
         qkv_stride,
         output_size * num_kv_chunks, // o_stride should consider num_kv_chunks
         head_dim,
         SEQ_LEN_PER_BLOCK,
         max_seq_len,
         page_size,
         max_tokens,
         "true", // PARTITION_KV
         num_kv_chunks);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[1],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f,");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->task_metadata.kv_idx);");
  return register_task_variant(TASK_PAGED_ATTENTION_SPLIT_KV_SM100,
                               code.to_string());
}

int TaskRegister::register_paged_attention_split_kv_merge_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_qo_heads_per_kv
  // params[1]: head_dim
  // params[2]: max_seq_len
  // params[3]: page_size
  // params[4]: num_kv_heads
  assert(params.size() == 5);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads_per_kv = params[0];
  int head_dim = params[1];
  int max_seq_len = params[2];
  int page_size = params[3];
  int num_kv_heads = params[4];

  int max_tokens = input_ops[0]->dtensor.dim[0];
  constexpr int SEQ_LEN_PER_BLOCK = 256;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();

  code.e("kernel::merge_splitkv<bfloat16, $, $, $, $, $, $, "
         "$, $, $>(",
         num_q_heads_per_kv,
         1,
         num_kv_heads,
         head_dim,
         max_tokens,
         true,
         ((max_seq_len + SEQ_LEN_PER_BLOCK - 1) / SEQ_LEN_PER_BLOCK),
         SEQ_LEN_PER_BLOCK,
         page_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->task_metadata.merge_task_offset);");
  return register_task_variant(TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100,
                               code.to_string());
}

int TaskRegister::register_paged_attention_split_kv_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_embed
  // params[4]: max_seq_len
  // params[5]: page_size
  // params[6]: num_kv_chunks
  assert(params.size() == 7);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 2;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3); // lse
  assert(output_ops[1]->output_tensors[0].num_dims == 3); // output_tmp

  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = input_ops[1]->output_tensors[0].dim[3];
  int output_size = head_dim * num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  int num_kv_chunks = params[6];
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);
  int max_tokens = input_ops[0]->dtensor.dim[0];
  constexpr int SEQ_LEN_PER_BLOCK = 128;

  // Cap MAX_TOKENS to fit in attention kernel LDS budget.
  // The attention kernel needs Q_ROWS = MAX_TOKENS * qo_per_kv shared memory
  // rows. The scheduler's prepare_next_batch also caps tokens per request via
  // MPK_MAX_TOKENS_PER_REQUEST to ensure runtime num_tokens <= MAX_TOKENS.
  //
  // LDS budget per generation (must match MAX_DYNAMIC_SHARED_MEMORY_SIZE in
  // runtime_header.h, minus WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE=3KB):
  //   MI300X (gfx942): 60 KB total -> 57 KB usable = 58368 bytes
  //   MI350X (gfx950): 155 KB total -> 152 KB usable = 155648 bytes
  // Detected at runtime via hipDeviceProp_t::sharedMemPerBlock.
  {
    int qo_per_kv = num_q_heads / num_kv_heads;
    constexpr int KV_TILE = 64;
    int lds_limit = 58368; // MI300X conservative default
#ifdef MIRAGE_BACKEND_USE_ROCM
    // Query the device's actual LDS budget once and reserve 5 KB
    // (3 KB worker static + 2 KB safety margin).
    static int cached_lds_limit = []() {
      size_t per_block = mirage::utils::get_max_shared_mem();
      if (per_block > 5 * 1024) {
        return static_cast<int>(per_block - 5 * 1024);
      }
      return 58368;
    }();
    lds_limit = cached_lds_limit;
#endif
    int per_qrow = head_dim * 2 + KV_TILE * 4 + head_dim * 4 + 8;
    int fixed = KV_TILE * head_dim * 2 + 256;
    int max_qrows = (lds_limit - fixed) / per_qrow;
    int max_tokens_lds = max_qrows / qo_per_kv;
    if (max_tokens_lds < 1) {
      max_tokens_lds = 1;
    }
    if (max_tokens > max_tokens_lds) {
      max_tokens = max_tokens_lds;
    }
  }

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::multitoken_paged_attention_split_kv_task_impl<bfloat16, $, "
         "$, $, $, $, $, "
         "$, $, $, $, $, $>(",
         num_q_heads / num_kv_heads, /* NUM_QO_HEADS */
         1,                          /* NUM_KV_HEADS */
         num_kv_heads,               /* NUM_QO_GROUPS */
         kv_stride,                  /* KV_CACHE_STRIDE */
         qkv_stride,                 /* QKV_STRIDE */
         output_size *
             num_kv_chunks, /* O_STRIDE (accounts for num_kv_chunks) */
         head_dim,          /* HEAD_DIM */
         SEQ_LEN_PER_BLOCK, /* SEQ_LEN_PER_BLOCK */
         max_seq_len,       /* MAX_SEQ_LEN */
         page_size,         /* PAGE_SIZE */
         max_tokens,        /* MAX_TOKENS */
         num_kv_chunks);    /* NUM_KV_CHUNKS */
  code.e("    task_desc->input_ptrs[0],");  // qkv
  code.e("    task_desc->input_ptrs[1],");  // k_cache
  code.e("    task_desc->input_ptrs[2],");  // v_cache
  code.e("    task_desc->output_ptrs[1],"); // output_tmp
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],"); // q_norm
  code.e("    task_desc->input_ptrs[4],"); // k_norm
  code.e("    task_desc->input_ptrs[5],"); // cos
  code.e("    task_desc->input_ptrs[6],"); // sin
  code.e("    1e-6f,");
  code.e("    1e-6f,");
  code.e("    task_desc->output_ptrs[0],"); // lse
  code.e("    task_desc->task_metadata.kv_idx);");
  return register_task_variant(TASK_PAGED_ATTENTION_SPLIT_KV_MI300,
                               code.to_string());
}

int TaskRegister::register_paged_attention_split_kv_merge_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_qo_heads_per_kv
  // params[1]: head_dim
  // params[2]: max_seq_len
  // params[3]: page_size
  // params[4]: num_kv_heads
  assert(params.size() == 5);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int num_q_heads_per_kv = params[0];
  int head_dim = params[1];
  int max_seq_len = params[2];
  int page_size = params[3];
  int num_kv_heads = params[4];

  int max_tokens = input_ops[0]->dtensor.dim[0];
  constexpr int SEQ_LEN_PER_BLOCK = 128;

  // Cap MAX_TOKENS to match split-kv attention kernel's LDS limit
  {
    constexpr int KV_TILE = 64;
    constexpr int LDS_LIMIT = 58368;
    int per_qrow = head_dim * 2 + KV_TILE * 4 + head_dim * 4 + 8;
    int fixed = KV_TILE * head_dim * 2 + 256;
    int max_qrows = (LDS_LIMIT - fixed) / per_qrow;
    int max_tokens_lds = max_qrows / num_q_heads_per_kv;
    if (max_tokens_lds < 1) {
      max_tokens_lds = 1;
    }
    if (max_tokens > max_tokens_lds) {
      max_tokens = max_tokens_lds;
    }
  }

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // Reuse the portable merge_splitkv kernel from ampere/
  code.e("kernel::merge_splitkv<bfloat16, $, $, $, $, $, $, "
         "$, $, $>(",
         num_q_heads_per_kv,
         1,
         num_kv_heads,
         head_dim,
         max_tokens,
         true,
         ((max_seq_len + SEQ_LEN_PER_BLOCK - 1) / SEQ_LEN_PER_BLOCK),
         SEQ_LEN_PER_BLOCK,
         page_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->task_metadata.merge_task_offset);");
  return register_task_variant(TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_MI300,
                               code.to_string());
}

int TaskRegister::register_kv_cache_update_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_embed
  // params[4]: max_seq_len
  // params[5]: page_size
  // params[6]: q_workspace_stride
  assert(params.size() == 7);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = input_ops[1]->output_tensors[0].dim[3];
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  int q_workspace_stride = params[6];
  int max_tokens = input_ops[0]->dtensor.dim[0];

  // Cap MAX_TOKENS to fit in LDS. kv_cache_update uses:
  //   S_Q = sizeof(bf16) * MAX_TOKENS * (num_q/num_kv) * head_dim
  //   S_K = sizeof(bf16) * MAX_TOKENS * head_dim
  // Total dynamic smem ≈ MAX_TOKENS * (qo_per_kv + 1) * head_dim * 2
  // Runtime also caps tokens per request via MPK_MAX_TOKENS_PER_REQUEST (8 on
  // AMD).
  {
    int qo_per_kv = num_q_heads / num_kv_heads;
    constexpr int LDS_LIMIT = 58368;                      // 60KB - 3KB reserved
    int per_token_bytes = (qo_per_kv + 1) * head_dim * 2; // bf16
    int overhead = 256; // alignment + reduction buffer
    int max_tokens_lds = (LDS_LIMIT - overhead) / per_token_bytes;
    if (max_tokens_lds < 1) {
      max_tokens_lds = 1;
    }
    if (max_tokens > max_tokens_lds) {
      max_tokens = max_tokens_lds;
    }
  }

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e(
      "kernel::kv_cache_update_impl<bfloat16, $, $, $, $, $, $, $, $, $, $>(",
      num_q_heads / num_kv_heads,           /* NUM_QO_HEADS */
      1,                                    /* NUM_KV_HEADS */
      num_kv_heads,                         /* NUM_QO_GROUPS */
      kv_stride,                            /* KV_CACHE_STRIDE */
      qkv_stride,                           /* QKV_STRIDE */
      head_dim,                             /* HEAD_DIM */
      max_seq_len,                          /* MAX_SEQ_LEN */
      page_size,                            /* PAGE_SIZE */
      max_tokens,                           /* MAX_TOKENS */
      q_workspace_stride);                  /* Q_WORKSPACE_STRIDE */
  code.e("    task_desc->input_ptrs[0],");  // qkv
  code.e("    task_desc->input_ptrs[1],");  // k_cache
  code.e("    task_desc->input_ptrs[2],");  // v_cache
  code.e("    task_desc->output_ptrs[0],"); // q_workspace
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],"); // q_norm
  code.e("    task_desc->input_ptrs[4],"); // k_norm
  code.e("    task_desc->input_ptrs[5],"); // cos
  code.e("    task_desc->input_ptrs[6],"); // sin
  code.e("    1e-6f,");
  code.e("    1e-6f);");
  return register_task_variant(TASK_KV_CACHE_UPDATE_MI300, code.to_string());
}

int TaskRegister::register_paged_attention_ck_fmha_split_kv_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: head_dim
  // params[3]: page_size
  // params[4]: max_seq_len
  // params[5]: num_kv_chunks
  // params[6]: q_workspace_stride
  // params[7]: kv_cache_stride
  // params[8]: max_num_requests
  // params[9]: sliding_window (0 = disabled)
  assert(params.size() == 10);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_outputs = 2;
  // Inputs: 3 (q_workspace, k_cache, v_cache) or 4 (+ sinks for GPT-OSS)
  int total_ops = (int)bgraph.operators.size();
  int num_inputs = total_ops - num_outputs;
  assert(num_inputs == 3 || num_inputs == 4);
  bool has_sinks = (num_inputs == 4);

  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = params[2];
  int page_size = params[3];
  int max_seq_len = params[4];
  int num_kv_chunks = params[5];
  int q_workspace_stride = params[6];
  int kv_cache_stride = params[7];
  int max_num_requests = params[8];
  int sliding_window = params[9];
  int num_qo_per_kv = num_q_heads / num_kv_heads;
  float scale_s = 1.0f / sqrtf((float)head_dim) *
                  1.44269504088896340736f; // CK_TILE_FMHA_FWD_FAST_EXP2=1:
                                           // scale includes log2(e)

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::paged_attention_ck_fmha_split_kv_impl<bfloat16, $, $, $, $, "
         "$, $, $, $>(",
         num_qo_per_kv,                     /* NUM_QO_PER_KV */
         head_dim,                          /* HEAD_DIM */
         page_size,                         /* PAGE_SIZE */
         max_seq_len,                       /* MAX_SEQ_LEN */
         num_kv_chunks,                     /* NUM_KV_CHUNKS */
         q_workspace_stride,                /* Q_WORKSPACE_STRIDE */
         kv_cache_stride,                   /* KV_CACHE_STRIDE_T */
         num_kv_heads);                     /* NUM_KV_HEADS_T */
  code.e("    task_desc->input_ptrs[0],");  // q_workspace
  code.e("    task_desc->input_ptrs[1],");  // k_cache
  code.e("    task_desc->input_ptrs[2],");  // v_cache
  code.e("    task_desc->output_ptrs[0],"); // o_acc
  code.e("    task_desc->output_ptrs[1],"); // lse_acc
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");        // request_id
  code.e("    task_desc->task_metadata.merge_task_offset,"); // kv_head_idx
  code.e("    task_desc->task_metadata.kv_idx,");            // kv_chunk_idx
  code.e("    $f,", scale_s);
  code.e("    $,", sliding_window); // sliding_window (0 = disabled)
  if (has_sinks) {
    code.e("    task_desc->input_ptrs[3]);"); // sinks (GPT-OSS per-head
                                              // attention sinks)
  } else {
    code.e("    nullptr);"); // no sinks
  }
  return register_task_variant(TASK_PAGED_ATTENTION_CK_FMHA_SPLIT_KV_MI300,
                               code.to_string());
}

int TaskRegister::register_paged_attention_ck_fmha_merge_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_qo_heads_per_kv
  // params[1]: head_dim
  // params[2]: max_seq_len
  // params[3]: page_size
  // params[4]: num_kv_heads
  // params[5]: num_kv_chunks
  assert(params.size() == 6);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  // num_inputs is 2 (lse, o) or 3 (lse, o, sinks). The Python wrapper appends
  // the sinks tensor as a 3rd input when GPT-OSS-style sink correction is
  // needed.
  int num_outputs = 1;
  int num_inputs = static_cast<int>(bgraph.operators.size()) - num_outputs;
  assert(num_inputs == 2 || num_inputs == 3);
  bool has_sinks = (num_inputs == 3);

  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int num_q_heads_per_kv = params[0];
  int head_dim = params[1];
  int max_seq_len = params[2];
  int page_size = params[3];
  int num_kv_heads = params[4];
  int num_kv_chunks = params[5];

  int max_tokens = input_ops[0]->dtensor.dim[0];

  constexpr int SEQ_LEN_PER_BLOCK = 128;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // Use dedicated CK FMHA merge kernel that handles separate lse/o and output
  // strides
  code.e("kernel::merge_splitkv_ck_fmha<bfloat16, $, $, $, $, $, $>(",
         num_q_heads_per_kv,
         num_kv_heads, // NUM_QO_GROUPS (for output stride)
         head_dim,
         num_kv_chunks,
         SEQ_LEN_PER_BLOCK,
         page_size);
  code.e(
      "    reinterpret_cast<float const*>(task_desc->input_ptrs[0]),"); // lse_acc
  code.e(
      "    reinterpret_cast<float const*>(task_desc->input_ptrs[1]),"); // o_acc
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    reinterpret_cast<bfloat16*>(task_desc->output_ptrs[0]),");
  if (has_sinks) {
    code.e("    task_desc->task_metadata.merge_task_offset,"); // kv_head_idx
    code.e("    task_desc->input_ptrs[2]);");                  // sinks_ptr
  } else {
    code.e("    task_desc->task_metadata.merge_task_offset);"); // kv_head_idx
  }
  return register_task_variant(TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_MI300,
                               code.to_string());
}

int TaskRegister::register_paged_attention_split_kv_hopper_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: max_seq_len
  // params[5]: page_size
  // params[6]: num_kv_chunks
  assert(params.size() == 7);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 2;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3); // lse
  assert(output_ops[1]->output_tensors[0].num_dims == 3); // output_tmp

  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = input_ops[1]->output_tensors[0].dim[3];
  int output_size = head_dim * num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  int num_kv_chunks = params[6];
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);
  int max_tokens = input_ops[0]->dtensor.dim[0];
  constexpr int SEQ_LEN_PER_BLOCK = 256;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::multitoken_paged_attention_hopper_impl<bfloat16, $, "
         "$, $, $, $, $, "
         "$, $, $, $, $, $, $>(",
         num_q_heads / num_kv_heads, /* NUM_QO_HEADS */
         1,                          /* NUM_KV_HEADS */
         num_kv_heads,               /* NUM_QO_GROUPS */
         kv_stride,                  /* KV_CACHE_STRIDE */
         qkv_stride,                 /* QKV_STRIDE */
         output_size *
             num_kv_chunks, /* O_STRIDE (should consider num_kv_chunks) */
         head_dim,          /* HEAD_DIM */
         SEQ_LEN_PER_BLOCK, /* SEQ_LEN */
         max_seq_len,       /* MAX_SEQ_LEN */
         page_size,         /* PAGE_SIZE */
         max_tokens,        /* MAX_TOKENS */
         "true",            /* PARTITION_KV */
         num_kv_chunks);    /* NUM_KV_CHUNKS */
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f,");
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[1],"); // output_tmp
  code.e("    task_desc->output_ptrs[0],"); // lse
  code.e("    task_desc->task_metadata.kv_idx);");
  return register_task_variant(TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER,
                               code.to_string());
}

} // namespace runtime
} // namespace mirage
