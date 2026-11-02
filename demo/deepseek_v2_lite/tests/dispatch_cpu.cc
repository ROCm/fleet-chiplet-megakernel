// CPU compile check of the Fleet glue: the `_execute_task` if-chain that runtime.cc's codegen
// emits for our task types, with the strings src/kernel/ds_task_register.cc produces at the real-model dims,
// compiled against a mock of the two Fleet structs the glue reads. tests/test_glue.py runs
//   clang++ -std=c++17 -DFLEET_CPU_SHIM -fsyntax-only -I<repo>/include/mirage/persistent_kernel/kernels -Itests tests/dispatch_cpu.cc
// and asserts every mocked field line below appears verbatim in Fleet's runtime_header.h.
#include "fleet_dispatch.cuh"

namespace mirage {
namespace runtime {
int const MAX_INPUTS_PER_TASK = 7;
int const MAX_OUTPUTS_PER_TASK = 3;
struct FullTaskDesc {
  union TaskMetadata {
    struct {
      int expert_offset; // Used for MoE
    };
    unsigned long long raw_payload;
  };
};
enum TaskType {
  TASK_TERMINATE = 0,
  TASK_GEMV_BF16 = 300,
  TASK_GEMV_BF16_SILU = 301,
  TASK_GEMV_BF16_RES = 302,
  TASK_GEMV_SPLITK = 303,
  TASK_ATTN_CHUNK_192_128 = 306,
  TASK_ATTN_MERGE = 307,
  TASK_MOE_TOPK = 308,
  TASK_MOE_EXPERT_W13_SILU = 309,
  TASK_MOE_EXPERT_W2 = 310,
  TASK_MOE_COMBINE = 311,
};
struct alignas(16) TaskDesc {
  TaskType task_type;
  unsigned variant_id;
  void *input_ptrs[MAX_INPUTS_PER_TASK];
  void *output_ptrs[MAX_OUTPUTS_PER_TASK];
  FullTaskDesc::TaskMetadata task_metadata;
};
struct RuntimeConfig {
  int *step;                    // Metadata for LLM serving
};
}  // namespace runtime
}  // namespace mirage
using namespace mirage::runtime;

// One branch per (task type, variant) of the DeepSeek-V2-Lite graph, T_MAX = 1056 (9 chunks), design.md 2.4-2.5
// (12 branches since 2026-09-14: gate+topk merged, MLA_PREP folded (11b); lm_head 512-row tiles are gemv_bf16 variant 2, Q-H).
// The strings after `{` are exactly what ds_task_register.cc emits (test_glue.py checks the ds_* names).
__device__ __forceinline__ void _execute_task(TaskDesc const *task_desc, RuntimeConfig const &runtime_config) {
  if (task_desc->task_type == TASK_GEMV_BF16 && task_desc->variant_id == 0) {          // QKV_PROJ
    dsk::ds_gemv<64, dsk::GEMV_NONE, dsk::bf16, 2048>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_GEMV_BF16 && task_desc->variant_id == 1) {   // KV_B_PROJ
    dsk::ds_kv_b_proj<1056, 64>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_GEMV_BF16 && task_desc->variant_id == 2) {   // LM_HEAD (200 x 512 rows)
    dsk::ds_gemv<512, dsk::GEMV_NONE, dsk::bf16, 2048>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_GEMV_BF16_SILU && task_desc->variant_id == 0) {   // GATE_UP_SILU
    dsk::ds_gemv<64, dsk::GEMV_SILU, dsk::bf16, 2048>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_GEMV_BF16_SILU && task_desc->variant_id == 1) {   // EXPERT_W13_SHARED (DAG 12a, 32 x 176 rows)
    dsk::ds_gemv<176, dsk::GEMV_SILU, dsk::bf16, 2048>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_GEMV_BF16 && task_desc->variant_id == 3) {   // EXPERT_W2_SHARED (DAG 12a, 32 x 64 rows, K 2816)
    dsk::ds_gemv<64, dsk::GEMV_NONE, dsk::bf16, 2816>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_GEMV_BF16 && task_desc->variant_id == 4) {   // Q_PROJ / KVA_PROJ + input norm (L6, 12e)
    dsk::ds_gemv_norm<64, dsk::GEMV_NONE, dsk::bf16, 2048>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_GEMV_BF16_RES && task_desc->variant_id == 1) {   // O_PROJ_RESIDUAL + merge (L6, 12e)
    dsk::ds_o_proj_merge<9, 128, 64>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_MOE_TOPK && task_desc->variant_id == 2) {   // MOE_GATE_TOPK + norm2 (L6, 12e)
    dsk::ds_moe_gate_topk_norm<64, 6, 2048>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_MOE_TOPK && task_desc->variant_id == 1) {   // MOE_TOPK (L5, 12e)
    dsk::ds_moe_topk<64, 6>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_GEMV_BF16_RES && task_desc->variant_id == 0) {   // O_PROJ_RESIDUAL
    dsk::ds_gemv<64, dsk::GEMV_RES, dsk::bf16, 2048>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_GEMV_SPLITK && task_desc->variant_id == 0) {   // DOWN_SPLITK
    dsk::ds_gemv<64, dsk::GEMV_NONE, float, 2736>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_ATTN_CHUNK_192_128 && task_desc->variant_id == 0) {   // ATTN_CHUNK
    dsk::ds_attn_chunk<1056, 128>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_ATTN_MERGE && task_desc->variant_id == 0) {   // ATTN_MERGE
    dsk::ds_attn_merge<9>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_MOE_TOPK && task_desc->variant_id == 0) {   // MOE_GATE_TOPK
    dsk::ds_moe_gate_topk<64, 6, 2048>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_MOE_EXPERT_W13_SILU && task_desc->variant_id == 0) {   // EXPERT_W13_SILU (7 x 32, 11a)
    dsk::ds_expert_w13<64, 2816, 2048, 88, 176, 6>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_MOE_EXPERT_W2 && task_desc->variant_id == 0) {   // EXPERT_W2 (7 x 32)
    dsk::ds_expert_w2<64, 2048, 1408, 2816, 64, 6>(task_desc, runtime_config);
  } else if (task_desc->task_type == TASK_MOE_COMBINE && task_desc->variant_id == 0) {   // MOE_COMBINE_RESIDUAL
    dsk::ds_moe_combine<64, 6, 2048>(task_desc, runtime_config);
  }
}
