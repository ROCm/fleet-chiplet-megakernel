// Offline gfx942 compile of every task kernel at the real DeepSeek-V2-Lite dims; no GPU, nothing is launched:
//   hipcc --offload-arch=gfx942 -std=c++17 -O3 -Wall -Werror -I<repo>/include/mirage/persistent_kernel/kernels -c tests/hip_smoke.hip -o /dev/null
// One __global__ per dsk:: instantiation of the default graph (dag=1, fold=1; kernels/switches.h defaults), i.e. the exact
// strings src/kernel/ds_task_register.cc emits into test.cu, called through a mock of the two Fleet structs the glue reads
// (as tests/dispatch_cpu.cc), plus the standalone fallback kernels (D9e, design §6 M1). -Wall catches HIP-isms,
// -Rpass-analysis=kernel-resource-usage prints VGPRs / LDS / scratch(spills) / occupancy per kernel
// T_MAX 1056 (9 chunks).
#include "argmax_partial.cuh"
#include "argmax_reduce.cuh"
#include "embed.cuh"
#include "fleet_dispatch.cuh"
#include "rmsnorm.cuh"
#include "splitk_reduce.cuh"

using namespace dsk;
struct TD {  // mirage::runtime::TaskDesc fields the glue reads
  void *input_ptrs[7];
  void *output_ptrs[3];
  struct { int expert_offset; } task_metadata;
};
struct RC { int *step; };  // mirage::runtime::RuntimeConfig
constexpr int HID = 2048, VOCAB_CHUNK = 1024, ARGMAX_PARTS = 100;
#define K __global__ void __launch_bounds__(WG_THREADS)

// default graph (build/graph/test.cu, 2026-09-15 container graph stage)
K s_qkv_norm(const TD *t, const RC *c) { ds_gemv_norm<64, GEMV_NONE, bf16, 2048, TD, RC>(t, *c); }
K s_lm_head_norm(const TD *t, const RC *c) { ds_gemv_norm<512, GEMV_NONE, bf16, 2048, TD, RC>(t, *c); }
K s_gate_up_norm(const TD *t, const RC *c) { ds_gemv_norm<64, GEMV_SILU, bf16, 2048, TD, RC>(t, *c); }
K s_shared_w13_norm(const TD *t, const RC *c) { ds_gemv_norm<176, GEMV_SILU, bf16, 2048, TD, RC>(t, *c); }
K s_shared_w2(const TD *t, const RC *c) { ds_gemv<64, GEMV_NONE, bf16, 2816, TD, RC>(t, *c); }
K s_splitk(const TD *t, const RC *c) { ds_gemv<64, GEMV_NONE, float, 2736, TD, RC>(t, *c); }
K s_kv_b_proj(const TD *t, const RC *c) { ds_kv_b_proj<1056, 64, TD, RC>(t, *c); }
K s_attn_chunk(const TD *t, const RC *c) { ds_attn_chunk<1056, 128, TD, RC>(t, *c); }
K s_o_proj_merge(const TD *t, const RC *c) { ds_o_proj_merge<9, 128, 64, TD, RC>(t, *c); }
K s_gate_topk_norm(const TD *t, const RC *c) { ds_moe_gate_topk_norm<64, 6, 2048, TD, RC>(t, *c); }
K s_expert_w13(const TD *t, const RC *c) { ds_expert_w13<64, 2816, 2048, 88, 176, 6, TD, RC>(t, *c); }
K s_expert_w2(const TD *t, const RC *c) { ds_expert_w2<64, 2048, 1408, 2816, 64, 6, TD, RC>(t, *c); }
K s_moe_combine(const TD *t, const RC *c) { ds_moe_combine<64, 6, 2048, TD, RC>(t, *c); }
// fallbacks for Fleet's own kernels
K s_splitk_reduce(const void *ws, const void *r, void *o) { splitk_reduce_task_impl<4>(ws, r, o, HID); }
K s_rmsnorm(const void *x, const void *w, void *o, int n) { rmsnorm_task_impl(x, w, o, n, 1e-6f); }
K s_embed(const void *tab, const long long *tok, int step, void *o) { embed_task_impl<HID>(tab, tok, step, o); }
K s_argmax_partial(const void *lg, void *v, void *i) { argmax_partial_task_impl<VOCAB_CHUNK>(lg, v, i); }
K s_argmax_reduce(const void *v, const void *i, void *t) { argmax_reduce_task_impl<VOCAB_CHUNK, ARGMAX_PARTS>(v, i, t); }
