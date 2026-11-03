/* Copyright 2026 Nikhil (github.com/badnikhil)
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

/* DeepSeek-V2-Lite task registration for Fleet/MPK (demo/deepseek_v2_lite).
 * Picked up by the CMake glob of src/; hooked from Graph::register_task (graph.cc) as the last `else if`. One branch per
 * register name (demo/deepseek_v2_lite/tests/test_glue.py CONTRACT): shape asserts, (num_inputs, num_outputs, TaskType) and
 * the one-line call into include/mirage/persistent_kernel/kernels/fleet_dispatch.cuh that the generated _execute_task
 * executes. Tensor order = inputs first, then outputs, in `_op()` order (fleet_ds/mirage_graph.py).
 */
#include "mirage/kernel/graph.h"
#include "mirage/kernel/task_register.h"
#include "mirage/threadblock/graph.h"
#include "mirage/type.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

namespace mirage {
namespace kernel {

namespace {
using mirage::runtime::TaskType;
namespace tb = mirage::threadblock;

std::string cat(std::initializer_list<std::string> parts) {
  std::string s;
  for (auto const &p : parts) s += p;
  return s;
}
std::string I(int v) { return std::to_string(v); }
std::string args(std::vector<int> const &v) {  // "<a, b, c>"
  std::string s = "<";
  for (size_t i = 0; i < v.size(); i++) s += (i ? ", " : "") + I(v[i]);
  return s + ">";
}
}  // namespace

// On a failed shape check: print the op's grid and every TB input's per-task / full dims, then abort. The per-task
// dim is dim / gcd(dim, grid) (src/threadblock/input_loader.cc), so a grid that does not divide a dim shows up here.
static void ds_dump(std::string const &name, tb::Graph const &bgraph, std::vector<tb::TBInputOp *> const &ops) {
  fprintf(stderr, "ds_register_task(\"%s\") grid (%d, %d, %d):\n", name.c_str(), bgraph.grid_dim.x, bgraph.grid_dim.y,
          bgraph.grid_dim.z);
  for (size_t i = 0; i < ops.size(); i++) {
    auto const &t = ops[i]->output_tensors[0];
    auto const &f = ops[i]->dtensor;
    fprintf(stderr, "  [%zu] per-task [", i);
    for (int d = 0; d < t.num_dims; d++) fprintf(stderr, "%s%d", d ? ", " : "", t.dim[d]);
    fprintf(stderr, "]  full [");
    for (int d = 0; d < f.num_dims; d++) fprintf(stderr, "%s%d", d ? ", " : "", f.dim[d]);
    fprintf(stderr, "]  dtype %d  map (%d, %d, %d)\n", (int)f.data_type, ops[i]->input_map.x, ops[i]->input_map.y,
            ops[i]->input_map.z);
  }
}
#define DS_CHECK(cond)                                                                        \
  do {                                                                                        \
    if (!(cond)) {                                                                            \
      ds_dump(name, bgraph, ops);                                                             \
      fprintf(stderr, "ds_task_register.cc:%d: shape check failed: %s\n", __LINE__, #cond);   \
      abort();                                                                                \
    }                                                                                         \
  } while (0)

// Returns false if `name` is not one of ours (graph.cc then falls through to its "Unsupported task" assert).
bool ds_register_task(std::string const &name,
                      tb::Graph const &bgraph,
                      std::vector<int> const &params,
                      std::tuple<int, int, TaskType, int> &cfg) {
  std::vector<tb::TBInputOp *> ops;
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    ops.push_back(static_cast<tb::TBInputOp *>(op));
  }
  const int n = (int)ops.size(), gx = bgraph.grid_dim.x, gy = bgraph.grid_dim.y;
  auto pd = [&](int i, int d) { return ops[i]->output_tensors[0].dim[d]; };  // per-task (partitioned) dim
  auto pn = [&](int i) { return ops[i]->output_tensors[0].num_dims; };
  auto fd = [&](int i, int d) { return ops[i]->dtensor.dim[d]; };  // full tensor dim
  auto numel = [&](int i) {  // full tensor elements: activations may be declared 2-D ([16,128] attn_o) and read flat
    long long e = 1;
    for (int d = 0; d < ops[i]->dtensor.num_dims; d++) e *= ops[i]->dtensor.dim[d];
    return e;
  };
  auto is_f32 = [&](int i) { return ops[i]->dtensor.data_type == mirage::type::DT_FLOAT32; };
  auto reg = [&](TaskType t, int n_in, int n_out, std::string const &call) {
    assert(n == n_in + n_out);
    int variant = runtime::TaskRegister::get_instance()->register_task_variant(
        t, "  " + call + "(task_desc, runtime_config);");
    cfg = std::make_tuple(n_in, n_out, t, variant);
    return true;
  };
  auto gemv = [&](TaskType t, int n_in, char const *epi, std::string const &out_t, char const *fn, int wi) {
    // in[n_in-1] = weight tile [ROWS, K] (or [1, ROWS, K] for the K-slice-major split-K layout); out0 [1, ROWS] per tile,
    // [1, ROWS/2] for the fused silu (gate/up row pairs -> one output); x is read flat and only needs >= K elements;
    // n_in == 3: in1 = the RMSNorm weight of the NORM prologue (decision 11b), K elements
    const int rows = pd(wi, pn(wi) - 2), k = pd(wi, pn(wi) - 1);
    const int out_cols = std::string(epi) == "GEMV_SILU" ? rows / 2 : rows;
    DS_CHECK(rows % 4 == 0 && k % 8 == 0 && numel(0) >= k && pd(n - 1, pn(n - 1) - 1) == out_cols);
    DS_CHECK(wi == 1 || (numel(1) == k && !is_f32(1)));
    return reg(t, n_in, 1, cat({fn, "<", I(rows), ", dsk::", epi, ", ", out_t, ", ", I(k), ">"}));
  };
  const bool is_gemv = name == "gemv_bf16";
  // KV_B_PROJ tile rows: 64 measured, L4 32 / 16 (12e)
  const auto kvb_rows_ok = [&] { return pd(2, 1) == 512 && gx * pd(2, 0) == 4096 && 256 % pd(2, 0) == 0 && pd(2, 0) % 8 == 0; };
  if (is_gemv && n == 7) {  // KV_B_PROJ: [qkva, kv_a_layernorm, kv_b_proj tile, cos, sin] -> writes K, V
    DS_CHECK(params.empty() && kvb_rows_ok() && fd(0, 1) == 16 * 192 + 512 + 64);
    DS_CHECK(fd(1, 0) == 512 && fd(3, 1) == 64 && fd(4, 1) == 64 && fd(3, 0) >= fd(5, 1));
    DS_CHECK(fd(5, 0) == 16 && fd(5, 2) == 192 && fd(6, 0) == 16 && fd(6, 2) == 128 && fd(5, 1) == fd(6, 1));
    return reg(runtime::TASK_GEMV_BF16, 5, 2, cat({"dsk::ds_kv_b_proj<", I(fd(5, 1)), ", ", I(pd(2, 0)), ">"}));
  }
  if (is_gemv && n == 8) {  // KV_B_PROJ in the DAG (12a): qkva whole (the kernel's offsets) + the kva view, dependency only
    DS_CHECK(params.empty() && kvb_rows_ok() && fd(0, 1) == 16 * 192 + 512 + 64);  // tile rows 64, L4: 32 / 16 (12e)
    DS_CHECK(fd(1, 0) == 512 && fd(3, 1) == 64 && fd(4, 1) == 64 && fd(3, 0) >= fd(6, 1) && fd(5, 1) == 512 + 64);
    DS_CHECK(fd(6, 0) == 16 && fd(6, 2) == 192 && fd(7, 0) == 16 && fd(7, 2) == 128 && fd(6, 1) == fd(7, 1));
    return reg(runtime::TASK_GEMV_BF16, 6, 2, cat({"dsk::ds_kv_b_proj<", I(fd(6, 1)), ", ", I(pd(2, 0)), ">"}));
  }
  // QKV_PROJ, LM_HEAD: [x, norm weight, w tile] -> out
  if (is_gemv && n == 4 && pn(1) == 1) return gemv(runtime::TASK_GEMV_BF16, 3, "GEMV_NONE", is_f32(3) ? "float" : "dsk::bf16", "dsk::ds_gemv_norm", 2);
  if (name == "gemv_bf16_silu" && n == 4) return gemv(runtime::TASK_GEMV_BF16_SILU, 3, "GEMV_SILU", "dsk::bf16", "dsk::ds_gemv_norm", 2);
  if (name == "gemv_bf16_res" && n == 5) {  // O_PROJ_RESIDUAL + merge: [part_o, lse, o_proj tile, x tile] -> x_attn; params [CHUNK]
    DS_CHECK(params.size() == 1 && n == 5 && is_f32(0) && is_f32(1) && fd(0, 1) == 16 && fd(0, 2) == 128);
    DS_CHECK(fd(1, 0) == fd(0, 0) && fd(1, 1) == 16 && pd(2, 0) % 8 == 0 && pd(2, 1) == 2048 && !is_f32(3) &&
             !is_f32(4) && fd(3, 1) == fd(4, 1) && pd(4, pn(4) - 1) == pd(2, 0));  // o_proj tile rows 64, L4 32 / 16 (12e)
    return reg(runtime::TASK_GEMV_BF16_RES, 4, 1, "dsk::ds_o_proj_merge" + args({fd(0, 0), params[0], pd(2, 0)}));
  }
  if (name == "gemv_splitk") {
    DS_CHECK(n == 3 && pn(1) == 3 && is_f32(2) && gy == fd(1, 0) && fd(0, 1) == fd(1, 0) * fd(1, 2));
    return gemv(runtime::TASK_GEMV_SPLITK, 2, "GEMV_NONE", "float", "dsk::ds_gemv", 1);
  }
  if (name == "attn_chunk_192_128" && n == 7) {  // [qkva, K, V, cos, sin] -> [part_o, lse]; params [CHUNK]
    DS_CHECK(params.size() == 1);
    const int chunk = params[0], t_max = fd(1, 1), n_chunks = (t_max + chunk - 1) / chunk;
    // in0: qkva (chain) or its q view (DAG 12a: 16 x 192, the same pointer); the kernel reads q = in0 + head * 192
    DS_CHECK(gx == 16 && gy == n_chunks && (fd(0, 1) == 16 * 192 + 512 + 64 || fd(0, 1) == 16 * 192) && fd(1, 0) == 16 &&
             fd(1, 2) == 192);
    DS_CHECK(fd(2, 1) == t_max && fd(2, 2) == 128 && fd(3, 1) == 64 && fd(4, 1) == 64 && fd(3, 0) >= t_max);
    DS_CHECK(fd(5, 0) == n_chunks && fd(5, 1) == 16 && fd(5, 2) == 128 && fd(6, 0) == n_chunks && fd(6, 1) == 16 &&
             is_f32(5) && is_f32(6));
    return reg(runtime::TASK_ATTN_CHUNK_192_128, 5, 2, cat({"dsk::ds_attn_chunk<", I(t_max), ", ", I(chunk), ">"}));
  }
  if (name == "moe_gate_topk" && n == 7) {  // [x_attn, norm w, gate [E, K], xn2 (written)] -> [logits fp32 [1, E], w fp32, idx int32]
    DS_CHECK(params.empty() && gx == 1 && !is_f32(1) && !is_f32(2) && !is_f32(3) && is_f32(4) && is_f32(5) &&
             !is_f32(6));
    const int e = fd(2, 0), k = fd(2, 1), topk = fd(5, 0);
    DS_CHECK(numel(0) >= k && numel(1) == k && numel(3) == k && fd(4, 1) == e && fd(6, 0) == topk && e % 16 == 0 &&
             k % 8 == 0);
    return reg(runtime::TASK_MOE_TOPK, 4, 3, "dsk::ds_moe_gate_topk_norm" + args({e, topk, k}));  // L6 (12e)
  }
  if (name == "moe_topk" && n == 3) {  // L5 (12e): [logits fp32 [1, E]] -> [w fp32 [TOPK], idx int32 [TOPK]]
    DS_CHECK(params.empty() && gx == 1 && is_f32(0) && is_f32(1) && !is_f32(2) && fd(1, 0) == fd(2, 0) && fd(0, 1) <= 64);
    return reg(runtime::TASK_MOE_TOPK, 1, 2, "dsk::ds_moe_topk" + args({fd(0, 1), fd(1, 0)}));
  }
  // Slot-indexed expert ops (10a): grid (TOPK + 1, N-tiles); in0 = idx [TOPK] int32, read at run time by the dispatch
  if (name == "moe_expert_w13_silu") {  // [idx, x, w13 [E, N_R, K], w13_s [N_S, K]] -> h flat
    DS_CHECK(params.empty() && n == 5 && !is_f32(0));
    const int topk = fd(0, 0), e = fd(2, 0), n_r = fd(2, 1), k = fd(2, 2), rows_r = pd(2, 1), rows_s = pd(3, 0);
    DS_CHECK((gx == topk + 1 || gx == topk) && gy == n_r / rows_r && gy == fd(3, 0) / rows_s);  // topk: DAG routed slots (12a)
    DS_CHECK(fd(1, 1) == k && fd(3, 1) == k && fd(4, 0) == e * (n_r / 2) + fd(3, 0) / 2);
    return reg(runtime::TASK_MOE_EXPERT_W13_SILU, 4, 1, "dsk::ds_expert_w13" + args({e, n_r, k, rows_r, rows_s, topk}));
  }
  if (name == "moe_expert_w2") {  // [idx, h flat, w2 [E, N, K_R], w2_s [N, K_S]] -> e_out [E+1, N], column tile pre-offset
    DS_CHECK(params.empty() && n == 5 && !is_f32(0));
    const int topk = fd(0, 0), e = fd(2, 0), nn = fd(2, 1), k_r = fd(2, 2), k_s = fd(3, 1), rows = pd(2, 1);
    DS_CHECK((gx == topk + 1 || gx == topk) && gy == nn / rows && pd(3, 0) == rows);  // topk: DAG routed slots (12a)
    DS_CHECK(fd(1, 0) == e * k_r + k_s && fd(3, 0) == nn && fd(4, 0) == e + 1 && pd(4, 0) == e + 1 && fd(4, 1) == nn &&
             pd(4, 1) == rows);
    return reg(runtime::TASK_MOE_EXPERT_W2, 4, 1, "dsk::ds_expert_w2" + args({e, nn, k_r, k_s, rows, topk}));
  }
  if (name == "moe_combine") {  // [e_out [E+1, N], w, idx, residual] (DAG 12a: + expert_out[E] view, dependency only) -> x
    DS_CHECK(params.empty() && (n == 5 || n == 6) && gx == 1);
    const int e = fd(0, 0) - 1, topk = fd(1, 0), nn = fd(0, 1);
    DS_CHECK(is_f32(1) && fd(2, 0) == topk && fd(3, 1) == nn && fd(n - 1, 1) == nn);
    if (n == 5) return reg(runtime::TASK_MOE_COMBINE, 4, 1, "dsk::ds_moe_combine" + args({e, topk, nn}));
    DS_CHECK(fd(4, 0) == 1 && fd(4, 1) == nn && !is_f32(4));
    return reg(runtime::TASK_MOE_COMBINE, 5, 1, "dsk::ds_moe_combine" + args({e, topk, nn}));
  }
  // Legacy: pre-D11b graph shapes (one Mirage build served the sibling perf branches until the fold merged); strings
  // are the old ones, so they only compile against an old kernels/fleet_dispatch.cuh. Delete with the next rebuild.
  if (is_gemv && n == 4) {  // old KV_B_PROJ: [c_kvn, kv_b_proj tile] -> K, V
    DS_CHECK(params.empty() && gx == 64 && pd(1, 0) == 64 && pd(1, 1) == 512 && fd(0, 1) == 512);
    return reg(runtime::TASK_GEMV_BF16, 2, 2, cat({"dsk::ds_kv_b_proj<", I(fd(2, 1)), ">"}));
  }
  if (is_gemv && n == 3) return gemv(runtime::TASK_GEMV_BF16, 2, "GEMV_NONE", is_f32(2) ? "float" : "dsk::bf16", "dsk::ds_gemv", 1);
  if (name == "gemv_bf16_silu" && n == 3) return gemv(runtime::TASK_GEMV_BF16_SILU, 2, "GEMV_SILU", "dsk::bf16", "dsk::ds_gemv", 1);
  if (name == "gemv_bf16_res" && n == 4) return gemv(runtime::TASK_GEMV_BF16_RES, 3, "GEMV_RES", "dsk::bf16", "dsk::ds_gemv", 1);
  if (name == "mla_prep" && n == 7) return reg(runtime::TASK_MLA_PREP, 4, 3, cat({"dsk::ds_mla_prep<", I(fd(6, 1)), ">"}));
  if (name == "attn_chunk_192_128" && n == 5)
    return reg(runtime::TASK_ATTN_CHUNK_192_128, 3, 2, cat({"dsk::ds_attn_chunk<", I(fd(1, 1)), ", ", I(params[0]), ">"}));
  if (name == "attn_merge" && n == 3) return reg(runtime::TASK_ATTN_MERGE, 2, 1, cat({"dsk::ds_attn_merge<", I(params[0]), ">"}));
  if (name == "moe_gate_topk" && n == 5) return reg(runtime::TASK_MOE_TOPK, 2, 3, "dsk::ds_moe_gate_topk" + args({fd(1, 0), fd(3, 0), fd(1, 1)}));
  return false;
}

}  // namespace kernel
}  // namespace mirage
