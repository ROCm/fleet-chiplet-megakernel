// C ABI over the task kernels for tests/test_kernels_cpu.py (ctypes). Build:
//   clang++ -x c++ -std=c++17 -O2 -fPIC -shared -DFLEET_CPU_SHIM -I../../../include/mirage/persistent_kernel/kernels -Itests tests/cpu_harness.cc -o <out>.so
// Every entry runs one emulated 256-thread workgroup (cpu_shim.h). Pointers follow the kernel contracts.
#include "argmax_partial.cuh"
#include "argmax_reduce.cuh"
#include "attn_chunk.cuh"
#include "attn_merge.cuh"
#include "embed.cuh"
#include "gemv_bf16.cuh"
#include "mla_prep.cuh"
#include "moe_combine.cuh"
#include "moe_topk.cuh"
#include "rmsnorm.cuh"
#include "splitk_reduce.cuh"
#include "switches.h"
#include "task_types.h"
#include <string>

using namespace dsk;
// real-model constants: 16 heads, 192/128 head dims, 512 latent, 64 rope, cache rows 1152
constexpr int H = 16, DN = 128, DR = 64, DC = 512, DV = 128, TMAX = 1152, CHUNK = 128, NCHUNK = TMAX / CHUNK;
constexpr int E = 64, TOPK = 6, HID = 2048, VOCAB_CHUNK = 1024, ARGMAX_PARTS = 100;

#define RUN(...) shim::run_block([&] { __VA_ARGS__; })
// epi: 0 none, 1 residual, 2 silu; out_fp32 only with epi 0 (split-K partial / gate logits). rows: 64, 128, 512 (lm_head, fp32 in the sim), 88/176/256 (silu, 11a).
// K is a template parameter of the kernel: every K the graph or the tests use is instantiated here.
template <int ROWS, int EPI, typename OutT>
static int gemv_k(const void *x, const void *w, void *out, const void *res, int K) {
#define GK(KK) case KK: RUN(gemv_bf16_task_impl<ROWS, EPI, OutT, KK>(x, w, out, res)); return 0;
  switch (K) { GK(512) GK(1408) GK(2048) GK(2736) GK(2816) GK(3648) GK(10944) default: return -1; }
#undef GK
}
extern "C" {
int k_gemv(const void *x, const void *w, void *out, const void *res, int K, int rows, int epi, int out_fp32) {
  switch (rows * 100 + epi * 10 + out_fp32) {
    case 6400: return gemv_k<64, GEMV_NONE, bf16>(x, w, out, res, K);
    case 401: return gemv_k<4, GEMV_NONE, float>(x, w, out, res, K);      // L5 gate tiles (16 x 4 rows)
    case 1600: return gemv_k<16, GEMV_NONE, bf16>(x, w, out, res, K);     // L4 tile rows
    case 3200: return gemv_k<32, GEMV_NONE, bf16>(x, w, out, res, K);
    case 6401: return gemv_k<64, GEMV_NONE, float>(x, w, out, res, K);
    case 6410: return gemv_k<64, GEMV_RES, bf16>(x, w, out, res, K);
    case 6420: return gemv_k<64, GEMV_SILU, bf16>(x, w, out, res, K);
    case 51201: return gemv_k<512, GEMV_NONE, float>(x, w, out, res, K);
    case 12800: return gemv_k<128, GEMV_NONE, bf16>(x, w, out, res, K);
    case 12820: return gemv_k<128, GEMV_SILU, bf16>(x, w, out, res, K);
    case 8820: return gemv_k<88, GEMV_SILU, bf16>(x, w, out, res, K);    // expert tile sweep (11a); L2 shared w13 (12e)
    case 3210: return gemv_k<32, GEMV_RES, bf16>(x, w, out, res, K);     // L4 O_PROJ
    case 1610: return gemv_k<16, GEMV_RES, bf16>(x, w, out, res, K);
    case 801: return gemv_k<8, GEMV_NONE, float>(x, w, out, res, K);     // L5 gate GEMV tiles (fp32 logits)
    case 1601: return gemv_k<16, GEMV_NONE, float>(x, w, out, res, K);
    case 3201: return gemv_k<32, GEMV_NONE, float>(x, w, out, res, K);
    case 17620: return gemv_k<176, GEMV_SILU, bf16>(x, w, out, res, K);
    case 25620: return gemv_k<256, GEMV_SILU, bf16>(x, w, out, res, K);
    default: return -1;
  }
}
// decision 12e L6 folds (perf/fold-full 9379248): NORM prologue gemv (epi/rows as k_gemv; K 512 or 2048; 176-row silu = the
// shared W13 tile of the DAG), O_PROJ_RESIDUAL with the merge prologue, gate + top-k with the norm2 prologue
int k_gemv_norm(const void *x, const void *nw, const void *w, void *out, const void *res, int K, int rows, int epi, int out_fp32, float eps) {
#define GN(R, EP, OT) \
  if (K == 2048) { RUN(gemv_bf16_norm_task_impl<R, EP, OT, 2048>(x, nw, w, out, res, eps)); return 0; } \
  if (K == 512) { RUN(gemv_bf16_norm_task_impl<R, EP, OT, 512>(x, nw, w, out, res, eps)); return 0; } \
  return -1;
  switch (rows * 100 + epi * 10 + out_fp32) {
    case 1600: { GN(16, GEMV_NONE, bf16) }
    case 3200: { GN(32, GEMV_NONE, bf16) }
    case 6400: { GN(64, GEMV_NONE, bf16) }
    case 6420: { GN(64, GEMV_SILU, bf16) }
    case 17620: { GN(176, GEMV_SILU, bf16) }
    case 51201: { GN(512, GEMV_NONE, float) }
    default: return -1;
  }
#undef GN
}
void k_o_proj_merge(const void *o_part, const void *lse, const void *w, void *out, const void *res, int T) {
  RUN(o_proj_merge_task_impl<H, NCHUNK, CHUNK, DV, 64>(o_part, lse, w, out, res, T));
}
int k_o_proj_merge_rows(const void *o_part, const void *lse, const void *w, void *out, const void *res, int T, int rows) {  // L4 x L6
  switch (rows) {
    case 16: RUN(o_proj_merge_task_impl<H, NCHUNK, CHUNK, DV, 16>(o_part, lse, w, out, res, T)); return 0;
    case 32: RUN(o_proj_merge_task_impl<H, NCHUNK, CHUNK, DV, 32>(o_part, lse, w, out, res, T)); return 0;
    case 64: RUN(o_proj_merge_task_impl<H, NCHUNK, CHUNK, DV, 64>(o_part, lse, w, out, res, T)); return 0;
    default: return -1;
  }
}
void k_moe_gate_topk_norm(const void *x, const void *nw, const void *gate, void *logits, void *w, void *idx, void *xn, float eps) {
  RUN(moe_gate_topk_task_impl<E, TOPK, HID>(x, nw, gate, logits, w, idx, xn, eps));
}
// decision 11b MLA_PREP fold: KV_B_PROJ tile (kv_a norm prologue + k_pe from tile 4h), ATTN_CHUNK with the q RoPE prologue
void k_kv_b_proj(const void *qkva, const void *nw, const void *w, const void *cs, const void *sn, void *out, void *kc, int n, int slot, int pos, float eps) {
  RUN(kv_b_proj_task_impl<H, DN, DR, DC, TMAX>(qkva, nw, w, cs, sn, out, kc, n, slot, pos, eps));
}
int k_kv_b_proj_rows(const void *qkva, const void *nw, const void *w, const void *cs, const void *sn, void *out, void *kc, int n, int slot,
                     int pos, float eps, int rows) {  // L4 (12e): tile rows 64 / 32 / 16
  switch (rows) {
    case 64: RUN(kv_b_proj_task_impl<H, DN, DR, DC, TMAX, 64>(qkva, nw, w, cs, sn, out, kc, n, slot, pos, eps)); return 0;
    case 32: RUN(kv_b_proj_task_impl<H, DN, DR, DC, TMAX, 32>(qkva, nw, w, cs, sn, out, kc, n, slot, pos, eps)); return 0;
    case 16: RUN(kv_b_proj_task_impl<H, DN, DR, DC, TMAX, 16>(qkva, nw, w, cs, sn, out, kc, n, slot, pos, eps)); return 0;
    default: return -1;
  }
}
void k_moe_topk(const void *logits, void *w, void *idx) { RUN(moe_topk_task_impl<E, TOPK>(logits, w, idx)); }  // L5 (12e)
void k_attn_chunk_rope(const void *qkva, const void *kc, const void *vc, void *o_part, void *lse, int head, int chunk, int T, float scale,
                       const void *cs, const void *sn, int pos) {
  RUN(attn_chunk_task_impl<H, TMAX, CHUNK, DN + DR, DV, true>(qkva, kc, vc, o_part, lse, head, chunk, T, scale, cs, sn, pos));
}
void k_splitk_reduce(const void *ws, const void *res, void *out, int N) { RUN(splitk_reduce_task_impl<4>(ws, res, out, N)); }
void k_rmsnorm(const void *x, const void *w, void *out, int N, float eps) { RUN(rmsnorm_task_impl(x, w, out, N, eps)); }
void k_mla_prep(const void *qkva, const void *nw, const void *cs, const void *sn, void *ckvn, void *q_rope, void *kc, int slot, int pos, float eps) {
  RUN(mla_prep_task_impl<H, DN, DR, DC, TMAX>(qkva, nw, cs, sn, ckvn, q_rope, kc, slot, pos, eps));
}
void k_attn_chunk(const void *q, const void *kc, const void *vc, void *o_part, void *lse, int head, int chunk, int T, float scale) {
  RUN(attn_chunk_task_impl<H, TMAX, CHUNK, DN + DR, DV>(q, kc, vc, o_part, lse, head, chunk, T, scale));
}
void k_attn_merge(const void *o_part, const void *lse, void *out, int head) { RUN(attn_merge_task_impl<H, NCHUNK, DV>(o_part, lse, out, head)); }
// gate GEMV + top-k in one task (decision 10c). Expert tiles are plain k_gemv calls with slot-derived pointers (10a).
void k_moe_gate_topk(const void *x, const void *gate, void *logits, void *w, void *idx) {
  RUN(moe_gate_topk_task_impl<E, TOPK, HID>(x, gate, logits, w, idx));
}
void k_moe_combine(const void *e_out, const void *w, const void *idx, const void *res, void *out) {
  RUN(moe_combine_task_impl<E, TOPK, HID>(e_out, w, idx, res, out));
}
void k_embed(const void *table, const long long *tokens, int step, void *out) { RUN(embed_task_impl<HID>(table, tokens, step, out)); }
void k_argmax_partial(const void *logits, void *val, void *idx) { RUN(argmax_partial_task_impl<VOCAB_CHUNK>(logits, val, idx)); }
void k_argmax_reduce(const void *val, const void *idx, void *tok) { RUN(argmax_reduce_task_impl<VOCAB_CHUNK, ARGMAX_PARTS>(val, idx, tok)); }
// the switch values this .so was built with (tests/test_kernels_cpu.py checks its -D flags arrived)
#define SW(n) + " " #n "=" + std::to_string(n)
const char *k_switches() {
  static const std::string s = std::string("") SW(DSK_COMBINE_BLOCK) SW(DSK_NORM_FIXED) SW(DSK_MERGE_FIXED) SW(DSK_ATTN_SCORE_HOIST) SW(DSK_GEMV_RPW);
  return s.c_str();
}
#undef SW
int k_task_type_count() { return TASK_RMSNORM - TASK_GEMV_BF16 + 1; }
}
