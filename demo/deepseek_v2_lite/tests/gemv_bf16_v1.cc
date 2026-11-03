// The pre-vectorisation gemv_bf16.cuh (commit be9a07d) verbatim + a C ABI, built by tests/test_kernels_cpu.py into its
// own .so so the vectorised kernel can be asserted torch.equal against it at every K and epilogue (
// same per-lane fp32 fma order per row => bit-identical). Not part of the product.
// out[r] = sum_k W[r,k] * x[k] for the ROWS rows of one tile, fp32 accumulate. Any K that is a multiple
// of 8 (512/1408/2048/2816/3648/10944 all are). Pointers are PRE-OFFSET to the tile by the caller:
// w = first row of the tile, out/res = the tile's first output (so scattered outputs such as the kv_b_proj
// rows into the K/V cache are just a different `out`). Split-K = this kernel with OutT=float and
// (x, w, K) pointing at the K-slice; splitk_reduce.cuh sums the partials.
// Epilogues mirror HF's bf16 op order: NONE -> bf16(acc); RES -> bf16(res + bf16(acc));
// SILU -> tile rows [0,ROWS/2) are gate rows, [ROWS/2,ROWS) their matching up rows (design.md 4.1),
//         h[j] = bf16(bf16(silu(bf16(g))) * bf16(u)) for j < ROWS/2.
// LDS: x staged as bf16 (2K bytes, 21.9 KB at K=10944) + ROWS fp32 partials.
#pragma once
#include "common.cuh"

namespace dsk {
enum GemvEpi { GEMV_NONE = 0, GEMV_RES = 1, GEMV_SILU = 2 };

template <int ROWS, int EPI, typename OutT>
__device__ __forceinline__ void gemv_bf16_task_impl(const void *x_ptr, const void *w_ptr, void *out_ptr,
                                                    const void *res_ptr, int K) {
  static_assert(ROWS % 4 == 0 && (EPI != GEMV_SILU || ROWS % 8 == 0), "rows split over 4 waves");
  extern __shared__ char smem[];
  const bf16 *x = static_cast<const bf16 *>(x_ptr);
  const bf16 *w = static_cast<const bf16 *>(w_ptr);
  const bf16 *res = static_cast<const bf16 *>(res_ptr);
  OutT *out = static_cast<OutT *>(out_ptr);
  bf16 *xs = reinterpret_cast<bf16 *>(smem);
  float *acc = reinterpret_cast<float *>(smem + ((K * 2 + 15) & ~15));

  for (int i = threadIdx.x * 8; i < K; i += WG_THREADS * 8) store8(xs + i, load8(x + i));
  __syncthreads();

  const int lane = threadIdx.x & (WAVE_SIZE - 1), wave = threadIdx.x / WAVE_SIZE;
  // One row per wave at a time (dependent fma chain, one 16B load in flight per lane);
  // vectorise to 2-4 rows per wave with independent accumulators once correctness is proven on GPU.
  for (int r = wave; r < ROWS; r += WG_THREADS / WAVE_SIZE) {
    const bf16 *wr = w + (size_t)r * K;
    float s = 0.f;
    for (int k = lane * 8; k < K; k += WAVE_SIZE * 8) {
      bf16x8 a = load8(wr + k), b = load8(xs + k);
#pragma unroll
      for (int j = 0; j < 8; j++) s = fmaf(bf2f(a.v[j]), bf2f(b.v[j]), s);
    }
    s = wave_sum(s);
    if (lane == 0) acc[r] = s;
  }
  __syncthreads();

  if constexpr (EPI == GEMV_SILU) {
    for (int j = threadIdx.x; j < ROWS / 2; j += WG_THREADS) {
      float g = bfround(acc[j]), u = bfround(acc[j + ROWS / 2]);
      out[j] = to_out<OutT>(bfround(silu(g)) * u);
    }
  } else {
    for (int r = threadIdx.x; r < ROWS; r += WG_THREADS) {
      float v = acc[r];
      if constexpr (EPI == GEMV_RES) v = bf2f(res[r]) + bfround(v);
      out[r] = to_out<OutT>(v);
    }
  }
}
}  // namespace dsk

#define RUN(...) shim::run_block([&] { __VA_ARGS__; })
using namespace dsk;
extern "C" int k_gemv(const void *x, const void *w, void *out, const void *res, int K, int rows, int epi, int out_fp32) {
  switch (rows * 100 + epi * 10 + out_fp32) {
    case 6400: RUN(gemv_bf16_task_impl<64, GEMV_NONE, bf16>(x, w, out, res, K)); return 0;
    case 6401: RUN(gemv_bf16_task_impl<64, GEMV_NONE, float>(x, w, out, res, K)); return 0;
    case 6410: RUN(gemv_bf16_task_impl<64, GEMV_RES, bf16>(x, w, out, res, K)); return 0;
    case 6420: RUN(gemv_bf16_task_impl<64, GEMV_SILU, bf16>(x, w, out, res, K)); return 0;
    case 12820: RUN(gemv_bf16_task_impl<128, GEMV_SILU, bf16>(x, w, out, res, K)); return 0;
    case 8820: RUN(gemv_bf16_task_impl<88, GEMV_SILU, bf16>(x, w, out, res, K)); return 0;
    default: return -1;
  }
}
