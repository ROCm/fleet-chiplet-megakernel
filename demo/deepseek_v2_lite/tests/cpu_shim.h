// Minimal HIP surface so kernels/*.cuh compile with plain `clang++ -x c++ -std=c++17 -DFLEET_CPU_SHIM`.
// One 256-thread workgroup is emulated as 256 ucontext fibers on ONE OS thread: __syncthreads() and wave
// shuffles yield to a round-robin scheduler, so kernels run unchanged and deterministically.
// Limits: one workgroup at a time; no atomics/fences/nontemporal
// semantics; a thread that returns early just leaves the barrier set (our kernels only skip uniformly);
// blockIdx is always 0; no real LDS size enforcement beyond SHIM_SMEM_BYTES.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <ucontext.h>

#define __device__
#define __host__
#define __global__
#define __forceinline__ inline
#define __shared__            // only dynamic LDS (`extern __shared__ char smem[]`) is supported
#define __launch_bounds__(...)

struct shim_dim3 { unsigned x, y, z; };
inline shim_dim3 threadIdx{0, 0, 0}, blockIdx{0, 0, 0}, blockDim{256, 1, 1};

namespace shim {
constexpr int NT = 256, WAVE = 64, NWAVE = NT / WAVE;
constexpr size_t STACK = 256 * 1024;
struct Rt {
  ucontext_t main, ctx[NT];
  char *stack[NT] = {};
  bool done[NT];
  int cur = 0, alive = 0, bar_n = 0, bar_gen = 0;
  int wbar_n[NWAVE], wbar_gen[NWAVE], walive[NWAVE];
  uint64_t xchg[2][NT];  // double-buffered by barrier-generation parity
  std::function<void()> body;
};
inline Rt rt;
inline int lane() { return rt.cur & (WAVE - 1); }
inline int wave() { return rt.cur / WAVE; }
inline void yield() { swapcontext(&rt.ctx[rt.cur], &rt.main); }
inline void release_if_complete(int &n, int &gen, int alive) {
  if (n > 0 && n >= alive) { n = 0; gen++; }
}
inline void block_bar() {
  int g = rt.bar_gen;
  rt.bar_n++;
  release_if_complete(rt.bar_n, rt.bar_gen, rt.alive);
  while (rt.bar_gen == g) yield();
}
inline void wave_bar() {
  int w = wave(), g = rt.wbar_gen[w];
  rt.wbar_n[w]++;
  release_if_complete(rt.wbar_n[w], rt.wbar_gen[w], rt.walive[w]);
  while (rt.wbar_gen[w] == g) yield();
}
// value of `v` held by lane `src` of my wave (all lanes of the wave must call this in lockstep)
template <class T> inline T wave_get(T v, int src) {
  int w = wave(), par = rt.wbar_gen[w] & 1;
  uint64_t u = 0; std::memcpy(&u, &v, sizeof v);
  rt.xchg[par][rt.cur] = u;
  wave_bar();
  u = rt.xchg[par][w * WAVE + src];
  std::memcpy(&v, &u, sizeof v);
  return v;
}
// serial-loop wave reduction: every lane publishes, one barrier, every lane folds the 64 slots
template <class T, class Op> inline T wave_reduce(T v, Op op) {
  int w = wave(), par = rt.wbar_gen[w] & 1;
  uint64_t u = 0; std::memcpy(&u, &v, sizeof v);
  rt.xchg[par][rt.cur] = u;
  wave_bar();
  T acc; std::memcpy(&acc, &rt.xchg[par][w * WAVE], sizeof acc);
  for (int i = 1; i < WAVE; i++) { T o; std::memcpy(&o, &rt.xchg[par][w * WAVE + i], sizeof o); acc = op(acc, o); }
  return acc;
}
inline void entry() {
  rt.body();
  rt.done[rt.cur] = true;
  rt.alive--; rt.walive[wave()]--;
  release_if_complete(rt.bar_n, rt.bar_gen, rt.alive);
  release_if_complete(rt.wbar_n[wave()], rt.wbar_gen[wave()], rt.walive[wave()]);
}
// Run `body` as one 256-thread workgroup.
inline void run_block(std::function<void()> body) {
  rt.body = std::move(body);
  rt.alive = NT; rt.bar_n = rt.bar_gen = 0;
  for (int w = 0; w < NWAVE; w++) { rt.wbar_n[w] = rt.wbar_gen[w] = 0; rt.walive[w] = WAVE; }
  for (int i = 0; i < NT; i++) {
    if (!rt.stack[i]) rt.stack[i] = (char *)std::malloc(STACK);
    getcontext(&rt.ctx[i]);
    rt.ctx[i].uc_stack.ss_sp = rt.stack[i];
    rt.ctx[i].uc_stack.ss_size = STACK;
    rt.ctx[i].uc_link = &rt.main;
    makecontext(&rt.ctx[i], entry, 0);
    rt.done[i] = false;
  }
  while (rt.alive > 0)
    for (int i = 0; i < NT; i++)
      if (!rt.done[i]) { rt.cur = i; threadIdx.x = i; swapcontext(&rt.main, &rt.ctx[i]); }
}
}  // namespace shim

inline void __syncthreads() { shim::block_bar(); }
template <class T> inline T __shfl_xor(T v, int mask, int width = 64) { (void)width; return shim::wave_get(v, shim::lane() ^ mask); }
template <class T> inline T __shfl_down(T v, int d, int width = 64) { int l = shim::lane(); return shim::wave_get(v, l + d < width ? l + d : l); }
template <class T> inline T __shfl(T v, int src, int width = 64) { (void)width; return shim::wave_get(v, src); }
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
inline float __fdividef(float a, float b) { return a / b; }
inline float __expf(float x) { return std::exp(x); }
