// Equivalence check for MPK_ARGMAX_DUAL_REDUCE.
//
// The LM head's g-group argmax reduces a (value, index) pair across lanes with
// two `__shfl_xor` steps, xor-16 then xor-32. The ported sequence replaces
// them with `v_permlane16_swap_b32` and `v_permlane32_swap_b32` plus a
// compare-select, on the claim that those two instructions *are* those two
// steps and that the swap-with-a-self-copy idiom reproduces the shuffle's
// lane-uniform result.
//
// Neither half of that claim is obvious, and the failure mode is quiet: a
// wrong swap direction still returns a plausible maximum, just occasionally
// the wrong lane's. So drive both sequences over the same inputs and compare
// (value, index) bit for bit -- heavy on exact ties, which is where the strict
// `>` tie-break decides the index and where a reassociation would show.
//
// Build:
//   hipcc -O2 --offload-arch=gfx950 test_argmax_dual_reduce.hip -o /tmp/t_amdr
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

// The sequence being replaced, lifted verbatim from the argmax kernel.
__device__ __forceinline__ void butterfly(float &val, int &idx) {
#pragma unroll
  for (int offset = 16; offset <= 32; offset <<= 1) {
    float other_val = __shfl_xor(val, offset, 64);
    int other_idx = __shfl_xor(idx, offset, 64);
    if (other_val > val) {
      val = other_val;
      idx = other_idx;
    }
  }
}

// The replacement, lifted verbatim from argmax_dual_wave_reduce_to_g0.
__device__ __forceinline__ void dual_swap_to_g0(float &val, int &idx) {
  float val_peer = val;
  int idx_peer = idx;
  asm volatile("s_nop 1\n"
               "v_permlane16_swap_b32 %[val], %[val_peer]\n"
               "v_permlane16_swap_b32 %[idx], %[idx_peer]\n"
               "v_cmp_gt_f32 vcc, %[val_peer], %[val]\n"
               "v_cndmask_b32 %[idx], %[idx], %[idx_peer], vcc\n"
               "v_cndmask_b32 %[val], %[val], %[val_peer], vcc\n"
               "v_mov_b32_e32 %[val_peer], %[val]\n"
               "v_mov_b32_e32 %[idx_peer], %[idx]\n"
               "s_nop 1\n"
               "v_permlane32_swap_b32 %[val], %[val_peer]\n"
               "v_permlane32_swap_b32 %[idx], %[idx_peer]\n"
               "v_cmp_gt_f32 vcc, %[val_peer], %[val]\n"
               "v_cndmask_b32 %[idx], %[idx], %[idx_peer], vcc\n"
               "v_cndmask_b32 %[val], %[val], %[val_peer], vcc"
               : [val] "+v"(val),
                 [idx] "+v"(idx),
                 [val_peer] "+v"(val_peer),
                 [idx_peer] "+v"(idx_peer)
               :
               : "vcc");
}

// One wave per block; each block reduces one 64-lane case with both sequences
// and writes all 64 lanes of each result. All 64 are written, not just the
// contracted g0, so the checker can hold the *value* to every lane while
// holding the index to g0 only -- a swap in the wrong direction would break
// the value everywhere and that must not be silently skipped.
__global__ void run(float const *__restrict__ in_val,
                    int const *__restrict__ in_idx,
                    float *__restrict__ out_val_a,
                    int *__restrict__ out_idx_a,
                    float *__restrict__ out_val_b,
                    int *__restrict__ out_idx_b) {
  int const lane = threadIdx.x;
  int const base = blockIdx.x * 64;

  float va = in_val[base + lane];
  int ia = in_idx[base + lane];
  butterfly(va, ia);
  out_val_a[base + lane] = va;
  out_idx_a[base + lane] = ia;

  float vb = in_val[base + lane];
  int ib = in_idx[base + lane];
  dual_swap_to_g0(vb, ib);
  out_val_b[base + lane] = vb;
  out_idx_b[base + lane] = ib;
}

#define CHECK(x)                                                               \
  do {                                                                         \
    hipError_t _e = (x);                                                       \
    if (_e != hipSuccess) {                                                    \
      printf("FAIL: %s -> %s\n", #x, hipGetErrorString(_e));                   \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  // 4096 cases: a third random, a third drawn from a tiny value set so exact
  // ties are common, and a third all-equal so every comparison is a tie and
  // the index is decided purely by the tie-break.
  int const CASES = 4096;
  int const N = CASES * 64;
  size_t const FB = N * sizeof(float), IB = N * sizeof(int);

  float *h_val = (float *)malloc(FB);
  int *h_idx = (int *)malloc(IB);
  srand(12345);
  for (int c = 0; c < CASES; c++) {
    int mode = c % 3;
    for (int l = 0; l < 64; l++) {
      int i = c * 64 + l;
      if (mode == 0) {
        h_val[i] = (float)(rand() % 20001 - 10000) * 0.001f;
      } else if (mode == 1) {
        static float const few[4] = {-1.0f, 0.0f, 0.5f, 2.0f};
        h_val[i] = few[rand() & 3];
      } else {
        h_val[i] = 1.25f;
      }
      // Distinct indices so a wrong-lane result is always visible.
      h_idx[i] = i;
    }
  }

  float *d_val, *d_va, *d_vb;
  int *d_idx, *d_ia, *d_ib;
  CHECK(hipMalloc(&d_val, FB));
  CHECK(hipMalloc(&d_va, FB));
  CHECK(hipMalloc(&d_vb, FB));
  CHECK(hipMalloc(&d_idx, IB));
  CHECK(hipMalloc(&d_ia, IB));
  CHECK(hipMalloc(&d_ib, IB));
  CHECK(hipMemcpy(d_val, h_val, FB, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_idx, h_idx, IB, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(
      run, dim3(CASES), dim3(64), 0, 0, d_val, d_idx, d_va, d_ia, d_vb, d_ib);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());

  float *h_va = (float *)malloc(FB), *h_vb = (float *)malloc(FB);
  int *h_ia = (int *)malloc(IB), *h_ib = (int *)malloc(IB);
  CHECK(hipMemcpy(h_va, d_va, FB, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(h_vb, d_vb, FB, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(h_ia, d_ia, IB, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(h_ib, d_ib, IB, hipMemcpyDeviceToHost));

  // The contract is g == 0 -- lanes 0..15 -- because that is the only group
  // the kernel reads (`if (g == 0) s_red_val[...] = thread_max`). Outside it
  // the index is allowed to differ, and it does: the swap idiom leaves the
  // *upper* member of each pair holding its partner's value in `val` and its
  // own in `peer`, so the strict `>` keeps the wrong side of an exact tie
  // there. Lanes 0..15 are the lower member at both steps, so they are
  // unaffected -- which is exactly what this asserts rather than assumes.
  int bad_val = 0, bad_idx = 0, shown = 0;      // g == 0, must be zero
  int oth_val = 0, oth_idx = 0;                 // g != 0, informational
  for (int i = 0; i < N; i++) {
    bool bv = __builtin_bit_cast(unsigned, h_va[i]) !=
              __builtin_bit_cast(unsigned, h_vb[i]);
    bool bi = h_ia[i] != h_ib[i];
    if ((i % 64) >= 16) {
      oth_val += bv;
      oth_idx += bi;
      continue;
    }
    bad_val += bv;
    bad_idx += bi;
    if ((bv || bi) && shown < 8) {
      printf("  case %d lane %2d: butterfly (%g, %d)  swap (%g, %d)\n",
             i / 64,
             i % 64,
             h_va[i],
             h_ia[i],
             h_vb[i],
             h_ib[i]);
      shown++;
    }
  }

  // The value is the maximum over all four groups and is lane-uniform, so it
  // must match in *every* lane even though only g0's index is contracted.
  printf("%s: %d cases, g0 (lanes 0-15) value mismatches %d, index "
         "mismatches %d\n",
         (bad_val || bad_idx || oth_val) ? "FAIL" : "PASS",
         CASES,
         bad_val,
         bad_idx);
  printf("       g1-g3 (lanes 16-63): value mismatches %d (must be 0), "
         "index mismatches %d (uncontracted)\n",
         oth_val,
         oth_idx);
  return (bad_val || bad_idx || oth_val) ? 1 : 0;
}
