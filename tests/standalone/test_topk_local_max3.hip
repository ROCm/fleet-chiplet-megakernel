// Equivalence check for MPK_TOPK_LOCAL_MAX3.
//
// The ported sequence replaces TopK's per-lane serial argmax with a
// v_max3_f32 reduction tree plus an equality-recovered index. Bit-exactness
// is not obvious: the serial scan uses strict `>` (lowest column wins a tie)
// while the tree recovers the index from v_cmp_eq, so the two only agree
// because the equality matches are visited in reverse. This drives both
// sequences over the same inputs -- heavy on exact ties, which is the case
// that decides it -- and compares (value, expert) bit for bit.
//
// Build:
//   hipcc -O2 --offload-arch=gfx950 test_topk_local_max3.hip -o /tmp/t_max3
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__device__ __forceinline__ void serial_argmax(float const *r,
                                              int const *c,
                                              float &mv,
                                              int &ex) {
  asm volatile("v_mov_b32 %[mv], %[r0]\n"
               "v_mov_b32 %[ex], %[c0]\n"
               "v_cmp_gt_f32 vcc, %[r1], %[mv]\n"
               "v_cndmask_b32 %[mv], %[mv], %[r1], vcc\n"
               "v_cndmask_b32 %[ex], %[ex], %[c1], vcc\n"
               "v_cmp_gt_f32 vcc, %[r2], %[mv]\n"
               "v_cndmask_b32 %[mv], %[mv], %[r2], vcc\n"
               "v_cndmask_b32 %[ex], %[ex], %[c2], vcc\n"
               "v_cmp_gt_f32 vcc, %[r3], %[mv]\n"
               "v_cndmask_b32 %[mv], %[mv], %[r3], vcc\n"
               "v_cndmask_b32 %[ex], %[ex], %[c3], vcc\n"
               "v_cmp_gt_f32 vcc, %[r4], %[mv]\n"
               "v_cndmask_b32 %[mv], %[mv], %[r4], vcc\n"
               "v_cndmask_b32 %[ex], %[ex], %[c4], vcc\n"
               "v_cmp_gt_f32 vcc, %[r5], %[mv]\n"
               "v_cndmask_b32 %[mv], %[mv], %[r5], vcc\n"
               "v_cndmask_b32 %[ex], %[ex], %[c5], vcc\n"
               "v_cmp_gt_f32 vcc, %[r6], %[mv]\n"
               "v_cndmask_b32 %[mv], %[mv], %[r6], vcc\n"
               "v_cndmask_b32 %[ex], %[ex], %[c6], vcc\n"
               "v_cmp_gt_f32 vcc, %[r7], %[mv]\n"
               "v_cndmask_b32 %[mv], %[mv], %[r7], vcc\n"
               "v_cndmask_b32 %[ex], %[ex], %[c7], vcc\n"
               : [mv] "=&v"(mv), [ex] "=&v"(ex)
               : [r0] "v"(r[0]), [r1] "v"(r[1]), [r2] "v"(r[2]),
                 [r3] "v"(r[3]), [r4] "v"(r[4]), [r5] "v"(r[5]),
                 [r6] "v"(r[6]), [r7] "v"(r[7]), [c0] "v"(c[0]),
                 [c1] "v"(c[1]), [c2] "v"(c[2]), [c3] "v"(c[3]),
                 [c4] "v"(c[4]), [c5] "v"(c[5]), [c6] "v"(c[6]),
                 [c7] "v"(c[7])
               : "vcc");
}

__device__ __forceinline__ void max3_argmax(float const *r,
                                            int const *c,
                                            float &mv,
                                            int &ex) {
  float m0, m1;
  asm volatile("v_max3_f32 %[m0], %[r0], %[r1], %[r2]\n"
               "v_max3_f32 %[m1], %[r3], %[r4], %[r5]\n"
               "v_max3_f32 %[mv], %[m0], %[m1], %[r6]\n"
               "v_max_f32 %[mv], %[mv], %[r7]\n"
               "v_mov_b32 %[ex], %[c7]\n"
               "v_cmp_eq_f32 vcc, %[r6], %[mv]\n"
               "v_cndmask_b32 %[ex], %[ex], %[c6], vcc\n"
               "v_cmp_eq_f32 vcc, %[r5], %[mv]\n"
               "v_cndmask_b32 %[ex], %[ex], %[c5], vcc\n"
               "v_cmp_eq_f32 vcc, %[r4], %[mv]\n"
               "v_cndmask_b32 %[ex], %[ex], %[c4], vcc\n"
               "v_cmp_eq_f32 vcc, %[r3], %[mv]\n"
               "v_cndmask_b32 %[ex], %[ex], %[c3], vcc\n"
               "v_cmp_eq_f32 vcc, %[r2], %[mv]\n"
               "v_cndmask_b32 %[ex], %[ex], %[c2], vcc\n"
               "v_cmp_eq_f32 vcc, %[r1], %[mv]\n"
               "v_cndmask_b32 %[ex], %[ex], %[c1], vcc\n"
               "v_cmp_eq_f32 vcc, %[r0], %[mv]\n"
               "v_cndmask_b32 %[ex], %[ex], %[c0], vcc\n"
               : [mv] "=&v"(mv), [ex] "=&v"(ex), [m0] "=&v"(m0), [m1] "=&v"(m1)
               : [r0] "v"(r[0]), [r1] "v"(r[1]), [r2] "v"(r[2]),
                 [r3] "v"(r[3]), [r4] "v"(r[4]), [r5] "v"(r[5]),
                 [r6] "v"(r[6]), [r7] "v"(r[7]), [c0] "v"(c[0]),
                 [c1] "v"(c[1]), [c2] "v"(c[2]), [c3] "v"(c[3]),
                 [c4] "v"(c[4]), [c5] "v"(c[5]), [c6] "v"(c[6]),
                 [c7] "v"(c[7])
               : "vcc");
}

// Mirrors the caller: k rounds of (argmax, blank the winner with -10000).
__global__ void compare(float const *__restrict__ inputs,
                        int n_cases,
                        int k,
                        unsigned *__restrict__ mismatches) {
  int case_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (case_id >= n_cases) {
    return;
  }
  float a[8], b[8];
  int col[8];
  int const start_col = (case_id % 16) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    a[i] = inputs[case_id * 8 + i];
    b[i] = a[i];
    col[i] = start_col + i;
  }

  for (int round = 0; round < k; ++round) {
    float va, vb;
    int ea, eb;
    serial_argmax(a, col, va, ea);
    max3_argmax(b, col, vb, eb);
    if (__float_as_uint(va) != __float_as_uint(vb) || ea != eb) {
      atomicAdd(mismatches, 1u);
    }
    float const neg = -10000.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (col[i] == ea) {
        a[i] = neg;
      }
      if (col[i] == eb) {
        b[i] = neg;
      }
    }
  }
}

int main() {
  int const n_cases = 1 << 20;
  int const k = 4;
  float *h = (float *)malloc(sizeof(float) * 8 * n_cases);

  // A quarter of the cases are drawn from a 3-value alphabet so exact ties are
  // the common case, not a rarity; the rest are BF16-rounded uniforms, which
  // is what the router actually produces.
  srand(1234);
  for (int i = 0; i < n_cases; ++i) {
    bool tie_heavy = (i % 4) == 0;
    for (int j = 0; j < 8; ++j) {
      float v;
      if (tie_heavy) {
        float const alphabet[3] = {0.25f, 0.5f, -10000.f};
        v = alphabet[rand() % 3];
      } else {
        v = (float)rand() / (float)RAND_MAX;
        unsigned u = __builtin_bit_cast(unsigned, v) & 0xffff0000u;
        v = __builtin_bit_cast(float, u);
      }
      h[i * 8 + j] = v;
    }
  }

  float *d_in;
  unsigned *d_mm;
  hipMalloc(&d_in, sizeof(float) * 8 * n_cases);
  hipMalloc(&d_mm, sizeof(unsigned));
  hipMemcpy(d_in, h, sizeof(float) * 8 * n_cases, hipMemcpyHostToDevice);
  hipMemset(d_mm, 0, sizeof(unsigned));

  compare<<<(n_cases + 255) / 256, 256>>>(d_in, n_cases, k, d_mm);
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) {
    printf("FAIL: %s\n", hipGetErrorString(e));
    return 1;
  }

  unsigned mm = 0;
  hipMemcpy(&mm, d_mm, sizeof(unsigned), hipMemcpyDeviceToHost);
  printf("cases: %d x %d rounds = %d comparisons\nmismatches: %u\n%s\n",
         n_cases, k, n_cases * k, mm, mm == 0 ? "PASS" : "FAIL");
  return mm == 0 ? 0 : 1;
}
