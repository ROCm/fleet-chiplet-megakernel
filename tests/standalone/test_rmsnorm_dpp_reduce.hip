// Equivalence check for MPK_RMSNORM_DPP_REDUCE.
//
// The RMSNorm prologues reduce a per-lane sum of squares with a six-step
// `__shfl_xor` butterfly (offsets 32/16/8/4/2/1) and then read lane 0 only.
// Every one of those steps has a VALU-native equivalent on gfx950 --
// permlane32_swap for xor-32, permlane16_swap for xor-16, and DPP row_shl for
// the four intra-row steps -- so the whole chain can leave the LDS data path.
//
// Two things need checking, and neither is safe to assume:
//
//   1. The lane algebra. row_shl accumulates toward lane 0 rather than leaving
//      the total in every lane, so the claim is only ever about lane 0. The
//      direction of the shift is exactly the kind of thing that is easy to get
//      backwards and still produce a plausible-looking number.
//   2. Bit-identity. This is a float sum, so it is only bit-identical if the
//      association tree is the same one the butterfly walked. A reassociation
//      would pass any "close enough" check and silently move the norm.
//
// Two candidate encodings are compared against the butterfly. Both use
// `s_nop 1` ahead of each stage to cover the two wait states Table 11 of the
// CDNA4 ISA requires between a VALU write and a DPP or PERMLANE read of that
// register. They differ in the four row_shl stages:
//
//   movadd -- `v_mov_b32_dpp` into a peer, then `v_add_f32`. Mirrors the
//             existing interleaved two-partial chain.
//   fused  -- one `v_add_f32` taking the DPP swizzle on src0. Four fewer
//             instructions; nothing consumes the peer separately, so the
//             move is pure overhead in the single-partial case.
//
// Build:
//   hipcc -O2 --offload-arch=gfx950 test_rmsnorm_dpp_reduce.hip -o /tmp/t_rdr
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

// The sequence being replaced, lifted verbatim from the RMSNorm prologues.
__device__ __forceinline__ void butterfly(float &ssq) {
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    ssq += __shfl_xor(ssq, offset);
  }
}

// Candidate A: peer move + add, matching the shape of the two-partial chain.
__device__ __forceinline__ void dpp_movadd(float &ssq) {
  float peer = ssq;
  asm volatile("s_nop 1\n"
               "v_permlane32_swap_b32 %[s], %[p]\n"
               "v_add_f32 %[s], %[s], %[p]\n"
               "v_mov_b32_e32 %[p], %[s]\n"
               "s_nop 1\n"
               "v_permlane16_swap_b32 %[s], %[p]\n"
               "v_add_f32 %[s], %[s], %[p]\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %[p], %[s] row_shl:8 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32 %[s], %[s], %[p]\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %[p], %[s] row_shl:4 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32 %[s], %[s], %[p]\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %[p], %[s] row_shl:2 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32 %[s], %[s], %[p]\n"
               "s_nop 1\n"
               "v_mov_b32_dpp %[p], %[s] row_shl:1 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "v_add_f32 %[s], %[s], %[p]"
               : [s] "+v"(ssq), [p] "+v"(peer));
}

// Candidate B: the row_shl stages fold the move into the add.
__device__ __forceinline__ void dpp_fused(float &ssq) {
  float peer = ssq;
  asm volatile("s_nop 1\n"
               "v_permlane32_swap_b32 %[s], %[p]\n"
               "v_add_f32 %[s], %[s], %[p]\n"
               "v_mov_b32_e32 %[p], %[s]\n"
               "s_nop 1\n"
               "v_permlane16_swap_b32 %[s], %[p]\n"
               "v_add_f32 %[s], %[s], %[p]\n"
               "s_nop 1\n"
               "v_add_f32_dpp %[s], %[s], %[s] row_shl:8 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "s_nop 1\n"
               "v_add_f32_dpp %[s], %[s], %[s] row_shl:4 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "s_nop 1\n"
               "v_add_f32_dpp %[s], %[s], %[s] row_shl:2 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1\n"
               "s_nop 1\n"
               "v_add_f32_dpp %[s], %[s], %[s] row_shl:1 row_mask:0xf "
               "bank_mask:0xf bound_ctrl:1"
               : [s] "+v"(ssq), [p] "+v"(peer));
}

__global__ void run(float const *__restrict__ in,
                    float *__restrict__ out_a,
                    float *__restrict__ out_b,
                    float *__restrict__ out_c) {
  int const lane = threadIdx.x;
  int const base = blockIdx.x * 64;

  float a = in[base + lane];
  butterfly(a);
  out_a[base + lane] = a;

  float b = in[base + lane];
  dpp_movadd(b);
  out_b[base + lane] = b;

  float c = in[base + lane];
  dpp_fused(c);
  out_c[base + lane] = c;
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
  // Sums of squares are non-negative and span a wide dynamic range, so the
  // cases mix magnitudes deliberately: reassociation only shows up when the
  // addends differ enough for the rounding to depend on the order.
  int const CASES = 8192;
  int const N = CASES * 64;
  size_t const FB = N * sizeof(float);

  float *h_in = (float *)malloc(FB);
  srand(9876);
  for (int c = 0; c < CASES; c++) {
    int mode = c % 4;
    for (int l = 0; l < 64; l++) {
      int i = c * 64 + l;
      float v;
      if (mode == 0) {
        v = (float)(rand() % 1000001) * 1e-3f; // uniform, same scale
      } else if (mode == 1) {
        // Mixed scale: one huge lane swamps the rest, so a different tree
        // rounds away a different set of small addends.
        v = (l == rand() % 64) ? 1e12f : (float)(rand() % 1000) * 1e-6f;
      } else if (mode == 2) {
        v = 1.0f; // exact, must be exact in both
      } else {
        // Powers of two straddling the mantissa boundary.
        v = ldexpf(1.0f, (rand() % 48) - 24);
      }
      h_in[i] = v;
    }
  }

  float *d_in, *d_a, *d_b, *d_c;
  CHECK(hipMalloc(&d_in, FB));
  CHECK(hipMalloc(&d_a, FB));
  CHECK(hipMalloc(&d_b, FB));
  CHECK(hipMalloc(&d_c, FB));
  CHECK(hipMemcpy(d_in, h_in, FB, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(run, dim3(CASES), dim3(64), 0, 0, d_in, d_a, d_b, d_c);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());

  float *h_a = (float *)malloc(FB), *h_b = (float *)malloc(FB),
        *h_c = (float *)malloc(FB);
  CHECK(hipMemcpy(h_a, d_a, FB, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(h_b, d_b, FB, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(h_c, d_c, FB, hipMemcpyDeviceToHost));

  // Lane 0 is the whole contract: `if (lane == 0) red[wave] = ssq`.
  int bad_b = 0, bad_c = 0, shown = 0;
  for (int c = 0; c < CASES; c++) {
    int i = c * 64;
    unsigned ua = __builtin_bit_cast(unsigned, h_a[i]);
    unsigned ub = __builtin_bit_cast(unsigned, h_b[i]);
    unsigned uc = __builtin_bit_cast(unsigned, h_c[i]);
    bad_b += (ua != ub);
    bad_c += (ua != uc);
    if ((ua != ub || ua != uc) && shown < 8) {
      printf("  case %d lane 0: butterfly %.9g  movadd %.9g  fused %.9g\n",
             c,
             h_a[i],
             h_b[i],
             h_c[i]);
      shown++;
    }
  }

  printf("%s: %d cases, lane-0 mismatches -- movadd %d, fused %d\n",
         (bad_b || bad_c) ? "FAIL" : "PASS",
         CASES,
         bad_b,
         bad_c);
  return (bad_b || bad_c) ? 1 : 0;
}
