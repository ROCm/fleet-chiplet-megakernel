/* Copyright 2025 CMU
 *
 * Regression test for the two gfx950 scaled-MFMA pipelining hazards.
 *
 * These hazards are SILENT: they corrupt roughly one iteration out of a
 * 22-iteration accumulation, which on an FP4-quantized model looks like
 * quantization noise rather than a bug. Nothing hangs, nothing faults, and
 * the model still emits fluent text. The only way to catch them is to
 * compare against a schedule that cannot race.
 *
 * Hazard 1 (WAR on MFMA source registers)
 *   The pipelined loop issues iteration N+1's ds_read into the SAME VGPRs
 *   that iteration N's MFMA is reading, and relies on the reads landing
 *   "late". lgkmcnt tracks when LDS data reaches the VGPR -- it says nothing
 *   about when the MFMA has finished sampling its sources. When LDS returns
 *   fast the write-back lands mid-MFMA and the op sees mixed-iteration
 *   operands. Intermittent: ~17-22% of launches.
 *
 * Hazard 2 (AccVGPR read before retirement)
 *   "s_nop 7; s_nop 0" is a 9-clock gap. The FP4xFP8 scaled MFMA is a
 *   32-cycle op on CDNA4, so v_accvgpr_read_b32 samples the accumulator
 *   mid-retirement. Deterministic: fails every launch. The accumulator
 *   retires progressively, so the signature depends on the gap: at 9 clocks
 *   a0..a2 are stale with a3 correct, at 10 clocks only a0 is stale.
 *
 *   Measured threshold on gfx950 is exactly 11 clocks -- 10 fails 300/300,
 *   11 passes 300/300. Note that LLVM's hazard recognizer emits precisely
 *   "s_nop 7; s_nop 2" (11 clocks) for intrinsic-based MFMAs elsewhere in
 *   this codebase, which corroborates the number. Inline asm bypasses that
 *   recognizer entirely -- that is why the hand-written tail sat 2 clocks
 *   short. We still pad to 32 rather than 11: the threshold is not
 *   architecturally documented, and the tail runs once per GEMM.
 *
 * The fix for hazard 1 is two disjoint operand banks (ping-pong), so no
 * register is ever both a live MFMA source and an in-flight LDS destination.
 * The fix for hazard 2 is "s_nop 15" x2 == 32 clocks.
 *
 * NOTE ON LOOP PARITY: the ping-pong loop's exit parity decides which bank
 * holds the final operands. An ODD iteration count falls out of the bank-1
 * half (final operands in bank 0); an EVEN count exits via the bank-0 half
 * (final operands in bank 1). A single-tail port gets one of the two cases
 * backwards and silently reintroduces the bug, so BOTH exit paths are spelled
 * out below. Production runs 23 iterations (REDUCTION_SIZE 2944 /
 * K_PER_MFMA 128) for QKV, W13 and W2 alike. Rebuild with -DMFMA_ITERS=22 to
 * exercise the even case.
 *
 * Build: ./build.sh   (or see the hipcc line in that script)
 * Run:   ./test_mfma_pipeline_hazards [launches]
 *
 * Exit code 0 = fixed schedule is bit-exact across all launches.
 * Exit code 1 = a hazard is present; the mfma_fixed kernel below is the
 *               schedule production code should match.
 */

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef MFMA_ITERS
// Production runs 23 (REDUCTION_SIZE 2944 / K_PER_MFMA 128). Both parities
// are exercised -- see the ITERS note in the header comment.
#define MFMA_ITERS 23
#endif

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,     \
                   hipGetErrorString(err));                                    \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

// LDS staging shared by every variant: A data 64B/iter, A scale 1B/iter,
// B data 128B/iter, B scale 1B/iter.
#define HAZ_PROLOGUE                                                           \
  extern __shared__ uint8_t smem[];                                            \
  int tid = threadIdx.x;                                                       \
  uint8_t *sAd = smem;                                                         \
  uint8_t *sAs = sAd + 64 * MFMA_ITERS;                                        \
  uint8_t *sBd = sAs + 64;                                                     \
  uint8_t *sBs = sBd + 128 * MFMA_ITERS;                                       \
  for (int i = tid; i < 64 * MFMA_ITERS; i += 256)                             \
    sAd[i] = Ad[i];                                                            \
  for (int i = tid; i < MFMA_ITERS; i += 256)                                  \
    sAs[i] = As[i];                                                            \
  for (int i = tid; i < 128 * MFMA_ITERS; i += 256)                            \
    sBd[i] = Bd[i];                                                            \
  for (int i = tid; i < MFMA_ITERS; i += 256)                                  \
    sBs[i] = Bs[i];                                                            \
  __syncthreads();                                                             \
  int grp = (tid & 63) >> 4;                                                   \
  unsigned w_addr = (unsigned)(uintptr_t)(sAd) + (unsigned)(grp * 16);         \
  unsigned ws_addr = (unsigned)(uintptr_t)(sAs);                               \
  unsigned t_addr = (unsigned)(uintptr_t)(sBd) + (unsigned)(grp * 16);         \
  unsigned ts_addr = (unsigned)(uintptr_t)(sBs);                               \
  float acc0, acc1, acc2, acc3;

#define HAZ_EPILOGUE                                                           \
  size_t off = ((size_t)blockIdx.x * 256 + tid) * 4;                           \
  out[off + 0] = acc0;                                                         \
  out[off + 1] = acc1;                                                         \
  out[off + 2] = acc2;                                                         \
  out[off + 3] = acc3;

#define HAZ_OUTS                                                               \
  [a0] "=v"(acc0), [a1] "=v"(acc1), [a2] "=v"(acc2), [a3] "=v"(acc3),          \
      [wa] "+v"(w_addr), [wsa] "+v"(ws_addr), [ta] "+v"(t_addr)

#define HAZ_INS [tsa] "v"(ts_addr), [im1] "n"(MFMA_ITERS - 1)

// Both operand banks are clobbered in every variant so the register
// allocator cannot reuse them and perturb the schedule under test.
#define HAZ_CLOBBER                                                            \
  "memory", "s13", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",        \
      "v15", "v16", "v17", "v18", "v19", "v22", "v23", "v24", "v25", "v26",    \
      "v27", "v28", "v29", "v32", "v33", "v34", "v35", "v36", "v37", "v38",    \
      "v39", "a0", "a1", "a2", "a3"

// ---------------------------------------------------------------------------
// BROKEN: the historical production schedule. Both hazards present.
// Kept so the test proves it is actually detecting something.
// ---------------------------------------------------------------------------
__global__ void mfma_broken(float *out, uint8_t const *Ad, uint8_t const *As,
                            uint8_t const *Bd, uint8_t const *Bs) {
  HAZ_PROLOGUE
  asm volatile(
      "v_accvgpr_write_b32 a0, 0\nv_accvgpr_write_b32 a1, 0\n"
      "v_accvgpr_write_b32 a2, 0\nv_accvgpr_write_b32 a3, 0\n"
      "ds_read_b128 v[22:25], %[wa]\nds_read_u8 v7, %[wsa]\n"
      "ds_read_b128 v[8:11], %[ta]\nds_read_b128 v[12:15], %[ta] offset:64\n"
      "ds_read_u8 v16, %[tsa]\ns_mov_b32 s13, 0\n"
      "HAZBAD%=:\n"
      "s_waitcnt lgkmcnt(0)\n"
      "v_add_u32_e32 %[wa], 64, %[wa]\nv_add_u32_e32 %[wsa], 1, %[wsa]\n"
      "v_add_u32_e32 %[ta], 0x80, %[ta]\ns_add_i32 s13, s13, 1\n"
      "v_add_u32_e32 v17, s13, %[tsa]\nds_read_u8 v17, v17\n"
      // HAZARD 1: these four reads target the MFMA's own source registers.
      "ds_read_b128 v[22:25], %[wa]\nds_read_u8 v7, %[wsa]\n"
      "ds_read_b128 v[8:11], %[ta]\nds_read_b128 v[12:15], %[ta] offset:64\n"
      "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], a[0:3], "
      "v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
      "s_waitcnt lgkmcnt(4)\nv_mov_b32_e32 v16, v17\n"
      "s_cmpk_lt_i32 s13, %[im1]\ns_cbranch_scc1 HAZBAD%=\n"
      "s_waitcnt lgkmcnt(0)\n"
      "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], a[0:3], "
      "v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
      // HAZARD 2: 9 clocks before reading a 32-cycle op's accumulator.
      "s_nop 7\ns_nop 0\n"
      "v_accvgpr_read_b32 %[a0], a0\nv_accvgpr_read_b32 %[a1], a1\n"
      "v_accvgpr_read_b32 %[a2], a2\nv_accvgpr_read_b32 %[a3], a3\n"
      : HAZ_OUTS
      : HAZ_INS
      : HAZ_CLOBBER);
  HAZ_EPILOGUE
}

// ---------------------------------------------------------------------------
// FIXED: two disjoint operand banks + a 32-clock retirement gap.
// This is the schedule production code should match.
// ---------------------------------------------------------------------------
__global__ void mfma_fixed(float *out, uint8_t const *Ad, uint8_t const *As,
                           uint8_t const *Bd, uint8_t const *Bs) {
  HAZ_PROLOGUE
  asm volatile(
      "v_accvgpr_write_b32 a0, 0\nv_accvgpr_write_b32 a1, 0\n"
      "v_accvgpr_write_b32 a2, 0\nv_accvgpr_write_b32 a3, 0\n"
      // Preload bank 0.
      "ds_read_b128 v[22:25], %[wa]\nds_read_u8 v7, %[wsa]\n"
      "ds_read_b128 v[8:11], %[ta]\nds_read_b128 v[12:15], %[ta] offset:64\n"
      "ds_read_u8 v16, %[tsa]\ns_mov_b32 s13, 0\n"
      "HAZFIX%=:\n"
      // --- consume bank 0, prefetch into bank 1 (disjoint registers) ---
      "s_waitcnt lgkmcnt(0)\n"
      "v_add_u32_e32 %[wa], 64, %[wa]\nv_add_u32_e32 %[wsa], 1, %[wsa]\n"
      "v_add_u32_e32 %[ta], 0x80, %[ta]\ns_add_i32 s13, s13, 1\n"
      "v_add_u32_e32 v17, s13, %[tsa]\nds_read_u8 v19, v17\n"
      "ds_read_b128 v[26:29], %[wa]\nds_read_u8 v18, %[wsa]\n"
      "ds_read_b128 v[32:35], %[ta]\nds_read_b128 v[36:39], %[ta] offset:64\n"
      "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], a[0:3], "
      "v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
      "s_cmpk_lt_i32 s13, %[im1]\ns_cbranch_scc0 HAZTAIL1%=\n"
      // --- consume bank 1, prefetch into bank 0 ---
      "s_waitcnt lgkmcnt(0)\n"
      "v_add_u32_e32 %[wa], 64, %[wa]\nv_add_u32_e32 %[wsa], 1, %[wsa]\n"
      "v_add_u32_e32 %[ta], 0x80, %[ta]\ns_add_i32 s13, s13, 1\n"
      "v_add_u32_e32 v17, s13, %[tsa]\nds_read_u8 v16, v17\n"
      "ds_read_b128 v[22:25], %[wa]\nds_read_u8 v7, %[wsa]\n"
      "ds_read_b128 v[8:11], %[ta]\nds_read_b128 v[12:15], %[ta] offset:64\n"
      "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], a[0:3], "
      "v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
      "s_cmpk_lt_i32 s13, %[im1]\ns_cbranch_scc1 HAZFIX%=\n"
      // Exited after a bank-1 iteration: final operands live in bank 0.
      "s_waitcnt lgkmcnt(0)\n"
      "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], a[0:3], "
      "v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
      "s_branch HAZDONE%=\n"
      "HAZTAIL1%=:\n"
      // Exited after a bank-0 iteration: final operands live in bank 1.
      "s_waitcnt lgkmcnt(0)\n"
      "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], a[0:3], "
      "v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"
      "HAZDONE%=:\n"
      // 32 clocks: matches the scaled-MFMA latency on CDNA4.
      "s_nop 15\ns_nop 15\n"
      "v_accvgpr_read_b32 %[a0], a0\nv_accvgpr_read_b32 %[a1], a1\n"
      "v_accvgpr_read_b32 %[a2], a2\nv_accvgpr_read_b32 %[a3], a3\n"
      : HAZ_OUTS
      : HAZ_INS
      : HAZ_CLOBBER);
  HAZ_EPILOGUE
}

// ---------------------------------------------------------------------------
// REFERENCE: no overlap whatsoever. Slow, obviously correct, used as truth.
// Every operand is fully resident before the MFMA issues, and the MFMA is
// fully retired before anything touches its registers or accumulator.
// ---------------------------------------------------------------------------
__global__ void mfma_reference(float *out, uint8_t const *Ad, uint8_t const *As,
                               uint8_t const *Bd, uint8_t const *Bs) {
  HAZ_PROLOGUE
  asm volatile(
      "v_accvgpr_write_b32 a0, 0\nv_accvgpr_write_b32 a1, 0\n"
      "v_accvgpr_write_b32 a2, 0\nv_accvgpr_write_b32 a3, 0\n"
      "s_mov_b32 s13, 0\n"
      "HAZREF%=:\n"
      "ds_read_b128 v[22:25], %[wa]\nds_read_u8 v7, %[wsa]\n"
      "ds_read_b128 v[8:11], %[ta]\nds_read_b128 v[12:15], %[ta] offset:64\n"
      "v_add_u32_e32 v17, s13, %[tsa]\nds_read_u8 v16, v17\n"
      "s_waitcnt lgkmcnt(0)\ns_nop 15\ns_nop 15\n"
      "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], a[0:3], "
      "v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
      "s_nop 15\ns_nop 15\ns_nop 15\ns_nop 15\n"
      "v_add_u32_e32 %[wa], 64, %[wa]\nv_add_u32_e32 %[wsa], 1, %[wsa]\n"
      "v_add_u32_e32 %[ta], 0x80, %[ta]\ns_add_i32 s13, s13, 1\n"
      "s_cmpk_lt_i32 s13, %[it]\ns_cbranch_scc1 HAZREF%=\n"
      "s_nop 15\ns_nop 15\ns_nop 15\ns_nop 15\n"
      "v_accvgpr_read_b32 %[a0], a0\nv_accvgpr_read_b32 %[a1], a1\n"
      "v_accvgpr_read_b32 %[a2], a2\nv_accvgpr_read_b32 %[a3], a3\n"
      : HAZ_OUTS
      : [tsa] "v"(ts_addr), [it] "n"(MFMA_ITERS)
      : HAZ_CLOBBER);
  HAZ_EPILOGUE
}

typedef void (*hazard_kernel_t)(float *, uint8_t const *, uint8_t const *,
                                uint8_t const *, uint8_t const *);

int main(int argc, char **argv) {
  int const launches = (argc > 1) ? std::atoi(argv[1]) : 300;
  int const grid = 64;

  size_t const szA = 64 * MFMA_ITERS, szAs = MFMA_ITERS;
  size_t const szB = 128 * MFMA_ITERS, szBs = MFMA_ITERS;
  std::vector<uint8_t> hA(szA), hAs(szAs), hB(szB), hBs(szBs);

  // Deterministic inputs. A is FP4 (e2m1) packed two per byte -- every nibble
  // is finite. B is FP8 e4m3, where exponent 0xF with a nonzero mantissa is
  // NaN, so the exponent is held in [1,13] to keep every byte finite and
  // normal. Random bytes here would decode to NaN and mask the comparison.
  unsigned state = 12345u;
  auto next = [&]() {
    state = state * 1664525u + 1013904223u;
    return state >> 16;
  };
  for (size_t i = 0; i < szA; i++) {
    hA[i] = (uint8_t)(next() & 0xff);
  }
  for (size_t i = 0; i < szB; i++) {
    unsigned r = next();
    hB[i] = (uint8_t)(((r & 1) << 7) | ((1 + ((r >> 1) % 13)) << 3) |
                      ((r >> 8) & 0x7));
  }
  // E8M0 scales clustered near 1.0 (bias 127).
  for (size_t i = 0; i < szAs; i++) {
    hAs[i] = (uint8_t)(124 + (next() % 7));
  }
  for (size_t i = 0; i < szBs; i++) {
    hBs[i] = (uint8_t)(124 + (next() % 7));
  }

  uint8_t *dA, *dAs, *dB, *dBs;
  float *dOut;
  HIP_CHECK(hipMalloc(&dA, szA));
  HIP_CHECK(hipMalloc(&dAs, szAs));
  HIP_CHECK(hipMalloc(&dB, szB));
  HIP_CHECK(hipMalloc(&dBs, szBs));
  HIP_CHECK(hipMalloc(&dOut, (size_t)grid * 1024 * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, hA.data(), szA, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dAs, hAs.data(), szAs, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hB.data(), szB, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dBs, hBs.data(), szBs, hipMemcpyHostToDevice));

  size_t const smem =
      64 * MFMA_ITERS + 64 + 128 * MFMA_ITERS + 64 + 256;

  std::vector<float> host((size_t)grid * 1024);
  auto launch = [&](hazard_kernel_t k, int blocks) {
    HIP_CHECK(hipMemset(dOut, 0, (size_t)blocks * 1024 * sizeof(float)));
    hipLaunchKernelGGL(k, dim3(blocks), dim3(256), smem, 0, dOut, dA, dAs, dB,
                       dBs);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(host.data(), dOut,
                        (size_t)blocks * 1024 * sizeof(float),
                        hipMemcpyDeviceToHost));
  };

  // Ground truth from the non-overlapping reference.
  launch(mfma_reference, 1);
  std::vector<float> truth(host.begin(), host.begin() + 1024);
  for (int i = 0; i < 1024; i++) {
    if (!std::isfinite(truth[i])) {
      std::printf("FAIL: reference produced a non-finite value at %d\n", i);
      return 2;
    }
  }

  std::printf("gfx950 scaled-MFMA pipeline hazard regression\n");
  std::printf("  iterations %d, grid %d, launches %d\n", MFMA_ITERS, grid,
              launches);
  std::printf("  reference lane0 accumulator: %.4f\n\n", truth[0]);

  struct Variant {
    char const *name;
    hazard_kernel_t kernel;
    bool must_match;
  } const variants[] = {
      {"reference (no overlap)", mfma_reference, true},
      {"fixed     (banked + 32clk)", mfma_fixed, true},
      {"broken    (historical)", mfma_broken, false},
  };

  int failures = 0;
  for (auto const &v : variants) {
    long wrong_blocks = 0, total_blocks = 0;
    double worst_rel = 0.0;
    for (int t = 0; t < launches; t++) {
      launch(v.kernel, grid);
      for (int b = 0; b < grid; b++) {
        total_blocks++;
        bool bad = false;
        for (int i = 0; i < 1024; i++) {
          float got = host[(size_t)b * 1024 + i];
          if (got != truth[i]) {
            bad = true;
            if (std::fabs(truth[i]) > 1e-6) {
              worst_rel = std::max(
                  worst_rel,
                  std::fabs((double)got - truth[i]) / std::fabs(truth[i]));
            }
          }
        }
        if (bad) {
          wrong_blocks++;
        }
      }
    }
    std::printf("  %-28s %7ld/%ld blocks differ", v.name, wrong_blocks,
                total_blocks);
    if (wrong_blocks) {
      std::printf("  (worst rel err %.2f%%)", 100.0 * worst_rel);
    }
    std::printf("\n");

    if (v.must_match && wrong_blocks != 0) {
      failures++;
    }
    // If the broken variant stops reproducing, this test has gone blind --
    // a toolchain or scheduler change may be hiding the hazard rather than
    // fixing it. Report loudly instead of passing quietly.
    if (!v.must_match && wrong_blocks == 0) {
      std::printf("  WARNING: the known-broken schedule now matches the "
                  "reference.\n"
                  "           This test can no longer prove it detects the "
                  "hazard.\n"
                  "           Inspect the emitted ISA before trusting a pass.\n");
    }
  }

  std::printf("\n");
  if (failures) {
    std::printf("FAIL: a hazard-free schedule did not reproduce the "
                "reference.\n"
                "      Compare the emitted ISA against the mfma_fixed kernel "
                "in this file:\n"
                "      hipcc --offload-arch=gfx950 -S "
                "test_mfma_pipeline_hazards.hip -o -\n");
    return 1;
  }
  std::printf("PASS: hazard-free schedules are bit-exact across all "
              "launches.\n");
  return 0;
}
