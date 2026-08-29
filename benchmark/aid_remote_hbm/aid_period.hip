// Is AID homing positional?
//
// Everything expensive tried so far moves memory. If instead the home AID is a
// function of the *low* address bits only, then which AID owns a given 4 KiB
// offset is identical in every 2 MiB page and in every allocation -- known
// ahead of time, with no classification, no page table and nothing to relocate.
// Weight data could be permuted once at load so each workgroup's bytes land on
// its own AID's offsets, and the kernel would pay only a static index swizzle.
//
// The hint that this might hold: aid_interleave's run-length distribution
// reproduced to two decimals across a 512 MiB and a 2 GiB buffer (51.34 /
// 25.28 / 15.07 / 6.02%). Two independent allocations agreeing that precisely
// is what a positional pattern looks like, not a hash over full physical
// addresses.
//
// Three questions, in increasing order of usefulness:
//
//   1. periodicity  - does side[i] == side[i+P] for some P <= 512 pages (2 MiB)?
//   2. reproducible - do two separate allocations agree at the same offset?
//   3. alignment    - does agreement hold relative to 2 MiB alignment?
//
// A yes to (2) is the one that matters: it means the map is a constant of the
// hardware and can be baked in.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_period.hip -o aid_period
//   HIP_VISIBLE_DEVICES=0 ./aid_period [MiB]

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

#define CHECK(x)                                                               \
  do {                                                                         \
    hipError_t _e = (x);                                                       \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,              \
              hipGetErrorString(_e));                                          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static constexpr size_t kPage = 4096;

__device__ inline unsigned get_xcc_id() {
  constexpr unsigned imm = ((4u - 1u) << 11) | (0u << 6) | 20u;
  return __builtin_amdgcn_s_getreg(imm);
}

__global__ void classify_parallel(char *base, size_t npages, int rounds,
                                  int per_set, int sets, int *claim,
                                  uint64_t *cycles, uint64_t spin_limit) {
  int x = (int)get_xcc_id();
  if (threadIdx.x != 0 || x >= sets * 2)
    return;
  int set = x >> 1, role = x & 1;
  int pid = -1;
  for (int i = 0; i < per_set; i++) {
    int slot = (set * per_set + i) * 2 + role;
    if (atomicCAS(&claim[slot], 0, 1) == 0) {
      pid = set * per_set + i;
      break;
    }
  }
  if (pid < 0)
    return;
  const int npairs = per_set * sets;
  for (size_t p = pid; p < npages; p += npairs) {
    int *flag = (int *)(base + p * kPage);
    uint64_t t0 = wall_clock64();
    bool ok = true;
    for (int r = 0; r < rounds && ok; r++) {
      int want = role == 0 ? r * 2 : r * 2 + 1;
      uint64_t guard = wall_clock64();
      while (__hip_atomic_load(flag, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT) != want) {
        if (wall_clock64() - guard > spin_limit) {
          ok = false;
          break;
        }
      }
      if (!ok)
        break;
      __hip_atomic_store(flag, want + 1, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
    if (role == 0)
      cycles[p] = ok ? wall_clock64() - t0 : 0;
  }
}

static double now_s() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

struct Buf {
  char *p;
  std::vector<uint8_t> side;
  std::vector<double> ns;
};

static void classify(Buf &b, size_t bytes, int cus, int wall_khz) {
  const size_t npages = bytes / kPage;
  CHECK(hipMemsetD8((hipDeviceptr_t)b.p, 0, bytes));
  int *d_claim = nullptr;
  uint64_t *d_cyc = nullptr;
  CHECK(hipMalloc(&d_claim, 4096 * sizeof(int)));
  CHECK(hipMalloc(&d_cyc, npages * sizeof(uint64_t)));
  CHECK(hipMemset(d_claim, 0, 4096 * sizeof(int)));
  CHECK(hipMemset(d_cyc, 0, npages * sizeof(uint64_t)));
  classify_parallel<<<cus, 64>>>(b.p, npages, 48, 32, 2, d_claim, d_cyc,
                                 (uint64_t)wall_khz * 1000);
  CHECK(hipDeviceSynchronize());
  std::vector<uint64_t> cyc(npages);
  CHECK(hipMemcpy(cyc.data(), d_cyc, npages * sizeof(uint64_t),
                  hipMemcpyDeviceToHost));
  b.ns.resize(npages);
  for (size_t i = 0; i < npages; i++)
    b.ns[i] = (double)cyc[i] * 1e6 / wall_khz / 48;
  std::vector<double> srt(b.ns);
  std::sort(srt.begin(), srt.end());
  const double th = (srt[srt.size() / 10] + srt[srt.size() * 9 / 10]) / 2;
  b.side.resize(npages);
  for (size_t i = 0; i < npages; i++)
    b.side[i] = b.ns[i] < th ? 0 : 1;
  CHECK(hipFree(d_claim));
  CHECK(hipFree(d_cyc));
}

int main(int argc, char **argv) {
  const size_t want_mib = (argc > 1) ? (size_t)std::atoll(argv[1]) : 256;
  CHECK(hipSetDevice(0));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
  const size_t bytes = want_mib << 20;
  const size_t npages = bytes / kPage;

  Buf A{}, B{};
  CHECK(hipMalloc(&A.p, bytes));
  CHECK(hipMalloc(&B.p, bytes));
  printf("two %zu MiB buffers: A=%p  B=%p\n", want_mib, (void *)A.p,
         (void *)B.p);
  printf("2 MiB alignment: A offset %zu KiB, B offset %zu KiB\n",
         ((uintptr_t)A.p & ((2 << 20) - 1)) / 1024,
         ((uintptr_t)B.p & ((2 << 20) - 1)) / 1024);

  double t0 = now_s();
  classify(A, bytes, prop.multiProcessorCount, wall_khz);
  classify(B, bytes, prop.multiProcessorCount, wall_khz);
  printf("classified both in %.2f s\n\n", now_s() - t0);

  // (1) periodicity within one buffer
  printf("periodicity of A: fraction with side[i] == side[i+P]\n");
  printf("  %10s %10s %10s\n", "P(pages)", "P(bytes)", "agree");
  double best = 0;
  size_t bestp = 0;
  for (size_t P : {1ul, 2ul, 4ul, 8ul, 16ul, 32ul, 64ul, 128ul, 256ul, 512ul,
                   1024ul, 2048ul}) {
    if (P >= npages)
      break;
    size_t same = 0;
    for (size_t i = 0; i + P < npages; i++)
      same += (A.side[i] == A.side[i + P]);
    double f = (double)same / (npages - P);
    printf("  %10zu %8zu KiB %9.1f%%%s\n", P, P * 4, 100.0 * f,
           f > 0.95 ? "   <- periodic" : "");
    if (f > best) {
      best = f;
      bestp = P;
    }
  }

  // (2) do two independent allocations agree at the same offset?
  size_t agree = 0;
  for (size_t i = 0; i < npages; i++)
    agree += (A.side[i] == B.side[i]);
  printf("\nA vs B at same buffer offset: %.1f%% agree\n",
         100.0 * agree / npages);

  // (3) same, but realigned so both start at a 2 MiB boundary
  const size_t offA = (((uintptr_t)A.p + (2 << 20) - 1) & ~((uintptr_t)(2 << 20) - 1) -
                       (uintptr_t)A.p) / kPage;
  const size_t offB = (((uintptr_t)B.p + (2 << 20) - 1) & ~((uintptr_t)(2 << 20) - 1) -
                       (uintptr_t)B.p) / kPage;
  size_t n2 = npages - std::max(offA, offB);
  size_t agree2 = 0;
  for (size_t i = 0; i < n2; i++)
    agree2 += (A.side[offA + i] == B.side[offB + i]);
  printf("A vs B aligned to 2 MiB boundaries: %.1f%% agree\n",
         100.0 * agree2 / n2);

  // A control: how self-consistent is the classifier at all? Re-classify A and
  // compare to itself. Anything above this floor is signal, below it is noise.
  Buf A2{};
  A2.p = A.p;
  classify(A2, bytes, prop.multiProcessorCount, wall_khz);
  size_t rep = 0;
  for (size_t i = 0; i < npages; i++)
    rep += (A.side[i] == A2.side[i]);
  printf("\ncontrol -- A re-classified against itself: %.1f%% agree\n",
         100.0 * rep / npages);
  printf("(that is the classifier's own repeatability; treat it as the "
         "ceiling)\n");

  printf("\nbest period found: P=%zu pages (%zu KiB) at %.1f%%\n", bestp,
         bestp * 4, 100.0 * best);
  return 0;
}
