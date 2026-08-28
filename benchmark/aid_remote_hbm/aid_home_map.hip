// Which AID homes a given address, and at what granularity?
//
// Classifier: bounce an uncached flag at the address between XCD0 and XCD1
// (both in the low chiplet group). If the address is homed near that group the
// round trip is fast (~700-800 ns); if it is homed on the other side it is slow
// (~1050-1210 ns). The split is wide and bimodal, so one run classifies one
// address. XCD4-XCD5 is used as a mirror check: it must come out inverted.
//
// A single-chiplet probe cannot see this. Repeated atomics from one chiplet
// keep the line owned locally and read ~94 ns no matter where it lives; the
// home only shows up when ownership has to move between chiplets.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_home_map.hip -o aid_home_map
//   HIP_VISIBLE_DEVICES=1 ./aid_home_map

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
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

__device__ inline unsigned get_xcc_id() {
  constexpr unsigned imm = ((4u - 1u) << 11) | (0u << 6) | 20u;
  return __builtin_amdgcn_s_getreg(imm);
}

__global__ void pingpong(int *flag, int xcc_a, int xcc_b, int rounds,
                         int *claim, uint64_t *cycles) {
  int x = (int)get_xcc_id();
  if ((x != xcc_a && x != xcc_b) || threadIdx.x != 0)
    return;
  int slot = (x == xcc_a) ? 0 : 1;
  if (atomicCAS(&claim[slot], 0, 1) != 0)
    return;

  if (slot == 0) {
    uint64_t t0 = wall_clock64();
    for (int r = 0; r < rounds; r++) {
      while (__hip_atomic_load(flag, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT) != r * 2) {
      }
      __hip_atomic_store(flag, r * 2 + 1, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
    cycles[0] = wall_clock64() - t0;
  } else {
    for (int r = 0; r < rounds; r++) {
      while (__hip_atomic_load(flag, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT) != r * 2 + 1) {
      }
      __hip_atomic_store(flag, r * 2 + 2, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

int main(int argc, char **argv) {
  int dev = (argc > 1) ? std::atoi(argv[1]) : 0;
  CHECK(hipSetDevice(dev));
  char bdf[32];
  CHECK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), dev));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, dev));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev));
  printf("device %d  %s  pci=%s\n\n", dev, prop.gcnArchName, bdf);

  const size_t span = 1ull << 30;
  char *arena = nullptr;
  CHECK(hipExtMallocWithFlags((void **)&arena, span, hipDeviceMallocUncached));
  int *d_claim = nullptr;
  uint64_t *d_cyc = nullptr;
  CHECK(hipMalloc(&d_claim, 2 * sizeof(int)));
  CHECK(hipMalloc(&d_cyc, sizeof(uint64_t)));

  const int rounds = 20000;
  auto probe = [&](size_t off, int a, int b) {
    int *flag = (int *)(arena + off);
    CHECK(hipMemset(flag, 0, sizeof(int)));
    CHECK(hipMemset(d_claim, 0, 2 * sizeof(int)));
    CHECK(hipMemset(d_cyc, 0, sizeof(uint64_t)));
    pingpong<<<prop.multiProcessorCount, 64>>>(flag, a, b, rounds, d_claim,
                                               d_cyc);
    CHECK(hipDeviceSynchronize());
    uint64_t cyc = 0;
    CHECK(hipMemcpy(&cyc, d_cyc, sizeof(uint64_t), hipMemcpyDeviceToHost));
    return (double)cyc * 1e6 / wall_khz / rounds;
  };

  // Calibrate the fast/slow threshold from the data itself.
  std::vector<double> cal;
  for (int i = 0; i < 16; i++)
    cal.push_back(probe((size_t)i * 4096, 0, 1));
  std::vector<double> s = cal;
  std::sort(s.begin(), s.end());
  double thresh = (s.front() + s.back()) / 2;
  printf("calibration over 16 pages: min %.0f  max %.0f  threshold %.0f ns\n\n",
         s.front(), s.back(), thresh);

  auto sweep = [&](const char *name, size_t stride, int n) {
    printf("== %s ==\n  ", name);
    std::vector<int> cls(n);
    for (int i = 0; i < n; i++) {
      double t = probe((size_t)i * stride, 0, 1);
      cls[i] = (t < thresh) ? 0 : 1;
      printf("%d", cls[i]);
      if ((i + 1) % 8 == 0)
        printf(" ");
      fflush(stdout);
    }
    printf("\n  (0 = homed near XCD0-3, 1 = homed near XCD4-7)\n");

    int flips = 0;
    for (int i = 1; i < n; i++)
      flips += (cls[i] != cls[i - 1]);
    int ones = 0;
    for (int c : cls)
      ones += c;
    printf("  %d/%d homed on side 1, %d transitions\n\n", ones, n, flips);
    return cls;
  };

  sweep("256 B stride within one page", 256, 32);
  sweep("1 KiB stride", 1024, 32);
  std::vector<int> pages = sweep("4 KiB stride (consecutive pages)", 4096, 64);
  sweep("2 MiB stride", 2ull << 20, 32);

  // Mirror check: the XCD4-XCD5 pair must invert the classification.
  printf("== mirror check with XCD4-XCD5 (must invert) ==\n");
  printf("  %-12s %10s %10s %8s %8s\n", "page", "XCD0-1", "XCD4-5", "cls01",
         "cls45");
  for (int i = 0; i < 8; i++) {
    size_t off = (size_t)i * 4096;
    double t01 = probe(off, 0, 1);
    double t45 = probe(off, 4, 5);
    printf("  %-12d %10.0f %10.0f %8d %8d\n", i, t01, t45,
           t01 < thresh ? 0 : 1, t45 < thresh ? 1 : 0);
  }

  CHECK(hipFree(arena));
  CHECK(hipFree(d_claim));
  CHECK(hipFree(d_cyc));
  return 0;
}
