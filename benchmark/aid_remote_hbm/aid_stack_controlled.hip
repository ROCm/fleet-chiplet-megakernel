// Locality at constant stack count.
//
// The NPS1-vs-NPS2 comparison in aid_nps_compare is confounded and its
// bandwidth conclusion was wrong. Both arms used 4 XCDs, but under NPS1 those
// XCDs read a buffer interleaved over all 8 HBM stacks while under NPS2 they
// were confined to their partition's 4. Stack count alone predicts about the
// 2x that was measured, so it was not evidence that NPS2 "costs bandwidth" --
// the two variables moved together.
//
// This holds stacks fixed. Everything runs under SPX/NPS1 on XCDs 0-3, which
// are AID0. Pages are classified by home AID and the same 4 XCDs then read:
//
//   local  : only AID0 pages  -> 4 stacks, 0 hops   (the NPS2 analogue)
//   remote : only AID1 pages  -> 4 stacks, every read crosses
//
// Same XCDs, same stack count, same page count, same gather pattern. The only
// difference is whether the traffic crosses the interconnect, so the ratio is
// locality and nothing else. `local` is also directly comparable to the NPS2
// partition number, since both are 4 XCDs over 4 local stacks -- if they agree,
// NPS2's lower bandwidth is entirely explained by stack count and costs nothing
// of its own.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_stack_controlled.hip -o aid_stack_controlled
//   HIP_VISIBLE_DEVICES=0 ./aid_stack_controlled [MiB]

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
static constexpr int kActiveXCD = 4; // XCDs 0-3 == AID0
typedef unsigned int v4u __attribute__((ext_vector_type(4)));

__device__ inline unsigned get_xcc_id() {
  constexpr unsigned imm = ((4u - 1u) << 11) | (0u << 6) | 20u;
  return __builtin_amdgcn_s_getreg(imm);
}

// Ping-pong a flag in each page between two XCDs that both sit on AID0, so the
// measured latency says whether the page is homed near AID0 or far from it.
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

// Only XCDs 0-3 work. Eight pages per trip for memory-level parallelism; the
// 256 threads cover each 4 KiB page with one 16 B non-temporal load each.
__global__ __launch_bounds__(256) void gather(const char *__restrict base,
                                              const int *__restrict list, int n,
                                              int nxcd, int iters,
                                              uint64_t *__restrict sink) {
  const int xcc = (int)get_xcc_id();
  if (xcc >= kActiveXCD)
    return;
  const int per = gridDim.x / nxcd;
  const int rank = xcc * per + blockIdx.x / nxcd;
  const int nranks = kActiveXCD * per;

  uint64_t acc = 0;
  for (int it = 0; it < iters; it++)
    for (int i = rank * 8; i < n; i += nranks * 8) {
      v4u v[8];
#pragma unroll
      for (int u = 0; u < 8; u++) {
        const v4u *p = (const v4u *)(base + (size_t)list[min(i + u, n - 1)] * kPage);
        v[u] = __builtin_nontemporal_load(p + threadIdx.x);
      }
#pragma unroll
      for (int u = 0; u < 8; u++)
        acc += v[u].x + v[u].w;
    }
  sink[blockIdx.x * 256 + threadIdx.x] = acc;
}

static double now_s() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  const size_t want_mib = (argc > 1) ? (size_t)std::atoll(argv[1]) : 4096;
  CHECK(hipSetDevice(0));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
  const int nxcd = prop.multiProcessorCount / 32;
  printf("CUs %d -> %d XCDs; working on XCDs 0-%d (AID0) only\n",
         prop.multiProcessorCount, nxcd, kActiveXCD - 1);
  if (nxcd < 8)
    printf("  NOTE: expected SPX/NPS1 (8 XCDs); classification needs both AIDs "
           "visible\n");

  const size_t bytes = want_mib << 20;
  const size_t npages = bytes / kPage;
  char *buf = nullptr;
  CHECK(hipMalloc(&buf, bytes));
  CHECK(hipMemsetD8((hipDeviceptr_t)buf, 0, bytes));

  int *d_claim = nullptr;
  uint64_t *d_cyc = nullptr;
  CHECK(hipMalloc(&d_claim, 4096 * sizeof(int)));
  CHECK(hipMalloc(&d_cyc, npages * sizeof(uint64_t)));
  CHECK(hipMemset(d_claim, 0, 4096 * sizeof(int)));
  CHECK(hipMemset(d_cyc, 0, npages * sizeof(uint64_t)));
  double t0 = now_s();
  classify_parallel<<<prop.multiProcessorCount, 64>>>(
      buf, npages, 48, 32, 2, d_claim, d_cyc, (uint64_t)wall_khz * 1000);
  CHECK(hipDeviceSynchronize());
  printf("classified %zu pages (%zu MiB) in %.2f s\n", npages, want_mib,
         now_s() - t0);

  std::vector<uint64_t> cyc(npages);
  CHECK(hipMemcpy(cyc.data(), d_cyc, npages * sizeof(uint64_t),
                  hipMemcpyDeviceToHost));
  std::vector<double> ns(npages);
  for (size_t i = 0; i < npages; i++)
    ns[i] = (double)cyc[i] * 1e6 / wall_khz / 48;
  std::vector<double> srt(ns);
  std::sort(srt.begin(), srt.end());
  const double th = (srt[srt.size() / 10] + srt[srt.size() * 9 / 10]) / 2;
  std::vector<int> aid0, aid1;
  for (size_t i = 0; i < npages; i++)
    (ns[i] < th ? aid0 : aid1).push_back((int)i);
  const int n = (int)std::min(aid0.size(), aid1.size());
  printf("  %zu pages near AID0, %zu far -> %d per arm (%.0f MiB each)\n",
         aid0.size(), aid1.size(), n, n * 4096.0 / 1048576.0);
  CHECK(hipMemsetD8((hipDeviceptr_t)buf, 1, bytes));

  int *d_near = nullptr, *d_far = nullptr;
  CHECK(hipMalloc(&d_near, (size_t)n * sizeof(int)));
  CHECK(hipMalloc(&d_far, (size_t)n * sizeof(int)));
  CHECK(hipMemcpy(d_near, aid0.data(), (size_t)n * sizeof(int),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_far, aid1.data(), (size_t)n * sizeof(int),
                  hipMemcpyHostToDevice));

  uint64_t *d_sink = nullptr;
  CHECK(hipMalloc(&d_sink, (size_t)8192 * 256 * sizeof(uint64_t)));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  const size_t moved = (size_t)n * kPage;

  auto bw = [&](int *list, int grid, int iters) {
    gather<<<grid, 256>>>(buf, list, n, nxcd, 1, d_sink);
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    gather<<<grid, 256>>>(buf, list, n, nxcd, iters, d_sink);
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    return (double)moved * iters / (ms * 1e6);
  };
  auto med3 = [&](int *list, int grid, int iters) {
    double v[3];
    for (int k = 0; k < 3; k++)
      v[k] = bw(list, grid, iters);
    std::sort(v, v + 3);
    return v[1];
  };

  printf("\n4 XCDs, 4 stacks in both arms -- only the crossing differs\n");
  printf("%-8s %14s %14s %10s\n", "K/XCD", "local(AID0)", "remote(AID1)",
         "loc/rem");
  for (int K : {32, 64, 128, 256}) {
    const int grid = nxcd * K;
    const int iters = std::max(4, (int)((8ull << 30) / moved));
    double l = med3(d_near, grid, iters);
    double r = med3(d_far, grid, iters);
    printf("%-8d %11.1f GB/s %11.1f %9.3fx\n", K, l, r, l / r);
  }
  printf("\nCompare `local` against the NPS2 partition (4 XCDs, 4 local\n"
         "stacks): if they match, NPS2's lower bandwidth is stack count\n"
         "alone and locality costs nothing.\n");
  return 0;
}
