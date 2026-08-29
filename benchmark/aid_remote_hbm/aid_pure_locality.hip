// The locality effect with the mechanism cost removed.
//
// aid_moe_bw pinned pages with the HIP VMM API, which forces one 4 KiB handle
// per page and costs 32% of read bandwidth to TLB pressure (VMM-mixed 4518 GB/s
// against hipMalloc 5981). That penalty belongs to the *mechanism*, not to AID
// locality, and it capped the AID-local arm at 4921 GB/s -- possibly before
// locality could show any benefit at all. Measuring the effect through it was a
// methodological error.
//
// This removes it. A plain hipMalloc buffer already has ~50/50 page homing, so
// instead of moving pages to match the readers, classify the pages and move the
// *readers* to match the pages: each XCD group reads only the pages already
// homed on its own AID. Ordinary 2 MiB-backed hipMalloc memory, no VMM, no
// handles, no startup cost -- and perfect locality.
//
// Three arms over the same buffer with the same gather pattern, differing only
// in which page list a group walks:
//
//   local  : XCD group g reads pages homed on AID g          (0% remote)
//   mixed  : each group reads an even blend of both lists    (50% remote)
//   remote : XCD group g reads pages homed on AID 1-g        (100% remote)
//
// `mixed` is the honest stand-in for stock hipMalloc, and because all three use
// an identical page-granular gather, any difference is homing alone.
//
// Why this matters: the arms in aid_moe_bw fit a model where the inter-AID link
// is worth ~2.9 TB/s. remote (100% crossing) measured 2881 GB/s, and hipMalloc
// at 5981 GB/s sends 50% across, i.e. 2990 GB/s -- right at that limit. If that
// is real, stock allocation is *already* interconnect-bound and locality should
// buy real bandwidth here.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_pure_locality.hip -o aid_pure_locality
//   HIP_VISIBLE_DEVICES=1 ./aid_pure_locality [MiB]

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
typedef unsigned int v4u __attribute__((ext_vector_type(4)));

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

// Page-granular gather. Four pages per trip so there is enough memory-level
// parallelism to reach saturation; within a page the 256 threads cover the
// 4 KiB with one 16 B non-temporal load each.
__global__ __launch_bounds__(256) void gather_stream(
    const char *__restrict base, const int *__restrict list, int n, int iters,
    uint64_t *__restrict sink) {
  const int x = (int)get_xcc_id();
  const int aid = x >> 2;
  const int bpx = gridDim.x / 8;
  const int rank = (x & 3) * bpx + blockIdx.x / 8;
  const int nranks = 4 * bpx;
  const int *my = list + (size_t)aid * n;

  uint64_t acc = 0;
  for (int it = 0; it < iters; it++) {
    for (int i = rank * 4; i < n; i += nranks * 4) {
      const v4u *p0 = (const v4u *)(base + (size_t)my[i] * kPage);
      const v4u *p1 =
          (const v4u *)(base + (size_t)my[min(i + 1, n - 1)] * kPage);
      const v4u *p2 =
          (const v4u *)(base + (size_t)my[min(i + 2, n - 1)] * kPage);
      const v4u *p3 =
          (const v4u *)(base + (size_t)my[min(i + 3, n - 1)] * kPage);
      v4u a = __builtin_nontemporal_load(p0 + threadIdx.x);
      v4u b = __builtin_nontemporal_load(p1 + threadIdx.x);
      v4u c = __builtin_nontemporal_load(p2 + threadIdx.x);
      v4u d = __builtin_nontemporal_load(p3 + threadIdx.x);
      acc += a.x + b.x + c.x + d.x;
      acc += a.w + b.w + c.w + d.w;
    }
  }
  sink[blockIdx.x * 256 + threadIdx.x] = acc;
}

static double now_s() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  const size_t want_mib = (argc > 1) ? (size_t)std::atoll(argv[1]) : 1024;
  const int dev = 0;
  CHECK(hipSetDevice(dev));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, dev));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev));
  printf("device %s  CUs=%d\n", prop.gcnArchName, prop.multiProcessorCount);

  const size_t bytes = want_mib << 20;
  const size_t npages = bytes / kPage;
  char *buf = nullptr;
  CHECK(hipMalloc(&buf, bytes));
  CHECK(hipMemsetD8((hipDeviceptr_t)buf, 1, bytes));
  printf("hipMalloc %zu MiB at %p (%zu pages), 2 MiB backed, no VMM\n",
         want_mib, (void *)buf, npages);

  int *d_claim = nullptr;
  uint64_t *d_cyc = nullptr;
  CHECK(hipMalloc(&d_claim, 4096 * sizeof(int)));
  CHECK(hipMalloc(&d_cyc, npages * sizeof(uint64_t)));
  CHECK(hipMemset(d_claim, 0, 4096 * sizeof(int)));
  CHECK(hipMemset(d_cyc, 0, npages * sizeof(uint64_t)));
  CHECK(hipMemsetD8((hipDeviceptr_t)buf, 0, bytes));
  double t0 = now_s();
  classify_parallel<<<prop.multiProcessorCount, 64>>>(
      buf, npages, 48, 32, 2, d_claim, d_cyc, (uint64_t)wall_khz * 1000);
  CHECK(hipDeviceSynchronize());
  printf("classified %zu pages in %.2f s (%.2f us/page)\n", npages,
         now_s() - t0, (now_s() - t0) * 1e6 / npages);

  std::vector<uint64_t> cyc(npages);
  CHECK(hipMemcpy(cyc.data(), d_cyc, npages * sizeof(uint64_t),
                  hipMemcpyDeviceToHost));
  std::vector<double> ns(npages);
  for (size_t i = 0; i < npages; i++)
    ns[i] = (double)cyc[i] * 1e6 / wall_khz / 48;
  std::vector<double> srt(ns);
  std::sort(srt.begin(), srt.end());
  const double th = (srt[srt.size() / 10] + srt[srt.size() * 9 / 10]) / 2;
  std::vector<int> side0, side1;
  for (size_t i = 0; i < npages; i++)
    (ns[i] < th ? side0 : side1).push_back((int)i);
  const int n = (int)std::min(side0.size(), side1.size());
  printf("  homing: %zu on AID0, %zu on AID1 -> using %d per side\n",
         side0.size(), side1.size(), n);
  printf("  p10 %.0f ns  p90 %.0f ns  threshold %.0f ns\n",
         srt[srt.size() / 10], srt[srt.size() * 9 / 10], th);
  CHECK(hipMemsetD8((hipDeviceptr_t)buf, 1, bytes));

  // A [2][n] page list for an arbitrary remote fraction. Sweeping this is the
  // point: MI300X has 4 AIDs so its SPX/NPS1 baseline is 75% remote, while this
  // 2-AID MI350 sits at 50%. The paper's 20% is the 75% -> 0% move; ours would
  // be 50% -> 0%. Only the shape of the curve says whether those are
  // comparable.
  auto make_list = [&](double remote_frac) {
    std::vector<int> v(2 * n);
    for (int i = 0; i < n; i++) {
      bool far = (i % 100) < (int)(remote_frac * 100 + 0.5);
      v[i] = far ? side1[i] : side0[i];
      v[n + i] = far ? side0[i] : side1[i];
    }
    return v;
  };
  const double fracs[] = {0.0, 0.25, 0.50, 0.75, 1.0};
  int *d_list[5];
  for (int f = 0; f < 5; f++) {
    auto v = make_list(fracs[f]);
    CHECK(hipMalloc(&d_list[f], 2 * n * sizeof(int)));
    CHECK(hipMemcpy(d_list[f], v.data(), 2 * n * sizeof(int),
                    hipMemcpyHostToDevice));
  }
  int *d_local = d_list[0], *d_mixed = d_list[2], *d_remote = d_list[4];

  uint64_t *d_sink = nullptr;
  CHECK(hipMalloc(&d_sink, (size_t)4096 * 256 * sizeof(uint64_t)));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  const size_t moved = (size_t)2 * n * kPage;

  auto bw = [&](int *list, int grid, int iters) {
    gather_stream<<<grid, 256>>>(buf, list, n, 1, d_sink);
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    gather_stream<<<grid, 256>>>(buf, list, n, iters, d_sink);
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

  printf("\n%-6s %12s %12s %12s %9s %9s\n", "grid", "local(0%rem)",
         "mixed(50%)", "remote(100%)", "loc/mix", "loc/rem");
  for (int grid : {240, 480, 960, 1920, 3840}) {
    const int iters = std::max(4, (int)((16ull << 30) / moved));
    double l = med3(d_local, grid, iters);
    double m = med3(d_mixed, grid, iters);
    double r = med3(d_remote, grid, iters);
    printf("%-6d %9.1f GB/s %9.1f %9.1f %8.3fx %8.3fx\n", grid, l, m, r, l / m,
           l / r);
  }
  // The curve. MI300X/NPS1 sits at 75% remote, MI350/NPS1 at 50%.
  printf("\nbandwidth vs remote fraction (grid 1920), and what each\n"
         "baseline would gain by moving to 0%% remote:\n");
  printf("  %8s %12s %10s\n", "remote", "GB/s", "0%/this");
  double at0 = 0;
  for (int f = 0; f < 5; f++) {
    double v = med3(d_list[f], 1920, std::max(4, (int)((16ull << 30) / moved)));
    if (f == 0)
      at0 = v;
    printf("  %7.0f%% %9.1f GB/s %9.3fx%s\n", fracs[f] * 100, v, at0 / v,
           fracs[f] == 0.75   ? "   <- MI300X NPS1 (paper)"
           : fracs[f] == 0.50 ? "   <- MI350 NPS1 (this part)"
                              : "");
  }

  printf("\nfleet's MoE phase runs at 4270 GB/s (52.7 MiB / 12.9 us).\n");
  return 0;
}
