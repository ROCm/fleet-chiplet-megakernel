// Is page-indirection cheap enough to pay for locality?
//
// Since homing is neither contiguous (aid_interleave: 0 of 10699 weight slabs
// are single-AID) nor positional (aid_period: two allocations agree only 71.4%
// at the same offset), the only remaining way to give a workgroup AID-local
// weights is to classify the buffer once at load, permute the weights so each
// workgroup's bytes sit on its own AID's pages, and have the kernel read
// through a page-index list.
//
// That is only worth doing if the indirection costs less than the locality
// gains, and previous attempts died exactly here: VMM cost 32% to buy 5%.
// Comparing a gather kernel against a streaming kernel would repeat the
// mistake, since the two differ in more than indirection.
//
// So: one kernel, three arms, each differing from the previous in exactly one
// thing.
//
//   contiguous  page = aid*n + i        natural ~50/50 homing, no index load
//   indexed     page = ident[aid*n + i] *the same pages in the same order*,
//                                       and the identity list makes the index
//                                       load the only difference -> its cost
//   local       page = local[aid*n + i] AID-sorted pages, so 0% remote -> the
//                                       gain, measured through the same
//                                       indirection
//
// contiguous->indexed is the mechanism's price; indexed->local is what it buys.
// The mechanism is viable iff the second exceeds the first.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_indirect_cost.hip -o aid_indirect_cost
//   HIP_VISIBLE_DEVICES=0 ./aid_indirect_cost [MiB]

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

// INDEXED=false computes the page number arithmetically; INDEXED=true loads it.
// Everything else is identical, so the delta is the index load and nothing else.
//
// n0/n1 are per-AID page counts and need not be equal. That matters: MI350
// NPS1 interleaves over *7* stacks (one is harvested), so the two AIDs own 3
// and 4 of them. Handing both XCD groups the same page count therefore starves
// the 4-stack side waiting on the 3-stack side, and the resulting imbalance
// looks exactly like a locality penalty. Sizing work 3:4 removes it.
template <bool INDEXED>
__global__ __launch_bounds__(256) void read_pages(
    const char *__restrict base, const int *__restrict list, int n0, int n1,
    int iters, uint64_t *__restrict sink) {
  const int xcc = (int)get_xcc_id();
  const int aid = xcc >> 2;
  const int per = gridDim.x / 8;
  const int rank = (xcc & 3) * per + blockIdx.x / 8;
  const int nranks = 4 * per;
  const int n = aid ? n1 : n0;
  const size_t abase = aid ? (size_t)n0 : 0;
  const int *my = list + abase;

  uint64_t acc = 0;
  for (int it = 0; it < iters; it++)
    for (int i = rank * 8; i < n; i += nranks * 8) {
      v4u v[8];
#pragma unroll
      for (int u = 0; u < 8; u++) {
        const int j = min(i + u, n - 1);
        const size_t page = INDEXED ? (size_t)my[j] : (abase + (size_t)j);
        const v4u *p = (const v4u *)(base + page * kPage);
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
  if (prop.multiProcessorCount / 32 != 8) {
    printf("needs SPX/NPS1 (8 XCDs); found %d CUs\n",
           prop.multiProcessorCount);
    return 1;
  }

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
  const double t_classify = now_s() - t0;

  std::vector<uint64_t> cyc(npages);
  CHECK(hipMemcpy(cyc.data(), d_cyc, npages * sizeof(uint64_t),
                  hipMemcpyDeviceToHost));
  std::vector<double> ns(npages);
  for (size_t i = 0; i < npages; i++)
    ns[i] = (double)cyc[i] * 1e6 / wall_khz / 48;
  std::vector<double> srt(ns);
  std::sort(srt.begin(), srt.end());
  const double th = (srt[srt.size() / 10] + srt[srt.size() * 9 / 10]) / 2;
  std::vector<int> a0, a1;
  for (size_t i = 0; i < npages; i++)
    (ns[i] < th ? a0 : a1).push_back((int)i);
  const int n = (int)std::min(a0.size(), a1.size());
  printf("%zu MiB, %zu pages: %zu on AID0, %zu on AID1 -> n=%d per side\n",
         want_mib, npages, a0.size(), a1.size(), n);
  printf("classify %.2f s (%.2f us/page) -> %.0f s extrapolated to 63 GiB of "
         "weights\n\n",
         t_classify, t_classify * 1e6 / npages,
         t_classify * (63.0 * 1024 / want_mib));
  CHECK(hipMemsetD8((hipDeviceptr_t)buf, 1, bytes));

  // Balanced arms give each AID the same page count; the proportional arm gives
  // each AID all of its own pages, so work tracks the 3:4 stack split.
  const int p0 = (int)a0.size(), p1 = (int)a1.size();
  std::vector<int> ident(2 * n), local_bal(2 * n), local_prop(p0 + p1);
  for (int i = 0; i < n; i++) {
    ident[i] = i;
    ident[n + i] = n + i;
    local_bal[i] = a0[i];
    local_bal[n + i] = a1[i];
  }
  for (int i = 0; i < p0; i++)
    local_prop[i] = a0[i];
  for (int i = 0; i < p1; i++)
    local_prop[p0 + i] = a1[i];

  int *d_ident = nullptr, *d_bal = nullptr, *d_prop = nullptr;
  CHECK(hipMalloc(&d_ident, 2 * n * sizeof(int)));
  CHECK(hipMalloc(&d_bal, 2 * n * sizeof(int)));
  CHECK(hipMalloc(&d_prop, (size_t)(p0 + p1) * sizeof(int)));
  CHECK(hipMemcpy(d_ident, ident.data(), 2 * n * sizeof(int),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_bal, local_bal.data(), 2 * n * sizeof(int),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_prop, local_prop.data(), (size_t)(p0 + p1) * sizeof(int),
                  hipMemcpyHostToDevice));

  uint64_t *d_sink = nullptr;
  CHECK(hipMalloc(&d_sink, (size_t)8192 * 256 * sizeof(uint64_t)));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));

  // mode: 0 contiguous, 1 indexed-identity, 2 local-balanced, 3 local-3:4
  auto run = [&](int mode, int grid, double *ns_per_page) {
    const int q0 = (mode == 3) ? p0 : n;
    const int q1 = (mode == 3) ? p1 : n;
    const size_t moved = (size_t)(q0 + q1) * kPage;
    const int iters = std::max(4, (int)((16ull << 30) / moved));
    auto launch = [&](int it) {
      if (mode == 0)
        read_pages<false><<<grid, 256>>>(buf, d_ident, q0, q1, it, d_sink);
      else if (mode == 1)
        read_pages<true><<<grid, 256>>>(buf, d_ident, q0, q1, it, d_sink);
      else if (mode == 2)
        read_pages<true><<<grid, 256>>>(buf, d_bal, q0, q1, it, d_sink);
      else
        read_pages<true><<<grid, 256>>>(buf, d_prop, q0, q1, it, d_sink);
    };
    double v[3];
    for (int k = 0; k < 3; k++) {
      launch(1);
      CHECK(hipDeviceSynchronize());
      CHECK(hipEventRecord(e0));
      launch(iters);
      CHECK(hipEventRecord(e1));
      CHECK(hipEventSynchronize(e1));
      float ms = 0;
      CHECK(hipEventElapsedTime(&ms, e0, e1));
      v[k] = (double)moved * iters / (ms * 1e6);
    }
    std::sort(v, v + 3);
    if (ns_per_page) {
      // Time for one page-read per active rank: the latency-sensitive view.
      const double bytes_per_rank_trip = 4.0 * kPage * 256.0 / 256.0;
      *ns_per_page = bytes_per_rank_trip / (v[1] * 1e9) * 1e9 * (4 * grid / 8);
    }
    return v[1];
  };

  printf("%-6s %12s %12s %12s %12s %9s %9s\n", "grid", "contiguous",
         "indexed", "local 1:1", "local 3:4", "idx cost", "3:4 gain");
  for (int grid : {240, 480, 960, 1920, 3840}) {
    double c = run(0, grid, nullptr);
    double x = run(1, grid, nullptr);
    double lb = run(2, grid, nullptr);
    double lp = run(3, grid, nullptr);
    printf("%-6d %9.1f GB/s %9.1f %9.1f %9.1f %8.3fx %8.3fx\n", grid, c, x, lb,
           lp, c / x, lp / x);
  }
  printf("\nlocal 1:1 hands both AIDs equal work and so is limited by the\n"
         "3-stack side. local 3:4 sizes work to the stack split. idx cost =\n"
         "contiguous/indexed (index load only); 3:4 gain = local3:4/indexed\n"
         "(homing only). Low grid is the latency-sensitive regime, which is\n"
         "where fleet's MoE phase sits at 74%% of achievable bandwidth.\n");
  return 0;
}
