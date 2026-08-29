// Does AID-local placement reproduce NPS2's latency without partitioning?
//
// aid_nps_compare measured the same dependent chase under both partition modes
// and found SPX/NPS1 ~400 ns with a visibly bimodal spread, DPX/NPS2 ~341 ns
// and unimodal -- NPS2 wins because a partition's memory is all homed on its
// own AID. If software placement really lands weights on the reading XCD's AID,
// running that same chase on SPX over an AID-local chain has to give the same
// ~341 ns. If it does not, the loads are not going where they are supposed to.
//
// So this is aid_nps_compare's chase, unchanged in every parameter that matters
// -- 64 B lines, 2048 dependent hops, one thread per worker, only XCDs 0-3
// active -- with one thing varied: which pages the chain runs through.
//
//   mixed      every page of the buffer, i.e. what any hipMalloc gives today.
//              XCDs 0-3 are on AID0, so ~57% of hops cross to AID1.
//   aid-local  only pages homed on AID0. XCDs 0-3 are on AID0, so no hop
//              crosses. This is the software equivalent of an NPS2 partition.
//
// Restricting to XCDs 0-3 is what makes the comparison exact: on SPX those four
// XCDs are precisely AID0, the same four an NPS2 partition would have had.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_local_chase.hip -o aid_local_chase
//   HIP_VISIBLE_DEVICES=0 ./aid_local_chase [GiB] [K]

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
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

static constexpr int kActiveXCD = 4; // XCDs 0-3, which on SPX are AID0
static constexpr size_t kLine = 64;
static constexpr size_t kPage = 4096;
static constexpr size_t kLinesPerPage = kPage / kLine; // 64
static constexpr uint64_t kStride = 1572869;           // as in aid_nps_compare

__device__ inline unsigned get_xcc_id() {
  constexpr unsigned imm = ((4u - 1u) << 11) | (0u << 6) | 20u;
  return __builtin_amdgcn_s_getreg(imm);
}

// Page-homing classifier, same ping-pong timing used everywhere else here.
__global__ void classify_kernel(char *base, size_t npages, int rounds,
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

// Byte offset of the i'th line of the chain. `pages` selects which pages the
// chain is allowed to use, so the same code lays down the mixed chain (pages =
// all) and the AID-local one (pages = AID0 only).
__device__ __forceinline__ uint64_t line_off(const int *pages, uint64_t L) {
  return (uint64_t)pages[L / kLinesPerPage] * kPage + (L % kLinesPerPage) * kLine;
}

__global__ void build_chain(char *__restrict base, const int *__restrict pages,
                            uint64_t nlines) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nlines)
    return;
  uint64_t cur = (i * kStride) % nlines;
  uint64_t nxt = ((i + 1) * kStride) % nlines;
  *(uint64_t *)(base + line_off(pages, cur)) = line_off(pages, nxt);
}

__global__ __launch_bounds__(256) void chase(const char *__restrict base,
                                             const int *__restrict pages,
                                             uint64_t nlines, int nxcd,
                                             int hops, uint64_t *__restrict out,
                                             uint64_t *__restrict sink) {
  const int xcc = (int)get_xcc_id();
  if (xcc >= kActiveXCD || threadIdx.x != 0)
    return;
  const int k = blockIdx.x / nxcd;
  const int worker = xcc * (gridDim.x / nxcd) + k;

  uint64_t idx = line_off(pages, ((uint64_t)worker * (nlines / 64 + 1)) % nlines);
  uint64_t t0 = wall_clock64();
  for (int h = 0; h < hops; h++)
    idx = *(const uint64_t *)(base + idx);
  uint64_t t1 = wall_clock64();
  out[worker] = (t1 - t0) / (uint64_t)hops;
  sink[worker] = idx;
}

int main(int argc, char **argv) {
  const size_t gib = (argc > 1) ? (size_t)std::atoll(argv[1]) : 16;
  const int K = (argc > 2) ? std::atoi(argv[2]) : 32;

  CHECK(hipSetDevice(0));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
  const int nxcd = prop.multiProcessorCount / 32;
  if (nxcd != 8) {
    printf("run this under SPX/NPS1; the point is to get NPS2's latency "
           "without partitioning (found %d XCDs)\n",
           nxcd);
    return 1;
  }
  const int grid = nxcd * K;
  const int active = kActiveXCD * K;
  printf("CUs %d -> %d XCDs (SPX/NPS1), grid %d, active workers %d on XCDs "
         "0-%d\n",
         prop.multiProcessorCount, nxcd, grid, active, kActiveXCD - 1);

  const size_t bytes = gib << 30;
  char *buf = nullptr;
  CHECK(hipMalloc(&buf, bytes));
  const size_t npages = bytes / kPage;

  int *d_claim = nullptr;
  uint64_t *d_cyc = nullptr;
  CHECK(hipMalloc(&d_claim, 4096 * sizeof(int)));
  CHECK(hipMalloc(&d_cyc, npages * sizeof(uint64_t)));
  CHECK(hipMemset(d_claim, 0, 4096 * sizeof(int)));
  CHECK(hipMemset(d_cyc, 0, npages * sizeof(uint64_t)));
  CHECK(hipMemsetD8((hipDeviceptr_t)buf, 0, bytes));
  classify_kernel<<<prop.multiProcessorCount, 64>>>(
      buf, npages, 48, 32, 2, d_claim, d_cyc, (uint64_t)wall_khz * 1000);
  CHECK(hipDeviceSynchronize());
  std::vector<uint64_t> cyc(npages);
  CHECK(hipMemcpy(cyc.data(), d_cyc, npages * sizeof(uint64_t),
                  hipMemcpyDeviceToHost));
  std::vector<double> ns(npages);
  for (size_t i = 0; i < npages; i++)
    ns[i] = (double)cyc[i] * 1e6 / wall_khz / 48;
  std::vector<double> srt(ns);
  std::sort(srt.begin(), srt.end());
  const double th = (srt[srt.size() / 10] + srt[srt.size() * 9 / 10]) / 2;
  std::vector<int> all(npages), aid0, aid0_strict;
  std::iota(all.begin(), all.end(), 0);
  for (size_t i = 0; i < npages; i++)
    if (ns[i] < th)
      aid0.push_back((int)i);
  // The threshold is only the midpoint of the p10/p90 ping-pong times, so pages
  // near it are the ones most likely mislabelled. Keep the clearly-fast third
  // as a control: if the residual gap to NPS2 is misclassification, this arm
  // closes it.
  const double th_strict = srt[srt.size() / 4];
  for (size_t i = 0; i < npages; i++)
    if (ns[i] < th_strict)
      aid0_strict.push_back((int)i);
  printf("buffer %zu GiB, %zu pages: %zu on AID0 (%.1f%%), %zu high-confidence "
         "(%.1f%%)\n\n",
         gib, npages, aid0.size(), 100.0 * aid0.size() / npages,
         aid0_strict.size(), 100.0 * aid0_strict.size() / npages);

  uint64_t *d_out = nullptr, *d_sink = nullptr;
  CHECK(hipMalloc(&d_out, (size_t)grid * sizeof(uint64_t)));
  CHECK(hipMalloc(&d_sink, (size_t)grid * 256 * sizeof(uint64_t)));

  auto measure = [&](const char *name, const std::vector<int> &pages) {
    int *d_pages = nullptr;
    CHECK(hipMalloc(&d_pages, pages.size() * sizeof(int)));
    CHECK(hipMemcpy(d_pages, pages.data(), pages.size() * sizeof(int),
                    hipMemcpyHostToDevice));
    uint64_t nlines = pages.size() * kLinesPerPage;
    while (std::__gcd(nlines, kStride) != 1)
      nlines--; // keep the walk a single cycle, as in aid_nps_compare
    CHECK(hipMemset(d_out, 0, (size_t)grid * sizeof(uint64_t)));
    build_chain<<<(unsigned)((nlines + 255) / 256), 256>>>(buf, d_pages,
                                                           nlines);
    CHECK(hipDeviceSynchronize());
    const int hops = 2048;
    chase<<<grid, 256>>>(buf, d_pages, nlines, nxcd, hops, d_out, d_sink);
    CHECK(hipDeviceSynchronize());
    std::vector<uint64_t> c(grid);
    CHECK(hipMemcpy(c.data(), d_out, (size_t)grid * sizeof(uint64_t),
                    hipMemcpyDeviceToHost));
    std::vector<double> lat;
    for (int i = 0; i < active; i++)
      if (c[i])
        lat.push_back((double)c[i] * 1e9 / (wall_khz * 1000.0));
    std::sort(lat.begin(), lat.end());
    double sum = 0;
    for (double v : lat)
      sum += v;
    const double p10 = lat[lat.size() / 10], p50 = lat[lat.size() / 2],
                 p90 = lat[lat.size() * 9 / 10];
    printf("%-10s working set %5.1f GiB  mean %6.1f ns  p10 %6.1f  p50 %6.1f  "
           "p90 %6.1f  spread %.3fx\n",
           name, pages.size() * kPage / 1073741824.0, sum / lat.size(), p10,
           p50, p90, p90 / p10);
    CHECK(hipFree(d_pages));
    return sum / lat.size();
  };

  printf("dependent chase, 2048 hops, %d workers on XCDs 0-3 (= AID0)\n",
         active);
  const double m = measure("mixed", all);
  const double l = measure("aid-local", aid0);
  const double s = measure("aid-local+", aid0_strict);
  printf("\naid-local is %.3fx lower latency than mixed on the same part, in\n"
         "the same partition mode, with only the chain's pages changed.\n",
         m / l);
  // A hop is either local or remote; solving the mean against those two ends
  // gives the fraction still crossing, which is the classifier's error rate.
  const double local_ns = s, remote_ns = 2.0 * m - s;
  if (remote_ns > local_ns)
    printf("taking aid-local+ as the local latency, the aid-local mean implies\n"
           "%.1f%% of its hops still cross to the far AID.\n",
           100.0 * (l - local_ns) / (remote_ns - local_ns));
  printf("\nreference, aid_nps_compare on this same chase:\n"
         "  SPX/NPS1  ~400 ns, spread 1.05-1.18 (bimodal: near and far AID)\n"
         "  DPX/NPS2  ~341 ns, spread 1.05-1.09 (unimodal: all local)\n"
         "aid-local should land on the NPS2 numbers. If it does not, the loads\n"
         "are not reaching the local stacks.\n");
  return 0;
}
