// What is the AID interleave granularity of a plain hipMalloc buffer?
//
// This decides whether AID-local weights need any remapping at all. fleet reads
// MoE weights as contiguous ~200 KB per-workgroup slabs. If homing alternates
// every 4 KiB, a slab necessarily straddles both AIDs and no choice of reader
// can make it local -- you must relocate pages, which is what cost 32% in
// aid_moe_bw. If instead the runs are long compared to a slab, every slab is
// *already* entirely on one AID, and making the read local is just a matter of
// picking which XCD reads which expert: a tile-map change, free.
//
// Classify each 4 KiB page by ping-pong latency, then report the run-length
// distribution and the detected period rather than assuming either answer.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_interleave.hip -o aid_interleave
//   HIP_VISIBLE_DEVICES=4 ./aid_interleave [MiB]

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <map>
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

int main(int argc, char **argv) {
  const size_t want_mib = (argc > 1) ? (size_t)std::atoll(argv[1]) : 512;
  CHECK(hipSetDevice(0));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));

  const size_t bytes = want_mib << 20;
  const size_t npages = bytes / kPage;
  char *buf = nullptr;
  CHECK(hipMalloc(&buf, bytes));
  CHECK(hipMemsetD8((hipDeviceptr_t)buf, 0, bytes));
  printf("buffer %zu MiB at %p (%zu pages of 4 KiB)\n", want_mib, (void *)buf,
         npages);

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
  printf("classified in %.2f s\n", now_s() - t0);

  std::vector<uint64_t> cyc(npages);
  CHECK(hipMemcpy(cyc.data(), d_cyc, npages * sizeof(uint64_t),
                  hipMemcpyDeviceToHost));
  std::vector<double> ns(npages);
  for (size_t i = 0; i < npages; i++)
    ns[i] = (double)cyc[i] * 1e6 / wall_khz / 48;
  std::vector<double> srt(ns);
  std::sort(srt.begin(), srt.end());
  const double th = (srt[srt.size() / 10] + srt[srt.size() * 9 / 10]) / 2;
  std::vector<uint8_t> side(npages);
  size_t n0 = 0;
  for (size_t i = 0; i < npages; i++) {
    side[i] = ns[i] < th ? 0 : 1;
    n0 += side[i] == 0;
  }
  printf("threshold %.0f ns -> %zu near, %zu far (%.1f%% / %.1f%%)\n\n", th, n0,
         npages - n0, 100.0 * n0 / npages, 100.0 * (npages - n0) / npages);

  printf("first 96 pages (N=near AID, F=far AID), 48 per row:\n  ");
  for (int i = 0; i < 96 && i < (int)npages; i++) {
    putchar(side[i] ? 'F' : 'N');
    if (i % 48 == 47)
      printf("\n  ");
  }
  printf("\n");

  std::map<size_t, size_t> hist;
  size_t run = 1, maxrun = 1;
  for (size_t i = 1; i < npages; i++) {
    if (side[i] == side[i - 1])
      run++;
    else {
      hist[run]++;
      maxrun = std::max(maxrun, run);
      run = 1;
    }
  }
  hist[run]++;
  printf("\nrun-length distribution (consecutive pages on the same AID):\n");
  printf("  %10s %10s %10s %12s\n", "run(pages)", "run(KiB)", "count",
         "share of buf");
  size_t shown = 0;
  std::vector<std::pair<size_t, size_t>> byfreq(hist.begin(), hist.end());
  std::sort(byfreq.begin(), byfreq.end(),
            [](auto &a, auto &b) { return a.second * a.first > b.second * b.first; });
  for (auto &kv : byfreq) {
    if (shown++ >= 12)
      break;
    printf("  %10zu %10zu %10zu %11.2f%%\n", kv.first, kv.first * 4, kv.second,
           100.0 * kv.first * kv.second / npages);
  }
  printf("  longest run: %zu pages (%zu KiB)\n", maxrun, maxrun * 4);

  // A fleet MoE workgroup slab is 200192 B ~= 49 pages. How many such slabs are
  // single-AID at their natural alignment?
  const size_t slab_pages = (200192 + kPage - 1) / kPage;
  size_t pure = 0, total = 0;
  for (size_t s = 0; s + slab_pages <= npages; s += slab_pages) {
    bool same = true;
    for (size_t i = 1; i < slab_pages; i++)
      if (side[s + i] != side[s])
        same = false;
    pure += same;
    total++;
  }
  printf("\nfleet MoE slabs (200192 B = %zu pages) entirely on one AID: "
         "%zu / %zu (%.1f%%)\n",
         slab_pages, pure, total, 100.0 * pure / total);

  // And for a few candidate granularities, what fraction of aligned blocks are
  // single-AID? If some size G is clean, weights can be made local by choosing
  // the reader at G granularity with no page remapping at all.
  printf("\naligned-block purity by size:\n");
  printf("  %10s %14s\n", "block", "single-AID");
  for (size_t bp : {1ul, 2ul, 4ul, 8ul, 16ul, 32ul, 64ul, 128ul, 256ul, 512ul}) {
    size_t p = 0, t = 0;
    for (size_t s = 0; s + bp <= npages; s += bp) {
      bool same = true;
      for (size_t i = 1; i < bp; i++)
        if (side[s + i] != side[s])
          same = false;
      p += same;
      t++;
    }
    printf("  %7zu KiB %13.1f%%\n", bp * 4, 100.0 * p / t);
  }
  return 0;
}
