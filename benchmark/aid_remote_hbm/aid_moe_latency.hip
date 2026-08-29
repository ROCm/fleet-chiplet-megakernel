// Weight-load *latency* under fleet's MoE access pattern.
//
// Every earlier benchmark here measured throughput: a streaming loop run flat
// out, reported in GB/s. That cannot show a latency win. When the memory
// system is the limiter, cutting latency just means requests queue longer and
// the throughput number lands in the same place. NPS2 measured 400 ns -> 341 ns
// for AID-local access, and none of the GB/s sweeps could see it.
//
// What the kernel actually does is load a 23552 B tile as 23 chunks of 1 KiB
// and then `s_waitcnt vmcnt(0)`. The tile is ready when the *slowest* of the 23
// returns. A contiguous slab spans ~6 pages with 4/7 of them homed on the other
// AID, so almost every tile waits on a remote chunk; AID-local placement makes
// all 23 local and removes that tail. So the metric is time-to-complete one
// tile load, not aggregate bandwidth.
//
// Three layouts, and for each the load is issued the same way:
//   contiguous  what fleet does today, mixed homing
//   local even  slabs split 1:1, uniform tile map and barrier unchanged
//   local 3:4   slabs split in the stack ratio, needs a reworked tile map
//
// Latency is reported idle and under background traffic, because the number
// that matters is loaded latency at the pressure fleet's MoE phase actually
// generates.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_moe_latency.hip -o aid_moe_latency
//   HIP_VISIBLE_DEVICES=0 ./aid_moe_latency

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
static constexpr int kSlabBytes = 200192;
static constexpr int kSlabPages = (kSlabBytes + 4095) / 4096;
static constexpr int kNSlabs = 128 * 46;
static constexpr int kTileBytes = 23552; // W13_TILE_ROWS * (W13_K/2)
static constexpr int kChunks = kTileBytes / 1024; // 23, as the asm hardcodes
typedef unsigned int v4u __attribute__((ext_vector_type(4)));

__device__ inline unsigned get_xcc_id() {
  constexpr unsigned imm = ((4u - 1u) << 11) | (0u << 6) | 20u;
  return __builtin_amdgcn_s_getreg(imm);
}

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

// Address of chunk c of tile t, resolved through page indices already staged in
// LDS. 1 KiB divides 4 KiB, so a chunk never straddles a page and maps to
// exactly one entry.
//
// The lookup must not be a global load here. Its result is the address of the
// data load, so fetching it from HBM inside the timed region serializes a full
// memory round trip in front of every chunk -- which is a measurement of the
// table, not of locality. The real kernel knows the slab long before the load
// and can stage these the same way.
__device__ __forceinline__ const char *chunk_addr(const char *base,
                                                  const int *lds_pages, int c,
                                                  int tile, int lane) {
  const int off = tile * kTileBytes + c * 1024 + lane * 16;
  return base + (size_t)lds_pages[off >> 12] * kPage + (off & 4095);
}

// One wave times how long a whole tile takes to arrive: issue all 23 chunks,
// wait for every one, stop the clock. That is what the kernel's
// `s_waitcnt vmcnt(0)` makes the MFMA pipeline wait for.
//
// Blocks past `n_measure` generate background traffic instead, so latency can
// be sampled at a realistic level of memory pressure rather than on an idle
// part. They spin until the measuring blocks are done so the load stays up for
// the whole window.
__global__ __launch_bounds__(256) void latency_kernel(
    const char *__restrict base, const int *__restrict table,
    const int *__restrict sl0, int n0, const int *__restrict sl1, int n1,
    int n_measure, int samples, uint64_t *__restrict out,
    int *__restrict done_flag, uint64_t *__restrict sink,
    unsigned long long *__restrict bg_iters) {
  const int xcc = (int)get_xcc_id();
  const int aid = xcc >> 2;
  const int *mine = aid ? sl1 : sl0;
  const int nmine = aid ? n1 : n0;
  if (nmine == 0)
    return;
  const int lane = threadIdx.x & 63;
  const int warp = threadIdx.x >> 6;

  constexpr int kBatch = 8;
  __shared__ int lds_pages[kBatch * kSlabPages];

  if (blockIdx.x < n_measure) {
    // Batch a few tiles per timed region: wall_clock64 ticks at ~100 MHz, so a
    // single ~400 ns tile would land on only a handful of ticks.
    for (int s = 0; s < samples; s++) {
      const int slab0 = ((blockIdx.x * samples + s) * 7) % nmine;

      // Stage this batch's page indices before the clock starts, so the timed
      // region contains only the weight loads.
      for (int i = threadIdx.x; i < kBatch * kSlabPages; i += 256) {
        const int slab = mine[(slab0 + (i / kSlabPages) * 131) % nmine];
        lds_pages[i] = table[(size_t)slab * kSlabPages + i % kSlabPages];
      }
      __syncthreads();

      const uint64_t t0 = wall_clock64();
      for (int b = 0; b < kBatch; b++) {
        v4u acc[kChunks];
#pragma unroll
        for (int c = 0; c < kChunks; c++)
          acc[c] = __builtin_nontemporal_load((const v4u *)chunk_addr(
              base, lds_pages + b * kSlabPages, c, warp, lane));
        // Wait for every chunk, exactly as the tile load does.
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        uint64_t h = 0;
#pragma unroll
        for (int c = 0; c < kChunks; c++)
          h += acc[c].x;
        sink[blockIdx.x * 256 + threadIdx.x] += h;
      }
      const uint64_t t1 = wall_clock64();
      if (threadIdx.x == 0)
        out[blockIdx.x * samples + s] = (t1 - t0) / kBatch;
      __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0)
      __hip_atomic_fetch_add(done_flag, 1, __ATOMIC_RELEASE,
                             __HIP_MEMORY_SCOPE_AGENT);
  } else {
    // Background traffic exists only to load the memory system, so it uses the
    // efficient streaming shape: one block-uniform table load per page, and
    // 256 threads x 16 B covering that page exactly. Staging through LDS with
    // barriers here caps the grid around 2900 GB/s, below the rate the phase
    // being modelled actually reaches.
    uint64_t acc = 0;
    unsigned long long n = 0;
    int s = (blockIdx.x * 31) % nmine;
    while (__hip_atomic_load(done_flag, __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_AGENT) < n_measure) {
      const int slab = mine[s];
      for (int p = 0; p < kSlabPages; p++) {
        const int off = p * (int)kPage + (int)threadIdx.x * 16;
        if (off >= kSlabBytes)
          break;
        const int pg = table[(size_t)slab * kSlabPages + p];
        acc += __builtin_nontemporal_load(
                   (const v4u *)(base + (size_t)pg * kPage +
                                 (size_t)threadIdx.x * 16))
                   .x;
      }
      s = (s + 131) % nmine;
      n++;
    }
    sink[blockIdx.x * 256 + threadIdx.x] = acc;
    if (threadIdx.x == 0)
      atomicAdd(bg_iters, n);
  }
}

int main() {
  CHECK(hipSetDevice(0));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
  if (prop.multiProcessorCount / 32 != 8) {
    printf("needs SPX/NPS1 (8 XCDs)\n");
    return 1;
  }

  const size_t local_bytes = (size_t)(kNSlabs * 5 / 4) * kSlabPages * kPage;
  char *buf = nullptr;
  CHECK(hipMalloc(&buf, local_bytes));
  const size_t npages = local_bytes / kPage;

  int *d_claim = nullptr;
  uint64_t *d_cyc = nullptr;
  CHECK(hipMalloc(&d_claim, 4096 * sizeof(int)));
  CHECK(hipMalloc(&d_cyc, npages * sizeof(uint64_t)));
  CHECK(hipMemset(d_claim, 0, 4096 * sizeof(int)));
  CHECK(hipMemset(d_cyc, 0, npages * sizeof(uint64_t)));
  CHECK(hipMemsetD8((hipDeviceptr_t)buf, 0, local_bytes));
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
  std::vector<int> free0, free1;
  for (size_t i = 0; i < npages; i++)
    (ns[i] < th ? free0 : free1).push_back((int)i);
  printf("AID0 %zu pages (%.1f%%), AID1 %zu -> stacks %.2f : %.2f\n\n",
         free0.size(), 100.0 * free0.size() / npages, free1.size(),
         7.0 * free0.size() / npages, 7.0 * free1.size() / npages);

  const int cap0 = (int)(free0.size() / kSlabPages);
  const int cap1 = (int)(free1.size() / kSlabPages);
  int n0p = std::min((int)((double)kNSlabs * free0.size() / npages), cap0);
  int n1p = std::min(kNSlabs - n0p, cap1);
  int n0e = std::min(kNSlabs / 2, cap0), n1e = std::min(kNSlabs / 2, cap1);

  auto build = [&](int n0, int n1, std::vector<int> &table,
                   std::vector<int> &sl0, std::vector<int> &sl1) {
    table.assign((size_t)kNSlabs * kSlabPages, 0);
    sl0.clear();
    sl1.clear();
    size_t t0 = 0, t1 = 0;
    for (int s = 0; s < n0 + n1; s++) {
      const bool on0 = (s < n0);
      (on0 ? sl0 : sl1).push_back(s);
      for (int p = 0; p < kSlabPages; p++)
        table[(size_t)s * kSlabPages + p] = on0 ? free0[t0++] : free1[t1++];
    }
  };
  std::vector<int> tab_p, sl0p, sl1p, tab_e, sl0e, sl1e;
  build(n0p, n1p, tab_p, sl0p, sl1p);
  build(n0e, n1e, tab_e, sl0e, sl1e);

  // Baseline goes through the identity table: slab s keeps its natural pages,
  // so homing is whatever the allocator gave, but the addressing path is the
  // same as the AID-local arms. Without this the comparison would be
  // indirection versus none rather than remote versus local.
  std::vector<int> tab_id((size_t)kNSlabs * kSlabPages);
  for (size_t i = 0; i < tab_id.size(); i++)
    tab_id[i] = (int)i;
  auto up = [&](const std::vector<int> &v) {
    int *d = nullptr;
    CHECK(hipMalloc(&d, v.size() * sizeof(int)));
    CHECK(hipMemcpy(d, v.data(), v.size() * sizeof(int), hipMemcpyHostToDevice));
    return d;
  };
  int *d_tab_p = up(tab_p), *d_sl0p = up(sl0p), *d_sl1p = up(sl1p);
  int *d_tab_e = up(tab_e), *d_sl0e = up(sl0e), *d_sl1e = up(sl1e);
  int *d_tab_id = up(tab_id);
  CHECK(hipMemsetD8((hipDeviceptr_t)buf, 1, local_bytes));

  const int kMeasure = 8, kSamples = 256;
  uint64_t *d_out = nullptr, *d_sink = nullptr;
  int *d_done = nullptr;
  CHECK(hipMalloc(&d_out, (size_t)kMeasure * kSamples * sizeof(uint64_t)));
  CHECK(hipMalloc(&d_sink, (size_t)8192 * 256 * sizeof(uint64_t)));
  CHECK(hipMalloc(&d_done, sizeof(int)));

  unsigned long long *d_iters = nullptr;
  CHECK(hipMalloc(&d_iters, sizeof(unsigned long long)));
  hipEvent_t ev0, ev1;
  CHECK(hipEventCreate(&ev0));
  CHECK(hipEventCreate(&ev1));

  // ns for one tile: 23 chunks x 1 KiB, issued together, all waited on.
  // Also reports the aggregate rate the whole grid sustained during the
  // window, so the row matching fleet's 4270 GB/s can be identified rather
  // than guessed at from block counts.
  auto run = [&](int mode, int grid, double pct[3], double *gbs) {
    CHECK(hipMemset(d_done, 0, sizeof(int)));
    CHECK(hipMemset(d_iters, 0, sizeof(unsigned long long)));
    CHECK(hipMemset(d_out, 0,
                    (size_t)kMeasure * kSamples * sizeof(uint64_t)));
    CHECK(hipEventRecord(ev0));
    if (mode == 0)
      latency_kernel<<<grid, 256>>>(buf, d_tab_id, d_sl0e, (int)sl0e.size(),
                                    d_sl1e, (int)sl1e.size(), kMeasure,
                                    kSamples, d_out, d_done, d_sink, d_iters);
    else if (mode == 1)
      latency_kernel<<<grid, 256>>>(buf, d_tab_e, d_sl0e, (int)sl0e.size(),
                                    d_sl1e, (int)sl1e.size(), kMeasure,
                                    kSamples, d_out, d_done, d_sink, d_iters);
    else
      latency_kernel<<<grid, 256>>>(buf, d_tab_p, d_sl0p, (int)sl0p.size(),
                                    d_sl1p, (int)sl1p.size(), kMeasure,
                                    kSamples, d_out, d_done, d_sink, d_iters);
    CHECK(hipEventRecord(ev1));
    CHECK(hipDeviceSynchronize());
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, ev0, ev1));
    unsigned long long iters = 0;
    CHECK(hipMemcpy(&iters, d_iters, sizeof(iters), hipMemcpyDeviceToHost));
    // Background: one whole slab per iteration. Measuring: 4 waves x one tile
    // each, kBatch tiles per sample.
    const double bytes = (double)iters * kSlabBytes +
                         (double)kMeasure * kSamples * 8 * 4.0 * kTileBytes;
    if (gbs)
      *gbs = bytes / (ms * 1e6);
    std::vector<uint64_t> h((size_t)kMeasure * kSamples);
    CHECK(hipMemcpy(h.data(), d_out, h.size() * sizeof(uint64_t),
                    hipMemcpyDeviceToHost));
    std::vector<double> v;
    for (uint64_t c : h)
      if (c)
        v.push_back((double)c * 1e6 / wall_khz);
    std::sort(v.begin(), v.end());
    if (v.empty()) {
      pct[0] = pct[1] = pct[2] = 0;
      return;
    }
    pct[0] = v[v.size() / 10];
    pct[1] = v[v.size() / 2];
    pct[2] = v[v.size() * 9 / 10];
  };

  printf("tile-load completion latency (p50, ns) and the rate the grid\n"
         "sustained while it was measured. 23 KiB tile, all chunks waited on.\n\n");
  printf("%-6s %18s %18s %18s %8s\n", "bg", "contiguous (today)", "local even",
         "local 3:4", "latency");
  printf("%-6s %8s %9s %8s %9s %8s %9s %8s\n", "blocks", "ns", "GB/s", "ns",
         "GB/s", "ns", "GB/s", "gain");
  const double kFleetGBs = 4270.0;
  int best = -1;
  double bd = 1e18;
  struct Row {
    int bg;
    double a, ag, b, bg_, c, cg;
  };
  std::vector<Row> rows;
  for (int bg : {0, 120, 240, 360, 480, 640, 800, 960, 1440, 1920, 3832}) {
    const int grid = kMeasure + bg;
    double a[3], b[3], c[3], ag = 0, bgs = 0, cg = 0;
    run(0, grid, a, &ag);
    run(1, grid, b, &bgs);
    run(2, grid, c, &cg);
    rows.push_back({bg, a[1], ag, b[1], bgs, c[1], cg});
    if (std::abs(ag - kFleetGBs) < bd) {
      bd = std::abs(ag - kFleetGBs);
      best = (int)rows.size() - 1;
    }
  }
  for (size_t i = 0; i < rows.size(); i++) {
    const Row &r = rows[i];
    printf("%-6d %8.0f %9.0f %8.0f %9.0f %8.0f %9.0f %8.3fx%s\n", r.bg, r.a,
           r.ag, r.b, r.bg_, r.c, r.cg, r.a / r.b,
           (int)i == best ? "  <- fleet" : "");
  }
  const Row &f = rows[best];
  printf("\nfleet's MoE phase reads 52.7 MiB in 12.9 us = %.0f GB/s. Closest\n"
         "row is %.0f GB/s:\n",
         kFleetGBs, f.ag);
  printf("  contiguous  %6.0f ns   %.0f GB/s\n", f.a, f.ag);
  printf("  local even  %6.0f ns   %.0f GB/s   latency %.3fx, bandwidth %.3fx\n",
         f.b, f.bg_, f.a / f.b, f.bg_ / f.ag);
  printf("  local 3:4   %6.0f ns   %.0f GB/s   latency %.3fx, bandwidth %.3fx\n",
         f.c, f.cg, f.a / f.c, f.cg / f.ag);
  printf("\nA tile is ready only when the slowest of its 23 chunks lands. A\n"
         "contiguous slab spans ~6 pages with 4/7 homed across the link, so\n"
         "nearly every tile waits on a remote chunk; AID-local placement makes\n"
         "all 23 local. The requirement is latency down without bandwidth\n"
         "dropping below what the phase needs.\n");
  return 0;
}
