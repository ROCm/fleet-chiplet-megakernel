// End-to-end prototype of AID-local MoE weights, on fleet's real shape.
//
// One layer of gpt-oss-120b W13: 128 experts x 46 workgroups = 5888 slabs of
// 200192 B, the exact tensor the kernel reads. Two layouts, same total bytes,
// same read pattern:
//
//   baseline   slabs laid out contiguously, addresses computed (what fleet
//              does today). Homing is whatever the allocator gave: ~3/7 of
//              each slab on one AID and 4/7 on the other.
//   aid-local  each slab is padded to 49 pages and placed on pages homed on
//              the AID of the XCD that will read it, addresses resolved
//              through a page table.
//
// Slabs are assigned to AIDs in the ratio of the AIDs' *page* counts rather
// than 1:1. That is not cosmetic: MI350 NPS1 interleaves over 7 stacks (one is
// harvested) split 3/4 between the AIDs, so an even split starves the 4-stack
// side behind the 3-stack side and shows up as a ~10% locality penalty that is
// really load imbalance. The ratio has to be measured, not assumed, because
// which stack is harvested is per-part.
//
// What this decides: whether to restructure the kernel's tuned
// `buffer_load_dwordx4 ... lds` walk, which currently strides +0x400 through a
// contiguous slab and would have to consult the table every 4th chunk.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_moe_layout_bench.hip -o aid_moe_layout_bench
//   HIP_VISIBLE_DEVICES=0 ./aid_moe_layout_bench

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
static constexpr int kSlabBytes = 200192;              // W13_WG_BYTES
static constexpr int kSlabPages = (kSlabBytes + 4095) / 4096; // 49
static constexpr int kNSlabs = 128 * 46;               // experts x workgroups
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

// Each block consumes whole slabs, 16 B per thread per step, exactly as the
// kernel's wave-linear walk does. TABLE=false computes the address from the
// slab stride; TABLE=true resolves each 4 KiB page through the table. Slabs are
// handed to blocks by AID so that a block on AID a only ever touches slabs
// assigned to a.
template <bool TABLE>
__global__ __launch_bounds__(256) void read_slabs(
    const char *__restrict base, const int *__restrict table,
    const int *__restrict slab_of_aid0, int n0, const int *__restrict slab_of_aid1,
    int n1, int iters, uint64_t *__restrict sink) {
  const int xcc = (int)get_xcc_id();
  const int aid = xcc >> 2;
  const int per = gridDim.x / 8;
  const int rank = (xcc & 3) * per + blockIdx.x / 8;
  const int nranks = 4 * per;
  const int *mine = aid ? slab_of_aid1 : slab_of_aid0;
  const int nmine = aid ? n1 : n0;

  // 256 threads x 16 B is exactly one 4 KiB page, so the table is consulted
  // once per page and the index is block-uniform -- a scalar load, 4 B of
  // table per 4096 B of weights. Putting the lookup any deeper would cost more
  // in index traffic than locality could ever return.
  uint64_t acc = 0;
  for (int it = 0; it < iters; it++)
    for (int s = rank; s < nmine; s += nranks) {
      const int slab = mine[s];
      for (int p = 0; p < kSlabPages; p++) {
        const int off = p * (int)kPage + (int)threadIdx.x * 16;
        if (off >= kSlabBytes)
          break;
        size_t addr;
        if (TABLE) {
          const int pg = table[(size_t)slab * kSlabPages + p];
          addr = (size_t)pg * kPage + (size_t)threadIdx.x * 16;
        } else {
          addr = (size_t)slab * kSlabBytes + (size_t)off;
        }
        v4u d = __builtin_nontemporal_load((const v4u *)(base + addr));
        acc += d.x + d.w;
      }
    }
  sink[blockIdx.x * 256 + threadIdx.x] = acc;
}

static double now_s() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
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

  // Baseline buffer: slabs back to back, as packed today.
  const size_t base_bytes = (size_t)kNSlabs * kSlabBytes;
  // AID-local buffer: each slab padded to whole pages so placement is
  // possible, plus slack. An even slab split needs half the slabs on AID0, but
  // AID0 only owns 42.9% of the pages, so the pool has to be over-allocated to
  // offer that many.
  const size_t local_bytes =
      (size_t)(kNSlabs * 5 / 4) * kSlabPages * kPage;
  printf("one layer W13: %d slabs x %d B = %.2f GiB\n", kNSlabs, kSlabBytes,
         base_bytes / 1073741824.0);
  printf("padded to %d pages/slab -> %.2f GiB (+%.2f%%)\n\n", kSlabPages,
         local_bytes / 1073741824.0,
         100.0 * (local_bytes - base_bytes) / base_bytes);

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
  double t0 = now_s();
  classify_kernel<<<prop.multiProcessorCount, 64>>>(
      buf, npages, 48, 32, 2, d_claim, d_cyc, (uint64_t)wall_khz * 1000);
  CHECK(hipDeviceSynchronize());
  const double t_cls = now_s() - t0;

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
  printf("classified %zu pages in %.2f s (%.2f s extrapolated to 63 GiB)\n",
         npages, t_cls, t_cls * (63.0 * 1073741824.0 / local_bytes));
  printf("  AID0 %zu pages (%.1f%%), AID1 %zu (%.1f%%)  -> ratio %.3f : %.3f\n",
         free0.size(), 100.0 * free0.size() / npages, free1.size(),
         100.0 * free1.size() / npages, 7.0 * free0.size() / npages,
         7.0 * free1.size() / npages);

  const int cap0 = (int)(free0.size() / kSlabPages);
  const int cap1 = (int)(free1.size() / kSlabPages);

  // Two placements to compare.
  //
  //  prop  slabs split in the stack ratio, 3:4. Bytes per stack are equal so
  //        both AIDs finish together, but the AID1 XCDs then do ~30% more
  //        compute than the AID0 XCDs, and the tile map and barrier have to
  //        change to hand out uneven tile counts.
  //  even  slabs split 1:1. The uniform tile map and barrier are untouched and
  //        compute is balanced, but AID0's 3 stacks must carry half the
  //        traffic. That is only viable if 3 stacks can sustain fleet's
  //        per-side 2135 GB/s -- which is the thing being measured.
  int n0p = std::min((int)((double)kNSlabs * free0.size() / npages), cap0);
  int n1p = std::min(kNSlabs - n0p, cap1);
  int n0e = kNSlabs / 2, n1e = kNSlabs - n0e;
  if (n0e > cap0 || n1e > cap1) {
    printf("not enough pages for an even split (%d/%d vs cap %d/%d)\n", n0e,
           n1e, cap0, cap1);
    return 1;
  }
  printf("  proportional split: %d / %d (%.1f%% / %.1f%%)\n", n0p, n1p,
         100.0 * n0p / kNSlabs, 100.0 * n1p / kNSlabs);
  printf("  even split:         %d / %d (50.0%% / 50.0%%)\n\n", n0e, n1e);

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

  auto up = [&](const std::vector<int> &v) {
    int *d = nullptr;
    CHECK(hipMalloc(&d, v.size() * sizeof(int)));
    CHECK(hipMemcpy(d, v.data(), v.size() * sizeof(int), hipMemcpyHostToDevice));
    return d;
  };
  int *d_tab_p = up(tab_p), *d_sl0p = up(sl0p), *d_sl1p = up(sl1p);
  int *d_tab_e = up(tab_e), *d_sl0e = up(sl0e), *d_sl1e = up(sl1e);

  CHECK(hipMemsetD8((hipDeviceptr_t)buf, 1, local_bytes));
  uint64_t *d_sink = nullptr;
  CHECK(hipMalloc(&d_sink, (size_t)8192 * 256 * sizeof(uint64_t)));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  const size_t moved = (size_t)kNSlabs * kSlabBytes;
  const int iters = std::max(4, (int)((12ull << 30) / moved));

  // mode 0: today -- contiguous bytes, even work split
  // mode 1: AID-local, even split (uniform tile map, 3 stacks carry half)
  // mode 2: AID-local, 3:4 split  (needs a reworked tile map and barrier)
  auto run = [&](int mode, int grid) {
    auto launch = [&](int it) {
      if (mode == 0)
        read_slabs<false><<<grid, 256>>>(buf, d_tab_e, d_sl0e, (int)sl0e.size(),
                                         d_sl1e, (int)sl1e.size(), it, d_sink);
      else if (mode == 1)
        read_slabs<true><<<grid, 256>>>(buf, d_tab_e, d_sl0e, (int)sl0e.size(),
                                        d_sl1e, (int)sl1e.size(), it, d_sink);
      else
        read_slabs<true><<<grid, 256>>>(buf, d_tab_p, d_sl0p, (int)sl0p.size(),
                                        d_sl1p, (int)sl1p.size(), it, d_sink);
    };
    launch(1);
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    launch(iters);
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    return (double)moved * iters / (ms * 1e6);
  };

  // Clocks drift under sustained load, so the arms are interleaved inside each
  // round rather than measured one arm at a time -- otherwise a downclock
  // lands entirely on whichever arm happened to be running and shows up as a
  // difference between layouts.
  constexpr int kRounds = 7;
  auto measure = [&](int grid, double out[3]) {
    double v[3][kRounds];
    for (int k = 0; k < kRounds; k++)
      for (int m = 0; m < 3; m++)
        v[m][k] = run(m, grid);
    for (int m = 0; m < 3; m++) {
      std::sort(v[m], v[m] + kRounds);
      out[m] = v[m][kRounds / 2];
    }
  };

  // fleet's MoE phase reads 52.7 MiB in 12.9 us = 4270 GB/s. Sweep finely
  // enough to see which arm leads at that rate rather than interpolating
  // across the crossover, which sits close to it.
  const double kFleetGBs = 4270.0;
  printf("%-6s %13s %13s %9s %13s %9s\n", "grid", "today", "local even",
         "gain", "local 3:4", "gain");
  const int grids[] = {160, 240, 320, 400, 480, 560, 640,
                       720, 800, 960, 1280, 1920, 3840};
  const int ngrids = (int)(sizeof(grids) / sizeof(grids[0]));
  std::vector<double> gb(ngrids), ge(ngrids), gp(ngrids);
  int best_i = 0;
  double best_d = 1e18;
  for (int i = 0; i < ngrids; i++) {
    double o[3];
    measure(grids[i], o);
    gb[i] = o[0];
    ge[i] = o[1];
    gp[i] = o[2];
    if (std::abs(o[0] - kFleetGBs) < best_d) {
      best_d = std::abs(o[0] - kFleetGBs);
      best_i = i;
    }
  }
  for (int i = 0; i < ngrids; i++)
    printf("%-6d %10.1f GB/s %10.1f %8.3fx %10.1f %8.3fx%s\n", grids[i], gb[i],
           ge[i], ge[i] / gb[i], gp[i], gp[i] / gb[i],
           i == best_i ? "  <- fleet" : "");
  const int best_grid = grids[best_i];
  const double bb = gb[best_i], be = ge[best_i], bp = gp[best_i];
  const double mib = 52.7 * 1048576.0;
  printf("\nfleet's MoE phase reads 52.7 MiB at %.0f GB/s (12.9 us), which is\n"
         "4 experts x 46 slabs x (200192 + 100096) B. Closest sweep point is\n"
         "grid %d at %.0f GB/s:\n",
         kFleetGBs, best_grid, bb);
  printf("  today      %7.2f us\n", mib / (bb * 1e9) * 1e6);
  printf("  local even %7.2f us  %.3fx  uniform tile map and barrier unchanged\n",
         mib / (be * 1e9) * 1e6, be / bb);
  printf("  local 3:4  %7.2f us  %.3fx  needs a reworked tile map and barrier\n",
         mib / (bp * 1e9) * 1e6, bp / bb);
  printf("\nthe two arms cross over as AID0's 3 stacks saturate, so which one\n"
         "to build depends on where fleet actually sits, not on peak numbers.\n");
  return 0;
}
