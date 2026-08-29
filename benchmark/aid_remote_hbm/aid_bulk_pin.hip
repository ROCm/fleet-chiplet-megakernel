// Is AID pinning affordable at model-weight scale?
//
// aid_alloc.hip proved a buffer can be pinned to one AID, but at a cost tuned
// for a 256 MiB probe: one wave pair walks every page in turn, 45 us/page. The
// MoE weights of gpt-oss-120b are 36 layers x 1.77 GB = 64 GB, or 15.5M pages,
// which at that rate is 12 minutes of probing before a single token is decoded
// -- and that is before the VMM calls themselves are counted.
//
// This measures the three rates that decide whether the full-model version is
// worth building, and extrapolates each to 64 GB:
//
//   A. hipMemCreate, at 4 KiB and at 2 MiB. One driver call per handle, so
//      15.5M small handles may dominate everything else on its own.
//   B. whether hipMemMap accepts a non-zero offset into a large handle. If it
//      does, the handle count collapses to a few dozen and only the map calls
//      scale with pages. CUDA historically required this to be zero.
//   C. hipMemMap, at 4 KiB. Two per page in the real flow (probe, then place).
//   D. a parallel classifier: N independent wave pairs instead of one. Pages
//      are independent addresses, so pairs need no barrier between them. Both
//      throughput and agreement against the serial classifier are reported --
//      speed is worthless if contention blurs the bimodal split it thresholds.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_bulk_pin.hip -o aid_bulk_pin
//   HIP_VISIBLE_DEVICES=1 ./aid_bulk_pin [probe_pages]

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
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
static constexpr size_t kMoeBytes = 64ull << 30; // 36 layers of MoE weights
static constexpr size_t kMoePages = kMoeBytes / kPage;

__device__ inline unsigned get_xcc_id() {
  constexpr unsigned imm = ((4u - 1u) << 11) | (0u << 6) | 20u;
  return __builtin_amdgcn_s_getreg(imm);
}

// Serial reference, as in aid_alloc.hip: one wave on each XCD, all pages.
__global__ void classify_serial(char *base, size_t npages, int rounds,
                                int xcc_a, int xcc_b, int *claim,
                                uint64_t *cycles) {
  int x = (int)get_xcc_id();
  if ((x != xcc_a && x != xcc_b) || threadIdx.x != 0)
    return;
  int slot = (x == xcc_a) ? 0 : 1;
  if (atomicCAS(&claim[slot], 0, 1) != 0)
    return;

  for (size_t p = 0; p < npages; p++) {
    int *flag = (int *)(base + p * kPage);
    if (slot == 0) {
      uint64_t t0 = wall_clock64();
      for (int r = 0; r < rounds; r++) {
        while (__hip_atomic_load(flag, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT) != r * 2) {
        }
        __hip_atomic_store(flag, r * 2 + 1, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
      }
      cycles[p] = wall_clock64() - t0;
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
}

// Parallel: `sets` disjoint XCD pairs, `per_set` wave pairs inside each, every
// pair walking its own stride of the page list. A pair that loses its partner
// would spin forever and wedge the device, so every wait is bounded and a
// timed-out page is reported as 0 rather than retried.
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

static void map_page(void *va, size_t off, hipMemGenericAllocationHandle_t h,
                     size_t handle_off, int dev) {
  CHECK(hipMemMap((char *)va + off, kPage, handle_off, h, 0));
  hipMemAccessDesc acc{};
  acc.location.type = hipMemLocationTypeDevice;
  acc.location.id = dev;
  acc.flags = hipMemAccessFlagsProtReadWrite;
  CHECK(hipMemSetAccess((char *)va + off, kPage, &acc, 1));
}

static double now_s() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void extrapolate(const char *what, double per_op_us, size_t ops) {
  double s = per_op_us * 1e-6 * ops;
  printf("      -> %.1f s for %zu ops over 64 GB%s\n", s, ops,
         s > 120 ? "   [PROHIBITIVE]" : (s > 20 ? "   [heavy]" : "   [ok]"));
}

int main(int argc, char **argv) {
  const size_t probe_pages = (argc > 1) ? (size_t)std::atoll(argv[1]) : 65536;
  const int dev = 0;
  CHECK(hipSetDevice(dev));

  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, dev));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev));
  size_t freeb = 0, totalb = 0;
  CHECK(hipMemGetInfo(&freeb, &totalb));
  printf("device %s  CUs=%d  free=%.1f GiB / %.1f GiB\n", prop.gcnArchName,
         prop.multiProcessorCount, freeb / 1073741824.0, totalb / 1073741824.0);
  printf("MoE weight target: %.0f GiB = %zu pages\n\n", kMoeBytes / 1073741824.0,
         kMoePages);

  hipMemAllocationProp aprop{};
  aprop.type = hipMemAllocationTypePinned;
  aprop.location.type = hipMemLocationTypeDevice;
  aprop.location.id = dev;

  size_t gran = 0;
  CHECK(hipMemGetAllocationGranularity(&gran, &aprop,
                                       hipMemAllocationGranularityMinimum));
  printf("[gran] minimum allocation granularity = %zu B\n\n", gran);

  // ── A. hipMemCreate rate ──────────────────────────────────────────────────
  printf("[A] hipMemCreate\n");
  {
    const size_t n = std::min<size_t>(probe_pages, 65536);
    std::vector<hipMemGenericAllocationHandle_t> h(n);
    double t0 = now_s();
    for (size_t i = 0; i < n; i++)
      CHECK(hipMemCreate(&h[i], kPage, &aprop, 0));
    double us = (now_s() - t0) * 1e6 / n;
    printf("  4 KiB handles: %zu in %.2f s -> %.1f us/handle\n", n,
           us * n * 1e-6, us);
    extrapolate("create", us, kMoePages);
    for (size_t i = 0; i < n; i++)
      CHECK(hipMemRelease(h[i]));

    const size_t big = 2ull << 20;
    const size_t nb = 256;
    std::vector<hipMemGenericAllocationHandle_t> hb(nb);
    t0 = now_s();
    for (size_t i = 0; i < nb; i++)
      CHECK(hipMemCreate(&hb[i], big, &aprop, 0));
    double usb = (now_s() - t0) * 1e6 / nb;
    printf("  2 MiB handles: %zu in %.3f s -> %.1f us/handle\n", nb,
           usb * nb * 1e-6, usb);
    extrapolate("create", usb, kMoeBytes / big);
    for (size_t i = 0; i < nb; i++)
      CHECK(hipMemRelease(hb[i]));
  }

  // ── B. non-zero offset into a large handle ────────────────────────────────
  // If this works, one 2 MiB handle covers 512 pages and the create cost above
  // stops mattering; only the map calls scale with page count.
  printf("\n[B] hipMemMap with non-zero handle offset\n");
  bool offset_ok = false;
  {
    const size_t big = 2ull << 20;
    hipMemGenericAllocationHandle_t h;
    CHECK(hipMemCreate(&h, big, &aprop, 0));
    void *va = nullptr;
    CHECK(hipMemAddressReserve(&va, kPage, 0, nullptr, 0));
    hipError_t e = hipMemMap(va, kPage, kPage * 3, h, 0);
    if (e == hipSuccess) {
      offset_ok = true;
      printf("  offset=3 pages into a 2 MiB handle: OK\n");
      CHECK(hipMemUnmap(va, kPage));
    } else {
      printf("  offset=3 pages into a 2 MiB handle: REJECTED (%s)\n",
             hipGetErrorString(e));
      printf("  -> one handle per page is mandatory; [A] 4 KiB rate governs\n");
    }
    CHECK(hipMemAddressFree(va, kPage));
    CHECK(hipMemRelease(h));
  }

  // ── C. hipMemMap rate ─────────────────────────────────────────────────────
  printf("\n[C] hipMemMap + hipMemSetAccess, 4 KiB\n");
  const size_t n = probe_pages;
  std::vector<hipMemGenericAllocationHandle_t> handles(n);
  for (size_t i = 0; i < n; i++)
    CHECK(hipMemCreate(&handles[i], kPage, &aprop, 0));
  void *probe_va = nullptr;
  CHECK(hipMemAddressReserve(&probe_va, n * kPage, 0, nullptr, 0));
  {
    double t0 = now_s();
    for (size_t i = 0; i < n; i++)
      map_page(probe_va, i * kPage, handles[i], 0, dev);
    double us = (now_s() - t0) * 1e6 / n;
    printf("  %zu pages in %.2f s -> %.1f us/page\n", n, us * n * 1e-6, us);
    extrapolate("map", us, kMoePages * 2); // probe pass + placement pass
  }
  CHECK(hipMemsetD8((hipDeviceptr_t)probe_va, 0, n * kPage));

  // ── D. classifier throughput and agreement ────────────────────────────────
  printf("\n[D] classifier, %zu pages, rounds=48\n", n);
  const int rounds = 48;
  const size_t ser_n = std::min<size_t>(n, 4096); // serial is too slow for all
  int *d_claim = nullptr;
  uint64_t *d_cyc = nullptr;
  CHECK(hipMalloc(&d_claim, 4096 * sizeof(int)));
  CHECK(hipMalloc(&d_cyc, n * sizeof(uint64_t)));

  auto run = [&](bool parallel, int per_set, int sets, size_t np) {
    CHECK(hipMemset(d_claim, 0, 4096 * sizeof(int)));
    CHECK(hipMemset(d_cyc, 0, n * sizeof(uint64_t)));
    CHECK(hipMemsetD8((hipDeviceptr_t)probe_va, 0, n * kPage));
    CHECK(hipDeviceSynchronize());
    double t0 = now_s();
    if (parallel)
      classify_parallel<<<prop.multiProcessorCount, 64>>>(
          (char *)probe_va, np, rounds, per_set, sets, d_claim, d_cyc,
          (uint64_t)wall_khz * 1000); // ~1 s guard
    else
      classify_serial<<<prop.multiProcessorCount, 64>>>(
          (char *)probe_va, np, rounds, 0, 1, d_claim, d_cyc);
    CHECK(hipDeviceSynchronize());
    double el = now_s() - t0;
    std::vector<uint64_t> c(np);
    CHECK(hipMemcpy(c.data(), d_cyc, np * sizeof(uint64_t),
                    hipMemcpyDeviceToHost));
    return std::make_pair(el, c);
  };

  auto ser = run(false, 0, 0, ser_n);
  printf("  serial          : %zu pages in %.2f s -> %6.1f us/page\n", ser_n,
         ser.first, ser.first * 1e6 / ser_n);
  extrapolate("classify", ser.first * 1e6 / ser_n, kMoePages);

  // Threshold from the serial run, then reuse it to score the parallel runs.
  std::vector<double> sns(ser_n);
  for (size_t i = 0; i < ser_n; i++)
    sns[i] = (double)ser.second[i] * 1e6 / wall_khz / rounds;
  std::vector<double> srt = sns;
  std::sort(srt.begin(), srt.end());
  double thresh = (srt[srt.size() / 10] + srt[srt.size() * 9 / 10]) / 2;
  printf("  serial p10 %.0f ns  p90 %.0f ns  threshold %.0f ns\n",
         srt[srt.size() / 10], srt[srt.size() * 9 / 10], thresh);

  for (int sets : {1, 2}) {
    for (int per_set : {8, 16, 32}) {
      auto par = run(true, per_set, sets, n);
      size_t timeouts = 0;
      for (size_t i = 0; i < ser_n; i++)
        if (par.second[i] == 0)
          timeouts++;
      std::vector<double> pns(ser_n);
      for (size_t i = 0; i < ser_n; i++)
        pns[i] = (double)par.second[i] * 1e6 / wall_khz / rounds;
      std::vector<double> psrt(pns);
      std::sort(psrt.begin(), psrt.end());
      double pth = (psrt[psrt.size() / 10] + psrt[psrt.size() * 9 / 10]) / 2;
      size_t agree = 0;
      for (size_t i = 0; i < ser_n; i++)
        if (par.second[i] && (pns[i] < pth) == (sns[i] < thresh))
          agree++;
      double us = par.first * 1e6 / n;
      printf("  par %dx%2d pairs : %zu pages in %.2f s -> %6.1f us/page  "
             "agree %zu/%zu (%.1f%%)  timeouts %zu  thr %.0f ns\n",
             sets, per_set, n, par.first, us, agree, ser_n,
             100.0 * agree / ser_n, timeouts, pth);
      extrapolate("classify", us, kMoePages);
    }
  }

  // ── E. is homing uniform inside one handle? ───────────────────────────────
  // This is the question that decides the whole approach. 4 KiB handles cost
  // 107 us each and 2 MiB handles cost 3.5 us, so if a large handle were homed
  // entirely on one AID the create cost collapses by 512x and full-model
  // pinning becomes free. Earlier probing showed *adjacent pages* of a
  // hipMalloc'd range landing on opposite sides, but never checked whether the
  // pages backing a single physical allocation are uniform.
  printf("\n[E] homing uniformity inside one handle\n");
  printf("  %10s %8s %10s  %s\n", "handle", "us/create", "pages/hdl",
         "uniform handles");
  for (size_t hsz : {(size_t)kPage, (size_t)(64 << 10), (size_t)(256 << 10),
                     (size_t)(2 << 20)}) {
    const size_t pages_per = hsz / kPage;
    const size_t want = 2048;
    const size_t nh = std::max<size_t>(1, want / pages_per);
    const size_t tot = nh * pages_per;

    std::vector<hipMemGenericAllocationHandle_t> hh(nh);
    double t0 = now_s();
    for (size_t i = 0; i < nh; i++)
      CHECK(hipMemCreate(&hh[i], hsz, &aprop, 0));
    double cus = (now_s() - t0) * 1e6 / nh;

    void *va = nullptr;
    CHECK(hipMemAddressReserve(&va, tot * kPage, 0, nullptr, 0));
    for (size_t i = 0; i < nh; i++) {
      CHECK(hipMemMap((char *)va + i * hsz, hsz, 0, hh[i], 0));
      hipMemAccessDesc acc{};
      acc.location.type = hipMemLocationTypeDevice;
      acc.location.id = dev;
      acc.flags = hipMemAccessFlagsProtReadWrite;
      CHECK(hipMemSetAccess((char *)va + i * hsz, hsz, &acc, 1));
    }
    CHECK(hipMemsetD8((hipDeviceptr_t)va, 0, tot * kPage));
    CHECK(hipMemset(d_claim, 0, 4096 * sizeof(int)));
    CHECK(hipMemset(d_cyc, 0, n * sizeof(uint64_t)));
    CHECK(hipDeviceSynchronize());
    classify_parallel<<<prop.multiProcessorCount, 64>>>(
        (char *)va, tot, rounds, 32, 2, d_claim, d_cyc,
        (uint64_t)wall_khz * 1000);
    CHECK(hipDeviceSynchronize());
    std::vector<uint64_t> c(tot);
    CHECK(hipMemcpy(c.data(), d_cyc, tot * sizeof(uint64_t),
                    hipMemcpyDeviceToHost));

    std::vector<double> v(tot);
    for (size_t i = 0; i < tot; i++)
      v[i] = (double)c[i] * 1e6 / wall_khz / rounds;
    std::vector<double> vs(v);
    std::sort(vs.begin(), vs.end());
    double th = (vs[vs.size() / 10] + vs[vs.size() * 9 / 10]) / 2;

    size_t uniform = 0, on0 = 0;
    for (size_t i = 0; i < nh; i++) {
      size_t s0 = 0;
      for (size_t p = 0; p < pages_per; p++)
        if (v[i * pages_per + p] < th)
          s0++;
      if (s0 == 0 || s0 == pages_per)
        uniform++;
      on0 += s0;
    }
    printf("  %8zuK %8.1f %10zu  %5zu/%-5zu (%5.1f%%)   side0 %.0f%%%s\n",
           hsz / 1024, cus, pages_per, uniform, nh, 100.0 * uniform / nh,
           100.0 * on0 / tot, (uniform == nh && pages_per > 1) ? "  <== !" : "");

    for (size_t i = 0; i < nh; i++)
      CHECK(hipMemUnmap((char *)va + i * hsz, hsz));
    CHECK(hipMemAddressFree(va, tot * kPage));
    for (size_t i = 0; i < nh; i++)
      CHECK(hipMemRelease(hh[i]));
  }

  // ── F. do the VMM calls parallelise across host threads? ──────────────────
  // 40 us is far too long for a driver call that allocates 4 KiB, which points
  // at a lock or a per-call zeroing rather than real work. If it scales with
  // host threads the full-model cost drops from ~12 min to something a load
  // path could absorb; if it is a global lock, it will not move at all.
  printf("\n[F] host-thread scaling of the VMM calls\n");
  printf("  %8s %14s %14s\n", "threads", "create us/hdl", "map us/page");
  for (int nt : {1, 4, 16, 32}) {
    const size_t per = 4096;
    const size_t tot = (size_t)nt * per;
    std::vector<hipMemGenericAllocationHandle_t> hh(tot);
    void *va = nullptr;
    CHECK(hipMemAddressReserve(&va, tot * kPage, 0, nullptr, 0));

    double t0 = now_s();
    {
      std::vector<std::thread> th;
      for (int t = 0; t < nt; t++)
        th.emplace_back([&, t] {
          for (size_t i = 0; i < per; i++)
            CHECK(hipMemCreate(&hh[t * per + i], kPage, &aprop, 0));
        });
      for (auto &x : th)
        x.join();
    }
    double cus = (now_s() - t0) * 1e6 / tot;

    t0 = now_s();
    {
      std::vector<std::thread> th;
      for (int t = 0; t < nt; t++)
        th.emplace_back([&, t] {
          for (size_t i = 0; i < per; i++)
            map_page(va, (t * per + i) * kPage, hh[t * per + i], 0, dev);
        });
      for (auto &x : th)
        x.join();
    }
    double mus = (now_s() - t0) * 1e6 / tot;
    printf("  %8d %14.1f %14.1f\n", nt, cus, mus);

    for (size_t i = 0; i < tot; i++)
      CHECK(hipMemUnmap((char *)va + i * kPage, kPage));
    CHECK(hipMemAddressFree(va, tot * kPage));
    for (size_t i = 0; i < tot; i++)
      CHECK(hipMemRelease(hh[i]));
  }

  printf("\n[verdict] full-model pinning needs, per 64 GB: "
         "%s creates + 2x maps + 1x classify\n",
         offset_ok ? "few (offset works)" : "15.5M (offset rejected)");
  return 0;
}
