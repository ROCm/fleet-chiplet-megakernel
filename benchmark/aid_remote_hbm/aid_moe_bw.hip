// Does AID-local placement still pay at the MoE phase's operating point?
//
// Every number this directory quotes for the local/remote gap -- 1.14x on NT
// streaming, 1.10x and 1.23x on aid_gemv -- was measured with the memory system
// nearly idle. aid_gemv's shipped phase runs at 296.9 GB/s. MPK_PHASE_SLOTS says
// fleet's real MoE phase moves 52.7 MiB in 12.9 us, which is 4270 GB/s, 53% of
// the 8 TB/s peak. That is a 14x different operating point, and the two regimes
// need not agree: idle memory is latency-bound, where a remote hop costs its
// full ~380 ns, while a loaded interconnect can either amplify the penalty
// (queueing on the inter-AID link) or hide it (every queue is full anyway).
//
// So this reproduces the MoE weight stream rather than a generic STREAM:
//
//   * MXFP4 weight slabs of exactly W13_WG_BYTES (200192) or W2_WG_BYTES
//     (100096), read contiguously and exactly once -- decode reuses nothing.
//   * the AID-invariant map's assignment: workgroup w belongs to AID
//     (w < n_wgs/2 ? 0 : 1), and only the four XCDs of that AID read it.
//   * a working set past the 224 MiB L3, so the memory-side cache cannot
//     absorb the traffic and make homing irrelevant (aid_alloc saw exactly
//     that at 16 MiB, where the effect vanished to 0.94x).
//
// Three arms off ONE pinned buffer, which is what makes the control tight:
// `local` and `remote` read the same physical pages and differ only in which
// XCD group reads which half, so no difference can be an artifact of 4 KiB VMM
// pages behaving unlike hipMalloc pages. `interleaved` is stock hipMalloc.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_moe_bw.hip -o aid_moe_bw
//   HIP_VISIBLE_DEVICES=1 ./aid_moe_bw [MiB] [wg_bytes]

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
static constexpr int kNumWgs = 46; // gpt-oss-120b, both phases

// Native vector, not HIP_vector_type: __builtin_nontemporal_load only accepts
// scalar or ext_vector operands, so uint4 is rejected.
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

// The MoE weight stream. `order` lists slab ids grouped by the AID that owns
// them, so a block only ever touches slabs belonging to its own AID (or the
// other one, under `invert`). Reads are non-temporal: decode never revisits a
// weight byte, and letting them allocate in L2 would evict the next tile.
__global__ __launch_bounds__(256) void moe_stream(const char *__restrict base,
                                                  const int *__restrict order,
                                                  size_t slab_bytes,
                                                  int per_aid, int invert,
                                                  int iters,
                                                  uint64_t *__restrict sink) {
  const int x = (int)get_xcc_id();
  int aid = x >> 2;
  if (invert)
    aid ^= 1;

  const int bpx = gridDim.x / 8;            // blocks per XCD
  const int rank = (x & 3) * bpx + blockIdx.x / 8;
  const int nranks = 4 * bpx;

  const size_t n16 = slab_bytes / 16;
  uint64_t acc = 0;
  for (int it = 0; it < iters; it++) {
    for (int i = rank; i < per_aid; i += nranks) {
      const int slab = order[aid * per_aid + i];
      const v4u *p = (const v4u *)(base + (size_t)slab * slab_bytes);
      for (size_t j = threadIdx.x; j < n16; j += 256) {
        v4u v = __builtin_nontemporal_load(p + j);
        acc += v.x + v.y + v.z + v.w;
      }
    }
  }
  sink[blockIdx.x * 256 + threadIdx.x] = acc;
}

static void map_page(void *va, size_t off, hipMemGenericAllocationHandle_t h,
                     int dev) {
  CHECK(hipMemMap((char *)va + off, kPage, 0, h, 0));
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

int main(int argc, char **argv) {
  const size_t want_mib = (argc > 1) ? (size_t)std::atoll(argv[1]) : 512;
  const size_t slab_bytes = (argc > 2) ? (size_t)std::atoll(argv[2]) : 200192;
  const int dev = 0;
  CHECK(hipSetDevice(dev));

  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, dev));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev));
  printf("device %s  CUs=%d\n", prop.gcnArchName, prop.multiProcessorCount);

  // Whole experts only, so the w = slab % 46 assignment stays balanced.
  size_t experts = (want_mib << 20) / (slab_bytes * kNumWgs);
  if (experts < 1)
    experts = 1;
  const size_t n_slabs = experts * kNumWgs;
  const size_t bytes = n_slabs * slab_bytes;
  const size_t n_pages = (bytes + kPage - 1) / kPage;
  const int half = kNumWgs / 2;

  printf("slab %zu B (wg_bytes), %zu experts x %d wgs = %zu slabs, "
         "%.0f MiB\n",
         slab_bytes, experts, kNumWgs, n_slabs, bytes / 1048576.0);
  printf("working set %.0f MiB vs 224 MiB L3: %s\n\n", bytes / 1048576.0,
         bytes > (224ull << 20) ? "past it" : "FITS -- homing will not matter");

  // Slab ids grouped by owning AID: w < 23 -> AID0.
  std::vector<int> order;
  order.reserve(n_slabs);
  for (int a = 0; a < 2; a++)
    for (size_t s = 0; s < n_slabs; s++) {
      int w = (int)(s % kNumWgs);
      if (((w < half) ? 0 : 1) == a)
        order.push_back((int)s);
    }
  const int per_aid = (int)(order.size() / 2);
  int *d_order = nullptr;
  CHECK(hipMalloc(&d_order, order.size() * sizeof(int)));
  CHECK(hipMemcpy(d_order, order.data(), order.size() * sizeof(int),
                  hipMemcpyHostToDevice));

  // ── the pinned buffer ─────────────────────────────────────────────────────
  hipMemAllocationProp aprop{};
  aprop.type = hipMemAllocationTypePinned;
  aprop.location.type = hipMemLocationTypeDevice;
  aprop.location.id = dev;

  std::vector<uint8_t> want(n_pages);
  size_t need[2] = {0, 0};
  for (size_t p = 0; p < n_pages; p++) {
    int w = (int)((p * kPage / slab_bytes) % kNumWgs);
    want[p] = (w < half) ? 0 : 1;
    need[want[p]]++;
  }

  // Two VMM buffers come out of this pool: the AID-sorted one, and a mixed
  // control that is VMM-backed but NOT sorted by AID. Without that control the
  // comparison against hipMalloc confounds homing with page size -- VMM forces
  // 4 KiB mappings while hipMalloc gets 2 MiB pages, and the TLB cost of the
  // former is not small.
  const size_t pool = (size_t)(n_pages * 2.40) + 2048;
  printf("pinning %zu pages (pool %zu)...\n", n_pages, pool);
  std::vector<hipMemGenericAllocationHandle_t> handles(pool);
  double t0 = now_s();
  for (size_t i = 0; i < pool; i++)
    CHECK(hipMemCreate(&handles[i], kPage, &aprop, 0));
  printf("  create %.1f s\n", now_s() - t0);

  void *probe_va = nullptr;
  CHECK(hipMemAddressReserve(&probe_va, pool * kPage, 0, nullptr, 0));
  t0 = now_s();
  for (size_t i = 0; i < pool; i++)
    map_page(probe_va, i * kPage, handles[i], dev);
  printf("  probe map %.1f s\n", now_s() - t0);
  CHECK(hipMemsetD8((hipDeviceptr_t)probe_va, 0, pool * kPage));

  int *d_claim = nullptr;
  uint64_t *d_cyc = nullptr;
  CHECK(hipMalloc(&d_claim, 4096 * sizeof(int)));
  CHECK(hipMalloc(&d_cyc, pool * sizeof(uint64_t)));
  CHECK(hipMemset(d_claim, 0, 4096 * sizeof(int)));
  CHECK(hipMemset(d_cyc, 0, pool * sizeof(uint64_t)));
  const int rounds = 48;
  t0 = now_s();
  classify_parallel<<<prop.multiProcessorCount, 64>>>(
      (char *)probe_va, pool, rounds, 32, 2, d_claim, d_cyc,
      (uint64_t)wall_khz * 1000);
  CHECK(hipDeviceSynchronize());
  printf("  classify %.2f s\n", now_s() - t0);

  std::vector<uint64_t> cyc(pool);
  CHECK(hipMemcpy(cyc.data(), d_cyc, pool * sizeof(uint64_t),
                  hipMemcpyDeviceToHost));
  std::vector<double> ns(pool);
  for (size_t i = 0; i < pool; i++)
    ns[i] = (double)cyc[i] * 1e6 / wall_khz / rounds;
  std::vector<double> srt(ns);
  std::sort(srt.begin(), srt.end());
  const double thresh = (srt[srt.size() / 10] + srt[srt.size() * 9 / 10]) / 2;
  std::vector<size_t> bucket[2];
  for (size_t i = 0; i < pool; i++)
    bucket[ns[i] < thresh ? 0 : 1].push_back(i);
  printf("  sides: %zu / %zu (need %zu / %zu)\n", bucket[0].size(),
         bucket[1].size(), need[0], need[1]);
  if (bucket[0].size() < need[0] || bucket[1].size() < need[1]) {
    fprintf(stderr, "pool short on one side\n");
    return 1;
  }

  void *pinned = nullptr;
  CHECK(hipMemAddressReserve(&pinned, n_pages * kPage, 0, nullptr, 0));
  size_t take[2] = {0, 0};
  std::vector<bool> used(pool, false);
  t0 = now_s();
  for (size_t p = 0; p < n_pages; p++) {
    size_t idx = bucket[want[p]][take[want[p]]++];
    used[idx] = true;
    CHECK(hipMemUnmap((char *)probe_va + idx * kPage, kPage));
    map_page(pinned, p * kPage, handles[idx], dev);
  }
  printf("  place %.1f s\n", now_s() - t0);

  // Mixed control: same VMM machinery, same 4 KiB granularity, pages dealt
  // alternately from the two sides so every workgroup is ~50/50 -- what
  // hipMalloc gives by accident.
  void *mixed = nullptr;
  CHECK(hipMemAddressReserve(&mixed, n_pages * kPage, 0, nullptr, 0));
  t0 = now_s();
  for (size_t p = 0; p < n_pages; p++) {
    int side = (int)(p & 1);
    if (take[side] >= bucket[side].size())
      side ^= 1;
    size_t idx = bucket[side][take[side]++];
    used[idx] = true;
    CHECK(hipMemUnmap((char *)probe_va + idx * kPage, kPage));
    map_page(mixed, p * kPage, handles[idx], dev);
  }
  printf("  place mixed %.1f s\n", now_s() - t0);
  for (size_t i = 0; i < pool; i++)
    if (!used[i]) {
      (void)hipMemUnmap((char *)probe_va + i * kPage, kPage);
      (void)hipMemRelease(handles[i]);
    }
  (void)hipMemAddressFree(probe_va, pool * kPage);
  CHECK(hipMemsetD8((hipDeviceptr_t)pinned, 1, bytes));
  CHECK(hipMemsetD8((hipDeviceptr_t)mixed, 1, bytes));

  void *plain = nullptr;
  CHECK(hipMalloc(&plain, bytes));
  CHECK(hipMemsetD8((hipDeviceptr_t)plain, 1, bytes));

  uint64_t *d_sink = nullptr;
  const int block = 256;
  CHECK(hipMalloc(&d_sink, (size_t)4096 * block * sizeof(uint64_t)));

  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));

  auto bw = [&](void *buf, int invert, int grid, int iters) {
    moe_stream<<<grid, block>>>((const char *)buf, d_order, slab_bytes, per_aid,
                                invert, 1, d_sink);
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    moe_stream<<<grid, block>>>((const char *)buf, d_order, slab_bytes, per_aid,
                                invert, iters, d_sink);
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    return (double)bytes * iters / (ms * 1e6);
  };

  // Sweep the grid: fleet runs 240 workers, but the operating point is what
  // matters, so find where this pattern actually saturates.
  auto med3 = [&](void *buf, int invert, int grid, int iters) {
    double v[3];
    for (int k = 0; k < 3; k++)
      v[k] = bw(buf, invert, grid, iters);
    std::sort(v, v + 3);
    return v[1];
  };

  printf("\n%-6s %11s %11s %11s %11s %8s %8s\n", "grid", "hipMalloc",
         "VMM-mixed", "VMM-local", "VMM-remote", "loc/mix", "loc/rem");
  for (int grid : {240, 960, 1920, 3840}) {
    const int iters = std::max(4, (int)((8ull << 30) / bytes));
    double h = med3(plain, 0, grid, iters);
    double m = med3(mixed, 0, grid, iters);
    double l = med3(pinned, 0, grid, iters);
    double r = med3(pinned, 1, grid, iters);
    printf("%-6d %8.1f GB/s %8.1f %8.1f %8.1f %8.3fx %8.3fx\n", grid, h, m, l,
           r, l / m, l / r);
  }

  printf("\nfleet's MoE phase runs at 4270 GB/s (52.7 MiB / 12.9 us).\n");
  return 0;
}
