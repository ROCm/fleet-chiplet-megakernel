// AID-pinned allocation without partition modes.
//
// The home AID of a physical page is observable at runtime (see aid_home_map)
// and HIP's virtual memory API on this part has a 4 KiB allocation granularity,
// which is exactly the granularity at which homing changes. That is enough to
// build a normal, contiguous buffer whose every page is homed on one chosen
// side:
//
//   1. reserve VA, create one 4 KiB physical handle per page, map them all
//   2. classify each page by bouncing a flag between two chiplets of one group
//   3. re-map only the matching handles, contiguously, into a second VA range
//   4. release the handles that did not match
//
// The caller gets a plain pointer. Nothing here needs CPX/NPS or root.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_alloc.hip -o aid_alloc
//   HIP_VISIBLE_DEVICES=1 ./aid_alloc [MiB]

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

__device__ inline unsigned get_xcc_id() {
  constexpr unsigned imm = ((4u - 1u) << 11) | (0u << 6) | 20u;
  return __builtin_amdgcn_s_getreg(imm);
}

__global__ void probe_xcc(int *out, int n) {
  if (blockIdx.x < (unsigned)n && threadIdx.x == 0)
    out[blockIdx.x] = (int)get_xcc_id();
}

// One launch classifies every page: a wave on xcc_a and a wave on xcc_b bounce
// a flag inside each page in turn. Pages are independent addresses, so the two
// waves stay in step without any extra barrier.
__global__ void classify_pages(char *base, size_t npages, int rounds, int xcc_a,
                               int xcc_b, int *claim, uint64_t *cycles) {
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

// NT read of the whole buffer, restricted to the chiplets of one group.
__global__ void stream_group(const uint64_t *data, size_t nbytes, int group,
                             int per_aid, int num_xcd, int iters,
                             uint64_t *sink) {
  int x = (int)get_xcc_id();
  if (x / per_aid != group)
    return;

  // blockIdx is round-robin across XCDs, so index within the group's blocks.
  const size_t n_local = gridDim.x / num_xcd;
  const size_t local = blockIdx.x / num_xcd;
  const size_t tid = local * blockDim.x + threadIdx.x;
  const size_t nthreads = n_local * blockDim.x;
  const size_t n64 = nbytes / sizeof(uint64_t);

  uint64_t acc = 0;
  for (int it = 0; it < iters; it++)
    for (size_t i = tid; i < n64; i += nthreads)
      acc += __builtin_nontemporal_load(data + i);
  sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

struct AidBuffer {
  void *ptr = nullptr;
  size_t bytes = 0;
  std::vector<hipMemGenericAllocationHandle_t> handles;
  void *reserved = nullptr;
  size_t reserved_bytes = 0;
};

static void map_range(void *va, size_t off, hipMemGenericAllocationHandle_t h,
                      int dev) {
  CHECK(hipMemMap((char *)va + off, kPage, 0, h, 0));
  hipMemAccessDesc acc{};
  acc.location.type = hipMemLocationTypeDevice;
  acc.location.id = dev;
  acc.flags = hipMemAccessFlagsProtReadWrite;
  CHECK(hipMemSetAccess((char *)va + off, kPage, &acc, 1));
}

int main(int argc, char **argv) {
  const size_t want_mib = (argc > 1) ? (size_t)std::atoll(argv[1]) : 64;
  const int dev = 0;
  CHECK(hipSetDevice(dev));

  char bdf[32];
  CHECK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), dev));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, dev));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev));
  printf("device %s  pci=%s\n", prop.gcnArchName, bdf);

  int *d_xcc = nullptr;
  const int nprobe = prop.multiProcessorCount * 2;
  CHECK(hipMalloc(&d_xcc, nprobe * sizeof(int)));
  probe_xcc<<<nprobe, 64>>>(d_xcc, nprobe);
  CHECK(hipDeviceSynchronize());
  std::vector<int> xcc(nprobe);
  CHECK(hipMemcpy(xcc.data(), d_xcc, nprobe * sizeof(int),
                  hipMemcpyDeviceToHost));
  CHECK(hipFree(d_xcc));
  int num_xcd = 0;
  for (int v : xcc)
    num_xcd = std::max(num_xcd, v + 1);
  const int per_aid = std::max(1, num_xcd / 2);

  const size_t want_pages = want_mib * 1024 * 1024 / kPage;
  // The split is near 50/50 but not exact (43/57 observed), so oversample.
  const size_t cand_pages = want_pages * 3;
  printf("requesting %zu MiB (%zu pages), probing %zu candidate pages\n\n",
         want_mib, want_pages, cand_pages);

  // 1. physical handles, one per page, all mapped contiguously for probing
  hipMemAllocationProp aprop{};
  aprop.type = hipMemAllocationTypePinned;
  aprop.location.type = hipMemLocationTypeDevice;
  aprop.location.id = dev;

  std::vector<hipMemGenericAllocationHandle_t> handles(cand_pages);
  for (size_t i = 0; i < cand_pages; i++)
    CHECK(hipMemCreate(&handles[i], kPage, &aprop, 0));

  void *probe_va = nullptr;
  CHECK(hipMemAddressReserve(&probe_va, cand_pages * kPage, 0, nullptr, 0));
  for (size_t i = 0; i < cand_pages; i++)
    map_range(probe_va, i * kPage, handles[i], dev);
  CHECK(hipMemsetD8((hipDeviceptr_t)probe_va, 0, cand_pages * kPage));

  // 2. classify
  int *d_claim = nullptr;
  uint64_t *d_cyc = nullptr;
  CHECK(hipMalloc(&d_claim, 2 * sizeof(int)));
  CHECK(hipMalloc(&d_cyc, cand_pages * sizeof(uint64_t)));
  CHECK(hipMemset(d_claim, 0, 2 * sizeof(int)));
  CHECK(hipMemset(d_cyc, 0, cand_pages * sizeof(uint64_t)));

  const int rounds = 48;
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  CHECK(hipEventRecord(e0));
  classify_pages<<<prop.multiProcessorCount, 64>>>(
      (char *)probe_va, cand_pages, rounds, 0, 1, d_claim, d_cyc);
  CHECK(hipEventRecord(e1));
  CHECK(hipEventSynchronize(e1));
  float probe_ms = 0;
  CHECK(hipEventElapsedTime(&probe_ms, e0, e1));

  std::vector<uint64_t> cyc(cand_pages);
  CHECK(hipMemcpy(cyc.data(), d_cyc, cand_pages * sizeof(uint64_t),
                  hipMemcpyDeviceToHost));
  std::vector<double> ns(cand_pages);
  for (size_t i = 0; i < cand_pages; i++)
    ns[i] = (double)cyc[i] * 1e6 / wall_khz / rounds;

  std::vector<double> srt = ns;
  std::sort(srt.begin(), srt.end());
  double lo = srt[srt.size() / 10], hi = srt[srt.size() * 9 / 10];
  double thresh = (lo + hi) / 2;
  printf("classified %zu pages in %.2f s (%.0f us/page)\n", cand_pages,
         probe_ms / 1000.0, probe_ms * 1000.0 / cand_pages);
  printf("  p10 %.0f ns   p90 %.0f ns   threshold %.0f ns\n", lo, hi, thresh);

  std::vector<size_t> side0, side1;
  for (size_t i = 0; i < cand_pages; i++)
    (ns[i] < thresh ? side0 : side1).push_back(i);
  printf("  near XCD0-%d: %zu pages    near XCD%d-%d: %zu pages\n\n",
         per_aid - 1, side0.size(), per_aid, num_xcd - 1, side1.size());

  if (side0.size() < want_pages || side1.size() < want_pages) {
    fprintf(stderr, "not enough pages on one side; increase the pool\n");
    return 1;
  }

  // 3. build one contiguous buffer per side out of matching handles
  auto build = [&](const std::vector<size_t> &pick) {
    void *va = nullptr;
    CHECK(hipMemAddressReserve(&va, want_pages * kPage, 0, nullptr, 0));
    for (size_t i = 0; i < want_pages; i++) {
      CHECK(hipMemUnmap((char *)probe_va + pick[i] * kPage, kPage));
      map_range(va, i * kPage, handles[pick[i]], dev);
    }
    return va;
  };
  void *local_buf = build(side0);
  void *remote_buf = build(side1);
  printf("built two %zu MiB buffers: local=%p remote=%p\n", want_mib, local_buf,
         remote_buf);

  // 4. verify: re-classify the finished buffers
  auto verify = [&](void *buf, const char *tag) {
    CHECK(hipMemsetD8((hipDeviceptr_t)buf, 0, want_pages * kPage));
    CHECK(hipMemset(d_claim, 0, 2 * sizeof(int)));
    const size_t n = std::min<size_t>(want_pages, 4096);
    classify_pages<<<prop.multiProcessorCount, 64>>>((char *)buf, n, rounds, 0,
                                                     1, d_claim, d_cyc);
    CHECK(hipDeviceSynchronize());
    std::vector<uint64_t> c(n);
    CHECK(hipMemcpy(c.data(), d_cyc, n * sizeof(uint64_t),
                    hipMemcpyDeviceToHost));
    size_t match = 0;
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
      double v = (double)c[i] * 1e6 / wall_khz / rounds;
      sum += v;
      if ((v < thresh) == (tag[0] == 'l'))
        match++;
    }
    printf("  %-8s buffer: %zu/%zu pages on the expected side, mean %.0f ns\n",
           tag, match, n, sum / n);
  };
  printf("\nverification (re-probing the finished buffers):\n");
  verify(local_buf, "local");
  verify(remote_buf, "remote");

  // 5. does it pay for bulk reads? stream both from the XCD0-3 group only.
  const int grid = 256, block = 256;
  uint64_t *d_sink = nullptr;
  CHECK(hipMalloc(&d_sink, (size_t)grid * block * sizeof(uint64_t)));
  const size_t bytes = want_pages * kPage;
  const int iters = std::max(4, (int)((512ull << 20) / bytes));

  auto bw = [&](void *buf, int group) {
    CHECK(hipMemset(d_sink, 0, (size_t)grid * block * sizeof(uint64_t)));
    stream_group<<<grid, block>>>((const uint64_t *)buf, bytes, group, per_aid,
                                  num_xcd, 1, d_sink); // warm
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    stream_group<<<grid, block>>>((const uint64_t *)buf, bytes, group, per_aid,
                                  num_xcd, iters, d_sink);
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    return (double)bytes * iters / (ms * 1e6);
  };

  // The control that matters: each group should prefer the buffer homed on its
  // own side. If one buffer were simply faster, both rows would agree.
  printf("\nNT streaming, %zu MiB, %d passes (GB/s):\n", want_mib, iters);
  printf("  %-22s %14s %14s   %s\n", "reader", "pages@XCD0-3", "pages@XCD4-7",
         "prefers");
  for (int g = 0; g < 2; g++) {
    double g0 = bw(local_buf, g);
    double g1 = bw(remote_buf, g);
    printf("  XCD%d-%-18d %14.1f %14.1f   %s (%.2fx)\n", g * per_aid,
           g * per_aid + per_aid - 1, g0, g1,
           g0 > g1 ? "XCD0-3 pages" : "XCD4-7 pages",
           std::max(g0, g1) / std::min(g0, g1));
  }

  return 0;
}
