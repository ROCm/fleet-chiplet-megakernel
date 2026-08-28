// Batch-1 GEMV with AID-pinned weights.
//
// Decode at batch size 1 is pure weight streaming, so it is the shape where
// remote-AID traffic should hurt most and where AID-pinned allocation should
// pay. y[n] = dot(W[n][:], x[:]), W is N x K floats, read once per call.
//
// Phase 1 runs one chiplet group against weights homed on its own side and
// against weights homed on the far side.
// Phase 2 is the version you would actually ship: all 8 chiplets working, with
// each group's half of W homed on its own side, versus a plain hipMalloc W.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_gemv.hip -o aid_gemv
//   HIP_VISIBLE_DEVICES=1 ./aid_gemv

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
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
static constexpr int kBlock = 256;

__device__ inline unsigned get_xcc_id() {
  constexpr unsigned imm = ((4u - 1u) << 11) | (0u << 6) | 20u;
  return __builtin_amdgcn_s_getreg(imm);
}

__global__ void probe_xcc(int *out, int n) {
  if (blockIdx.x < (unsigned)n && threadIdx.x == 0)
    out[blockIdx.x] = (int)get_xcc_id();
}

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

// One block per row at a time; the block strides over K with NT loads.
__device__ inline void gemv_rows(const float *W, const float *x, float *y,
                                 int row_begin, int row_end, int K, int slot,
                                 int nslots) {
  __shared__ float red[kBlock];
  for (int n = row_begin + slot; n < row_end; n += nslots) {
    const float *w = W + (size_t)(n - row_begin) * K;
    float acc = 0.f;
    for (int k = threadIdx.x; k < K; k += kBlock)
      acc += __builtin_nontemporal_load(w + k) * x[k];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s)
        red[threadIdx.x] += red[threadIdx.x + s];
      __syncthreads();
    }
    if (threadIdx.x == 0)
      y[n] = red[0];
  }
}

// Phase 1: only the chiplets of `group` work, over all N rows of one buffer.
__global__ void gemv_one_group(const float *W, const float *x, float *y, int N,
                               int K, int group, int per_aid, int num_xcd) {
  if ((int)get_xcc_id() / per_aid != group)
    return;
  gemv_rows(W, x, y, 0, N, K, blockIdx.x / num_xcd, gridDim.x / num_xcd);
}

// Phase 2: every chiplet works. Group g takes the g-th half of the rows and
// reads it from Wg, so each group can be pointed at its own local pages.
__global__ void gemv_split(const float *W0, const float *W1, const float *x,
                           float *y, int N, int K, int per_aid, int num_xcd) {
  int g = (int)get_xcc_id() / per_aid;
  const float *W = g ? W1 : W0;
  int half = N / 2;
  gemv_rows(W, x, y, g * half, g * half + half, K, blockIdx.x / num_xcd,
            gridDim.x / num_xcd);
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

int main() {
  const int dev = 0;
  CHECK(hipSetDevice(dev));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, dev));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev));
  char bdf[32];
  CHECK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), dev));

  int *d_xcc = nullptr;
  const int nprobe = prop.multiProcessorCount * 2;
  CHECK(hipMalloc(&d_xcc, nprobe * sizeof(int)));
  probe_xcc<<<nprobe, 64>>>(d_xcc, nprobe);
  CHECK(hipDeviceSynchronize());
  std::vector<int> xv(nprobe);
  CHECK(hipMemcpy(xv.data(), d_xcc, nprobe * sizeof(int),
                  hipMemcpyDeviceToHost));
  CHECK(hipFree(d_xcc));
  int num_xcd = 0;
  for (int v : xv)
    num_xcd = std::max(num_xcd, v + 1);
  const int per_aid = std::max(1, num_xcd / 2);

  const int K = 8192;
  const int N = 16384;
  const size_t half_bytes = (size_t)(N / 2) * K * sizeof(float);
  const size_t half_pages = half_bytes / kPage;
  printf("device %s  pci=%s  XCDs=%d\n", prop.gcnArchName, bdf, num_xcd);
  printf("GEMV  N=%d K=%d fp32   W = %zu MiB total, %zu MiB per group\n\n",
         N, K, 2 * half_bytes >> 20, half_bytes >> 20);

  // --- build one AID-local buffer per side out of 4 KiB VMM handles ---
  const size_t cand = half_pages * 3;
  hipMemAllocationProp ap{};
  ap.type = hipMemAllocationTypePinned;
  ap.location.type = hipMemLocationTypeDevice;
  ap.location.id = dev;

  std::vector<hipMemGenericAllocationHandle_t> h(cand);
  for (size_t i = 0; i < cand; i++)
    CHECK(hipMemCreate(&h[i], kPage, &ap, 0));
  void *pool = nullptr;
  CHECK(hipMemAddressReserve(&pool, cand * kPage, 0, nullptr, 0));
  for (size_t i = 0; i < cand; i++)
    map_page(pool, i * kPage, h[i], dev);
  CHECK(hipMemsetD8((hipDeviceptr_t)pool, 0, cand * kPage));

  int *d_claim = nullptr;
  uint64_t *d_cyc = nullptr;
  CHECK(hipMalloc(&d_claim, 2 * sizeof(int)));
  CHECK(hipMalloc(&d_cyc, cand * sizeof(uint64_t)));
  CHECK(hipMemset(d_claim, 0, 2 * sizeof(int)));
  const int rounds = 48;
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  CHECK(hipEventRecord(e0));
  classify_pages<<<prop.multiProcessorCount, 64>>>((char *)pool, cand, rounds,
                                                   0, 1, d_claim, d_cyc);
  CHECK(hipEventRecord(e1));
  CHECK(hipEventSynchronize(e1));
  float probe_ms = 0;
  CHECK(hipEventElapsedTime(&probe_ms, e0, e1));

  std::vector<uint64_t> cyc(cand);
  CHECK(hipMemcpy(cyc.data(), d_cyc, cand * sizeof(uint64_t),
                  hipMemcpyDeviceToHost));
  std::vector<double> ns(cand);
  for (size_t i = 0; i < cand; i++)
    ns[i] = (double)cyc[i] * 1e6 / wall_khz / rounds;
  std::vector<double> srt = ns;
  std::sort(srt.begin(), srt.end());
  double thresh = (srt[cand / 10] + srt[cand * 9 / 10]) / 2;
  std::vector<size_t> s0, s1;
  for (size_t i = 0; i < cand; i++)
    (ns[i] < thresh ? s0 : s1).push_back(i);
  printf("probed %zu pages in %.1f s -> %zu near XCD0-%d, %zu near XCD%d-%d\n",
         cand, probe_ms / 1000.0, s0.size(), per_aid - 1, s1.size(), per_aid,
         num_xcd - 1);
  if (s0.size() < half_pages || s1.size() < half_pages) {
    fprintf(stderr, "pool too small\n");
    return 1;
  }

  auto build = [&](const std::vector<size_t> &pick) {
    void *va = nullptr;
    CHECK(hipMemAddressReserve(&va, half_bytes, 0, nullptr, 0));
    for (size_t i = 0; i < half_pages; i++) {
      CHECK(hipMemUnmap((char *)pool + pick[i] * kPage, kPage));
      map_page(va, i * kPage, h[pick[i]], dev);
    }
    return (float *)va;
  };
  float *W0 = build(s0); // homed near XCD0-3
  float *W1 = build(s1); // homed near XCD4-7
  printf("built W0=%p (side 0)  W1=%p (side 1)\n\n", (void *)W0, (void *)W1);

  // --- data ---
  std::vector<float> hw(half_bytes / sizeof(float));
  for (size_t i = 0; i < hw.size(); i++)
    hw[i] = (float)((i * 1103515245u + 12345u) % 17) * 0.03125f - 0.25f;
  CHECK(hipMemcpy(W0, hw.data(), half_bytes, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(W1, hw.data(), half_bytes, hipMemcpyHostToDevice));

  float *x = nullptr, *y = nullptr;
  CHECK(hipMalloc(&x, (size_t)K * sizeof(float)));
  CHECK(hipMalloc(&y, (size_t)N * sizeof(float)));
  std::vector<float> hx(K);
  for (int i = 0; i < K; i++)
    hx[i] = (float)((i % 13) - 6) * 0.125f;
  CHECK(hipMemcpy(x, hx.data(), K * sizeof(float), hipMemcpyHostToDevice));

  // A plain hipMalloc W for the baseline: same size, ~50/50 across the AIDs.
  float *Wp = nullptr;
  CHECK(hipMalloc(&Wp, 2 * half_bytes));
  CHECK(hipMemcpy(Wp, hw.data(), half_bytes, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(Wp + half_bytes / sizeof(float), hw.data(), half_bytes,
                  hipMemcpyHostToDevice));

  const int grid = 512;
  const int reps = 20;
  auto timeit = [&](auto launch) {
    launch();
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    for (int r = 0; r < reps; r++)
      launch();
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    return ms / reps;
  };

  auto checksum = [&](int n) {
    std::vector<float> hy(n);
    CHECK(hipMemcpy(hy.data(), y, n * sizeof(float), hipMemcpyDeviceToHost));
    double s = 0;
    for (int i = 0; i < n; i++)
      s += hy[i];
    return s;
  };

  printf("== phase 1: only XCD0-%d works, %zu MiB of W ==\n", per_aid - 1,
         half_bytes >> 20);
  double ref = 0;
  for (int c = 0; c < 2; c++) {
    const float *W = c ? W1 : W0;
    CHECK(hipMemset(y, 0, (size_t)N * sizeof(float)));
    float ms = timeit([&] {
      gemv_one_group<<<grid, kBlock>>>(W, x, y, N / 2, K, 0, per_aid, num_xcd);
    });
    double sum = checksum(N / 2);
    if (!c)
      ref = sum;
    printf("  W homed %-13s %8.3f ms   %7.1f GB/s   checksum %.6e%s\n",
           c ? "on far side" : "on its side", ms,
           half_bytes / (ms * 1e6), sum,
           c ? (std::abs(sum - ref) < 1e-3 * std::abs(ref) ? "  (matches)"
                                                          : "  MISMATCH")
             : "");
  }

  printf("\n== phase 2: all %d chiplets, %zu MiB of W ==\n", num_xcd,
         2 * half_bytes >> 20);
  CHECK(hipMemset(y, 0, (size_t)N * sizeof(float)));
  float ms_base = timeit([&] {
    gemv_split<<<grid, kBlock>>>(Wp, Wp + half_bytes / sizeof(float), x, y, N, K,
                                 per_aid, num_xcd);
  });
  double sum_base = checksum(N);

  CHECK(hipMemset(y, 0, (size_t)N * sizeof(float)));
  float ms_pin = timeit([&] {
    gemv_split<<<grid, kBlock>>>(W0, W1, x, y, N, K, per_aid, num_xcd);
  });
  double sum_pin = checksum(N);

  // Same pages, same allocator, assignment inverted: each group now reads the
  // far side. Isolates locality from any effect of VMM pages themselves.
  CHECK(hipMemset(y, 0, (size_t)N * sizeof(float)));
  float ms_anti = timeit([&] {
    gemv_split<<<grid, kBlock>>>(W1, W0, x, y, N, K, per_aid, num_xcd);
  });
  double sum_anti = checksum(N);

  auto row = [&](const char *tag, float ms, double sum) {
    printf("  %-19s %8.3f ms   %7.1f GB/s   checksum %.6e%s\n", tag, ms,
           2 * half_bytes / (ms * 1e6), sum,
           std::abs(sum - sum_base) < 1e-3 * std::abs(sum_base) ? "  (matches)"
                                                                : "  MISMATCH");
  };
  row("plain hipMalloc W", ms_base, sum_base);
  row("AID-pinned W", ms_pin, sum_pin);
  row("inverted (control)", ms_anti, sum_anti);
  printf("  pinned vs hipMalloc %.2fx     pinned vs inverted %.2fx\n",
         ms_base / ms_pin, ms_anti / ms_pin);
  return 0;
}
