// Local vs remote AID HBM / fabric on MI350/MI355.
//
// Machine: 8 XCDs (chiplets), 2 AIDs, HBM stacks attached to AIDs, NPS1 (SPX).
// GTAC'25 paper 48 (Starscream): a CU fetching from HBM on the other AID pays
// extra interconnect latency and AID power vs the stacks on its own AID.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_remote_hbm.hip -o aid_remote_hbm
//   HIP_VISIBLE_DEVICES=0 ./aid_remote_hbm

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
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

__device__ inline unsigned get_xcc_id() {
  constexpr unsigned imm = ((4u - 1u) << 11) | (0u << 6) | 20u;
  return __builtin_amdgcn_s_getreg(imm);
}

__global__ void probe_xcc(int *out, int n) {
  int i = blockIdx.x;
  if (i < n && threadIdx.x == 0)
    out[i] = (int)get_xcc_id();
}

// One wave on xcc_a ping-pongs an HBM flag with one wave on xcc_b.
__global__ void pingpong_kernel(int *flag, int xcc_a, int xcc_b, int rounds,
                                int *claim, uint64_t *cycles_out) {
  int x = (int)get_xcc_id();
  if (x != xcc_a && x != xcc_b)
    return;
  if (threadIdx.x != 0)
    return;
  int slot = (x == xcc_a) ? 0 : 1;
  if (atomicCAS(&claim[slot], 0, 1) != 0)
    return;

  if (x == xcc_a) {
    uint64_t t0 = wall_clock64();
    for (int r = 0; r < rounds; r++) {
      while (__hip_atomic_load(flag, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT) != (r * 2)) {
      }
      __hip_atomic_store(flag, r * 2 + 1, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
    uint64_t t1 = wall_clock64();
    cycles_out[0] = t1 - t0;
  } else {
    for (int r = 0; r < rounds; r++) {
      while (__hip_atomic_load(flag, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT) != (r * 2 + 1)) {
      }
      __hip_atomic_store(flag, r * 2 + 2, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

// NT STREAM. aid_sel: 0 = stacks 0-3, 1 = stacks 4-7, -1 = all 8.
// Assumes 256B HBM channel interleave and 8 stacks (4 per AID on MI350).
__global__ void stream_stacks_kernel(const uint64_t *data, size_t nbytes,
                                     int aid_sel, int target_xcc, int num_xcd,
                                     int iters, uint64_t *sink) {
  if ((int)get_xcc_id() != target_xcc)
    return;

  constexpr size_t kLine = 256;
  const size_t nlines = nbytes / kLine;
  const int n_local = gridDim.x / num_xcd;
  const int local = blockIdx.x / num_xcd;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  uint64_t acc = 0;

  for (int it = 0; it < iters; it++) {
    for (size_t i = (size_t)local; i < nlines; i += (size_t)n_local) {
      int stack = (int)(i & 7);
      int aid = stack >> 2; // 0-3 -> AID0, 4-7 -> AID1
      if (aid_sel >= 0 && aid != aid_sel)
        continue;
      const uint64_t *p =
          (const uint64_t *)((const char *)data + i * kLine);
      // 256B = 32 x uint64; stripe threads across the line
      for (int k = tid; k < 32; k += nthreads)
        acc += __builtin_nontemporal_load(p + k);
    }
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// Same STREAM but 2MiB page coloring (hipMalloc recommended granule).
__global__ void stream_2mb_kernel(const uint64_t *data, size_t nbytes,
                                  int aid_sel, int target_xcc, int num_xcd,
                                  int iters, uint64_t *sink) {
  if ((int)get_xcc_id() != target_xcc)
    return;

  constexpr size_t kPage = 2ull << 20;
  constexpr size_t kChunk = 256;
  const size_t npage = nbytes / kPage;
  const int n_local = gridDim.x / num_xcd;
  const int local = blockIdx.x / num_xcd;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  uint64_t acc = 0;

  for (int it = 0; it < iters; it++) {
    for (size_t pg = (size_t)local; pg < npage; pg += (size_t)n_local) {
      int aid = (int)(pg & 1);
      if (aid_sel >= 0 && aid != aid_sel)
        continue;
      const char *base = (const char *)data + pg * kPage;
      const size_t nchunk = kPage / kChunk;
      for (size_t c = (size_t)tid; c < nchunk; c += (size_t)nthreads) {
        const uint64_t *p = (const uint64_t *)(base + c * kChunk);
#pragma unroll
        for (int k = 0; k < 32; k++)
          acc += __builtin_nontemporal_load(p + k);
      }
    }
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

struct PowerSampler {
  std::string path;
  std::atomic<bool> stop{false};
  std::atomic<int64_t> sum_mw{0};
  std::atomic<int> n{0};

  explicit PowerSampler(std::string p) : path(std::move(p)) {}

  double read_w() const {
    std::ifstream f(path);
    long uW = 0;
    if (!(f >> uW) || uW <= 0)
      return 0.0;
    return uW / 1e6;
  }

  void run() {
    while (!stop.load()) {
      double w = read_w();
      if (w > 0) {
        sum_mw.fetch_add((int64_t)(w * 1000.0));
        n.fetch_add(1);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  double avg_w() const {
    int k = n.load();
    return k ? (sum_mw.load() / 1000.0) / k : 0.0;
  }
};

static std::string power_path_for_device(int dev) {
  char bdf[32];
  CHECK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), dev));
  std::string id = bdf;
  if (id.size() < 12)
    id = std::string("0000:") + id;
  std::string cmd = "ls /sys/bus/pci/devices/" + id + "/hwmon/hwmon*/power1_input 2>/dev/null";
  FILE *pipe = popen(cmd.c_str(), "r");
  if (!pipe)
    return {};
  char buf[256] = {};
  std::string path;
  if (fgets(buf, sizeof(buf), pipe)) {
    path = buf;
    while (!path.empty() && (path.back() == '\n' || path.back() == ' '))
      path.pop_back();
  }
  pclose(pipe);
  return path;
}

static void run_pingpong(int grid, int xcc_a, int xcc_b, int rounds, int wall_khz,
                         const std::string &pwr, double idle_w, const char *label) {
  int *d_flag = nullptr, *d_claim = nullptr;
  uint64_t *d_cyc = nullptr;
  CHECK(hipExtMallocWithFlags((void **)&d_flag, 4096, hipDeviceMallocUncached));
  CHECK(hipMalloc(&d_claim, 2 * sizeof(int)));
  CHECK(hipMalloc(&d_cyc, sizeof(uint64_t)));
  CHECK(hipMemset(d_flag, 0, sizeof(int)));
  CHECK(hipMemset(d_claim, 0, 2 * sizeof(int)));
  CHECK(hipMemset(d_cyc, 0, sizeof(uint64_t)));

  PowerSampler samp(pwr);
  std::thread th;
  if (!pwr.empty())
    th = std::thread([&] { samp.run(); });

  hipEvent_t ev0, ev1;
  CHECK(hipEventCreate(&ev0));
  CHECK(hipEventCreate(&ev1));
  CHECK(hipEventRecord(ev0));
  pingpong_kernel<<<grid, 64>>>(d_flag, xcc_a, xcc_b, rounds, d_claim, d_cyc);
  CHECK(hipEventRecord(ev1));
  CHECK(hipEventSynchronize(ev1));
  float ms = 0;
  CHECK(hipEventElapsedTime(&ms, ev0, ev1));
  CHECK(hipEventDestroy(ev0));
  CHECK(hipEventDestroy(ev1));

  if (!pwr.empty()) {
    samp.stop = true;
    th.join();
  }

  uint64_t cyc = 0;
  CHECK(hipMemcpy(&cyc, d_cyc, sizeof(uint64_t), hipMemcpyDeviceToHost));
  int claim[2] = {};
  CHECK(hipMemcpy(claim, d_claim, 2 * sizeof(int), hipMemcpyDeviceToHost));
  double ns = (wall_khz > 0) ? (double)cyc * 1e6 / wall_khz / rounds : 0;
  double watts = pwr.empty() ? 0 : samp.avg_w();
  printf("  %-28s  XCD %d <-> %d  %6.1f ns/round  %6.0f W (idle %.0f)  "
         "claimed=%d,%d  wall=%.2f ms\n",
         label, xcc_a, xcc_b, ns, watts, idle_w, claim[0], claim[1], ms);

  CHECK(hipFree(d_flag));
  CHECK(hipFree(d_claim));
  CHECK(hipFree(d_cyc));
}

int main(int argc, char **argv) {
  int dev = 0;
  if (argc > 1)
    dev = std::atoi(argv[1]);
  CHECK(hipSetDevice(dev));

  char bdf[32];
  CHECK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), dev));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, dev));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev));

  printf("device %d  %s  arch=%s  CUs=%d  VRAM=%.1f GiB  wall=%.1f MHz  pci=%s\n",
         dev, prop.name, prop.gcnArchName, prop.multiProcessorCount,
         prop.totalGlobalMem / (1024.0 * 1024 * 1024), wall_khz / 1000.0, bdf);

  const int nprobe = prop.multiProcessorCount * 2;
  int *d_xcc = nullptr;
  CHECK(hipMalloc(&d_xcc, nprobe * sizeof(int)));
  probe_xcc<<<nprobe, 64>>>(d_xcc, nprobe);
  CHECK(hipDeviceSynchronize());
  std::vector<int> xcc(nprobe);
  CHECK(hipMemcpy(xcc.data(), d_xcc, nprobe * sizeof(int), hipMemcpyDeviceToHost));
  CHECK(hipFree(d_xcc));

  int max_xcc = 0;
  std::vector<int> xcc_count(16, 0);
  for (int v : xcc) {
    if (v >= 0 && v < 16) {
      xcc_count[v]++;
      max_xcc = std::max(max_xcc, v);
    }
  }
  const int num_xcd = max_xcc + 1;
  const int xcd_per_aid = std::max(1, num_xcd / 2);
  printf("XCDs=%d  (AID = XCD / %d). HBM is attached to AIDs, not to XCDs.\n",
         num_xcd, xcd_per_aid);
  for (int i = 0; i < num_xcd; i++)
    printf("  XCD %d  AID %d  blocks=%d\n", i, i / xcd_per_aid, xcc_count[i]);

  std::string pwr = power_path_for_device(dev);
  double idle_w = pwr.empty() ? 0 : PowerSampler(pwr).read_w();
  if (pwr.empty())
    fprintf(stderr, "warning: no power1_input for this GPU\n");
  else
    printf("power: %s  idle=%.0f W\n", pwr.c_str(), idle_w);

  const int grid = 256;
  const int block = 256;
  const int target_xcc = 0;
  const int same_aid_xcc = 1;              // AID 0
  const int remote_aid_xcc = xcd_per_aid;  // XCD 4 on MI350 = AID 1

  printf("\n== 1. HBM flag ping-pong (one chiplet each) ==\n");
  printf("Same AID: both XCDs share the AID-local fabric to that AID's HBM.\n"
         "Cross AID: the flag round-trip crosses the inter-AID interconnect.\n");
  const int rounds = 2000000;
  run_pingpong(grid, target_xcc, same_aid_xcc, rounds, wall_khz, pwr, idle_w,
               "same AID (XCD0-XCD1)");
  run_pingpong(grid, target_xcc, remote_aid_xcc, rounds, wall_khz, pwr, idle_w,
               "remote AID (XCD0-XCD4)");

  const size_t bytes = 4ull << 30;
  uint64_t *d_data = nullptr;
  uint64_t *d_sink = nullptr;
  CHECK(hipMalloc(&d_data, bytes));
  CHECK(hipMalloc(&d_sink, (size_t)grid * block * sizeof(uint64_t)));
  CHECK(hipMemset(d_data, 1, bytes));

  auto stream_case = [&](const char *label, bool page2mb, int aid_sel, int iters) {
    CHECK(hipMemset(d_sink, 0, (size_t)grid * block * sizeof(uint64_t)));
    PowerSampler samp(pwr);
    std::thread th;
    if (!pwr.empty())
      th = std::thread([&] { samp.run(); });
    hipEvent_t a, b;
    CHECK(hipEventCreate(&a));
    CHECK(hipEventCreate(&b));
    CHECK(hipEventRecord(a));
    if (page2mb)
      stream_2mb_kernel<<<grid, block>>>(d_data, bytes, aid_sel, target_xcc,
                                         num_xcd, iters, d_sink);
    else
      stream_stacks_kernel<<<grid, block>>>(d_data, bytes, aid_sel, target_xcc,
                                            num_xcd, iters, d_sink);
    CHECK(hipEventRecord(b));
    CHECK(hipEventSynchronize(b));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, a, b));
    CHECK(hipEventDestroy(a));
    CHECK(hipEventDestroy(b));
    if (!pwr.empty()) {
      samp.stop = true;
      th.join();
    }
    double frac = (aid_sel < 0) ? 1.0 : 0.5;
    double gbps = (ms > 0) ? (bytes * frac * iters) / (ms * 1e6) : 0;
    printf("  %-36s  %7.1f GB/s  %6.0f W  (%.2f s)\n", label, gbps,
           pwr.empty() ? 0 : samp.avg_w(), ms / 1000.0);
  };

  printf("\n== 2. NT STREAM from XCD 0, 256B stack groups (4 stacks/AID) ==\n");
  stream_case("all stacks (NPS1 mix)", false, -1, 40);
  stream_case("stacks 0-3 (AID 0 if 256B RR)", false, 0, 40);
  stream_case("stacks 4-7 (AID 1 if 256B RR)", false, 1, 40);

  printf("\n== 3. NT STREAM from XCD 0, 2MiB page groups ==\n");
  stream_case("even 2MiB pages", true, 0, 40);
  stream_case("odd  2MiB pages", true, 1, 40);

  printf("\nIf ping-pong (1) is slower/higher-power cross-AID, that is the "
         "paper's remote-AID hop.\n"
         "STREAM (2-3) splits only if NPS1 physical interleave matches that "
         "coloring; hashed interleave looks even.\n");

  CHECK(hipFree(d_data));
  CHECK(hipFree(d_sink));
  return 0;
}
