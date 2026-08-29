// One binary, two partition modes, identical work: what does NPS2 actually buy?
//
// Run it unchanged under SPX/NPS1 (8 XCDs, 252 GiB, memory hash-interleaved
// across both AIDs so ~50% of reads are remote) and under DPX/NPS2 (4 XCDs,
// 126 GiB, all memory homed on the partition's own AID, 0% remote). The delta
// is the value of AID locality with no software mechanism involved at all --
// no VMM, no page remapping, no startup cost.
//
// Making the two comparable takes some care, because the modes do not offer the
// same machine:
//
//   * XCD count differs (8 vs 4). So only XCDs 0-3 ever do work; blocks landing
//     on 4-7 exit immediately. Both modes then run on exactly 4 XCDs.
//   * Block->XCD assignment is round-robin (blockIdx.x % nxcd), verified on
//     both, so the grid is sized nxcd*K to give every XCD exactly K blocks.
//     Active workers = 4*K in both modes.
//   * Buffer size and access pattern are fixed, and must fit the 126 GiB
//     partition.
//
// Two regimes, because they answer different questions and can disagree:
//
//   latency   - dependent pointer chase, one thread per worker. Sensitive to
//               the remote hop, insensitive to aggregate bandwidth. NPS2 should
//               win here and the NPS1 distribution should be visibly bimodal
//               (near vs far AID) while NPS2 is unimodal.
//   bandwidth - saturating non-temporal streaming. Here NPS2 is *not*
//               obviously better: it removes the crossings but also confines
//               the partition to half the HBM stacks, while NPS1's 4 XCDs draw
//               on both AIDs' stacks. Which effect dominates is the measurement.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_nps_compare.hip -o aid_nps_compare
//   HIP_VISIBLE_DEVICES=<dev> ./aid_nps_compare [GiB] [K]

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

typedef unsigned int v4u __attribute__((ext_vector_type(4)));

static constexpr int kActiveXCD = 4; // both modes have at least this many
static constexpr size_t kLine = 64;
// Coprime-with-nlines stride so the walk is a single full-length cycle. Large
// enough (~96 MiB) that no prefetcher or row buffer helps and successive hops
// land on unrelated pages, hence on both AIDs under NPS1.
static constexpr uint64_t kStride = 1572869;

__device__ inline unsigned get_xcc_id() {
  constexpr unsigned imm = ((4u - 1u) << 11) | (0u << 6) | 20u;
  return __builtin_amdgcn_s_getreg(imm);
}

// Lay the cycle down: the line visited at step i points at the line for i+1.
__global__ void build_chain(uint64_t *__restrict base, uint64_t nlines) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nlines)
    return;
  uint64_t cur = (i * kStride) % nlines;
  uint64_t nxt = ((i + 1) * kStride) % nlines;
  base[cur * (kLine / 8)] = nxt * (kLine / 8);
}

__global__ __launch_bounds__(256) void chase(const uint64_t *__restrict base,
                                             uint64_t nlines, int nxcd,
                                             int hops, uint64_t *__restrict out,
                                             uint64_t *__restrict sink) {
  const int xcc = (int)get_xcc_id();
  if (xcc >= kActiveXCD || threadIdx.x != 0)
    return;
  const int k = blockIdx.x / nxcd;   // rank within this XCD
  const int worker = xcc * (gridDim.x / nxcd) + k;

  // Spread starting points around the cycle so workers are not in lockstep.
  uint64_t idx =
      (((uint64_t)worker * (nlines / 64 + 1)) % nlines) * (kLine / 8);
  uint64_t t0 = wall_clock64();
  for (int h = 0; h < hops; h++)
    idx = base[idx];
  uint64_t t1 = wall_clock64();
  out[worker] = (t1 - t0) / (uint64_t)hops;
  sink[worker] = idx;
}

// HBM here is ~8 TB/s, so a single 16 B load per thread per trip cannot keep the
// memory system busy -- each thread stalls on its one outstanding miss. UNROLL
// independent loads per trip give enough memory-level parallelism to cover the
// ~350 ns latency; the addresses are independent so they all issue before any
// result is consumed.
#define UNROLL 8

__global__ __launch_bounds__(256) void stream(const char *__restrict base,
                                              size_t bytes, int nxcd, int iters,
                                              uint64_t *__restrict sink) {
  const int xcc = (int)get_xcc_id();
  if (xcc >= kActiveXCD)
    return;
  const int per = gridDim.x / nxcd;
  const int k = blockIdx.x / nxcd;
  const int worker = xcc * per + k;
  const int nworkers = kActiveXCD * per;

  const size_t vecs = bytes / 16;
  const size_t span = (size_t)nworkers * 256 * UNROLL;
  const v4u *p = (const v4u *)base;
  uint64_t acc = 0;
  for (int it = 0; it < iters; it++)
    for (size_t b = (size_t)worker * 256 * UNROLL + threadIdx.x; b < vecs;
         b += span) {
      v4u v[UNROLL];
#pragma unroll
      for (int u = 0; u < UNROLL; u++)
        v[u] = __builtin_nontemporal_load(p + min(b + (size_t)u * 256, vecs - 1));
#pragma unroll
      for (int u = 0; u < UNROLL; u++)
        acc += v[u].x + v[u].w;
    }
  sink[blockIdx.x * 256 + threadIdx.x] = acc;
}

int main(int argc, char **argv) {
  const size_t gib = (argc > 1) ? (size_t)std::atoll(argv[1]) : 8;
  const int K = (argc > 2) ? std::atoi(argv[2]) : 32;

  CHECK(hipSetDevice(0));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
  const int nxcd = prop.multiProcessorCount / 32;
  const int grid = nxcd * K;
  const int active = kActiveXCD * K;

  size_t freeb = 0, totb = 0;
  CHECK(hipMemGetInfo(&freeb, &totb));
  printf("CUs %d -> %d XCDs, %.0f GiB visible  [%s]\n", prop.multiProcessorCount,
         nxcd, totb / 1073741824.0,
         nxcd == 8 ? "SPX/NPS1 expected" : "partitioned/NPS2 expected");
  printf("grid %d (K=%d per XCD), active workers %d on XCDs 0-%d\n", grid, K,
         active, kActiveXCD - 1);

  const size_t bytes = gib << 30;
  char *buf = nullptr;
  CHECK(hipMalloc(&buf, bytes));
  CHECK(hipMemsetD8((hipDeviceptr_t)buf, 1, bytes));
  printf("buffer %zu GiB\n\n", gib);

  // ---- latency ----
  uint64_t nlines = bytes / kLine;
  while (std::__gcd(nlines, kStride) != 1)
    nlines--; // keep the walk a single cycle
  uint64_t *d_out = nullptr, *d_sink = nullptr;
  CHECK(hipMalloc(&d_out, (size_t)grid * sizeof(uint64_t)));
  CHECK(hipMalloc(&d_sink, (size_t)grid * 256 * sizeof(uint64_t)));
  CHECK(hipMemset(d_out, 0, (size_t)grid * sizeof(uint64_t)));
  build_chain<<<(unsigned)((nlines + 255) / 256), 256>>>((uint64_t *)buf,
                                                         nlines);
  CHECK(hipDeviceSynchronize());

  const int hops = 2048;
  chase<<<grid, 256>>>((const uint64_t *)buf, nlines, nxcd, hops, d_out,
                       d_sink);
  CHECK(hipDeviceSynchronize());
  std::vector<uint64_t> cyc(grid);
  CHECK(hipMemcpy(cyc.data(), d_out, (size_t)grid * sizeof(uint64_t),
                  hipMemcpyDeviceToHost));
  std::vector<double> lat;
  for (int i = 0; i < active; i++)
    if (cyc[i])
      lat.push_back((double)cyc[i] * 1e9 / (wall_khz * 1000.0));
  std::sort(lat.begin(), lat.end());
  double sum = 0;
  for (double v : lat)
    sum += v;
  printf("LATENCY  dependent chase, %d hops x %zu workers\n", hops, lat.size());
  printf("  mean %.1f ns   p10 %.1f   p50 %.1f   p90 %.1f   spread(p90/p10) "
         "%.3fx\n",
         sum / lat.size(), lat[lat.size() / 10], lat[lat.size() / 2],
         lat[lat.size() * 9 / 10],
         lat[lat.size() * 9 / 10] / lat[lat.size() / 10]);

  // ---- bandwidth ----
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  CHECK(hipMemsetD8((hipDeviceptr_t)buf, 1, bytes));
  auto bw = [&](int iters) {
    stream<<<grid, 256>>>(buf, bytes, nxcd, 1, d_sink);
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    stream<<<grid, 256>>>(buf, bytes, nxcd, iters, d_sink);
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    return (double)bytes * iters / (ms * 1e6);
  };
  double v[3];
  for (int i = 0; i < 3; i++)
    v[i] = bw(4);
  std::sort(v, v + 3);
  printf("\nBANDWIDTH  streaming read, %d active workers\n", active);
  printf("  %.1f GB/s  (median of 3)\n", v[1]);
  printf("\nSUMMARY xcds=%d lat_mean_ns=%.1f lat_p50_ns=%.1f spread=%.3f "
         "bw_gbs=%.1f\n",
         nxcd, sum / lat.size(), lat[lat.size() / 2],
         lat[lat.size() * 9 / 10] / lat[lat.size() / 10], v[1]);
  return 0;
}
