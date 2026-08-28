// NEGATIVE RESULT -- this probe is blind. Use aid_home_map.hip instead.
//
// Hammering one address with atomics from a single chiplet leaves the line
// owned by that chiplet, so the request never travels to the home node and
// every address reads back the same ~94 ns from every XCD. The home only shows
// up when ownership has to move between chiplets, which aid_home_map does with
// a two-chiplet ping-pong and which resolves the same addresses into a 1.7x
// bimodal split. Kept as a record of the failure mode.
//
// Is a physical address owned by one AID, and can software see which?
//
// For each probed address we time a dependent self-loop load (uncached, so
// every load is a real HBM round trip) from a chiplet on AID 0 and from a
// chiplet on AID 1, and take the difference.
//
//   addr owned by AID 0  ->  XCD0 fast, XCD4 slow  ->  delta > 0
//   addr owned by AID 1  ->  XCD0 slow, XCD4 fast  ->  delta < 0
//   addr striped across both -> delta ~ 0 for every address
//
// A bimodal delta means AID ownership is an address property and a NUMA-aware
// allocator is possible. A unimodal delta at ~0 means the interleave is finer
// than a cache line and no allocation can be made AID-local.
//
// The XCD0/XCD1 pair (same AID) is the control: it must come out flat.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_page_map.hip -o aid_page_map
//   HIP_VISIBLE_DEVICES=1 ./aid_page_map

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
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
  if (blockIdx.x < (unsigned)n && threadIdx.x == 0)
    out[blockIdx.x] = (int)get_xcc_id();
}

// chain[off] == off, so an atomic add of 0 returns off and the chain stays on
// the probed address. Agent-scope atomics are resolved at the address's home
// point, so this times the fabric round trip to whichever AID owns it. A plain
// load would just sit in L1/L2 (~56 ns) and hide the topology.
__device__ void probe_all(uint32_t *chain, const uint64_t *offsets, int nlines,
                          int iters, uint64_t *out, uint32_t *sink) {
  for (int l = 0; l < nlines; l++) {
    uint32_t idx = (uint32_t)(offsets[l] / sizeof(uint32_t));
    for (int w = 0; w < 32; w++)
      idx = __hip_atomic_fetch_add(&chain[idx], 0u, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
    uint64_t t0 = wall_clock64();
#pragma unroll 1
    for (int i = 0; i < iters; i++)
      idx = __hip_atomic_fetch_add(&chain[idx], 0u, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
    uint64_t t1 = wall_clock64();
    out[l] = t1 - t0;
    sink[l] = idx; // keeps the chain live
  }
}

// One wave on each of two chiplets, run strictly one after the other so they
// never contend for the same line.
__global__ void two_xcd_probe(uint32_t *chain, const uint64_t *offsets,
                              int nlines, int iters, int xcc_a, int xcc_b,
                              int *claim, int *done, uint64_t *out_a,
                              uint64_t *out_b, uint32_t *sink) {
  int x = (int)get_xcc_id();
  if ((x != xcc_a && x != xcc_b) || threadIdx.x != 0)
    return;
  int slot = (x == xcc_a) ? 0 : 1;
  if (atomicCAS(&claim[slot], 0, 1) != 0)
    return;

  if (slot == 0) {
    probe_all(chain, offsets, nlines, iters, out_a, sink);
    __hip_atomic_store(done, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  } else {
    while (__hip_atomic_load(done, __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_AGENT) == 0) {
    }
    probe_all(chain, offsets, nlines, iters, out_b, sink + nlines);
  }
}

// Every XCD probes the same addresses, one XCD at a time. If an address were
// homed on one AID, XCDs 0-3 and 4-7 would land in two visibly separate groups.
__global__ void all_xcd_probe(uint32_t *chain, const uint64_t *offsets,
                              int nlines, int iters, int num_xcd, int *claim,
                              int *turn, uint64_t *out, uint32_t *sink) {
  int x = (int)get_xcc_id();
  if (threadIdx.x != 0 || x >= num_xcd)
    return;
  if (atomicCAS(&claim[x], 0, 1) != 0)
    return;

  while (__hip_atomic_load(turn, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) !=
         x) {
  }
  probe_all(chain, offsets, nlines, iters, out + (size_t)x * nlines,
            sink + (size_t)x * nlines);
  __hip_atomic_store(turn, x + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

struct Stats {
  double mean, sd, lo, hi;
};

static Stats stats_of(const std::vector<double> &v) {
  Stats s{0, 0, v[0], v[0]};
  for (double x : v) {
    s.mean += x;
    s.lo = std::min(s.lo, x);
    s.hi = std::max(s.hi, x);
  }
  s.mean /= v.size();
  for (double x : v)
    s.sd += (x - s.mean) * (x - s.mean);
  s.sd = std::sqrt(s.sd / v.size());
  return s;
}

int main(int argc, char **argv) {
  int dev = (argc > 1) ? std::atoi(argv[1]) : 0;
  CHECK(hipSetDevice(dev));

  char bdf[32];
  CHECK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), dev));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, dev));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev));
  printf("device %d  %s  pci=%s  CUs=%d  wall=%.1f MHz\n", dev, prop.gcnArchName,
         bdf, prop.multiProcessorCount, wall_khz / 1000.0);

  const int nprobe = prop.multiProcessorCount * 2;
  int *d_xcc = nullptr;
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
  const int xcd_per_aid = std::max(1, num_xcd / 2);
  printf("XCDs=%d, %d per AID\n\n", num_xcd, xcd_per_aid);

  // Uncached buffer so loads bypass L2 and reach the memory fabric.
  const size_t bytes = 1ull << 30;
  uint32_t *d_chain = nullptr;
  CHECK(hipExtMallocWithFlags((void **)&d_chain, bytes,
                              hipDeviceMallocUncached));

  const int nlines = 128;
  const int iters = 4000;

  struct Sweep {
    const char *name;
    size_t stride;
  };
  const Sweep sweeps[] = {
      {"256 B steps (48 KiB span)", 256},
      {"4 KiB steps (768 KiB span)", 4096},
      {"2 MiB steps (384 MiB span)", 2ull << 20},
  };

  uint64_t *d_off = nullptr, *d_a = nullptr, *d_b = nullptr;
  uint32_t *d_sink = nullptr;
  int *d_claim = nullptr, *d_done = nullptr;
  CHECK(hipMalloc(&d_off, nlines * sizeof(uint64_t)));
  CHECK(hipMalloc(&d_a, nlines * sizeof(uint64_t)));
  CHECK(hipMalloc(&d_b, nlines * sizeof(uint64_t)));
  CHECK(hipMalloc(&d_sink, 2 * nlines * sizeof(uint32_t)));
  CHECK(hipMalloc(&d_claim, 2 * sizeof(int)));
  CHECK(hipMalloc(&d_done, sizeof(int)));

  auto run_pair = [&](const std::vector<uint64_t> &offs, int xcc_a, int xcc_b,
                      std::vector<double> *lat_a, std::vector<double> *lat_b) {
    CHECK(hipMemcpy(d_off, offs.data(), offs.size() * sizeof(uint64_t),
                    hipMemcpyHostToDevice));
    CHECK(hipMemset(d_claim, 0, 2 * sizeof(int)));
    CHECK(hipMemset(d_done, 0, sizeof(int)));
    two_xcd_probe<<<prop.multiProcessorCount, 64>>>(
        d_chain, d_off, (int)offs.size(), iters, xcc_a, xcc_b, d_claim, d_done,
        d_a, d_b, d_sink);
    CHECK(hipDeviceSynchronize());
    std::vector<uint64_t> ca(offs.size()), cb(offs.size());
    CHECK(hipMemcpy(ca.data(), d_a, ca.size() * sizeof(uint64_t),
                    hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(cb.data(), d_b, cb.size() * sizeof(uint64_t),
                    hipMemcpyDeviceToHost));
    lat_a->resize(offs.size());
    lat_b->resize(offs.size());
    for (size_t i = 0; i < offs.size(); i++) {
      (*lat_a)[i] = (double)ca[i] * 1e6 / wall_khz / iters;
      (*lat_b)[i] = (double)cb[i] * 1e6 / wall_khz / iters;
    }
  };

  for (const Sweep &sw : sweeps) {
    std::vector<uint64_t> offs(nlines);
    for (int i = 0; i < nlines; i++)
      offs[i] = (uint64_t)i * sw.stride;
    // Seed the self-loop for every probed address.
    std::vector<uint32_t> seed(nlines);
    for (int i = 0; i < nlines; i++) {
      uint32_t w = (uint32_t)(offs[i] / sizeof(uint32_t));
      seed[i] = w;
      CHECK(hipMemcpy(d_chain + w, &seed[i], sizeof(uint32_t),
                      hipMemcpyHostToDevice));
    }

    printf("== %s ==\n", sw.name);
    for (int cross = 0; cross < 2; cross++) {
      const int xcc_a = 0;
      const int xcc_b = cross ? xcd_per_aid : 1;
      std::vector<double> la, lb;
      run_pair(offs, xcc_a, xcc_b, &la, &lb);

      std::vector<double> delta(nlines);
      for (int i = 0; i < nlines; i++)
        delta[i] = lb[i] - la[i];
      Stats sa = stats_of(la), sb = stats_of(lb), sd = stats_of(delta);

      // Bimodality: split at the midpoint of the range and compare clusters.
      std::vector<double> sorted = delta;
      std::sort(sorted.begin(), sorted.end());
      double mid = (sorted.front() + sorted.back()) / 2;
      double lo_sum = 0, hi_sum = 0;
      int lo_n = 0, hi_n = 0;
      for (double d : delta) {
        if (d < mid) {
          lo_sum += d;
          lo_n++;
        } else {
          hi_sum += d;
          hi_n++;
        }
      }
      double gap = (lo_n && hi_n) ? (hi_sum / hi_n - lo_sum / lo_n) : 0;

      printf("  XCD0 vs XCD%-2d (%-9s)  lat %6.1f / %6.1f ns   "
             "delta mean %+6.1f sd %4.1f range [%+.0f,%+.0f]  "
             "cluster gap %5.1f (%d/%d)\n",
             xcc_b, cross ? "other AID" : "same AID", sa.mean, sb.mean, sd.mean,
             sd.sd, sd.lo, sd.hi, gap, lo_n, hi_n);

      if (cross) {
        // If a single address bit picks the AID, it shows up here.
        printf("      per-bit delta split (bit: mean_delta[bit=1] - "
               "mean_delta[bit=0]):\n      ");
        for (int b = 8; b <= 26; b++) {
          double s1 = 0, s0 = 0;
          int n1 = 0, n0 = 0;
          for (int i = 0; i < nlines; i++) {
            if ((offs[i] >> b) & 1) {
              s1 += delta[i];
              n1++;
            } else {
              s0 += delta[i];
              n0++;
            }
          }
          if (n1 && n0)
            printf("b%d %+.0f  ", b, s1 / n1 - s0 / n0);
        }
        printf("\n");
      }
    }
    printf("\n");
  }

  // Latency from every chiplet to a handful of fixed addresses.
  {
    const int k = 6;
    std::vector<uint64_t> offs = {0,          4096,       1ull << 20,
                                  17ull << 20, 129ull << 20, 400ull << 20};
    offs.resize(k);
    for (int i = 0; i < k; i++) {
      uint32_t w = (uint32_t)(offs[i] / sizeof(uint32_t));
      CHECK(hipMemcpy(d_chain + w, &w, sizeof(uint32_t), hipMemcpyHostToDevice));
    }
    CHECK(hipMemcpy(d_off, offs.data(), k * sizeof(uint64_t),
                    hipMemcpyHostToDevice));

    uint64_t *d_all = nullptr;
    uint32_t *d_asink = nullptr;
    int *d_claim8 = nullptr, *d_turn = nullptr;
    CHECK(hipMalloc(&d_all, (size_t)num_xcd * k * sizeof(uint64_t)));
    CHECK(hipMalloc(&d_asink, (size_t)num_xcd * k * sizeof(uint32_t)));
    CHECK(hipMalloc(&d_claim8, num_xcd * sizeof(int)));
    CHECK(hipMalloc(&d_turn, sizeof(int)));
    CHECK(hipMemset(d_claim8, 0, num_xcd * sizeof(int)));
    CHECK(hipMemset(d_turn, 0, sizeof(int)));

    all_xcd_probe<<<prop.multiProcessorCount, 64>>>(
        d_chain, d_off, k, iters, num_xcd, d_claim8, d_turn, d_all, d_asink);
    CHECK(hipDeviceSynchronize());

    std::vector<uint64_t> all((size_t)num_xcd * k);
    CHECK(hipMemcpy(all.data(), d_all, all.size() * sizeof(uint64_t),
                    hipMemcpyDeviceToHost));

    printf("== atomic latency (ns) from every chiplet to a fixed address ==\n");
    printf("  %-14s", "offset");
    for (int x = 0; x < num_xcd; x++)
      printf(" XCD%d", x);
    printf("   AID0avg AID1avg  split\n");
    for (int i = 0; i < k; i++) {
      printf("  %-14s", (std::to_string(offs[i] >> 10) + " KiB").c_str());
      double a0 = 0, a1 = 0;
      for (int x = 0; x < num_xcd; x++) {
        double ns = (double)all[(size_t)x * k + i] * 1e6 / wall_khz / iters;
        printf(" %4.0f", ns);
        (x < xcd_per_aid ? a0 : a1) += ns / xcd_per_aid;
      }
      printf("   %7.1f %7.1f  %+5.1f\n", a0, a1, a1 - a0);
    }
    printf("\n");

    CHECK(hipFree(d_all));
    CHECK(hipFree(d_asink));
    CHECK(hipFree(d_claim8));
    CHECK(hipFree(d_turn));
  }

  CHECK(hipFree(d_chain));
  CHECK(hipFree(d_off));
  CHECK(hipFree(d_a));
  CHECK(hipFree(d_b));
  CHECK(hipFree(d_sink));
  CHECK(hipFree(d_claim));
  CHECK(hipFree(d_done));
  return 0;
}
