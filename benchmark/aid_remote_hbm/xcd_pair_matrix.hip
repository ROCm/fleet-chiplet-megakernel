// Round-trip latency between every pair of chiplets.
//
// Allocation placement cannot be steered toward an AID in SPX/NPS1 (see
// aid_page_map). What software still controls is which chiplets talk to each
// other. This measures that directly: an uncached flag bounced between one
// wave on chiplet A and one on chiplet B, for all 28 pairs.
//
// The output tells you which worker pairings a cross-XCD barrier or
// producer/consumer hand-off should avoid.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/xcd_pair_matrix.hip -o xcd_pair_matrix
//   HIP_VISIBLE_DEVICES=1 ./xcd_pair_matrix

#include <hip/hip_runtime.h>

#include <algorithm>
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

__global__ void pingpong(int *flag, int xcc_a, int xcc_b, int rounds,
                         int *claim, uint64_t *cycles) {
  int x = (int)get_xcc_id();
  if ((x != xcc_a && x != xcc_b) || threadIdx.x != 0)
    return;
  int slot = (x == xcc_a) ? 0 : 1;
  if (atomicCAS(&claim[slot], 0, 1) != 0)
    return;

  if (slot == 0) {
    uint64_t t0 = wall_clock64();
    for (int r = 0; r < rounds; r++) {
      while (__hip_atomic_load(flag, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT) != r * 2) {
      }
      __hip_atomic_store(flag, r * 2 + 1, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
    cycles[0] = wall_clock64() - t0;
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

int main(int argc, char **argv) {
  int dev = (argc > 1) ? std::atoi(argv[1]) : 0;
  CHECK(hipSetDevice(dev));

  char bdf[32];
  CHECK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), dev));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, dev));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev));
  printf("device %d  %s  pci=%s\n", dev, prop.gcnArchName, bdf);

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
  const int per_aid = std::max(1, num_xcd / 2);

  int *d_flag = nullptr, *d_claim = nullptr;
  uint64_t *d_cyc = nullptr;
  CHECK(hipExtMallocWithFlags((void **)&d_flag, 4096, hipDeviceMallocUncached));
  CHECK(hipMalloc(&d_claim, 2 * sizeof(int)));
  CHECK(hipMalloc(&d_cyc, sizeof(uint64_t)));

  const int rounds = 300000;
  std::vector<double> m((size_t)num_xcd * num_xcd, 0.0);

  for (int a = 0; a < num_xcd; a++) {
    for (int b = a + 1; b < num_xcd; b++) {
      CHECK(hipMemset(d_flag, 0, sizeof(int)));
      CHECK(hipMemset(d_claim, 0, 2 * sizeof(int)));
      CHECK(hipMemset(d_cyc, 0, sizeof(uint64_t)));
      pingpong<<<prop.multiProcessorCount, 64>>>(d_flag, a, b, rounds, d_claim,
                                                 d_cyc);
      CHECK(hipDeviceSynchronize());
      uint64_t cyc = 0;
      CHECK(hipMemcpy(&cyc, d_cyc, sizeof(uint64_t), hipMemcpyDeviceToHost));
      double ns = (double)cyc * 1e6 / wall_khz / rounds;
      m[(size_t)a * num_xcd + b] = ns;
      m[(size_t)b * num_xcd + a] = ns;
    }
  }

  printf("\nround-trip ns between chiplets (AID = XCD/%d)\n\n     ", per_aid);
  for (int b = 0; b < num_xcd; b++)
    printf("XCD%d ", b);
  printf("\n");
  for (int a = 0; a < num_xcd; a++) {
    printf("XCD%d ", a);
    for (int b = 0; b < num_xcd; b++) {
      if (a == b)
        printf("   . ");
      else
        printf("%4.0f ", m[(size_t)a * num_xcd + b]);
    }
    printf("  %s\n", a < per_aid ? "AID0" : "AID1");
  }

  double same = 0, cross = 0;
  int ns_ = 0, nc = 0;
  for (int a = 0; a < num_xcd; a++)
    for (int b = a + 1; b < num_xcd; b++) {
      bool sameaid = (a / per_aid) == (b / per_aid);
      if (sameaid) {
        same += m[(size_t)a * num_xcd + b];
        ns_++;
      } else {
        cross += m[(size_t)a * num_xcd + b];
        nc++;
      }
    }
  same /= ns_;
  cross /= nc;
  printf("\nsame-AID pairs  : %6.1f ns (n=%d)\n", same, ns_);
  printf("cross-AID pairs : %6.1f ns (n=%d)\n", cross, nc);
  printf("penalty         : %+.1f ns (%.2fx)\n", cross - same, cross / same);

  // Does the fast/slow grouping follow the flag's address? If some addresses
  // make the 0-3 group the fast one, homing is address-dependent and software
  // could pick addresses near its chiplets. If the grouping never moves, it is
  // a fixed property of the chiplets and no address choice helps.
  CHECK(hipFree(d_flag));
  const size_t span = 2ull << 30;
  char *d_arena = nullptr;
  CHECK(hipExtMallocWithFlags((void **)&d_arena, span, hipDeviceMallocUncached));

  const size_t offsets[] = {0,           4096,        1ull << 20,  3ull << 20,
                            64ull << 20, 257ull << 20, 512ull << 20,
                            1023ull << 20, 1ull << 30, (1ull << 30) + 4096,
                            1536ull << 20, 2047ull << 20};
  const int noff = (int)(sizeof(offsets) / sizeof(offsets[0]));
  const int sweep_rounds = 150000;

  struct Pair {
    int a, b;
    const char *tag;
  };
  const Pair pairs[] = {
      {0, 1, "0-1 (both low group)"},
      {4, 5, "4-5 (both high group)"},
      {0, 4, "0-4 (across groups)"},
  };

  printf("\n== flag address sweep: does homing follow the address? ==\n");
  printf("  %-24s", "offset");
  for (const Pair &p : pairs)
    printf(" %10s", p.tag ? (std::string("XCD") + std::to_string(p.a) + "-" +
                             std::to_string(p.b))
                                .c_str()
                          : "");
  printf("\n");

  for (int o = 0; o < noff; o++) {
    int *flag = (int *)(d_arena + offsets[o]);
    printf("  %-24s", (std::to_string(offsets[o] >> 20) + " MiB").c_str());
    for (const Pair &p : pairs) {
      CHECK(hipMemset(flag, 0, sizeof(int)));
      CHECK(hipMemset(d_claim, 0, 2 * sizeof(int)));
      CHECK(hipMemset(d_cyc, 0, sizeof(uint64_t)));
      pingpong<<<prop.multiProcessorCount, 64>>>(flag, p.a, p.b, sweep_rounds,
                                                 d_claim, d_cyc);
      CHECK(hipDeviceSynchronize());
      uint64_t cyc = 0;
      CHECK(hipMemcpy(&cyc, d_cyc, sizeof(uint64_t), hipMemcpyDeviceToHost));
      printf(" %10.0f", (double)cyc * 1e6 / wall_khz / sweep_rounds);
    }
    printf("\n");
  }
  CHECK(hipFree(d_arena));
  CHECK(hipMalloc(&d_flag, sizeof(int))); // keep the free below well-formed

  CHECK(hipFree(d_flag));
  CHECK(hipFree(d_claim));
  CHECK(hipFree(d_cyc));
  return 0;
}
