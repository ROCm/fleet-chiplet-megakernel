// How does hipMalloc lay pages out, and is the AID home pattern predictable?
//
// If the home were a function of low physical address bits and hipMalloc handed
// back physically contiguous, 2 MiB-aligned memory, the per-page pattern would
// repeat every 512 pages and aid_alloc could compute homes instead of probing
// them at 45 us/page. This checks that.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/hipmalloc_pages.hip -o hipmalloc_pages
//   HIP_VISIBLE_DEVICES=1 ./hipmalloc_pages

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
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

int main() {
  CHECK(hipSetDevice(0));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));

  printf("== hipMalloc pointer alignment ==\n");
  for (size_t sz : {(size_t)256, (size_t)4096, (size_t)(64 << 10),
                    (size_t)(1 << 20), (size_t)(2 << 20), (size_t)(8 << 20),
                    (size_t)(64 << 20), (size_t)(256 << 20)}) {
    void *p = nullptr;
    CHECK(hipMalloc(&p, sz));
    size_t a = (size_t)p;
    int align = 0;
    while (align < 30 && ((a >> align) & 1) == 0)
      align++;
    printf("  %8zu KiB -> %p  (aligned to %d KiB)\n", sz >> 10, p,
           (1 << align) >> 10);
    CHECK(hipFree(p));
  }

  const size_t npages = 4096; // 16 MiB
  const int rounds = 48;
  int *d_claim = nullptr;
  uint64_t *d_cyc = nullptr;
  CHECK(hipMalloc(&d_claim, 2 * sizeof(int)));
  CHECK(hipMalloc(&d_cyc, npages * sizeof(uint64_t)));

  auto classify = [&](void *buf, size_t n) {
    CHECK(hipMemset(buf, 0, n * kPage));
    CHECK(hipMemset(d_claim, 0, 2 * sizeof(int)));
    CHECK(hipMemset(d_cyc, 0, n * sizeof(uint64_t)));
    classify_pages<<<prop.multiProcessorCount, 64>>>((char *)buf, n, rounds, 0,
                                                     1, d_claim, d_cyc);
    CHECK(hipDeviceSynchronize());
    std::vector<uint64_t> c(n);
    CHECK(hipMemcpy(c.data(), d_cyc, n * sizeof(uint64_t),
                    hipMemcpyDeviceToHost));
    std::vector<double> ns(n);
    for (size_t i = 0; i < n; i++)
      ns[i] = (double)c[i] * 1e6 / wall_khz / rounds;
    std::vector<double> s = ns;
    std::sort(s.begin(), s.end());
    double th = (s[n / 10] + s[n * 9 / 10]) / 2;
    std::vector<int> cls(n);
    for (size_t i = 0; i < n; i++)
      cls[i] = ns[i] < th ? 0 : 1;
    return cls;
  };

  void *buf = nullptr;
  CHECK(hipMalloc(&buf, npages * kPage));
  printf("\n== 16 MiB hipMalloc buffer at %p ==\n", buf);
  std::vector<int> cls = classify(buf, npages);

  int ones = 0, flips = 0;
  for (size_t i = 0; i < npages; i++) {
    ones += cls[i];
    if (i && cls[i] != cls[i - 1])
      flips++;
  }
  printf("  %d/%zu pages on side 1, %d transitions\n", ones, npages, flips);

  printf("  first 64 pages: ");
  for (int i = 0; i < 64; i++) {
    printf("%d", cls[i]);
    if ((i + 1) % 8 == 0)
      printf(" ");
  }
  printf("\n");

  std::map<int, int> runs;
  for (size_t i = 0; i < npages;) {
    size_t j = i;
    while (j < npages && cls[j] == cls[i])
      j++;
    runs[(int)(j - i)]++;
    i = j;
  }
  printf("  run lengths: ");
  for (auto &kv : runs)
    printf("%dx%d ", kv.second, kv.first);
  printf("\n");

  printf("\n== periodicity (match fraction of cls[i] vs cls[i+P]) ==\n");
  std::vector<std::pair<double, int>> best;
  for (int P = 1; P <= 1024; P++) {
    int m = 0, n = 0;
    for (size_t i = 0; i + P < npages; i++, n++)
      m += (cls[i] == cls[i + P]);
    best.push_back({(double)m / n, P});
  }
  std::sort(best.rbegin(), best.rend());
  printf("  top periods: ");
  for (int i = 0; i < 8; i++)
    printf("P=%d %.2f  ", best[i].second, best[i].first);
  printf("\n  P=512 (2 MiB): %.3f    P=256 (1 MiB): %.3f\n",
         best[std::find_if(best.begin(), best.end(),
                           [](auto &p) { return p.second == 512; }) -
               best.begin()]
             .first,
         best[std::find_if(best.begin(), best.end(),
                           [](auto &p) { return p.second == 256; }) -
               best.begin()]
             .first);

  // Does a second allocation reproduce the pattern at the same VA offset?
  void *buf2 = nullptr;
  CHECK(hipMalloc(&buf2, npages * kPage));
  std::vector<int> cls2 = classify(buf2, npages);
  int same = 0, ones2 = 0;
  for (size_t i = 0; i < npages; i++) {
    same += (cls[i] == cls2[i]);
    ones2 += cls2[i];
  }
  printf("\n== second 16 MiB allocation at %p ==\n", buf2);
  printf("  first 64 pages: ");
  for (int i = 0; i < 64; i++) {
    printf("%d", cls2[i]);
    if ((i + 1) % 8 == 0)
      printf(" ");
  }
  printf("\n  %d/%zu on side 1; agrees with first allocation on %d/%zu pages "
         "(%.0f%%, chance is 50%%)\n",
         ones2, npages, same, npages, 100.0 * same / npages);

  // Both buffers show the same side-1 count and run-length profile, which
  // suggests one underlying sequence sampled at different physical offsets.
  // If so, some shift should line them up almost perfectly.
  int best_shift = 0;
  double best_frac = 0;
  for (int sh = -2048; sh <= 2048; sh++) {
    int m = 0, n = 0;
    for (size_t i = 0; i < npages; i++) {
      long j = (long)i + sh;
      if (j < 0 || j >= (long)npages)
        continue;
      m += (cls[i] == cls2[j]);
      n++;
    }
    if (n > 512 && (double)m / n > best_frac) {
      best_frac = (double)m / n;
      best_shift = sh;
    }
  }
  printf("  best alignment: shift %d pages -> %.1f%% agreement\n", best_shift,
         100 * best_frac);
  printf("  %s\n",
         best_frac > 0.95
             ? "=> one fixed sequence; probe a few pages for phase, predict "
               "the rest"
             : "=> not a shifted copy; homes must be probed per allocation");

  // If that sequence repeats, one probe characterises all of VRAM and the
  // allocator never has to probe again. Search well past 2 MiB.
  const size_t big_pages = 32768; // 128 MiB
  void *big = nullptr;
  CHECK(hipMalloc(&big, big_pages * kPage));
  CHECK(hipFree(d_cyc));
  CHECK(hipMalloc(&d_cyc, big_pages * sizeof(uint64_t)));
  printf("\n== period search over %zu MiB at %p ==\n", big_pages * kPage >> 20,
         big);
  std::vector<int> b = classify(big, big_pages);

  auto match_at = [&](int P) {
    int m = 0, n = 0;
    for (size_t i = 0; i + P < big_pages; i++, n++)
      m += (b[i] == b[i + P]);
    return (double)m / n;
  };

  // Round-robin across 8 stacks at 4 KiB, with stacks split 4/4 between the
  // AIDs, would make the side sequence periodic at 8 pages (and show runs of
  // 4). These are the periods that hypothesis predicts.
  printf("  small periods: ");
  for (int P : {2, 4, 8, 16, 32, 64, 128, 512})
    printf("P=%d %.2f  ", P, match_at(P));
  printf("\n");

  std::pair<double, int> top{0, 0};
  for (int P = 1; P <= 16384; P++) {
    double f = match_at(P);
    if (f > top.first)
      top = {f, P};
  }
  printf("  harmonics of best: ");
  for (int k = 1; k <= 3; k++)
    if (top.second * k <= 16384)
      printf("P=%d %.3f  ", top.second * k, match_at(top.second * k));
  printf("\n");
  printf("  best period P=%d pages (%.2f MiB) -> %.1f%% match\n", top.second,
         top.second * 4096 / 1048576.0, 100 * top.first);
  printf("  (this value moves between runs -- it reflects how the buddy "
         "allocator\n   laid out this allocation, not a hardware period)\n");

  // Granularity: if the AID bit round-robined across stacks at 256 B, a single
  // 4 KiB page would contain both sides. Probe 16 sub-page offsets per page.
  printf("\n== sub-page probe: 16 x 256 B within each of 8 pages ==\n");
  void *sub = nullptr;
  CHECK(hipMalloc(&sub, 8 * kPage));
  CHECK(hipFree(d_cyc));
  CHECK(hipMalloc(&d_cyc, 128 * sizeof(uint64_t)));
  CHECK(hipMemset(sub, 0, 8 * kPage));
  CHECK(hipMemset(d_claim, 0, 2 * sizeof(int)));
  CHECK(hipMemset(d_cyc, 0, 128 * sizeof(uint64_t)));
  // Stride the flag by 256 B instead of 4096 B by lying about the page size:
  // treat the region as 128 "pages" of 256 B.
  classify_pages<<<prop.multiProcessorCount, 64>>>((char *)sub, 0, rounds, 0, 1,
                                                   d_claim, d_cyc);
  CHECK(hipDeviceSynchronize());
  {
    std::vector<double> v(128);
    for (int i = 0; i < 128; i++) {
      CHECK(hipMemset(d_claim, 0, 2 * sizeof(int)));
      classify_pages<<<prop.multiProcessorCount, 64>>>(
          (char *)sub + (size_t)i * 256, 1, rounds, 0, 1, d_claim, d_cyc);
      CHECK(hipDeviceSynchronize());
      uint64_t c = 0;
      CHECK(hipMemcpy(&c, d_cyc, sizeof(uint64_t), hipMemcpyDeviceToHost));
      v[i] = (double)c * 1e6 / wall_khz / rounds;
    }
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    double th = (s[12] + s[115]) / 2;
    for (int p = 0; p < 8; p++) {
      printf("  page %d: ", p);
      int ones = 0;
      for (int k = 0; k < 16; k++) {
        int c = v[p * 16 + k] < th ? 0 : 1;
        ones += c;
        printf("%d", c);
      }
      printf("   %s\n", (ones == 0 || ones == 16) ? "uniform" : "MIXED");
    }
  }

  return 0;
}
