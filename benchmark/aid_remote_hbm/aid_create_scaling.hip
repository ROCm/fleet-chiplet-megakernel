// Does hipMemCreate stay linear as the live 4 KiB handle count grows?
//
// aid_bulk_pin measures the 4 KiB create rate at a few thousand handles and
// extrapolates the full-model cost from it, but the three numbers it happened
// to collect were 42.6 us at 2K live handles, 47.3 at 4K and 107.7 at 16K.
// If that is a trend rather than noise, the 64 GB estimate built on a flat
// 47 us is meaningless and full-model pinning is far worse than it looks.
//
// Creates handles in batches, reporting the *marginal* rate of each batch
// against the live count going into it, then extrapolates the 16.7M-handle
// total both ways.
//
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 \
//     benchmark/aid_remote_hbm/aid_create_scaling.hip -o aid_create_scaling
//   HIP_VISIBLE_DEVICES=1 ./aid_create_scaling [max_handles]

#include <hip/hip_runtime.h>

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
static constexpr size_t kMoePages = (64ull << 30) / kPage;

static double now_s() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  const size_t max_h = (argc > 1) ? (size_t)std::atoll(argv[1]) : (1u << 20);
  CHECK(hipSetDevice(0));

  size_t freeb = 0, totalb = 0;
  CHECK(hipMemGetInfo(&freeb, &totalb));
  printf("free %.1f GiB / %.1f GiB; creating up to %zu 4 KiB handles "
         "(%.1f GiB)\n\n",
         freeb / 1073741824.0, totalb / 1073741824.0, max_h,
         max_h * kPage / 1073741824.0);

  hipMemAllocationProp aprop{};
  aprop.type = hipMemAllocationTypePinned;
  aprop.location.type = hipMemLocationTypeDevice;
  aprop.location.id = 0;

  std::vector<hipMemGenericAllocationHandle_t> h;
  h.reserve(max_h);

  printf("  %10s %10s %12s %14s\n", "live", "batch", "us/create",
         "-> 16.7M (s)");
  size_t batch = 4096;
  double t_all = now_s();
  while (h.size() < max_h) {
    size_t n = std::min(batch, max_h - h.size());
    size_t live = h.size();
    double t0 = now_s();
    for (size_t i = 0; i < n; i++) {
      hipMemGenericAllocationHandle_t x;
      CHECK(hipMemCreate(&x, kPage, &aprop, 0));
      h.push_back(x);
    }
    double us = (now_s() - t0) * 1e6 / n;
    printf("  %10zu %10zu %12.1f %14.0f\n", live, n, us,
           us * 1e-6 * kMoePages);
    fflush(stdout);
    if (h.size() >= 4 * batch)
      batch *= 2;
  }
  double total_s = now_s() - t_all;
  printf("\n  %zu handles in %.1f s (mean %.1f us)\n", h.size(), total_s,
         total_s * 1e6 / h.size());

  double t0 = now_s();
  for (auto x : h)
    CHECK(hipMemRelease(x));
  printf("  release: %.1f s (%.1f us/handle)\n", now_s() - t0,
         (now_s() - t0) * 1e6 / h.size());
  return 0;
}
