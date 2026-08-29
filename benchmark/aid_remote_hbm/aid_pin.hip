// AID-pinned allocation for the MoE expert weights, as a C ABI for demo.py.
//
// The tile map (MPK_MOE_AID_INVARIANT_MAP) fixes which AID *reads* weight
// workgroup w. This puts the bytes of workgroup w on that AID's HBM, which is
// the half that actually moves traffic. Layout is [E][n_wgs][wg_bytes], so for
// a byte at offset o the owning workgroup is (o % (n_wgs*wg_bytes)) / wg_bytes
// and the target AID is 0 for w < half_wgs, else 1.
//
// Cost, from aid_bulk_pin: hipMemCreate is ~47 us per 4 KiB handle and does not
// parallelise (a global driver lock -- 32 threads make it 9x worse), homing is
// per 4 KiB regardless of allocation size (0 of 4 2 MiB handles are on one
// AID), and hipMemMap rejects a non-zero handle offset. So one handle per page
// is mandatory and this runs at roughly 30 s per layer, once per process.
//
// Build:
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 -fPIC -shared \
//     benchmark/aid_remote_hbm/aid_pin.hip -o libaid_pin.so

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

static constexpr size_t kPage = 4096;

#define CK(x)                                                                  \
  do {                                                                         \
    hipError_t _e = (x);                                                       \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "[aid_pin] %s:%d %s -> %s\n", __FILE__, __LINE__, #x,    \
              hipGetErrorString(_e));                                          \
      return nullptr;                                                          \
    }                                                                          \
  } while (0)

__device__ inline unsigned get_xcc_id() {
  constexpr unsigned imm = ((4u - 1u) << 11) | (0u << 6) | 20u;
  return __builtin_amdgcn_s_getreg(imm);
}

// `sets` disjoint XCD pairs x `per_set` wave pairs, each walking its own stride
// of the page list. Validated at 100% agreement with the serial classifier in
// aid_bulk_pin. Every wait is bounded so a lost partner cannot wedge the GPU.
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

static double now_s() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

struct Pinned {
  void *va;
  size_t bytes;
  std::vector<hipMemGenericAllocationHandle_t> handles;
};

static std::vector<Pinned> g_live;
static int g_verbose = 1;

static bool map_one(void *va, size_t off, hipMemGenericAllocationHandle_t h,
                    int dev) {
  if (hipMemMap((char *)va + off, kPage, 0, h, 0) != hipSuccess)
    return false;
  hipMemAccessDesc acc{};
  acc.location.type = hipMemLocationTypeDevice;
  acc.location.id = dev;
  acc.flags = hipMemAccessFlagsProtReadWrite;
  return hipMemSetAccess((char *)va + off, kPage, &acc, 1) == hipSuccess;
}

extern "C" {

void aid_pin_set_verbose(int v) { g_verbose = v; }

// Copies `nbytes` from `src` into a fresh buffer whose every page is homed on
// the AID that the tile map will have read it from. `invert` swaps the two
// halves -- the control arm: same VMM pages, same everything, only the AID
// assignment reversed, so a difference cannot be an artifact of 4 KiB VMM
// pages behaving unlike hipMalloc pages.
void *aid_pin_alloc(const void *src, size_t nbytes, size_t wg_bytes, int n_wgs,
                    int half_wgs, int invert, double *out_setup_s,
                    double *out_frac_correct) {
  const int dev = 0;
  const double t_start = now_s();
  const size_t n_pages = (nbytes + kPage - 1) / kPage;
  const size_t expert_bytes = wg_bytes * (size_t)n_wgs;

  // Desired side per page, from the layout alone.
  std::vector<uint8_t> want(n_pages);
  size_t need[2] = {0, 0};
  for (size_t p = 0; p < n_pages; p++) {
    size_t off = p * kPage;
    int wg = (int)((off % expert_bytes) / wg_bytes);
    int aid = (wg < half_wgs) ? 0 : 1;
    if (invert)
      aid ^= 1;
    want[p] = (uint8_t)aid;
    need[aid]++;
  }

  hipMemAllocationProp aprop{};
  aprop.type = hipMemAllocationTypePinned;
  aprop.location.type = hipMemLocationTypeDevice;
  aprop.location.id = dev;

  hipDeviceProp_t prop;
  CK(hipGetDeviceProperties(&prop, dev));
  int wall_khz = 0;
  CK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev));

  // The natural split is near 43/57, so a pool sized to the larger need plus
  // margin is what guarantees both buckets fill.
  const size_t pool = (size_t)(n_pages * 1.40) + 1024;
  std::vector<hipMemGenericAllocationHandle_t> handles(pool);
  double t0 = now_s();
  for (size_t i = 0; i < pool; i++)
    CK(hipMemCreate(&handles[i], kPage, &aprop, 0));
  const double t_create = now_s() - t0;

  void *probe_va = nullptr;
  CK(hipMemAddressReserve(&probe_va, pool * kPage, 0, nullptr, 0));
  t0 = now_s();
  for (size_t i = 0; i < pool; i++)
    if (!map_one(probe_va, i * kPage, handles[i], dev)) {
      fprintf(stderr, "[aid_pin] probe map failed at %zu\n", i);
      return nullptr;
    }
  const double t_map1 = now_s() - t0;
  CK(hipMemsetD8((hipDeviceptr_t)probe_va, 0, pool * kPage));

  int *d_claim = nullptr;
  uint64_t *d_cyc = nullptr;
  CK(hipMalloc(&d_claim, 4096 * sizeof(int)));
  CK(hipMalloc(&d_cyc, pool * sizeof(uint64_t)));
  CK(hipMemset(d_claim, 0, 4096 * sizeof(int)));
  CK(hipMemset(d_cyc, 0, pool * sizeof(uint64_t)));

  const int rounds = 48;
  t0 = now_s();
  classify_parallel<<<prop.multiProcessorCount, 64>>>(
      (char *)probe_va, pool, rounds, 32, 2, d_claim, d_cyc,
      (uint64_t)wall_khz * 1000);
  CK(hipDeviceSynchronize());
  const double t_class = now_s() - t0;

  std::vector<uint64_t> cyc(pool);
  CK(hipMemcpy(cyc.data(), d_cyc, pool * sizeof(uint64_t),
               hipMemcpyDeviceToHost));
  CK(hipFree(d_claim));
  CK(hipFree(d_cyc));

  std::vector<double> ns(pool);
  for (size_t i = 0; i < pool; i++)
    ns[i] = (double)cyc[i] * 1e6 / wall_khz / rounds;
  std::vector<double> srt(ns);
  std::sort(srt.begin(), srt.end());
  const double thresh = (srt[srt.size() / 10] + srt[srt.size() * 9 / 10]) / 2;

  std::vector<size_t> bucket[2];
  for (size_t i = 0; i < pool; i++)
    bucket[ns[i] < thresh ? 0 : 1].push_back(i);

  if (bucket[0].size() < need[0] || bucket[1].size() < need[1]) {
    fprintf(stderr,
            "[aid_pin] pool short: have %zu/%zu side0, %zu/%zu side1\n",
            bucket[0].size(), need[0], bucket[1].size(), need[1]);
    return nullptr;
  }

  void *va = nullptr;
  CK(hipMemAddressReserve(&va, n_pages * kPage, 0, nullptr, 0));
  size_t take[2] = {0, 0};
  std::vector<hipMemGenericAllocationHandle_t> used(n_pages);
  std::vector<bool> consumed(pool, false);
  t0 = now_s();
  for (size_t p = 0; p < n_pages; p++) {
    size_t idx = bucket[want[p]][take[want[p]]++];
    consumed[idx] = true;
    if (hipMemUnmap((char *)probe_va + idx * kPage, kPage) != hipSuccess ||
        !map_one(va, p * kPage, handles[idx], dev)) {
      fprintf(stderr, "[aid_pin] placement failed at page %zu\n", p);
      return nullptr;
    }
    used[p] = handles[idx];
  }
  const double t_map2 = now_s() - t0;

  for (size_t i = 0; i < pool; i++)
    if (!consumed[i]) {
      hipMemUnmap((char *)probe_va + i * kPage, kPage);
      hipMemRelease(handles[i]);
    }
  CK(hipMemAddressFree(probe_va, pool * kPage));

  // Verify before the copy, while the buffer is still scratch: re-classify the
  // finished range and score it against the layout's intent. Reusing the probe
  // threshold would be circular, so this thresholds its own distribution.
  double frac = -1.0;
  {
    const size_t nv = std::min<size_t>(n_pages, 1u << 16);
    int *vc = nullptr;
    uint64_t *vy = nullptr;
    CK(hipMalloc(&vc, 4096 * sizeof(int)));
    CK(hipMalloc(&vy, nv * sizeof(uint64_t)));
    CK(hipMemset(vc, 0, 4096 * sizeof(int)));
    CK(hipMemset(vy, 0, nv * sizeof(uint64_t)));
    CK(hipMemsetD8((hipDeviceptr_t)va, 0, nv * kPage));
    classify_parallel<<<prop.multiProcessorCount, 64>>>(
        (char *)va, nv, rounds, 32, 2, vc, vy, (uint64_t)wall_khz * 1000);
    CK(hipDeviceSynchronize());
    std::vector<uint64_t> vcyc(nv);
    CK(hipMemcpy(vcyc.data(), vy, nv * sizeof(uint64_t),
                 hipMemcpyDeviceToHost));
    CK(hipFree(vc));
    CK(hipFree(vy));
    std::vector<double> vns(nv);
    for (size_t i = 0; i < nv; i++)
      vns[i] = (double)vcyc[i] * 1e6 / wall_khz / rounds;
    std::vector<double> vs(vns);
    std::sort(vs.begin(), vs.end());
    const double vth = (vs[vs.size() / 10] + vs[vs.size() * 9 / 10]) / 2;
    size_t hit = 0;
    for (size_t i = 0; i < nv; i++)
      if ((vns[i] < vth ? 0 : 1) == (int)want[i])
        hit++;
    frac = (double)hit / nv;
  }

  CK(hipMemcpy(va, src, nbytes, hipMemcpyDeviceToDevice));
  CK(hipDeviceSynchronize());

  const double setup = now_s() - t_start;
  if (out_setup_s)
    *out_setup_s = setup;
  if (out_frac_correct)
    *out_frac_correct = frac;

  if (g_verbose)
    fprintf(stderr,
            "[aid_pin] %.0f MiB: %zu pages (%zu/%zu sides), pool %zu, "
            "create %.1fs map %.1fs+%.1fs classify %.2fs total %.1fs  "
            "verified %.1f%%%s\n",
            nbytes / 1048576.0, n_pages, need[0], need[1], pool, t_create,
            t_map1, t_map2, t_class, setup, 100.0 * frac,
            invert ? "  [INVERTED]" : "");

  g_live.push_back({va, n_pages * kPage, std::move(used)});
  return va;
}

void aid_pin_free_all() {
  for (auto &b : g_live) {
    for (size_t p = 0; p < b.bytes / kPage; p++)
      hipMemUnmap((char *)b.va + p * kPage, kPage);
    for (auto h : b.handles)
      hipMemRelease(h);
    hipMemAddressFree(b.va, b.bytes);
  }
  g_live.clear();
}

} // extern "C"
