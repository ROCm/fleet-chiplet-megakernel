// Host-side support for AID-local MoE weights.
//
// The kernel reads each workgroup's weight slab as a contiguous 200192 B run
// (W13; 100096 for W2). MI350 NPS1 homes memory per 4 KiB page as a mod-7 hash
// of the *physical* page number, and the two AIDs own 3 and 4 of the 7 stacks,
// so a 49-page slab is always split across both AIDs -- aid_interleave found 0
// of 10699 slabs single-AID. Neither reordering readers nor any compile-time
// map fixes that (aid_period: two allocations agree only 71.4% at the same
// offset), so the data has to be placed deliberately, once, at load.
//
// This exposes the two things the host needs for that:
//
//   aid_classify_pages  label every 4 KiB page of a buffer by home AID
//   aid_verify_pages    re-measure a set of pages and report how many are on
//                       the AID they were meant to be on
//
// Classification is ~0.84 us/page, so ~14 s for 63 GiB of weights, paid once
// per model load. It cannot be cached across runs because it depends on the
// physical pages this allocation happened to get.
//
// Build as a shared object for ctypes:
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 -fPIC -shared \
//     benchmark/aid_remote_hbm/aid_layout.hip -o libaid_layout.so

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

static constexpr size_t kPage = 4096;

__device__ inline unsigned get_xcc_id() {
  constexpr unsigned imm = ((4u - 1u) << 11) | (0u << 6) | 20u;
  return __builtin_amdgcn_s_getreg(imm);
}

// Two XCDs on AID0 ping-pong a flag in each page. A page homed near AID0
// answers fast, one homed on AID1 pays the crossing on every round trip, and
// 48 rounds is enough separation to threshold cleanly.
__global__ void classify_kernel(char *base, size_t npages, int rounds,
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

// Classification writes to the pages it probes, so the caller's data must not
// be live yet. Returns 0 on success, and fills side_out with 0 (AID0) or 1.
extern "C" int aid_classify_pages(void *base, size_t nbytes,
                                  unsigned char *side_out, int *n_aid0,
                                  double *seconds) {
  const size_t npages = nbytes / kPage;
  if (!base || npages == 0)
    return -1;

  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess)
    return -2;
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess)
    return -2;
  int wall_khz = 0;
  if (hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev) !=
      hipSuccess)
    return -2;

  int *d_claim = nullptr;
  uint64_t *d_cyc = nullptr;
  if (hipMalloc(&d_claim, 4096 * sizeof(int)) != hipSuccess)
    return -3;
  if (hipMalloc(&d_cyc, npages * sizeof(uint64_t)) != hipSuccess) {
    hipFree(d_claim);
    return -3;
  }
  hipMemset(d_claim, 0, 4096 * sizeof(int));
  hipMemset(d_cyc, 0, npages * sizeof(uint64_t));
  hipMemsetD8((hipDeviceptr_t)base, 0, nbytes);

  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  hipEventRecord(e0);
  classify_kernel<<<prop.multiProcessorCount, 64>>>(
      (char *)base, npages, 48, 32, 2, d_claim, d_cyc,
      (uint64_t)wall_khz * 1000);
  hipEventRecord(e1);
  if (hipEventSynchronize(e1) != hipSuccess) {
    hipFree(d_claim);
    hipFree(d_cyc);
    return -4;
  }
  float ms = 0;
  hipEventElapsedTime(&ms, e0, e1);
  if (seconds)
    *seconds = ms / 1e3;

  std::vector<uint64_t> cyc(npages);
  hipMemcpy(cyc.data(), d_cyc, npages * sizeof(uint64_t), hipMemcpyDeviceToHost);
  hipFree(d_claim);
  hipFree(d_cyc);
  hipEventDestroy(e0);
  hipEventDestroy(e1);

  std::vector<double> ns(npages), srt(npages);
  for (size_t i = 0; i < npages; i++)
    ns[i] = (double)cyc[i] * 1e6 / wall_khz / 48;
  srt = ns;
  std::sort(srt.begin(), srt.end());
  const double th = (srt[srt.size() / 10] + srt[srt.size() * 9 / 10]) / 2;

  int c0 = 0;
  for (size_t i = 0; i < npages; i++) {
    side_out[i] = ns[i] < th ? 0 : 1;
    c0 += side_out[i] == 0;
  }
  if (n_aid0)
    *n_aid0 = c0;
  return 0;
}

// Independent check that a built layout is what it claims: re-classify and
// count how many of the listed pages really sit on `want_aid`. Destroys the
// contents of the listed pages, so run it before the weights are written.
extern "C" int aid_verify_pages(void *base, size_t nbytes, const int *pages,
                                int npages_listed, int want_aid,
                                double *frac_correct) {
  const size_t npages = nbytes / kPage;
  std::vector<unsigned char> side(npages);
  int c0 = 0;
  double secs = 0;
  int rc = aid_classify_pages(base, nbytes, side.data(), &c0, &secs);
  if (rc != 0)
    return rc;
  int ok = 0;
  for (int i = 0; i < npages_listed; i++) {
    int p = pages[i];
    if (p >= 0 && (size_t)p < npages && side[p] == (unsigned char)want_aid)
      ok++;
  }
  if (frac_correct)
    *frac_correct = npages_listed ? (double)ok / npages_listed : 0.0;
  return 0;
}
