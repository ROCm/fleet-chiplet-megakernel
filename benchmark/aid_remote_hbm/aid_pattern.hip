// The AID map of every buffer is one universal 224-byte pattern plus a shift.
//
// MI350 NPS1 homes memory per 4 KiB page as a mod-7 hash, but the interleave
// unit is 1 MB (256 pages), so the label sequence repeats every 7 MB = 1792
// pages -- measured 100.00% over 1 GiB, and every allocation's map is a pure
// rotation of the same pattern (100.00% agreement at k=768, k=1536).
//
// So the whole map of a 63 GiB weight buffer is:
//
//     AID(page p) = kAidPattern[(p + rot) mod 1792]
//
// 1792 bits of universal constant plus one integer per buffer. That replaces
// the per-page table that had to live in HBM and cost 1.67 us/layer to read:
// 224 bytes fits in LDS, and because every slab uses the same offsets, a
// workgroup can resolve its page list once at kernel start and then address
// with plain base+offset for the rest of the run.
//
// This does NOT make a contiguous slab single-AID: the longest same-AID run is
// 6 pages (24 KiB) and a W13 slab needs 49, so the data still has to be
// scattered into the AID's pages. What changed is that knowing where those
// pages are is now free.
//
// Build:
//   hipcc -O3 -std=c++17 --offload-arch=gfx950 -fPIC -shared \
//     benchmark/aid_remote_hbm/aid_pattern.hip -o libaid_pattern.so

#include "aid_layout.hip"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr size_t kAidPeriod = 1792;  // 7 MiB / 4 KiB

// Learn the universal pattern from a scratch buffer. Needs >= 2 periods so the
// period claim can be checked rather than assumed.
extern "C" int aid_learn_pattern(void *scratch, size_t nbytes,
                                 unsigned char *pattern_out,
                                 double *period_match_out) {
  const size_t npages = nbytes / 4096;
  if (!scratch || !pattern_out || npages < 2 * kAidPeriod)
    return -1;

  std::vector<unsigned char> side(npages);
  int c0 = 0;
  double secs = 0;
  int rc = aid_classify_pages(scratch, nbytes, side.data(), &c0, &secs);
  if (rc != 0)
    return rc;

  size_t ok = 0, tot = 0;
  for (size_t p = 0; p + kAidPeriod < npages; p++) {
    ok += (side[p] == side[p + kAidPeriod]);
    tot++;
  }
  if (period_match_out)
    *period_match_out = tot ? (double)ok / tot : 0.0;

  memcpy(pattern_out, side.data(), kAidPeriod);
  return 0;
}

// Find a buffer's rotation by classifying only `probe_pages` pages and
// correlating against the known pattern. The pattern is effectively random, so
// a couple hundred pages pin the shift uniquely; this is milliseconds, not the
// 14 s a full classification of the weights would take.
extern "C" int aid_find_rotation(void *buf, size_t probe_pages,
                                 const unsigned char *pattern, int *rot_out,
                                 double *confidence_out) {
  if (!buf || !pattern || probe_pages < 64)
    return -1;
  const size_t nbytes = probe_pages * 4096;
  std::vector<unsigned char> side(probe_pages);
  int c0 = 0;
  double secs = 0;
  int rc = aid_classify_pages(buf, nbytes, side.data(), &c0, &secs);
  if (rc != 0)
    return rc;

  size_t best_k = 0, best_hits = 0, second_hits = 0;
  for (size_t k = 0; k < kAidPeriod; k++) {
    size_t hits = 0;
    for (size_t p = 0; p < probe_pages; p++)
      hits += (side[p] == pattern[(p + k) % kAidPeriod]);
    if (hits > best_hits) {
      second_hits = best_hits;
      best_hits = hits;
      best_k = k;
    } else if (hits > second_hits) {
      second_hits = hits;
    }
  }
  if (rot_out)
    *rot_out = (int)best_k;
  // Margin over the runner-up: near 1.0 for the true shift, ~0.5 otherwise.
  if (confidence_out)
    *confidence_out = (double)best_hits / probe_pages;
  return (best_hits == probe_pages) ? 0 : 1;
}

// Drop-in replacement for aid_classify_pages that reads only the first two
// periods and tiles the result over the rest of the buffer.
//
// Probing 2 periods rather than 1 is what makes this checkable instead of
// assumed: the second period is compared against the first, and a buffer whose
// map is not periodic reports a low match and can be rejected by the caller
// rather than silently mislabelled. 14 MiB of probing replaces 63 GiB, so a
// classification that cost ~14 s costs a few ms, and build_layout retries the
// sizing loop up to four times.
extern "C" int aid_classify_fast(void *base, size_t nbytes,
                                 unsigned char *side_out, int *n_aid0,
                                 double *seconds, double *period_match_out) {
  const size_t npages = nbytes / 4096;
  if (!base || !side_out || npages < 2 * kAidPeriod)
    return -1;

  const size_t probe = 2 * kAidPeriod;
  std::vector<unsigned char> head(probe);
  int c0 = 0;
  double secs = 0;
  int rc = aid_classify_pages(base, probe * 4096, head.data(), &c0, &secs);
  if (rc != 0)
    return rc;

  size_t ok = 0;
  for (size_t p = 0; p < kAidPeriod; p++)
    ok += (head[p] == head[p + kAidPeriod]);
  const double pm = (double)ok / kAidPeriod;
  if (period_match_out)
    *period_match_out = pm;
  if (pm < 0.999)
    return 2;  // not periodic: caller should fall back to a full classify

  int total0 = 0;
  for (size_t p = 0; p < npages; p++) {
    unsigned char v = head[p % kAidPeriod];
    side_out[p] = v;
    total0 += (v == 0);
  }
  if (n_aid0)
    *n_aid0 = total0;
  if (seconds)
    *seconds = secs;
  return 0;
}

// Page -> AID, once the rotation is known. Pure arithmetic; no memory traffic
// beyond the 224-byte pattern, which is small enough to sit in LDS.
extern "C" int aid_of_page(const unsigned char *pattern, int rot, size_t page) {
  return pattern[(page + (size_t)rot) % kAidPeriod];
}

// The page indices within one period that belong to `want_aid`, in order.
// 768 entries for AID0, 1024 for AID1. This is the table a workgroup resolves
// once: its k-th 4 KiB chunk lives at
//     page = kAidPeriod * (k / count) + idx[k % count]
extern "C" int aid_period_pages(const unsigned char *pattern, int rot,
                                int want_aid, uint16_t *idx_out, int *count_out) {
  if (!pattern || !idx_out)
    return -1;
  int n = 0;
  for (size_t p = 0; p < kAidPeriod; p++)
    if (pattern[(p + (size_t)rot) % kAidPeriod] == (unsigned char)want_aid)
      idx_out[n++] = (uint16_t)p;
  if (count_out)
    *count_out = n;
  return 0;
}
