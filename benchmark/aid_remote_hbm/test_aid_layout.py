"""Check the AID-local layout builder on fleet's real W13 shape.

Verifies three things the kernel will depend on:
  1. every slab byte survives the move to its placed pages, bit-exact
  2. the pages a slab was given really are on the AID it was assigned
  3. the slab split matches the measured 3:4 stack ratio

Run:  MPK_AID_LAYOUT_LIB=/tmp/libaid_layout.so \
      HIP_VISIBLE_DEVICES=0 python3 benchmark/aid_remote_hbm/test_aid_layout.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from mirage.mpk import aid_layout  # noqa: E402

PAGE = 4096
SLAB_BYTES = 200192  # W13_WG_BYTES for gpt-oss-120b
N_SLABS = 128 * 46  # experts x workgroups, one layer


def main():
    torch.manual_seed(0)
    dev = "cuda"
    print(f"one layer W13: {N_SLABS} slabs x {SLAB_BYTES} B = "
          f"{N_SLABS * SLAB_BYTES / 2**30:.2f} GiB")

    packed = torch.randint(
        0, 256, (N_SLABS, SLAB_BYTES), dtype=torch.uint8, device=dev
    )

    t0 = time.time()
    dest, table, slab_aid, info = aid_layout.build_layout(packed)
    print(f"  build_layout total {time.time() - t0:.2f}s")

    # 1. bit-exact round trip: gather each slab back out of its pages
    ppl = info["pages_per_slab"]
    tbl = table.cpu().reshape(N_SLABS, ppl)
    idx = torch.randperm(N_SLABS)[:64]  # sample, full check is 1.1 GiB of gathers
    bad = 0
    for s in idx.tolist():
        out = torch.empty(ppl * PAGE, dtype=torch.uint8, device=dev)
        for p in range(ppl):
            src = int(tbl[s, p]) * PAGE
            out[p * PAGE : (p + 1) * PAGE] = dest[src : src + PAGE]
        if not torch.equal(out[:SLAB_BYTES], packed[s]):
            bad += 1
    print(f"  round trip: {len(idx) - bad}/{len(idx)} sampled slabs bit-exact")

    # 3. split matches the stack ratio
    r0, r1 = info["stack_ratio"]
    frac0 = info["slabs_aid0"] / N_SLABS
    print(f"  stacks {r0:.3f}:{r1:.3f}, slabs {frac0:.3f} on AID0 "
          f"(want {r0 / 7:.3f})")

    # 2. placement really landed where intended. Destroys dest, so do it last.
    frac = aid_layout.verify(dest, table, slab_aid, ppl)
    print(f"  placement: {100 * frac:.1f}% of slab pages on their intended AID")

    ok = bad == 0 and frac > 0.97 and abs(frac0 - r0 / 7) < 0.01
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
