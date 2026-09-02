#!/usr/bin/env python3
"""Property-check `shuffle_w13_workgroups_kmajor`, the host half of
MPK_W13_KMAJOR_RECYCLE.

Why this exists
---------------
This shuffle is a TWO-SIDED contract with the kernel, and the failure mode is
the worst kind: applying it without -DMPK_W13_KMAJOR_RECYCLE, or the reverse,
is not a build error and not a crash. Every lane still reads a valid byte of
the tile, just the wrong one, so the model emits plausible wrong logits. The
same class of bug as MPK_OPROJ_KMAJOR, which is why both halves are derived
from one YAML entry.

An end-to-end run catches a *gross* error (garbage text), but a permutation
that is subtly wrong on one axis can still read fluent. These properties are
checkable without a GPU, in a second, and they pin the transform exactly:

  1. shape preserved
  2. it is a PERMUTATION -- the multiset of bytes is unchanged, so this can
     never be a requantization
  3. the scale suffix is untouched (scales stay row-major)
  4. the data prefix actually moved (guards against a silent no-op, which
     would be an invisible half-configuration)
  5. it round-trips under the inverse permutation, which is the strongest
     statement that the axis order is what we think it is
  6. fleet's own wg_bytes-inference agrees with an explicit reduction on the
     interleaved layout -- i.e. we have not diverged from upstream where the
     two are comparable
  7. it works on a data-only section (split_scales=True), which fleet has no
     equivalent of
  8. every guard fires rather than mis-permuting

Property 5 is the one that would have caught a transposed pair of axes, and
property 2 is the one that would have caught an accidental requantization.

Usage:
    python3 tools/check_w13_kmajor.py        # exits nonzero on any failure
"""

import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fleet_megakernel_vllm.mxfp4_pack import (  # noqa: E402
    shuffle_w13_workgroups_kmajor)

# gpt-oss-120b's W13: OPW=128 (measured; see configs/gpt_oss_120b.yaml) and
# K = padded hidden = 2944 = 23 * 128.
E, WG, OPW, K = 2, 3, 128, 2944


def main():
    data_b = OPW * (K // 2)
    sc_b = OPW * (K // 32)
    torch.manual_seed(0)
    packed = torch.randint(0, 256, (E, WG, data_b + sc_b), dtype=torch.uint8)
    out = shuffle_w13_workgroups_kmajor(packed, OPW, reduction=K)

    checks = []

    checks.append(("shape preserved", out.shape == packed.shape))
    checks.append(("is a permutation, not a requantization",
                   torch.equal(out.flatten().sort().values,
                               packed.flatten().sort().values)))
    checks.append(("scale suffix untouched",
                   torch.equal(out[..., data_b:], packed[..., data_b:])))
    checks.append(("data prefix actually moved (not a silent no-op)",
                   not torch.equal(out[..., :data_b], packed[..., :data_b])))

    # Inverse: [k128, quarter, row, byte] -> [row, k128, quarter, byte],
    # per 16-row MFMA tile.
    tiles = OPW // 16
    inv = out[..., :data_b].reshape(E, WG, tiles, K // 128, 4, 16, 16)
    inv = inv.permute(0, 1, 2, 5, 3, 4, 6).reshape(E, WG, data_b)
    checks.append(("round-trips under the inverse permutation",
                   torch.equal(inv, packed[..., :data_b])))

    checks.append(("fleet's wg_bytes inference agrees on interleaved scales",
                   torch.equal(shuffle_w13_workgroups_kmajor(packed, OPW), out)))

    d_only = packed[..., :data_b].contiguous()
    checks.append(("works on a data-only section (split_scales=True)",
                   torch.equal(shuffle_w13_workgroups_kmajor(d_only, OPW, K),
                               out[..., :data_b])))

    guards = [
        ("OPW != 128", lambda: shuffle_w13_workgroups_kmajor(packed, 64, K)),
        ("K not K128-aligned",
         lambda: shuffle_w13_workgroups_kmajor(packed, OPW, K + 1)),
        ("rank-2 input",
         lambda: shuffle_w13_workgroups_kmajor(packed[0], OPW, K)),
        ("reduction too large for the record",
         lambda: shuffle_w13_workgroups_kmajor(packed, OPW, K * 4)),
    ]
    for name, fn in guards:
        try:
            fn()
            checks.append((f"guard fires: {name}", False))
        except ValueError:
            checks.append((f"guard fires: {name}", True))

    bad = 0
    for name, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")
        bad += not ok

    print(f"\n{'PASS' if not bad else f'FAIL ({bad} of {len(checks)})'}"
          f" -- shuffle_w13_workgroups_kmajor, OPW={OPW} K={K}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
