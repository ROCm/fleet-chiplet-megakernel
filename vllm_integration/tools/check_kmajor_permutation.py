#!/usr/bin/env python3
"""Assert the K-major O-proj shuffle is a pure permutation of the packed tile.

`shuffle_oproj_workgroups_kmajor` claims to be bit-exact: it reorders which
byte sits at which address, and changes no byte's value. That claim is worth a
gate rather than a comment, because the failure mode if it is wrong is not a
crash -- every lane still reads a valid byte, just the wrong one, and the model
emits plausible wrong logits. See the MPK_OPROJ_KMAJOR block in
gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh.

Three properties, all checkable on the CPU with no model and no GPU:

  1. the data prefix is a permutation -- same byte multiset, per workgroup
     (per-workgroup, not global: a global multiset match would also accept a
     shuffle that leaked bytes between workgroups)
  2. the scale suffix is untouched, byte for byte. The kernel reads scales as
     scalars off a row-major base in both layouts, so a shuffle that also moved
     them would be wrong even though the data prefix looked right.
  3. the permutation inverts -- applying the documented inverse recovers the
     input exactly, which is what makes "the same bytes reach the same lane"
     true rather than merely plausible.

Usage:
    ./check_kmajor_permutation.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from fleet_megakernel_vllm.mxfp4_pack import (  # noqa: E402
    shuffle_oproj_workgroups_kmajor)

OPW = 16
# GPT-OSS O-proj: K = num_q_heads * head_dim = 64 * 64.
REDUCTION = 4096
N_WGS = 2880 // OPW  # hidden_size rows / rows per workgroup
DATA_BYTES = OPW * (REDUCTION // 2)
WG_BYTES = DATA_BYTES + OPW * (REDUCTION // 32)


def unshuffle(out, output_per_wg=OPW):
    """The documented inverse: [k128, quarter, row] -> [row, k128, quarter]."""
    n_wgs, wg_bytes = out.shape
    reduction = (wg_bytes * 32) // (output_per_wg * 17)
    data_bytes = output_per_wg * (reduction // 2)
    k128 = reduction // 128
    data = out[:, :data_bytes].reshape(n_wgs, k128, 4, output_per_wg, 16)
    data = data.permute(0, 3, 1, 2, 4).reshape(n_wgs, data_bytes)
    return torch.cat([data, out[:, data_bytes:]], dim=1).contiguous()


def main():
    torch.manual_seed(0)
    src = torch.randint(0, 256, (N_WGS, WG_BYTES), dtype=torch.uint8)
    out = shuffle_oproj_workgroups_kmajor(src, OPW)

    ok = True

    if out.shape != src.shape:
        print("FAIL shape: %s -> %s" % (tuple(src.shape), tuple(out.shape)))
        ok = False

    # 1. per-workgroup byte multiset of the data prefix
    a = src[:, :DATA_BYTES].sort(dim=1).values
    b = out[:, :DATA_BYTES].sort(dim=1).values
    if not torch.equal(a, b):
        bad = (a != b).any(dim=1).nonzero().flatten().tolist()
        print("FAIL data prefix is not a per-workgroup permutation "
              "(%d workgroups differ, first: %s)" % (len(bad), bad[:8]))
        ok = False
    else:
        print("OK   data prefix: byte multiset preserved in all %d workgroups"
              % N_WGS)

    # 2. scale suffix untouched
    if not torch.equal(src[:, DATA_BYTES:], out[:, DATA_BYTES:]):
        print("FAIL scale suffix was modified -- it must stay row-major")
        ok = False
    else:
        print("OK   scale suffix: %d bytes/wg identical"
              % (WG_BYTES - DATA_BYTES))

    # 3. round-trip
    if not torch.equal(unshuffle(out), src):
        print("FAIL inverse permutation does not recover the input")
        ok = False
    else:
        print("OK   round-trip: unshuffle(shuffle(x)) == x")

    # the shuffle must actually do something -- a no-op would pass 1-3
    if torch.equal(src, out):
        print("FAIL shuffle is a no-op; the layout did not change")
        ok = False
    else:
        moved = (src[:, :DATA_BYTES] != out[:, :DATA_BYTES]).float().mean()
        print("OK   layout changed: %.1f%% of data bytes moved address"
              % (100.0 * moved))

    # rank-3 [1, n_wgs, wg_bytes] is what pack_mxfp4_workgroup returns
    r3 = shuffle_oproj_workgroups_kmajor(src.unsqueeze(0), OPW)
    if r3.ndim != 3 or not torch.equal(r3.squeeze(0), out):
        print("FAIL rank-3 input did not give the same tile back at rank 3")
        ok = False
    else:
        print("OK   rank-3 [1, n_wgs, wg_bytes] agrees with rank-2")

    print("\n%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
