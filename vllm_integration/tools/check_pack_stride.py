#!/usr/bin/env python3
"""Verify pack_mxfp4_workgroup's row_stride_blocks against the packed baseline.

The stride path must satisfy two properties, and only the first is obvious:

  1. With row_stride_blocks == target_num_blocks (or None), the output is
     BYTE-IDENTICAL to the baseline packer. A default that is not a no-op is a
     silent corruption of every existing slab.
  2. With a wider stride, slicing the appended pad back off each row recovers
     the baseline data bytes exactly, and the scale section is untouched. This
     is the property the kernel relies on: it strides by K_STRIDE/2 to find a
     row but reduces only K_REDUCE columns, so the extra bytes must sit at the
     END of a row and must be zero.

Run:
  python3 tools/check_pack_stride.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from titan_vllm.mxfp4_pack import pack_mxfp4_workgroup  # noqa: E402


def check(name, E, out_dim, nblk_true, nblk_reduce, nblk_stride, opw):
    g = torch.Generator().manual_seed(0)
    blocks = torch.randint(0, 256, (E, out_dim, nblk_true, 16),
                           dtype=torch.uint8, generator=g)
    scales = torch.randint(0, 256, (E, out_dim, nblk_true),
                           dtype=torch.uint8, generator=g)

    base = pack_mxfp4_workgroup(blocks, scales, output_per_wg=opw,
                                target_num_blocks=nblk_reduce)
    same = pack_mxfp4_workgroup(blocks, scales, output_per_wg=opw,
                                target_num_blocks=nblk_reduce,
                                row_stride_blocks=nblk_reduce)
    wide = pack_mxfp4_workgroup(blocks, scales, output_per_wg=opw,
                                target_num_blocks=nblk_reduce,
                                row_stride_blocks=nblk_stride)

    wgs = out_dim // opw
    exp_base = (E, wgs, opw * nblk_reduce * 16 + opw * nblk_reduce)
    exp_wide = (E, wgs, opw * nblk_stride * 16 + opw * nblk_reduce)

    ok = True
    ok &= _say(f"{name}: baseline shape {tuple(base.shape)}",
               tuple(base.shape) == exp_base)
    ok &= _say(f"{name}: explicit stride == reduce is a no-op",
               torch.equal(base, same))
    ok &= _say(f"{name}: wide shape {tuple(wide.shape)}",
               tuple(wide.shape) == exp_wide)

    # Split both into their data / scale halves and compare row by row.
    b_data = base[:, :, :opw * nblk_reduce * 16].reshape(
        E, wgs, opw, nblk_reduce * 16)
    b_sc = base[:, :, opw * nblk_reduce * 16:]
    w_data = wide[:, :, :opw * nblk_stride * 16].reshape(
        E, wgs, opw, nblk_stride * 16)
    w_sc = wide[:, :, opw * nblk_stride * 16:]

    ok &= _say(f"{name}: wide row[:reduce] recovers baseline data",
               torch.equal(w_data[:, :, :, :nblk_reduce * 16], b_data))
    ok &= _say(f"{name}: wide row pad bytes are zero",
               not w_data[:, :, :, nblk_reduce * 16:].any())
    ok &= _say(f"{name}: scale section unchanged by stride",
               torch.equal(b_sc, w_sc))
    return ok


def _say(label, cond):
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}")
    return cond


def main():
    # Real GPT-OSS 120B geometry: hidden 2880 true -> 2944 reduced (92 blocks)
    # -> 3072 strided (96 blocks), which is what vLLM's allocation looks like.
    # E is cut to 2 experts; the packing is per-expert so the count does not
    # change what is being tested.
    ok = True
    ok &= check("W13", E=2, out_dim=5888, nblk_true=90, nblk_reduce=92,
                nblk_stride=96, opw=128)
    ok &= check("W2", E=2, out_dim=2944, nblk_true=90, nblk_reduce=92,
                nblk_stride=96, opw=64)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
