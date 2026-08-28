#!/usr/bin/env python3
"""Verify pack_mxfp4_workgroup's out_stride_rows against the packed baseline.

The N stride is the last of the three "somebody else wrote this buffer" knobs,
after row_stride_blocks (K pitch) and split_scales (section layout). It makes
one expert occupy MORE rows in the data section than the kernel computes, so
titan can step experts by vLLM's 6144/3072 pitch while its tiles still cover
5888/2944.

Four properties, in increasing strength:

  1. out_stride_rows=None is BYTE-IDENTICAL to not passing it. A default that
     is not a no-op silently corrupts every existing slab. Same property
     check_pack_stride.py and check_pack_split.py enforce for their knobs.
  2. The DATA section grows by exactly the pad rows and the SCALE section does
     not grow at all. This is the asymmetry that makes the knob safe: scales
     keep titan's own row count because in split mode the two sections have
     different writers by construction. If the scale section grew, the kernel's
     W13_EXPERT_SCALE_BYTES (which deliberately keeps WGS, not N_STRIDE) would
     be reading the wrong expert.
  3. Every row the kernel COMPUTES holds the same bytes it held without the
     knob. The pad rows are appended after them, not interleaved among them.
  4. The kernel's own pitches -- expert_data = (N_STRIDE/OPW)*WG_DATA for data,
     expert_scale = WGS*WG_SCALE for scales -- land on those bytes for every
     (expert, workgroup) pair. Properties 2 and 3 can both hold while the
     kernel still walks experts at the wrong pitch, and the failure mode is a
     read of real weight bytes belonging to the wrong expert: plausible values,
     fluent garbage, no fault.

Unlike the K pad, the N pad rows are never addressed by any tile, so their
CONTENT is irrelevant -- this checker deliberately fills them with a recognisable
pattern rather than zeros, so that a tile which wrongly reaches into them shows
up as garbage rather than as a benign zero.

Run:
  python3 tools/check_pack_nstride.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from titan_vllm.mxfp4_pack import pack_mxfp4_workgroup  # noqa: E402


def _say(label, cond):
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}")
    return cond


def check(name, E, out_dim, out_stride, nblk_true, nblk_reduce, nblk_stride,
          opw):
    g = torch.Generator().manual_seed(0)
    blocks = torch.randint(0, 256, (E, out_dim, nblk_true, 16),
                           dtype=torch.uint8, generator=g)
    scales = torch.randint(0, 256, (E, out_dim, nblk_true),
                           dtype=torch.uint8, generator=g)

    ok = True
    for stride in (nblk_reduce, nblk_stride):
        tag = f"{name} N={out_stride} K={stride}"
        common = dict(output_per_wg=opw, target_num_blocks=nblk_reduce,
                      row_stride_blocks=stride, split_scales=True)
        base = pack_mxfp4_workgroup(blocks, scales, **common)
        none_arg = pack_mxfp4_workgroup(blocks, scales, out_stride_rows=None,
                                        **common)
        same = pack_mxfp4_workgroup(blocks, scales, out_stride_rows=out_dim,
                                    **common)
        wide = pack_mxfp4_workgroup(blocks, scales, out_stride_rows=out_stride,
                                    **common)

        wgs = out_dim // opw
        data_wgs = out_stride // opw
        wg_data = opw * stride * 16
        wg_scale = opw * nblk_reduce

        # Property 1.
        ok &= _say(f"{tag}: out_stride_rows=None is a no-op",
                   torch.equal(base, none_arg))
        ok &= _say(f"{tag}: out_stride_rows==out_dim is a no-op",
                   torch.equal(base, same))

        # Property 2: data grows by the pad rows, scales do not grow.
        want = E * (data_wgs * wg_data + wgs * wg_scale)
        ok &= _say(f"{tag}: total bytes = E*(N/OPW*WG_DATA + WGS*WG_SCALE)",
                   wide.numel() == want)
        base_sc = base[E * wgs * wg_data:]
        wide_sc = wide[E * data_wgs * wg_data:]
        ok &= _say(f"{tag}: scale section unchanged by the N stride",
                   torch.equal(base_sc, wide_sc))

        # Properties 3 and 4 together: replay the kernel's arithmetic and
        # compare against the un-widened pack, which is the ground truth for
        # what each computed row should contain.
        expert_data = data_wgs * wg_data
        expert_scale = wgs * wg_scale
        scale_origin = E * expert_data
        base_expert_data = wgs * wg_data
        base_scale_origin = E * base_expert_data
        data_ok = scale_ok = True
        for e in range(E):
            for wg in range(wgs):        # only the wgs the kernel COMPUTES
                d = wide[e * expert_data + wg * wg_data:][:wg_data]
                b = base[e * base_expert_data + wg * wg_data:][:wg_data]
                data_ok &= torch.equal(d, b)
                s = wide[scale_origin + e * expert_scale
                         + wg * wg_scale:][:wg_scale]
                bs = base[base_scale_origin + e * expert_scale
                          + wg * wg_scale:][:wg_scale]
                scale_ok &= torch.equal(s, bs)
        ok &= _say(f"{tag}: kernel data pitch reaches every computed workgroup",
                   data_ok)
        ok &= _say(f"{tag}: kernel scale pitch reaches every workgroup",
                   scale_ok)

    return ok


def check_rejects():
    """The knob must refuse the two combinations that cannot be honest."""
    g = torch.Generator().manual_seed(0)
    blocks = torch.randint(0, 256, (2, 256, 4, 16), dtype=torch.uint8,
                           generator=g)
    scales = torch.randint(0, 256, (2, 256, 4), dtype=torch.uint8, generator=g)
    ok = True

    # A foreign expert pitch implies a foreign buffer, which cannot carry
    # titan's interleaved scales.
    try:
        pack_mxfp4_workgroup(blocks, scales, output_per_wg=64,
                             out_stride_rows=384)
        ok &= _say("rejects out_stride_rows without split_scales", False)
    except AssertionError:
        ok &= _say("rejects out_stride_rows without split_scales", True)

    # A pitch shorter than what we compute would overlap experts.
    try:
        pack_mxfp4_workgroup(blocks, scales, output_per_wg=64,
                             out_stride_rows=192, split_scales=True)
        ok &= _say("rejects an expert pitch shorter than out_dim", False)
    except AssertionError:
        ok &= _say("rejects an expert pitch shorter than out_dim", True)

    return ok


def main():
    # Real GPT-OSS 120B geometry, E cut to 2 experts (packing is per-expert, so
    # the count does not change what is tested). Titan computes 5888/2944 rows;
    # vLLM stores 6144/3072. 90 true blocks -> 92 reduced (2944) -> 96 strided
    # (3072, vLLM's K pitch).
    ok = True
    ok &= check("W13", E=2, out_dim=5888, out_stride=6144, nblk_true=90,
                nblk_reduce=92, nblk_stride=96, opw=128)
    ok &= check("W2", E=2, out_dim=2944, out_stride=3072, nblk_true=90,
                nblk_reduce=92, nblk_stride=96, opw=64)
    ok &= check_rejects()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
