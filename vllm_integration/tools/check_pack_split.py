#!/usr/bin/env python3
"""Verify pack_mxfp4_workgroup's split_scales against the interleaved baseline.

Split mode moves the scale bytes out of the per-workgroup interleave and into
one contiguous section after all the data. Three properties, and only the first
is obvious:

  1. split_scales=False (the default) is BYTE-IDENTICAL to the baseline packer.
     A default that is not a no-op is a silent corruption of every existing slab.
     This is the same property check_pack_stride.py enforces for the stride knob.
  2. Split mode is a PERMUTATION, not a re-encode: the data section holds
     exactly the interleaved mode's data bytes, in the same order, and the scale
     section holds exactly its scale bytes, in the same order. Nothing is
     re-quantized, re-padded, or re-ordered within a section. If this holds, any
     output difference in a split-mode run is an addressing bug in the kernel,
     not a packing difference.
  3. The section pitches the kernel computes actually land on those bytes. This
     replays the kernel's own address arithmetic --
       data:  base + e*(WGS*WG_DATA) + wg*WG_DATA
       scale: base + E*(WGS*WG_DATA) + e*(WGS*WG_SCALE) + wg*WG_SCALE
     -- against the packed tensor. Property 2 can hold while the kernel still
     reads the wrong rows, because the two sections have DIFFERENT per-workgroup
     pitches in split mode and the same pitch in interleaved mode; conflating
     them reads real scale bytes for the wrong output rows, which produces
     plausible values and fluent garbage rather than a fault.

The knob is checked with and without a wider row stride, since split scales and
strided data rows are independent and will be combined when fleet_mk aliases
vLLM's allocation.

Run:
  python3 tools/check_pack_split.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fleet_megakernel_vllm.mxfp4_pack import pack_mxfp4_workgroup  # noqa: E402


def check(name, E, out_dim, nblk_true, nblk_reduce, nblk_stride, opw):
    g = torch.Generator().manual_seed(0)
    blocks = torch.randint(0, 256, (E, out_dim, nblk_true, 16),
                           dtype=torch.uint8, generator=g)
    scales = torch.randint(0, 256, (E, out_dim, nblk_true),
                           dtype=torch.uint8, generator=g)

    ok = True
    for stride in (nblk_reduce, nblk_stride):
        tag = f"{name} stride={stride}"
        inter = pack_mxfp4_workgroup(blocks, scales, output_per_wg=opw,
                                     target_num_blocks=nblk_reduce,
                                     row_stride_blocks=stride)
        default_off = pack_mxfp4_workgroup(blocks, scales, output_per_wg=opw,
                                           target_num_blocks=nblk_reduce,
                                           row_stride_blocks=stride,
                                           split_scales=False)
        split = pack_mxfp4_workgroup(blocks, scales, output_per_wg=opw,
                                     target_num_blocks=nblk_reduce,
                                     row_stride_blocks=stride,
                                     split_scales=True)

        wgs = out_dim // opw
        wg_data = opw * stride * 16
        wg_scale = opw * nblk_reduce

        ok &= _say(f"{tag}: split_scales=False is a no-op",
                   torch.equal(inter, default_off))
        ok &= _say(f"{tag}: split total bytes match interleaved",
                   split.numel() == inter.numel())
        ok &= _say(f"{tag}: split is flat 1-D",
                   split.ndim == 1)

        # Property 2: same bytes, section by section.
        i_data = inter[:, :, :wg_data].reshape(-1)
        i_sc = inter[:, :, wg_data:].reshape(-1)
        s_data = split[:E * wgs * wg_data]
        s_sc = split[E * wgs * wg_data:]
        ok &= _say(f"{tag}: split data section == interleaved data bytes",
                   torch.equal(s_data, i_data))
        ok &= _say(f"{tag}: split scale section == interleaved scale bytes",
                   torch.equal(s_sc, i_sc))

        # Property 3: the kernel's pitches reach the right workgroup. Checking
        # every (expert, workgroup) pair is cheap at this size and catches an
        # off-by-one that a spot check at wg=0 cannot -- wg=0 has offset 0 under
        # both a correct and a wrong pitch.
        expert_data = wgs * wg_data
        expert_scale = wgs * wg_scale
        scale_origin = E * expert_data
        data_ok = scale_ok = True
        for e in range(E):
            for wg in range(wgs):
                d = split[e * expert_data + wg * wg_data:][:wg_data]
                s = split[scale_origin + e * expert_scale + wg * wg_scale:][:wg_scale]
                data_ok &= torch.equal(d, inter[e, wg, :wg_data])
                scale_ok &= torch.equal(s, inter[e, wg, wg_data:])
        ok &= _say(f"{tag}: kernel data pitch reaches every workgroup", data_ok)
        ok &= _say(f"{tag}: kernel scale pitch reaches every workgroup", scale_ok)

    return ok


def _say(label, cond):
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}")
    return cond


def main():
    # Real GPT-OSS 120B geometry, E cut to 2 experts (packing is per-expert, so
    # the count does not change what is tested). 90 true blocks -> 92 reduced
    # (2944) -> 96 strided (3072, vLLM's pitch).
    ok = True
    ok &= check("W13", E=2, out_dim=5888, nblk_true=90, nblk_reduce=92,
                nblk_stride=96, opw=128)
    ok &= check("W2", E=2, out_dim=2944, nblk_true=90, nblk_reduce=92,
                nblk_stride=96, opw=64)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
