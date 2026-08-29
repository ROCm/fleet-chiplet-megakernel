#!/usr/bin/env python3
"""Verify pack_mxfp4_workgroup's `section` argument against the full packed buffer.

`section` is what makes aliasing possible in practice. The three stride/layout
knobs (row_stride_blocks, split_scales, out_stride_rows) let fleet_mk DESCRIBE a
buffer vLLM wrote; `section` lets fleet_mk avoid REBUILDING it. Without it, packing
a scale slab for aliased data would still materialize the 60 GiB data section
that is being aliased -- which is the entire cost this change exists to remove.

Five properties:

  1. section="both" is BYTE-IDENTICAL to not passing it. Same default-is-a-no-op
     property the other three knobs are held to.
  2. section="data" is exactly the leading data bytes of "both", and
     section="scales" is exactly the trailing scale bytes. Concatenated they
     reproduce "both" whole -- so the sections are a partition, not a
     recomputation that happens to look similar.
  3. **section="scales" does not read `blocks` at all.** Checked by packing twice
     with completely different block CONTENT and requiring identical output. This
     is the property the memory saving rests on: if any data byte reached the
     output, the caller could not hand in a foreign tensor and skip the copy.
  4. The kernel's scale arithmetic -- e*EXPERT_SCALE + wg*WG_SCALE, with NO
     E*EXPERT_DATA origin term, because the data lives in another buffer -- lands
     on the right bytes for every (expert, workgroup) pair.
  5. Both stride knobs are inert on a scales-only pack. A foreign K pitch and a
     foreign N pitch describe the DATA section; if either moved the scale
     section, fleet_mk's scales and vLLM's data would disagree about which expert
     is which.

Run:
  python3 tools/check_pack_section.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fleet_megakernel_vllm.mxfp4_pack import pack_mxfp4_workgroup  # noqa: E402


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
    # Same shape, deliberately different bytes -- property 3's control.
    other = torch.randint(0, 256, (E, out_dim, nblk_true, 16),
                          dtype=torch.uint8, generator=g)

    ok = True
    for stride, nstride in ((nblk_reduce, out_dim), (nblk_stride, out_stride)):
        tag = f"{name} N={nstride} K={stride}"
        common = dict(output_per_wg=opw, target_num_blocks=nblk_reduce,
                      row_stride_blocks=stride, out_stride_rows=nstride,
                      split_scales=True)
        both = pack_mxfp4_workgroup(blocks, scales, **common)
        both_explicit = pack_mxfp4_workgroup(blocks, scales, section="both",
                                             **common)
        data = pack_mxfp4_workgroup(blocks, scales, section="data", **common)
        sc = pack_mxfp4_workgroup(blocks, scales, section="scales", **common)
        sc_other = pack_mxfp4_workgroup(other, scales, section="scales",
                                        **common)

        wgs = out_dim // opw
        data_wgs = nstride // opw
        wg_data = opw * stride * 16
        wg_scale = opw * nblk_reduce
        split = E * data_wgs * wg_data

        # Property 1.
        ok &= _say(f"{tag}: section='both' is the default",
                   torch.equal(both, both_explicit))

        # Property 2: the two sections partition the whole buffer.
        ok &= _say(f"{tag}: section='data' == leading data bytes",
                   data.numel() == split and torch.equal(data, both[:split]))
        ok &= _say(f"{tag}: section='scales' == trailing scale bytes",
                   sc.numel() == E * wgs * wg_scale
                   and torch.equal(sc, both[split:]))
        ok &= _say(f"{tag}: data ++ scales reproduces the whole buffer",
                   torch.equal(torch.cat([data, sc]), both))

        # Property 3: the scale section is independent of the data content.
        ok &= _say(f"{tag}: section='scales' ignores blocks entirely",
                   torch.equal(sc, sc_other))

        # Property 4: kernel arithmetic, with the data origin term absent.
        scale_ok = True
        for e in range(E):
            for wg in range(wgs):
                s = sc[e * wgs * wg_scale + wg * wg_scale:][:wg_scale]
                bs = both[split + e * wgs * wg_scale
                          + wg * wg_scale:][:wg_scale]
                scale_ok &= torch.equal(s, bs)
        ok &= _say(f"{tag}: scale pitch reaches every workgroup from its own base",
                   scale_ok)

    # Property 5: neither stride knob perturbs a scales-only pack.
    plain = pack_mxfp4_workgroup(
        blocks, scales, output_per_wg=opw, target_num_blocks=nblk_reduce,
        split_scales=True, section="scales")
    strided = pack_mxfp4_workgroup(
        blocks, scales, output_per_wg=opw, target_num_blocks=nblk_reduce,
        row_stride_blocks=nblk_stride, out_stride_rows=out_stride,
        split_scales=True, section="scales")
    ok &= _say(f"{name}: K and N strides are inert on a scales-only pack",
               torch.equal(plain, strided))
    return ok


def check_rejects():
    """A section only exists once the interleave is gone."""
    g = torch.Generator().manual_seed(0)
    blocks = torch.randint(0, 256, (2, 256, 4, 16), dtype=torch.uint8,
                           generator=g)
    scales = torch.randint(0, 256, (2, 256, 4), dtype=torch.uint8, generator=g)
    ok = True

    for bad in ("data", "scales"):
        try:
            pack_mxfp4_workgroup(blocks, scales, output_per_wg=64, section=bad)
            ok &= _say(f"rejects section={bad!r} without split_scales", False)
        except AssertionError:
            ok &= _say(f"rejects section={bad!r} without split_scales", True)

    try:
        pack_mxfp4_workgroup(blocks, scales, output_per_wg=64,
                             split_scales=True, section="scale")
        ok &= _say("rejects a misspelled section name", False)
    except AssertionError:
        ok &= _say("rejects a misspelled section name", True)

    return ok


def main():
    # Real GPT-OSS 120B geometry, E cut to 2 experts. Fleet MK computes 5888/2944
    # rows and reduces 2944 (92 blocks); vLLM stores 6144/3072 rows at a 3072
    # (96-block) K pitch.
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
