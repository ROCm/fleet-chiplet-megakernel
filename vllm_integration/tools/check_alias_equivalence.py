#!/usr/bin/env python3
"""Prove fleet_mk's packed DATA section equals vLLM's plain expert tensor, byte for byte.

This is the load-bearing claim of the aliasing step. Everything else -- the three
stride knobs, the section split, the separate allocations -- exists to make
fleet_mk's addressing match a buffer someone else wrote. None of it matters unless
the bytes fleet_mk WOULD have packed are the bytes vLLM ALREADY has, at the same
offsets. If they are, the pack is redundant and the pointer can simply be
redirected; if they are not, redirecting it reads plausible garbage silently.

The claim, precisely: with

    row_stride_blocks = K_STRIDE/32   (vLLM's row pitch, 96 blocks = 3072 values)
    out_stride_rows   = N_STRIDE      (vLLM's expert pitch, 6144 / 3072 rows)
    split_scales=True, section="data"

`pack_mxfp4_workgroup` emits a buffer of exactly vLLM's size that agrees with
`vllm_tensor.reshape(-1)` at every byte the kernel reads.

It is NOT byte-identical, and this checker deliberately does not claim that: the
N pad rows differ, because fleet_mk zero-fills them and vLLM has whatever it has.
That is the K-pad/N-pad asymmetry -- K pad columns are summed by the MFMA and so
must be zero, while N pad rows are addressed by no tile at all. So the checker
proves the stronger-in-practice pair: the sizes match exactly, and every
DIFFERING byte lies inside an unaddressed pad row.

Why that is even plausible, and where it could fail:

  * A workgroup's data block is OPW consecutive rows at the full stored pitch.
    Laid out expert-major then workgroup-major, that is just rows in order --
    the SAME order a row-major [E, rows, bytes] tensor already has. The
    workgroup grouping is a reinterpretation, not a permutation.
  * It fails the moment any widening is not a pure tail-append: if the K pad
    were inserted anywhere but the end of each row, or the N pad anywhere but
    after each expert's rows, the orders diverge.
  * It also fails if fleet_mk's computed extent runs past vLLM's allocation, which
    is checked here explicitly -- the buffer-rsrc extent bounds-checks every
    buffer_load, so an over-long extent turns an out-of-range read into a
    wrong-address read rather than a zero.

The vLLM side is SYNTHESIZED here rather than loaded: a [E, N_STRIDE, K_STRIDE/2]
row-major uint8 tensor is exactly what vLLM holds (probe_vllm_expert_layout.py
confirmed the shapes and that the K pad is zero), and using a synthetic one lets
this run in a second with no model and no GPU. The zero-pad property is asserted
against the real tensors at pack time, not here.

Run:
  python3 tools/check_alias_equivalence.py
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
    # vLLM's buffer: row-major [E, stored_rows, stored_bytes], pad included.
    g = torch.Generator().manual_seed(0)
    foreign = torch.randint(0, 256, (E, out_stride, nblk_stride * 16),
                            dtype=torch.uint8, generator=g)

    # The K pad must be zero -- those columns ARE summed by the MFMA. The N pad
    # is deliberately left as random bytes: no tile addresses those rows, so if
    # the equivalence still holds with garbage there, it holds for anything.
    foreign[:, :, nblk_reduce * 16:] = 0

    # What fleet_mk would pack, given the same logical weights. Recover them from
    # the foreign buffer so the two sides cannot disagree about content.
    blocks = foreign.view(E, out_stride, nblk_stride, 16)[:, :out_dim,
                                                          :nblk_reduce]
    scales = torch.randint(0, 256, (E, out_dim, nblk_reduce), dtype=torch.uint8,
                           generator=g)

    packed = pack_mxfp4_workgroup(
        blocks.contiguous(), scales, output_per_wg=opw,
        target_num_blocks=nblk_reduce, row_stride_blocks=nblk_stride,
        out_stride_rows=out_stride, split_scales=True, section="data")

    ok = True
    flat = foreign.reshape(-1)
    ok &= _say(f"{name}: packed data section has vLLM's exact byte count",
               packed.numel() == flat.numel())

    # The two buffers are NOT byte-identical, and must not be claimed to be:
    # fleet_mk writes zeros into the N pad rows, vLLM writes whatever it wrote.
    # The honest claim is that they agree everywhere the kernel READS, and
    # differ only in rows no tile addresses. Both halves are checked, because
    # "they differ somewhere" is only benign if the somewhere is exactly the pad.
    diff = (packed != flat).nonzero().flatten()
    pad_lo = out_dim * nblk_stride * 16          # first pad byte within expert 0
    expert_span = out_stride * nblk_stride * 16
    in_pad = ((diff % expert_span) >= pad_lo)
    ok &= _say(f"{name}: every differing byte lies in an unaddressed pad row",
               bool(in_pad.all()))
    n_pad_bytes = E * (out_stride - out_dim) * nblk_stride * 16
    ok &= _say(f"{name}: the addressed bytes are identical "
               f"({flat.numel() - n_pad_bytes} of {flat.numel()})",
               diff.numel() <= n_pad_bytes)

    # The extent the kernel will hand to make_w_buffer_rsrc must not exceed the
    # allocation it is describing.
    expert_bytes = (out_stride // opw) * opw * nblk_stride * 16
    ok &= _say(f"{name}: kernel expert extent fits vLLM's allocation",
               E * expert_bytes == foreign.numel())

    # Every workgroup the kernel addresses must land on the right rows of the
    # foreign tensor. This is the check that would catch a permutation the
    # whole-buffer compare above cannot: it compares against the ROWS, not
    # against the flattened order.
    wg_data = opw * nblk_stride * 16
    rows_ok = True
    for e in range(E):
        for wg in range(out_dim // opw):
            got = packed[e * expert_bytes + wg * wg_data:][:wg_data]
            want = foreign[e, wg * opw:(wg + 1) * opw].reshape(-1)
            rows_ok &= torch.equal(got, want)
    ok &= _say(f"{name}: every computed workgroup maps to its own rows", rows_ok)

    # Sanity: the pad rows really are being stepped over, i.e. the pitch is the
    # stored one. If fleet_mk used its computed row count as the pitch instead, the
    # last expert would start (out_stride - out_dim) * E rows too early.
    ok &= _say(f"{name}: expert pitch is the STORED row count, not the computed",
               expert_bytes == out_stride * nblk_stride * 16
               and expert_bytes > out_dim * nblk_reduce * 16)
    return ok


def check_scales_stay_local():
    """The scale section must NOT be claimed to match vLLM's.

    vLLM's TRITON backend deletes w13_weight_scale/w2_weight_scale in
    process_weights_after_loading (the swizzled scales move inside the precision
    configs), so there is nothing to alias and fleet_mk keeps packing its own. This
    checks the packer keeps the scale section at fleet_mk's OWN row count and
    reduction pitch even while the data section runs at vLLM's -- i.e. that a
    foreign data pitch cannot leak into the scale addressing.
    """
    E, out_dim, out_stride, nblk_reduce, nblk_stride, opw = 2, 2944, 3072, 92, 96, 64
    g = torch.Generator().manual_seed(1)
    blocks = torch.randint(0, 256, (E, out_dim, nblk_reduce, 16),
                           dtype=torch.uint8, generator=g)
    scales = torch.randint(0, 256, (E, out_dim, nblk_reduce), dtype=torch.uint8,
                           generator=g)
    sc = pack_mxfp4_workgroup(
        blocks, scales, output_per_wg=opw, target_num_blocks=nblk_reduce,
        row_stride_blocks=nblk_stride, out_stride_rows=out_stride,
        split_scales=True, section="scales")
    return _say("scale section keeps fleet_mk's own row count and pitch",
                sc.numel() == E * out_dim * nblk_reduce
                and torch.equal(sc, scales.reshape(-1)))


def main():
    # Real GPT-OSS 120B geometry, E cut to 2. vLLM stores 6144/3072 rows at a
    # 3072-value (96-block) K pitch; fleet_mk computes 5888/2944 and reduces 2944.
    ok = True
    ok &= check("W13", E=2, out_dim=5888, out_stride=6144, nblk_true=90,
                nblk_reduce=92, nblk_stride=96, opw=128)
    ok &= check("W2", E=2, out_dim=2944, out_stride=3072, nblk_true=90,
                nblk_reduce=92, nblk_stride=96, opw=64)
    ok &= check_scales_stay_local()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
