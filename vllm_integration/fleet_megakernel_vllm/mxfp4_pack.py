"""MXFP4 quantize/pad/pack primitives -- fleet_mk's own copy.

Vendored verbatim from mirage's `demo/gpt_oss/demo.py` (functions `pad_weight_1d`,
`pad_weight_2d`, `quantize_bf16_to_mxfp4`, `pack_mxfp4_workgroup`) so fleet_megakernel_vllm
has no Python import dependency on the mirage checkout. Previously these were
pulled at runtime through `_load_gptoss_demo_module`, which executed mirage's demo
module -- dragging in its model code and its `sys.path` insert -- purely to reach
four pure-tensor helpers.

**These are a byte-exact copy, not a reimplementation.** The packed layout they
produce is what the megakernel's address arithmetic assumes, so a "cleanup" here
that changes a threshold, a rounding rule, or the data/scale concat order does not
fail the build -- it produces silently wrong logits. If a change is ever needed,
re-verify with `tools/hash_packed.py`, which hashes every packed slot.

Layout note for `pack_mxfp4_workgroup`: per workgroup the slab holds
`OPW x K_half` data bytes immediately followed by `OPW x num_blocks` scale bytes,
so `wg_bytes = OPW*K_half + OPW*num_blocks`. That interleaving is why fleet_mk cannot
simply point at vLLM's two separate plain tensors -- see the wiki article on
single-copy weight sourcing. `split_scales=True` removes the interleave (all data,
then all scales) and is the step that makes aliasing possible; it must be matched
by `-DMPK_MOE_SPLIT_SCALES=1` in the MoE kernel, and nothing checks that it is.
"""

import torch


def pad_weight_1d(w: torch.Tensor, target_size: int, pad_value: float = 0.0):
    """Pad a 1D weight vector to target size."""
    if w.shape[0] == target_size:
        return w
    padded = torch.full((target_size,), pad_value, dtype=w.dtype, device=w.device)
    padded[:w.shape[0]] = w
    return padded


def pad_weight_2d(w: torch.Tensor, target_rows: int = None, target_cols: int = None):
    """Pad a 2D weight matrix to target dimensions with zeros."""
    rows, cols = w.shape
    if target_rows is None:
        target_rows = rows
    if target_cols is None:
        target_cols = cols
    if rows == target_rows and cols == target_cols:
        return w
    padded = torch.zeros(target_rows, target_cols, dtype=w.dtype, device=w.device)
    padded[:rows, :cols] = w
    return padded


def pad_weight_3d(w: torch.Tensor, target_dim1: int = None, target_dim2: int = None):
    """Pad a 3D weight matrix [E, D1, D2] to target dimensions with zeros."""
    E, D1, D2 = w.shape
    if target_dim1 is None:
        target_dim1 = D1
    if target_dim2 is None:
        target_dim2 = D2
    if D1 == target_dim1 and D2 == target_dim2:
        return w
    padded = torch.zeros(E, target_dim1, target_dim2, dtype=w.dtype, device=w.device)
    padded[:, :D1, :D2] = w
    return padded


# FP4 E2M1 magnitude lookup for quantization (index = nibble value 0-7)
_FP4_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def quantize_bf16_to_mxfp4(weight: torch.Tensor,
                           target_out_dim: int = None,
                           target_reduction: int = None) -> tuple:
    """Quantize BF16 weight [out_dim, in_dim] to MXFP4 blocks+scales.

    Uses MXFP4 format: FP4 E2M1 values with shared E8M0 block scales (32 elements/block).

    Args:
        weight: BF16 tensor [out_dim, in_dim]
        target_out_dim: pad output dim to this value
        target_reduction: pad reduction dim to this value (must be multiple of 32)

    Returns:
        blocks: uint8 [1, out_dim, num_blocks, 16] - packed FP4 nibbles
        scales: uint8 [1, out_dim, num_blocks] - E8M0 block scales
    """
    assert weight.ndim == 2
    out_dim, in_dim = weight.shape

    # Pad if needed
    if target_out_dim is not None and out_dim < target_out_dim:
        weight = torch.nn.functional.pad(weight, (0, 0, 0, target_out_dim - out_dim))
        out_dim = target_out_dim
    if target_reduction is not None and in_dim < target_reduction:
        weight = torch.nn.functional.pad(weight, (0, target_reduction - in_dim))
        in_dim = target_reduction

    assert in_dim % 32 == 0, f"in_dim {in_dim} must be multiple of 32"
    num_blocks = in_dim // 32

    # Work in float32
    w = weight.float().reshape(out_dim, num_blocks, 32)

    # Per-block max absolute value
    max_abs = w.abs().amax(dim=2)  # [out_dim, num_blocks]

    # Compute E8M0 scale exponent: scale = 2^(e-127)
    # Want: max_abs <= 6.0 * 2^(e-127), so e >= log2(max_abs/6) + 127
    safe_max = max_abs.clamp(min=2**-126)
    scale_exp_f = torch.ceil(torch.log2(safe_max / 6.0)) + 127.0
    scale_exp = scale_exp_f.clamp(0, 254).to(torch.int32)
    scale_exp[max_abs == 0] = 0
    scales_out = scale_exp.to(torch.uint8)  # [out_dim, num_blocks]

    # Scale value: 2^(scale_exp - 127)
    scale_val = torch.pow(2.0, (scale_exp.float() - 127.0))  # [out_dim, num_blocks]

    # Normalize values by scale
    w_norm = w / scale_val.unsqueeze(-1).clamp(min=2**-126)  # [out_dim, num_blocks, 32]

    # Quantize to nearest FP4 E2M1 magnitude
    w_sign = w_norm.sign()
    w_abs = w_norm.abs()

    # Round to nearest FP4 using midpoint thresholds
    # FP4 values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    nibble = torch.zeros_like(w_abs, dtype=torch.uint8)
    nibble[w_abs >= 0.25] = 1   # 0.5
    nibble[w_abs >= 0.75] = 2   # 1.0
    nibble[w_abs >= 1.25] = 3   # 1.5
    nibble[w_abs >= 1.75] = 4   # 2.0
    nibble[w_abs >= 2.50] = 5   # 3.0
    nibble[w_abs >= 3.50] = 6   # 4.0
    nibble[w_abs >= 5.00] = 7   # 6.0
    # Set sign bit (bit 3) for negative values
    nibble[w_sign < 0] |= 8

    # Pack pairs of nibbles into bytes: byte = lo_nibble | (hi_nibble << 4)
    even = nibble[:, :, 0::2]  # [out_dim, num_blocks, 16]
    odd = nibble[:, :, 1::2]   # [out_dim, num_blocks, 16]
    packed = (even | (odd << 4)).to(torch.uint8)  # [out_dim, num_blocks, 16]

    # Add batch dim=1 for compatibility with pack_mxfp4_workgroup
    return packed.unsqueeze(0).contiguous(), scales_out.unsqueeze(0).contiguous()


def pack_mxfp4_workgroup(blocks: torch.Tensor, scales: torch.Tensor,
                         output_per_wg: int = 16,
                         target_out_dim: int = None,
                         target_num_blocks: int = None,
                         row_stride_blocks: int = None,
                         out_stride_rows: int = None,
                         split_scales: bool = False,
                         section: str = "both") -> torch.Tensor:
    """Repack MXFP4 blocks+scales into workgroup layout for the MXFP4 GEMV kernel.

    Args:
        blocks: uint8 [E, out_dim, num_blocks, 16] - packed FP4 nibbles
        scales: uint8 [E, out_dim, num_blocks] - E8M0 block scales
        output_per_wg: output rows per workgroup (default 16)
        target_out_dim: pad output dim to this value (for MFMA alignment)
        target_num_blocks: pad num_blocks to this value (for reduction alignment)
        row_stride_blocks: pitch of a stored DATA row, in 16-byte blocks. Defaults
            to target_num_blocks, i.e. rows packed exactly as wide as the
            reduction. Set it wider to reproduce a buffer someone else padded
            (vLLM stores GPT-OSS rows 96 blocks apart while the MFMA still
            reduces 92) -- see MPK_MOE_K_STRIDE in the MoE kernel. Scales are
            NOT restrided: the kernel indexes them at K_REDUCE/32 either way.
        out_stride_rows: rows one expert occupies in the DATA section. Defaults
            to target_out_dim, i.e. experts packed exactly as tall as the kernel
            computes. Set it taller to reproduce a buffer someone else padded on
            the OUTPUT axis (vLLM stores GPT-OSS experts 6144/3072 rows apart
            while fleet_mk computes 5888/2944) -- see MPK_MOE_N_STRIDE. Requires
            split_scales, and like row_stride_blocks it moves ONLY the data
            section: the scale section keeps fleet_mk's own row count, because in
            split mode the two sections have different writers by construction.
        split_scales: emit ALL data followed by ALL scales, instead of
            interleaving them per workgroup. Must match the MoE kernel's
            -DMPK_MOE_SPLIT_SCALES. The byte CONTENT of each section is
            identical either way -- only the interleave changes -- which is what
            lets the two modes be diffed section-against-section.
        section: which of the two split sections to emit -- "both" (default),
            "data", or "scales". Requires split_scales, since the sections are
            only separable once the interleave is gone.

            "scales" exists so fleet_mk can build a scale-only slab for a data
            section it does NOT own: when the data pointer aliases vLLM's
            w13_weight/w2_weight, packing the data bytes would materialize a
            second 60 GiB copy of the very thing being aliased, defeating the
            point. So the whole data path is skipped -- not just the final
            concat, but the target_out_dim / target_num_blocks / row_stride /
            out_stride widenings too, none of which would otherwise be free.
            `blocks` is then read for its shape only, and the caller may hand in
            the foreign tensor unmodified.

    Returns:
        uint8 tensor. Interleaved (default): [E, expert_wgs, wg_bytes], per
        workgroup [data: OPW * K_STRIDE/2][scales: OPW * num_blocks].
        Split: flat [E * (expert_wgs*wg_data + expert_wgs*wg_scale)] with the
        two sections contiguous, data first. Flat because the two sections have
        different per-workgroup sizes, so there is no honest 3-D shape; the
        kernel walks it with explicit pitches. With `section` set, the same flat
        buffer truncated to just that section.
    """
    assert section in ("both", "data", "scales"), f"bad section {section!r}"
    assert section == "both" or split_scales, (
        f"section={section!r} asks for one of the two split sections, but "
        f"split_scales=False interleaves them per workgroup -- there are no "
        f"separable sections to pick from")
    want_data = section != "scales"
    want_scales = section != "data"
    E, out_dim, num_blocks, B = blocks.shape
    assert B == 16, f"Expected 16 bytes per block, got {B}"
    assert scales.shape == (E, out_dim, num_blocks)

    # Pad output dimension if needed. The `want_data` guards below keep the row
    # and block COUNTS updating exactly as they always did -- the scale section's
    # geometry is derived from them -- while skipping the concats that would
    # allocate a data section the caller has said it does not want.
    if target_out_dim is not None and out_dim < target_out_dim:
        pad_rows = target_out_dim - out_dim
        if want_data:
            blocks = torch.cat([blocks,
                torch.zeros(E, pad_rows, num_blocks, 16, dtype=torch.uint8, device=blocks.device)], dim=1)
        scales = torch.cat([scales,
            torch.zeros(E, pad_rows, num_blocks, dtype=torch.uint8, device=scales.device)], dim=1)
        out_dim = target_out_dim

    # Pad num_blocks (reduction dimension) if needed
    if target_num_blocks is not None and num_blocks < target_num_blocks:
        pad_blks = target_num_blocks - num_blocks
        if want_data:
            blocks = torch.cat([blocks,
                torch.zeros(E, out_dim, pad_blks, 16, dtype=torch.uint8, device=blocks.device)], dim=2)
        scales = torch.cat([scales,
            torch.zeros(E, out_dim, pad_blks, dtype=torch.uint8, device=scales.device)], dim=2)
        num_blocks = target_num_blocks

    assert out_dim % output_per_wg == 0, \
        f"out_dim {out_dim} must be divisible by output_per_wg {output_per_wg}"

    # Widen the stored data row if asked. This appends dead blocks to the END of
    # every row, which is exactly what a foreign padder produces: the reduction
    # still stops at num_blocks, the next row just starts further along. Only the
    # data side moves -- scales keep their own num_blocks pitch, matching the
    # kernel, which derives row_scale_base from K_REDUCE and row_data_base from
    # K_STRIDE independently.
    if row_stride_blocks is not None:
        assert row_stride_blocks >= num_blocks, (
            f"row stride {row_stride_blocks} blocks is narrower than the "
            f"{num_blocks}-block reduction -- rows would overlap")
        if row_stride_blocks > num_blocks and want_data:
            blocks = torch.cat([blocks, torch.zeros(
                E, out_dim, row_stride_blocks - num_blocks, 16,
                dtype=torch.uint8, device=blocks.device)], dim=2)

    # Make the stored expert taller if asked. Same shape of change as the row
    # stride one axis over: dead ROWS appended after the ones the kernel
    # computes, so the next EXPERT starts further along. Unlike the K pad these
    # rows are never addressed at all -- no tile covers them -- so their content
    # is irrelevant rather than required-zero. Data only; `scales` keeps out_dim.
    data_rows = out_dim
    if out_stride_rows is not None:
        assert split_scales, (
            "out_stride_rows describes a foreign expert pitch, which implies a "
            "foreign buffer; that buffer cannot also carry fleet_mk's per-workgroup "
            "interleaved scales. Pass split_scales=True.")
        assert out_stride_rows >= out_dim, (
            f"expert pitch {out_stride_rows} rows is shorter than the {out_dim} "
            f"rows the kernel computes -- experts would overlap")
        if out_stride_rows > out_dim:
            if want_data:
                blocks = torch.cat([blocks, torch.zeros(
                    E, out_stride_rows - out_dim, blocks.shape[2], 16,
                    dtype=torch.uint8, device=blocks.device)], dim=1)
            data_rows = out_stride_rows

    expert_wgs = out_dim // output_per_wg
    data_wgs = data_rows // output_per_wg
    stride_blocks = row_stride_blocks or num_blocks
    K_half = stride_blocks * 16  # stored bytes per row (16 bytes per block)
    wg_data_bytes = output_per_wg * K_half
    wg_scale_bytes = output_per_wg * num_blocks
    wg_bytes = wg_data_bytes + wg_scale_bytes

    # Reshape blocks: [E, data_rows, stride_blocks, 16] -> [E, data_wgs, OPW, K_half]
    # Skipped entirely when the data section is not wanted: `blocks` then still
    # holds the caller's original (un-widened) shape, so this reshape would be
    # both wasteful and wrong.
    if want_data:
        data = blocks.reshape(E, data_wgs, output_per_wg, -1)  # [E, wgs, OPW, K_half]
        data_flat = data.reshape(E, data_wgs, wg_data_bytes)

    # Reshape scales: [E, out_dim, num_blocks] -> [E, expert_wgs, OPW, num_blocks]
    sc = scales.reshape(E, expert_wgs, output_per_wg, num_blocks)
    sc_flat = sc.reshape(E, expert_wgs, wg_scale_bytes)

    if split_scales:
        # [all data][all scales], each section expert-major then workgroup-major.
        # Flattening a contiguous [E, wgs, bytes] gives exactly that order, so the
        # kernel reaches a workgroup's scales with
        #   base + E*EXPERT_DATA + e*EXPERT_SCALE + wg*WG_SCALE
        # where EXPERT_DATA is data-only. Both sections hold the same bytes as the
        # interleaved mode, in the same within-section order -- only the interleave
        # is gone.
        if not want_data:
            # Scale section alone, in the same expert-major/workgroup-major order
            # it would occupy inside the full buffer. The kernel reaches it with
            # its own base pointer plus e*EXPERT_SCALE + wg*WG_SCALE -- the
            # E*EXPERT_DATA term drops out because the data lives elsewhere.
            packed = sc_flat.reshape(-1)
            assert packed.numel() == E * expert_wgs * wg_scale_bytes
            return packed.contiguous()
        if not want_scales:
            packed = data_flat.reshape(-1)
            assert packed.numel() == E * data_wgs * wg_data_bytes
            return packed.contiguous()
        packed = torch.cat([data_flat.reshape(-1), sc_flat.reshape(-1)])
        assert packed.numel() == E * (data_wgs * wg_data_bytes
                                      + expert_wgs * wg_scale_bytes)
        return packed.contiguous()

    # Concatenate: [E, wgs, wg_data_bytes + wg_scale_bytes]
    packed = torch.cat([data_flat, sc_flat], dim=2)
    assert packed.shape == (E, expert_wgs, wg_bytes)
    return packed.contiguous()


def shuffle_oproj_workgroups_kmajor(packed: torch.Tensor,
                                    output_per_wg: int = 16) -> torch.Tensor:
    """Repack packed O-proj workgroups into lane-native K128 fragments.

    Ported from fleet's `demo/gpt_oss/demo.py`, which is the definition of the
    layout `-DMPK_OPROJ_KMAJOR` compiles the kernel to expect. The only change
    is that this accepts the rank-3 `[1, n_wgs, wg_bytes]` that
    `pack_mxfp4_workgroup` returns as well as fleet's rank-2, and gives back
    whichever rank it was handed.

    The data prefix goes from ``[row, k128, quarter, 16B]`` to
    ``[k128, quarter, row, 16B]``; the scale suffix stays row-major. Since
    ``lane_id == quarter * 16 + row`` for the 16x16x128 MFMA A operand, the new
    order puts lane L's fragment at byte ``L * 16`` of its K128 block, so a
    wave reads 64 consecutive 16-byte chunks instead of 16-way-conflicting on a
    2048-byte row stride. See the MPK_OPROJ_KMAJOR block in
    gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh.

    This is a permutation, not a requantization -- every lane still receives
    exactly the bytes it received before, so results are bit-exact.

    **Host and kernel must agree.** Applying this without -DMPK_OPROJ_KMAJOR,
    or the reverse, is not a build error and not a crash: every lane reads a
    valid byte, just the wrong one. The symptom is silently wrong logits.
    """
    squeezed = False
    if packed.ndim == 3:
        if packed.shape[0] != 1:
            raise ValueError(
                f"expected a single-expert O-proj tile, got E={packed.shape[0]}")
        packed = packed.squeeze(0)
        squeezed = True
    if packed.ndim != 2:
        raise ValueError(
            f"expected a rank-2 [n_wgs, wg_bytes] O-proj tile, got "
            f"{tuple(packed.shape)}")
    if output_per_wg != 16:
        raise ValueError(
            "the K-major O-proj layout is defined for 16-row tiles only "
            f"(got output_per_wg={output_per_wg}); the kernel-side "
            "static_assert enforces the same thing")

    n_wgs, wg_bytes = packed.shape
    # Solve for the reduction size: each row costs K/2 data bytes + K/32
    # scale bytes, so wg_bytes = OPW * K * (1/2 + 1/32).
    reduction = (wg_bytes * 32) // (output_per_wg * 17)
    data_bytes = output_per_wg * (reduction // 2)
    if data_bytes + output_per_wg * (reduction // 32) != wg_bytes:
        raise ValueError(
            f"wg_bytes={wg_bytes} is not a valid MXFP4 record for "
            f"output_per_wg={output_per_wg}")
    k128_blocks = reduction // 128

    data = packed[:, :data_bytes].reshape(
        n_wgs, output_per_wg, k128_blocks, 4, 16)
    # [wg, row, k128, quarter, byte] -> [wg, k128, quarter, row, byte]
    data = data.permute(0, 2, 3, 1, 4).reshape(n_wgs, data_bytes)
    out = torch.cat([data, packed[:, data_bytes:]], dim=1).contiguous()
    return out.unsqueeze(0).contiguous() if squeezed else out
