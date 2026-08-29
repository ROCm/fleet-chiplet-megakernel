#!/usr/bin/env python3
"""Compare raw source tensors before they are packed.

`hash_packed.py` gates the *packed* bytes, which is the right final check but a
poor diagnostic: a packed slot is a quantized, padded, workgroup-interleaved slab,
so a hash mismatch says "something upstream moved" and nothing more. When fleet_mk's
weights are re-sourced from vLLM's live modules instead of mirage's reference
model, the useful question is asked one level earlier -- is *this* bf16 matrix the
same matrix? -- and the useful answer is a max-abs-diff, not a boolean.

The distinction matters for RoPE in particular. vLLM builds `cos_sin_cache` in
fp32 with `yarn_get_mscale(factor) * attn_factor`; mirage builds bf16 tables with
transformers' `attention_scaling`. Those can agree to ~1e-3 (bf16 rounding) while
never hashing equal, and a hash-only gate would report a failure that is not one.

Used from inside the engine process (the reference model is only alive there),
under FLEET_MK_SRC_DIFF -- see model.py's `_fleet_mk_pack_weights`.
"""

import torch


def diff_tensors(named_pairs, label="", atol=0.0, out=None):
    """Report per-entry agreement for [(name, reference, candidate), ...].

    Shapes are compared first: a mismatch there is a sourcing error (wrong slice,
    wrong module) and no numeric comparison is meaningful. Otherwise both sides are
    cast to fp32 -- comparing a bf16 reference against an fp32 candidate in bf16
    would hide exactly the precision question being asked.

    Returns (num_exact, num_within_atol, num_bad).
    """
    lines, exact, close, bad = [], 0, 0, 0
    for name, ref, cand in named_pairs:
        if ref is None or cand is None:
            lines.append(f"  {name:<22} MISSING ref={ref is not None} "
                         f"cand={cand is not None}")
            bad += 1
            continue
        if tuple(ref.shape) != tuple(cand.shape):
            lines.append(f"  {name:<22} SHAPE {tuple(ref.shape)} vs "
                         f"{tuple(cand.shape)}")
            bad += 1
            continue
        if not ref.is_floating_point():
            # MXFP4 blocks and E8M0 scales are uint8 and can reach ~1 GB per
            # layer; casting them to fp32 to subtract would allocate 8x the
            # tensor for a question that is purely "are these the same bytes".
            same = torch.equal(ref.detach().cpu(), cand.detach().cpu())
            if same:
                exact += 1
                lines.append(f"  {name:<22} EXACT   dtype {ref.dtype} "
                             f"({ref.numel()} bytes)")
            else:
                bad += 1
                ne = (ref.detach().cpu() != cand.detach().cpu())
                first = int(ne.flatten().nonzero()[0].item())
                idx = tuple(int(i) for i in torch.unravel_index(
                    torch.tensor(first), ne.shape))
                lines.append(f"  {name:<22} DIFFERS {int(ne.sum())} of "
                             f"{ne.numel()} bytes; first at {idx}: "
                             f"ref={int(ref[idx])} cand={int(cand[idx])}")
            continue
        a = ref.detach().float()
        b = cand.detach().to(a.device).float()
        d = (a - b).abs()
        mx = d.max().item()
        if mx == 0.0:
            exact += 1
            lines.append(f"  {name:<22} EXACT   dtype {ref.dtype}/{cand.dtype}")
        elif mx <= atol:
            close += 1
            denom = a.abs().max().item() or 1.0
            lines.append(f"  {name:<22} close   max|d|={mx:.3e} "
                         f"rel={mx / denom:.3e}  dtype {ref.dtype}/{cand.dtype}")
        else:
            bad += 1
            denom = a.abs().max().item() or 1.0
            lines.append(f"  {name:<22} DIFFERS max|d|={mx:.3e} "
                         f"rel={mx / denom:.3e}  dtype {ref.dtype}/{cand.dtype}")
            # A magnitude alone cannot distinguish "wrong values" from "right
            # values, wrong position" -- and for tables indexed by position those
            # are entirely different bugs. Report where the worst disagreement is
            # and what both sides hold there.
            flat = int(d.argmax().item())
            idx = tuple(int(i) for i in torch.unravel_index(
                torch.tensor(flat), d.shape))
            lines.append(f"      worst at {idx}: ref={a[idx].item():+.6f} "
                         f"cand={b[idx].item():+.6f}")
            if a.ndim == 2:
                rowbad = (d.max(dim=1).values > atol).nonzero()
                first = int(rowbad[0].item()) if rowbad.numel() else -1
                lines.append(f"      first differing row {first}; "
                             f"ref[{first}][:4]="
                             f"{[round(v, 5) for v in a[first][:4].tolist()]} "
                             f"cand[{first}][:4]="
                             f"{[round(v, 5) for v in b[first][:4].tolist()]}")

    head = (f"[FLEET_MK_SRC_DIFF] {label}: {exact} exact, {close} within "
            f"atol={atol}, {bad} differing")
    body = "\n".join([head] + lines)
    if out:
        with open(out, "a") as f:
            f.write(body + "\n")
    print(body, flush=True)
    return exact, close, bad
