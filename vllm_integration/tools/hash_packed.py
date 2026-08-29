#!/usr/bin/env python3
"""Hash fleet_mk's packed weight slots, and diff two such hash dumps.

The single-copy weight-sourcing work rewrites *where* fleet_mk's weights come from
(mirage's reference model -> vLLM's live modules) while claiming the packed bytes
are unchanged. That claim is only worth anything if it is checked mechanically:
a slot that silently changes produces fluent-but-wrong text, which no amount of
reading the output catches.

Two modes:

  Dump   -- set FLEET_MK_HASH_PACKED=<path> on a normal harness/bench run. The mixin
            writes one `slot_index<TAB>sha256<TAB>shape<TAB>dtype` line per packed
            tensor, in pointer-table order.

  Diff   -- python3 tools/hash_packed.py <baseline.tsv> <candidate.tsv>
            Reports slots whose hash moved, grouped by the per-layer slot position
            so "slot 8 differs on every layer" reads as one finding rather than 36.

Per-layer slot names are from packing_moe.pack_moe_layer; the trailing entries
(final_norm, lm_head, lm_head_bias, cos, sin) are appended after the layer slots.
"""

import sys

MOE_SLOT_NAMES = [
    "qkv_weight", "qkv_bias", "oproj_weight", "oproj_bias",
    "norm_pre", "norm_post", "router_w", "router_b",
    "gate_up_w", "down_w", "gate_up_bias", "down_bias", "sinks",
]


def hash_tensors(named_tensors, path):
    """Write `name<TAB>sha256<TAB>shape<TAB>dtype` for each (name, tensor).

    Called from the mixin under FLEET_MK_HASH_PACKED; kept here so the hashing rule
    lives next to the diff that consumes it.
    """
    import hashlib

    import torch
    with open(path, "w") as f:
        for name, t in named_tensors:
            if t is None:
                f.write(f"{name}\t<none>\t-\t-\n")
                continue
            c = t.detach().contiguous().cpu()
            # .numpy() rejects bfloat16 (no numpy dtype), and most fleet_mk slots
            # are bf16 norms/biases. view(uint8) hashes the raw storage bytes,
            # which is what we actually want to compare anyway.
            digest = hashlib.sha256(
                c.view(torch.uint8).numpy().tobytes()).hexdigest()
            f.write(f"{name}\t{digest}\t{tuple(c.shape)}\t{c.dtype}\n")
    print(f"[hash_packed] wrote {len(named_tensors)} slot hashes to {path}",
          flush=True)


def _load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                rows[parts[0]] = tuple(parts[1:])
    return rows


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    a, b = _load(sys.argv[1]), _load(sys.argv[2])

    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    common = [k for k in a if k in b]
    diff = [k for k in common if a[k][0] != b[k][0]]

    print(f"baseline slots: {len(a)}   candidate slots: {len(b)}   "
          f"common: {len(common)}")
    if only_a:
        print(f"  MISSING in candidate ({len(only_a)}): {only_a[:8]}")
    if only_b:
        print(f"  NEW in candidate ({len(only_b)}): {only_b[:8]}")

    if not diff:
        print("  ALL COMMON SLOT HASHES IDENTICAL")
    else:
        # Group "layer<N>.<slot>" by slot so a systematic difference reads as one
        # line instead of 36 near-identical ones.
        by_slot = {}
        for k in diff:
            slot = k.split(".", 1)[1] if k.startswith("layer") and "." in k else k
            by_slot.setdefault(slot, []).append(k)
        print(f"  {len(diff)} SLOT(S) DIFFER, grouped by slot name:")
        for slot, keys in sorted(by_slot.items()):
            ex = sorted(keys)[0]
            print(f"    {slot:<14} {len(keys):>3} instance(s)   e.g. {ex}")
            print(f"        base {a[ex][0][:16]} shape={a[ex][1]} dtype={a[ex][2]}")
            print(f"        cand {b[ex][0][:16]} shape={b[ex][1]} dtype={b[ex][2]}")

    shape_diff = [k for k in common if a[k][1:] != b[k][1:]]
    if shape_diff:
        print(f"  NOTE {len(shape_diff)} slot(s) differ in shape/dtype "
              f"(e.g. {shape_diff[0]})")
    return 1 if (diff or only_a or only_b) else 0


if __name__ == "__main__":
    sys.exit(main())
