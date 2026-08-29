#!/usr/bin/env python3
"""Per-slot GPU byte budget for fleet_mk's packed weights, split by ALIASED vs OWNED.

Answers "how many copies of each weight class are resident" with arithmetic that
can be checked against the [FLEET_MK_MEM] before/after deltas the harness prints,
instead of eyeballing a total. Every number here is derived from the same
formulas fleet_mk_generate.py uses to emit the kernel constants, so a divergence
between this and the measured delta is a real discrepancy, not a modelling gap.

Three classes, and the distinction is the whole point:

  ALIASED   fleet_mk's pointer table points at vLLM's own allocation. Zero added
            bytes. Only the MoE expert DATA qualifies -- see
            fleet_megakernel_vllm/model.py:199.
  OWNED     fleet_mk allocates these. They are NOT duplicates of a live vLLM
            tensor: either vLLM deleted its version (MoE scales, dropped in
            process_weights_after_loading) or fleet_mk needs a different numeric
            format than vLLM holds (attention/lm_head, bf16 -> MXFP4).
  SHADOWED  fleet_mk owns a copy AND vLLM still holds its own of the same logical
            weight, in a different format. This is the only class where "two
            copies" is literally true, and it is what the linear layers are.

Usage:
    python3 tools/mem_budget.py                    # gpt_oss_120b defaults
    python3 tools/mem_budget.py --measured 4.382   # check against a run
"""
import argparse

GiB = 1024 ** 3


def mxfp4_bytes(rows, k):
    """Packed MXFP4: k/2 data bytes + k/32 E8M0 scale bytes per row."""
    assert k % 32 == 0, k
    return rows * (k // 2 + k // 32)


def budget(*, num_layers=36, hidden=2880, padded_hidden=2944,
           num_q=64, num_kv=8, head_dim=64, num_experts=128,
           moe_inter=2880, padded_moe_inter=2944,
           vocab=201088, padded_vocab=201216,
           w13_opw=128, w2_opw=64,
           vllm_pad=3072):
    ph = padded_hidden
    pi = padded_moe_inter
    qkv_out = num_q * head_dim + 2 * num_kv * head_dim      # 5120
    attn_red = num_q * head_dim                             # 4096
    w13_out = 2 * pi                                        # 5888
    w2_out = pi                                             # 2944
    bf16 = lambda n: 2 * n

    # ── per-layer slots, in pack_moe_layer order ─────────────────────────────
    # (name, bytes, class). MoE data/scale rows are split out because only the
    # data half is aliasable.
    per_layer = [
        ("[0] qkv weight   MXFP4", mxfp4_bytes(qkv_out, ph),        "SHADOWED"),
        ("[1] qkv bias     bf16",  bf16(qkv_out),                   "SHADOWED"),
        ("[2] oproj weight MXFP4", mxfp4_bytes(ph, attn_red),       "SHADOWED"),
        ("[3] oproj bias   bf16",  bf16(ph),                        "SHADOWED"),
        ("[4] norm_pre     bf16",  bf16(ph),                        "SHADOWED"),
        ("[5] norm_post    bf16",  bf16(ph),                        "SHADOWED"),
        ("[6] router w     bf16",  bf16(num_experts * ph),          "SHADOWED"),
        ("[7] router b     bf16",  bf16(num_experts),               "SHADOWED"),
        ("[10] w13 bias    bf16",  bf16(num_experts * w13_out),     "SHADOWED"),
        ("[11] w2 bias     bf16",  bf16(num_experts * w2_out),      "SHADOWED"),
        ("[12] attn sinks  bf16",  bf16(num_q),                     "SHADOWED"),
        # Expert DATA: aliased, so fleet_mk adds zero. Sized at vLLM's pitch, which
        # is what the pointer table now spans.
        ("[8] w13 data     MXFP4", num_experts * 2 * vllm_pad * (vllm_pad // 2),
                                                                    "ALIASED"),
        ("[9] w2 data      MXFP4", num_experts * vllm_pad * (vllm_pad // 2),
                                                                    "ALIASED"),
        # Expert SCALES: fleet_mk's own, at fleet_mk's own row count (N stride moves
        # only the data section -- mxfp4_pack.py:257-266).
        ("[8s] w13 scales  E8M0",  num_experts * w13_out * (ph // 32),
                                                                    "OWNED"),
        ("[9s] w2 scales   E8M0",  num_experts * w2_out * (pi // 32),
                                                                    "OWNED"),
    ]

    rows = [(n, b * num_layers, c) for (n, b, c) in per_layer]
    # ── global slots ─────────────────────────────────────────────────────────
    rows += [
        ("lm_head          MXFP4", mxfp4_bytes(padded_vocab, ph),   "SHADOWED"),
        ("lm_head bias     bf16",  bf16(padded_vocab),              "SHADOWED"),
    ]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", type=float, default=None,
                    help="GiB delta from [FLEET_MK_MEM] after-pack, to check against")
    a = ap.parse_args()

    rows = budget()
    w = max(len(n) for n, _, _ in rows)
    tot = {}
    print(f"{'slot':<{w}}  {'bytes':>16}  {'GiB':>8}  class")
    print("-" * (w + 42))
    for name, b, cls in sorted(rows, key=lambda r: (r[2], -r[1])):
        tot[cls] = tot.get(cls, 0) + b
        print(f"{name:<{w}}  {b:>16,}  {b/GiB:>8.3f}  {cls}")
    print("-" * (w + 42))
    for cls in ("ALIASED", "OWNED", "SHADOWED"):
        print(f"{cls:<{w}}  {tot.get(cls,0):>16,}  {tot.get(cls,0)/GiB:>8.3f}")
    added = tot.get("OWNED", 0) + tot.get("SHADOWED", 0)
    print(f"{'fleet_mk adds (OWNED+SHADOWED)':<{w}}  {added:>16,}  {added/GiB:>8.3f}")
    if a.measured is not None:
        d = added / GiB - a.measured
        print(f"{'measured [FLEET_MK_MEM] delta':<{w}}  {'':>16}  "
              f"{a.measured:>8.3f}\n{'residual (model - measured)':<{w}}  "
              f"{'':>16}  {d:>8.3f}")


if __name__ == "__main__":
    main()
