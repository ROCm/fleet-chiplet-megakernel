#!/usr/bin/env python3
"""Aggregate fleet's [FUSED_PHASE] device printf into per-phase medians.

Fleet's fused-layer header prints one line per (layer, xcd) when built with
-DMPK_ENABLE_DEVICE_TASK_TIMING. A 438-token run emits ~126k of them, which is
far too many to read and far too few to eyeball -- hence this.

The reason this exists rather than a one-off grep: the question that keeps
coming back is "which phase absorbed the regression", and answering it needs
the SAME reduction applied to two logs. A throwaway parse gives two numbers
you cannot line up.

Two things it does that a naive mean does not:

  * Medians, not means. The tail of any decode run carries multi-ms outliers
    (allocator, eviction); a mean over 126k samples is dominated by them.
  * Splits EVEN vs ODD layers when --alternating is passed. Under fleet's
    header the sliding window is a template parameter, so even and odd layers
    execute two DIFFERENT instantiations of the layer body. If a regression
    lives in one body and not the other, a pooled median hides it completely.

Layer parity is not in the printf, so it is recovered from position: the
lines arrive in layer order within a token, and there are NUM_LAYERS * 8
(xcd) lines per token. --layers/--xcds set that shape.

  * --by-token bins the run along its own time axis. This exists because a
    pooled median silently assumes every token costs the same, and under a
    data-dependent MoE that is false: routing follows the text, so a run whose
    output degenerates into repetition settles onto a small set of experts
    whose weights then stay resident. Comparing a coherent run against a
    degenerate one compares cache states, not code. The trajectory shows that
    directly -- a phase that drifts across the run is a locality effect, a
    phase that is flat is a real cost.

Usage:
  fused_phase_stats.py LOG [LOG ...] [--alternating] [--layers 36] [--xcds 8]
  fused_phase_stats.py --compare A.log B.log [--alternating]
  fused_phase_stats.py LOG --by-token [--bins 20]
"""
import argparse
import re
import statistics
import sys

# [FUSED_PHASE] xcd=0 qkv_attn=.. xcd_barrier=.. oproj_topk=.. moe=.. total=..
#   | qkv_gemm=.. qkv_bar=.. attn=.. merge=..(m=..+f=..) wait=..
FIELD = re.compile(r"(\w+)=(-?\d+\.?\d*)")

PHASES = ["qkv_attn", "xcd_barrier", "oproj_topk", "moe", "total",
          "qkv_gemm", "qkv_bar", "attn", "merge", "m", "f", "wait"]


def parse(path, layers, xcds, alternating):
    """Return {phase: [values]} or, if alternating, {(parity, phase): [values]}."""
    buckets = {}
    n = 0
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if "[FUSED_PHASE]" not in line:
                continue
            fields = dict(FIELD.findall(line))
            if "xcd" not in fields:
                continue
            # Position within the token determines the layer; the printf
            # carries xcd but not layer.
            layer = (n // xcds) % layers
            n += 1
            key_prefix = ("even" if layer % 2 == 0 else "odd",) if alternating else ()
            for ph in PHASES:
                if ph not in fields:
                    continue
                v = float(fields[ph])
                if v < 0:          # header emits -1 for "not measured here"
                    continue
                buckets.setdefault(key_prefix + (ph,), []).append(v)
    return buckets, n


def med(vals):
    return statistics.median(vals) if vals else float("nan")


def report(path, buckets, n, alternating):
    print(f"\n=== {path}  ({n} FUSED_PHASE lines) ===")
    if not alternating:
        print(f"{'phase':<14}{'median':>10}{'n':>10}")
        for ph in PHASES:
            v = buckets.get((ph,))
            if v:
                print(f"{ph:<14}{med(v):>10.2f}{len(v):>10}")
        return
    print(f"{'phase':<14}{'even(A)':>10}{'odd(B)':>10}{'delta':>10}{'n even':>9}")
    for ph in PHASES:
        e = buckets.get(("even", ph))
        o = buckets.get(("odd", ph))
        if not e and not o:
            continue
        me, mo = med(e), med(o)
        print(f"{ph:<14}{me:>10.2f}{mo:>10.2f}{mo - me:>10.2f}{len(e or []):>9}")


def compare(a, b, layers, xcds, alternating):
    ba, na = parse(a, layers, xcds, alternating)
    bb, nb = parse(b, layers, xcds, alternating)
    report(a, ba, na, alternating)
    report(b, bb, nb, alternating)
    print(f"\n=== delta: {b} minus {a} (pooled) ===")
    print(f"{'phase':<14}{'A':>10}{'B':>10}{'delta':>10}")
    for ph in PHASES:
        va = [v for k, v in ba.items() if k[-1] == ph for v in v]
        vb = [v for k, v in bb.items() if k[-1] == ph for v in v]
        if not va or not vb:
            continue
        print(f"{ph:<14}{med(va):>10.2f}{med(vb):>10.2f}{med(vb) - med(va):>10.2f}")


def by_token(path, layers, xcds, bins):
    """Median per phase over `bins` equal slices of the run, in token order.

    One token is layers*xcds lines. Reports the first and last bin side by
    side so drift is readable without squinting at the whole table.
    """
    per_token = {}          # token index -> {phase: [values]}
    n = 0
    lines_per_token = layers * xcds
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if "[FUSED_PHASE]" not in line:
                continue
            fields = dict(FIELD.findall(line))
            if "xcd" not in fields:
                continue
            tok = n // lines_per_token
            n += 1
            d = per_token.setdefault(tok, {})
            for ph in PHASES:
                if ph not in fields:
                    continue
                v = float(fields[ph])
                if v < 0:
                    continue
                d.setdefault(ph, []).append(v)

    ntok = len(per_token)
    if ntok < bins:
        bins = max(1, ntok)
    size = ntok / bins
    print(f"\n=== {path}  ({n} lines, {ntok} tokens, {bins} bins) ===")
    header = f"{'bin':<6}{'tok':>8}" + "".join(f"{ph:>11}" for ph in
                                              ("qkv_gemm", "moe", "oproj_topk",
                                               "attn", "xcd_barrier", "total"))
    print(header)
    firsts, lasts = {}, {}
    for b in range(bins):
        lo, hi = int(b * size), int((b + 1) * size)
        agg = {}
        for t in range(lo, hi):
            for ph, vs in per_token.get(t, {}).items():
                agg.setdefault(ph, []).extend(vs)
        row = f"{b:<6}{lo:>8}"
        for ph in ("qkv_gemm", "moe", "oproj_topk", "attn", "xcd_barrier", "total"):
            m = med(agg.get(ph, []))
            row += f"{m:>11.2f}"
            if b == 0:
                firsts[ph] = m
            if b == bins - 1:
                lasts[ph] = m
        print(row)
    print(f"\n{'phase':<14}{'first bin':>11}{'last bin':>11}{'drift':>11}")
    for ph in ("qkv_gemm", "moe", "oproj_topk", "attn", "xcd_barrier", "total"):
        print(f"{ph:<14}{firsts[ph]:>11.2f}{lasts[ph]:>11.2f}"
              f"{lasts[ph] - firsts[ph]:>11.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--xcds", type=int, default=8)
    ap.add_argument("--alternating", action="store_true",
                    help="split even/odd layers (the two template instantiations)")
    ap.add_argument("--compare", action="store_true",
                    help="two logs: report each, then the pooled delta")
    ap.add_argument("--by-token", action="store_true",
                    help="median per phase over time-ordered bins (locality drift)")
    ap.add_argument("--bins", type=int, default=20)
    args = ap.parse_args()

    if args.by_token:
        for path in args.logs:
            by_token(path, args.layers, args.xcds, args.bins)
        return
    if args.compare:
        if len(args.logs) != 2:
            sys.exit("--compare takes exactly two logs")
        compare(args.logs[0], args.logs[1], args.layers, args.xcds, args.alternating)
        return
    for path in args.logs:
        b, n = parse(path, args.layers, args.xcds, args.alternating)
        report(path, b, n, args.alternating)


if __name__ == "__main__":
    main()
