#!/usr/bin/env python3
"""Compare titan decode runs: latency, first token, and generated text.

Titan's decode is NOT bit-deterministic across runs of the same binary (see
the fleet-header wiki article), so a single run of each arm cannot resolve a
sub-1% latency delta and a text-hash difference is NOT by itself evidence of a
correctness change. This script reports the within-arm spread alongside the
between-arm delta so the two can be told apart.

Usage:
    python3 tools/compare_runs.py --arm base /tmp/base*.log \
                                  --arm shadow /tmp/shadow_run*.log

Each log is a full demo_gpt_oss_120b.py stdout capture. Reads the summary
"Non-outlier avg" line -- never the tail FWD_PASS lines.
"""

import argparse
import hashlib
import re
import statistics
import sys

# Anchor for locating the generated text inside the log. The chat template
# always opens with this, so find() on it gives the start of the completion.
TEXT_ANCHOR = "systemYou are ChatGPT"


def parse_log(path):
    """Pull latency, first decoded token, and generated text out of one log."""
    with open(path, errors="replace") as fh:
        raw = fh.read()

    out = {"path": path, "avg": None, "decode_avg": None, "tok1": None,
           "text": None, "fault": False}

    # A memory-access fault or an abort leaves a truncated log that still
    # contains valid-looking timing lines above it. Flag it rather than
    # silently averaging in a crashed run.
    if "Memory access fault" in raw or "HSA_STATUS_ERROR" in raw:
        out["fault"] = True

    for line in raw.split("\n"):
        if "Non-outlier avg" in line:
            m = re.search(r"([\d.]+)\s*ms", line)
            if m:
                out["avg"] = float(m.group(1))
        elif "Decode avg" in line:
            m = re.search(r"([\d.]+)\s*ms", line)
            if m:
                out["decode_avg"] = float(m.group(1))
        elif "iter 1:" in line and "next_token" in line:
            m = re.search(r"next_token=(\d+)", line)
            if m:
                out["tok1"] = int(m.group(1))

    i = raw.find(TEXT_ANCHOR)
    if i >= 0:
        # Text runs to the next summary banner; fall back to end of log.
        rest = raw[i:]
        end = min((p for p in (rest.find("\n====="),
                               rest.find("\nDecode avg"),
                               rest.find("\n[TITAN_TIME]"))
                   if p > 0), default=len(rest))
        out["text"] = rest[:end]
    return out


def common_prefix(strings):
    strings = [s for s in strings if s]
    if not strings:
        return 0
    n = min(len(s) for s in strings)
    p = 0
    while p < n and len({s[p] for s in strings}) == 1:
        p += 1
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", action="append", required=True,
                    metavar=("NAME", "LOG"),
                    help="arm name followed by one or more log paths")
    args = ap.parse_args()

    arms = {}
    for entry in args.arm:
        arms[entry[0]] = [parse_log(p) for p in entry[1:]]

    print(f"{'arm':10s} {'n':>2s} {'non-outlier avg (ms)':>22s} "
          f"{'spread':>8s} {'tok1':>7s} {'texts':>6s}")
    print("-" * 66)

    summary = {}
    for name, runs in arms.items():
        avgs = [r["avg"] for r in runs if r["avg"] is not None]
        toks = {r["tok1"] for r in runs if r["tok1"] is not None}
        hashes = {hashlib.sha256(r["text"].encode()).hexdigest()[:12]
                  for r in runs if r["text"]}
        faults = sum(1 for r in runs if r["fault"])
        if not avgs:
            print(f"{name:10s} -- no latency parsed --")
            continue
        best, med = min(avgs), statistics.median(avgs)
        spread = max(avgs) - best
        summary[name] = {"best": best, "median": med, "spread": spread,
                         "avgs": avgs}
        toks_s = ",".join(str(t) for t in sorted(toks)) or "?"
        print(f"{name:10s} {len(avgs):2d} "
              f"best {best:6.3f}  med {med:6.3f} {spread:8.3f} "
              f"{toks_s:>7s} {len(hashes):6d}")
        if faults:
            print(f"{'':10s}   !! {faults} run(s) hit a memory fault")
        if len(toks) > 1:
            print(f"{'':10s}   !! first token DIFFERS across runs -- "
                  f"correctness, not noise")
        if len(hashes) > 1:
            p = common_prefix([r["text"] for r in runs])
            print(f"{'':10s}   note: {len(hashes)} distinct texts, "
                  f"common prefix {p} chars (titan is non-deterministic)")

    names = list(summary)
    if len(names) == 2:
        a, b = names
        d = summary[b]["median"] - summary[a]["median"]
        noise = max(summary[a]["spread"], summary[b]["spread"])
        print()
        print(f"delta ({b} - {a}) on medians: {d:+.3f} ms")
        print(f"largest within-arm spread   : {noise:.3f} ms")
        if abs(d) <= noise:
            print("VERDICT: delta is within run-to-run noise -- not resolvable. "
                  "Add replicates or measure a phase directly.")
        else:
            print(f"VERDICT: delta exceeds noise "
                  f"({'regression' if d > 0 else 'improvement'}).")

    # A first-token change is the one signal that survives non-determinism.
    all_toks = {r["tok1"] for runs in arms.values() for r in runs
                if r["tok1"] is not None}
    if len(all_toks) > 1:
        print(f"\n!! first token differs ACROSS arms: {sorted(all_toks)} "
              f"-- investigate before trusting any latency number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
