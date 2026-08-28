#!/usr/bin/env python3
"""Summarize a titan_vllm harness/bench log: per-token timing, footprint, output.

Every step of the single-copy weight-sourcing work has to be compared against a
baseline on three axes -- latency, memory footprint, and generated text -- so
this reads all three out of one log instead of eyeballing tail lines. The
non-outlier average is the number to quote (see the "Use decode avg" rule); the
raw tail lines drift and reading them by hand has produced wrong calls before.

Outliers are trimmed with a MAD rule rather than a fixed percentile: the first
token after a JIT compile can be several hundred microseconds high, and a
percentile trim on a short run either keeps it or throws away good samples.

Usage:
  python3 tools/run_stats.py /tmp/s0_base_027.log
  python3 tools/run_stats.py /tmp/a.log /tmp/b.log     # compare two runs
"""

import re
import statistics
import sys

TIME_RE = re.compile(
    r"\[TITAN_TIME\]\s+embed=([\d.]+)us\s+layers=([\d.]+)us\s+"
    r"tail=([\d.]+)us\s+total=([\d.]+)us")
MEM_RE = re.compile(r"\[TITAN_MEM\] (.+?):\s+(.*)")
IDS_RE = re.compile(r"generated token ids =====\n\[(.*?)\]", re.S)
TEXT_RE = re.compile(r"===== \[\w+\] text =====\n(.*?)(?:\nINFO |\Z)", re.S)


def _trim_outliers(xs):
    """Drop samples more than 3 MADs from the median. Returns (kept, dropped)."""
    if len(xs) < 4:
        return xs, []
    med = statistics.median(xs)
    mad = statistics.median([abs(x - med) for x in xs]) or 1e-9
    kept = [x for x in xs if abs(x - med) <= 3 * 1.4826 * mad]
    dropped = [x for x in xs if abs(x - med) > 3 * 1.4826 * mad]
    return (kept or xs), dropped


def parse(path):
    with open(path, errors="replace") as f:
        blob = f.read()

    rows = [tuple(float(g) for g in m.groups()) for m in TIME_RE.finditer(blob)]
    mem = MEM_RE.findall(blob)

    ids_m = IDS_RE.search(blob)
    ids = ([int(t) for t in ids_m.group(1).replace("\n", " ").split(",") if t.strip()]
           if ids_m else [])
    text_m = TEXT_RE.search(blob)
    text = text_m.group(1).strip() if text_m else ""

    return dict(path=path, rows=rows, mem=mem, ids=ids, text=text)


def report(r):
    print(f"=== {r['path']} ===")
    if not r["rows"]:
        print("  no [TITAN_TIME] lines -- kernel timing not enabled or run died")
    else:
        totals = [x[3] for x in r["rows"]]
        kept, dropped = _trim_outliers(totals)
        print(f"  tokens timed:      {len(totals)}")
        print(f"  Non-outlier avg:   {statistics.mean(kept):.1f} us/token"
              f"   ({statistics.mean(kept) / 1000:.3f} ms)")
        print(f"  median / min / max {statistics.median(totals):.1f}"
              f" / {min(totals):.1f} / {max(totals):.1f} us")
        if dropped:
            print(f"  dropped outliers:  {len(dropped)} "
                  f"({', '.join(f'{d:.0f}' for d in dropped[:6])})")
        for i, name in enumerate(("embed", "layers", "tail")):
            vals = [x[i] for x in r["rows"]]
            print(f"    {name:<7} avg {statistics.mean(vals):8.1f} us")
    for k, v in r["mem"]:
        print(f"  MEM {k}: {v}")
    if r["ids"]:
        print(f"  first token:       {r['ids'][0]}"
              f"{'  <-- expected 200005/35644' if r['ids'][0] not in (200005, 35644) else ''}")
        print(f"  tokens generated:  {len(r['ids'])}")
    if r["text"]:
        head = r["text"][:200].replace("\n", " ")
        print(f"  text: {head}{'...' if len(r['text']) > 200 else ''}")
    print()
    return r


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    parsed = [report(parse(p)) for p in sys.argv[1:]]

    if len(parsed) == 2:
        a, b = parsed
        if a["rows"] and b["rows"]:
            am = statistics.mean(_trim_outliers([x[3] for x in a["rows"]])[0])
            bm = statistics.mean(_trim_outliers([x[3] for x in b["rows"]])[0])
            print(f"=== delta ===\n  {am:.1f} -> {bm:.1f} us "
                  f"({(bm - am) / am * 100:+.2f}%)")
        # Text differs run to run (MoE reduction order); report where, do not
        # treat divergence as failure -- compare against the within-config noise
        # floor, which is as early as token 3 on this model.
        n = min(len(a["ids"]), len(b["ids"]))
        div = next((i for i in range(n) if a["ids"][i] != b["ids"][i]), None)
        print(f"  first token divergence: "
              f"{div if div is not None else 'none in ' + str(n)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
