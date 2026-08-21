"""Summarize a seq-len sweep run by run_gpt_oss_seqlen_sweep.sh.

TPOT is the MEDIAN of the per-iteration [FWD_PASS] samples that fall in the
decode phase. Getting to that took two wrong turns, both recorded here because
each produced numbers that looked fine:

  1. [FWD_PASS_TOTAL] avg_ms -- untruncated, but it blends prefill and decode,
     so it is not TPOT.
  2. A two-point differential on [FWD_PASS_TOTAL] total_ms across two decode
     lengths. This subtracts two runs' prefills, and prefill variance scales
     with prefill length: ~1% of a 10.7 s prefill is ~107 ms of noise against
     ~1000 ms of signal. Repeating ctx 4096 gave 2.845 then 3.173 ms/tok, and
     ctx 8192 came out FASTER than 4096. At 32k the noise approaches the whole
     signal.

The per-iter ring works because of how it truncates. It is emitted by deferred
printf at kernel exit and the HIP printf FIFO drops the FRONT of the dump, so
what survives is the TAIL -- which at these context lengths is entirely decode
(verified: every surviving sample has iter > prompt_len and
num_active_tokens == 1). Spread within a run is sd ~0.006 ms, and two
independent repeats of ctx 4096/8192 agreed to 0.5%.

Median, not mean: the first decode iteration after prefill is an outlier (9.93
vs 2.47 ms at ctx 512) and at short contexts few enough samples survive that it
moves the mean by 16%.

Also hashes the generated token ids at each point, so a plausible latency
produced by a broken kernel is visible rather than silently tabulated.
"""
import glob
import hashlib
import json
import os
import re
import statistics
import sys


def parse_log(path):
    with open(path, errors="replace") as f:
        txt = f.read()
    out = {}
    m = re.search(r"\[WALL\] prompt_tokens=(\d+) generated_tokens=(\d+) "
                  r"total_ms=([\d.]+)", txt)
    if m:
        out["prompt_tokens"] = int(m.group(1))
        out["generated_tokens"] = int(m.group(2))
        out["wall_total_ms"] = float(m.group(3))
    # Untruncated device accumulator. Blended prefill+decode, so it is not
    # TPOT, but dropped=0 confirms the run completed every iteration and it is
    # the right clock for prefill cost.
    m = re.search(r"\[FWD_PASS_TOTAL\] iters=(\d+) total_ms=([\d.]+) "
                  r"avg_ms=([\d.]+) dropped=(\d+)", txt)
    if m:
        out["dev_iters"] = int(m.group(1))
        out["dev_total_ms"] = float(m.group(2))
        out["dev_avg_ms"] = float(m.group(3))
        out["dev_dropped"] = int(m.group(4))
    out["samples"] = [
        (int(i), float(t), int(n)) for i, t, n in re.findall(
            r"\[FWD_PASS\] iter=(\d+) time_ms=([\d.]+) "
            r"num_active_tokens=(\d+)", txt)
    ]
    m = re.search(r"ck_fmha_num_kv_chunks=(\d+)", txt)
    if m:
        out["kv_chunks"] = int(m.group(1))
    return out


def txthash(path):
    """Hash the CONTINUATION, not demo.py's "text" field.

    "text" is prompt+continuation, so at a 32k prompt its hash is ~99.99%
    prompt and would look stable even if every generated token changed.
    "token_ids" is the continuation alone (capped at MAX_SAVE_TOKENS).
    """
    if not os.path.exists(path):
        return None, None, None
    with open(path) as f:
        j = json.load(f)
    ids = j.get("token_ids", [])
    h = hashlib.md5(json.dumps(ids).encode()).hexdigest()[:12]
    return h, len(ids), ids


def decode_stats(p):
    """Median per-iter latency over the decode-phase ring samples."""
    prompt = p.get("prompt_tokens")
    if prompt is None:
        return None
    # iter > prompt_tokens puts us past prefill; num_active_tokens == 1 is the
    # decode shape. Requiring both means a mislabeled sample cannot slip in.
    dec = [t for i, t, n in p.get("samples", []) if i > prompt and n == 1]
    if not dec:
        return None
    return {"n": len(dec), "median_ms": statistics.median(dec),
            "min_ms": min(dec), "max_ms": max(dec),
            "sd_ms": statistics.pstdev(dec) if len(dec) > 1 else 0.0}


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "outputs/gpt_oss/seqlen_sweep"
    points = {}
    for log in glob.glob(os.path.join(d, "fleet_*_g*.log")):
        m = re.search(r"fleet_(\d+)_g(\d+)\.log$", log)
        if not m:
            continue
        ctx, gen = int(m.group(1)), int(m.group(2))
        r = parse_log(log)
        r["gen"], r["log"] = gen, log
        h, n, ids = txthash(log[:-4] + "_tokens.json")
        r["txthash"], r["n_saved"], r["ids"] = h, n, ids
        points.setdefault(ctx, {})[gen] = r

    rows = []
    for ctx in sorted(points):
        gens = sorted(points[ctx])
        row = {"ctx": ctx, "points": {}}
        for g in gens:
            p = points[ctx][g]
            row["points"][g] = {"dev_total_ms": p.get("dev_total_ms"),
                                "dev_avg_ms": p.get("dev_avg_ms"),
                                "dev_dropped": p.get("dev_dropped"),
                                "wall_total_ms": p.get("wall_total_ms"),
                                "generated": p.get("generated_tokens"),
                                "n_saved": p.get("n_saved"),
                                "txthash": p.get("txthash"),
                                "decode": decode_stats(p)}
            row.setdefault("kv_chunks", p.get("kv_chunks"))

        # TPOT from the longest decode run available: more surviving ring
        # samples, and further from the prefill boundary.
        for g in reversed(gens):
            st = row["points"][g]["decode"]
            if st and points[ctx][g].get("dev_dropped") == 0:
                row["tpot_ms"] = st["median_ms"]
                row["tpot_sd_ms"] = st["sd_ms"]
                row["tpot_n"] = st["n"]
                row["tpot_gen"] = g
                # Prefill cost, from the untruncated accumulator.
                p = points[ctx][g]
                row["prefill_ms"] = (p["dev_total_ms"]
                                     - p["generated_tokens"] * st["median_ms"])
                break
        # Too few samples to trust a median.
        if row.get("tpot_n", 0) and row["tpot_n"] < 5:
            row["suspect"] = (f"only {row['tpot_n']} decode ring samples "
                              f"survived; median is fragile")

        # Consistency check across the two decode lengths at the same context:
        # the short run's continuation must be a prefix of the long run's.
        # Decode is greedy and deterministic, so anything else means the extra
        # decode steps perturbed earlier ones -- e.g. a KV-cache or paging bug
        # that only bites past some step count.
        if len(gens) >= 2:
            a = points[ctx][gens[0]].get("ids") or []
            b = points[ctx][gens[-1]].get("ids") or []
            k = min(len(a), len(b))
            if k and a[:k] != b[:k]:
                row["prefix_mismatch"] = next(
                    i for i in range(k) if a[i] != b[i])
        rows.append(row)

    hdr = (f"{'ctx':>8} {'TPOT ms/tok':>12} {'sd':>7} {'n':>4} "
           f"{'prefill ms':>11} {'chunks':>7}  {'idhash':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tp = f"{r['tpot_ms']:.3f}" if "tpot_ms" in r else "n/a"
        sd = f"{r['tpot_sd_ms']:.4f}" if "tpot_sd_ms" in r else "-"
        n = str(r.get("tpot_n", "-"))
        pf = f"{r['prefill_ms']:.0f}" if "prefill_ms" in r else "-"
        hi = max(r["points"]) if r["points"] else None
        hh = r["points"][hi]["txthash"] if hi is not None else None
        print(f"{r['ctx']:>8} {tp:>12} {sd:>7} {n:>4} {pf:>11} "
              f"{str(r.get('kv_chunks', '-')):>7}  {str(hh):>12}")
    for r in rows:
        if "suspect" in r:
            print(f"WARNING ctx {r['ctx']}: {r['suspect']}")
        if "prefix_mismatch" in r:
            print(f"WARNING ctx {r['ctx']}: the two decode lengths diverge at "
                  f"generated token {r['prefix_mismatch']}; greedy decode "
                  f"should be a prefix")

    for r in rows:                     # ids are for the checks, not the record
        for p in r.get("points", {}).values():
            p.pop("ids", None)
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {os.path.join(d, 'summary.json')}")

    dead = [r["ctx"] for r in rows
            if any(p.get("txthash") == "d41d8cd98f00"
                   for p in r["points"].values())]
    if dead:
        print(f"WARNING: empty generated text at ctx {dead} -- latency there "
              f"is meaningless")


if __name__ == "__main__":
    main()
