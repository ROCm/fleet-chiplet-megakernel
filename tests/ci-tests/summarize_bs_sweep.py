#!/usr/bin/env python3
"""Summarize the batch-size sweep: TPOT and throughput vs concurrent requests.

Reads the fleet_b<B>_g<GEN>.log pairs written by run_gpt_oss_bs_sweep.sh and
differences the device clock across the two decode lengths.

WHY DIFFERENCE, AND WHY THE DEVICE CLOCK

demo.py's `Decode avg` misclassifies prefill iterations as decode at B>1 (it
assumes ceil(prompt_len / max_num_batched_tokens) prefill iterations, which only
holds at B=1), so it is not usable here. [FWD_PASS_TOTAL] is the untruncated
device-side accumulator over every iteration -- prefill included -- so a single
point is also a blend. Differencing two decode lengths at the same prompt
cancels prefill and all setup exactly:

    ms_per_step  = (total_ms(HI) - total_ms(LO)) / (HI - LO)
    ms_per_token = ms_per_step / B

`iters` is reported alongside because iters/step is the diagnostic that matters
at B>1: it is 1.00 when every megakernel pass emits a token for every request,
and above 1.00 when prefill contends with decode for the admission budget.

With --packing, summarizes the pack_t<T>.log files instead: one request, T
prompt tokens per iteration. That run is prefill-dominated rather than
decode-dominated, so it subtracts the decode tail out of the accumulator
instead of differencing two lengths:

    prefill_ms = total_ms - generated_tokens * median_decode_ms

The decode median comes from [FWD_PASS] samples with num_active_tokens == 1,
which is safe because the tail is the LAST thing to run and the ring buffer
truncates from the front. Reading prefill out of that same ring is NOT safe --
for a 512-token prompt it keeps only a trailing window.

usage:
  python3 tests/ci-tests/summarize_bs_sweep.py outputs/gpt_oss/bs_sweep
  python3 tests/ci-tests/summarize_bs_sweep.py --packing outputs/gpt_oss/bs_sweep
"""
import json
import os
import re
import statistics
import sys


def parse(path):
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as f:
        txt = f.read()
    m = re.search(
        r"\[FWD_PASS_TOTAL\] iters=(\d+) total_ms=([\d.]+) "
        r"avg_ms=([\d.]+) dropped=(\d+)",
        txt,
    )
    if not m:
        return None
    out = {
        "iters": int(m.group(1)),
        "total_ms": float(m.group(2)),
        "dropped": int(m.group(4)),
    }
    w = re.search(r"prompt_tokens=(\d+) generated_tokens=(\d+)", txt)
    if w:
        out["prompt_tokens"] = int(w.group(1))
        out["generated_tokens"] = int(w.group(2))
    dec = [
        float(t)
        for t, n in re.findall(
            r"\[FWD_PASS\] iter=\d+ time_ms=([\d.]+) num_active_tokens=(\d+)",
            txt,
        )
        if int(n) == 1
    ]
    if dec:
        out["decode_ms"] = statistics.median(dec)
    return out


def packing(d):
    ts = sorted(
        {
            int(m.group(1))
            for f in os.listdir(d)
            if (m := re.match(r"pack_t(\d+)\.log$", f))
        }
    )
    rows = []
    for t in ts:
        r = parse(f"{d}/pack_t{t}.log")
        if not r:
            print(f"T={t}: no [FWD_PASS_TOTAL], skipped")
            continue
        if r["dropped"]:
            print(f"T={t}: dropped iterations, results not trustworthy")
        if "decode_ms" not in r or "generated_tokens" not in r:
            print(f"T={t}: no decode tail to subtract, skipped")
            continue
        gen = r["generated_tokens"]
        pre_ms = r["total_ms"] - gen * r["decode_ms"]
        pre_it = r["iters"] - gen
        rows.append(
            {
                "tokens_per_iter": t,
                "prefill_ms": round(pre_ms, 1),
                "prefill_iters": pre_it,
                "ms_per_iter": round(pre_ms / pre_it, 3),
                "ms_per_prompt_token": round(pre_ms / r["prompt_tokens"], 3),
                "decode_ms": round(r["decode_ms"], 3),
            }
        )

    hdr = ["tokens_per_iter", "prefill_ms", "prefill_iters", "ms_per_iter",
           "ms_per_prompt_token", "decode_ms"]
    print("  ".join(f"{h:>19}" for h in hdr))
    for r in rows:
        print("  ".join(f"{r[h]:>19}" for h in hdr))

    out = f"{d}/packing_summary.json"
    with open(out, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\nwrote {out}")


def main():
    argv = sys.argv[1:]
    if "--packing" in argv:
        argv.remove("--packing")
        return packing(argv[0] if argv else "outputs/gpt_oss/bs_sweep")
    d = argv[0] if argv else "outputs/gpt_oss/bs_sweep"
    lo = int(os.environ.get("GEN_LO", 128))
    hi = int(os.environ.get("GEN_HI", 256))

    bs = sorted(
        {
            int(m.group(1))
            for f in os.listdir(d)
            if (m := re.match(r"fleet_b(\d+)_g\d+\.log$", f))
        }
    )

    rows = []
    for b in bs:
        a = parse(f"{d}/fleet_b{b}_g{lo}.log")
        z = parse(f"{d}/fleet_b{b}_g{hi}.log")
        if not a or not z:
            print(f"B={b}: missing a point, skipped")
            continue
        if a["dropped"] or z["dropped"]:
            print(f"B={b}: dropped iterations, results not trustworthy")
        ms_step = (z["total_ms"] - a["total_ms"]) / (hi - lo)
        iters_step = (z["iters"] - a["iters"]) / (hi - lo)
        rows.append(
            {
                "batch_size": b,
                "ms_per_step": round(ms_step, 3),
                "ms_per_token": round(ms_step / b, 3),
                "iters_per_step": round(iters_step, 3),
                "ms_per_iter": round(ms_step / iters_step, 3),
                "tokens_per_s": round(1000.0 * b / ms_step, 1),
            }
        )

    hdr = ["batch_size", "ms_per_step", "iters_per_step", "ms_per_iter",
           "ms_per_token", "tokens_per_s"]
    print("  ".join(f"{h:>14}" for h in hdr))
    for r in rows:
        print("  ".join(f"{r[h]:>14}" for h in hdr))

    out = f"{d}/summary.json"
    with open(out, "w") as f:
        json.dump({"gen_lo": lo, "gen_hi": hi, "rows": rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
