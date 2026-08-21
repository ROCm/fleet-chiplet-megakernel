#!/usr/bin/env python3
"""Plot decode TPOT vs context length for Fleet and vLLM on the same axes.

Inputs:
  --fleet   summary.json written by summarize_seqlen_sweep.py
  --vllm    directory of `vllm bench latency --output-json` files named
            vllm_<ctx>_o<outlen>.json, two output lengths per context

vLLM's `bench latency` reports END-TO-END latency for one batch, prefill
included, so TPOT is a two-point difference at the same input length:
(latency@HI - latency@LO) / (HI - LO). Both points pay the same prefill, so it
cancels. Fleet's number comes from the decode-phase [FWD_PASS] ring samples
directly -- see summarize_seqlen_sweep.py for why the median and not the mean.

usage:
  python3 tests/ci-tests/plot_seqlen_sweep.py \
      --fleet outputs/gpt_oss/seqlen_sweep/summary.json \
      --vllm  outputs/gpt_oss/vllm_sweep \
      --out   docs/img/seqlen_sweep.svg
"""
import argparse
import glob
import json
import os
import re


def load_fleet(path):
    d = json.load(open(path))
    rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
    return {int(r["ctx"]): float(r["tpot_ms"]) for r in rows}


def load_vllm(d, lo, hi):
    ctxs = sorted(
        {
            int(re.search(r"vllm_(\d+)_o", os.path.basename(f)).group(1))
            for f in glob.glob(os.path.join(d, "vllm_*.json"))
        }
    )
    out = {}
    for n in ctxs:
        try:
            a = json.load(open(f"{d}/vllm_{n}_o{lo}.json"))["avg_latency"]
            b = json.load(open(f"{d}/vllm_{n}_o{hi}.json"))["avg_latency"]
        except (OSError, KeyError):
            continue
        out[n] = (b - a) / (hi - lo) * 1000.0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fleet", required=True)
    p.add_argument("--vllm")
    p.add_argument("--vllm-lo", type=int, default=1)
    p.add_argument("--vllm-hi", type=int, default=129)
    p.add_argument("--out", default="docs/img/seqlen_sweep.svg")
    p.add_argument(
        "--title", default="GPT-OSS 120B decode latency vs context, batch 1, MI355X"
    )
    a = p.parse_args()

    fleet = load_fleet(a.fleet)
    vllm = load_vllm(a.vllm, a.vllm_lo, a.vllm_hi) if a.vllm else {}

    print(f"{'ctx':>7} {'Fleet ms':>9} {'vLLM ms':>9} {'speedup':>8}")
    for n in sorted(set(fleet) | set(vllm)):
        f, v = fleet.get(n), vllm.get(n)
        sp = f"{v / f:.2f}x" if f and v else "--"
        print(
            f"{n:>7} {f if f else float('nan'):9.3f} "
            f"{v if v else float('nan'):9.3f} {sp:>8}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if vllm:
        xs = sorted(vllm)
        ax.plot(xs, [vllm[x] for x in xs], "o-", color="#888888", label="vLLM")
    xs = sorted(fleet)
    ax.plot(xs, [fleet[x] for x in xs], "o-", color="#c0392b", label="Fleet (MPK)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    # log y: the two curves differ by ~40x at 32k, which on a linear axis
    # flattens the faster one into the x-axis and hides its own scaling.
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("context length (KV tokens)")
    ax.set_ylabel("decode latency (ms/token, log scale)")
    ax.set_title(a.title)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    for x in xs:
        if x in fleet and x in vllm:
            ax.annotate(
                f"{vllm[x] / fleet[x]:.1f}x",
                (x, (fleet[x] * vllm[x]) ** 0.5),
                ha="center",
                va="center",
                fontsize=8,
                color="#444444",
            )
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
