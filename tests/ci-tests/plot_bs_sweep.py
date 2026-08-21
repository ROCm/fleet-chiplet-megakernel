#!/usr/bin/env python3
"""Plot decode latency and throughput vs batch size (concurrent requests).

Input:
  --bs   summary.json written by summarize_bs_sweep.py

Two panels, because the batching story needs both. Per-token latency is what a
single user feels and it rises with B; aggregate throughput is what the GPU
delivers and it rises too. Plotting only one of them makes batching look either
purely good or purely bad.

usage:
  python3 tests/ci-tests/plot_bs_sweep.py \
      --bs  outputs/gpt_oss/bs_sweep/summary.json \
      --out docs/img/bs_sweep.svg
"""
import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", required=True)
    p.add_argument("--out", default="docs/img/bs_sweep.svg")
    p.add_argument(
        "--title",
        default=(
            "GPT-OSS 120B decode vs concurrent requests, short context, MI355X"
        ),
    )
    a = p.parse_args()

    d = json.load(open(a.bs))
    rows = sorted(d["rows"] if isinstance(d, dict) else d,
                  key=lambda r: r["batch_size"])
    xs = [r["batch_size"] for r in rows]
    tpot = [r["ms_per_token"] for r in rows]
    step = [r["ms_per_step"] for r in rows]
    tps = [r["tokens_per_s"] for r in rows]

    hdr = f"{'B':>4} {'ms/step':>9} {'ms/token':>9} {'tok/s':>9}"
    print(hdr)
    for r in rows:
        print(
            f"{r['batch_size']:>4} {r['ms_per_step']:9.3f} "
            f"{r['ms_per_token']:9.3f} {r['tokens_per_s']:9.1f}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    ax0.plot(xs, step, "o-", color="#888888", label="per iteration (all B)")
    ax0.plot(xs, tpot, "o-", color="#c0392b", label="per token (per request)")
    ax0.set_xscale("log", base=2)
    ax0.set_xticks(xs)
    ax0.set_xticklabels([str(x) for x in xs])
    ax0.set_ylim(bottom=0)
    ax0.set_xlabel("concurrent requests (B)")
    ax0.set_ylabel("decode latency (ms)")
    ax0.set_title("latency")
    ax0.grid(alpha=0.3, which="both")
    ax0.legend()

    ax1.plot(xs, tps, "o-", color="#c0392b", label="Fleet (MPK)")
    # Ideal scaling from B=1: what perfect batching would give, i.e. the same
    # per-iteration cost serving B times as many tokens. The gap to it is the
    # MoE's expert-weight traffic, which grows with the number of DISTINCT
    # experts the batch activates.
    ax1.plot(
        xs,
        [tps[0] * x for x in xs],
        "--",
        color="#bbbbbb",
        label="linear scaling from B=1",
    )
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(x) for x in xs])
    ax1.set_yscale("log")
    ax1.set_xlabel("concurrent requests (B)")
    ax1.set_ylabel("aggregate throughput (tokens/s, log scale)")
    ax1.set_title("throughput")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend()

    fig.suptitle(a.title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
