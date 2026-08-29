"""Compare the token streams of two or more harness runs.

Why this exists: after a change like a PAGE_SIZE switch, the question is always
"did the output change, and does the change mean anything?" -- and eyeballing two
lists of token ids answers neither. Divergence *position* is not by itself a
signal, because fleet_mk's MoE W2 accumulates 4 experts into workspace_f32 with a
float atomicAdd, whose summation order varies run to run. Two runs of the SAME
build can therefore diverge, and routinely do.

So the discriminator is not A-vs-B, it is:

    spread WITHIN a config  vs  spread BETWEEN configs

If run-to-run divergence inside one config is as early and as common as the
divergence between two configs, the configs are indistinguishable at this
sample size and the change is output-neutral. Only a between-config divergence
that is consistently EARLIER than the within-config spread is evidence the
change altered the computation.

Usage:
    python -m fleet_megakernel_vllm.compare_runs \\
        --group ps128 /tmp/a1.txt /tmp/a2.txt \\
        --group ps16  /tmp/b1.txt /tmp/b2.txt

Each file is a harness stdout capture; the token ids are read from the line
following the "generated token ids" banner.
"""

import argparse
import ast
import itertools
import os


BANNER = "generated token ids"


def read_token_ids(path):
    """Pull the token-id list out of a harness stdout capture.

    Returns None rather than raising when the banner is absent: a run that
    crashed or hung still has a log worth naming in the report, and a hard
    failure here would hide the other runs' results.
    """
    with open(path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()
    for i, line in enumerate(lines):
        if BANNER in line:
            for cand in lines[i + 1:]:
                cand = cand.strip()
                if cand.startswith("["):
                    try:
                        return ast.literal_eval(cand)
                    except (ValueError, SyntaxError):
                        return None
            return None
    return None


def first_divergence(a, b):
    """Index of the first differing token, or None if one is a prefix of the other.

    A prefix relationship is not a divergence: the runs agreed for as long as
    they were both asked to generate.
    """
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def summarize(label, streams):
    """Pairwise first-divergence for every pair within one group."""
    out = []
    for (na, a), (nb, b) in itertools.combinations(streams, 2):
        d = first_divergence(a, b)
        out.append((f"{os.path.basename(na)} vs {os.path.basename(nb)}", d))
    return out


def fmt(d, n):
    return f"identical ({n} tokens)" if d is None else f"diverge at {d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", action="append", metavar=("NAME", "FILE"),
                    required=True,
                    help="a config name followed by one or more run logs")
    args = ap.parse_args()

    groups = {}
    for g in args.group:
        name, files = g[0], g[1:]
        streams = []
        for f in files:
            ids = read_token_ids(f)
            if ids is None:
                print(f"  !! {f}: no token ids found (crashed run?)")
                continue
            streams.append((f, ids))
        groups[name] = streams

    n_tok = min((len(s) for st in groups.values() for _, s in st), default=0)

    print("\n=== within-config spread (same build, run to run) ===")
    print("    This is the noise floor. fleet_mk's MoE W2 sums 4 experts with a")
    print("    float atomicAdd, so summation order -- and thus the last bits of")
    print("    every logit -- varies between runs of one build.")
    for name, streams in groups.items():
        print(f"\n  [{name}]  {len(streams)} runs")
        for pair, d in summarize(name, streams):
            print(f"    {pair}: {fmt(d, n_tok)}")

    print("\n=== between-config spread ===")
    print("    Compare against the noise floor above, NOT against zero.")
    for (na, sa), (nb, sb) in itertools.combinations(groups.items(), 2):
        print(f"\n  [{na}] vs [{nb}]")
        for fa, a in sa:
            for fb, b in sb:
                d = first_divergence(a, b)
                print(f"    {os.path.basename(fa)} vs {os.path.basename(fb)}: "
                      f"{fmt(d, n_tok)}")

    print("\n=== first token (hard correctness gate) ===")
    print("    35644 == 'analysis'. This one IS deterministic: it is the first")
    print("    decode off a freshly prefilled cache, and every known corruption")
    print("    of the KV path has changed it.")
    for name, streams in groups.items():
        for f, s in streams:
            head = s[:3]
            ok = "OK" if 35644 in head else "MISMATCH"
            print(f"  [{name}] {os.path.basename(f)}: {head} {ok}")
    print()


if __name__ == "__main__":
    main()
