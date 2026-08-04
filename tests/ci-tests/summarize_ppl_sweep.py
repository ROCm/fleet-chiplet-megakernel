#!/usr/bin/env python3
"""Tabulate a perplexity-vs-sequence-length sweep produced by
run_gpt_oss_ppl_sweep.sh. Writes summary.md and summary.csv next to the dumps.

Usage: summarize_ppl_sweep.py <sweep-dir>

Reads <mode>_<len>.json. Entropy is reported alongside perplexity on purpose:
numeric noise flattens the softmax, which *lowers* NLL at positions the model
gets wrong, so perplexity alone can make a noisier run look like a better one.
"""
import csv
import json
import os
import re
import sys


def main(d):
    rows = {}
    for fn in sorted(os.listdir(d)):
        m = re.fullmatch(r"(mpk|torch)_(\d+)\.json", fn)
        if not m:
            continue
        mode, n = m.group(1), int(m.group(2))
        with open(os.path.join(d, fn)) as f:
            j = json.load(f)
        ent = j.get("entropy") or []
        top1, tgts = j.get("top1") or [], j.get("targets") or []
        acc = (sum(1 for a, b in zip(top1, tgts) if int(a) == int(b))
               / len(top1)) if top1 else None
        rows.setdefault(n, {})[mode] = {
            "ppl": j.get("perplexity"),
            "nll": j.get("mean_nll"),
            "scored": j.get("scored_positions"),
            "ent": (sum(ent) / len(ent)) if ent else None,
            "acc": acc,
            "top1": top1,
            "per_pos": j.get("per_position_nll") or [],
        }

    if not rows:
        print(f"No <mode>_<len>.json dumps found in {d}")
        return 1

    # Each length scores a longer prefix of the same corpus, so the full-slice
    # perplexity moves with what text got included, not just with context
    # length -- WikiText-2 gets markedly harder around position 512 (6.3 nats
    # vs 3.6 for the first 511), which shows up as rising perplexity on the
    # *Torch reference* too. Restricting every run to the positions all runs
    # share removes that and leaves only the effect of longer context.
    shortest = min(rows)
    common = min(
        (len(v["per_pos"]) for r in rows.values() for v in r.values()
         if v["per_pos"]),
        default=0,
    )
    for r in rows.values():
        for v in r.values():
            pp = v["per_pos"][:common]
            v["ppl_common"] = (
                __import__("math").exp(sum(pp) / len(pp)) if pp else None
            )

    def f(x, spec="8.3f"):
        return format(x, spec) if isinstance(x, (int, float)) else "     n/a"

    out = []
    out.append(f"| seq len | scored | MPK ppl | Torch ppl | ratio | "
               f"MPK ppl@{common} | Torch ppl@{common} | "
               f"MPK ent | Torch ent | MPK acc | Torch acc | agree |")
    out.append("|--------:|-------:|--------:|----------:|------:|"
               "----------:|------------:|"
               "--------:|----------:|--------:|----------:|------:|")
    csv_rows = []
    for n in sorted(rows):
        r = rows[n]
        mk, tc = r.get("mpk", {}), r.get("torch", {})
        ratio = (mk.get("ppl") / tc["ppl"]
                 if mk.get("ppl") and tc.get("ppl") else None)
        agree = None
        if mk.get("top1") and tc.get("top1"):
            k = min(len(mk["top1"]), len(tc["top1"]))
            agree = sum(1 for i in range(k)
                        if int(mk["top1"][i]) == int(tc["top1"][i])) / k
        out.append(
            f"| {n:7d} | {mk.get('scored') or tc.get('scored') or 0:6d} "
            f"| {f(mk.get('ppl'))} | {f(tc.get('ppl'))} | {f(ratio, '6.3f')} "
            f"| {f(mk.get('ppl_common'), '10.3f')} "
            f"| {f(tc.get('ppl_common'), '12.3f')} "
            f"| {f(mk.get('ent'), '7.3f')} | {f(tc.get('ent'), '9.3f')} "
            f"| {f(mk.get('acc'), '7.4f')} | {f(tc.get('acc'), '9.4f')} "
            f"| {f(agree, '6.4f')} |"
        )
        csv_rows.append({
            "seq_len": n, "scored": mk.get("scored") or tc.get("scored"),
            "mpk_ppl": mk.get("ppl"), "torch_ppl": tc.get("ppl"),
            "ratio": ratio,
            "mpk_ppl_common_prefix": mk.get("ppl_common"),
            "torch_ppl_common_prefix": tc.get("ppl_common"),
            "mpk_entropy": mk.get("ent"),
            "torch_entropy": tc.get("ent"), "mpk_acc": mk.get("acc"),
            "torch_acc": tc.get("acc"), "top1_agreement": agree,
        })

    table = "\n".join(out)
    print("\nPerplexity vs sequence length (WikiText-2 prefix)\n")
    print(table)
    print("\nppl/ent = perplexity / mean predictive entropy (nats). "
          "acc = argmax vs corpus targets.\nagree = MPK and Torch pick the "
          "same argmax. Blank Torch cells = the reference\nattention "
          "([64,n,n] f32) does not fit at that length.")
    print(f"\nppl@{common} is the SAME first {common} positions in every run "
          f"-- read this column for the\neffect of context length. The plain "
          f"ppl column also moves with which text each\nslice includes "
          f"(WikiText-2 gets harder past ~512 tokens), which is why the Torch\n"
          f"reference rises too.")

    with open(os.path.join(d, "summary.md"), "w") as fh:
        fh.write("# GPT-OSS 120B perplexity vs sequence length\n\n"
                 "Corpus: WikiText-2 raw test, first N tokens. MPK is "
                 "prefill-only (teacher forced).\n\n" + table + "\n")
    with open(os.path.join(d, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nWrote {d}/summary.md and {d}/summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else "outputs/gpt_oss/ppl_sweep"))
