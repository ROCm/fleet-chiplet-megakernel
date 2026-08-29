#!/usr/bin/env python3
"""Token-level determinism and PyTorch-agreement check for a Fleet MK demo.

Usage:
  ./check_determinism.py --demo demo_gpt_oss_120b.py \\
      --model-path /home/claudeuser/models/gpt-oss-120b --runs 3

  # also compare against the PyTorch path (much slower, one extra run)
  ./check_determinism.py --demo demo_gpt_oss_120b.py \\
      --model-path /home/claudeuser/models/gpt-oss-120b --runs 3 --vs-pytorch

WHAT COUNTS AS A FAILURE -- READ THIS BEFORE "FIXING" A DIVERGENCE
------------------------------------------------------------------
Bit-exact reproducibility is NOT the bar for the MoE path, and treating it as
one sends you chasing a race that does not exist.

MoE W2 accumulates 4 experts into workspace_f32 with float atomicAdd
(gang_moe_fused_mxfp4_mi300.cuh:1438-1442, 1627-1631). Float addition is not
associative and expert arrival order varies run to run, so near-ties in the
argmax flip. Divergent-but-coherent prose is the EXPECTED steady state.

Verified 2026-08-04: mirage, a different engine on the same weights, diverges
between two runs at char 169 -- EARLIER than Fleet MK's char 315. Reproduce with:

  MIRAGE_HOME=/home/claudeuser/mirage \\
  LD_LIBRARY_PATH=/home/claudeuser/mirage/python/mirage:$LD_LIBRARY_PATH \\
  HIP_VISIBLE_DEVICES=0 USE_FP8_ACT=1 python3 demo/gpt_oss/demo.py --use-mirage \\
    --model-path /home/claudeuser/models/gpt-oss-120b \\
    --prompt "Tell me the history of america" --max-seq-length 120

Divergence POSITION is not the discriminator either. A near-tie can flip the
5th logged token just as easily as the 315th character -- observed 2026-08-04,
where one run gave 31064 (' asks') and two gave 10648 (' wants') at iteration 5,
and 31064 is exactly what the PyTorch reference produces. Both are correct
decodes of a near-tie; neither is corruption.

What this script fails on is the CORRUPTION SIGNATURE from the embedding
barrier race (recorded in the project wiki -- search its articles for
"embedding barrier race"), which is qualitative, not positional:

  1. iteration 1 emits the PREFILL token (200005) instead of 35644 -- means
     239 workers read the previous token's embedding, i.e. a device-wide write
     guarded only by a workgroup-scoped __syncthreads.
  2. doubled words in the prose ("to to", "of of") -- same root cause.

Divergence in token identity or text is reported as INFO.

Qwen3-8B dense has no MoE float atomics and IS bit-reproducible, so for that
demo any text divergence is a genuine finding -- pass --strict-text to enforce.
"""

import argparse
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))

# The demos print one line per decoded token for the first few, but the full
# sequence only appears in the decoded text block. Parse the per-iter lines --
# they carry the token IDs, which is what we want to compare (text can be
# identical while IDs differ, and vice versa, through tokenizer quirks).
ITER_RE = re.compile(r"^\s*iter (\d+): next_token=(\d+)")

# The token the PREFILL emits. Seeing it again at decode iteration 1 is the
# embedding-race signature: the other 239 workers read the stale residual.
PREFILL_TOKEN = 200005

DOUBLED_RE = re.compile(r"\b(\w+) \1\b")


def corruption_signature(toks, body):
    """Qualitative corruption checks. Empty list = clean.

    Deliberately does NOT look at run-to-run divergence -- see module docstring.
    """
    bad = []
    if toks and toks[0] == PREFILL_TOKEN:
        bad.append(f"iteration 1 re-emitted the prefill token {PREFILL_TOKEN} "
                   f"(embedding visible to only one worker)")
    dbl = DOUBLED_RE.findall(body)
    # "that that" and "had had" are legitimate English; require a repeat that is
    # not one of the handful of words that genuinely double.
    dbl = [w for w in dbl if w.lower() not in {"that", "had", "the"}]
    if dbl:
        bad.append(f"doubled words in output: {sorted(set(dbl))[:5]}")
    return bad


def run_once(demo, model_path, max_seq, extra, log_path):
    cmd = [sys.executable, demo,
           "--model-path", model_path,
           "--prompt", "Tell me the history of america",
           "--max-seq-length", str(max_seq)] + extra
    env = dict(os.environ, HIP_VISIBLE_DEVICES="0")
    with open(log_path, "w") as f:
        subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                       timeout=280, env=env)
    with open(log_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    toks = [int(m.group(2)) for m in
            (ITER_RE.match(ln) for ln in text.split("\n")) if m]
    # The demos only log the first few iters, so the token list alone would miss
    # a late divergence -- which is exactly the interesting case. The decoded
    # text block between the two ==== rules covers the whole sequence.
    body = ""
    parts = text.split("=" * 60)
    if len(parts) >= 3:
        body = parts[1]
    return toks, body


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--max-seq-length", type=int, default=150)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--vs-pytorch", action="store_true",
                    help="Also run --pytorch-only and compare against it.")
    ap.add_argument("--strict-text", action="store_true",
                    help="Fail on full-text divergence too. Correct for dense "
                         "models (Qwen3-8B); WRONG for GPT-OSS MoE, whose float "
                         "atomicAdd accumulation is legitimately non-associative "
                         "-- see the module docstring.")
    args = ap.parse_args()

    seqs, bodies = [], []
    for i in range(args.runs):
        toks, body = run_once(args.demo, args.model_path, args.max_seq_length,
                              [], f"/tmp/det_{i}.log")
        print(f"run {i}: {len(toks)} tokens logged, first 8 = {toks[:8]}")
        seqs.append(toks)
        bodies.append(body)

    ok = True

    # The real gate: qualitative corruption, checked per run independently.
    for i, (toks, body) in enumerate(zip(seqs, bodies)):
        bad = corruption_signature(toks, body)
        if bad:
            ok = False
            for b in bad:
                print(f"run {i}: CORRUPTION -- {b}")

    for i in range(1, len(seqs)):
        d = first_divergence(seqs[0], seqs[i])
        if d is not None:
            print(f"run 0 vs run {i}: token diverges (INFO) at logged index {d} "
                  f"({seqs[0][d] if d < len(seqs[0]) else '-'} vs "
                  f"{seqs[i][d] if d < len(seqs[i]) else '-'})")
            if args.strict_text:
                ok = False
            continue
        # Logged tokens agree. Full-text divergence past that point is benign
        # under float-atomic MoE accumulation -- report it, but only fail if
        # the caller asserts this model should be bit-reproducible.
        if bodies[0] == bodies[i]:
            print(f"run 0 vs run {i}: IDENTICAL (logged tokens + full text)")
        else:
            c = first_divergence(bodies[0], bodies[i])
            tag = "TEXT DIVERGES" if args.strict_text else "text diverges (INFO)"
            print(f"run 0 vs run {i}: {tag} at char {c}")
            print(f"    run0: ...{bodies[0][max(0,c-60):c+60]!r}")
            print(f"    run{i}: ...{bodies[i][max(0,c-60):c+60]!r}")
            if args.strict_text:
                ok = False

    if args.vs_pytorch:
        pt, _ = run_once(args.demo, args.model_path, args.max_seq_length,
                         ["--pytorch-only"], "/tmp/det_pytorch.log")
        print(f"pytorch: {len(pt)} tokens logged, first 8 = {pt[:8]}")
        d = first_divergence(seqs[0], pt)
        if d is None:
            print(f"fleet_mk vs pytorch: IDENTICAL ({len(pt)} tokens)")
        else:
            print(f"fleet_mk vs pytorch: DIVERGES at index {d} "
                  f"({seqs[0][d] if d < len(seqs[0]) else '-'} vs "
                  f"{pt[d] if d < len(pt) else '-'}); "
                  f"{d} token prefix agrees")

    print()
    if ok:
        print("PASS (no corruption signature; divergence, if any, is benign)")
    else:
        print("FAIL (corruption signature present -- a real cross-worker race)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
