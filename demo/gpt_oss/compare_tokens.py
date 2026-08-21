#!/usr/bin/env python3
"""Compare generated token streams between two runs of the correctness sweep.

Usage:
    python compare_tokens.py /tmp/gptoss_correctness 1gpu mp2

Why exact match is NOT the pass criterion under tensor parallelism
------------------------------------------------------------------
TP splits the o_proj reduction across ranks, so the summation order of the
attention output differs from single-GPU. In bf16 that changes low-order bits,
and under greedy decode a single near-tie logit flips -- after which the two
sequences are legitimately different text, both correct. A TP implementation
that matched single-GPU bit-for-bit would be the surprise. Observed prefix
agreement on correct 2-GPU runs ranged from 3 to 66 tokens.

What IS checked
---------------
A content keyword per prompt. This was chosen after two other criteria were
measured and rejected:

  * Exact token match -- fails on every correct TP run (see above).
  * Distinct-token ratio / repetition -- catches only one of the two observed
    failure modes. The broken AR#1 build degenerated repetitively ("The The
    This is a bit of. The 1.0.") and scores low, but the broken AR#2 build
    emitted high-entropy word salad ("isos-history Pers diagramHEL Ve Esc")
    which scores 0.89 distinct against 0.78 for correct text -- i.e. the metric
    ranks the garbage ABOVE the good output. It cannot be used as a gate.

A wrong collective corrupts the residual stream, and the model then never
reaches the answer. Checking that it does is crude but empirically separates
every good run from every broken one seen so far. Prefix agreement and
repetition are still printed as diagnostics -- just not used to pass or fail.
"""
import json
import os
import sys

# Substrings that any correct generation for the corresponding prompt in
# run_correctness_suite.sh must contain (case-insensitive). Keep index-aligned
# with PROMPTS in that script.
EXPECTED = {
    0: ["paris"],
    1: ["scatter"],
    2: ["prime"],
    3: ["stack", "queue"],
}


def load(d, tag, i):
    p = os.path.join(d, f"{tag}_p{i}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def repetition(ids):
    """(distinct ratio, longest run of an immediately repeating bigram)."""
    if not ids:
        return 0.0, 0
    longest = run = 0
    for k in range(2, len(ids)):
        if ids[k] == ids[k - 2]:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return len(set(ids)) / len(ids), longest


def check(entry, i):
    """(ok, missing_keywords) for one generation."""
    text = entry.get("text", "").lower()
    missing = [k for k in EXPECTED.get(i, []) if k not in text]
    return not missing, missing


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    d, ref_tag, cand_tag = sys.argv[1:4]

    i = 0
    n_cmp = n_exact = n_pass = 0
    while True:
        ref = load(d, ref_tag, i)
        cand = load(d, cand_tag, i)
        if ref is None and cand is None:
            break
        if ref is None or cand is None:
            print(f"prompt {i}: MISSING "
                  f"({ref_tag}={'ok' if ref else 'absent'}, "
                  f"{cand_tag}={'ok' if cand else 'absent'})")
            i += 1
            continue

        a, b = ref["token_ids"], cand["token_ids"]
        n_cmp += 1
        agree = 0
        for k in range(min(len(a), len(b))):
            if a[k] != b[k]:
                break
            agree += 1

        ref_ok, ref_missing = check(ref, i)
        ok, missing = check(cand, i)
        exact = agree == len(a) == len(b)
        n_exact += exact
        n_pass += ok

        ra, la = repetition(a)
        rb, lb = repetition(b)
        print(f"prompt {i}: {'PASS' if ok else 'FAIL'} "
              f"agree@{agree}/{min(len(a), len(b))}"
              f"{' (exact)' if exact else ''}")
        if missing:
            print(f"    {cand_tag} missing expected: {missing}")
        if not ref_ok:
            print(f"    WARNING: reference {ref_tag} also missing "
                  f"{ref_missing} -- check EXPECTED[{i}], not the candidate")
        print(f"    {ref_tag:>6}: distinct={ra:.2f} max_bigram_run={la} "
              f"n={len(a)}")
        print(f"    {cand_tag:>6}: distinct={rb:.2f} max_bigram_run={lb} "
              f"n={len(b)}")
        i += 1

    print(f"\n{n_pass}/{n_cmp} prompts pass; "
          f"{n_exact}/{n_cmp} bit-exact vs {ref_tag}")
    print("Exact match is not required under TP -- see module docstring.")
    return 0 if n_cmp and n_pass == n_cmp else 1


if __name__ == "__main__":
    sys.exit(main())
