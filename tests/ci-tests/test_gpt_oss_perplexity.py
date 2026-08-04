"""GPT-OSS 120B perplexity: Mirage (MPK) vs the Torch reference.

Companion to test_gpt_oss_inference_output.py. That test gates on generated
tokens, which is a thresholded single-prompt proxy: it says "the two paths
agree" or "they don't" and nothing in between. This one gates on a continuous,
corpus-wide number, so a numerical regression that only bends the distribution
(rather than flipping the argmax) still shows up.

Both dumps come from demo.py with PPL_MODE=1, which loads a corpus as one long
prompt and runs prefill only. Inside the prompt the megakernel never overwrites
tokens[] (persistent_kernel.cuh guards the writeback on step+1 >= prompt_len),
so every position conditions on the reference prefix -- prefill IS teacher
forcing, which is exactly what perplexity needs.

Produce the inputs first (from repo root):
    PPL_MODE=1 python3 demo/gpt_oss/demo.py --model-path <m> \
        --max-num-batched-tokens 1 --ppl-max-tokens 512 \
        --ppl-out outputs/gpt_oss/torch_ppl.json
    PPL_MODE=1 python3 demo/gpt_oss/demo.py --model-path <m> --use-mirage \
        --max-num-batched-tokens 1 --ppl-max-tokens 512 \
        --ppl-out outputs/gpt_oss/mpk_ppl.json

Why a ceiling and not an exact value: MXFP4 decode is not bit-deterministic, so
the MPK number moves run to run. On a 512-token WikiText-2 slice the observed
spread was ~92-100 against a Torch reference of ~36. The defaults below leave
headroom above that spread while still catching a real blowup (a broken logits
sink lands in the hundreds or worse -- an early partition-offset bug in the sink
scored 64.5 against a Torch 8.15 on a 64-token slice).

Tunables (env):
    GPT_OSS_OUTPUT_DIR     dir holding the two json dumps (default outputs/gpt_oss)
    GPT_OSS_PPL_MAX        absolute MPK perplexity ceiling      (default 150)
    GPT_OSS_PPL_RATIO_MAX  max MPK/Torch perplexity ratio       (default 3.5)
    GPT_OSS_PPL_MIN_SCORED min scored positions to accept a run (default 64)
"""
import json
import math
import os
import pytest

DEFAULT_OUTPUT_DIR = os.environ.get(
    "GPT_OSS_OUTPUT_DIR", os.path.join("outputs", "gpt_oss")
)
TORCH_PPL = os.path.join(DEFAULT_OUTPUT_DIR, "torch_ppl.json")
MPK_PPL = os.path.join(DEFAULT_OUTPUT_DIR, "mpk_ppl.json")

PPL_MAX = float(os.environ.get("GPT_OSS_PPL_MAX", "150"))
PPL_RATIO_MAX = float(os.environ.get("GPT_OSS_PPL_RATIO_MAX", "3.5"))
MIN_SCORED = int(os.environ.get("GPT_OSS_PPL_MIN_SCORED", "64"))


def _load_ppl(path):
    if not os.path.exists(path):
        pytest.fail(
            f"Missing perplexity file: {path}. Run demo/gpt_oss/demo.py with "
            f"PPL_MODE=1 --max-num-batched-tokens 1 --ppl-out {path} "
            f"(and --use-mirage for the MPK dump) first."
        )
    with open(path) as f:
        data = json.load(f)
    ppl = data.get("perplexity")
    if not isinstance(ppl, (int, float)):
        pytest.fail(f"'perplexity' missing or not a number in {path}")
    if not math.isfinite(ppl):
        pytest.fail(
            f"Non-finite perplexity ({ppl}) in {path} -- the logits sink "
            f"produced inf/nan, which is a kernel bug, not a quality result."
        )
    if ppl < 1.0:
        pytest.fail(
            f"Perplexity {ppl} < 1 in {path}, which is impossible for a "
            f"cross-entropy over a real distribution. The scoring slice or the "
            f"vocab truncation is wrong."
        )
    return ppl, data


def _top1_agreement(a, b):
    """Fraction of positions where the two runs pick the same argmax token."""
    if not a or not b:
        return None
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if int(a[i]) == int(b[i])) / n


def test_gpt_oss_mpk_perplexity():
    torch_ppl, torch_meta = _load_ppl(TORCH_PPL)
    mpk_ppl, mpk_meta = _load_ppl(MPK_PPL)

    # The two numbers are only comparable if they scored the same text. A
    # silent corpus/slice mismatch would otherwise read as a quality delta.
    for key in ("corpus", "corpus_desc", "corpus_tokens", "scored_positions"):
        if torch_meta.get(key) != mpk_meta.get(key):
            pytest.fail(
                f"Dumps scored different slices: {key} is "
                f"{torch_meta.get(key)!r} (torch) vs {mpk_meta.get(key)!r} "
                f"(mpk). Re-run both with the same --ppl-corpus and "
                f"--ppl-max-tokens."
            )

    n_scored = mpk_meta.get("scored_positions", 0)
    if n_scored < MIN_SCORED:
        pytest.fail(
            f"Only {n_scored} scored positions (require >= {MIN_SCORED}); "
            f"too few for the mean NLL to be stable. Raise --ppl-max-tokens."
        )

    ratio = mpk_ppl / torch_ppl
    agreement = _top1_agreement(mpk_meta.get("top1"), torch_meta.get("top1"))

    # Reported, not gated. Top-1 agreement is the sharper discriminator of the
    # two, but it is also the noisier one across runs of the same build, so
    # gating on it would flake. Read it when the perplexity gate fires.
    agree_str = f"{agreement:.2%}" if agreement is not None else "n/a"
    print(
        f"[gpt-oss perplexity] scored {n_scored} positions of "
        f"{mpk_meta.get('corpus')}: mpk_ppl={mpk_ppl:.4f} "
        f"(ceiling {PPL_MAX}), torch_ppl={torch_ppl:.4f}, "
        f"ratio={ratio:.3f} (max {PPL_RATIO_MAX}), "
        f"mpk/torch top-1 agreement={agree_str}"
    )

    if mpk_ppl > PPL_MAX:
        pytest.fail(
            f"MPK perplexity {mpk_ppl:.4f} exceeds ceiling {PPL_MAX} on "
            f"{n_scored} positions (torch reference {torch_ppl:.4f}, "
            f"ratio {ratio:.3f}, top-1 agreement {agree_str}). Either a "
            f"numerical regression reached the LM head or the logits sink is "
            f"writing the wrong columns -- check the per-column coverage "
            f"diagnostic the demo prints in PPL_MODE."
        )

    if ratio > PPL_RATIO_MAX:
        pytest.fail(
            f"MPK perplexity {mpk_ppl:.4f} is {ratio:.3f}x the Torch "
            f"reference {torch_ppl:.4f} (max {PPL_RATIO_MAX}) over "
            f"{n_scored} positions, top-1 agreement {agree_str}. The absolute "
            f"ceiling passed, so the corpus may just be hard -- but the two "
            f"paths disagree by more than MXFP4 run-to-run drift explains."
        )
