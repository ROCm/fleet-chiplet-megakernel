"""GPT-OSS 120B per-stage layer comparison: MPK vs the matched Torch reference.

Companion to test_gpt_oss_perplexity.py. That test gates on one corpus-wide
number, which tells you *that* the model got worse and nothing about *where*.
This one gates per stage inside a decoder layer, so a numerical regression
names the op that caused it.

That distinction is not hypothetical. The split-KV LSE unit bug (fixed in
49f446b) showed up as 8.2% error at layer-0 attention while the layer output
was only 4.8% off and the headline perplexity moved between windows by more
than the bug did. A per-stage gate would have caught it on the commit that
introduced it; the perplexity gate did not.

HOW IT WORKS
------------
Both arms run demo.py with PPL_MODE=1, PPL_STAGE_DUMP=<path> and --max-layers L:

  * Reference arm: PPL_MXFP4_MATCH=1 PPL_FP8_ACT=1 registers forward hooks on
    each decoder layer and saves `L<i>.ln1/.attn/.ln2/.mlp/.out`.
  * MPK arm: --use-mirage saves the live megakernel scratch buffers.

The MPK buffers are single-token scratch reused every iteration, so after the
run they hold the LAST layer's values for the LAST token. `--max-layers L`
therefore pins the comparison to reference layer index L-1. We test L=1 and
L=2 deliberately: GPT-OSS alternates sliding_attention (even layers) and
full_attention (odd), so those two cover both attention kinds, which is why
two layers is the default here.

WHICH STAGES ARE GATED, AND WHY NOT MORE
----------------------------------------
Every pair below was established by cross-correlating every MPK buffer against
every reference stage, not by matching names:

    attn (L=1 only)      <-> L0.attn    cos 0.9998  -- see residual note
    rmsnorm_out_moe      <-> L<i>.ln2   cos 0.9990  (next best 0.5606)
    mlp_weighted_sum_out <-> L<i>.out   cos 0.9999  (next best 0.9155)

`rmsnorm_out_moe` is the loosest of the three, and by design: with the W13
prequant on (the default at bs=1) that buffer holds FP8 E4M3 + E8M0 block
scales, not BF16, so the compared values carry one extra quantization the
other two stages do not. demo.py decodes the layout before dumping -- reading
those bytes as BF16 gives cos 0.0, which reads as catastrophe rather than as
the format mismatch it is. L1.ln2 lands at 0.99901 against a 0.999 WARN line,
so this stage will flip to WARN on small FP changes; that is the format's
noise, not a regression. The stage that matters downstream is layer_out, which
is bit-identical with the prequant on and off (0.999871 both ways) -- the
prequant is exact by construction and only the handoff buffer differs.
MPK_W13_PREQUANT=0 restores the BF16 buffer and moves ln2 to 0.9994.

The attention pair needs care. MPK's `attn_proj_out` carries the residual
already added, so it is compared as `attn_proj_out - <layer input>`. At depth 1
the layer input is `embed_out` and the pair lands at 0.9998. At depth 2 the
layer input is layer 0's output, which MPK's single-token scratch no longer
holds, so the subtraction is wrong there (0.637) and the attention stage is
gated at depth 1 only. Depth 2 still covers full_attention via ln2/layer_out.

The others are deliberately NOT gated:

  * `attn_in`, `moe_gate_out`, `mlp_final` are all exactly zero -- dead
    buffers in the fused full-layer path. A threshold on a zero tensor gates
    nothing while looking like coverage.
  * `attn_out` is pre-o_proj and head-major (4096 wide); the reference's
    self_attn hook fires post-o_proj (2880). Different tensors.
  * `rmsnorm_out` has no exact counterpart among the reference hook points --
    its best match (0.956) is a *neighbouring* stage, so any threshold would
    measure hook placement rather than kernel correctness.

Extending coverage means adding hook points to the reference that line up with
MPK's actual buffers, not loosening a threshold until a mismatched pair passes.

THRESHOLDS, AND THE FAULT INJECTION THAT SET THEM
-------------------------------------------------
Measured on this box, 64-token WikiText-2 window. The BUG column is the same
build with MPK_LSE_LOG_BUG=1, which restores the pre-49f446b split-KV LSE unit
bug -- the real defect this suite exists to catch:

    depth  stage      fixed cos    bug cos   fixed rel   bug rel
    L=1    attn       0.999848   0.998377    1.75e-02   8.20e-02
    L=1    ln2        0.999823   0.997701    1.89e-02   7.29e-02
    L=1    layer_out  0.999958   0.999159    9.47e-03   4.79e-02
    L=2    ln2        0.999403   0.992377    3.48e-02   1.27e-01
    L=2    layer_out  0.999871   0.995638    1.62e-02   1.13e-01

Cosine alone is a poor discriminator here: at L=1 layer_out the bug only moves
it 0.99996 -> 0.99916, because the residual and the MoE dilute an attention
error that is 8.2% at the source. Relative RMSE separates 4-7x on every stage,
so BOTH are gated and rel_rmse is the tighter of the two.

Gates: cos >= 0.995, rel_rmse <= 0.055. The rel_rmse ceiling sits ~1.6x above
the worst fixed measurement (0.0348) and ~1.3x below the mildest bug
measurement (0.0479). That is a real but not generous margin -- if this flakes,
widen it with a measured number and re-run the fault injection, do not guess:

    MPK_LSE_LOG_BUG=1 tests/ci-tests/run_ci_tests_gpt_oss_layer_compare.sh

and confirm it FAILS. A gate nobody has watched go red is not known to work.

RUNNING IT
----------
    tests/ci-tests/run_ci_tests_gpt_oss_layer_compare.sh

or, if the dumps already exist:

    GPT_OSS_STAGE_DIR=/tmp/stagecmp pytest -q -s \
        tests/ci-tests/test_gpt_oss_layer_compare.py
"""
import os

import pytest

torch = pytest.importorskip("torch")

STAGE_DIR = os.environ.get("GPT_OSS_STAGE_DIR",
                           os.path.join("outputs", "gpt_oss", "stage"))

# Unpadded hidden size. MPK pads 2880 -> 2944; the tail is not a model output
# and comparing it would mix real error with padding noise.
HIDDEN = int(os.environ.get("GPT_OSS_HIDDEN", "2880"))

FAIL_COS = float(os.environ.get("GPT_OSS_STAGE_FAIL_COS", "0.995"))
WARN_COS = float(os.environ.get("GPT_OSS_STAGE_WARN_COS", "0.999"))
FAIL_REL = float(os.environ.get("GPT_OSS_STAGE_FAIL_REL", "0.055"))

# (stage label, reference hook suffix, MPK buffer name, residual to subtract)
#
# `residual` names an MPK buffer holding the layer input, which MPK's
# attn_proj_out has already accumulated but the reference's self_attn hook has
# not. None means compare directly.
STAGE_PAIRS = [
    ("ln2", "ln2", "rmsnorm_out_moe", None),
    ("layer_out", "out", "mlp_weighted_sum_out", None),
]

# Depth 1 only: at depth 2 the layer input is layer 0's output, which the
# single-token scratch no longer holds, so `embed_out` is the wrong residual.
STAGE_PAIRS_L1_ONLY = [
    ("attn", "attn", "attn_proj_out", "embed_out"),
]

# Buffers that are exactly zero in the fused path. Asserted dead so that if one
# ever comes alive we notice and can gate it, instead of it silently staying
# uncovered.
KNOWN_DEAD = ("attn_in", "moe_gate_out", "mlp_final")


def _load(path):
    if not os.path.exists(path):
        pytest.skip(
            f"Missing stage dump {path}. Generate both arms first with "
            f"tests/ci-tests/run_ci_tests_gpt_oss_layer_compare.sh"
        )
    return torch.load(path, map_location="cpu")


def _flat(t):
    return t.float().reshape(-1)[:HIDDEN]


def _cos(a, b):
    a, b = _flat(a), _flat(b)
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()


def _rel_rmse(ref, got):
    ref, got = _flat(ref), _flat(got)
    denom = ref.norm().item()
    if denom == 0.0:
        pytest.fail("Reference stage tensor is all zeros -- the hook did not "
                    "fire, so there is nothing to compare against.")
    return ((ref - got).norm() / denom).item()


@pytest.mark.parametrize("num_layers", [1, 2])
def test_gpt_oss_layer_stages(num_layers):
    """Compare MPK's last-layer buffers against the reference at that layer."""
    ref = _load(os.path.join(STAGE_DIR, f"ref_L{num_layers}.pt"))
    mpk = _load(os.path.join(STAGE_DIR, f"mpk_L{num_layers}.pt"))

    layer_idx = num_layers - 1
    kind = "sliding_attention" if layer_idx % 2 == 0 else "full_attention"

    pairs = list(STAGE_PAIRS)
    if num_layers == 1:
        pairs = STAGE_PAIRS_L1_ONLY + pairs

    failures = []
    for label, ref_suffix, mpk_name, residual in pairs:
        ref_key = f"L{layer_idx}.{ref_suffix}"
        if ref_key not in ref:
            pytest.fail(
                f"Reference dump has no {ref_key}; it holds {sorted(ref)}. "
                f"The hook set in demo.py changed -- update STAGE_PAIRS."
            )
        for name in (mpk_name, residual):
            if name is not None and name not in mpk:
                pytest.fail(
                    f"MPK dump has no {name}; it holds {sorted(mpk)}. The "
                    f"buffer was renamed or dropped from the PPL_STAGE_DUMP "
                    f"list in demo.py -- update STAGE_PAIRS."
                )

        got = mpk[mpk_name].float()
        if residual is not None:
            got = got - mpk[residual].float()

        cos = _cos(ref[ref_key], got)
        rel = _rel_rmse(ref[ref_key], got)
        bad = cos < FAIL_COS or rel > FAIL_REL
        status = "FAIL" if bad else ("OK" if cos >= WARN_COS else "WARN")
        note = f" (minus {residual})" if residual else ""
        print(
            f"[layer-compare] L{layer_idx} ({kind}) {status:4s} "
            f"{label:10s} cos={cos:.6f} rel_rmse={rel:.4e} "
            f"({ref_key} vs {mpk_name}{note})"
        )
        if bad:
            failures.append(
                f"{label}: cos={cos:.6f} (min {FAIL_COS}), "
                f"rel_rmse={rel:.4e} (max {FAIL_REL}), "
                f"{ref_key} vs {mpk_name}{note}"
            )

    if failures:
        pytest.fail(
            f"Layer {layer_idx} ({kind}) diverges from the matched Torch "
            f"reference:\n  " + "\n  ".join(failures) + "\n"
            f"Both arms ran the same tokens with MXFP4 weights and FP8 "
            f"activations, so this is kernel error, not quantization. "
            f"See docs/mpk/correctness-infra.md for how to localize it."
        )


def test_known_dead_buffers_still_dead():
    """The fused path leaves some scratch buffers unwritten.

    Gating on a zero tensor is coverage theater, so those stages are excluded
    from STAGE_PAIRS. This test pins that exclusion to a fact: if a buffer
    starts carrying real values, this fails and the pair should be added.
    """
    mpk = _load(os.path.join(STAGE_DIR, "mpk_L2.pt"))
    alive = [
        name for name in KNOWN_DEAD
        if name in mpk and mpk[name].float().reshape(-1).norm().item() > 0.0
    ]
    if alive:
        pytest.fail(
            f"Buffers {alive} are no longer zero in the fused path. They were "
            f"excluded from the per-stage gate *because* they were dead. Add "
            f"a reference hook point for each and put them in STAGE_PAIRS."
        )
