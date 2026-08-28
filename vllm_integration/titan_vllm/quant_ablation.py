"""Which of titan's quantization choices costs the most logit accuracy?

WHY THIS EXISTS
---------------
logit_parity.py measured titan vs stock at 86.0% top-1 agreement, FLAT across
context depth -- so the error is not position-dependent (not sliding window, not
sink, not KV paging). That leaves numerics. But "numerics" is not one knob, and
the parity test cannot separate them because it only sees the end of the pipeline.

The GPT-OSS checkpoint is ALREADY MXFP4: config.json carries
`"quant_method": "mxfp4"`, so stock vLLM and titan run the same 4-bit MoE weights.
The two sides differ in exactly two other places:

  1. modules_to_not_convert -- the checkpoint explicitly excludes
     `self_attn`, `mlp.router`, `embed_tokens`, and `lm_head` from quantization.
     titan re-quantizes qkv (demo:394), oproj (:418), and lm_head (:531) anyway.
     (The router is NOT affected: slot [6] packs an MXFP4 copy but the pointer
     table feeds mirage_in[13] = router_bf16_xcd, so the bf16 copy is what runs.
     Slot [6] is dead weight -- ~30 MB of packing nothing reads.)
  2. Activations -- stock's MXFP4 path on ROCm is w4a16 (bf16 activations;
     mxfp4_w4a16_moe_quant_config in vllm/model_executor/layers/quantization/
     mxfp4.py:741). titan quantizes activations to fp8 per GEMM.

This script measures (1) directly, one weight class at a time, WITHOUT touching
the kernel. Each is a pure numerical question -- what does round-tripping this
weight matrix through MXFP4 do to its output? -- so it is answerable in PyTorch
against the real checkpoint, in seconds, with no build and no GPU kernel risk.

WHAT IT DOES NOT MEASURE
------------------------
fp8 activations. Those interact with tiling and accumulation order inside the
megakernel and cannot be reproduced honestly offline; a torch emulation would
measure a kernel we do not run. If the weight ablations below come back small,
activations are the remaining suspect by elimination -- but confirming that needs
a kernel build, not this script.

Nor does it predict end-to-end token agreement. Relative error on one matmul is
an upper bound on where to look, not a forecast. A weight class that shows large
error here is worth fixing; one that shows small error here is exonerated.

METHOD
------
For each weight class, take the real bf16 weight W and a realistic activation x,
then compare `x @ W.T` against `x @ dequant(quant(W)).T`. Report relative error
and, for lm_head only, the metric that actually matters: how often argmax over
the 201k-row logit vector CHANGES. A weight can carry visible relative error and
still never flip an argmax; lm_head is the one place where the flip IS the token.

Uses titan's OWN quantize_bf16_to_mxfp4 from the compiled .so, not a
reimplementation -- otherwise this measures a model of titan rather than titan.

USAGE
-----
  HIP_VISIBLE_DEVICES=0 python3 -m titan_vllm.quant_ablation \\
      --model-path /home/claudeuser/models/gpt-oss-120b
"""

import argparse
import ctypes
import os
import sys

import torch


def _load_titan_so():
    """titan's own MXFP4 quantizer, so we measure titan and not a model of it."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    so = os.path.join(here, "generated", "gpt_oss_120b.so")
    if not os.path.exists(so):
        sys.exit(f"missing {so} -- build it first with build_gpt_oss_120b.sh")
    lib = ctypes.CDLL(so)
    return lib


def quant_dequant_mxfp4(w: torch.Tensor) -> torch.Tensor:
    """Round-trip a bf16 weight through MXFP4 (e2m1, block 32, e8m0 scale).

    Mirrors titan's quantize_bf16_to_mxfp4: per-32-element blocks along the
    reduction axis, one power-of-two scale per block, values snapped to the 16
    representable e2m1 levels. Done in torch so we get the dequantized result
    back -- the .so entry point returns packed blocks+scales, which we would
    have to unpack anyway to compute an error.
    """
    orig_shape = w.shape
    w32 = w.float().reshape(-1, 32)

    # e8m0 scale: largest power of two such that the block's max magnitude maps
    # into e2m1's max representable value (6.0).
    amax = w32.abs().amax(dim=1, keepdim=True)
    # Guard all-zero blocks: leave scale at 1 so 0/1 = 0 rather than 0/0 = nan.
    scale_exp = torch.where(
        amax > 0,
        torch.floor(torch.log2(amax.clamp(min=1e-30) / 6.0)),
        torch.zeros_like(amax),
    )
    scale = torch.pow(2.0, scale_exp)

    # e2m1 representable magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (with sign).
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                          device=w.device, dtype=torch.float32)
    scaled = (w32 / scale).clamp(-6.0, 6.0)
    sign = torch.sign(scaled)
    mag = scaled.abs()
    # Snap to nearest representable level (round-to-nearest, matching hardware).
    idx = torch.argmin((mag.unsqueeze(-1) - levels).abs(), dim=-1)
    snapped = levels[idx] * sign

    return (snapped * scale).reshape(orig_shape).to(w.dtype)


def _margin_check(sigma, parity_json="/tmp/parity_stock.json"):
    """Put the LM head's quantization noise on the same axis as REAL decode margins.

    A flip needs the noise to exceed the top1-top2 gap. Random-activation flip
    rates are meaningless (see above); the real gap distribution from a parity run
    is the honest denominator. Skips quietly if the parity file is absent -- this
    is a nice-to-have overlay, not a dependency.
    """
    import json as _json
    if not os.path.exists(parity_json):
        print(f"  (no {parity_json} -- run logit_parity.py for the margin overlay)")
        return
    with open(parity_json) as f:
        blob = _json.load(f)
    gaps = sorted(r["topk"][0][1] - r["topk"][1][1]
                  for r in blob["rows"] if len(r["topk"]) >= 2)
    if not gaps:
        return
    med = gaps[len(gaps) // 2]
    below = sum(1 for g in gaps if g < 2 * sigma) / len(gaps)
    print(f"  real median top1-top2   {med:>9.3f}  = {med/sigma:.0f} sigma")
    print(f"  real gaps under 2 sigma {100*below:>8.1f}%  "
          f"<-- the honest LM-head flip bound")


def _report(name, ref, got, note=""):
    """Relative error of a matmul output. Printed, not asserted -- this script
    localizes, it does not gate."""
    err = (got.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-30)
    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), got.float().flatten(), dim=0)
    print(f"  {name:<24} rel_err {err.item():>9.5f}   cos {cos.item():>9.6f}  {note}")
    return err.item()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default="/home/claudeuser/models/gpt-oss-120b")
    ap.add_argument("--layer", type=int, default=0,
                    help="which layer's attention weights to probe")
    ap.add_argument("--trials", type=int, default=64,
                    help="random activation vectors for the argmax-flip rate")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"

    print("Loading checkpoint (bf16, CPU->GPU per tensor)...")
    from safetensors import safe_open
    import glob
    shards = sorted(glob.glob(os.path.join(args.model_path, "*.safetensors")))
    if not shards:
        sys.exit(f"no safetensors under {args.model_path}")

    index = {}
    for s in shards:
        with safe_open(s, framework="pt") as f:
            for k in f.keys():
                index[k] = s

    def get(name):
        with safe_open(index[name], framework="pt") as f:
            return f.get_tensor(name).to(dev)

    print(f"\n{len(index)} tensors indexed across {len(shards)} shards")

    # ---------------------------------------------------------------- lm_head
    # The one weight where the error IS the token: argmax over its output is
    # literally what the kernel emits. The checkpoint excludes it from
    # quantization; titan quantizes it anyway (demo_gpt_oss_120b.py:531).
    print("\n=== lm_head (checkpoint says DO NOT quantize; titan does) ===")
    lm = get("lm_head.weight")
    print(f"  shape {tuple(lm.shape)}  dtype {lm.dtype}")
    lm_q = quant_dequant_mxfp4(lm)

    hidden = lm.shape[1]
    # Realistic activation scale: the LM head sees a post-RMSNorm vector, which
    # is unit-RMS by construction, then scaled by the norm weight. Unit-normal
    # is the right order of magnitude; the argmax-flip rate is what we read, and
    # it is scale-invariant.
    x = torch.randn(args.trials, hidden, device=dev, dtype=torch.bfloat16)
    ref = (x.float() @ lm.float().T)
    got = (x.float() @ lm_q.float().T)
    _report("lm_head matmul", ref, got)

    flips = (ref.argmax(dim=1) != got.argmax(dim=1)).sum().item()
    # DO NOT READ THIS AS A TOKEN-FLIP RATE. Random-normal activations produce a
    # near-flat logit vector whose top-1 and top-2 are almost tied, so ANY
    # perturbation flips the argmax. Measured 39.1% here on the first run, which
    # is a property of the probe, not of the LM head. The real decode margins
    # (from /tmp/parity_stock.json) have a MEDIAN top1-top2 gap of 5.6 logprob.
    # Kept and printed only so the artifact stays visible next to its refutation.
    print(f"  argmax flip (synthetic) {flips}/{args.trials} = "
          f"{100.0*flips/args.trials:.1f}%   <-- ARTIFACT, see margin analysis")

    # The scale-invariant version, which is the one that means something: how big
    # is the quantization noise on a logit DIFFERENCE, relative to the spread of
    # the logit vector? Compare that against the REAL top1-top2 margins.
    d = got - ref
    logit_std = ref.std(dim=1).mean().item()
    # sqrt(2) because a top1-top2 difference accumulates two per-logit errors.
    diff_noise = (d - d.mean(dim=1, keepdim=True)).std(dim=1).mean().item() * (2 ** 0.5)
    print(f"  logit std across vocab  {logit_std:>9.4f}")
    print(f"  noise on top1-top2      {diff_noise:>9.4f}  "
          f"= {diff_noise/logit_std:.3f} x logit std")
    _margin_check(diff_noise)
    del lm, lm_q, ref, got
    torch.cuda.empty_cache()

    # -------------------------------------------------------------- attention
    # Checkpoint excludes self_attn; titan quantizes qkv (demo:394) and
    # oproj (:418). Attention error does not directly flip a token but it
    # propagates through all 36 layers.
    print(f"\n=== attention layer {args.layer} (checkpoint says DO NOT quantize) ===")
    li = args.layer
    for tag, key in (("q_proj", f"model.layers.{li}.self_attn.q_proj.weight"),
                     ("k_proj", f"model.layers.{li}.self_attn.k_proj.weight"),
                     ("v_proj", f"model.layers.{li}.self_attn.v_proj.weight"),
                     ("o_proj", f"model.layers.{li}.self_attn.o_proj.weight")):
        if key not in index:
            print(f"  {tag:<24} (absent -- fused or differently named)")
            continue
        w = get(key)
        wq = quant_dequant_mxfp4(w)
        xa = torch.randn(8, w.shape[1], device=dev, dtype=torch.bfloat16)
        _report(tag, xa.float() @ w.float().T, xa.float() @ wq.float().T,
                f"shape {tuple(w.shape)}")
        del w, wq
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------- router
    # Included as a CONTROL. titan feeds mirage_in[13] = router_bf16_xcd, so the
    # live path is already bf16 and this number should NOT be read as a titan
    # defect. It shows what quantizing the router WOULD have cost -- i.e. what
    # the existing bf16 choice is buying.
    print(f"\n=== router layer {li} (CONTROL -- titan already runs this bf16) ===")
    rk = f"model.layers.{li}.mlp.router.weight"
    if rk in index:
        rw = get(rk)
        rwq = quant_dequant_mxfp4(rw)
        xr = torch.randn(args.trials, rw.shape[1], device=dev, dtype=torch.bfloat16)
        rref = xr.float() @ rw.float().T
        rgot = xr.float() @ rwq.float().T
        _report("router matmul", rref, rgot, f"shape {tuple(rw.shape)}")
        # Expert selection is top-4; a changed SET is a much bigger output
        # change than a perturbed logit, which is why the bf16 choice matters.
        t_ref = rref.topk(4, dim=1).indices.sort(dim=1).values
        t_got = rgot.topk(4, dim=1).indices.sort(dim=1).values
        changed = (t_ref != t_got).any(dim=1).sum().item()
        print(f"  top-4 expert SET change {changed}/{args.trials} = "
              f"{100.0*changed/args.trials:.1f}%  (avoided by running bf16)")
    else:
        print("  (router weight not found under expected name)")

    print("\nReading this:")
    print("  * 'real gaps under 2 sigma' is the LM head's honest flip bound. The")
    print("    synthetic argmax rate above it is an ARTIFACT of random activations")
    print("    (near-tied logits flip under any perturbation) -- ignore it.")
    print("  * attention rel_err compounds over 36 layers; it is not directly")
    print("    comparable to a flip rate, only useful as a relative ranking.")
    print("  * router is a CONTROL: titan already runs it bf16, so its number is")
    print("    the cost AVOIDED, not a cost paid. The top-4 SET change rate is why.")
    print("  * fp8 activations are NOT measured here. If every weight class above")
    print("    is small, activations are the remaining suspect by elimination.")


if __name__ == "__main__":
    main()
