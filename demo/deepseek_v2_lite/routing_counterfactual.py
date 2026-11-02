#!/usr/bin/env python
"""Is a teacher-forced logit dip of the megakernel caused by its expert routing? (docs/gpu-log.md session 5)

For every step s, HF eager bf16 on the GPU with HF's own cache rows 0..P+s-1 (kv_cache_final) and HF's token s, one decode
step, twice:
  control  HF's own routing: must reproduce results/m0/reference/logits.safetensors[s] (proves the rebuilt cache is exact)
  forced   every MoE gate returns the megakernel's top-6 ids of that step (results/m4/tf_capture.safetensors, run_e2e.py
           teacher-forced), weights = HF's own softmax scores at those ids; nothing else changes
  forced_w as forced, but the weights are ours too (softmax of our captured fp32 gate logits at our ids = what moe_topk
           computes), so the whole routing decision is the megakernel's and only the kernels' arithmetic differs
If routing explains a dip: cos(forced, megakernel) is back at the level of the clean steps while cos(forced, HF) reproduces the dip.
  routing_counterfactual.py            -> results/m4/routing_counterfactual.json
"""
import argparse, json, os, sys, time

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = os.path.dirname(os.path.abspath(__file__))
P = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "DeepSeek-Coder-V2-Lite-Base"))
    ap.add_argument("--ref", default=os.path.join(ROOT, "results", "m0", "reference"))
    ap.add_argument("--capture", default=os.path.join(ROOT, "results", "m4", "tf_capture.safetensors"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "m4", "routing_counterfactual.json"))
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, DynamicCache
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 attn_implementation="eager").cuda().eval()
    cfg, layers = model.config, model.model.layers
    moe = list(range(cfg.first_k_dense_replace, cfg.num_hidden_layers))
    ref_logits = safe_open(os.path.join(args.ref, "logits.safetensors"), "pt").get_tensor("logits")
    fin = safe_open(os.path.join(args.ref, "kv_cache_final.safetensors"), "pt", device="cuda")
    kv = [(fin.get_tensor(f"layer{L}.K")[None], fin.get_tensor(f"layer{L}.V")[None]) for L in range(cfg.num_hidden_layers)]
    inputs = json.load(open(os.path.join(args.ref, "prompt_ids.json")))["step_inputs"]
    capf = safe_open(args.capture, "pt")
    ours, our_ids, our_gl = capf.get_tensor("logits"), capf.get_tensor("topk_idx"), capf.get_tensor("gate_logits")  # [S,V], [S,26,6], [S,26,64]
    force, force_w = {}, {}  # layer -> ids LongTensor [1, 6] (and weights [1, 6]) while a forced pass runs

    def gate_hook(m, i, o, L):
        if L not in force:
            return None
        h = i[0].view(-1, i[0].shape[-1])
        scores = F.linear(h.type(torch.float32), m.weight.type(torch.float32), None).softmax(-1, dtype=torch.float32)
        return force[L], force_w[L] if L in force_w else scores.gather(1, force[L]), o[2]

    hooks = [layers[L].mlp.gate.register_forward_hook(lambda m, i, o, L=L: gate_hook(m, i, o, L)) for L in moe]
    cos = lambda a, b: F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()
    rows = []
    with torch.no_grad():
        for s in range(ours.shape[0]):
            out = {}
            for mode in ("control", "forced", "forced_w"):
                cache = DynamicCache()
                for L, (K, V) in enumerate(kv):
                    cache.update(K[:, :, :P + s], V[:, :, :P + s], L)
                force.clear(), force_w.clear()
                if mode != "control":
                    force.update({L: our_ids[s, j].long().view(1, -1).cuda() for j, L in enumerate(moe)})
                if mode == "forced_w":
                    force_w.update({L: our_gl[s, j].float().softmax(-1)[our_ids[s, j].long()].view(1, -1).cuda() for j, L in enumerate(moe)})
                lg = model(input_ids=torch.tensor([[inputs[s]]], device="cuda"), past_key_values=cache, use_cache=True,
                           attention_mask=torch.ones(1, P + s + 1, dtype=torch.long, device="cuda")).logits[0, -1].float().cpu()
                out[mode] = lg
            ref, c, f, fw, o = ref_logits[s], out["control"], out["forced"], out["forced_w"], ours[s]
            rows.append(dict(step=s, control_equal_ref=bool(torch.equal(c, ref)), control_max_abs=(c - ref).abs().max().item(),
                             cos_ours_ref=cos(o, ref), cos_forced_ref=cos(f, ref), cos_ours_forced=cos(o, f), cos_forced_w_ref=cos(fw, ref), cos_ours_forced_w=cos(o, fw),
                             argmax_ref=int(ref.argmax()), argmax_forced=int(f.argmax()), argmax_ours=int(o.argmax())))
            r = rows[-1]
            print(f"{time.strftime('%H:%M:%S')} step {s:2d}: control==ref {r['control_equal_ref']} (max|d| {r['control_max_abs']:.2e}); "
                  f"cos ours-ref {r['cos_ours_ref']:.6f}  forced-ref {r['cos_forced_ref']:.6f}  ours-forced {r['cos_ours_forced']:.6f}  "
                  f"forced_w-ref {r['cos_forced_w_ref']:.6f}  ours-forced_w {r['cos_ours_forced_w']:.6f}", flush=True)
    for h in hooks:
        h.remove()
    json.dump(dict(cmd=" ".join(sys.argv), date=time.strftime("%Y-%m-%dT%H:%M:%S"), capture=args.capture, steps=rows,
                   method=__doc__.strip()), open(args.out, "w"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
