#!/usr/bin/env python
"""Correctness reference for every milestone (docs/design.md §5): HF eager bf16 on the GPU.

Runs the fixed 1,024-token prompt through deepseek-ai/DeepSeek-Coder-V2-Lite-Base, then 32 greedy decode
steps one token at a time (exactly what the megakernel does), and saves under results/m0/reference/:

  prompt_ids.json              the 1,024 prompt token ids (+ the 32 generated ids, prefill argmax = token 1024)
  kv_cache_prefill.safetensors layer{L}.K [16,1024,192] / layer{L}.V [16,1024,128] bf16 — the DynamicCache
                               after prefill, i.e. what the megakernel is handed (design §4.2, decision 1)
  kv_cache_final.safetensors   same after the 32 steps, T=1056 — reference rows for every appended position
  step0_hidden.safetensors     first decode token (position 1024): embed [2048], layer{L}.out [2048] for L=0..26
                               (residual stream after each decoder layer, pre final-norm), final_norm [2048]
  logits.safetensors           logits fp32 [32,102400], one row per decode step, teacher-forced on HF's own
                               greedy tokens (a greedy run IS its own teacher-forced sequence)
  routing_steps.safetensors    gate_logits fp32 [32,26,64]: the MoE router's fp32 gate logits per decode step and MoE
                               layer 1..26 (HF MoEGate's own F.linear on its own input, recomputed after the step's timing
                               from the captured tensors; its top-6 is asserted equal to HF's topk_idx) — the e2e
                               routing gate's reference (run_e2e.py teacher-forced, design §5.2 near-tie escape)
  meta.json                    command, git hash, dtype, timings, versions

Prompt: prompts/prompt_1024.txt. If absent it is generated deterministically (seed 0) as synthetic Python
source and truncated to exactly 1,024 tokens; the ids are the canonical artifact (prompts/prompt_1024_ids.json).

Usage: .venv/bin/python reference_prefill.py [--model models/DeepSeek-Coder-V2-Lite-Base] [--steps 32]
Time: ~1 min load (31 GB from disk) + seconds of compute. GPU only.
"""
import argparse, json, os, random, subprocess, sys, time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPT_TXT, PROMPT_IDS = (os.path.join(ROOT, "prompts", f) for f in ("prompt_1024.txt", "prompt_1024_ids.json"))
P = 1024


def synth_prompt(seed=0, n_funcs=400):
    """Deterministic pseudo-Python: enough tokens (>1024) for the tokenizer to cut at exactly 1,024."""
    rng = random.Random(seed)
    names = ["alpha", "beta", "gamma", "delta", "count", "total", "index", "value", "score", "weight"]
    ops = ["+", "-", "*", "//", "%"]
    out = ["# synthetic reference prompt, seed %d\nimport math\n\n" % seed]
    for i in range(n_funcs):
        a, b, c = rng.sample(names, 3)
        out.append(
            f"def f{i}({a}, {b}):\n"
            f"    {c} = {a} {rng.choice(ops)} {b} {rng.choice(ops)} {rng.randint(1, 99)}\n"
            f"    if {c} > {rng.randint(0, 500)}:\n"
            f"        return {c} - {rng.randint(1, 9)}\n"
            f"    return math.floor({c} / {rng.randint(1, 9)})\n\n"
        )
    return "".join(out)


def prompt_ids(tok):
    if not os.path.exists(PROMPT_TXT):
        os.makedirs(os.path.dirname(PROMPT_TXT), exist_ok=True)
        open(PROMPT_TXT, "w").write(synth_prompt())
    ids = tok(open(PROMPT_TXT).read(), add_special_tokens=True)["input_ids"]
    assert len(ids) >= P, f"prompt has {len(ids)} tokens, need >= {P}"
    ids = ids[:P]
    if os.path.exists(PROMPT_IDS):
        old = json.load(open(PROMPT_IDS))
        assert old == ids, "prompts/prompt_1024_ids.json no longer matches prompt_1024.txt — delete one"
    else:
        json.dump(ids, open(PROMPT_IDS, "w"))
    return ids


def cache_tensors(cache, nlayers):
    d = {}
    for L in range(nlayers):
        d[f"layer{L}.K"] = cache.key_cache[L][0].contiguous().cpu()   # [16,T,192]
        d[f"layer{L}.V"] = cache.value_cache[L][0].contiguous().cpu()  # [16,T,128]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "DeepSeek-Coder-V2-Lite-Base"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "m0", "reference"))
    ap.add_argument("--steps", type=int, default=32)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    assert torch.cuda.is_available(), "GPU required (design §5.1: the reference is HF eager bf16 on the same MI300X)"
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
    ).cuda().eval()
    cfg = model.config
    nl = cfg.num_hidden_layers
    t_load = time.time() - t0
    ids = prompt_ids(tok)

    # hooks: residual stream after each decoder layer + embedding + final norm (captured on every forward,
    # we keep the ones from decode step 0)
    cap = {}
    layers = model.model.layers
    hooks = [layers[L].register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(f"layer{L}.out", o[0][0, -1].clone())) for L in range(nl)]
    hooks.append(model.model.embed_tokens.register_forward_hook(lambda m, i, o: cap.__setitem__("embed", o[0, -1].clone())))
    hooks.append(model.model.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("final_norm", o[0, -1].clone())))
    # router: keep references to each MoE gate's input and topk_idx (no launch inside the timed step); logits after it
    moe = list(range(cfg.first_k_dense_replace, nl))
    gin = {}
    hooks += [layers[L].mlp.gate.register_forward_hook(lambda m, i, o, L=L: gin.__setitem__(L, (i[0], o[0]))) for L in moe]

    with torch.no_grad():
        x = torch.tensor([ids], device="cuda")
        cache = DynamicCache()
        t1 = time.time()
        out = model(input_ids=x, attention_mask=torch.ones_like(x), past_key_values=cache, use_cache=True)
        torch.cuda.synchronize(); t_prefill = time.time() - t1
        cache = out.past_key_values
        assert cache.get_seq_length() == P
        k = cache.key_cache[0]
        assert tuple(k.shape) == (1, cfg.num_attention_heads, P, cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), k.shape
        save_file(cache_tensors(cache, nl), os.path.join(args.out, "kv_cache_prefill.safetensors"))
        nxt = int(out.logits[0, -1].float().argmax())  # token 1024's input (design §9 "prompt token boundary")

        gen, logits, step0, t_steps, routing = [], [], None, [], []
        for s in range(args.steps):
            cap.clear()
            t2 = time.time()
            # DeepSeek's eager attention asserts a mask even for q_len 1, where transformers would pass None
            out = model(input_ids=torch.tensor([[nxt]], device="cuda"), attention_mask=torch.ones(1, P + s + 1, dtype=torch.long, device="cuda"),
                        past_key_values=cache, use_cache=True)
            torch.cuda.synchronize(); t_steps.append(time.time() - t2)
            cache = out.past_key_values
            lg = out.logits[0, -1].float()
            g = []
            for L in moe:  # MoEGate.forward: F.linear(h.view(-1, h).type(float32), weight.type(float32)), last row = this token
                h, idx = gin[L]
                gl = F.linear(h.view(-1, h.shape[-1]).type(torch.float32), layers[L].mlp.gate.weight.type(torch.float32), None)[-1]
                assert set(gl.softmax(-1).topk(cfg.num_experts_per_tok).indices.tolist()) == set(idx.view(-1, cfg.num_experts_per_tok)[-1].tolist()), (s, L)
                g.append(gl)
            routing.append(torch.stack(g).cpu())
            if s == 0:
                step0 = {k: v.cpu() for k, v in cap.items()}
            gen.append(nxt); logits.append(lg.cpu()); nxt = int(lg.argmax())
        assert cache.get_seq_length() == P + args.steps
        save_file(cache_tensors(cache, nl), os.path.join(args.out, "kv_cache_final.safetensors"))
    for h in hooks: h.remove()

    save_file(step0, os.path.join(args.out, "step0_hidden.safetensors"))
    save_file({"logits": torch.stack(logits)}, os.path.join(args.out, "logits.safetensors"))
    save_file({"gate_logits": torch.stack(routing)}, os.path.join(args.out, "routing_steps.safetensors"))  # [steps, 26, 64]
    tokens = gen + [nxt]  # gen[s] = input of step s (gen[0] = argmax of prefill); tokens[1:] = argmax of logits[s]
    assert all(int(logits[s].argmax()) == tokens[s + 1] for s in range(args.steps))
    json.dump({"prompt_ids": ids, "generated_ids": tokens[1:], "step_inputs": gen,
               "generated_text": tok.decode(tokens[1:])}, open(os.path.join(args.out, "prompt_ids.json"), "w"), indent=0)
    import transformers
    meta = {
        "cmd": " ".join(sys.argv), "cwd": os.getcwd(),
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "model": args.model, "dtype": "bfloat16", "attn": "eager", "P": P, "steps": args.steps,
        "torch": torch.__version__, "hip": torch.version.hip, "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(0),
        "t_load_s": t_load, "t_prefill_s": t_prefill, "t_step_ms": [round(t * 1e3, 2) for t in t_steps],
        "hf_eager_ms_per_token_median": round(sorted(t_steps)[len(t_steps) // 2] * 1e3, 2),
    }
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(f"tokens: {tokens[1:]}\ntext: {meta and json.load(open(os.path.join(args.out, 'prompt_ids.json')))['generated_text']!r}")
    print(f"load {t_load:.0f}s prefill {t_prefill*1e3:.0f}ms decode median {meta['hf_eager_ms_per_token_median']} ms/token -> {args.out}")


if __name__ == "__main__":
    main()
