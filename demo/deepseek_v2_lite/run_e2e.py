#!/usr/bin/env python
"""M4 evidence: 32 greedy tokens end to end through the megakernel (embed -> 27 layers -> head -> argmax ->
prepare_next_batch loop, one launch), vs reference_prefill.py's HF eager run.

  run_e2e.py                       # -> results/m4/e2e.json (+ task_graph_rank0.json, test_rank0.cu)
  run_e2e.py --runs 5 --warmup 1 --event-timing   # per-iteration P50/P95 from MPK_EVENT_TIMING
  run_e2e.py --dry-run             # laptop: plan + arena shapes (same as run_layer.py --layers 0-26)

Two measurements (docs/design.md 5.2):
  free-running: one mpk() call decodes positions P..P+31, feeding its own argmax back (Fleet's prepare_next_batch);
                raw token match vs HF (count + first divergence), same tokens on every run (determinism), ms/token.
  teacher-forced: for each step s, HF's cache rows P..P+s-1 and HF's token s are forced and ONE iteration is run
                (runtime max_seq_length = P+s+2, set per launch); the bf16 logits vs HF's fp32 logits[s]: cosine >= 0.9995,
                argmax equal unless HF's top-2 margin < 0.1. Routing: after each step every MoE layer's top-6 ids and fp32 gate
                logits (arena L{l}.topk_idx / L{l}.logits) vs HF's (reference_prefill.py routing_steps.safetensors): ids exact,
                or the design §5.2 near-tie escape exactly as tests/test_sim.py::routing_check applies it (swap accepted only
                when HF's logit margin between the swapped experts < the measured gate-logit max|d| at that layer).
Runtime max_seq_length = P + steps + 1 so prepare_next_batch stops after exactly `steps` iterations and writes
tokens[P+1 .. P+steps] (persistent_kernel.cuh: stop when step + 2 >= max_seq_length; a write needs step+1 < max_seq_length).
Per-iteration times (--event-timing): P50/P95 over iterations >= 2 of each launch; iteration 1 (carries the schedulers'
pass 1) reported separately (decision 10e). Launch path + watchdog: run_layer.launch().
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_layer as RL  # noqa: E402
from run_layer import BF, DIMS, GATE, H, P, ROOT  # noqa: E402

NL = DIMS["n_layers"]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def load_all(model_dir, ref_dir, t_max, steps, dev):
    from safetensors import safe_open
    layers, sh = RL.load_layers(model_dir, 0, NL - 1, dev)
    cfg = RL.cfg_real()
    cos, sin = RL.rope_tables(cfg, t_max, dev)
    glob = {"embed_tokens": sh.get("model.embed_tokens.weight").to(dev), "lm_head": sh.get("lm_head.weight").to(dev),
            "model.norm": sh.get("model.norm.weight").to(dev), "rope_cos": cos, "rope_sin": sin}
    pre = safe_open(os.path.join(ref_dir, "kv_cache_prefill.safetensors"), "pt")
    fin = safe_open(os.path.join(ref_dir, "kv_cache_final.safetensors"), "pt")
    ck, cv, fk, fv = [], [], [], []
    for l in range(NL):
        K, V = torch.zeros(H, t_max, 192, dtype=BF, device=dev), torch.zeros(H, t_max, 128, dtype=BF, device=dev)
        K[:, :P], V[:, :P] = pre.get_tensor(f"layer{l}.K").to(dev), pre.get_tensor(f"layer{l}.V").to(dev)
        ck.append(K), cv.append(V)
        fk.append(fin.get_slice(f"layer{l}.K")[:, P:P + steps].to(dev))  # HF's rows for the 32 decoded positions
        fv.append(fin.get_slice(f"layer{l}.V")[:, P:P + steps].to(dev))
    ids = json.load(open(os.path.join(ref_dir, "prompt_ids.json")))
    logits = safe_open(os.path.join(ref_dir, "logits.safetensors"), "pt", device=dev).get_tensor("logits")
    meta = json.load(open(os.path.join(ref_dir, "meta.json")))
    rp = os.path.join(ref_dir, "routing_steps.safetensors")
    ref = dict(step_inputs=ids["step_inputs"], generated_ids=ids["generated_ids"], logits=logits, final_k=fk, final_v=fv,
               hf_eager_ms_per_token=meta.get("hf_eager_ms_per_token_median"),
               gate_logits=safe_open(rp, "pt", device=dev).get_tensor("gate_logits") if os.path.exists(rp) else None)  # [steps, 26, 64]
    return layers, glob, ck, cv, ref, cfg


def build(args, dev="cuda", perfetto=None, event_timing=False):
    """Everything up to a compiled, launchable kernel. Returns (mpk, arena, meta, ref, tg, plan)."""
    import mirage as mi
    os.makedirs(args.out, exist_ok=True)
    print(f"[e2e] loading 27 layers + head from {args.model}", flush=True)
    layers, glob, ck, cv, ref, cfg = load_all(args.model, args.ref, args.t_max, args.steps, dev)
    perf = torch.zeros(30000 * 1280, dtype=torch.uint64, device=dev) if perfetto else None
    if event_timing:
        os.environ["MPK_EVENT_TIMING"] = "1"
    t0 = time.time()
    mpk, arena, meta = RL.make_mpk(mi, layers, glob, ck, cv, args.t_max, P + args.steps + 1, first_k_dense=1, dev=dev, perfetto=perf)
    if perf is not None:
        mpk.trace_name = os.path.join(args.out, "e2e.trace")
    print(f"[e2e] graph built in {time.time() - t0:.1f}s; compiling (minutes)", flush=True)
    t0 = time.time()
    mpk.compile(output_dir=args.out)
    os.environ["MPK_EVENT_TIMING"] = "0"
    mpk.compile_s = time.time() - t0
    tg = json.load(open(os.path.join(args.out, "task_graph_rank0.json")))
    plan = RL.MG.layer_plan(args.t_max)
    mpk.watchdog = (tg, plan)  # labels for run_layer.watchdog_abort
    print(f"[e2e] compiled in {mpk.compile_s:.0f}s: {len(tg['all_tasks'])} tasks, {len(tg['all_events'])} events", flush=True)
    return mpk, arena, meta, ref, tg, plan


def free_running(mpk, meta, ref, steps, runs, warmup, tg=None, plan=None, event_timing=False):
    """(warmup + runs) launches of the 32-token loop; the runtime max_seq_length and a fresh event-timing buffer are
    set per launch (run_layer.launch)."""
    tok0, msl = ref["step_inputs"][0], P + steps + 1
    times, toks, iters, iters1, first = [], [], [], [], None
    for r in range(warmup + runs):
        ms = RL.launch(mpk, meta, P, tok0, max_seq_length=msl)
        gen = meta["tokens"][0, P + 1:P + steps + 1].tolist()
        if r >= warmup:
            times.append(ms)
            toks.append(gen)
            if event_timing:
                its = RL.event_timing(mpk, tg, plan)
                iters1 += [it["iteration_us"] for it in its[:1]]
                iters += [it["iteration_us"] for it in its[1:]]
                first = first or its
        print(f"  run {r}{' (warmup)' if r < warmup else ''}: {ms:.2f} ms = {ms / steps:.3f} ms/token; tokens {gen[:6]}...", flush=True)
    gen, want = toks[0], ref["generated_ids"][:steps]
    div = next((i for i, (x, y) in enumerate(zip(gen, want)) if x != y), None)
    res = dict(runs=runs, warmup=warmup, steps=steps, launch_ms=times, ms_per_token_launch=[t / steps for t in times],
               ms_per_token_p50=pct([t / steps for t in times], 0.5), ms_per_token_p95=pct([t / steps for t in times], 0.95),
               tokens=gen, ref_tokens=want, match_count=sum(x == y for x, y in zip(gen, want)), first_divergence=div,
               deterministic=all(t == gen for t in toks), hf_eager_ms_per_token=ref["hf_eager_ms_per_token"])
    if iters:
        res.update(iteration_us=iters, iteration_p50_us=pct(iters, 0.5), iteration_p95_us=pct(iters, 0.95),
                   iteration1_us=iters1, iteration1_p50_us=pct(iters1, 0.5), tokens_per_s=1e6 / pct(iters, 0.5),
                   iteration_method="MPK_EVENT_TIMING END_OF_TASK_GRAPH firings; P50/P95 over iterations >= 2 of each launch, iteration 1 separately (decision 10e)",
                   boundaries_first_iteration=first[0]["boundaries"] if first else None)
    return res


def routing_step(arena, gate_logits, topk=6):
    """One step, every MoE layer: our top-6 + fp32 gate logits vs HF's. Rule = tests/test_sim.py::routing_check (design
    §5.2, D5/D7h): ids equal, or a swap whose HF logit margin (max ref[dropped] - min ref[added]) < gate-logit max|d|."""
    fk, mism, errs = DIMS["first_k_dense"], [], []
    for j, r in enumerate(gate_logits):
        l = fk + j
        ours, ref_ids = arena[f"L{l}.topk_idx"].flatten().tolist(), r.float().softmax(-1).topk(topk).indices.tolist()
        err = (arena[f"L{l}.logits"].flatten().float() - r.float()).abs().max().item()
        errs.append(err)
        s, rr = set(ours), set(ref_ids)
        if s != rr:
            dropped, added = sorted(rr - s), sorted(s - rr)
            top = r.float().sort(descending=True).values
            margin = (r[dropped].float().max() - r[added].float().min()).item()
            mism.append(dict(layer=l, ref_ids=sorted(rr), ids=sorted(s), dropped=dropped, added=added, ref_margin_6_7=(top[5] - top[6]).item(),
                             swap_margin=margin, gate_logit_err=err, escape=margin < err))
    return dict(layers_exact=len(gate_logits) - len(mism), layers=len(gate_logits), gate_logit_err=errs, mismatches=mism,
                ok=all(m["escape"] for m in mism))


def teacher_forced(mpk, arena, meta, ref, steps, out_dir=None):
    """Step s: HF's cache rows P..P+s-1, HF's token s, one iteration; logits vs HF logits[s]. Our per-step logits, top-6 ids
    and gate logits -> <out_dir>/tf_capture.safetensors (routing_counterfactual.py reads them)."""
    out, cap = [], dict(logits=[], topk_idx=[], gate_logits=[])
    moe = range(DIMS["first_k_dense"], NL)
    for s in range(steps):
        for l in range(NL):
            arena[f"L{l}.K"][:, P:P + s] = ref["final_k"][l][:, :s]
            arena[f"L{l}.V"][:, P:P + s] = ref["final_v"][l][:, :s]
        ms = RL.launch(mpk, meta, P + s, ref["step_inputs"][s], max_seq_length=P + s + 2)
        lg, want = arena["logits"].flatten(), ref["logits"][s]
        cap["logits"].append(lg.float().cpu())
        cap["topk_idx"].append(torch.stack([arena[f"L{l}.topk_idx"].flatten().int() for l in moe]).cpu())
        cap["gate_logits"].append(torch.stack([arena[f"L{l}.logits"].flatten().float() for l in moe]).cpu())
        m = RL.metrics(lg, want, GATE["op"])
        top2 = want.topk(2).values
        m.update(step=s, ms=ms, argmax=int(meta["output_tokens"][0, 0]), ref_argmax=ref["generated_ids"][s],
                 ref_top2_margin=(top2[0] - top2[1]).item())
        m["ok"] = m["cosine"] >= GATE["logit_cos"] and (m["argmax"] == m["ref_argmax"] or m["ref_top2_margin"] < 0.1)
        if ref["gate_logits"] is not None:
            m["routing"] = routing_step(arena, ref["gate_logits"][s])
        out.append(m)
        rt = m.get("routing")
        print(f"  step {s:2d}: cos {m['cosine']:.6f} argmax {m['argmax']} ref {m['ref_argmax']} margin {m['ref_top2_margin']:.3f} {'ok' if m['ok'] else 'FAIL'}"
              + (f"; routing {rt['layers_exact']}/{rt['layers']} exact" + "".join(
                  f", L{x['layer']} {'NEAR-TIE ESCAPE' if x['escape'] else 'MISMATCH'} ref {x['ref_ids']} ours {x['ids']} swap margin {x['swap_margin']:.3e} "
                  f"(6th-7th {x['ref_margin_6_7']:.3e}) vs gate-logit err {x['gate_logit_err']:.3e}" for x in rt["mismatches"]) if rt else ""), flush=True)
    if out_dir:
        from safetensors.torch import save_file
        save_file({k: torch.stack(v) for k, v in cap.items()}, os.path.join(out_dir, "tf_capture.safetensors"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=RL.REF_DIR)
    ap.add_argument("--model", default=RL.MODEL_DIR)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--t-max", type=int, default=1056)
    ap.add_argument("--runs", type=int, default=5, help="timed free-running launches (design.md 9: >= 5 x 32 iterations)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "m4"))
    ap.add_argument("--event-timing", action="store_true", help="MPK_EVENT_TIMING: per-iteration and per-boundary times")
    ap.add_argument("--perfetto", action="store_true", help="Fleet on-device profiler (NOTE: it caps a launch at 1 iteration)")
    ap.add_argument("--no-teacher-forced", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.dry_run:
        RL.dry_run(0, NL - 1, args.t_max, P + args.steps + 1)
        return 0
    mpk, arena, meta, ref, tg, plan = build(args, perfetto=args.perfetto, event_timing=args.event_timing)
    print(f"[e2e] free-running {args.steps} tokens x {args.warmup + args.runs} launches", flush=True)
    fr = free_running(mpk, meta, ref, args.steps, args.runs, args.warmup, tg, plan, args.event_timing)
    print(f"[e2e] match {fr['match_count']}/{args.steps}, first divergence {fr['first_divergence']}, "
          f"P50 {fr['ms_per_token_p50']:.3f} ms/token (launch-level), deterministic {fr['deterministic']}"
          + (f"; iterations >= 2: P50 {fr['iteration_p50_us']:.0f} us P95 {fr['iteration_p95_us']:.0f} us, iteration 1 P50 {fr['iteration1_p50_us']:.0f} us"
             if "iteration_p50_us" in fr else ""), flush=True)
    tf = None if args.no_teacher_forced else teacher_forced(mpk, arena, meta, ref, args.steps, args.out)
    res = dict(cmd=" ".join(sys.argv), git=RL.git_hash(), date=time.strftime("%Y-%m-%dT%H:%M:%S"), steps=args.steps, pos=P,
               t_max=args.t_max, max_seq_length=P + args.steps + 1, device=torch.cuda.get_device_name(0),
               workers=mpk.num_workers, schedulers=mpk.num_local_schedulers, compile_s=mpk.compile_s,
               graph=dict(tasks=len(tg["all_tasks"]), events=len(tg["all_events"]), ops=len(plan)),
               gates=GATE, free_running=fr, teacher_forced=tf)
    res["pass"] = (tf is None or all(m["ok"] for m in tf)) and (fr["first_divergence"] is None or tf is not None and tf[fr["first_divergence"]]["ok"])
    if tf:  # every gate on its own (design §5.2); `pass` above is unchanged
        res["gates_pass"] = dict(tokens=fr["match_count"] == args.steps, argmax=all(m["argmax"] == m["ref_argmax"] or m["ref_top2_margin"] < 0.1 for m in tf),
                                 logit_cos=all(m["cosine"] >= GATE["logit_cos"] for m in tf),
                                 routing=all(m["routing"]["ok"] for m in tf) if "routing" in tf[0] else None)
        if "routing" in tf[0]:
            misses = [m["step"] for m in tf if m["cosine"] < GATE["logit_cos"]]
            esc = {m["step"]: [x["layer"] for x in m["routing"]["mismatches"] if x["escape"]] for m in tf if m["routing"]["mismatches"]}
            res["routing_summary"] = dict(
                steps_all_exact=sum(1 for m in tf if not m["routing"]["mismatches"]), steps_with_escape=esc,
                unexplained_mismatches=[dict(step=m["step"], **x) for m in tf for x in m["routing"]["mismatches"] if not x["escape"]],
                cosine_misses=misses,
                cosine_misses_explained={s: bool(esc.get(s)) and tf[s]["routing"]["ok"] for s in misses},
                escapes_without_cosine_miss=[s for s in esc if s not in misses],
                rule="explained = the step has >= 1 accepted near-tie escape and no rejected mismatch (tests/test_sim.py::routing_check)")
    json.dump(res, open(os.path.join(args.out, "e2e.json"), "w"), indent=1)
    print(f"[e2e] {'PASS' if res['pass'] else 'FAIL'} -> {args.out}/e2e.json")
    mpk.finalize()
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
