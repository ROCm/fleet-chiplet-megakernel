#!/usr/bin/env python
"""Latency, launches, traffic, bandwidth, occupancy for the launched decode graph (brief's metric list).

  bench.py                          # 20 warmup + 100 timed 32-token launches (each event-timed)
                                            #   -> results/bench/bench.json
  bench.py --rocprof results/profile/trace   # parse a rocprofv3/rocprof kernel trace (profile.sh trace)
                                            #   -> results/bench/rocprof.json: dispatches per launch, VGPR/SGPR/LDS/scratch,
                                            #      occupancy bounds
  bench.py --hf                     # HF eager baseline: kernel launches and ms per decode token (torch.profiler)
                                            #   -> results/bench/hf_baseline.json

How each number is obtained (also written into the JSON):
  latency      wall time of the launch call (one launch = `steps` iterations, run_layer.launch): P50/P95 over launches
               of ms/token. Per-iteration P50/P95: MPK_EVENT_TIMING timestamps of the END_OF_TASK_GRAPH event
               (s_memrealtime, 100 MHz), every timed launch x (steps - 1) samples over iterations >= 2; iteration 1 of
               each launch (carries the schedulers' pass 1) reported separately (decision 10e). The
               timing build adds one atomic + one clock read per event firing (~376/iteration); rerun with
               --no-event-timing to confirm the launch-level numbers are unchanged.
  launches     one mpk() call per `steps` tokens (1/32 per token); by construction that call is 3 kernel dispatches:
               prepare_kernel + worker_kernel + scheduler_kernel (persistent_kernel.cuh launch_persistent_kernel), i.e.
               3/steps per token; measured by the rocprof kernel trace (--rocprof). HF eager: --hf counts device kernels
               per decode step.
  traffic      bytes/token = weights a token reads (all attention/dense/gate/shared tensors + 6 of 64 routed experts per
               MoE layer + lm_head + norms + 1 embedding row) + KV cache read at T (27 x 16 x T x 320 x 2 B) + design.md 4.3's
               45 MB of descriptors/activations/counters. Counted from the attached tensors, not typed in.
  bandwidth    bytes/token / P50 time. rocprof --pmc FETCH_SIZE/WRITE_SIZE would measure it directly but serializes the
               worker/scheduler kernel pair (deadlock, see persistent_kernel.cuh init comment), so it is not used.
  occupancy    from the kernel trace's resource columns: LDS 57 KB/workgroup -> 1 workgroup (4 waves) per CU on 64 KB LDS,
               VGPR bound = 512 / vgprs (gfx942 unified arch+acc file, 8 waves/SIMD max), scratch > 0 = spills.
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_layer as RL  # noqa: E402
import run_e2e as E2E  # noqa: E402
from run_e2e import P, NL, pct  # noqa: E402

OUT = os.path.join(RL.ROOT, "results", "bench")
LAUNCHES_PER_CALL = 3  # prepare_kernel + worker_kernel + scheduler_kernel
META_BYTES = 45e6  # design.md 4.3: task descriptors + activations + counters


WEIGHTS = ("input_layernorm", "post_attention_layernorm", "kv_a_layernorm", "W_qkva", "kv_b_proj", "o_proj",
           "mlp.gate_up", "mlp.down_proj", "mlp.gate", "mlp.w13_s", "mlp.w2_s",
           "W_q", "W_kva")  # relayout_layer names read in full; W_q / W_kva = the DAG's views of W_qkva (session 8 counted 4.84 GB without them)


def bytes_per_token(arena, t):
    """Weights a token touches (from the attached tensors) + cache at T + metadata."""
    w, cache = 0, 0
    for n, x in arena.items():
        nb, tail = x.numel() * x.element_size(), n.split(".", 1)[-1]
        if tail in ("mlp.w13", "mlp.w2"):
            w += nb * 6 // 64  # superset graph: 58 of 64 experts skip without reading their weights
        elif n == "embed_tokens":
            w += x.shape[1] * x.element_size()  # one row
        elif n.endswith((".K", ".V")):
            cache += x.shape[0] * t * x.shape[2] * x.element_size()
        elif n in ("lm_head", "model_norm") or tail in WEIGHTS:
            w += nb
    return dict(weights=w, kv_cache=cache, metadata=META_BYTES, total=w + cache + META_BYTES, T=t)


def bench(args):
    mpk, arena, meta, ref, tg, plan = E2E.build(args, event_timing=not args.no_event_timing)
    tok0, steps, msl = ref["step_inputs"][0], args.steps, P + args.steps + 1
    print(f"[bench] {args.warmup} warmup + {args.runs} timed launches x {steps} tokens", flush=True)
    for _ in range(args.warmup):
        RL.launch(mpk, meta, P, tok0, max_seq_length=msl)
    times, it1, its, bounds = [], [], [], []
    for r in range(args.runs):
        times.append(RL.launch(mpk, meta, P, tok0, max_seq_length=msl))  # per launch: runtime bound + fresh timing buffer
        if not args.no_event_timing:
            ev = RL.event_timing(mpk, tg, plan)
            it1 += [it["iteration_us"] for it in ev[:1]]
            its += [it["iteration_us"] for it in ev[1:]]
            bounds += [b for it in ev[1:] for b in it["boundaries"]]
        if r % 20 == 19:
            print(f"  launch {r + 1}/{args.runs}: {times[-1]:.1f} ms" + (f", iteration P50 so far {pct(its, 0.5):.0f} us" if its else ""), flush=True)
    gen = meta["tokens"][0, P + 1:P + steps + 1].tolist()
    per_tok = [t / steps for t in times]
    res = dict(cmd=" ".join(sys.argv), knobs={k: RL.DIMS[k] for k in RL.KNOBS}, git=RL.git_hash(), date=time.strftime("%Y-%m-%dT%H:%M:%S"), device=torch.cuda.get_device_name(0),
               steps=steps, warmup=args.warmup, runs=args.runs, launch_ms=times,
               ms_per_token=dict(p50=pct(per_tok, 0.5), p95=pct(per_tok, 0.95), mean=statistics.fmean(per_tok), min=min(per_tok),
                                 method="wall ms of the launch call / steps, per launch (includes prepare_kernel and iteration 1)"),
               launches_per_call=LAUNCHES_PER_CALL, launches_per_token=LAUNCHES_PER_CALL / steps, mpk_calls_per_token=1 / steps,
               launches_method="one mpk() call per `steps` tokens; the call is prepare_kernel + worker_kernel + scheduler_kernel dispatches",
               tokens_match_ref=sum(x == y for x, y in zip(gen, ref["generated_ids"])),
               hf_eager_ms_per_token=ref["hf_eager_ms_per_token"], bytes_per_token=bytes_per_token(arena, args.t_max))
    if its:
        by = {}
        for b in bounds:
            k = f"{b['type']}" if b["layer"] < 0 else f"L{b['layer']}.{b['type']}"
            by.setdefault(k, []).append(b["us"])
        res["iteration_us"] = dict(samples=len(its), p50=pct(its, 0.5), p95=pct(its, 0.95), min=min(its), all=its,
                                   method="MPK_EVENT_TIMING: END_OF_TASK_GRAPH firings, s_memrealtime 100 MHz; iterations >= 2 of every timed launch (decision 10e)")
        res["iteration1_us"] = dict(samples=len(it1), p50=pct(it1, 0.5), p95=pct(it1, 0.95), all=it1,
                                    method="first iteration of each launch: carries the schedulers' pass 1 over all tasks")
        res["tokens_per_s"] = 1e6 / pct(its, 0.5)
        res["boundary_us_p50"] = {k: pct(v, 0.5) for k, v in by.items()}
        res["top_boundaries"] = sorted(res["boundary_us_p50"].items(), key=lambda kv: -kv[1])[:15]
    t_p50 = (res.get("iteration_us") or {}).get("p50", res["ms_per_token"]["p50"] * 1e3)
    res["achieved_bandwidth_TBps"] = res["bytes_per_token"]["total"] / (t_p50 * 1e-6) / 1e12
    res["achieved_bandwidth_method"] = "bytes_per_token.total / P50 iteration time (launch-level P50 if no event timing)"
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "bench.json"), "w"), indent=1)
    print(f"[bench] ms/token P50 {res['ms_per_token']['p50']:.3f} P95 {res['ms_per_token']['p95']:.3f} (launch-level); "
          + (f"iterations >= 2: P50 {res['iteration_us']['p50']:.0f} us P95 {res['iteration_us']['p95']:.0f} us "
             f"({res['tokens_per_s']:.0f} tok/s), iteration 1: P50 {res['iteration1_us']['p50']:.0f} us; " if "iteration_us" in res else "")
          + f"{res['bytes_per_token']['total'] / 1e9:.2f} GB/token -> {res['achieved_bandwidth_TBps']:.2f} TB/s -> {args.out}/bench.json", flush=True)
    mpk.finalize()


# ---- rocprof kernel trace --------------------------------------------------------------------------------

COLS = dict(name=("kernel_name", "kernelname", "name"), begin=("start_timestamp", "beginns", "begin_ns"),
            end=("end_timestamp", "endns", "end_ns"), lds=("lds_block_size", "lds"), scratch=("scratch_size", "scr"),
            vgpr=("vgpr_count", "arch_vgpr", "vgpr"), agpr=("accum_vgpr_count", "accum_vgpr", "agpr"), sgpr=("sgpr_count", "sgpr"),
            grid=("grid_size_x", "grd", "grid_size"), wg=("workgroup_size_x", "wgr", "workgroup_size"),
            calls=("calls",), avg=("averagens", "average_ns"))  # rocprof v1 --stats: one row per kernel name


def codeobj_resources(notes):
    """`llvm-readelf --notes` of the gfx942 code object (profile.sh trace writes it) -> per kernel base name:
    VGPR/AGPR/SGPR, static LDS, scratch bytes per lane, spills. Exact and static, unlike a tracer's columns."""
    import re
    out = {}
    for blk in open(notes).read().split("- .agpr_count:")[1:]:
        f = lambda key, cast=int: (lambda m: cast(m.group(1)) if m else None)(re.search(r"\.%s:\s+(\S+)" % key, blk))
        name = re.search(r"\.name:\s+_Z(\d+)(\w+)", blk)  # Itanium mangling: _Z<len><identifier>...
        if name:
            out[name.group(2)[:int(name.group(1))]] = dict(vgpr=f("vgpr_count"), agpr=int(blk.split()[0]), sgpr=f("sgpr_count"), lds=f("group_segment_fixed_size"),
                                                          scratch=f("private_segment_fixed_size"), spill_vgpr=f("vgpr_spill_count"), spill_sgpr=f("sgpr_spill_count"))
    return out


def parse_kernel_trace(paths):
    """rocprofv3 kernel_trace.csv or rocprof results.csv -> per kernel name: dispatches, us stats, resources."""
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rd = csv.DictReader(f)
            low = {c.lower().strip(): c for c in rd.fieldnames}
            col = {k: next((low[a] for a in alts if a in low), None) for k, alts in COLS.items()}
            for r in rd:
                g = lambda k, cast=float: cast(r[col[k]]) if col[k] and r[col[k]] not in ("", None) else None
                us = (g("end") - g("begin")) / 1e3 if col["begin"] and col["end"] else (g("avg") / 1e3 if col["avg"] else None)
                rows += [dict(name=r[col["name"]], us=us, lds=g("lds", int), scratch=g("scratch", int), vgpr=g("vgpr", int), agpr=g("agpr", int),
                              sgpr=g("sgpr", int), grid=g("grid", int), wg=g("wg", int))] * (g("calls", int) if col["calls"] else 1)
    per = {}
    for r in rows:
        d = per.setdefault(r["name"], dict(dispatches=0, us=[], **{k: r[k] for k in ("lds", "scratch", "vgpr", "agpr", "sgpr", "grid", "wg")}))
        d["dispatches"] += 1
        if r["us"] is not None:
            d["us"].append(r["us"])
    for d in per.values():
        us = d.pop("us")
        d["us_p50"], d["us_max"] = (pct(us, 0.5), max(us)) if us else (None, None)
        vg = (d["vgpr"] or 0) + (d["agpr"] or 0)
        d["occupancy"] = dict(waves_per_simd_vgpr_bound=min(8, 512 // max(vg, 1)) if vg else None,
                              workgroups_per_cu_lds_bound=(65536 // d["lds"]) if d["lds"] else None,
                              spills=bool(d["scratch"]), note="256-thread workgroup = 4 waves = 1 per SIMD; 57 KB LDS -> 1 WG/CU")
    return dict(kernels=per, total_dispatches=len(rows), files=paths)


def rocprof(args):
    paths = sorted(glob.glob(os.path.join(args.rocprof, "**", "*kernel_trace.csv"), recursive=True)  # rocprofv3
                   or glob.glob(os.path.join(args.rocprof, "**", "results.csv"), recursive=True)  # rocprof v1 per-dispatch (not results.stats.csv)
                   or [p for p in glob.glob(os.path.join(args.rocprof, "**", "*.csv"), recursive=True)
                       if any(a in open(p).readline().lower() for a in COLS["name"])])
    assert paths, f"no kernel trace CSV under {args.rocprof} (profile.sh trace)"
    res = parse_kernel_trace(paths)
    mk = [k for k in res["kernels"] if "worker_kernel" in k or "scheduler_kernel" in k or "prepare_kernel" in k]
    notes = os.path.join(os.path.dirname(os.path.abspath(args.rocprof.rstrip("/"))), "codeobj", "notes.txt")
    if os.path.exists(notes):  # the tracer's resource columns are empty in stats mode; the code object has the exact numbers
        co = codeobj_resources(notes)
        for k in mk:
            d = res["kernels"][k]
            for key, v in co.get(k.split("(")[0], {}).items():
                d[key] = v if d.get(key) is None else d[key]
            vg = (d["vgpr"] or 0) + (d["agpr"] or 0)
            d["occupancy"].update(waves_per_simd_vgpr_bound=min(8, 512 // max(vg, 1)) if vg else None, spills=bool(d["scratch"]) or bool(d.get("spill_vgpr")),
                                  source=os.path.relpath(notes, RL.ROOT))
        res["codeobj"] = co
    calls = max((res["kernels"][k]["dispatches"] for k in mk), default=0)
    res.update(megakernel_names=mk, mpk_calls=calls, launches_per_call=(sum(res["kernels"][k]["dispatches"] for k in mk) / calls) if calls else None,
               cmd=" ".join(sys.argv))
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "rocprof.json"), "w"), indent=1)
    for k in mk:
        d = res["kernels"][k]
        print(f"  {k[:60]:<60} x{d['dispatches']} p50 {d['us_p50']} us vgpr {d['vgpr']} agpr {d['agpr']} sgpr {d['sgpr']} lds {d['lds']} scratch {d['scratch']}")
    print(f"[bench] {res['total_dispatches']} dispatches, {calls} mpk() calls, {res['launches_per_call']} launches/call -> {args.out}/rocprof.json")


# ---- HF eager baseline -----------------------------------------------------------------------------------

def hf_baseline(args):
    from transformers import AutoModelForCausalLM, DynamicCache
    from torch.profiler import ProfilerActivity, profile
    ids = json.load(open(os.path.join(args.ref, "prompt_ids.json")))
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 attn_implementation="eager").cuda().eval()
    x = torch.tensor([ids["prompt_ids"]], device="cuda")
    with torch.no_grad():
        cache = DynamicCache()
        out = model(input_ids=x, attention_mask=torch.ones_like(x), past_key_values=cache, use_cache=True)  # DeepSeek eager asserts a mask
        cache, nxt = out.past_key_values, ids["step_inputs"][0]
        ms, launches = [], []
        for s in range(8):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                e0.record()
                out = model(input_ids=torch.tensor([[nxt]], device="cuda"), attention_mask=torch.ones(1, x.shape[1] + s + 1, dtype=torch.long, device="cuda"),
                            past_key_values=cache, use_cache=True)
                e1.record()
                torch.cuda.synchronize()
            cache, nxt = out.past_key_values, int(out.logits[0, -1].float().argmax())
            ms.append(e0.elapsed_time(e1))
            launches.append(sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA))
    res = dict(cmd=" ".join(sys.argv), device=torch.cuda.get_device_name(0), steps=8, ms=ms, ms_p50=pct(ms, 0.5),
               kernel_launches_per_token=launches, kernel_launches_p50=pct(launches, 0.5),
               method="torch.profiler CUDA events per decode step (kernels + memcpys), cuda events for ms")
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "hf_baseline.json"), "w"), indent=1)
    print(f"[bench] HF eager: {res['ms_p50']:.2f} ms/token, {res['kernel_launches_p50']} launches/token -> {args.out}/hf_baseline.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=RL.REF_DIR)
    ap.add_argument("--model", default=RL.MODEL_DIR)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--t-max", type=int, default=1056)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--no-event-timing", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--rocprof", metavar="DIR", help="parse a kernel trace instead of running")
    ap.add_argument("--hf", action="store_true", help="HF eager baseline instead of running the megakernel")
    args = ap.parse_args(argv)
    if args.rocprof:
        rocprof(args)
    elif args.hf:
        hf_baseline(args)
    else:
        bench(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
