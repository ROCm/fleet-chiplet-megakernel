#!/usr/bin/env python
"""Measure the per-task dispatch floor of the whole-token graph, and lm_head tile options.

  dispatch_probe.py --timing --out results/perf/timing     # diagnostic build: + -DMPK_ENABLE_TIMING (decision 10g)
  dispatch_probe.py --parse-timing results/perf/timing     # its [TIMING] worker lines (<out>/probe.log) -> probe.json
  dispatch_probe.py --lm-rows 128 --out results/perf/lm128 # an lm_head tile option (DIMS["lm_rows"], D10b = 64)
  dispatch_probe.py --w13-tiles 22 --w2-tiles 32 --out results/perf/sweep_w13_22_w2_32   # expert tiling (11a)

Whole token (layers 0-26 + head), warmup + runs launches of `steps` iterations each, always MPK_EVENT_TIMING (the benchmark
build). Unlike run_layer.event_timing (labels by producer order, shifted after a gcd split) every event is
labelled by the op owning its first producer task (tasks sit in plan order from index 2), so an op's span = the sum of its
events' intervals, exact. Per-op P50 over iterations >= 2. The TIMING build is diagnostic only: run nothing else on it.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_layer as RL  # noqa: E402
import run_e2e as E2E  # noqa: E402
from run_e2e import P, pct  # noqa: E402
from fleet_ds.graph import KNOBS  # noqa: E402

TIMING_RE = re.compile(r"\[TIMING\] worker=(\d+) tasks=(\d+) poll_iters=(\d+) dep_iters=(\d+) poll_cycles=(\d+) "
                       r"dep_cycles=(\d+) exec_cycles=(\d+) signal_cycles=(\d+)")


def op_of_event(tg, plan):
    """event -> plan index of the op whose tasks trigger it (task 0 TERMINATE, 1 BEGIN, then the plan in order); for a
    multi-producer DAG event (12a) the op of its first producer task."""
    starts, i = [], 2
    for _, _, g in plan:
        starts.append(i)
        i += g[0] * g[1] * g[2]
    assert i == len(tg["all_tasks"]), (i, len(tg["all_tasks"]))
    out = {}
    for t_idx in range(2, len(tg["all_tasks"])):
        e = tg["all_tasks"][t_idx]["trigger_event"] & 0xFFFFFFFF
        if e not in out:
            out[e] = int(np.searchsorted(starts, t_idx, side="right")) - 1
    return out


def raw_iterations(mpk, n_ev):
    buf = np.zeros(n_ev * 100 * 2, dtype=np.uint64)
    n = mpk.get_event_timing_func(buf.ctypes.data, n_ev * 100)
    its, cur = [], []
    for e, ts in sorted(((int(buf[2 * i]), int(buf[2 * i + 1])) for i in range(n)), key=lambda x: x[1]):
        cur.append((e, ts))
        if e == n_ev - 1:
            its.append(cur)
            cur = []
    return its


def probe(args):
    if args.lm_rows:
        RL.DIMS["lm_rows"] = args.lm_rows  # the dict fleet_ds.mirage_graph plans and tiles from
    for k in ("w13_tiles", "w2_tiles"):  # 11a expert tile sweep (relayout_layer reads w13_tiles too)
        if getattr(args, k):
            RL.DIMS[k] = getattr(args, k)
    if args.timing:
        os.environ["MPK_TIMING"] = "1"  # persistent_kernel.py compile(): -DMPK_ENABLE_TIMING
    mpk, arena, meta, ref, tg, plan = E2E.build(args, event_timing=True)
    os.environ["MPK_TIMING"] = "0"
    n_ev, ev_op = len(tg["all_events"]), op_of_event(tg, plan)
    waits = {}  # event -> the events its producer tasks wait on (12a DAG spans)
    for t in tg["all_tasks"][2:]:
        waits.setdefault(t["trigger_event"] & 0xFFFFFFFF, set()).add(t["dependent_event"] & 0xFFFFFFFF)
    n_layers = 1 + max(l for l, _, _ in plan)
    last_op = {l: max(i for i, (ll, _, _) in enumerate(plan) if ll == l) for l in range(n_layers)}
    last_ev = {l: next(e for e, o in ev_op.items() if o == last_op[l]) for l in range(n_layers)}
    layer_wall = {}
    tok0, msl = ref["step_inputs"][0], P + args.steps + 1
    spans, gcd_iv, iters, iter1, launch_ms, match, per_launch = {}, [], [], [], [], [], []
    for r in range(args.warmup + args.runs):
        ms = RL.launch(mpk, meta, P, tok0, max_seq_length=msl)
        gen = meta["tokens"][0, P + 1:P + args.steps + 1].tolist()
        its = raw_iterations(mpk, n_ev)
        print(f"{time.strftime('%H:%M:%S')} [probe] launch {r}{' (warmup)' if r < args.warmup else ''}: {ms:.2f} ms, "
              f"{len(its)} iterations, tokens match {sum(a == b for a, b in zip(gen, ref['generated_ids']))}/{args.steps}", flush=True)
        launch_ms.append(ms)
        if r < args.warmup:
            continue
        match.append(sum(a == b for a, b in zip(gen, ref["generated_ids"])))
        per_launch.append([(it[-1][1] - it[0][1]) / 100 for it in its[1:]])  # iterations >= 2 of this launch (noise: P50 per launch group)
        for k, it in enumerate(its):
            (iter1 if k == 0 else iters).append((it[-1][1] - it[0][1]) / 100)
            if k == 0:
                continue
            per, at = {}, dict(it)
            lm = [ts for e, ts in it if plan[ev_op.get(e, 0)][1] == "LM_HEAD"]
            gcd_iv += [(b - a) / 100 for a, b in zip(lm, lm[1:])]
            for (_, ts), (e, ts2) in zip(it, it[1:]):  # span = firing - the latest event its producer tasks waited on
                d = [at[x] for x in waits.get(e, ()) if x in at]  # (chain: the previous firing; DAG: the op's own start)
                per[ev_op[e]] = max(per.get(ev_op[e], 0.0), (ts2 - (max(d) if d else ts)) / 100)
            for l in range(1, n_layers):  # MoE layer wall time: its last op's event minus the previous layer's
                if last_ev.get(l) in at and last_ev.get(l - 1) in at:
                    layer_wall.setdefault(l, []).append((at[last_ev[l]] - at[last_ev[l - 1]]) / 100)
            for o, us in per.items():
                spans.setdefault(o, []).append(us)
    n_sched = mpk.num_local_schedulers
    ops = []
    for o, (l, typ, g) in enumerate(plan):
        if o in spans:
            ops.append(dict(op=o, layer=l, type=typ, tasks=g[0] * g[1] * g[2], events=sum(1 for v in ev_op.values() if v == o),
                            p50_us=pct(spans[o], 0.5)))
    by = {}
    for x in ops:
        cls = "head" if x["layer"] < 0 else "layer0" if x["layer"] == 0 else "moe"
        by.setdefault((cls, x["type"], x["tasks"], x["events"]), []).append(x["p50_us"])
    table = [dict(cls=c, type=t, tasks=n, events=e, p50_us=statistics.fmean(v), layers=len(v),
                  us_per_task_per_sched=statistics.fmean(v) / max(n / n_sched, 1)) for (c, t, n, e), v in by.items()]
    lm = next(x for x in table if x["type"] == "LM_HEAD")
    res = dict(cmd=" ".join(sys.argv), git=RL.git_hash(), date=time.strftime("%Y-%m-%dT%H:%M:%S"), timing_build=args.timing,
               lm_rows=RL.DIMS["lm_rows"], w13_tiles=RL.DIMS["w13_tiles"], w2_tiles=RL.DIMS["w2_tiles"],
               knobs={k: RL.DIMS[k] for k in KNOBS}, steps=args.steps, warmup=args.warmup, runs=args.runs, schedulers=n_sched,
               workers=mpk.num_workers, graph=dict(tasks=len(tg["all_tasks"]), events=n_ev, ops=len(plan)), compile_s=mpk.compile_s,
               launch_ms=launch_ms, tokens_match=match,
               iteration_us=dict(p50=pct(iters, 0.5), p95=pct(iters, 0.95), n=len(iters), method="iterations >= 2 of each timed launch"),
               iteration1_us=dict(p50=pct(iter1, 0.5), n=len(iter1)), tokens_per_s=1e6 / pct(iters, 0.5),
               iteration_us_per_launch=per_launch,
               head_us=sum(x["p50_us"] for x in table if x["cls"] == "head" and x["type"] != "EMBEDDING"),
               moe_layer_us=sum(x["p50_us"] for x in table if x["cls"] == "moe"), layer0_us=sum(x["p50_us"] for x in table if x["cls"] == "layer0"),
               moe_layer_wall_us=statistics.fmean(pct(v, 0.5) for v in layer_wall.values()) if layer_wall else None,
               lm_head=dict(tasks=lm["tasks"], events=lm["events"], p50_us=lm["p50_us"], gcd_interval_p50_us=pct(gcd_iv, 0.5),
                            tasks_per_event=lm["tasks"] / lm["events"],
                            us_per_task_per_sched_from_gcd=(pct(gcd_iv, 0.5) / (lm["tasks"] / lm["events"] / n_sched)) if gcd_iv else None),
               per_op_type=sorted(table, key=lambda x: (x["cls"], -x["p50_us"])), per_op=ops,
               method="event -> op by its first producer task; op span = sum of its events' intervals; P50 over iterations >= 2; "
                      "MoE rows = mean over layers 1-26 of the per-layer P50")
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "probe.json"), "w"), indent=1)
    print(f"[probe] knobs {res['knobs']} lm_rows {res['lm_rows']}: MoE layer wall {res['moe_layer_wall_us']} us; iteration P50 {res['iteration_us']['p50']:.0f} us P95 {res['iteration_us']['p95']:.0f} us; "
          f"head {res['head_us']:.0f} us (LM_HEAD {lm['p50_us']:.1f} us, {lm['tasks']} tasks, {lm['events']} events, gcd interval P50 "
          f"{res['lm_head']['gcd_interval_p50_us']}); MoE layer {res['moe_layer_us']:.1f} us; tokens {match} -> {args.out}/probe.json", flush=True)
    for x in res["per_op_type"]:
        print(f"  {x['cls']:<6} {x['type']:<22} tasks {x['tasks']:>5} events {x['events']:>3} P50 {x['p50_us']:8.2f} us  "
              f"{x['us_per_task_per_sched']:.2f} us/(task/sched)", flush=True)
    mpk.finalize()


def parse_timing(out):
    """[TIMING] lines (workers 0-7, lane 0, printed at each launch's TERMINATE) -> per launch, per worker; clock64 is
    calibrated per launch as max over workers of (poll + dep + exec + signal) cycles / launch wall us (untimed descriptor
    loads make it a slight under-estimate of cycles per us)."""
    res = json.load(open(os.path.join(out, "probe.json")))
    rows = [tuple(map(int, m.groups())) for m in TIMING_RE.finditer(open(os.path.join(out, "probe.log")).read())]
    blocks = [rows[i:i + 8] for i in range(0, len(rows), 8)]
    assert len(blocks) == len(res["launch_ms"]), (len(rows), len(res["launch_ms"]))
    launches = []
    for ms, blk in zip(res["launch_ms"], blocks):
        cpu = max(p + d + e + s for _, _, _, _, p, d, e, s in blk) / (ms * 1e3)  # cycles per us
        launches.append(dict(launch_ms=ms, cycles_per_us=cpu, workers=[dict(
            worker=w, tasks=n, poll_iters=pi, dep_iters=di, poll_us=p / cpu, dep_us=d / cpu, exec_us=e / cpu, signal_us=s / cpu,
            exec_us_per_task=e / cpu / n, signal_us_per_task=s / cpu / n, dep_us_per_task=d / cpu / n, poll_us_per_task=p / cpu / n)
            for w, n, pi, di, p, d, e, s in blk]))
    timed = launches[res["warmup"]:]
    mean = lambda k: statistics.fmean(x[k] for lau in timed for x in lau["workers"])
    res["timing"] = dict(launches=launches, per_task_mean_us={k: mean(k) for k in ("exec_us_per_task", "signal_us_per_task", "dep_us_per_task", "poll_us_per_task")},
                         tasks_per_worker_per_launch=mean("tasks"), cycles_per_us=statistics.fmean(x["cycles_per_us"] for x in timed),
                         method=parse_timing.__doc__.strip())
    json.dump(res, open(os.path.join(out, "probe.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res["timing"].items() if k != "launches"}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=RL.REF_DIR)
    ap.add_argument("--model", default=RL.MODEL_DIR)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--t-max", type=int, default=1056)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--lm-rows", type=int, default=None, help="lm_head tile rows (D10b: 64)")
    ap.add_argument("--w13-tiles", type=int, default=None, help="EXPERT_W13_SILU N-tiles per slot (11a; 44)")
    ap.add_argument("--w2-tiles", type=int, default=None, help="EXPERT_W2 N-tiles per slot (11a; 32)")
    ap.add_argument("--timing", action="store_true", help="diagnostic -DMPK_ENABLE_TIMING build (decision 10g)")
    ap.add_argument("--parse-timing", metavar="OUT", help="parse OUT/probe.log [TIMING] lines into OUT/probe.json")
    ap.add_argument("--out", default=os.path.join(RL.ROOT, "results", "perf", "probe"))
    args = ap.parse_args(argv)
    if args.parse_timing:
        parse_timing(args.parse_timing)
    else:
        probe(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
