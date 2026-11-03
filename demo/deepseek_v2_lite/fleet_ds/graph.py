"""Per-token decode task graph for DeepSeek-V2-Lite in Fleet's TaskDesc/EventDesc shape, emitted as JSON.
Local cross-check only (design D4: the real graph comes from Mirage's builder). Op order and
per-op grids are exactly docs/design.md §2.4-2.5 (decision 8f: the harness follows the doc, not the other way).

The graph is a straight chain of *ops* (decision 4): every op's first input is the previous op's first
output, so Mirage's linear-chain assert holds. An op is a stage of tasks launched by one event; the stage
triggers the next op's event (num_triggers == stage size, first/last_task_id == launched range, half-open).
GEMVs are tiled into 64-row tickets (decisions 3, 8e); the shared expert shares the routed grid, so its
w13 tiles are 2x the routed rows (8c; 176 vs 88 since 11a); silu is fused into the w13 epilogue (8a); only layer 0's down-proj is split-K (8b).
Expert tasks are slot-indexed (10a, 2026-09-14; the superset-with-skip graph of decision 2 was the first measured
version): task = (top-k slot, N-tile), the expert index is idx[slot] read at run time; gate + top-k are one task (10c).

Defaults (2026-09-15): dag=1, fold=1 = the session-8 measured best (docs/gpu-log.md; with kernels/switches.h COMBINE_BLOCK 1,
NORM_FIXED 1, MERGE_FIXED 2): the DAG of decision 12a with the L6 norm/merge folds, 20,492 tasks / 298 ops / 246 events per token
(design-final-fixed §2.4-2.5). FLEET_KNOBS=dag=0,fold=0 (and the switches at 0) builds the straight chain below, the measured history
up to session 7 (20,979 tasks / 301 ops / 302 events).

Chain per-layer op order (design.md §2.4 op numbers; decision numbers in brackets):
  1 RMS_NORM(x)                -> xn
  2 QKV_PROJ(xn)               -> qkva       [q_proj | kv_a_proj_with_mqa] = 3648 rows, 57 tiles
  3 KV_B_PROJ(qkva)            -> kv_cache   prologue kv_a_layernorm(c_kv); 4096 rows, 64 tiles, scattered into
                                             K[h,p,0:128] / V[h,p,:]; tile 4h writes K[h,p,128:192] = RoPE(k_pe) [1, 11b]
  4 ATTN_CHUNK(kv_cache, qkva) -> attn_part  prologue q RoPE of its head; 128-token chunk x head               [7b, 11b]
  (MLA_PREP folded, decision 11b; the RMSNorm and ATTN_MERGE folds were measured slower and are not in the graph)
  6 ATTN_MERGE(attn_part)      -> attn_o     LSE merge, one task per head                                  [7b]
  7 O_PROJ_RESIDUAL(attn_o, x) -> x_attn     2048 rows, 32 tiles
  8 RMS_NORM(x_attn)           -> xn2
  dense (layer 0):
  9 GATE_UP_SILU(xn2)          -> h          21888 rows, 342 tiles, silu*up in the epilogue                  [8a]
 10 DOWN_SPLITK(h)             -> down_part  32 N-tiles x 4 K-splits                                        [7e, 8b]
 11 REDUCE_RESIDUAL(down_part, x_attn) -> x
  MoE (layers 1..26):
  9 MOE_GATE_TOPK(xn2)         -> routing    gate GEMV fp32 [64] + top-6: w[6], idx[6] (+ logits kept)     [10c]
 10 EXPERT_W13_SILU(routing, xn2) -> h       32 N-tiles x 7 slots (88-row routed / 176-row shared tiles, 11a sweep) (6 routed: expert idx[slot]; slot 6 = shared) [10a, 7a, 8a, 8c]
 11 EXPERT_W2(h, routing)      -> expert_out 32 N-tiles x 7 slots (shared down unsplit)                    [10a, 8b]
 12 MOE_COMBINE_RESIDUAL(expert_out, routing, x_attn) -> x   fp32 sum routed -> bf16 -> + shared (HF order) [8d]
Model tail: RMS_NORM -> LM_HEAD (200 tasks x 512 rows [Q-H 2026-09-14, measured; 1,600 x 64 under 10b, 320 x 320 under 7g]) -> ARGMAX_PARTIAL (1024-logit chunks) -> ARGMAX_REDUCE.
Scratch buffers are per layer, L{l}.* [7f].
Run: python -m fleet_ds.graph [t_max] > graph.json  (counts on stderr)
"""
import json
import os
import sys
from collections import Counter

DIMS = dict(n_layers=27, n_heads=16, n_experts=64, first_k_dense=1, vocab=102400, hidden=2048, chunk=128,
            topk=6, split_k=4, lm_rows=512, tile=64, argmax_chunk=1024, w13_tiles=32, w2_tiles=32,
            dag=1,  # 12a: 1 = the DAG (Q/KVA split, the shared expert as its own ops), default since session 8; 0 = the chain
            # 12e-12g switches (0 = off): L2 shared-expert tiles (DAG), L3 = w13_tiles 44, L4 attn_rows 32/16 (Q/KVA/KV_B/O_PROJ tile
            # rows), L5 gate_tiles (gate GEMV as its own op of that many tiles + a top-k task), L6 fold (norm/merge prologues, default 1)
            shared_w13_tiles=0, shared_w2_tiles=0, attn_rows=64, gate_tiles=0, fold=1,
            q_rows=3072, kv_a_rows=576, kv_b_rows=4096, dense_inter=10944, moe_inter=1408, shared_inter=2816)


KNOBS = ("dag", "shared_w13_tiles", "shared_w2_tiles", "w13_tiles", "w2_tiles", "attn_rows", "gate_tiles", "fold", "lm_rows")
# FLEET_KNOBS="dag=1,attn_rows=16,..." sets them for every script (run_layer, run_e2e, bench, dispatch_probe, build_graph.sh)
for _kv in filter(None, os.environ.get("FLEET_KNOBS", "").split(",")):
    _k, _v = _kv.split("=")
    assert _k in KNOBS, f"FLEET_KNOBS: unknown knob {_k!r}, one of {KNOBS}"
    DIMS[_k] = int(_v)
BAD_ATTN_ROWS = {256: "attn_rows=256 decodes incorrectly (5/32 tokens, docs/gpu-log.md session 8 x6, results/status.yaml known_failures)"}
assert DIMS["attn_rows"] not in BAD_ATTN_ROWS, BAD_ATTN_ROWS.get(DIMS["attn_rows"])


def build(t_max=1056, **kw):
    d = {**DIMS, **kw}
    dag, fold, ar, gt = d["dag"], d["fold"], d["attn_rows"], d["gate_tiles"]
    assert ar not in BAD_ATTN_ROWS, BAD_ATTN_ROWS[ar]
    n13s, n2s = d["shared_w13_tiles"] or d["w13_tiles"], d["shared_w2_tiles"] or d["w2_tiles"]
    assert dag or (n13s == d["w13_tiles"] and n2s == d["w2_tiles"]), "own shared-expert tiles need the DAG (L2)"
    n_chunks = -(-t_max // d["chunk"])
    tile = d["tile"]
    tasks, events, ops = [], [{"id": 0, "type": "LAUNCH_TASKS", "num_triggers": 1, "producers": None, "first_task_id": 2}], []
    tasks.append({"id": 0, "type": "TERMINATE", "layer": -1, "op": None, "inputs": [], "outputs": [],
                  "dependent_event": None, "trigger_event": None})
    tasks.append({"id": 1, "type": "BEGIN_TASK_GRAPH", "layer": -1, "op": None, "inputs": [], "outputs": [],
                  "dependent_event": None, "trigger_event": 0})

    writers, anc = {}, []  # buffer -> op ids that wrote it; op id -> ancestor op ids

    def op(layer, typ, inputs, outputs, metas=({},)):
        """One op = one stage; metas = per-task metadata (len == task count). Its producers are the writers of its input
        buffers minus writers that are ancestors of another writer (the same rule as src/kernel/runtime.cc register_mugraph
        register_mugraph, decision 12a). Each producer task triggers exactly one event (Fleet's TaskDesc has one
        trigger_event), so ops with the same producer set share one event (fan-out, their tasks adjacent) and an
        event's num_triggers counts every producer task (multi-producer). On the chain graph this is exactly the chain."""
        oid, first = len(ops), len(tasks)
        w = set().union(*(writers.get(n, ()) for n in inputs))
        prod = sorted(w - set().union(*(anc[i] for i in w)))
        anc.append(w.union(*(anc[i] for i in w)))
        if not prod:
            dep = events[0]
        else:
            dep = next((e for e in events[1:] if e["producers"] == prod), None)
            if dep is not None:  # fan-out: another consumer of the same producers extends the launched range
                assert dep["last_task_id"] == first, (typ, layer, "fan-out consumers of one event must be adjacent")
            else:
                ptasks = [t for i in prod for t in tasks[ops[i]["first_task_id"]:ops[i]["last_task_id"]]]
                assert all(t["trigger_event"] is None for t in ptasks), (typ, layer, "a producer already triggers another event")
                dep = {"id": len(events), "type": "LAUNCH_TASKS", "num_triggers": len(ptasks), "producers": prod,
                       "first_task_id": first}
                for t in ptasks:
                    t["trigger_event"] = dep["id"]
                events.append(dep)
        o = {"id": oid, "type": typ, "layer": layer, "inputs": inputs, "outputs": outputs, "producers": prod,
             "first_task_id": first, "last_task_id": first + len(metas)}
        for m in metas:
            tasks.append({"id": len(tasks), "type": typ, "layer": layer, "op": oid, "inputs": inputs,
                          "outputs": outputs, "dependent_event": dep["id"], "trigger_event": None, **m})
        dep["last_task_id"] = len(tasks)
        for n in dict.fromkeys(outputs):
            writers.setdefault(n, []).append(oid)
        ops.append(o)

    def tiles(rows, rows_per=tile, **extra):
        assert rows % rows_per == 0, (rows, rows_per)
        return [{"row0": r, "rows": rows_per, **extra} for r in range(0, rows, rows_per)]

    def slot_tiles(rows, rows_shared, n, n_s):
        """Slot-indexed (decision 10a): task = (top-k slot, N-tile), the expert index is idx[slot] at run time; slot
        `topk` = the shared expert (7a), always on, weight 1.0, on the same grid.x so its tiles are rows_shared/rows_per
        wide (8c). n = N-tiles per slot (11a: w13_tiles / w2_tiles). No empty tickets (decision 2's superset had 58 x 76)."""
        assert rows % n == 0 and rows_shared % n_s == 0, (rows, rows_shared, n, n_s)
        out = [{"slot": s, **t} for s in range(d["topk"]) for t in tiles(rows, rows // n)]
        out += [{"slot": d["topk"], "shared": True, "weight": 1.0, **t} for t in tiles(rows_shared, rows_shared // n_s)]
        return out

    op(-1, "EMBEDDING", ["token", "embed_tokens"], ["x0"])
    x = "x0"
    for l in range(d["n_layers"]):
        b = lambda n: f"L{l}.{n}"
        dense = l < d["first_k_dense"]
        if fold:  # L6: every Q/KVA/QKV tile recomputes rmsnorm(x) as its prologue (11b structure, perf/fold-full)
            xin = [x, b("input_layernorm")]
        else:
            op(l, "RMS_NORM", [x, b("input_layernorm")], [b("xn")])
            xin = [b("xn")]
        if dag:  # 12a: q rows feed ATTN_CHUNK, kv_a rows feed KV_B_PROJ; the same tiles of W_qkva (row0 global)
            op(l, "Q_PROJ", [*xin, b("W_qkva")], [b("q")], tiles(d["q_rows"], ar))
            op(l, "KVA_PROJ", [*xin, b("W_qkva")], [b("kva")],
               [{**t, "row0": t["row0"] + d["q_rows"]} for t in tiles(d["kv_a_rows"], ar if d["kv_a_rows"] % ar == 0 else tile)])  # 576 rows: 64-row tiles when attn_rows does not divide
            qkv = [b("q"), b("kva")]
        else:
            op(l, "QKV_PROJ", [*xin, b("W_qkva")], [b("qkva")], tiles(d["q_rows"] + d["kv_a_rows"], ar))
            qkv = [b("qkva")] * 2
        op(l, "KV_B_PROJ", [qkv[1], b("kv_a_layernorm"), b("kv_b_proj"), "rope_table"], [b("kv_cache")],
           tiles(d["kv_b_rows"], ar))
        op(l, "ATTN_CHUNK", [b("kv_cache"), qkv[0], "rope_table"], [b("attn_part")],
           [{"head": h, "chunk": c, "chunk_tokens": d["chunk"]} for c in range(n_chunks) for h in range(d["n_heads"])])
        if fold:  # L6: the LSE merge of the 16 heads is the O_PROJ tiles' prologue
            op(l, "O_PROJ_RESIDUAL", [b("attn_part"), b("o_proj"), x], [b("x_attn")], tiles(d["hidden"], ar))
        else:
            op(l, "ATTN_MERGE", [b("attn_part")], [b("attn_o")], [{"head": h} for h in range(d["n_heads"])])
            op(l, "O_PROJ_RESIDUAL", [b("attn_o"), b("o_proj"), x], [b("x_attn")], tiles(d["hidden"], ar))
        if not fold or (gt and not dense):  # L6 + L5: norm2 stays a task (no gate tile can own the xn2 write), 12e
            op(l, "RMS_NORM", [b("x_attn"), b("post_attention_layernorm")], [b("xn2")])
            x2 = [b("xn2")]
        else:
            x2 = [b("x_attn"), b("post_attention_layernorm")]
        if dense:
            op(l, "GATE_UP_SILU", [*x2, b("mlp.gate_up")], [b("h")], tiles(2 * d["dense_inter"]))
            op(l, "DOWN_SPLITK", [b("h"), b("mlp.down_proj")], [b("down_part")],
               [{**t, "k_split": k, "k_splits": d["split_k"]} for t in tiles(d["hidden"]) for k in range(d["split_k"])])
            op(l, "REDUCE_RESIDUAL", [b("down_part"), b("x_attn")], [b("x")])
        else:
            w13 = slot_tiles(2 * d["moe_inter"], 2 * d["shared_inter"], d["w13_tiles"], n13s)
            w2 = slot_tiles(d["hidden"], d["hidden"], d["w2_tiles"], n2s)
            if gt:  # L5: gate GEMV as its own op of gt fp32 tiles, then top-k as one task
                op(l, "MOE_GATE", [*x2, b("mlp.gate")], [b("logits")], tiles(d["n_experts"], d["n_experts"] // gt))
            else:  # L6 (x2 = x_attn + norm weight): norm2 is the prologue and the task publishes xn2 for the routed experts
                op(l, "MOE_GATE_TOPK", [*x2, b("mlp.gate")], [b("routing"), b("logits")] + ([b("xn2")] if len(x2) == 2 else []))
            if dag:  # 12a: the shared expert (slot 6, 7a) is its own w13 -> w2 path next to gate/top-k and the routed slots
                op(l, "EXPERT_W13_SHARED", [*x2, b("mlp.w13_s")], [b("h_s")], [t for t in w13 if t.get("shared")])
            if gt:
                op(l, "MOE_TOPK", [b("logits")], [b("routing")])
            if dag:
                op(l, "EXPERT_W13_SILU", [b("routing"), b("xn2"), b("mlp.w13")], [b("h")], [t for t in w13 if not t.get("shared")])
                op(l, "EXPERT_W2_SHARED", [b("h_s"), b("mlp.w2_s")], [b("expert_out_s")], [t for t in w2 if t.get("shared")])
                op(l, "EXPERT_W2", [b("h"), b("routing"), b("mlp.w2")], [b("expert_out")], [t for t in w2 if not t.get("shared")])
                eo = [b("expert_out"), b("expert_out_s")]
            else:
                op(l, "EXPERT_W13_SILU", [b("routing"), b("xn2"), b("mlp.w13"), b("mlp.w13_s")], [b("h")], w13)
                op(l, "EXPERT_W2", [b("h"), b("routing"), b("mlp.w2"), b("mlp.w2_s")], [b("expert_out")], w2)
                eo = [b("expert_out")]
            op(l, "MOE_COMBINE_RESIDUAL", [*eo, b("routing"), b("x_attn")], [b("x")])
        x = b("x")
    if fold:  # L6: the model norm is the LM_HEAD tiles' prologue (perf/fold-full)
        op(-1, "LM_HEAD", [x, "model.norm", "lm_head"], ["logits"], tiles(d["vocab"], d["lm_rows"]))
    else:
        op(-1, "RMS_NORM", [x, "model.norm"], ["xn_final"])
        op(-1, "LM_HEAD", ["xn_final", "lm_head"], ["logits"], tiles(d["vocab"], d["lm_rows"]))
    op(-1, "ARGMAX_PARTIAL", ["logits"], ["argmax_part"], tiles(d["vocab"], d["argmax_chunk"]))
    op(-1, "ARGMAX_REDUCE", ["argmax_part"], ["token"])
    sinks = [t for t in tasks[2:] if t["trigger_event"] is None]  # every task triggers one event; sinks trigger END
    for t in sinks:
        t["trigger_event"] = len(events)
    events.append({"id": len(events), "type": "END_OF_TASK_GRAPH", "num_triggers": len(sinks), "producers": [],
                   "first_task_id": 0, "last_task_id": 0})
    return {"tasks": tasks, "events": events, "ops": ops}


def counts(g):
    per_layer = Counter(t["layer"] for t in g["tasks"])
    per_layer_ops = Counter(o["layer"] for o in g["ops"])
    slots = sum(1 for t in g["tasks"] if "slot" in t)
    lines = [f"layer {l}: {per_layer[l]} tasks, {per_layer_ops[l]} ops/events" for l in sorted(per_layer)]
    lines.append(f"per token: {len(g['tasks'])} tasks ({len(g['tasks']) - 2} launched, {slots} slot-indexed expert "
                 f"tickets, no empty ones), {len(g['ops'])} ops, {len(g['events'])} events")
    return "\n".join(lines)


if __name__ == "__main__":
    g = build(*map(int, sys.argv[1:]))
    print(counts(g), file=sys.stderr)
    json.dump(g, sys.stdout, indent=1)
