import glob, json, os, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fleet_ds.graph import DIMS, build, counts  # noqa: E402

# Numbers as written in docs/design-final-fixed.md §2.4-2.5 (decision 8f: the harness must reproduce the doc, not define it).
# Default (2026-09-15) = the session-8 measured best: FLEET_KNOBS dag=1,fold=1, the DAG of decision 12a with the L6 norm/merge folds.
DEFAULT, CHAIN = dict(dag=1, fold=1), dict(dag=0, fold=0)
DOC_MOE_LAYER_TASKS = 747       # §2.4: the chain's 765 minus 2 RMS_NORM and 16 ATTN_MERGE tasks (folded into prologues, L6)
DOC_MOE_LAYER_OPS = 11          # §2.4 (the chain's 11 ops: QKV_PROJ -> Q_PROJ + KVA_PROJ, shared expert own w13/w2, 3 folds)
DOC_MOE_LAYER_EVENTS = 9        # §2.4: Q_PROJ + KVA_PROJ and MOE_GATE_TOPK + EXPERT_W13_SHARED share one event each (fan-out)
DOC_DENSE_LAYER_TASKS = 768     # §2.5: 786 - 18
DOC_DENSE_LAYER_OPS = 8         # §2.5: Q, KVA, KV_B, ATTN_CHUNK, O_PROJ, GATE_UP_SILU, DOWN_SPLITK, REDUCE_RESIDUAL
DOC_DENSE_LAYER_EVENTS = 7      # §2.5: one fan-out (Q + KVA)
DOC_HEAD_TASKS = 301            # §2.5: LM_HEAD 200 (model norm as prologue) + ARGMAX_PARTIAL 100 + ARGMAX_REDUCE 1
DOC_PER_TOKEN_TASKS = 20492     # §2.5: 1 + 768 + 26 x 747 + 301
DOC_PER_TOKEN_OPS = 298         # §2.5: 11 x 26 + 8 + 4 (embedding + 3 head ops)
DOC_PER_TOKEN_EVENTS = 246      # §2.5: 9 x 26 + 7 + 4 + END (builder events; Mirage §3.2: 347)
DOC_MIRAGE_TASKS, DOC_MIRAGE_EVENTS = 20494, 347  # task_graph.json (+ TERMINATE, BEGIN); session 8 g1 results/perf/dag_g1_layers
MIRAGE_EXTRA_EVENTS = 101       # §3.2: LM_HEAD -> ARGMAX_PARTIAL gcd 100 events (+99), GATE_UP -> DOWN_SPLITK 2 (+1), TERMINATE (+1)
DOC_MOE_GRID = {"Q_PROJ": 48, "KVA_PROJ": 9, "KV_B_PROJ": 64, "ATTN_CHUNK": 144, "O_PROJ_RESIDUAL": 32, "MOE_GATE_TOPK": 1,
                "EXPERT_W13_SHARED": 32, "EXPERT_W13_SILU": 6 * 32, "EXPERT_W2_SHARED": 32, "EXPERT_W2": 6 * 32,
                "MOE_COMBINE_RESIDUAL": 1}  # §2.4 "Grid / count" column
DOC_DENSE_GRID = {"GATE_UP_SILU": 342, "DOWN_SPLITK": 128, "REDUCE_RESIDUAL": 1}  # §2.5
DOC_MOE_PRODUCERS = {"Q_PROJ": [], "KVA_PROJ": [], "KV_B_PROJ": ["KVA_PROJ"], "ATTN_CHUNK": ["Q_PROJ", "KV_B_PROJ"],
                     "O_PROJ_RESIDUAL": ["ATTN_CHUNK"], "MOE_GATE_TOPK": ["O_PROJ_RESIDUAL"], "EXPERT_W13_SHARED": ["O_PROJ_RESIDUAL"],
                     "EXPERT_W13_SILU": ["MOE_GATE_TOPK"], "EXPERT_W2_SHARED": ["EXPERT_W13_SHARED"], "EXPERT_W2": ["EXPERT_W13_SILU"],
                     "MOE_COMBINE_RESIDUAL": ["EXPERT_W2_SHARED", "EXPERT_W2"]}  # producers inside the layer; Q/KVA: the previous layer's last op

# The straight chain (FLEET_KNOBS=dag=0,fold=0; the measured history up to session 7, design §2.4-2.5 "chain" rows).
CHAIN_MOE_LAYER_TASKS = 765     # §2.4 table total (11 ops, D10a/D10c/D11b): 315 (ops 1-7) + 1 gate+topk + 7 x 32 + 7 x 32 + 1 (850 with 7 x 44, 11a)
CHAIN_MOE_LAYER_REAL = 765      # every task does work: slot-indexed expert tickets (D10a; the superset had 5,259 / 851)
CHAIN_MOE_LAYER_EMPTY = 0       # §2.4: no empty tickets any more (was 58 unselected experts x 76 = 4,408)
CHAIN_MOE_LAYER_OPS = 11        # §2.4 op boundaries per MoE layer (13 before D10c, 12 before the D11b MLA_PREP fold)
CHAIN_DENSE_LAYER_TASKS = 786   # §2.5: 315 + 342 + 128 + 1
CHAIN_DENSE_LAYER_OPS = 10      # §2.5: 7 + gate_up_silu + down_splitk + reduce
CHAIN_HEAD_TASKS = 302          # §2.5: 1 + 200 + 100 + 1 (Q-H: lm_head 200 x 512 rows; 1,702 under D10b)
CHAIN_PER_TOKEN_TASKS = 20979   # §2.5: 1 + 786 + 26 x 765 + 302 (21,006 before D11b, 23,190 before 11a, 24,590 under D10b)
CHAIN_PER_TOKEN_REAL = 20979    # §2.5
CHAIN_PER_TOKEN_EMPTY = 0       # §2.5 (was 26 x 4408 = 114,608)
CHAIN_PER_TOKEN_BARRIERS = 301  # §2.5: 11 x 26 + 10 + 5 (embedding + 4 head ops); 328 before D11b
CHAIN_MIRAGE_EVENTS = 403       # §3.2, results/perf/mla_l0-26
CHAIN_MOE_GRID = {"QKV_PROJ": 57, "KV_B_PROJ": 64, "ATTN_CHUNK": 144, "ATTN_MERGE": 16, "O_PROJ_RESIDUAL": 32,
                  "MOE_GATE_TOPK": 1, "EXPERT_W13_SILU": 32 * 7, "EXPERT_W2": 32 * 7}  # §2.4 "Grid / count" column
CHAIN_DENSE_GRID = DOC_DENSE_GRID


def test_chain_graph():
    g = build(**CHAIN)
    tasks, events, ops = g["tasks"], g["events"], g["ops"]
    assert [t["id"] for t in tasks] == list(range(len(tasks)))
    assert [e["id"] for e in events] == list(range(len(events)))
    assert [o["id"] for o in ops] == list(range(len(ops)))

    # events: num_triggers recount == tasks that trigger it; launched range is exactly the dependents
    trig = Counter(t["trigger_event"] for t in tasks if t["trigger_event"] is not None)
    for e in events:
        if e["type"] == "END_OF_TASK_GRAPH":
            continue
        assert trig[e["id"]] == e["num_triggers"], e
        launched = tasks[e["first_task_id"]:e["last_task_id"]]
        assert launched and all(t["dependent_event"] == e["id"] for t in launched), e
    # DAG: every task waits on an event created before the one it triggers; all but TERMINATE/BEGIN launched once
    for t in tasks[2:]:
        assert t["dependent_event"] < t["trigger_event"]
    assert sum(e["last_task_id"] - e["first_task_id"] for e in events) == len(tasks) - 2

    # chain (decision 4): each op's first input is the previous op's first output; op i has event i+1
    for prev, o in zip(ops, ops[1:]):
        assert o["inputs"][0] == prev["outputs"][0], (prev, o)
        assert tasks[o["first_task_id"]]["dependent_event"] == prev["id"] + 1
    assert len(events) == len(ops) + 1

    # counts EQUAL docs/design.md §2.4-2.5
    per_layer = Counter(t["layer"] for t in tasks)
    per_layer_ops = Counter(o["layer"] for o in ops)
    assert per_layer[0] == CHAIN_DENSE_LAYER_TASKS and per_layer_ops[0] == CHAIN_DENSE_LAYER_OPS
    assert all(per_layer[l] == CHAIN_MOE_LAYER_TASKS and per_layer_ops[l] == CHAIN_MOE_LAYER_OPS for l in range(1, 27))
    assert per_layer[-1] == 2 + 1 + CHAIN_HEAD_TASKS  # TERMINATE, BEGIN, embedding, head
    assert len(tasks) - 2 == CHAIN_PER_TOKEN_TASKS
    assert len(ops) == CHAIN_PER_TOKEN_BARRIERS and len(events) == CHAIN_PER_TOKEN_BARRIERS + 1
    assert len(events) + MIRAGE_EXTRA_EVENTS == CHAIN_MIRAGE_EVENTS
    grid = Counter((t["layer"], t["type"]) for t in tasks)
    for typ, n in CHAIN_MOE_GRID.items():
        assert all(grid[l, typ] == n for l in range(1, 27)), typ
    for typ, n in CHAIN_DENSE_GRID.items():
        assert grid[0, typ] == n, typ
    # slot-indexed (decision 10a): per MoE layer 6 routed slots x 76 tickets (expert = idx[slot] at run time) + 76 shared
    routed = sum(1 for t in tasks if "slot" in t and not t.get("shared"))
    shared = sum(1 for t in tasks if t.get("shared"))
    assert routed == 26 * 6 * 64 and shared == 26 * 64 and not any("expert" in t or "skip_unless_routed" in t for t in tasks)
    assert all(t["slot"] == 6 for t in tasks if t.get("shared")) and all(t["slot"] < 6 for t in tasks if "slot" in t and not t.get("shared"))
    assert CHAIN_PER_TOKEN_EMPTY == CHAIN_MOE_LAYER_EMPTY == 0
    assert CHAIN_PER_TOKEN_TASKS == CHAIN_PER_TOKEN_REAL and CHAIN_MOE_LAYER_TASKS == CHAIN_MOE_LAYER_REAL
    # tile geometry: 64-row tickets (8e); w13 88 rows, shared 176 (8c, 11a sweep); lm_head 200 x 512 (Q-H)
    assert all(t["rows"] == 64 for t in tasks if t["type"] in ("QKV_PROJ", "KV_B_PROJ", "O_PROJ_RESIDUAL",
                                                                 "GATE_UP_SILU", "DOWN_SPLITK", "EXPERT_W2"))
    assert all(t["rows"] == (176 if t.get("shared") else 88) for t in tasks if t["type"] == "EXPERT_W13_SILU")  # 32 tiles (11a)
    assert sum(1 for t in tasks if t["type"] == "LM_HEAD") == 200 and all(t["rows"] == 512 for t in tasks if t["type"] == "LM_HEAD")
    assert sum(1 for t in tasks if "k_split" in t) == 128 and all(t["layer"] == 0 for t in tasks if "k_split" in t)
    assert not any(t["type"] == "SILU_MUL" for t in tasks)  # 8a: fused into the w13 epilogue
    # per-layer scratch names are distinct across layers (7f)
    names = [n for o in ops for n in dict.fromkeys(o["outputs"]) if n.startswith("L")]
    assert all(n.startswith(f"L{o['layer']}.") for o in ops for n in o["outputs"] if n.startswith("L"))
    assert len(set(names)) == 27 * 8 + 2 + 26 * 4  # 8 common (c_kvn, q_rope gone with MLA_PREP, D11b) + (h, down_part) dense / (logits, routing, h, expert_out) MoE
    print("\n" + counts(g))


def test_default_graph():
    """The repository default (2026-09-15) = the session-8 measured best, the DAG + L6 folds: counts equal design §2.4-2.5, Fleet's
    event model holds, the tiles are the chain's minus the folded ops, and Mirage's own build of it (session 8 g1) had these counts."""
    assert {k: DIMS[k] for k in DEFAULT} == DEFAULT
    g, chain = build(), build(**CHAIN)
    check_events(g)
    tasks, events, ops = g["tasks"], g["events"], g["ops"]
    per_layer, per_layer_ops = Counter(t["layer"] for t in tasks), Counter(o["layer"] for o in ops)
    first = {}  # event -> layer of its first producer task
    for t in tasks[2:]:
        first.setdefault(t["trigger_event"], t["layer"])
    ev_layer = Counter(first.values())
    assert (per_layer[0], per_layer_ops[0], ev_layer[0]) == (DOC_DENSE_LAYER_TASKS, DOC_DENSE_LAYER_OPS, DOC_DENSE_LAYER_EVENTS)
    assert all((per_layer[l], per_layer_ops[l], ev_layer[l]) == (DOC_MOE_LAYER_TASKS, DOC_MOE_LAYER_OPS, DOC_MOE_LAYER_EVENTS) for l in range(1, 27))
    assert per_layer[-1] == 2 + 1 + DOC_HEAD_TASKS and per_layer_ops[-1] == 4 and ev_layer[-1] == 4
    assert (len(tasks) - 2, len(ops), len(events)) == (DOC_PER_TOKEN_TASKS, DOC_PER_TOKEN_OPS, DOC_PER_TOKEN_EVENTS)
    assert (len(tasks), len(events) + MIRAGE_EXTRA_EVENTS) == (DOC_MIRAGE_TASKS, DOC_MIRAGE_EVENTS)
    grid = Counter((t["layer"], t["type"]) for t in tasks)
    assert all({typ: grid[l, typ] for typ in DOC_MOE_GRID} == DOC_MOE_GRID for l in range(1, 27))
    assert sum(DOC_MOE_GRID.values()) == DOC_MOE_LAYER_TASKS and {typ: grid[0, typ] for typ in DOC_DENSE_GRID} == DOC_DENSE_GRID
    assert not any(t["type"] in ("RMS_NORM", "ATTN_MERGE", "QKV_PROJ") for t in tasks)
    # same tiles as the chain minus its RMS_NORM / ATTN_MERGE tasks (Q/KVA = QKV rows, shared expert = slot 6)
    base = {"Q_PROJ": "QKV_PROJ", "KVA_PROJ": "QKV_PROJ", "EXPERT_W13_SHARED": "EXPERT_W13_SILU", "EXPERT_W2_SHARED": "EXPERT_W2"}
    tile = lambda t: (t["layer"], base.get(t["type"], t["type"]), t.get("row0"), t.get("rows"), t.get("slot"), t.get("head"), t.get("chunk"), t.get("k_split"))
    assert Counter(map(tile, tasks[2:])) == Counter(map(tile, (t for t in chain["tasks"][2:] if t["type"] not in ("RMS_NORM", "ATTN_MERGE"))))
    # DAG structure per MoE layer (12a): producers, fan-out, multi-producer trigger counts; L6 prologue inputs
    for l in range(1, 27):
        mine = [o for o in ops if o["layer"] == l]
        assert {o["type"]: [ops[i]["type"] for i in o["producers"] if ops[i]["layer"] == l] for o in mine} == DOC_MOE_PRODUCERS
        assert mine[0]["producers"] == mine[1]["producers"] == [[o for o in ops if o["layer"] == l - 1][-1]["id"]]  # the previous layer's last op
        e = {o["type"]: tasks[o["first_task_id"]]["dependent_event"] for o in mine}
        assert e["Q_PROJ"] == e["KVA_PROJ"] and e["MOE_GATE_TOPK"] == e["EXPERT_W13_SHARED"]  # fan-out
        assert [events[e[k]]["num_triggers"] for k in ("Q_PROJ", "ATTN_CHUNK", "MOE_GATE_TOPK", "MOE_COMBINE_RESIDUAL")] == [1, 48 + 64, 32, 192 + 32]
        o = {x["type"]: x["inputs"] for x in mine}
        assert o["Q_PROJ"][1] == o["KVA_PROJ"][1] == f"L{l}.input_layernorm" and o["O_PROJ_RESIDUAL"][0] == f"L{l}.attn_part"
        assert o["MOE_GATE_TOPK"][:2] == o["EXPERT_W13_SHARED"][:2] == [f"L{l}.x_attn", f"L{l}.post_attention_layernorm"]
    assert [o["inputs"][:2] for o in ops if o["type"] == "LM_HEAD"] == [["L26.x", "model.norm"]]
    # tile geometry and slot tickets as the chain (8e, 8c, 11a, Q-H, 10a)
    assert all(t["rows"] == 64 for t in tasks if t["type"] in ("Q_PROJ", "KVA_PROJ", "KV_B_PROJ", "O_PROJ_RESIDUAL", "GATE_UP_SILU",
                                                                 "DOWN_SPLITK", "EXPERT_W2", "EXPERT_W2_SHARED"))
    assert all(t["rows"] == 88 for t in tasks if t["type"] == "EXPERT_W13_SILU") and all(t["rows"] == 176 for t in tasks if t["type"] == "EXPERT_W13_SHARED")
    assert sum(1 for t in tasks if t["type"] == "LM_HEAD") == 200 and all(t["rows"] == 512 for t in tasks if t["type"] == "LM_HEAD")
    assert sum(1 for t in tasks if "slot" in t and not t.get("shared")) == 26 * 6 * 64 and sum(1 for t in tasks if t.get("shared")) == 26 * 64
    names = {n for o in ops for n in o["outputs"] if n.startswith("L")}
    assert len(names) == 27 * 7 + 26 * 6 + 1  # (x_attn, attn_part, x, kv_cache, kva, h, q) + MoE (logits, routing, xn2, expert_out, h_s, expert_out_s) + down_part
    # Mirage's build of the default on the box (session 8 g1; results/ is local evidence, absent in a fresh clone)
    for f in glob.glob(os.path.join(ROOT, "results", "perf", "dag_g1_layers", "*.json")):
        m = json.load(open(f))["graph"]
        assert (m["tasks"], m["ops"], m["events"]) == (DOC_MIRAGE_TASKS, DOC_PER_TOKEN_OPS, DOC_MIRAGE_EVENTS), f
        assert all(DIMS[k] == v for k, v in m["knobs"].items()), m["knobs"]
    print("\n" + counts(g))


# Plain DAG counts (dag=1, fold=0: design D12a, docs/gpu-log.md session 8 step s1): same 20,979 tasks and tiles as the chain;
# QKV_PROJ -> Q_PROJ + KVA_PROJ, the shared expert as its own w13/w2 ops. Per MoE layer 14 ops / 12 events, layer 0 11 / 10, head + embedding 5 / 5, + END.
DAG_MOE_LAYER_OPS, DAG_MOE_LAYER_EVENTS, DAG_DENSE_LAYER_OPS, DAG_PER_TOKEN_OPS, DAG_PER_TOKEN_EVENTS = 14, 12, 11, 380, 328
DAG_MOE_PRODUCERS = {"Q_PROJ": ["RMS_NORM"], "KVA_PROJ": ["RMS_NORM"], "KV_B_PROJ": ["KVA_PROJ"],
                     "ATTN_CHUNK": ["Q_PROJ", "KV_B_PROJ"], "ATTN_MERGE": ["ATTN_CHUNK"], "O_PROJ_RESIDUAL": ["ATTN_MERGE"],
                     "MOE_GATE_TOPK": ["RMS_NORM"], "EXPERT_W13_SHARED": ["RMS_NORM"], "EXPERT_W13_SILU": ["MOE_GATE_TOPK"],
                     "EXPERT_W2_SHARED": ["EXPERT_W13_SHARED"], "EXPERT_W2": ["EXPERT_W13_SILU"],
                     "MOE_COMBINE_RESIDUAL": ["EXPERT_W2_SHARED", "EXPERT_W2"], "RMS_NORM": ["O_PROJ_RESIDUAL"]}  # op-id order; the 2nd norm


def check_events(g):
    """Fleet's worker model on any graph: every launched task waits on ONE event and triggers ONE event; each event's
    num_triggers == the tasks that trigger it (END included); an event launches exactly the tasks that wait on it, as one
    contiguous range (fan-out consumers adjacent); the producers of an op's event are the op's derived producers and all
    come earlier (acyclic)."""
    tasks, events, ops = g["tasks"], g["events"], g["ops"]
    trig = Counter(t["trigger_event"] for t in tasks[1:])
    assert all(t["trigger_event"] is not None and t["dependent_event"] is not None for t in tasks[2:])
    for e in events:
        assert trig[e["id"]] == e["num_triggers"], e
        if e["type"] != "END_OF_TASK_GRAPH":
            launched = tasks[e["first_task_id"]:e["last_task_id"]]
            assert launched and all(t["dependent_event"] == e["id"] for t in launched), e
    assert sum(e["last_task_id"] - e["first_task_id"] for e in events) == len(tasks) - 2
    prod_ops = {e["id"]: sorted({t["op"] for t in tasks[2:] if t["trigger_event"] == e["id"]}) for e in events}
    for o in ops:
        dep = {t["dependent_event"] for t in tasks[o["first_task_id"]:o["last_task_id"]]}
        assert len(dep) == 1, o
        p = prod_ops[dep.pop()]
        assert p == o["producers"] or (not o["producers"] and o["id"] == 0), (o["type"], o["layer"], p, o["producers"])
        assert all(i < o["id"] for i in p)
    for t in tasks[2:]:
        assert t["dependent_event"] < t["trigger_event"]


def test_dag():
    chain, g = build(**CHAIN), build(dag=1, fold=0)
    check_events(chain)
    check_events(g)
    assert all(o["producers"] == [o["id"] - 1] for o in chain["ops"][1:])  # the derived producers of the chain are the chain
    tasks, ops = g["tasks"], g["ops"]
    assert len(tasks) == len(chain["tasks"]) and len(ops) == DAG_PER_TOKEN_OPS and len(g["events"]) == DAG_PER_TOKEN_EVENTS
    tile = lambda t: (t["layer"], t["type"].replace("Q_PROJ", "QKV_PROJ").replace("KVA_PROJ", "QKV_PROJ").replace("_SHARED", "_SILU" if "W13" in t["type"] else ""),
                      t.get("row0"), t.get("rows"), t.get("slot"), t.get("head"), t.get("chunk"), t.get("k_split"))
    assert Counter(map(tile, tasks[2:])) == Counter(map(tile, chain["tasks"][2:]))
    per_layer_ops = Counter(o["layer"] for o in ops)
    ev_layer = Counter(tasks[i]["layer"] for i in {min(t["id"] for t in tasks[2:] if t["trigger_event"] == e["id"]) for e in g["events"][1:]})
    assert per_layer_ops[0] == DAG_DENSE_LAYER_OPS and ev_layer[0] == DAG_DENSE_LAYER_OPS - 1
    for l in range(1, 27):
        mine = [o for o in ops if o["layer"] == l]
        assert len(mine) == DAG_MOE_LAYER_OPS and ev_layer[l] == DAG_MOE_LAYER_EVENTS
        assert {o["type"]: [ops[i]["type"] for i in o["producers"]] for o in mine[1:]} == DAG_MOE_PRODUCERS
        e = {o["type"]: tasks[o["first_task_id"]]["dependent_event"] for o in mine}
        assert e["Q_PROJ"] == e["KVA_PROJ"] and e["MOE_GATE_TOPK"] == e["EXPERT_W13_SHARED"]  # fan-out
        assert g["events"][e["ATTN_CHUNK"]]["num_triggers"] == 48 + 64 and g["events"][e["MOE_COMBINE_RESIDUAL"]]["num_triggers"] == 192 + 32


def test_chunks_scale_with_t_max():
    for kw in ({}, CHAIN):
        g = build(t_max=128, **kw)
        assert sum(1 for t in g["tasks"] if t["type"] == "ATTN_CHUNK") == 27 * 16
