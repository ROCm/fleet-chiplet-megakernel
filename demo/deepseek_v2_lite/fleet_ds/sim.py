"""CPU simulator of Fleet's runtime over the per-token task graph (design §8).

Graph = fleet_ds.graph.build() (it carries buffer names, per-task metadata and event ids; mirage_graph.layer_plan
only has grids and is asserted equal to it by tests/test_mirage_graph.py). Buffer arena = dict name -> torch
tensor named as mirage_graph.build() names its tensors (design.md 4.1/4.3, relayout_layer for weights).

What is mirrored: every task of the token pre-assigned to a worker queue (round-robin like Fleet's scheduler on
LAUNCH_DEPENDENT_TASKS, or a seeded shuffle); a worker runs its queue in order, blocking on
all_event_counters[dependent_event] >= num_triggers; after the task it increments trigger_event; END_OF_TASK_GRAPH = prepare_next_batch (tokens[slot+1] = output_token). Each task is the real
kernel (kernels/*.cuh under the CPU shim, via tests/cpu_harness.cc + ctypes) called with the pointer arithmetic
of kernels/fleet_dispatch.cuh (bid -> pre-offset pointers; expert tasks read idx[slot] from the arena at run time, decision 10a).
Hang = no runnable task anywhere -> SimHang names the event. Data readiness (decision 12a DAG): before a task runs, every
earlier op that writes one of its input buffers must be complete (asserted from buffer names, independent of the events).
Not mirrored: XCD-local counters/fences, LDS limits, atomics, timing.
"""
import ctypes
import os
import random
import subprocess
import tempfile
from collections import defaultdict

import torch

from . import reference as R
from .graph import DIMS

H, DN, DR, DC, DV, TMAX, CHUNK = 16, 128, 64, 512, 128, 1152, 128  # baked into cpu_harness.cc
NCHUNK = TMAX // CHUNK
HID, E, TOPK, TILE, LM_ROWS, VCHUNK = DIMS["hidden"], DIMS["n_experts"], DIMS["topk"], DIMS["tile"], DIMS["lm_rows"], DIMS["argmax_chunk"]
MOE_I, SH_I, KSLICE, SPLITK = DIMS["moe_inter"], DIMS["shared_inter"], DIMS["dense_inter"] // DIMS["split_k"], DIMS["split_k"]
EPS = 1e-6
BF = torch.bfloat16
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNELS = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "include", "mirage", "persistent_kernel", "kernels")
CPU_FLAGS = ["-DFLEET_CPU_SHIM", "-I" + KERNELS, "-I" + os.path.join(ROOT, "tests")]  # cpu_shim.h sits next to the tests


def build_harness(out=None):
    """clang++ tests/cpu_harness.cc under the CPU shim -> ctypes.CDLL (same build as tests/test_kernels_cpu.py)."""
    out = out or os.path.join(tempfile.mkdtemp(prefix="fleet_sim_"), "k.so")
    subprocess.check_call(["clang++", "-x", "c++", "-std=c++17", "-O2", "-fPIC", "-shared", *CPU_FLAGS,
                           os.path.join(ROOT, "tests", "cpu_harness.cc"), "-o", out])
    return ctypes.CDLL(out)


def ptr(t, off=0):
    """Element-offset pointer: the pre-offset base the runtime hands a task."""
    return ctypes.c_void_p(t.data_ptr() + off * t.element_size())


def make_arena(layers, glob, cache_k, cache_v):
    """layers: relayout_layer() dicts; glob: embed_tokens, lm_head, model.norm, rope_cos/rope_sin [TMAX,64] bf16,
    tokens int64 [TMAX]; cache_k[l] [16,TMAX,192], cache_v[l] [16,TMAX,128]. Scratch as design.md 4.3."""
    a = {"embed_tokens": glob["embed_tokens"], "lm_head": glob["lm_head"], "model_norm": glob["model.norm"],
         "rope_cos": glob["rope_cos"], "rope_sin": glob["rope_sin"], "tokens": glob["tokens"],
         "x0": torch.zeros(HID, dtype=BF), "xn_final": torch.zeros(HID, dtype=BF),
         "logits": torch.zeros(DIMS["vocab"]), "argmax_part_value": torch.zeros(DIMS["vocab"] // VCHUNK),
         "argmax_part_index": torch.zeros(DIMS["vocab"] // VCHUNK, dtype=torch.int64),
         "output_token": torch.zeros(1, dtype=torch.int64)}
    for l, w in enumerate(layers):
        b = lambda n: f"L{l}.{n}"
        a.update({b(k): v for k, v in w.items()})
        a[b("K")], a[b("V")] = cache_k[l], cache_v[l]
        for n, shape, dt in (("xn", HID, BF), ("qkva", DIMS["q_rows"] + DIMS["kv_a_rows"], BF), 
                             ("part_o", (NCHUNK, H, DV), torch.float32),
                             ("lse", (NCHUNK, H), torch.float32), ("attn_o", (H, DV), BF), ("x_attn", HID, BF),
                             ("xn2", HID, BF), ("x", HID, BF)):
            a[b(n)] = torch.zeros(shape, dtype=dt)
        if "mlp.gate_up" in w:  # dense
            a[b("h")] = torch.zeros(DIMS["dense_inter"], dtype=BF)
            a[b("down_part")] = torch.zeros(SPLITK, HID)
        else:
            a[b("logits")] = torch.zeros(E)
            a[b("topk_w")], a[b("topk_idx")] = torch.zeros(TOPK), torch.zeros(TOPK, dtype=torch.int32)
            a[b("h")] = torch.zeros(E * MOE_I + SH_I, dtype=BF)
            a[b("expert_out")] = torch.zeros(E + 1, HID, dtype=BF)
    return a


def run_task(K, t, a, slot, T, scale):
    """One TaskDesc -> one kernel call. Pointers = kernels/fleet_dispatch.cuh's arithmetic from the task's bid (metadata)."""
    typ, l = t["type"], t["layer"]
    b = lambda n: a[f"L{l}.{n}"]
    x_in = "x0" if l == 0 else f"L{l - 1}.x"
    r0 = t.get("row0", 0)
    if typ == "EMBEDDING":
        K.k_embed(ptr(a["embed_tokens"]), ptr(a["tokens"]), slot, ptr(a["x0"]))
    elif typ in ("QKV_PROJ", "Q_PROJ", "KVA_PROJ") and len(t["inputs"]) == 3:  # L6 (12e): input norm as the tiles' prologue
        assert K.k_gemv_norm(ptr(a[x_in]), ptr(b("input_layernorm")), ptr(b("W_qkva"), r0 * HID), ptr(b("qkva"), r0), None,
                             HID, t["rows"], 0, 0, ctypes.c_float(EPS)) == 0
    elif typ == "RMS_NORM":
        src = a[t["inputs"][0]]  # graph.py residual names (x0 / L{l}.x / L{l}.x_attn) == arena names
        w, out = (a["model_norm"], a["xn_final"]) if l < 0 else (b("post_attention_layernorm"), b("xn2")) \
            if t["inputs"][0].endswith(".x_attn") else (b("input_layernorm"), b("xn"))
        K.k_rmsnorm(ptr(src), ptr(w), ptr(out), HID, ctypes.c_float(EPS))
    elif typ in ("QKV_PROJ", "Q_PROJ", "KVA_PROJ"):  # DAG (12a): the q / kv_a rows of the same W_qkva tiles
        assert K.k_gemv(ptr(b("xn")), ptr(b("W_qkva"), r0 * HID), ptr(b("qkva"), r0), None, HID, t["rows"], 0, 0) == 0
    elif typ == "KV_B_PROJ":  # tile n = bid.x: a head's 256 rows are tph tiles; sub < tph/2 -> K[h][slot][0:128], else V[h][slot];
        rows = t["rows"]      # sub == 0 writes k_pe (11b); rows 64 (tph 4) measured, 32 / 16 = L4 (12e)
        tph, n = (DN + DV) // rows, r0 // rows
        h, sub = n // tph, n % tph
        out = ptr(b("K"), (h * TMAX + slot) * (DN + DR) + sub * rows) if sub < tph // 2 else \
            ptr(b("V"), (h * TMAX + slot) * DV + (sub - tph // 2) * rows)
        assert K.k_kv_b_proj_rows(ptr(b("qkva")), ptr(b("kv_a_layernorm")), ptr(b("kv_b_proj"), n * rows * DC), ptr(a["rope_cos"]),
                                  ptr(a["rope_sin"]), out, ptr(b("K")), n, slot, slot, ctypes.c_float(EPS), rows) == 0
    elif typ == "ATTN_CHUNK":  # q RoPE prologue (11b)
        K.k_attn_chunk_rope(ptr(b("qkva")), ptr(b("K")), ptr(b("V")), ptr(b("part_o")), ptr(b("lse")), t["head"], t["chunk"],
                            T, ctypes.c_float(scale), ptr(a["rope_cos"]), ptr(a["rope_sin"]), slot)
    elif typ == "ATTN_MERGE":
        K.k_attn_merge(ptr(b("part_o")), ptr(b("lse")), ptr(b("attn_o")), t["head"])
    elif typ == "O_PROJ_RESIDUAL" and t["inputs"][0].endswith(".attn_part"):  # L6: LSE merge prologue (perf/fold-full harness)
        assert K.k_o_proj_merge_rows(ptr(b("part_o")), ptr(b("lse")), ptr(b("o_proj"), r0 * HID), ptr(b("x_attn"), r0),
                                     ptr(a[x_in], r0), T, t["rows"]) == 0  # rows 64, L4 32 / 16 (perf/kernels export)
    elif typ == "O_PROJ_RESIDUAL":
        assert K.k_gemv(ptr(b("attn_o")), ptr(b("o_proj"), r0 * HID), ptr(b("x_attn"), r0), ptr(a[x_in], r0), HID, t["rows"], 1, 0) == 0
    elif typ == "GATE_UP_SILU" and len(t["inputs"]) == 3:  # L6: norm2 prologue
        assert K.k_gemv_norm(ptr(b("x_attn")), ptr(b("post_attention_layernorm")), ptr(b("mlp.gate_up"), r0 * HID),
                             ptr(b("h"), r0 // 2), None, HID, TILE, 2, 0, ctypes.c_float(EPS)) == 0
    elif typ == "GATE_UP_SILU":  # 64 interleaved rows -> 32 h values
        assert K.k_gemv(ptr(b("xn2")), ptr(b("mlp.gate_up"), r0 * HID), ptr(b("h"), r0 // 2), None, HID, TILE, 2, 0) == 0
    elif typ == "DOWN_SPLITK":  # down_proj [4, 2048, 2736]: y = K-slice, x = N-tile; fp32 partial ws[y][x*64]
        y = t["k_split"]
        assert K.k_gemv(ptr(b("h"), y * KSLICE), ptr(b("mlp.down_proj"), (y * HID + r0) * KSLICE),
                        ptr(b("down_part"), y * HID + r0), None, KSLICE, TILE, 0, 1) == 0
    elif typ == "REDUCE_RESIDUAL":
        K.k_splitk_reduce(ptr(b("down_part")), ptr(b("x_attn")), ptr(b("x")), HID)
    elif typ == "MOE_GATE_TOPK" and len(t["inputs"]) == 3:  # L6: norm2 prologue, publishes xn2
        K.k_moe_gate_topk_norm(ptr(b("x_attn")), ptr(b("post_attention_layernorm")), ptr(b("mlp.gate")), ptr(b("logits")),
                               ptr(b("topk_w")), ptr(b("topk_idx")), ptr(b("xn2")), ctypes.c_float(EPS))
    elif typ == "MOE_GATE":  # L5 (12e): the gate GEMV as fp32 tiles of its own op
        assert K.k_gemv(ptr(b("xn2")), ptr(b("mlp.gate"), r0 * HID), ptr(b("logits"), r0), None, HID, t["rows"], 0, 1) == 0
    elif typ == "MOE_TOPK":  # L5: top-k over the gate op's logits
        K.k_moe_topk(ptr(b("logits")), ptr(b("topk_w")), ptr(b("topk_idx")))
    elif typ == "MOE_GATE_TOPK":  # one task: gate GEMV (fp32 logits) + top-k (decision 10c)
        K.k_moe_gate_topk(ptr(b("xn2")), ptr(b("mlp.gate")), ptr(b("logits")), ptr(b("topk_w")), ptr(b("topk_idx")))
    elif typ in ("EXPERT_W13_SILU", "EXPERT_W13_SHARED"):  # slot = bid.x, tile = bid.y; e = idx[slot] read at run time (10a); slot 6 = shared; rows per tile from the graph (11a)
        slot, rows = t["slot"], t["rows"]
        tile = r0 // rows
        if slot < TOPK:
            e = int(b("topk_idx")[slot])  # the pointer arithmetic fleet_dispatch.cuh::ds_expert_w13 does on the device
            w, out = ptr(b("mlp.w13"), (e * 2 * MOE_I + tile * rows) * HID), ptr(b("h"), e * MOE_I + tile * rows // 2)
        else:
            w, out = ptr(b("mlp.w13_s"), tile * rows * HID), ptr(b("h"), E * MOE_I + tile * rows // 2)
        if len(t["inputs"]) == 3 and typ == "EXPERT_W13_SHARED":  # L6: norm2 prologue on the shared tiles
            assert K.k_gemv_norm(ptr(b("x_attn")), ptr(b("post_attention_layernorm")), w, out, None, HID, rows, 2, 0, ctypes.c_float(EPS)) == 0
        else:
            assert K.k_gemv(ptr(b("xn2")), w, out, None, HID, rows, 2, 0) == 0
    elif typ in ("EXPERT_W2", "EXPERT_W2_SHARED"):
        slot, rows = t["slot"], t["rows"]
        tile = r0 // rows
        if slot < TOPK:
            e = int(b("topk_idx")[slot])
            w, x, Kd = ptr(b("mlp.w2"), (e * HID + tile * rows) * MOE_I), ptr(b("h"), e * MOE_I), MOE_I
        else:
            e = E
            w, x, Kd = ptr(b("mlp.w2_s"), tile * rows * SH_I), ptr(b("h"), E * MOE_I), SH_I
        assert K.k_gemv(x, w, ptr(b("expert_out"), e * HID + tile * rows), None, Kd, rows, 0, 0) == 0
    elif typ == "MOE_COMBINE_RESIDUAL":
        K.k_moe_combine(ptr(b("expert_out")), ptr(b("topk_w")), ptr(b("topk_idx")), ptr(b("x_attn")), ptr(b("x")))
    elif typ == "LM_HEAD" and len(t["inputs"]) == 3:  # L6: model norm prologue
        assert K.k_gemv_norm(ptr(a[t["inputs"][0]]), ptr(a["model_norm"]), ptr(a["lm_head"], r0 * HID), ptr(a["logits"], r0),
                             None, HID, LM_ROWS, 0, 1, ctypes.c_float(EPS)) == 0
    elif typ == "LM_HEAD":  # fp32 logits: our argmax_partial fallback reads float (Fleet's reads the bf16 the graph declares)
        assert K.k_gemv(ptr(a["xn_final"]), ptr(a["lm_head"], r0 * HID), ptr(a["logits"], r0), None, HID, LM_ROWS, 0, 1) == 0
    elif typ == "ARGMAX_PARTIAL":
        c = r0 // VCHUNK
        K.k_argmax_partial(ptr(a["logits"], r0), ptr(a["argmax_part_value"], c), ptr(a["argmax_part_index"], c))
    elif typ == "ARGMAX_REDUCE":
        K.k_argmax_reduce(ptr(a["argmax_part_value"]), ptr(a["argmax_part_index"]), ptr(a["output_token"]))
    elif typ != "BEGIN_TASK_GRAPH":
        raise ValueError(typ)


def replay(g, task_us, barrier_us, n_workers=296, fifo=True):
    """Timed replay of Fleet's placement over graph g. Position p = task id goes to
    stripe (p - 2) mod n_workers; the XCD scheduler's carried round-robin index only relabels workers inside the XCD, so the
    stripe decides which tasks share one FIFO queue. A task starts at max(the end of the previous task on its queue [fifo],
    its dependent event + barrier_us); an event fires when its last producer task ends. task_us[id] = run time of the task.
    Returns (token us = END firing, head-of-line waits {(layer, op type): [tasks delayed, max delay us, {types ahead}]},
    peak MoE-layer tasks ready or running at one instant = the worker demand of layers >= 1; > n_workers means some must
    queue; layer 0's 342-task GATE_UP_SILU wraps in every graph and is left out):
    a wait is recorded when the queue's previous task (a straggler of another op, or the same op wrapping) ends after
    this task's event + barrier."""
    prod = defaultdict(list)
    for t in g["tasks"][2:]:
        prod[t["trigger_event"]].append(t["id"])
    fire, end, free, last, waits, edges = {0: 0.0}, {}, [0.0] * n_workers, [None] * n_workers, {}, []
    for t in g["tasks"][2:]:  # id order is topological (producers precede consumers)
        e = t["dependent_event"]
        if e not in fire:
            fire[e] = max(end[i] for i in prod[e])
        w = (t["id"] - 2) % n_workers
        ready = fire[e] + barrier_us
        start = max(free[w], ready) if fifo else ready
        if start > ready + 1e-9:
            rec = waits.setdefault((t["layer"], t["type"]), [0, 0.0, set()])
            rec[0], rec[1] = rec[0] + 1, max(rec[1], start - ready)
            rec[2].add(g["tasks"][last[w]]["type"])
        end[t["id"]] = start + task_us[t["id"]]
        free[w], last[w] = max(free[w], end[t["id"]]), t["id"]
        if t["layer"] >= 1:
            edges += [(ready, 1), (end[t["id"]], -1)]
    peak, n = 0, 0
    for _, dn in sorted(edges):  # ends sort before starts at the same instant
        n += dn
        peak = max(peak, n)
    return max(end[i] for i in prod[g["events"][-1]["id"]]), waits, peak


class SimHang(RuntimeError):
    def __init__(self, event, counter, need, stuck):
        self.event, self.counter, self.need, self.stuck = event, counter, need, stuck
        t = stuck[0]
        super().__init__(f"hang: no runnable task; {len(stuck)} stuck, first task {t['id']} {t['type']} L{t['layer']} "
                         f"waits on event {event} (counter {counter} < num_triggers {need})")


def simulate(g, arena, K, cfg, slot, n_workers=8, seed=None, only_layer=None):
    """Run one token of graph g on the arena. seed=None: round-robin assignment and worker polling (Fleet);
    seed=int: shuffled assignment and polling order (order-independence check). only_layer=l: run that layer's
    ops alone (their first dependent event is treated as already fired). Returns the trace [(task_id, worker)]."""
    ev = g["events"]
    tasks = [t for t in g["tasks"] if t["type"] != "TERMINATE" and (only_layer is None or t["layer"] == only_layer)]
    rng = random.Random(seed) if seed is not None else None
    queues = [[] for _ in range(n_workers)]
    for i, t in enumerate(tasks):
        queues[rng.randrange(n_workers) if rng else i % n_workers].append(t)
    counters = defaultdict(int)
    triggered = {t["trigger_event"] for t in tasks}
    for t in tasks:  # ops whose producer is outside the run (only_layer): dependency already satisfied
        if t["dependent_event"] is not None and t["dependent_event"] not in triggered:
            counters[t["dependent_event"]] = ev[t["dependent_event"]]["num_triggers"]
    T, scale, heads, trace, left = slot + 1, R.softmax_scale(cfg), [0] * n_workers, [], len(tasks)
    todo = defaultdict(int)  # op -> tasks not yet run; ops outside the run count as done
    for t in tasks:
        todo[t["op"]] += 1
    writers = defaultdict(list)
    for o in g["ops"]:
        for n in dict.fromkeys(o["outputs"]):
            writers[n].append(o["id"])
    while left:
        progressed = False
        order = list(range(n_workers))
        if rng:
            rng.shuffle(order)
        for w in order:
            if heads[w] == len(queues[w]):
                continue
            t = queues[w][heads[w]]
            d = t["dependent_event"]
            if d is not None and counters[d] < ev[d]["num_triggers"]:
                continue
            late = [o for n in t["inputs"] for o in writers[n] if o < t["op"] and todo[o]]
            assert not late, f"task {t['id']} {t['type']} L{t['layer']} runs before producer op(s) {late} finished"
            run_task(K, t, arena, slot, T, scale)
            todo[t["op"]] -= 1
            heads[w] += 1
            counters[t["trigger_event"]] += 1
            trace.append((t["id"], w))
            progressed, left = True, left - 1
        if not progressed:
            stuck = sorted((queues[w][heads[w]] for w in range(n_workers) if heads[w] < len(queues[w])), key=lambda t: t["id"])
            d = stuck[0]["dependent_event"]
            raise SimHang(d, counters[d], ev[d]["num_triggers"], stuck)
    end = ev[tasks[-1]["trigger_event"]]
    assert counters[end["id"]] == end["num_triggers"], (counters[end["id"]], end)
    if end["type"] == "END_OF_TASK_GRAPH":  # prepare_next_batch: output token becomes the next input
        arena["tokens"][slot + 1] = arena["output_token"][0]
    return trace
