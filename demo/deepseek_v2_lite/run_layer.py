#!/usr/bin/env python
"""M1-M3 evidence: ONE decode token through layers [a..b] of the Fleet megakernel vs the HF reference dumps.

  run_layer.py --layers 1            # M2: layer 1 alone           -> results/m2/layer_1-1.json
  run_layer.py --layers 1-3          # M3: consecutive layers      -> results/m3/layer_1-3.json
  run_layer.py --layers 0-26         # M3: the whole token         -> results/m3/layer_0-26.json (+ logits)
  run_layer.py --layers 1 --torch-fallback   # A/B: same arena through fleet_ds.reference on the GPU
  run_layer.py --layers 1-3 --dry-run        # laptop: plan + arena shapes, no ROCm, no model

Inputs: results/m0/reference/ (reference_prefill.py: prefill cache, per-layer hidden states of decode
step 0, final cache, logits) and models/DeepSeek-Coder-V2-Lite-Base/ (safetensors, mmap'd per layer).
Graph: fleet_ds.mirage_graph.build() for the layer subset; scratch buffers are torch tensors we allocate
(new_tensor is redirected to attach_input) so every per-op output can be read back after the launch.
HF's layer-a input enters through the graph's own EMBEDDING op: the "embedding table" is a 2-row tensor
whose row 0 is the hidden state, and the decoded token id is 0. The chain (embed -> layers -> norm ->
lm_head -> argmax) is therefore the production graph, just shorter.
One iteration: PersistentKernel(max_seq_length = pos + 2) makes Fleet's prepare_next_batch stop after one
token (it stops when step + 2 >= max_seq_length). Fleet admits the request from tokens[0] and our kernels
decode at p = step, so tokens[0] = tokens[pos] = the token and step = prompt_lengths = pos.
Launch path (launch()): mpk_debug.c is spliced into Mirage's launcher source at compile time; it
runs the launch on a detached thread under a host watchdog (MPK_WATCHDOG_S, default 60 s: on timeout the
runtime's counters/queues/step are read over a side stream, the stuck event named, the process aborted) and
sets the RUNTIME max_seq_length per launch, so no second init_func is needed. A second init_func used to stall the
next launch (leaked stream pair, docs/gpu-log.md session 4); upstream PR ROCm/fleet-chiplet-megakernel#9
destroys the old pair first, and `--reinit` is now the regression check for it.
Gates: docs/design.md 5.2 (per-op 2 %, per-layer 5 % + cosine 0.999, logits cosine 0.9995).
"""
import argparse
import ctypes
import functools
import json
import os
import subprocess
import sys
import time
import types

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, "ref"), os.path.join(ROOT, "tests")]
REPO = os.path.dirname(os.path.dirname(ROOT))  # the fleet-chiplet-megakernel checkout
os.environ.setdefault("MIRAGE_HOME", REPO)
os.environ.setdefault("AMDGPU_TARGETS", "gfx942")  # PersistentKernel.compile() defaults to gfx950
from fleet_ds import mirage_graph as MG  # noqa: E402
from fleet_ds import reference as R  # noqa: E402
from fleet_ds.graph import DIMS, KNOBS  # noqa: E402

P, HID, H, E, BF = 1024, DIMS["hidden"], DIMS["n_heads"], DIMS["n_experts"], torch.bfloat16
REF_DIR = os.path.join(ROOT, "results", "m0", "reference")
MODEL_DIR = os.path.join(ROOT, "models", "DeepSeek-Coder-V2-Lite-Base")
GATE = dict(op=2e-2, layer=5e-2, layer_cos=0.999, logit_cos=0.9995)  # design.md 5.2
META = ["step", "tokens", "input_tokens", "output_tokens", "num_new_tokens", "prompt_lengths", "qo_indptr_buffer",
        "paged_kv_indptr_buffer", "paged_kv_indices_buffer", "paged_kv_last_page_len_buffer"]  # init_func order
FLEET_TYPES = {101: "EMBEDDING", 119: "RMS_NORM", 132: "REDUCE_RESIDUAL", 110: "ARGMAX_PARTIAL", 111: "ARGMAX_REDUCE"}
WATCHDOG_S = float(os.environ.get("MPK_WATCHDOG_S", "60"))  # launch watchdog (mpk_debug.c)


def parse_layers(s):
    a, _, b = s.partition("-")
    a, b = int(a), int(b or a)
    assert 0 <= a <= b < DIMS["n_layers"], s
    return a, b


def cfg_real():
    return types.SimpleNamespace(**R.COMMON, **R.REAL)


def git_hash():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def metrics(out, ref, gate):
    """design.md 5.2: max|d| <= gate * max|ref| + 1e-3, plus cosine."""
    o, r = out.float().flatten(), ref.float().flatten()
    err, scale = (o - r).abs().max().item(), r.abs().max().item()
    cos = F.cosine_similarity(o, r, dim=0).item() if o.numel() > 1 else float(o.item() == r.item())
    return dict(max_abs=err, max_ref=scale, rel=err / max(scale, 1e-30), cosine=cos, gate=gate, ok=err <= gate * scale + 1e-3)


# ---- weights / reference dumps ---------------------------------------------------------------------------

class Shards:
    """safetensors index -> per-layer HF state_dict, mmap'd (only the requested tensors are read)."""
    def __init__(self, model_dir):
        from safetensors import safe_open
        self.dir, self.open, self.safe_open = model_dir, {}, safe_open
        self.map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]

    def get(self, key):
        f = self.map[key]
        if f not in self.open:
            self.open[f] = self.safe_open(os.path.join(self.dir, f), "pt")
        return self.open[f].get_tensor(key)

    def layer(self, l):
        pre = f"model.layers.{l}."
        return {k[len(pre):]: self.get(k) for k in self.map if k.startswith(pre)}


def load_layers(model_dir, a, b, dev):
    """relayout_layer() dicts on `dev` for layers a..b, one layer resident on the host at a time."""
    sh, out = Shards(model_dir), []
    for l in range(a, b + 1):
        t0 = time.time()
        w = MG.relayout_layer(sh.layer(l), dense=l < DIMS["first_k_dense"])
        out.append({k: v.to(dev) for k, v in w.items()})
        print(f"  layer {l}: {sum(v.numel() * 2 for v in w.values()) / 2**20:.0f} MiB in {time.time() - t0:.1f}s", flush=True)
    return out, sh


def rope_tables(cfg, t_max, dev):
    cs = [R.rope_cos_sin(torch.tensor(float(p)), cfg, BF) for p in range(t_max)]
    return torch.stack([c for c, _ in cs]).to(dev), torch.stack([s for _, s in cs]).to(dev)


def load_ref(ref_dir, a, b, t_max, dev):
    """HF dumps: layer-a input, layer outputs a..b (decode step 0), prefill caches padded to t_max, final rows."""
    from safetensors import safe_open
    ids = json.load(open(os.path.join(ref_dir, "prompt_ids.json")))
    hid = safe_open(os.path.join(ref_dir, "step0_hidden.safetensors"), "pt", device=dev)
    pre = safe_open(os.path.join(ref_dir, "kv_cache_prefill.safetensors"), "pt")
    fin = safe_open(os.path.join(ref_dir, "kv_cache_final.safetensors"), "pt")
    x_in = hid.get_tensor("embed" if a == 0 else f"layer{a - 1}.out")
    outs = {l: hid.get_tensor(f"layer{l}.out") for l in range(a, b + 1)}
    ck, cv, k_row, v_row = [], [], {}, {}
    for l in range(a, b + 1):
        K, V = torch.zeros(H, t_max, 192, dtype=BF, device=dev), torch.zeros(H, t_max, 128, dtype=BF, device=dev)
        K[:, :P], V[:, :P] = pre.get_tensor(f"layer{l}.K").to(dev), pre.get_tensor(f"layer{l}.V").to(dev)
        ck.append(K), cv.append(V)
        k_row[l], v_row[l] = fin.get_slice(f"layer{l}.K")[:, P:P + 1][:, 0].to(dev), fin.get_slice(f"layer{l}.V")[:, P:P + 1][:, 0].to(dev)
    logits0 = safe_open(os.path.join(ref_dir, "logits.safetensors"), "pt", device=dev).get_slice("logits")[0:1][0]
    return dict(x_in=x_in, outs=outs, cache_k=ck, cache_v=cv, k_row=k_row, v_row=v_row, logits0=logits0,
                tok=ids["step_inputs"][0], gen=ids["generated_ids"], final_norm=hid.get_tensor("final_norm"))


# ---- graph + runtime -------------------------------------------------------------------------------------

class _subset:
    """build() plans DIMS['n_layers'] layers from layer 0; run the plan for our slice instead (restored on exit).
    Patches module state; a `plan_kw` passthrough in build() would replace this."""
    def __init__(self, n_layers, first_k_dense):
        self.kw = dict(n_layers=n_layers, first_k_dense=first_k_dense)

    def __enter__(self):
        self.saved = (MG.layer_plan, {k: DIMS[k] for k in self.kw})
        MG.layer_plan = functools.partial(self.saved[0], **self.kw)
        DIMS.update(self.kw)

    def __exit__(self, *_):
        MG.layer_plan = self.saved[0]
        DIMS.update(self.saved[1])


def splice_hardcode(pk):
    """mpk_debug.c into Mirage's launcher source: compile() writes cuda_code + persistent_kernel.HARD_CODE
    (a module global read at call time), so the ref clone stays untouched."""
    if "launch_async_func" in pk.HARD_CODE:
        return
    funcs, entries = open(os.path.join(ROOT, "mpk_debug.c")).read().split("// ---- ModuleMethods entries ----\n")
    anchor = "static PyMethodDef ModuleMethods[] = {\n"
    assert pk.HARD_CODE.count(anchor) == 1, "Mirage HARD_CODE drifted: ModuleMethods anchor not found"
    pk.HARD_CODE = pk.HARD_CODE.replace(anchor, funcs + anchor + entries)


def make_mpk(mi, layers, glob, cache_k, cache_v, t_max, max_seq_length, first_k_dense, dev="cuda", perfetto=None):
    """PersistentKernel + graph for the layer slice. Returns (mpk, arena, meta): arena = every tensor the graph
    touches by mirage_graph name (weights, caches, scratch), meta = Fleet's runtime tensors (step, tokens, ...)."""
    splice_hardcode(sys.modules[mi.PersistentKernel.__module__])
    meta = dict(step=torch.zeros(1, dtype=torch.int32, device=dev),
                tokens=torch.zeros(1, max_seq_length, dtype=torch.int64, device=dev),
                input_tokens=torch.zeros(1, 1, dtype=torch.int64, device=dev),
                output_tokens=torch.zeros(1, 1, dtype=torch.int64, device=dev),
                num_new_tokens=torch.ones(1, dtype=torch.int32, device=dev),
                prompt_lengths=torch.zeros(1, dtype=torch.int32, device=dev),
                qo_indptr_buffer=torch.zeros(2, dtype=torch.int32, device=dev),
                paged_kv_indptr_buffer=torch.zeros(2, dtype=torch.int32, device=dev),
                paged_kv_indices_buffer=torch.zeros(1, dtype=torch.int32, device=dev),
                paged_kv_last_page_len_buffer=torch.zeros(1, dtype=torch.int32, device=dev))
    glob = dict(glob, input_tokens=meta["input_tokens"], output_tokens=meta["output_tokens"])
    workers, scheds = mi.get_configurations_from_gpu(0)
    mpk = mi.PersistentKernel(  # demo_30B_A3B.py constructor (mirage_graph.main)
        mode="offline", world_size=1, mpi_rank=0, num_workers=workers, num_local_schedulers=scheds,
        num_remote_schedulers=0, max_seq_length=max_seq_length, max_num_batched_requests=1, max_num_batched_tokens=1,
        max_num_pages=1, page_size=max_seq_length, eos_token_id=-1, meta_tensors=meta, profiler_tensor=perfetto,
        trace_name="", spec_decode_config=None, use_cutlass_kernel=False)
    arena, attach = {}, mpk.attach_input
    td = {mi.bfloat16: torch.bfloat16, mi.float32: torch.float32, mi.int32: torch.int32, mi.int64: torch.int64}

    def attach_input(torch_tensor, name=None):
        arena[name] = torch_tensor
        return attach(torch_tensor=torch_tensor, name=name)

    def new_tensor(dims, strides=None, dtype=mi.bfloat16, name=None, io_category="cuda_tensor"):
        return attach_input(torch.zeros(dims, dtype=td[dtype], device=dev), name)  # ours, not Mirage's: readable

    mpk.attach_input, mpk.new_tensor = attach_input, new_tensor
    with _subset(len(layers), first_k_dense):
        MG.build(mpk, mi, layers, glob, cache_k, cache_v, t_max=t_max)
    return mpk, arena, meta


def reinit(mpk, meta, max_seq_length):
    """A SECOND init_func (re-mallocs the descriptors, re-reads task_graph.json, creates new streams). Without
    upstream PR #9 (not in this branch) the launch after it never returned (the leaked stream pair exhausts HIP's
    hardware queues, docs/gpu-log.md session 4); the patch destroys the old pair first. Kept for `run_layer.py --reinit`
    (regression check, under the watchdog). launch(max_seq_length=...) sets the runtime bound without it."""
    mpk.init_func([meta[k].data_ptr() for k in META], mpk.profiler_tensor.data_ptr() if mpk.profiler_tensor is not None else 0,
                  0, mpk.num_workers, mpk.num_local_schedulers, 0, max_seq_length, 1, mpk.eos_token_id)


def launch(mpk, meta, pos, tok, max_seq_length=None):
    """One mpk() call: decodes tokens at positions pos.. until step + 2 >= max_seq_length (persistent_kernel.cuh
    prepare_next_batch; the RUNTIME bound, set per launch through mpk_debug.c — no second
    init_func needed; before persistent_kernel_cuh.patch one stalled the next launch: docs/gpu-log.md session 4). None keeps the previous bound. Returns the
    launch call's wall ms (prepare_kernel + every iteration). A host watchdog (MPK_WATCHDOG_S, default 60 s)
    dumps the runtime state and aborts if the kernel never returns."""
    ext = mpk.init_func.__self__  # the compiled launcher module: mpk_debug.c's functions live there too
    mpk.init_request_func()  # init_kernel: step = 0, request slots free, next_request_id = 0
    if max_seq_length is not None:
        ext.set_max_seq_length_func(max_seq_length)
        ext.reset_event_timing_func()  # a fresh MPK_EVENT_TIMING buffer for this launch
    meta["step"].fill_(pos)
    meta["prompt_lengths"].fill_(pos)
    meta["tokens"][0, 0] = tok  # admission path embeds tokens[0] (prepare_next_batch step 3)
    meta["tokens"][0, pos] = tok  # what the position semantics say; kept for readers of tokens[]
    torch.cuda.synchronize()
    if mpk.profiler_tensor is not None:  # the perfetto export lives in mpk.__call__: blocking, no watchdog
        t0 = time.time()
        mpk()
        torch.cuda.synchronize()
        ms = (time.time() - t0) * 1e3
    else:
        ext.launch_async_func(torch.cuda.current_stream().cuda_stream)
        deadline = time.time() + WATCHDOG_S
        while (ms := ext.launch_poll_func()) < 0:
            if time.time() > deadline:
                watchdog_abort(mpk, meta, pos)
            time.sleep(0.0005)
    # Fleet's host printf banner and the device printf lines sit in C stdio, which is fully buffered under `tee`
    # and flushed only at exit: without this they land AFTER our PASS line and read as "launched, then silence".
    ctypes.CDLL(None).fflush(None)
    return ms


def watchdog_abort(mpk, meta, pos):
    """The launch did not return within MPK_WATCHDOG_S: read the runtime's host-visible state over a side stream
    (mpk_debug.c debug_state_func), name the event the graph is stuck on, then abort (the spinning
    kernels die with the process)."""
    st = mpk.init_func.__self__.debug_state_func(5.0)
    tg, plan = getattr(mpk, "watchdog", (None, None))
    say = lambda *a: print(*a, flush=True)
    say(f"\n[WATCHDOG] launch did not return within {WATCHDOG_S:.0f} s (pos {pos}); device copies that failed: {st['copy_failures']}")
    for k in ("num_events", "num_workers", "num_schedulers", "max_seq_length", "eos_token_id", "total_num_requests", "step",
              "request_ids", "next_request_id", "qo_indptr", "page_queue_head", "page_queue_tail", "input_tokens", "output_tokens",
              "worker_xcd_ready_count", "event_timing_count", "sched_last_ready", "sched_next_free"):
        say(f"  {k}: {st[k]}")
    nw, wl = st["num_workers"], st["worker_last_ready"]
    if wl:
        say(f"  worker_last_ready (local queues): min {min(wl[:nw])} max {max(wl[:nw])} sum {sum(wl[:nw])}")
    cnt, trig = st["event_counters"], st["num_triggers"]
    ne = len(cnt or [])
    label, prod, order = event_labels(tg, plan) if tg else ({}, {}, list(range(ne)))
    name = lambda e: f"{e} {label.get(e, ('?', '?', ''))[:2]}"
    if cnt and trig:
        k = max(c // t for c, t in zip(cnt, trig) if t)  # iterations the furthest event has completed
        need = [t * (k + 1) for t in trig]
        stuck = [e for e in order if trig[e] and cnt[e] < need[e]]
        say(f"  furthest event completed {k} iteration(s): {ne - len(stuck)} events fired in iteration {k + 1}, {len(stuck)} not yet")
        xl, xt, nx = st["xcd_local_event_counters"], st["xcd_event_num_tasks"], st["num_xcds"]
        for e in stuck[:4]:
            i0, tt = prod.get(e, (None, None))
            say(f"  stuck on event {name(e)} task_type {tt} first task {i0}: count {cnt[e]} / {need[e]} (num_triggers {trig[e]})"
                + (f"; per-XCD local {[xl[x * ne + e] for x in range(nx)]} of {[xt[x * ne + e] for x in range(nx)]}" if xl and xt else ""))
            if tg and i0 is not None:
                dep = tg["all_tasks"][i0]["dependent_event"] & 0xFFFFFFFF
                say(f"    its tasks wait on event {name(dep)}: count {cnt[dep] if dep < ne else '-'}")
    tail = st["event_timing_tail"]
    if tail:
        pairs = list(zip(tail[0::2], tail[1::2]))
        say("  last events fired (event [layer, op] +us): " + ", ".join(f"{name(e)} +{(ts - pts) / 100:.0f}" for (_, pts), (e, ts) in zip(pairs, pairs[1:])))
    if st["copy_failures"] == 0:
        host, side = torch.empty(meta["tokens"].shape, dtype=meta["tokens"].dtype, pin_memory=True), torch.cuda.Stream()
        with torch.cuda.stream(side):
            host.copy_(meta["tokens"], non_blocking=True)
        side.synchronize()
        say(f"  tokens[{pos}..{pos + 8}]: {host[0, pos:pos + 8].tolist()}")
    say("[WATCHDOG] aborting (results not written)")
    ctypes.CDLL(None).fflush(None)
    os._exit(3)


def event_labels(tg, plan):
    """event -> (layer, type, grid) of the op that triggers it, event -> (first producer task, task_type), and the
    events in producer order (BEGIN's event, op 0's, ..., END). Labels shift by one after a gcd split."""
    n_ev = len(tg["all_events"])
    prod = {}
    for i, t in enumerate(tg["all_tasks"]):
        e = t["trigger_event"] & 0xFFFFFFFF
        if 0 <= e < n_ev and e not in prod:
            prod[e] = (i, t["task_type"])
    order = [e for e, _ in sorted(prod.items(), key=lambda kv: kv[1][0])]
    return {e: plan[r - 1] for r, e in enumerate(order) if 1 <= r <= len(plan)}, prod, order


def event_timing(mpk, tg, plan):
    """MPK_EVENT_TIMING buffer -> per iteration: [{event, layer, type, tasks, us}] and iteration_us.
    An entry is (event index, s_memrealtime @100 MHz) written when an event fires = all the tasks that trigger it
    are done. Event k+1 is triggered by op k (one event per op boundary), so the time from event k firing to
    event k+1 firing is op k's span, labelled from the trigger side (after register_mugraph's downgrade only
    event 1 keeps a launch range). BEGIN_TASK_GRAPH triggers the first event of an iteration; the last event is
    END_OF_TASK_GRAPH."""
    import numpy as np
    n_ev = len(tg["all_events"])
    buf = np.zeros(n_ev * 100 * 2, dtype=np.uint64)
    n = mpk.get_event_timing_func(buf.ctypes.data, n_ev * 100)
    entries = sorted(((int(buf[2 * i]), int(buf[2 * i + 1])) for i in range(n)), key=lambda e: e[1])
    if not entries:
        return []
    label, prod, _ = event_labels(tg, plan)
    end = n_ev - 1
    iters, cur = [], []
    for e, ts in entries:
        cur.append((e, ts))
        if e == end:
            iters.append(cur)
            cur = []
    out = []
    for it in iters:
        rows = []
        for (_, ts), (e2, ts2) in zip(it, it[1:]):
            l, typ, grid = label.get(e2, (-1, f"event{e2}", (1, 1, 1)))
            rows.append(dict(event=e2, layer=l, type=typ, task_type=prod.get(e2, (None, None))[1],
                             tasks=grid[0] * grid[1] * grid[2], us=(ts2 - ts) / 100))
        out.append(dict(iteration_us=(it[-1][1] - it[0][1]) / 100, boundaries=rows))
    return out


# ---- checks ----------------------------------------------------------------------------------------------

def hf_views(arena, i, dense):
    from test_sim import HFViews  # HF-named views over the device layout (no copy of the layer)
    return HFViews({k.split(".", 1)[1]: v for k, v in arena.items() if k.startswith(f"L{i}.")}, dense)


def per_op(arena, i, dense, w, cfg, pos, x_in):
    """Every op of the layer vs torch on the megakernel's OWN inputs (design.md 5.2 per-op gate, teacher-forced
    op by op so nothing accumulates). Keys are the op names of design.md 2.4."""
    b, eps, g, res = (lambda n: arena[f"L{i}.{n}"]), cfg.rms_norm_eps, GATE["op"], {}
    chk = lambda name, out, ref: res.__setitem__(name, metrics(out, ref, g))
    chk("RMS_NORM", b("xn"), R.rmsnorm(x_in, w["input_layernorm.weight"], eps))
    xn = b("xn").flatten()
    chk("QKV_PROJ", b("qkva"), F.linear(xn, b("W_qkva")))
    qkva = b("qkva").flatten()
    q, c, k_pe = qkva[:3072].view(H, 192), qkva[3072:3584], qkva[3584:]
    cos, sin = (t.to(xn.device) for t in R.rope_cos_sin(torch.tensor(float(pos)), cfg, BF))
    # decision 11b: MLA_PREP is folded (kv_a norm + k_pe in KV_B_PROJ, q RoPE in ATTN_CHUNK), so its checks move there
    chk("KV_B_PROJ.k_pe", b("K")[:, pos, 128:], R.apply_rope(k_pe, cos, sin).expand(H, -1))
    k_new, v_new = R.kv_expand(R.rmsnorm(c, w["self_attn.kv_a_layernorm.weight"], eps), b("K")[0, pos, 128:], w, cfg)
    chk("KV_B_PROJ.K", b("K")[:, pos, :128], k_new[:, :128])
    chk("KV_B_PROJ.V", b("V")[:, pos], v_new)
    chk("ATTN_CHUNK+MERGE", b("attn_o"), R.attention(q[:, :128], R.apply_rope(q[:, 128:], cos, sin), w, cfg, b("K")[:, :pos + 1], b("V")[:, :pos + 1]))
    chk("O_PROJ_RESIDUAL", b("x_attn"), x_in + R.o_proj(b("attn_o"), w))
    x_attn = b("x_attn").flatten()
    chk("RMS_NORM.2", b("xn2"), R.rmsnorm(x_attn, w["post_attention_layernorm.weight"], eps))
    xn2 = b("xn2").flatten()
    if dense:
        chk("GATE_UP_SILU", b("h"), F.silu(F.linear(xn2, w["mlp.gate_proj.weight"])) * F.linear(xn2, w["mlp.up_proj.weight"]))
        chk("DOWN_SPLITK+REDUCE", b("x"), x_attn + F.linear(b("h").flatten(), w["mlp.down_proj.weight"]))
    else:
        chk("MOE_GATE", b("logits"), F.linear(xn2.float(), w["mlp.gate.weight"].float()))
        tw, ti = R.route(xn2, w, cfg)
        ids_k, ids_r = sorted(b("topk_idx").tolist()), sorted(ti.tolist())
        res["MOE_TOPK"] = dict(ids=ids_k, ref_ids=ids_r, ids_exact=ids_k == ids_r,
                               w_max_abs=(b("topk_w").sort().values - tw.sort().values).abs().max().item(),
                               ref_gap_6_7=(lambda s: (s[5] - s[6]).item())(torch.softmax(F.linear(xn2.float(), w["mlp.gate.weight"].float()), -1).sort(descending=True).values),
                               ok=ids_k == ids_r)
        y, _, _ = R.moe(xn2, w, cfg)
        chk("EXPERT_W13+W2+COMBINE", b("x"), x_attn + y)
    return res


def compare(arena, ref, a, b, cfg, torch_fallback):
    """Per layer: output vs HF dump (5 % + cosine), appended K/V rows (2 %), optional per-op + torch A/B."""
    per_layer, x = {}, ref["x_in"]
    for i, l in enumerate(range(a, b + 1)):
        dense, out = l < DIMS["first_k_dense"], arena[f"L{i}.x"].flatten()
        r = dict(out=metrics(out, ref["outs"][l], GATE["layer"]),
                 K_row=metrics(arena[f"L{i}.K"][:, P], ref["k_row"][l], GATE["op"]),
                 V_row=metrics(arena[f"L{i}.V"][:, P], ref["v_row"][l], GATE["op"]))
        r["out"]["ok"] = r["out"]["ok"] and r["out"]["cosine"] >= GATE["layer_cos"]
        if not dense:
            r["routing_ids"] = arena[f"L{i}.topk_idx"].tolist()
        if torch_fallback:
            w = hf_views(arena, i, dense)
            with torch.device(out.device):  # reference.py allocates without a device argument
                r["ops"] = per_op(arena, i, dense, w, cfg, P, x)
                ck, cv = arena[f"L{i}.K"].clone(), arena[f"L{i}.V"].clone()
                tf_out, routing = R.decode_step(x, w, cfg, ck, cv, P, dense)
            r["torch_vs_hf"] = metrics(tf_out, ref["outs"][l], GATE["layer"])
            r["fleet_vs_torch"] = metrics(out, tf_out, GATE["layer"])
            if routing is not None:
                r["routing_exact_vs_torch"] = sorted(routing[1].tolist()) == sorted(r["routing_ids"])
        r["ok"] = r["out"]["ok"] and r["K_row"]["ok"] and r["V_row"]["ok"]
        per_layer[l] = r
        x = out  # next layer sees the megakernel's own output (error vs depth, design.md 6 M3)
    return per_layer


# ---- dry run (laptop) ------------------------------------------------------------------------------------

class _T:
    def __init__(self, name, shape, dtype):
        self.name, self.shape, self.dtype = name, tuple(shape), dtype


class _TB:
    def __init__(self, grid, *_):
        self.grid, self.inputs = grid, []

    def new_input(self, t, m, *_):
        self.inputs.append((t.name, m))


class _KN:
    def __init__(self, ops):
        self.ops = ops

    def customized(self, tensors, tb):
        pass

    def register_task(self, tb, name, params):
        self.ops.append(dict(name=name, grid=tb.grid, tensors=tb.inputs, params=params))


class FakeMPK:
    """Records what build() would hand to Mirage: tensors (name, shape, dtype) and ops (register name, grid)."""
    def __init__(self):
        self.arena, self.ops, self.kn_graph = {}, [], _KN([])
        self.kn_graph.ops = self.ops

    def attach_input(self, torch_tensor, name=None):
        self.arena[name] = torch_tensor
        return _T(name, torch_tensor.shape, str(torch_tensor.dtype).replace("torch.", ""))

    def new_tensor(self, dims, strides=None, dtype="bf16", name=None, io_category="cuda_tensor"):
        return self.attach_input(torch.empty(dims, dtype=dict(bf16=BF, f32=torch.float32, i32=torch.int32, i64=torch.int64)[dtype], device="meta"), name)

    def _fleet(self, name, kw):
        ts = [v for v in kw.values() if isinstance(v, _T)] + [t for v in kw.values() if isinstance(v, tuple) for t in v if isinstance(t, _T)]
        self.ops.append(dict(name=name, grid=kw["grid_dim"], tensors=[(t.name, None) for t in ts], params=[]))

    FLEET = dict(embed_layer="embedding", rmsnorm_layer="rmsnorm", splitk_reduce_layer="splitk_reduce_mi300",
                 argmax_partial_layer="argmax_partial", argmax_reduce_layer="argmax_reduce")  # register names

    def __getattr__(self, k):
        if k in self.FLEET:
            return lambda **kw: self._fleet(self.FLEET[k], kw)
        raise AttributeError(k)


def dry_run(a, b, t_max, max_seq_length):
    """build() on the meta device with a recording stand-in for Mirage. Prints the plan and the arena."""
    mi = types.SimpleNamespace(bfloat16="bf16", float32="f32", int32="i32", int64="i64")
    MG.TBGraph, MG.CyTBGraph = (lambda cy: cy), _TB
    layers, glob = MG.no_gpu_weights()
    layers = layers[a:b + 1]
    glob["embed_tokens"] = torch.empty(2, HID, dtype=BF)  # the HF hidden state enters as row 0 (see module doc)
    empty = lambda *s, dt=BF: torch.empty(*s, dtype=dt, device="meta")
    glob.update(rope_cos=empty(t_max, 64), rope_sin=empty(t_max, 64), input_tokens=empty(1, 1, dt=torch.int64),
                output_tokens=empty(1, 1, dt=torch.int64))
    ck, cv = [empty(H, t_max, 192) for _ in layers], [empty(H, t_max, 128) for _ in layers]
    mpk, fk = FakeMPK(), 1 if a == 0 else 0
    with _subset(len(layers), fk):
        MG.build(mpk, mi, layers, glob, ck, cv, t_max=t_max)
        plan = MG.layer_plan(t_max)
    assert len(plan) == len(mpk.ops), (len(plan), len(mpk.ops))
    n_tasks = sum(g[0] * g[1] * g[2] for _, _, g in plan)
    print(f"plan: layers {a}-{b} ({len(layers)} layers, first_k_dense={fk}), 1 token at pos {P}, t_max {t_max} "
          f"({-(-t_max // 128)} chunks), max_seq_length {max_seq_length} -> {max_seq_length - P - 1} iteration(s)")
    print(f"{'op':>4} {'layer':>5} {'type':<20} {'register':<22} {'grid':<12} {'tasks':>6}  tensors")
    for k, ((l, typ, grid), op) in enumerate(zip(plan, mpk.ops)):
        assert MG.TASK[typ] == op["name"], (typ, op["name"])
        print(f"{k:>4} {l:>5} {typ:<20} {op['name']:<22} {str(grid):<12} {grid[0] * grid[1] * grid[2]:>6}  "
              + " ".join(n for n, _ in op["tensors"]))
    print(f"total ops {len(plan)}, tasks {n_tasks} (+ TERMINATE, BEGIN_TASK_GRAPH), events {len(plan) + 1}")
    tot = dict(weights=0, caches=0, scratch=0)
    print(f"\narena ({len(mpk.arena)} tensors, mirage_graph names):")
    for n, t in mpk.arena.items():
        nbytes = t.numel() * t.element_size()
        kind = "caches" if n.endswith((".K", ".V")) else "weights" if t.device.type == "cpu" else "scratch"
        tot[kind] += nbytes
        if kind != "weights" or "." not in n or n.split(".", 1)[0] in ("L0",) or n in ("embed_tokens", "lm_head", "model_norm"):
            print(f"  {n:<28} {str(tuple(t.shape)):<20} {str(t.dtype).replace('torch.', ''):<9} {nbytes / 2**20:9.2f} MiB")
    print("  ... (L1.. weights: same shapes as L0's MoE/attention tensors)")
    print(f"weights {tot['weights'] / 2**30:.2f} GiB, caches {tot['caches'] / 2**20:.0f} MiB, scratch {tot['scratch'] / 2**20:.1f} MiB")
    print(f"runtime: tokens[1,{max_seq_length}] int64, step = prompt_lengths = {P}, tokens[0] = tokens[{P}] = 0 (row 0 of the 2-row table)")
    return plan, mpk


# ---- main ------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", default="1", help="a or a-b (0 = dense layer; milestone layer >= 1)")
    ap.add_argument("--ref", default=REF_DIR)
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--t-max", type=int, default=1056, help="cache rows = prompt + 32 (9 chunks of 128)")
    ap.add_argument("--out", default=None, help="default results/m2 (one layer) or results/m3 (a range)")
    ap.add_argument("--torch-fallback", action="store_true", help="A/B: fleet_ds.reference on the GPU + per-op gates")
    ap.add_argument("--event-timing", action="store_true", help="compile with MPK_EVENT_TIMING: per-boundary times")
    ap.add_argument("--perfetto", action="store_true", help="Fleet's on-device profiler -> <out>/trace.perfetto-trace")
    ap.add_argument("--dry-run", action="store_true", help="no ROCm/model: print plan + arena shapes and exit")
    ap.add_argument("--reinit", action="store_true", help="a second init_func before the launch: regression check for upstream PR #9 (stalls without it; the watchdog dumps)")
    args = ap.parse_args(argv)
    assert not (args.torch_fallback and (DIMS["dag"] or DIMS["fold"])), \
        "--torch-fallback checks the chain's per-op buffers (xn, W_qkva, attn_o): run it with FLEET_KNOBS=dag=0,fold=0"
    a, b = parse_layers(args.layers)
    max_seq_length = P + 2  # one iteration
    if args.dry_run:
        dry_run(a, b, args.t_max, max_seq_length)
        return 0
    import mirage as mi
    out_dir = args.out or os.path.join(ROOT, "results", "m2" if a == b else "m3")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"layer_{a}-{b}"
    dev, cfg = "cuda", cfg_real()
    print(f"[{tag}] loading reference dumps from {args.ref}", flush=True)
    ref = load_ref(args.ref, a, b, args.t_max, dev)
    print(f"[{tag}] loading weights for layers {a}-{b} from {args.model}", flush=True)
    layers, sh = load_layers(args.model, a, b, dev)
    cos, sin = rope_tables(cfg, args.t_max, dev)
    glob = {"embed_tokens": torch.stack([ref["x_in"], torch.zeros_like(ref["x_in"])]),  # row 0 = HF layer-a input
            "lm_head": sh.get("lm_head.weight").to(dev), "model.norm": sh.get("model.norm.weight").to(dev),
            "rope_cos": cos, "rope_sin": sin}
    perf = torch.zeros(30000 * 1280, dtype=torch.uint64, device=dev) if args.perfetto else None
    if args.event_timing:
        os.environ["MPK_EVENT_TIMING"] = "1"
    t0 = time.time()
    mpk, arena, meta = make_mpk(mi, layers, glob, ref["cache_k"], ref["cache_v"], args.t_max, max_seq_length,
                                first_k_dense=1 if a == 0 else 0, dev=dev, perfetto=perf)
    if perf is not None:
        mpk.trace_name = os.path.join(out_dir, f"{tag}.trace")
    print(f"[{tag}] graph built in {time.time() - t0:.1f}s; compiling (minutes)", flush=True)
    t0 = time.time()
    mpk.compile(output_dir=out_dir)
    compile_s = time.time() - t0
    os.environ["MPK_EVENT_TIMING"] = "0"  # we read the buffer ourselves (event_timing)
    tg = json.load(open(os.path.join(out_dir, "task_graph_rank0.json")))
    with _subset(len(layers), 1 if a == 0 else 0):
        plan = MG.layer_plan(args.t_max)
    print(f"[{tag}] compiled in {compile_s:.0f}s: {len(tg['all_tasks'])} tasks, {len(tg['all_events'])} events; launching", flush=True)
    mpk.watchdog = (tg, plan)  # labels for watchdog_abort
    if args.reinit:
        reinit(mpk, meta, max_seq_length)
    ms = launch(mpk, meta, P, 0)  # token 0 = row 0 of the 2-row table = HF's layer-a input
    print(f"[{tag}] launch {ms:.3f} ms (one iteration incl. prepare_kernel)", flush=True)
    per_layer = compare(arena, ref, a, b, cfg, args.torch_fallback)
    res = dict(cmd=" ".join(sys.argv), git=git_hash(), date=time.strftime("%Y-%m-%dT%H:%M:%S"), layers=[a, b], pos=P,
               t_max=args.t_max, max_seq_length=max_seq_length, device=torch.cuda.get_device_name(0),
               workers=mpk.num_workers, schedulers=mpk.num_local_schedulers, compile_s=compile_s,
               graph=dict(tasks=len(tg["all_tasks"]), events=len(tg["all_events"]), ops=len(plan), knobs={k: DIMS[k] for k in KNOBS},
                          files=[os.path.join(out_dir, f) for f in ("task_graph_rank0.json", "test_rank0.cu")]),
               launch_ms=ms, gates=GATE, per_layer=per_layer)
    if b == DIMS["n_layers"] - 1:  # whole token: the head output is meaningful
        lg = arena["logits"].flatten()
        res["logits"] = metrics(lg, ref["logits0"], GATE["op"])
        res["logits"]["ok"] = res["logits"]["cosine"] >= GATE["logit_cos"]
        if "xn_final" in arena:  # L6 fold: the model norm is the LM_HEAD tiles' prologue (no buffer); the logits gate covers it
            res["final_norm"] = metrics(arena["xn_final"], ref["final_norm"], GATE["op"])
        res["argmax"] = dict(fleet=int(meta["output_tokens"][0, 0]), ref=ref["gen"][0], next_token_written=int(meta["tokens"][0, P + 1]),
                             ref_top2_margin=(lambda v: (v[0] - v[1]).item())(ref["logits0"].topk(2).values))
        res["argmax"]["ok"] = res["argmax"]["fleet"] == res["argmax"]["ref"] or res["argmax"]["ref_top2_margin"] < 0.1
    if args.event_timing:
        res["event_timing"] = event_timing(mpk, tg, plan)
    res["pass"] = all(r["ok"] for r in per_layer.values()) and res.get("logits", {"ok": True})["ok"]
    json.dump(res, open(os.path.join(out_dir, f"{tag}.json"), "w"), indent=1)
    torch.save({f"L{l}": dict(fleet=arena[f"L{i}.x"].cpu(), ref=ref["outs"][l].cpu()) for i, l in enumerate(range(a, b + 1))},
               os.path.join(out_dir, f"{tag}_diff.pt"))
    for l, r in per_layer.items():
        o = r["out"]
        print(f"  layer {l:2d}: out max|d| {o['max_abs']:.3e} (|ref| {o['max_ref']:.2e}, rel {o['rel']:.2e}, cos {o['cosine']:.6f}) "
              f"K {r['K_row']['rel']:.1e} V {r['V_row']['rel']:.1e}  {'PASS' if r['ok'] else 'FAIL'}"
              + ("" if "ops" not in r else "  ops: " + " ".join(f"{k}={'ok' if v['ok'] else 'FAIL'}" for k, v in r["ops"].items())))
    if "logits" in res:
        print(f"  logits cos {res['logits']['cosine']:.6f} argmax {res['argmax']}")
    print(f"[{tag}] {'PASS' if res['pass'] else 'FAIL'} -> {out_dir}/{tag}.json")
    mpk.finalize()
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
