"""Whole decode token through fleet_ds/sim.py (graph + kernels + events + pointer offsets) vs fleet_ds.reference.
(a) real layer-1 weights, HF 64-token prefill, one simulated MoE layer (850 slot-indexed tasks) vs decode_step / HF;
(b) 3-layer mini model (dense + 2 MoE, random at real dims) vs decode_token incl. embed/lm_head/argmax;
(c) hang detection; (d) worker-assignment shuffle -> identical bits. RAM-gated like test_reference."""
import os, re, shutil, sys, time
import pytest, torch, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "ref"), os.path.join(ROOT, "tests")]
from fleet_ds import reference as R, sim  # noqa: E402
from fleet_ds.graph import DIMS, build  # noqa: E402
from fleet_ds.mirage_graph import relayout_layer  # noqa: E402
from test_reference import COMMON, P, REAL, make_layer, mem_avail_gb, run_hf  # noqa: E402
from hf.configuration_deepseek import DeepseekV2Config  # noqa: E402
from test_real_layer1 import SHARDS, load  # noqa: E402

BF = torch.bfloat16
HID, E, TMAX = sim.HID, sim.E, sim.TMAX
pytestmark = pytest.mark.skipif(shutil.which("clang++") is None, reason="clang++ not installed")


@pytest.fixture(scope="module")
def K():
    return sim.build_harness()


def rope_tables(cfg):
    cs, sn = zip(*[R.rope_cos_sin(p, cfg, BF) for p in range(TMAX)])
    return torch.stack(cs).contiguous(), torch.stack(sn).contiguous()


def close(out, ref, tol, name):
    err, scale = (out.float() - ref.float()).abs().max().item(), ref.float().abs().max().item()
    cos = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    print(f"  {name}: max|d|={err:.3e} max|ref|={scale:.3e} rel={err / scale:.3e} cosine={cos:.6f}")
    assert err <= tol * scale + 1e-3, (name, err, scale)
    return err, cos


# ---- (a) one real MoE layer ---------------------------------------------------------------------------------

@pytest.mark.skipif(not SHARDS, reason="no complete checkpoint shard under models/ (fetch_model.sh)")
@pytest.mark.skipif(mem_avail_gb() < 2.8, reason="RAM (~2.6 GB peak)")
@pytest.mark.parametrize("knobs", [dict(dag=0, fold=0), {}], ids=["chain", "default"])  # the chain and the default (12a DAG + L6 folds)
def test_real_layer1_sim(K, knobs):
    w = load("model.layers.1.")
    cfg, layer = make_layer(REAL, BF, dense=False)
    layer.load_state_dict(w, assign=True)  # share storage: RAM
    emb = load("model.")["embed_tokens.weight"]
    torch.manual_seed(1)
    x = emb[torch.randint(emb.shape[0], (1, P + 1))]
    del emb
    hf_out, _, cache, _ = run_hf(layer, x, 0, BF)
    rl = relayout_layer(w, dense=False)
    del layer, w
    w = HFViews(rl, dense=False)  # the reference reads the device layout back (no second copy)
    ck, cv = torch.zeros(16, TMAX, 192, dtype=BF), torch.zeros(16, TMAX, 128, dtype=BF)
    ck[:, :P], cv[:, :P] = cache.key_cache[0][0, :, :P], cache.value_cache[0][0, :, :P]  # HF prefill handoff (decision 1)
    ref_k, ref_v = ck[:, :P + 1].clone(), cv[:, :P + 1].clone()
    ck0, cv0 = ck.clone(), cv.clone()
    cos, sin = rope_tables(cfg)
    glob = dict(embed_tokens=torch.zeros(1, HID, dtype=BF), lm_head=torch.zeros(1, HID, dtype=BF), tokens=torch.zeros(TMAX, dtype=torch.int64),
                rope_cos=cos, rope_sin=sin, **{"model.norm": torch.zeros(HID, dtype=BF)})
    a = sim.make_arena([rl], glob, [ck], [cv])
    a["x0"][:] = x[0, P]
    a["L0.h"].fill_(7.0); a["L0.expert_out"].fill_(7.0)  # sentinels: skipped experts must not touch their slots
    g = build(n_layers=1, first_k_dense=0, **knobs)
    t0 = time.time()
    trace = sim.simulate(g, a, K, cfg, P, only_layer=0)
    print(f"\n[real layer1 sim] {len(trace)} tasks in {time.time() - t0:.1f}s")
    ref, (tw, ti) = R.decode_step(x[0, P], w, cfg, ref_k, ref_v, P, dense=False)
    err, c = close(a["L0.x"], ref, 5e-2, "layer out vs reference")
    close(a["L0.x"], hf_out, 5e-2, "layer out vs HF eager")
    assert c >= 0.999
    close(ck[:, P], ref_k[:, P], 2e-2, "new K row"); close(cv[:, P], ref_v[:, P], 2e-2, "new V row")
    assert torch.equal(ck[:, :P], ck0[:, :P]) and torch.equal(cv[:, :P], cv0[:, :P]) and torch.equal(ck[:, P + 1:], ck0[:, P + 1:])
    idx = a["L0.topk_idx"].tolist()
    assert sorted(idx) == sorted(ti.tolist()) and len(set(idx)) == 6
    skipped = [e for e in range(E) if e not in idx]  # slot-indexed (10a): never dispatched, so never touched
    hv = a["L0.h"].view(-1)
    assert all((hv[e * 1408:(e + 1) * 1408] == 7.0).all() for e in skipped) and (a["L0.expert_out"][skipped] == 7.0).all()
    assert (hv[:E * 1408].view(E, 1408)[a["L0.topk_idx"].long()] != 7.0).any(dim=1).all()
    assert (a["L0.lse"][-(-(P + 1) // 128):] == -float("inf")).all()  # empty chunks signalled but wrote no data


# ---- (b)-(d) 3-layer mini model at real dims ---------------------------------------------------------------

class HFViews(dict):
    """HF-named view of a relayout_layer() dict for the reference; expert gate/up de-interleaved on demand
    (only the 6 routed experts are ever materialised, so no second copy of the layer)."""
    def __init__(self, w, dense):
        super().__init__({"input_layernorm.weight": w["input_layernorm"], "post_attention_layernorm.weight": w["post_attention_layernorm"],
                          "self_attn.kv_a_layernorm.weight": w["kv_a_layernorm"], "self_attn.q_proj.weight": w["W_qkva"][:3072],
                          "self_attn.kv_a_proj_with_mqa.weight": w["W_qkva"][3072:], "self_attn.kv_b_proj.weight": w["kv_b_proj"],
                          "self_attn.o_proj.weight": w["o_proj"]})
        self.w = w
        if dense:
            gu = w["mlp.gate_up"].view(-1, 2, 32, HID)
            self["mlp.gate_proj.weight"], self["mlp.up_proj.weight"] = gu[:, 0].reshape(-1, HID), gu[:, 1].reshape(-1, HID)
            self["mlp.down_proj.weight"] = w["mlp.down_proj"].transpose(0, 1).reshape(HID, -1)
        else:
            self["mlp.gate.weight"] = w["mlp.gate"]
            gu = w["mlp.w13_s"].view(-1, 2, 2816 // (DIMS["shared_w13_tiles"] or DIMS["w13_tiles"]), HID)
            self["mlp.shared_experts.gate_proj.weight"], self["mlp.shared_experts.up_proj.weight"] = gu[:, 0].reshape(-1, HID), gu[:, 1].reshape(-1, HID)
            self["mlp.shared_experts.down_proj.weight"] = w["mlp.w2_s"]

    def __missing__(self, k):
        e, which = re.fullmatch(r"mlp\.experts\.(\d+)\.(\w+)_proj\.weight", k).groups()
        if which == "down":
            return self.w["mlp.w2"][int(e)]
        return self.w["mlp.w13"][int(e)].view(-1, 2, 1408 // DIMS["w13_tiles"], HID)[:, 0 if which == "gate" else 1].reshape(-1, HID)


def rand_layer(dense, seed):
    """relayout_layer()-shaped random weights straight in device layout (no HF module: RAM)."""
    torch.manual_seed(seed)
    r = lambda *s: torch.randn(*s, dtype=BF).mul_(s[-1] ** -0.5)  # bf16 from the start: no fp32 temporaries
    u = lambda n: torch.empty(n).uniform_(0.5, 1.5).to(BF)
    w = {"input_layernorm": u(HID), "post_attention_layernorm": u(HID), "kv_a_layernorm": u(512),
         "W_qkva": r(3648, HID), "kv_b_proj": r(4096, 512), "o_proj": r(HID, HID)}
    if dense:
        w["mlp.gate_up"], w["mlp.down_proj"] = r(2 * 10944, HID), r(4, HID, 2736)
    else:
        w["mlp.gate"], w["mlp.w13"], w["mlp.w2"] = r(E, HID), r(E, 2816, HID), r(E, HID, 1408)
        w["mlp.w13_s"], w["mlp.w2_s"] = r(5632, HID), r(HID, 2816)
    return w


NL = int(os.environ.get("FLEET_SIM_LAYERS", 3))  # dense + (NL-1) MoE; ~1.15 GB per MoE layer, 3 -> ~3.2 GB peak RSS


@pytest.fixture(scope="module")
def mini(K):
    """NL layers (dense, MoE...), random prefill cache of 64 tokens, arena + reference weights."""
    if mem_avail_gb() < 1.1 * NL + 0.4:
        pytest.skip(f"RAM: need ~{1.1 * NL + 0.4:.1f} GB available (FLEET_SIM_LAYERS={NL})")
    cfg = DeepseekV2Config(**COMMON, **REAL)
    layers = [rand_layer(l == 0, 100 + l) for l in range(NL)]
    torch.manual_seed(5)
    ck = [torch.randn(16, TMAX, 192).to(BF) for _ in range(NL)]
    cv = [torch.randn(16, TMAX, 128).to(BF) for _ in range(NL)]
    cos, sin = rope_tables(cfg)
    emb = torch.zeros(cfg.vocab_size, HID, dtype=BF)  # calloc: only the rows we write become resident
    emb[4242] = torch.randn(HID)
    glob = dict(embed_tokens=emb, lm_head=torch.randn(cfg.vocab_size, HID, dtype=BF).mul_(HID ** -0.5),
                rope_cos=cos, rope_sin=sin, tokens=torch.zeros(TMAX, dtype=torch.int64),
                **{"model.norm": torch.empty(HID).uniform_(0.5, 1.5).to(BF)})
    glob["tokens"][P] = 4242
    a = sim.make_arena(layers, glob, ck, cv)
    ref_w = {"model.embed_tokens.weight": glob["embed_tokens"], "model.norm.weight": glob["model.norm"],
             "lm_head.weight": glob["lm_head"], "layers": [HFViews(w, l == 0) for l, w in enumerate(layers)]}
    return cfg, a, ref_w, build(n_layers=NL)  # the default graph: DAG + L6 folds (2026-09-15)


FOLDED = re.compile(r"(L\d+\.(xn|attn_o)|L0\.xn2|xn_final)")  # chain-only buffers: the default's L6 folds never write them


def snapshot(a):
    return {k: v.clone() for k, v in a.items() if v.numel() < 2 ** 24 and not FOLDED.fullmatch(k)}  # scratch, caches, small weights


@pytest.fixture(scope="module")
def run0(K, mini):
    cfg, a, ref_w, g = mini
    t0 = time.time()
    trace = sim.simulate(g, a, K, cfg, P)
    print(f"\n[mini token sim] {len(trace)} tasks in {time.time() - t0:.1f}s")
    return snapshot(a), trace


def routing_check(sim_idx, sim_logits, ref_idx, ref_logits, l):
    """design §5.2 routing gate (D5/D7h): expert ids identical, except a near-tie: the sim may swap experts only where
    the reference's own gate-logit margin between them is below the measured gate-logit error (the error is layers
    0..l-1 fed forward). Every escape is printed, never silent."""
    s, r = set(sim_idx.tolist()), set(ref_idx.tolist())
    err = (sim_logits - ref_logits).abs().max().item()
    top = ref_logits.sort(descending=True).values
    print(f"  L{l} routing: gate-logit max|d|={err:.3e}, ref 6th-7th logit margin={float(top[5] - top[6]):.3e}, "
          f"{'ids equal' if s == r else 'ids differ'}")
    if s != r:
        dropped, added = sorted(r - s), sorted(s - r)
        margin = (ref_logits[dropped].max() - ref_logits[added].min()).item()
        print(f"  L{l} NEAR-TIE ESCAPE: ref {sorted(r)} sim {sorted(s)} (dropped {dropped}, added {added}): "
              f"ref logit margin {margin:.3e} < gate-logit error {err:.3e}")
        assert margin < err, (l, dropped, added, margin, err)


def test_mini_token_vs_decode_token(K, mini, run0, monkeypatch):
    cfg, a, ref_w, g = mini
    snap, trace = run0
    caches = [(a[f"L{l}.K"][:, :P + 1].clone(), a[f"L{l}.V"][:, :P + 1].clone()) for l in range(NL)]
    for ck, cv in caches:
        ck[:, P] = 0; cv[:, P] = 0
    gate = []  # the reference's fp32 gate logits per MoE layer (route() keeps the top-6 only; the escape needs the 7th)
    monkeypatch.setattr(R, "route", lambda xn, w, c, _r=R.route: (gate.append(F.linear(xn.float(), w["mlp.gate.weight"].float())), _r(xn, w, c))[1])
    tok, logits, routing, hidden = R.decode_token(4242, ref_w, cfg, caches, P)
    for l in range(NL):
        _, c = close(snap[f"L{l}.x"], hidden[l], 5e-2, f"L{l} residual out (error accumulates over layers)")
        assert c >= 0.999
        close(snap[f"L{l}.K"][:, P], caches[l][0][:, P], 2e-2, f"L{l} new K row")
        close(snap[f"L{l}.V"][:, P], caches[l][1][:, P], 2e-2, f"L{l} new V row")
        if routing[l] is not None:
            routing_check(snap[f"L{l}.topk_idx"], snap[f"L{l}.logits"], routing[l][1], gate[l - cfg.first_k_dense_replace], l)
    err, cos = close(snap["logits"], logits, 5e-2, "logits")
    assert cos >= 0.9995
    ours = snap["output_token"].item()
    assert ours == snap["tokens"][P + 1].item()  # prepare_next_batch wrote our own token
    top2 = logits.topk(2).values
    d = (snap["logits"].float() - logits.float()).abs()
    print(f"  argmax {ours} (ref {tok}), ref top-2 margin {float(top2[0] - top2[1]):.3e}, |d| at ref/our argmax "
          f"{float(d[tok]):.3e}/{float(d[ours]):.3e}, max|d| {float(d.max()):.3e}")
    if ours != tok:  # near-tie escape (design 5.2, as routing_check): the swap must be explained by the logit error
        margin, err = float(logits[tok] - logits[ours]), float(d[tok] + d[ours])
        print(f"  NEAR-TIE ESCAPE argmax: ref logit margin {margin:.3e} < logit error {err:.3e}")
        assert margin < err, (ours, tok, margin, err)


def test_hang_detection(K, mini):
    cfg, a, ref_w, g = mini
    g2 = {**g, "events": [dict(e) for e in g["events"]]}
    ev = g["tasks"][g["ops"][3]["first_task_id"]]["dependent_event"]  # fires after layer-0 KVA_PROJ (op 2 of the default DAG), gates KV_B_PROJ (11b)
    g2["events"][ev]["num_triggers"] += 1
    with pytest.raises(sim.SimHang) as ex:
        sim.simulate(g2, a, K, cfg, P)
    print(f"\n  {ex.value}")
    assert ex.value.event == ev and ex.value.counter == g["events"][ev]["num_triggers"] == ex.value.need - 1
    assert ex.value.stuck[0]["type"] == "KV_B_PROJ"


@pytest.mark.parametrize("knobs, seed", [(dict(dag=0, fold=0), None), (dict(dag=0, fold=0), 7), (dict(dag=1, fold=0), 7)],
                         ids=["chain", "chain-shuffled", "dag-nofold"])
def test_chain_bit_identical_to_default(K, mini, run0, knobs, seed):
    """The measured history stays reproducible: the straight chain (D4) and the plain DAG (12a, no folds) compute every arena tensor
    bit for bit like the default graph (DAG: split ops, multi-producer + fan-out events, readiness asserted by the sim; L6 folds),
    which test_mini_token_vs_decode_token checks against the reference."""
    cfg, a, ref_w, g = mini
    snap, _ = run0
    gd = build(n_layers=NL, **knobs)
    trace = sim.simulate(gd, a, K, cfg, P, seed=seed)
    assert len(trace) == len(gd["tasks"]) - 1
    for k, v in snapshot(a).items():
        assert torch.equal(v, snap[k]), k


def test_dag_hang_and_early_consumer(K, mini, run0):
    """A multi-producer event one trigger short hangs (SimHang names it); one that fires early lets the consumer run before
    its producers finish, which the sim's data-readiness assertion catches."""
    cfg, a, ref_w, gd = mini  # the default DAG (+ L6 folds)
    attn = next(o for o in gd["ops"] if o["layer"] == 1 and o["type"] == "ATTN_CHUNK")
    comb = next(o for o in gd["ops"] if o["layer"] == 1 and o["type"] == "MOE_COMBINE_RESIDUAL")
    ev = lambda o: gd["tasks"][o["first_task_id"]]["dependent_event"]
    for o, delta in ((attn, 1), (comb, 1 - 224)):  # seed 1: uneven queues (round-robin polling keeps producers in step)
        g2 = {**gd, "events": [dict(e) for e in gd["events"]]}
        g2["events"][ev(o)]["num_triggers"] += delta
        if delta > 0:
            with pytest.raises(sim.SimHang) as ex:
                sim.simulate(g2, a, K, cfg, P, only_layer=1)
            assert ex.value.event == ev(o) and ex.value.stuck[0]["type"] == "ATTN_CHUNK"
        else:
            with pytest.raises(AssertionError, match="runs before producer"):
                sim.simulate(g2, a, K, cfg, P, only_layer=1, seed=1)


def retile(t, old, new):
    """Gate/up rows interleaved in `old`-row blocks -> `new`-row blocks, in place, expert by expert (RAM)."""
    for v in (t if t.dim() == 3 else t[None]):
        gu = v.view(-1, 2, old, HID)
        g, u = gu[:, 0].reshape(-1, HID).clone(), gu[:, 1].reshape(-1, HID).clone()
        v.copy_(torch.stack([g.view(-1, new, HID), u.view(-1, new, HID)], 1).reshape(-1, HID))


@pytest.mark.parametrize("knobs", [dict(fold=0, attn_rows=32), dict(fold=0, attn_rows=16), dict(fold=0, gate_tiles=8),
                                   dict(fold=0, shared_w13_tiles=64, shared_w2_tiles=64, w13_tiles=44),
                                   dict(fold=1, gate_tiles=8, attn_rows=16, shared_w13_tiles=64, shared_w2_tiles=64, w13_tiles=44)])
def test_dag_switches_bit_identical(K, mini, run0, knobs):
    """12e switches on the DAG (L2 shared-expert tiles, L3 w13 44, L4 16/32-row attention tiles, L5 gate tiles + top-k task,
    L6 folds): every arena tensor bit for bit as the default run (== the chain, test_chain_bit_identical_to_default). L2/L3 re-tile the w13 gate/up interleave in place for the
    run (restored after); L6 needs the fold kernels' harness exports (perf/kernels)."""
    if knobs.get("fold") and not all(hasattr(K, f) for f in ("k_gemv_norm", "k_o_proj_merge_rows", "k_moe_gate_topk_norm")):
        pytest.skip("L6 fold kernels arrive with perf/kernels (12e)")
    cfg, a, ref_w, g = mini
    snap, _ = run0
    w13 = [(a[f"L{l}.mlp.w13"], 1408 // DIMS["w13_tiles"], 1408 // knobs.get("w13_tiles", DIMS["w13_tiles"])) for l in range(1, NL)]
    w13 += [(a[f"L{l}.mlp.w13_s"], 2816 // DIMS["w13_tiles"], 2816 // knobs.get("shared_w13_tiles", DIMS["w13_tiles"])) for l in range(1, NL)]
    for t, old, new in w13:
        retile(t, old, new)
    try:
        sim.simulate(build(n_layers=NL, dag=1, **knobs), a, K, cfg, P, seed=3)
        for k, v in snapshot(a).items():
            assert k.endswith("mlp.w13_s") or torch.equal(v, snap[k]), k
    finally:
        for t, old, new in w13:
            retile(t, new, old)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_order_independence(K, mini, run0, seed):
    cfg, a, ref_w, g = mini
    snap, trace0 = run0
    trace = sim.simulate(g, a, K, cfg, P, seed=seed)
    assert [i for i, _ in trace] != [i for i, _ in trace0]
    for k, v in snapshot(a).items():
        assert torch.equal(v, snap[k]), k
