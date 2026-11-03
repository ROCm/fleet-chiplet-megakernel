"""DeepSeek-V2-Lite batch-1 decode graph for Mirage/Fleet's PersistentKernel (design D4).

Same call pattern as demo/qwen3/demo_30B_A3B.py: `mpk.attach_input` for torch
tensors, `mpk.new_tensor` for scratch, and per op a TBGraph whose only nodes are `new_input(tensor,
input_map, forloop_dim, True)` followed by `kn_graph.customized(...)` + `kn_graph.register_task(tb, name,
params)`. `input_map = (x, y, z)` names which tensor dim grid.x/y/z partitions (-1 = whole tensor); the
runtime pre-offsets that task's base pointer by bid * dim/grid (runtime.cc print_task_graph L1052).
Op order and grids are docs/design.md 2.4-2.5; `layer_plan()` is the Mirage-free part and is asserted
equal to fleet_ds/graph.py by tests/test_mirage_graph.py. Task-type names, tensor order (inputs first,
then outputs -- Fleet's TaskDesc packing) and metadata are the contract with kernels/fleet_dispatch.cuh (checked by tests/test_glue.py);
Fleet's own kernels are reused where design.md 2.2 says so.

Chain rule (runtime.cc register_mugraph, assert num_shared_tensors >= 1): every op consumes at least one
output of the previous op. That tensor is always mapped (-1,-1,-1) on the consumer side so the gcd event
partition collapses to ONE event per op boundary (design.md: every boundary is a full barrier).

Run on the MI300X box: see build_graph.sh.
"""
import argparse
import math
import os

import torch

from .graph import DIMS

# Mirage register_task names (design §2.2). Fleet's existing ones (embedding, rmsnorm,
# splitk_reduce_mi300, argmax_*) are reused unchanged; the rest are ours (kernels/*.cuh).
TASK = dict(
    EMBEDDING="embedding", RMS_NORM="rmsnorm", QKV_PROJ="gemv_bf16", Q_PROJ="gemv_bf16", KVA_PROJ="gemv_bf16",
    EXPERT_W13_SHARED="gemv_bf16_silu", EXPERT_W2_SHARED="gemv_bf16", MOE_GATE="gemv_bf16", MOE_TOPK="moe_topk",
    KV_B_PROJ="gemv_bf16", ATTN_CHUNK="attn_chunk_192_128", ATTN_MERGE="attn_merge",
    O_PROJ_RESIDUAL="gemv_bf16_res", GATE_UP_SILU="gemv_bf16_silu", DOWN_SPLITK="gemv_splitk",
    REDUCE_RESIDUAL="splitk_reduce_mi300", MOE_GATE_TOPK="moe_gate_topk",
    EXPERT_W13_SILU="moe_expert_w13_silu", EXPERT_W2="moe_expert_w2", MOE_COMBINE_RESIDUAL="moe_combine",
    LM_HEAD="gemv_bf16", ARGMAX_PARTIAL="argmax_partial", ARGMAX_REDUCE="argmax_reduce")


def layer_plan(t_max=1056, **kw):
    """Mirage-independent op list: [(layer, type, grid)] in emission order. prod(grid) == task count. Knobs as graph.DIMS
    (12a dag, 12e L2-L6): the op list is graph.build()'s, asserted by tests/test_mirage_graph.py."""
    d = {**DIMS, **kw}
    n_chunks = -(-t_max // d["chunk"])
    tile, ar, fold, gt, dag = d["tile"], d["attn_rows"], d["fold"], d["gate_tiles"], d["dag"]
    n13s, n2s = d["shared_w13_tiles"] or d["w13_tiles"], d["shared_w2_tiles"] or d["w2_tiles"]
    one = (1, 1, 1)
    plan = [(-1, "EMBEDDING", one)]
    for l in range(d["n_layers"]):
        dense = l < d["first_k_dense"]
        plan += [] if fold else [(l, "RMS_NORM", one)]
        plan += ([(l, "Q_PROJ", (d["q_rows"] // ar, 1, 1)), (l, "KVA_PROJ", (d["kv_a_rows"] // (ar if d["kv_a_rows"] % ar == 0 else tile), 1, 1))] if dag else
                 [(l, "QKV_PROJ", ((d["q_rows"] + d["kv_a_rows"]) // ar, 1, 1))])
        plan += [(l, "KV_B_PROJ", (d["kv_b_rows"] // ar, 1, 1)), (l, "ATTN_CHUNK", (d["n_heads"], n_chunks, 1))]
        plan += [] if fold else [(l, "ATTN_MERGE", (d["n_heads"], 1, 1))]
        plan += [(l, "O_PROJ_RESIDUAL", (d["hidden"] // ar, 1, 1))]
        plan += [(l, "RMS_NORM", one)] if not fold or (gt and not dense) else []
        if dense:
            plan += [(l, "GATE_UP_SILU", (2 * d["dense_inter"] // tile, 1, 1)),
                     (l, "DOWN_SPLITK", (d["hidden"] // tile, d["split_k"], 1)),
                     (l, "REDUCE_RESIDUAL", one)]
            continue
        # grid.x = top-k slot (0-5 routed, 6 = shared in the chain), grid.y = N-tile: Fleet's expert_offset packing (10a)
        plan += [(l, "MOE_GATE", (gt, 1, 1))] if gt else [(l, "MOE_GATE_TOPK", one)]
        plan += [(l, "EXPERT_W13_SHARED", (n13s, 1, 1))] if dag else []
        plan += [(l, "MOE_TOPK", one)] if gt else []
        if dag:  # 12a: the shared expert is its own w13 -> w2 path (plain GEMVs), routed slots 0-5
            plan += [(l, "EXPERT_W13_SILU", (d["topk"], d["w13_tiles"], 1)), (l, "EXPERT_W2_SHARED", (n2s, 1, 1)),
                     (l, "EXPERT_W2", (d["topk"], d["w2_tiles"], 1))]
        else:
            plan += [(l, "EXPERT_W13_SILU", (d["topk"] + 1, d["w13_tiles"], 1)), (l, "EXPERT_W2", (d["topk"] + 1, d["w2_tiles"], 1))]
        plan += [(l, "MOE_COMBINE_RESIDUAL", one)]
    plan += [] if fold else [(-1, "RMS_NORM", one)]
    plan += [(-1, "LM_HEAD", (d["vocab"] // d["lm_rows"], 1, 1)),
             (-1, "ARGMAX_PARTIAL", (d["vocab"] // d["argmax_chunk"], 1, 1)),
             (-1, "ARGMAX_REDUCE", one)]
    return plan


def plan_counts(plan):
    return [(l, typ, math.prod(grid)) for l, typ, grid in plan]


# ---------------------------------------------------------------------------------------------------
# Host weight re-layout (design.md 4.1). Pure torch, done once at load, excluded from timing.

def interleave_gate_up(gate, up, block):
    """[2I,K] with gate/up rows in alternating `block`-row blocks so one tile holds matching pairs (8a)."""
    I, K = gate.shape
    return torch.stack([gate.view(I // block, block, K), up.view(I // block, block, K)], 1).reshape(2 * I, K)


def relayout_layer(sd, dense):
    """HF per-layer state_dict (keys without 'model.layers.L.') -> device tensors named as design.md 4.1."""
    w = {"input_layernorm": sd["input_layernorm.weight"],
         "post_attention_layernorm": sd["post_attention_layernorm.weight"],
         "kv_a_layernorm": sd["self_attn.kv_a_layernorm.weight"],
         "W_qkva": torch.cat([sd["self_attn.q_proj.weight"], sd["self_attn.kv_a_proj_with_mqa.weight"]], 0),
         "kv_b_proj": sd["self_attn.kv_b_proj.weight"], "o_proj": sd["self_attn.o_proj.weight"]}
    if dense:
        w["mlp.gate_up"] = interleave_gate_up(sd["mlp.gate_proj.weight"], sd["mlp.up_proj.weight"], 32)
        # [4, 2048, 2736]: K-slice-major so each split-K tile is a contiguous [64, 2736] block (gemv_bf16.cuh
        # assumes row stride == K). Same bytes as HF's [2048, 10944].
        S, dp = DIMS["split_k"], sd["mlp.down_proj.weight"]
        w["mlp.down_proj"] = dp.view(dp.shape[0], S, dp.shape[1] // S).transpose(0, 1)
    else:
        E, n = DIMS["n_experts"], DIMS["w13_tiles"]  # gate/up block = half a w13 tile (11a)
        w["mlp.gate"] = sd["mlp.gate.weight"]
        w["mlp.w13"] = torch.stack([interleave_gate_up(sd[f"mlp.experts.{e}.gate_proj.weight"],
                                                       sd[f"mlp.experts.{e}.up_proj.weight"], DIMS["moe_inter"] // n) for e in range(E)])
        w["mlp.w2"] = torch.stack([sd[f"mlp.experts.{e}.down_proj.weight"] for e in range(E)])
        w["mlp.w13_s"] = interleave_gate_up(sd["mlp.shared_experts.gate_proj.weight"],  # own tiles under L2 (12e)
                                            sd["mlp.shared_experts.up_proj.weight"], DIMS["shared_inter"] // (DIMS["shared_w13_tiles"] or n))
        w["mlp.w2_s"] = sd["mlp.shared_experts.down_proj.weight"]
    return {k: v.contiguous() for k, v in w.items()}


# ---------------------------------------------------------------------------------------------------
# Mirage graph

def _op(mpk, name, tensors, maps, grid, params=None):
    """One chain op, exactly what every PersistentKernel.*_layer does under the hood."""
    tb = TBGraph(CyTBGraph(grid, (256, 1, 1), 1, 64))
    for t, m in zip(tensors, maps):
        tb.new_input(t, m, -1, True)
    mpk.kn_graph.customized(list(tensors), tb)
    mpk.kn_graph.register_task(tb, name, params or [])


WHOLE = (-1, -1, -1)
ROWS_X = (0, -1, -1)   # grid.x partitions dim 0 (weight rows -> 64-row tile, pre-offset base_ptr)
ROWS_Y = (-1, 0, -1)   # grid.y partitions dim 0
COLS_X = (1, -1, -1)   # grid.x partitions dim 1 ([1, N] outputs)


def build(mpk, mi, layers, glob, cache_k, cache_v, t_max=1056):
    """layers: list of relayout_layer() dicts (torch, on GPU); glob: embed_tokens, lm_head, model.norm,
    rope_cos, rope_sin, input_tokens, output_tokens; cache_k[l] [16,T,192], cache_v[l] [16,T,128] bf16."""
    d, bf16, f32 = DIMS, mi.bfloat16, mi.float32
    n_chunks = -(-t_max // d["chunk"])
    H, hid, E = d["n_heads"], d["hidden"], d["n_experts"]
    A = mpk.attach_input
    N = lambda name, dims, dtype=bf16: mpk.new_tensor(dims=dims, dtype=dtype, name=name, io_category="cuda_tensor")
    plan = iter(layer_plan(t_max))

    def G(typ):  # next grid from the plan; build() cannot drift from layer_plan()
        l, t, grid = next(plan)
        assert t == typ, (t, typ)
        return grid

    tok = A(glob["input_tokens"], "input_token")
    cos, sin = A(glob["rope_cos"], "rope_cos"), A(glob["rope_sin"], "rope_sin")
    x = N("x0", (1, hid))
    mpk.embed_layer(input=tok, weight=A(glob["embed_tokens"], "embed_tokens"), output=x,
                    grid_dim=G("EMBEDDING"), block_dim=(256, 1, 1), input_source=1)

    fold, gt, dag = d["fold"], d["gate_tiles"], d["dag"]
    for l, w in enumerate(layers):
        b = lambda n: f"L{l}.{n}"  # C++ identifiers in test.cu get dots -> "_" (runtime_cc.patch); keys keep the dots
        W = lambda n: A(w[n], b(n))
        dense = l < d["first_k_dense"]
        dev = w["W_qkva"].device
        # DAG (12a): a buffer written by two sibling ops is one torch tensor attached whole AND as row-major views, each its
        # own Mirage tensor: register_mugraph derives producers from tensor guids, so a reader of the whole (no writer) plus
        # a dependency-only view waits exactly on the ops it needs; every task keeps the chain's pointers (same kernels)
        T = lambda name, t, *views: [A(t, name)] + [A(v, f"{name}_{s}") for s, v in views]
        # per-layer scratch (7f)
        if dag:
            qt = torch.zeros(1, d["q_rows"] + d["kv_a_rows"], dtype=torch.bfloat16, device=dev)
            qkva, q, kva = T(b("qkva"), qt, ("q", qt[0, :d["q_rows"]].view(1, -1)), ("kva", qt[0, d["q_rows"]:].view(1, -1)))
        else:
            qkva = q = N(b("qkva"), (1, d["q_rows"] + d["kv_a_rows"]))
        K, V = A(cache_k[l], b("K")), A(cache_v[l], b("V"))
        # attention partials in Fleet's merge_splitkv layout: o_part[(chunk*H + head)*128], lse[chunk*H + head]
        part_o, lse = N(b("part_o"), (n_chunks, H, 128), f32), N(b("lse"), (n_chunks, H), f32)
        x_attn, x_out = N(b("x_attn"), (1, hid)), N(b("x"), (1, hid))

        # 1-8: attention block, ops as design.md 2.4; L6 (fold, 12e): input norm and the LSE merge are prologues (perf/fold-full)
        if fold:
            xin = [x, W("input_layernorm")]
        else:
            xn = N(b("xn"), (1, hid))
            mpk.rmsnorm_layer(input=x, weight=W("input_layernorm"), output=xn, grid_dim=G("RMS_NORM"), block_dim=(256, 1, 1))
            xin = [xn]
        xm = [WHOLE] * len(xin)
        # decision 11b: MLA_PREP folded -- kv_a_layernorm + k_pe RoPE in the KV_B_PROJ tiles, q RoPE in ATTN_CHUNK
        if dag:  # 12a: q rows || kv_a rows -> KV_B_PROJ; KV_B reads qkva whole (kernel offsets) + the kva view (dependency)
            wq = w["W_qkva"]
            _op(mpk, TASK["Q_PROJ"], [*xin, A(wq[:d["q_rows"]], b("W_q")), q], [*xm, ROWS_X, COLS_X], G("Q_PROJ"))
            _op(mpk, TASK["KVA_PROJ"], [*xin, A(wq[d["q_rows"]:], b("W_kva")), kva], [*xm, ROWS_X, COLS_X], G("KVA_PROJ"))
            _op(mpk, TASK["KV_B_PROJ"], [qkva, W("kv_a_layernorm"), W("kv_b_proj"), cos, sin, kva, K, V],
                [WHOLE, WHOLE, ROWS_X, WHOLE, WHOLE, WHOLE, WHOLE, WHOLE], G("KV_B_PROJ"))
        else:
            _op(mpk, TASK["QKV_PROJ"], [*xin, W("W_qkva"), qkva], [*xm, ROWS_X, COLS_X], G("QKV_PROJ"))
            _op(mpk, TASK["KV_B_PROJ"], [qkva, W("kv_a_layernorm"), W("kv_b_proj"), cos, sin, K, V],
                [WHOLE, WHOLE, ROWS_X, WHOLE, WHOLE, WHOLE, WHOLE], G("KV_B_PROJ"))
        # kernels index by head = bid.x / chunk = bid.y themselves (kernels/fleet_dispatch.cuh), so nothing is pre-offset; q = qkva's first
        # 3072 values (the same pointer; its own tensor in the DAG so ATTN_CHUNK waits on Q_PROJ + KV_B_PROJ)
        _op(mpk, TASK["ATTN_CHUNK"], [q, K, V, cos, sin, part_o, lse], [WHOLE] * 7, G("ATTN_CHUNK"), [d["chunk"]])
        if fold:
            _op(mpk, TASK["O_PROJ_RESIDUAL"], [part_o, lse, W("o_proj"), x, x_attn], [WHOLE, WHOLE, ROWS_X, COLS_X, COLS_X],
                G("O_PROJ_RESIDUAL"), [d["chunk"]])
        else:
            attn_o = N(b("attn_o"), (H, 128))
            _op(mpk, TASK["ATTN_MERGE"], [part_o, lse, attn_o], [WHOLE] * 3, G("ATTN_MERGE"), [n_chunks])
            _op(mpk, TASK["O_PROJ_RESIDUAL"], [attn_o, W("o_proj"), x, x_attn], [WHOLE, ROWS_X, COLS_X, COLS_X], G("O_PROJ_RESIDUAL"))
        pln = W("post_attention_layernorm")
        xn2 = N(b("xn2"), (1, hid)) if not (fold and dense) else None
        if not fold or (gt and not dense):  # L6 + L5: norm2 stays a task (no gate tile can own the xn2 write), 12e
            mpk.rmsnorm_layer(input=x_attn, weight=pln, output=xn2, grid_dim=G("RMS_NORM"), block_dim=(256, 1, 1))
            x2, x2m = [xn2], [WHOLE]
        else:
            x2, x2m = [x_attn, pln], [WHOLE, WHOLE]
        if dense:  # 9-11 (design.md 2.5): fused silu gate_up, split-K 4 down, reduce + residual
            h, ws = N(b("h"), (1, d["dense_inter"])), N(b("down_part"), (d["split_k"], hid), f32)
            _op(mpk, TASK["GATE_UP_SILU"], [*x2, W("mlp.gate_up"), h], [*x2m, ROWS_X, COLS_X], G("GATE_UP_SILU"))
            # down_proj is [4, 2048, 2736] (relayout_layer): grid.y = K-slice (dim 0), grid.x = 64-row tile (dim 1)
            _op(mpk, TASK["DOWN_SPLITK"], [h, W("mlp.down_proj"), ws], [(-1, 1, -1), (1, 0, -1), (1, 0, -1)],
                G("DOWN_SPLITK"))
            mpk.splitk_reduce_layer(workspace=ws, residual=x_attn, output=x_out, k_splits=d["split_k"],
                                    grid_dim=G("REDUCE_RESIDUAL"), block_dim=(256, 1, 1))
        else:  # 9-12: gate + top-k in one task (ours: logits fp32, w[6], idx[6]; 10c), slot-indexed w13/w2 (10a), combine in HF order
            logits = N(b("logits"), (1, E), f32)
            topk_w, topk_idx = N(b("topk_w"), (d["topk"],), f32), N(b("topk_idx"), (d["topk"],), mi.int32)
            if dag:  # 12a: h_s / expert_out_s = the shared expert's slices of h / expert_out (views, as qkva above)
                ht, et = (torch.zeros(s, dtype=torch.bfloat16, device=dev) for s in ((E * d["moe_inter"] + d["shared_inter"],), (E + 1, hid)))
                h, h_s = T(b("h"), ht, ("s", ht[E * d["moe_inter"]:].view(1, -1)))
                e_out, e_out_s = T(b("expert_out"), et, ("s", et[E].view(1, -1)))
            else:
                h = N(b("h"), (E * d["moe_inter"] + d["shared_inter"],))  # routed e at e*1408, shared at 64*1408
                e_out = N(b("expert_out"), (E + 1, hid))
            gate, w13_s, w2_s = W("mlp.gate"), W("mlp.w13_s"), W("mlp.w2_s")
            if gt:  # L5 (12e): the gate GEMV as gt fp32 tiles of its own op, top-k as one task below
                _op(mpk, TASK["MOE_GATE"], [xn2, gate, logits], [WHOLE, ROWS_X, COLS_X], G("MOE_GATE"))
            elif fold:  # L6: norm2 is the prologue; the task writes xn2 for the experts (declared an input: Fleet allows 3 outputs)
                _op(mpk, TASK["MOE_GATE_TOPK"], [x_attn, pln, gate, xn2, logits, topk_w, topk_idx], [WHOLE] * 7, G("MOE_GATE_TOPK"))
            else:
                _op(mpk, TASK["MOE_GATE_TOPK"], [xn2, gate, logits, topk_w, topk_idx], [WHOLE] * 5, G("MOE_GATE_TOPK"))
            # slot = bid.x (0-5: expert idx[slot], read at run time; 6 = shared), tile = bid.y (Fleet's expert_offset
            # packing); the expert offset into w13/w2/h/e_out is added by the generated _execute_task, the tile offset by
            # the input map (e_out: grid.y partitions its columns, the expert row cannot be a pre-offset any more)
            if dag:  # 12a: shared w13 alongside gate/top-k (plain fused-silu GEMV, the chain's slot-6 call; L6: + norm2 prologue)
                _op(mpk, TASK["EXPERT_W13_SHARED"], [*x2, w13_s, h_s], [*x2m, ROWS_X, COLS_X], G("EXPERT_W13_SHARED"))
            if gt:
                _op(mpk, TASK["MOE_TOPK"], [logits, topk_w, topk_idx], [WHOLE] * 3, G("MOE_TOPK"))
            _op(mpk, TASK["EXPERT_W13_SILU"], [topk_idx, xn2, W("mlp.w13"), w13_s, h],
                [WHOLE, WHOLE, (-1, 1, -1), ROWS_Y, WHOLE], G("EXPERT_W13_SILU"))
            if dag:  # shared w2 (K 2816) into expert_out[64]
                _op(mpk, TASK["EXPERT_W2_SHARED"], [h_s, w2_s, e_out_s], [WHOLE, ROWS_X, COLS_X], G("EXPERT_W2_SHARED"))
            _op(mpk, TASK["EXPERT_W2"], [topk_idx, h, W("mlp.w2"), w2_s, e_out],
                [WHOLE, WHOLE, (-1, 1, -1), ROWS_Y, (-1, 1, -1)], G("EXPERT_W2"))
            dep = [e_out_s] if dag else []  # dependency-only operand: combine reads expert_out[64] through e_out
            _op(mpk, TASK["MOE_COMBINE_RESIDUAL"], [e_out, topk_w, topk_idx, x_attn, *dep, x_out], [WHOLE] * (5 + len(dep)),
                G("MOE_COMBINE_RESIDUAL"))
        x = x_out

    # head (design.md 2.5): norm, lm_head 200 x 512 rows (Q-H), argmax partial over 1024-logit chunks, reduce; L6: norm prologue
    logits = N("logits", (1, d["vocab"]))
    pv, pi = N("argmax_part_value", (1, 100)), N("argmax_part_index", (1, 100), mi.int64)
    if fold:
        _op(mpk, TASK["LM_HEAD"], [x, A(glob["model.norm"], "model_norm"), A(glob["lm_head"], "lm_head"), logits],
            [WHOLE, WHOLE, ROWS_X, COLS_X], G("LM_HEAD"))
    else:
        xn_f = N("xn_final", (1, hid))
        mpk.rmsnorm_layer(input=x, weight=A(glob["model.norm"], "model_norm"), output=xn_f,
                          grid_dim=G("RMS_NORM"), block_dim=(256, 1, 1))
        _op(mpk, TASK["LM_HEAD"], [xn_f, A(glob["lm_head"], "lm_head"), logits], [WHOLE, ROWS_X, COLS_X], G("LM_HEAD"))
    mpk.argmax_partial_layer(input=logits, output=(pv, pi), grid_dim=G("ARGMAX_PARTIAL"), block_dim=(256, 1, 1))
    mpk.argmax_reduce_layer(input=(pv, pi), output=A(glob["output_tokens"], "output_token"),
                            grid_dim=G("ARGMAX_REDUCE"), block_dim=(256, 1, 1))
    assert next(plan, None) is None


def no_gpu_weights():
    """Real-dim weights without a checkpoint: HF decoder layers on the meta device -> relayout_layer() (the same code
    the GPU path runs) -> torch.empty per tensor (pages are never touched, so the 31 GB stays virtual)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ref"))
    from hf.configuration_deepseek import DeepseekV2Config  # noqa: E402  (ref/hf: setup_local.sh)
    from hf.modeling_deepseek import DeepseekV2DecoderLayer  # noqa: E402
    from .reference import COMMON, REAL
    cfg, bf16 = DeepseekV2Config(**COMMON, **REAL), torch.bfloat16
    empty = lambda d: {k: torch.empty(v.shape, dtype=bf16) for k, v in d.items()}
    with torch.device("meta"):
        sd = {dense: DeepseekV2DecoderLayer(cfg, 0 if dense else 1).state_dict() for dense in (True, False)}
    layers = [empty(relayout_layer(sd[l < DIMS["first_k_dense"]], dense=l < DIMS["first_k_dense"]))
              for l in range(DIMS["n_layers"])]
    V, hid = DIMS["vocab"], DIMS["hidden"]
    glob = {"embed_tokens": torch.empty(V, hid, dtype=bf16), "lm_head": torch.empty(V, hid, dtype=bf16),
            "model.norm": torch.empty(hid, dtype=bf16)}
    return layers, glob


def hipcc_offline(mpk, out_dir, cuda_code):
    """PersistentKernel.compile() (persistent_kernel.py:2404-2573) minus the import + init_func, which need a GPU:
    the same hipcc command line, output <out_dir>/test_rank0.so. Used by `--no-gpu --compile`."""
    import shutil
    import subprocess
    import sysconfig
    from mirage.kernel import get_key_paths
    from mirage.mpk.persistent_kernel import HARD_CODE, get_compile_command
    cu, so = os.path.join(out_dir, "test_rank0.cu"), os.path.join(out_dir, "test_rank0.so")
    with open(cu, "w") as f:
        f.write(cuda_code + HARD_CODE)
    scheme = sysconfig.get_default_scheme() if hasattr(sysconfig, "get_default_scheme") else sysconfig._get_default_scheme()
    py_inc = sysconfig.get_paths(scheme="posix_prefix" if scheme == "posix_local" else scheme)["include"]
    _, inc, deps = get_key_paths()
    cmd = get_compile_command(mpk=mpk, target_cc=mpk.target_cc, cc=shutil.which("hipcc"), file_name=cu,
                              py_include_dir=py_inc, mirage_home_path=os.environ["MIRAGE_HOME"], mirage_inc_path=inc,
                              mirage_deps_path=deps, nvshmem_inc_path=None, nvshmem_lib_path=None, mpi_inc_path=None,
                              mpi_lib_path=None, py_so_path=so, profiling=False, use_nvshmem=False,
                              num_workers=mpk.num_workers, num_local_schedulers=mpk.num_local_schedulers,
                              num_remote_schedulers=mpk.num_remote_schedulers, use_cutlass_kernel=False)
    print("hipcc:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    print(f"wrote {so}")


def main():
    """Mirror of demo_30B_A3B.py's --use-mirage path: load HF weights, re-layout, build, emit, compile."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-ai/DeepSeek-Coder-V2-Lite-Base")
    p.add_argument("--max-seq-length", type=int, default=1056)
    p.add_argument("--output-dir", default="build/graph")
    p.add_argument("--compile", action="store_true", help="also hipcc the megakernel (needs the task kernels)")
    p.add_argument("--no-gpu", action="store_true", help="shapes only: meta-device HF layers, torch.empty weights, "
                   "CPU torch, 296+8 workers; --compile then runs hipcc without importing")
    args = p.parse_args()
    T, bf16 = args.max_seq_length, torch.bfloat16
    if args.no_gpu:
        torch.version.hip = torch.version.hip or "no-gpu"  # PersistentKernel: is_rocm -> target_cc 94 without a device
    import mirage as mi
    if args.no_gpu:
        dev, alloc = "cpu", torch.empty
        layers, glob = no_gpu_weights()
        cos = sin = torch.empty(T, 64, dtype=bf16)
        workers = (296, 8)  # get_configurations_from_gpu(0) on 304 CUs (python/mirage/utils.py:53-56)
    else:
        from transformers import AutoModelForCausalLM
        from .reference import rope_cos_sin
        dev, alloc = "cuda", torch.zeros
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=bf16, trust_remote_code=True,
                                                     device_map="cuda")
        layers = []
        for l, layer in enumerate(model.model.layers):
            layers.append(relayout_layer(layer.state_dict(), dense=l < DIMS["first_k_dense"]))
            del layer.mlp
            torch.cuda.empty_cache()
        cs = [rope_cos_sin(torch.tensor(float(p)), model.config, bf16) for p in range(T)]
        cos, sin = torch.stack([c for c, _ in cs]), torch.stack([s for _, s in cs])  # [T, 64] each
        glob = dict(embed_tokens=model.model.embed_tokens.weight, lm_head=model.lm_head.weight,
                    **{"model.norm": model.model.norm.weight})
        workers = mi.get_configurations_from_gpu(0)
    cache_k = [alloc(DIMS["n_heads"], T, 192, dtype=bf16, device=dev) for _ in layers]
    cache_v = [alloc(DIMS["n_heads"], T, 128, dtype=bf16, device=dev) for _ in layers]
    step = torch.tensor([0], dtype=torch.int32, device=dev)
    tokens = torch.zeros(1, T, dtype=torch.int64, device=dev)
    glob.update(rope_cos=cos.to(dev, bf16), rope_sin=sin.to(dev, bf16),
                input_tokens=torch.zeros(1, 1, dtype=torch.int64, device=dev),
                output_tokens=torch.zeros(1, 1, dtype=torch.int64, device=dev))
    mpk = mi.PersistentKernel(  # constructor copied from demo_30B_A3B.py
        mode="offline", world_size=1, mpi_rank=0, num_workers=workers[0], num_local_schedulers=workers[1],
        num_remote_schedulers=0, max_seq_length=T, max_num_batched_requests=1,
        max_num_batched_tokens=1, max_num_pages=1, page_size=T, eos_token_id=-1,
        meta_tensors={"step": step, "tokens": tokens, "input_tokens": glob["input_tokens"],
                      "output_tokens": glob["output_tokens"],
                      "num_new_tokens": torch.zeros(1, dtype=torch.int32, device=dev),
                      "prompt_lengths": torch.zeros(1, dtype=torch.int32, device=dev),
                      "qo_indptr_buffer": torch.empty(2, dtype=torch.int32, device=dev),
                      "paged_kv_indptr_buffer": torch.empty(2, dtype=torch.int32, device=dev),
                      "paged_kv_indices_buffer": torch.empty(1, dtype=torch.int32, device=dev),
                      "paged_kv_last_page_len_buffer": torch.empty(1, dtype=torch.int32, device=dev)},
        profiler_tensor=None, trace_name="", spec_decode_config=None, use_cutlass_kernel=False)
    build(mpk, mi, layers, glob, cache_k, cache_v, t_max=T)
    results = mpk.kn_graph.generate_task_graph(num_gpus=1, my_gpu_id=0)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "task_graph.json"), "w") as f:
        f.write(results["json_file"])
    with open(os.path.join(args.output_dir, "test.cu"), "w") as f:
        f.write(results["cuda_code"])
    print(f"wrote {args.output_dir}/task_graph.json and test.cu")
    if args.compile and args.no_gpu:
        hipcc_offline(mpk, args.output_dir, results["cuda_code"])
    elif args.compile:
        mpk.compile(output_dir=args.output_dir)


try:  # Mirage is only present on the GPU box; layer_plan()/relayout_layer() must import without it
    from mirage.core import CyTBGraph
    from mirage.kernel import TBGraph
except ImportError:  # pragma: no cover
    pass

if __name__ == "__main__":
    main()
