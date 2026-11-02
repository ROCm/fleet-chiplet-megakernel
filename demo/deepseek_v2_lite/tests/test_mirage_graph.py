import os, sys

import pytest, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fleet_ds.graph import build  # noqa: E402
from fleet_ds.mirage_graph import TASK, layer_plan, plan_counts  # noqa: E402
from tests.test_graph import (CHAIN, CHAIN_DENSE_GRID, CHAIN_MOE_GRID, CHAIN_MOE_LAYER_OPS, CHAIN_MOE_LAYER_TASKS,  # noqa: E402
                              CHAIN_PER_TOKEN_BARRIERS, CHAIN_PER_TOKEN_TASKS, DOC_DENSE_GRID, DOC_MOE_GRID, DOC_MOE_LAYER_OPS,
                              DOC_MOE_LAYER_TASKS, DOC_PER_TOKEN_OPS, DOC_PER_TOKEN_TASKS)


@pytest.mark.parametrize("kw, n_ops, n_tasks, moe_ops, moe_tasks, moe_grid, dense_grid", [
    ({}, DOC_PER_TOKEN_OPS, DOC_PER_TOKEN_TASKS, DOC_MOE_LAYER_OPS, DOC_MOE_LAYER_TASKS, DOC_MOE_GRID, DOC_DENSE_GRID),
    (CHAIN, CHAIN_PER_TOKEN_BARRIERS, CHAIN_PER_TOKEN_TASKS, CHAIN_MOE_LAYER_OPS, CHAIN_MOE_LAYER_TASKS, CHAIN_MOE_GRID, CHAIN_DENSE_GRID)],
    ids=["default", "chain"])
def test_plan_equals_graph_py(kw, n_ops, n_tasks, moe_ops, moe_tasks, moe_grid, dense_grid):
    """The Mirage-free plan (op order + grids) is exactly graph.py's op list, hence design-final-fixed 2.4-2.5, for the default
    (DAG + folds) and the chain."""
    ops = build(**kw)["ops"]
    got = plan_counts(layer_plan(**kw))
    assert got == [(o["layer"], o["type"], o["last_task_id"] - o["first_task_id"]) for o in ops]
    assert len(got) == n_ops and sum(n for _, _, n in got) == n_tasks
    moe = [(t, n) for l, t, n in got if l == 1]
    assert len(moe) == moe_ops and sum(n for _, n in moe) == moe_tasks
    assert all(dict(moe)[t] == n for t, n in moe_grid.items())
    assert all(dict((t, n) for l, t, n in got if l == 0)[t] == n for t, n in dense_grid.items())
    assert set(TASK) >= {t for _, t, _ in got}  # every op has a register_task name


def test_dag_plan_equals_graph_py():
    """Decision 12a without the folds (session 8 step s1): the DAG op list of layer_plan() is graph.py's (same ops, same order)."""
    got = plan_counts(layer_plan(dag=1, fold=0))
    assert got == [(o["layer"], o["type"], o["last_task_id"] - o["first_task_id"]) for o in build(dag=1, fold=0)["ops"]]
    assert sum(n for _, _, n in got) == CHAIN_PER_TOKEN_TASKS and set(TASK) >= {t for _, t, _ in got}
    grids = {t: g for l, t, g in layer_plan(dag=1, fold=0) if l == 1}
    assert grids["EXPERT_W13_SILU"] == (6, 32, 1) and grids["EXPERT_W13_SHARED"] == (32, 1, 1) and grids["Q_PROJ"] == (48, 1, 1)


@pytest.mark.parametrize("knobs", [dict(dag=0, fold=1), dict(dag=0, fold=0, gate_tiles=8), dict(dag=1, fold=1),
                                   *(dict(dag=1, shared_w13_tiles=64, shared_w2_tiles=64, w13_tiles=44, attn_rows=r, gate_tiles=8, fold=f)
                                     for r in (32, 16) for f in (0, 1))])
def test_switch_plans_equal_graph_py(knobs):
    """12e switches: layer_plan() == graph.py's op list for every switch combination the GPU plan builds."""
    got = plan_counts(layer_plan(**knobs))
    assert got == [(o["layer"], o["type"], o["last_task_id"] - o["first_task_id"]) for o in build(**knobs)["ops"]]
    assert set(TASK) >= {t for _, t, _ in got}


def test_expert_grid_is_fleet_packing():
    # grid.x = top-k slot, grid.y = N-tile, as runtime.cc packs expert_offset = (bid.y<<16)|bid.x: the chain has 6 routed slots + the
    # shared expert (slot 6), the default DAG (12a) the 6 routed slots (the shared expert is its own ops)
    for kw, slots in ((CHAIN, 7), ({}, 6)):
        grids = {t: g for l, t, g in layer_plan(**kw) if l == 1}
        assert grids["EXPERT_W13_SILU"] == (slots, 32, 1) and grids["EXPERT_W2"] == (slots, 32, 1) and grids["MOE_GATE_TOPK"] == (1, 1, 1)
        assert grids["ATTN_CHUNK"] == (16, 9, 1)
        assert {t: g for l, t, g in layer_plan(**kw) if l == 0}["DOWN_SPLITK"] == (32, 4, 1)


def test_no_gpu_weights_cover_the_checkpoint():
    """--no-gpu weights (meta-device HF layers -> relayout_layer -> torch.empty) are every byte of the real
    checkpoint: sum == metadata.total_size of model.safetensors.index.json, design.md 4.1 shapes."""
    pytest.importorskip("transformers")
    if not os.path.isfile(os.path.join(ROOT, "ref", "hf", "modeling_deepseek.py")):
        pytest.skip("ref/hf missing (setup_local.sh)")
    from fleet_ds.mirage_graph import no_gpu_weights
    layers, glob = no_gpu_weights()
    nbytes = sum(v.numel() * 2 for L in layers for v in L.values()) + sum(v.numel() * 2 for v in glob.values())
    assert nbytes == 31412968448
    assert all(v.device.type == "cpu" and v.dtype == torch.bfloat16 for L in layers for v in L.values())
    assert tuple(layers[0]["mlp.down_proj"].shape) == (4, 2048, 2736) and tuple(layers[0]["W_qkva"].shape) == (3648, 2048)
    assert tuple(layers[1]["mlp.w13"].shape) == (64, 2816, 2048) and tuple(layers[1]["mlp.w13_s"].shape) == (5632, 2048)
