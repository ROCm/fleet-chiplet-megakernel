"""Fleet DSL: Multi-level task API for chiplet GPUs.

Usage:
    import mirage.mpk.dsl as dsl

    g = dsl.Graph(mpk)
    x = g.input(torch_tensor, "name")
    out = g.intermediate((bs, hidden), name="out")
    dsl.rmsnorm(x, weight, out)
    dsl.linear(x, w, out)  # auto-selects fleet strategy
"""

from .graph import Graph, Tensor
from .ops import (
    rmsnorm,
    embed,
    linear,
    linear_silu,
    silu_mul,
    silu_mul_linear,
    argmax,
    identity,
    paged_attention_split_kv,
    paged_attention_ck_fmha,
)
from .cost_model import select_strategy, select_k_splits
from .decorators import task, distribute, affinity
from .types import Tile, Partition, Chunk

__all__ = [
    # Graph
    "Graph",
    "Tensor",
    # Ops
    "rmsnorm",
    "embed",
    "linear",
    "linear_silu",
    "silu_mul",
    "argmax",
    "identity",
    "paged_attention_split_kv",
    "paged_attention_ck_fmha",
    # Cost model
    "select_strategy",
    "select_k_splits",
    # Decorators
    "task",
    "distribute",
    "affinity",
    # Types
    "Tile",
    "Partition",
    "Chunk",
]
