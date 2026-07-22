"""Fleet: A chiplet-aware programming model for persistent GPU kernels.

This module provides the top-level API described in the ASPLOS paper,
including decorators for multi-level task specification, distribution
across XCDs, and cross-operator placement constraints.

Usage:
    import mirage.mpk.fleet as fleet

    # Decorator-based task definition (Listing 1)
    @fleet.task(level="xcd", tiling="cooperative_3d", l2_budget="4MB")
    def coop_linear(x, w, output, **kw):
        return fleet.ops.linear(x, w, output, strategy="gang_coop", **kw)

    @fleet.task(level="cu")
    def rmsnorm(x, weight, output):
        return fleet.ops.rmsnorm(x, weight, output)

    @fleet.affinity(placement="inherit_xcd")
    @fleet.task(level="warp")
    def activation(input, output):
        return fleet.ops.silu_mul(input, output)

    # Graph construction
    g = fleet.Graph(mpk)
    x = g.input(tensor, "name")
    out = g.intermediate((bs, hidden), name="out")
    rmsnorm(x, weight, out)
    coop_linear(x, w, out)
"""

# Decorators — the paper's core API
from .dsl.decorators import task, distribute, affinity

# Type annotations for function signatures
from .dsl.types import Tile, Partition, Chunk

# Graph and tensor management
from .dsl.graph import Graph, Tensor

# Pre-built ops (used inside decorated functions)
from .dsl import ops

# Cost model utilities
from .dsl.cost_model import select_strategy, select_k_splits

__all__ = [
    # Decorators
    "task",
    "distribute",
    "affinity",
    # Types
    "Tile",
    "Partition",
    "Chunk",
    # Graph
    "Graph",
    "Tensor",
    # Ops
    "ops",
    # Cost model
    "select_strategy",
    "select_k_splits",
]
