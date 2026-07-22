"""Decorator-based API for the Fleet programming model.

Provides @task, @distribute, and @affinity decorators matching the paper's
Listing 1 (Task Specification DSL). These decorators attach scheduling
metadata to functions; the underlying ops read this metadata to select
the appropriate dispatch strategy.

Usage:
    from mirage.mpk.dsl.decorators import task, distribute, affinity

    @task(level="xcd", tiling="cooperative_3d", l2_budget="4MB")
    def linear(x, w, output, **kw):
        return ops.linear(x, w, output, **kw)

    @distribute(dim=0, num_xcds=8)
    def qkv_proj(x, w_qkv, output, **kw):
        return linear(x, w_qkv, output, **kw)

    @affinity(placement="inherit_xcd")
    @task(level="warp")
    def silu_mul(input, output):
        return ops.silu_mul(input, output)
"""

from functools import wraps


def _parse_budget(budget):
    """Parse an L2 budget string like '4MB' into bytes."""
    if budget is None:
        return None
    if isinstance(budget, (int, float)):
        return int(budget)
    s = str(budget).strip().upper()
    if s.endswith("MB"):
        return int(float(s[:-2]) * 1024 * 1024)
    if s.endswith("KB"):
        return int(float(s[:-2]) * 1024)
    if s.endswith("B"):
        return int(s[:-1])
    return int(s)


class task:
    """Annotate a function with its hardware scheduling level.

    Args:
        level: Hardware scope — "warp", "cu", "xcd", or "auto".
            - "warp":  element-wise ops (SiLU, residual add). Standard dispatch.
            - "cu":    single-block ops (attention head, RMSNorm). Standard dispatch.
            - "xcd":   cache-scoped ops (GEMM partition). Fleet dispatch (8 tasks).
            - "auto":  let the cost model decide.
        tiling: Tiling strategy for XCD-level tasks.
            - "cooperative_3d": 3D tile decomposition (m_tile, k_split, n_tile)
                                for intra-XCD L2 weight reuse.
            - "split_k":        K-splitting across workers within each XCD.
            - "auto":           let the cost model decide.
        l2_budget: L2 cache budget per XCD. Accepts bytes (int) or
                   strings like "4MB". Used by the cost model to decide
                   whether cooperative tiling is needed.
    """

    def __init__(self, level="auto", tiling="auto", l2_budget=None):
        self.level = level
        self.tiling = tiling
        self.l2_budget = l2_budget

    def __call__(self, fn):
        meta = getattr(fn, "_fleet_meta", {})
        meta.update({
            "level": self.level,
            "tiling": self.tiling,
            "l2_budget": _parse_budget(self.l2_budget),
        })
        fn._fleet_meta = meta
        return fn

    def __repr__(self):
        return (f"@task(level={self.level!r}, tiling={self.tiling!r}, "
                f"l2_budget={self.l2_budget!r})")


class distribute:
    """Distribute an operator across XCDs.

    Wraps a task function to partition its weight tensor along
    `dim` across `num_xcds` chiplets. Each XCD processes
    a 1/num_xcds slice of the weight, keeping it in local L2.

    Args:
        dim: Dimension to partition (0 = rows/output dim).
        num_xcds: Number of XCDs to partition across (default 8).

    Usage:
        @distribute(dim=0, num_xcds=8)
        def qkv_proj(x, w_qkv, output, **kw):
            return linear(x, w_qkv, output, **kw)
    """

    def __init__(self, dim=0, num_xcds=8):
        self.dim = dim
        self.num_xcds = num_xcds

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        meta = getattr(fn, "_fleet_meta", {})
        meta["partition"] = {
            "dim": self.dim,
            "num_xcds": self.num_xcds,
        }
        wrapper._fleet_meta = meta
        return wrapper

    def __repr__(self):
        return f"@distribute(dim={self.dim}, num_xcds={self.num_xcds})"


class affinity:
    """Declare cross-operator XCD affinity constraints.

    Specifies that a task should run on the same XCD as its
    input producer, enabling L2 reuse of intermediate activations.

    Args:
        placement: Placement strategy.
            - "inherit_xcd": run on the same XCD as the producing task.
              Currently accepted as a scheduling hint; full enforcement
              requires scheduler support (future work).

    Usage:
        @affinity(placement="inherit_xcd")
        @task(level="warp")
        def silu_mul(input, output):
            return ops.silu_mul(input, output)
    """

    def __init__(self, placement="inherit_xcd"):
        self.placement = placement

    def __call__(self, fn):
        meta = getattr(fn, "_fleet_meta", {})
        meta["placement"] = self.placement
        fn._fleet_meta = meta
        return fn

    def __repr__(self):
        return f"@affinity(placement={self.placement!r})"


def get_meta(fn):
    """Read the scheduling metadata attached by decorators."""
    return getattr(fn, "_fleet_meta", {})


def strategy_from_meta(meta):
    """Map decorator metadata to a linear dispatch strategy.

    Returns one of: "standard", "gang", "gang_coop", "gang_splitk", "auto".
    """
    level = meta.get("level", "auto")
    tiling = meta.get("tiling", "auto")

    if level in ("warp", "cu"):
        return "standard"
    if level == "xcd":
        if tiling == "cooperative_3d":
            return "gang_coop"
        if tiling == "split_k":
            return "gang_splitk"
        return "gang"
    return "auto"
