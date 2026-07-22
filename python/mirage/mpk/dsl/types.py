"""Type annotations for the Fleet DSL.

These types are used in function signatures to document data partitioning
and tiling intent. They are not enforced at runtime — they serve as
documentation and optional static analysis targets.

Usage (matches paper Listing 1):
    def linear(x: Tile[M, K], w: Partition[N//8, K]) -> Tile[M, N//8]:
        return gemm(x, w)
"""


class Tile:
    """A tiled view of a tensor.

    Tile[M, K] declares a tile with M rows and K columns.
    Used for activation inputs that are broadcast (not partitioned).
    """

    def __class_getitem__(cls, dims):
        if isinstance(dims, tuple):
            return cls(*dims)
        return cls(dims)

    def __init__(self, *dims):
        self.dims = dims

    def __repr__(self):
        return f"Tile[{', '.join(str(d) for d in self.dims)}]"


class Partition:
    """A partitioned slice of a tensor.

    Partition[N//8, K] declares a per-XCD partition: the full tensor
    has N rows, and each of 8 XCDs gets N//8 rows.
    Used for weight matrices partitioned across chiplets.
    """

    def __class_getitem__(cls, dims):
        if isinstance(dims, tuple):
            return cls(*dims)
        return cls(dims)

    def __init__(self, *dims):
        self.dims = dims

    def __repr__(self):
        return f"Partition[{', '.join(str(d) for d in self.dims)}]"


class Chunk:
    """A KV cache chunk for attention.

    Declares that this input is a chunk of the KV cache,
    with one chunk per XCD for cache affinity.
    """

    def __repr__(self):
        return "Chunk"
