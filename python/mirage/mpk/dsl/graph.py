"""Graph context, Tensor wrapper, and WorkspacePool for the Gang DSL."""

import torch
import mirage as mi


class Tensor:
    """Thin wrapper around Mirage DTensor with shape tracking."""

    def __init__(self, dtensor, name, graph):
        self.dt = dtensor
        self.name = name
        self.graph = graph

    def dim(self, i):
        return self.dt.dim(i)

    @property
    def num_dims(self):
        return self.dt.num_dims

    @property
    def shape(self):
        return tuple(self.dt.dim(i) for i in range(self.dt.num_dims))

    def __repr__(self):
        return f"Tensor({self.name}, shape={self.shape})"


class Graph:
    """Session wrapper around PersistentKernel with auto workspace management.

    Usage:
        g = Graph(mpk)
        x = g.input(torch_tensor, "x")
        out = g.intermediate((bs, hidden), name="out")
    """

    def __init__(self, mpk):
        self.mpk = mpk
        self._ws_pool = {}    # (rows, cols, dtype_str) -> DTensor
        self._dc_pool = {}    # n_blocks -> DTensor
        self._alloc_idx = 0

    def input(self, torch_tensor, name):
        """Attach an existing torch tensor as a graph input."""
        dt = self.mpk.attach_input(torch_tensor=torch_tensor, name=name)
        return Tensor(dt, name, self)

    def intermediate(self, dims, dtype=mi.bfloat16, name=None):
        """Allocate a new intermediate tensor."""
        dt = self.mpk.new_tensor(dims=dims, dtype=dtype, name=name)
        return Tensor(dt, name or f"_intermediate_{self._alloc_idx}", self)

    def shuffle_inputs(self, dtensors, num_groups, name):
        """Shuffle (interleave) multiple DTensors for QKV/GateUp weight packing."""
        dt = self.mpk.shuffle_tensors(
            inputs=dtensors,
            shuffled_dim=0,
            num_groups=num_groups,
            name=name,
        )
        return Tensor(dt, name, self)

    def fuse_inputs(self, dtensors, name):
        """Concatenate multiple DTensors along dim 0."""
        dt = self.mpk.fuse_tensors(
            inputs=dtensors,
            fused_dim=0,
            num_groups=1,
            name=name,
        )
        return Tensor(dt, name, self)

    def get_workspace(self, rows, cols, dtype=torch.float32, hint="ws"):
        """Get or create a workspace tensor (cached by shape+dtype).

        Workspaces are write-only scratch buffers zeroed by the kernel,
        so sharing the same tensor across layers is safe.
        """
        dtype_str = str(dtype)
        key = (rows, cols, dtype_str)
        if key not in self._ws_pool:
            t = torch.zeros((rows, cols), dtype=dtype, device="cuda")
            name = f"_dsl_{hint}_{self._alloc_idx}"
            self._alloc_idx += 1
            self._ws_pool[key] = self.mpk.attach_input(
                torch_tensor=t, name=name,
            )
        return self._ws_pool[key]

    def get_counter(self, n_blocks, hint="dc"):
        """Get or create a done-counter tensor (cached by size).

        Done counters are atomically incremented by workers and reset
        by the last K-split worker, so sharing across layers is safe.
        """
        if n_blocks not in self._dc_pool:
            t = torch.zeros((n_blocks, 1), dtype=torch.int32, device="cuda")
            name = f"_dsl_{hint}_{self._alloc_idx}"
            self._alloc_idx += 1
            self._dc_pool[n_blocks] = self.mpk.attach_input(
                torch_tensor=t, name=name,
            )
        return self._dc_pool[n_blocks]
