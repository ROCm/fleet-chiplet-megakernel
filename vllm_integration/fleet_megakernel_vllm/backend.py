"""A fleet-mk-shaped KV cache backend for vLLM 0.27.x.

## Why this file exists

fleet_mk's megakernel reads the KV cache as a flat `[entries, kv_cache_stride]`
bf16 array per layer, where `entries = num_blocks * page_size` and
`kv_cache_stride = num_kv_heads * head_dim`. That aliases -- with no copy and no
index arithmetic -- onto a KV pool physically laid out as

    (2, num_blocks, block_size, num_kv_heads, head_size)

which is exactly what vLLM's ROCm aiter backends allocated through 0.25.x. In
0.26 they switched to a *packed* shape,

    (num_blocks, num_kv_heads, block_size, 2 * head_size)

with K and V interleaved in the innermost dim. Neither `t[..., :hs]` nor
`t[..., hs:]` is contiguous there, and no reshape recovers fleet_mk's flat view, so
the zero-copy binding cannot survive on the stock shape.

## What was assumed, and why it was wrong

The obvious reading -- and the one recorded in this project's notes for a while
-- was that the packed layout is a *requirement* of the 0.26+ kernels, so
binding fleet_mk to a modern vLLM would mean writing a custom attention kernel.
That is false, and it was worth testing rather than believing:

  * `ops.reshape_and_cache_flash` is stride-aware. It writes correctly into
    non-contiguous, arbitrarily-strided K and V views.
  * `triton_unified_attention` takes all four K and V cache strides as explicit
    kernel parameters (`stride_k_cache_0..3` from `k.stride(0..3)`).
  * Stock 0.27.1 *already* hands both of those non-contiguous inputs: its own
    `_split_kv_cache` is `kv_cache.transpose(1, 2).split(head_size, dim=-1)`,
    whose outputs have strides like `(16384, 128, 2048, 1)`.

So stride-generality is load-bearing in stock vLLM, not an accident that might
silently regress. A full write-then-read round-trip on both layouts was verified
bit-identical (max abs diff exactly 0.0) -- see check_backend.py.

That reduces the port to this file: declare fleet_mk's shape, and split it on the
leading axis instead of transposing. No new kernel.

## Why it subclasses the aiter backend but does not use aiter

`RocmAiterUnifiedAttentionImpl` is the one impl on 0.27.1 that factors the KV
split into a single `_split_kv_cache` method, which `forward`,
`do_kv_cache_update` and the fused-rope paths all route through. Overriding that
one method redirects the entire layer onto fleet_mk's layout with the stock
`forward` untouched. `TritonAttentionImpl` inlines the same split at three
separate call sites, so subclassing it would mean copying ~100 lines of
`forward` and re-copying them on every vLLM bump.

The catch: that class's `__init__` does `from aiter.ops.triton.unified_attention
import unified_attention`, and the installed aiter (0.1.5.dev157) is a prebuilt
extension linked against torch 2.9 / ROCm 7.0. Under this venv's torch 2.11 it
fails at import with `undefined symbol: _ZN3c103hip21warn_or_error_on_syncEv`.

Rather than rebuild aiter, substitute vLLM's own Triton `unified_attention` --
which is the same algorithm, ships in-tree, and is *already* what the parent
class falls back to for non-causal attention. The one adaptation needed is the
descale convention: the aiter kernel takes scalar `k_descale`/`v_descale`, the
Triton kernel wants them expanded to `(num_seqs, num_kv_heads)`. `_UnifiedAttn`
below does exactly that and nothing else.

## The trap this file deliberately avoids

`RocmAttentionBackend` -- the base class of the aiter backends -- still
*returns* fleet_mk's `(2, nb, bs, n_kv, hs)` shape, which makes it look like a free
ride. It is not usable: its `do_kv_cache_update` routes through
`PagedAttention.split_kv_cache`, which re-views K as
`(nb, n_kv, hs // x, -1, x)` with `x = 16 // element_size()` and writes via
`reshape_and_cache`. That x-reordered K layout is not something fleet_mk's flat
view can alias, so the shape would match while the contents were scrambled.
**Shape alone is not the test; the write path is.** This backend therefore
subclasses the *unified* backend (whose write path is the stride-aware
`reshape_and_cache_flash`) and changes only the shape and the split.
"""

import torch

from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl
from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
    RocmAiterUnifiedAttentionBackend,
    RocmAiterUnifiedAttentionImpl,
)


class _UnifiedAttn:
    """vLLM's Triton unified attention behind the aiter kernel's call signature.

    The two differ in exactly one respect: aiter takes `k_descale`/`v_descale`
    as the layer's scalar scale tensors, while the Triton kernel indexes them
    per (sequence, kv head) and so needs them expanded to
    `(num_seqs, num_kv_heads)`. Stock vLLM does this expansion itself at its own
    Triton call site; this class hoists it so the parent's `forward` can stay
    unmodified.
    """

    def __call__(self, *, q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k,
                 max_seqlen_k, softmax_scale, causal, alibi_slopes, window_size,
                 block_table, softcap, q_descale, k_descale, v_descale, sinks,
                 output_scale):
        from vllm.v1.attention.ops.triton_unified_attention import (
            unified_attention as triton_unified_attention,
        )

        # k is the split K cache, (num_blocks, block_size, num_kv_heads, hs), so
        # shape[2] is the kv-head count -- the same expression stock vLLM uses.
        descale_shape = (cu_seqlens_q.shape[0] - 1, k.shape[2])
        triton_unified_attention(
            q=q, k=k, v=v, out=out,
            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k, max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale, causal=causal,
            alibi_slopes=alibi_slopes, window_size=window_size,
            block_table=block_table, softcap=softcap,
            q_descale=q_descale,
            k_descale=k_descale.expand(descale_shape),
            v_descale=v_descale.expand(descale_shape),
            sinks=sinks, output_scale=output_scale,
        )


class FleetMKUnifiedAttentionImpl(RocmAiterUnifiedAttentionImpl):
    """The aiter unified impl, on fleet_mk's KV layout, without the aiter import."""

    def __init__(self, *args, **kwargs):
        # Skip RocmAiterUnifiedAttentionImpl.__init__ -- its only content beyond
        # the super() call is the aiter import (unloadable here, see module
        # docstring) plus these two attributes.
        RocmAttentionImpl.__init__(self, *args, **kwargs)
        self.unified_attention = _UnifiedAttn()
        self.supports_quant_query_input = True

    def _split_kv_cache(self, kv_cache: torch.Tensor):
        # (2, B, N, H, hs) -> ((B, N, H, hs), (B, N, H, hs))
        #
        # Both halves are contiguous, i.e. strictly easier for the downstream
        # kernels than the transposed views the stock path already feeds them,
        # and each reshapes to fleet_mk's flat [B*N, H*hs] with no copy.
        return kv_cache[0], kv_cache[1]

    # The fused rope / qk-norm-rope KV-update paths are aiter-op fast paths
    # (`rocm_aiter_ops.is_enabled()`), which would import aiter at call time.
    # Declining them costs nothing that matters here -- fleet_mk does its own RoPE
    # inside the megakernel, and these only ever run on the prefill path, where
    # vLLM falls back to the unfused rope + reshape_and_cache_flash sequence.
    def fused_rope_kvcache_supported(self):
        return False

    def fused_qk_norm_rope_kvcache_supported(self):
        return False


class FleetMKAttentionBackend(RocmAiterUnifiedAttentionBackend):
    """Triton unified attention over a KV pool in fleet_mk's split layout."""

    @staticmethod
    def get_name() -> str:
        # Must match the AttentionBackendEnum member this is registered under,
        # because Attention.__init__ does
        # `AttentionBackendEnum[self.attn_backend.get_name()]` and a name with
        # no matching member raises KeyError at layer construction.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[FleetMKUnifiedAttentionImpl]:
        return FleetMKUnifiedAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        # K and V split on the leading axis, each (B, N, H, hs) contiguous --
        # so kv[0] and kv[1] each reshape to fleet_mk's flat
        # [num_blocks * block_size, num_kv_heads * head_size] with no copy.
        return (2, num_blocks, block_size, num_kv_heads, head_size)


def register_fleet_mk_backend():
    """Register FleetMKAttentionBackend as the CUSTOM attention backend.

    `register_backend` is a documented public API on 0.26+; `CUSTOM` is a
    placeholder enum member (value None) that exists precisely for this. Absent
    on 0.11.x, where fleet_mk instead binds to the stock aiter layout directly --
    so the import is local and the caller decides whether to require it.

    Registration is not selection: the engine must also be launched with
    `attention_backend=AttentionBackendEnum.CUSTOM` (harness.py).
    """
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(AttentionBackendEnum.CUSTOM,
                     "fleet_megakernel_vllm.backend.FleetMKAttentionBackend")
    return AttentionBackendEnum.CUSTOM
