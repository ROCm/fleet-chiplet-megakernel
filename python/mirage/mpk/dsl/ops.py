"""Op functions for the Gang DSL.

Each function wraps one or more mpk.*_layer() calls, auto-computing
grid_dim/block_dim and allocating workspaces as needed.
"""

import math
import torch

from .cost_model import select_strategy, select_k_splits


# ---------------------------------------------------------------------------
# Element-wise / simple ops
# ---------------------------------------------------------------------------

def rmsnorm(x, weight, output):
    """RMSNorm: one task per token row."""
    g = x.graph
    mpk = g.mpk
    mpk.rmsnorm_layer(
        input=x.dt, weight=weight.dt, output=output.dt,
        grid_dim=(mpk.max_num_batched_tokens, 1, 1),
        block_dim=(128, 1, 1),
    )
    return output


def embed(x, weight, output, input_source=1):
    """Embedding lookup."""
    g = x.graph
    mpk = g.mpk
    mpk.embed_layer(
        input=x.dt, weight=weight.dt, output=output.dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
        input_source=input_source,
    )
    return output


def silu_mul(input, output):
    """SiLU-gated multiplication: input has 2x the columns of output."""
    g = input.graph
    mpk = g.mpk
    # input = [bs, 2*intermediate], output = [bs, intermediate]
    # Each task handles 128 output elements → reads 256 input elements
    out_cols = output.dim(1) if output.num_dims == 2 else output.dim(0)
    n_tasks = out_cols // 128
    mpk.silu_mul_layer(
        input=input.dt, output=output.dt,
        grid_dim=(n_tasks, 1, 1),
        block_dim=(128, 1, 1),
    )
    return output


def silu_mul_linear(input, weight, output, residual=None):
    """Fused SiLU×mul + down-projection linear (+ optional residual add).

    Combines silu_mul and linear into a single task, eliminating the
    intermediate activation round-trip through HBM.

    Args:
        input:    Tensor [bs, 2*intermediate] (gate+up concatenated)
        weight:   Tensor [hidden, intermediate] (down_proj weight)
        output:   Tensor [bs, hidden]
        residual: optional Tensor [bs, hidden] for fused residual add
    """
    g = input.graph
    mpk = g.mpk
    N = weight.dim(0)  # hidden_size
    grid_x = N // 64

    if residual is not None:
        mpk.silu_mul_linear_with_residual_layer(
            input=input.dt, weight=weight.dt, residual=residual.dt,
            output=output.dt,
            grid_dim=(grid_x, 1, 1),
            block_dim=(256, 1, 1),
        )
    else:
        mpk.silu_mul_linear_layer(
            input=input.dt, weight=weight.dt, output=output.dt,
            grid_dim=(grid_x, 1, 1),
            block_dim=(256, 1, 1),
        )
    return output


def linear_silu(input, weight, output, output_stride=None, m_tiles=None,
                tile_n=64):
    """Gang linear with fused SiLU+mul (gate+up GEMM fusion).

    Takes interleaved gate+up weight [gate_up_size, K] and produces
    [bs, intermediate_size] output with SiLU+mul applied in-kernel.
    Eliminates the separate silu_mul step entirely.

    Args:
        input:        Tensor [bs, K]
        weight:       Tensor [gate_up_size, K] (interleaved gate+up)
        output:       Tensor [bs, intermediate_size]
        output_stride: stride in output columns (default = intermediate_size)
        m_tiles:      M-tile count (default = max(1, bs // 16))
        tile_n:       N-tile size (default 64)
    """
    g = input.graph
    mpk = g.mpk
    bs = mpk.max_num_batched_tokens
    inter_size = output.dim(1) if output.num_dims == 2 else output.dim(0)
    stride = output_stride if output_stride is not None else inter_size
    mt = m_tiles or max(1, bs // 16)

    mpk.gang_linear_silu_layer(
        input=input.dt, weight=weight.dt, output=output.dt,
        tile_n=tile_n, output_stride=stride,
        m_tiles=mt, block_dim=(256, 1, 1),
    )
    return output


def argmax(input, part_val, part_idx, output, num_tasks=None):
    """Two-phase argmax: partial reduction then final reduce."""
    g = input.graph
    mpk = g.mpk
    nt = num_tasks or input.dim(1) // 1024
    mpk.argmax_partial_layer(
        input=input.dt,
        output=(part_val.dt, part_idx.dt),
        grid_dim=(nt, 1, 1),
        block_dim=(128, 1, 1),
    )
    mpk.argmax_reduce_layer(
        input=(part_val.dt, part_idx.dt),
        output=output.dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    return output


def identity(input, output, dependent=None):
    """Identity copy (for scheduling dependencies)."""
    g = input.graph
    mpk = g.mpk
    last_dim = input.num_dims - 1
    size = input.dim(last_dim)
    n_tasks = size // 64 if size >= 64 else 1
    mpk.identity_layer(
        input=input.dt, output=output.dt,
        grid_dim=(n_tasks, 1, 1),
        block_dim=(128, 1, 1),
        dependent_tensor=dependent.dt if dependent else None,
    )
    return output


# ---------------------------------------------------------------------------
# Linear dispatch
# ---------------------------------------------------------------------------

def _grid_for_standard_linear(output_size, target_cc):
    """Compute grid_dim[0] for standard (non-gang) linear layers."""
    if target_cc == 94:
        # MI300X: always use 64-column tiles
        return output_size // 64
    # NVIDIA paths
    if output_size / 96 > 400:
        assert output_size % 256 == 0
        return output_size // 256
    if output_size % 96 == 0:
        return 96
    if output_size % 64 == 0:
        return 64
    return output_size // 64


def linear(input, weight, output, residual=None, strategy="auto",
           k_splits=None, output_stride=None, m_tiles=None, tile_n=64,
           num_tasks=None):
    """Unified linear dispatch.

    Auto-selects between standard, gang, gang_coop, and gang_splitk
    strategies based on the cost model (or override with strategy=).

    Args:
        input:   Tensor [batch, K]
        weight:  Tensor [N, K]
        output:  Tensor [batch, N] (or wider if output_stride > N)
        residual: optional Tensor [batch, N] for fused residual add
        strategy: "auto", "standard", "gang", "gang_coop", "gang_splitk"
        k_splits: override K-split count (for coop/splitk)
        output_stride: stride in output columns (default = weight.dim(0))
        m_tiles: override M-tile count (for gang)
        tile_n: N-tile size for gang (default 64)
        num_tasks: override grid_dim[0] for standard linear
    """
    g = input.graph
    mpk = g.mpk
    N = weight.dim(0)
    K = weight.dim(1)
    bs = mpk.max_num_batched_tokens
    stride = output_stride if output_stride is not None else N

    if strategy == "auto":
        strategy = select_strategy(N, K, bs, mpk.target_cc)

    if strategy == "gang_coop":
        _linear_gang_coop(g, mpk, input, weight, output, residual,
                          N, K, bs, stride, k_splits)
    elif strategy == "gang_splitk":
        _linear_gang_splitk(g, mpk, input, weight, output, residual,
                            N, K, bs, stride, k_splits)
    elif strategy == "gang":
        _linear_gang(g, mpk, input, weight, output, residual,
                     N, bs, stride, m_tiles, tile_n)
    else:
        _linear_standard(g, mpk, input, weight, output, residual, N,
                         num_tasks)

    return output


def _linear_gang_coop(g, mpk, input, weight, output, residual,
                      N, K, bs, stride, k_splits):
    """Gang cooperative 3D-tiled linear."""
    m_per_tile = 16
    m_tiles = math.ceil(bs / m_per_tile)
    padded = m_tiles * m_per_tile
    chunk_n = N // 8
    n_tiles = chunk_n // 64

    ks = k_splits or select_k_splits(K, m_tiles)

    ws = g.get_workspace(padded, N, torch.float32, f"coop_{N}")
    dc = g.get_counter(m_tiles * n_tiles * 8, f"coop_dc_{N}")

    if residual is not None:
        mpk.gang_coop_linear_with_residual_layer(
            input=input.dt, weight=weight.dt, residual=residual.dt,
            workspace=ws, done_counter=dc, output=output.dt,
            k_splits=ks, output_stride=stride,
            block_dim=(256, 1, 1),
        )
    else:
        mpk.gang_coop_linear_layer(
            input=input.dt, weight=weight.dt,
            workspace=ws, done_counter=dc, output=output.dt,
            k_splits=ks, output_stride=stride,
            block_dim=(256, 1, 1),
        )


def _linear_gang_splitk(g, mpk, input, weight, output, residual,
                         N, K, bs, stride, k_splits):
    """Gang split-K linear (per-XCD weight partition + K-splitting)."""
    chunk_n = N // 8
    n_tiles = chunk_n // 64

    ks = k_splits or select_k_splits(K, n_tiles)

    ws = g.get_workspace(bs, N, torch.float32, f"splitk_{N}")
    dc = g.get_counter(n_tiles * 8, f"splitk_dc_{N}")

    if residual is not None:
        mpk.gang_splitk_linear_with_residual_layer(
            input=input.dt, weight=weight.dt, residual=residual.dt,
            workspace=ws, done_counter=dc, output=output.dt,
            k_splits=ks, output_stride=stride,
            block_dim=(256, 1, 1),
        )
    else:
        mpk.gang_splitk_linear_layer(
            input=input.dt, weight=weight.dt,
            workspace=ws, done_counter=dc, output=output.dt,
            k_splits=ks, output_stride=stride,
            block_dim=(256, 1, 1),
        )


def _linear_gang(g, mpk, input, weight, output, residual,
                  N, bs, stride, m_tiles, tile_n):
    """Gang linear (per-XCD weight partition, no K-splitting)."""
    mt = m_tiles or 1

    if residual is not None:
        mpk.gang_linear_with_residual_layer(
            input=input.dt, weight=weight.dt, residual=residual.dt,
            output=output.dt, tile_n=tile_n, output_stride=stride,
            m_tiles=mt, block_dim=(256, 1, 1),
        )
    else:
        mpk.gang_linear_layer(
            input=input.dt, weight=weight.dt, output=output.dt,
            tile_n=tile_n, output_stride=stride,
            m_tiles=mt, block_dim=(256, 1, 1),
        )


def _linear_standard(g, mpk, input, weight, output, residual, N,
                     num_tasks=None):
    """Standard linear (no gang, per-block N-tile)."""
    grid_x = num_tasks or _grid_for_standard_linear(N, mpk.target_cc)
    block_dim = (256, 1, 1) if mpk.target_cc == 94 else (128, 1, 1)

    if residual is not None:
        mpk.linear_with_residual_layer(
            input=input.dt, weight=weight.dt, residual=residual.dt,
            output=output.dt,
            grid_dim=(grid_x, 1, 1), block_dim=block_dim,
        )
    else:
        mpk.linear_layer(
            input=input.dt, weight=weight.dt, output=output.dt,
            grid_dim=(grid_x, 1, 1), block_dim=block_dim,
        )


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def paged_attention_split_kv(input, k_cache, v_cache, q_norm, k_norm,
                              cos_pe, sin_pe, lse, attn_tmp, attn_out,
                              num_q_heads, num_kv_chunks, num_kv_heads,
                              head_dim, gang=None):
    """Split-KV paged attention + merge.

    Args:
        input:       Tensor [bs, fused_qkv_dim]
        k_cache:     Tensor [pages, page_size, kv_heads, head_dim]
        v_cache:     Tensor [pages, page_size, kv_heads, head_dim]
        q_norm:      Tensor [head_dim] or None
        k_norm:      Tensor [head_dim] or None
        cos_pe:      Tensor [seq_len, head_dim] or None
        sin_pe:      Tensor [seq_len, head_dim] or None
        lse:         Tensor [bs, chunks*qo_per_kv, kv_heads]
        attn_tmp:    Tensor [bs, chunks, hidden]  (split-KV intermediates)
        attn_out:    Tensor [bs, hidden]
        num_q_heads: total Q heads (before GQA grouping)
        num_kv_chunks: number of KV cache chunks
        num_kv_heads: number of KV heads
        head_dim:    dimension per head
        gang:        force gang mode (None = auto based on target_cc)
    """
    g = input.graph
    mpk = g.mpk
    # Gang attention is disabled: only 2% improvement, and gang_attention
    # kernel headers are not yet ported to this branch.
    use_gang = gang if gang is not None else False

    # Unwrap DTensors, handling None for optional inputs
    q_norm_dt = q_norm.dt if q_norm is not None else None
    k_norm_dt = k_norm.dt if k_norm is not None else None
    cos_dt = cos_pe.dt if cos_pe is not None else None
    sin_dt = sin_pe.dt if sin_pe is not None else None

    if use_gang:
        mpk.gang_paged_attention_split_kv_layer(
            input=input.dt, k_cache=k_cache.dt, v_cache=v_cache.dt,
            q_norm=q_norm_dt, k_norm=k_norm_dt,
            cos_pos_embed=cos_dt, sin_pos_embed=sin_dt,
            lse=lse.dt, output=attn_tmp.dt,
            attention_params=(num_q_heads, num_kv_chunks),
            block_dim=(128, 1, 1),
        )
        mpk.gang_paged_attention_split_kv_merge_layer(
            lse=lse.dt, output_tmp=attn_tmp.dt, output=attn_out.dt,
            attention_params=(num_q_heads, head_dim, num_kv_heads),
            block_dim=(128, 1, 1),
        )
    else:
        mpk.paged_attention_split_kv_layer(
            input=input.dt, k_cache=k_cache.dt, v_cache=v_cache.dt,
            q_norm=q_norm_dt, k_norm=k_norm_dt,
            cos_pos_embed=cos_dt, sin_pos_embed=sin_dt,
            lse=lse.dt, output=attn_tmp.dt,
            attention_params=(num_q_heads, num_kv_chunks),
            grid_dim=(mpk.max_num_batched_requests, num_kv_heads,
                      num_kv_chunks),
            block_dim=(128, 1, 1),
        )
        mpk.paged_attention_split_kv_merge_layer(
            lse=lse.dt, output_tmp=attn_tmp.dt, output=attn_out.dt,
            attention_params=(num_q_heads, head_dim),
            grid_dim=(mpk.max_num_batched_requests, num_kv_heads, 1),
            block_dim=(128, 1, 1),
        )

    return attn_out


def paged_attention_ck_fmha(input, k_cache, v_cache, q_norm, k_norm,
                             cos_pe, sin_pe, q_workspace, o_acc, lse_acc,
                             attn_out, num_q_heads, num_kv_heads, head_dim,
                             num_kv_chunks=1):
    """CK FMHA paged attention (3-phase: KV update → FMHA → merge).

    More efficient than split-KV: fewer tasks, fused KV cache update.

    Args:
        input:       Tensor [bs, fused_qkv_dim]
        k_cache:     Tensor [pages, page_size, kv_heads, head_dim]
        v_cache:     Tensor [pages, page_size, kv_heads, head_dim]
        q_norm:      Tensor [head_dim]
        k_norm:      Tensor [head_dim]
        cos_pe:      Tensor [seq_len, head_dim]
        sin_pe:      Tensor [seq_len, head_dim]
        q_workspace: Tensor [bs, q_heads * head_dim] (bf16)
        o_acc:       Tensor [bs, kv_heads * chunks * qo_per_kv * head_dim] (fp32)
        lse_acc:     Tensor [bs, kv_heads * chunks * qo_per_kv] (fp32)
        attn_out:    Tensor [bs, q_heads * head_dim]
        num_q_heads: total Q heads
        num_kv_heads: number of KV heads
        head_dim:    dimension per head
        num_kv_chunks: KV chunks (default 1 for CK FMHA)
    """
    g = input.graph
    mpk = g.mpk

    q_norm_dt = q_norm.dt if q_norm is not None else None
    k_norm_dt = k_norm.dt if k_norm is not None else None
    cos_dt = cos_pe.dt if cos_pe is not None else None
    sin_dt = sin_pe.dt if sin_pe is not None else None

    # Phase A: KV cache update + Q preprocessing
    mpk.kv_cache_update_layer(
        input=input.dt,
        k_cache=k_cache.dt, v_cache=v_cache.dt,
        q_norm=q_norm_dt, k_norm=k_norm_dt,
        cos_pos_embed=cos_dt, sin_pos_embed=sin_dt,
        q_workspace=q_workspace.dt,
        grid_dim=(mpk.max_num_batched_requests, num_kv_heads, 1),
        block_dim=(256, 1, 1),
    )
    # Phase B: CK FMHA
    mpk.paged_attention_ck_fmha_layer(
        q_workspace=q_workspace.dt,
        k_cache=k_cache.dt, v_cache=v_cache.dt,
        o_acc=o_acc.dt, lse_acc=lse_acc.dt,
        attention_params=(num_q_heads, num_kv_chunks,
                          mpk.max_num_batched_requests),
        grid_dim=(mpk.max_num_batched_requests, num_kv_heads, num_kv_chunks),
        block_dim=(256, 1, 1),
    )
    # Phase C: Merge
    mpk.paged_attention_ck_fmha_merge_layer(
        lse=lse_acc.dt, output_tmp=o_acc.dt, output=attn_out.dt,
        attention_params=(num_q_heads, head_dim,
                          num_kv_chunks, num_kv_heads),
        grid_dim=(mpk.max_num_batched_requests, num_kv_heads, 1),
        block_dim=(256, 1, 1),
    )
    return attn_out
