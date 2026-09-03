"""Spec-driven runtime glue for the fleet_mk megakernel: ctypes loader, buffer pool,
pointer-table builders, and a decode driver.

Everything here is parameterized by a `ModelSpec` (spec.py) so one launch path
serves every fleet_mk model. The decode hot path (a single prebuilt-ptr-table kernel
launch) is identical across models; all per-model variation lives in setup:
which .so/launch symbol to load, buffer sizing, and how the pointer table is laid
out (dense gate/up/down vs MoE experts). Weight packing is the caller's job
(packing.py / packing_moe.py).
"""

import ctypes
import math
import os

import torch


# ── kernel loader ────────────────────────────────────────────────────────────
def load_kernel(spec):
    """Load the compiled fleet_mk .so and declare the ctypes ABI for `spec`.

    The launch ABI is identical across models up to `argmax_output`; past it each
    family appends its own trailing args, carried on the spec as a tuple of
    ctypes types (`extra_launch_args`). Dense appends only timing_buf; the fused
    MoE kernel also takes embed_weight, cur_token_id and decode_ctrl because the
    embedding lookup moved on-device.

    This must match the emitted C signature exactly. A ctypes list that is short
    or transposed is stack corruption, not a clean error -- which is why the arg
    list is data on the spec rather than a bool tested here.
    """
    lib = ctypes.CDLL(spec.so_path)

    init = getattr(lib, spec.init_symbol)
    init.restype = None
    init.argtypes = []

    launch = getattr(lib, spec.launch_symbol)
    launch.restype = None
    argtypes = [
        ctypes.c_int,       # num_active_tokens
        ctypes.c_float,     # attn_scale
        ctypes.c_void_p,    # cos_ptr
        ctypes.c_void_p,    # sin_ptr
        ctypes.c_void_p,    # qo_indptr
        ctypes.c_void_p,    # kv_indptr
        ctypes.c_void_p,    # kv_indices
        ctypes.c_void_p,    # kv_last_page_len
        ctypes.c_void_p,    # ptr_table
        ctypes.c_void_p,    # counter_buf
        ctypes.c_void_p,    # lm_norm_weight
        ctypes.c_void_p,    # lm_norm_scratch
        ctypes.c_void_p,    # lm_mxfp4_weight
        ctypes.c_void_p,    # lm_bias
        ctypes.c_void_p,    # argmax_output
    ]
    argtypes.extend(spec.extra_launch_args)
    argtypes.append(ctypes.c_void_p)       # stream
    launch.argtypes = argtypes

    fin = getattr(lib, spec.finalize_symbol)
    fin.restype = None
    fin.argtypes = []
    return lib


# Backwards-compatible alias (old callers / __init__ re-export).
def load_qwen3_kernel(so_path):
    from .spec import load_spec
    spec = load_spec("Qwen3ForCausalLM")
    spec.so_path = so_path
    return load_kernel(spec)


# ── workspace + paged-KV buffers ─────────────────────────────────────────────
class FleetMKBuffers:
    """All workspace + paged-KV buffers the megakernel reads/writes, sized from
    `spec`. Per-layer K/V caches are separate bf16 tensors of shape
    [max_num_pages*page_size, kv_cache_stride]; with alloc_kv=False they are left
    None and aliased onto vLLM's kv_cache via set_kv_aliases() (zero-copy path).
    """

    def __init__(self, spec, bs=1, max_num_pages=16, device="cuda",
                 alloc_kv=True):
        self.spec = spec
        self.bs = bs
        self.max_num_pages = max_num_pages
        self.page_size = spec.page_size
        self.device = device

        S = spec
        # The kernel runs on the MFMA-aligned padded hidden dim; for dense this
        # equals hidden_size, for MoE it is wider (2944 vs 2880). Hidden-width
        # workspace buffers use it so the MoE kernel's padded indexing lands.
        pw = S.padded_hidden_size
        num_kv_entries = max_num_pages * S.page_size
        z = lambda *s, dt=torch.bfloat16: torch.zeros(*s, dtype=dt, device=device)

        # Per-layer paged KV cache (aliasable).
        if alloc_kv:
            self.buf_k_cache = [z(num_kv_entries, S.kv_cache_stride)
                                for _ in range(S.num_layers)]
            self.buf_v_cache = [z(num_kv_entries, S.kv_cache_stride)
                                for _ in range(S.num_layers)]
        else:
            self.buf_k_cache = [None] * S.num_layers
            self.buf_v_cache = [None] * S.num_layers

        # Residual ping-pong pair (layer 0 input = embedding, in residual_a).
        self.buf_residual_a = z(bs, pw)
        self.buf_residual_b = z(bs, pw)

        # Norm scratch.
        self.buf_norm_scratch1 = z(bs, pw)
        self.buf_norm_scratch2 = z(bs, pw)

        # QKV GEMM output.
        self.buf_qkv_output = z(bs, S.qkv_output_size)

        # Attention.
        self.buf_q_workspace = z(bs, S.num_q_heads * S.head_dim)
        self.buf_attn_out = z(bs, S.num_q_heads * S.head_dim)
        self.buf_oproj_out = z(bs, pw)

        # Split-KV accumulation (same formula for dense + MoE).
        o_acc_dim = S.num_kv_heads * S.num_kv_chunks * S.q_per_kv * S.head_dim
        self.buf_o_acc_f32 = z(bs, o_acc_dim, dt=torch.float32)
        lse_dim = S.num_kv_heads * S.num_kv_chunks * S.q_per_kv
        self.buf_lse_acc = z(bs, lse_dim, dt=torch.float32)

        # LM head. The kernel's fused final-RMSNorm writes the normed hidden into
        # lm_norm_scratch at padded width; vLLM's lm_head reads the [:hidden] slice.
        self.buf_argmax_out = torch.zeros(1, dtype=torch.int64, device=device)
        self.buf_lm_norm_scratch = z(bs, pw)
        # Logits sink for the fused MXFP4 lm_head. The kernel's argmax epilogue
        # already computes every logit and discards all but the max, so filling
        # this costs one 402 KB write; the bf16 lm_head it replaces on the vLLM
        # path costs a 1.16 GB weight read (0.200 ms/token measured). MoE only:
        # the dense kernel has no logits_output in its ABI.
        self.buf_logits = z(bs, S.padded_vocab_size) if S.is_moe else None

        # MLP buffers -- family specific.
        if S.is_moe:
            self._alloc_moe(z)
        else:
            self.buf_gateup_scratch = z(bs, S.gateup_output_size)
            self.buf_swiglu_out = z(bs, S.intermediate_size)

        # Counters. Layout, in order, and every region past the per-layer blocks
        # is addressed by the kernel with an ABSOLUTE offset -- under-allocating
        # any of them is an unchecked atomicAdd past the end of the buffer, which
        # presents as a memory fault at a plausible but meaningless address rather
        # than as an error here. (This is the COUNTERS_PER_LAYER 40 vs 103 bug in
        # a different guise; see spec.py.)
        #
        #   [per-layer blocks | rank | decode-iter | embed barrier | ILB slack]
        #
        # The MoE kernel's EMBED_BARRIER_BASE is
        # `NUM_LAYERS*COUNTERS_PER_LAYER + NUM_XCDS*16 + 16`, i.e. it assumes the
        # rank block then one decode-iter cache line precede it.
        counter_total = (S.num_layers * S.counters_per_layer
                         + S.rank_counter_ints
                         + S.counter_tail_ints)
        self.buf_counter = torch.zeros(counter_total, dtype=torch.int32, device=device)
        self.rank_counter_offset = S.num_layers * S.counters_per_layer
        self.rank_counter_slice = self.buf_counter[
            self.rank_counter_offset:self.rank_counter_offset + S.rank_counter_ints]

        # Subphase timing (dense build only; MoE has no timing slots).
        timing_total = max(
            1, S.num_layers * S.timing_slots_per_layer + S.timing_tail_slots)
        self.buf_timing = torch.zeros(timing_total, dtype=torch.int64, device=device)

    def _alloc_moe(self, z):
        """MoE-only workspace (router + per-expert SwiGLU + attn-block residual),
        sized from spec and matching the demo.py reference byte-for-byte. All the
        hidden/intermediate-width tensors run at padded_hidden_size (== padded
        intermediate for GPT-OSS). workspace_f32 and moe_barrier are re-zeroed each
        decode step (see decode_step); the rest are overwritten by the kernel."""
        S = self.spec
        bs = self.bs
        dev = self.device
        pw = S.padded_hidden_size
        n_exp = S.num_experts
        top_k = S.num_experts_per_tok

        # f32 residual accumulator. Fleet's W2 epilogue writes
        # (b * MOE_WS_SLOTS + slot) * HIDDEN, with MOE_WS_SLOTS == top_k, so this
        # must be top_k slabs wide -- a single hidden-width buffer is an
        # unchecked overrun. Starts zero; the kernel owns it after that.
        self.buf_workspace_f32 = torch.zeros(
            bs, top_k * pw, dtype=torch.float32, device=dev)
        # Per-layer attention-block residual (layers 1+ read this as their residual).
        self.buf_attn_proj_out = z(bs, pw)
        # QKV's ResAdd+RMSNorm output. Distinct from the residual buffers on
        # purpose -- see the ptr-table comment at output[0].
        self.buf_x_output = z(bs, pw)
        # Router / expert routing.
        self.buf_topk_weight = torch.zeros(bs, top_k, dtype=torch.float32, device=dev)
        self.buf_routing_indices = torch.zeros(n_exp, bs, dtype=torch.int32, device=dev)
        self.buf_active_expert_ids = torch.zeros(n_exp + 1, dtype=torch.int32, device=dev)
        self.buf_moe_routing_wt = self.buf_topk_weight  # alias (demo.py)
        # Per-(token, selected-expert) SwiGLU output, intermediate at padded width.
        self.buf_swiglu_out = z(bs, top_k, pw)
        # Router logits scratch (bf16; ptr table offsets it by 2 bytes/expert per XCD).
        self.buf_logits_scratch = z(bs, n_exp)
        # Per-expert barrier. Fleet indexes MOE_BAR_STRIDE = 10 * 16 ints per
        # expert (8 per-XCD release flags + arrival on line 8). 16 ints/expert
        # aliases experts into each other and puts the arrival counter off the
        # allocation. NOT zeroed between decode steps: W13->W2 expected is
        # layer_idx + 1 on a counter fleet never resets. Matches demo.py.
        self.buf_moe_barrier = torch.zeros(160 * n_exp, dtype=torch.int32, device=dev)

    def set_kv_aliases(self, k_tensors, v_tensors):
        """Point the per-layer K/V caches at externally-owned tensors (vLLM's KV).

        Each tensor must be a contiguous bf16 view of shape
        [num_entries, kv_cache_stride] so the megakernel's flat
        `entry*kv_cache_stride + kv_head*head_dim + d` indexing lands in place.
        """
        S = self.spec
        assert len(k_tensors) == S.num_layers and len(v_tensors) == S.num_layers
        for t in (*k_tensors, *v_tensors):
            assert t.is_contiguous() and t.dtype == torch.bfloat16
            assert t.shape[-1] == S.kv_cache_stride
        self.buf_k_cache = list(k_tensors)
        self.buf_v_cache = list(v_tensors)


# ── pointer-table builders ───────────────────────────────────────────────────
def build_ptr_table_dense(spec, weight_ptrs_host, buffers):
    """Build the [num_xcds * num_layers * ptrs_per_layer] pointer table (dense).

    weight_ptrs_host is a flat list of num_layers*weights_per_layer int addresses
    (the 12-slot per-layer order from packing.pack_layer_weights). Per-XCD weight
    slices are offset into the MXFP4-packed base pointers exactly as the kernel
    indexes them. Returns an int64 CUDA tensor. Byte-identical to the pre-refactor
    build_ptr_table for Qwen3.
    """
    S = spec
    g = S.gemm
    assert len(weight_ptrs_host) == S.num_layers * S.weights_per_layer
    b = buffers
    ptr_table_host = []
    for xcd in range(S.num_xcds):
        for li in range(S.num_layers):
            base = li * S.weights_per_layer
            qkv_weight_base = weight_ptrs_host[base + 0]
            qkv_bias = weight_ptrs_host[base + 1]
            oproj_weight_base = weight_ptrs_host[base + 2]
            oproj_bias = weight_ptrs_host[base + 3]
            norm_w1 = weight_ptrs_host[base + 4]
            norm_w2 = weight_ptrs_host[base + 5]
            gateup_weight_base = weight_ptrs_host[base + 6]
            gateup_bias = weight_ptrs_host[base + 7]
            down_weight_base = weight_ptrs_host[base + 8]
            down_bias = weight_ptrs_host[base + 9]
            q_norm_w = weight_ptrs_host[base + 10]
            k_norm_w = weight_ptrs_host[base + 11]

            qkv_weight_xcd = qkv_weight_base + xcd * g["qkv_n_wgs_per_xcd"] * g["qkv_wg_bytes"]
            oproj_weight_xcd = oproj_weight_base + xcd * g["oproj_n_wgs_per_xcd"] * g["oproj_wg_bytes"]
            gateup_weight_xcd = gateup_weight_base + xcd * g["gateup_n_wgs_per_xcd"] * g["gateup_wg_bytes"]
            down_weight_xcd = down_weight_base + xcd * g["down_n_wgs_per_xcd"] * g["down_wg_bytes"]

            counter_ptr = b.buf_counter.data_ptr() + li * S.counters_per_layer * 4

            # Ping-pong residual: even layers read A/write B; odd reverse.
            if li % 2 == 0:
                residual_ptr = b.buf_residual_a.data_ptr()
                layer_output_ptr = b.buf_residual_b.data_ptr()
            else:
                residual_ptr = b.buf_residual_b.data_ptr()
                layer_output_ptr = b.buf_residual_a.data_ptr()

            input_ptrs = [
                residual_ptr,                    # [0]  RESIDUAL
                norm_w1,                         # [1]  NORM_W1
                b.buf_norm_scratch1.data_ptr(),  # [2]  NORM_SCRATCH1
                qkv_weight_xcd,                  # [3]  QKV_WEIGHT (per-XCD)
                qkv_bias,                        # [4]  QKV_BIAS
                b.buf_lse_acc.data_ptr(),        # [5]  LSE_ACC
                b.buf_o_acc_f32.data_ptr(),      # [6]  O_ACC_F32
                oproj_weight_xcd,                # [7]  OPROJ_WEIGHT (per-XCD)
                oproj_bias,                      # [8]  OPROJ_BIAS
                norm_w2,                         # [9]  NORM_W2
                b.buf_norm_scratch2.data_ptr(),  # [10] NORM_SCRATCH2
                gateup_weight_xcd,               # [11] GATEUP_WEIGHT (per-XCD)
                gateup_bias,                     # [12] GATEUP_BIAS
                down_weight_xcd,                 # [13] DOWN_WEIGHT (per-XCD)
                down_bias,                       # [14] DOWN_BIAS
                counter_ptr,                     # [15] COUNTER_BUF
                q_norm_w,                        # [16] Q_NORM_WEIGHT
                k_norm_w,                        # [17] K_NORM_WEIGHT
            ]
            output_ptrs = [
                b.buf_qkv_output.data_ptr(),     # [0]  QKV_OUTPUT
                b.buf_k_cache[li].data_ptr(),    # [1]  K_CACHE
                b.buf_v_cache[li].data_ptr(),    # [2]  V_CACHE
                b.buf_q_workspace.data_ptr(),    # [3]  Q_WORKSPACE
                b.buf_attn_out.data_ptr(),       # [4]  ATTN_OUT
                b.buf_oproj_out.data_ptr(),      # [5]  OPROJ_OUT
                b.buf_gateup_scratch.data_ptr(), # [6]  GATEUP_SCRATCH
                b.buf_swiglu_out.data_ptr(),     # [7]  SWIGLU_OUT
                layer_output_ptr,                # [8]  LAYER_OUTPUT (ping-pong)
            ]
            assert len(input_ptrs) == S.ptrs_in
            assert len(output_ptrs) == S.ptrs_out
            ptr_table_host.extend(input_ptrs)
            ptr_table_host.extend(output_ptrs)

    assert len(ptr_table_host) == S.num_xcds * S.num_layers * S.ptrs_per_layer
    return torch.tensor(ptr_table_host, dtype=torch.long, device=b.device)


def build_ptr_table(spec, weight_ptrs_host, buffers, moe_scale_ptrs=None):
    """Dispatch to the family pointer-table builder.

    `moe_scale_ptrs` carries the per-layer expert scale-section addresses when
    those live in their own allocations; the dense family has no such split and
    must not be handed one.
    """
    if spec.is_moe:
        from .packing_moe import build_ptr_table_moe
        return build_ptr_table_moe(spec, weight_ptrs_host, buffers,
                                   moe_scale_ptrs)
    assert moe_scale_ptrs is None, "dense models have no MoE scale sections"
    return build_ptr_table_dense(spec, weight_ptrs_host, buffers)


# ── decode driver ────────────────────────────────────────────────────────────
class FleetMKDecoder:
    """Drives a single decode step through the megakernel.

    Holds the loaded lib, buffer pool, pointer table, RoPE cos/sin tables (indexed
    by absolute position, so they must span max_seq_length), and packed LM-head
    weights. The launch symbol and the trailing launch args both come from `spec`.

    Who does the embedding lookup differs by family, and this is load-bearing:

      * dense -- the caller embeds and passes the [hidden_size] bf16 vector;
        decode_step copies it into residual_a.
      * MoE  -- the *kernel* embeds, from `embed_weight` + a token id, straight
        into layer 0's residual (ptr_table slot 1, which IS residual_a). A
        host-side copy there would just be overwritten. So the MoE path needs
        `embed_weight` at construction and a token id per step, and it cannot
        honour a caller-supplied embedding vector at all.
    """

    def __init__(self, spec, lib, buffers, ptr_table, cos_padded, sin_padded,
                 lm_norm_weight, lm_head_packed, lm_head_bias,
                 embed_weight=None):
        self.spec = spec
        self.lib = lib
        self.launch = getattr(lib, spec.launch_symbol)
        self.b = buffers
        self.ptr_table = ptr_table
        self.cos_padded = cos_padded
        self.sin_padded = sin_padded
        self.lm_norm_weight = lm_norm_weight
        self.lm_head_packed = lm_head_packed
        self.lm_head_bias = lm_head_bias
        self.embed_weight = embed_weight
        assert not (spec.is_moe and embed_weight is None), (
            "the fused MoE kernel does its own embedding lookup and needs the "
            "embedding matrix; pass embed_weight=")
        # DecodeControl, the 4th trailing launch arg. The kernel takes the
        # pointer but never dereferences it on the single-step path (it is for
        # the on-GPU multi-token decode loop). Allocated rather than passed as
        # null so that path stays available without an ABI change.
        if spec.is_moe:
            self.buf_decode_ctrl = torch.zeros(
                48, dtype=torch.uint8, device=buffers.device)
        # Plain greedy requests can consume the token Fleet already selects in
        # its LM-head epilogue.  In that mode there is no reason to materialize
        # 201216 bf16 logits for vLLM to convert and argmax again.
        self.use_kernel_argmax = (
            spec.is_moe and os.environ.get("FLEET_MK_GREEDY_ARGMAX") == "1")
        self.persist_n = (
            max(1, int(os.environ.get("FLEET_MK_PERSIST", "1")))
            if self.use_kernel_argmax else 1)
        self.last_chunk_n = 1
        self.decode_epoch = 0
        if spec.is_moe and self.persist_n > 1:
            self.buf_tokens_out = torch.zeros(
                self.persist_n, dtype=torch.int32, device=buffers.device)

        self.attn_scale = (1.0 / math.sqrt(spec.head_dim)) * 1.44269504088896340736

        dev = buffers.device
        self.qo_indptr = torch.zeros(2, dtype=torch.int32, device=dev)
        self.qo_indptr[1] = 1
        self.kv_indptr = torch.zeros(2, dtype=torch.int32, device=dev)
        self.kv_indices = torch.zeros(buffers.max_num_pages, dtype=torch.int32, device=dev)
        self.kv_last_page_len = torch.zeros(1, dtype=torch.int32, device=dev)
        if spec.is_moe and self.persist_n > 1:
            ctrl_i32 = self.buf_decode_ctrl.view(torch.int32)
            ctrl_i64 = self.buf_decode_ctrl.view(torch.int64)
            ctrl_i32[2] = -1
            ctrl_i64[2] = self.kv_indptr.data_ptr()
            ctrl_i64[3] = self.kv_last_page_len.data_ptr()
            ctrl_i64[4] = self.buf_tokens_out.data_ptr()
            ctrl_i64[5] = 0

        # Optional per-step megakernel timing (FLEET_MK_PROFILE=1).
        self._prof_start = torch.cuda.Event(enable_timing=True)
        self._prof_end = torch.cuda.Event(enable_timing=True)
        self._prof_ms = 0.0
        self._prof_n = 0

    def decode_step(self, embed_vec, cur_pos, block_table=None, stream=None,
                    sync=True, token_id=None):
        """Run one decode step for absolute position cur_pos.

        embed_vec: [padded_hidden_size] or [1, padded_hidden_size] bf16 CUDA tensor
        (embedding of the token being decoded, zero-padded past hidden_size), or
        None on the MoE path, where the kernel embeds itself and `token_id` is
        required instead. cur_pos: cached positions. block_table: optional int
        tensor of physical page ids (vLLM's block_table, 1 block == 1 fleet_mk page
        when block_size == page_size); None -> identity map (standalone/demo path).
        sync=False enqueues without a host sync (vLLM path, which ignores fleet_mk's
        fused argmax and reads buf_lm_norm_scratch).
        """
        b = self.b
        S = self.spec
        page_size = b.page_size
        if stream is None:
            stream = torch.cuda.current_stream()

        if S.is_moe:
            assert token_id is not None, (
                "MoE decode embeds on-device: pass token_id=, not embed_vec")
        else:
            # Layer 0 reads its residual from residual_a; the embedding vector
            # arrives already padded to padded_hidden_size (dense: pw == hidden).
            b.buf_residual_a.copy_(embed_vec.reshape(1, S.padded_hidden_size))
        # Do NOT zero rank_counter, per-layer counters, moe_barrier, or
        # workspace_f32 between steps. Titan's kernel (titan_phases.cuh) reused
        # per-layer slots and needed a reset; fleet's barriers are monotonic
        # (demo.py: "NO need to zero between iterations"). Zeroing rank_counter
        # pins decode_iter at 0 so task_layer_idx restarts and later tokens
        # pre-satisfy barriers. Zeroing moe_barrier hangs W13->W2 at layer 1.

        # Do not cross a paged-KV block inside one launch. The in-kernel
        # metadata advance is safe within a page; changing kv_indptr at a page
        # boundary races readers in other workgroups.
        chunk_n = min(self.persist_n, page_size - (cur_pos % page_size))
        self.last_chunk_n = chunk_n
        last_pos = cur_pos + chunk_n - 1
        num_pages_used = (last_pos + page_size) // page_size
        self.kv_indptr[0] = 0
        self.kv_indptr[1] = num_pages_used
        if block_table is None:
            for p in range(num_pages_used):
                self.kv_indices[p] = p
        else:
            self.kv_indices[:num_pages_used].copy_(
                block_table[:num_pages_used].to(torch.int32))
        self.kv_last_page_len[0] = (cur_pos % page_size) + 1
        if S.is_moe and self.persist_n > 1:
            ctrl_i32 = self.buf_decode_ctrl.view(torch.int32)
            ctrl_i32[0] = chunk_n
            ctrl_i32[1] = cur_pos
            ctrl_i32[3] = self.decode_epoch + 1

        args = [
            1, self.attn_scale,
            self.cos_padded.data_ptr(), self.sin_padded.data_ptr(),
            self.qo_indptr.data_ptr(), self.kv_indptr.data_ptr(),
            self.kv_indices.data_ptr(), self.kv_last_page_len.data_ptr(),
            self.ptr_table.data_ptr(), b.buf_counter.data_ptr(),
            self.lm_norm_weight.data_ptr(), b.buf_lm_norm_scratch.data_ptr(),
            self.lm_head_packed.data_ptr(), self.lm_head_bias.data_ptr(),
            b.buf_argmax_out.data_ptr(),
        ]
        # Trailing args, in the order spec.extra_launch_args declares to ctypes.
        # These two lists are the same fact twice; keep them together so a change
        # to one is visibly a change to the other. A short or transposed list here
        # is stack corruption, not a clean error -- hence the length assert.
        if S.is_moe:
            # The kernel always computes argmax.  Materialize the full logit row
            # only when vLLM needs its general sampler; plain greedy consumes
            # buf_argmax_out directly through greedy.py.
            args.append(0 if self.use_kernel_argmax
                        else b.buf_logits.data_ptr())   # logits_output
        args.append(b.buf_timing.data_ptr())          # timing_buf
        if S.is_moe:
            args.append(self.embed_weight.data_ptr())  # embed_weight
            args.append(int(token_id))                 # cur_token_id
            args.append(self.buf_decode_ctrl.data_ptr())  # decode_ctrl
        assert len(args) == 15 + len(S.extra_launch_args), (
            len(args), len(S.extra_launch_args))
        args.append(stream.cuda_stream)

        profile = os.environ.get("FLEET_MK_PROFILE")
        if profile:
            self._prof_start.record(stream)
        self.launch(*args)
        self.decode_epoch += chunk_n
        if profile:
            # Pure GPU time of the megakernel launch, isolated from vLLM per-step
            # overhead (lm_head + sampler + scheduler). Forces a per-step sync.
            self._prof_end.record(stream)
            self._prof_end.synchronize()
            self._prof_ms += self._prof_start.elapsed_time(self._prof_end)
            self._prof_n += 1
            if self._prof_n % 32 == 0:
                print(f"[fleet_mk] megakernel GPU avg: "
                      f"{self._prof_ms / self._prof_n:.3f} ms/token "
                      f"over {self._prof_n} steps", flush=True)
            return None
        if not sync:
            return None
        torch.cuda.synchronize()
        return b.buf_argmax_out[0].item()
