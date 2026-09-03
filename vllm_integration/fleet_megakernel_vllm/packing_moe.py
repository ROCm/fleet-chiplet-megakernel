"""MXFP4 weight packing + pointer-table build for the fleet_mk GPT-OSS (MoE) kernel.

Ported verbatim (parameterized by a ModelSpec) from the known-good standalone
demo.py so the vLLM plugin packs MoE weights byte-identically to the 2.13 ms/token
reference. Two differences from demo.py, both because vLLM owns the paged KV cache:

  * The per-layer packed weight list has 13 slots, not demo's 15 -- the k_cache /
    v_cache slots are dropped here and aliased zero-copy from vLLM's kv_cache in
    build_ptr_table_moe (mirroring how the dense path drops KV from its 12).
  * Expert weights are sourced as raw MXFP4 blocks/scales (the pre-transform HF
    safetensors layout mirage packs from); vLLM swizzles+deletes its own copy, so
    the caller passes the raw tensors in.

Slot order (matches build_ptr_table_moe indexing):
  [0] qkv_weight(MXFP4)  [1] qkv_bias(wg)   [2] oproj_weight(MXFP4) [3] oproj_bias(wg)
  [4] norm_pre           [5] norm_post      [6] router_weight(bf16) [7] router_bias
  [8] gate_up_weight(W13,MXFP4) [9] down_weight(W2,MXFP4) [10] w13_bias(bf16)
  [11] w2_bias(bf16)     [12] attn_sinks
"""

import torch

# NOTE: an expert-slab "guard pad" diagnostic used to live here -- four env
# knobs that grew each MoE slab by N dead bytes to test whether the MoE fault
# under SPLIT_SCALES + K_STRIDE=3072 was a read walking off the end of a
# section. It was removed on 2026-08-27 after its own control disproved the
# hypothesis: allocating the same bytes as a SEPARATE tensor, leaving the slab
# byte-identical at the same address, suppressed the crash just as well. Every
# "pad fixes it" result was the caching allocator being reshuffled.
#
# What the fault actually is, measured over repeated runs rather than single
# ones: it reproduces about 2 runs in 6 with the W13 direct-to-LDS prefetch on
# and 0 in 6 with it off at identical knobs and extents. It is a race, not an
# over-read -- see the wiki article on the prefetch fault. Do not reintroduce a
# pad here; a single passing run means nothing at this crash rate.


def _moe_dims(spec):
    """Padded dims the MoE kernel runs on. hidden and intermediate both pad to
    padded_hidden_size (2944) for GPT-OSS -- demo.py's PADDED_HIDDEN_SIZE ==
    PADDED_INTERMEDIATE_SIZE."""
    ph = spec.padded_hidden_size          # 2944 (both hidden and intermediate pad here)
    return ph, ph


def _moe_data(pack, layout, which, blocks, scales, foreign=None):
    """The expert DATA slot for "w13" or "w2".

    Returns `foreign` unchanged when aliasing: vLLM's w13_weight / w2_weight
    already hold exactly these bytes at exactly these offsets, so packing them
    would materialize a byte-for-byte 60 GiB duplicate of the thing being
    aliased. The tensor itself is returned rather than its address so the caller
    keeps a reference in the same list it keeps fleet_mk's own tensors in -- vLLM
    will not free it, but a slot whose lifetime rule differs from its
    neighbours' is exactly what breaks when someone later reorders the list.

    Without a split there are no separable sections, so the whole interleaved
    slab is emitted and `section` is not passed at all.
    """
    if foreign is not None:
        return foreign
    kw = layout.pack_kwargs(which)
    return pack(blocks, scales,
                **(dict(kw, section="data") if layout.split_scales else kw))


def _moe_scales(pack, layout, which, blocks, scales):
    """The expert SCALE slot for "w13" or "w2", or None when interleaved.

    Always fleet_mk's own bytes, aliased or not: vLLM's TRITON backend deletes
    w13_weight_scale / w2_weight_scale in process_weights_after_loading (the
    swizzled scales move inside the precision configs), so there is nothing on
    that side to point at.

    `blocks` is read for its shape only -- `section="scales"` skips the whole
    data path, which is what keeps this cheap enough to do while aliasing.
    """
    if not layout.split_scales:
        return None
    return pack(blocks, scales, section="scales", **layout.pack_kwargs(which))


def pack_moe_layer(*, w_q, w_k, w_v, q_bias, k_bias, v_bias, w_o, o_bias,
                   norm_pre, norm_post, router_w, router_b,
                   gate_up_blocks, gate_up_scales, gate_up_bias,
                   down_blocks, down_scales, down_bias, sinks,
                   spec, packers, layout,
                   foreign_w13=None, foreign_w2=None):
    """Pack one GPT-OSS decoder layer into the 13-slot fleet_mk weight list.

    Attention weights (w_q/w_k/w_v/w_o + biases), norms, router, sinks are raw HF
    bf16 [out,in] CUDA tensors. Expert weights come as raw MXFP4 blocks/scales
    (experts.gate_up_proj_blocks/scales, experts.down_proj_blocks/scales) plus bf16
    biases.

    `layout` is the MoeLayout describing the expert buffer's pitches and section
    split (fleet_megakernel_vllm/moe_layout.py). `foreign_w13` / `foreign_w2` are vLLM's own
    expert data tensors, passed only when aliasing them instead of packing a
    second copy.

    Returns `(weights, moe_scales)`:
      * `weights` -- 13 CUDA tensors in slot order (caller keeps refs alive).
      * `moe_scales` -- `(w13_scale, w2_scale)` when the layout splits the
        sections into their own allocations, else `(None, None)`. These are NOT
        appended to `weights`: that list is indexed by a fixed
        `spec.weights_per_layer` stride the pointer table and the kernel both
        depend on, so growing it would shift every slot after position 9.
    """
    pad_1d = packers["pad_weight_1d"]
    pad_2d = packers["pad_weight_2d"]
    quantize = packers["quantize_bf16_to_mxfp4"]
    pack = packers["pack_mxfp4_workgroup"]

    S = spec
    PH, PI = _moe_dims(S)
    num_kv = S.num_kv_heads
    q_per_kv = S.q_per_kv
    hd = S.head_dim
    num_q = S.num_q_heads
    n_exp = S.num_experts
    inter = S.intermediate_size
    hidden = S.hidden_size
    fused_qkv = S.qkv_output_size
    attn_reduction = num_q * hd
    qkv_opw = S.gemm["qkv_output_per_wg"]
    oproj_opw = S.gemm["oproj_output_per_wg"]
    w13_opw = S.gemm["w13_output_per_wg"]
    w2_opw = S.gemm["w2_output_per_wg"]
    dev = w_q.device
    z1 = lambda n: torch.zeros(n, dtype=torch.bfloat16, device=dev)
    out = []

    # ── [0] QKV weight: interleave Q/K/V by KV groups, pad cols to PH, MXFP4 ──
    wq = pad_2d(w_q, target_cols=PH)
    wk = pad_2d(w_k, target_cols=PH)
    wv = pad_2d(w_v, target_cols=PH)
    chunks = []
    for g in range(num_kv):
        chunks.append(wq[g * q_per_kv * hd:(g + 1) * q_per_kv * hd])
        chunks.append(wk[g * hd:(g + 1) * hd])
        chunks.append(wv[g * hd:(g + 1) * hd])
    w_qkv = torch.cat(chunks, dim=0).contiguous()
    qb, qs = quantize(w_qkv)
    out.append(pack(qb, qs, output_per_wg=qkv_opw, target_out_dim=fused_qkv,
                    target_num_blocks=PH // 32))  # [0]

    # ── [1] QKV bias: same KV-group interleave, packed to [n_wgs, qkv_opw] ──
    qb_ = q_bias if q_bias is not None else z1(num_q * hd)
    kb_ = k_bias if k_bias is not None else z1(num_kv * hd)
    vb_ = v_bias if v_bias is not None else z1(num_kv * hd)
    bchunks = []
    for g in range(num_kv):
        bchunks.append(qb_[g * q_per_kv * hd:(g + 1) * q_per_kv * hd])
        bchunks.append(kb_[g * hd:(g + 1) * hd])
        bchunks.append(vb_[g * hd:(g + 1) * hd])
    qkv_bias = torch.cat(bchunks, dim=0).contiguous()
    n_wgs = fused_qkv // qkv_opw
    out.append(qkv_bias.reshape(n_wgs, qkv_opw).contiguous())  # [1]

    # ── [2] O-proj weight: pad rows->PH, cols->num_q*hd, MXFP4 ──
    wo = pad_2d(w_o, target_rows=PH, target_cols=attn_reduction)
    ob_blk, ob_sc = quantize(wo)
    o_packed = pack(ob_blk, ob_sc, output_per_wg=oproj_opw, target_out_dim=PH,
                    target_num_blocks=attn_reduction // 32)
    if S.gemm.get("oproj_kmajor"):
        # The .so was compiled with -DMPK_OPROJ_KMAJOR, so the kernel reads this
        # tile as [k128, quarter, row, 16B] rather than [row, k128, quarter,
        # 16B]. Repack to match. Skipping this against a K-major .so is not a
        # crash -- every lane still reads a valid byte of the tile, just the
        # wrong one -- so it would surface as plausible wrong logits. Both the
        # flag and this branch come off build.extra_defines (spec.py), which is
        # what keeps them from drifting apart. The driver does the same thing at
        # its own O-proj pack site; the two entry points must agree byte for
        # byte or they build different slabs for the same kernel.
        from .mxfp4_pack import shuffle_oproj_workgroups_kmajor
        o_packed = shuffle_oproj_workgroups_kmajor(o_packed, oproj_opw)
    out.append(o_packed)  # [2]

    # ── [3] O-proj bias: pad to PH, packed to [n_wgs, oproj_opw] ──
    ob = o_bias if o_bias is not None else z1(PH)
    ob = pad_1d(ob, PH)
    out.append(ob.reshape(PH // oproj_opw, oproj_opw).contiguous())  # [3]

    # ── [4][5] RMSNorm weights (pad to PH, zero-pad) ──
    out.append(pad_1d(norm_pre, PH, pad_value=0.0))   # [4]
    out.append(pad_1d(norm_post, PH, pad_value=0.0))  # [5]

    # ── [6] Router weight (bf16, pad cols to PH) + [7] Router bias ──
    out.append(pad_2d(router_w, target_cols=PH).contiguous())  # [6]
    out.append(router_b.unsqueeze(0).contiguous())             # [7]

    # ── [8] W13 (gate_up) MXFP4 + [9] W2 (down) MXFP4 ────────────────────────
    # Data sections; pitches and section split come from `layout`, and under
    # aliasing the data is vLLM's tensor rather than a packed copy. See
    # fleet_megakernel_vllm/moe_layout.py for why each knob exists, and
    # tools/check_alias_equivalence.py for the proof that the aliased bytes are
    # the bytes fleet_mk would have packed.
    w13_data = _moe_data(pack, layout, "w13", gate_up_blocks,
                          gate_up_scales, foreign=foreign_w13)
    if S.gemm.get("w13_kmajor"):
        # MPK_W13_KMAJOR_RECYCLE changes the lane order inside each W13 tile.
        # A foreign vLLM slab is row-major and cannot be transformed in place;
        # reject aliasing rather than materializing a hidden ~60 GiB copy.
        if foreign_w13 is not None:
            raise ValueError(
                "MPK_W13_KMAJOR_RECYCLE is incompatible with "
                "FLEET_MK_MOE_ALIAS_VLLM=1; disable aliasing or the flag")
        from .mxfp4_pack import shuffle_w13_workgroups_kmajor
        w13_data = shuffle_w13_workgroups_kmajor(
            w13_data, output_per_wg=w13_opw, reduction=PH)
    out.append(w13_data)                                       # [8]
    out.append(_moe_data(pack, layout, "w2", down_blocks, down_scales,
                         foreign=foreign_w2))                   # [9]
    # Scale sections, kept OUT of `out` -- see the docstring: `out` is indexed by
    # a fixed weights_per_layer stride, so appending here would shift slot 10 on.
    moe_scales = (_moe_scales(pack, layout, "w13", gate_up_blocks, gate_up_scales),
                  _moe_scales(pack, layout, "w2", down_blocks, down_scales))

    # ── [10] W13 bias (bf16, per expert, padded to 2*PI) ──
    gu_bias = torch.zeros(n_exp, 2 * PI, dtype=torch.bfloat16, device=dev)
    gu_bias[:, :2 * inter] = gate_up_bias.to(device=dev, dtype=torch.bfloat16)
    out.append(gu_bias)  # [10]

    # ── [11] W2 bias (bf16, per expert, padded to PH) ──
    dp_bias = torch.zeros(n_exp, PH, dtype=torch.bfloat16, device=dev)
    dp_bias[:, :hidden] = down_bias.to(device=dev, dtype=torch.bfloat16)
    out.append(dp_bias)  # [11]

    # ── [12] Attention sinks ──
    out.append(sinks.to(device=dev).contiguous())  # [12]

    assert len(out) == S.weights_per_layer, (len(out), S.weights_per_layer)
    return out, moe_scales


def pack_lm_head_moe(lm_head_weight, spec, packers):
    """Pad + MXFP4-pack the LM head. Returns ([n_wgs, wg_bytes] packed, zero_bias).

    Padded to spec.padded_vocab_size rows (== the kernel's compiled LM_VOCAB_SIZE)
    and padded_hidden_size cols; output_per_wg=64 (LM_OUTPUT_PER_WG).
    """
    pad_2d = packers["pad_weight_2d"]
    quantize = packers["quantize_bf16_to_mxfp4"]
    pack = packers["pack_mxfp4_workgroup"]
    PH = spec.padded_hidden_size
    padded = pad_2d(lm_head_weight, target_rows=spec.padded_vocab_size,
                    target_cols=PH)
    blocks, scales = quantize(padded)
    packed = pack(blocks, scales, output_per_wg=64).squeeze(0)
    if spec.gemm.get("lm_head_kmajor"):
        from .mxfp4_pack import shuffle_lm_head_record_kmajor
        packed = shuffle_lm_head_record_kmajor(packed, output_per_wg=64)
    zero_bias = torch.zeros(1, spec.padded_vocab_size, dtype=torch.bfloat16,
                            device=lm_head_weight.device)
    return packed, zero_bias


def build_ptr_table_moe(spec, weight_ptrs_host, buffers, moe_scale_ptrs=None):
    """Build the [num_xcds * num_layers * 38] MoE pointer table.

    weight_ptrs_host: flat list of num_layers*13 int addresses (pack_moe_layer
    order). Ported from demo_gpt_oss_120b.py's ptr-table build: 26 mirage_in +
    11 mirage_out + 1 trailing layer_output per layer per XCD, with per-XCD MXFP4
    weight-slice offsets baked in. K/V cache pointers come from the zero-copy
    aliases in `buffers`, not the weight list.

    moe_scale_ptrs: per-layer `(w13_scale_addr, w2_scale_addr)` when the expert
    scales live in their own allocations (split mode), else None. Kept as a
    separate argument rather than two more entries in weight_ptrs_host because
    that list is indexed by a fixed `weights_per_layer` stride the kernel also
    depends on.

    Returns an int64 CUDA tensor.
    """
    S = spec
    b = buffers
    assert len(weight_ptrs_host) == S.num_layers * S.weights_per_layer, (
        len(weight_ptrs_host), S.num_layers, S.weights_per_layer)
    assert moe_scale_ptrs is None or len(moe_scale_ptrs) == S.num_layers, (
        len(moe_scale_ptrs), S.num_layers)

    PH = S.padded_hidden_size
    num_q = S.num_q_heads
    hd = S.head_dim
    attn_reduction = num_q * hd
    qkv_opw = S.gemm["qkv_output_per_wg"]
    oproj_opw = S.gemm["oproj_output_per_wg"]

    # Per-XCD weight-slice strides (bytes) + wg counts, matching demo.py.
    qkv_wg_bytes = qkv_opw * (PH // 2 + PH // 32)
    qkv_n_wgs_per_xcd = (S.qkv_output_size // qkv_opw) // S.num_xcds
    oproj_wg_bytes = oproj_opw * (attn_reduction // 2 + attn_reduction // 32)
    oproj_n_wgs_per_xcd = (PH // oproj_opw) // S.num_xcds
    router_experts_per_xcd = S.num_experts // S.num_xcds

    ptr_table_host = []
    for xcd in range(S.num_xcds):
        for li in range(S.num_layers):
            base = li * S.weights_per_layer
            qkv_w = weight_ptrs_host[base + 0]
            qkv_b = weight_ptrs_host[base + 1]
            oproj_w = weight_ptrs_host[base + 2]
            oproj_b = weight_ptrs_host[base + 3]
            norm_pre = weight_ptrs_host[base + 4]
            norm_post = weight_ptrs_host[base + 5]
            router_w = weight_ptrs_host[base + 6]
            router_b = weight_ptrs_host[base + 7]
            gate_up_w = weight_ptrs_host[base + 8]
            down_w = weight_ptrs_host[base + 9]
            w13_bias = weight_ptrs_host[base + 10]
            w2_bias = weight_ptrs_host[base + 11]
            attn_sinks = weight_ptrs_host[base + 12]
            # Own allocation in split mode; the data base otherwise -- see the
            # slot [24]/[25] comment below for why the fallback is the data
            # base rather than 0.
            w13_scale, w2_scale = (moe_scale_ptrs[li] if moe_scale_ptrs
                                   else (gate_up_w, down_w))

            # Per-XCD offsets into the MXFP4-packed / bf16 weight bases.
            qkv_w_xcd = qkv_w + xcd * qkv_n_wgs_per_xcd * qkv_wg_bytes
            qkv_b_xcd = qkv_b + xcd * qkv_n_wgs_per_xcd * qkv_opw * 2
            oproj_w_xcd = oproj_w + xcd * oproj_n_wgs_per_xcd * oproj_wg_bytes
            oproj_b_xcd = oproj_b + xcd * oproj_n_wgs_per_xcd * oproj_opw * 2
            router_w_xcd = router_w + xcd * router_experts_per_xcd * PH * 2
            router_b_xcd = router_b + xcd * router_experts_per_xcd * 2
            logits_xcd = b.buf_logits_scratch.data_ptr() + xcd * router_experts_per_xcd * 2

            # ONE block shared by every layer -- NOT `+ li * counters_per_layer`.
            # That per-layer stride was correct for titan's own barriers
            # (titan_phases.cuh), which zeroed and reused slots. Fleet's layer
            # body derives expected = task_layer_idx + 1 against counters that
            # are never reset, and the standalone demo already passes the same
            # pointer at every layer. Per-layer zeroed blocks hang on the
            # FIRST decode: layer 2 waits for expected=2 with observed=1.
            counter_ptr = b.buf_counter.data_ptr()
            qkv_barrier_ptr = counter_ptr + S.slot_qkv_barrier * 4

            # MoE residual scheme: layer 0 reads residual_a; layers 1+ read
            # attn_proj_out (the prior layer's MoE-block residual).
            residual = (b.buf_residual_a.data_ptr() if li == 0
                        else b.buf_attn_proj_out.data_ptr())

            input_ptrs = [
                b.buf_workspace_f32.data_ptr(),   # [0]  workspace_f32
                residual,                         # [1]  residual
                norm_pre,                         # [2]  norm_weight_pre
                b.buf_norm_scratch1.data_ptr(),   # [3]  norm_scratch_pre
                qkv_w_xcd,                        # [4]  qkv_weight (XCD)
                qkv_b_xcd,                        # [5]  qkv_bias (XCD)
                attn_sinks,                       # [6]  attn_sinks
                qkv_barrier_ptr,                  # [7]  qkv_barrier (in counters)
                b.buf_lse_acc.data_ptr(),         # [8]  lse_acc
                oproj_w_xcd,                      # [9]  oproj_weight (XCD)
                oproj_b_xcd,                      # [10] oproj_bias (XCD)
                norm_post,                        # [11] norm_weight_post
                b.buf_norm_scratch2.data_ptr(),   # [12] norm_scratch_post
                router_w_xcd,                     # [13] router_weight (XCD)
                router_b_xcd,                     # [14] router_bias (XCD)
                logits_xcd,                       # [15] logits_scratch (XCD)
                counter_ptr,                      # [16] oproj_counters (per-layer)
                gate_up_w,                        # [17] moe_gate_up_weight
                down_w,                           # [18] moe_down_weight
                w13_bias,                         # [19] moe_w13_bias
                w2_bias,                          # [20] moe_w2_bias
                b.buf_moe_barrier.data_ptr(),     # [21] moe_barrier
                b.buf_swiglu_out.data_ptr(),      # [22] moe_swiglu_out
                b.buf_o_acc_f32.data_ptr(),       # [23] o_acc_f32
                # MoE scale bases, dereferenced only by a .so built
                # -DMPK_MOE_SPLIT_SCALES=1.
                #
                # Interleaved (no split): there IS no scale section. A
                # workgroup's scales sit inside its own data block and the
                # kernel reaches them from the data base alone, so these fall
                # back to the data base -- a mapped address the kernel never
                # reads. Deliberately not 0 and not omitted: the table is sized
                # from len(MIRAGE_IN), so leaving them out would not shorten it,
                # it would leave two slots holding whatever the previous layer
                # wrote, which faults confusingly the day someone flips the knob.
                #
                # Split: fleet_mk's own scale-only slab, in its own allocation.
                # These are NOT vLLM's w13_weight_scale / w2_weight_scale -- the
                # TRITON backend deletes those in process_weights_after_loading
                # (the swizzled scales move into the precision configs), so
                # there is nothing there to point at. Only the DATA side is
                # aliasable, and it is aliased through slots [17]/[18] above,
                # which read the same weight_ptrs_host entries either way.
                w13_scale,                        # [24] moe_gate_up_scale
                w2_scale,                         # [25] moe_down_scale
            ]
            output_ptrs = [
                # [0] is a DEDICATED intermediate (mirage's mlp_weighted_sum_out),
                # not the residual: QKV writes its ResAdd+RMSNorm result here while
                # [5]/input[1] still hold the residual the MoE block needs.
                b.buf_x_output.data_ptr(),        # [0]  x_output
                b.buf_k_cache[li].data_ptr(),     # [1]  k_cache (aliased vLLM KV)
                b.buf_v_cache[li].data_ptr(),     # [2]  v_cache (aliased vLLM KV)
                b.buf_q_workspace.data_ptr(),     # [3]  q_workspace
                b.buf_attn_out.data_ptr(),        # [4]  o_acc
                b.buf_attn_proj_out.data_ptr(),   # [5]  attn_proj_out (next residual)
                b.buf_topk_weight.data_ptr(),     # [6]  topk_weight
                b.buf_routing_indices.data_ptr(), # [7]  routing_indices
                b.buf_active_expert_ids.data_ptr(),# [8] active_expert_ids
                b.buf_moe_routing_wt.data_ptr(),  # [9]  moe_routing_weight (== [6])
                b.buf_workspace_f32.data_ptr(),   # [10] moe_workspace_f32 (== input[0])
            ]
            # Trailing slot: destination for the last layer's fused ResAdd. Same
            # buffer as output[5]; the kernel reads it from a fixed slot rather
            # than special-casing the last layer.
            layer_output = [b.buf_attn_proj_out.data_ptr()]

            assert len(input_ptrs) == S.ptrs_in
            assert len(output_ptrs) == S.ptrs_out
            assert len(layer_output) == S.ptrs_extra
            ptr_table_host.extend(input_ptrs)
            ptr_table_host.extend(output_ptrs)
            ptr_table_host.extend(layer_output)

    assert len(ptr_table_host) == S.num_xcds * S.num_layers * S.ptrs_per_layer
    return torch.tensor(ptr_table_host, dtype=torch.long, device=b.device)
