"""MXFP4 weight packing for the titan dense (Qwen3-family) megakernel.

Extracted verbatim (parameterized into functions) from demo_qwen3.py so the demo
and the vLLM plugin pack weights identically. The low-level MXFP4 quantize/pack
primitives live in mxfp4_pack.py, vendored into titan; nothing here reaches into
a mirage checkout any more.

Every dimension comes from the ModelSpec argument rather than module constants.
These same numbers drive the GEMM tiling inside the kernel, so packing to one
value and tiling to another produces garbage output with no crash -- and a
second copy of them in this file is exactly how that drift happens. packing_moe
was already written this way; this module now matches it.
"""

import torch


def import_mirage_packers(mirage_dir=None):
    """The pad/quantize/pack helpers, now titan's own (see mxfp4_pack.py).

    Name and signature kept so callers (mixin._setup_titan, the demos) do not
    change; `mirage_dir` is accepted and ignored. These four functions used to be
    executed out of mirage's demo module through a `_load_gptoss_demo_module`
    handle, which also served the reference-model load; both are gone. Vendoring
    them is what lets titan_vllm run with no mirage checkout on sys.path.
    """
    from . import mxfp4_pack
    return {
        "pad_weight_1d": mxfp4_pack.pad_weight_1d,
        "pad_weight_2d": mxfp4_pack.pad_weight_2d,
        "quantize_bf16_to_mxfp4": mxfp4_pack.quantize_bf16_to_mxfp4,
        "pack_mxfp4_workgroup": mxfp4_pack.pack_mxfp4_workgroup,
    }


def pack_layer_weights(w_q, w_k, w_v, w_o, w_gate, w_up, w_down,
                       norm_w1, norm_w2, q_norm_w, k_norm_w, packers, spec,
                       q_bias=None, k_bias=None, v_bias=None, o_bias=None):
    """Pack one dense decoder layer into the 12-slot titan weight list.

    All weight tensors are the raw HF [out, in] bf16 matrices on CUDA. Returns a
    list of 12 CUDA tensors in the exact per-layer order the ptr table expects:
      [0] qkv_weight (MXFP4)   [1] qkv_bias        [2] oproj_weight (MXFP4)
      [3] oproj_bias           [4] norm_w1         [5] norm_w2
      [6] gateup_weight (MXFP4)[7] gateup_bias     [8] down_weight (MXFP4)
      [9] down_bias            [10] q_norm_weight  [11] k_norm_weight
    """
    quantize = packers["quantize_bf16_to_mxfp4"]
    pack = packers["pack_mxfp4_workgroup"]
    out = []

    S = spec
    HIDDEN_SIZE = S.hidden_size
    INTERMEDIATE_SIZE = S.intermediate_size
    HEAD_DIM = S.head_dim
    NUM_KV_HEADS = S.num_kv_heads
    Q_PER_KV = S.q_per_kv
    QKV_OUTPUT_SIZE = S.qkv_output_size
    GATEUP_OUTPUT_SIZE = S.gateup_output_size
    OUTPUT_PER_WG = S.output_per_wg
    GATEUP_OPW = S.gateup_opw
    OPROJ_REDUCTION = S.num_q_heads * S.head_dim

    # [0] QKV weight: interleave Q/K/V by KV groups
    qkv_chunks = []
    for g in range(NUM_KV_HEADS):
        qkv_chunks.append(w_q[g * Q_PER_KV * HEAD_DIM:(g + 1) * Q_PER_KV * HEAD_DIM])
        qkv_chunks.append(w_k[g * HEAD_DIM:(g + 1) * HEAD_DIM])
        qkv_chunks.append(w_v[g * HEAD_DIM:(g + 1) * HEAD_DIM])
    w_qkv = torch.cat(qkv_chunks, dim=0).contiguous()
    assert w_qkv.shape == (QKV_OUTPUT_SIZE, HIDDEN_SIZE), w_qkv.shape
    qkv_blocks, qkv_scales = quantize(w_qkv)
    out.append(pack(qkv_blocks, qkv_scales, output_per_wg=OUTPUT_PER_WG,
                    target_out_dim=QKV_OUTPUT_SIZE,
                    target_num_blocks=HIDDEN_SIZE // 32))  # [0]

    # [1] QKV bias (Qwen3 has no attention bias -> zeros unless provided)
    qkv_bias = torch.zeros(QKV_OUTPUT_SIZE, dtype=torch.bfloat16, device="cuda")
    if q_bias is not None:
        bias_chunks = []
        for g in range(NUM_KV_HEADS):
            bias_chunks.append(q_bias[g * Q_PER_KV * HEAD_DIM:(g + 1) * Q_PER_KV * HEAD_DIM])
            bias_chunks.append(k_bias[g * HEAD_DIM:(g + 1) * HEAD_DIM])
            bias_chunks.append(v_bias[g * HEAD_DIM:(g + 1) * HEAD_DIM])
        qkv_bias = torch.cat(bias_chunks, dim=0).contiguous()
    out.append(qkv_bias)  # [1]

    # [2] O-proj weight
    o_blocks, o_scales = quantize(w_o)
    out.append(pack(o_blocks, o_scales, output_per_wg=OUTPUT_PER_WG,
                    target_out_dim=HIDDEN_SIZE,
                    target_num_blocks=OPROJ_REDUCTION // 32))  # [2]

    # [3] O-proj bias
    ob = o_bias if o_bias is not None else torch.zeros(
        HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    out.append(ob)  # [3]

    # [4][5] RMSNorm weights
    out.append(norm_w1.contiguous())  # [4]
    out.append(norm_w2.contiguous())  # [5]

    # [6] GateUp weight: interleave gate/up in 128-row blocks (64 gate + 64 up)
    n_wgs = INTERMEDIATE_SIZE // (GATEUP_OPW // 2)  # 192
    w_gateup = torch.empty(GATEUP_OUTPUT_SIZE, HIDDEN_SIZE,
                           dtype=torch.bfloat16, device="cuda")
    for wg in range(n_wgs):
        start = wg * (GATEUP_OPW // 2)
        end = start + (GATEUP_OPW // 2)
        dst = wg * GATEUP_OPW
        w_gateup[dst:dst + GATEUP_OPW // 2] = w_gate[start:end]
        w_gateup[dst + GATEUP_OPW // 2:dst + GATEUP_OPW] = w_up[start:end]
    w_gateup = w_gateup.contiguous()
    gu_blocks, gu_scales = quantize(w_gateup)
    out.append(pack(gu_blocks, gu_scales, output_per_wg=GATEUP_OPW,
                    target_out_dim=GATEUP_OUTPUT_SIZE,
                    target_num_blocks=HIDDEN_SIZE // 32))  # [6]

    # [7] GateUp bias (zeros)
    out.append(torch.zeros(GATEUP_OUTPUT_SIZE, dtype=torch.bfloat16, device="cuda"))  # [7]

    # [8] Down weight
    d_blocks, d_scales = quantize(w_down)
    out.append(pack(d_blocks, d_scales, output_per_wg=OUTPUT_PER_WG,
                    target_out_dim=HIDDEN_SIZE,
                    target_num_blocks=INTERMEDIATE_SIZE // 32))  # [8]

    # [9] Down bias (zeros)
    out.append(torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"))  # [9]

    # [10][11] QK-norm weights
    out.append(q_norm_w.contiguous())  # [10]
    out.append(k_norm_w.contiguous())  # [11]
    return out


def pack_lm_head(lm_head_weight, packers, spec):
    """Pad + MXFP4-pack the LM head; returns [n_wgs, LM_WG_BYTES] packed tensor."""
    pad_weight_2d = packers["pad_weight_2d"]
    quantize = packers["quantize_bf16_to_mxfp4"]
    pack = packers["pack_mxfp4_workgroup"]
    padded = pad_weight_2d(lm_head_weight, target_rows=spec.padded_vocab_size,
                           target_cols=spec.hidden_size)
    blocks, scales = quantize(padded)
    return pack(blocks, scales, output_per_wg=spec.output_per_wg).squeeze(0)
