#!/usr/bin/env python3
"""GPT-OSS 120B MoE inference demo on MI300X using Mirage persistent kernel.

Usage:
    # PyTorch reference (no Mirage):
    HIP_VISIBLE_DEVICES=7 python3 demo/gpt_oss/demo.py --model-path <path>

    # Mirage persistent kernel:
    HIP_VISIBLE_DEVICES=7 python3 demo/gpt_oss/demo.py --model-path <path> --use-mirage
"""

from models.modeling_gpt_oss import GptOssForCausalLM, _dequantize_mxfp4, swigluoai
from transformers import AutoTokenizer, AutoConfig
import torch
import torch.distributed as dist
import argparse
import os
import math
import json

# Local model directory or HF repo id. Override with --model-path or the
# GPT_OSS_MODEL_PATH env var.
DEFAULT_MODEL_PATH = os.environ.get("GPT_OSS_MODEL_PATH", "openai/gpt-oss-120b")

# CI correctness-dump defaults. Torch vs Mirage token dumps land here for
# tests/ci-tests/test_gpt_oss_inference_output.py.
DEFAULT_SAVE_DIR = os.path.join("outputs", "gpt_oss")
MAX_SAVE_TOKENS = 100

# GPT-OSS 120B dimensions
# hidden_size=2880, intermediate_size=2880
# Pad hidden_size=2880 to next multiple of 128 for MFMA alignment
PADDED_HIDDEN_SIZE = 2944
PADDED_INTERMEDIATE_SIZE = 2944


def grid_for_rmsnorm_linear_layer(size: int):
    if size % 64 == 0:
        return size // 64
    raise ValueError(f"Size {size} not supported for rmsnorm_linear")


def compute_dynamic_splitk(output_size, n_per_block, reduction_size, num_workers, k_per_block=256):
    """Compute optimal split-K to maximize CU utilization."""
    tile_num = output_size // n_per_block
    if tile_num >= num_workers:
        return 1
    ideal = max(1, math.ceil(num_workers / tile_num))
    k_tiles = reduction_size // k_per_block
    best = 1
    best_diff = abs(1 - ideal)
    for s in range(2, k_tiles + 1):
        if k_tiles % s == 0:
            diff = abs(s - ideal)
            if diff < best_diff or (diff == best_diff and s > best):
                best = s
                best_diff = diff
    return best


def max_factor_leq_n(m: int, n: int) -> int:
    max_factor = 1
    i = 1
    while i * i <= m:
        if m % i == 0:
            if i <= n:
                max_factor = max(max_factor, i)
            if m // i <= n:
                max_factor = max(max_factor, m // i)
        i += 1
    return max_factor


def pad_weight_1d(w: torch.Tensor, target_size: int, pad_value: float = 0.0):
    """Pad a 1D weight vector to target size."""
    if w.shape[0] == target_size:
        return w
    padded = torch.full((target_size,), pad_value, dtype=w.dtype, device=w.device)
    padded[:w.shape[0]] = w
    return padded


def pad_weight_2d(w: torch.Tensor, target_rows: int = None, target_cols: int = None):
    """Pad a 2D weight matrix to target dimensions with zeros."""
    rows, cols = w.shape
    if target_rows is None:
        target_rows = rows
    if target_cols is None:
        target_cols = cols
    if rows == target_rows and cols == target_cols:
        return w
    padded = torch.zeros(target_rows, target_cols, dtype=w.dtype, device=w.device)
    padded[:rows, :cols] = w
    return padded


def pad_weight_3d(w: torch.Tensor, target_dim1: int = None, target_dim2: int = None):
    """Pad a 3D weight matrix [E, D1, D2] to target dimensions with zeros."""
    E, D1, D2 = w.shape
    if target_dim1 is None:
        target_dim1 = D1
    if target_dim2 is None:
        target_dim2 = D2
    if D1 == target_dim1 and D2 == target_dim2:
        return w
    padded = torch.zeros(E, target_dim1, target_dim2, dtype=w.dtype, device=w.device)
    padded[:, :D1, :D2] = w
    return padded


# FP4 E2M1 magnitude lookup for quantization (index = nibble value 0-7)
_FP4_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def quantize_bf16_to_mxfp4(weight: torch.Tensor,
                             target_out_dim: int = None,
                             target_reduction: int = None) -> tuple:
    """Quantize BF16 weight [out_dim, in_dim] to MXFP4 blocks+scales.

    Uses MXFP4 format: FP4 E2M1 values with shared E8M0 block scales (32 elements/block).

    Args:
        weight: BF16 tensor [out_dim, in_dim]
        target_out_dim: pad output dim to this value
        target_reduction: pad reduction dim to this value (must be multiple of 32)

    Returns:
        blocks: uint8 [1, out_dim, num_blocks, 16] - packed FP4 nibbles
        scales: uint8 [1, out_dim, num_blocks] - E8M0 block scales
    """
    assert weight.ndim == 2
    out_dim, in_dim = weight.shape
    device = weight.device

    # Pad if needed
    if target_out_dim is not None and out_dim < target_out_dim:
        weight = torch.nn.functional.pad(weight, (0, 0, 0, target_out_dim - out_dim))
        out_dim = target_out_dim
    if target_reduction is not None and in_dim < target_reduction:
        weight = torch.nn.functional.pad(weight, (0, target_reduction - in_dim))
        in_dim = target_reduction

    assert in_dim % 32 == 0, f"in_dim {in_dim} must be multiple of 32"
    num_blocks = in_dim // 32

    # Work in float32
    w = weight.float().reshape(out_dim, num_blocks, 32)

    # Per-block max absolute value
    max_abs = w.abs().amax(dim=2)  # [out_dim, num_blocks]

    # Compute E8M0 scale exponent: scale = 2^(e-127)
    # Want: max_abs <= 6.0 * 2^(e-127), so e >= log2(max_abs/6) + 127
    safe_max = max_abs.clamp(min=2**-126)
    scale_exp_f = torch.ceil(torch.log2(safe_max / 6.0)) + 127.0
    scale_exp = scale_exp_f.clamp(0, 254).to(torch.int32)
    scale_exp[max_abs == 0] = 0
    scales_out = scale_exp.to(torch.uint8)  # [out_dim, num_blocks]

    # Scale value: 2^(scale_exp - 127)
    scale_val = torch.pow(2.0, (scale_exp.float() - 127.0))  # [out_dim, num_blocks]

    # Normalize values by scale
    w_norm = w / scale_val.unsqueeze(-1).clamp(min=2**-126)  # [out_dim, num_blocks, 32]

    # Quantize to nearest FP4 E2M1 magnitude
    w_sign = w_norm.sign()
    w_abs = w_norm.abs()

    # Round to nearest FP4 using midpoint thresholds
    # FP4 values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    nibble = torch.zeros_like(w_abs, dtype=torch.uint8)
    nibble[w_abs >= 0.25] = 1   # 0.5
    nibble[w_abs >= 0.75] = 2   # 1.0
    nibble[w_abs >= 1.25] = 3   # 1.5
    nibble[w_abs >= 1.75] = 4   # 2.0
    nibble[w_abs >= 2.50] = 5   # 3.0
    nibble[w_abs >= 3.50] = 6   # 4.0
    nibble[w_abs >= 5.00] = 7   # 6.0
    # Set sign bit (bit 3) for negative values
    nibble[w_sign < 0] |= 8

    # Pack pairs of nibbles into bytes: byte = lo_nibble | (hi_nibble << 4)
    even = nibble[:, :, 0::2]  # [out_dim, num_blocks, 16]
    odd = nibble[:, :, 1::2]   # [out_dim, num_blocks, 16]
    packed = (even | (odd << 4)).to(torch.uint8)  # [out_dim, num_blocks, 16]

    # Add batch dim=1 for compatibility with pack_mxfp4_workgroup
    return packed.unsqueeze(0).contiguous(), scales_out.unsqueeze(0).contiguous()


def pack_mxfp4_workgroup(blocks: torch.Tensor, scales: torch.Tensor,
                          output_per_wg: int = 16,
                          target_out_dim: int = None,
                          target_num_blocks: int = None) -> torch.Tensor:
    """Repack MXFP4 blocks+scales into workgroup layout for the MXFP4 GEMV kernel.

    Args:
        blocks: uint8 [E, out_dim, num_blocks, 16] - packed FP4 nibbles
        scales: uint8 [E, out_dim, num_blocks] - E8M0 block scales
        output_per_wg: output rows per workgroup (default 16)
        target_out_dim: pad output dim to this value (for MFMA alignment)
        target_num_blocks: pad num_blocks to this value (for reduction alignment)

    Returns:
        uint8 [E, expert_wgs, wg_bytes] tensor in workgroup layout:
        Per workgroup: [data: OPW * K/2 bytes][scales: OPW * num_blocks bytes]
    """
    E, out_dim, num_blocks, B = blocks.shape
    assert B == 16, f"Expected 16 bytes per block, got {B}"
    assert scales.shape == (E, out_dim, num_blocks)

    # Pad output dimension if needed
    if target_out_dim is not None and out_dim < target_out_dim:
        pad_rows = target_out_dim - out_dim
        blocks = torch.cat([blocks,
            torch.zeros(E, pad_rows, num_blocks, 16, dtype=torch.uint8, device=blocks.device)], dim=1)
        scales = torch.cat([scales,
            torch.zeros(E, pad_rows, num_blocks, dtype=torch.uint8, device=scales.device)], dim=1)
        out_dim = target_out_dim

    # Pad num_blocks (reduction dimension) if needed
    if target_num_blocks is not None and num_blocks < target_num_blocks:
        pad_blks = target_num_blocks - num_blocks
        blocks = torch.cat([blocks,
            torch.zeros(E, out_dim, pad_blks, 16, dtype=torch.uint8, device=blocks.device)], dim=2)
        scales = torch.cat([scales,
            torch.zeros(E, out_dim, pad_blks, dtype=torch.uint8, device=scales.device)], dim=2)
        num_blocks = target_num_blocks

    assert out_dim % output_per_wg == 0, \
        f"out_dim {out_dim} must be divisible by output_per_wg {output_per_wg}"

    expert_wgs = out_dim // output_per_wg
    K_half = num_blocks * 16  # K/2 bytes per row (16 bytes per block)
    wg_data_bytes = output_per_wg * K_half
    wg_scale_bytes = output_per_wg * num_blocks
    wg_bytes = wg_data_bytes + wg_scale_bytes

    # Reshape blocks: [E, out_dim, num_blocks, 16] -> [E, expert_wgs, OPW, K_half]
    data = blocks.reshape(E, expert_wgs, output_per_wg, -1)  # [E, wgs, OPW, K_half]

    # Reshape scales: [E, out_dim, num_blocks] -> [E, expert_wgs, OPW, num_blocks]
    sc = scales.reshape(E, expert_wgs, output_per_wg, num_blocks)

    # Concatenate data and scales per workgroup
    # data: [E, wgs, OPW, K_half] -> flatten last 2 dims -> [E, wgs, OPW*K_half]
    data_flat = data.reshape(E, expert_wgs, wg_data_bytes)
    # scales: [E, wgs, OPW, num_blocks] -> flatten last 2 dims -> [E, wgs, OPW*num_blocks]
    sc_flat = sc.reshape(E, expert_wgs, wg_scale_bytes)

    # Concatenate: [E, wgs, wg_data_bytes + wg_scale_bytes]
    packed = torch.cat([data_flat, sc_flat], dim=2)
    assert packed.shape == (E, expert_wgs, wg_bytes)
    return packed.contiguous()


def pack_bf16_workgroup(weight_bf16: torch.Tensor,
                        output_per_wg: int) -> torch.Tensor:
    """Pack BF16 weight [out_dim, K] into workgroup layout [n_wgs, OPW*K*2] bytes.

    Each workgroup: OPW rows x K bf16 values = OPW*K*2 bytes (no scales section).
    """
    out_dim, K = weight_bf16.shape
    assert out_dim % output_per_wg == 0
    n_wgs = out_dim // output_per_wg
    # [n_wgs, OPW, K] bf16 -> view as uint8 [n_wgs, OPW*K*2]
    return weight_bf16.reshape(n_wgs, output_per_wg, K).contiguous().view(
        torch.uint8).reshape(n_wgs, -1)


def dequant_mxfp4_to_bf16(blocks: torch.Tensor, scales: torch.Tensor,
                           target_out_dim: int = None,
                           target_reduction: int = None) -> torch.Tensor:
    """Dequantize MXFP4 (FP4 E2M1 + E8M0 scales) to bf16.

    Args:
        blocks: uint8 [E, out_dim, num_blocks, 16] - packed FP4 nibbles
        scales: uint8 [E, out_dim, num_blocks] - E8M0 block scales
        target_out_dim: pad output dim to this value
        target_reduction: pad K (= num_blocks * 32) to this value

    Returns:
        bf16 [E, out_dim, K] weight tensor
    """
    E, out_dim, num_blocks, B = blocks.shape
    assert B == 16, f"Expected 16 bytes per block, got {B}"

    # FP4 E2M1 lookup table
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                        dtype=torch.float32, device=blocks.device)

    # Extract low and high nibbles
    low = (blocks & 0x0F).long()    # [E, out_dim, num_blocks, 16]
    high = (blocks >> 4).long()     # [E, out_dim, num_blocks, 16]

    # Dequant nibbles via LUT
    low_vals = lut[low]             # [E, out_dim, num_blocks, 16] float32
    high_vals = lut[high]           # [E, out_dim, num_blocks, 16] float32

    # Interleave: each byte -> 2 values (low nibble first, high nibble second)
    vals = torch.stack([low_vals, high_vals], dim=-1)  # [..., 16, 2]
    vals = vals.reshape(E, out_dim, num_blocks, 32)    # [..., 32]

    # Apply E8M0 block scales: 2^(scale - 127)
    scale_factors = torch.pow(2.0, scales.float() - 127.0)  # [E, out_dim, num_blocks]
    vals = vals * scale_factors.unsqueeze(-1)  # broadcast to [..., 32]

    # Reshape to [E, out_dim, K]
    K = num_blocks * 32
    result = vals.reshape(E, out_dim, K).bfloat16()

    # Pad output dimension if needed
    if target_out_dim is not None and out_dim < target_out_dim:
        pad_rows = target_out_dim - out_dim
        result = torch.cat([result,
            torch.zeros(E, pad_rows, K, dtype=torch.bfloat16, device=result.device)], dim=1)
        out_dim = target_out_dim

    # Pad reduction dimension if needed
    if target_reduction is not None and K < target_reduction:
        pad_k = target_reduction - K
        result = torch.cat([result,
            torch.zeros(E, out_dim, pad_k, dtype=torch.bfloat16, device=result.device)], dim=2)

    return result.contiguous()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true", help="Use Mirage kernels")
    parser.add_argument("--use-triton", action="store_true",
                        help="Use triton_kernels (vLLM's MXFP4 MoE) for fast decode")
    parser.add_argument("--use-aiter", action="store_true",
                        help="Use AITER fused MoE (CK-tile MXFP4 path)")
    parser.add_argument("--max-num-batched-tokens", default=1, type=int)
    parser.add_argument("--max-num-batched-requests", default=1, type=int)
    parser.add_argument("--page-size", default=4096, type=int)
    parser.add_argument("--max-num-pages", default=16, type=int)
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--trace-name", default="", help="Perfetto trace output name")
    parser.add_argument("--profiling", action="store_true")
    parser.add_argument("--max-seq-length", default=512, type=int)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument(
        "--save-tokens", nargs="?", const="auto", default=None,
        help=("Dump generated token_ids to JSON for the correctness test. If the "
              "path is omitted, saves to outputs/gpt_oss/{torch_output.json|"
              "mpk_output.json}."),
    )
    parser.add_argument("--split-kv-cache", action="store_true", default=True)
    parser.add_argument("--no-split-kv-cache", action="store_false", dest="split_kv_cache")
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Only use first N layers (for memory-constrained testing)")
    parser.add_argument("--verify", action="store_true",
                        help="Run both PyTorch and Mirage, compare intermediates")
    args = parser.parse_args()
    if args.verify:
        args.use_mirage = True
    if args.use_aiter:
        args.use_mirage = False

    # Resolve where to dump generated tokens for the correctness test.
    if args.save_tokens:
        if args.save_tokens == "auto":
            fn = "mpk_output.json" if args.use_mirage else "torch_output.json"
            save_path = os.path.join(DEFAULT_SAVE_DIR, fn)
        else:
            save_path = args.save_tokens
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    else:
        save_path = None

    # Force cutlass off on ROCm/MI300X
    use_cutlass_kernel = False
    if getattr(torch.version, "hip", None):
        use_cutlass_kernel = False

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        rank = comm.Get_rank()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
    except ImportError:
        world_size = 1
        rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(rank)

    with torch.device("cuda"):
        model = GptOssForCausalLM.from_pretrained(
            args.model_path, world_size,
            max_num_pages=args.max_num_pages, page_size=args.page_size
        ).to(dtype=torch.bfloat16, device="cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    config = model.config
    num_layers = config.num_hidden_layers
    if args.max_layers is not None:
        num_layers = min(num_layers, args.max_layers)
        print(f"Using {num_layers} layers (out of {config.num_hidden_layers})")
    total_num_requests = 1 if not args.use_mirage else args.max_num_batched_requests
    tokens = torch.full((total_num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda")

    # Tokenize prompt (apply chat template if available)
    text = args.prompt
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([formatted], return_tensors="pt", add_special_tokens=False).to("cuda")
        print(f"Chat template applied: {len(model_inputs.input_ids[0])} tokens")
    else:
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    for r in range(total_num_requests):
        for i in range(model_inputs.input_ids.shape[-1]):
            tokens[r, i] = model_inputs.input_ids[0, i]
    prompt_lengths = torch.full(
        (total_num_requests,), model_inputs.input_ids.shape[-1],
        dtype=torch.int, device="cuda"
    )

    # Position embeddings
    positions = torch.arange(args.max_seq_length).unsqueeze(0).to("cuda")
    position_embeddings = model.model.rotary_emb(positions)

    # Tensors for decode loop
    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    prev_pos = 0

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    step = torch.full((total_num_requests,), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests,), 1, dtype=torch.int32, device="cuda")

    # Model dimensions
    hidden_size = config.hidden_size         # 2880
    intermediate_size = config.intermediate_size  # 2880
    num_q_heads = config.num_attention_heads  # 64
    num_kv_heads = config.num_key_value_heads  # 8
    num_local_q_heads = num_q_heads // world_size
    num_local_kv_heads = num_kv_heads // world_size
    head_dim = config.head_dim                # 64
    fused_qkv_dim = (num_local_q_heads + 2 * num_local_kv_heads) * head_dim  # 5120
    num_experts = config.num_local_experts    # 128
    num_experts_per_tok = config.num_experts_per_tok  # 4

    # Per-layer sliding window: even layers use sliding_window, odd use full attention
    layer_types = getattr(config, 'layer_types', None)
    sw_value = getattr(config, 'sliding_window', None) or 0
    per_layer_sliding_window = []
    for li in range(config.num_hidden_layers):
        if layer_types is not None:
            sw = sw_value if layer_types[li] == "sliding_attention" else 0
        else:
            sw = sw_value if li % 2 == 0 else 0
        per_layer_sliding_window.append(sw)
    if sw_value > 0:
        n_sw = sum(1 for sw in per_layer_sliding_window if sw > 0)
        print(f"Sliding window: {sw_value} tokens, {n_sw}/{config.num_hidden_layers} layers")

    # RMSNorm compensation factor: padding zeros change the variance denominator.
    # RMS_padded = sqrt(sum(x^2)/N_padded) < RMS_orig = sqrt(sum(x^2)/N_orig)
    # So padded output is LARGER by sqrt(N_padded/N_orig).
    # To compensate, multiply weights by sqrt(N_orig/N_padded) to cancel the inflation.
    rmsnorm_scale_factor = math.sqrt(hidden_size / PADDED_HIDDEN_SIZE)

    print(f"GPT-OSS 120B: hidden={hidden_size}, intermediate={intermediate_size}, "
          f"experts={num_experts}, top_k={num_experts_per_tok}, "
          f"Q_heads={num_q_heads}, KV_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"Padding: hidden {hidden_size} -> {PADDED_HIDDEN_SIZE}, "
          f"intermediate {intermediate_size} -> {PADDED_INTERMEDIATE_SIZE}")
    print(f"RMSNorm scale factor: {rmsnorm_scale_factor:.6f}")

    # === Triton kernels MoE setup (vLLM's fast MXFP4 path) ===
    if args.use_triton:
        print("\n=== Setting up triton_kernels MoE (vLLM MXFP4 path) ===")
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import _swizzle_mxfp4
        from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig
        from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a16_moe_quant_config
        from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
            triton_kernel_fused_experts, make_routing_data,
        )

        num_warps = 8

        for li in range(num_layers):
            layer = model.model.layers[li]
            experts = layer.mlp.experts

            # Reshape blocks to [E, N, K//2]
            gu_blocks = experts.gate_up_proj_blocks.to("cuda")
            gu_scales = experts.gate_up_proj_scales.to("cuda")
            dp_blocks = experts.down_proj_blocks.to("cuda")
            dp_scales = experts.down_proj_scales.to("cuda")

            w13_raw = gu_blocks.reshape(num_experts, 2*intermediate_size, -1).contiguous()
            w13_scale_raw = gu_scales.contiguous()
            w2_raw = dp_blocks.reshape(num_experts, hidden_size, -1).contiguous()
            w2_scale_raw = dp_scales.contiguous()

            # Pad to 3072 (hidden and inter both need K//32 divisible by 8)
            w13_w = torch.zeros(num_experts, 2*PADDED_INTERMEDIATE_SIZE, PADDED_HIDDEN_SIZE//2,
                                dtype=torch.uint8, device="cuda")
            w13_w[:, :2*intermediate_size, :w13_raw.shape[2]] = w13_raw
            w13_s = torch.zeros(num_experts, 2*PADDED_INTERMEDIATE_SIZE, PADDED_HIDDEN_SIZE//32,
                                dtype=torch.uint8, device="cuda")
            w13_s[:, :2*intermediate_size, :w13_scale_raw.shape[2]] = w13_scale_raw

            w2_w = torch.zeros(num_experts, PADDED_HIDDEN_SIZE, PADDED_INTERMEDIATE_SIZE//2,
                               dtype=torch.uint8, device="cuda")
            w2_w[:, :hidden_size, :w2_raw.shape[2]] = w2_raw
            w2_s = torch.zeros(num_experts, PADDED_HIDDEN_SIZE, PADDED_INTERMEDIATE_SIZE//32,
                               dtype=torch.uint8, device="cuda")
            w2_s[:, :hidden_size, :w2_scale_raw.shape[2]] = w2_scale_raw

            # Biases (must be float32 for triton_kernels)
            gu_bias = torch.zeros(num_experts, 2*PADDED_INTERMEDIATE_SIZE,
                                  dtype=torch.float32, device="cuda")
            gu_bias[:, :2*intermediate_size] = experts.gate_up_proj_bias.data.to("cuda").float()
            dp_bias = torch.zeros(num_experts, PADDED_HIDDEN_SIZE,
                                  dtype=torch.float32, device="cuda")
            dp_bias[:, :hidden_size] = experts.down_proj_bias.data.to("cuda").float()

            # Swizzle (transpose + GFX950MXScaleLayout)
            w13_swiz, w13_flex, w13_sc_swiz = _swizzle_mxfp4(w13_w, w13_s, num_warps)
            w2_swiz, w2_flex, w2_sc_swiz = _swizzle_mxfp4(w2_w, w2_s, num_warps)

            # Build quant config
            qcfg = mxfp4_w4a16_moe_quant_config(
                w1_scale=PrecisionConfig(weight_scale=w13_sc_swiz,
                                         flex_ctx=FlexCtx(rhs_data=w13_flex)),
                w2_scale=PrecisionConfig(weight_scale=w2_sc_swiz,
                                         flex_ctx=FlexCtx(rhs_data=w2_flex)),
                w1_bias=gu_bias,
                w2_bias=dp_bias,
            )

            # Free intermediate padded tensors
            del w13_w, w13_s, w2_w, w2_s, w13_raw, w13_scale_raw, w2_raw, w2_scale_raw
            del gu_blocks, gu_scales, dp_blocks, dp_scales

            # Store on the MLP module for forward()
            layer.mlp._triton_w13 = w13_swiz
            layer.mlp._triton_w2 = w2_swiz
            layer.mlp._triton_qcfg = qcfg
            layer.mlp._triton_num_experts = num_experts
            layer.mlp._triton_topk = num_experts_per_tok

            # Free original MXFP4 blocks
            experts.gate_up_proj_blocks = None
            experts.gate_up_proj_scales = None
            experts.down_proj_blocks = None
            experts.down_proj_scales = None
            experts.gate_up_proj_bias = None
            experts.down_proj_bias = None

            if li % 6 == 0 or li == num_layers - 1:
                mem_used = torch.cuda.memory_allocated() / 1e9
                print(f"  Layer {li}: swizzled OK (GPU mem: {mem_used:.1f} GB)")

            torch.cuda.empty_cache()

        # Monkey-patch GptOssMLP.forward to use triton_kernels
        import types
        from models.modeling_gpt_oss import GptOssMLP

        def triton_mlp_forward(self, hidden_states):
            """Fast MoE forward using triton_kernels (same as vLLM)."""
            # Router: get logits and top-k
            router = self.router
            hs_flat = hidden_states.reshape(-1, router.hidden_dim)
            router_logits = torch.nn.functional.linear(hs_flat, router.weight, router.bias)

            # Clamp logits to prevent NaN propagation
            router_logits = router_logits.clamp(-1e4, 1e4)

            # Top-k + softmax (same as original GptOssRouter)
            topk_values, topk_ids = torch.topk(router_logits, self._triton_topk, dim=-1)
            topk_weights = torch.softmax(topk_values.float(), dim=-1)

            # Build routing objects
            routing_data, gather_indx, scatter_indx = make_routing_data(
                topk_ids.to(torch.int16),
                topk_weights.to(torch.bfloat16),
                num_local_experts=self._triton_num_experts,
            )

            # Pad input to PADDED_HIDDEN_SIZE
            bs = hs_flat.shape[0]
            x_padded = torch.zeros(bs, PADDED_HIDDEN_SIZE, dtype=hs_flat.dtype, device=hs_flat.device)
            x_padded[:, :router.hidden_dim] = hs_flat

            # Call triton_kernel_fused_experts
            out_padded = triton_kernel_fused_experts(
                output_tensor=None,
                hidden_states=x_padded,
                w1=self._triton_w13,
                w2=self._triton_w2,
                routing_data=routing_data,
                gather_indx=gather_indx,
                scatter_indx=scatter_indx,
                activation="swigluoai",
                quant_config=self._triton_qcfg,
                apply_router_weight_on_input=False,
                global_num_experts=self._triton_num_experts,
            )

            # Slice back to original hidden_size and reshape
            out = out_padded[:, :router.hidden_dim]
            return out.view_as(hidden_states), router_logits

        for li in range(num_layers):
            layer = model.model.layers[li]
            layer.mlp.forward = types.MethodType(triton_mlp_forward, layer.mlp)

        # Also patch decoder layer forward for timing
        from models.modeling_gpt_oss import GptOssDecoderLayer
        _orig_layer_forward = GptOssDecoderLayer.forward

        _layer_profile_times = {'attn': [], 'moe': [], 'other': []}
        _layer_profile_step = [0]

        def profiled_layer_forward(self, hidden_states, position_embeddings=None, step=None):
            profile = (_layer_profile_step[0] == 5)
            if profile:
                torch.cuda.synchronize()
                import time
                t0 = time.perf_counter()

            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            if profile:
                torch.cuda.synchronize()
                t1 = time.perf_counter()

            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                step=step,
            )

            if profile:
                torch.cuda.synchronize()
                t2 = time.perf_counter()

            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

            if profile:
                torch.cuda.synchronize()
                t3 = time.perf_counter()

            hidden_states, router_logits = self.mlp(hidden_states)

            if profile:
                torch.cuda.synchronize()
                t4 = time.perf_counter()

            hidden_states = residual + hidden_states

            if profile:
                _layer_profile_times['other'].append((t1-t0)*1000 + (t3-t2)*1000)
                _layer_profile_times['attn'].append((t2-t1)*1000)
                _layer_profile_times['moe'].append((t4-t3)*1000)

            return hidden_states, router_logits

        for li in range(num_layers):
            layer = model.model.layers[li]
            layer.forward = types.MethodType(profiled_layer_forward, layer)

        print(f"  Patched {num_layers} layers with profiled layer forward\n")

        # Add per-component profiling
        _triton_profile_step = [0]  # mutable int in list for closure
        _triton_profile_times = {}  # component -> total_ms

        def triton_profile_mlp_forward(self, hidden_states):
            """Profiled version of triton MoE forward."""
            step = _triton_profile_step[0]
            profile = (step == 5)  # profile 5th decode step

            if profile:
                torch.cuda.synchronize()
                import time
                t0 = time.perf_counter()

            router = self.router
            hs_flat = hidden_states.reshape(-1, router.hidden_dim)
            router_logits = torch.nn.functional.linear(hs_flat, router.weight, router.bias)
            router_logits = router_logits.clamp(-1e4, 1e4)

            if profile:
                torch.cuda.synchronize()
                t1 = time.perf_counter()

            topk_values, topk_ids = torch.topk(router_logits, self._triton_topk, dim=-1)
            topk_weights = torch.softmax(topk_values.float(), dim=-1)
            routing_data, gather_indx, scatter_indx = make_routing_data(
                topk_ids.to(torch.int16),
                topk_weights.to(torch.bfloat16),
                num_local_experts=self._triton_num_experts,
            )

            if profile:
                torch.cuda.synchronize()
                t2 = time.perf_counter()

            bs = hs_flat.shape[0]
            x_padded = torch.zeros(bs, PADDED_HIDDEN_SIZE, dtype=hs_flat.dtype, device=hs_flat.device)
            x_padded[:, :router.hidden_dim] = hs_flat

            out_padded = triton_kernel_fused_experts(
                output_tensor=None,
                hidden_states=x_padded,
                w1=self._triton_w13,
                w2=self._triton_w2,
                routing_data=routing_data,
                gather_indx=gather_indx,
                scatter_indx=scatter_indx,
                activation="swigluoai",
                quant_config=self._triton_qcfg,
                apply_router_weight_on_input=False,
                global_num_experts=self._triton_num_experts,
            )

            if profile:
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                _triton_profile_times.setdefault('moe_router', []).append((t1-t0)*1000)
                _triton_profile_times.setdefault('moe_routing', []).append((t2-t1)*1000)
                _triton_profile_times.setdefault('moe_gemm', []).append((t3-t2)*1000)

            out = out_padded[:, :router.hidden_dim]
            return out.view_as(hidden_states), router_logits

    # === AITER fused MoE setup (CK-tile MXFP4 path) ===
    if args.use_aiter:
        import types
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe as aiter_fused_moe

        HIDDEN_PAD = PADDED_HIDDEN_SIZE - hidden_size
        INTER_PAD = PADDED_INTERMEDIATE_SIZE - intermediate_size

        print(f"\n=== Setting up AITER fused MoE (CK-tile MXFP4 path) ===")
        print(f"  hidden={hidden_size}, padded={PADDED_HIDDEN_SIZE}, pad={HIDDEN_PAD}")
        print(f"  inter={intermediate_size}, padded={PADDED_INTERMEDIATE_SIZE}, pad={INTER_PAD}")

        for li in range(num_layers):
            layer = model.model.layers[li]
            experts = layer.mlp.experts

            # Load raw MXFP4 blocks+scales
            gu_blocks = experts.gate_up_proj_blocks.to("cuda")
            gu_scales = experts.gate_up_proj_scales.to("cuda")
            dp_blocks = experts.down_proj_blocks.to("cuda")
            dp_scales = experts.down_proj_scales.to("cuda")

            # Reshape to [E, N, K_packed]
            w1_interleaved = gu_blocks.reshape(num_experts, 2 * intermediate_size, -1).contiguous()
            w1_sc_interleaved = gu_scales.reshape(num_experts, 2 * intermediate_size, -1).contiguous()
            w2_raw = dp_blocks.reshape(num_experts, hidden_size, -1).contiguous()
            w2_sc_raw = dp_scales.contiguous()

            # De-interleave gate/up: model stores [gate[0],up[0],gate[1],up[1],...]
            # AITER expects concatenated: [gate[0],...,gate[N-1],up[0],...,up[N-1]]
            w1_raw = torch.cat([w1_interleaved[:, 0::2, :],
                                w1_interleaved[:, 1::2, :]], dim=1).contiguous()
            w1_sc_raw = torch.cat([w1_sc_interleaved[:, 0::2, :],
                                   w1_sc_interleaved[:, 1::2, :]], dim=1).contiguous()
            del w1_interleaved, w1_sc_interleaved

            # Pad to PADDED dimensions
            w1 = torch.zeros(num_experts, 2 * PADDED_INTERMEDIATE_SIZE, PADDED_HIDDEN_SIZE // 2,
                             dtype=torch.uint8, device="cuda")
            w1[:, :2 * intermediate_size, :w1_raw.shape[2]] = w1_raw
            w1_scale = torch.zeros(num_experts, 2 * PADDED_INTERMEDIATE_SIZE, PADDED_HIDDEN_SIZE // 32,
                                   dtype=torch.uint8, device="cuda")
            w1_scale[:, :2 * intermediate_size, :w1_sc_raw.shape[2]] = w1_sc_raw

            w2 = torch.zeros(num_experts, PADDED_HIDDEN_SIZE, PADDED_INTERMEDIATE_SIZE // 2,
                             dtype=torch.uint8, device="cuda")
            w2[:, :hidden_size, :w2_raw.shape[2]] = w2_raw
            w2_scale = torch.zeros(num_experts, PADDED_HIDDEN_SIZE, PADDED_INTERMEDIATE_SIZE // 32,
                                   dtype=torch.uint8, device="cuda")
            w2_scale[:, :hidden_size, :w2_sc_raw.shape[2]] = w2_sc_raw

            # Biases (float32 for CK-tile) — de-interleave gate/up bias
            bias1_interleaved = experts.gate_up_proj_bias.data.to("cuda").float()
            bias1_concat = torch.cat([bias1_interleaved[:, 0::2],
                                      bias1_interleaved[:, 1::2]], dim=1)
            bias1 = torch.zeros(num_experts, 2 * PADDED_INTERMEDIATE_SIZE,
                                dtype=torch.float32, device="cuda")
            bias1[:, :2 * intermediate_size] = bias1_concat
            del bias1_interleaved, bias1_concat
            bias2 = torch.zeros(num_experts, PADDED_HIDDEN_SIZE,
                                dtype=torch.float32, device="cuda")
            bias2[:, :hidden_size] = experts.down_proj_bias.data.to("cuda").float()

            # Store on module
            layer.mlp._aiter_w1 = w1
            layer.mlp._aiter_w2 = w2
            layer.mlp._aiter_w1_scale = w1_scale
            layer.mlp._aiter_w2_scale = w2_scale
            layer.mlp._aiter_bias1 = bias1
            layer.mlp._aiter_bias2 = bias2
            layer.mlp._aiter_num_experts = num_experts
            layer.mlp._aiter_topk = num_experts_per_tok

            # Free original weights
            experts.gate_up_proj_blocks = None
            experts.gate_up_proj_scales = None
            experts.down_proj_blocks = None
            experts.down_proj_scales = None
            experts.gate_up_proj_bias = None
            experts.down_proj_bias = None

            if li % 6 == 0 or li == num_layers - 1:
                mem_used = torch.cuda.memory_allocated() / 1e9
                print(f"  Layer {li}: prepared OK (GPU mem: {mem_used:.1f} GB)")

            torch.cuda.empty_cache()

        # Monkey-patch MLP forward to use AITER
        from models.modeling_gpt_oss import GptOssMLP

        def aiter_mlp_forward(self, hidden_states):
            """MoE forward using AITER fused_moe (CK-tile MXFP4)."""
            router = self.router
            hs_flat = hidden_states.reshape(-1, router.hidden_dim)
            router_logits = torch.nn.functional.linear(hs_flat, router.weight, router.bias)
            router_logits = router_logits.clamp(-1e4, 1e4)

            topk_values, topk_ids = torch.topk(router_logits, self._aiter_topk, dim=-1)
            topk_ids = topk_ids.to(torch.int32)
            topk_weights = torch.softmax(topk_values.float(), dim=-1)

            # Pad input to PADDED_HIDDEN_SIZE
            bs = hs_flat.shape[0]
            x_padded = torch.zeros(bs, PADDED_HIDDEN_SIZE, dtype=hs_flat.dtype, device=hs_flat.device)
            x_padded[:, :router.hidden_dim] = hs_flat

            out = aiter_fused_moe(
                hidden_states=x_padded,
                w1=self._aiter_w1,
                w2=self._aiter_w2,
                topk_weight=topk_weights,
                topk_ids=topk_ids,
                activation=ActivationType.Swiglu,
                quant_type=QuantType.per_1x32,
                w1_scale=self._aiter_w1_scale,
                w2_scale=self._aiter_w2_scale,
                hidden_pad=HIDDEN_PAD,
                intermediate_pad=INTER_PAD,
                bias1=self._aiter_bias1,
                bias2=self._aiter_bias2,
            )

            return out[:, :router.hidden_dim].view_as(hidden_states), router_logits

        for li in range(num_layers):
            layer = model.model.layers[li]
            layer.mlp.forward = types.MethodType(aiter_mlp_forward, layer.mlp)

        # Profiled layer forward (same as triton path)
        from models.modeling_gpt_oss import GptOssDecoderLayer
        _orig_layer_forward = GptOssDecoderLayer.forward

        _aiter_profile_times = {'attn': [], 'moe': [], 'other': []}
        _aiter_profile_step = [0]

        def aiter_profiled_layer_forward(self, hidden_states, position_embeddings=None, step=None):
            profile = (_aiter_profile_step[0] == 5)
            if profile:
                torch.cuda.synchronize()
                import time
                t0 = time.perf_counter()

            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            if profile:
                torch.cuda.synchronize()
                t1 = time.perf_counter()

            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                step=step,
            )

            if profile:
                torch.cuda.synchronize()
                t2 = time.perf_counter()

            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

            if profile:
                torch.cuda.synchronize()
                t3 = time.perf_counter()

            hidden_states, router_logits = self.mlp(hidden_states)

            if profile:
                torch.cuda.synchronize()
                t4 = time.perf_counter()

            hidden_states = residual + hidden_states

            if profile:
                _aiter_profile_times['other'].append((t1 - t0) * 1000 + (t3 - t2) * 1000)
                _aiter_profile_times['attn'].append((t2 - t1) * 1000)
                _aiter_profile_times['moe'].append((t4 - t3) * 1000)

            return hidden_states, router_logits

        for li in range(num_layers):
            layer = model.model.layers[li]
            layer.forward = types.MethodType(aiter_profiled_layer_forward, layer)

        print(f"  Patched {num_layers} layers with AITER MoE + profiled forward\n")

    if args.use_mirage:
        import mirage as mi

        # Gang dispatch is required for MXFP4 MoE kernels on MI300/MI350
        os.environ.setdefault("USE_GANG", "1")

        # Pad vocab_size to facilitate task graph creation
        # GPT-OSS vocab_size = 201088, round up to multiple of 256
        padded_vocab_size = ((config.vocab_size + 255) // 256) * 256  # 201216
        lm_head_weight = torch.cat(
            (
                # Pad lm_head output_dim (hidden_size -> PADDED_HIDDEN_SIZE)
                pad_weight_2d(model.lm_head.weight, target_cols=PADDED_HIDDEN_SIZE),
                torch.zeros(
                    (padded_vocab_size - config.vocab_size, PADDED_HIDDEN_SIZE),
                    device="cuda",
                ),
            ),
            0,
        )
        assert lm_head_weight.stride()[0] == PADDED_HIDDEN_SIZE
        vocab_size = padded_vocab_size

        # Quantize LM head to MXFP4 for FP4×FP8 MFMA (3.7x less HBM traffic)
        # OPW must be >= 64 (4 waves × 16 rows/MFMA tile = 64 minimum)
        lm_head_output_per_wg = 64
        lm_blocks, lm_scales = quantize_bf16_to_mxfp4(lm_head_weight)
        lm_head_packed = pack_mxfp4_workgroup(
            lm_blocks, lm_scales, output_per_wg=lm_head_output_per_wg,
        ).squeeze(0)  # [n_wgs, wg_bytes]
        print(f"LM head MXFP4: {lm_head_weight.shape} BF16 ({lm_head_weight.numel()*2/1e6:.0f} MB) "
              f"-> {lm_head_packed.shape} packed ({lm_head_packed.numel()/1e6:.0f} MB)")

        num_kv_cache_chunks = max(1, (args.max_seq_length + 127) // 128)
        use_ck_fmha = int(os.environ.get("USE_CK_FMHA", "1")) == 1
        # Split-KV chunks for the CK FMHA decode kernel. >1 parallelizes the KV
        # tile loop across (kv_head, chunk_idx) blocks and runs a merge step;
        # required code paths exist in paged_attention_decode_minimal_hd64_mi300.cuh
        # (chunk-aware partition) and merge_splitkv.cuh (with optional sinks).
        #
        # Chunks are claimed by xcd_rank inside the fused full-layer gang task
        # (gang_full_layer_fused_mi300.cuh: `if (xcd_rank < NUM_KV_CHUNKS)`), so
        # NUM_KV_CHUNKS must never exceed the workers available on one XCD --
        # otherwise the chunk barrier never reaches NUM_KV_CHUNKS-1, the merge
        # never fires, and the megakernel deadlocks.
        #
        # Attention time per chunk is proportional to seqlen/NUM_KV_CHUNKS, so
        # scale chunks with sequence length to keep decode latency flat. The
        # decode kernel already stamps LSE=-inf for chunks that get no KV tiles,
        # so over-provisioning chunks at short seqlen is safe (just wasteful).
        #
        # KV_TILE=64 in paged_attention_decode_minimal_hd64_mi300.cuh; aim for
        # >=2 tiles per chunk so a chunk is worth its merge overhead.
        _nw, _ = mi.get_configurations_from_gpu(rank)
        MAX_KV_CHUNKS = _nw // 8  # workers per XCD (240/8 = 30 on MI350)
        _env_chunks = os.environ.get("CK_FMHA_NUM_KV_CHUNKS")
        if _env_chunks is not None:
            ck_fmha_num_kv_chunks = int(_env_chunks)
        else:
            _kv_tiles = max(1, (args.max_seq_length + 63) // 64)
            ck_fmha_num_kv_chunks = max(8, min(MAX_KV_CHUNKS, _kv_tiles // 2))
        assert ck_fmha_num_kv_chunks >= 1
        use_split_attn_chunks = (ck_fmha_num_kv_chunks > 1)
        fuse_full_layer = os.environ.get("FUSE_FULL_LAYER", "1") == "1"
        if fuse_full_layer and ck_fmha_num_kv_chunks > MAX_KV_CHUNKS:
            raise ValueError(
                f"CK_FMHA_NUM_KV_CHUNKS={ck_fmha_num_kv_chunks} exceeds the "
                f"{MAX_KV_CHUNKS} workers per XCD available to claim chunks in "
                f"the fused full-layer gang task; the split-KV merge would "
                f"never fire and the kernel would hang."
            )
        print(f"[CFG] max_seq_length={args.max_seq_length} "
              f"ck_fmha_num_kv_chunks={ck_fmha_num_kv_chunks} "
              f"(max {MAX_KV_CHUNKS})")
        fuse_tail = os.environ.get("FUSE_TAIL", "0") == "1"

        if args.profiling:
            profiler_tensor = torch.zeros(
                30000 * 1280, dtype=torch.uint64, device="cuda"
            ).contiguous()
        else:
            profiler_tensor = None

        num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
        qo_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        paged_kv_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        paged_kv_indices_buffer = torch.zeros(
            args.max_num_pages, dtype=torch.int32, device="cuda")
        paged_kv_last_page_len_buffer = torch.empty(
            args.max_num_batched_requests, dtype=torch.int32, device="cuda")

        mpk = mi.PersistentKernel(
            mode="offline",
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=args.max_seq_length,
            max_num_batched_requests=args.max_num_batched_requests,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_pages=args.max_num_pages,
            page_size=args.page_size,
            eos_token_id=config.eos_token_id if not args.ignore_eos else 0x7FFFFFFF,
            meta_tensors={
                "step": step,
                "tokens": tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "num_new_tokens": num_new_tokens,
                "prompt_lengths": prompt_lengths,
                "qo_indptr_buffer": qo_indptr_buffer,
                "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
                "paged_kv_indices_buffer": paged_kv_indices_buffer,
                "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
            },
            profiler_tensor=profiler_tensor,
            trace_name=args.trace_name,
            spec_decode_config=None,
            use_cutlass_kernel=use_cutlass_kernel
        )

        bs = args.max_num_batched_tokens

        # --- Intermediate tensors ---
        x = mpk.attach_input(torch_tensor=input_tokens, name="input_token")
        # Mirage kv_cache_update kernel reads cos/sin with stride HEAD_DIM per position,
        # but GPT-OSS rotary embedding produces only HEAD_DIM/2 elements per position.
        # Pad to HEAD_DIM so the kernel reads correct values for all positions.
        cos_raw = position_embeddings[0][0, :args.max_seq_length, :]  # [seq_len, head_dim/2]
        sin_raw = position_embeddings[1][0, :args.max_seq_length, :]  # [seq_len, head_dim/2]
        print(f"[DEBUG] cos_raw shape: {cos_raw.shape}, head_dim: {head_dim}")
        cos_padded = torch.nn.functional.pad(cos_raw, (0, head_dim - cos_raw.shape[-1]))  # [seq_len, head_dim]
        sin_padded = torch.nn.functional.pad(sin_raw, (0, head_dim - sin_raw.shape[-1]))  # [seq_len, head_dim]
        print(f"[DEBUG] cos_padded shape: {cos_padded.shape}, cos_padded[0,:5]: {cos_padded[0,:5]}, cos_padded[1,:5]: {cos_padded[1,:5]}")
        cos_pos_embed = mpk.attach_input(
            torch_tensor=cos_padded.contiguous(),
            name="cos_position_embedding",
        )
        sin_pos_embed = mpk.attach_input(
            torch_tensor=sin_padded.contiguous(),
            name="sin_position_embedding",
        )

        # Intermediate tensors: always use attach_input (PyTorch-backed) to avoid
        # megakernel internal memory aliasing that causes wrong output.
        verify_tensors = {}
        _tensor_refs = {}  # prevent GC of backing tensors
        def make_tensor(name, dims, torch_dtype=torch.bfloat16):
            t = torch.zeros(dims, dtype=torch_dtype, device="cuda")
            _tensor_refs[name] = t
            if args.verify:
                verify_tensors[name] = t
            return mpk.attach_input(torch_tensor=t, name=name)
        y = make_tensor("embed_out", (bs, PADDED_HIDDEN_SIZE))
        rmsnorm_out = make_tensor("rmsnorm_out", (bs, PADDED_HIDDEN_SIZE))
        attn_in = make_tensor("attn_in", (bs, fused_qkv_dim))
        # CK FMHA workspaces
        num_qo_per_kv = num_local_q_heads // num_local_kv_heads
        q_ws_stride = num_local_q_heads * head_dim
        ck_fmha_q_ws_tensor = torch.zeros(
            bs, q_ws_stride, dtype=torch.bfloat16, device="cuda")
        ck_fmha_q_ws = mpk.attach_input(
            torch_tensor=ck_fmha_q_ws_tensor, name="ck_fmha_q_workspace")
        lse_dim1 = num_local_kv_heads * ck_fmha_num_kv_chunks * num_qo_per_kv
        ck_fmha_lse_acc_tensor = torch.zeros(
            bs, lse_dim1, dtype=torch.float32, device="cuda")
        ck_fmha_lse_acc = mpk.attach_input(
            torch_tensor=ck_fmha_lse_acc_tensor, name="ck_fmha_lse_acc")

        attn_out = make_tensor("attn_out", (bs, num_local_q_heads * head_dim))
        # When CK_FMHA_NUM_KV_CHUNKS > 1 the decode kernel writes per-chunk float
        # partials into ck_fmha_o_acc; merge step combines them into attn_out.
        if use_split_attn_chunks or fuse_full_layer:
            o_acc_dim1 = num_local_kv_heads * ck_fmha_num_kv_chunks * num_qo_per_kv * head_dim
            ck_fmha_o_acc_tensor = torch.zeros(
                bs, o_acc_dim1, dtype=torch.float32, device="cuda")
            ck_fmha_o_acc = mpk.attach_input(
                torch_tensor=ck_fmha_o_acc_tensor, name="ck_fmha_o_acc")
        else:
            ck_fmha_o_acc = None
        attn_proj_out = make_tensor("attn_proj_out", (bs, PADDED_HIDDEN_SIZE))
        if world_size > 1:
            allreduce_buf = mpk.new_tensor(
                dims=(world_size, bs, PADDED_HIDDEN_SIZE),
                dtype=mi.bfloat16,
                name="all_reduce_buf",
                io_category="nvshmem_tensor",
            )
            attn_allreduce_out = mpk.new_tensor(
                dims=(bs, PADDED_HIDDEN_SIZE),
                dtype=mi.bfloat16,
                name="attn_allreduce_out",
                io_category="nvshmem_tensor",
            )
        else:
            allreduce_buf = make_tensor("all_reduce_buf", (world_size, bs, PADDED_HIDDEN_SIZE))
            attn_allreduce_out = make_tensor("attn_allreduce_out", (bs, PADDED_HIDDEN_SIZE))
        rmsnorm_out_moe = make_tensor("rmsnorm_out_moe", (bs, PADDED_HIDDEN_SIZE))
        moe_gate_out = make_tensor("moe_gate_out", (bs, num_experts))
        moe_routing_indices = make_tensor("moe_routing_indices", (num_experts, bs), torch_dtype=torch.int32)
        moe_mask = make_tensor("moe_mask", (num_experts + 1,), torch_dtype=torch.int32)
        moe_topk_weight = make_tensor("moe_topk_weight", (bs, num_experts_per_tok), torch_dtype=torch.float32)
        # Atomic counter for fused router+TopK gang task (single int32, init to 0)
        router_topk_counter = make_tensor("router_topk_counter", (1,), torch_dtype=torch.int32)
        # Hierarchical barrier for fused O-PROJ+TopK kernel.
        # Layout (HIER_STRIDE=16 int32 = 64 bytes = 1 cache line per entry):
        #   [0*16]: XCD0 arrive, [1*16]: XCD1 arrive, ... [7*16]: XCD7 arrive
        #   [8*16]: global_arrive (leader count)
        #   [9*16]: topk_counter
        # Total: 160 int32 = 640 bytes (10 cache lines, no false sharing)
        fuse_oproj_moe = os.environ.get("FUSE_OPROJ_MOE", "0") == "1"
        fuse_oproj_topk = os.environ.get("FUSE_OPROJ_TOPK", "1") == "1"
        if fuse_full_layer:
            fuse_oproj_moe = True
        if fuse_oproj_moe:
            fuse_oproj_topk = True  # fused O-proj+MoE implies fused O-proj+TopK
        if fuse_oproj_topk:
            # 704 int32: full-layer fusion counter buffer layout:
            #   0..18*16-1:  type 215 counters
            #   19*16 (304): attn_global_counter (cross-XCD sync)
            #   20*16 (320): qkv_epoch[0..7] per-XCD epoch flags
            #   28*16 (448): chunk_barrier[0..7] per-XCD chunk arrival
            #   36*16 (576): attn_xcd_release[0..7] per-XCD release flags
            counter_size = 896 if fuse_tail else 832
            oproj_topk_counters = make_tensor("oproj_topk_counters", (counter_size,), torch_dtype=torch.int32)
        # Hierarchical barrier for fused QKV+Attention kernel [16 int32]:
        # [0..7]: per-XCD QKV arrival counters, [8]: global leader count
        fuse_qkv_attn = os.environ.get("FUSE_QKV_ATTN", "1") == "1"
        # Gang QKV+Attn fusion uses the chunks=1 internal attention path; for
        # chunks>1 we fall back to separate QKV→attn→merge tasks.
        if use_split_attn_chunks:
            fuse_qkv_attn = False
        if fuse_full_layer:
            fuse_qkv_attn = True
        if fuse_qkv_attn:
            qkv_attn_barrier = make_tensor("qkv_attn_barrier", (16,), torch_dtype=torch.int32)
        # Hierarchical barrier for fused W13+W2 kernel [160*E int32].
        # Per expert, 10 slots of one 64-byte cache line (16 int32) each:
        #   [x*16] for x in 0..7: per-XCD release flag (st_wt, bypasses L2)
        #   [8*16]: global_arrive (atomic, lives in L2)
        #   [9*16]: reserved
        # The one-line-per-slot spacing is required, not padding: an L2-resident
        # atomic sharing a line with write-through release flags can write the
        # stale L2 copy back over them, reverting releases that already
        # happened and deadlocking the W2 workers for that expert. See the
        # layout note in gang_moe_fused_mxfp4_mi300.cuh (MOE_BAR_*).
        moe_fused_barrier = make_tensor("moe_fused_barrier", (160 * num_experts,), torch_dtype=torch.int32)
        # W13+SwiGLU fused output: [bs, top_k, padded_intermediate]
        # (SwiGLU is fused into W13 epilogue — no separate mlp_mid buffer)
        swiglu_out = make_tensor("swiglu_out", (bs, num_experts_per_tok, PADDED_INTERMEDIATE_SIZE))
        # W2 output: [bs, top_k, padded_hidden]
        mlp_out = make_tensor("mlp_out", (bs, num_experts_per_tok, PADDED_HIDDEN_SIZE))
        # F32 workspace for W2 atomicAdd: replaces MulSumAdd standalone task
        moe_workspace_f32 = make_tensor("moe_workspace_f32", (bs, PADDED_HIDDEN_SIZE), torch_dtype=torch.float32)
        mlp_weighted_sum_out = make_tensor("mlp_weighted_sum_out", (bs, PADDED_HIDDEN_SIZE))
        mlp_final = make_tensor("mlp_final", (bs, PADDED_HIDDEN_SIZE))
        # Argmax — fused into LM head GEMM (type 218, norm-once):
        # each worker writes one (bf16 max, int64 abs_idx). num_workers = 240.
        argmax_part_value = make_tensor("argmax_part_value", (bs, mpk.num_workers))
        argmax_part_index = make_tensor("argmax_part_index", (bs, mpk.num_workers), torch_dtype=torch.int64)
        argmax_out = mpk.attach_input(torch_tensor=output_tokens, name="output_token")
        if fuse_tail:
            argmax_in = make_tensor("argmax_in", (bs, vocab_size))

        # Split-K workspace for linear_with_residual on MI300
        # Must include done counter space: per XCD we need n_tiles_per_xcd ints
        # after the float32 workspace. Add 8 floats/XCD (32 bytes) for padding.
        is_rocm = bool(getattr(torch.version, "hip", None))
        if is_rocm:
            n_blocks = PADDED_HIDDEN_SIZE // 64
            # Workspace: 3072 data floats + 64 done counter floats (8 per XCD)
            splitk_ws_size = PADDED_HIDDEN_SIZE + 64  # 3136 = 392 * 8
            splitk_ws_torch = torch.zeros(
                (bs, splitk_ws_size), dtype=torch.float32, device="cuda")
            splitk_dc_torch = torch.zeros(
                (n_blocks, 1), dtype=torch.int32, device="cuda")
            splitk_workspace = mpk.attach_input(
                torch_tensor=splitk_ws_torch, name="splitk_workspace")
            splitk_done_counter = mpk.attach_input(
                torch_tensor=splitk_dc_torch, name="splitk_done_counter")

        # --- Prepare MoE weight tensors (MXFP4 packed for gang kernel) ---
        # W13: interleaved gate_up with OPW=64 (N-parallel: 4 waves x 16 rows)
        # W2: separate down weights, also OPW=64
        # SwiGLU is fused into W2's input quantization step (no separate task)
        w13_output_per_wg = 128  # W13: 48→24 tiles/XCD, max 1 tile/worker (no stragglers)
        w2_output_per_wg = 64   # W2: 24 tiles/XCD (OPW=128 regressed 7% even with prefetch)
        print(f"Packing MXFP4 MoE expert weights ({num_layers} layers, "
              f"W13_OPW={w13_output_per_wg}, W2_OPW={w2_output_per_wg})...")
        moe_gate_up_proj_weights = []  # [E, expert_wgs, wg_bytes] uint8
        moe_down_proj_weights = []     # [E, expert_wgs, wg_bytes] uint8
        moe_gate_up_proj_biases = []   # [E, 2*padded_inter] bf16
        moe_down_proj_biases = []      # [E, padded_hidden] bf16
        w13_target_num_blocks = PADDED_HIDDEN_SIZE // 32
        w2_target_num_blocks = PADDED_INTERMEDIATE_SIZE // 32
        for li in range(num_layers):
            layer = model.model.layers[li]
            experts = layer.mlp.experts

            # gate_up_proj: blocks [E, 2*inter, nb, 16], scales [E, 2*inter, nb]
            gu_packed = pack_mxfp4_workgroup(
                experts.gate_up_proj_blocks.to("cuda"),
                experts.gate_up_proj_scales.to("cuda"),
                output_per_wg=w13_output_per_wg,
                target_out_dim=2 * PADDED_INTERMEDIATE_SIZE,
                target_num_blocks=w13_target_num_blocks,
            )
            moe_gate_up_proj_weights.append(gu_packed)

            # down_proj: blocks [E, hidden, nb, 16], scales [E, hidden, nb]
            dp_packed = pack_mxfp4_workgroup(
                experts.down_proj_blocks.to("cuda"),
                experts.down_proj_scales.to("cuda"),
                output_per_wg=w2_output_per_wg,
                target_out_dim=PADDED_HIDDEN_SIZE,
                target_num_blocks=w2_target_num_blocks,
            )
            moe_down_proj_weights.append(dp_packed)

            # Biases: pad to padded dimensions (2D for gang kernel)
            gu_bias = experts.gate_up_proj_bias.data.to("cuda")  # [E, 2*inter]
            if gu_bias.shape[1] < 2 * PADDED_INTERMEDIATE_SIZE:
                gu_bias = torch.nn.functional.pad(
                    gu_bias, (0, 2 * PADDED_INTERMEDIATE_SIZE - gu_bias.shape[1]))
            moe_gate_up_proj_biases.append(gu_bias.contiguous())

            dp_bias = experts.down_proj_bias.data.to("cuda")  # [E, hidden]
            if dp_bias.shape[1] < PADDED_HIDDEN_SIZE:
                dp_bias = torch.nn.functional.pad(
                    dp_bias, (0, PADDED_HIDDEN_SIZE - dp_bias.shape[1]))
            moe_down_proj_biases.append(dp_bias.contiguous())

            # Free MXFP4 buffers for this layer to save memory
            # Keep them if --verify is set so PyTorch reference can still run
            if not args.verify:
                experts.gate_up_proj_blocks = None
                experts.gate_up_proj_scales = None
                experts.down_proj_blocks = None
                experts.down_proj_scales = None
            torch.cuda.synchronize()

        print(f"  Packed {num_layers} layers: gate_up {list(moe_gate_up_proj_weights[0].shape)}, "
              f"down {list(moe_down_proj_weights[0].shape)}")

        # --- Build task graph ---
        # Embed layer
        embed_weight = pad_weight_2d(
            model.model.embed_tokens.weight,
            target_cols=PADDED_HIDDEN_SIZE,
        )
        w = mpk.attach_input(torch_tensor=embed_weight, name="embed_tokens")
        mpk.embed_layer(
            input=x,
            weight=w,
            output=y,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
            input_source=1,
        )
        x = y

        # Keep references to ALL per-layer weight tensors to prevent PyTorch
        # from reusing GPU memory (causes pointer collisions in Mirage runtime).
        _layer_weight_refs = []

        def _attach_input_keep(torch_tensor, name):
            """attach_input + keep tensor alive to prevent pointer reuse."""
            _layer_weight_refs.append(torch_tensor)
            return mpk.attach_input(torch_tensor=torch_tensor, name=name)

        fused_tail_done = False
        for i in range(num_layers):
            layer = model.model.layers[i]
            # === Attention block ===
            # RMSNorm — pad weight; kernel uses actual_hidden_dim for RMS mean
            # (avoids bf16 rounding error from scale factor)
            norm_w_padded = pad_weight_1d(
                layer.input_layernorm.weight,
                PADDED_HIDDEN_SIZE, pad_value=0.0
            )
            w_norm = _attach_input_keep(
                norm_w_padded,
                f"layer_{i}_input_layernorm",
            )
            # NOTE: standalone rmsnorm_layer is fused into the QKV gang
            # linear below — every gang worker computes the same RMSNorm
            # prologue locally. Saves the dispatch barrier.

            # QKV projection (padded_hidden -> fused_qkv_dim) using MXFP4
            # Pad Q/K/V weight reduction dims and interleave by KV groups
            w_q = pad_weight_2d(
                layer.self_attn.q_proj.weight,
                target_cols=PADDED_HIDDEN_SIZE,
            )
            w_k = pad_weight_2d(
                layer.self_attn.k_proj.weight,
                target_cols=PADDED_HIDDEN_SIZE,
            )
            w_v = pad_weight_2d(
                layer.self_attn.v_proj.weight,
                target_cols=PADDED_HIDDEN_SIZE,
            )
            # Interleave Q/K/V by KV groups (same layout as shuffle_tensors)
            q_per_kv = num_local_q_heads // num_local_kv_heads
            qkv_chunks = []
            for g in range(num_local_kv_heads):
                qkv_chunks.append(w_q[g*q_per_kv*head_dim:(g+1)*q_per_kv*head_dim])
                qkv_chunks.append(w_k[g*head_dim:(g+1)*head_dim])
                qkv_chunks.append(w_v[g*head_dim:(g+1)*head_dim])
            w_qkv_shuffled = torch.cat(qkv_chunks, dim=0).contiguous()
            qkv_out_size = w_qkv_shuffled.shape[0]  # fused_qkv_dim
            # Quantize/pack weights for workgroup layout
            qkv_output_per_wg = 64  # 10 tiles/XCD fits in 30 workers, kvupd fusion needs OPW==head_dim
            use_bf16_native = os.environ.get("USE_FP16_ACT") == "1"
            if use_bf16_native:
                # BF16 native weights: store pre-dequanted BF16 in workgroup layout
                w_qkv_packed = pack_bf16_workgroup(w_qkv_shuffled, output_per_wg=qkv_output_per_wg)
                w_qkv_mxfp4 = _attach_input_keep(
                    w_qkv_packed, f"layer_{i}_qkv_bf16")
            else:
                qkv_blocks, qkv_scales = quantize_bf16_to_mxfp4(w_qkv_shuffled)
                w_qkv_packed = pack_mxfp4_workgroup(
                    qkv_blocks, qkv_scales, output_per_wg=qkv_output_per_wg,
                ).squeeze(0)  # [n_wgs, wg_bytes]
                w_qkv_mxfp4 = _attach_input_keep(
                    w_qkv_packed, f"layer_{i}_qkv_mxfp4")
            # QKV bias: shuffle Q/K/V biases to match interleaved weight layout
            q_bias = layer.self_attn.q_proj.bias.data.to("cuda")
            k_bias = layer.self_attn.k_proj.bias.data.to("cuda")
            v_bias = layer.self_attn.v_proj.bias.data.to("cuda")
            q_bias_grouped = q_bias.reshape(num_local_kv_heads, q_per_kv * head_dim)
            k_bias_grouped = k_bias.reshape(num_local_kv_heads, head_dim)
            v_bias_grouped = v_bias.reshape(num_local_kv_heads, head_dim)
            qkv_bias = torch.cat([q_bias_grouped, k_bias_grouped, v_bias_grouped], dim=1)
            qkv_bias = qkv_bias.reshape(1, -1).contiguous()  # [1, fused_qkv_dim]
            w_qkv_bias = _attach_input_keep(qkv_bias, f"layer_{i}_qkv_bias")

            # Attention
            # GPT-OSS doesn't have QK norm — pass None to disable it
            w_q_norm = None
            w_k_norm = None
            k_cache = _attach_input_keep(model.model.kv_cache[0][i], f"layer_{i}_k_cache")
            v_cache = _attach_input_keep(model.model.kv_cache[1][i], f"layer_{i}_v_cache")

            # Per-head attention sinks (GPT-OSS specific)
            w_sinks = _attach_input_keep(
                layer.self_attn.sinks.data.to("cuda"), f"layer_{i}_sinks"
            )

            # kv_stride: stride between consecutive positions in paged cache
            # k_cache layout: [num_pages, page_size, kv_heads, head_dim]
            kv_stride = num_local_kv_heads * head_dim

            if use_ck_fmha and args.split_kv_cache and qkv_output_per_wg >= head_dim:
                # Fused QKV + KV cache update: epilogue applies RoPE and
                # writes Q→q_workspace, K/V→paged caches directly.
                if world_size > 1:
                    mpk.gang_rmsnorm_linear_mxfp4_bias_kvupd_layer(
                        norm_input=x,
                        norm_weight=w_norm,
                        norm_output=rmsnorm_out,
                        mxfp4_weight=w_qkv_mxfp4,
                        bias=w_qkv_bias,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        q_workspace=ck_fmha_q_ws,
                        actual_hidden_dim=hidden_size,
                        output_per_wg=qkv_output_per_wg,
                        head_dim=head_dim,
                        num_q_per_kv=q_per_kv,
                        kv_stride=kv_stride,
                        q_ws_stride=q_ws_stride,
                        block_dim=(256, 1, 1),
                    )
                elif fuse_full_layer and world_size == 1:
                    # Full-layer fused: QKV+Attn+O-proj+TopK+MoE in one dispatch
                    # O-proj weight prep (moved up from below)
                    w_o = pad_weight_2d(
                        layer.self_attn.o_proj.weight,
                        target_rows=PADDED_HIDDEN_SIZE,
                    )
                    o_bias = pad_weight_1d(
                        layer.self_attn.o_proj.bias.data.to("cuda"),
                        PADDED_HIDDEN_SIZE
                    ).unsqueeze(0).contiguous()
                    w_o_bias = _attach_input_keep(o_bias, f"layer_{i}_o_bias")
                    o_output_per_wg = 16
                    o_blocks, o_scales = quantize_bf16_to_mxfp4(w_o)
                    w_o_packed = pack_mxfp4_workgroup(
                        o_blocks, o_scales, output_per_wg=o_output_per_wg,
                    ).squeeze(0)
                    w_o_mxfp4 = _attach_input_keep(
                        w_o_packed, f"layer_{i}_o_proj_mxfp4")
                    # MoE weight prep
                    post_norm_w_padded = pad_weight_1d(
                        layer.post_attention_layernorm.weight,
                        PADDED_HIDDEN_SIZE, pad_value=0.0
                    )
                    w_norm_moe = _attach_input_keep(
                        post_norm_w_padded, f"layer_{i}_post_attn_layernorm",
                    )
                    w_moe_gate = pad_weight_2d(
                        layer.mlp.router.weight,
                        target_cols=PADDED_HIDDEN_SIZE,
                    )
                    w_moe_gate_t = _attach_input_keep(w_moe_gate, f"layer_{i}_moe_gate")
                    router_bias_t = layer.mlp.router.bias.data.to("cuda").unsqueeze(0).contiguous()
                    w_router_bias = _attach_input_keep(router_bias_t, f"layer_{i}_router_bias")
                    w_gatedup = _attach_input_keep(
                        moe_gate_up_proj_weights[i], f"layer_{i}_gate_up_proj"
                    )
                    w13_bias = _attach_input_keep(
                        moe_gate_up_proj_biases[i], f"layer_{i}_gate_up_bias"
                    )
                    w_down_proj = _attach_input_keep(
                        moe_down_proj_weights[i], f"layer_{i}_down_proj"
                    )
                    w2_bias = _attach_input_keep(
                        moe_down_proj_biases[i], f"layer_{i}_down_bias"
                    )
                    if fuse_tail and i == num_layers - 1:
                        # Last layer: fuse tail (resadd + LM head + argmax) into type 217
                        final_norm_w_padded = pad_weight_1d(
                            model.model.norm.weight,
                            PADDED_HIDDEN_SIZE, pad_value=0.0
                        )
                        w_lm_norm = _attach_input_keep(
                            final_norm_w_padded, "model_norm_weight")
                        w_lm_proj_mxfp4 = _attach_input_keep(
                            lm_head_packed, "lm_head_mxfp4")
                        lm_head_zero_bias = torch.zeros(
                            1, vocab_size, dtype=torch.bfloat16, device="cuda")
                        w_lm_bias = _attach_input_keep(
                            lm_head_zero_bias, "lm_head_bias")
                        mpk.gang_full_layer_with_lmhead_fused_layer(
                            # QKV+Attn inputs
                            workspace_f32=moe_workspace_f32,
                            residual=x,
                            norm_weight_pre=w_norm,
                            norm_scratch_pre=rmsnorm_out,
                            qkv_weight=w_qkv_mxfp4,
                            qkv_bias=w_qkv_bias,
                            sinks=w_sinks,
                            qkv_barrier=qkv_attn_barrier,
                            lse_acc=ck_fmha_lse_acc,
                            # O-proj+TopK inputs
                            oproj_weight=w_o_mxfp4,
                            oproj_bias=w_o_bias,
                            norm_weight_post=w_norm_moe,
                            norm_scratch_post=rmsnorm_out_moe,
                            router_weight=w_moe_gate_t,
                            router_bias=w_router_bias,
                            logits_scratch=moe_gate_out,
                            oproj_counters=oproj_topk_counters,
                            # MoE inputs
                            gate_up_weight=w_gatedup,
                            down_weight=w_down_proj,
                            w13_bias=w13_bias,
                            w2_bias=w2_bias,
                            moe_barrier=moe_fused_barrier,
                            swiglu_out=swiglu_out,
                            o_acc_f32=ck_fmha_o_acc,
                            # LM head inputs (4 extra)
                            lm_norm_weight=w_lm_norm,
                            lm_norm_scratch=rmsnorm_out,
                            lm_mxfp4_weight=w_lm_proj_mxfp4,
                            lm_bias=w_lm_bias,
                            # QKV+Attn outputs
                            x_output=mlp_weighted_sum_out,
                            k_cache=k_cache,
                            v_cache=v_cache,
                            q_workspace=ck_fmha_q_ws,
                            o_acc=attn_out,
                            # O-proj+TopK+MoE outputs
                            attn_proj_out=attn_proj_out,
                            topk_weight=moe_topk_weight,
                            routing_indices=moe_routing_indices,
                            active_expert_ids=moe_mask,
                            routing_weight_moe=moe_topk_weight,
                            moe_workspace_f32=moe_workspace_f32,
                            # LM head outputs (2 extra)
                            lm_logits=argmax_in,
                            argmax_output=argmax_out,
                            # Parameters
                            actual_hidden_dim=hidden_size,
                            qkv_output_per_wg=qkv_output_per_wg,
                            oproj_output_per_wg=o_output_per_wg,
                            head_dim=head_dim,
                            num_q_per_kv=q_per_kv,
                            kv_stride=kv_stride,
                            q_ws_stride=q_ws_stride,
                            num_kv_chunks=ck_fmha_num_kv_chunks,
                            num_kv_heads=num_local_kv_heads,
                            num_experts=num_experts,
                            topk_k=num_experts_per_tok,
                            lm_output_per_wg=lm_head_output_per_wg,
                            lm_output_stride=vocab_size,
                            sliding_window=per_layer_sliding_window[i],
                            w13_output_per_wg=w13_output_per_wg,
                            w2_output_per_wg=w2_output_per_wg,
                            block_dim=(256, 1, 1),
                        )
                        fused_tail_done = True
                    else:
                        mpk.gang_full_layer_fused_layer(
                            # QKV+Attn inputs
                            workspace_f32=moe_workspace_f32,
                            residual=x,
                            norm_weight_pre=w_norm,
                            norm_scratch_pre=rmsnorm_out,
                            qkv_weight=w_qkv_mxfp4,
                            qkv_bias=w_qkv_bias,
                            sinks=w_sinks,
                            qkv_barrier=qkv_attn_barrier,
                            lse_acc=ck_fmha_lse_acc,
                            # O-proj+TopK inputs
                            oproj_weight=w_o_mxfp4,
                            oproj_bias=w_o_bias,
                            norm_weight_post=w_norm_moe,
                            norm_scratch_post=rmsnorm_out_moe,
                            router_weight=w_moe_gate_t,
                            router_bias=w_router_bias,
                            logits_scratch=moe_gate_out,
                            oproj_counters=oproj_topk_counters,
                            # MoE inputs
                            gate_up_weight=w_gatedup,
                            down_weight=w_down_proj,
                            w13_bias=w13_bias,
                            w2_bias=w2_bias,
                            moe_barrier=moe_fused_barrier,
                            swiglu_out=swiglu_out,
                            o_acc_f32=ck_fmha_o_acc,
                            # QKV+Attn outputs
                            x_output=mlp_weighted_sum_out,
                            k_cache=k_cache,
                            v_cache=v_cache,
                            q_workspace=ck_fmha_q_ws,
                            o_acc=attn_out,
                            # O-proj+TopK+MoE outputs
                            attn_proj_out=attn_proj_out,
                            topk_weight=moe_topk_weight,
                            routing_indices=moe_routing_indices,
                            active_expert_ids=moe_mask,
                            routing_weight_moe=moe_topk_weight,
                            moe_workspace_f32=moe_workspace_f32,
                            # Parameters
                            actual_hidden_dim=hidden_size,
                            qkv_output_per_wg=qkv_output_per_wg,
                            oproj_output_per_wg=o_output_per_wg,
                            head_dim=head_dim,
                            num_q_per_kv=q_per_kv,
                            kv_stride=kv_stride,
                            q_ws_stride=q_ws_stride,
                            num_kv_chunks=ck_fmha_num_kv_chunks,
                            num_kv_heads=num_local_kv_heads,
                            num_experts=num_experts,
                            topk_k=num_experts_per_tok,
                            sliding_window=per_layer_sliding_window[i],
                            w13_output_per_wg=w13_output_per_wg,
                            w2_output_per_wg=w2_output_per_wg,
                            block_dim=(256, 1, 1),
                        )
                    x = attn_proj_out
                    # Last layer needs explicit residual add (f32→bf16)
                    if i == num_layers - 1 and not fused_tail_done:
                        mpk.moe_residual_add_f32_layer(
                            workspace_f32=moe_workspace_f32,
                            residual=x,
                            output=mlp_weighted_sum_out,
                            grid_dim=(1, 1, 1),
                            block_dim=(256, 1, 1),
                        )
                        x = mlp_weighted_sum_out
                    # Skip O-proj/TopK/MoE tasks below — already fused
                    continue
                elif fuse_qkv_attn:
                    # Fused QKV + Attention: single gang task with internal barrier
                    mpk.gang_qkv_attn_fused_layer(
                        workspace_f32=moe_workspace_f32,
                        residual=x,
                        x_output=mlp_weighted_sum_out,
                        norm_weight=w_norm,
                        norm_scratch=rmsnorm_out,
                        mxfp4_weight=w_qkv_mxfp4,
                        bias=w_qkv_bias,
                        sinks=w_sinks,
                        barrier=qkv_attn_barrier,
                        lse_acc=ck_fmha_lse_acc,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        q_workspace=ck_fmha_q_ws,
                        o_acc=attn_out,
                        actual_hidden_dim=hidden_size,
                        output_per_wg=qkv_output_per_wg,
                        head_dim=head_dim,
                        num_q_per_kv=q_per_kv,
                        kv_stride=kv_stride,
                        q_ws_stride=q_ws_stride,
                        num_kv_chunks=ck_fmha_num_kv_chunks,
                        num_kv_heads=num_local_kv_heads,
                        sliding_window=per_layer_sliding_window[i],
                        block_dim=(256, 1, 1),
                    )
                    x = mlp_weighted_sum_out
                else:
                    mpk.gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_layer(
                        workspace_f32=moe_workspace_f32,
                        residual=x,
                        x_output=mlp_weighted_sum_out,
                        norm_weight=w_norm,
                        norm_scratch=rmsnorm_out,
                        mxfp4_weight=w_qkv_mxfp4,
                        bias=w_qkv_bias,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        q_workspace=ck_fmha_q_ws,
                        actual_hidden_dim=hidden_size,
                        output_per_wg=qkv_output_per_wg,
                        head_dim=head_dim,
                        num_q_per_kv=q_per_kv,
                        kv_stride=kv_stride,
                        q_ws_stride=q_ws_stride,
                        block_dim=(256, 1, 1),
                    )
                    x = mlp_weighted_sum_out

                if not fuse_qkv_attn or i == 0 or world_size > 1:
                    # Separate attention task (layer 0 or unfused path).
                    # For chunks>1, write float partials to ck_fmha_o_acc and
                    # apply sinks in the merge step (decode kernel only fuses
                    # sinks in the chunks==1 branch).
                    attn_o_target = ck_fmha_o_acc if use_split_attn_chunks else attn_out
                    attn_sinks_arg = None if use_split_attn_chunks else w_sinks
                    mpk.paged_attention_ck_fmha_layer(
                        q_workspace=ck_fmha_q_ws,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        o_acc=attn_o_target,
                        lse_acc=ck_fmha_lse_acc,
                        attention_params=(num_local_q_heads,
                                         ck_fmha_num_kv_chunks,
                                         mpk.max_num_batched_requests),
                        grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, ck_fmha_num_kv_chunks),
                        block_dim=(256, 1, 1),
                        sinks=attn_sinks_arg,
                        sliding_window=per_layer_sliding_window[i],
                    )
                    if use_split_attn_chunks:
                        mpk.paged_attention_ck_fmha_merge_layer(
                            lse=ck_fmha_lse_acc,
                            output_tmp=ck_fmha_o_acc,
                            output=attn_out,
                            attention_params=(num_local_q_heads, head_dim,
                                             ck_fmha_num_kv_chunks,
                                             num_local_kv_heads),
                            grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
                            block_dim=(256, 1, 1),
                            sinks=w_sinks,
                        )
            else:
                # Non-fused path: QKV writes to attn_in, paged_attention does KV update
                if i == 0 or world_size > 1:
                    mpk.gang_rmsnorm_linear_mxfp4_bias_layer(
                        norm_input=x,
                        norm_weight=w_norm,
                        norm_output=rmsnorm_out,
                        mxfp4_weight=w_qkv_mxfp4,
                        bias=w_qkv_bias,
                        output=attn_in,
                        actual_hidden_dim=hidden_size,
                        output_per_wg=qkv_output_per_wg,
                        output_stride=qkv_out_size,
                        block_dim=(256, 1, 1),
                    )
                else:
                    mpk.gang_resaddf32_rmsnorm_linear_mxfp4_bias_layer(
                        workspace_f32=moe_workspace_f32,
                        residual=x,
                        x_output=mlp_weighted_sum_out,
                        norm_weight=w_norm,
                        norm_scratch=rmsnorm_out,
                        mxfp4_weight=w_qkv_mxfp4,
                        bias=w_qkv_bias,
                        qkv_output=attn_in,
                        actual_hidden_dim=hidden_size,
                        output_per_wg=qkv_output_per_wg,
                        output_stride=qkv_out_size,
                        block_dim=(256, 1, 1),
                    )
                    x = mlp_weighted_sum_out
                # KV cache update: RoPE + write K/V to paged cache + Q to workspace
                mpk.kv_cache_update_layer(
                    input=attn_in,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    q_norm=w_q_norm,
                    k_norm=w_k_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    q_workspace=ck_fmha_q_ws,
                    grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
                    block_dim=(256, 1, 1),
                )
                # CK FMHA attention
                mpk.paged_attention_ck_fmha_layer(
                    q_workspace=ck_fmha_q_ws,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    o_acc=attn_out,
                    lse_acc=ck_fmha_lse_acc,
                    attention_params=(num_local_q_heads,
                                     ck_fmha_num_kv_chunks,
                                     mpk.max_num_batched_requests),
                    grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, ck_fmha_num_kv_chunks),
                    block_dim=(256, 1, 1),
                    sinks=w_sinks,
                    sliding_window=per_layer_sliding_window[i],
                )

            # O projection + residual (attn_out -> padded_hidden + residual)
            w_o = pad_weight_2d(
                layer.self_attn.o_proj.weight,
                target_rows=PADDED_HIDDEN_SIZE,
            )
            # O-proj bias: pad from hidden_size to PADDED_HIDDEN_SIZE
            o_bias = pad_weight_1d(
                layer.self_attn.o_proj.bias.data.to("cuda"),
                PADDED_HIDDEN_SIZE
            ).unsqueeze(0).contiguous()  # [1, PADDED_HIDDEN_SIZE]
            w_o_bias = _attach_input_keep(o_bias, f"layer_{i}_o_bias")

            if is_rocm:
                # Quantize/pack O-proj weight
                o_output_per_wg = 16
                if use_bf16_native:
                    w_o_packed = pack_bf16_workgroup(w_o, output_per_wg=o_output_per_wg)
                    w_o_mxfp4 = _attach_input_keep(
                        w_o_packed, f"layer_{i}_o_proj_bf16")
                else:
                    o_blocks, o_scales = quantize_bf16_to_mxfp4(w_o)
                    w_o_packed = pack_mxfp4_workgroup(
                        o_blocks, o_scales, output_per_wg=o_output_per_wg,
                    ).squeeze(0)  # [n_wgs, wg_bytes]
                    w_o_mxfp4 = _attach_input_keep(
                        w_o_packed, f"layer_{i}_o_proj_mxfp4")

            # === MoE block weight prep (needed before fused path) ===
            post_norm_w_padded = pad_weight_1d(
                layer.post_attention_layernorm.weight,
                PADDED_HIDDEN_SIZE, pad_value=0.0
            )
            w_norm_moe = _attach_input_keep(
                post_norm_w_padded, f"layer_{i}_post_attn_layernorm",
            )
            w_moe_gate = pad_weight_2d(
                layer.mlp.router.weight,
                target_cols=PADDED_HIDDEN_SIZE,
            )
            w_moe_gate_t = _attach_input_keep(w_moe_gate, f"layer_{i}_moe_gate")
            router_bias = layer.mlp.router.bias.data.to("cuda").unsqueeze(0).contiguous()  # [1, 128]
            w_router_bias = _attach_input_keep(router_bias, f"layer_{i}_router_bias")

            if is_rocm and fuse_oproj_moe and world_size == 1:
                # Fused O-PROJ + TopK + MoE in a single gang task (type 215)
                w_gatedup = _attach_input_keep(
                    moe_gate_up_proj_weights[i], f"layer_{i}_gate_up_proj"
                )
                w13_bias = _attach_input_keep(
                    moe_gate_up_proj_biases[i], f"layer_{i}_gate_up_bias"
                )
                w_down_proj = _attach_input_keep(
                    moe_down_proj_weights[i], f"layer_{i}_down_proj"
                )
                w2_bias = _attach_input_keep(
                    moe_down_proj_biases[i], f"layer_{i}_down_bias"
                )
                mpk.gang_oproj_topk_moe_fused_layer(
                    # O-PROJ inputs
                    input=attn_out,
                    oproj_weight=w_o_mxfp4,
                    residual=x,
                    oproj_bias=w_o_bias,
                    norm_weight=w_norm_moe,
                    norm_output=rmsnorm_out_moe,
                    router_weight=w_moe_gate_t,
                    router_bias=w_router_bias,
                    logits_scratch=moe_gate_out,
                    counters=oproj_topk_counters,
                    # MoE inputs
                    gate_up_weight=w_gatedup,
                    down_weight=w_down_proj,
                    w13_bias=w13_bias,
                    w2_bias=w2_bias,
                    moe_barrier=moe_fused_barrier,
                    swiglu_out=swiglu_out,
                    # Outputs
                    oproj_output=attn_proj_out,
                    topk_weight=moe_topk_weight,
                    routing_indices=moe_routing_indices,
                    active_expert_ids=moe_mask,
                    routing_weight_moe=moe_topk_weight,
                    workspace_f32=moe_workspace_f32,
                    # Parameters
                    output_per_wg=o_output_per_wg,
                    output_stride=PADDED_HIDDEN_SIZE,
                    actual_hidden_dim=hidden_size,
                    num_experts=num_experts,
                    topk_k=num_experts_per_tok,
                    w13_output_per_wg=w13_output_per_wg,
                    w2_output_per_wg=w2_output_per_wg,
                    block_dim=(256, 1, 1),
                )
                x = attn_proj_out
            elif is_rocm and fuse_oproj_topk and world_size == 1:
                # Fused O-PROJ + RMSNorm + Router + TopK in single gang task
                mpk.gang_linear_mxfp4_res_bias_rmsnorm_topk_layer(
                    input=attn_out,
                    mxfp4_weight=w_o_mxfp4,
                    residual=x,
                    oproj_bias=w_o_bias,
                    norm_weight=w_norm_moe,
                    norm_output=rmsnorm_out_moe,
                    router_weight=w_moe_gate_t,
                    router_bias=w_router_bias,
                    logits_scratch=moe_gate_out,
                    counters=oproj_topk_counters,
                    output=attn_proj_out,
                    topk_weight=moe_topk_weight,
                    routing_indices=moe_routing_indices,
                    active_expert_ids=moe_mask,
                    output_per_wg=o_output_per_wg,
                    output_stride=PADDED_HIDDEN_SIZE,
                    actual_hidden_dim=hidden_size,
                    num_experts=num_experts,
                    topk_k=num_experts_per_tok,
                    block_dim=(256, 1, 1),
                )
                x = attn_proj_out
            else:
                if is_rocm:
                    mpk.gang_linear_mxfp4_res_bias_layer(
                        input=attn_out,
                        mxfp4_weight=w_o_mxfp4,
                        residual=x,
                        bias=w_o_bias,
                        output=attn_proj_out,
                        output_per_wg=o_output_per_wg,
                        output_stride=PADDED_HIDDEN_SIZE,
                        block_dim=(256, 1, 1),
                    )
                else:
                    w_o_t = _attach_input_keep(w_o, f"layer_{i}_o_proj")
                    mpk.gang_linear_with_residual_layer(
                        input=attn_out,
                        weight=w_o_t,
                        residual=x,
                        output=attn_proj_out,
                        tile_n=64,
                        output_stride=PADDED_HIDDEN_SIZE,
                        block_dim=(256, 1, 1),
                    )
                # DEBUG: verify O-proj bias padding
                if i == 0 and args.verify:
                    print(f"[O-PROJ BIAS CHECK] shape: {o_bias.shape}")
                    print(f"  bias[2878:2882]: {o_bias[0, 2878:2882].float().tolist()}")
                    print(f"  bias pad [2880:] all zero: {(o_bias[0, 2880:].abs().max().item() == 0)}")
                x = attn_proj_out

                if world_size > 1:
                    mpk.allreduce_layer(
                        input=attn_proj_out,
                        buffer=allreduce_buf,
                        output=attn_allreduce_out,
                        grid_dim=(PADDED_HIDDEN_SIZE // 64, 1, 1),
                        block_dim=(128, 1, 1),
                    )
                    x = attn_allreduce_out

                if world_size == 1:
                    # Fused RMSNorm + Router linear + TopK softmax
                    mpk.gang_rmsnorm_linear_bias_topk_layer(
                        norm_input=x,
                        norm_weight=w_norm_moe,
                        norm_output=rmsnorm_out_moe,
                        linear_weight=w_moe_gate_t,
                        bias=w_router_bias,
                        logits_scratch=moe_gate_out,
                        gang_counter=router_topk_counter,
                        topk_weight=moe_topk_weight,
                        routing_indices=moe_routing_indices,
                        active_expert_ids=moe_mask,
                        actual_hidden_dim=hidden_size,
                        tile_n=1,
                        output_stride=num_experts,
                        num_experts_per_tok=num_experts_per_tok,
                        block_dim=(256, 1, 1),
                    )
                else:
                    # Multi-GPU: keep separate tasks (allreduce between them)
                    mpk.gang_rmsnorm_linear_bias_layer(
                        norm_input=x,
                        norm_weight=w_norm_moe,
                        norm_output=rmsnorm_out_moe,
                        linear_weight=w_moe_gate_t,
                        bias=w_router_bias,
                        output=moe_gate_out,
                        actual_hidden_dim=hidden_size,
                        tile_n=16,
                        output_stride=num_experts,
                        block_dim=(256, 1, 1),
                    )
                    mpk.moe_topk_softmax_routing_layer(
                        input=moe_gate_out,
                        output=(moe_topk_weight, moe_routing_indices, moe_mask),
                        grid_dim=(1, 1, 1),
                        block_dim=(256, 1, 1),
                    )

            if not (is_rocm and fuse_oproj_moe and world_size == 1):
                # Fused W13+SwiGLU+W2 gang MXFP4 (single task, per-expert barrier)
                # Phase-ordered: all W13 tiles before all W2 tiles.
                # Eliminates scheduler gap between W13→W2 events.
                # (Skipped when fuse_oproj_moe: MoE is inside the fused task.)
                w_gatedup = _attach_input_keep(
                    moe_gate_up_proj_weights[i], f"layer_{i}_gate_up_proj"
                )
                w13_bias = _attach_input_keep(
                    moe_gate_up_proj_biases[i], f"layer_{i}_gate_up_bias"
                )
                w_down_proj = _attach_input_keep(
                    moe_down_proj_weights[i], f"layer_{i}_down_proj"
                )
                w2_bias = _attach_input_keep(
                    moe_down_proj_biases[i], f"layer_{i}_down_bias"
                )
                mpk.gang_moe_fused_mxfp4_layer(
                    input=rmsnorm_out_moe,
                    gate_up_weight=w_gatedup,
                    down_weight=w_down_proj,
                    moe_routing_indices=moe_routing_indices,
                    moe_mask=moe_mask,
                    w13_bias=w13_bias,
                    w2_bias=w2_bias,
                    routing_weight=moe_topk_weight,
                    swiglu_out=swiglu_out,
                    workspace_f32=moe_workspace_f32,
                    barrier=moe_fused_barrier,
                    w13_output_per_wg=w13_output_per_wg,
                    w2_output_per_wg=w2_output_per_wg,
                    block_dim=(256, 1, 1),
                )

            # MoE residual add: W2 epilogue already did routing_weight*result
            # → atomicAdd to moe_workspace_f32. Just add residual + zero workspace.
            # For single-GPU layers 0..(n-2), defer into next layer's QKV prologue
            # (resaddf32 variant). Last layer + multi-GPU run standalone.
            if i == num_layers - 1 or world_size > 1:
                mpk.moe_residual_add_f32_layer(
                    workspace_f32=moe_workspace_f32,
                    residual=x,
                    output=mlp_weighted_sum_out,
                    grid_dim=(1, 1, 1),
                    block_dim=(256, 1, 1),
                )
                x = mlp_weighted_sum_out

            if world_size > 1:
                mpk.allreduce_layer(
                    input=mlp_weighted_sum_out,
                    buffer=allreduce_buf,
                    output=mlp_final,
                    grid_dim=(PADDED_HIDDEN_SIZE // 64, 1, 1),
                    block_dim=(256, 1, 1),
                )
                x = mlp_final

        if not fused_tail_done:
            # Final RMSNorm + LM head (fused MXFP4: FP4 weights × FP8 activations)
            final_norm_w_padded = pad_weight_1d(
                model.model.norm.weight,
                PADDED_HIDDEN_SIZE, pad_value=0.0
            )
            w_norm = mpk.attach_input(
                torch_tensor=final_norm_w_padded, name="model_norm_weight"
            )
            w_proj_mxfp4 = mpk.attach_input(
                torch_tensor=lm_head_packed, name="lm_head_mxfp4")
            lm_head_zero_bias = torch.zeros(
                1, vocab_size, dtype=torch.bfloat16, device="cuda")
            w_lm_bias = mpk.attach_input(
                torch_tensor=lm_head_zero_bias, name="lm_head_bias")
            # Fused LM head GEMM + argmax (type 218): each tile writes
            # per-tile (max_val, rel_idx) instead of logits to HBM.
            # Eliminates 393KB logits write + 393KB logits read.
            mpk.gang_rmsnorm_linear_mxfp4_bias_argmax_layer(
                norm_input=x,
                norm_weight=w_norm,
                norm_output=rmsnorm_out,
                mxfp4_weight=w_proj_mxfp4,
                bias=w_lm_bias,
                argmax_part_value=argmax_part_value,
                argmax_part_index=argmax_part_index,
                actual_hidden_dim=hidden_size,
                output_per_wg=lm_head_output_per_wg,
                output_stride=vocab_size,
                block_dim=(256, 1, 1),
            )
            mpk.argmax_reduce_layer(
                input=(argmax_part_value, argmax_part_index),
                output=argmax_out,
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )

        # Generate task graph and compile
        num_ops = len(mpk.kn_graph.cygraph.get_graph_structure())
        print(f"DEBUG: kn_graph has {num_ops} operators before generate_task_graph")
        results = mpk.kn_graph.generate_task_graph(num_gpus=world_size, my_gpu_id=rank)
        with open(f"task_graph_{rank}.json", "w") as f:
            f.write(results["json_file"])
        with open(f"kernel_{rank}.cu", "w") as f:
            f.write(results["cuda_code"])

        mpk.compile(output_dir=args.output_dir)

        # Set RoPE cos/sin pointers in RuntimeConfig (used by fused QKV+KV_UPD tasks)
        if args.split_kv_cache:
            print(f"[DEBUG] Setting RoPE tables: cos_padded ptr=0x{cos_padded.data_ptr():x} shape={cos_padded.shape}, sin_padded ptr=0x{sin_padded.data_ptr():x} shape={sin_padded.shape}")
            mpk.set_rope_tables(cos_padded, sin_padded)

    # --- Execution loop ---
    stream = torch.cuda.Stream()
    warmup = 0
    output_len = args.max_new_tokens if args.max_new_tokens is not None else (
        tokens.size(1) - prompt_lengths[0].item()
    )
    output_len = max(0, min(output_len, tokens.size(1) - prompt_lengths[0].item()))

    if not args.use_mirage:
        prompt_len = prompt_lengths[0].item()
        decode_limit = prompt_len + output_len
        for cur_pos in range(prompt_len, decode_limit):
            step.fill_(cur_pos - 1)
            input_ids = tokens[:, prev_pos:cur_pos]
            cos_embeddings = position_embeddings[0][:, prev_pos:cur_pos]
            sin_embeddings = position_embeddings[1][:, prev_pos:cur_pos]
            if args.use_triton:
                decode_step = cur_pos - prompt_len
                _triton_profile_step[0] = decode_step
                _layer_profile_step[0] = decode_step
            if args.use_aiter:
                _aiter_profile_step[0] = cur_pos - prompt_len
            logits = model.forward(
                input_ids=input_ids,
                position_embeddings=(cos_embeddings, sin_embeddings),
                step=step,
            )
            next_token = logits.argmax(dim=-1)
            next_token = next_token[0, -1]
            tokens[0, cur_pos] = next_token
            prev_pos = cur_pos
            if next_token == config.eos_token_id and not args.ignore_eos:
                break
            if cur_pos == prompt_len + warmup:
                torch.cuda.synchronize()
                starter.record()

        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        end_idx = prev_pos + 1
        generated_ids = tokens[:, :end_idx]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        print(
            "Prompt length {}, generate length {}, per-token latency {} ms".format(
                prompt_len, cur_pos - prompt_len, run_time / max(1, cur_pos - prompt_len)
            )
        )
        if save_path and rank == 0:
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            out = {
                "token_ids": tokens[0, prompt_len:slice_end].tolist(),
                "text": tokenizer.decode(tokens[0, :end_idx], skip_special_tokens=True),
                "generate_length": max(0, end_idx - prompt_len),
                "mode": "torch",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")
        if args.use_triton and _layer_profile_times and _layer_profile_times['attn']:
            print("\n=== Layer profiling (step 5, 36 layers) ===")
            for comp in ['attn', 'moe', 'other']:
                times = _layer_profile_times[comp]
                if times:
                    total = sum(times)
                    avg = total / len(times)
                    print(f"  {comp}: total={total:.2f}ms, avg/layer={avg:.3f}ms")
            grand_total = sum(sum(v) for v in _layer_profile_times.values())
            print(f"  Grand total (36 layers): {grand_total:.2f}ms")
        if args.use_triton and _triton_profile_times:
            print("\n=== MoE detail (step 5, per-layer) ===")
            for comp, times in sorted(_triton_profile_times.items()):
                total = sum(times)
                avg = total / len(times) if times else 0
                print(f"  {comp}: total={total:.2f}ms, avg={avg:.3f}ms")
            total_step = sum(sum(v) for v in _triton_profile_times.values())
            print(f"  MoE breakdown total: {total_step:.2f}ms")
        if args.use_aiter and _aiter_profile_times and _aiter_profile_times['attn']:
            print("\n=== AITER Layer profiling (step 5, 36 layers) ===")
            for comp in ['attn', 'moe', 'other']:
                times = _aiter_profile_times[comp]
                if times:
                    total = sum(times)
                    avg = total / len(times)
                    print(f"  {comp}: total={total:.2f}ms, avg/layer={avg:.3f}ms")
            grand_total = sum(sum(v) for v in _aiter_profile_times.values())
            print(f"  Grand total (36 layers): {grand_total:.2f}ms")
    else:
        # Capture device printf (which writes to fd 1, bypassing sys.stdout)
        # so we can parse [FWD_PASS] iter=N time_ms=X lines and split
        # prefill vs decode totals after mpk() returns.
        import os, sys, tempfile, re
        sys.stdout.flush()
        sys.stderr.flush()
        _fwd_pass_log = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".fwdlog", delete=False
        )
        _fwd_pass_log_path = _fwd_pass_log.name
        _saved_stdout_fd = os.dup(1)
        os.dup2(_fwd_pass_log.fileno(), 1)

        starter.record()
        mpk()
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        # Save profiler tensor if profiling was enabled
        if profiler_tensor is not None:
            _prof_path = "profile_output.pt"
            torch.save(profiler_tensor.cpu(), _prof_path)
            print(f"Profiler tensor saved to {_prof_path}")

        # Restore stdout, then re-emit captured text so the user still sees it.
        sys.stdout.flush()
        os.dup2(_saved_stdout_fd, 1)
        os.close(_saved_stdout_fd)
        _fwd_pass_log.flush()
        _fwd_pass_log.seek(0)
        _captured = _fwd_pass_log.read()
        _fwd_pass_log.close()
        # Print non-FWD_PASS lines only (FWD_PASS still parsed for summary)
        for _line in _captured.splitlines(keepends=True):
            if "[FWD_PASS]" not in _line:
                sys.stdout.write(_line)
        sys.stdout.flush()

        # Parse [FWD_PASS] iter=N time_ms=X lines into a dict (dedupe by iter
        # in case multiple schedulers print the same iteration).
        _fwd_times = {}
        for _m in re.finditer(
            r"\[FWD_PASS\] iter=(\d+) time_ms=([\d.]+)", _captured
        ):
            _fwd_times[int(_m.group(1))] = float(_m.group(2))

        # The device-side per-iter ring holds FWDPASS_LOG_MAX (8192) samples.
        # Longer runs drop the tail, and since per-iter latency grows with
        # sequence length, averaging only what survived understates the real
        # number. [FWD_PASS_TOTAL] is accumulated over every iteration, so
        # prefer it whenever samples were dropped.
        _fwd_dropped = 0
        _fwd_total_avg = None
        _fwd_total_iters = 0
        _m_tot = re.search(
            r"\[FWD_PASS_TOTAL\] iters=(\d+) total_ms=[\d.]+ "
            r"avg_ms=([\d.]+) dropped=(\d+)",
            _captured,
        )
        if _m_tot:
            _fwd_total_iters = int(_m_tot.group(1))
            _fwd_total_avg = float(_m_tot.group(2))
            _fwd_dropped = int(_m_tot.group(3))

        #print("tokens.shape = ", tokens.shape, flush=True)
        #print("All tokens:", tokens[0].tolist())
        #print("Step:", step.tolist())
        for r in range(total_num_requests):
            generated_ids = tokens[r, : step[r] + 1]
            valid_ids = generated_ids[generated_ids >= 0]
            # Debug: print first few generated tokens
            prompt_len_r = prompt_lengths[r].item()
            gen_ids = valid_ids[prompt_len_r:prompt_len_r+10]
            #print(f"First 10 generated token IDs: {gen_ids.tolist()}")
            #for tid in gen_ids.tolist():
            #    print(f"  {tid} -> '{tokenizer.decode([tid])}'")
            response = tokenizer.decode(valid_ids, skip_special_tokens=True)
            print(response)

        if save_path and rank == 0:
            gen0 = tokens[0, : step[0].item() + 1]
            gen0 = gen0[gen0 >= 0]
            pl0 = prompt_lengths[0].item()
            end0 = gen0.numel()
            slice_end = min(end0, pl0 + MAX_SAVE_TOKENS)
            out = {
                "token_ids": gen0[pl0:slice_end].tolist(),
                "text": tokenizer.decode(gen0[:end0], skip_special_tokens=True),
                "generate_length": max(0, end0 - pl0),
                "mode": "mpk",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

        prompt_len = prompt_lengths[0].item()
        total_tokens = step.max().item() + 1
        generated_tokens = total_tokens - prompt_len

        prefill_iterations = math.ceil(prompt_len / args.max_num_batched_tokens)
        decode_iterations = generated_tokens
        total_iterations = prefill_iterations + decode_iterations

        if total_iterations > 0:
            avg_time_per_iter = run_time / total_iterations
        else:
            avg_time_per_iter = 0

        # Per-iter timings come from the device-side scheduler clock and
        # cover only the steady-state forward pass (no setup/JIT). Sum
        # them separately for prefill and decode.
        # Iter numbering: iter=1 is the first prepare step (no compute);
        # FWD_PASS for iter=N reports the time elapsed BETWEEN
        # END_OF_TASK_GRAPH N-1 and N. So real per-iter times are
        # iter=2..total_iterations+1, mapped to logical iter 1..total.
        _prefill_total = 0.0
        _prefill_count = 0
        _decode_total = 0.0
        _decode_count = 0
        for _it, _t in _fwd_times.items():
            # Map: kernel iter=2 corresponds to the FIRST forward pass
            # (logical iter 1). Skip iter=1 entry if present (it has no
            # prev clock, so it's actually filtered out in the kernel).
            _logical = _it - 1
            if _logical < 1:
                continue
            if _logical <= prefill_iterations:
                _prefill_total += _t
                _prefill_count += 1
            elif _logical <= total_iterations:
                _decode_total += _t
                _decode_count += 1

        print("=" * 80)
        print(f"[Wall-time average — same number printed three ways]")
        print(f"  avg_per_iter (run_time / {total_iterations}): "
              f"{avg_time_per_iter:.3f} ms")
        print(f"  Combined: {total_tokens} tokens, per-token latency: "
              f"{run_time / total_tokens:.3f} ms")
        print("-" * 80)
        print(f"[Steady-state per-iter (device clock, prefill vs decode split)]")
        if _prefill_count > 0:
            _pp_avg = _prefill_total / _prefill_count
            print(f"  Prefill: {prompt_len} tokens in {_prefill_count}/{prefill_iterations} "
                  f"iters ~= {_prefill_total:.1f}ms total "
                  f"(avg {_pp_avg:.3f}ms/iter)")
        else:
            print(f"  Prefill: no FWD_PASS samples captured "
                  f"(expected {prefill_iterations} iters)")
        if _decode_count > 0:
            _dd_avg = _decode_total / _decode_count
            print(f"  Decode:  {generated_tokens} tokens in {_decode_count}/{decode_iterations} "
                  f"iters ~= {_decode_total:.1f}ms total "
                  f"(avg {_dd_avg:.3f}ms/iter)")
        else:
            print(f"  Decode:  no FWD_PASS samples captured "
                  f"(expected {decode_iterations} iters)")
        # Min/max for decode to show how much per-iter ramps with seq len.
        if _decode_count > 0:
            _decode_samples = [_t for _it, _t in _fwd_times.items()
                               if _it - 1 > prefill_iterations
                               and _it - 1 <= total_iterations]
            print(f"  Decode per-iter range: min={min(_decode_samples):.3f}ms "
                  f"max={max(_decode_samples):.3f}ms")
        if _fwd_dropped > 0:
            print("-" * 80)
            print(f"  NOTE: device per-iter ring overflowed — {_fwd_dropped} of "
                  f"{_fwd_total_iters} samples dropped. The prefill/decode "
                  f"splits above cover only the first "
                  f"{_fwd_total_iters - _fwd_dropped} iterations and understate "
                  f"latency (per-iter grows with seq len).")
            print(f"  All-iteration device average: {_fwd_total_avg:.3f}ms/iter "
                  f"over {_fwd_total_iters} iters")
        print("=" * 80)

        # === Verification: compare Mirage intermediates with PyTorch reference ===
        if args.verify and verify_tensors:
            print("\n" + "=" * 80)
            print("VERIFICATION: Comparing Mirage intermediates with PyTorch reference")
            print("=" * 80)
            torch.cuda.synchronize()

            # Debug: check q_workspace and k_cache values
            q_ws_nz = (ck_fmha_q_ws_tensor.abs() > 1e-6).sum().item()
            print(f"  [DEBUG] q_workspace non-zero: {q_ws_nz} / {ck_fmha_q_ws_tensor.numel()}")
            print(f"  [DEBUG] q_workspace[:8]: {ck_fmha_q_ws_tensor[0, :8].float().tolist()}")
            k_cache_t = model.model.kv_cache[0][0]
            v_cache_t = model.model.kv_cache[1][0]
            k_nz = (k_cache_t[:, 0, :].abs() > 1e-6).sum().item()
            print(f"  [DEBUG] k_cache[page0,:,head0,:] non-zero: {k_nz} / {k_cache_t[:, 0, :].numel()}")
            print(f"  [DEBUG] k_cache[0,0,0,:8]: {k_cache_t[0, 0, :8].float().tolist()}")

            # Run PyTorch for the first decode step
            # The first decode step processes token at position = prompt_len - 1
            # (the last prompt token, generating the first output token)
            prompt_len_v = prompt_lengths[0].item()
            # Reset step for PyTorch path
            step_pt = torch.full((1,), prompt_len_v - 1, dtype=torch.int32, device="cuda")
            input_id = tokens[0, prompt_len_v - 1].unsqueeze(0).unsqueeze(0)  # [1, 1]
            cos_emb = position_embeddings[0][:, prompt_len_v - 1:prompt_len_v]
            sin_emb = position_embeddings[1][:, prompt_len_v - 1:prompt_len_v]

            # Step through PyTorch manually to get intermediates
            # Use the LAST layer for comparison since verify tensors capture the last layer's data
            last_layer_idx = num_layers - 1
            layer = model.model.layers[last_layer_idx]

            # 1. Embedding
            pt_embed = model.model.embed_tokens(input_id)  # [1, 1, 2880]
            pt_embed_padded = torch.zeros(1, PADDED_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
            pt_embed_padded[0, :hidden_size] = pt_embed[0, 0]

            # Compare embedding
            mg_embed = verify_tensors["embed_out"]
            print(f"\n--- Embedding ---")
            print(f"  PT embed[:8]: {pt_embed_padded[0, :8].float().tolist()}")
            print(f"  MG embed[:8]: {mg_embed[0, :8].float().tolist()}")
            diff = (pt_embed_padded - mg_embed).abs().max().item()
            print(f"  Max abs diff: {diff:.6f}")

            # 2. RMSNorm (pre-attention) — compute from embedding (known good)
            # NOTE: rmsnorm_out verify tensor is aliased (shared with final RMSNorm),
            # so we compute the expected pre-attention RMSNorm from embedding here.
            x_pad = pt_embed_padded.float()  # [1, 3072]
            sum_sq = (x_pad ** 2).sum(dim=-1, keepdim=True)
            rms_rcp = torch.rsqrt(sum_sq / hidden_size + 1e-5)  # eps=1e-5
            norm_w = pad_weight_1d(
                layer.input_layernorm.weight,
                PADDED_HIDDEN_SIZE, pad_value=0.0
            )
            pt_rmsnorm_padded = (x_pad * rms_rcp * norm_w.float()).bfloat16()
            print(f"\n--- RMSNorm (pre-attention, computed from embedding) ---")
            print(f"  PT rmsnorm[:8]: {pt_rmsnorm_padded[0, :8].float().tolist()}")

            # Also compute using PyTorch's RMSNorm module for reference
            pt_norm_in = pt_embed_padded[:, :hidden_size].unsqueeze(0)  # [1, 1, 2880]
            pt_norm_out = layer.input_layernorm(pt_norm_in)  # [1, 1, 2880]
            print(f"  PT module rmsnorm[:8]: {pt_norm_out[0, 0, :8].float().tolist()}")

            # 3. QKV projection — compute from PT RMSNorm
            mg_attn_in = verify_tensors.get("attn_in")
            # Compute expected QKV using padded RMSNorm result
            w_q = pad_weight_2d(layer.self_attn.q_proj.weight, target_cols=PADDED_HIDDEN_SIZE)
            w_k = pad_weight_2d(layer.self_attn.k_proj.weight, target_cols=PADDED_HIDDEN_SIZE)
            w_v = pad_weight_2d(layer.self_attn.v_proj.weight, target_cols=PADDED_HIDDEN_SIZE)
            # Mirage shuffles QKV into interleaved [kv_head_groups] format
            # For GPT-OSS: 64 Q heads, 8 KV heads, head_dim=64
            # Shuffled: [Q0..Q7, K0, V0, Q8..Q15, K1, V1, ...] per group of 8 Q + 1 K + 1 V
            q_per_kv = num_q_heads // num_kv_heads  # 8
            qkv_chunks = []
            for g in range(num_kv_heads):
                qkv_chunks.append(w_q[g*q_per_kv*head_dim:(g+1)*q_per_kv*head_dim])
                qkv_chunks.append(w_k[g*head_dim:(g+1)*head_dim])
                qkv_chunks.append(w_v[g*head_dim:(g+1)*head_dim])
            w_qkv_shuffled = torch.cat(qkv_chunks, dim=0)  # [fused_qkv_dim, 3072]

            pt_qkv = (pt_rmsnorm_padded.float() @ w_qkv_shuffled.float().T).bfloat16()  # [1, fused_qkv_dim]
            # Add QKV bias (also shuffled)
            b_q = layer.self_attn.q_proj.bias.data.to("cuda")
            b_k = layer.self_attn.k_proj.bias.data.to("cuda")
            b_v = layer.self_attn.v_proj.bias.data.to("cuda")
            bias_chunks = []
            for g in range(num_kv_heads):
                bias_chunks.append(b_q[g*q_per_kv*head_dim:(g+1)*q_per_kv*head_dim])
                bias_chunks.append(b_k[g*head_dim:(g+1)*head_dim])
                bias_chunks.append(b_v[g*head_dim:(g+1)*head_dim])
            b_qkv_shuffled = torch.cat(bias_chunks, dim=0)  # [fused_qkv_dim]
            pt_qkv = pt_qkv + b_qkv_shuffled.unsqueeze(0)

            print(f"\n--- QKV Projection ---")
            print(f"  PT qkv[:8]:  {pt_qkv[0, :8].float().tolist()}")
            if mg_attn_in is not None:
                print(f"  MG attn_in[:8]: {mg_attn_in[0, :8].float().tolist()}")
                diff = (pt_qkv[0] - mg_attn_in[0]).abs().max().item()
                print(f"  Max abs diff: {diff:.6f}")
                # Find where largest diffs are
                diffs = (pt_qkv[0] - mg_attn_in[0]).abs().float()
                top_diff_vals, top_diff_idx = diffs.topk(5)
                print(f"  Top-5 diff indices: {top_diff_idx.tolist()}")
                print(f"  Top-5 diff values: {top_diff_vals.tolist()}")
            else:
                print(f"  MG attn_in: NOT CAPTURED")

            # 3b. Manual attention from MG attn_in (which matches PT perfectly)
            mg_attn_out = verify_tensors.get("attn_out")
            mg_attn_in_v = verify_tensors.get("attn_in")
            if mg_attn_out is not None and mg_attn_in_v is not None:
                from models.modeling_gpt_oss import apply_rotary_pos_emb_neox
                # Parse QKV from interleaved layout
                qkv = mg_attn_in_v[0]  # [5120]
                q_per_kv = num_q_heads // num_kv_heads  # 8
                q_heads = []
                k_heads_list = []
                v_heads_list = []
                for g in range(num_kv_heads):
                    base = g * (q_per_kv + 2) * head_dim
                    for h in range(q_per_kv):
                        q_heads.append(qkv[base + h * head_dim : base + (h+1) * head_dim])
                    k_heads_list.append(qkv[base + q_per_kv * head_dim : base + (q_per_kv+1) * head_dim])
                    v_heads_list.append(qkv[base + (q_per_kv+1) * head_dim : base + (q_per_kv+2) * head_dim])
                q_all = torch.stack(q_heads)  # [64, 64]
                k_new = torch.stack(k_heads_list)  # [8, 64]
                v_new = torch.stack(v_heads_list)  # [8, 64]

                # Apply NeoX RoPE at position = (prompt_len - 1) for first decode
                pos = prompt_len_v - 1  # position of last prompt token
                half = head_dim // 2
                c = cos_emb[0, 0, :half].float()  # cos for this position
                s = sin_emb[0, 0, :half].float()   # sin for this position

                q1, q2 = q_all[:, :half].float(), q_all[:, half:].float()
                q_rot = torch.cat([q1 * c - q2 * s, q1 * s + q2 * c], dim=-1).bfloat16()
                k1, k2 = k_new[:, :half].float(), k_new[:, half:].float()
                k_rot = torch.cat([k1 * c - k2 * s, k1 * s + k2 * c], dim=-1).bfloat16()

                # Sequential prefill: build KV cache from scratch
                # For each prior token, compute QKV and apply RoPE to K
                kv_k_manual = torch.zeros(prompt_len_v, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
                kv_v_manual = torch.zeros(prompt_len_v, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")

                for t_idx in range(prompt_len_v):
                    t_emb = model.model.embed_tokens(tokens[:1, t_idx:t_idx+1])  # [1,1,2880]
                    t_normed = layer.input_layernorm(t_emb)  # [1,1,2880]
                    t_q = layer.self_attn.q_proj(t_normed[0,0])  # [4096]
                    t_k = layer.self_attn.k_proj(t_normed[0,0])  # [512]
                    t_v = layer.self_attn.v_proj(t_normed[0,0])  # [512]
                    # Reshape K, V
                    t_k = t_k.view(num_kv_heads, head_dim)
                    t_v = t_v.view(num_kv_heads, head_dim)
                    # Apply NeoX RoPE to K at position t_idx
                    ct = position_embeddings[0][0, t_idx, :half].float()
                    st = position_embeddings[1][0, t_idx, :half].float()
                    tk1, tk2 = t_k[:, :half].float(), t_k[:, half:].float()
                    t_k_rot = torch.cat([tk1 * ct - tk2 * st, tk1 * st + tk2 * ct], dim=-1).bfloat16()
                    kv_k_manual[t_idx] = t_k_rot
                    kv_v_manual[t_idx] = t_v

                # Now compute attention: Q_rot * K_cache^T, softmax, V_cache
                scale = 1.0 / (head_dim ** 0.5)
                manual_attn_out = torch.zeros(num_q_heads * head_dim, dtype=torch.bfloat16, device="cuda")

                sinks_data = layer.self_attn.sinks.data.to("cuda")  # [num_q_heads]

                for kv_h in range(num_kv_heads):
                    k_cache_h = kv_k_manual[:prompt_len_v, kv_h].float()  # [seq_len, head_dim]
                    v_cache_h = kv_v_manual[:prompt_len_v, kv_h].float()
                    for qh_local in range(q_per_kv):
                        qh = kv_h * q_per_kv + qh_local
                        q_h = q_rot[qh].float()
                        scores = (q_h @ k_cache_h.T) * scale
                        scores_max = scores.max()
                        scores_exp = torch.exp(scores - scores_max)
                        scores_sum = scores_exp.sum()
                        attn_w = scores_exp / scores_sum
                        out_h = attn_w @ v_cache_h
                        # Apply sink correction
                        lse = torch.log(scores_sum) + scores_max
                        sink_val = sinks_data[qh].float()
                        correction = torch.sigmoid(lse - sink_val)
                        out_h_with_sink = out_h * correction
                        # Store
                        manual_attn_out[qh * head_dim : (qh+1) * head_dim] = out_h_with_sink.bfloat16()

                print(f"\n--- Manual Attention (from MG attn_in, with sinks) ---")
                print(f"  Manual attn[:8]: {manual_attn_out[:8].float().tolist()}")
                print(f"  MG attn_out[:8]: {mg_attn_out[0, :8].float().tolist()}")
                diff_manual = (manual_attn_out - mg_attn_out[0]).abs().max().item()
                print(f"  Max abs diff: {diff_manual:.6f}")
                if diff_manual > 0.01:
                    diffs = (manual_attn_out - mg_attn_out[0]).abs().float()
                    top_vals, top_idx = diffs.topk(5)
                    print(f"  Top-5 diff indices: {top_idx.tolist()}")
                    print(f"  Top-5 diff values: {top_vals.tolist()}")
                    # Check if it's sink-related: compute without sinks
                    manual_no_sink = torch.zeros(num_q_heads * head_dim, dtype=torch.bfloat16, device="cuda")
                    for kv_h in range(num_kv_heads):
                        k_cache_h = kv_k_manual[:prompt_len_v, kv_h].float()
                        v_cache_h = kv_v_manual[:prompt_len_v, kv_h].float()
                        for qh_local in range(q_per_kv):
                            qh = kv_h * q_per_kv + qh_local
                            q_h = q_rot[qh].float()
                            scores = (q_h @ k_cache_h.T) * scale
                            attn_w = torch.softmax(scores, dim=-1)
                            out_h = attn_w @ v_cache_h
                            manual_no_sink[qh * head_dim : (qh+1) * head_dim] = out_h.bfloat16()
                    diff_no_sink = (manual_no_sink - mg_attn_out[0]).abs().max().item()
                    print(f"\n  WITHOUT sinks:")
                    print(f"  Manual no-sink[:8]: {manual_no_sink[:8].float().tolist()}")
                    print(f"  Max abs diff (no sink vs MG): {diff_no_sink:.6f}")
                    # Check per KV-group
                    for g in range(min(4, num_kv_heads)):
                        start = g * q_per_kv * head_dim
                        end = (g+1) * q_per_kv * head_dim
                        g_diff_sink = (manual_attn_out[start:end] - mg_attn_out[0, start:end]).abs().max().item()
                        g_diff_nosink = (manual_no_sink[start:end] - mg_attn_out[0, start:end]).abs().max().item()
                        print(f"  KV group {g}: with_sink_diff={g_diff_sink:.4f} no_sink_diff={g_diff_nosink:.4f}")

            # 4. Attention output — compute PT attention using sequential KV cache
            if mg_attn_out is not None:
                # Run PT layer's attention for proper comparison
                # We need to use the model's own KV cache to match sequential prefill
                from models.modeling_gpt_oss import naive_attention_with_sinks, apply_rotary_pos_emb_neox
                # Compute attention from embedding through the full PT attention path
                # Save and restore KV cache
                kv_k_tmp = model.model.kv_cache[0].clone()
                kv_v_tmp = model.model.kv_cache[1].clone()
                model.model.kv_cache[0].zero_()
                model.model.kv_cache[1].zero_()
                with torch.inference_mode():
                    prev_p = 0
                    step_tmp = torch.tensor([0], device="cuda")
                    # Run sequential prefill through layer 0's attention only
                    for cur_p in range(1, prompt_len_v + 1):
                        step_tmp.fill_(cur_p - 1)
                        in_ids = tokens[:1, prev_p:cur_p]
                        c_emb = position_embeddings[0][:, prev_p:cur_p]
                        s_emb = position_embeddings[1][:, prev_p:cur_p]
                        # Compute through embedding + layernorm + attention
                        emb = model.model.embed_tokens(in_ids)
                        normed = layer.input_layernorm(emb)
                        attn_out_pt = layer.self_attn(normed, position_embeddings=(c_emb, s_emb), step=step_tmp)
                        prev_p = cur_p
                # attn_out_pt is the last step's attention output [1, 1, 2880]
                # Pad to compare with Mirage [1, 4096]
                pt_attn_padded = torch.zeros(num_q_heads * head_dim, dtype=torch.bfloat16, device="cuda")
                # The PT attn output is AFTER o_proj, so we can't directly compare with MG attn_out
                # MG attn_out is BEFORE o_proj (the raw attention output)
                # Let's compare the o_proj+residual result instead
                pt_oproj_res = attn_out_pt[0, 0] + pt_embed[0, 0]  # [2880]
                model.model.kv_cache[0].copy_(kv_k_tmp)
                model.model.kv_cache[1].copy_(kv_v_tmp)
                del kv_k_tmp, kv_v_tmp
                print(f"\n--- PT Attention (sequential prefill, with sinks + RoPE) ---")
                pt_oproj_padded = torch.zeros(PADDED_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
                pt_oproj_padded[:hidden_size] = pt_oproj_res
                mg_aproj = verify_tensors.get("attn_proj_out")
                if mg_aproj is not None:
                    print(f"  PT o_proj+res[:8]: {pt_oproj_padded[:8].float().tolist()}")
                    print(f"  MG attn_proj[:8]: {mg_aproj[0, :8].float().tolist()}")
                    diff = (pt_oproj_padded - mg_aproj[0]).abs().max().item()
                    print(f"  Max abs diff (o_proj+res): {diff:.6f}")
                    if diff > 0.01:
                        diffs = (pt_oproj_padded - mg_aproj[0]).abs().float()
                        top_diff_vals, top_diff_idx = diffs.topk(5)
                        print(f"  Top-5 diff indices: {top_diff_idx.tolist()}")
                        print(f"  Top-5 diff values: {top_diff_vals.tolist()}")
                nonzero = (mg_attn_out.abs() > 1e-6).sum().item()
                print(f"  MG attn_out non-zero: {nonzero} / {mg_attn_out.numel()}")

            # 4b. O-proj + residual (manual computation from attn_out)
            mg_attn_proj = verify_tensors.get("attn_proj_out")
            if mg_attn_out is not None and mg_attn_proj is not None:
                # O-proj: attn_out [1, 4096] @ o_proj.weight.T [4096, 2880] + o_proj.bias
                w_o = layer.self_attn.o_proj.weight.to("cuda")  # [2880, 4096]
                b_o = layer.self_attn.o_proj.bias.data.to("cuda")  # [2880]
                # Step 1: Just the matmul (no bias, no residual)
                pt_o_gemm = (mg_attn_out[0].float() @ w_o.float().T).bfloat16()  # [2880]
                # Also use padded weight (what Mirage actually uses)
                w_o_padded = pad_weight_2d(layer.self_attn.o_proj.weight, target_rows=PADDED_HIDDEN_SIZE)
                pt_o_gemm_padded = (mg_attn_out[0].float() @ w_o_padded.float().T).bfloat16()  # [3072]
                # Subtract bias and residual from MG result to isolate GEMM
                o_bias_padded = pad_weight_1d(b_o, PADDED_HIDDEN_SIZE)
                mg_embed_out = verify_tensors.get("embed_out")
                mg_gemm_only = mg_attn_proj[0].float() - o_bias_padded.float()
                if mg_embed_out is not None:
                    mg_gemm_only = mg_gemm_only - mg_embed_out[0].float()

                print(f"\n--- O-proj GEMM only (no bias/residual) ---")
                print(f"  PT gemm[:8]: {pt_o_gemm_padded[:8].float().tolist()}")
                print(f"  MG gemm[:8]: {mg_gemm_only[:8].tolist()}")
                gemm_diff = (pt_o_gemm_padded.float() - mg_gemm_only).abs()
                print(f"  Max GEMM diff: {gemm_diff.max().item():.6f}")
                print(f"  Mean GEMM diff: {gemm_diff[:hidden_size].mean().item():.6f}")
                if gemm_diff.max().item() > 0.1:
                    top_vals, top_idx = gemm_diff.topk(5)
                    print(f"  Top-5 diff indices: {top_idx.tolist()}")
                    print(f"  Top-5 diff values: {top_vals.tolist()}")
                    # Check per-XCD (384 cols each)
                    for xcd in range(8):
                        s, e = xcd * 384, (xcd + 1) * 384
                        xcd_diff = gemm_diff[s:e].max().item()
                        print(f"  XCD {xcd} [{s}:{e}]: max_diff={xcd_diff:.4f}")

                # Step 2: Full o_proj + residual + bias
                pt_o_proj = pt_o_gemm + b_o
                pt_o_with_res = pt_o_proj + pt_embed[0, 0]  # [2880]
                pt_o_padded = torch.zeros(PADDED_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
                pt_o_padded[:hidden_size] = pt_o_with_res

                print(f"\n--- O-proj + residual + bias ---")
                print(f"  PT o_proj+res[:8]: {pt_o_padded[:8].float().tolist()}")
                print(f"  MG attn_proj[:8]: {mg_attn_proj[0, :8].float().tolist()}")
                diff = (pt_o_padded - mg_attn_proj[0]).abs().max().item()
                print(f"  Max abs diff: {diff:.6f}")
                nonzero = (mg_attn_proj.abs() > 1e-6).sum().item()
                print(f"  Non-zero elements: {nonzero} / {mg_attn_proj.numel()}")
                pad_max = mg_attn_proj[0, hidden_size:].abs().max().item()
                print(f"  PAD REGION [2880:3072] max: {pad_max:.6f}")
                if diff > 0.1:
                    diffs = (pt_o_padded - mg_attn_proj[0]).abs().float()
                    top_diff_vals, top_diff_idx = diffs.topk(5)
                    print(f"  Top-5 diff indices: {top_diff_idx.tolist()}")
                    print(f"  Top-5 diff values: {top_diff_vals.tolist()}")

            # 6. Post-attention RMSNorm — compute PT from O-proj+residual output
            mg_rmsnorm_moe = verify_tensors.get("rmsnorm_out_moe")
            if mg_attn_proj is not None and mg_rmsnorm_moe is not None:
                # Compute post-attn RMSNorm from Mirage's attn_proj_out (verified above)
                x_f = mg_attn_proj[0].float()
                sum_sq = (x_f ** 2).sum()
                rms_rcp = torch.rsqrt(sum_sq / hidden_size + 1e-5)
                post_norm_w = pad_weight_1d(
                    layer.post_attention_layernorm.weight,
                    PADDED_HIDDEN_SIZE, pad_value=0.0
                )
                pt_rmsnorm_moe = (x_f * rms_rcp * post_norm_w.float()).bfloat16()
                print(f"\n--- Post-attention RMSNorm ---")
                print(f"  PT rmsnorm_moe[:8]: {pt_rmsnorm_moe[:8].float().tolist()}")
                print(f"  MG rmsnorm_moe[:8]: {mg_rmsnorm_moe[0, :8].float().tolist()}")
                diff = (pt_rmsnorm_moe - mg_rmsnorm_moe[0]).abs().max().item()
                print(f"  Max abs diff: {diff:.6f}")

                # 7. MoE gate — compute from PT RMSNorm
                w_gate = pad_weight_2d(layer.mlp.router.weight, target_cols=PADDED_HIDDEN_SIZE)
                b_gate = layer.mlp.router.bias.data.to("cuda")
                pt_gate = (pt_rmsnorm_moe.float() @ w_gate.float().T).bfloat16() + b_gate
                pt_gate_from_mg = (mg_rmsnorm_moe[0].float() @ w_gate.float().T).bfloat16() + b_gate
                print(f"\n--- MoE Gate Logits ---")
                pt_top_vals, pt_top_idx = pt_gate.float().topk(4)
                print(f"  PT top-4 experts: {pt_top_idx.tolist()}")
                print(f"  PT top-4 logits: {pt_top_vals.tolist()}")
                mg_top_vals, mg_top_idx = pt_gate_from_mg.float().topk(4)
                print(f"  PT(from MG rmsnorm) top-4 experts: {mg_top_idx.tolist()}")
                print(f"  PT(from MG rmsnorm) top-4 logits: {mg_top_vals.tolist()}")
                # Gate stats from PT computation
                g = pt_gate.float()
                print(f"  PT gate stats: min={g.min().item():.4f} max={g.max().item():.4f} mean={g.mean().item():.4f}")
                # Note: moe_gate_out verify tensor is zeroed by routing kernel (see comment above)
            else:
                if mg_rmsnorm_moe is not None:
                    print(f"\n--- Post-attention RMSNorm ---")
                    print(f"  MG rmsnorm_moe[:8]: {mg_rmsnorm_moe[0, :8].float().tolist()}")
                print(f"\n--- MoE Gate Logits ---")
                mg_gate = verify_tensors.get("moe_gate_out")
                if mg_gate is not None:
                    g = mg_gate[0].float()
                    top_vals, top_idx = g.topk(4)
                    print(f"  Top-4 experts: {top_idx.tolist()}")
                    print(f"  Top-4 logits: {top_vals.tolist()}")
                    print(f"  (NOTE: gate verify tensor is zeroed by routing kernel)")

            # 8. MoE routing
            mg_routing = verify_tensors.get("moe_routing_indices")
            mg_weights = verify_tensors.get("moe_topk_weight")
            mg_mask = verify_tensors.get("moe_mask")
            if mg_routing is not None and mg_weights is not None:
                print(f"\n--- MoE Routing ---")
                print(f"  Routing weights: {mg_weights[0].tolist()}")
                # Find which experts are active (routing_indices != 0)
                active = (mg_routing[:, 0] != 0).nonzero().squeeze(-1)
                print(f"  Active experts: {active.tolist()}")
                for e in active.tolist():
                    slot = mg_routing[e, 0].item()
                    print(f"    Expert {e} -> slot {slot - 1} (raw={slot})")
            if mg_mask is not None:
                count = mg_mask[num_experts].item()
                print(f"  moe_mask count (mask[{num_experts}]): {count}")
                print(f"  moe_mask compact list [0:{count}]: {mg_mask[:count].tolist()}")
                print(f"  moe_mask full [0:10]: {mg_mask[:10].tolist()}")
                print(f"  moe_mask[-5:]: {mg_mask[-5:].tolist()}")

            # 8b. Check W13 input (rmsnorm_out_moe is the input to gate AND W13)
            if mg_rmsnorm_moe is not None:
                nz = (mg_rmsnorm_moe[0].abs() > 1e-6).sum().item()
                print(f"\n--- W13 Input Check ---")
                print(f"  rmsnorm_out_moe nonzero: {nz} / {mg_rmsnorm_moe.shape[-1]}")
                print(f"  rmsnorm_out_moe norm: {mg_rmsnorm_moe[0].float().norm().item():.4f}")

            # 9. MoE outputs

            # 9c. W2 output (mlp_out)
            mg_mlp_out = verify_tensors.get("mlp_out")
            if mg_mlp_out is not None:
                print(f"\n--- W2 (down_proj) Output ---")
                for k in range(num_experts_per_tok):
                    vals = mg_mlp_out[0, k, :8].float().tolist()
                    nonzero_k = (mg_mlp_out[0, k].abs() > 1e-6).sum().item()
                    print(f"  Slot {k}[:8]: {vals}  (nonzero: {nonzero_k}/{mg_mlp_out.shape[-1]})")

            # 9d. MoE workspace_f32 (atomicAdd accumulator used in fused path)
            mg_ws_f32 = verify_tensors.get("moe_workspace_f32")
            if mg_ws_f32 is not None:
                print(f"\n--- MoE workspace_f32 (atomicAdd output) ---")
                ws_vals = mg_ws_f32[0, :8].float().tolist()
                ws_nonzero = (mg_ws_f32[0].abs() > 1e-6).sum().item()
                ws_norm = mg_ws_f32[0, :hidden_size].float().norm().item()
                print(f"  ws_f32[:8]: {ws_vals}")
                print(f"  nonzero: {ws_nonzero}/{mg_ws_f32.shape[-1]}, norm: {ws_norm:.4f}")
                print(f"  pad region [{hidden_size}:] max: {mg_ws_f32[0, hidden_size:].abs().max().item():.6f}")

                # Note: mlp_mid (W13 output) is also available for debugging if needed

            # 10. Full PyTorch MoE computation from dequantized MXFP4 weights
            mg_final = verify_tensors.get("mlp_weighted_sum_out")
            if mg_attn_proj is not None and mg_rmsnorm_moe is not None and mg_final is not None:
                mg_routing = verify_tensors.get("moe_routing_indices")
                mg_weights_t = verify_tensors.get("moe_topk_weight")
                active = (mg_routing[:, 0] != 0).nonzero().squeeze(-1)

                pt_moe_input = mg_rmsnorm_moe[0]  # [3072] padded
                pt_moe_out = torch.zeros(PADDED_HIDDEN_SIZE, dtype=torch.float32, device="cuda")
                layer = model.model.layers[last_layer_idx]
                experts_ref = layer.mlp.experts

                for e_idx in active.tolist():
                    slot = mg_routing[e_idx, 0].item() - 1
                    weight = mg_weights_t[0, slot].item()
                    # W13: dequant gate_up from blocks/scales
                    gu_blk = experts_ref.gate_up_proj_blocks[e_idx:e_idx+1].to("cuda")
                    gu_sc = experts_ref.gate_up_proj_scales[e_idx:e_idx+1].to("cuda")
                    gate_up_w = dequant_mxfp4_to_bf16(gu_blk, gu_sc,
                        target_out_dim=2*PADDED_INTERMEDIATE_SIZE,
                        target_reduction=PADDED_HIDDEN_SIZE)[0]
                    gate_up = (pt_moe_input.float() @ gate_up_w.float().T).bfloat16()
                    gu_bias = moe_combined_biases[last_layer_idx][e_idx, :2*PADDED_INTERMEDIATE_SIZE]
                    gate_up = gate_up + gu_bias
                    # SwigluOAI
                    activated = swigluoai(gate_up.unsqueeze(0)).squeeze(0)
                    # W2: dequant down from blocks/scales
                    dp_blk = experts_ref.down_proj_blocks[e_idx:e_idx+1].to("cuda")
                    dp_sc = experts_ref.down_proj_scales[e_idx:e_idx+1].to("cuda")
                    down_w = dequant_mxfp4_to_bf16(dp_blk, dp_sc,
                        target_out_dim=PADDED_HIDDEN_SIZE,
                        target_reduction=PADDED_INTERMEDIATE_SIZE)[0]
                    down_out = (activated.float() @ down_w.float().T).bfloat16()
                    dp_bias = moe_combined_biases[last_layer_idx][e_idx, 2*PADDED_INTERMEDIATE_SIZE:]
                    down_out = down_out + dp_bias
                    pt_moe_out += down_out.float() * weight
                    del gate_up_w, down_w, gu_blk, gu_sc, dp_blk, dp_sc

                pt_layer_out = pt_moe_out.bfloat16() + mg_attn_proj[0]  # residual

                print(f"\n--- MoE Weighted Sum + Residual (layer output) ---")
                print(f"  PT layer_out[:8]: {pt_layer_out[:8].float().tolist()}")
                print(f"  MG final[:8]: {mg_final[0, :8].float().tolist()}")
                diff = (pt_layer_out - mg_final[0]).abs().max().item()
                print(f"  Max abs diff: {diff:.6f}")
                if diff > 0.01:
                    diffs = (pt_layer_out - mg_final[0]).abs().float()
                    top_diff_vals, top_diff_idx = diffs.topk(5)
                    print(f"  Top-5 diff indices: {top_diff_idx.tolist()}")
                    print(f"  Top-5 diff values: {top_diff_vals.tolist()}")
                nonzero = (mg_final.abs() > 1e-6).sum().item()
                print(f"  Non-zero elements: {nonzero} / {mg_final.numel()}")
                # What token would PT MoE produce?
                x_f = pt_layer_out.float()
                sum_sq = (x_f ** 2).sum()
                rms_rcp = torch.rsqrt(sum_sq / hidden_size + 1e-5)
                final_norm_w = model.model.norm.weight.to("cuda")
                norm_w_pad_v = torch.zeros(PADDED_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
                norm_w_pad_v[:hidden_size] = final_norm_w
                final_normed = (x_f * rms_rcp * norm_w_pad_v.float()).bfloat16()
                logits_pt_moe = (final_normed.float() @ lm_head_weight.float().T).bfloat16()
                print(f"  PT MoE argmax: {logits_pt_moe.argmax().item()} ({tokenizer.decode([logits_pt_moe.argmax().item()])!r})")
                print(f"  Mirage argmax: {tokens[0, prompt_lengths[0].item()].item()} ({tokenizer.decode([tokens[0, prompt_lengths[0].item()].item()])!r})")

            # 11. Final RMSNorm + LM head + Argmax comparison
            # Use Mirage's final hidden state to compute logits in PyTorch
            mg_final_hs = verify_tensors.get("mlp_weighted_sum_out")
            if mg_final_hs is not None:
                print(f"\n--- Final RMSNorm + LM Head (PyTorch from Mirage hidden) ---")
                # Final RMSNorm (matching what Mirage does)
                x_f = mg_final_hs[0].float()
                sum_sq = (x_f ** 2).sum()
                rms_rcp = torch.rsqrt(sum_sq / hidden_size + 1e-5)
                final_norm_w = model.model.norm.weight.to("cuda")
                norm_w_pad_v = torch.zeros(PADDED_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
                norm_w_pad_v[:hidden_size] = final_norm_w
                final_normed = (x_f * rms_rcp * norm_w_pad_v.float()).bfloat16()
                print(f"  PT Final normed[:8]: {final_normed[:8].float().tolist()}")
                # Compare with Mirage's rmsnorm_out (shared tensor used for final norm)
                mg_rmsnorm = verify_tensors.get("rmsnorm_out")
                if mg_rmsnorm is not None:
                    print(f"  MG rmsnorm_out[:8]: {mg_rmsnorm[0, :8].float().tolist()}")
                    rnorm_diff = (final_normed - mg_rmsnorm[0]).abs().max().item()
                    print(f"  Final RMSNorm max abs diff: {rnorm_diff:.6f}")
                    if rnorm_diff > 0.01:
                        diffs = (final_normed - mg_rmsnorm[0]).abs().float()
                        top_diff_vals, top_diff_idx = diffs.topk(5)
                        print(f"  Top-5 diff indices: {top_diff_idx.tolist()}")
                        print(f"  Top-5 diff values: {top_diff_vals.tolist()}")
                        # Check if the hidden states are identical
                        print(f"  MG hidden[:8]: {mg_final_hs[0, :8].float().tolist()}")
                        # Check padded region
                        print(f"  MG rmsnorm pad region max: {mg_rmsnorm[0, hidden_size:].abs().max().item():.6f}")
                # LM head
                logits_pt = (final_normed.float() @ lm_head_weight.float().T).bfloat16()
                top_vals, top_idx = logits_pt.float().topk(5)
                print(f"  PT-computed top-5 from Mirage hidden:")
                for v, i in zip(top_vals.tolist(), top_idx.tolist()):
                    print(f"    {i} ({tokenizer.decode([i])!r}): {v:.4f}")
                print(f"  PT-computed argmax: {logits_pt.argmax().item()} ({tokenizer.decode([logits_pt.argmax().item()])!r})")
                print(f"  Mirage produced: {tokens[0, prompt_lengths[0].item()].item()} ({tokenizer.decode([tokens[0, prompt_lengths[0].item()].item()])!r})")

                # Compare Mirage's argmax_in (logits) with PT-computed logits
                mg_argmax_in = verify_tensors.get("argmax_in")
                if mg_argmax_in is not None:
                    print(f"\n--- Argmax Input (Logits) Comparison ---")
                    mg_logits = mg_argmax_in[0]
                    print(f"  PT logits[:8]: {logits_pt[:8].float().tolist()}")
                    print(f"  MG logits[:8]: {mg_logits[:8].float().tolist()}")
                    logit_diff = (logits_pt.float() - mg_logits.float()).abs()
                    print(f"  Logits max abs diff: {logit_diff.max().item():.6f}")
                    print(f"  Logits mean abs diff: {logit_diff.mean().item():.6f}")
                    # Where does Mirage's argmax think the max is?
                    mg_top_vals, mg_top_idx = mg_logits.float().topk(5)
                    print(f"  MG logits top-5 indices: {mg_top_idx.tolist()}")
                    print(f"  MG logits top-5 values: {mg_top_vals.tolist()}")
                    # What does Mirage's logit say at position 2637 vs 87844?
                    print(f"  MG logit[2637]={mg_logits[2637].float().item():.4f} PT logit[2637]={logits_pt[2637].float().item():.4f}")
                    print(f"  MG logit[87844]={mg_logits[87844].float().item():.4f} PT logit[87844]={logits_pt[87844].float().item():.4f}")
                    # Check if MG logits are all zeros or something weird
                    print(f"  MG logits nonzero: {(mg_logits.abs() > 1e-6).sum().item()} / {mg_logits.numel()}")
                    print(f"  MG logits stats: min={mg_logits.float().min().item():.4f} max={mg_logits.float().max().item():.4f} mean={mg_logits.float().mean().item():.4f}")

                # Also check argmax partial/reduce tensors
                mg_part_val = verify_tensors.get("argmax_part_value")
                mg_part_idx = verify_tensors.get("argmax_part_index")
                if mg_part_val is not None and mg_part_idx is not None:
                    print(f"\n--- Argmax Partial Results ---")
                    print(f"  Part values[:8]: {mg_part_val[0, :8].float().tolist()}")
                    print(f"  Part indices[:8]: {mg_part_idx[0, :8].tolist()}")
                    # Find which partition has the max
                    max_part = mg_part_val[0].float().argmax().item()
                    print(f"  Max partition: {max_part}, value: {mg_part_val[0, max_part].float().item():.4f}, index: {mg_part_idx[0, max_part].item()}")
                    # Final output token
                    print(f"  Output token: {output_tokens[0].item()}")

            # Run full PyTorch model (all layers) for this decode step and compare
            try:
                print(f"\n--- PyTorch full model reference (all {num_layers} layers) ---")
                # Save current model state
                original_layers = list(model.model.layers)
                # Save KV cache state
                kv_k_save = model.model.kv_cache[0].clone()
                kv_v_save = model.model.kv_cache[1].clone()
                # Reset KV cache and run sequential prefill
                model.model.kv_cache[0].zero_()
                model.model.kv_cache[1].zero_()
                model.model.layers = torch.nn.ModuleList(original_layers[:num_layers])
                with torch.inference_mode():
                    prev_pos = 0
                    step_ref = torch.tensor([0], device="cuda")
                    for cur_pos in range(1, prompt_len_v + 1):
                        step_ref.fill_(cur_pos - 1)
                        ref_ids = tokens[:1, prev_pos:cur_pos]
                        ref_cos = position_embeddings[0][:, prev_pos:cur_pos]
                        ref_sin = position_embeddings[1][:, prev_pos:cur_pos]
                        ref_logits = model.forward(
                            input_ids=ref_ids,
                            position_embeddings=(ref_cos, ref_sin),
                            step=step_ref,
                        )
                        prev_pos = cur_pos
                model.model.layers = torch.nn.ModuleList(original_layers)
                # Compare KV caches before restoring
                print(f"\n--- KV Cache Comparison (Mirage vs PyTorch) ---")
                for li in range(num_layers):
                    for cache_type, (mg_kv, pt_kv) in enumerate([
                        (kv_k_save, model.model.kv_cache[0]),
                        (kv_v_save, model.model.kv_cache[1])
                    ]):
                        ct = "K" if cache_type == 0 else "V"
                        for pos in range(prompt_len_v):
                            mg_val = mg_kv[li, 0, pos]  # [num_heads, head_dim]
                            pt_val = pt_kv[li, 0, pos]
                            diff = (mg_val.float() - pt_val.float()).abs().max().item()
                            if diff > 0.01 or pos == 0:
                                print(f"  Layer {li} {ct} pos={pos}: max_diff={diff:.6f}"
                                      f"  mg[:3]={mg_val[0,:3].float().tolist()}"
                                      f"  pt[:3]={pt_val[0,:3].float().tolist()}")
                # Restore KV cache
                model.model.kv_cache[0].copy_(kv_k_save)
                model.model.kv_cache[1].copy_(kv_v_save)
                del kv_k_save, kv_v_save
                pt_token = ref_logits[0, -1].argmax().item()
                mg_token = output_tokens[0].item()
                print(f"  PT token: {pt_token} ({tokenizer.decode([pt_token])!r})")
                print(f"  MG token: {mg_token} ({tokenizer.decode([mg_token])!r})")
                print(f"  Match: {pt_token == mg_token}")
                # Compare logits
                pt_top_vals, pt_top_idx = ref_logits[0, -1].float().topk(5)
                print(f"  PT top-5: {[(i, tokenizer.decode([i])) for i, v in zip(pt_top_idx.tolist(), pt_top_vals.tolist())]}")
                mg_argmax_in = verify_tensors.get("argmax_in")
                if mg_argmax_in is not None:
                    mg_top_vals, mg_top_idx = mg_argmax_in[0].float().topk(5)
                    print(f"  MG top-5: {[(i, tokenizer.decode([i])) for i, v in zip(mg_top_idx.tolist(), mg_top_vals.tolist())]}")
                    real_vocab = ref_logits.shape[-1]
                    logit_diff = (ref_logits[0, -1].float() - mg_argmax_in[0, :real_vocab].float()).abs()
                    print(f"  Logit max diff: {logit_diff.max().item():.4f}")
                    print(f"  Logit mean diff: {logit_diff.mean().item():.6f}")
            except Exception as e:
                import traceback
                print(f"  PT full model reference FAILED: {e}")
                traceback.print_exc()

            # === Pad region check for shared intermediates ===
            print(f"\n{'='*80}")
            print(f"PAD REGION CHECK (positions {hidden_size}..{PADDED_HIDDEN_SIZE-1})")
            print(f"{'='*80}")
            for tname in ["embed_out", "rmsnorm_out", "attn_proj_out",
                          "rmsnorm_out_moe", "mlp_weighted_sum_out"]:
                t = verify_tensors.get(tname)
                if t is not None and t.dim() == 2 and t.shape[1] == PADDED_HIDDEN_SIZE:
                    pad_max = t[0, hidden_size:].abs().max().item()
                    pad_norm = t[0, hidden_size:].float().norm().item()
                    print(f"  {tname}: pad_max={pad_max:.6f}  pad_norm={pad_norm:.4f}")
            # Also check mlp_out per-slot
            mg_mlp_out = verify_tensors.get("mlp_out")
            if mg_mlp_out is not None:
                for k in range(mg_mlp_out.shape[1]):
                    pad_max_k = mg_mlp_out[0, k, hidden_size:].abs().max().item()
                    print(f"  mlp_out slot {k}: pad_max={pad_max_k:.6f}")

            print("\n" + "=" * 80)
            print("VERIFICATION COMPLETE")
            print("=" * 80)

    if world_size > 1:
        dist.destroy_process_group()
