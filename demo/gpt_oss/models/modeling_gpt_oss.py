# Copyright 2025 Mirage Team. Adapted from HuggingFace GPT-OSS implementation.
# Licensed under the Apache License, Version 2.0.
"""GPT-OSS 120B MoE model for Mirage inference."""

import math
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers import GptOssConfig

# Import triton RoPE lazily to avoid circular imports
try:
    from demo.qwen3.models.rope import apply_rotary_pos_emb_triton
except ImportError:
    apply_rotary_pos_emb_triton = None


class GptOssRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class GptOssRotaryEmbedding(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type="cuda", enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # GPT-OSS uses neox-style: only half rotation, no doubling
            emb = freqs
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=torch.bfloat16), sin.to(dtype=torch.bfloat16)


def apply_rotary_pos_emb_neox(q, k, cos, sin, unsqueeze_dim=1):
    """Apply rotary embeddings (neox style - first/second half split)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    first_half_q, second_half_q = torch.chunk(q, 2, dim=-1)
    first_half_k, second_half_k = torch.chunk(k, 2, dim=-1)
    q_embed = torch.cat([
        first_half_q * cos - second_half_q * sin,
        second_half_q * cos + first_half_q * sin,
    ], dim=-1)
    k_embed = torch.cat([
        first_half_k * cos - second_half_k * sin,
        second_half_k * cos + first_half_k * sin,
    ], dim=-1)
    return q_embed, k_embed


# MXFP4 dequantization LUT: E2M1 format
# Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6
_FP4_LUT = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def _dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor,
                       dtype=torch.bfloat16) -> torch.Tensor:
    """Dequantize MXFP4 packed expert weights to BF16.

    Args:
        blocks: uint8 [E, out_dim, num_blocks, 16] — packed FP4 nibbles (2 per byte)
        scales: uint8 [E, out_dim, num_blocks] — E8M0 block scales

    Returns:
        BF16 tensor [E, in_dim, out_dim] where in_dim = num_blocks * 32
        Transposed: columns = original rows for matmul compatibility.
    """
    device = blocks.device
    lut = torch.tensor(_FP4_LUT, dtype=dtype, device=device)

    # scales: E8M0 -> exponent (scale = 2^(e-127))
    exp = scales.to(torch.int32) - 127  # [E, out_dim, num_blocks]

    E, out_dim, num_blocks, B = blocks.shape
    assert B == 16, f"Expected 16 bytes per block, got {B}"

    # Flatten for efficient processing
    flat_blocks = blocks.reshape(-1, B)  # [E*out_dim*num_blocks, 16]
    flat_exp = exp.reshape(-1, 1)        # [E*out_dim*num_blocks, 1]

    # Unpack nibbles: each byte has lo (bits 0-3) and hi (bits 4-7)
    idx_lo = (flat_blocks & 0x0F).to(torch.long)
    idx_hi = (flat_blocks >> 4).to(torch.long)

    # LUT lookup
    out = torch.empty(flat_blocks.shape[0], B * 2, dtype=dtype, device=device)
    out[:, 0::2] = lut[idx_lo]
    out[:, 1::2] = lut[idx_hi]

    # Apply block scale: multiply by 2^(exp)
    torch.ldexp(out, flat_exp, out=out)

    # Reshape: [E, out_dim, num_blocks * 32] -> [E, out_dim, in_dim]
    out = out.reshape(E, out_dim, num_blocks * 32)

    # Transpose last two dims: [E, in_dim, out_dim] for matmul (x @ W)
    return out.transpose(1, 2).contiguous()


def swigluoai(gate_up, alpha=1.702, limit=7.0):
    """GPT-OSS activation: interleaved gate/up with clamped SiLU variant."""
    gate = gate_up[..., ::2]
    up = gate_up[..., 1::2]
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu


def naive_attention_with_sinks(
    q, key_cache, value_cache, kv_len, layer_idx, sinks,
    is_causal=True, enable_gqa=True, sliding_window=None,
):
    """Attention with per-head sink parameters (GPT-OSS specific)."""
    k = key_cache[layer_idx, 0, :kv_len, :, :]
    v = value_cache[layer_idx, 0, :kv_len, :, :]

    q_for_sdpa = q.permute(1, 0, 2)    # [num_q_heads, q_len, head_dim]
    k_for_sdpa = k.permute(1, 0, 2)    # [num_kv_heads, kv_len, head_dim]
    v_for_sdpa = v.permute(1, 0, 2)    # [num_kv_heads, kv_len, head_dim]

    num_q_heads = q_for_sdpa.shape[0]
    num_kv_heads = k_for_sdpa.shape[0]
    num_kv_groups = num_q_heads // num_kv_heads

    # Repeat KV for GQA
    if num_kv_groups > 1:
        k_for_sdpa = k_for_sdpa.repeat_interleave(num_kv_groups, dim=0)
        v_for_sdpa = v_for_sdpa.repeat_interleave(num_kv_groups, dim=0)

    q_len = q_for_sdpa.shape[1]
    scaling = q_for_sdpa.shape[-1] ** -0.5

    # Compute attention scores (cast to float32 for numerical stability)
    attn_weights = torch.matmul(
        q_for_sdpa.float(), k_for_sdpa.float().transpose(-2, -1)
    ) * scaling

    # Apply causal mask
    if is_causal and q_len > 1:
        causal_mask = torch.triu(
            torch.full((q_len, kv_len), float("-inf"), device=q.device, dtype=torch.float32),
            diagonal=kv_len - q_len + 1
        )
        attn_weights = attn_weights + causal_mask.unsqueeze(0)

    # Apply sliding window mask if needed
    if sliding_window is not None:
        row_idx = torch.arange(q_len, device=q.device).unsqueeze(1) + (kv_len - q_len)
        col_idx = torch.arange(kv_len, device=q.device).unsqueeze(0)
        sw_mask = (row_idx - col_idx) >= sliding_window
        attn_weights = attn_weights.masked_fill(sw_mask.unsqueeze(0), float("-inf"))

    # Append sinks: [num_heads, q_len, 1]
    sink_logits = sinks.float().reshape(-1, 1, 1).expand(-1, q_len, 1)
    combined = torch.cat([attn_weights, sink_logits], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1, dtype=torch.float32)
    scores = probs[..., :-1]  # drop sink

    attn_output = torch.matmul(scores, v_for_sdpa.float()).to(q.dtype)
    attn_output = attn_output.permute(1, 0, 2)  # [q_len, num_heads, head_dim]
    return attn_output


class GptOssAttention(nn.Module):
    def __init__(self, config: GptOssConfig, kv_cache, layer_idx: int, world_size: int):
        super().__init__()
        self.world_size = world_size
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.local_qkv_size = (self.num_heads // world_size) * self.head_dim
        self.key_cache, self.value_cache = kv_cache

        # GPT-OSS has bias on all attention projections
        self.q_proj = nn.Linear(self.hidden_size, (self.num_heads // world_size) * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, (self.num_key_value_heads // world_size) * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, (self.num_key_value_heads // world_size) * self.head_dim, bias=True)
        self.o_proj = nn.Linear((self.num_heads // world_size) * self.head_dim, self.hidden_size, bias=True)

        # Per-head sink parameters
        self.sinks = nn.Parameter(torch.empty(self.num_heads // world_size))

        # Sliding window: even layers only
        layer_types = getattr(config, 'layer_types', None)
        if layer_types is not None:
            self.sliding_window = config.sliding_window if layer_types[layer_idx] == "sliding_attention" else None
        else:
            self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else None

        self.rotary_emb = GptOssRotaryEmbedding(config=config)

    def forward(self, hidden_states, position_embeddings=None, step=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads // self.world_size, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads // self.world_size, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads // self.world_size, self.head_dim)

        cos, sin = position_embeddings

        # Apply rotary embeddings (neox style)
        query_states, key_states = apply_rotary_pos_emb_neox(
            query_states, key_states, cos, sin, unsqueeze_dim=2
        )

        # Update KV cache
        if q_len > 1:
            self.key_cache[self.layer_idx, 0, :q_len] = key_states[0]
            self.value_cache[self.layer_idx, 0, :q_len] = value_states[0]
        else:
            self.key_cache[self.layer_idx, 0, step] = key_states[0]
            self.value_cache[self.layer_idx, 0, step] = value_states[0]

        q = query_states[0]
        if q_len > 1:
            kv_len = q_len
        else:
            kv_len = step.item() + 1

        attn_output = naive_attention_with_sinks(
            q, self.key_cache, self.value_cache, kv_len, self.layer_idx,
            self.sinks,
            is_causal=(q_len > 1),
            enable_gqa=True,
            sliding_window=self.sliding_window,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.local_qkv_size)
        attn_output = self.o_proj(attn_output)

        if self.world_size > 1:
            dist.all_reduce(attn_output)

        return attn_output


class GptOssExperts(nn.Module):
    """MoE expert block with swigluoai activation.

    Expert weights are stored in MXFP4 format (uint8 blocks + uint8 scales)
    to fit the 120B model in GPU memory. Dequantization to BF16 happens
    on-the-fly per expert during forward pass.
    """
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # MXFP4 storage: blocks (packed nibbles) and scales (E8M0 exponents)
        # gate_up_proj: [E, 2*inter, num_blocks, 16] blocks + [E, 2*inter, num_blocks] scales
        # down_proj: [E, hidden, num_blocks, 16] blocks + [E, hidden, num_blocks] scales
        # These are registered as buffers (non-trainable, persistent)
        # Actual shapes set during from_pretrained loading
        self.register_buffer('gate_up_proj_blocks', None)
        self.register_buffer('gate_up_proj_scales', None)
        self.register_buffer('down_proj_blocks', None)
        self.register_buffer('down_proj_scales', None)
        # Biases remain in BF16 (small: [E, dim])
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_size))
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        # Fallback: if BF16 weights are loaded directly (e.g. for testing)
        self._gate_up_proj_bf16 = None
        self._down_proj_bf16 = None

    def _get_gate_up_weight(self, expert_idx):
        """Get gate_up weight for one expert, dequantizing from MXFP4 if needed."""
        if self._gate_up_proj_bf16 is not None:
            return self._gate_up_proj_bf16[expert_idx]
        blocks = self.gate_up_proj_blocks[expert_idx:expert_idx+1]  # [1, out_dim, nb, 16]
        scales = self.gate_up_proj_scales[expert_idx:expert_idx+1]  # [1, out_dim, nb]
        w = _dequantize_mxfp4(blocks, scales)  # [1, in_dim, out_dim]
        return w[0].to(torch.bfloat16)  # [in_dim, out_dim] = [hidden, 2*inter]

    def _get_down_weight(self, expert_idx):
        """Get down_proj weight for one expert, dequantizing from MXFP4 if needed."""
        if self._down_proj_bf16 is not None:
            return self._down_proj_bf16[expert_idx]
        blocks = self.down_proj_blocks[expert_idx:expert_idx+1]  # [1, out_dim, nb, 16]
        scales = self.down_proj_scales[expert_idx:expert_idx+1]  # [1, out_dim, nb]
        w = _dequantize_mxfp4(blocks, scales)  # [1, in_dim, out_dim]
        return w[0].to(torch.bfloat16)  # [in_dim, out_dim] = [inter, hidden]

    def forward(self, hidden_states, router_indices, routing_weights):
        batch_size = hidden_states.shape[0]
        hidden_states_flat = hidden_states.reshape(-1, self.hidden_size)
        num_tokens = hidden_states_flat.shape[0]
        num_experts = routing_weights.shape[1]

        next_states = torch.zeros_like(hidden_states_flat)

        expert_mask = F.one_hot(router_indices, num_classes=num_experts + 1).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0].item()
            if expert_idx == num_experts:
                continue
            _, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_flat[token_idx].to(torch.bfloat16)
            # Dequantize expert weights on-the-fly (one expert at a time)
            gate_up_w = self._get_gate_up_weight(expert_idx)  # [hidden, 2*inter]
            gate_up = current_state @ gate_up_w + self.gate_up_proj_bias[expert_idx]
            del gate_up_w
            activated = swigluoai(gate_up).to(torch.bfloat16)
            down_w = self._get_down_weight(expert_idx)  # [inter, hidden]
            out = activated @ down_w + self.down_proj_bias[expert_idx]
            del down_w
            weighted_output = out * routing_weights[token_idx, expert_idx, None]
            next_states.index_add_(0, token_idx, weighted_output.to(hidden_states_flat.dtype))

        return next_states.view(batch_size, -1, self.hidden_size)


class GptOssRouter(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.bias = nn.Parameter(torch.empty(self.num_experts))

    def forward(self, hidden_states):
        hidden_states_flat = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states_flat, self.weight, self.bias)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_top_value = F.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices, router_logits


class GptOssMLP(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.router = GptOssRouter(config)
        self.experts = GptOssExperts(config)

    def forward(self, hidden_states):
        router_scores, router_indices, router_logits = self.router(hidden_states)
        routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        return routed_out, router_logits


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config: GptOssConfig, kv_cache, layer_idx: int, world_size: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GptOssAttention(config, kv_cache, layer_idx, world_size)
        self.mlp = GptOssMLP(config)
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings=None, step=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            step=step,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, router_logits


class GptOssPreTrainedModel(PreTrainedModel):
    config_class = GptOssConfig

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GptOssRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, GptOssExperts):
            module.gate_up_proj_bias.data.zero_()
            module.down_proj_bias.data.zero_()
        elif isinstance(module, GptOssAttention):
            module.sinks.data.normal_(mean=0.0, std=std)
        elif isinstance(module, GptOssRouter):
            module.weight.data.normal_(mean=0.0, std=std)
            module.bias.data.normal_(mean=0.0, std=std)


class GptOssModel(GptOssPreTrainedModel):
    def __init__(self, config: GptOssConfig, world_size: int, max_num_pages: int, page_size: int):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        key_cache = torch.empty(
            (config.num_hidden_layers, max_num_pages, page_size,
             config.num_key_value_heads // world_size, config.head_dim),
            dtype=torch.bfloat16, device="cuda",
        )
        value_cache = torch.empty(
            (config.num_hidden_layers, max_num_pages, page_size,
             config.num_key_value_heads // world_size, config.head_dim),
            dtype=torch.bfloat16, device="cuda",
        )
        self.kv_cache = (key_cache, value_cache)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            GptOssDecoderLayer(config, self.kv_cache, layer_idx, world_size)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GptOssRotaryEmbedding(config=config)

        self.post_init()

    def forward(self, input_ids, position_embeddings=None, step=None):
        hidden_states = self.embed_tokens(input_ids)

        all_router_logits = []
        for layer in self.layers:
            hidden_states, router_logits = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                step=step,
            )
            all_router_logits.append(router_logits)

        hidden_states = self.norm(hidden_states)
        return hidden_states, all_router_logits


class GptOssForCausalLM(GptOssPreTrainedModel):
    def __init__(self, config: GptOssConfig, world_size: int = 1,
                 max_num_pages: int = 16, page_size: int = 4096):
        super().__init__(config)
        self.model = GptOssModel(config, world_size, max_num_pages, page_size)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, world_size=1,
                        max_num_pages=16, page_size=4096, **kwargs):
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config, world_size=world_size,
                    max_num_pages=max_num_pages, page_size=page_size)

        import glob
        import os
        from safetensors.torch import load_file

        if os.path.isdir(pretrained_model_name_or_path):
            model_path = pretrained_model_name_or_path
        else:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(pretrained_model_name_or_path)

        safetensor_files = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensor files found in {model_path}")

        # Collect MXFP4 blocks/scales for deferred dequantization
        # Key: "model.layers.N.mlp.experts.{gate_up_proj,down_proj}"
        # Value: {"blocks": tensor, "scales": tensor}
        mxfp4_pending = {}

        for sf_file in safetensor_files:
            state_dict = load_file(sf_file, device="cpu")
            mapped = {}
            for name, tensor in state_dict.items():
                # Handle MXFP4 expert weights: _blocks and _scales
                if name.endswith("_blocks") or name.endswith("_scales"):
                    # e.g. "model.layers.0.mlp.experts.gate_up_proj_blocks"
                    #   -> base = "model.layers.0.mlp.experts.gate_up_proj"
                    #   -> suffix = "blocks"
                    if name.endswith("_blocks"):
                        base = name[:-len("_blocks")]
                        suffix = "blocks"
                    else:
                        base = name[:-len("_scales")]
                        suffix = "scales"
                    if base not in mxfp4_pending:
                        mxfp4_pending[base] = {}
                    mxfp4_pending[base][suffix] = tensor
                    continue
                mapped[name] = tensor

            missing, unexpected = model.load_state_dict(mapped, strict=False)
            if unexpected:
                print(f"  Unexpected keys from {os.path.basename(sf_file)}: {unexpected[:5]}...")

        # Store MXFP4 expert weights directly (no dequantization to BF16!)
        # This keeps expert weights at ~65GB instead of ~230GB
        if mxfp4_pending:
            print(f"Loading {len(mxfp4_pending)} MXFP4 expert weight tensors (keeping FP4 format)...")
            for base_name, parts in sorted(mxfp4_pending.items()):
                if "blocks" not in parts or "scales" not in parts:
                    print(f"  WARNING: incomplete MXFP4 pair for {base_name}")
                    continue
                # Set blocks/scales directly on the expert module
                # base_name: "model.layers.N.mlp.experts.gate_up_proj" or ".down_proj"
                # Navigate to the module and set the buffer attribute
                parts_list = base_name.split(".")
                # parts_list: ['model', 'layers', 'N', 'mlp', 'experts', 'gate_up_proj' or 'down_proj']
                module = model
                for p in parts_list[:-1]:
                    if p.isdigit():
                        module = module[int(p)]
                    else:
                        module = getattr(module, p)
                attr_name = parts_list[-1]
                module.register_buffer(attr_name + "_blocks", parts["blocks"])
                module.register_buffer(attr_name + "_scales", parts["scales"])
                E, out_dim, num_blocks, B = parts["blocks"].shape
                in_dim = num_blocks * 32  # 32 FP4 values per block
                print(f"  {base_name}: MXFP4 [{E}, {out_dim}, {in_dim}] "
                      f"({parts['blocks'].numel() + parts['scales'].numel()} bytes)")

        return model

    @torch.inference_mode()
    def forward(self, input_ids, position_embeddings=None, step=None):
        hidden_states, all_router_logits = self.model(
            input_ids=input_ids,
            position_embeddings=position_embeddings,
            step=step,
        )
        logits = self.lm_head(hidden_states[:, -1:, :])
        return logits
