"""Pure-torch reference for ONE batch-1 decode step of a DeepSeek-V2(-Lite) decoder layer.

Primary path (`decode_step`, design D1): naive MLA over the expanded HF DynamicCache layout
K[H,T,nope+rope] / V[H,T,v]. `decode_step_absorbed` keeps the absorbed [c_kv|k_pe] form (future work).
MoE (decision 7a): the shared expert is expert index n_routed_experts, always selected with weight 1.0.

Weights are a flat dict with HF per-layer names (`layer.state_dict()` or safetensors with the
`model.layers.L.` prefix stripped). `cfg` is a DeepseekV2Config-like object (attribute access).
All activations are in the dtype of `x`; matmuls are plain torch (fp32 accumulate, output dtype).
"""
import math
import torch
import torch.nn.functional as F

# DeepSeek-Coder-V2-Lite-Base config.json, split so tests can pair COMMON with SMALL dims (tests/test_reference.py)
# and fleet_ds/mirage_graph.py --no-gpu can build the HF layers at REAL dims on the meta device without a checkpoint.
ROPE = dict(type="yarn", factor=40, original_max_position_embeddings=4096, beta_fast=32, beta_slow=1,
            mscale=0.707, mscale_all_dim=0.707)
COMMON = dict(q_lora_rank=None, first_k_dense_replace=1, topk_method="greedy", n_group=1, topk_group=1,
              norm_topk_prob=False, routed_scaling_factor=1.0, rms_norm_eps=1e-6, rope_theta=10000,
              rope_scaling=ROPE, max_position_embeddings=8192, n_shared_experts=2, hidden_act="silu",
              attn_implementation="eager")
REAL = dict(hidden_size=2048, num_attention_heads=16, num_key_value_heads=16, kv_lora_rank=512, qk_nope_head_dim=128,
            qk_rope_head_dim=64, v_head_dim=128, n_routed_experts=64, num_experts_per_tok=6,
            moe_intermediate_size=1408, intermediate_size=10944)


def rmsnorm(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)  # HF: weight * hidden.to(input_dtype)


def yarn_inv_freq(cfg):
    dim, base, rs = cfg.qk_rope_head_dim, cfg.rope_theta, cfg.rope_scaling
    i = torch.arange(0, dim, 2, dtype=torch.float32) / dim
    extra = 1.0 / base**i  # ops mirror HF bit-for-bit: 1 ulp in inv_freq is visible at pos ~1e4
    inter = 1.0 / (rs["factor"] * base**i)
    omax = rs["original_max_position_embeddings"]
    corr = lambda rot: dim * math.log(omax / (rot * 2 * math.pi)) / (2 * math.log(base))
    low = max(math.floor(corr(rs["beta_fast"])), 0)
    high = min(math.ceil(corr(rs["beta_slow"])), dim - 1)
    mask = 1.0 - torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    return inter * (1 - mask) + extra * mask


def yarn_mscale(cfg):
    rs = cfg.rope_scaling
    get = lambda m: 0.1 * m * math.log(rs["factor"]) + 1.0 if rs["factor"] > 1 else 1.0
    return get(rs["mscale"]) / get(rs["mscale_all_dim"])


def softmax_scale(cfg):
    s = (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) ** -0.5
    m = cfg.rope_scaling.get("mscale_all_dim", 0)
    if m:
        m = 0.1 * m * math.log(cfg.rope_scaling["factor"]) + 1.0
        s *= m * m
    return s


def rope_cos_sin(pos, cfg, dtype):
    """cos/sin [rope_dim] in HF 'half' layout, fp32 math then cast (HF builds the table in fp32)."""
    f = pos * yarn_inv_freq(cfg)
    emb = torch.cat([f, f]) * yarn_mscale(cfg)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin):
    """x [..., d] in the ORIGINAL (kv_a_proj / q_proj) layout. Returns HF's permuted+rotated layout:
    x is first permuted [x0,x1,...] -> [x0,x2,..,x1,x3,..] then rotated half-style. The cache holds this layout."""
    d = x.shape[-1]
    x = x.reshape(*x.shape[:-1], d // 2, 2).transpose(-1, -2).reshape(*x.shape[:-1], d)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def apply_rope_interleaved(x, cos, sin):
    """Same rotation applied in place on adjacent pairs (no permute) — what a kernel would do.
    Output layout differs from apply_rope by the fixed permutation; dot products are identical."""
    d = x.shape[-1]
    c, s = cos[: d // 2], sin[: d // 2]
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = xe * c - xo * s
    out[..., 1::2] = xo * c + xe * s
    return out


# ---- shared attention helpers -------------------------------------------------------------------

def q_rope(xn, w, cfg, pos):
    """q_proj -> per-head (q_nope [H,nope], q_pe roped [H,rope])."""
    H, dn = cfg.num_attention_heads, cfg.qk_nope_head_dim
    q = F.linear(xn, w["self_attn.q_proj.weight"]).view(H, -1)
    cos, sin = rope_cos_sin(pos, cfg, xn.dtype)
    return q[:, :dn], apply_rope(q[:, dn:], cos, sin)


def kv_new(xn, w, cfg, pos):
    """Latent row for this token: (c_kv normalised [kv_lora], k_pe roped [rope])."""
    ckv = F.linear(xn, w["self_attn.kv_a_proj_with_mqa.weight"])
    c, k_pe = ckv.split([cfg.kv_lora_rank, cfg.qk_rope_head_dim], -1)
    c = rmsnorm(c, w["self_attn.kv_a_layernorm.weight"], cfg.rms_norm_eps)
    cos, sin = rope_cos_sin(pos, cfg, xn.dtype)
    return c, apply_rope(k_pe, cos, sin)


def kv_expand(c, k_pe, w, cfg):
    """kv_b_proj on the latent -> HF cache rows K_new [H, nope+rope] = [k_nope | k_pe (broadcast)], V_new [H, v]."""
    H, dn = cfg.num_attention_heads, cfg.qk_nope_head_dim
    kv = F.linear(c, w["self_attn.kv_b_proj.weight"]).view(H, -1)
    k_nope, v = kv[:, :dn], kv[:, dn:]
    return torch.cat([k_nope, k_pe.expand(H, -1)], -1), v


def o_proj(o, w):
    return F.linear(o.reshape(-1), w["self_attn.o_proj.weight"])


def attention(q_nope, q_pe, w, cfg, cache_k, cache_v):
    """Ordinary MHA (HF eager order): q [H,nope+rope] . K [H,T,nope+rope], fp32 softmax -> dtype, P @ V."""
    q = torch.cat([q_nope, q_pe], -1)
    s = torch.einsum("hd,htd->ht", q, cache_k) * softmax_scale(cfg)
    p = torch.softmax(s, -1, dtype=torch.float32).to(q.dtype)
    return torch.einsum("ht,htv->hv", p, cache_v)


def attention_absorbed(q_nope, q_pe, w, cfg, cache_c, cache_r):
    """Absorbed MLA: q_abs = W_UK^T q_nope, scores over [C|R], value = C, then W_UV. Returns o [H, v]."""
    H, dn = cfg.num_attention_heads, cfg.qk_nope_head_dim
    wkv = w["self_attn.kv_b_proj.weight"].view(H, -1, cfg.kv_lora_rank)
    W_UK, W_UV = wkv[:, :dn], wkv[:, dn:]  # [H, dn, r], [H, dv, r]
    q_abs = torch.einsum("hdr,hd->hr", W_UK, q_nope)  # [H, r]
    s = (q_abs.float() @ cache_c.float().T + q_pe.float() @ cache_r.float().T) * softmax_scale(cfg)
    p = torch.softmax(s, -1).to(q_nope.dtype)  # [H, T]
    o_lat = p @ cache_c  # [H, r]
    return torch.einsum("hvr,hr->hv", W_UV, o_lat)


# ---- MLP / MoE ----------------------------------------------------------------------------------

def mlp(x, wg, wu, wd):
    return F.linear(F.silu(F.linear(x, wg)) * F.linear(x, wu), wd)


def expert_weights(w, cfg, e):
    """Expert e in 0..n_routed-1 is routed; e == n_routed is the shared expert (decision 7a)."""
    p = f"mlp.experts.{e}." if e < cfg.n_routed_experts else "mlp.shared_experts."
    return w[p + "gate_proj.weight"], w[p + "up_proj.weight"], w[p + "down_proj.weight"]


def route(xn, w, cfg):
    """fp32 gate -> softmax -> top-k (no renorm). Returns (weights fp32 [k], idx int64 [k])."""
    logits = F.linear(xn.float(), w["mlp.gate.weight"].float())
    tw, ti = torch.topk(torch.softmax(logits, -1), cfg.num_experts_per_tok, sorted=False)
    return tw * cfg.routed_scaling_factor, ti


def moe(xn, w, cfg):
    """n_routed+1 experts; the shared one (index n_routed) is always selected with weight 1.0.
    Combine mirrors HF exactly: fp32 weighted sum of routed outputs -> dtype, then + shared in dtype."""
    tw, ti = route(xn, w, cfg)
    E = cfg.n_routed_experts
    y, shared = torch.zeros(xn.shape, dtype=torch.float32), None
    for wt, e in list(zip(tw.tolist(), ti.tolist())) + [(1.0, E)]:
        out = mlp(xn, *expert_weights(w, cfg, e))
        if e == E:
            shared = wt * out  # weight 1.0, added in dtype after the routed fp32 sum is cast (HF order)
        else:
            y += wt * out.float()
    return y.to(xn.dtype) + shared, tw, ti


def _step(x, w, cfg, dense, attn_o):
    x = x + o_proj(attn_o, w)
    xn = rmsnorm(x, w["post_attention_layernorm.weight"], cfg.rms_norm_eps)
    if dense:
        return x + mlp(xn, w["mlp.gate_proj.weight"], w["mlp.up_proj.weight"], w["mlp.down_proj.weight"]), None
    y, tw, ti = moe(xn, w, cfg)
    return x + y, (tw, ti)


def decode_step(x, w, cfg, cache_k, cache_v, slot, dense, pos=None):
    """One layer, one token, naive attention over the expanded HF cache (decision 1).
    x [hidden]. cache_k [H, T_max, nope+rope], cache_v [H, T_max, v] (DynamicCache key/value_cache[l][0]);
    this token's rows go in `slot` (= tokens already cached) and are RoPE'd at `pos` (default: slot).
    Returns (x_out, routing) where routing = (topk_weight, topk_idx) or None for the dense layer."""
    pos = slot if pos is None else pos
    xn = rmsnorm(x, w["input_layernorm.weight"], cfg.rms_norm_eps)
    q_nope, q_pe = q_rope(xn, w, cfg, pos)
    cache_k[:, slot], cache_v[:, slot] = kv_expand(*kv_new(xn, w, cfg, pos), w, cfg)
    o = attention(q_nope, q_pe, w, cfg, cache_k[:, : slot + 1], cache_v[:, : slot + 1])
    return _step(x, w, cfg, dense, o)


def decode_step_absorbed(x, w, cfg, cache_c, cache_r, slot, dense, pos=None):
    """Same layer step with the absorbed [c_kv|k_pe] cache: cache_c [T_max, kv_lora], cache_r [T_max, rope]."""
    pos = slot if pos is None else pos
    xn = rmsnorm(x, w["input_layernorm.weight"], cfg.rms_norm_eps)
    q_nope, q_pe = q_rope(xn, w, cfg, pos)
    cache_c[slot], cache_r[slot] = kv_new(xn, w, cfg, pos)
    o = attention_absorbed(q_nope, q_pe, w, cfg, cache_c[: slot + 1], cache_r[: slot + 1])
    return _step(x, w, cfg, dense, o)


def decode_token(token, w, cfg, caches, slot, pos=None):
    """Whole model, one token: embed -> every layer (dense while l < first_k_dense_replace) -> final norm ->
    lm_head -> argmax. `w` = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
    "layers": [per-layer flat dicts as decode_step takes]}; caches = [(cache_k, cache_v)] per layer, written in
    place at `slot`. Returns (next_token int, logits fp32 [vocab], [routing per layer], [x after each layer])."""
    x, routing, hidden = w["model.embed_tokens.weight"][token], [], []
    for l, (lw, (ck, cv)) in enumerate(zip(w["layers"], caches)):
        x, r = decode_step(x, lw, cfg, ck, cv, slot, l < cfg.first_k_dense_replace, pos)
        routing.append(r), hidden.append(x)
    logits = F.linear(rmsnorm(x, w["model.norm.weight"], cfg.rms_norm_eps), w["lm_head.weight"]).float()
    return int(torch.argmax(logits)), logits, routing, hidden
