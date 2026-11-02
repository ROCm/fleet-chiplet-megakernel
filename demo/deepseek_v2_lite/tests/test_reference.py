"""fleet_ds.reference vs HF DeepseekV2DecoderLayer eager, random weights, no download.
naive = decision-1 path over the expanded HF cache; absorbed = decode_step_absorbed (future work)."""
import os, sys
import pytest, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "ref")]
from hf.configuration_deepseek import DeepseekV2Config  # noqa: E402
from hf.modeling_deepseek import DeepseekV2DecoderLayer  # noqa: E402
from transformers import DynamicCache  # noqa: E402
from fleet_ds import reference as R  # noqa: E402

from fleet_ds.reference import COMMON, REAL, ROPE  # noqa: E402,F401
SMALL = dict(hidden_size=256, num_attention_heads=4, num_key_value_heads=4, kv_lora_rank=64, qk_nope_head_dim=32,
             qk_rope_head_dim=16, v_head_dim=32, n_routed_experts=8, num_experts_per_tok=2,
             moe_intermediate_size=128, intermediate_size=128)
REAL = dict(hidden_size=2048, num_attention_heads=16, num_key_value_heads=16, kv_lora_rank=512, qk_nope_head_dim=128,
            qk_rope_head_dim=64, v_head_dim=128, n_routed_experts=64, num_experts_per_tok=6,
            moe_intermediate_size=1408, intermediate_size=10944)
DT = {"fp32": torch.float32, "bf16": torch.bfloat16}
TOL = {"fp32": 1e-4, "bf16": 2e-2}  # bf16: observed ~5e-3 relative to |out|~1
P = 64  # prefill tokens


def mem_avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2**20
    return 0


def make_layer(dims, dtype, dense, seed=0):
    torch.manual_seed(seed)
    cfg = DeepseekV2Config(**COMMON, **dims)
    layer = DeepseekV2DecoderLayer(cfg, layer_idx=0 if dense else 1)
    layer.self_attn.layer_idx = 0  # DynamicCache indexes by layer_idx; we only have one layer
    rot = layer.self_attn.rotary_emb  # HF slices cos[:kv_seq_len] so position_ids >= T would IndexError
    rot.forward = lambda x, seq_len=None: (rot.cos_cached.to(x.dtype), rot.sin_cached.to(x.dtype))
    for n in (layer.input_layernorm, layer.post_attention_layernorm, layer.self_attn.kv_a_layernorm):
        n.weight.data.uniform_(0.5, 1.5)
    return cfg, layer.to(dtype).eval()


def run_hf(layer, x, pos0, dtype):
    """Prefill P tokens then decode 1. Returns (decode out [hidden], c_kv hook outputs, cache, gate input)."""
    ckv, gate_in = [], []
    hooks = [layer.self_attn.kv_a_layernorm.register_forward_hook(lambda m, i, o: ckv.append(o)),
             layer.post_attention_layernorm.register_forward_hook(lambda m, i, o: gate_in.append(o))]
    cache = DynamicCache()
    mask = torch.full((P, P), torch.finfo(dtype).min, dtype=dtype).triu(1)[None, None]
    with torch.no_grad():
        layer(x[:, :P], attention_mask=mask, position_ids=torch.arange(pos0, pos0 + P)[None],
              past_key_value=cache, use_cache=True)
        out = layer(x[:, P:], attention_mask=torch.zeros(1, 1, 1, P + 1, dtype=dtype),
                    position_ids=torch.tensor([[pos0 + P]]), past_key_value=cache, use_cache=True)[0]
    for h in hooks:
        h.remove()
    return out[0, 0], ckv, cache, gate_in[-1][0, 0]


def run_ref(layer, cfg, x, pos0, dtype, ckv, cache, dense, path):
    w = layer.state_dict()
    H, dn, dr, dv = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
    if path == "naive":  # cache = HF DynamicCache rows as-is (direct prefill handoff)
        cache_k = torch.zeros(H, P + 1, dn + dr, dtype=dtype)
        cache_v = torch.zeros(H, P + 1, dv, dtype=dtype)
        cache_k[:, :P] = cache.key_cache[0][0, :, :P]
        cache_v[:, :P] = cache.value_cache[0][0, :, :P]
        step = R.decode_step
    else:
        cache_k = torch.zeros(P + 1, cfg.kv_lora_rank, dtype=dtype)
        cache_v = torch.zeros(P + 1, dr, dtype=dtype)
        cache_k[:P] = ckv[0][0]
        cache_v[:P] = cache.key_cache[0][0, 0, :P, dn:]
        step = R.decode_step_absorbed
    with torch.no_grad():
        out, routing = step(x[0, P], w, cfg, cache_k, cache_v, P, dense, pos=pos0 + P)
    return out, routing, cache_k, cache_v


def cases():
    yield from [("small", d, dense, pos0) for d in DT for dense in (False, True) for pos0 in (0, 5000)]
    yield ("real", "bf16", True, 0)
    yield ("real", "bf16", False, 0)
    yield ("real", "fp32", True, 0)
    yield pytest.param("real", "fp32", False, 0, marks=pytest.mark.skipif(mem_avail_gb() < 6, reason="RAM"))


@pytest.mark.parametrize("path", ["naive", "absorbed"])
@pytest.mark.parametrize("size,dt,dense,pos0", list(cases()))
def test_decode_step(path, size, dt, dense, pos0):
    dims, dtype = (SMALL if size == "small" else REAL), DT[dt]
    cfg, layer = make_layer(dims, dtype, dense)
    torch.manual_seed(1)
    x = torch.randn(1, P + 1, cfg.hidden_size).to(dtype)
    hf_out, ckv, cache, gate_in = run_hf(layer, x, pos0, dtype)
    out, routing, cache_k, cache_v = run_ref(layer, cfg, x, pos0, dtype, ckv, cache, dense, path)

    err = (out.float() - hf_out.float()).abs().max().item()
    scale = hf_out.float().abs().max().item()
    print(f"\n[{path} {size} {dt} dense={dense} pos0={pos0}] max|err|={err:.3e} max|out|={scale:.3e} rel={err/scale:.3e}")
    assert err / scale < TOL[dt]

    if path == "naive":  # new cache rows == HF key_cache/value_cache rows for this token (all heads)
        hf_k, hf_v = cache.key_cache[0][0, :, P], cache.value_cache[0][0, :, P]
        for ours, hf in ((cache_k[:, P], hf_k), (cache_v[:, P], hf_v)):
            d = (ours.float() - hf.float()).abs().max().item()
            print(f"  cache row max|diff|={d:.3e} equal={torch.equal(ours, hf)}")
            assert d < TOL[dt]
        assert torch.equal(cache_k[:, P], hf_k) or dt == "bf16"
        assert torch.equal(cache_v[:, P], hf_v) or dt == "bf16"
    else:
        cache_c, cache_r = cache_k, cache_v

        # new-token cache row == what HF produced (c_kv from hook, k_pe from key_cache)
        assert torch.equal(cache_c[P], ckv[1][0, 0]) or dt == "bf16"
        assert (cache_c[P].float() - ckv[1][0, 0].float()).abs().max() < TOL[dt]
        assert (cache_r[P].float() - cache.key_cache[0][0, 0, P, cfg.qk_nope_head_dim:].float()).abs().max() < TOL[dt]
        # handoff check: key_cache[...,:nope] == W_UK,h @ c_kv_hook
        wkv = layer.self_attn.kv_b_proj.weight.view(cfg.num_attention_heads, -1, cfg.kv_lora_rank)
        k_nope = torch.einsum("hdr,tr->htd", wkv[:, :cfg.qk_nope_head_dim], ckv[0][0])
        assert (k_nope.float() - cache.key_cache[0][0, :, :P, :cfg.qk_nope_head_dim].float()).abs().max() < TOL[dt]

    if not dense:  # routing: exact expert set and weights
        tw, ti = routing
        hf_ti, hf_tw, _ = layer.mlp.gate(gate_in[None, None])
        assert sorted(ti.tolist()) == sorted(hf_ti[0].tolist()), (ti, hf_ti)
        ours = dict(zip(ti.tolist(), tw.tolist()))
        for e, wt in zip(hf_ti[0].tolist(), hf_tw[0].tolist()):
            assert abs(ours[e] - wt) < 1e-5 * (1 if dt == "fp32" else 100)


def test_constants_match_doc03():
    cfg = DeepseekV2Config(**COMMON, **REAL)
    assert abs(R.softmax_scale(cfg) - 0.114721) < 1e-6
    assert R.yarn_mscale(cfg) == 1.0
    f = R.yarn_inv_freq(cfg)
    assert torch.allclose(f[:3], torch.tensor([1.0, 0.7499, 0.5623]), atol=1e-4)
    assert torch.allclose(f[29:], torch.tensor([5.93e-6, 4.45e-6, 3.33e-6]), rtol=2e-3)
    assert torch.allclose(f[:11], 1.0 / 10000 ** (torch.arange(0, 22, 2) / 64))  # i<=10 unscaled
    assert torch.allclose(f[23:], 1.0 / (40 * 10000 ** (torch.arange(46, 64, 2) / 64)))  # i>=23 /40


def test_interleaved_rope_equivalent():
    """The kernel may rotate adjacent pairs in place; scores are unchanged."""
    cfg = DeepseekV2Config(**COMMON, **REAL)
    torch.manual_seed(0)
    q, k = torch.randn(16, 64), torch.randn(64)
    for pos in (0, 1, 1000, 100000):
        cos, sin = R.rope_cos_sin(pos, cfg, torch.float32)
        s_hf = R.apply_rope(q, cos, sin) @ R.apply_rope(k, cos, sin)
        s_il = R.apply_rope_interleaved(q, cos, sin) @ R.apply_rope_interleaved(k, cos, sin)
        assert torch.allclose(s_hf, s_il, atol=1e-4), pos
