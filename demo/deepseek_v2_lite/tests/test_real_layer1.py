"""Real DeepSeek-Coder-V2-Lite weights, layer 1 (MoE): HF eager vs fleet_ds.reference. Skipped unless a checkpoint shard is under models/ (fetch_model.sh)."""
import glob, os
import pytest, torch
from safetensors import safe_open
from test_reference import COMMON, REAL, P, TOL, make_layer, run_hf, run_ref

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARDS = glob.glob(os.path.join(ROOT, "models", "*.safetensors")) + glob.glob(os.path.join(ROOT, "models", "*", "*.safetensors"))


def _complete(path):  # a shard mid-download has a valid header but the file is short
    try:
        with safe_open(path, "pt") as f:
            f.get_slice(next(iter(f.keys())))
        return True
    except Exception:
        return False


SHARDS = [s for s in SHARDS if _complete(s)]
pytestmark = pytest.mark.skipif(not SHARDS, reason="no complete checkpoint shard under models/ (fetch_model.sh)")


def load(prefix):
    out = {}
    for s in SHARDS:
        with safe_open(s, "pt") as f:  # mmap; only the requested tensors are read
            out.update({k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)})
    return out


def test_real_layer1_bf16():
    w = load("model.layers.1.")
    assert len(w) == 203 and w["self_attn.q_proj.weight"].dtype == torch.bfloat16
    cfg, layer = make_layer(REAL, torch.bfloat16, dense=False)
    layer.load_state_dict(w)
    emb = load("model.")["embed_tokens.weight"]
    torch.manual_seed(1)
    x = emb[torch.randint(emb.shape[0], (1, P + 1))]  # real embedding rows of random tokens
    hf_out, ckv, cache, gate_in = run_hf(layer, x, 0, torch.bfloat16)
    out, (tw, ti), cache_k, cache_v = run_ref(layer, cfg, x, 0, torch.bfloat16, ckv, cache, dense=False, path="naive")
    err, scale = (out.float() - hf_out.float()).abs().max().item(), hf_out.float().abs().max().item()
    print(f"\n[real layer1 bf16] max|err|={err:.3e} max|out|={scale:.3e} rel={err/scale:.3e} equal={torch.equal(out, hf_out)}")
    assert err / scale < TOL["bf16"]
    # decision 1: our appended rows == HF key_cache/value_cache rows for this token
    assert (cache_k[:, P].float() - cache.key_cache[0][0, :, P].float()).abs().max() < TOL["bf16"]
    assert (cache_v[:, P].float() - cache.value_cache[0][0, :, P].float()).abs().max() < TOL["bf16"]
    hf_ti, hf_tw, _ = layer.mlp.gate(gate_in[None, None])
    assert sorted(ti.tolist()) == sorted(hf_ti[0].tolist())
    assert torch.allclose(tw[ti.argsort()], hf_tw[0][hf_ti[0].argsort()].float(), atol=1e-3)
