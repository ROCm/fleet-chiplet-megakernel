"""ModelSpec: the single per-model source of truth for the titan vLLM adapter.

`load_spec(vllm_arch)` returns a `ModelSpec` carrying everything the
model-agnostic machinery (buffers, pointer table, decode ABI) needs to vary per
model, while the decode hot path stays identical across models.

Two families today:
  * dense  (Qwen3-8B):  qwen3_8b_launch, QK-norm, dense gate/up/down MLP.
  * moe    (GPT-OSS):   gpt_oss_120b_launch, attn sinks, 128-expert top-4 MLP.

## Why this file no longer parses YAML

It used to, with its own private schema: `raw["kernel"]["page_size"]`,
`m["num_attention_heads"]`, `m["padded_hidden_size"]`, and so on. Every one of
those keys has since been renamed or moved, so `load_spec` raised
`KeyError: 'kernel'` on both architectures -- the plugin could not build a spec
at all.

That is not a bug to patch key-by-key. It is the same class of failure the
generator work already fixed twice: a constant maintained in two places drifts,
and the drift surfaces as garbage output rather than an error. A *schema*
maintained in two places drifts the same way.

So the spec is now built from `titan_generate.load_and_validate()` -- the exact
function that emits the kernel. Whatever the YAML says, whatever derivation
rules apply, the plugin sees precisely what the compiled `.cuh` was generated
from, because it is the same object. Renaming a YAML key can no longer break
the plugin silently; the generator and the plugin move together or not at all.

Three constants that used to be literals here are consequences of that:

  * `counters_per_layer` -- shipped as `40 * 16` while the kernel had moved to
    `103 * 16`. The kernel derives `rank_counters` from the header's value while
    Python sizes the allocation, so the mismatch was an unchecked atomicAdd
    ~36000 ints past the end of the buffer. Same constant, same bug, fixed on
    the generator side in ca48d19.
  * `ptrs_per_layer` -- the MoE kernel indexes 26 in + 11 out + 1 trailing
    layer-output slot = 38, not `ptrs_in + ptrs_out`. `cfg.ptrs_in/ptrs_out`
    (19/12) are vestigial for MoE; the emitted kernel is built from the
    MIRAGE_IN / MIRAGE_OUT tables. See `_moe_spec`.
  * `padded_vocab_size` -- was hardcoded to 201216 to paper over a stale YAML
    `vocab_size: 152064`. The YAML now says 201088 and the generator pads it,
    so the override is gone.
"""

import ctypes
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

_TITAN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_DIR = os.path.join(_TITAN_ROOT, "configs")
_GENERATED_DIR = os.path.join(_TITAN_ROOT, "generated")


def _load_cfg(config_stem):
    """Parse configs/<stem>.yaml through the generator's own loader.

    Imported lazily and with the titan root on sys.path: titan_generate.py lives
    at the repo root, not in this package, and importing it at module scope would
    make `import titan_vllm` depend on cwd.
    """
    if _TITAN_ROOT not in sys.path:
        sys.path.insert(0, _TITAN_ROOT)
    import titan_generate

    return titan_generate.load_and_validate(
        os.path.join(_CONFIG_DIR, f"{config_stem}.yaml"))


def _qkv_barrier_slot():
    """SLOT_QKV_BARRIER as an int offset, from the generator's region list.

    Same reason as `_load_cfg`: this offset is a running sum over
    `titan_generate.COUNTER_REGIONS`, and re-declaring it here would let it drift
    silently -- a wrong value does not crash, it corrupts a live barrier.
    """
    if _TITAN_ROOT not in sys.path:
        sys.path.insert(0, _TITAN_ROOT)
    import titan_generate

    return titan_generate.counter_slots()["SLOT_QKV_BARRIER_NEW"][0] * 16


def _mirage_ptr_counts():
    """(ptrs_in, ptrs_out) from the generator's own MIRAGE_IN / MIRAGE_OUT tables.

    These same two lists emit the kernel's MIRAGE_IN_COUNT / MIRAGE_OUT_COUNT, so
    reading them here means build_ptr_table_moe cannot disagree with the kernel it
    is feeding. A count that is too small leaves a stale pointer in the tail slots;
    too large shifts every subsequent layer's base. Neither faults -- both produce
    wrong output from a table that looks well-formed.
    """
    if _TITAN_ROOT not in sys.path:
        sys.path.insert(0, _TITAN_ROOT)
    import titan_generate

    return len(titan_generate.MIRAGE_IN), len(titan_generate.MIRAGE_OUT)


@dataclass
class ModelSpec:
    # ── identity ──
    name: str
    family: str                     # "dense" | "moe"
    config_stem: str                # configs/<config_stem>.yaml

    # ── model dims ──
    num_layers: int
    hidden_size: int
    padded_hidden_size: int         # MFMA-aligned hidden dim the kernel runs on
    intermediate_size: int
    vocab_size: int
    padded_vocab_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    q_per_kv: int
    qkv_output_size: int            # num_q*hd + 2*num_kv*hd
    gateup_output_size: int         # 2*intermediate (dense); unused for moe
    rope_theta: float
    max_position_embeddings: int
    sliding_window: Optional[int]

    # ── attention extras ──
    has_qk_norm: bool               # dense Qwen3 per-head q/k RMSNorm
    has_sinks: bool                 # moe GPT-OSS attention sinks

    # ── MoE ──
    is_moe: bool
    num_experts: int
    num_experts_per_tok: int

    # ── hardware / GPU layout ──
    num_xcds: int
    workers_per_xcd: int
    num_kv_chunks: int
    output_per_wg: int
    gateup_opw: int

    # ── paged KV ──
    page_size: int
    kv_cache_stride: int            # num_kv_heads * head_dim

    # ── pointer table / weight layout ──
    weights_per_layer: int
    ptrs_in: int
    ptrs_out: int
    # Slots per layer beyond ptrs_in + ptrs_out. The fused MoE kernel appends a
    # trailing layer_output slot (SLOT_LAYER_OUTPUT) that only the last layer's
    # ResAdd reads, so its table is 24 + 11 + 1 = 36 wide. Dense has none.
    ptrs_extra: int

    # ── counters / timing ──
    counters_per_layer: int
    # Cache-line ints the kernel addresses PAST the per-layer blocks and the
    # rank-counter block, with absolute offsets. Dense needs none; the MoE kernel
    # needs a decode-iter line, the embed barrier (1 global + num_xcds per-XCD),
    # and slack for the TITAN_ILB_TIMING probe. See TitanBuffers for the map.
    counter_tail_ints: int
    # Int offset of the per-XCD QKV barrier WITHIN a layer's counter block. The
    # MoE pointer table passes it as input_ptrs[7], derived from the layer's
    # counter base -- it is not a standalone allocation. Taken from the
    # generator's COUNTER_REGIONS running sum, the same list the kernel's
    # SLOT_QKV_BARRIER_NEW is emitted from. Dense does not use it.
    slot_qkv_barrier: int
    timing_slots_per_layer: int
    timing_tail_slots: int

    # ── ABI ──
    launch_symbol: str
    init_symbol: str
    finalize_symbol: str
    # Trailing launch args past `argmax_output`, in order, as ctypes types.
    # Dense passes only timing_buf; the fused MoE kernel added logits_output,
    # embed_weight, cur_token_id and decode_ctrl. A ctypes list that disagrees
    # with the C signature is stack corruption, not a clean error, so this is
    # derived from the family rather than a bool.
    extra_launch_args: tuple
    so_path: str

    # ── dense GEMM tiling (per-XCD wg counts + MXFP4 wg-byte strides) ──
    # Populated only for the dense family; the MoE pointer-table builder derives
    # its own expert tiling. Kept as a dict so the dataclass stays family-uniform.
    gemm: dict = field(default_factory=dict)

    # ── derived ──
    @property
    def ptrs_per_layer(self):
        return self.ptrs_in + self.ptrs_out + self.ptrs_extra

    @property
    def rank_counter_ints(self):
        return self.num_xcds * 16

    @property
    def attn_reduction(self):
        return self.num_q_heads * self.head_dim


def _dense_spec(cfg, config_stem, so_path):
    hidden = cfg.hidden_size
    inter = cfg.intermediate_size
    num_q = cfg.num_q_heads
    num_kv = cfg.num_kv_heads
    head_dim = cfg.head_dim
    padded_vocab = cfg.padded_vocab_size
    num_xcds = cfg.num_xcds
    output_per_wg = cfg.output_per_wg
    gateup_opw = cfg.gateup_opw            # 64 gate + 64 up per workgroup (fused)

    qkv_output = cfg.qkv_output_size
    gateup_output = cfg.gateup_output_size
    attn_reduction = cfg.oproj_reduction

    # Per-XCD workgroup counts + MXFP4 wg-byte strides. Taken from cfg, which
    # computes them with the formulas the kernel was emitted from.
    gemm = {
        "qkv_n_wgs_per_xcd": cfg.qkv_n_wgs_per_xcd,
        "oproj_n_wgs_per_xcd": cfg.oproj_n_wgs_per_xcd,
        "gateup_n_wgs_per_xcd": cfg.gateup_n_wgs_per_xcd,
        "down_n_wgs_per_xcd": cfg.down_n_wgs_per_xcd,
        "lm_n_wgs_per_xcd": cfg.lm_n_wgs_per_xcd,
        "qkv_wg_bytes": cfg.qkv_wg_bytes,
        "oproj_wg_bytes": cfg.oproj_wg_bytes,
        "gateup_wg_bytes": cfg.gateup_wg_bytes,
        "down_wg_bytes": cfg.down_wg_bytes,
        "lm_wg_bytes": cfg.lm_wg_bytes,
    }

    stem = cfg.name_clean
    return ModelSpec(
        name=cfg.name,
        family="dense",
        config_stem=config_stem,
        num_layers=cfg.num_layers,
        hidden_size=hidden,
        padded_hidden_size=cfg.padded_hidden_size,
        intermediate_size=inter,
        vocab_size=cfg.vocab_size,
        padded_vocab_size=padded_vocab,
        num_q_heads=num_q,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        q_per_kv=cfg.q_per_kv,
        qkv_output_size=qkv_output,
        gateup_output_size=gateup_output,
        rope_theta=float(cfg.rope_theta),
        max_position_embeddings=cfg.max_position_embeddings,
        sliding_window=cfg.sliding_window or None,
        has_qk_norm=cfg.has_qk_norm,
        has_sinks=False,
        is_moe=False,
        num_experts=0,
        num_experts_per_tok=0,
        num_xcds=num_xcds,
        workers_per_xcd=cfg.workers_per_xcd,
        num_kv_chunks=cfg.num_kv_chunks,
        output_per_wg=output_per_wg,
        gateup_opw=gateup_opw,
        page_size=cfg.page_size,
        kv_cache_stride=cfg.kv_cache_stride,
        weights_per_layer=12,
        ptrs_in=cfg.ptrs_in,
        ptrs_out=cfg.ptrs_out,
        ptrs_extra=0,
        counters_per_layer=cfg.counters_per_layer,
        counter_tail_ints=0,
        slot_qkv_barrier=0,
        timing_slots_per_layer=12,
        timing_tail_slots=4,
        launch_symbol=f"{stem}_launch",
        init_symbol=f"{stem}_init",
        finalize_symbol=f"{stem}_finalize",
        extra_launch_args=(ctypes.c_void_p,),   # timing_buf
        so_path=so_path,
        gemm=gemm,
    )


def _moe_spec(cfg, config_stem, so_path):
    num_q = cfg.num_q_heads
    num_kv = cfg.num_kv_heads
    head_dim = cfg.head_dim
    hidden = cfg.hidden_size
    _moe_ptrs_in, _moe_ptrs_out = _mirage_ptr_counts()

    # MoE GEMM tiling. oproj/w13/w2 are MEASURED values, not derivations -- see
    # the `measured:` block in the YAML for what each alternative cost. cfg
    # carries them through and cross-checks the derivable ones.
    gemm = {
        "qkv_output_per_wg": cfg.output_per_wg,
        "oproj_output_per_wg": cfg.oproj_opw,
        "w13_output_per_wg": cfg.w13_output_per_wg,
        "w2_output_per_wg": cfg.w2_output_per_wg,
    }

    stem = cfg.name_clean
    return ModelSpec(
        name=cfg.name,
        family="moe",
        config_stem=config_stem,
        num_layers=cfg.num_layers,
        hidden_size=hidden,
        padded_hidden_size=cfg.padded_hidden_size,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        padded_vocab_size=cfg.padded_vocab_size,
        num_q_heads=num_q,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        q_per_kv=cfg.q_per_kv,
        qkv_output_size=cfg.qkv_output_size,
        gateup_output_size=0,
        rope_theta=float(cfg.rope_theta),
        max_position_embeddings=cfg.max_position_embeddings,
        sliding_window=cfg.sliding_window or None,
        has_qk_norm=cfg.has_qk_norm,
        has_sinks=True,
        is_moe=True,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        num_xcds=cfg.num_xcds,
        workers_per_xcd=cfg.workers_per_xcd,
        num_kv_chunks=cfg.num_kv_chunks,
        output_per_wg=cfg.output_per_wg,
        gateup_opw=cfg.w13_output_per_wg,
        page_size=cfg.page_size,
        kv_cache_stride=cfg.kv_cache_stride,
        # 13 packed weight tensors per layer. The demo packs 14: it also packs an
        # MXFP4 router at [6], which nothing reads -- the fused kernel routes with
        # the bf16 router GEMV (demo slot [8], our slot [6]). Order here:
        # [0]qkv_w [1]qkv_bias [2]oproj_w [3]oproj_bias [4]norm_pre [5]norm_post
        # [6]router_w_bf16 [7]router_b [8]gate_up_w [9]down_w [10]w13_bias
        # [11]w2_bias [12]attn_sinks. K/V are aliased zero-copy from vLLM.
        weights_per_layer=13,
        # 24 in + 11 out + 1 trailing layer_output, read from titan_generate's
        # MIRAGE_IN / MIRAGE_OUT tables -- the same lists that emit the kernel's
        # MIRAGE_IN_COUNT / MIRAGE_OUT_COUNT. NOT cfg.ptrs_in/ptrs_out (19/12):
        # those are the dense counts and are vestigial for MoE.
        ptrs_in=_moe_ptrs_in,
        ptrs_out=_moe_ptrs_out,
        ptrs_extra=1,
        counters_per_layer=cfg.counters_per_layer,
        # 16 decode-iter + (1 + num_xcds)*16 embed barrier + 20*16 ILB slack,
        # matching demo_gpt_oss_120b.py. The ILB slack is always allocated so a
        # TITAN_ILB_TIMING build needs no Python change; it costs 1.3 KB.
        counter_tail_ints=16 + (1 + cfg.num_xcds) * 16 + 20 * 16,
        slot_qkv_barrier=_qkv_barrier_slot(),
        timing_slots_per_layer=14,
        timing_tail_slots=4,
        launch_symbol=f"{stem}_launch",
        init_symbol=f"{stem}_init",
        finalize_symbol=f"{stem}_finalize",
        # The fused kernel moved the embedding lookup on-device, so the launch
        # takes three more args than the dense one past timing_buf -- plus
        # logits_output, which precedes timing_buf. Order here must match the
        # emitted C signature, not intuition: logits_output is FIRST.
        extra_launch_args=(
            ctypes.c_void_p,   # logits_output
            ctypes.c_void_p,   # timing_buf
            ctypes.c_void_p,   # embed_weight
            ctypes.c_int,      # cur_token_id
            ctypes.c_void_p,   # decode_ctrl
        ),
        so_path=so_path,
        gemm=gemm,
    )


# vLLM architecture name -> (config stem, family builder). The .so and the
# launch/init/finalize symbols are all named after the config stem by the
# generator, so they are derived rather than listed -- one less place to drift.
_REGISTRY = {
    "Qwen3ForCausalLM": ("qwen3_8b", _dense_spec),
    "GptOssForCausalLM": ("gpt_oss_120b", _moe_spec),
}


def supported_arches():
    return tuple(_REGISTRY.keys())


def load_spec(vllm_arch):
    """Build the ModelSpec for a vLLM architecture name.

    TITAN_SO overrides the .so path; otherwise generated/<default>.so.
    """
    if vllm_arch not in _REGISTRY:
        raise KeyError(
            f"no titan spec for vLLM arch {vllm_arch!r}; "
            f"known: {sorted(_REGISTRY)}")
    config_stem, builder = _REGISTRY[vllm_arch]
    cfg = _load_cfg(config_stem)
    so_path = os.environ.get(
        "TITAN_SO", os.path.join(_GENERATED_DIR, f"{cfg.name_clean}.so"))
    return builder(cfg, config_stem, so_path)
