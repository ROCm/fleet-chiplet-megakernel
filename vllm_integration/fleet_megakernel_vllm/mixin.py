"""FleetMKModelMixin: the model-agnostic machinery that runs decode through fleet_mk's
fused megakernel while leaving prefill on vLLM's stock path.

This holds everything that does *not* vary per model: kernel load, workspace
buffer alloc, RoPE tables, zero-copy KV aliasing, and prefill/decode routing --
all parameterized by a `ModelSpec` (spec.py). The per-model pieces are supplied by
thin subclasses (model.py) via three hooks:

  * class attribute `FLEET_MK_ARCH` -- the vLLM architecture name load_spec() keys on.
  * `_fleet_mk_layer_attn(li)`       -- the vLLM Attention module for layer li
                                     (dense: layers[li].self_attn.attn;
                                      moe:   layers[li].attn.attn).
  * `_fleet_mk_pack_weights(...)`    -- MXFP4-pack this model's weights into fleet_mk's
                                     layout (dense 12-slot vs MoE 13-slot).
  * `_fleet_mk_embed_tokens(ids)`    -- token embedding lookup (dense: embed_tokens;
                                     moe: embedding); the mixin zero-pads it.

MRO note: subclasses must list the mixin FIRST
(`class T(FleetMKModelMixin, StockModel)`) so the mixin's forward()/load_weights()
override the stock model's, and `super().forward()` still reaches the stock model.
"""

import math
import os

import torch

from vllm.forward_context import get_forward_context

from .runtime import FleetMKBuffers, FleetMKDecoder, build_ptr_table, load_kernel
from .spec import load_spec


def build_rope_tables(head_dim, rope_theta, max_pos, device):
    """HF-style RoPE cos/sin tables of shape [max_pos, head_dim].

    Matches transformers' rotary embedding (emb = cat(freqs, freqs)), the same
    source the standalone demos feed to the megakernel.
    """
    inv_freq = 1.0 / (rope_theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)                 # [max_pos, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)          # [max_pos, head_dim]
    return (emb.cos().to(torch.bfloat16).contiguous(),
            emb.sin().to(torch.bfloat16).contiguous())


class FleetMKModelMixin:
    # Subclasses set this to the vLLM arch name (e.g. "Qwen3ForCausalLM").
    FLEET_MK_ARCH = None

    # ── state init (call from subclass __init__ after super().__init__) ───────
    def _fleet_mk_setup_state(self, vllm_config):
        self._max_model_len = vllm_config.model_config.max_model_len
        # HF model id/path -- MoE packing reloads the raw (un-swizzled) weights.
        self._fleet_mk_model_path = vllm_config.model_config.model
        self.fleet_mk = None             # FleetMKDecoder, built on first decode
        self._fleet_mk_weight_refs = []  # keep packed tensors alive
        self._fleet_mk_kv_bound = False  # aliased vLLM KV + ptr_table built yet?
        self._fleet_mk_spec = None

    # ── weight loading + fleet_mk setup ─────────────────────────────────────────
    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        self._setup_fleet_mk()
        return loaded

    def _setup_fleet_mk(self):
        from .packing import import_mirage_packers

        S = load_spec(self.FLEET_MK_ARCH)
        self._fleet_mk_spec = S
        dev = next(self.parameters()).device
        packers = import_mirage_packers()

        # Weight-footprint instrumentation. The whole point of sourcing fleet_mk's
        # weights from vLLM's live modules is to drop the second full-checkpoint
        # load, so the footprint has to be MEASURED, not inferred from tensor
        # arithmetic. Off by default (a reset_peak_memory_stats would clobber
        # whatever else is measuring); FLEET_MK_MEM_TRACE=1 to enable.
        mem_trace = os.environ.get("FLEET_MK_MEM_TRACE") == "1"
        if mem_trace:
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            mem_before = torch.cuda.memory_allocated(dev)
            print(f"[FLEET_MK_MEM] before pack: allocated="
                  f"{mem_before / 2**30:.3f} GiB", flush=True)

        # Model-specific packing (subclass): returns the flat per-layer weight
        # pointer list + tensors to keep alive + LM-head/final-norm tensors.
        packed = self._fleet_mk_pack_weights(packers, S, dev)

        if mem_trace:
            torch.cuda.synchronize(dev)
            after = torch.cuda.memory_allocated(dev)
            peak = torch.cuda.max_memory_allocated(dev)
            print(f"[FLEET_MK_MEM] after pack:  allocated={after / 2**30:.3f} GiB"
                  f"  (+{(after - mem_before) / 2**30:.3f} GiB)", flush=True)
            print(f"[FLEET_MK_MEM] peak during pack: "
                  f"{peak / 2**30:.3f} GiB", flush=True)
        self._weight_ptrs_host = packed["weight_ptrs_host"]
        # Per-layer expert scale-section addresses, when those sections live in
        # their own allocations. None for dense, and for MoE without the split.
        # Carried separately from weight_ptrs_host because that list is indexed
        # by a fixed weights_per_layer stride the kernel shares.
        self._moe_scale_ptrs = packed.get("moe_scale_ptrs")
        self._fleet_mk_weight_refs.extend(packed["weight_refs"])
        # Held here rather than in weight_refs: the hash gate below derives slot
        # names from len(weight_refs) // num_layers, so anything appended there
        # renames every slot instead of failing.
        self._fleet_mk_weight_refs.extend(packed.get("moe_scale_refs") or [])
        self._final_norm_w = packed["final_norm_w"]
        self._lm_head_packed = packed["lm_head_packed"]
        self._lm_head_bias = packed["lm_head_bias"]
        self._fleet_mk_weight_refs += [
            self._final_norm_w, self._lm_head_packed, self._lm_head_bias]

        # Byte-level gate for the weight-sourcing rework. Hashing the packed
        # tensors here -- rather than inside a packer -- checks what the kernel
        # will actually read, including anything the pack path appended. Only the
        # per-layer slots are named; trailing tensors keep their own names.
        if os.environ.get("FLEET_MK_HASH_PACKED"):
            # tools/ is a sibling of fleet_megakernel_vllm/, not a package on the path when
            # vLLM spawns the engine from another cwd.
            import sys
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from tools.hash_packed import MOE_SLOT_NAMES, hash_tensors
            refs = packed["weight_refs"]
            per_layer = len(refs) // S.num_layers if S.num_layers else 0
            named = []
            for i, t in enumerate(refs):
                li, slot = divmod(i, per_layer) if per_layer else (0, i)
                sname = (MOE_SLOT_NAMES[slot]
                         if per_layer == len(MOE_SLOT_NAMES) else f"slot{slot}")
                named.append((f"layer{li:02d}.{sname}", t))
            named += [("final_norm_w", self._final_norm_w),
                      ("lm_head_packed", self._lm_head_packed),
                      ("lm_head_bias", self._lm_head_bias),
                      ("rope_cos", packed.get("cos")),
                      ("rope_sin", packed.get("sin"))]
            hash_tensors(named, os.environ["FLEET_MK_HASH_PACKED"])

        # Workspace buffers. KV caches are aliased onto vLLM's kv_cache on the
        # first decode (zero-copy); max_num_pages sizes only kv_indices.
        max_num_pages = max(1, math.ceil(self._max_model_len / S.page_size))
        self._fleet_mk_buffers = FleetMKBuffers(
            S, bs=1, max_num_pages=max_num_pages, device=dev, alloc_kv=False)

        # RoPE tables. Plain RoPE (cat(freqs, freqs)) is correct for models whose
        # rotary embedding has no scaling (Qwen3). Models with rope_scaling (GPT-OSS
        # YaRN) must instead feed the reference model's actual cos/sin, which the
        # packer returns; use those verbatim when present.
        if packed.get("cos") is not None and packed.get("sin") is not None:
            self._cos, self._sin = packed["cos"], packed["sin"]
            self._fleet_mk_weight_refs += [self._cos, self._sin]
        else:
            self._cos, self._sin = build_rope_tables(
                S.head_dim, S.rope_theta, self._max_model_len + 1, dev)

        self._lib = load_kernel(S)
        getattr(self._lib, S.init_symbol)()

    # ── zero-copy KV: alias vLLM's KV into fleet_mk buffers on first decode ──────
    def _ensure_fleet_mk_kv_bound(self):
        """Alias fleet_mk's per-layer K/V onto vLLM's kv_cache and build the ptr table.

        vLLM allocates the KV cache after load_weights(), so this runs once,
        lazily, on the first decode. With an aiter flash backend the physical
        layout is [2, num_blocks, block_size, num_kv_heads, head_size] contiguous,
        so kv_cache[0]/[1] reshape to fleet_mk's flat [entries, kv_cache_stride] K/V
        with no copy. Requires block_size == PAGE_SIZE (1 vLLM block == 1 fleet_mk
        page).
        """
        if self._fleet_mk_kv_bound:
            return
        S = self._fleet_mk_spec
        k_aliases, v_aliases = [], []
        for li in range(S.num_layers):
            attn = self._fleet_mk_layer_attn(li)
            if li == 0:
                self._assert_flash_backend(attn)
            kvc = self._layer_kv_tensor(attn)
            assert kvc.dim() == 5 and kvc.shape[0] == 2, (
                f"fleet_mk zero-copy needs a (2, num_blocks, block_size, n_kv, "
                f"head_size) KV layout; got {tuple(kvc.shape)}. Launch with the "
                f"aiter unified attention backend.")
            _, num_blocks, block_size, n_kv, head_size = kvc.shape
            assert block_size == S.page_size, (
                f"fleet_mk PAGE_SIZE={S.page_size} but vLLM block_size={block_size}; "
                f"relaunch with block_size={S.page_size}.")
            assert n_kv * head_size == S.kv_cache_stride, (
                f"layer {li}: vLLM KV stride {n_kv}*{head_size}={n_kv*head_size} "
                f"!= fleet_mk kv_cache_stride {S.kv_cache_stride}.")
            entries = num_blocks * block_size
            k_alias = kvc[0].reshape(entries, S.kv_cache_stride)
            v_alias = kvc[1].reshape(entries, S.kv_cache_stride)
            # Zero-copy invariant: the reshape must be a view, not a copy, or the
            # vLLM KV pool is silently duplicated and decode reads stale KV.
            assert (k_alias.data_ptr() == kvc[0].data_ptr()
                    and v_alias.data_ptr() == kvc[1].data_ptr()), (
                f"layer {li}: fleet_mk KV alias is a copy, not a view of vLLM's "
                f"kv_cache -- zero-copy broken.")
            k_aliases.append(k_alias)
            v_aliases.append(v_alias)
        self._fleet_mk_buffers.set_kv_aliases(k_aliases, v_aliases)

        ptr_table = build_ptr_table(S, self._weight_ptrs_host,
                                    self._fleet_mk_buffers, self._moe_scale_ptrs)
        self.fleet_mk = FleetMKDecoder(
            S, self._lib, self._fleet_mk_buffers, ptr_table, self._cos, self._sin,
            self._final_norm_w, self._lm_head_packed, self._lm_head_bias,
            embed_weight=self._fleet_mk_embed_weight() if S.is_moe else None)
        self._fleet_mk_kv_bound = True

    @staticmethod
    def _layer_kv_tensor(attn):
        """This layer's whole KV allocation, across vLLM's two spellings.

        The most dangerous seam in the 0.11 -> 0.27 port. On 0.11.x
        `attn.kv_cache` is a *list* indexed by virtual engine, so `kv_cache[0]`
        meant "this layer's cache". On 0.27.x `Attention.__init__` sets
        `self.kv_cache = torch.tensor([])` and `bind_kv_cache` replaces it with
        the tensor itself, so `kv_cache[0]` is the K half of the first *block*.
        Both index without raising and both yield a tensor -- the old spelling
        on a new vLLM silently binds fleet_mk to 1/num_blocks of the pool at the
        wrong rank, which is wrong KV with no error anywhere.

        So discriminate on type, never on indexability, and let the caller's
        `dim() == 5 and shape[0] == 2` assert catch anything unexpected.
        """
        kvc = attn.kv_cache
        if isinstance(kvc, (list, tuple)):        # 0.11.x: per virtual engine
            assert len(kvc) == 1, (
                f"fleet_mk decode assumes a single virtual engine; got "
                f"{len(kvc)} KV caches on this layer.")
            kvc = kvc[0]
        assert isinstance(kvc, torch.Tensor) and kvc.numel() > 0, (
            f"layer KV cache is {type(kvc).__name__} with "
            f"{getattr(kvc, 'numel', lambda: '?')()} elements -- vLLM has not "
            f"bound it yet. _ensure_fleet_mk_kv_bound must run on the first real "
            f"decode, after bind_kv_cache.")
        return kvc

    @staticmethod
    def _assert_flash_backend(attn):
        # AiterFlashAttentionBackend / RocmAiterUnifiedAttentionBackend: 0.25 and
        # earlier allocated fleet_mk's split layout directly. FleetMKAttentionBackend
        # (backend.py) is how that layout is recovered on 0.26+, where the stock
        # aiter shape went packed -- same stride-aware write path, different
        # get_kv_cache_shape. All three write KV via reshape_and_cache_flash and
        # are non-reordered, which is what zero-copy actually requires.
        name = getattr(attn.attn_backend, "__name__", "")
        ok = {"AiterFlashAttentionBackend", "RocmAiterUnifiedAttentionBackend",
              "FleetMKAttentionBackend"}
        assert name in ok, (
            f"fleet_mk zero-copy KV requires an aiter flash backend "
            f"(reshape_and_cache_flash, non-reordered); got {name!r}. Relaunch "
            f"with VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 "
            f"(and, on vLLM 0.26+, attention_backend=CUSTOM).")

    # ── forward: route prefill vs decode ─────────────────────────────────────
    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None):
        attn_md = self._get_attn_metadata()

        # No metadata (profiling/dummy run) or multi-token query -> stock path.
        if attn_md is None or getattr(attn_md, "max_query_len", 2) != 1:
            return super().forward(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)

        # Decode: one token, batch size 1. Alias vLLM's KV on the first decode.
        self._ensure_fleet_mk_kv_bound()
        S = self._fleet_mk_spec
        cur_pos = int(positions[0].item())

        # Embedding. The fused MoE kernel looks the token up on-device (worker
        # (0,0) writes it into layer 0's residual behind a device-wide barrier),
        # so it takes a token id and any host-side vector we prepared would just
        # be overwritten. The dense kernel expects the vector.
        embed, token_id = None, None
        if S.is_moe:
            assert inputs_embeds is None, (
                "fleet_mk MoE decode embeds on-device and cannot accept "
                "inputs_embeds; run this request through the stock path.")
            token_id = int(input_ids[0].item())
        else:
            if inputs_embeds is not None:
                embed = inputs_embeds[0]
            else:
                embed = self._fleet_mk_embed_tokens(input_ids)[0]
            # The kernel runs on the padded hidden dim; zero-pad past hidden_size
            # (dense: pw == hidden_size, so this is a no-op today).
            pw = S.padded_hidden_size
            if embed.shape[-1] != pw:
                padded = torch.zeros(pw, dtype=embed.dtype, device=embed.device)
                padded[:S.hidden_size] = embed
                embed = padded
        # Runs the megakernel (fuses all layers + final RMSNorm), reading prefill
        # KV and appending this token's KV in vLLM's block. It writes the
        # post-final-norm hidden state into buf_lm_norm_scratch, and -- on the MoE
        # path -- the full bf16 logit row into buf_logits, which compute_logits
        # below returns in place of vLLM's own bf16 lm_head. The sampler
        # (temperature/top-p/penalties) still runs on top, so this replaces only
        # the redundant GEMV, not sampling.
        block_table = attn_md.block_table[0]
        self.fleet_mk.decode_step(embed, cur_pos, block_table=block_table,
                               sync=False, token_id=token_id)
        self._fleet_mk_fused_logits = S.is_moe
        # Slice off the MoE padding so the returned width matches vLLM's lm_head.
        if self.fleet_mk.use_kernel_argmax:
            # The greedy sampler consumes buf_argmax_out, not hidden states.
            # Returning a one-element placeholder avoids cloning 2880 bf16
            # values and makes vLLM's logits_indices gather trivial.
            return self._fleet_mk_buffers.buf_logits[:input_ids.shape[0], :1]
        hs = self._fleet_mk_buffers.buf_lm_norm_scratch[:input_ids.shape[0], :S.hidden_size]
        return hs.clone()

    def compute_logits(self, hidden_states):
        """Return fleet_mk's fused logits when the last forward() was a fleet_mk decode.

        The megakernel's argmax epilogue computes `sum + bias` for every vocab
        entry and throws all but the max away; with logits_output non-null it also
        stores that row. Recomputing it with vLLM's bf16 lm_head is a 1.16 GB
        weight read (0.200 ms/token measured) for a value fleet_mk already had, so
        the fused row is strictly cheaper and, being the same arithmetic on the
        same MXFP4 weights the argmax used, cannot disagree with fleet_mk's own
        greedy pick.

        The flag is consumed (not just read) so any path that reaches
        compute_logits without a preceding fleet_mk decode -- prefill, a dummy run,
        a stock-path fallback -- falls through to vLLM's lm_head. Silently
        returning a stale row there would be wrong output with no error.
        """
        if not getattr(self, "_fleet_mk_fused_logits", False):
            return super().compute_logits(hidden_states)
        self._fleet_mk_fused_logits = False
        S = self._fleet_mk_spec
        n = hidden_states.shape[0]
        if self.fleet_mk.use_kernel_argmax:
            # vLLM passes this object unchanged to Sampler.forward.  The payload
            # is intentionally one element: greedy.py consumes the attached
            # device token and never reads logits.  Unsupported sampler features
            # fail there with a precise error instead of using a bogus row.
            tagged = self._fleet_mk_buffers.buf_logits[:n, :1]
            tagged._fleet_mk_argmax = self._fleet_mk_buffers.buf_argmax_out
            return tagged
        return self._fleet_mk_buffers.buf_logits[:n, :S.vocab_size]

    # ── helpers ──────────────────────────────────────────────────────────────
    def _get_attn_metadata(self):
        try:
            md = get_forward_context().attn_metadata
        except AssertionError:
            return None
        if md is None:
            return None
        if isinstance(md, dict):
            if not md:
                return None
            return next(iter(md.values()))
        return md

    # ── subclass hooks ───────────────────────────────────────────────────────
    def _fleet_mk_layer_attn(self, li):
        raise NotImplementedError

    def _fleet_mk_pack_weights(self, packers, spec, dev):
        raise NotImplementedError

    def _fleet_mk_embed_tokens(self, input_ids):
        """Return the (unpadded) token embeddings [num_tokens, hidden_size].

        Uses whichever spelling the installed vLLM has: `embed_input_ids` on
        0.27.x, `get_input_embeddings` on 0.11.x. Both route to the right
        submodule (dense: model.embed_tokens; MoE/GPT-OSS: model.embedding).
        A rename, not a semantic change -- but 0.27.x removed the old name
        outright, so binding to one spelling breaks the other version. The mixin
        zero-pads the result to padded_hidden_size. Subclasses may override if a
        model needs special handling.

        Unused on the MoE path -- see _fleet_mk_embed_weight.
        """
        return self._fleet_mk_embed_fn()(input_ids)

    def _fleet_mk_embed_fn(self):
        """The model's token-embedding callable, across vLLM spellings."""
        for name in ("embed_input_ids", "get_input_embeddings"):
            fn = getattr(self, name, None)
            if fn is not None:
                return fn
        raise AttributeError(
            f"{type(self).__name__} exposes neither embed_input_ids (vLLM "
            f"0.27+) nor get_input_embeddings (0.11.x); override "
            f"_fleet_mk_embed_tokens.")

    def _fleet_mk_embed_weight(self):
        """The raw embedding matrix, for kernels that do the lookup on-device.

        The kernel indexes it as [vocab_size, hidden_size] bf16 with a plain row
        stride, so it must be the unsharded, contiguous weight -- a
        tensor-parallel shard or a non-contiguous view reads the wrong rows and
        produces fluent-but-wrong text rather than an error.
        """
        mod = getattr(self.model, "embedding", None)
        if mod is None:
            mod = getattr(self.model, "embed_tokens", None)
        # Report what the model *does* expose rather than probing for a specific
        # spelling: this message previously interpolated self.get_input_embeddings,
        # which 0.27.x removed -- so the diagnostic would itself raise
        # AttributeError, replacing a clear failure with a confusing one.
        assert mod is not None and hasattr(mod, "weight"), (
            f"could not find the embedding module on {type(self).__name__}; "
            f"override _fleet_mk_embed_weight (model has: "
            f"{[n for n, _ in self.model.named_children()]})")
        w = mod.weight.data
        assert w.is_contiguous() and w.dim() == 2, (w.shape, w.is_contiguous())
        return w
