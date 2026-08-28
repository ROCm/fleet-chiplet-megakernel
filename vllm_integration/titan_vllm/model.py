"""Thin titan model subclasses: dense (Qwen3) and MoE (GPT-OSS).

All the shared machinery (kernel load, buffers, RoPE, zero-copy KV aliasing,
prefill/decode routing) lives in TitanModelMixin (mixin.py), parameterized by a
ModelSpec (spec.py). Each subclass supplies only what is model-specific:

  * TITAN_ARCH            -- vLLM arch name load_spec() keys on.
  * _titan_layer_attn(li) -- where the per-layer Attention module lives.
  * _titan_pack_weights() -- MXFP4-pack this model's weights into titan's layout.

The mixin is listed FIRST in each base list so its forward()/load_weights()
override the stock model's while super() still reaches the stock model.
"""

import torch

from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

from .mixin import TitanModelMixin
from .packing import pack_layer_weights, pack_lm_head


class TitanQwen3ForCausalLM(TitanModelMixin, Qwen3ForCausalLM):
    """Dense Qwen3-8B: fused qkv/gate_up projections, per-head QK-norm."""

    TITAN_ARCH = "Qwen3ForCausalLM"

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._titan_setup_state(vllm_config)

    def _titan_layer_attn(self, li):
        return self.model.layers[li].self_attn.attn

    def _titan_pack_weights(self, packers, spec, dev):
        q_end = spec.num_q_heads * spec.head_dim              # 4096
        k_end = q_end + spec.num_kv_heads * spec.head_dim     # 5120
        v_end = k_end + spec.num_kv_heads * spec.head_dim     # 6144

        weight_ptrs_host, refs = [], []
        for li in range(spec.num_layers):
            layer = self.model.layers[li]
            attn = layer.self_attn
            qkv_w = attn.qkv_proj.weight.data                 # [6144, 4096] fused
            w_q = qkv_w[:q_end]
            w_k = qkv_w[q_end:k_end]
            w_v = qkv_w[k_end:v_end]
            w_o = attn.o_proj.weight.data                     # [4096, 4096]
            gu_w = layer.mlp.gate_up_proj.weight.data         # [24576, 4096] fused
            w_gate = gu_w[:spec.intermediate_size]
            w_up = gu_w[spec.intermediate_size:]
            w_down = layer.mlp.down_proj.weight.data          # [4096, 12288]

            packed = pack_layer_weights(
                w_q, w_k, w_v, w_o, w_gate, w_up, w_down,
                layer.input_layernorm.weight.data,
                layer.post_attention_layernorm.weight.data,
                attn.q_norm.weight.data,
                attn.k_norm.weight.data,
                packers, spec,
            )
            refs.extend(packed)
            weight_ptrs_host.extend(t.data_ptr() for t in packed)

        final_norm_w = self.model.norm.weight.data.contiguous()
        lm_head_packed = pack_lm_head(self.lm_head.weight.data, packers, spec)
        lm_head_bias = torch.zeros(
            1, spec.padded_vocab_size, dtype=torch.bfloat16, device=dev)
        return dict(
            weight_ptrs_host=weight_ptrs_host,
            weight_refs=refs,
            final_norm_w=final_norm_w,
            lm_head_packed=lm_head_packed,
            lm_head_bias=lm_head_bias,
        )


def _make_gptoss_class():
    """Build the MoE subclass lazily so a missing GptOss arch in the installed
    vLLM doesn't break importing this module for the dense path."""
    from vllm.model_executor.models.gpt_oss import GptOssForCausalLM

    class TitanGptOssForCausalLM(TitanModelMixin, GptOssForCausalLM):
        """MoE GPT-OSS 120B: 128 experts top-4, attention sinks, per-layer SW."""

        TITAN_ARCH = "GptOssForCausalLM"

        def __init__(self, *, vllm_config, prefix: str = ""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self._titan_setup_state(vllm_config)

        def _titan_layer_attn(self, li):
            return self.model.layers[li].attn.attn

        def _titan_pack_weights(self, packers, spec, dev):
            # Every tensor here comes from vLLM's own live modules. Titan used to
            # load a second full 120B (mirage's reference GptOssForCausalLM) purely
            # to pack from; that copy is gone, so the checkpoint is read once.
            #
            # This works because of *when* we run: base_loader calls
            # load_weights() -- whose tail invokes _setup_titan -- before
            # process_weights_after_loading(), so vLLM's parameters are still the
            # raw checkpoint values, not the post-load swizzle. Sourcing them here
            # is byte-identical to re-reading the checkpoint, and was gated as
            # such: 470 non-expert and 216 expert tensors compared bit-exact
            # against the reference model before it was deleted.
            from .moe_layout import MoeLayout
            from .packing_moe import pack_lm_head_moe, pack_moe_layer

            cos, sin = self._titan_rope_from_vllm(spec, dev)

            # Expert-buffer geometry: row pitch, expert pitch, section split,
            # and whether the data section is aliased from vLLM rather than
            # packed. Printed unconditionally -- three of those knobs are
            # compiled into the .so and nothing verifies agreement at runtime,
            # so this banner is the only place a mismatch is visible before it
            # becomes fluent-looking garbage.
            layout = MoeLayout(spec)
            print(layout.describe(), flush=True)

            weight_ptrs_host, refs = [], []
            scale_refs, moe_scale_ptrs = [], []
            for li in range(spec.num_layers):
                vlayer = self.model.layers[li]      # vLLM TransformerBlock
                vattn = vlayer.attn                 # OAIAttention
                w_q, w_k, w_v, q_b, k_b, v_b = self._titan_split_qkv(vattn, spec)
                packed, scales = pack_moe_layer(
                    w_q=w_q, w_k=w_k, w_v=w_v,
                    q_bias=q_b, k_bias=k_b, v_bias=v_b,
                    w_o=vattn.o_proj.weight.data,
                    o_bias=None if vattn.o_proj.bias is None else vattn.o_proj.bias.data,
                    norm_pre=vlayer.input_layernorm.weight.data,
                    norm_post=vlayer.post_attention_layernorm.weight.data,
                    router_w=vlayer.mlp.router.weight.data,
                    router_b=vlayer.mlp.router.bias.data,
                    sinks=vattn.sinks.data,
                    spec=spec,
                    packers=packers,
                    layout=layout,
                    **self._titan_experts_from_vllm(li, spec),
                    **self._titan_foreign_experts(li, layout),
                )
                refs.extend(packed)
                weight_ptrs_host.extend(t.data_ptr() for t in packed)
                # Scale sections are kept in their own ref list, NOT appended to
                # `refs`: TITAN_HASH_PACKED names slots by `len(refs) //
                # num_layers`, so two extra entries per layer would silently
                # rename every slot rather than fail. They still need a
                # reference held -- they are ordinary titan allocations and
                # nothing else holds them.
                scale_refs.extend(t for t in scales if t is not None)
                moe_scale_ptrs.append(
                    None if scales[0] is None
                    else (scales[0].data_ptr(), scales[1].data_ptr()))

            final_norm_w = packers["pad_weight_1d"](
                self.model.norm.weight.data, spec.padded_hidden_size,
                pad_value=0.0).contiguous()
            lm_head_packed, lm_head_bias = pack_lm_head_moe(
                self.lm_head.weight.data, spec, packers)

            return dict(
                weight_ptrs_host=weight_ptrs_host,
                weight_refs=refs,
                moe_scale_refs=scale_refs,
                moe_scale_ptrs=moe_scale_ptrs if moe_scale_ptrs[0] else None,
                final_norm_w=final_norm_w,
                lm_head_packed=lm_head_packed,
                lm_head_bias=lm_head_bias,
                cos=cos,
                sin=sin,
            )

        def _titan_foreign_experts(self, li, layout):
            """vLLM's own expert DATA tensors, when aliasing them.

            `{}` otherwise, so the call site reads the same either way.

            These are the tensors titan would otherwise re-pack byte for byte.
            Titan's hook runs BEFORE `process_weights_after_loading`, so what is
            captured here is the raw checkpoint layout -- and the address must
            therefore survive that call, since the kernel dereferences it much
            later. It does: `tools/probe_vllm_expert_layout.py` measured the
            storage before and after and found both the address and every byte
            unchanged. ROCm's `StridedLayout` swizzle is identity and the only
            shape change is a transposed *view*, so a linear reader still sees
            `[E, rows, K/2]` row-major -- titan's own order.

            That was measured rather than read off the source because the
            failure mode is silent: the AITER path de-interleaves w13 with an
            in-place `copy_`, which keeps `data_ptr` identical while rewriting
            every byte, so an address-only check would call it unchanged and
            send the kernel at permuted rows.

            `.data` is taken, not `.contiguous()`. Only the storage address is
            ever used, and materializing anything here would be the 60 GiB copy
            this exists to avoid.
            """
            if not layout.alias:
                return {}
            exp = self.model.layers[li].mlp.experts.routed_experts
            return dict(foreign_w13=exp.w13_weight.data,
                        foreign_w2=exp.w2_weight.data)

        def _titan_experts_from_vllm(self, li, spec):
            """The six raw MXFP4 expert tensors, sliced out of vLLM's padded ones.

            vLLM (0.27.1) rounds both hidden and intermediate up to 3072 and keeps
            the block bytes on a flat axis, so its shapes are

              w13_weight [E, 2*3072, 3072/2]   w13_weight_scale [E, 2*3072, 3072/32]
              w2_weight  [E, 3072,   3072/2]   w2_weight_scale  [E, 3072,   3072/32]

            while `pack_mxfp4_workgroup` wants the checkpoint's own
            [E, out, num_blocks, 16] blocks at the *true* dims. Two adjustments,
            both pure views:

              * split the byte axis back into (num_blocks, 16) -- 16 bytes is one
                MXFP4 block of 32 values, so this regroups, it does not reorder;
              * slice off vLLM's pad rows/blocks and let pack re-pad to titan's
                2944. Adopting 3072 instead would add ~8.9% MoE weight traffic in
                a bandwidth-bound phase and trip the K-keyed static_asserts on
                fleet's linear-load paths.

            Row order is untouched: `hf_to_vllm_mapper` renames
            gate_up_proj_blocks -> w13_weight with no permutation, so the gate/up
            interleave the kernel expects is already there. This whole function was
            gated byte-for-byte against mirage's reference model -- 216 of 216
            tensors exact across 36 layers -- before that model was deleted.
            """
            exp = self.model.layers[li].mlp.experts.routed_experts
            inter, hidden = spec.intermediate_size, spec.hidden_size
            nb_h, nb_i = hidden // 32, inter // 32       # true block counts

            def blocks(t, rows, nblk):
                E, _, nbytes = t.shape
                return t.view(E, -1, nbytes // 16, 16)[:, :rows, :nblk]

            return dict(
                gate_up_blocks=blocks(exp.w13_weight.data, 2 * inter, nb_h),
                gate_up_scales=exp.w13_weight_scale.data[:, :2 * inter, :nb_h],
                gate_up_bias=exp.w13_bias.data[:, :2 * inter],
                down_blocks=blocks(exp.w2_weight.data, hidden, nb_i),
                down_scales=exp.w2_weight_scale.data[:, :hidden, :nb_i],
                down_bias=exp.w2_bias.data[:, :hidden],
            )

        def _titan_split_qkv(self, vattn, spec):
            """Split vLLM's fused QKVParallelLinear back into q/k/v weight+bias.

            pack_moe_layer re-interleaves Q/K/V by KV group itself, so it wants the
            three matrices separately -- the same shape mirage's reference model
            exposes as q_proj/k_proj/v_proj. QKVParallelLinear stacks them in
            [q | k | v] row order (see OAIAttention.forward's qkv.split), so plain
            row slices recover them; at TP=1 there is no per-rank shuffle to undo.
            """
            q_end = spec.num_q_heads * spec.head_dim               # 4096
            k_end = q_end + spec.num_kv_heads * spec.head_dim      # 4608
            v_end = k_end + spec.num_kv_heads * spec.head_dim      # 5120
            w = vattn.qkv_proj.weight.data
            assert w.shape[0] == v_end == spec.qkv_output_size, \
                (tuple(w.shape), v_end, spec.qkv_output_size)
            b = None if vattn.qkv_proj.bias is None else vattn.qkv_proj.bias.data
            return (w[:q_end], w[q_end:k_end], w[k_end:v_end],
                    None if b is None else b[:q_end],
                    None if b is None else b[q_end:k_end],
                    None if b is None else b[k_end:v_end])

        def _titan_rope_from_vllm(self, spec, dev):
            """cos/sin tables from vLLM's own YaRN rotary embedding.

            vLLM caches `cat(cos, sin)` along the last dim, both halves rotary_dim/2
            wide -- GPT-OSS is un-doubled neox, exactly what titan's kernel indexes.
            The cache is built in fp32 and is far longer than we need (it spans
            max_position_embeddings x scaling_factor), so slice to the positions this
            engine can actually reach and cast to titan's bf16.
            """
            rope = self.model.layers[0].attn.rotary_emb
            cache = rope.cos_sin_cache                     # [N, rotary_dim] fp32
            rows = self._max_model_len + 1
            assert cache.shape[0] >= rows, (cache.shape, rows)
            cos_raw, sin_raw = cache[:rows].chunk(2, dim=-1)
            pad = spec.head_dim - cos_raw.shape[-1]
            cos = torch.nn.functional.pad(cos_raw, (0, pad)).to(
                device=dev, dtype=torch.bfloat16).contiguous()
            sin = torch.nn.functional.pad(sin_raw, (0, pad)).to(
                device=dev, dtype=torch.bfloat16).contiguous()
            return cos, sin

    return TitanGptOssForCausalLM
