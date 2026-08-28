"""M1/M2 bring-up + parity harness.

Runs Qwen3-8B greedy generation through vLLM and prints the generated token ids
and text. Enable titan via `VLLM_PLUGINS=titan`; disable (stock baseline) via
`VLLM_PLUGINS=` (empty). Compare the two token streams for parity.

Titan's zero-copy KV path requires an aiter flash attention backend
(reshape_and_cache_flash, non-reordered) and block_size == titan PAGE_SIZE; the
harness reads PAGE_SIZE from the model's spec and sets both when the plugin is on.
Override via TITAN_BLOCK_SIZE. TITAN_TEMP / TITAN_SEED drive vLLM's sampler.
TITAN_MODEL picks the model (dense Qwen3 or MoE GPT-OSS).

Usage:
  VLLM_PLUGINS=       python -m titan_vllm.harness   # stock baseline
  VLLM_PLUGINS=titan  python -m titan_vllm.harness   # titan decode (aiter)
"""

import os

# MUST precede `from vllm import ...`: vLLM resolves the ROCm attention backend at
# import time, so setting these inside main() lands too late and decode dies with
# "got 'RocmAttentionBackend'" from _assert_flash_backend. Unconditional (not gated
# on VLLM_PLUGINS) because the stock baseline must run the SAME backend as titan or
# the A/B compares two different attention implementations. See _assert_flash_backend
# in mixin.py for why only the aiter backends are zero-copy compatible.
os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION", "1")

from vllm import LLM, SamplingParams  # noqa: E402


def _titan_page_size(model):
    """titan PAGE_SIZE (== required vLLM block_size) for the selected model."""
    from titan_vllm.spec import load_spec
    name = model.lower()
    arch = ("GptOssForCausalLM" if "gpt-oss" in name or "gpt_oss" in name
            else "Qwen3ForCausalLM")
    return load_spec(arch).page_size


def _custom_attention_backend():
    """AttentionBackendEnum.CUSTOM, or None on a vLLM without the registry.

    0.11.x has no backend registry and no `attention_backend` engine arg; its
    stock aiter backends already allocate titan's split KV layout, so there is
    nothing to override and passing the kwarg would be a TypeError.
    """
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except ImportError:
        return None
    return AttentionBackendEnum.CUSTOM


def main():
    prompt = os.environ.get("TITAN_PROMPT", "Tell me the history of America.")
    # 20 was enough for token-id parity but truncates mid-sentence, which reads as
    # "no output" when you are eyeballing the text. 300 gets past the analysis
    # channel into the actual answer. Parity checks only compare the leading ids,
    # so a longer default costs nothing but a few seconds.
    max_tokens = int(os.environ.get("TITAN_MAX_TOKENS", "1000"))
    model = os.environ.get("TITAN_MODEL", "Qwen/Qwen3-8B")
    temperature = float(os.environ.get("TITAN_TEMP", "0.0"))
    seed = int(os.environ.get("TITAN_SEED", "0"))
    titan_on = bool(os.environ.get("VLLM_PLUGINS"))

    llm_kwargs = dict(
        model=model,
        max_num_seqs=1,
        enforce_eager=True,
        max_model_len=2048,
        dtype="bfloat16",
        # vLLM sizes the KV pool as a fraction of *total* VRAM and refuses to
        # start if that much is not free, so the default 0.9 fails outright when
        # anything else shares the GPU -- including another container, whose
        # allocation shows up in the free-memory check but not in `ps`. The
        # error names the utilization, not the neighbour, which reads like a
        # titan bug and is not one. Overridable so a quiet GPU can still use 0.9.
        gpu_memory_utilization=float(
            os.environ.get("TITAN_GPU_MEM_UTIL", "0.9")),
    )
    if titan_on:
        # Select the aiter FA backend on V1/ROCm (gfx9): it writes KV via
        # reshape_and_cache_flash (non-reordered) in titan's layout. This is chosen
        # by the AITER + AITER_MHA flags, NOT VLLM_ATTENTION_BACKEND (that string
        # forces the V0 engine). block_size == PAGE_SIZE so 1 vLLM block == 1 titan
        # page (enables zero-copy KV aliasing).
        #
        # Use the aiter UNIFIED-ATTENTION backend (Triton unified_attention), not
        # the MHA/flash one. Both write KV via reshape_and_cache_flash in titan's
        # non-reordered [2, num_blocks, block_size, n_kv, head] layout, so both are
        # zero-copy compatible -- but they cap block_size very differently:
        #   * MHA (rocm_aiter_fa) calls aiter paged_attention_v1, whose kernel
        #     static_asserts BLOCK_SIZE <= k_thread_per_block (== 32) -> unusable
        #     for large pages (this is why vLLM's ROCm default block_size is 16).
        #   * Unified (rocm_aiter_unified_attn) uses a Triton kernel needing
        #     block_size*1024 B of LDS; the 160 KB cap gives block_size <= 160.
        # block_size == PAGE_SIZE so 1 vLLM block == 1 titan page (zero-copy KV).
        # PAGE_SIZE is now 16 -- vLLM's own ROCm default -- so this override is
        # a no-op in practice and titan binds to a stock-configured engine. It
        # stays explicit because the zero-copy assert depends on it holding.
        # (The two AITER env vars themselves are set at module import above -- vLLM
        # reads them before this function runs.)
        llm_kwargs["block_size"] = int(
            os.environ.get("TITAN_BLOCK_SIZE", str(_titan_page_size(model))))
        # Hybrid models (GPT-OSS: sliding-window on even layers, full on odd) make
        # vLLM's hybrid KV-cache manager overlay the two attention types on SHARED
        # physical KV memory with SEPARATE per-group block tables. titan's zero-copy
        # decode uses ONE block table for all layers and assumes distinct per-layer
        # KV tensors -- valid only for a uniform (single-group) cache. Disabling the
        # hybrid manager unifies every layer to FullAttentionSpec: one shared block
        # table, one KV tensor per layer. titan's own kernel still applies the
        # sliding-window mask, so only the KV-memory saving is lost (negligible at
        # our context lengths). No-op for already-uniform models (Qwen3).
        llm_kwargs["disable_hybrid_kv_cache_manager"] = True
        # On vLLM 0.26+ the stock aiter backends allocate the packed KV layout
        # (num_blocks, n_kv, block_size, 2*head_size), which titan's flat view
        # cannot alias. plugin.register() registers TitanAttentionBackend under
        # the CUSTOM enum member -- but registration is not selection, so ask
        # for it here. It subclasses the aiter unified backend and changes only
        # get_kv_cache_shape and _split_kv_cache, so the attention math and the
        # KV write path are stock; see backend.py.
        be = _custom_attention_backend()
        if be is not None:
            llm_kwargs["attention_backend"] = be

    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    out = llm.generate(
        [text],
        SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed),
    )
    gen = out[0].outputs[0]
    label = "titan" if os.environ.get("VLLM_PLUGINS") else "stock"
    print(f"\n===== [{label}] generated token ids =====")
    print(list(gen.token_ids))
    print(f"===== [{label}] text =====")
    print(gen.text)


if __name__ == "__main__":
    main()
