"""Decode-latency benchmark: fleet_mk vs stock vLLM (Qwen3-8B, greedy).

Measures per-decode-token latency with a prefill-cancelling method: time
generate() for a short and a long output of the SAME prompt, then
  ms/token = 1000 * (T_long - T_short) / (N_long - N_short)
so the (identical) prefill + fixed call overhead cancels out.

  VLLM_PLUGINS=       python -m fleet_megakernel_vllm.bench   # stock baseline
  VLLM_PLUGINS=fleet_mk  python -m fleet_megakernel_vllm.bench   # fleet_mk decode
"""

import os

# MUST precede `from vllm import ...` -- vLLM resolves the ROCm attention backend at
# import time. See harness.py for the full rationale. Unconditional so the stock
# baseline runs the SAME attention backend as fleet_mk; otherwise the A/B silently
# compares two different attention implementations, not two decode paths.
os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION", "1")

import time  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402


def _fleet_mk_page_size(model):
    """fleet_mk PAGE_SIZE (== required vLLM block_size) for the selected model."""
    from fleet_megakernel_vllm.spec import load_spec
    name = model.lower()
    arch = ("GptOssForCausalLM" if "gpt-oss" in name or "gpt_oss" in name
            else "Qwen3ForCausalLM")
    return load_spec(arch).page_size


def _custom_attention_backend():
    """AttentionBackendEnum.CUSTOM, or None on a vLLM without the registry.

    Mirrors harness.py. Without this, a 0.27.x fleet_mk run silently falls back to
    whatever backend the platform picks (TRITON_ATTN here, since aiter does not
    import under this venv's torch) and binds fleet_mk's KV against that backend's
    layout -- so the measurement would be of a configuration that never ran
    correctly. 0.11.x has no registry and needs no override; passing the kwarg
    there is a TypeError.
    """
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except ImportError:
        return None
    return AttentionBackendEnum.CUSTOM


def _time_generate(llm, text, n_tokens):
    # ignore_eos already guarantees n_tokens outputs.  min_tokens=n_tokens
    # was redundant and installs an argmax-changing EOS mask, which correctly
    # makes Fleet's device-side greedy fast path reject the request.
    sp = SamplingParams(temperature=0.0, max_tokens=n_tokens,
                        ignore_eos=True, seed=0)
    t0 = time.perf_counter()
    out = llm.generate([text], sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    produced = len(out[0].outputs[0].token_ids)
    return dt, produced


def main():
    prompt = os.environ.get("FLEET_MK_PROMPT", "Tell me the history of America.")
    model = os.environ.get("FLEET_MK_MODEL", "Qwen/Qwen3-8B")
    n_short = int(os.environ.get("BENCH_SHORT", "8"))
    n_long = int(os.environ.get("BENCH_LONG", "128"))
    reps = int(os.environ.get("BENCH_REPS", "3"))
    fleet_mk_on = bool(os.environ.get("VLLM_PLUGINS"))
    # enforce_eager=1 disables hipGraph capture. It used to be hardcoded on, which
    # silently handicapped the stock baseline: stock vLLM's decode is a long tail of
    # small kernel launches and graph capture is most of its performance. fleet_mk's
    # decode is ONE megakernel launch, so it has almost nothing to gain -- meaning
    # eager mode flatters fleet_mk by a large factor. Default off; set BENCH_EAGER=1 to
    # reproduce the old numbers. Always state which mode a number came from.
    eager = os.environ.get("BENCH_EAGER", "0") == "1"

    llm_kwargs = dict(
        model=model,
        max_num_seqs=1,
        enforce_eager=eager,
        max_model_len=2048,
        dtype="bfloat16",
        # Same knob harness.py exposes. Without it the default utilization
        # leaves too little room after a 131 GiB 120B load and KV-cache init
        # fails during engine startup -- which looks like a kernel fault in the
        # log but is a configuration error before any fleet_mk code runs.
        gpu_memory_utilization=float(
            os.environ.get("FLEET_MK_GPU_MEM_UTIL", "0.9")),
    )
    if fleet_mk_on:
        if int(os.environ.get("FLEET_MK_PERSIST", "1")) > 1:
            llm_kwargs["async_scheduling"] = False
        # The two AITER env vars are set at module import (vLLM reads them before
        # this runs). Only the vLLM-constructor settings belong here.
        llm_kwargs["block_size"] = int(
            os.environ.get("FLEET_MK_BLOCK_SIZE", str(_fleet_mk_page_size(model))))
        # Unify hybrid (GPT-OSS sliding+full) layers into one uniform KV-cache group
        # so fleet_mk's single-block-table zero-copy decode reads the right per-layer
        # KV. No-op for uniform models (Qwen3). See harness.py for the full rationale.
        llm_kwargs["disable_hybrid_kv_cache_manager"] = True
        # Registration (in plugin.py) is not selection -- the engine must also be
        # launched asking for CUSTOM. See harness.py and the 0.27.1 port article.
        be = _custom_attention_backend()
        if be is not None:
            llm_kwargs["attention_backend"] = be

    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    # Warmup (JIT/compile/allocs) so the first timed run isn't penalized.
    _time_generate(llm, text, n_long)

    short_times, long_times = [], []
    for _ in range(reps):
        ts, ns = _time_generate(llm, text, n_short)
        tl, nl = _time_generate(llm, text, n_long)
        assert ns == n_short and nl == n_long, (ns, nl)
        short_times.append(ts)
        long_times.append(tl)

    ts = min(short_times)   # best-of-reps (least noise)
    tl = min(long_times)
    steps = n_long - n_short
    ms_per_tok = 1000.0 * (tl - ts) / steps
    label = "fleet_mk" if fleet_mk_on else "stock"

    print(f"\n===== [{label}] decode latency =====")
    print(f"graph capture         : {'OFF (enforce_eager)' if eager else 'ON'}")
    print(f"prompt tokens         : {len(tok(text).input_ids)}")
    print(f"short/long out tokens : {n_short} / {n_long}  ({steps} decode steps diffed)")
    print(f"T_short (best of {reps})  : {ts*1000:8.2f} ms")
    print(f"T_long  (best of {reps})  : {tl*1000:8.2f} ms")
    print(f"decode latency        : {ms_per_tok:8.3f} ms/token")
    print(f"decode throughput     : {1000.0/ms_per_tok:8.2f} tok/s")


if __name__ == "__main__":
    main()
