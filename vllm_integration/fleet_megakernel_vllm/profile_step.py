"""Attribute fleet_mk's per-decode-token time under vLLM to specific GPU kernels.

`bench.py` says *how much* a decode token costs end-to-end; this says *where* it
goes. The gap it exists to explain: the megakernel's own GPU time is ~2.38 ms
(FLEET_MK_PROFILE=1) against ~3.21 ms end-to-end, so ~0.83 ms/token is everything
else -- vLLM's bf16 lm_head, the sampler, the scheduler, and host round-trips.
Guessing which of those dominates is exactly the mistake this repo keeps paying
for, so: profile it.

  VLLM_PLUGINS=fleet_mk FLEET_MK_MODEL=... python -m fleet_megakernel_vllm.profile_step

Reports, per decode token:
  * top GPU kernels by total device time, with the megakernel identified
  * the non-megakernel GPU total -- the part a fused-logits ABI could remove
  * wall time minus GPU time -- host overhead that fusing logits would NOT remove

Writes the raw chrome trace to /tmp/fleet_megakernel_vllm_trace.json for Perfetto.
"""

import os
import time

import torch
from torch.profiler import ProfilerActivity, profile
from vllm import LLM, SamplingParams


# The megakernel launches under this name; everything else on the decode stream
# is vLLM's own work (lm_head GEMV, sampler elementwise, KV bookkeeping).
_MEGAKERNEL_HINTS = ("gpt_oss", "qwen3", "llama3", "fleet_mk", "megakernel")


def _is_megakernel(name):
    low = name.lower()
    return any(h in low for h in _MEGAKERNEL_HINTS)


def _fleet_mk_page_size(model):
    from fleet_megakernel_vllm.spec import load_spec
    name = model.lower()
    arch = ("GptOssForCausalLM" if "gpt-oss" in name or "gpt_oss" in name
            else "Qwen3ForCausalLM")
    return load_spec(arch).page_size


def main():
    model = os.environ.get("FLEET_MK_MODEL", "Qwen/Qwen3-8B")
    prompt = os.environ.get("FLEET_MK_PROMPT", "Tell me the history of America.")
    n_tokens = int(os.environ.get("PROFILE_TOKENS", "64"))
    fleet_mk_on = bool(os.environ.get("VLLM_PLUGINS"))
    # Graph capture must be OFF: fleet_mk cannot be captured today (the .item()
    # host syncs in mixin.forward), and a captured graph would also collapse the
    # per-kernel attribution this script exists to produce.
    llm_kwargs = dict(model=model, max_num_seqs=1, enforce_eager=True,
                      max_model_len=2048, dtype="bfloat16")
    if fleet_mk_on:
        os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
        os.environ.setdefault("VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION", "1")
        llm_kwargs["block_size"] = int(
            os.environ.get("FLEET_MK_BLOCK_SIZE", str(_fleet_mk_page_size(model))))
        llm_kwargs["disable_hybrid_kv_cache_manager"] = True

    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(temperature=0.0, max_tokens=n_tokens,
                        min_tokens=n_tokens, ignore_eos=True, seed=0)

    def run(n):
        """Profile one generate() of n tokens -> (wall_ms, {kernel: device_ms}).

        Only true device-side events are summed. torch's key_averages also lists
        the CPU-side op that launched a kernel (e.g. `_rocm_C::wvSplitK` next to
        `wvSplitK_hf_sml_<...>`), and both carry device time -- summing all of
        them double-counts every kernel behind a custom op.
        """
        sp_n = SamplingParams(temperature=0.0, max_tokens=n, min_tokens=n,
                              ignore_eos=True, seed=0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            llm.generate([text], sp_n, use_tqdm=False)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000.0
        ks = {}
        for e in prof.key_averages():
            if str(getattr(e, "device_type", "")).endswith("CUDA"):
                dev = getattr(e, "self_device_time_total", 0) or 0
                if dev > 0:
                    ks[e.key] = ks.get(e.key, 0.0) + dev / 1000.0
        return wall, ks, prof

    # Prefill-cancelling, same method as bench.py: the short and long runs share
    # an identical prefill, so differencing removes it along with fixed per-call
    # overhead. Without this, prefill's stock-path kernels (vllm::moe_forward,
    # _matmul_ogs_*) are amortized across the decode tokens and masquerade as
    # decode cost -- they are not on the decode path at all.
    n_short = int(os.environ.get("PROFILE_SHORT", "8"))
    w_short, k_short, _ = run(n_short)
    w_long, k_long, prof = run(n_tokens)
    prof.export_chrome_trace("/tmp/fleet_megakernel_vllm_trace.json")

    steps = n_tokens - n_short
    by_kernel = {}
    for k in set(k_short) | set(k_long):
        d = k_long.get(k, 0.0) - k_short.get(k, 0.0)
        if d > 1e-6:
            by_kernel[k] = d / steps
    wall_pt = (w_long - w_short) / steps

    total_gpu = sum(by_kernel.values())
    mega = sum(v for k, v in by_kernel.items() if _is_megakernel(k))
    other = total_gpu - mega

    label = "fleet_mk" if fleet_mk_on else "stock"
    print(f"\n===== [{label}] per-decode-token GPU attribution =====")
    print(f"prefill-cancelled     : {n_short} vs {n_tokens} tokens ({steps} steps)")
    print(f"wall (PROFILED, high) : {wall_pt:8.3f} ms/token")
    print(f"GPU total             : {total_gpu:8.3f} ms/token")
    print(f"  megakernel          : {mega:8.3f} ms/token")
    print(f"  everything else     : {other:8.3f} ms/token   "
          f"<- lm_head + sampler + KV bookkeeping")
    print(f"host / not-on-GPU     : {wall_pt-total_gpu:8.3f} ms/token   "
          f"<- scheduler, launch, syncs (INFLATED by the profiler; compare"
          f" against bench.py wall, not this)")

    print(f"\n--- decode-path GPU kernels (ms/token, >0.5% of GPU total) ---")
    for k, v in sorted(by_kernel.items(), key=lambda kv: -kv[1]):
        if v < total_gpu * 0.005:
            continue
        tag = "  [MEGAKERNEL]" if _is_megakernel(k) else ""
        print(f"{v:8.3f}  {k[:88]}{tag}")

    print("\ntrace: /tmp/fleet_megakernel_vllm_trace.json (open in Perfetto)")


if __name__ == "__main__":
    main()
