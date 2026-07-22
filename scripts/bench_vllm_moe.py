#!/usr/bin/env python3
"""Measure vLLM decode-only latency for Qwen3-30B-A3B MoE using RequestOutput.metrics."""

def main():
    from vllm import LLM, SamplingParams
    MODEL = "Qwen/Qwen3-30B-A3B"
    PROMPT = (
        "Explain the key differences between RISC and CISC processor architectures, "
        "including their instruction set design philosophies, pipeline implementations, "
        "memory access patterns, register file organization, microcode usage, and how "
        "modern processors have evolved to blur the traditional boundaries between "
        "these two approaches in terms of performance, power efficiency, and overall "
        "transistor utilization."
    )
    llm = LLM(model=MODEL, dtype="bfloat16", enforce_eager=True,
              max_model_len=2048, gpu_memory_utilization=0.9,
              enable_prefix_caching=False, disable_log_stats=False)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024,
                        ignore_eos=True)
    for bs in [1, 2, 4, 8]:
        prompts = [PROMPT] * bs
        for _ in range(2):
            llm.generate(prompts, sampling_params=sp, use_tqdm=False)
        decode_per_iter = []
        for i in range(5):
            outputs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
            iter_decode = []
            for out in outputs:
                m = out.metrics
                if m is None:
                    continue
                decode_s = m.last_token_ts - m.first_token_ts
                num_decode_tokens = m.num_generation_tokens - 1
                if num_decode_tokens > 0:
                    iter_decode.append((decode_s / num_decode_tokens) * 1000)
            if iter_decode:
                decode_per_iter.append(sum(iter_decode) / len(iter_decode))
        if decode_per_iter:
            avg = sum(decode_per_iter) / len(decode_per_iter)
            print(f"BS={bs}  decode_TPOT={avg:.3f}ms")
        else:
            print(f"BS={bs}  ERROR: metrics None")

if __name__ == '__main__':
    main()
