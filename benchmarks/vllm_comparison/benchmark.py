#!/usr/bin/env python3
"""
Benchmark: vLLM vs Mirage per-token latency comparison.

Reproduces the Figure 9 experiment from the Mirage Persistent Kernel paper
(arxiv 2512.22219):
  - Offline batched inference
  - Input: 64 tokens, Output: 1024 tokens
  - Batch sizes: 1, 2, 4, 8, 16
  - Models: Qwen3-8B (default), Llama2-7B, Qwen3-30B-A3B

Usage:
    python benchmarks/vllm_comparison/benchmark.py
    python benchmarks/vllm_comparison/benchmark.py --model Qwen/Qwen3-8B --batch-sizes 1,2,4,8,16
    python benchmarks/vllm_comparison/benchmark.py --vllm-only
    python benchmarks/vllm_comparison/benchmark.py --mirage-only
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIRAGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEMO_DIR = os.path.join(MIRAGE_ROOT, "demo", "qwen3")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# ── vLLM benchmark ──────────────────────────────────────────────────────────

def run_vllm_benchmark(model, input_len, output_len, batch_size,
                       num_iters, num_warmup, enable_chunked_prefill=False):
    """Run vLLM offline inference and return per-token latency, TTFT, TPOT."""
    from vllm import LLM, SamplingParams

    mode = "chunked prefill" if enable_chunked_prefill else "continuous batching"
    print(f"  [vLLM] Loading model {model} ({mode}) ...")
    llm = LLM(model=model, enforce_eager=True,
              enable_chunked_prefill=enable_chunked_prefill)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=output_len,
    )

    # Create dummy prompts with random token IDs
    dummy_token_ids = np.random.randint(
        10000, size=(batch_size, input_len)
    )
    dummy_prompts = [
        {"prompt_token_ids": ids.tolist()} for ids in dummy_token_ids
    ]

    def run_once():
        start = time.perf_counter()
        llm.generate(dummy_prompts, sampling_params=sampling_params,
                     use_tqdm=False)
        end = time.perf_counter()
        return end - start

    def run_once_ttft():
        """Run with max_tokens=1 to measure TTFT (prefill only)."""
        sp1 = SamplingParams(temperature=0.0, top_p=1.0,
                             ignore_eos=True, max_tokens=1)
        start = time.perf_counter()
        llm.generate(dummy_prompts, sampling_params=sp1, use_tqdm=False)
        end = time.perf_counter()
        return end - start

    # Warmup
    print(f"  [vLLM] Warming up ({num_warmup} iters) ...")
    for _ in range(num_warmup):
        run_once()

    # Benchmark - full generation
    print(f"  [vLLM] Benchmarking ({num_iters} iters, bs={batch_size}) ...")
    latencies = []
    for _ in range(num_iters):
        latencies.append(run_once())

    # Benchmark - TTFT (1 token generation = prefill + 1 decode)
    print(f"  [vLLM] Measuring TTFT ({num_iters} iters) ...")
    ttft_latencies = []
    for _ in range(num_iters):
        ttft_latencies.append(run_once_ttft())

    avg_batch_latency_s = np.mean(latencies)
    # Divide by total tokens (prefill + decode) to match Mirage's methodology
    total_tokens = input_len + output_len
    per_token_ms = (avg_batch_latency_s * 1000) / total_tokens
    ttft_ms = np.mean(ttft_latencies) * 1000
    # TPOT = (total - TTFT) / (output_len - 1)
    tpot_ms = (avg_batch_latency_s * 1000 - ttft_ms) / max(output_len - 1, 1)

    print(f"  [vLLM] avg batch latency: {avg_batch_latency_s*1000:.1f} ms, "
          f"per-token: {per_token_ms:.3f} ms")
    print(f"  [vLLM] TTFT: {ttft_ms:.3f} ms, TPOT: {tpot_ms:.3f} ms")

    # Cleanup to free GPU memory before Mirage runs
    del llm
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return {
        "system": "vllm",
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "per_token_latency_ms": per_token_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "avg_batch_latency_s": avg_batch_latency_s,
        "batch_latencies_s": latencies,
        "ttft_latencies_s": ttft_latencies,
        "num_iters": num_iters,
        "mode": mode,
    }


# ── Mirage benchmark ────────────────────────────────────────────────────────

def _run_mirage_once(model, input_len, output_len, batch_size, prompt):
    """Run a single Mirage demo.py invocation and return parsed output dict."""
    max_seq_len = input_len + output_len
    num_pages = 16

    cmd = [
        sys.executable, "demo.py",
        "--use-mirage",
        "--model", model,
        "--max-num-batched-requests", str(batch_size),
        "--max-num-batched-tokens", str(batch_size),
        "--max-new-tokens", str(output_len),
        "--max-seq-length", str(max_seq_len),
        "--max-num-pages", str(num_pages),
        "--page-size", "4096",
        "--ignore-eos",
        "--temperature", "0.0",
        "--prompt", prompt,
    ]

    env = os.environ.copy()
    cuda_bin = "/usr/local/cuda/bin"
    if cuda_bin not in env.get("PATH", ""):
        env["PATH"] = cuda_bin + ":" + env.get("PATH", "")
    env["CUDACXX"] = os.path.join(cuda_bin, "nvcc")
    if "MIRAGE_HOME" not in env:
        env["MIRAGE_HOME"] = MIRAGE_ROOT

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        cwd=DEMO_DIR, env=env,
    )
    output = result.stdout + "\n" + result.stderr
    latency = parse_mirage_latency(output)
    prompt_len = parse_field(output, r'Prompt length\s+(\d+)')
    gen_len = parse_field(output, r'generate length\s+(\d+)')
    fwd_pass_times = parse_fwd_pass_times(output)
    ttft_direct = parse_ttft(output)
    return latency, prompt_len, gen_len, fwd_pass_times, ttft_direct, output


def run_mirage_benchmark(model, input_len, output_len, batch_size):
    """Run Mirage via demo.py subprocess and return per-token latency, TTFT, TPOT."""
    prompt = build_prompt(model, input_len)

    print(f"  [Mirage] Running bs={batch_size} (full generation) ...")
    try:
        latency, prompt_len, gen_len, fwd_pass_times, ttft_direct, output = \
            _run_mirage_once(model, input_len, output_len, batch_size, prompt)

        if latency is None:
            print(f"  [Mirage] WARNING: could not parse latency from output")
            print(f"  [Mirage] Last 500 chars: {output[-500:]}")

        # Compute TTFT/TPOT from GPU-measured per-iteration times
        effective_prompt_len = prompt_len or input_len
        ttft_ms, tpot_ms = compute_ttft_tpot(
            fwd_pass_times, ttft_direct, effective_prompt_len, batch_size)

        if ttft_ms is None and latency is not None:
            # Fallback: estimate from per-token latency
            prefill_iters = (effective_prompt_len + batch_size - 1) // batch_size
            ttft_ms = latency * prefill_iters
            tpot_ms = latency
            print(f"  [Mirage] TTFT: {ttft_ms:.1f} ms (estimated), TPOT: {tpot_ms:.3f} ms (estimated)")
        elif ttft_ms is not None:
            print(f"  [Mirage] TTFT: {ttft_ms:.1f} ms (GPU-measured), TPOT: {tpot_ms:.3f} ms (GPU-measured)")
            if fwd_pass_times:
                print(f"  [Mirage] {len(fwd_pass_times)} iterations captured, "
                      f"first 5: {[f'{fwd_pass_times.get(i, 0):.2f}' for i in range(min(5, len(fwd_pass_times)))]}")

        return {
            "system": "mirage",
            "batch_size": batch_size,
            "input_len": input_len,
            "output_len": output_len,
            "per_token_latency_ms": latency,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "prompt_length": prompt_len,
            "generate_length": gen_len,
        }
    except subprocess.TimeoutExpired:
        print(f"  [Mirage] TIMEOUT at bs={batch_size}")
        return {
            "system": "mirage",
            "batch_size": batch_size,
            "per_token_latency_ms": None,
            "error": "timeout",
        }
    except Exception as e:
        print(f"  [Mirage] ERROR: {e}")
        return {
            "system": "mirage",
            "batch_size": batch_size,
            "per_token_latency_ms": None,
            "error": str(e),
        }


def parse_mirage_latency(output):
    """Extract per-token latency (ms) from demo.py output."""
    # Mirage mode: "per-token latency (both prefill and decode): 3.456 ms"
    m = re.search(
        r'per-token latency[^0-9]*?([0-9.]+)\s*ms', output, re.IGNORECASE
    )
    return float(m.group(1)) if m else None


def parse_fwd_pass_times(output):
    """Parse [FWD_PASS] iter=N time_ms=X.XXX lines from GPU kernel output.

    Returns a dict mapping iter_number -> time_ms.
    """
    times = {}
    for m in re.finditer(r'\[FWD_PASS\]\s*iter=(\d+)\s+time_ms=([0-9.]+)', output):
        iter_num = int(m.group(1))
        time_ms = float(m.group(2))
        times[iter_num] = time_ms
    return times


def parse_ttft(output):
    """Parse [TTFT] time_ms=X.XXX from GPU kernel output."""
    m = re.search(r'\[TTFT\]\s*time_ms=([0-9.]+)', output)
    return float(m.group(1)) if m else None


def compute_ttft_tpot(fwd_pass_times, ttft_direct, prompt_len, batch_size):
    """Compute TTFT and TPOT from per-iteration GPU times.

    TTFT = directly measured from kernel (time until first output token).
    TPOT = median of decode iterations (after prefill).

    With max_num_batched_tokens = batch_size, prefill processes
    batch_size tokens per iteration for the first request.
    """
    # Use directly-measured TTFT if available
    ttft = ttft_direct

    if not fwd_pass_times:
        return ttft, None

    prefill_iters = (prompt_len + batch_size - 1) // batch_size

    # If no direct TTFT, sum prefill iteration times
    if ttft is None:
        ttft = 0.0
        for i in range(prefill_iters):
            if i in fwd_pass_times:
                ttft += fwd_pass_times[i]
            else:
                ttft = None
                break

    # Collect decode iteration times for TPOT
    decode_times = []
    max_iter = max(fwd_pass_times.keys())
    for i in range(prefill_iters, max_iter + 1):
        if i in fwd_pass_times:
            decode_times.append(fwd_pass_times[i])

    if decode_times:
        tpot = float(np.median(decode_times))
    else:
        tpot = None

    return ttft, tpot


def parse_field(output, pattern):
    m = re.search(pattern, output)
    return int(m.group(1)) if m else None


_prompt_cache = {}

def build_prompt(model, target_tokens):
    """Build a text prompt that tokenizes to exactly target_tokens tokens.

    Accounts for demo.py's chat template (system message + special tokens)
    so that the TOTAL prompt length as seen by the model is target_tokens.
    """
    key = (model, target_tokens)
    if key in _prompt_cache:
        return _prompt_cache[key]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)

    # demo.py wraps the prompt in a chat template:
    #   [system msg] + [user: prompt] + [generation prompt]
    # We need to find user text whose total template length == target_tokens.
    def template_len(user_text):
        messages = [
            {"role": "system",
             "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": user_text},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return len(tokenizer.encode(text))

    base = (
        "The history of artificial intelligence spans decades of research "
        "and innovation across many disciplines including computer science "
        "mathematics neuroscience and cognitive psychology from early symbolic "
        "systems through modern deep learning and large language models "
    ) * 20

    # Binary search for the right number of raw token IDs to include
    # so that the full chat template tokenizes to exactly target_tokens.
    lo, hi = 1, len(tokenizer.encode(base))
    best_prompt = ""
    best_diff = float("inf")
    for _ in range(40):
        mid = (lo + hi) // 2
        ids = tokenizer.encode(base)[:mid]
        candidate = tokenizer.decode(ids, skip_special_tokens=True)
        tl = template_len(candidate)
        diff = tl - target_tokens
        if abs(diff) < abs(best_diff):
            best_diff = diff
            best_prompt = candidate
        if tl == target_tokens:
            break
        elif tl < target_tokens:
            lo = mid + 1
        else:
            hi = mid - 1

    actual = template_len(best_prompt)
    print(f"  Prompt: {actual} tokens (target: {target_tokens})")

    _prompt_cache[key] = best_prompt
    return best_prompt


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM vs Mirage per-token latency (Figure 9)"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-8B",
        help="HuggingFace model name (default: Qwen/Qwen3-8B)"
    )
    parser.add_argument(
        "--input-len", type=int, default=64,
        help="Input prompt length in tokens (default: 64)"
    )
    parser.add_argument(
        "--output-len", type=int, default=1024,
        help="Output generation length in tokens (default: 1024)"
    )
    parser.add_argument(
        "--batch-sizes", default="1,2,4,8,16",
        help="Comma-separated batch sizes (default: 1,2,4,8,16)"
    )
    parser.add_argument(
        "--num-iters", type=int, default=10,
        help="Number of benchmark iterations for vLLM (default: 10)"
    )
    parser.add_argument(
        "--num-warmup", type=int, default=3,
        help="Number of warmup iterations for vLLM (default: 3)"
    )
    parser.add_argument(
        "--vllm-only", action="store_true",
        help="Only run vLLM benchmark"
    )
    parser.add_argument(
        "--mirage-only", action="store_true",
        help="Only run Mirage benchmark"
    )
    parser.add_argument(
        "--chunked-prefill", action="store_true",
        help="Enable chunked prefill for vLLM (default: continuous batching)"
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Path to save JSON results (default: auto-generated in results/)"
    )
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    run_vllm = not args.mirage_only
    run_mirage = not args.vllm_only

    print("=" * 70)
    print("Benchmark: vLLM vs Mirage (Figure 9 reproduction)")
    print("=" * 70)
    print(f"  Model:       {args.model}")
    print(f"  Input len:   {args.input_len}")
    print(f"  Output len:  {args.output_len}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  vLLM iters:  {args.num_warmup} warmup + {args.num_iters} bench")
    print(f"  Run vLLM:    {run_vllm}")
    print(f"  Run Mirage:  {run_mirage}")
    print("=" * 70)

    vllm_results = {}
    mirage_results = {}

    # Run vLLM for all batch sizes first (single model load)
    if run_vllm:
        mode = "chunked prefill" if args.chunked_prefill else "continuous batching"
        print(f"\n>>> vLLM benchmarks ({mode})")
        for bs in batch_sizes:
            r = run_vllm_benchmark(
                args.model, args.input_len, args.output_len, bs,
                args.num_iters, args.num_warmup,
                enable_chunked_prefill=args.chunked_prefill,
            )
            vllm_results[bs] = r

    # Run Mirage for all batch sizes
    if run_mirage:
        print("\n>>> Mirage benchmarks")
        for bs in batch_sizes:
            r = run_mirage_benchmark(
                args.model, args.input_len, args.output_len, bs,
            )
            mirage_results[bs] = r

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("Per-Token Latency (ms) — lower is better")
    if run_vllm:
        vllm_mode = vllm_results.get(batch_sizes[0], {}).get("mode", "continuous batching")
        print(f"  vLLM mode: {vllm_mode}")
    print(f"  Mirage: max_num_batched_tokens = batch_size")
    print("-" * 110)
    header = f"{'BS':<5}"
    if run_vllm:
        header += f"{'vLLM (ms)':<12}"
    if run_mirage:
        header += f"{'Mirage (ms)':<14}"
    if run_vllm and run_mirage:
        header += f"{'Speedup':<10}"
    header += f"{'vLLM TTFT':<12}{'vLLM TPOT':<12}" if run_vllm else ""
    header += f"{'Mirage TTFT':<14}{'Mirage TPOT':<14}" if run_mirage else ""
    print(header)
    print("-" * 110)

    for bs in batch_sizes:
        line = f"{bs:<5}"

        vr = vllm_results.get(bs, {})
        mr = mirage_results.get(bs, {})

        if run_vllm:
            v_lat = f"{vr.get('per_token_latency_ms', 0):.3f}" if vr.get("per_token_latency_ms") else "FAIL"
            line += f"{v_lat:<12}"
        if run_mirage:
            m_lat = f"{mr.get('per_token_latency_ms', 0):.3f}" if mr.get("per_token_latency_ms") else "FAIL"
            line += f"{m_lat:<14}"
        if run_vllm and run_mirage:
            v_val = vr.get("per_token_latency_ms")
            m_val = mr.get("per_token_latency_ms")
            if v_val and m_val:
                speedup = v_val / m_val
                line += f"{speedup:.2f}x     "
            else:
                line += "N/A       "
        if run_vllm:
            v_ttft = f"{vr.get('ttft_ms', 0):.1f}" if vr.get("ttft_ms") else "-"
            v_tpot = f"{vr.get('tpot_ms', 0):.3f}" if vr.get("tpot_ms") else "-"
            line += f"{v_ttft:<12}{v_tpot:<12}"
        if run_mirage:
            m_ttft = f"{mr.get('ttft_ms', 0):.1f}" if mr.get("ttft_ms") else "-"
            m_tpot = f"{mr.get('tpot_ms', 0):.3f}" if mr.get("tpot_ms") else "-"
            line += f"{m_ttft:<14}{m_tpot:<14}"
        print(line)

    print("=" * 110)
    print("Note: Mirage TTFT = GPU-measured time until first output token (includes chunked prefill)")
    print("      Mirage TPOT = median decode iteration time (GPU-measured)")

    # ── Save results ─────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_json:
        json_path = args.output_json
    else:
        json_path = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.json")

    all_results = {
        "timestamp": timestamp,
        "config": {
            "model": args.model,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "batch_sizes": batch_sizes,
            "num_iters": args.num_iters,
            "num_warmup": args.num_warmup,
        },
        "vllm": {str(bs): _serialize(r) for bs, r in vllm_results.items()},
        "mirage": {str(bs): _serialize(r) for bs, r in mirage_results.items()},
    }

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {json_path}")


def _serialize(d):
    """Make dict JSON-serializable (convert numpy types)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, list):
            out[k] = [x.item() if isinstance(x, (np.floating, np.integer)) else x
                       for x in v]
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
