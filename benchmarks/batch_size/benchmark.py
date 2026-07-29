#!/usr/bin/env python3
"""
Benchmark: Mirage vs non-Mirage across batch sizes
"""

import subprocess
import os
import json
import time
import re
from datetime import datetime

BATCH_SIZES = [1, 2, 4, 8, 16]
OUTPUT_TOKENS = 256  # Target output tokens
MAX_SEQ_LENGTH = 300  # Controls actual output (prompt + output <= this)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEMO_DIR = "/root/mirage/demo/qwen3"

# Short prompt that fits within limits
PROMPT = "Give me a short introduction to large language model."


def run_benchmark(batch_size, use_mirage):
    cmd = [
        "python", "demo.py",
        "--max-new-tokens", str(OUTPUT_TOKENS),
        "--max-num-batched-requests", str(batch_size),
        "--max-num-batched-tokens", str(batch_size),
        "--max-seq-length", str(MAX_SEQ_LENGTH),
        "--ignore-eos",
        "--prompt", PROMPT,
    ]
    if use_mirage:
        cmd.append("--use-mirage")

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = "2"

    print(f"Running: batch={batch_size}, mirage={use_mirage}")

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env, cwd=DEMO_DIR)
        elapsed = time.time() - start
        output = result.stdout + result.stderr

        # Parse latency
        latency = None
        prompt_len = None
        gen_len = None

        for line in output.split('\n'):
            if "per-token latency" in line.lower():
                m = re.search(r'Prompt length\s+(\d+)', line)
                if m: prompt_len = int(m.group(1))
                m = re.search(r'generate length\s+(\d+)', line)
                if m: gen_len = int(m.group(1))
                m = re.search(r'per-token latency[^0-9]*([0-9.]+)\s*ms', line, re.IGNORECASE)
                if m: latency = float(m.group(1))

        return {
            "batch_size": batch_size,
            "use_mirage": use_mirage,
            "prompt_length": prompt_len,
            "generated_tokens": gen_len,
            "per_token_latency_ms": latency,
            "total_time_s": elapsed,
            "success": latency is not None,
            "error": None if latency else result.stderr[-300:] if result.stderr else "No latency found"
        }
    except Exception as e:
        return {"batch_size": batch_size, "use_mirage": use_mirage, "success": False, "error": str(e)}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.json")

    print(f"Benchmark started: {timestamp}")
    print(f"Batch sizes: {BATCH_SIZES}, Output tokens: {OUTPUT_TOKENS}")

    results = []
    for batch_size in BATCH_SIZES:
        for use_mirage in [False, True]:
            r = run_benchmark(batch_size, use_mirage)
            results.append(r)
            lat = f"{r['per_token_latency_ms']:.2f}ms" if r.get('per_token_latency_ms') else "FAILED"
            print(f"  -> {lat}")

        with open(results_file, 'w') as f:
            json.dump({"timestamp": timestamp, "results": results}, f, indent=2)

    # Summary
    print("\n" + "="*70)
    print(f"{'Batch':<6} {'No Mirage':<15} {'Mirage':<15} {'Speedup':<10}")
    print("="*70)
    for i in range(0, len(results), 2):
        nm, m = results[i], results[i+1]
        nm_lat = f"{nm['per_token_latency_ms']:.2f}ms" if nm.get('per_token_latency_ms') else "FAIL"
        m_lat = f"{m['per_token_latency_ms']:.2f}ms" if m.get('per_token_latency_ms') else "FAIL"
        if nm.get('per_token_latency_ms') and m.get('per_token_latency_ms'):
            speedup = f"{nm['per_token_latency_ms']/m['per_token_latency_ms']:.2f}x"
        else:
            speedup = "N/A"
        print(f"{nm['batch_size']:<6} {nm_lat:<15} {m_lat:<15} {speedup:<10}")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
