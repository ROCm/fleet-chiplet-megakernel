#!/usr/bin/env bash
# Profiling the persistent megakernel on the MI300X. One mpk() call = 3 dispatches (prepare_kernel, worker_kernel,
# scheduler_kernel); the worker kernel is ONE 296-block dispatch that lives for all 32 tokens (~100 ms), so a kernel
# trace shows one long bar, not 138k tasks. Inside-the-kernel structure comes from Fleet's own instrumentation.
#
# Usage: profile.sh trace|counters|perfetto|occupancy [extra args for the python script]
#   trace      rocprofv3 --kernel-trace (fallback: rocprof --hip-trace --stats) around bench.py (1 warm-up + 2 launches) +
#              llvm-readelf --notes of the compiled code object (exact VGPR/SGPR/LDS/scratch per kernel):
#              results/profile/trace/**/*kernel_trace.csv -> bench.py --rocprof -> results/profile/rocprof.json
#              = dispatch count per mpk() call, per-dispatch duration, VGPR/AGPR/SGPR/LDS/scratch of each kernel.
#   counters   second pass, hardware counters (FETCH_SIZE, WRITE_SIZE, TCC hit/miss, SQ_WAVES). KNOWN: rocprofv3 --pmc
#              serializes dispatches, and the worker and scheduler kernels must run concurrently -> the split-mode
#              megakernel DEADLOCKS under --pmc (persistent_kernel.cuh, init comment). Wrapped in `timeout`; if it
#              hangs, the derived bandwidth in results/bench/latency.json is the number to report. Counter names
#              differ by ROCm version: `rocprofv3 -L` lists them.
#   perfetto   Fleet's on-device task profiler (MPK_ENABLE_PROFILING): run_e2e.py --perfetto writes
#              results/m4/e2e.trace.perfetto-trace (+ .pt raw) -> open in https://ui.perfetto.dev. Caveat: the
#              profiling build stops after ONE iteration (prepare_next_batch uses profiling_num_iters), so this is a
#              one-token trace of every task on every worker; needs tg4perfetto (pip install git+https://github.com/flashinfer-ai/tg4perfetto.git).
#   occupancy  static resource usage of the compiled megakernel from the trace CSV (same as `trace`, parse only).
#
# Reading a single 100 ms+ kernel: the per-token time is NOT the kernel duration / 32 unless the launch and the
# prepare_kernel are negligible; use bench.py's per-iteration numbers (MPK_EVENT_TIMING) for ms/token and the
# per-boundary table for where the time goes (which op, which layer). A kernel trace answers only: how many
# dispatches, how long the whole loop lived, what resources the kernel was compiled with.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="${ROCM_PATH:-/opt/rocm}/bin:$PATH"
export AMDGPU_TARGETS=gfx942 MIRAGE_HOME="${MIRAGE_HOME:-$(cd ../.. && pwd)}"
PY=${PYTHON:-.venv/bin/python}; [ -x "$PY" ] || PY=python3
mode=${1:-help}; shift || true
mkdir -p results/profile results/bench

case "$mode" in
  trace)
    # rocprofv3 1.1 aborts on this box: torch's kineto registers with rocprofiler-sdk after the tool's configuration
    # window ("api registration failed with error code 16", results/profile/trace.log 2026-09-14). Try it, then legacy
    # rocprof (roctracer; rocprofv2 wants a different CLI and adds nothing); the parser accepts every CSV with a kernel-name column.
    out=results/profile/trace; rm -rf "$out"; mkdir -p "$out"
    cmd="$PY bench.py --warmup 1 --runs 2 --no-event-timing --out results/profile $*"
    mkdir -p "$out/v3" "$out/v1"
    if command -v rocprofv3 >/dev/null && rocprofv3 --kernel-trace --stats -d "$out/v3" -o run --output-format csv -- $cmd \
        && ls "$out"/v3/*/*kernel_trace.csv >/dev/null 2>&1; then echo "trace: rocprofv3"
    elif command -v rocprof >/dev/null && { rocprof --hip-trace --stats -o "$out/v1/results.csv" $cmd || true; } \
        && ls "$out"/v1/results.stats.csv >/dev/null 2>&1; then echo "trace: rocprof (v1, --hip-trace --stats: our kernels are seen at the HIP API; the plain dispatch mode missed them)"
    else echo "no tracer produced a CSV under $out"; exit 1
    fi
    # exact per-kernel resources from the compiled code object (the tracers' columns are empty for these dispatches)
    co=results/profile/codeobj; rm -rf "$co"; mkdir -p "$co"; llvm=/opt/rocm/llvm/bin
    $llvm/llvm-objcopy --dump-section .hip_fatbin="$co/fatbin.bin" permanent_output_dir/test.cpython-38-x86_64-linux-gnu.so /dev/null
    tgt=$($llvm/clang-offload-bundler --list --type=o --input="$co/fatbin.bin" | grep gfx942 | head -1)
    $llvm/clang-offload-bundler --unbundle --type=o --targets="$tgt" --input="$co/fatbin.bin" --output="$co/gfx942.o"
    $llvm/llvm-readelf --notes "$co/gfx942.o" > "$co/notes.txt"; rm -f "$co/fatbin.bin" "$co/gfx942.o"
    $PY bench.py --rocprof "$out" --out results/profile ;;
  counters)
    out=results/profile/counters; rm -rf "$out"
    echo "WARNING: --pmc serializes dispatches; split worker/scheduler kernels can deadlock. 15 min timeout."
    pmc="${PMC:-FETCH_SIZE WRITE_SIZE TCC_HIT_sum TCC_MISS_sum SQ_WAVES SQ_BUSY_CYCLES}"
    if command -v rocprofv3 >/dev/null; then
      timeout 900 rocprofv3 --pmc $pmc --kernel-trace -d "$out" -o run --output-format csv -- \
        $PY run_layer.py --layers 1 --out results/profile/counters_run "$@" || echo "counters pass failed/timed out (expected on split mode)"
    else
      printf 'pmc: %s\n' "$pmc" > "$out.txt"
      timeout 900 rocprof -i "$out.txt" -o "$out/results.csv" \
        $PY run_layer.py --layers 1 --out results/profile/counters_run "$@" || echo "counters pass failed/timed out (expected on split mode)"
    fi
    ls -la "$out" 2>/dev/null || true ;;
  perfetto)
    $PY run_e2e.py --perfetto --runs 1 --warmup 0 --no-teacher-forced --out results/profile/perfetto "$@"
    ls -la results/profile/perfetto/*.perfetto-trace ;;
  occupancy)
    $PY bench.py --rocprof results/profile/trace ;;
  *)
    sed -n 2,26p "$0"; exit 1 ;;
esac
