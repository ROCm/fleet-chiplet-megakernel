# Running Fleet MK decode inside vLLM

Prefill stays on stock vLLM. Decode (`max_query_len == 1`) is one Fleet MK
megakernel launch that reads and writes vLLM's paged KV cache zero-copy.

Maintained model: **GPT-OSS 120B** (MoE). Dense Qwen3-8B is registered but is
not the path these numbers come from.

## Scope of this integration

`fleet_megakernel_vllm` currently wraps the older Titan-style sequential
36-layer kernel: 240 workers, no persistent DAG scheduler, and one HIP launch
per output token. It uses Fleet's GEMM/MoE headers and optimizations, but it is
not the native Fleet `PersistentKernel` used by `demo/gpt_oss/demo.py`.

Native Fleet's measured 1.683 ms device iteration uses 248 workers plus eight
scheduler blocks and keeps the offline decode loop inside one persistent DAG
launch. Therefore 1.792 ms for this wrapper and 1.683 ms for native Fleet are
not measurements of the same kernel behind different Python front ends.

The path toward native performance is a separate native integration: allocate
vLLM's split KV layout, bind it into the real `PersistentKernel`, and run native
online decode. Further tuning of this wrapper cannot recover the DAG scheduler
or fused-layer overlap.

## Versions (verified 2026-09-03)

| Piece | Version / pin |
|---|---|
| Plugin | `vllm_integration/fleet_megakernel_vllm` (entry point `fleet_mk`) |
| vLLM | **0.27.1** (`v0.27.1`, ROCm wheel / source build) |
| Python | 3.12 venv (do **not** mix with system vLLM 0.11.1) |
| PyTorch | 2.11.x + ROCm (the 0.27.1 ABI floor) |
| ROCm | 7.0 driver; the 0.27.1 venv may bundle HIP 7.1 and still run |
| GPU | AMD MI350 / MI355 (**gfx950**), batch size 1 |
| Attention | `attention_backend=CUSTOM` (`FleetMKAttentionBackend`) |
| Block size | **16** (must equal Fleet MK `PAGE_SIZE`) |
| Graph capture | **off** (`enforce_eager=True` / `BENCH_EAGER=1`) |

0.11.1 still works from the same plugin (stock aiter layout, no CUSTOM enum).
0.26+ packed KV is why 0.27.1 needs CUSTOM. See `INTEGRATION.md` section 5b.

On this machine the 0.27.1 interpreter is `/home/claudeuser/venv-vllm027/bin/python3`.
System python keeps vLLM 0.11.1 + torch 2.9 and must not be used for this path.

## One-time setup

From the repo, after `generated/gpt_oss_120b.so` exists
(`vllm_integration/build_gpt_oss_120b.sh`):

```bash
cd vllm_integration
# the 0.27.1 interpreter, not system python
/path/to/venv-vllm027/bin/python3 -m pip install -e fleet_megakernel_vllm \
  --no-deps --no-build-isolation

# entry point must be visible or VLLM_PLUGINS=fleet_mk silently no-ops
/path/to/venv-vllm027/bin/python3 -c \
  "import importlib.metadata as m; print([e.name for e in m.entry_points(group='vllm.general_plugins')])"
# expect: fleet_mk
```

`harness.py` / `bench.py` set these **before** `import vllm` (too late in `main()`):

- `VLLM_ROCM_USE_AITER=1`
- `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`

Do not set `VLLM_ATTENTION_BACKEND=ROCM_AITER_FA`; that forces the removed V0
engine.

## Run (GPT-OSS 120B)

Always `HIP_VISIBLE_DEVICES=0`. Always `ulimit -c 0`. After a hang, `rm -f /tmp/gpucore*`
and reap the `VLLM::EngineCore` child (killing the parent does not free ~227 GiB).

```bash
cd vllm_integration
export FLEET_MK_MODEL=/path/to/gpt-oss-120b   # local weights
PY=/path/to/venv-vllm027/bin/python3

# greedy generate (prefill = stock vLLM, decode + argmax = Fleet MK)
HIP_VISIBLE_DEVICES=0 VLLM_PLUGINS=fleet_mk FLEET_MK_GREEDY_ARGMAX=1 \
  FLEET_MK_TEMP=0 \
  $PY -m fleet_megakernel_vllm.harness

# stock vLLM baseline (same backend/block size knobs, plugin off)
HIP_VISIBLE_DEVICES=0 VLLM_PLUGINS= \
  $PY -m fleet_megakernel_vllm.harness

# decode latency (prefill-cancelling 8 vs 64/128 tokens)
# BENCH_EAGER=1 is required at 120B; graph capture hits a host sync in mixin.forward
HIP_VISIBLE_DEVICES=0 VLLM_PLUGINS=fleet_mk FLEET_MK_GREEDY_ARGMAX=1 \
  BENCH_EAGER=1 \
  $PY -m fleet_megakernel_vllm.bench
```

Useful env:

| Env | Default | Meaning |
|---|---|---|
| `FLEET_MK_MODEL` | `Qwen/Qwen3-8B` | set to the 120B checkpoint |
| `FLEET_MK_MAX_TOKENS` | 1000 (`harness`) | generation length |
| `FLEET_MK_TEMP` / `FLEET_MK_SEED` | 0.0 / 0 | sampler |
| `FLEET_MK_PROMPT` | `Tell me the history of America.` | |
| `FLEET_MK_GPU_MEM_UTIL` | 0.9 | KV init fails if this over-commits free VRAM |
| `FLEET_MK_BLOCK_SIZE` | spec `page_size` (16) | must match the `.so` |
| `FLEET_MK_SO` | `generated/gpt_oss_120b.so` | pin a specific kernel |
| `FLEET_MK_GREEDY_ARGMAX` | unset | set to `1` to consume Fleet's in-kernel argmax directly |
| `BENCH_EAGER` | 0 in `bench.py` | **set to 1** for 120B |
| `BENCH_SHORT` / `BENCH_LONG` / `BENCH_REPS` | 8 / 128 / 3 | |

Wrapper that reaps EngineCore and polls VRAM between arms:

```bash
cd vllm_integration
./tools/run_vllm_arm.sh fleet_mk 32 180    # plugin on, 32 tokens, 180s cap
./tools/run_vllm_arm.sh stock 32 180       # plugin off
```

## What a good run looks like

Engine log must show:

- `Resolved architecture: GptOssForCausalLM`
- `Using CUSTOM backend`
- `enforce_eager`
- `block_size: 16`

Greedy first token on the default prompt is **35644**. Coherent analysis-channel
text follows. Do not judge bit-identical repeats: MoE W2 uses float atomics.

Measured 2026-09-03, gfx950 GPU 0, vLLM 0.27.1, `BENCH_EAGER=1`,
`FLEET_MK_GREEDY_ARGMAX=1`, 8 vs 64 tokens (best of five):
**1.983 ms/token** through vLLM, up from 2.032 ms/token. The optimized
Titan-style wrapper kernel measures **1.792 ms/device iteration**. Native Fleet
measured **1.683 ms/device iteration** and **1.794 ms/token wall time** on this
host. Of the ~0.30 ms device-to-vLLM gap, about 0.11 ms is the different kernel
control plane and about 0.19 ms is per-token vLLM launch/synchronization—not a
second sampler or KV copy.

The direct argmax path is deliberately limited to batch-size-one plain greedy
requests. Logprobs, penalties, token masks/biases, `min_tokens`, speculative
decoding, and thinking-budget processing require vLLM's general logits sampler;
unset `FLEET_MK_GREEDY_ARGMAX` for those requests.

## Counter contract (why a Titan-era plugin hang)

Titan's kernel (`titan_phases.cuh`) used **per-layer** counter blocks and re-zeroed
rank / MoE barrier / workspace every decode step. Fleet's layer body derives
`expected = task_layer_idx + 1` against **one shared monotonic block** that is
never reset (`demo_gpt_oss_120b.py`).

Copying the Titan plugin into this tree left `packing_moe.py` on the per-layer
stride. First vLLM decode then hung at layer 2 (`expected=2`, `observed=1`) while
prefill (stock) succeeded. The plugin now matches the standalone demo: shared
`counter_ptr`, MoE barrier `160 * num_experts`, workspace `top_k * hidden`, no
per-step zero of those buffers.
