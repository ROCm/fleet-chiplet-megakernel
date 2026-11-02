# DeepSeek-Coder-V2-Lite batch-1 decode on AMD MI300X

Batch-1, bf16, greedy decode of [deepseek-ai/DeepSeek-Coder-V2-Lite-Base](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base)
(27 layers: 1 dense + 26 MoE with 64 routed + 2 shared experts, MLA attention) as one persistent megakernel on the Fleet
runtime. The whole 32-token generation is one `mpk()` call; 298 ops / 20,492 tasks per token. The stock Hugging Face model
runs the prefill, only decode goes through the megakernel.

What this adds to the repository:

| where | what |
|---|---|
| `include/mirage/persistent_kernel/runtime_header.h` | `TaskType` 300-315 for the new tasks |
| `src/kernel/ds_task_register.cc`, hook in `src/kernel/graph.cc` | registration of the new task names: shape checks, `(inputs, outputs, type)`, dispatch string |
| `src/kernel/runtime.cc` | `register_mugraph` accepts a DAG (an op may consume tensors of any earlier op: multi-producer events, fan-out); bid packing into `expert_offset` for the new types; codegen additions |
| `include/mirage/persistent_kernel/kernels/` | the task kernels (namespace `dsk`): tiled bf16 GEMV with epilogues, MLA attention chunk + LSE merge, RMSNorm, MoE gate + top-k, slot-indexed expert GEMVs, HF-order combine, argmax/embedding fallbacks; `fleet_dispatch.cuh` = TaskDesc/RuntimeConfig to kernel arguments |
| `demo/deepseek_v2_lite/` (this directory) | graph builder, reference, run / gate / benchmark scripts, CPU tests |

Fleet's own `embedding`, `rmsnorm`, `splitk_reduce_mi300`, `argmax_partial` / `argmax_reduce` tasks are reused; everything
else in the graph is one of the new kernels. The kernels are plain bf16-as-uint16 with fp32 fma (no gfx950-only
conversion instructions), so they build for gfx942; nothing in them is MI300X-specific beyond the tile sizes, which
were chosen by measurement on that GPU.

## Results (one AMD MI300X, ROCm 7.2.4, 1,024-token prompt, 32 greedy tokens)

| metric | result |
|---|---|
| Time per output token, P50 / P95 (iterations >= 2) | **3.76 ms / 3.77 ms** (first iteration of a launch 5.4 ms; launch-level 3.93 ms) |
| Tokens per second | **266** |
| HF eager bf16 on the same GPU | 69.35 ms per token, 3,853 kernel launches per token |
| GPU launches | 1 persistent-kernel call per 32 tokens (3 dispatches: prepare, worker, scheduler) |
| Correctness, all 27 layers vs HF eager bf16 | every layer within gate (worst layer rel. error 1.6 %), logits cosine 0.99997, same next token |
| Free-running decode | **32 / 32 tokens identical to HF**, same tokens on every one of 5 runs |
| Teacher-forced logit cosine (gate 0.9995) | misses at 4 of 32 steps (0.99913, 0.99895, 0.99771, 0.99914); the argmax matches at all 32. All four are expert-routing near-ties: HF with the megakernel's routing forced reproduces its logits at cosine >= 0.99992 (`routing_counterfactual.py`). Reported as a failing gate, not relaxed |
| Memory traffic / achieved bandwidth | 5.24 GB per token / 1.39 TB/s (derived: bytes counted from the attached tensors / P50) |
| Straight chain vs DAG, back to back | 4,250 us -> 3,741 us per token (-12 %) from letting independent ops overlap (Q || KV path, shared expert || gate + routed experts) |

Where the time goes (diagnostic `-DMPK_ENABLE_TIMING` build, worker view, per task): 61 % waiting on the previous op's
dependency counter, 32 % executing, 6 % polling an empty queue, 1 % signalling. The remaining levers are runtime-side
(poll latency, cheaper completion signalling, fewer critical-path events), not kernel-side.

### Limitations

- Batch 1, bf16, one GPU, no tensor parallelism, batching or speculative decoding.
- Fixed 1,024-token context (`--t-max 1056`: 9 attention chunks of 128 tokens; the attention tile width is a template
  parameter, the chunk count follows `t_max`).
- Prefill is out of scope: `reference_prefill.py` runs the HF model and hands its `DynamicCache` (K `[16, T, 192]`,
  V `[16, T, 128]` per layer, bf16) to the megakernel unchanged. Absorbed-MLA caching is validated in the CPU reference
  only (`fleet_ds/reference.py::decode_step_absorbed`), not in the kernels.
- Tested only on MI300X (gfx942, 304 CUs -> 296 workers + 8 schedulers). Not run on MI350/MI355; the tile sizes
  (`fleet_ds/graph.py` `DIMS`) and `AMDGPU_TARGETS` are the knobs to revisit there.
- The design target of 2.5 ms per token was not reached.

## Build (gfx942)

Fleet's build defaults to gfx950 (`CMakeLists.txt`, `python/mirage/mpk/persistent_kernel.py`, `python/mirage/kernel.py`);
the one switch is `AMDGPU_TARGETS`, exported for the build and for every `PersistentKernel.compile()`:

```bash
cd fleet-chiplet-megakernel
export AMDGPU_TARGETS=gfx942 MIRAGE_HOME=$PWD
pip install -e . -v --no-build-isolation --no-deps      # MAX_JOBS=<n> limits make -j on small hosts
```

`MPK_USE_CK_FMHA` must stay undefined for gfx942 (that path uses gfx950-only builtins). The DeepSeek remote code needs
`transformers==4.44.2` (newer versions break `trust_remote_code` for this model), plus `safetensors`, `accelerate`,
`huggingface_hub`. On the box we used `deps/composable_kernel` at `ac18460` (a 2026-05 `develop` commit; the rocm-7.2 tags
are too old for `linear_ck_mi300.cuh`, 2026-06+ `develop` too new), `deps/json` v3.12.0, `deps/cutlass` v4.7.1.

`src/kernel/ds_task_register.cc` is picked up by the CMake glob of `src/`; a checkout that was built before it existed
needs a clean `build/`.

## Run

From this directory, with the model and the HF reference in place:

```bash
./setup_local.sh                                   # CPU venv + the HF remote code into ref/hf (also for the tests)
./fetch_model.sh                                   # 31.4 GB checkpoint into models/DeepSeek-Coder-V2-Lite-Base/
python -u reference_prefill.py                     # HF eager bf16 reference: prefill cache, 32 greedy steps, per-layer dumps
python -u run_layer.py --layers 1 --event-timing   # one MoE layer vs HF, per-layer gates          -> results/m2/
python -u run_layer.py --layers 0-26 --event-timing   # all 27 layers in one launch                -> results/m3/
python -u run_e2e.py --event-timing                # 32 greedy tokens + teacher-forced gates        -> results/m4/e2e.json
python -u bench.py                                 # P50 / P95, tok/s, bytes per token, bandwidth   -> results/bench/bench.json
python -u dispatch_probe.py --out results/perf/probe   # per-op time for the whole token
./profile.sh trace                                 # rocprof kernel trace + code-object resources
```

`run_layer.py` / `run_e2e.py --dry-run` print the plan and arena shapes on any machine (no ROCm, no model).
`build_graph.sh` generates `task_graph.json` + `test.cu` the way `demo/qwen3/demo_30B_A3B.py` does and cross-checks the
events against the local builder; `COMPILE=1` also runs hipcc.

Every `PersistentKernel.compile()` recompiles into `./permanent_output_dir/`; remove it when flags change. The run scripts
set `AMDGPU_TARGETS=gfx942` and `MIRAGE_HOME` (the repo root) if unset.

Knobs: `FLEET_KNOBS=dag=0,fold=0` builds the straight-chain graph of the earlier measurements (`fleet_ds/graph.py` `KNOBS`);
`include/mirage/persistent_kernel/kernels/switches.h` holds the kernel switches (all values bit-identical, tested).

`mpk_debug.c` is spliced into Mirage's generated launcher at compile time by `run_layer.py` (not into the repository): a
non-blocking launch under a host watchdog that dumps the runtime's event counters and queue heads on a hang, and a
per-launch runtime `max_seq_length` so the scripts never call `init_func` twice (a second `init_func` leaks the
worker/scheduler stream pair and hangs the next launch; fix in ROCm/fleet-chiplet-megakernel#9, not part of this branch).

## Tests (CPU only)

```bash
./setup_local.sh && .venv/bin/pytest -q tests           # FLEET_SIM_LAYERS=2 for a smaller simulator run
```

- `test_reference.py`: `fleet_ds.reference` vs the HF decoder layer, random weights, fp32 and bf16, small and real dims.
- `test_graph.py`, `test_mirage_graph.py`: task / op / event counts, DAG producers, chain property, the Mirage builder's
  op list equals the local builder's.
- `test_kernels_cpu.py`: every kernel compiled with `clang++` under `tests/cpu_shim.h` (a 256-thread workgroup as
  fibers) vs the reference at real shapes; GEMV bit-identical to its pre-vectorisation version; every switch bit-identical.
- `test_sim.py`: `fleet_ds/sim.py` runs the whole task graph (events, pointer offsets, the real kernels) on the CPU: one
  MoE layer with real layer-1 weights if a checkpoint shard is under `models/`, a 3-layer mini model, hang detection.
- `test_glue.py`: enum, register functions, dispatch strings and the mocked Fleet fields agree with the headers in this
  checkout; `src/kernel/*.cc` syntax-checked against Fleet's host headers (needs `nlohmann/json.hpp`).
- `test_scripts_smoke.py`: every script parses, `--dry-run` plans.
- `tests/hip_smoke.hip`: one `__global__` per kernel instantiation of the default graph for an offline
  `hipcc --offload-arch=gfx942 -fsyntax-only` (needs ROCm; see its header).

## Layout

```
fleet_ds/graph.py          per-token task graph in Fleet's TaskDesc/EventDesc shape (local builder, knobs in DIMS / KNOBS)
fleet_ds/mirage_graph.py   the same graph through Mirage's PersistentKernel (attach_input / new_tensor / register_task)
fleet_ds/reference.py      pure-torch reference decode step, bit-exact vs HF
fleet_ds/sim.py            CPU simulator of the Fleet runtime over the graph, running the real kernels
run_layer.py               one token through layers a..b vs the HF dumps (also the shared launch / arena / gate code)
run_e2e.py, bench.py, dispatch_probe.py, routing_counterfactual.py, reference_prefill.py, profile.sh, build_graph.sh
prompts/                   the fixed 1,024-token prompt (text and ids)
tests/                     CPU tests, the CPU shim and harness, the offline hipcc smoke file
```

Comments in the code cite decisions as `D4`, `10a`, `12a` and sessions of a GPU log: these refer to the design document
and measurement log of the project this was developed in, kept as provenance for the tile sizes and op folds.
