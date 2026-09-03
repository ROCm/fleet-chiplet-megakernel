# fleet_mk in vLLM — integration & architecture

How the **fleet_mk** fused decode megakernels (AMD MI350 / gfx950 / ROCm) are plugged
into vLLM, how the decode path works, how KV is shared zero-copy, and measured
latency vs stock vLLM.

Runs on **vLLM 0.11.x and 0.27.x** from one codebase — see §5b for what differs
between them. Sections dated 2026-08-04 were measured on 0.11.1 at page size 128
and are kept as recorded; page size is now 16 (§5) and 0.27.1 is verified (§5b).

Two architectures are supported, both emitted from `fleet_mk_generate.py` + a YAML
config: dense **Qwen3-8B** and MoE **GPT-OSS 120B**. GPT-OSS is the maintained
path and the one all numbers below were measured on.

---

## 1. What fleet_mk is

fleet_mk is a single **fused persistent HIP megakernel**: one GPU launch
(`<model>_launch`) runs *all* 36 transformer layers + final RMSNorm + LM head +
argmax for **one decode step** (one generated token, batch size 1). Weights are
MXFP4, activations fp8. The compiled artifact is `generated/<model>.so`
(built from `generated/<model>_launch.hip` + `generated/<model>_kernel.cuh` via
`build_<model>.sh`) — e.g. `generated/gpt_oss_120b.so`.

fleet_mk handles **decode only**. Prefill (processing the prompt) stays on vLLM's
stock path.

## 2. How it's integrated

vLLM discovers fleet_mk through a `vllm.general_plugins` entry point (`pyproject.toml`)
that calls `plugin.register()` once at engine startup. `register()` swaps each stock
model class fleet_mk supports for a subclass:

```
plugin.register()  ->  ModelRegistry.register_model("Qwen3ForCausalLM",
                                                     FleetMKQwen3ForCausalLM)
                       ModelRegistry.register_model("GptOssForCausalLM",
                                                     FleetMKGptOssForCausalLM)
```

Registering both is harmless when only one is served — vLLM instantiates whichever
architecture the loaded checkpoint declares. The GPT-OSS registration is wrapped in
a try/except so a vLLM build without `gpt_oss.py` still gets the dense path.

Each subclass is thin: everything model-agnostic lives in `FleetMKModelMixin`
(`mixin.py`), parameterized by a `ModelSpec` (`spec.py`). The mixin overrides three
seams on the stock model:

| Override | Role |
|---|---|
| `load_weights()` | after `super()` loads the standard modules, MXFP4-pack them into fleet_mk's layout, build RoPE tables, load `<model>.so`, allocate workspace buffers (`_setup_fleet_mk`). |
| `forward()` | route **prefill → `super().forward()`**, **decode → `fleet_mk.decode_step()`**. |
| `compute_logits()` | after a fleet_mk **MoE** decode, return the bf16 logit row the megakernel's argmax epilogue already computed (§8.2); otherwise stock — vLLM's bf16 `lm_head` over the hidden state fleet_mk returns. vLLM's sampler runs on top either way. |

Enable with `VLLM_PLUGINS=fleet_mk`; disable (stock baseline) with `VLLM_PLUGINS=`.

### Module map

| File | Responsibility |
|---|---|
| `plugin.py` | entry point; registers the model overrides. |
| `model.py` | `FleetMKQwen3ForCausalLM` / `FleetMKGptOssForCausalLM`: per-model weight packing and the layer-attention accessor. Everything else is inherited. |
| `mixin.py` | `FleetMKModelMixin`: kernel load, buffers, RoPE, zero-copy KV binding, prefill/decode routing — all model-agnostic. |
| `spec.py` | `ModelSpec`: every dimension and layout constant, derived from `configs/<model>.yaml` through `fleet_mk_generate.load_and_validate()`. |
| `runtime.py` | ctypes ABI loader, `FleetMKBuffers` (workspace + aliasable KV), `build_ptr_table`, `FleetMKDecoder.decode_step`. |
| `packing.py` | dense MXFP4 weight packing into fleet_mk's per-XCD layout. |
| `packing_moe.py` | MoE weight packing + the 36-slot per-layer pointer table. |
| `harness.py` | single-request greedy/sampler bring-up + parity harness. |
| `bench.py` | decode-latency benchmark (fleet_mk vs stock). |

**There is no `constants.py`.** Every number comes from `spec.py`, which parses the
same YAML the code generator does — and reaches into `fleet_mk_generate` directly for
the two facts the YAML does not carry (`counter_slots()` for the QKV barrier offset,
`MIRAGE_IN`/`MIRAGE_OUT` for the pointer-table shape). A second hand-maintained copy
of these constants is exactly how the plugin silently drifted from the kernel
before; the file was deleted rather than updated.

## 3. Prefill vs decode routing

`forward()` reads vLLM's attention metadata and branches on query length:

- **`attn_metadata is None`** (profiling/dummy run) or **`max_query_len != 1`**
  (prefill: many prompt tokens in one query) → `super().forward()`. vLLM's stock
  attention runs and writes the prompt's K/V into the paged KV cache.
- **`max_query_len == 1`** (decode: exactly one new token) → fleet_mk.
  `decode_step(embed, cur_pos, block_table)` runs the megakernel.

On decode, fleet_mk writes the post-final-norm hidden state into
`buf_lm_norm_scratch` and `forward()` returns `hs.clone()`. fleet_mk's fused **argmax**
is still ignored — vLLM's full sampler runs on top, so temperature / top-p /
penalties / seeds all work. On the MoE path the fused **logits** are not ignored:
the epilogue also writes the bf16 row into `buf_logits`, which `compute_logits()`
returns in place of vLLM's `lm_head` GEMV (§8.2). Sampling is unaffected; only the
redundant matrix-vector product is skipped.

## 4. Zero-copy KV (the "M2" work)

### The problem it solves

Attention needs every past token's K/V (the **KV cache**; vLLM reserves ~184 GiB
here). Originally vLLM owned the KV cache in *its* layout, while fleet_mk's kernel
read/wrote its *own* separate K/V buffers. A per-request gather-copy
(`_mirror_prefill_kv`, now deleted) reshuffled vLLM's prefill KV into fleet_mk's
buffers before decode. That meant **two KV stores** (vLLM's pool wasted during
decode) plus a **per-request copy**.

Zero-copy KV makes fleet_mk read/write **vLLM's KV pool directly** — one store, no
copy.

### How it works

The trick is to force a vLLM attention backend whose *physical KV layout is
byte-identical to fleet_mk's*, then alias pointers:

- The **aiter** backends (`rocm_aiter_fa`, `rocm_aiter_unified_attn`) allocate KV as
  `(2, num_blocks, block_size, num_kv_heads, head_size)` — the leading `2` makes
  K and V each contiguous — and write it via `reshape_and_cache_flash`
  (the *non-reordered* layout). Reshaped to `[num_blocks*block_size, 1024]` this is
  byte-identical to fleet_mk's flat `[entries, num_kv_heads*head_dim]` K (and V)
  buffer. (The default Triton/Rocm backends reorder K or interleave the `2` in the
  middle — incompatible with a single flat pointer.)
- With **`block_size == PAGE_SIZE`**, one vLLM block == one fleet_mk page, so fleet_mk's
  page-indirection table `kv_indices` is driven directly from vLLM's `block_table`.

Binding happens lazily on the **first decode** (`_ensure_fleet_mk_kv_bound`), because
vLLM allocates the KV cache *after* `load_weights()`:

```python
kvc = attn.kv_cache[0]                        # [2, num_blocks, block_size, n_kv, head]
k_alias = kvc[0].reshape(entries, n_kv*head)  # view, not copy
v_alias = kvc[1].reshape(entries, n_kv*head)
assert k_alias.data_ptr() == kvc[0].data_ptr()  # zero-copy invariant guard
...
buffers.set_kv_aliases(k_aliases, v_aliases)  # fleet_mk's K/V point at vLLM's pool
ptr_table = build_ptr_table(weight_ptrs, buffers)
```

During decode the megakernel appends the new token's K/V into vLLM's block
(`rope_kv_update`, indexed through `kv_indices <- block_table`) and reads all prior
KV in place.

## 5. Constraints & configuration

Zero-copy requires `block_size == PAGE_SIZE`. Both aiter backends that share
fleet_mk's KV layout **cap `block_size`**, which forced fleet_mk's `PAGE_SIZE` down:

| Backend | Mechanism | `block_size` ceiling |
|---|---|---|
| `rocm_aiter_fa` (MHA) | `paged_attention_v1` → `static_assert(BLOCK_SIZE <= k_thread_per_block)`, `k_thread_per_block = NWARPS*8 = 32` | **≤ 32** |
| `rocm_aiter_unified_attn` (Triton) | LDS use = `block_size * 1024 B` vs 160 KB cap | **≤ 160** |

vLLM independently caps `block_size` at 256 in its own config
(`BlockSize = Literal[1, 8, 16, 32, 64, 128, 256]`, `vllm/config/cache.py`), so the
standalone kernel's original `PAGE_SIZE = 4096` could never bind.

**Chosen: `PAGE_SIZE = block_size = 16`** — vLLM's *default* ROCm block size,
unified-attention backend. Running at the default is the point: fleet_mk binds to a
stock-configured engine with no `block_size` override, so no caller has to know
fleet_mk's page size. A ≤512-token sequence spans ≤32 fleet_mk pages.

128 was the previous choice (largest power-of-two under the unified backend's LDS
budget) and is still the ceiling that matters if a larger page is ever wanted.

`PAGE_SIZE` is a compiled `static constexpr`, and also a *template parameter* to
both `rope_kv_update` and `paged_attention_minimal_decode_hd64` — so the driver's
`page_size` and the kernel's `PAGE_SIZE` must change together or KV addressing goes
silently wrong. Changing it is now **one line**, `page_size:` in
`configs/<model>.yaml`, followed by `fleet_mk_generate.py --output-dir …` and a
rebuild; the generator drives all three emitted sites. (The older integration
hand-edited four files, which is why they drifted.)

**It is free.** Each step measured back-to-back against the previous build on the
same GPU:

| model | `PAGE_SIZE` | decode | registers |
|---|---|---:|---|
| GPT-OSS 120B | 4096 | 2.501 ms/tok | 210 VGPR / 12 SGPR spill / 0 VGPR spill / 2 waves per SIMD |
| GPT-OSS 120B | 128 | 2.491 ms/tok | identical |
| GPT-OSS 120B | **16** | ~2.37 ms/tok in-kernel¹ | 226 VGPR / 12 SGPR spill / 0 VGPR spill / 2 waves per SIMD / 512 B LDS — identical to its own 128 build |
| Qwen3-8B | 4096 | 6.798 ms/tok | 241 VGPR / 2 SGPR spill / 0 VGPR spill / 2 waves per SIMD / 360 B LDS |
| Qwen3-8B | **128** | **6.768 ms/tok** | identical |

¹ 128→16 was measured under vLLM from the kernel's own `[FLEET_MK_TIME]` per-token
totals — ~2360–2397 µs at 128 against ~2356–2386 µs at 16, i.e. inside the
run-to-run band. These are not `bench.py` "Non-outlier avg" numbers and should not
be quoted as the headline. The 4096/128 rows above are, which is why the columns
are not directly comparable across rows.

`qwen3_8b.yaml` is still at 128; the dense path has no vLLM-default requirement
driving it down, and moving it would cost a rebuild for no measured gain.

That flatness is what the code predicts — `PAGE_SIZE` only feeds address
arithmetic (`kv_indices[tok / PAGE_SIZE]`, `tok % PAGE_SIZE`) and `seqlen_k`; it
sizes no LDS and sets no unroll factor. Do not expect a page-size sweep to pay.
What 16 *does* change is that a 512-token sequence spans 32 pages instead of 4, so
`kv_indices` is 8× longer and the per-page indirection runs 8× as often — address
arithmetic, not bandwidth, and evidently below the noise floor here.

A side benefit worth keeping: at 4096 the demos' ~150-token runs fit entirely
inside page 0, so cross-page addressing was never exercised on the default path.
At 16 they span many pages.

### Output at 16 is fluent, and the divergence is ambient, not page-size (2026-08-24)

The 4096→128 change left the token stream *identical*, so the natural expectation
was that 128→16 would too. It did not, and that looked like a regression until it
was measured properly. `fleet_megakernel_vllm/compare_runs.py` exists for exactly this:

| comparison | earliest first-divergence |
|---|---|
| **within** page-16 (3 runs of one build) | **3**, 3, 13 |
| **between** page-128 and page-16 | 3, 13, 26 |

The earliest divergence seen anywhere is *within* page-16 — two runs of a single
identical build disagreeing at token 3. The between-config spread is bounded below
by that noise floor, so the two page sizes are indistinguishable at this sample
size. All four runs emit `35644` and are fluent and correctly structured.

The rule this encodes: **compare between-config divergence against the
within-config noise floor, never against zero.** An A/B of one run per config
cannot tell a real regression from ambient nondeterminism, and at this sample size
would have condemned a change that is fine.

### Required launch settings (set automatically by `harness.py` / `bench.py`)

```
VLLM_PLUGINS=fleet_mk
VLLM_ROCM_USE_AITER=1
VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1     # selects the unified aiter backend
LLM(..., block_size=16)                      # == fleet_mk PAGE_SIZE (vLLM's ROCm default)
```

`_assert_flash_backend` and the shape/`block_size` asserts in
`_ensure_fleet_mk_kv_bound` fail loud if a mismatched backend or block size is used
(a silent layout mismatch would produce garbage tokens).

> Note: `VLLM_ATTENTION_BACKEND=ROCM_AITER_FA` is **not** used — that string forces
> the removed V0 engine. Backend selection is via the `VLLM_ROCM_USE_AITER*` flags.

### 5b. vLLM version support (0.11.x and 0.27.x from one codebase)

Verified on **0.11.1** and **0.27.1**; `harness.py` picks the right path from what
the installed vLLM exposes, so nothing below needs a flag.

| | 0.11.x | 0.27.x |
|---|---|---|
| KV layout, stock aiter | fleet_mk's split | **packed** `(nb, n_kv, bs, 2*hs)` |
| backend used | stock `rocm_aiter_unified_attn` | `FleetMKAttentionBackend` as `CUSTOM` |
| selection | `VLLM_ROCM_USE_AITER*` env | `LLM(attention_backend=CUSTOM)` |
| `attn.kv_cache` | `list` per virtual engine | the tensor itself |
| embedding method | `get_input_embeddings` | `embed_input_ids` |

On 0.26+ the stock aiter pool went packed, which fleet_mk's flat view cannot alias.
`backend.py` restores the split layout: it subclasses the aiter unified backend
and overrides only `get_kv_cache_shape` and `_split_kv_cache`, so the attention
math and the KV write path stay stock. Attention output is bit-identical to the
packed layout (0.0 max abs diff, with and without sinks) — the 0.26+ kernels are
stride-aware and never required packing. `plugin.register()` registers it;
**registration is not selection**, so `harness.py` also passes
`attention_backend=CUSTOM`.

Three seams change silently rather than raising, which is why each is guarded:

1. **`attn.kv_cache[0]`** meant "this layer's cache" on 0.11 and means "K half of
   block 0" on 0.27. Both index cleanly. `_layer_kv_tensor` discriminates on
   **type**, never on indexability.
2. **`get_input_embeddings` is gone** on 0.27 — a total rename, not an addition.
   `_fleet_mk_embed_fn` probes for both spellings.
3. **aiter is optional now.** The parent impl imports it in `__init__`, and a
   prebuilt aiter linked against a different torch fails with `undefined symbol:
   _ZN3c103hip21warn_or_error_on_syncEv`. `FleetMKUnifiedAttentionImpl` binds
   vLLM's in-tree Triton `unified_attention` instead (the same kernel the parent
   already uses on its non-causal branch), so no aiter install is needed on
   0.27.x.

Gate before running anything: `python -m fleet_megakernel_vllm.check_backend` — 5 checks,
including that the impl **constructs** (an earlier version called
`_split_kv_cache` unbound, so `__init__` never ran and the aiter import escaped
to engine startup) and that attention matches the stock layout numerically.

## 6. Running it

The operational checklist (versions, venv, env vars, hang recovery) lives in
[`RUN.md`](RUN.md). Summary:

Install the plugin into the **vLLM 0.27.1** interpreter so the `fleet_mk` entry
point is discoverable (`VLLM_PLUGINS=fleet_mk` silently no-ops otherwise — you
get stock vLLM and a plausible-looking result):

```bash
# 0.27.1 venv, not system python (which may still be 0.11.1)
/path/to/venv-vllm027/bin/python3 -m pip install -e fleet_megakernel_vllm \
  --no-deps --no-build-isolation
# verify:
/path/to/venv-vllm027/bin/python3 -c "import importlib.metadata as m; print([e.name for e in m.entry_points(group='vllm.general_plugins')])"
```

```bash
cd vllm_integration
export FLEET_MK_MODEL=/path/to/gpt-oss-120b
PY=/path/to/venv-vllm027/bin/python3
# stock baseline
HIP_VISIBLE_DEVICES=0 VLLM_PLUGINS=  $PY -m fleet_megakernel_vllm.harness
# fleet_mk decode (greedy). Prefill is stock vLLM; decode is the megakernel.
HIP_VISIBLE_DEVICES=0 VLLM_PLUGINS=fleet_mk FLEET_MK_TEMP=0 $PY -m fleet_megakernel_vllm.harness
# sampler (two seeds diverge at temp>0)
HIP_VISIBLE_DEVICES=0 VLLM_PLUGINS=fleet_mk FLEET_MK_TEMP=0.8 FLEET_MK_SEED=0 $PY -m fleet_megakernel_vllm.harness
# latency: BENCH_EAGER=1 is required at 120B (graph capture hits a host sync)
HIP_VISIBLE_DEVICES=0 VLLM_PLUGINS=fleet_mk BENCH_EAGER=1 $PY -m fleet_megakernel_vllm.bench
```

`tools/run_vllm_arm.sh fleet_mk 32 180` wraps the harness, kills orphan
`VLLM::EngineCore` children, and polls VRAM before the next arm.

Comparing two configurations (page size, quant, backend) — capture several runs of
*each* build, then:

```bash
python3 -m fleet_megakernel_vllm.compare_runs \
  --group ps128 /tmp/a1.txt /tmp/a2.txt \
  --group ps16  /tmp/b1.txt /tmp/b2.txt /tmp/b3.txt
```

`HIP_VISIBLE_DEVICES=0` **is** honored — verified 2026-08-04, all runs above landed
on GPU 0. An earlier revision of this document claimed HIP ignores it and prescribed
`CUDA_VISIBLE_DEVICES=1`; that was wrong, and it conflicted with this repo's
standing "always use GPU 0" convention.

## 7. Correctness verification (2026-08-04)

Prompt *"Tell me the history of America."*, greedy, 20 tokens. **Both**
architectures now run end-to-end through vLLM.

1. **Backend**: log shows the aiter unified-attention backend on the V1 engine, for
   both models.
2. **Zero-copy aliasing**: the `data_ptr` invariant guard passes for all 36 layers
   (view, not copy), and the `block_size == PAGE_SIZE` assert passes at 128. Both
   models — the dense path could not bind this until `configs/qwen3_8b.yaml`
   dropped to 128.
3. **GPT-OSS output**: `analysisThe user asks: "Tell me the history of America."
   This is a broad request` — coherent, correct channel structure, first token
   `35644` (`'analysis'`), matching the standalone demo's correctness gate.
4. **Qwen3 output**: `<think>\nOkay, the user is asking for the history of America.
   I need to cover the key` — token ids identical to the standalone
   `demo_qwen3_8b.py` run, which is the tighter check (same kernel, same weights,
   different driver).
5. **Stability**: two identical GPT-OSS fleet_mk runs produced bit-identical streams.
   **Superseded 2026-08-24** — that was luck, not a property. Three runs of one
   build at page-16 diverge from each other as early as token 3 (see §5). Do not
   treat a stable stream as a gate; the note below already said so, and the
   page-16 work confirmed it.
6. **Parity vs stock**: GPT-OSS 15/20 tokens identical, Qwen3 14/20, each then
   branching to a fluent synonym (`"This is"` vs `"That's"`; `"I need to cover the
   key"` vs `"Let me start by breaking down"`).

On (6): a difference is expected and is not by itself a defect — fleet_mk runs MXFP4
weights and fp8 activations against stock's bf16, so logits differ slightly and a
near-tie eventually resolves the other way. The discriminator is *qualitative*, per
`check_determinism.py`: corruption shows up as re-emitting a prefill token, doubled
words, or a degenerate repeat, not as a fluent synonym 14 tokens in. Divergence
*position* is not a signal. The reason is that fleet_mk's MoE W2 accumulates 4 experts
into `workspace_f32` with float `atomicAdd`, so summation order — and the last bits
of every logit — varies run to run. Bit-reproducibility is therefore not a property
of this kernel at all.

When comparing two *configurations*, use `compare_runs.py` and read the
between-config divergence against the within-config noise floor from repeated runs
of a single build. A one-run-per-config A/B cannot distinguish the two.

## 8. Latency (bs=1, gfx950, 2026-08-04)

Prefill-cancelling method (`bench.py`): time greedy generate for a short and long
output of the same prompt, difference out the identical prefill:
`ms/token = 1000*(T_long − T_short)/(N_long − N_short)`, 8 vs 128 tokens, best of 3.

**Always state which graph-capture mode a number came from.** `enforce_eager`
disables hipGraph capture, and it is *not* neutral between the two sides: stock
vLLM's decode is a long tail of small kernel launches, so graph capture is most of
its performance, while fleet_mk's decode is ONE megakernel launch and has almost
nothing to gain. Eager mode therefore flatters fleet_mk by a large factor.

| model | stock, graphs ON | stock, eager | fleet_mk (eager) | |
|---|---:|---:|---:|---|
| **GPT-OSS 120B** | **4.658 ms/tok (215 tok/s)** | 26.113 ms/tok | **2.921 ms/tok (342 tok/s)** | **1.59× faster** |
| **Qwen3-8B** | **4.469 ms/tok (224 tok/s)** | 5.720 ms/tok (175 tok/s) | 7.282 ms/tok (137 tok/s) | 1.63× **slower** |

Correcting the baseline moved *both* rows against fleet_mk: GPT-OSS from 8.3× to
1.45×, and Qwen3 from 1.27× slower to **1.63× slower**. GPT-OSS then recovered to
1.59× when the fused-logits change (§8.2) removed vLLM's redundant bf16 `lm_head`,
3.211 → 2.921. Qwen3 is dense and does not have the fused path.

Worth noting on its own: stock decodes 120B-MoE and 8B-dense at nearly the same
rate (4.658 vs 4.469 ms/tok). At bs=1 with graphs, stock's decode is dominated by
per-layer launch and dispatch, not by weight traffic — GPT-OSS activates only a
few experts per token, so its *active* parameter count is not 15× Qwen3's.

An earlier revision of this document reported GPT-OSS as **8.3× faster than
stock**. That number was an artifact: `bench.py` hardcoded `enforce_eager=True`,
so it compared graph-captured fleet_mk against a stock baseline running with graphs
disabled. Stock GPT-OSS with graphs is **4.658 ms/token**, not 26.113 — a 5.6×
swing that belonged entirely to the baseline. The real margin is ~1.45×.

The comparison is still not like-for-like, and in the direction that *disfavours*
fleet_mk's headline: **fleet_mk cannot be graph-captured today** (§8.1), so its column is
eager. Since fleet_mk has little to gain from capture this is a small effect, but it
is unmeasured, and the table says "eager" until it is measured.

**The two architectures land on opposite sides of stock, and that is the finding.**
GPT-OSS runs the fused single-barrier pipeline; Qwen3 still runs the older
multi-barrier dense kernel that the generator was re-converged against but whose
*body* was never rewritten. The dense number is essentially unchanged from the
pre-rebase measurement below (7.33 → 7.28), confirming the regression is in the
kernel, not the plugin.

### 8.1 fleet_mk cannot be hipGraph-captured (open)

Running `bench.py` without `BENCH_EAGER=1` crashes during engine init:

```
File "fleet_megakernel_vllm/mixin.py", line 180, in forward
    cur_pos = int(positions[0].item())
torch.AcceleratorError: HIP error: operation not permitted when stream is capturing
```

`.item()` is a device→host sync, which HIP forbids inside a capture region. There
are two such syncs in `forward` (`cur_pos` and, on the MoE path, `token_id`), and
removing them is more than deleting the `.item()`:

- `cur_pos` drives **host-side control flow** in `decode_step` — `num_pages_used =
  (cur_pos + page_size) // page_size` sizes a *dynamic-length* slice of
  `kv_indices`. A captured graph freezes whatever length was recorded.
- `token_id` is passed **by value** as a launch argument, so a captured graph would
  decode the same token forever.

`token_id` is already solved in the kernel ABI and just unused by the plugin:
`GptConfig` carries a `cur_token_ptr` alongside `cur_token_id`, and the kernel does
`int cur_token = config.cur_token_ptr ? *config.cur_token_ptr : config.cur_token_id;`
— a device-side read, which is capture-safe. The plugin passes `nullptr` today.
`cur_pos` is the harder half and needs the page-count arithmetic moved on-device.

**Bound on the payoff first.** With `FLEET_MK_PROFILE=1` the megakernel's own GPU time
is ~2.38 ms against 2.921 ms end-to-end, so *everything* host-side — scheduler,
sampler, launch, round-trip — is ~0.54 ms/token (was ~0.83 before §8.2 removed the
bf16 `lm_head`). That is the ceiling on what capture could recover, and capture
would only take part of it.

### 8.2 Fused logits (landed)

The megakernel's argmax epilogue computes `sum + bias` for **every** vocab entry
and discards all but the max. We used to throw that away and let vLLM's own bf16
`lm_head` recompute it — a 1.16 GB weight read (201088 × 2880 × 2 B), measured at
**0.200 ms/token** as `wvSplitK_hf_sml_<__hip_bfloat16,…>`, against 315 MB of
MXFP4 weights already resident in the megakernel.

`GptConfig` now carries an optional `logits_output`. Null (the standalone default)
keeps the argmax-only tail; non-null makes the epilogue also store the bf16 row —
a 402 KB write. It is a null check rather than an `#ifdef` so one build serves both
drivers. Since the stored value is the same `val` the argmax compares, at the same
absolute vocab index, the fused row cannot disagree with fleet_mk's own greedy pick.

`compute_logits()` returns the fused row only when the preceding `forward()` was a
fleet_mk decode, and **consumes** that flag: prefill, dummy runs and stock-path
fallbacks fall through to vLLM's `lm_head`. Returning a stale row there would be
wrong output with no error.

| | before | after |
|---|---:|---:|
| vLLM decode | 3.211 ms/tok | **2.921 ms/tok** |
| non-megakernel GPU | ~0.30 ms/tok | 0.096 ms/tok |
| standalone driver | 2.520 ms/tok | 2.502 ms/tok (unchanged) |
| kernel registers | 210 VGPR | 226 VGPR, 0 spills, 2 waves/SIMD |

The extra 16 VGPRs did not cost occupancy, and the standalone path pays only the
predicated branch.

### Where the time goes — GPT-OSS

| Component | per token |
|---|---:|
| fleet_mk megakernel (GPU, from the kernel's own `[FLEET_MK_TIME]` line) | ~2.38 ms |
| standalone driver total (`demo_gpt_oss_120b.py`) | 2.50 ms |
| **through vLLM** | **2.92 ms** |

The plugin costs **~0.42 ms/token** over the standalone driver — vLLM's scheduler,
the sampler, and the host round-trip. It was ~0.68 ms before §8.2.

Qwen3 shows the same shape: 6.85 ms standalone → 7.28 ms through vLLM, i.e. ~0.43 ms
of plugin overhead. Since the two models pay a similar absolute cost through
different kernels, the overhead is a property of the integration, not of either
kernel — and it is not what makes the dense path slow.

### The earlier conclusion, and what survives of it

An earlier revision recorded stock 5.98 vs fleet_mk 7.33 ms/token on Qwen3-8B and
concluded *"the megakernel itself is the bottleneck… fleet_mk is ~23% slower than
stock."* **For the dense path that is still true** — reverified here at 5.72 vs
7.28 (both eager). What changed is that it no longer generalizes: GPT-OSS's fused
kernel is faster than stock, so "the megakernel is the bottleneck" is now a
statement about which kernel, not about fleet_mk.

The integration-level lever named there — emitting logits from fleet_mk to skip vLLM's
redundant bf16 `lm_head` — **landed** (§8.2) and was worth 0.290 ms on GPT-OSS,
inside the 0.4–0.6 ms estimated. It does not apply to the dense path, whose kernel
has no `logits_output` in its ABI, and it would not close the dense gap anyway:
porting the fused pipeline to the dense kernel is the change that would.

## 8.3 Fleet kernel + vLLM plugin (2026-09-03)

The 2.921 ms/token number in section 8 was Titan-under-vLLM, before this tree
switched the kernel body to fleet's headers. After that move the plugin still
used Titan's per-layer counter blocks. Prefill (stock) succeeded; the **first**
megakernel decode hung (GPU 100%, `kfd_wait_on_events`) — layer 2 waiting for
`expected=2` with `observed=1`.

The standalone driver had already been updated to fleet's contract (one shared
monotonic counter block, MoE barrier 160 ints/expert, no per-step zero). The
plugin now matches that driver. Greedy tok1=35644, coherent text, vLLM **0.27.1**,
`BENCH_EAGER=1`, 8 vs 64 tokens: **2.032 ms/token** through vLLM (standalone
driver on the same `.so` is ~1.88 ms/token).

