# -*- coding: utf-8 -*-
# Slide bodies. Each entry: (tag, html). Figures are spliced in by build.py
# via the {FIGn} placeholder, taken verbatim from titan-architecture.html.
#
# Deliberate content constraints (from review):
#   - SHORT titles.
#   - NO explicit performance numbers in the prose. The results figure carries
#     them; everything else states direction ("worse", "the largest overhead").
#     Structural constants (8 XCDs, 36 layers, 8 phases) are architecture, not
#     measurements, and stay.
#   - Spine: problem with dynamic dispatch -> static approach and its goal ->
#     how Titan achieves it -> vLLM + zero-copy KV -> results.

SLIDES = []
def S(tag, html): SLIDES.append((tag, html))

# ─────────────────────────────────────────────────────────── 1. the problem
S("Problem", '''
<h2>Mirage: dynamic task dispatch is not free</h2>

<p>The megakernel systems we build on (MPK / Mirage) run a <b>task graph inside the
kernel</b>:</p>
<ul style="margin:10px 0">
<li>Work is cut into <b>tasks</b>, each with a <code>dependent_event</code> and a
    <code>trigger_event</code>.</li>
<li><b>Scheduler workgroups</b> pop ready events into <b>per-worker queues</b>.</li>
<li>A <b>worker</b> polls its queue, loads a <code>TaskDesc</code>, runs it, bumps an event counter.</li>
</ul>
<p>That machinery buys real things: ragged batches, load balancing, and any workload whose <em>tile
count</em> is only known at runtime — <b>sparse attention</b> above all. Under <b>expert parallelism</b>
across GPUs it earns its keep twice over: top-<i>k</i> routing sends a data-dependent number of tokens
to each expert, so without rebalancing some GPUs sit idle while others queue.</p>

<p><b>None of that applies to dense decode at small batch sizes</b> — and it is still paid for, at
every task boundary, per layer, per token.</p>
''')

# ─────────────────────────────────────────────────────────────────── 2. title
S("", '''
<div style="margin:auto 0">
<h1 style="font-size:46px">Titan</h1>
<p class="sub" style="font-size:20px;max-width:870px">A <b>static generator</b> for fused
decode kernels. One launch runs the whole model, and what each worker does is
decided when the kernel is generated — not while it executes.</p>

<hr class="rule">

<div class="stat-row">
  <div class="stat"><div class="v">no scheduler</div><div class="k">work → worker is compile-time</div></div>
  <div class="stat"><div class="v">no task queue</div><div class="k">no dispatch on the critical path</div></div>
  <div class="stat"><div class="v">one YAML</div><div class="k">to add a model</div></div>
  <div class="stat"><div class="v">one launch</div><div class="k">per token, all layers inside</div></div>
</div>

<p class="muted" style="font-size:14px;margin-top:26px">
AMD MI350X · gfx950 · ROCm &nbsp;·&nbsp; GPT-OSS 120B (MoE, MXFP4) and Qwen3-8B (dense)
&nbsp;·&nbsp; batch size 1, decode only &nbsp;·&nbsp; served through vLLM</p>
</div>
''')

# ─────────────────────────────────────────────────────────── 3. the approach
S("Approach", '''
<h2>Make it static, and generate it</h2>

<div class="box note" style="margin-top:6px">
<p class="big" style="margin:0">If the fast configuration of a dynamic system is the one where
dispatch has been precomputed by hand, emit that end state directly — and emit it from a
<em>description of the model</em>, so it costs nothing to do again for the next one.</p>
</div>

<div class="two" style="margin-top:16px">
<div>
<p><b>Static.</b> No task graph, no scheduler, no queues. Each worker derives the tiles it owns
from <code>static constexpr</code> arithmetic on its own id, and ordering is a fixed sequence of
phases separated by explicit barriers.</p>
<p><b>Generated.</b> That mapping comes from a YAML description — shapes, head counts, quantization,
tile widths — written out as HIP source against the existing device library.</p>
</div>
<div>
<div class="box good" style="margin:0">
<span class="lbl">The goal</span>
<p>Support a new model <b>without writing kernels</b>. The generator can be cheap precisely because
the library is expensive.</p>
</div>
<p class="muted" style="font-size:14.5px;margin-top:12px">The claim is that static is
<em>sufficient at this operating point</em> — not that runtime scheduling is wrong in general.</p>
</div>
</div>
''')

# ─────────────────────────────────────────────────────────── 4. the split
S("How", '''
<h2>What is generated, what is not</h2>
{FIG10}
<div class="two" style="gap:26px;margin-top:2px">
<div>
<p style="margin:0 0 6px"><b>Generated per model, from YAML</b></p>
<ul style="margin:0;font-size:15px">
<li>Derived constants: tile counts, workers per XCD, strides, padded sizes</li>
<li>The layer loop, the end-of-layer barrier, the LM-head tail</li>
<li>One template instantiation per layer body, with its argument list</li>
<li>Host driver, weight packing, build flags, plugin constants</li>
</ul>
</div>
<div>
<p style="margin:0 0 6px"><b>Device function library — hand-written, shared by every model</b></p>
<ul style="margin:0;font-size:15px">
<li>MFMA GEMM mainloops; MXFP4 and FP8 paths</li>
<li>Fused epilogues: SwiGLU, bias, residual add, quantize</li>
<li>Split-KV attention and its merge, RoPE, RMSNorm, TopK</li>
<li>The barrier primitives</li>
</ul>
</div>
</div>
<p class="muted" style="font-size:14.5px;margin:10px 0 0">Upstream reaches this library through a
superoptimizer; Titan reaches it through a table. <b>Neither generates a kernel body.</b></p>
''')

# ─────────────────────────────────────────────────────────── 5. the example
S("How", '''
<h2>What the generator emits</h2>

<div class="two" style="gap:20px">
<div>
<p class="muted" style="margin:0 0 6px;font-size:13.5px">You write this —
<code>configs/qwen3_8b.yaml</code>, near enough entire:</p>
<pre style="margin:0;padding:9px 12px"><code style="font-size:11.2px;line-height:1.44">model:
  arch: dense
  num_layers: 36
  hidden_size: 4096
  num_q_heads: 32
  num_kv_heads: 8
  head_dim: 128
  activation: swiglu
gpu:
  num_xcds: 8
  workers_per_xcd: 30
quantization:
  weight_format: mxfp4
  output_per_wg: 64</code></pre>
</div>
<div>
<p class="muted" style="margin:0 0 6px;font-size:13.5px">You get this — derived, never typed, and
asserted against the device header:</p>
<pre style="margin:0;padding:9px 12px"><code style="font-size:11.2px;line-height:1.44">static constexpr int NUM_XCDS        = 8;
static constexpr int WORKERS_PER_XCD = 30;
static constexpr int HIDDEN_SIZE     = 2944;
static constexpr int NUM_KV_HEADS    = 8;

static constexpr int OPROJ_N_WGS =
        HIDDEN_SIZE / OPROJ_OPW;
static constexpr int OPROJ_N_WGS_PER_XCD =
        OPROJ_N_WGS / NUM_XCDS;
static constexpr int W13_N_WGS =
        W13_OUTPUT_SIZE / W13_OPW;</code></pre>
</div>
</div>
<p class="muted" style="font-size:14.5px;margin:10px 0 0">Those constants <em>are</em> the work
decomposition — what replaces the task graph. Only the tile widths are measured; every other
constant is derived from them by arithmetic.</p>
<div class="box note" style="margin:10px 0 0;padding:10px 15px">
<span class="lbl">The mechanism, deliberately boring</span>
<p>YAML into a validated config object, then string substitution. <b>No IR, no pattern matching, no
autotuner</b> — the derivation is arithmetic, and ~30 asserts encode what the device library assumes,
so an unsupported shape fails at generate time rather than hanging. The counter layout is
<em>parsed</em> out of the device header rather than restated, so it cannot drift.</p>
</div>
''')

# ────────────────────────────────────────────────── 6. the whole emitted kernel
S("How", '''
<h2>The whole model is one kernel</h2>

<p class="muted" style="margin:0 0 7px;font-size:13.5px">Everything the generator emits for
GPT-OSS 120B, abbreviated but structurally complete.</p>

<div class="two" style="gap:20px">
<div>
<pre style="margin:0;padding:9px 13px"><code style="font-size:11.5px;line-height:1.5">__global__ void <b>model_kernel</b>(cfg, ptr_table)
{
  <span class="muted">// who am I — arithmetic, no descriptor</span>
  int <b>tile</b> = xcd * WORKERS_PER_XCD + rank;

  <b>embed</b>(cfg.cur_token);
  <b>barrier_global</b>(embed_done);

  for (layer = 0; layer &lt; <b>NUM_LAYERS</b>; layer++)
    <b>full_layer_fused</b>&lt;<span class="muted">tile widths, counts,</span>
        <span class="muted">head dims, DECODE_ONLY</span>&gt;(
        ptrs[layer], <b>tile</b>, window(layer));

  <b>barrier_global</b>(layer_done);
  <b>rmsnorm</b> → <b>lm_head</b> → <b>argmax</b>;
}</code></pre>
</div>
<div>
<p style="margin-top:0">A model description becomes <b>one templated kernel</b> whose body is calls
into the device library — one per op, with the derived constants as template arguments. Every
<b>bold</b> name already existed in the library.</p>
<p>The generator's whole job is the constant list, the loop, the barriers, and this argument list.
Nothing is searched for; nothing is scheduled.</p>
<div class="box note" style="margin:11px 0 0;padding:10px 15px">
<span class="lbl">Template arguments, not runtime arguments</span>
<p>Tile widths and counts are <code>constexpr</code>, so the compiler unrolls against them and
allocates registers for the shape it will actually run — spending at compile time what the dynamic
version spends per token.</p>
</div>
<p class="muted" style="font-size:13.5px;margin:11px 0 0">Next: what one of those calls expands
to.</p>
</div>
</div>
''')

# ─────────────────────────────────────────────────────────── 7. execution model
S("How", '''
<h2>Inside one layer: the library, called in order</h2>

<div class="two" style="gap:22px">
<div>
<pre style="margin:0;padding:9px 13px"><code style="font-size:11.5px;line-height:1.5"><span class="muted">// this worker's share of layer L, on XCD x</span>

<b>ResAdd→RMSNorm→QKV→RoPE→KV()</b>
<b>qkv_done()</b>      <span class="muted">// 10 tiles/XCD arrive · XCD-local</span>

<b>Attention()</b>      <span class="muted">// KV chunk `rank`</span>
<b>chunks_done()</b>   <span class="muted">// 8 chunks arrive · last one merges</span>

<b>Merge()</b>          <span class="muted">// 8 partials, + sinks</span>
<b>attn_done()</b>     <span class="muted">// 8 merges arrive · frees all 240</span>
                <span class="muted">// O-proj DMA runs inside this wait</span>

<b>O-proj→ResAdd→RMSNorm→Router→TopK()</b>
<b>routing_done()</b>  <span class="muted">// 1 WG per expert (128) · last picks top-4</span>

<b>W13→SwiGLU→W2→accumulate()</b>
<b>layer_done()</b>    <span class="muted">// all 240 arrive · drains the layer</span></code></pre>
</div>
<div>
<p style="margin-top:0">Every name above is a <b>hand-written device function</b>:</p>
<pre style="margin:0;padding:8px 13px"><code style="font-size:11px;line-height:1.45"><span class="muted">// every library function: 256 threads, 4×wave64</span>
template&lt;int HEAD_DIM, int NUM_Q_PER_KV,
         int NUM_KV_HEADS, int PAGE_SIZE,
         bool HAS_QK_NORM&gt;
__device__ void <b>rope_kv_update</b>(
    qkv_output, cos, sin, q_workspace,
    <b>k_cache</b>, <b>v_cache</b>,   <span class="muted">// vLLM's pages</span>
    kv_indptr, <b>kv_indices</b>, …,
    <b>kv_head_idx</b>)      <span class="muted">// = xcd_id</span></code></pre>

</div>
</div>
''')

# ─────────────────────────────────────────────────────────── 8. adding a model
S("How", '''
<h2>Adding a model: Mistral-7B, start to finish</h2>

<div class="two" style="gap:20px">
<div>
<p class="muted" style="margin:0 0 6px;font-size:13.5px"><b>1.</b> Transcribe
<code>config.json</code> into a YAML. Nothing here is a decision — it is the model card:</p>
<pre style="margin:0;padding:8px 12px"><code style="font-size:10.9px;line-height:1.4">model:
  name: mistral-7b
  arch: dense
  num_layers: 32
  hidden_size: 4096
  intermediate_size: 14336
  num_q_heads: 32
  num_kv_heads: 8
  head_dim: 128
  activation: swiglu
gpu:
  num_xcds: 8
  workers_per_xcd: 30
quantization:
  weight_format: mxfp4
  output_per_wg: 64</code></pre>

<p class="muted" style="margin:9px 0 6px;font-size:13.5px"><b>2.</b> Run the generator:</p>
<pre style="margin:0;padding:8px 12px"><code style="font-size:10.9px;line-height:1.4">$ python3 titan_generate.py \\
      configs/mistral_7b.yaml</code></pre>
</div>

<div>
<p class="muted" style="margin:0 0 6px;font-size:13.5px"><b>3.</b> It derives the constants, checks
~30 asserts, and emits <b>everything needed to run</b> — nothing else is written by hand:</p>
<table style="font-size:12.6px;margin:0">
<tbody>
<tr><td style="width:47%"><code>mistral_7b_kernel.cuh</code><br>
    <span class="muted">787 lines</span></td>
    <td>The constants, the layer loop, the barriers, the LM-head tail</td></tr>
<tr><td><code>mistral_7b_launch.hip</code><br><span class="muted">152 lines</span></td>
    <td><code>extern "C"</code> entry point: grid, block, LDS size, one launch, and the
        CPython wrapper Python calls it through</td></tr>
<tr><td><code>demo_mistral_7b.py</code><br><span class="muted">686 lines</span></td>
    <td>Host driver — loads HF weights, <b>quantizes and packs them to the layout the kernel
        addresses</b>, decodes. Also builds the <b>pointer table</b>: every weight and scratch
        address, indexed <code>[XCD][layer][slot]</code>, so the kernel resolves all 32 layers
        without host contact</td></tr>
<tr><td><code>build_mistral_7b.sh</code><br><span class="muted">49 lines</span></td>
    <td>The <code>hipcc</code> line: arch, include paths, every <code>-D</code></td></tr>
</tbody>
</table>

<p class="muted" style="margin:8px 0 6px;font-size:13.5px"><b>4.</b> Build, then run:</p>
<pre style="margin:0;padding:8px 12px"><code style="font-size:10.9px;line-height:1.4">$ bash build_mistral_7b.sh
$ python3 demo_mistral_7b.py --model-path &lt;hf&gt;</code></pre>
</div>
</div>

''')

# ─────────────────────────────────────────────────────────── 11. vLLM
S("Serving", '''
<h2>Prefill on vLLM, decode on the megakernel</h2>

<div class="two" style="gap:22px">
<div>
<p style="margin-top:0">vLLM discovers a <code>vllm.general_plugins</code> entry point at startup
and calls <code>register()</code>, which swaps the stock architecture for a subclass.
<b>No fork, no patch:</b></p>
<pre style="margin:8px 0 0;padding:9px 13px"><code style="font-size:11.5px;line-height:1.5">ModelRegistry.<b>register_model</b>(
    "Qwen3ForCausalLM", TitanQwen3ForCausalLM)

class TitanQwen3ForCausalLM(
        <b>TitanModelMixin</b>, Qwen3ForCausalLM): …</code></pre>
<p style="margin:9px 0 0">Mixin first in the MRO, so its <code>forward()</code> shadows the stock
model's and <code>super().forward()</code> still reaches it. One predicate decides everything:</p>
<pre style="margin:8px 0 0;padding:9px 13px"><code style="font-size:11.5px;line-height:1.5">def forward(self, input_ids, positions, …):
  md = self._get_attn_metadata()

  <span class="muted"># anything that is not a single-token decode</span>
  if md is None or md.max_query_len != 1:
      return <b>super().forward</b>(…)   <span class="muted"># stock vLLM</span>

  <span class="muted"># decode: one token, batch 1</span>
  self._ensure_titan_kv_bound()   <span class="muted"># once → next slide</span>
  self.titan.<b>decode_step</b>(embed, cur_pos,
        block_table=md.block_table[0])</code></pre>
<p class="muted" style="margin:8px 0 0;font-size:13.5px">Prefill is compute-bound and already well
served. The megakernel claims only the memory-bound case it was built for.</p>
</div>
<div>
<p style="margin-top:0">Titan inherits everything it deliberately left out: continuous batching,
the scheduler, preemption, sampling, the OpenAI API, paged KV, prefix caching.</p>
<div class="box warn" style="margin:12px 0 0;padding:11px 15px">
<span class="lbl">The handoff is a pointer, not a copy</span>
<p>Decode reads the KV prefill just wrote, in place — which works only because one vLLM block is
exactly one Titan page, so vLLM's block table drives the kernel's page indirection directly.</p>
</div>
<p class="muted" style="font-size:14.5px;margin-top:10px">The scheduler, sampler and host
round-trip sit between kernel time and end-to-end latency; the results slide separates them rather
than quoting the kernel alone.</p>
</div>
</div>
''')

# ─────────────────────────────────────────────────────────── 12. zero-copy KV
S("Serving", '''
<h2>Zero-copy KV</h2>

<p>The KV cache is the largest allocation in the system, and vLLM lays it out the serving stack's way
while the kernel wants one flat buffer. The obvious bridge — a per-request gather-copy — would put
two full KV stores and a reshuffle on the decode path, every token.</p>

{FIG12}

<pre style="margin:6px 0 4px;padding:8px 13px"><code style="font-size:11.3px;line-height:1.45"><span class="muted">def <b>_ensure_titan_kv_bound</b>(self):   # first decode only — vLLM allocates KV after load_weights</span>
k_alias = attn.kv_cache[0][0].reshape(entries, KV_STRIDE)   <span class="muted"># contiguous reshape = <b>view</b></span>
assert k_alias.data_ptr() == kvc[0].data_ptr()   <span class="muted"># or it silently copied</span>
ptr_table[… + <b>K_CACHE</b>] = k_alias.data_ptr()   <span class="muted"># same slot the demo fills</span></code></pre>

<div class="two" style="gap:20px;margin-top:11px">
<div>
<div class="box warn" style="margin:0">
<span class="lbl">The price: one vLLM block = one Titan page</span>
<p>The engine is launched at <code>block_size = PAGE_SIZE</code>, which also picks the backend: aiter's
MHA path static_asserts <code>BLOCK_SIZE &lt;= 32</code> — why the ROCm default is 16 — so decode runs
on the unified path, which stages a page-sized tile in LDS and tops out at 160.</p>
</div>
</div>
<div>
<div class="box bad" style="margin:0">
<span class="lbl">And prefill inherits it</span>
<p>One pool, one block size — so prefill runs there too, off its tuned default. <b>Cost to prefill
unmeasured.</b> No vLLM code changed; one config value did.</p>
</div>
</div>
</div>
''')

# ─────────────────────────────────────────────────────────── 13. results
S("Results", '''
<h2>Where it lands</h2>
{FIG13}
<div class="two">
<div>
<p>Titan is substantially faster than stock vLLM on the MoE model, and still behind the dynamic
system it is built on — half that gap the price of being a real serving backend, half the kernel's
barriers.</p>
<p>The dense model runs the <em>older</em> multi-barrier kernel — the fused pipeline was never
ported to it — and lands on the wrong side of stock. Its number barely moved across a full plugin
rebase, placing the regression in the kernel, not the integration.</p>
</div>
<div>
<div class="box bad" style="margin:0">
<span class="lbl">Read this before quoting any of it</span>
<p><code>enforce_eager</code> is <b>not neutral between the two sides</b>. Stock vLLM's decode is a
long tail of small launches, so graph capture is most of its performance; Titan's is one launch with
nothing to gain. An earlier revision reported a far larger speedup that belonged entirely to a
handicapped baseline.</p>
</div>
<p class="muted" style="font-size:14.5px;margin-top:12px">Correctness is gated on an exact token
id, not perplexity — a wrong weight layout produces fluent nonsense at full speed.</p>
</div>
</div>
''')

# ─────────────────────────────────────────────────────────── 14. negative results
S("Results", '''
<h2>What did not work</h2>

<p class="muted" style="margin:2px 0 10px">Each is recorded in the model YAML so the next person
does not re-run it. These shaped the design more than the successes.</p>

<table style="font-size:14.5px;margin:0">
<tbody>
<tr><td style="width:47%">Widen the MoE W13 tile to remove all padding tiles</td>
    <td><b>Worse.</b> W13 wants fat tiles — LDS weight reuse across MFMA rounds.</td></tr>
<tr><td>Widen the W2 tile to match it</td>
    <td><b>Worse, in the opposite direction.</b> W2 wants thin tiles: more parallelism to hide
        latency.</td></tr>
<tr><td>Fuse SwiGLU into the W2 quantization step</td>
    <td><b>Large regression.</b> Transcendental cost. More fusion is not monotonically
        better.</td></tr>
<tr><td>Fuse the router GEMV into the preceding RMSNorm</td>
    <td><b>Worse.</b> Loses the quantized path.</td></tr>
<tr><td>Finer-grained per-expert MoE barriers</td>
    <td><b>Failed.</b> Pipeline-drain overhead exceeded the ordering saved.</td></tr>
<tr><td>Delete the inter-layer barrier</td>
    <td><b>Faster, wrong output.</b> Included because it is the tempting one.</td></tr>
<tr><td>Prefetch the SwiGLU bias — an apparently free win</td>
    <td><b>No change.</b> The wait moved to the next stall; the enclosing span was
        constant.</td></tr>
</tbody>
</table>

<div class="box note" style="margin:12px 0 0;padding:10px 16px">
<span class="lbl">The pair that matters is the pair that disagrees</span>
<p>The two MoE tile-width constants pull in <b>opposite</b> directions. Unifying them is the most
tempting wrong move available — and exactly what an autotuner searching one shared tile-size
parameter would do.</p>
</div>
''')

# ─────────────────────────────────────────────────────────── 15. limitations
S("Limitations", '''
<h2>Where this falls over</h2>

<div class="two">
<div>
<p style="margin-top:4px"><b>Scope of the evaluation:</b></p>
<ul style="margin:8px 0">
<li><b>Batch size 1 only.</b> The whole static argument is conditioned on it, and we have not
    measured where ragged batches make the dynamic machinery start earning its cost.</li>
<li><b>One GPU.</b> Every fence is agent-scope; tensor parallelism needs new primitives.</li>
<li><b>Two models, one target.</b> Both are 8-KV-head; the fused path ties KV heads to XCDs.</li>
<li><b>One prompt, greedy decode.</b></li>
</ul>
<p class="muted" style="font-size:14.5px">Also open: the plugin cannot yet be graph-captured,
leaving host time on the table that no kernel work can reach.</p>
</div>
<div>
<div class="box bad" style="margin-top:4px">
<span class="lbl">The thesis, inverted</span>
<p>The generator is cheap <em>because</em> the library is expensive, and nothing here makes writing
that library cheaper. A search that could produce competitive MFMA mainloops would move the
boundary — that, not the assembling layer, is the open problem.</p>
</div>
<div class="box warn" style="margin:14px 0 0">
<span class="lbl">Where dynamic is the right answer</span>
<p>Ragged batches, and <b>sparse attention</b> — which breaks the static argument even at batch size 1,
because the block count per step is genuinely data-dependent. Also any tile shape nobody has
written: Titan rejects those at validate time rather than running them slowly, which is correct
behaviour and also the honest limit of a static system.</p>
</div>
</div>
</div>
''')

# ────────────────────────────────────────────────────────────── 15. next steps
S("Future work", '''
<h2>What we would build next</h2>

<table style="font-size:13.5px;margin:6px 0 0">
<thead><tr><th style="width:26%">Idea</th><th style="width:39%">What it takes</th>
<th>What decides whether it pays</th></tr></thead>
<tbody>
<tr><td><b>Tensor-parallel the expert FFN</b></td>
    <td>Replicate attention and the KV cache, shard only the MoE intermediate. One all-reduce per
        layer, on a buffer that already exists. Needs <b>system-scope</b> fences; every fence today
        is agent-scope.</td>
    <td>The die is already under-subscribed at batch size 1. Sharding removes <em>bytes per GPU</em>,
        not serial work, so the win is bounded by how bandwidth-bound MoE really is.
        Sublinear by construction.</td></tr>
<tr><td><b>Graph-capture the vLLM path</b></td>
    <td>The standalone decode loop is captured; the plugin is not, because launch state lives in
        file-scope statics. Make it per-device and capture per rank.</td>
    <td>Nothing technical. It is host time no kernel work can reach — and the largest remaining
        <em>known</em> win.</td></tr>
<tr><td><b>The fused pipeline on the dense path</b></td>
    <td>Dense models still run the unfused per-op sequence; the eight-phase layer is MoE-only.</td>
    <td>The real generality test — a second <em>architecture</em>, not a second model of the same
        one. Until it passes, the YAML claim is weaker than it sounds.</td></tr>
<tr><td><b>Find the crossover</b></td>
    <td>Ragged batches, and sparse attention. Measure where dynamic dispatch starts earning its
        cost rather than asserting that it does not.</td>
    <td>Nothing — this one we simply owe. The static argument is <em>conditioned</em> on batch size
        1, and we have never measured where that ends.</td></tr>
</tbody>
</table>

<div class="box note" style="margin:10px 0 0;padding:9px 16px">
<span class="lbl">Each of these gets a stopping rule before it gets built</span>
<p>Two GPUs answer the tensor-parallel question; if that is not a real win over one, the remaining
degrees do not get built. The negative results are cheap only because they were <em>designed</em>
to be cheap to abandon.</p>
</div>
''')

# ─────────────────────────────────────────────────────────── 16. related work
S("Related work", '''
<h2>Where this sits</h2>

<table style="font-size:13.5px;margin:8px 0">
<thead><tr><th>System</th><th>How the megakernel is built</th><th>Relation to this work</th></tr></thead>
<tbody>
<tr><td><b>Hazy Research</b><br><span class="cite">Spector et al., 2025</span></td>
    <td>Hand-written on-GPU <b>interpreter</b>; instructions request and release shared-memory
        pages</td>
    <td>Established the bs=1 case. Dynamic at the instruction level — though their throughput
        follow-up builds the schedule <b>ahead of time on the CPU</b>.</td></tr>
<tr><td><b>Mirage / MPK</b><br><span class="cite">Cheng et al., arXiv:2512.22219</span></td>
    <td><b>Compiler plus in-kernel runtime.</b> SM-level task graph, decentralized scheduling, JIT
        task launch for data-dependent attention</td>
    <td><b>The system we build on.</b> We consume its device library and delete its scheduler.</td></tr>
<tr><td><b>Kog monokernel</b><br><span class="cite">Kog Labs, MI300X</span></td>
    <td><b>Programmer-managed compile-time partition.</b> Fixed CU assignment per stage, no dynamic
        task scheduling</td>
    <td>Independent convergence on the static choice, same vendor — but by hand, for one
        model.</td></tr>
<tr><td><b>CODA</b><br><span class="cite">Guo et al., arXiv:2605.19269</span></td>
    <td>Transformer blocks as <b>GEMM-epilogue programs</b> over a fixed mainloop</td>
    <td>Prior work on the epilogue fusion our device library uses. Stops at the kernel boundary;
        one kernel per token removes it.</td></tr>
<tr><td><b>FlashInfer · CK/CUTLASS · Triton</b></td>
    <td>Kernel libraries and DSLs — the layer <em>below</em> all of the above</td>
    <td>The layer that holds the performance. Our claim is about <em>who calls it</em>.</td></tr>
</tbody>
</table>

<div class="box note" style="margin:10px 0 0;padding:10px 16px">
<span class="lbl">The axis, stated carefully</span>
<p>All four call a hand-written device library — <b>that is not the distinction</b>. The axis is
<b>who decides the work assignment, and when</b>. Hazy and MPK decide at runtime; Kog at compile
time by hand, per model. Titan decides at compile time and <b>generates that decision from a model
description</b> — which makes the static choice reusable instead of bespoke.</p>
</div>
''')

# ─────────────────────────────────────────────────────────── 17. takeaways
S("Takeaways", '''
<h2>Takeaways</h2>

<div class="box good" style="margin:12px 0;padding:11px 17px">
<span class="lbl">1 — At batch size 1, dynamic dispatch is a cost that buys nothing</span>
<p style="margin:0;font-size:17px">Queues, event counters and descriptor loads sit on the critical
path of a workload whose work assignment was knowable before launch — and the dynamic system's
fastest build precomputes its dispatch away.</p>
</div>

<div class="box note" style="margin:12px 0;padding:11px 17px">
<span class="lbl">2 — The performance is in the device library, so the layer above can be a table</span>
<p style="margin:0;font-size:17px">Both front ends bottom out in the same hand-written kernels;
what either emits is derived constants and a template instantiation. So adding a model can be a YAML
file instead of a search, at no cost in performance.</p>
</div>

<div class="box warn" style="margin:12px 0;padding:11px 17px">
<span class="lbl">3 — Static is a trade, and the price is legible</span>
<p style="margin:0;font-size:17px">Deleting the launches deletes the free ordering they provided,
and the barriers that replace it are the largest thing Titan adds. We report that next to the
speedup, with the negative results that produced the design.</p>
</div>

<hr class="rule" style="margin:12px 0 6px">
<p class="cite" style="margin:0">Engineering reference — repository shape, reproduction commands,
every measurement: <a href="titan-architecture.html"><code>titan-architecture.html</code></a>.
&nbsp;·&nbsp; Sources —
<a href="https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles">Hazy Research megakernel</a> ·
<a href="https://arxiv.org/abs/2512.22219">MPK, arXiv:2512.22219</a> ·
<a href="https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/">Kog monokernel</a> ·
CODA, arXiv:2605.19269</p>
''')
