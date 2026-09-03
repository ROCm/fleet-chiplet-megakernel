#!/usr/bin/env python3
"""Fleet MK code generator: reads a YAML model config and generates 4 files
for a working persistent kernel on MI350.

Generated files:
  1. generated/{name}_kernel.cuh  -- The persistent kernel
  2. generated/{name}_launch.hip  -- C launch wrapper
  3. demo_{name}.py               -- Python driver
  4. build_{name}.sh              -- Build script

Usage:
    python3 fleet_mk_generate.py configs/qwen3_8b.yaml
"""

import argparse
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import yaml


# ============================================================================
# ModelConfig dataclass
# ============================================================================

@dataclass
class ModelConfig:
    # --- From YAML model section ---
    name: str = ""
    arch: str = "dense"
    num_layers: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    num_q_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    activation: str = "swiglu"
    has_qk_norm: bool = False
    has_bias: bool = False
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 40960

    # --- From YAML gpu section ---
    target: str = "gfx950"
    num_xcds: int = 8
    workers_per_xcd: int = 30
    page_size: int = 4096
    max_seq_len: int = 512
    num_kv_chunks: int = 8
    # gfx950 exposes 160 KB LDS per workgroup, of which 155648 B is available at
    # runtime. A hardware property, not a model one -- hence the default, and
    # hence `gpu.lds_bytes` being optional in the YAML. It is a config field
    # rather than a bare literal because it appears at six sites in the MoE
    # launch wrapper and disagreement between any two of them is a launch
    # failure at best and a silent LDS overrun at worst.
    lds_bytes: int = 155648

    # --- From YAML quantization section ---
    weight_format: str = "mxfp4"
    output_per_wg: int = 64
    oproj_opw: int = 0               # O-proj OPW (0 = auto-compute)
    gateup_opw: int = 128

    # --- From YAML moe section (only when arch == "moe") ---
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0

    # --- From YAML attention section ---
    sliding_window: int = 0
    # "uniform" (one window for every layer) or "alternating" (even layers get
    # the window, odd layers are unlimited). Deliberately NOT a general
    # per-layer pattern: the byte target is a single ternary, and generalizing
    # past these two cases buys nothing today.
    sliding_window_pattern: str = "uniform"

    # --- From YAML build section ---
    opt_level: str = "O2"
    rdc: bool = False
    extra_defines: list = field(default_factory=list)
    # Which megakernel header tree to compile against: "mirage" or "fleet".
    # These are NOT interchangeable -- the trailing runtime argument of the
    # fused layer body is sliding_window_override in mirage and task_layer_idx
    # in fleet, and neither header diagnoses receiving the other's. The dense
    # configs stay on "mirage"; only gpt-oss has been ported to fleet's
    # contract, so the default must not move.
    headers: str = "mirage"
    header_dir: str = "/home/claudeuser/mirage"
    # Emit the FLEET_MK_NO_W13_PREFETCH shell escape hatch (fleet arm only). This
    # is not just extra_defines with a flag: the point is an A/B pair that
    # differs in NOTHING but the prefetch, which a plain define cannot give.
    w13_prefetch_toggle: bool = False

    # --- From YAML measured section ---
    # Tuning constants that came out of a search, not a formula. Values that CAN
    # be derived are still declared here and cross-checked against the
    # derivation in load_and_validate, so a silent drift fails the load.
    w13_output_per_wg: int = 128
    w2_output_per_wg: int = 64
    moe_total_tiles_per_xcd: int = 0

    # --- Derived constants (computed in load_and_validate) ---
    name_clean: str = ""
    name_title: str = ""
    padded_vocab_size: int = 0
    padded_hidden_size: int = 0       # hidden_size padded to mfma_align
    padded_intermediate_size: int = 0  # intermediate_size padded to mfma_align
    mfma_align: int = 128             # MFMA tile alignment
    q_per_kv: int = 0
    total_workers: int = 0
    qkv_output_size: int = 0
    gateup_output_size: int = 0
    oproj_reduction: int = 0

    qkv_n_wgs: int = 0
    qkv_n_wgs_per_xcd: int = 0
    oproj_n_wgs: int = 0
    oproj_n_wgs_per_xcd: int = 0
    gateup_n_wgs: int = 0
    gateup_n_wgs_per_xcd: int = 0
    down_n_wgs: int = 0
    down_n_wgs_per_xcd: int = 0
    lm_n_wgs: int = 0
    lm_n_wgs_per_xcd: int = 0

    qkv_wg_bytes: int = 0
    oproj_wg_bytes: int = 0
    gateup_wg_bytes: int = 0
    down_wg_bytes: int = 0
    lm_wg_bytes: int = 0

    # MoE-specific derived constants
    padded_moe_intermediate_size: int = 0
    w13_output_size: int = 0          # 2 * padded_moe_intermediate for SwiGLU
    w13_n_wgs: int = 0
    w13_wg_bytes: int = 0
    w2_n_wgs: int = 0
    w2_wg_bytes: int = 0
    # MFMA K iterations across the MoE reduction: 16x16x128 tiles, so
    # padded_hidden_size / 128. Derived rather than written into the driver's
    # prose by hand, because that prose sits next to the K-stride knob and the
    # whole point of the note is that the reduction is the thing that does NOT
    # move when the stride does.
    moe_k_mfma_iters: int = 0
    router_n_wgs: int = 0
    router_wg_bytes: int = 0

    kv_cache_stride: int = 0
    q_workspace_stride: int = 0

    ptrs_in: int = 0
    ptrs_out: int = 9
    ptrs_per_layer: int = 0

    # Parsed from kernels/device_functions.cuh, never declared here -- see
    # parse_counters_per_layer().
    counters_per_layer: int = 0


def _pad_up(val, align):
    """Round val up to next multiple of align."""
    return ((val + align - 1) // align) * align


# ============================================================================
# Counter buffer layout (fused MoE path)
# ============================================================================
#
# The per-layer counter block is described ONCE, here, as an ordered list of
# (slot_name, cache_lines, description). Everything downstream -- the C++
# `SLOT_*` constants, the comment map above them, and the Python-side buffer
# sizing in the demo -- is emitted from this list.
#
# Why it matters: these offsets appear in BOTH the kernel .cuh and
# demo_gpt_oss_120b.py. A mismatch does not fail to build and does not crash;
# it corrupts a live barrier and produces garbage tokens. Hand-maintaining two
# copies of a running sum is exactly the kind of thing that drifts.
#
# Each entry is (slot_name, cache_lines, map_text, range_override).
#   slot_name       -- None means mirage addresses the region internally and
#                      there is no named Fleet MK constant; it still consumes
#                      cache lines, so it still moves everything after it.
#   map_text        -- prose for the comment map. May contain {p0}, {p1}, ...
#                      placeholders, where {pN} is this region's own first cache
#                      line plus N, so a description that names sub-offsets
#                      cannot go stale when a region above it changes size.
#   range_override  -- the comment map's first entry was hand-written as
#                      "[0..9*16-1]" rather than the regular "[0*16..9*16]".
#                      Kept as an override so the numbers stay derived and the
#                      bytes stay identical; there is exactly one of these.
COUNTER_REGIONS = [
    (None,                     10, "OProj Mechanism C (10 cache lines)",
     "[0..9*16-1]"),
    (None,                      9, "routing_ready (9 cache lines)", None),
    (None,                      1, "attn_global_counter", None),
    (None,                      8, "qkv_epoch (8 per-XCD)", None),
    (None,                      8, "chunk_barrier, mirage's site (8 per-XCD)", None),
    (None,                      8, "attn_release (8 per-XCD)", None),
    # Everything from here to the end of the layer barrier is owned by FLEET's
    # layer body, which fleet_mk compiles against. Fleet moved the chunk barrier
    # from 28 to 48 (it needs 8*NUM_REQS lines and at NUM_REQS>=3 the old site
    # grew through the fused-tail literals) and added a layer barrier above it
    # at FULL_LAYER_LAYER_BARRIER_SLOT(NUM_REQS) = 48*16 + 128*NUM_REQS for 272
    # ints. Sized here at NUM_REQS=1. Mirage's 28..35 site above is kept
    # reserved too, so one map is valid against either header.
    (None,                      4, "fleet fused-tail literals "
                                   "(moe/resadd/lmhead done)", None),
    (None,                      8, "chunk_barrier, fleet's site (8 * NUM_REQS)", None),
    (None,                     17, "fleet layer barrier "
                                   "(local {p0}..{p7}, global {p8}, "
                                   "release {p9}..{p16})", None),
    ("SLOT_QKV_BARRIER_NEW",    8,
     "qkv_barrier arrival (input_ptrs[7], 8 ints padded to 8 cache lines)",
     None),
    ("SLOT_LAYER_DONE_NEW",     1, "layer_done (end-of-layer barrier)", None),
    ("SLOT_LAYER_LOCAL_NEW",    8,
     "layer_local (per-XCD local arrive, 8 cache lines)", None),
    ("SLOT_TAIL_LMHEAD_NEW",    1, "tail_lmhead", None),
    ("SLOT_TAIL_ARGMAX_NEW",    3, "tail_argmax", None),
    ("SLOT_LAYER_RELEASE_NEW",  8,
     "layer_release (per-XCD release flags, 8 cache lines)", None),
    ("SLOT_LAYER_DONE_GLOBAL",  1, "layer_done_global", None),
]


def parse_counters_per_layer() -> int:
    """COUNTERS_PER_LAYER read out of kernels/device_functions.cuh.

    The header is the single source of truth -- mirage's device code sizes its
    per-layer counter block from it, and the generated kernel and demo both
    have to agree or they scribble on a live barrier. Parsing beats
    re-declaring: the generator cannot drift from a header it reads.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "kernels", "device_functions.cuh")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    m = re.search(
        r"^static constexpr int COUNTERS_PER_LAYER\s*=\s*(\d+)\s*\*\s*(\d+)\s*;",
        src, re.M)
    assert m, f"COUNTERS_PER_LAYER not found in {path}"
    return int(m.group(1)) * int(m.group(2))


def counter_slots() -> dict:
    """{slot_name: (cache_line_index, cache_lines)} from the running sum."""
    slots, off = {}, 0
    for name, lines, _, _ in COUNTER_REGIONS:
        if name is not None:
            slots[name] = (off, lines)
        off += lines
    return slots


def counter_lines_used() -> int:
    return sum(lines for _, lines, _, _ in COUNTER_REGIONS)


def _region_bounds(marker: str) -> tuple:
    """(first, last) cache line of the region whose map text starts with marker.

    Lets the prose around the map cite sub-ranges by name instead of by literal,
    so a region changing size cannot leave a paragraph quietly describing the
    old layout. The regions cited this way are all unnamed (mirage/fleet own
    them internally), which is why there is no SLOT_* constant to look up.
    """
    off = 0
    for _, lines, text, _ in COUNTER_REGIONS:
        if text is not None and text.startswith(marker):
            return off, off + lines - 1
        off += lines
    raise KeyError(marker)


def _fleet_mk_owned_first() -> int:
    """First cache line Fleet MK owns: everything below belongs to the layer body."""
    off = 0
    for name, lines, _, _ in COUNTER_REGIONS:
        if name is not None:
            return off
        off += lines
    raise AssertionError("no Fleet MK-owned counter region")


def check_counter_region_collisions(cfg: ModelConfig) -> None:
    """Reject fleet flags that allocate counter lines Fleet MK already owns.

    The counter buffer is shared: the layer body addresses every counter as an
    offset from one `oproj&#95;counters&#95;base` pointer, so a fleet flag that claims a
    new region claims it out of the SAME arithmetic our SLOT&#95;* constants use.
    COUNTER&#95;REGIONS reserves fleet's known regions, but a flag added to
    build.extra&#95;defines later can introduce one that is not reserved, and the
    failure mode is the usual one for this buffer -- not a build error and not a
    crash, but two barriers sharing memory so neither can complete.

    MPK&#95;ROUTER&#95;XCD&#95;FOLD is the live example. It publishes per-XCD O-proj
    slice-ready flags at

        FULL&#95;LAYER&#95;OPROJ&#95;XCD&#95;READY&#95;SLOT(n) = 48*16 + 128*n + 272

    (gang&#95;full&#95;layer&#95;fused&#95;mi300.cuh:81, with the matching pointer computed at
    gang&#95;linear&#95;mxfp4&#95;res&#95;bias&#95;rmsnorm&#95;topk&#95;mi300.cuh:416). At NUM&#95;REQS=1 that
    is int 1168 = cache line 73 -- exactly the first line Fleet MK owns, which
    today is SLOT&#95;QKV&#95;BARRIER&#95;NEW. Both headers ARE in our include set, so this
    is one YAML line away from being live; it is not live today only because the
    flag is absent. Fleet does not hit it because its own layer body has no
    barrier there and its demo.py sizes counter&#95;size with a trailing +128.

    Checked here, where the error can name the YAML key, rather than left to be
    discovered as garbage tokens. Fixing it is a layout change (move Fleet MK's
    slots up by 8 lines, which fits: COUNTERS&#95;PER&#95;LAYER has headroom), not a
    flag change -- so the guard says that instead of just refusing.
    """
    if "MPK_ROUTER_XCD_FOLD" not in cfg.extra_defines:
        return
    fold_first = _region_bounds("fleet layer barrier")[1] + 1
    owned_first = _fleet_mk_owned_first()
    if fold_first + 8 <= owned_first:
        return
    collided = next(n for n, (off, _) in counter_slots().items()
                    if off == owned_first)
    raise ValueError(
        f"MPK_ROUTER_XCD_FOLD allocates 8 cache lines at "
        f"FULL_LAYER_OPROJ_XCD_READY_SLOT(1) = line {fold_first} "
        f"({fold_first}..{fold_first + 7}), but Fleet MK's own slots start at "
        f"line {owned_first} ({collided}). They would share memory and neither "
        f"barrier could complete -- silently, as wrong tokens. Reserve the "
        f"region in COUNTER_REGIONS (which moves Fleet MK's slots up 8 lines) "
        f"before adding this flag to build.extra_defines.")


def emit_counter_ownership() -> str:
    """The one-sentence split between the layer body's half and Fleet MK's."""
    first = _fleet_mk_owned_first()
    return (f"lines 0..{first - 1} belong to the layer body and Fleet MK must not "
            f"touch them, and\n"
            f"// lines {first}..{counter_lines_used() - 1} are Fleet MK's own.")


def emit_counter_reserved_note(cfg: ModelConfig) -> str:
    """Why the reserved half is sized for fleet, on the fleet arm only.

    Both fleet's chunk-barrier site and mirage's are reserved, so one map is
    valid against either header -- the cost is 8 dead cache lines and the
    benefit is that flipping build.headers cannot silently move a live barrier.
    """
    if cfg.headers != "fleet":
        return ""
    mir_c0, mir_c1 = _region_bounds("chunk_barrier, mirage's site")
    fl_c0, _ = _region_bounds("chunk_barrier, fleet's site")
    tail0, tail1 = _region_bounds("fleet fused-tail literals")
    lb0, lb1 = _region_bounds("fleet layer barrier")
    return f'''//
// The reserved half is sized for FLEET's layer body, not mirage's, because the
// -I paths point at fleet. Fleet moved the chunk barrier from {mir_c0} to {fl_c0} (it
// needs 8*NUM_REQS lines and at NUM_REQS>=3 the old site grew through the
// fused-tail literals at {tail0}..{tail1}) and added a layer barrier above it at
// FULL_LAYER_LAYER_BARRIER_SLOT(NUM_REQS) = {fl_c0}*16 + 128*NUM_REQS, 272 ints.
// At NUM_REQS=1 that is {lb0}..{lb1}. Mirage's {mir_c0}..{mir_c1} chunk barrier is kept reserved
// as well so the same map is valid against either header.
//
'''


def emit_driver_qkv_slot_note(cfg: ModelConfig) -> str:
    """Why SLOT_QKV_BARRIER sits where it does, for the driver's copy.

    The kernel's counter map is not visible from the driver, and this constant
    is the driver's ONLY view of it -- it is handed to the layer body as
    input_ptrs[7]. So the enumeration is repeated here rather than cross
    referenced, with every boundary derived from COUNTER_REGIONS so the two
    copies cannot disagree.

    Fleet arm only: on mirage's headers there is no fused-tail literal region to
    collide with, and the paragraph would describe a layout that build does not
    have.
    """
    if cfg.headers != "fleet":
        return ""
    body_last = _fleet_mk_owned_first() - 1
    op0, op1 = _region_bounds("OProj Mechanism C")
    rr0, rr1 = _region_bounds("routing_ready")
    ag0, _ = _region_bounds("attn_global_counter")
    qe0, qe1 = _region_bounds("qkv_epoch")
    mc0, mc1 = _region_bounds("chunk_barrier, mirage's site")
    ar0, ar1 = _region_bounds("attn_release")
    ft0, ft1 = _region_bounds("fleet fused-tail literals")
    fc0, fc1 = _region_bounds("chunk_barrier, fleet's site")
    lb0, lb1 = _region_bounds("fleet layer barrier")
    return f'''#
# This is the pointer handed to the layer body as input_ptrs[7] (the per-XCD
# QKV epoch arrival line), so it must not alias any slot the layer body itself
# uses. The layer body owns lines {op0}..{body_last}: {op0}..{op1} oproj, {rr0}..{rr1} routing_ready,
# {ag0} attn_global, {qe0}..{qe1} qkv_epoch, {mc0}..{mc1} chunk barrier (mirage's site),
# {ar0}..{ar1} attn_release, {ft0}..{ft1} fused-tail literals, {fc0}..{fc1} chunk barrier
# (fleet's site, 8*NUM_REQS), {lb0}..{lb1} fleet's layer barrier. {ft0}*16 -- the value
# this used to hold, chosen against mirage's map alone -- lands on fleet's
# FUSED_TAIL_MOE_DONE_SLOT, so the QKV epoch and one of the layer body's own
# barriers would share memory and neither could complete.
# Must stay equal to SLOT_QKV_BARRIER_NEW in {cfg.name_clean}_kernel.cuh.
'''


def emit_counter_total_note(cfg: ModelConfig) -> str:
    """Tie the map's line count back to COUNTERS_PER_LAYER, and flag the
    slots that outlive their use.

    Fleet MK's own layer-barrier slots stay ALLOCATED under fleet even though the
    barrier itself is compiled out -- deallocating them would renumber
    everything above and buy 18 cache lines nobody is short of.
    """
    return f'''// Total {counter_lines_used()} cache lines = COUNTERS_PER_LAYER. Fleet MK's own layer barrier
// (layer_done / layer_local / layer_release / layer_done_global) is still live
// here because mirage's layer body has no layer barrier of its own; it is
// deleted when the -I paths move to fleet, whose body ends with one.
'''


def emit_decode_iter_note(cfg: ModelConfig) -> str:
    """Why the rank counter's quotient is a valid decode-iteration index."""
    if cfg.headers != "fleet":
        return ""
    return '''\
    // The raw value carries the launch index for free: this counter is a host
    // allocation that is never reset, one launch is one decode step, and
    // exactly WORKERS_PER_XCD workers bump each XCD's line per launch (the
    // `% WORKERS_PER_XCD` below already depends on that). So the quotient is
    // the same number on every worker of every XCD within a launch, and it is
    // monotonic across the whole run -- which is what fleet's layer body needs
    // for task_layer_idx. Derived here rather than plumbed from Python so
    // there is no second copy to drift.
    //
    // "One launch is one decode step" is the load-bearing clause, and it is
    // exactly what a persistent decode loop breaks: the atomic fires once per
    // worker per LAUNCH, so across N steps inside one launch the quotient
    // would freeze at its entry value. Frozen is not merely stale -- fleet
    // reads it as "no layer has run yet", so every barrier from step 2 onward
    // is already satisfied and the workers stream past data nobody wrote.
    // That surfaces as fluent-but-wrong text, not as a crash. When
    // decode_ctrl->iter_base_p1 is non-zero the host owns the epoch instead
    // and this quotient is unused; the atomic still supplies xcd_rank either
    // way, which is the half that is genuinely per-launch.
'''


def emit_decode_iter_decl(cfg: ModelConfig) -> str:
    return "    __shared__ int s_decode_iter;\n" if cfg.headers == "fleet" else ""


def emit_xcd_rank_atomic(cfg: ModelConfig) -> str:
    """The rank atomic. On the fleet arm the same atomic yields both values."""
    if cfg.headers != "fleet":
        return ("        s_xcd_rank = "
                "atomicAdd(&rank_counters[xcd_id * 16], 1) % WORKERS_PER_XCD;\n")
    return '''\
        int raw = atomicAdd(&rank_counters[xcd_id * 16], 1);
        s_xcd_rank = raw % WORKERS_PER_XCD;
        s_decode_iter = raw / WORKERS_PER_XCD;
'''


def emit_decode_iter_read(cfg: ModelConfig) -> str:
    """decode_iter_base: host-supplied epoch when offered, else the atomic's.

    The BASE, not the epoch: this launch may cover several decode steps, and
    the loop adds `iter` to get the epoch of each one. The two agree token for
    token at one step per launch -- measured, not assumed (2.044 ms median
    either way, same tok1=35644).

    The select is a scalar branch on a value uniform across the whole grid, so
    it costs one s_cmp and no divergence. Reading decode_ctrl unconditionally
    would fault the three callers that pass a null pointer for it.
    """
    if cfg.headers != "fleet":
        return ""
    return '''\
    int decode_iter_base = (decode_ctrl && decode_ctrl->iter_base_p1)
                               ? decode_ctrl->iter_base_p1 - 1
                               : s_decode_iter;
'''


def emit_decode_iter_step(cfg: ModelConfig) -> str:
    """This iteration's epoch, inside the loop body."""
    if cfg.headers != "fleet":
        return ""
    return "        int decode_iter = decode_iter_base + iter;\n"


def emit_persist_iters(cfg: ModelConfig) -> str:
    """How many decode steps this launch covers.

    One launch of this kernel costs ~320 us/token that has nothing to do with
    the model: a rocprofv3 kernel+HIP trace of 430 steady-state tokens measured
    2024.9 us inside the dispatch against a 1893 us in-kernel s_memrealtime
    reading (~132 us of wave ramp-up and drain that the device timer cannot
    see), plus a 188.0 us median gap between one dispatch ending and the next
    beginning. Looping inside the launch pays that once per chunk instead of
    once per token.

    Zero means "not supplied", so every caller that passes a zeroed or null
    DecodeControl -- which is all of them until the driver opts in -- gets
    exactly one step and the pre-loop behaviour.
    """
    if cfg.headers != "fleet":
        return "    int const n_decode_iters = 1;\n\n"
    return '''\
    int const n_decode_iters =
        (decode_ctrl && decode_ctrl->max_decode_tokens > 0)
            ? decode_ctrl->max_decode_tokens : 1;

'''


def emit_persist_prologue(cfg: ModelConfig) -> str:
    """Per-iteration prologue: consume the last step's token, advance the KV.

    A port of decode_bridge_kernel (the 1-block kernel the FLEET_MK_PIPE graph
    path already runs between launches), moved inside the loop. It runs on
    thread 0 of worker (0,0) only, and needs no barrier of its own:

      - argmax -> cur_token is read by THE SAME THREAD that wrote it last
        iteration, so no cross-worker visibility is required. The broadcast to
        the rest of the workgroup is the __syncthreads() below, and worker
        (0,0) is the only worker that gathers the embedding.
      - kv_indptr / kv_last_page_len are read by all 240 workers, but they are
        written here BEFORE (0,0) arrives at the embed barrier, which ends in
        buffer_inv. No worker can reach layer 0 without (0,0) having arrived,
        so every reader acquires them there. This is the same producer ->
        embed-barrier -> consumer edge the embedding itself already uses.

    Both metadata ints reach the layer body as raw `int const *` -- the KV
    write slot and the RoPE angle both come from a compute_pos lambda that
    re-reads them every layer -- so a device-side write does propagate. Nothing
    is cached in a register across the loop.

    Guarded on iter > 0: the first step's token and metadata come from the host
    exactly as before, which is what keeps n_decode_iters == 1 bit-identical.
    """
    if cfg.headers != "fleet":
        return ""
    return '''\
        if (iter > 0 && xcd_id == 0 && xcd_rank == 0) {
            // Every thread of this workgroup loads the token itself rather than
            // having tid 0 broadcast it through LDS. A single __shared__ int
            // here takes static LDS from 736 to 744, and 744 % 16 == 8
            // misaligns the dynamic-LDS base the MoE and QKV staging phases
            // index from -- the +1.09 ms/token regression. Measured at 744 on
            // the first two builds of this loop; this is what took it back to
            // 736. The load is a broadcast of one address across 256 threads,
            // which the scalar cache serves once.
            //
            // Restricted to worker (0,0) because it is the only worker that
            // reads cur_token for anything that matters -- it alone gathers the
            // embedding. The other 239 keep their launch-time value, which is
            // all touch_lds_pad wants of it, exactly as at one step per launch.
            // Confining the read this way also keeps the loop correct after
            // stage 3 removes the end-of-body barrier: tid 0 of THIS workgroup
            // wrote argmax_output in the previous iteration's tail, so the
            // ordering is workgroup-local and needs no device-wide barrier.
            cur_token = static_cast<int>(
                static_cast<long long *>(config.argmax_output)[0]);
            if (tid == 0) {
                int pos = decode_ctrl->start_pos + iter;
                if (decode_ctrl->token_output_buf) {
                    decode_ctrl->token_output_buf[iter - 1] = cur_token;
                }
                decode_ctrl->kv_indptr[1] = (pos + PAGE_SIZE) / PAGE_SIZE;
                decode_ctrl->kv_last_page_len[0] = (pos % PAGE_SIZE) + 1;
            }
        }

'''


def emit_persist_epilogue(cfg: ModelConfig) -> str:
    """End-of-body barrier, on the multi-step path only.

    The argument that this barrier is unnecessary is a real one, and it is
    written out in emit_persist_prologue: every producer -> consumer edge
    across the loop boundary is already covered, the last of them by the embed
    barrier at the top of the next iteration. But "I reasoned that no barrier
    is needed" is exactly the class of claim this kernel has been burned by,
    and the failure mode here is fluent-but-wrong text rather than a crash --
    the kind that survives a casual read of the output. So the loop is proven
    correct with the barrier in place, and the barrier is removed afterwards
    against a measurement (stage 3), not against an argument.

    It is emitted inside `if (n_decode_iters > 1)` rather than outside, so the
    one-step path executes not one extra instruction and stays bit-identical.

    The expected value is a plain local, not the __shared__ the embed barrier
    uses for the same job. That is not a style choice: a single extra
    __shared__ int takes static LDS from 736 to 744, and 744 % 16 == 8
    misaligns the dynamic-LDS base the MoE and QKV staging phases index from --
    the +1.09 ms/token regression, which the build reports as its "LDS Size
    [bytes/block]" line and nothing else would catch. Measured at 744 on the
    first build of this loop.

    It is correct because barrier_global reads `expected` only inside
    `if (tid == 0)`, the same thread that computes it here. Padding LDS back to
    752 instead would not work -- LLVM strips padding that is never read, which
    is the whole reason touch_lds_pad exists.
    """
    if cfg.headers != "fleet":
        return ""
    return '''
        // Provisional: see the note on this emitter. Removed in stage 3 on a
        // measurement, not on the argument that it is redundant.
        if (n_decode_iters > 1) {
            int *loop_done  = counter_buf + SLOT_LOOP_DONE;
            int *loop_local = counter_buf + SLOT_LOOP_LOCAL;
            // Local, not __shared__: see the LDS note on this emitter.
            // barrier_global reads `expected` on tid 0 only.
            int loop_expected = 0;
            if (tid == 0) {
                int cur = __atomic_load_n(loop_done, __ATOMIC_RELAXED);
                loop_expected = ((cur / NUM_XCDS) + 1) * NUM_XCDS;
            }
            fleet_mk::barrier_global(loop_done, loop_expected, TOTAL_WORKERS,
                                  loop_local, xcd_id, WORKERS_PER_XCD);
        }
'''


def emit_touch_lds_pad(cfg: ModelConfig) -> str:
    """Keep the LDS pad words live, once per decode step.

    This is the whole 3.61 -> 2.548 ms fix: without a live reference LLVM strips
    the padding, static LDS falls back to 728, and 728 % 16 == 8 misaligns the
    dynamic-LDS base that the MoE and QKV staging phases index from.
    """
    if cfg.headers != "fleet":
        return ""
    return '''
        // Hold the window slot's padding live so static LDS stays 736 (a
        // multiple of 16). This is the +1.09 ms fix -- see the note at the
        // LayerWindowSlot declaration. One-time, not per layer.
        fleet_mk::touch_lds_pad(tid, cur_token);
'''


def emit_full_window(cfg: ModelConfig) -> str:
    """FLEET_MK_FULL_WINDOW, the finite stand-in for the unlimited window.

    Only emitted on the fleet arm, where the window is a template parameter and
    the odd (full-attention) layers need a value that keeps them on the same
    code path as the sliding ones.
    """
    if cfg.headers != "fleet":
        return ""
    return f'''// Window value used for the "full attention" (odd) layers. Any value >=
// MAX_SEQ_LEN is arithmetically identical to the unlimited window 0, since
// every guard in fleet's attention path tests `seqlen_k > sliding_window` and
// seqlen_k tops out at MAX_SEQ_LEN - 1. Passing a finite window instead of 0
// keeps all {cfg.num_layers} layers on one code path -- that is worth ~0.8 ms/token.
static constexpr int FLEET_MK_FULL_WINDOW = MAX_SEQ_LEN;
'''


def emit_counter_map(cfg: ModelConfig) -> str:
    """The `//   [a*16..b*16] : description` comment map, ranges computed."""
    out, off = [], 0
    for _, lines, text, override in COUNTER_REGIONS:
        if text is None:
            off += lines
            continue
        if override is not None:
            rng = override
        elif lines == 1:
            rng = f"[{off}*16]"
        else:
            rng = f"[{off}*16..{off + lines - 1}*16]"
        text = text.format(**{f"p{i}": off + i for i in range(lines)})
        out.append(f"//   {rng:<15}: {text}")
        off += lines
    return "\n".join(out)


# Emission order of the C++ SLOT_* constants, which is NOT the layout order --
# the header lists the end-of-layer barrier slots first. Comments are
# templates over the computed offsets so the "53..60" style ranges cannot go
# stale when a region above them changes size.
COUNTER_SLOT_EMISSION = [
    ("SLOT_LAYER_DONE_NEW",    "global arrival counter ({n} cache line)"),
    ("SLOT_LAYER_LOCAL_NEW",
     "per-XCD local arrival ({n} cache lines: {first}..{last})"),
    ("SLOT_TAIL_LMHEAD_NEW",   None),
    ("SLOT_TAIL_ARGMAX_NEW",   None),
    ("SLOT_QKV_BARRIER_NEW",   None),
    ("SLOT_LAYER_RELEASE_NEW",
     "per-XCD release flags ({n} cache lines: {first}..{last})"),
    ("SLOT_LAYER_DONE_GLOBAL",
     "global done counter for release ({n} cache line)"),
]


def emit_counter_slots(cfg: ModelConfig) -> str:
    """The `static constexpr int SLOT_* = N * 16;` block."""
    slots = counter_slots()
    width = max(len(n) for n, _ in COUNTER_SLOT_EMISSION)
    out = []
    for name, comment in COUNTER_SLOT_EMISSION:
        first, lines = slots[name]
        decl = f"static constexpr int {name:<{width}} = {first} * 16;"
        if comment is None:
            out.append(decl)
        else:
            out.append(decl + "  // " + comment.format(
                n=lines, first=first, last=first + lines - 1))
    return "\n".join(out)


# ============================================================================
# Mirage pointer table -- ONE definition, three consumers
# ============================================================================
# The kernel's MIRAGE_IN_COUNT / MIRAGE_OUT_COUNT / PTRS_PER_LAYER /
# SLOT_LAYER_OUTPUT and the demo's 10368-entry table build are four views of
# the same fact. They were maintained by hand in two files; the failure mode is
# not a build error but an off-by-one that hands mirage the wrong buffer, so
# the tokens are garbage and nothing reports why.
#
# Each entry is (python_expr, comment). Order IS the ABI -- mirage indexes
# input_ptrs[]/output_ptrs[] positionally, so moving a line here silently
# rewires the kernel. The comment carries the mirage-side name.
MIRAGE_IN = [
    ("buf_moe_workspace_f32.data_ptr(),", "workspace_f32"),
    ("buf_residual.data_ptr() if li == 0 else buf_oproj_out.data_ptr(),",
     "residual"),
    ("norm_w1,", "norm_weight_pre"),
    ("buf_norm_scratch1.data_ptr(),", "norm_scratch_pre"),
    ("qkv_weight_xcd,", "qkv_weight"),
    ("qkv_bias_xcd,", "qkv_bias"),
    ("attn_sinks,", "attn_sinks (per-layer, bf16)"),
    ("qkv_barrier_ptr,", "qkv_barrier"),
    ("buf_lse_acc.data_ptr(),", "lse_acc"),
    ("oproj_weight_xcd,", "oproj_weight"),
    ("oproj_bias_xcd,", "oproj_bias"),
    ("norm_w2,", "norm_weight_post"),
    ("buf_norm_scratch2.data_ptr(),", "norm_scratch_post"),
    ("router_bf16_xcd,", "router_weight (bf16 per-XCD)"),
    ("router_bias_xcd,", "router_bias (per-XCD)"),
    ("logits_scratch_xcd,", "logits_scratch (per-XCD)"),
    ("counter_ptr,", "oproj_counters (counter base)"),
    ("w13_weight_base,", "moe_gate_up_weight"),
    ("w2_weight_base,", "moe_down_weight"),
    ("w13_bias,", "moe_w13_bias"),
    ("w2_bias,", "moe_w2_bias"),
    ("buf_moe_barrier.data_ptr(),", "moe_barrier"),
    ("buf_swiglu_out.data_ptr(),", "moe_swiglu_out"),
    ("buf_o_acc_f32.data_ptr(),", "o_acc_f32"),
    # Slots 24/25 were appended rather than inserted next to [17]/[18]: order is
    # the ABI, and every index below 24 is already baked into the layer body
    # (input_ptrs[0..23]) and into the kernel's `layer_ptrs[16]` counter fetch.
    # Appending kept all of those correct for free -- which is also why the
    # fleet flip could repurpose these two without touching anything else.
    #
    # What they carry now depends on FLEET_MK_QKV_PF, and the emitted note below
    # is the full account of why. It is emitted, not left here as generator
    # source, because the reader who needs it is looking at the pointer table
    # in the driver, not at this list.
    ("(next_qkv_weight_xcd if FLEET_MK_QKV_PF else w13_scale_base),", ""),
    ("(prefetched_qkv_weight_xcd if FLEET_MK_QKV_PF else w2_scale_base),", ""),
]

# Emitted verbatim above the pointer-table entry it annotates. Kept as a module
# constant rather than a third tuple element so the entry list stays a uniform
# (expr, comment) table, and kept verbatim rather than transcribed for the same
# reason as FLEET_SHIM_PREAMBLE -- it records an A/B measurement and two
# superseded conclusions, and re-typing prose is how the generator and the
# artifact drifted apart in the first place.
PTR_PRE_NOTES = {
    ("in", 24): """# [24]/[25] -- FLEET'S QKV-PREFETCH HAND-OFF.
#
# These used to carry fleet_mk's MoE scale bases, which was a real
# collision and the root cause of the "+1.09 ms dual-body
# penalty". Fleet reads these two slots at exactly five sites
# (gang_full_layer_fused_mi300.cuh :458, :1693, :1719, :1912,
# :1917) and every one of them is the QKV weight prefetch --
# see the block above the loop where these are computed.
#
# Nothing on the MoE side wants them: fleet's MoE call site
# (:1259-1271) passes input_ptrs[12,17,18,19,20,21,22] only,
# and gang_moe_fused_mxfp4_kernel_mi300 takes 12 parameters
# with no scale pointers at all -- it reads MXFP4 scales inline
# from the packed slab. MPK_MOE_SPLIT_SCALES has ZERO
# occurrences in fleet's tree, so fleet_mk's -D for it was already
# being silently ignored. The scale bases are still computed
# above because the non-fleet build path passes them; under
# fleet's header they simply have no consumer.
#
# What the old values did: slot 24 was a valid non-null address,
# so the staging DMA fired on all 36 layers and pulled MoE scale
# bytes into the QKV LDS region, while slot 25 never equalled
# slot 4, so weights_preloaded stayed false and QKV re-read from
# HBM anyway. Full cost, zero benefit.
#
# MEASURED, A/B on ONE .so with only these two integers changed
# (-DMPK_ENABLE_DEVICE_TASK_TIMING, medians over 126144 phase
# lines each, tools/fused_phase_stats.py --compare):
#
#     phase          collided    fixed    delta
#     qkv_gemm         15.40     13.50    -1.90
#     qkv_attn         22.00     20.20    -1.80
#     xcd_barrier     225.30    218.10    -7.20
#     moe              45.60     45.60     0.00
#     total           312.40    303.50    -8.90
#
# End to end that is ~11 us/token (layers 3390.5 -> 3379.7 us,
# non-timing build), text coherent, next_token=35644 on iter 1.
#
# NOTE WHAT THIS REFUTES. The collision was written up as the
# root cause of the "+1.09 ms dual-body penalty", on the theory
# that the bogus DMA was polluting both weight-staging phases.
# It is not: moe moves 0.00 us. The prefetch is real and worth
# having, but it is a ~11 us/token effect, two orders of
# magnitude short of 1.09 ms. The dual-body cost remains
# unexplained -- see dual-layer-body-cost.md, and do not treat
# this as having closed it.
#
# NEGATIVE RESULT, superseded but recorded: nulling both slots
# appeared to fail 3 runs of 3. That verdict was invalid. Those
# runs carried FLEET_MK_MOE_SPLIT_SCALES/K_STRIDE/N_STRIDE env
# values against a .so built WITHOUT the matching -D flags, and
# that mismatch faults deterministically at iter 1 (next_token
# =-1, HIP error 700) on the shipping .so too. The knobs and the
# build flags must agree; the demo prints both for exactly this
# reason.""",
}

MIRAGE_OUT = [
    ("buf_x_output.data_ptr(),",
     "x_output (separate intermediate, NOT residual)"),
    ("buf_k_cache[li].data_ptr(),", "k_cache"),
    ("buf_v_cache[li].data_ptr(),", "v_cache"),
    ("buf_q_workspace.data_ptr(),", "q_workspace"),
    ("buf_attn_out.data_ptr(),", "o_acc (attn output)"),
    ("buf_oproj_out.data_ptr(),", "attn_proj_out (= next layer's residual)"),
    ("buf_topk_weight.data_ptr(),", "topk_weight"),
    ("buf_routing_indices.data_ptr(),", "routing_indices"),
    ("buf_active_expert_ids.data_ptr(),", "active_expert_ids"),
    ("buf_topk_weight.data_ptr(),", "moe_routing_weight (= topk_weight)"),
    ("buf_moe_workspace_f32.data_ptr(),", "moe_workspace_f32"),
]

# +1 for the trailing layer_output slot, used only by the last layer's ResAdd.
MIRAGE_PTRS_PER_LAYER = len(MIRAGE_IN) + len(MIRAGE_OUT) + 1


# ============================================================================
# The gang_full_layer_fused_kernel_mi300 call -- order is data
# ============================================================================
# Every one of the 23 template parameters is an int or a bool, and all 24
# runtime arguments are int/float/pointer. Transposing any two of them compiles
# without a diagnostic and produces garbage tokens. So the order lives in a
# list, checked against the arity mirage declares, rather than in 47 lines of
# hand-aligned text that a careless edit can permute.
#
# Names are mirage's, from
# include/mirage/persistent_kernel/tasks/mi300/gang_full_layer_fused_mi300.cuh
# (template at :49-72, signature at :74-102). Keep them in sync with that file.
MIRAGE_TEMPLATE_PARAMS = [
    ("QKV_BATCH_SIZE",        "1"),
    ("QKV_OUTPUT_PER_WG",     "OUTPUT_PER_WG"),
    ("QKV_REDUCTION_SIZE",    "HIDDEN_SIZE"),
    ("ACTUAL_HIDDEN_DIM",     "ACTUAL_HIDDEN_DIM"),
    ("HEAD_DIM",              "HEAD_DIM"),
    ("NUM_Q_PER_KV",          "NUM_Q_PER_KV"),
    ("PAGE_SIZE",             "PAGE_SIZE"),
    ("MAX_SEQ_LEN",           "MAX_SEQ_LEN"),
    ("NUM_KV_CHUNKS",         "NUM_KV_CHUNKS"),
    ("Q_WORKSPACE_STRIDE",    "Q_WORKSPACE_STRIDE"),
    ("KV_CACHE_STRIDE",       "KV_CACHE_STRIDE"),
    ("NUM_KV_HEADS",          "NUM_KV_HEADS"),
    # Emitted as the macro's (SW_) parameter, never a literal. Which value
    # reaches it is the whole subject of the dispatch ladder below: on the
    # mirage arm it is always 0 and the real window arrives as the
    # sliding_window_override runtime argument; on the fleet arm there is no
    # such argument, so the window must be a template value -- either two
    # compile-time instantiations, or one instantiation plus the shim that
    # forwards fleet_mk::g_layer_sliding_window. Baking a literal in here would
    # foreclose all three.
    ("SLIDING_WINDOW",        "(SW_)"),
    ("HAS_SINKS",             "1"),
    ("OPROJ_OUTPUT_PER_WG",   "OPROJ_OPW"),
    ("OPROJ_REDUCTION_SIZE",  "OPROJ_REDUCTION"),
    ("NUM_EXPERTS",           "NUM_EXPERTS"),
    ("TOPK_K",                "NUM_TOPK"),
    ("MOE_INTERMEDIATE_SIZE", "MOE_INTERMEDIATE_SIZE"),
    ("MOE_HIDDEN_SIZE",       "HIDDEN_SIZE"),
    ("MOE_W13_OUTPUT_PER_WG", "W13_OPW"),
    ("MOE_W2_OUTPUT_PER_WG",  "W2_OPW"),
    ("DECODE_ONLY",           "true"),
]

# (mirage parameter name, expression). The name is documentation only -- the
# call site passes these positionally, with no /*name=*/ comments.
MIRAGE_RUNTIME_ARGS = [
    ("input_ptrs",              "mirage_in"),
    ("output_ptrs",             "mirage_out"),
    ("cos_ptr",                 "config.cos_ptr"),
    ("sin_ptr",                 "config.sin_ptr"),
    ("qo_indptr",               "config.qo_indptr"),
    ("kv_indptr",               "config.kv_indptr"),
    ("kv_indices",              "config.kv_indices"),
    ("kv_last_page_len",        "config.kv_last_page_len"),
    ("num_active_tokens",       "config.num_active_tokens"),
    ("qkv_n_wgs_per_xcd",       "QKV_N_WGS_PER_XCD"),
    ("kv_stride",               "KV_CACHE_STRIDE"),
    ("q_ws_stride",             "Q_WORKSPACE_STRIDE"),
    ("attn_scale",              "config.attn_scale"),
    ("total_qkv_tiles_per_xcd", "QKV_N_WGS_PER_XCD"),
    ("oproj_n_wgs_per_xcd",     "OPROJ_N_WGS_PER_XCD"),
    ("oproj_output_stride",     "HIDDEN_SIZE"),
    ("router_tile_n",           "ROUTER_TILE_N"),
    ("total_oproj_tiles",       "OPROJ_N_WGS"),
    ("total_topk_tiles",        "TOTAL_TOPK_TILES"),
    ("oproj_tiles_per_xcd",     "OPROJ_TILES_PER_XCD"),
    ("moe_total_tiles_per_xcd", "MOE_TOTAL_TILES_PER_XCD"),
    ("workers_per_xcd",         "WORKERS_PER_XCD"),
    ("tile_idx",                "tile_idx"),
    # The trailing argument means DIFFERENT THINGS in the two header trees and
    # neither diagnoses receiving the other's, so the name here is mirage's but
    # the expression is a local whose definition the dispatch block picks:
    # sliding_window_override (mirage) or task_layer_idx (fleet).
    ("sliding_window_override", "layer_trailing_arg"),
]

# Arity as declared by mirage. A change on either side should stop the build
# here, not surface as wrong tokens.
assert len(MIRAGE_TEMPLATE_PARAMS) == 23, len(MIRAGE_TEMPLATE_PARAMS)
assert len(MIRAGE_RUNTIME_ARGS) == 24, len(MIRAGE_RUNTIME_ARGS)


# Column layout of the emitted table. The hand-written original is not quite
# uniform -- three lines drifted -- and since demo_gpt_oss_120b.py is a byte
# target, the drift is reproduced rather than tidied. Cleaning it up is a
# separate, deliberate change; doing it here would hide it inside a refactor.
#   MIRAGE_IN[1]  overflows the comment column (it carries a conditional), and
#                 the author left one space after "[1]" instead of two.
#   MIRAGE_IN[8]  and [23] sit one column right of everything else.
_PTR_COMMENT_COL = 35
_PTR_INDEX_COL = 5
_PTR_COL_FIXUPS = {("in", 8): (36, 5), ("in", 23): (36, 5),
                   ("in", 1): (_PTR_COMMENT_COL, 4),
                   # 24/25 carry no name, just the index, so the index field is
                   # zero-width; 24's expression is short enough to still pad.
                   ("in", 24): (61, 0), ("in", 25): (_PTR_COMMENT_COL, 0)}


def emit_mirage_ptr_list(which: str, indent: str = " " * 16) -> str:
    """The body of the `mirage_in = [` / `mirage_out = [` literal."""
    entries = MIRAGE_IN if which == "in" else MIRAGE_OUT
    lines = []
    for i, (expr, comment) in enumerate(entries):
        note = PTR_PRE_NOTES.get((which, i))
        if note is not None:
            lines += [indent + s if s else indent.rstrip()
                      for s in note.split("\n")]
        pad, iw = _PTR_COL_FIXUPS.get((which, i),
                                      (_PTR_COMMENT_COL, _PTR_INDEX_COL))
        col = expr.ljust(pad) if len(expr) < pad else expr + "  "
        lines.append(f"{indent}{col}# {f'[{i}]'.ljust(iw)}{comment}")
    return "\n".join(lines)


def mirage_out_idx(name: str) -> int:
    """Index of a mirage output pointer by its comment name.

    The kernel indexes mirage_out[] with bare integers. Looking them up by name
    means a reordering of MIRAGE_OUT moves the kernel's index with it instead
    of leaving it pointing at whatever slid into that slot.
    """
    for i, (_, comment) in enumerate(MIRAGE_OUT):
        if comment.split(" ")[0] == name:
            return i
    raise KeyError(f"no mirage_out named '{name}'; have "
                   f"{[c.split(' ')[0] for _, c in MIRAGE_OUT]}")


# ---------------------------------------------------------------------------
# Counter regions that live AFTER the per-layer blocks.
#
# The driver allocates one flat int32 buffer; the kernel indexes into it with
# compile-time offsets. EMBED_BARRIER_BASE in the kernel and counter_total_ints
# in the driver are the same running sum written in two languages, so they are
# emitted from this one list. Get them out of step and the embedding barrier
# either lands inside the last layer's counters or past the end of the
# allocation -- both silent.
#
# (name, expr, inline_comment, pre_comment_lines)
TRAILING_COUNTERS = [
    ("RANK_COUNTER_INTS", "NUM_XCDS * 16", None, []),
    ("DECODE_ITER_COUNTER_INTS", "16",
     "1 cache line for decode iteration barrier", []),
    ("EMBED_BARRIER_INTS", "(1 + NUM_XCDS) * 16", None, [
        "Embedding-write barrier: 1 global cache line + NUM_XCDS per-XCD lines.",
        "Kept outside the per-layer blocks, whose {lines} cache lines are fully used.",
        "Must match SLOT_EMBED_DONE / SLOT_EMBED_LOCAL in the kernel.",
    ]),
    ("ILB_PROBE_INTS", "20 * 16", None, [
        "Slack for the FLEET_MK_ILB_TIMING diagnostic probe barrier (compiled out in",
        "production). Always allocated so a timing build needs no Python change.",
    ]),
    ("LOOP_BARRIER_INTS", "(1 + NUM_XCDS) * 16", None, [
        "End-of-decode-step barrier for the persistent loop: 1 global cache line",
        "+ NUM_XCDS per-XCD lines, same shape as the embedding barrier.",
        "Appended LAST so it cannot shift EMBED_BARRIER_BASE, which is the sum of",
        "everything declared above it.",
        "Allocated unconditionally -- one decode step per launch never touches it,",
        "and 144 ints of a multi-megabyte buffer is not worth a conditional.",
    ]),
]
# The region the kernel's EMBED_BARRIER_BASE must point at. Everything declared
# before it in TRAILING_COUNTERS is part of that offset.
_EMBED_REGION = "EMBED_BARRIER_INTS"
_LOOP_REGION = "LOOP_BARRIER_INTS"


def emit_trailing_counters(cfg: ModelConfig, indent: str = " " * 4) -> str:
    """The driver's trailing-region declarations plus the total."""
    lines = []
    for name, expr, inline, pre in TRAILING_COUNTERS:
        for c in pre:
            lines.append(f"{indent}# {c.format(lines=cfg.counters_per_layer // 16)}")
        tail = f"  # {inline}" if inline else ""
        lines.append(f"{indent}{name} = {expr}{tail}")
    terms = ["NUM_LAYERS * COUNTERS_PER_LAYER"] + [n for n, _, _, _ in
                                                   TRAILING_COUNTERS]
    lines.append(_wrap_sum(f"{indent}counter_total_ints = (", terms))
    return "\n".join(lines)


def _wrap_sum(head: str, terms: list) -> str:
    """`head` + terms joined by ` + `, greedy-wrapped at 79 columns.

    Continuation lines lead with the operator and align under the opening
    paren, which is what the hand-written driver does.
    """
    cont = " " * len(head)
    out, cur = [], head + terms[0]
    for t in terms[1:]:
        if len(cur) + 3 + len(t) <= 79:
            cur += f" + {t}"
        else:
            out.append(cur)
            cur = f"{cont}+ {t}"
    out.append(cur + ")")
    return "\n".join(out)


def emit_embed_barrier_base(indent: str = " " * 4) -> str:
    """The kernel's EMBED_BARRIER_BASE expression, from the same list."""
    return _region_base(_EMBED_REGION, indent)


def emit_loop_barrier_base(indent: str = " " * 4) -> str:
    """The kernel's LOOP_BARRIER_BASE expression, from the same list."""
    return _region_base(_LOOP_REGION, indent)


def _region_base(region: str, indent: str) -> str:
    """Offset of one TRAILING_COUNTERS region: the sum of everything before it.

    Same running sum the driver computes in Python, so the two cannot drift.
    """
    terms = ["NUM_LAYERS * fleet_mk::COUNTERS_PER_LAYER"]
    for name, expr, _, _ in TRAILING_COUNTERS:
        if name == region:
            return f"{indent}{' + '.join(terms)};"
        terms.append(expr)
    raise KeyError(f"no trailing-counter region named '{region}'")


# ---------------------------------------------------------------------------
# Per-layer weight slots.
#
# The demo packs NUM_LAYERS * WEIGHTS_PER_LAYER tensors into one flat list and
# later unpacks them by integer offset from `base = li * WEIGHTS_PER_LAYER`.
# Those two sites are the same fact written twice: an append inserted in the
# middle shifts every later offset, and nothing complains -- the assert on the
# total length still passes, and the kernel just receives the wrong buffer.
#
# The appends themselves are NOT contiguous in the driver; each one follows the
# packing code that produces its tensor. So this table drives three things: the
# WEIGHTS_PER_LAYER count, the `# [N] name` comment on each scattered append,
# and the one contiguous unpack block.
#
# (append_expr, unpack_var, comment)
WEIGHT_SLOTS = [
    ("qkv_packed",             "qkv_weight_base",     "qkv_weight"),
    ("qkv_bias",               "qkv_bias",            "qkv_bias"),
    ("o_packed",               "oproj_weight_base",
     "oproj_weight (OPW={cfg.oproj_opw})"),
    ("o_bias",                 "oproj_bias",          "oproj_bias"),
    ("norm_w1.contiguous()",   "norm_w1",             "norm_weight_pre"),
    ("norm_w2.contiguous()",   "norm_w2",             "norm_weight_post"),
    ("r_packed",               "router_weight_base",  "router_weight (MXFP4)"),
    ("r_bias",                 "router_bias",         "router_bias"),
    ("w_router_bf16.contiguous()", "router_weight_bf16", "router_weight_bf16"),
    ("gu_packed",              "w13_weight_base",
     "w13_weight [E, w13_n_wgs, wg_bytes]"),
    ("w13_bias",               "w13_bias",            "w13_bias"),
    ("dp_packed",              "w2_weight_base",
     "w2_weight [E, w2_n_wgs, wg_bytes]"),
    ("w2_bias",                "w2_bias",             "w2_bias"),
    ("w_sinks",                "attn_sinks",          "attn_sinks"),
]


def weight_slot(name: str) -> int:
    """Index of a weight slot by its unpack variable name."""
    for i, (_, var, _) in enumerate(WEIGHT_SLOTS):
        if var == name:
            return i
    raise KeyError(f"no weight slot named '{name}'; have "
                   f"{[v for _, v, _ in WEIGHT_SLOTS]}")


def emit_weight_append(cfg: ModelConfig, name: str,
                       indent: str = " " * 8) -> str:
    """One `weight_tensors.append(...)  # [N] comment` line."""
    i = weight_slot(name)
    expr, _, comment = WEIGHT_SLOTS[i]
    return (f"{indent}weight_tensors.append({expr})  "
            f"# [{i}] {comment.format(cfg=cfg)}")


def emit_weight_unpack(indent: str = " " * 12) -> str:
    """The contiguous `name = weight_ptrs_host[base + N]` block."""
    return "\n".join(f"{indent}{var} = weight_ptrs_host[base + {i}]"
                     for i, (_, var, _) in enumerate(WEIGHT_SLOTS))


# Width of the emitted macro. Every continued line is padded to exactly this
# many columns with the backslash in the last one, so the continuations form a
# straight edge -- and so a future argument longer than the current widest one
# fails the assert in emit_mirage_call instead of silently jagging the block.
MACRO_WIDTH = 80


def emit_mirage_call(cfg: ModelConfig, indent: str = " " * 16) -> str:
    """The full templated call, template args commented, runtime args bare.

    Emitted as a `#define FLEET_MK_LAYER_BODY(SW_)` rather than as a bare call,
    because the fleet arm needs the SAME 47-argument list at up to four call
    sites (two compile-time windows, the one-body shim path, the legacy
    two-instantiation ladder). Writing it once and letting the preprocessor
    stamp it out is the only form where those sites cannot drift in an
    argument -- and a transposed argument here compiles clean and produces
    garbage tokens.
    """
    lines = ["#define FLEET_MK_LAYER_BODY(SW_)",
             f"{indent[:-4]}kernel::gang_full_layer_fused_kernel_mi300<"]
    for i, (name, val) in enumerate(MIRAGE_TEMPLATE_PARAMS):
        tail = ">(" if i == len(MIRAGE_TEMPLATE_PARAMS) - 1 else ","
        lines.append(f"{indent}/*{name}=*/{val}{tail}")
    for i, (_, val) in enumerate(MIRAGE_RUNTIME_ARGS):
        lines.append(f"{indent}{val}" + ("," if i < len(MIRAGE_RUNTIME_ARGS) - 1
                                         else ")"))
    # The terminator carries no backslash and no semicolon: the macro is
    # invoked as `FLEET_MK_LAYER_BODY(0);`, so the semicolon belongs to the call.
    body, last = lines[:-1], lines[-1]
    widest = max(len(s) for s in body)
    assert widest < MACRO_WIDTH, f"macro line {widest} >= {MACRO_WIDTH} cols"
    return "\n".join([s.ljust(MACRO_WIDTH - 1) + "\\" for s in body] + [last])


def emit_layer_sliding_window(cfg: ModelConfig) -> str:
    """The per-layer attention-window expression, as a C++ initializer.

    Two cases only, matching `sliding_window_pattern`. GPT-OSS alternates
    (even = sliding, odd = full); everything else uses one window for every
    layer. Deliberately not generalized to an arbitrary per-layer pattern --
    that would need a lookup table in the kernel, and no config wants one.
    """
    if cfg.sliding_window_pattern == "alternating":
        return "(layer & 1) == 0 ? SLIDING_WINDOW : 0"
    return "SLIDING_WINDOW"


def emit_layer_dispatch(cfg: ModelConfig) -> str:
    """The whole per-layer dispatch: trailing-arg choice, macro, ladder.

    The fleet arm's #ifdef ladder is emitted UNCONDITIONALLY, not gated on
    cfg.headers. That is deliberate: the arms select on FLEET_MK_FLEET_HEADERS,
    which is a compile-time define, and the mirage configs must keep compiling
    the mirage arm out of the same source. Making the generator pick would put
    the choice in two places -- YAML and -D -- that could disagree, and the
    failure mode of disagreeing is silent wrong output, not a build error.
    """
    return "\n".join([
        FLEET_DISPATCH_PRE.format(sw=emit_layer_sliding_window(cfg)),
        emit_mirage_call(cfg),
        FLEET_DISPATCH_POST,
    ])


# ---------------------------------------------------------------------------
# Host ABI: the C entrypoints in launch.hip and the ctypes argtypes in the
# driver.
#
# These two are the same list written twice, in two files, in two languages,
# with no compiler between them. ctypes does not validate against the .so --
# it packs whatever the Python list says and calls. Get one entry out of step
# and the callee reads a pointer where an int was pushed. That is stack
# corruption, and because the driver selects among four entrypoints by env var,
# it reproduces under one code path and not the other three.
#
# (c_decl, ctypes_type, name, comment_override)
#   c_decl -- the C parameter text, minus the name
#   ctypes_type -- what the driver packs
#   comment_override -- when the two files' trailing comments differ

# The 20 parameters every entrypoint takes, in order.
ABI_CORE = [
    ("int",                    "c_int",    "num_active_tokens", None),
    ("float",                  "c_float",  "attn_scale",        None),
    ("void *",                 "c_void_p", "cos_ptr",           None),
    ("void *",                 "c_void_p", "sin_ptr",           None),
    ("int *",                  "c_void_p", "qo_indptr",         None),
    ("int *",                  "c_void_p", "kv_indptr",         None),
    ("int *",                  "c_void_p", "kv_indices",        None),
    ("int *",                  "c_void_p", "kv_last_page_len",  None),
    ("void **",                "c_void_p", "ptr_table",         None),
    ("void **",                "c_void_p", "counter_buf_vp",    "counter_buf"),
    ("void *",                 "c_void_p", "lm_norm_weight",    None),
    ("void *",                 "c_void_p", "lm_norm_scratch",   None),
    ("void *",                 "c_void_p", "lm_mxfp4_weight",   None),
    ("void *",                 "c_void_p", "lm_bias",           None),
    ("void *",                 "c_void_p", "argmax_output",     None),
    ("void *",                 "c_void_p", "logits_output",
     "logits_output (bf16 [padded_vocab]; null = argmax only)"),
    ("unsigned long long *",   "c_void_p", "timing_buf",        None),
    ("void *",                 "c_void_p", "embed_weight",      None),
    ("int",                    "c_int",    "cur_token_id",      None),
    ("DecodeControl *",        "c_void_p", "decode_ctrl",       None),
    ("hipStream_t",            "c_void_p", "stream",            None),
]

# Per-entrypoint tails. A `None` c_decl marks a comment-only divider line,
# which both files carry.
ABI_TAILS = {
    "launch": [],
    "graph_capture": [
        (None, None, "Zero buffer info", None),
        ("void *", "c_void_p", "counter_buf_raw",       None),
        ("size_t", "c_size_t", "counter_buf_bytes",     None),
        ("void *", "c_void_p", "workspace_buf_raw",     None),
        ("size_t", "c_size_t", "workspace_buf_bytes",   None),
        ("void *", "c_void_p", "moe_barrier_buf_raw",   None),
        ("size_t", "c_size_t", "moe_barrier_buf_bytes", None),
    ],
    "pipe_capture": [
        (None, None, "Bridge kernel args", None),
        ("void *", "c_void_p", "moe_barrier_raw",     None),
        ("int",    "c_int",    "moe_barrier_ints",    None),
        ("int *",  "c_void_p", "cur_token_ptr",       None),
        ("int *",  "c_void_p", "token_output_buf",    None),
        ("int *",  "c_void_p", "step_counter",        None),
        ("int",    "c_int",    "start_pos",           None),
        ("int",    "c_int",    "page_size",           None),
        (None, None, "Workspace memset", None),
        ("void *", "c_void_p", "workspace_buf_raw",   None),
        ("size_t", "c_size_t", "workspace_buf_bytes", None),
    ],
    "decode_loop": [
        (None, None, "Decode loop params", None),
        ("int",   "c_int",    "total_steps",      None),
        ("int",   "c_int",    "start_pos",        None),
        ("int",   "c_int",    "page_size",        None),
        ("int *", "c_void_p", "token_output_buf",
         "token_output_buf (host-pinned)"),
    ],
}

# Two entrypoints rename a core parameter without changing its type or
# position. Kept as an explicit override rather than four near-copies of
# ABI_CORE, so the shared prefix stays literally shared.
ABI_RENAMES = {
    "decode_loop": {"cur_token_id": "first_token_id"},
}

# The driver's trailing comment on decode_ctrl is longer in exactly one place.
ABI_CTYPES_COMMENT_OVERRIDES = {
    "launch": {"decode_ctrl":
               "decode_ctrl (GPU pointer to DecodeControl struct)"},
}

# The C side is nearly comment-free -- the parameter names carry the meaning
# there, whereas the ctypes list is a column of bare `ctypes.c_void_p` and
# needs every one. Only these two say anything the name does not.
ABI_C_COMMENTS = {
    "decode_loop": {"token_output_buf":
                    "host-pinned or device buffer for output tokens"},
}


def abi_params(entry: str):
    """(c_decl, ctypes_type, name, comment) for one entrypoint, in order."""
    renames = ABI_RENAMES.get(entry, {})
    out = []
    for decl, ct, name, override in ABI_CORE + ABI_TAILS[entry]:
        name = renames.get(name, name)
        out.append((decl, ct, name, override or name))
    return out


def emit_ctypes_argtypes(entry: str, indent: str = " " * 8) -> str:
    """The body of a `lib.<prefix>_<entry>.argtypes = [` literal.

    Column layout matches the hand-written driver: the ctypes type is padded
    to a fixed width so the trailing comments line up.
    """
    overrides = ABI_CTYPES_COMMENT_OVERRIDES.get(entry, {})
    lines = []
    for decl, ct, name, comment in abi_params(entry):
        if decl is None:
            lines.append(f"{indent}# {comment}")
            continue
        lines.append(f"{indent}{('ctypes.' + ct + ',').ljust(20)}"
                     f"# {overrides.get(name, comment)}")
    return "\n".join(lines)


def emit_c_signature(entry: str, indent: str = " " * 4) -> str:
    """The parameter list of one C entrypoint, closing paren included.

    The other half of emit_ctypes_argtypes. `void *` and friends carry their
    own trailing space in ABI_CORE, so the star binds to the name the way the
    hand-written file writes it.
    """
    params = abi_params(entry)
    real = [p for p in params if p[0] is not None]
    last = real[-1]
    comments = ABI_C_COMMENTS.get(entry, {})
    lines = []
    for decl, _, name, comment in params:
        if decl is None:
            lines.append(f"{indent}// {comment}")
            continue
        sep = ")" if (decl, name) == (last[0], last[2]) else ","
        tail = f"  // {comments[name]}" if name in comments else ""
        space = "" if decl.endswith("*") else " "
        lines.append(f"{indent}{decl}{space}{name}{sep}{tail}")
    return "\n".join(lines)


def load_and_validate(config_path: str) -> ModelConfig:
    """Parse YAML, compute all derived constants, validate."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = ModelConfig()

    # Model section
    m = raw["model"]
    cfg.name = m["name"]
    cfg.arch = m.get("arch", "dense")
    cfg.num_layers = m["num_layers"]
    cfg.hidden_size = m["hidden_size"]
    cfg.intermediate_size = m["intermediate_size"]
    cfg.vocab_size = m["vocab_size"]
    cfg.num_q_heads = m["num_q_heads"]
    cfg.num_kv_heads = m["num_kv_heads"]
    cfg.head_dim = m["head_dim"]
    cfg.activation = m.get("activation", "swiglu")
    cfg.has_qk_norm = m.get("has_qk_norm", False)
    cfg.has_bias = m.get("has_bias", False)
    cfg.rope_theta = m.get("rope_theta", 1000000.0)
    cfg.rms_norm_eps = m.get("rms_norm_eps", 1e-6)
    cfg.max_position_embeddings = m.get("max_position_embeddings", 40960)

    # MoE section (only for arch == "moe")
    if cfg.arch == "moe":
        moe = raw["moe"]
        cfg.num_experts = moe["num_experts"]
        cfg.num_experts_per_tok = moe["num_experts_per_tok"]
        cfg.moe_intermediate_size = moe["moe_intermediate_size"]
        cfg.shared_expert_intermediate_size = moe.get(
            "shared_expert_intermediate_size", 0)

    # Attention section (optional; absent means no sliding window)
    att = raw.get("attention", {})
    cfg.sliding_window = att.get("sliding_window", 0)
    cfg.sliding_window_pattern = att.get("sliding_window_pattern", "uniform")

    # Build section (optional; defaults match the dense build scripts, which
    # round-trip byte-identically on -O2 / no-rdc / no extra defines)
    b = raw.get("build", {})
    cfg.opt_level = b.get("opt_level", "O2")
    cfg.rdc = b.get("rdc", False)
    cfg.extra_defines = list(b.get("extra_defines", []))
    cfg.headers = b.get("headers", "mirage")
    if cfg.headers not in ("mirage", "fleet"):
        raise ValueError(
            f"build.headers must be 'mirage' or 'fleet', got {cfg.headers!r}")
    cfg.header_dir = b.get(
        "header_dir",
        "/home/claudeuser/fleet-chiplet-megakernel"
        if cfg.headers == "fleet" else "/home/claudeuser/mirage")
    cfg.w13_prefetch_toggle = b.get("w13_prefetch_toggle", False)

    # Measured section (optional). See the YAML comments for what each
    # alternative cost -- those numbers are the reason these are not derived.
    meas = raw.get("measured", {})
    cfg.oproj_opw = meas.get("oproj_output_per_wg", 0)  # 0 = auto-derive
    cfg.w13_output_per_wg = meas.get("w13_output_per_wg", 128)
    cfg.w2_output_per_wg = meas.get("w2_output_per_wg", 64)
    cfg.moe_total_tiles_per_xcd = meas.get("moe_total_tiles_per_xcd", 0)

    # GPU section
    g = raw["gpu"]
    cfg.target = g["target"]
    cfg.num_xcds = g["num_xcds"]
    cfg.workers_per_xcd = g["workers_per_xcd"]
    cfg.page_size = g["page_size"]
    cfg.max_seq_len = g["max_seq_len"]
    cfg.num_kv_chunks = g["num_kv_chunks"]
    cfg.lds_bytes = g.get("lds_bytes", cfg.lds_bytes)

    # Quantization section
    q = raw["quantization"]
    cfg.weight_format = q["weight_format"]
    cfg.output_per_wg = q["output_per_wg"]
    # oproj_opw is NOT read here: it is a measured value, not a quantization
    # property. It lives in the `measured:` section (0 there = auto-derive).
    cfg.gateup_opw = q.get("gateup_opw", 128)

    # --- Derived constants ---
    cfg.name_clean = cfg.name.replace("-", "_")
    # Title: first part, title-cased (e.g. "qwen3-8b" -> "Qwen3")
    cfg.name_title = cfg.name.split("-")[0].title()

    # Pad hidden/intermediate to MFMA alignment (128) for non-power-of-2 dims
    # For most models (4096, 12288) this is a no-op; for GPT-OSS (2880) it pads to 2944
    cfg.padded_hidden_size = _pad_up(cfg.hidden_size, cfg.mfma_align)
    cfg.padded_intermediate_size = _pad_up(cfg.intermediate_size, cfg.mfma_align)

    # Pad vocab to be divisible by output_per_wg * num_xcds (for even LM head split)
    lm_align = cfg.output_per_wg * cfg.num_xcds  # typically 512
    cfg.padded_vocab_size = _pad_up(cfg.vocab_size, lm_align)
    cfg.q_per_kv = cfg.num_q_heads // cfg.num_kv_heads
    cfg.total_workers = cfg.num_xcds * cfg.workers_per_xcd

    cfg.qkv_output_size = (cfg.num_q_heads * cfg.head_dim
                           + 2 * cfg.num_kv_heads * cfg.head_dim)

    # Use padded sizes for GEMM dimension calculations
    ph = cfg.padded_hidden_size
    pi = cfg.padded_intermediate_size

    if cfg.arch == "dense":
        if cfg.activation == "swiglu":
            cfg.gateup_output_size = 2 * pi
        else:
            cfg.gateup_output_size = pi

    cfg.oproj_reduction = cfg.num_q_heads * cfg.head_dim

    # Workgroup counts (use padded dimensions for output splits)
    cfg.qkv_n_wgs = cfg.qkv_output_size // cfg.output_per_wg
    cfg.qkv_n_wgs_per_xcd = cfg.qkv_n_wgs // cfg.num_xcds

    # Auto-compute O-proj OPW: find largest OPW (64, 32, 16) that evenly splits
    if cfg.oproj_opw == 0:
        for try_opw in [64, 32, 16]:
            n = ph // try_opw
            if ph % try_opw == 0 and n % cfg.num_xcds == 0:
                cfg.oproj_opw = try_opw
                break
        assert cfg.oproj_opw > 0, \
            f"Cannot find OPW for O-proj with padded_hidden={ph}"

    cfg.oproj_n_wgs = ph // cfg.oproj_opw
    cfg.oproj_n_wgs_per_xcd = cfg.oproj_n_wgs // cfg.num_xcds

    if cfg.arch == "dense":
        cfg.gateup_n_wgs = cfg.gateup_output_size // cfg.gateup_opw
        cfg.gateup_n_wgs_per_xcd = cfg.gateup_n_wgs // cfg.num_xcds
        cfg.down_n_wgs = ph // cfg.output_per_wg
        cfg.down_n_wgs_per_xcd = cfg.down_n_wgs // cfg.num_xcds

    cfg.lm_n_wgs = cfg.padded_vocab_size // cfg.output_per_wg
    cfg.lm_n_wgs_per_xcd = cfg.lm_n_wgs // cfg.num_xcds

    # MXFP4 byte sizes per workgroup: OPW * (K/2 + K/32)
    # K is the reduction dimension (padded)
    cfg.qkv_wg_bytes = cfg.output_per_wg * (ph // 2 + ph // 32)
    cfg.oproj_wg_bytes = cfg.oproj_opw * (cfg.oproj_reduction // 2
                                         + cfg.oproj_reduction // 32)
    if cfg.arch == "dense":
        cfg.gateup_wg_bytes = cfg.gateup_opw * (ph // 2 + ph // 32)
        cfg.down_wg_bytes = cfg.output_per_wg * (pi // 2 + pi // 32)
    cfg.lm_wg_bytes = cfg.output_per_wg * (ph // 2 + ph // 32)

    # MoE-specific derived constants
    if cfg.arch == "moe":
        cfg.padded_moe_intermediate_size = _pad_up(
            cfg.moe_intermediate_size, cfg.mfma_align)
        pmi = cfg.padded_moe_intermediate_size
        cfg.w13_output_size = 2 * pmi  # SwiGLU: gate + up interleaved
        cfg.w13_n_wgs = cfg.w13_output_size // cfg.w13_output_per_wg
        cfg.w13_wg_bytes = cfg.w13_output_per_wg * (ph // 2 + ph // 32)
        cfg.w2_n_wgs = ph // cfg.w2_output_per_wg
        cfg.w2_wg_bytes = cfg.w2_output_per_wg * (pmi // 2 + pmi // 32)
        assert ph % 128 == 0, (
            f"padded_hidden_size ({ph}) must be a multiple of the 16x16x128 "
            f"MFMA K tile for the MoE reduction")
        cfg.moe_k_mfma_iters = ph // 128
        # Router: hidden -> num_experts (small GEMM, one WG does all experts)
        cfg.router_n_wgs = _pad_up(cfg.num_experts, cfg.output_per_wg) // cfg.output_per_wg
        cfg.router_wg_bytes = cfg.output_per_wg * (ph // 2 + ph // 32)

        # Fused MoE tiles per XCD. This one IS derivable, so the measured value
        # gets a real cross-check rather than being taken on faith: W13 tiles
        # are padded up to a whole multiple of TOTAL_WORKERS, W2 tiles are not.
        # The formula reproduces all three historical values -- 53 (W13=128,
        # W2=64), 83 (64/64), 42 (128/128).
        _w13_tiles = cfg.num_experts_per_tok * (cfg.w13_output_size
                                                // cfg.w13_output_per_wg)
        _w13_padded = _pad_up(_w13_tiles, cfg.total_workers)
        _w2_tiles = cfg.num_experts_per_tok * (ph // cfg.w2_output_per_wg)
        _derived_tiles = _pad_up(_w13_padded + _w2_tiles,
                                 cfg.num_xcds) // cfg.num_xcds
        if cfg.moe_total_tiles_per_xcd == 0:
            cfg.moe_total_tiles_per_xcd = _derived_tiles
        else:
            assert cfg.moe_total_tiles_per_xcd == _derived_tiles, (
                f"measured.moe_total_tiles_per_xcd="
                f"{cfg.moe_total_tiles_per_xcd} disagrees with the derivation "
                f"({_derived_tiles}) for W13_OPW={cfg.w13_output_per_wg}, "
                f"W2_OPW={cfg.w2_output_per_wg}. One of the two is stale; "
                f"changing an OPW changes this number.")

    # Attention strides
    cfg.kv_cache_stride = cfg.num_kv_heads * cfg.head_dim
    cfg.q_workspace_stride = cfg.q_per_kv * cfg.head_dim

    # Counter block. The header allocates more cache lines than COUNTER_REGIONS
    # names -- the tail is deliberate headroom, so this is a fits-check, not an
    # equality. Overflowing it would scribble on the NEXT layer's barriers.
    cfg.counters_per_layer = parse_counters_per_layer()
    assert counter_lines_used() * 16 <= cfg.counters_per_layer, (
        f"COUNTER_REGIONS needs {counter_lines_used()} cache lines but "
        f"device_functions.cuh allocates {cfg.counters_per_layer // 16}. "
        f"Raise COUNTERS_PER_LAYER in the header first -- overflowing it "
        f"corrupts the next layer's counters.")
    check_counter_region_collisions(cfg)

    # Pointer table layout depends on arch
    if cfg.arch == "dense":
        cfg.ptrs_in = 16 + (2 if cfg.has_qk_norm else 0)
        cfg.ptrs_out = 9
    elif cfg.arch == "moe":
        # MoE input ptrs: RESIDUAL, NORM_W1, NORM_SCRATCH1, QKV_WEIGHT, QKV_BIAS,
        #   LSE_ACC, O_ACC_F32, OPROJ_WEIGHT, OPROJ_BIAS,
        #   NORM_W2, NORM_SCRATCH2, ROUTER_WEIGHT, ROUTER_BIAS,
        #   W13_WEIGHT, W13_BIAS, W2_WEIGHT, W2_BIAS,
        #   MOE_BARRIER, COUNTER_BUF [+ Q_NORM_WEIGHT, K_NORM_WEIGHT]
        cfg.ptrs_in = 19 + (2 if cfg.has_qk_norm else 0)
        # MoE output ptrs: QKV_OUTPUT, K_CACHE, V_CACHE, Q_WORKSPACE, ATTN_OUT,
        #   OPROJ_OUT, ROUTING_INDICES, TOPK_WEIGHT, ACTIVE_EXPERT_IDS,
        #   SWIGLU_OUT, MOE_WORKSPACE_F32, LAYER_OUTPUT
        cfg.ptrs_out = 12
    cfg.ptrs_per_layer = cfg.ptrs_in + cfg.ptrs_out

    # --- Validation ---
    assert cfg.arch in ("dense", "moe"), f"Unknown arch '{cfg.arch}'"
    assert cfg.weight_format == "mxfp4", f"Only 'mxfp4' supported, got '{cfg.weight_format}'"
    assert cfg.qkv_output_size % cfg.output_per_wg == 0, \
        f"QKV output size {cfg.qkv_output_size} not divisible by OPW {cfg.output_per_wg}"
    assert cfg.qkv_n_wgs % cfg.num_xcds == 0, \
        f"QKV WGs {cfg.qkv_n_wgs} not divisible by num_xcds {cfg.num_xcds}"
    assert ph % cfg.output_per_wg == 0
    assert cfg.oproj_n_wgs % cfg.num_xcds == 0
    if cfg.arch == "dense":
        assert cfg.gateup_output_size % cfg.gateup_opw == 0
        assert cfg.gateup_n_wgs % cfg.num_xcds == 0
        assert cfg.down_n_wgs % cfg.num_xcds == 0
    assert cfg.padded_vocab_size % cfg.output_per_wg == 0
    assert cfg.lm_n_wgs % cfg.num_xcds == 0
    assert ph % 32 == 0, f"padded_hidden_size ({ph}) must be multiple of 32 for MXFP4"
    assert cfg.num_q_heads % cfg.num_kv_heads == 0
    if cfg.arch == "moe":
        assert cfg.num_experts > 0
        assert cfg.num_experts_per_tok > 0
        assert cfg.moe_intermediate_size > 0

        # Structural constraints of the mirage fused kernel. These are NOT
        # tunable -- each one names the code that hardcodes it, so a future
        # model config fails at load time instead of producing garbage.
        assert cfg.num_kv_heads == cfg.num_xcds, (
            f"The fused MoE path requires num_kv_heads ({cfg.num_kv_heads}) == "
            f"num_xcds ({cfg.num_xcds}): mirage hardcodes kv_head_idx = xcd_id "
            f"at gang_full_layer_fused_mi300.cuh:260,297.")
        assert cfg.head_dim == 64, (
            f"The fused MoE path requires head_dim == 64, got {cfg.head_dim}: "
            f"paged_attention_minimal_decode_hd64 static_asserts it.")
        assert cfg.w13_output_size % cfg.w13_output_per_wg == 0
        assert ph % cfg.w2_output_per_wg == 0

    assert cfg.sliding_window_pattern in ("uniform", "alternating"), (
        f"sliding_window_pattern '{cfg.sliding_window_pattern}' is not "
        f"emittable; only 'uniform' and 'alternating' have emitter arms.")

    # Print summary
    print(f"Loaded config: {cfg.name} ({cfg.arch})")
    print(f"  Layers={cfg.num_layers}, Hidden={cfg.hidden_size}"
          f"{f' (padded {ph})' if ph != cfg.hidden_size else ''}, "
          f"Intermediate={cfg.intermediate_size}"
          f"{f' (padded {pi})' if pi != cfg.intermediate_size else ''}")
    print(f"  Heads: {cfg.num_q_heads}Q / {cfg.num_kv_heads}KV, head_dim={cfg.head_dim}")
    print(f"  QKV output={cfg.qkv_output_size}"
          f"{f', GateUp output={cfg.gateup_output_size}' if cfg.arch == 'dense' else ''}")
    print(f"  Padded vocab={cfg.padded_vocab_size}, LM WGs/XCD={cfg.lm_n_wgs_per_xcd}")
    print(f"  PTRS_IN={cfg.ptrs_in}, PTRS_OUT={cfg.ptrs_out}, "
          f"PTRS_PER_LAYER={cfg.ptrs_per_layer}")
    if cfg.has_qk_norm:
        print(f"  QK normalization: enabled")
    if cfg.arch == "moe":
        print(f"  MoE: {cfg.num_experts} experts, top-{cfg.num_experts_per_tok}, "
              f"expert_intermediate={cfg.moe_intermediate_size}"
              f"{f' (padded {cfg.padded_moe_intermediate_size})' if cfg.padded_moe_intermediate_size != cfg.moe_intermediate_size else ''}")
        print(f"  W13: {cfg.w13_n_wgs} WGs (OPW={cfg.w13_output_per_wg}), "
              f"W2: {cfg.w2_n_wgs} WGs (OPW={cfg.w2_output_per_wg})")

    return cfg


# ============================================================================
# generate_build
# ============================================================================

def emit_build_flag(enabled: bool, flag: str) -> str:
    """One optional hipcc flag line, or nothing at all.

    Returns the trailing line-continuation and newline with the flag, so a
    disabled flag leaves no blank line behind. The defaults in ModelConfig are
    chosen so every dense config renders these to "" and its build script stays
    byte-identical -- the GPT-OSS build opts in through the YAML `build:`
    section instead of the flags being turned on globally.
    """
    return f"    {flag} \\\n" if enabled else ""


def emit_extra_defines(cfg: ModelConfig) -> str:
    """The YAML build.extra_defines list as -D lines.

    Empty by default. These are correctness- or performance-load-bearing:
    MPK_W13_LDS_PREFETCH is one of the three flags commit f2354a7 records as
    affecting the 2.520 ms/tok result, and FLEET_MK_ENABLE_LEGACY is what the
    dense kernel needs for gemm_mxfp4 and rope_kv_update to be visible.
    """
    return "".join(f"    -D{d} \\\n" for d in cfg.extra_defines)


def emit_oproj_kmajor_shuffle(cfg: ModelConfig) -> str:
    """The host half of MPK_OPROJ_KMAJOR, derived from the same YAML entry.

    MPK_OPROJ_KMAJOR is a TWO-SIDED contract: the flag rewrites the kernel's
    LDS address arithmetic, and the host must repack the O-proj tile to match.
    Get one side without the other and it is not a build error and not a
    crash -- every lane still reads a valid byte of the tile, just the wrong
    one -- so the model emits plausible wrong logits. Fleet's own demo carries
    the same warning.

    Which is why this is DERIVED from build.extra_defines rather than being a
    second knob someone has to remember to flip in step. The flag being in the
    list is what puts the shuffle in the driver; there is no way to express
    the broken half-configuration.

    The kernel-side static_assert requires a 16-row tile, and so does the
    shuffle (lane_id spans exactly 16 rows x 4 K-quarters, which is what makes
    lane_id * 16 the fragment address). Assert it here too so the mismatch is
    a generation-time error naming oproj_output_per_wg, rather than a
    compile-time error inside a header.
    """
    if "MPK_OPROJ_KMAJOR" not in cfg.extra_defines:
        return ""
    if cfg.oproj_opw != 16:
        raise ValueError(
            f"MPK_OPROJ_KMAJOR needs oproj_output_per_wg=16, got "
            f"{cfg.oproj_opw}; the kernel-side static_assert says the same")
    return (
        "        # MPK_OPROJ_KMAJOR is on: repack [row, k128, quarter, 16B] ->\n"
        "        # [k128, quarter, row, 16B] so lane L's fragment sits at byte\n"
        "        # L*16 of its K128 block and the wave reads 64 consecutive\n"
        "        # 16-byte chunks instead of 16-way-conflicting on the 2048-byte\n"
        "        # row stride. A permutation, not a requantization -- bit-exact.\n"
        "        o_packed = shuffle_oproj_workgroups_kmajor(o_packed, OPROJ_OPW)\n")


def emit_oproj_kmajor_import(cfg: ModelConfig) -> str:
    """Import the shuffle only where it is used, so the dense drivers are
    byte-identical to what they were before this flag existed."""
    if "MPK_OPROJ_KMAJOR" not in cfg.extra_defines:
        return ""
    return ",\n    shuffle_oproj_workgroups_kmajor"


def emit_lm_head_kmajor_import(cfg: ModelConfig) -> str:
    if "MPK_LM_HEAD_KMAJOR" not in cfg.extra_defines:
        return ""
    return ",\n    shuffle_lm_head_record_kmajor"


def emit_lm_head_kmajor_shuffle(cfg: ModelConfig) -> str:
    if "MPK_LM_HEAD_KMAJOR" not in cfg.extra_defines:
        return ""
    return (
        "    lm_head_packed = shuffle_lm_head_record_kmajor(\n"
        "        lm_head_packed, output_per_wg=OUTPUT_PER_WG)\n")


def emit_w13_kmajor_shuffle(cfg: ModelConfig) -> str:
    """The host half of MPK_W13_KMAJOR_RECYCLE, derived from the same YAML
    entry -- exactly the arrangement emit_oproj_kmajor_shuffle exists for, and
    for the same reason: a one-sided setting is silently wrong logits, not a
    build error.

    The kernel-side static_asserts are `W13_OUTPUT_PER_WG == 128` and a
    K128-aligned reduction. Assert both here so a mismatch is a
    generation-time error naming the YAML key, not a compile error inside a
    2000-line header.

    Upstream additionally #errors this flag against MPK_W13_T1_SPLIT_LDS_STAGE
    (the recycle path REPLACES the split-stage path). That would be a build
    failure rather than silent corruption, but it costs a two-minute compile to
    discover, so it is checked here too.
    """
    if "MPK_W13_KMAJOR_RECYCLE" not in cfg.extra_defines:
        return ""
    if cfg.w13_output_per_wg != 128:
        raise ValueError(
            f"MPK_W13_KMAJOR_RECYCLE needs w13_output_per_wg=128, got "
            f"{cfg.w13_output_per_wg}; the kernel-side static_assert "
            f"`W13_OUTPUT_PER_WG == 128 && NUM_WARPS == 4` says the same")
    # The reduction is the PADDED hidden size -- the driver's HIDDEN&#95;SIZE,
    # which is what the emitted call passes. cfg.hidden_size is the unpadded
    # 2880 and is NOT K128-aligned, so checking that one would reject a
    # configuration the kernel accepts.
    if cfg.padded_hidden_size % 128 != 0:
        raise ValueError(
            f"MPK_W13_KMAJOR_RECYCLE needs a K128-aligned reduction, but "
            f"padded_hidden_size={cfg.padded_hidden_size} is not a multiple "
            f"of 128")
    # The kernel hard-codes the fragment-wait schedule for 23 K128 iterations
    # (`static_assert(W13_MFMA_ITERS == 23, "re-audit canonical fragment waits
    # when W13 K changes")`). A different K builds, but the waits would be
    # wrong -- so fail here, where the error can name the YAML key.
    if cfg.padded_hidden_size // 128 != 23:
        raise ValueError(
            f"MPK_W13_KMAJOR_RECYCLE's fragment-wait schedule is written for "
            f"W13_MFMA_ITERS=23, but padded_hidden_size="
            f"{cfg.padded_hidden_size} gives {cfg.padded_hidden_size // 128}; "
            f"the kernel-side static_assert says re-audit the waits")
    if "MPK_W13_T1_SPLIT_LDS_STAGE" in cfg.extra_defines:
        raise ValueError(
            "MPK_W13_KMAJOR_RECYCLE replaces the W13 tile-1 split-stage path; "
            "remove MPK_W13_T1_SPLIT_LDS_STAGE from extra_defines (the header "
            "#errors on the pair)")
    return (
        "        # MPK_W13_KMAJOR_RECYCLE is on: within each 16-row MFMA tile,\n"
        "        # repack [row, k128, quarter, 16B] -> [k128, quarter, row, 16B]\n"
        "        # so the recycle schedule retires lane-contiguous fragments.\n"
        "        # A permutation, not a requantization -- bit-exact. The kernel\n"
        "        # half is the -D flag; both come from one YAML entry because a\n"
        "        # one-sided setting is silently wrong logits, not a build error.\n"
        "        # K is passed explicitly: under split scales the wg_bytes\n"
        "        # equation fleet solves has no solution (tools/check_w13_kmajor.py).\n"
        "        gu_packed = shuffle_w13_workgroups_kmajor(\n"
        "            gu_packed, W13_OPW, reduction=HIDDEN_SIZE)\n")


def emit_w13_kmajor_import(cfg: ModelConfig) -> str:
    """Import the shuffle only where it is used, so every other driver stays
    byte-identical to what it was before this flag existed."""
    if "MPK_W13_KMAJOR_RECYCLE" not in cfg.extra_defines:
        return ""
    return ",\n    shuffle_w13_workgroups_kmajor"


def emit_header_root(cfg: ModelConfig) -> str:
    """The shell variable naming the megakernel header tree, plus its rationale.

    Two arms, because the two trees are not interchangeable and the difference
    is not something a compiler can see. Keeping BOTH emittable is deliberate:
    the dense configs still build against mirage and round-trip byte-identically
    against their on-disk scripts, so this switch cannot silently rewrite them.
    """
    if cfg.headers == "fleet":
        return f'''\
# Fleet MK compiles against fleet's header tree ONLY -- ROCm/fleet-chiplet-megakernel,
# branch amd_mi355_gpt_oss120b. Nothing here may point at mirage
# (github.com/sangeeta0201/mirage): that is a personal fork, and fleet_mk carrying a
# build-time dependency on it is exactly what this flip removes. Fleet is public
# and is the single source of truth.
#
# Fleet's tree is used OUT OF THE BOX -- it is on the include path, never copied
# and never edited. A vendored copy would be a fork that silently rots; -I at
# fleet's own checkout means future fleet work lands automatically.
#
# The old -I"$HIP_COMPAT" line is deleted rather than repointed: that directory
# does not exist in either tree, so it was always a dead path.
FLEET_DIR="${{FLEET_DIR:-{cfg.header_dir}}}"
'''
    return f'MIRAGE_DIR="${{MIRAGE_DIR:-{cfg.header_dir}}}"\n'


def emit_include_setup(cfg: ModelConfig) -> str:
    """MPK_INCLUDE, and on the mirage arm the (dead) HIP_COMPAT alongside it."""
    var = "FLEET_DIR" if cfg.headers == "fleet" else "MIRAGE_DIR"
    out = f'MPK_INCLUDE="${var}/include/mirage/persistent_kernel"\n'
    if cfg.headers != "fleet":
        out += f'HIP_COMPAT="${var}/include/mirage/hip_compat"\n'
    return out


def emit_fleet_define_note(cfg: ModelConfig) -> str:
    """Why the fleet arm hand-supplies three defines fleet's Python normally does.

    Emitted only on the fleet arm, and load-bearing as documentation: two of the
    three have no in-header default, so omitting either is a silent miscompile
    rather than a build error. The note names that fact next to the flags.
    """
    if cfg.headers != "fleet":
        return ""
    return '''
# Three defines fleet's Python driver (persistent_kernel.py) normally supplies and
# fleet_mk must now supply itself. Verified against fleet @ f3c40ed: NEITHER of the
# first two has an in-header default -- the only #define of
# MPK_MAX_NUM_BATCHED_REQUESTS in the tree is commented out
# (persistent_kernel.cuh:556), and MPK_PREFETCH_NEXT_QKV is a bare #ifdef at
# three sites. Omitting either is a SILENT miscompile, not a build error:
#
#   FLEET_MK_FLEET_HEADERS         selects fleet's calling convention in the
#                               generated kernel -- the trailing runtime arg is
#                               task_layer_idx in fleet, sliding_window_override
#                               in mirage, and neither header diagnoses receiving
#                               the other's. Also selects the two-instantiation
#                               form, since fleet takes the window as a template
#                               parameter only.
#   MPK_MAX_NUM_BATCHED_REQUESTS=1   fleet_mk is bs=1 decode.
#   MPK_PREFETCH_NEXT_QKV       default ON in fleet; the reason for the migration.
#
# MPK_NO_LAYER_BARRIER is deliberately NOT passed: fleet's cross-XCD layer
# barrier is the one that survives, and fleet_mk's own has been removed.
'''


def emit_w13_prefetch_toggle(cfg: ModelConfig) -> str:
    """The FLEET_MK_NO_W13_PREFETCH escape hatch, as a shell if/else.

    A plain entry in build.extra_defines cannot be turned OFF without editing
    the generated script, and an A/B that also has to edit the script is an A/B
    that drifts. This is the only supported way to get two builds differing in
    exactly the prefetch.
    """
    if not cfg.w13_prefetch_toggle:
        return ""
    return '''
# W13 direct-to-LDS weight prefetch is ON by default. Set FLEET_MK_NO_W13_PREFETCH=1
# to build the HBM-direct path instead -- the only supported way to get an A/B
# pair that differs ONLY in the prefetch, which matters because the MoE fault
# under investigation reproduces about one run in three. Comparing a prefetch-ON
# build against a prefetch-OFF build that also differs in K stride or scale
# split cannot attribute anything.
if [ -n "$FLEET_MK_NO_W13_PREFETCH" ]; then
  W13_PREFETCH_FLAG=""
else
  W13_PREFETCH_FLAG="-DMPK_W13_LDS_PREFETCH"
fi
'''


def emit_include_paths(cfg: ModelConfig) -> str:
    """The -I list. The mirage arm carries the dead HIP_COMPAT path it always had."""
    var = "FLEET_DIR" if cfg.headers == "fleet" else "MIRAGE_DIR"
    out = f'    -I"${var}/include" \\\n'
    if cfg.headers != "fleet":
        out += '    -I"$HIP_COMPAT" \\\n'
    return out


def emit_fleet_defines(cfg: ModelConfig) -> str:
    """The three defines fleet's Python driver would otherwise supply.

    Deliberately NOT left to build.extra_defines: they are not free-form tuning,
    they are the fleet calling convention, and two of the three fail silently
    when absent. Deriving them from `headers: fleet` makes them impossible to
    drop while flipping the header tree, and keeps them in lockstep with the
    comment block above that explains what each one does.
    """
    if cfg.headers != "fleet":
        return ""
    return (
        "    -DFLEET_MK_FLEET_HEADERS \\\n"
        "    -DMPK_MAX_NUM_BATCHED_REQUESTS=1 \\\n"
        "    -DMPK_PREFETCH_NEXT_QKV \\\n"
    )


def emit_w13_prefetch_flag(cfg: ModelConfig) -> str:
    """Expand the shell variable the toggle above sets, if there is one."""
    return "    $W13_PREFETCH_FLAG \\\n" if cfg.w13_prefetch_toggle else ""


def generate_build(cfg: ModelConfig) -> str:
    nc = cfg.name_clean
    return f'''\
#!/bin/bash
# Auto-generated by fleet_mk_generate.py
# Fleet MK build script for {cfg.name} on {cfg.target}
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLEET_MK_DIR="$SCRIPT_DIR"
GEN_DIR="${{1:-$SCRIPT_DIR/generated}}"

ROCM_PATH="${{ROCM_PATH:-/opt/rocm}}"
HIPCC="${{ROCM_PATH}}/bin/hipcc"

{emit_header_root(cfg)}\
CK_DIR="${{CK_DIR:-/home/claudeuser/composable_kernel_mirage}}"

{emit_include_setup(cfg)}\
{emit_fleet_define_note(cfg)}\
{emit_w13_prefetch_toggle(cfg)}\

echo "=== Fleet MK {cfg.name} Build ==="
echo "  Source: $GEN_DIR/{nc}_launch.hip"
echo "  Target: {cfg.target}"
echo "  Output: $GEN_DIR/{nc}.so"

$HIPCC -x hip "$GEN_DIR/{nc}_launch.hip" \\
    -{cfg.opt_level} \\
    -std=c++17 \\
    -fPIC \\
{emit_build_flag(cfg.rdc, "-fgpu-rdc")}\
    --offload-arch={cfg.target} \\
    -I"$FLEET_MK_DIR/kernels" \\
    -I"$FLEET_MK_DIR/generated" \\
    -I"$MPK_INCLUDE" \\
{emit_include_paths(cfg)}\
    -I"$ROCM_PATH/include" \\
    -I"$CK_DIR/include" \\
    -D__HIP_PLATFORM_AMD__=1 \\
    -DMIRAGE_AMD_MI300 \\
    -DMIRAGE_BACKEND_USE_ROCM \\
    -DMPK_USE_CK_FMHA \\
    -DFLEET_MK_TARGET_GFX950 \\
    -DMPK_TARGET_CC=95 \\
{emit_extra_defines(cfg)}\
{emit_fleet_defines(cfg)}\
{emit_w13_prefetch_flag(cfg)}\
    -DCK_TILE_FMHA_FWD_FAST_EXP2=1 \\
    ${{FLEET_MK_SUBPHASE_TIMING:+-DFLEET_MK_SUBPHASE_TIMING}} \\
    ${{FLEET_MK_TIMER_PRINT:+-DFLEET_MK_TIMER_PRINT}} \\
    ${{FLEET_MK_EXTRA_DEFINES:+$FLEET_MK_EXTRA_DEFINES}} \\
    -Rpass-analysis=kernel-resource-usage \\
    -shared \\
    -o "$GEN_DIR/{nc}.so"

echo "=== Build complete: $GEN_DIR/{nc}.so ==="
'''


# ============================================================================
# generate_launch
# ============================================================================

def generate_launch(cfg: ModelConfig) -> str:
    """Emit the host-side launch wrapper .hip for `cfg`. Dispatches on arch --
    see generate_kernel for why the two arms are separate functions."""
    if cfg.arch == "dense":
        return generate_launch_dense(cfg)
    if cfg.arch == "moe":
        return generate_launch_fused_moe(cfg)
    raise ValueError(f"Unknown arch '{cfg.arch}'")


def generate_launch_fused_moe(cfg: ModelConfig) -> str:
    """MoE launch wrapper: hipGraph trio, BridgeArgs/decode_bridge_kernel, the
    pipelined graph and the C-side decode_loop -- none of which the dense arm
    has. Built backwards from the on-disk launch wrapper.

    The plan gated this file only on building and still measuring 2.52 ms/tok.
    The verbatim seed reached byte-identity for free, so check_roundtrip.py
    gates it on bytes as well -- see the note there before loosening that.
    """
    return f"""/* Auto-generated by fleet_mk_generate.py
 * Fleet MK: Launch wrapper for {cfg.name}
 *
 * Build: hipcc -x hip {cfg.name_clean}_launch.hip [flags] -shared -o {cfg.name_clean}.so
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "{cfg.name_clean}_kernel.cuh"

// ============================================================================
// Initialization (no-op: counter buffer managed by Python)
// ============================================================================
extern "C" void {cfg.name_clean}_init() {{
    printf("[Fleet MK/Gpt] Initialized (counter buffer managed externally)\\n");
}}

// ============================================================================
// Kernel launch
// ============================================================================
extern "C" void {cfg.name_clean}_launch(
{emit_c_signature('launch')}
{{
    GptConfig config;
    config.num_active_tokens  = num_active_tokens;
    config.attn_scale         = attn_scale;
    config.cos_ptr            = cos_ptr;
    config.sin_ptr            = sin_ptr;
    config.qo_indptr          = qo_indptr;
    config.kv_indptr          = kv_indptr;
    config.kv_indices         = kv_indices;
    config.kv_last_page_len   = kv_last_page_len;
    config.lm_norm_weight     = lm_norm_weight;
    config.lm_norm_scratch    = lm_norm_scratch;
    config.lm_mxfp4_weight    = lm_mxfp4_weight;
    config.lm_bias            = lm_bias;
    config.argmax_output      = argmax_output;
    config.logits_output      = logits_output;
    config.timing_buf         = timing_buf;
    config.embed_weight       = embed_weight;
    config.cur_token_id       = cur_token_id;
    config.cur_token_ptr      = nullptr;

    int *counter_buf = reinterpret_cast<int *>(counter_buf_vp);

    // {cfg.total_workers} threadblocks = {cfg.workers_per_xcd} workers/XCD x {cfg.num_xcds} XCDs
    constexpr int LDS_SIZE = {cfg.lds_bytes};

    dim3 grid({cfg.name_clean}::TOTAL_WORKERS, 1, 1);
    dim3 block(256, 1, 1);

    hipLaunchKernelGGL(
        {cfg.name_clean}_kernel,
        grid, block,
        LDS_SIZE,
        stream,
        config,
        ptr_table,
        counter_buf,
        decode_ctrl);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {{
        fprintf(stderr, "{cfg.name_clean}_launch: kernel launch failed: %s\\n",
                hipGetErrorString(err));
    }}
}}

// ============================================================================
// hipGraph-based launch for reduced overhead
// ============================================================================
static hipGraph_t       s_graph = nullptr;
static hipGraphExec_t   s_graph_exec = nullptr;
static hipGraphNode_t   s_kernel_node = nullptr;
static GptConfig        s_captured_config;
static void           **s_captured_ptr_table = nullptr;
static int             *s_captured_counter_buf = nullptr;
static DecodeControl   *s_captured_decode_ctrl = nullptr;

// Static kernel args array — must outlive graph launches
static void *s_kernel_args[4] = {{nullptr, nullptr, nullptr, nullptr}};

#define HIP_CHECK(call) do {{ \\
    hipError_t _e = (call); \\
    if (_e != hipSuccess) {{ \\
        fprintf(stderr, "hipGraph: %s failed: %s (line %d)\\n", \\
                #call, hipGetErrorString(_e), __LINE__); \\
        return; \\
    }} \\
}} while(0)

extern "C" void {cfg.name_clean}_graph_capture(
{emit_c_signature('graph_capture')}
{{
    // Build config
    s_captured_config.num_active_tokens  = num_active_tokens;
    s_captured_config.attn_scale         = attn_scale;
    s_captured_config.cos_ptr            = cos_ptr;
    s_captured_config.sin_ptr            = sin_ptr;
    s_captured_config.qo_indptr          = qo_indptr;
    s_captured_config.kv_indptr          = kv_indptr;
    s_captured_config.kv_indices         = kv_indices;
    s_captured_config.kv_last_page_len   = kv_last_page_len;
    s_captured_config.lm_norm_weight     = lm_norm_weight;
    s_captured_config.lm_norm_scratch    = lm_norm_scratch;
    s_captured_config.lm_mxfp4_weight    = lm_mxfp4_weight;
    s_captured_config.lm_bias            = lm_bias;
    s_captured_config.argmax_output      = argmax_output;
    s_captured_config.logits_output      = logits_output;
    s_captured_config.timing_buf         = timing_buf;
    s_captured_config.embed_weight       = embed_weight;
    s_captured_config.cur_token_id       = cur_token_id;
    s_captured_config.cur_token_ptr      = nullptr;

    s_captured_ptr_table = ptr_table;
    s_captured_counter_buf = reinterpret_cast<int *>(counter_buf_vp);
    s_captured_decode_ctrl = decode_ctrl;

    // Set up static kernel args pointers
    s_kernel_args[0] = &s_captured_config;
    s_kernel_args[1] = &s_captured_ptr_table;
    s_kernel_args[2] = &s_captured_counter_buf;
    s_kernel_args[3] = &s_captured_decode_ctrl;

    // Build graph manually (not via stream capture) for full control
    HIP_CHECK(hipGraphCreate(&s_graph, 0));

    // -- Memset nodes (counter + workspace; MoE barrier merged into counter) --
    hipGraphNode_t memset_nodes[3];
    int num_memset_nodes = 0;

    hipMemsetParams mset_params[3];
    memset(&mset_params, 0, sizeof(mset_params));

    // Counter buffer (includes MoE barrier if merged)
    mset_params[0].dst = counter_buf_raw;
    mset_params[0].value = 0;
    mset_params[0].elementSize = 1;
    mset_params[0].width = counter_buf_bytes;
    mset_params[0].height = 1;
    HIP_CHECK(hipGraphAddMemsetNode(&memset_nodes[num_memset_nodes++], s_graph, nullptr, 0, &mset_params[0]));

    // Workspace buffer
    mset_params[1].dst = workspace_buf_raw;
    mset_params[1].value = 0;
    mset_params[1].elementSize = 1;
    mset_params[1].width = workspace_buf_bytes;
    mset_params[1].height = 1;
    HIP_CHECK(hipGraphAddMemsetNode(&memset_nodes[num_memset_nodes++], s_graph, nullptr, 0, &mset_params[1]));

    // MoE barrier buffer (optional — may be merged into counter buffer)
    if (moe_barrier_buf_bytes > 0) {{
        mset_params[2].dst = moe_barrier_buf_raw;
        mset_params[2].value = 0;
        mset_params[2].elementSize = 1;
        mset_params[2].width = moe_barrier_buf_bytes;
        mset_params[2].height = 1;
        HIP_CHECK(hipGraphAddMemsetNode(&memset_nodes[num_memset_nodes++], s_graph, nullptr, 0, &mset_params[2]));
    }}

    // -- Kernel node (depends on all memsets) --
    hipKernelNodeParams kparams;
    memset(&kparams, 0, sizeof(kparams));
    kparams.func = (void *){cfg.name_clean}_kernel;
    kparams.gridDim = dim3({cfg.name_clean}::TOTAL_WORKERS, 1, 1);
    kparams.blockDim = dim3(256, 1, 1);
    kparams.sharedMemBytes = {cfg.lds_bytes};
    kparams.kernelParams = s_kernel_args;
    kparams.extra = nullptr;

    HIP_CHECK(hipGraphAddKernelNode(&s_kernel_node, s_graph,
                                     memset_nodes, num_memset_nodes, &kparams));

    // Instantiate
    HIP_CHECK(hipGraphInstantiate(&s_graph_exec, s_graph, nullptr, nullptr, 0));

    printf("[Fleet MK/Gpt] hipGraph created: %d nodes (%d memset + 1 kernel), kernel node=%p\\n",
           num_memset_nodes + 1, num_memset_nodes, s_kernel_node);
}}

extern "C" void {cfg.name_clean}_graph_launch(
    int cur_token_id,
    hipStream_t stream)
{{
    if (!s_graph_exec) {{
        fprintf(stderr, "{cfg.name_clean}_graph_launch: graph not captured!\\n");
        return;
    }}

    // Update cur_token_id in the static config
    s_captured_config.cur_token_id = cur_token_id;

    // Update kernel node params (s_kernel_args[0] already points to s_captured_config)
    hipKernelNodeParams kparams;
    memset(&kparams, 0, sizeof(kparams));
    kparams.func = (void *){cfg.name_clean}_kernel;
    kparams.gridDim = dim3({cfg.name_clean}::TOTAL_WORKERS, 1, 1);
    kparams.blockDim = dim3(256, 1, 1);
    kparams.sharedMemBytes = {cfg.lds_bytes};
    kparams.kernelParams = s_kernel_args;
    kparams.extra = nullptr;

    hipGraphExecKernelNodeSetParams(s_graph_exec, s_kernel_node, &kparams);

    hipGraphLaunch(s_graph_exec, stream);
}}

extern "C" void {cfg.name_clean}_graph_destroy() {{
    if (s_graph_exec) {{
        hipGraphExecDestroy(s_graph_exec);
        s_graph_exec = nullptr;
    }}
    if (s_graph) {{
        hipGraphDestroy(s_graph);
        s_graph = nullptr;
    }}
    s_kernel_node = nullptr;
    printf("[Fleet MK/Gpt] hipGraph destroyed\\n");
}}

#undef HIP_CHECK

// ============================================================================
// Bridge kernel: runs between decode steps to prepare next iteration
// Reads argmax result, writes next token, updates KV metadata, zeros MoE barrier
// ============================================================================
struct BridgeArgs {{
    long long *argmax_output;   // read: last argmax result
    int       *cur_token_ptr;   // write: next token for main kernel
    int       *kv_indptr;       // write: kv_indptr[1] = num_pages
    int       *kv_last_page_len;// write: kv_last_page_len[0] = pos_in_page
    int       *moe_barrier;     // write: zero {16 * cfg.num_experts} ints
    int       *token_output_buf;// write: store each token for host readback
    int       *step_counter;    // read/write: current step index (atomicAdd)
    int        page_size;       // const: {cfg.page_size}
    int        moe_barrier_ints;// const: {16 * cfg.num_experts}
    int        start_pos;       // const: initial cur_pos
}};

__global__ void decode_bridge_kernel(BridgeArgs args) {{
    int tid = threadIdx.x;

    // Thread 0: read argmax, store token, update KV metadata
    if (tid == 0) {{
        int next_token = static_cast<int>(args.argmax_output[0]);
        *args.cur_token_ptr = next_token;

        // Increment step counter and compute cur_pos
        int step = atomicAdd(args.step_counter, 1);
        int cur_pos = args.start_pos + step + 1;

        // Store token for host readback
        args.token_output_buf[step] = next_token;

        // Update KV metadata
        int num_pages = (cur_pos + args.page_size) / args.page_size;
        args.kv_indptr[1] = num_pages;
        args.kv_last_page_len[0] = (cur_pos % args.page_size) + 1;
    }}

    // All threads: zero MoE barrier ({16 * cfg.num_experts} ints / 256 threads = {16 * cfg.num_experts // 256} per thread)
    for (int i = tid; i < args.moe_barrier_ints; i += 256) {{
        args.moe_barrier[i] = 0;
    }}
}}

// ============================================================================
// Pipelined graph: bridge_kernel → main_kernel (no host sync between tokens)
// ============================================================================
static hipGraph_t       s_pipe_graph = nullptr;
static hipGraphExec_t   s_pipe_graph_exec = nullptr;
static BridgeArgs       s_bridge_args;
static void            *s_bridge_kernel_args[1] = {{nullptr}};

// Separate graph for the first step (no bridge, just memsets + kernel)
static hipGraph_t       s_first_graph = nullptr;
static hipGraphExec_t   s_first_graph_exec = nullptr;

#define HIP_CHECK(call) do {{ \\
    hipError_t _e = (call); \\
    if (_e != hipSuccess) {{ \\
        fprintf(stderr, "hipGraph: %s failed: %s (line %d)\\n", \\
                #call, hipGetErrorString(_e), __LINE__); \\
        return; \\
    }} \\
}} while(0)

extern "C" void {cfg.name_clean}_pipe_capture(
{emit_c_signature('pipe_capture')}
{{
    // Build main kernel config (with cur_token_ptr for GPU-side token reading)
    s_captured_config.num_active_tokens  = num_active_tokens;
    s_captured_config.attn_scale         = attn_scale;
    s_captured_config.cos_ptr            = cos_ptr;
    s_captured_config.sin_ptr            = sin_ptr;
    s_captured_config.qo_indptr          = qo_indptr;
    s_captured_config.kv_indptr          = kv_indptr;
    s_captured_config.kv_indices         = kv_indices;
    s_captured_config.kv_last_page_len   = kv_last_page_len;
    s_captured_config.lm_norm_weight     = lm_norm_weight;
    s_captured_config.lm_norm_scratch    = lm_norm_scratch;
    s_captured_config.lm_mxfp4_weight    = lm_mxfp4_weight;
    s_captured_config.lm_bias            = lm_bias;
    s_captured_config.argmax_output      = argmax_output;
    s_captured_config.logits_output      = logits_output;
    s_captured_config.timing_buf         = timing_buf;
    s_captured_config.embed_weight       = embed_weight;
    s_captured_config.cur_token_id       = cur_token_id;
    s_captured_config.cur_token_ptr      = cur_token_ptr;

    s_captured_ptr_table = ptr_table;
    s_captured_counter_buf = reinterpret_cast<int *>(counter_buf_vp);
    s_captured_decode_ctrl = decode_ctrl;

    // Set up bridge args
    s_bridge_args.argmax_output    = reinterpret_cast<long long *>(argmax_output);
    s_bridge_args.cur_token_ptr    = cur_token_ptr;
    s_bridge_args.kv_indptr        = kv_indptr;
    s_bridge_args.kv_last_page_len = kv_last_page_len;
    s_bridge_args.moe_barrier      = reinterpret_cast<int *>(moe_barrier_raw);
    s_bridge_args.token_output_buf = token_output_buf;
    s_bridge_args.step_counter     = step_counter;
    s_bridge_args.page_size        = page_size;
    s_bridge_args.moe_barrier_ints = moe_barrier_ints;
    s_bridge_args.start_pos        = start_pos;

    // Set up static kernel args
    s_kernel_args[0] = &s_captured_config;
    s_kernel_args[1] = &s_captured_ptr_table;
    s_kernel_args[2] = &s_captured_counter_buf;
    s_kernel_args[3] = &s_captured_decode_ctrl;

    s_bridge_kernel_args[0] = &s_bridge_args;

    // ---- First-step graph: memsets + main kernel (no bridge) ----
    HIP_CHECK(hipGraphCreate(&s_first_graph, 0));

    hipGraphNode_t first_memset_nodes[2];
    hipMemsetParams mset_first[2];
    memset(&mset_first, 0, sizeof(mset_first));

    // MoE barrier memset
    mset_first[0].dst = moe_barrier_raw;
    mset_first[0].value = 0;
    mset_first[0].elementSize = 1;
    mset_first[0].width = moe_barrier_ints * sizeof(int);
    mset_first[0].height = 1;
    HIP_CHECK(hipGraphAddMemsetNode(&first_memset_nodes[0], s_first_graph, nullptr, 0, &mset_first[0]));

    // Workspace memset (only needed for first step if kernel doesn't zero it)
    // Actually workspace is zeroed by kernel's ResAdd, skip workspace memset
    int first_memset_count = 1;

    hipKernelNodeParams kparams_first;
    memset(&kparams_first, 0, sizeof(kparams_first));
    kparams_first.func = (void *){cfg.name_clean}_kernel;
    kparams_first.gridDim = dim3({cfg.name_clean}::TOTAL_WORKERS, 1, 1);
    kparams_first.blockDim = dim3(256, 1, 1);
    kparams_first.sharedMemBytes = {cfg.lds_bytes};
    kparams_first.kernelParams = s_kernel_args;
    kparams_first.extra = nullptr;

    hipGraphNode_t first_kernel_node;
    HIP_CHECK(hipGraphAddKernelNode(&first_kernel_node, s_first_graph,
                                     first_memset_nodes, first_memset_count, &kparams_first));
    HIP_CHECK(hipGraphInstantiate(&s_first_graph_exec, s_first_graph, nullptr, nullptr, 0));

    // ---- Pipeline graph: bridge → main kernel ----
    HIP_CHECK(hipGraphCreate(&s_pipe_graph, 0));

    // Bridge kernel node (no dependencies — runs first)
    hipKernelNodeParams bridge_params;
    memset(&bridge_params, 0, sizeof(bridge_params));
    bridge_params.func = (void *)decode_bridge_kernel;
    bridge_params.gridDim = dim3(1, 1, 1);
    bridge_params.blockDim = dim3(256, 1, 1);
    bridge_params.sharedMemBytes = 0;
    bridge_params.kernelParams = s_bridge_kernel_args;
    bridge_params.extra = nullptr;

    hipGraphNode_t bridge_node;
    HIP_CHECK(hipGraphAddKernelNode(&bridge_node, s_pipe_graph, nullptr, 0, &bridge_params));

    // Main kernel node (depends on bridge)
    hipKernelNodeParams kparams_pipe;
    memset(&kparams_pipe, 0, sizeof(kparams_pipe));
    kparams_pipe.func = (void *){cfg.name_clean}_kernel;
    kparams_pipe.gridDim = dim3({cfg.name_clean}::TOTAL_WORKERS, 1, 1);
    kparams_pipe.blockDim = dim3(256, 1, 1);
    kparams_pipe.sharedMemBytes = {cfg.lds_bytes};
    kparams_pipe.kernelParams = s_kernel_args;
    kparams_pipe.extra = nullptr;

    hipGraphNode_t pipe_kernel_node;
    HIP_CHECK(hipGraphAddKernelNode(&pipe_kernel_node, s_pipe_graph,
                                     &bridge_node, 1, &kparams_pipe));
    HIP_CHECK(hipGraphInstantiate(&s_pipe_graph_exec, s_pipe_graph, nullptr, nullptr, 0));

    printf("[Fleet MK/Gpt] Pipelined graph captured: first_graph(1 memset + kernel), pipe_graph(bridge + kernel)\\n");
}}

extern "C" void {cfg.name_clean}_pipe_launch_first(hipStream_t stream) {{
    if (!s_first_graph_exec) {{
        fprintf(stderr, "{cfg.name_clean}_pipe_launch_first: graph not captured!\\n");
        return;
    }}
    hipGraphLaunch(s_first_graph_exec, stream);
}}

extern "C" void {cfg.name_clean}_pipe_launch_step(hipStream_t stream) {{
    if (!s_pipe_graph_exec) {{
        fprintf(stderr, "{cfg.name_clean}_pipe_launch_step: graph not captured!\\n");
        return;
    }}
    hipGraphLaunch(s_pipe_graph_exec, stream);
}}

extern "C" void {cfg.name_clean}_pipe_launch_all(hipStream_t stream, int total_steps) {{
    if (!s_first_graph_exec || !s_pipe_graph_exec) {{
        fprintf(stderr, "{cfg.name_clean}_pipe_launch_all: graph not captured!\\n");
        return;
    }}
    hipGraphLaunch(s_first_graph_exec, stream);
    for (int i = 1; i < total_steps; i++) {{
        hipGraphLaunch(s_pipe_graph_exec, stream);
    }}
}}

extern "C" void {cfg.name_clean}_pipe_destroy() {{
    if (s_pipe_graph_exec) {{ hipGraphExecDestroy(s_pipe_graph_exec); s_pipe_graph_exec = nullptr; }}
    if (s_pipe_graph)      {{ hipGraphDestroy(s_pipe_graph); s_pipe_graph = nullptr; }}
    if (s_first_graph_exec){{ hipGraphExecDestroy(s_first_graph_exec); s_first_graph_exec = nullptr; }}
    if (s_first_graph)     {{ hipGraphDestroy(s_first_graph); s_first_graph = nullptr; }}
    printf("[Fleet MK/Gpt] Pipelined graph destroyed\\n");
}}

#undef HIP_CHECK

// ============================================================================
// C-side decode loop: eliminates Python per-token overhead
// Launches the kernel repeatedly from C, reading argmax between iterations.
// ============================================================================
extern "C" void {cfg.name_clean}_decode_loop(
{emit_c_signature('decode_loop')}
{{
    GptConfig config;
    config.num_active_tokens  = num_active_tokens;
    config.attn_scale         = attn_scale;
    config.cos_ptr            = cos_ptr;
    config.sin_ptr            = sin_ptr;
    config.qo_indptr          = qo_indptr;
    config.kv_indptr          = kv_indptr;
    config.kv_indices         = kv_indices;
    config.kv_last_page_len   = kv_last_page_len;
    config.lm_norm_weight     = lm_norm_weight;
    config.lm_norm_scratch    = lm_norm_scratch;
    config.lm_mxfp4_weight    = lm_mxfp4_weight;
    config.lm_bias            = lm_bias;
    config.argmax_output      = argmax_output;
    config.logits_output      = logits_output;
    config.timing_buf         = timing_buf;
    config.embed_weight       = embed_weight;
    config.cur_token_id       = first_token_id;
    config.cur_token_ptr      = nullptr;

    int *counter_buf = reinterpret_cast<int *>(counter_buf_vp);

    constexpr int LDS_SIZE = {cfg.lds_bytes};
    dim3 grid({cfg.name_clean}::TOTAL_WORKERS, 1, 1);
    dim3 block(256, 1, 1);

    long long argmax_host;
    int cur_pos = start_pos;

    for (int step = 0; step < total_steps; step++) {{
        // Update KV metadata on GPU (async H2D before kernel)
        int num_pages = (cur_pos + page_size) / page_size;
        int last_page_len = (cur_pos % page_size) + 1;
        hipMemcpyAsync(&kv_indptr[1], &num_pages, sizeof(int),
                        hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(&kv_last_page_len[0], &last_page_len, sizeof(int),
                        hipMemcpyHostToDevice, stream);

        // Launch kernel (queued after memcpys on same stream)
        hipLaunchKernelGGL(
            {cfg.name_clean}_kernel,
            grid, block,
            LDS_SIZE,
            stream,
            config,
            ptr_table,
            counter_buf,
            decode_ctrl);

        // Wait for kernel and read argmax from GPU
        hipMemcpyAsync(&argmax_host, argmax_output, sizeof(long long),
                        hipMemcpyDeviceToHost, stream);
        hipStreamSynchronize(stream);

        int next_token = static_cast<int>(argmax_host);
        token_output_buf[step] = next_token;

        // Update for next iteration
        config.cur_token_id = next_token;
        cur_pos++;
    }}
}}

// ============================================================================
// Finalization (no-op)
// ============================================================================
extern "C" void {cfg.name_clean}_finalize() {{
    printf("[Fleet MK/Gpt] Finalized\\n");
}}
"""


def generate_launch_dense(cfg: ModelConfig) -> str:
    """Dense launch wrapper. Round-trips byte-identically against
    generated/qwen3_8b_launch.hip and generated/llama3_8b_launch.hip."""
    nc = cfg.name_clean
    nt = cfg.name_title
    ns = nc  # namespace name (same as name_clean)

    return f'''\
/* Auto-generated by fleet_mk_generate.py
 * Fleet MK: Launch wrapper for {cfg.name}
 *
 * Build: hipcc -x hip {nc}_launch.hip [flags] -shared -o {nc}.so
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "{nc}_kernel.cuh"

// ============================================================================
// Initialization (no-op: counter buffer managed by Python)
// ============================================================================
extern "C" void {nc}_init() {{
    printf("[Fleet MK/{nt}] Initialized (counter buffer managed externally)\\n");
}}

// ============================================================================
// Kernel launch
// ============================================================================
extern "C" void {nc}_launch(
    int num_active_tokens,
    float attn_scale,
    void *cos_ptr,
    void *sin_ptr,
    int *qo_indptr,
    int *kv_indptr,
    int *kv_indices,
    int *kv_last_page_len,
    void **ptr_table,
    void **counter_buf_vp,
    void *lm_norm_weight,
    void *lm_norm_scratch,
    void *lm_mxfp4_weight,
    void *lm_bias,
    void *argmax_output,
    unsigned long long *timing_buf,
    hipStream_t stream)
{{
    {nt}Config config;
    config.num_active_tokens  = num_active_tokens;
    config.attn_scale         = attn_scale;
    config.cos_ptr            = cos_ptr;
    config.sin_ptr            = sin_ptr;
    config.qo_indptr          = qo_indptr;
    config.kv_indptr          = kv_indptr;
    config.kv_indices         = kv_indices;
    config.kv_last_page_len   = kv_last_page_len;
    config.lm_norm_weight     = lm_norm_weight;
    config.lm_norm_scratch    = lm_norm_scratch;
    config.lm_mxfp4_weight    = lm_mxfp4_weight;
    config.lm_bias            = lm_bias;
    config.argmax_output      = argmax_output;
    config.timing_buf         = timing_buf;

    int *counter_buf = reinterpret_cast<int *>(counter_buf_vp);

    // {cfg.total_workers} threadblocks = {cfg.workers_per_xcd} workers/XCD x {cfg.num_xcds} XCDs
    constexpr int LDS_SIZE = {cfg.lds_bytes};

    dim3 grid({ns}::TOTAL_WORKERS, 1, 1);
    dim3 block(256, 1, 1);

    hipLaunchKernelGGL(
        {nc}_kernel,
        grid, block,
        LDS_SIZE,
        stream,
        config,
        ptr_table,
        counter_buf);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {{
        fprintf(stderr, "{nc}_launch: kernel launch failed: %s\\n",
                hipGetErrorString(err));
    }}
}}

// ============================================================================
// Finalization (no-op)
// ============================================================================
extern "C" void {nc}_finalize() {{
    printf("[Fleet MK/{nt}] Finalized\\n");
}}

// ============================================================================
// Python module
// ============================================================================
#ifdef FLEET_MK_PYTHON_MODULE
#include <Python.h>

static PyObject* py_{nc}_init(PyObject *self, PyObject *args) {{
    {nc}_init();
    Py_RETURN_NONE;
}}

static PyObject* py_{nc}_launch(PyObject *self, PyObject *args) {{
    int num_active_tokens;
    float attn_scale;
    unsigned long long cos_p, sin_p, qo_p, kv_ind_p, kv_idx_p, kv_lpl_p;
    unsigned long long ptr_table_p, counter_buf_p;
    unsigned long long lm_norm_w_p, lm_norm_scratch_p, lm_mxfp4_w_p, lm_bias_p, argmax_out_p;
    unsigned long long timing_buf_p;
    unsigned long long stream_p;

    if (!PyArg_ParseTuple(args, "ifKKKKKKKKKKKKKKK",
            &num_active_tokens, &attn_scale,
            &cos_p, &sin_p, &qo_p, &kv_ind_p, &kv_idx_p, &kv_lpl_p,
            &ptr_table_p, &counter_buf_p,
            &lm_norm_w_p, &lm_norm_scratch_p, &lm_mxfp4_w_p, &lm_bias_p, &argmax_out_p,
            &timing_buf_p,
            &stream_p)) {{
        return NULL;
    }}

    {nc}_launch(
        num_active_tokens, attn_scale,
        (void *)cos_p, (void *)sin_p,
        (int *)qo_p, (int *)kv_ind_p, (int *)kv_idx_p, (int *)kv_lpl_p,
        (void **)ptr_table_p, (void **)counter_buf_p,
        (void *)lm_norm_w_p, (void *)lm_norm_scratch_p,
        (void *)lm_mxfp4_w_p, (void *)lm_bias_p, (void *)argmax_out_p,
        (unsigned long long *)timing_buf_p,
        (hipStream_t)stream_p);

    Py_RETURN_NONE;
}}

static PyObject* py_{nc}_finalize(PyObject *self, PyObject *args) {{
    {nc}_finalize();
    Py_RETURN_NONE;
}}

static PyMethodDef {nt}Methods[] = {{
    {{"init", py_{nc}_init, METH_NOARGS, "Initialize {nt} Fleet MK"}},
    {{"launch", py_{nc}_launch, METH_VARARGS, "Launch {nt} Fleet MK kernel"}},
    {{"finalize", py_{nc}_finalize, METH_NOARGS, "Finalize {nt} Fleet MK"}},
    {{NULL, NULL, 0, NULL}}
}};

static struct PyModuleDef {nc}module = {{
    PyModuleDef_HEAD_INIT, "{nc}_fleet_mk", NULL, -1, {nt}Methods
}};

PyMODINIT_FUNC PyInit_{nc}_fleet_mk(void) {{
    return PyModule_Create(&{nc}module);
}}
#endif // FLEET_MK_PYTHON_MODULE
'''


# ============================================================================
# generate_kernel
# ============================================================================


# ============================================================================
# The per-layer dispatch block, verbatim
# ============================================================================
# Everything between `layer_counters` and the last-layer tail. Kept as module
# constants for the same reason as FLEET_SHIM_PREAMBLE: it is 200 lines of C++
# whose braces and #ifdef ladders would have to be escaped into an f-string,
# and hand-transcribing it is precisely how the generator and the .cuh drifted
# apart. The one value that varies by config -- the sliding-window expression
# on the non-fleet arm of the #ifdef -- is a {sw} placeholder.
#
# The comment bodies are load-bearing: they record two superseded conclusions
# (the "+1.09 ms dual-body penalty" and the "LDS alignment is ruled out"
# bullet) that cost days to disprove. Do not trim them.
FLEET_DISPATCH_PRE = r'''            // GPT-OSS alternates attention types per layer: even layers use a
            // 128-token sliding window, odd layers use full attention.
            //
            // The window is a TEMPLATE parameter, so this branches over two
            // instantiations of the layer body rather than passing the window
            // at runtime. That is what fleet's own codegen emits -- its
            // permanent_output_dir/test.cu has exactly two instantiations of
            // this function, identical but for SLIDING_WINDOW 0 and 128 -- and
            // fleet's trailing runtime argument is task_layer_idx, not a
            // window override, so the runtime path does not exist there at
            // all. Two bodies where there was one; the register and code-size
            // cost is measured in the build's kernel-resource-usage output.
            //
            // The two headers disagree about what the TRAILING runtime
            // argument means, and neither would diagnose passing the other's:
            //
            //   mirage  int sliding_window_override = -1   (-1 = use template)
            //   fleet   int task_layer_idx                 (monotonic counter)
            //
            // Fleet derives four release values from task_layer_idx
            // (routing_ready, attn_release, qkv_epoch, and the MoE W13->W2
            // barrier via an LDS slot), against counters that are host
            // allocations never reset between launches. So it must count
            // (decode_iter * NUM_LAYERS + layer) -- monotonic across the whole
            // run and identical on every worker -- exactly as fleet's own
            // dispatcher computes (pc_iter - 1) * num_layers + ml. Passing a
            // per-token value that restarts at 0 would re-satisfy every
            // barrier from the first layer onward.
            //
            // decode_iter comes from the dedicated counter line past the
            // per-layer blocks; it is bumped once per token by worker (0,0).
#ifdef FLEET_MK_FLEET_HEADERS
            int const layer_trailing_arg = decode_iter * NUM_LAYERS + layer;
#else
            int const layer_trailing_arg = {sw};
#endif
            // SUPERSEDED (2026-08-28): there is no two-instantiation cost.
            //
            // A long A/B series once attributed +1.05 ms/token to "executing
            // two distinct copies of this body in one token", and could not
            // explain it -- the penalty was fixed rather than proportional,
            // register counts and occupancy matched, and the never-executed
            // two-body arm was fast. Every one of those arms was confounded:
            // the fast ones were arms where LLVM dead-stripped
            // fleet_mk::g_layer_sliding_window (static LDS 720) and the slow ones
            // were arms that kept it (728). See the LayerWindowSlot note at the
            // top of this file for the controlled experiment that separates
            // them; the cause is static LDS size mod 16, not the body count.
            // dual-layer-body-cost.md in the wiki records the dead end.'''

FLEET_DISPATCH_POST = r'''
            // RESOLVED (2026-08-28). What was called the "+1.09 ms dual-body
            // penalty" was a measurement artifact; see the LayerWindowSlot note
            // at the top of this file. Kept here because two findings from the
            // hunt stand on their own:
            //
            //   * A REAL BUG, fixed: fleet_mk was writing MoE scale pointers into
            //     input_ptrs[24]/[25], which under fleet's header are the
            //     QKV-prefetch slots (see the note at those entries in
            //     demo_gpt_oss_120b.py). Worth qkv_gemm 15.40 -> 13.50 us,
            //     moe 0.00 -- ~11 us/token. Real, but two orders of magnitude
            //     short of the gap it was briefly credited with.
            //
            //   * A WRONG BULLET, recorded so it is not repeated: this block
            //     used to rule LDS alignment out on the grounds that "static
            //     LDS is 720 vs 728 but the dynamic-LDS base constants are
            //     0x2d0 in both, so no ds_ offset moves." Those constants are
            //     offsets from the dynamic base, not absolute -- they cannot
            //     move, and their equality says nothing about where the base
            //     itself sits. The 8-byte shift was the entire effect. When an
            //     ablation says a structural cause is ruled out, check that the
            //     evidence could have detected it.
#if defined(FLEET_MK_FLEET_HEADERS) && !defined(FLEET_MK_TWO_BODY_LEGACY)
            // ONE BODY, CORRECT WINDOWS. The template argument below is a
            // sentinel: the shim installed at the top of this file discards
            // it and forwards fleet_mk::g_layer_sliding_window instead, so a
            // single instantiation serves both layer types.
            //
            // GPT-OSS alternates: even layers use a 128-token sliding window,
            // odd layers full attention. Published here, before the body, by
            // every thread of every worker -- it is __shared__, so each
            // workgroup needs its own copy, and the value is a pure function
            // of `layer`, identical on all of them.
#ifdef FLEET_MK_TWO_CONST_WINDOWS
            // THE FIX. Two COMPILE-TIME instantiations, 128 and
            // FLEET_MK_FULL_WINDOW(=MAX_SEQ_LEN=512). Both are > 0, so in BOTH
            // bodies the compiler can prove `sliding_window == 0` false and
            // dead-strip the 21 KB __attn_wave_local_scan_hd64 path
            // (decode_minimal_hd64:754). 512 is arithmetically identical to
            // the unlimited window 0 at this MAX_SEQ_LEN -- see the note on
            // FLEET_MK_FULL_WINDOW -- so semantics are exact.
            if ((layer & 1) == 0) {
                FLEET_MK_LAYER_BODY(SLIDING_WINDOW);
            } else {
                FLEET_MK_LAYER_BODY(FLEET_MK_FULL_WINDOW);
            }
#else
            // Single writer plus a barrier. This IS needed, despite the
            // value being a pure function of `layer`: the window ALTERNATES,
            // so without ordering a fast wave can publish layer N+1's window
            // into the slot while a slow wave is still reading layer N's, and
            // the two differ. The barrier-free form was tried while chasing
            // the +1.09 ms (on the theory that a per-layer rendezvous was
            // serialising skewed waves) -- it measured 3.656 ms, i.e. no help
            // at all, because the cause was static LDS size. Reverted; the
            // cost of being right here is one workgroup barrier per layer.
            if (tid == 0) {
#ifdef FLEET_MK_PIN_RUNTIME_WINDOW
                // Diagnostic: still a RUNTIME value through the same shared
                // channel, but the same value on every layer. Separates
                // "runtime vs compile-time constant" from "value varies
                // across layers".  cur_token is a runtime value the compiler
                // cannot fold, so `& 0` keeps the store unfoldable while
                // pinning the value.
                // NOTE: the original form here was
                //   FLEET_MK_PIN_RUNTIME_WINDOW + (cur_token & 0)
                // which LLVM folds to a literal (x & 0 == 0), so it did NOT
                // test runtime-ness at all -- the build reported LDS 720, i.e.
                // the shared store was dead-stripped entirely. Launder the
                // value through an asm barrier instead: the compiler must
                // materialise it in a register and cannot constant-fold it,
                // while the value is still 128 on EVERY layer.
                int fleet_mk_w = FLEET_MK_PIN_RUNTIME_WINDOW;
                asm volatile("" : "+v"(fleet_mk_w));
                fleet_mk::g_layer_sliding_window = fleet_mk_w;
#else
                // Odd (full-attention) layers publish FLEET_MK_FULL_WINDOW rather
                // than 0. Any window >= MAX_SEQ_LEN is arithmetically identical
                // to full attention here, because every consumer guards on
                // `seqlen_k > sliding_window` and seqlen_k never reaches
                // MAX_SEQ_LEN:
                //   split_kv_mi300.cuh:288       kv_start stays 0
                //   decode_minimal_hd64.cuh:671  kv_start stays 0
                //   decode_minimal_hd64.cuh:953  win_first <= 0, mask is a
                //                                no-op (guarded win_first > 0)
                //   split_kv_mi300.cuh:555       left_size = W-1 >= seqlen_k-1,
                //                                so the causal mask is unclipped
                // The point is that it keeps every layer on the SAME
                // `sliding_window > 0` code path.
                //
                // ALTERNATION IS FREE. It was blamed for ~0.8 ms/token for two
                // investigation rounds; that was an artifact of the LDS-size
                // confound described at the LayerWindowSlot declaration. A
                // pinned runtime 128 on every layer measured 3.612 ms, i.e. no
                // better than alternating -- the cost tracked static LDS mod
                // 16, never the value. FLEET_MK_ODD_WINDOW survives as a knob for
                // re-testing that separation, default 0.
// Default 0 = fleet's true unlimited window, which is also what enables its
// wave-local scan fast path (decode_minimal_hd64:754). FLEET_MK_FULL_WINDOW(512)
// is arithmetically equivalent but disables that path, so it is a diagnostic
// only.
#ifndef FLEET_MK_ODD_WINDOW
#define FLEET_MK_ODD_WINDOW 0
#endif
                fleet_mk::g_layer_sliding_window =
                    ((layer & 1) == 0) ? SLIDING_WINDOW : FLEET_MK_ODD_WINDOW;
#endif
            }
            __syncthreads();
            FLEET_MK_LAYER_BODY(0);
#endif // FLEET_MK_TWO_CONST_WINDOWS
#elif defined(FLEET_MK_FLEET_HEADERS) && defined(FLEET_MK_MEASURE_ONE_BODY)
            // MEASUREMENT ONLY -- THIS BUILD PRODUCES WRONG OUTPUT.
            //
            // One instantiation under fleet's header, which means full
            // attention on the 18 even layers that need the 128-token window.
            // The text WILL be wrong; do not read it as a correctness signal
            // and do not ship this .so.
            //
            // Purpose: the fleet switch cost +1.19 ms/token, and two things
            // changed at once -- the forced second layer body and everything
            // else fleet's headers bring (scratch 0 -> 36 B/lane, LDS
            // 536 -> 728, AGPRs 6 -> 17, the prefetch, fleet's layer barrier).
            // This arm holds fleet's headers fixed and removes ONLY the second
            // body. It served its purpose and the answer was that the body
            // count is not the term at all: the LDS figure in that very list
            // (728, not a multiple of 16) was. Retained as an A/B harness.
            FLEET_MK_LAYER_BODY(0);
#elif defined(FLEET_MK_FLEET_HEADERS)
            // Fleet's header takes the window ONLY as a template parameter
            // (declared :118, used once at :610, where it is handed to
            // paged_attention_ck_fmha_split_kv_impl as a runtime argument --
            // exactly what mirage does at its :265, but with no override
            // parameter to reach it). So under fleet's header two
            // instantiations are forced. LEGACY PATH -- the shim above
            // avoids this; kept only for A/B against it.
            if ((layer & 1) == 0) {
                FLEET_MK_LAYER_BODY(SLIDING_WINDOW);
            } else {
                FLEET_MK_LAYER_BODY(0);
            }
#else
            // Mirage's header exposes sliding_window_override, so ONE body
            // serves both layer types without any shim.
            FLEET_MK_LAYER_BODY(0);
#endif
#undef FLEET_MK_LAYER_BODY
'''


# The fleet shim preamble, verbatim. Kept as a module constant rather than
# inlined in the f-string because it is C++ that contains braces and backslash
# continuations on nearly every line -- escaping it into the f-string would make
# every future edit a transcription exercise, which is exactly how the .cuh and
# the generator drifted apart in the first place.
FLEET_SHIM_PREAMBLE = r'''
// ============================================================================
// ONE LAYER BODY UNDER FLEET'S HEADER
// ============================================================================
// Fleet takes the sliding window as a TEMPLATE parameter
// (gang_full_layer_fused_mi300.cuh:118), which is what forced fleet_mk to
// instantiate the whole 26 KB layer body twice -- SW=128 for even layers,
// SW=0 for odd. (The second body was long believed to cost +1.05 ms/token;
// it does not -- that A/B was confounded by static LDS size, see the
// LayerWindowSlot note below. One body is still the right shape, just for
// code-size and clarity reasons rather than latency ones.)
//
// But the window is a template parameter in NAME ONLY. It is used at exactly
// ONE place in fleet's header (:610), where it is handed to
// paged_attention_ck_fmha_split_kv_impl as its 14th RUNTIME argument -- and
// every consumer below that point takes it as a plain `int sliding_window`
// and tests it at runtime (split_kv :288, minimal_hd64 :671/:754/:953).
// Nothing branches on it at compile time. So the two instantiations are not
// a fast path and a slow path; they are the same machine code with one
// immediate differing, and there is no semantic reason to emit both.
//
// Fleet's tree must not be edited, and the template parameter is the only
// channel it exposes -- so fleet_mk interposes on the call instead. The real
// definition is parsed FIRST under its real name (the include just below;
// `#pragma once` then makes fleet's own include of it at :40 a no-op), then
// the name is macro-redirected to a shim for the duration of the fused
// header. Fleet's :610 expands to the shim, which discards the template-
// derived window and forwards fleet_mk's per-layer runtime one.
//
// The redirect is scoped to a single include and is #undef'd immediately.
// Only gang_full_layer_fused_mi300.cuh calls this name in fleet_mk's include
// chain -- gang_qkv_attn_fused_mi300.cuh and gang_attention_mi300.cuh also
// call it, but neither is reachable from here.
//
// These four are fleet's own includes (gang_full_layer_fused_mi300.cuh:37-40),
// reproduced here in fleet's order so the preprocessor state fleet's body sees
// is identical by construction. `#pragma once` then makes fleet's own :37-40
// no-ops, and the only thing changed for that body is the name redirect below.
#include "tasks/mi300/gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh"
#include "tasks/mi300/gang_moe_fused_mxfp4_mi300.cuh"
#include "tasks/mi300/gang_rmsnorm_linear_mxfp4_bias_mi300.cuh"
#include "tasks/mi300/paged_attention_ck_fmha_split_kv_mi300.cuh"

namespace fleet_mk {
// Per-layer sliding window, published by the layer loop before the body runs
// and read inside it by the attention phase. Workgroup-scoped: every worker
// sets it for itself under a __syncthreads(), so no cross-worker ordering is
// involved. FLEET_MK_SHIM_CONST_WINDOW ablates it to a constant to isolate the
// shim + include reorder from the shared-variable channel.
//
// STATIC LDS SIZE MOD 16 IS LOAD-BEARING, and it is the whole +1.09 ms story.
//
// A bare `__shared__ int` here makes static LDS 728 bytes. 728 % 16 == 8, so
// the DYNAMIC LDS block -- which starts immediately after the static block and
// is what the MoE and QKV GEMM phases index -- begins 8 bytes past a 16-byte
// boundary, splitting every 16-byte ds_read/ds_write in those phases. Measured
// (tools/fused_phase_stats.py --compare): moe 25.2 -> 45.3 us and qkv_gemm
// 6.4 -> 13.5 us, both almost exactly 2x, both FLAT across layers and tokens,
// and neither phase touches attention at all.
//
// THE WINDOW WAS NEVER THE VARIABLE. Every "fast" arm in the investigation was
// one where LLVM happened to dead-strip this variable (LDS 720, a multiple of
// 16); every "slow" arm kept it (728). That coincidence produced, in turn, a
// false dual-body theory and a false alternation theory. The three arms that
// killed them, all measured:
//   * SHIM_CONST_WINDOW=128 + LDS_KEEPALIVE -- compile-time window, uniform
//     attention work, store held live at 728:            3.624 ms  (SLOW)
//   * LDS_PAD_ONLY, 1 word -- compile-time window, ZERO per-layer ds_write,
//     one padding word outside the layer loop, LDS 728:  3.673 ms  (SLOW)
//   * LDS_PAD_ONLY, 4 words -- identical but LDS 736:    2.552 ms  (FAST)
// Same window, same attention, same per-layer LDS traffic in the last two;
// only the block size mod 16 differs.
//
// TRAP THAT COST AN ENTIRE INVESTIGATION: the original "pinned runtime window"
// control arm wrote `CONST + (cur_token & 0)`, which LLVM folds to a literal,
// so it never tested runtime-ness -- the store was dead-stripped and the arm
// silently became a compile-time-constant arm. Use asm volatile("" : "+v"(x))
// to launder a value past constant folding, and always confirm intent against
// the build's reported "LDS Size [bytes/block]".
//
// THE FIX: pad the slot out to a full 16 bytes so the static block goes
// 720 -> 736, both multiples of 16, and the dynamic base stays aligned.
//
// The padding must be LIVE. Neither __align__(16) nor `alignas(16)` with a
// dead `int pad[3]` works: alignment sets this variable's own offset but LLVM
// strips unreferenced padding, so the block SIZE stayed at 728 and both
// attempts silently changed nothing. touch_lds_pad() below writes and reads
// every pad word through an asm barrier, once, outside the layer loop.
// Verified by the build's own "LDS Size [bytes/block]" line: it must read 736.
struct alignas(16) LayerWindowSlot {
  int window;
  int pad[3];
};
__shared__ LayerWindowSlot g_layer_window_slot;
#define g_layer_sliding_window g_layer_window_slot.window

// Keep the pad words from being dead-stripped. Costs three ds_write + three
// ds_read on thread 0 for the whole decode step, not per layer.
__device__ __forceinline__ void touch_lds_pad(int tid, int seed) {
  if (tid == 0) {
    int v = seed;
    asm volatile("" : "+v"(v));
#pragma unroll
    for (int i = 0; i < 3; i++) g_layer_window_slot.pad[i] = v + i;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 3; i++) {
    int r = g_layer_window_slot.pad[i];
    asm volatile("" : : "v"(r));
  }
}

} // namespace fleet_mk

namespace kernel {
// Signature mirrors paged_attention_ck_fmha_split_kv_impl exactly (10 template
// parameters, 16 runtime arguments). Argument 14 -- the window fleet derived
// from its template parameter -- is accepted and ignored.
template <typename T,
          int NUM_QO_PER_KV,
          int HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE_T,
          int NUM_KV_HEADS_T,
          bool DECODE_ONLY = false>
__device__ __forceinline__ void
    fleet_mk_split_kv_runtime_window(void const *q_workspace_ptr,
                                  void *paged_k_cache_ptr,
                                  void *paged_v_cache_ptr,
                                  void *o_acc_ptr,
                                  void *lse_acc_ptr,
                                  int const *qo_indptr_buffer_ptr,
                                  int const *paged_kv_indptr_buffer_ptr,
                                  int const *paged_kv_indices_buffer_ptr,
                                  int const *paged_kv_last_page_len_buffer_ptr,
                                  int16_t request_id,
                                  int kv_head_idx,
                                  int kv_chunk_idx,
                                  float scale_s,
                                  int /*template_sliding_window -- ignored*/,
                                  void const *sinks_ptr = nullptr) {
  paged_attention_ck_fmha_split_kv_impl<T,
                                        NUM_QO_PER_KV,
                                        HEAD_DIM,
                                        PAGE_SIZE,
                                        MAX_SEQ_LEN,
                                        NUM_KV_CHUNKS,
                                        Q_WORKSPACE_STRIDE,
                                        KV_CACHE_STRIDE_T,
                                        NUM_KV_HEADS_T,
                                        DECODE_ONLY>(
      q_workspace_ptr,
      paged_k_cache_ptr,
      paged_v_cache_ptr,
      o_acc_ptr,
      lse_acc_ptr,
      qo_indptr_buffer_ptr,
      paged_kv_indptr_buffer_ptr,
      paged_kv_indices_buffer_ptr,
      paged_kv_last_page_len_buffer_ptr,
      request_id,
      kv_head_idx,
      kv_chunk_idx,
      scale_s,
#ifdef FLEET_MK_SHIM_CONST_WINDOW
      FLEET_MK_SHIM_CONST_WINDOW,
#elif defined(FLEET_MK_CLAMP_WINDOW)
      // Forward the per-layer window through a CLAMP the compiler can see:
      // the result provably lies in [SLIDING_WINDOW, FLEET_MK_FULL_WINDOW], so it
      // is provably > 0. That lets LLVM fold `sliding_window == 0` to false and
      // dead-strip the 21 KB __attn_wave_local_scan_hd64 path
      // (decode_minimal_hd64:754) even though the VALUE is a runtime load.
      // Both legal values (128 and 512) survive the clamp unchanged, so this
      // is semantically a no-op -- 512 is identical to the unlimited window at
      // MAX_SEQ_LEN=512 (see FLEET_MK_FULL_WINDOW).
      // (Literals, not SLIDING_WINDOW / FLEET_MK_FULL_WINDOW: those constants are
      // declared below this shim. They are asserted equal at their definition.)
      (fleet_mk::g_layer_sliding_window < 128
           ? 128
           : (fleet_mk::g_layer_sliding_window > 512
                  ? 512
                  : fleet_mk::g_layer_sliding_window)),
#else
      fleet_mk::g_layer_sliding_window,
#endif
      sinks_ptr);
}

// Same shim, but the window reaches the attention impl as a COMPILE-TIME
// constant on both sides of a runtime branch. This specializes only the
// attention function (~12.9 KB) rather than the whole 26.5 KB layer body, so
// there is still exactly ONE gang_full_layer_fused_kernel_mi300.
//
// Motivation: with a single body, a constant window runs 2.527 ms (all 128)
// / 2.603 ms (all 0), but alternating per layer runs 3.35 ms -- worse than
// either constant, so it is not attention work. The suspicion is that a
// genuinely-runtime window defeats specialization inside the attention path,
// and that the fast "pinned runtime" arm (2.556 ms) was silently constant-
// folded by LLVM (single constant store to a shared variable).
template <typename T,
          int NUM_QO_PER_KV,
          int HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE_T,
          int NUM_KV_HEADS_T,
          bool DECODE_ONLY = false>
__device__ __forceinline__ void
    fleet_mk_split_kv_specialized_window(void const *q_workspace_ptr,
                                      void *paged_k_cache_ptr,
                                      void *paged_v_cache_ptr,
                                      void *o_acc_ptr,
                                      void *lse_acc_ptr,
                                      int const *qo_indptr_buffer_ptr,
                                      int const *paged_kv_indptr_buffer_ptr,
                                      int const *paged_kv_indices_buffer_ptr,
                                      int const *paged_kv_last_page_len_ptr,
                                      int16_t request_id,
                                      int kv_head_idx,
                                      int kv_chunk_idx,
                                      float scale_s,
                                      int /*template window -- ignored*/,
                                      void const *sinks_ptr = nullptr) {
#define FLEET_MK_CALL_ATTN(WIN_)                                                    paged_attention_ck_fmha_split_kv_impl<T,                                                                             NUM_QO_PER_KV,                                                                 HEAD_DIM,                                                                      PAGE_SIZE,                                                                     MAX_SEQ_LEN,                                                                   NUM_KV_CHUNKS,                                                                 Q_WORKSPACE_STRIDE,                                                            KV_CACHE_STRIDE_T,                                                             NUM_KV_HEADS_T,                                                                DECODE_ONLY>(q_workspace_ptr,                                                               paged_k_cache_ptr,                                                             paged_v_cache_ptr,                                                             o_acc_ptr,                                                                     lse_acc_ptr,                                                                   qo_indptr_buffer_ptr,                                                          paged_kv_indptr_buffer_ptr,                                                      paged_kv_indices_buffer_ptr,                                                      paged_kv_last_page_len_ptr,                                                      request_id,                                                                    kv_head_idx,                                                                   kv_chunk_idx,                                                                  scale_s,                                                                       (WIN_),                                                                        sinks_ptr)
  if (fleet_mk::g_layer_sliding_window == 0) {
    FLEET_MK_CALL_ATTN(0);
  } else {
    FLEET_MK_CALL_ATTN(128);
  }
#undef FLEET_MK_CALL_ATTN
}
} // namespace kernel

// FLEET_MK_NO_SHIM keeps the early split-KV include but does NOT redirect the
// name, so fleet's body compiles exactly as it does in the shipping build.
// Bisects "include reorder" against "shim".
#if defined(FLEET_MK_FLEET_HEADERS) && !defined(FLEET_MK_NO_SHIM)
#ifdef FLEET_MK_SPECIALIZE_WINDOW
#define paged_attention_ck_fmha_split_kv_impl                                  \
  kernel::fleet_mk_split_kv_specialized_window
#else
#define paged_attention_ck_fmha_split_kv_impl                                  \
  kernel::fleet_mk_split_kv_runtime_window
#endif
#endif
'''

FLEET_SHIM_UNDEF = r'''#if defined(FLEET_MK_FLEET_HEADERS) && !defined(FLEET_MK_NO_SHIM)
#undef paged_attention_ck_fmha_split_kv_impl
#endif
'''


def emit_fleet_shim(cfg: ModelConfig) -> str:
    """Macro interposition on fleet's attention callee, fleet arm only.

    Fleet exposes the sliding window as a template parameter only, and names the
    attention callee UNQUALIFIED at gang_full_layer_fused_mi300.cuh:610. Since
    fleet's tree is used out of the box and must not be edited, fleet_mk parses the
    real definition first, redirects the name to a shim for the duration of one
    include, and #undef's it immediately after. One instantiation then serves
    both layer parities.

    UNPROTECTED COUPLING: if fleet ever qualifies or renames that call, the
    redirect stops applying and fleet_mk silently runs full attention on the 18
    sliding-window layers. That is fluent wrong output with no compile error.
    """
    return FLEET_SHIM_PREAMBLE if cfg.headers == "fleet" else ""


def emit_fleet_shim_undef(cfg: ModelConfig) -> str:
    """Close the redirect opened by emit_fleet_shim."""
    return FLEET_SHIM_UNDEF if cfg.headers == "fleet" else ""


def generate_kernel(cfg: ModelConfig) -> str:
    """Emit the persistent kernel .cuh for `cfg`.

    Dispatches on architecture. The two arms emit fundamentally different
    pipelines -- not two settings of one pipeline -- which is why this is a
    dispatcher and not a wall of `if cfg.arch ==` ternaries inside one
    f-string. The dense arm is the original device-function pipeline; the MoE
    arm calls mirage's fused full-layer kernel.
    """
    if cfg.arch == "dense":
        return generate_kernel_dense(cfg)
    if cfg.arch == "moe":
        return generate_kernel_fused_moe(cfg)
    raise ValueError(f"Unknown arch '{cfg.arch}'")


def generate_kernel_fused_moe(cfg: ModelConfig) -> str:
    """MoE (GPT-OSS 120B) pipeline: one barrier per layer, delegating the whole
    layer body to mirage's gang_full_layer_fused_kernel_mi300.

    THIS IS THE MAINTAINED SOURCE of generated/gpt_oss_120b_kernel.cuh, the
    artifact that measures 2.520 ms/tok. Edit here, never the .cuh -- the gate
    is byte-identity, checked by `check_roundtrip.py --strict`.

    Built backwards: the hand-written .cuh was pasted in verbatim (brace- and
    backslash-escaped) and then parameterized one literal at a time, re-running
    the gate after each substitution. Anything still a literal below is a
    literal because parameterizing it was not shown to buy anything -- see the
    "preserve verbatim" notes at the substitution sites.
    """
    return f'''\
/* Auto-generated by fleet_mk_generate.py
 * Fleet MK: Persistent kernel for gpt-oss-120b (MoE) on MI350X
 *
 * Calls mirage's gang_full_layer_fused_kernel_mi300 directly for each layer,
 * eliminating any pointer/barrier mapping bugs. The layer loop + end-of-layer
 * barrier + tail (LM head) are Fleet MK's own code.
 */
#pragma once

#include "common.cuh"

// Device function library (barriers, helpers)
#include "device_functions.cuh"

// Mirage full-layer fused kernel (includes all sub-kernels)
#include "tasks/common/utils.cuh"
#ifndef NUM_THREADS
#define NUM_THREADS 256
#endif
#include "tasks/ampere/merge_splitkv.cuh"
// Minimal hd64 attention must be included BEFORE split_kv (which calls it)
#include "tasks/mi300/paged_attention_decode_minimal_hd64_mi300.cuh"
{emit_fleet_shim(cfg)}\
#include "tasks/mi300/gang_full_layer_fused_mi300.cuh"
{emit_fleet_shim_undef(cfg)}\

// ============================================================================
// gpt-oss-120b Architecture Constants
// ============================================================================
namespace gpt_oss_120b {{

// Model dimensions
static constexpr int NUM_LAYERS = {cfg.num_layers};
static constexpr int HIDDEN_SIZE = {cfg.padded_hidden_size};
static constexpr int ACTUAL_HIDDEN_DIM = {cfg.hidden_size};  // for RMSNorm mean
static constexpr int INTERMEDIATE_SIZE = {cfg.padded_intermediate_size};
static constexpr int VOCAB_SIZE = {cfg.vocab_size};
static constexpr int PADDED_VOCAB_SIZE = {cfg.padded_vocab_size};  // next multiple of 256

// Attention
static constexpr int NUM_Q_HEADS = {cfg.num_q_heads};
static constexpr int NUM_KV_HEADS = {cfg.num_kv_heads};
static constexpr int HEAD_DIM = {cfg.head_dim};
static constexpr int NUM_Q_PER_KV = {cfg.q_per_kv};  // {cfg.num_q_heads} / {cfg.num_kv_heads}

// QKV GEMM output: Q({cfg.oproj_reduction}) + K({cfg.kv_cache_stride}) + V({cfg.kv_cache_stride}) = {cfg.qkv_output_size}
static constexpr int QKV_OUTPUT_SIZE = NUM_Q_HEADS * HEAD_DIM
                                     + 2 * NUM_KV_HEADS * HEAD_DIM;  // {cfg.qkv_output_size}

// GPU layout
static constexpr int NUM_XCDS = {cfg.num_xcds};
static constexpr int WORKERS_PER_XCD = {cfg.workers_per_xcd};
static constexpr int TOTAL_WORKERS = NUM_XCDS * WORKERS_PER_XCD;

// All GEMMs use OPW={cfg.output_per_wg} for QKV, OPW={cfg.oproj_opw} for OProj
static constexpr int OUTPUT_PER_WG = {cfg.output_per_wg};

// QKV GEMM
static constexpr int QKV_N_WGS = QKV_OUTPUT_SIZE / OUTPUT_PER_WG;     // {cfg.qkv_n_wgs}
static constexpr int QKV_N_WGS_PER_XCD = QKV_N_WGS / NUM_XCDS;       // {cfg.qkv_n_wgs_per_xcd}

// O-proj GEMM
static constexpr int OPROJ_REDUCTION = NUM_Q_HEADS * HEAD_DIM;         // {cfg.oproj_reduction}
static constexpr int OPROJ_OPW = {cfg.oproj_opw};
static constexpr int OPROJ_N_WGS = HIDDEN_SIZE / OPROJ_OPW;            // {cfg.oproj_n_wgs}
static constexpr int OPROJ_N_WGS_PER_XCD = OPROJ_N_WGS / NUM_XCDS;    // {cfg.oproj_n_wgs_per_xcd}
static constexpr int OPROJ_TILES_PER_XCD = OPROJ_N_WGS_PER_XCD;       // {cfg.oproj_n_wgs_per_xcd}

// MoE expert GEMMs
static constexpr int NUM_EXPERTS = {cfg.num_experts};
static constexpr int NUM_TOPK = {cfg.num_experts_per_tok};
static constexpr int MOE_INTERMEDIATE_SIZE = {cfg.padded_moe_intermediate_size};

static constexpr int W13_OPW = {cfg.w13_output_per_wg};
static constexpr int W13_OUTPUT_SIZE = 2 * MOE_INTERMEDIATE_SIZE;      // {cfg.w13_output_size}
static constexpr int W13_N_WGS = W13_OUTPUT_SIZE / W13_OPW;           // {cfg.w13_n_wgs}
static constexpr int W2_OPW = {cfg.w2_output_per_wg};
static constexpr int W2_N_WGS = HIDDEN_SIZE / W2_OPW;                 // {cfg.w2_n_wgs}

// Router
static constexpr int ROUTER_TILE_N = NUM_EXPERTS / NUM_XCDS;  // {cfg.num_experts // cfg.num_xcds}
static constexpr int TOTAL_TOPK_TILES = NUM_EXPERTS;           // {cfg.num_experts}

// Fused MoE tile count per XCD (matches mirage: OPW={cfg.w13_output_per_wg} for W13)
static constexpr int MOE_TOTAL_TILES_PER_XCD = {cfg.moe_total_tiles_per_xcd};

// LM head
static constexpr int LM_N_WGS = PADDED_VOCAB_SIZE / OUTPUT_PER_WG;    // {cfg.lm_n_wgs}
static constexpr int LM_N_WGS_PER_XCD = LM_N_WGS / NUM_XCDS;         // {cfg.lm_n_wgs_per_xcd}

// Attention
static constexpr int PAGE_SIZE = {cfg.page_size};
// GPT-OSS alternates attention types: even layers are "sliding_attention" with a
// {cfg.sliding_window}-token window, odd layers are "full_attention" (window 0 = unlimited).
static constexpr int SLIDING_WINDOW = {cfg.sliding_window};
static constexpr int NUM_KV_CHUNKS = {cfg.num_kv_chunks};
static constexpr int MAX_SEQ_LEN = {cfg.max_seq_len};
{emit_full_window(cfg)}\
static constexpr int KV_CACHE_STRIDE = NUM_KV_HEADS * HEAD_DIM;       // {cfg.kv_cache_stride}
static constexpr int Q_WORKSPACE_STRIDE = NUM_Q_PER_KV * HEAD_DIM;    // {cfg.q_workspace_stride}

// Pointer table layout: pre-computed mirage format per-XCD per-layer
// {len(MIRAGE_IN)} mirage_in + {len(MIRAGE_OUT)} mirage_out + 1 layer_output = {MIRAGE_PTRS_PER_LAYER}
static constexpr int MIRAGE_IN_COUNT = {len(MIRAGE_IN)};
static constexpr int MIRAGE_OUT_COUNT = {len(MIRAGE_OUT)};
static constexpr int PTRS_PER_LAYER = MIRAGE_IN_COUNT + MIRAGE_OUT_COUNT + 1;  // {MIRAGE_PTRS_PER_LAYER}

// Counter layout per layer (must match COUNTERS_PER_LAYER in device_functions.cuh)
// The layer body addresses every counter as an offset from one
// "oproj_counters_base" pointer (input_ptrs[16]), so this block is SHARED with
// it: {emit_counter_ownership()}
{emit_counter_reserved_note(cfg)}\
{emit_counter_map(cfg)}
{emit_counter_total_note(cfg)}\
{emit_counter_slots(cfg)}
// Embedding-write barrier lives AFTER the per-layer blocks and the rank /
// decode-iter counters. The per-layer block's {cfg.counters_per_layer // 16} cache lines are fully
// allocated (see SLOT_* in device_functions.cuh), so reusing an index there
// silently corrupts a live barrier.
static constexpr int EMBED_BARRIER_BASE =
{emit_embed_barrier_base()}
static constexpr int SLOT_EMBED_DONE  = EMBED_BARRIER_BASE;          // global (1 cache line)
static constexpr int SLOT_EMBED_LOCAL = EMBED_BARRIER_BASE + 16;     // per-XCD (8 cache lines)

// End-of-decode-step barrier, used only when one launch covers several steps.
// Its own region, past every other, so adding it moved no existing offset.
static constexpr int LOOP_BARRIER_BASE =
{emit_loop_barrier_base()}
static constexpr int SLOT_LOOP_DONE  = LOOP_BARRIER_BASE;            // global (1 cache line)
static constexpr int SLOT_LOOP_LOCAL = LOOP_BARRIER_BASE + 16;       // per-XCD (8 cache lines)

// MoE barrier size (for zeroing between decode iterations)
static constexpr int MOE_BARRIER_INTS = 16 * NUM_EXPERTS;  // 2048

// Pointer table slot indices (pre-computed mirage format)
// mirage_in[0..{len(MIRAGE_IN) - 1}] are at slots 0..{len(MIRAGE_IN) - 1}
// mirage_out[0..{len(MIRAGE_OUT) - 1}] are at slots {len(MIRAGE_IN)}..{len(MIRAGE_IN) + len(MIRAGE_OUT) - 1}
// layer_output is at slot {len(MIRAGE_IN) + len(MIRAGE_OUT)}
static constexpr int SLOT_LAYER_OUTPUT = MIRAGE_IN_COUNT + MIRAGE_OUT_COUNT; // {len(MIRAGE_IN) + len(MIRAGE_OUT)}

}} // namespace gpt_oss_120b


// ============================================================================
// On-GPU decode loop control block (lives in GPU global memory)
// ============================================================================
struct DecodeControl {{
    int max_decode_tokens;   // how many tokens to generate
    int start_pos;           // cur_pos for first decode token
    int eos_token_id;        // stop on this token (-1 = never)
    // Decode-step epoch, biased by one so that 0 means "not supplied".
    //
    // fleet's task_layer_idx must be the EXACT count of layer-executions that
    // precede this one -- not merely monotonic. Its barriers derive
    // `expected = layer_counter + 1` against counters that are never reset, so
    // an epoch that runs ahead of the arrivals waits on a value nothing will
    // ever write, and all 240 workers hang.
    //
    // The kernel's own `rank_counter / WORKERS_PER_XCD` supplies that count for
    // free while one launch is one decode step. It stops being the answer the
    // moment a launch covers N steps, because it counts LAUNCHES -- hence this
    // field. It is biased so a zeroed DecodeControl (what every pre-existing
    // caller passes) selects the atomic-derived value and is bit-identical to
    // the behaviour before this field existed.
    int iter_base_p1;        // 0 = derive from rank counter; else epoch + 1
    int *kv_indptr;          // writable: kernel updates per iteration
    int *kv_last_page_len;   // writable: kernel updates per iteration
    int *token_output_buf;   // [max_decode_tokens] output token IDs
    int *num_generated;      // output: how many tokens actually generated
}};

// ============================================================================
// gpt-oss-120b Runtime Config (kernel argument — passed by value, UNCHANGED)
// ============================================================================
struct GptConfig {{
    int num_active_tokens;
    float attn_scale;

    void const *cos_ptr;
    void const *sin_ptr;

    int const *qo_indptr;
    int const *kv_indptr;
    int const *kv_indices;
    int const *kv_last_page_len;

    void const *lm_norm_weight;
    void       *lm_norm_scratch;
    void const *lm_mxfp4_weight;
    void const *lm_bias;
    void       *argmax_output;
    // Optional bf16 logits sink, [padded_vocab_size]. Null (the standalone
    // default) keeps the argmax-only tail: the epilogue already computes every
    // logit and throws all but the max away. vLLM needs the full row for its
    // sampler, and its own bf16 lm_head costs a 1.16 GB weight read to
    // recompute what this kernel just discarded -- 0.200 ms/token measured,
    // against 315 MB for the MXFP4 weights already resident here. Non-null =>
    // the epilogue also stores, which is a 402 KB write (201216 x bf16).
    void       *logits_output;

    unsigned long long *timing_buf;

    // Embedding: passed to kernel so we skip Python zero+copy per token
    void const *embed_weight;   // [vocab_size, ACTUAL_HIDDEN_DIM] bf16
    int         cur_token_id;   // first token to embed into residual (used if cur_token_ptr==nullptr)
    int        *cur_token_ptr;  // if non-null, read token from GPU memory (for pipelined decode)
}};


// ============================================================================
// LM head GEMM + inline argmax (separated to reduce register pressure)
// ============================================================================
__device__ __noinline__ void
gpt_oss_120b_lmhead_gemm_argmax(
    const void *lm_norm_scratch, const void *lm_mxfp4_weight,
    const void *lm_bias_ptr, int num_active_tokens,
    float *argmax_packed_base, int xcd_id, int xcd_rank,
    unsigned short *logits_out) {{

    using namespace gpt_oss_120b;
    int const tid = threadIdx.x;

    constexpr int LM_NUM_BLOCKS_32 = HIDDEN_SIZE / 32;
    constexpr int LM_WG_DATA_BYTES = OUTPUT_PER_WG * (HIDDEN_SIZE / 2);
    constexpr int LM_WG_SCALE_BYTES = OUTPUT_PER_WG * LM_NUM_BLOCKS_32;
    constexpr int LM_WG_BYTES_TOTAL = LM_WG_DATA_BYTES + LM_WG_SCALE_BYTES;
    constexpr int K_PER_MFMA = 128;
    constexpr int MFMA_ITERS = HIDDEN_SIZE / K_PER_MFMA;
    constexpr int NUM_WAVES = 4;
    constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;

    extern __shared__ char _lm_smem[];
    uint8_t *s_tok_fp8 = (uint8_t *)_lm_smem;
    uint8_t *s_tok_scales = s_tok_fp8 + HIDDEN_SIZE;

    // FP8 quantize the normalized input
    kernel::_gang_wave_parallel_fp8_quant<HIDDEN_SIZE>(
        (const unsigned short *)lm_norm_scratch, s_tok_fp8, s_tok_scales);

    const uint8_t *lm_W = (const uint8_t *)lm_mxfp4_weight
        + static_cast<int64_t>(xcd_id) * LM_N_WGS_PER_XCD * LM_WG_BYTES_TOTAL;
    const unsigned short *lm_bias = (const unsigned short *)lm_bias_ptr
        + xcd_id * LM_N_WGS_PER_XCD * OUTPUT_PER_WG;

    int const warp_id = tid >> 6;
    int const lane_id = tid & 63;
    int const col = lane_id & 15;
    int const g = lane_id >> 4;

    float thread_max = -1e30f;
    long long thread_max_idx = -1;
    int partition_start = xcd_id * LM_N_WGS_PER_XCD * OUTPUT_PER_WG;

    int lm_total_tiles = LM_N_WGS_PER_XCD * num_active_tokens;
    for (int lm_t = xcd_rank; lm_t < lm_total_tiles; lm_t += WORKERS_PER_XCD) {{
        int wg_idx = lm_t % LM_N_WGS_PER_XCD;
        uint8_t const *wg_data = lm_W + static_cast<int64_t>(wg_idx) * LM_WG_BYTES_TOTAL;
        uint8_t const *wg_scales = wg_data + LM_WG_DATA_BYTES;

        for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {{
            int wave_tile = warp_id + tile_iter * NUM_WAVES;
            int w_row = wave_tile * 16 + col;
            int const row_scale_base = w_row * LM_NUM_BLOCKS_32;
#ifdef MPK_LM_HEAD_KMAJOR
            constexpr int LM_TILE_DATA = 16 * (HIDDEN_SIZE / 2);
            constexpr int LM_K_STRIDE = 16 * (K_PER_MFMA / 2);
            uint8_t const *tile_data = wg_data + wave_tile * LM_TILE_DATA;
            int const data_lane_offset = lane_id * 16;
#define FLEET_MK_LM_W_FRAG(KI) (*((const kernel::i32x8_t *)(tile_data + data_lane_offset + (KI) * LM_K_STRIDE)))
#else
            int const row_data_base = w_row * (HIDDEN_SIZE / 2);
#define FLEET_MK_LM_W_FRAG(KI) (*((const kernel::i32x8_t *)(wg_data + row_data_base + (KI) * 64 + g * 16)))
#endif

            kernel::f32x4_t acc = {{0.0f, 0.0f, 0.0f, 0.0f}};
            kernel::i32x8_t a0 = FLEET_MK_LM_W_FRAG(0);
            int sa0 = (int)wg_scales[row_scale_base + 0 * 4 + g];
            kernel::i32x8_t a1 = FLEET_MK_LM_W_FRAG(1);
            int sa1 = (int)wg_scales[row_scale_base + 1 * 4 + g];
            kernel::i32x8_t a2 = FLEET_MK_LM_W_FRAG(2);
            int sa2 = (int)wg_scales[row_scale_base + 2 * 4 + g];
            kernel::i32x8_t a3 = FLEET_MK_LM_W_FRAG(3);
            int sa3 = (int)wg_scales[row_scale_base + 3 * 4 + g];

            #pragma unroll 1
            for (int ki = 0; ki < MFMA_ITERS; ki += 4) {{
                {{
                    kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
                    int sb = (int)s_tok_scales[ki];
                    acc = kernel::_gang_mfma_f4xf8(a0, b, acc, sa0, sb);
                }}
                if (ki + 4 < MFMA_ITERS) {{
                    int kt = (ki + 4) * K_PER_MFMA;
                    a0 = FLEET_MK_LM_W_FRAG(ki + 4);
                    sa0 = (int)wg_scales[row_scale_base + kt / 32 + g];
                }}
                {{
                    kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
                    int sb = (int)s_tok_scales[ki + 1];
                    acc = kernel::_gang_mfma_f4xf8(a1, b, acc, sa1, sb);
                }}
                if (ki + 5 < MFMA_ITERS) {{
                    int kt = (ki + 5) * K_PER_MFMA;
                    a1 = FLEET_MK_LM_W_FRAG(ki + 5);
                    sa1 = (int)wg_scales[row_scale_base + kt / 32 + g];
                }}
                {{
                    kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
                    int sb = (int)s_tok_scales[ki + 2];
                    acc = kernel::_gang_mfma_f4xf8(a2, b, acc, sa2, sb);
                }}
                if (ki + 6 < MFMA_ITERS) {{
                    int kt = (ki + 6) * K_PER_MFMA;
                    a2 = FLEET_MK_LM_W_FRAG(ki + 6);
                    sa2 = (int)wg_scales[row_scale_base + kt / 32 + g];
                }}
                if (ki + 3 < MFMA_ITERS) {{
                    kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
                    int sb = (int)s_tok_scales[ki + 3];
                    acc = kernel::_gang_mfma_f4xf8(a3, b, acc, sa3, sb);
                }}
                if (ki + 7 < MFMA_ITERS) {{
                    int kt = (ki + 7) * K_PER_MFMA;
                    a3 = FLEET_MK_LM_W_FRAG(ki + 7);
                    sa3 = (int)wg_scales[row_scale_base + kt / 32 + g];
                }}
            }}

#undef FLEET_MK_LM_W_FRAG

            // Argmax epilogue
            if (col == 0) {{
                for (int i = 0; i < 4; i++) {{
                    int out_n = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
                    float sum = acc[i];
                    unsigned bt = (unsigned)lm_bias[out_n] << 16;
                    float bv;
                    __builtin_memcpy(&bv, &bt, 4);
                    float val = sum + bv;
                    long long abs_idx = (long long)(partition_start + out_n);
                    // Optional logits store. Same value the argmax compares, at
                    // the same absolute vocab index -- so this cannot disagree
                    // with the token the argmax picks. bf16 to match what vLLM's
                    // own lm_head would have produced. Predicated on a null
                    // check, not #ifdef: one build serves both drivers, and the
                    // standalone path passes null and pays only the branch.
                    if (logits_out != nullptr) {{
                        unsigned vb;
                        __builtin_memcpy(&vb, &val, 4);
                        logits_out[abs_idx] = (unsigned short)(vb >> 16);
                    }}
                    if (val > thread_max) {{
                        thread_max = val;
                        thread_max_idx = abs_idx;
                    }}
                }}
            }}
        }}
    }}

    // Per-worker warp reduce
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {{
        float other_val = __shfl_xor(thread_max, offset, 64);
        unsigned int idx_lo = static_cast<unsigned int>(thread_max_idx & 0xFFFFFFFF);
        unsigned int idx_hi = static_cast<unsigned int>((thread_max_idx >> 32) & 0xFFFFFFFF);
        unsigned int other_lo = __shfl_xor(idx_lo, offset, 64);
        unsigned int other_hi = __shfl_xor(idx_hi, offset, 64);
        long long other_idx = (static_cast<long long>(other_hi) << 32) | other_lo;
        if (other_val > thread_max) {{
            thread_max = other_val;
            thread_max_idx = other_idx;
        }}
    }}

    // Cross-warp reduce via shared memory
    __shared__ float s_max_vals[4];
    __shared__ long long s_max_idxs[4];
    if (lane_id == 0) {{
        s_max_vals[warp_id] = thread_max;
        s_max_idxs[warp_id] = thread_max_idx;
    }}
    __syncthreads();

    if (tid == 0) {{
        float best_val = -1e30f;
        long long best_idx = -1;
        for (int w = 0; w < 4; w++) {{
            if (s_max_vals[w] > best_val) {{
                best_val = s_max_vals[w];
                best_idx = s_max_idxs[w];
            }}
        }}
        unsigned long long *argmax_packed =
            reinterpret_cast<unsigned long long *>(&argmax_packed_base[xcd_id * 4]);
        unsigned long long old_packed = __atomic_load_n(argmax_packed, __ATOMIC_RELAXED);
        while (true) {{
            int old_val_bits = static_cast<int>(old_packed & 0xFFFFFFFF);
            float old_val;
            __builtin_memcpy(&old_val, &old_val_bits, 4);
            if (best_val <= old_val) break;
            int new_val_bits;
            __builtin_memcpy(&new_val_bits, &best_val, 4);
            int new_idx_bits = static_cast<int>(best_idx);
            unsigned long long new_packed =
                (static_cast<unsigned long long>(static_cast<unsigned int>(new_idx_bits)) << 32)
                | static_cast<unsigned int>(new_val_bits);
            if (__atomic_compare_exchange_n(argmax_packed, &old_packed, new_packed,
                                             true, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                break;
        }}
    }}
}}

// ============================================================================
// The gpt-oss-120b persistent kernel
// Calls mirage's gang_full_layer_fused_kernel_mi300 for each layer,
// with a Fleet MK-specific end-of-layer barrier and tail (LM head).
// ============================================================================
__global__ void __launch_bounds__(256)
gpt_oss_120b_kernel(
    GptConfig config,
    void **ptr_table,       // [NUM_XCDS * NUM_LAYERS * PTRS_PER_LAYER]
    int  *counter_buf,
    DecodeControl *decode_ctrl)  // on-GPU decode loop control (in GPU memory)
{{
    using namespace gpt_oss_120b;
    // -- Worker identification --
    int xcd_id = get_xcd_id();
    int tid = threadIdx.x;

    // XCD-local rank via atomic counter
    int *rank_counters = counter_buf + NUM_LAYERS * fleet_mk::COUNTERS_PER_LAYER;

{emit_decode_iter_note(cfg)}\
    __shared__ int s_xcd_rank;
{emit_decode_iter_decl(cfg)}\
    if (tid == 0) {{
{emit_xcd_rank_atomic(cfg)}\
    }}
    __syncthreads();
    int xcd_rank = s_xcd_rank;
{emit_decode_iter_read(cfg)}\

    // Global tile_idx for mirage orchestrator (encodes xcd_id + xcd_rank)
    int tile_idx = xcd_id * WORKERS_PER_XCD + xcd_rank;

    // Mirage pointer arrays (populated per layer directly from ptr_table)
    __shared__ void *mirage_in[{len(MIRAGE_IN)}];
    __shared__ void *mirage_out[{len(MIRAGE_OUT)}];
    __shared__ int  *s_layer_counters;

    // Pointer table base for this XCD
    int xcd_table_base = xcd_id * NUM_LAYERS * PTRS_PER_LAYER;

    // Tail counter pointers
    int *tail_counters = counter_buf + (NUM_LAYERS - 1) * fleet_mk::COUNTERS_PER_LAYER;
    int *lmhead_done   = tail_counters + SLOT_TAIL_LMHEAD_NEW;
    float *argmax_packed_base = reinterpret_cast<float *>(
        tail_counters + SLOT_TAIL_ARGMAX_NEW);

    // Read token from GPU pointer if available, else from config
    int cur_token = config.cur_token_ptr
        ? *config.cur_token_ptr : config.cur_token_id;

{emit_persist_iters(cfg)}\
    // ================================================================
    // Decode step loop
    // ================================================================
    for (int iter = 0; iter < n_decode_iters; iter++) {{
{emit_decode_iter_step(cfg)}\
{emit_persist_prologue(cfg)}\
        unsigned long long _embed_t0 = __builtin_amdgcn_s_memrealtime();

        // -- Embedding: write cur_token embedding into layer 0's residual buffer --
        // Only worker (0,0) writes it, so the other 239 workers MUST wait on a
        // device-wide barrier before reading it in layer 0. __syncthreads() is
        // workgroup-scoped and does not order across workers: without this
        // barrier the other workers read the PREVIOUS token's embedding, which
        // shows up as duplicated tokens in the generated text.
        // SLOT_EMBED_* are absolute offsets into counter_buf (dedicated region
        // past the per-layer blocks), not per-layer offsets.
        int *embed_done  = counter_buf + SLOT_EMBED_DONE;
        int *embed_local = counter_buf + SLOT_EMBED_LOCAL;
        __shared__ int s_embed_expected;
        if (tid == 0) {{
            int cur = __atomic_load_n(embed_done, __ATOMIC_RELAXED);
            s_embed_expected = ((cur / NUM_XCDS) + 1) * NUM_XCDS;
        }}
        __syncthreads();
        {{
            volatile __hip_bfloat16 *d_res =
                (volatile __hip_bfloat16 *)ptr_table[xcd_table_base + 1];
            const __hip_bfloat16 *d_emb =
                (const __hip_bfloat16 *)config.embed_weight;

            if (xcd_id == 0 && xcd_rank == 0) {{
                const __hip_bfloat16 *tok_emb = d_emb +
                    static_cast<int64_t>(cur_token) * ACTUAL_HIDDEN_DIM;
                for (int i = tid; i < ACTUAL_HIDDEN_DIM; i += 256) {{
                    d_res[i] = tok_emb[i];
                }}
                for (int i = ACTUAL_HIDDEN_DIM + tid; i < HIDDEN_SIZE; i += 256) {{
                    d_res[i] = __hip_bfloat16(0);
                }}
            }}
        }}
        __syncthreads();
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        // Ends with buffer_inv, so the embedding is visible to every worker.
        fleet_mk::barrier_global(embed_done, s_embed_expected, TOTAL_WORKERS,
                              embed_local, xcd_id, WORKERS_PER_XCD);

        unsigned long long _embed_t1 = __builtin_amdgcn_s_memrealtime();

        // ================================================================
        // Layer loop (36 layers)
        // ================================================================
        unsigned long long _layer_t0 = __builtin_amdgcn_s_memrealtime();
#ifdef FLEET_MK_ILB_TIMING
        unsigned long long _ilb_accum = 0;
#endif
{emit_touch_lds_pad(cfg)}
        for (int layer = 0; layer < NUM_LAYERS; layer++) {{
            // Load pre-computed mirage pointers directly from ptr_table
            {{
                void **layer_ptrs = &ptr_table[xcd_table_base + layer * PTRS_PER_LAYER];
                if (tid < MIRAGE_IN_COUNT) {{
                    mirage_in[tid] = layer_ptrs[tid];
                }} else if (tid < MIRAGE_IN_COUNT + MIRAGE_OUT_COUNT) {{
                    mirage_out[tid - MIRAGE_IN_COUNT] = layer_ptrs[tid];
                }}
                if (tid == 35) {{
                    s_layer_counters = static_cast<int *>(layer_ptrs[16]);
                }}
            }}
            __syncthreads();
            int *layer_counters = s_layer_counters;

{emit_layer_dispatch(cfg)}
            // Last layer: global barrier + ResAdd before LM head
            if (layer == NUM_LAYERS - 1) {{
                int *layer_done = layer_counters + SLOT_LAYER_DONE_NEW;
                int *layer_local = layer_counters + SLOT_LAYER_LOCAL_NEW;
                __shared__ int s_layer_expected;
                if (tid == 0) {{
                    int cur = __atomic_load_n(layer_done, __ATOMIC_RELAXED);
                    s_layer_expected = ((cur / NUM_XCDS) + 1) * NUM_XCDS;
                }}
                __syncthreads();
                fleet_mk::barrier_global(layer_done, s_layer_expected, TOTAL_WORKERS,
                                      layer_local, xcd_id, WORKERS_PER_XCD);

                // ResAdd: workspace_f32 + oproj_out → layer_output, and zero
                // workspace_f32 for the next iteration.
                // Parallel across all 240 workers is safe here: each worker owns a
                // disjoint column stride, so no worker reads a line another zeroes.
                // (Mirage runs this single-worker, but that serializes 2944 elements
                // into one workgroup and costs ~1.1ms/token here.)
                {{
                    float *ws_f32 = (float *)mirage_out[{mirage_out_idx('moe_workspace_f32')}];
                    const unsigned short *oproj = (const unsigned short *)mirage_out[{mirage_out_idx('attn_proj_out')}];
                    __shared__ void *s_layer_output_ptr;
                    if (tid == 0) {{
                        int last_layer_base = xcd_table_base + (NUM_LAYERS - 1) * PTRS_PER_LAYER;
                        s_layer_output_ptr = ptr_table[last_layer_base + SLOT_LAYER_OUTPUT];
                    }}
                    __syncthreads();
                    unsigned short *layer_out = (unsigned short *)s_layer_output_ptr;
                    int global_rank = xcd_id * WORKERS_PER_XCD + xcd_rank;
                    for (int tok = 0; tok < config.num_active_tokens; tok++) {{
                        for (int col = global_rank * 256 + tid; col < HIDDEN_SIZE;
                             col += TOTAL_WORKERS * 256) {{
#ifdef FLEET_MK_FLEET_HEADERS
                            // Fleet's MoE workspace is per-(token, topk slot), not
                            // per-token: W2's epilogue writes plain st_wt stores to
                            // (b * MOE_WS_SLOTS + slot) * HIDDEN_SIZE instead of
                            // atomicAdd-ing every expert into one slab
                            // (moe_ws_layout.cuh). So the expert contributions arrive
                            // UNSUMMED and this fold is what adds them up.
                            //
                            // Reading slot 0 alone -- which is what the mirage-era
                            // code below does -- is wrong twice over, and neither
                            // failure announces itself:
                            //   * the residual is short by three quarters of the MoE
                            //     output, and
                            //   * slots 1..3 are never zeroed, so last iteration's
                            //     values survive into the next one and compound.
                            // Symptom is a correct first token followed by text that
                            // degenerates over a few dozen tokens -- exactly what the
                            // first fleet-headers run produced.
                            float f = 0.0f;
                            for (int s = 0; s < kernel::MOE_WS_SLOTS; s++) {{
                                int off = kernel::moe_ws_offset(tok, s, HIDDEN_SIZE) + col;
                                f += ws_f32[off];
                                // Zero every slot, not just the one folded: layer 0's
                                // QKV prologue on the NEXT iteration adds this
                                // workspace to the embedding before any MoE has run,
                                // so anything left here is read as real output.
                                ws_f32[off] = 0.0f;
                            }}
#else
                            float f = ws_f32[tok * HIDDEN_SIZE + col];
#endif
                            if (col < ACTUAL_HIDDEN_DIM) {{
                                unsigned bt = (unsigned)oproj[tok * HIDDEN_SIZE + col] << 16;
                                float rv;
                                __builtin_memcpy(&rv, &bt, 4);
                                float sum = f + rv;
                                layer_out[tok * HIDDEN_SIZE + col] = kernel::_gang_float_to_bf16(sum);
                            }}
#ifndef FLEET_MK_FLEET_HEADERS
                            ws_f32[tok * HIDDEN_SIZE + col] = 0.0f;
#endif
                        }}
                    }}
                }}
            }}
            // Inter-layer barrier (skip on last layer — used global barrier above).
            //
            // Under fleet's headers this is DEAD and must not run: fleet's fused
            // layer ends with its own cross-XCD layer barrier
            // (gang_full_layer_fused_mi300.cuh:1318, active because we do not pass
            // MPK_NO_LAYER_BARRIER). Running both is a double rendezvous per layer
            // -- 36 extra barriers per token, and every worker waiting twice.
            // Decision on record: fleet's barrier survives, fleet_mk's is deleted.
            //
            // The LAST-layer global barrier above is NOT redundant and stays: it
            // guards fleet_mk's own parallel ResAdd, which fleet has no equivalent of.
#ifndef FLEET_MK_FLEET_HEADERS
            if (layer < NUM_LAYERS - 1) {{
                int *layer_done = layer_counters + SLOT_LAYER_DONE_NEW;
                int *layer_local = layer_counters + SLOT_LAYER_LOCAL_NEW;
                int *layer_release = layer_counters + SLOT_LAYER_RELEASE_NEW;
#ifdef FLEET_MK_ILB_TIMING
                unsigned long long _ilb0 = __builtin_amdgcn_s_memrealtime();
#endif
                fleet_mk::barrier_wt_release_no_wbl2(layer_done, layer_local,
                                                  layer_release, xcd_id, WORKERS_PER_XCD);
#ifdef FLEET_MK_ILB_TIMING
                // Accumulate instead of printing per layer: 36 layers x 8 XCDs
                // of printf would perturb the very thing being measured.
                _ilb_accum += __builtin_amdgcn_s_memrealtime() - _ilb0;
#endif
            }}
#endif  // !FLEET_MK_FLEET_HEADERS
        }} // end layer loop

        unsigned long long _layer_t1 = __builtin_amdgcn_s_memrealtime();

        // ================================================================
        // Tail: RMSNorm + LM head GEMM + argmax
        // ================================================================
        __shared__ int s_lmhead_expected;
        if (tid == 0) {{
            int cur_lmhead = __atomic_load_n(lmhead_done, __ATOMIC_RELAXED);
            s_lmhead_expected = ((cur_lmhead / TOTAL_WORKERS) + 1) * TOTAL_WORKERS;
        }}
        __syncthreads();

        asm volatile("buffer_inv" ::: "memory");

        // Initialize per-XCD argmax slots
        if (xcd_rank == 0 && tid == 0) {{
            float neg_inf = -1e30f;
            int neg_inf_bits;
            __builtin_memcpy(&neg_inf_bits, &neg_inf, 4);
            unsigned long long init_packed =
                (static_cast<unsigned long long>(0xFFFFFFFFu) << 32)
                | static_cast<unsigned int>(neg_inf_bits);
            reinterpret_cast<unsigned long long *>(&argmax_packed_base[xcd_id * 4])[0] = init_packed;
        }}

        // RMSNorm on last layer output
        {{
            __shared__ void *s_lyr_out;
            if (tid == 0) {{
                int last_layer_base = xcd_table_base + (NUM_LAYERS - 1) * PTRS_PER_LAYER;
                s_lyr_out = ptr_table[last_layer_base + SLOT_LAYER_OUTPUT];
            }}
            __syncthreads();
            fleet_mk::rmsnorm<HIDDEN_SIZE, ACTUAL_HIDDEN_DIM>(
                s_lyr_out,
                config.lm_norm_weight,
                config.lm_norm_scratch);
        }}

        // LM head GEMM with inline argmax
        gpt_oss_120b_lmhead_gemm_argmax(
            config.lm_norm_scratch, config.lm_mxfp4_weight,
            config.lm_bias, config.num_active_tokens,
            argmax_packed_base, xcd_id, xcd_rank,
            (unsigned short *)config.logits_output);

        // LM head global barrier
        fleet_mk::barrier_global_local(lmhead_done, s_lmhead_expected, TOTAL_WORKERS);

        // Cross-XCD argmax reduce (worker 0 on XCD 0)
        if (xcd_id == 0 && xcd_rank == 0 && tid == 0) {{
            float best_val = -1e30f;
            long long best_idx = -1;
            for (int x = 0; x < NUM_XCDS; x++) {{
                unsigned long long packed = reinterpret_cast<unsigned long long *>(
                    &argmax_packed_base[x * 4])[0];
                int val_bits = static_cast<int>(packed & 0xFFFFFFFF);
                float v;
                __builtin_memcpy(&v, &val_bits, 4);
                int idx_bits = static_cast<int>(packed >> 32);
                if (v > best_val) {{
                    best_val = v;
                    best_idx = static_cast<long long>(idx_bits);
                }}
            }}
            static_cast<long long *>(config.argmax_output)[0] = best_idx;
        }}
        __syncthreads();

        // Timing printout -- OPT-IN, build with FLEET_MK_TIMER_PRINT=1.
        //
        // This is a device-side printf, i.e. a HOSTCALL: the wave stalls while a
        // host thread drains the buffer. It costs 0.164 ms/token, measured as
        // 3 interleaved rounds of the same .so built both ways --
        //   printon  med 2.052  spread 0.010
        //   printoff med 1.888  spread 0.014
        // both tok1=35644, coherent text, "delta exceeds noise (improvement)".
        // Reproduce with:
        //   tools/ab_interleave.sh printon <on.so> printoff <off.so> 3
        //   tools/compare_runs.py --arm printon /tmp/printon_run*.log \\
        //                         --arm printoff /tmp/printoff_run*.log
        //
        // It hid for as long as it did because it sits AFTER _tail_t1, so it is
        // invisible to the very timer it reports, and because at one decode step
        // per launch the stall overlaps the launch gap and cancels out of every
        // A/B that keeps it on both arms -- which was every A/B this project has
        // run. It surfaced only inside the persistent loop, where it fires once
        // per decode STEP rather than once per launch, so N of them land in one
        // launch and the gap stops amortising (158/168/156/199 us at N=1/2/4/8
        // instead of ~158/N). The loop found it; the loop does not need it.
        //
        // Fleet does not pay this. Its three ungated printfs are a once-per-run
        // teardown summary and two host-side calls, and its per-iteration
        // [FWD_PASS] dump is gated behind MPK_QUIET_FWDPASS with a comment
        // making the same point (persistent_kernel.cuh:3663). Leaving ours on by
        // default meant every fleet-vs-fleet_mk number handicapped our own side.
        //
        // So: OFF by default, and this is a real latency change, not just an
        // instrumentation default. Turn it on for attribution work --
        // tools/layers_timer_stats.py and the embed/layers/tail split need it --
        // but never quote a latency number from a build that has it on, for the
        // same reason no number is quoted from a FLEET_MK_WORKER_STATE build.
#ifdef FLEET_MK_TIMER_PRINT
        if (xcd_id == 0 && xcd_rank == 0 && tid == 0) {{
            unsigned long long _tail_t1 = __builtin_amdgcn_s_memrealtime();
            double embed_us  = (double)(_embed_t1 - _embed_t0) * 10.0 / 1000.0;
            double layers_us = (double)(_layer_t1 - _layer_t0) * 10.0 / 1000.0;
            double tail_us   = (double)(_tail_t1 - _layer_t1) * 10.0 / 1000.0;
            double total_us  = (double)(_tail_t1 - _embed_t0) * 10.0 / 1000.0;
#ifdef FLEET_MK_ILB_TIMING
            double ilb_us = (double)_ilb_accum * 10.0 / 1000.0;
            printf("[FLEET_MK_TIME] embed=%.1fus layers=%.1fus tail=%.1fus total=%.1fus ilb=%.1fus\\n",
                   embed_us, layers_us, tail_us, total_us, ilb_us);
#else
            printf("[FLEET_MK_TIME] embed=%.1fus layers=%.1fus tail=%.1fus total=%.1fus\\n",
                   embed_us, layers_us, tail_us, total_us);
#endif
#ifdef FLEET_MK_QKV_DRAIN_TIMING
            // Sum over every QKV worker/layer, so report the per-call mean and
            // the per-token total for XCD 0 alone (drains overlap across XCDs).
            unsigned long long dsum = kernel::g_fleet_mk_qkv_drain;
            unsigned long long dcnt = kernel::g_fleet_mk_qkv_drain_cnt;
            if (dcnt) {{
                printf("[FLEET_MK_QKV_DRAIN] calls=%llu mean=%.3fus total_all=%.1fus\\n",
                       dcnt, (double)dsum * 10.0 / 1000.0 / (double)dcnt,
                       (double)dsum * 10.0 / 1000.0);
            }}
            kernel::g_fleet_mk_qkv_drain = 0;
            kernel::g_fleet_mk_qkv_drain_cnt = 0;
#endif
#ifdef FLEET_MK_MOE_W2WAIT_TIMING
            {{
                unsigned long long ws = kernel::g_fleet_mk_w2_wait;
                unsigned long long wc = kernel::g_fleet_mk_w2_wait_cnt;
                if (wc) {{
                    // 184 W2 tiles/layer x 36 layers = 6624 calls/token expected.
                    // mean = time one W2 tile sits blocked on the W13 barrier.
                    printf("[FLEET_MK_W2WAIT] calls=%llu mean=%.3fus\\n",
                           wc, (double)ws * 10.0 / 1000.0 / (double)wc);
                }}
                kernel::g_fleet_mk_w2_wait = 0;
                kernel::g_fleet_mk_w2_wait_cnt = 0;
            }}
#endif
#ifdef FLEET_MK_MOE_SPLIT_TIMING
            {{
                // Per-tile wall time for each MoE phase. Expected counts per
                // token: W13 = 180 real tiles x 36, W2 = 184 x 36. The means
                // are per-tile, so they overlap across the 240 workers — they
                // bound, not sum to, the 21.6us/layer MoE phase.
                unsigned long long a = kernel::g_fleet_mk_moe_w13;
                unsigned long long ac = kernel::g_fleet_mk_moe_w13_cnt;
                unsigned long long b = kernel::g_fleet_mk_moe_w2;
                unsigned long long bc = kernel::g_fleet_mk_moe_w2_cnt;
                unsigned long long d = kernel::g_fleet_mk_moe_w13drain;
                unsigned long long e = kernel::g_fleet_mk_moe_w13drain1;
                unsigned long long q = kernel::g_fleet_mk_moe_w13quant;
                unsigned long long qc = kernel::g_fleet_mk_moe_w13quant_cnt;
                unsigned long long m = kernel::g_fleet_mk_moe_w13mfma;
                unsigned long long pr = kernel::g_fleet_mk_moe_w13pro;
                unsigned long long ep = kernel::g_fleet_mk_moe_w13epi;
                unsigned long long f0 = kernel::g_fleet_mk_moe_w13mf0;
                unsigned long long f1 = kernel::g_fleet_mk_moe_w13mf1;
                unsigned long long md = kernel::g_fleet_mk_moe_w13mid;
                unsigned long long en = kernel::g_fleet_mk_moe_w13ent;
                unsigned long long i0 = kernel::g_fleet_mk_moe_w13iss0;
                unsigned long long i1 = kernel::g_fleet_mk_moe_w13iss1;
                unsigned long long sw = kernel::g_fleet_mk_moe_w13swi0;
                double s = qc ? 10.0 / 1000.0 / (double)qc : 0.0;
                if (ac && bc) {{
                    printf("[FLEET_MK_MOESPLIT] w13 %.3fus = pro %.3f + quant %.3f + drain0 %.3f + mfma %.3f + epi %.3f | mfma = mf0 %.3f + mid %.3f + drain1 %.3f + mf1 %.3f | pro = ent %.3f + iss0 %.3f | mid = iss1 %.3f + swi0 %.3f | w2 %.3fus\\n",
                           (double)a * 10.0 / 1000.0 / (double)ac,
                           (double)pr * s, (double)q * s, (double)d * s,
                           (double)m * s, (double)ep * s,
                           (double)f0 * s, (double)md * s, (double)e * s, (double)f1 * s,
                           (double)en * s, (double)i0 * s,
                           (double)i1 * s, (double)sw * s,
                           (double)b * 10.0 / 1000.0 / (double)bc);
                }}
                kernel::g_fleet_mk_moe_w13ent = 0;
                kernel::g_fleet_mk_moe_w13iss0 = 0;
                kernel::g_fleet_mk_moe_w13iss1 = 0;
                kernel::g_fleet_mk_moe_w13swi0 = 0;
                kernel::g_fleet_mk_moe_w13drain = 0;
                kernel::g_fleet_mk_moe_w13drain1 = 0;
                kernel::g_fleet_mk_moe_w13pro = 0;
                kernel::g_fleet_mk_moe_w13epi = 0;
                kernel::g_fleet_mk_moe_w13mf0 = 0;
                kernel::g_fleet_mk_moe_w13mf1 = 0;
                kernel::g_fleet_mk_moe_w13mid = 0;
                kernel::g_fleet_mk_moe_w13quant = 0;
                kernel::g_fleet_mk_moe_w13quant_cnt = 0;
                kernel::g_fleet_mk_moe_w13mfma = 0;
                kernel::g_fleet_mk_moe_w13 = 0;
                kernel::g_fleet_mk_moe_w13_cnt = 0;
                kernel::g_fleet_mk_moe_w2 = 0;
                kernel::g_fleet_mk_moe_w2_cnt = 0;
            }}
#endif
#ifdef FLEET_MK_OPROJ_SPLIT_TIMING
            {{
                // Sub-phase split of the fused OProj+RMSNorm+Router+TopK task.
                // Expected count: OPROJ_TOTAL_TILES x 36 layers. gemm and rtopk
                // are real work; bar is the Mechanism C wait, i.e. pure loss.
                unsigned long long g = kernel::g_fleet_mk_op_gemm;
                unsigned long long gc = kernel::g_fleet_mk_op_gemm_cnt;
                unsigned long long b = kernel::g_fleet_mk_op_bar;
                unsigned long long bc = kernel::g_fleet_mk_op_bar_cnt;
                unsigned long long r = kernel::g_fleet_mk_op_rtopk;
                unsigned long long rc = kernel::g_fleet_mk_op_rtopk_cnt;
                unsigned long long l = kernel::g_fleet_mk_op_barlast;
                unsigned long long lc = kernel::g_fleet_mk_op_barlast_cnt;
                unsigned long long rr = kernel::g_fleet_mk_op_rr;
                unsigned long long rrc = kernel::g_fleet_mk_op_rr_cnt;
                unsigned long long ri = kernel::g_fleet_mk_op_rridle;
                unsigned long long ric = kernel::g_fleet_mk_op_rridle_cnt;
                if (gc) {{
                    printf("[FLEET_MK_OPSPLIT] calls=%llu | gemm=%.3fus bar=%.3fus rtopk=%.3fus | barlast=%.3fus | rr=%.3fus(n=%llu) rridle=%.3fus(n=%llu)\\n",
                           gc, (double)g * 10.0 / 1000.0 / (double)gc,
                           bc ? (double)b * 10.0 / 1000.0 / (double)bc : 0.0,
                           rc ? (double)r * 10.0 / 1000.0 / (double)rc : 0.0,
                           lc ? (double)l * 10.0 / 1000.0 / (double)lc : 0.0,
                           rrc ? (double)rr * 10.0 / 1000.0 / (double)rrc : 0.0, rrc,
                           ric ? (double)ri * 10.0 / 1000.0 / (double)ric : 0.0, ric);
                }}
                kernel::g_fleet_mk_op_barlast = 0;
                kernel::g_fleet_mk_op_barlast_cnt = 0;
                kernel::g_fleet_mk_op_rr = 0;
                kernel::g_fleet_mk_op_rr_cnt = 0;
                kernel::g_fleet_mk_op_rridle = 0;
                kernel::g_fleet_mk_op_rridle_cnt = 0;
                kernel::g_fleet_mk_op_gemm = 0;
                kernel::g_fleet_mk_op_gemm_cnt = 0;
                kernel::g_fleet_mk_op_bar = 0;
                kernel::g_fleet_mk_op_bar_cnt = 0;
                kernel::g_fleet_mk_op_rtopk = 0;
                kernel::g_fleet_mk_op_rtopk_cnt = 0;
            }}
#endif
        }}
#endif  // FLEET_MK_TIMER_PRINT

        // Zero MoE barrier for next iteration (saves external memset)
        {{
            int *moe_bar = static_cast<int *>(ptr_table[xcd_table_base + 21]);
            int global_rank = xcd_id * WORKERS_PER_XCD + xcd_rank;
            for (int i = global_rank * 256 + tid; i < MOE_BARRIER_INTS;
                 i += TOTAL_WORKERS * 256) {{
                moe_bar[i] = 0;
            }}
        }}
{emit_persist_epilogue(cfg)}\
    }} // end decode step loop
}}
'''


def generate_kernel_dense(cfg: ModelConfig) -> str:
    """Dense (Qwen3, Llama3) pipeline: 7 barriers/layer, built entirely from
    device_functions.cuh. Round-trips byte-identically against
    generated/qwen3_8b_kernel.cuh and generated/llama3_8b_kernel.cuh -- see
    check_roundtrip.py. Do not edit without re-running it.
    """
    nc = cfg.name_clean
    ns = nc  # namespace
    nt = cfg.name_title
    parts = []

    # ── Header ──
    parts.append(f'''\
/* Auto-generated by fleet_mk_generate.py
 * Fleet MK: Persistent kernel for {cfg.name} (Dense) using device function library
 *
 * Built ENTIRELY from device_functions.cuh -- no mirage fused kernel calls.
 * Demonstrates two-level fusion with composable device functions:
 *   Level 1 (CODA-style): Epilogues fuse element-wise ops into GEMM accumulators
 *   Level 2 (Persistent):  All phases share data via HBM within one persistent kernel
 *
 * Dense layer structure:
 *   Phase 1: RMSNorm -> QKV GEMM (EpilogueStore) -> barrier -> RoPE + KV update
 *   Phase 2: Attention (CK FMHA hd={cfg.head_dim}, split-KV) + merge
 *   Phase 3: O-proj GEMM (EpilogueResAdd with layer input) -> barrier -> RMSNorm
 *   Phase 4: GateUp GEMM (EpilogueSwiGLU) -> barrier -> Down GEMM (EpilogueResAdd)
 *   Tail:    RMSNorm -> LM head GEMM + argmax
 */
#pragma once

#include "common.cuh"

// Device function library (composable epilogues + GEMM mainloop + barriers)
#include "device_functions.cuh"

// Mirage sub-kernels: CK FMHA attention + split-KV merge
#include "tasks/common/utils.cuh"
#ifndef NUM_THREADS
#define NUM_THREADS 256
#endif
#include "tasks/ampere/merge_splitkv.cuh"
#include "tasks/mi300/paged_attention_ck_fmha_split_kv_mi300.cuh"
''')

    # ── Constants namespace ──
    parts.append(f'''\
// ============================================================================
// {cfg.name} Architecture Constants
// ============================================================================
namespace {ns} {{

// Model dimensions
static constexpr int NUM_LAYERS = {cfg.num_layers};
static constexpr int HIDDEN_SIZE = {cfg.padded_hidden_size};
static constexpr int ACTUAL_HIDDEN_DIM = {cfg.hidden_size};  // for RMSNorm mean
static constexpr int INTERMEDIATE_SIZE = {cfg.padded_intermediate_size};
static constexpr int VOCAB_SIZE = {cfg.vocab_size};
static constexpr int PADDED_VOCAB_SIZE = {cfg.padded_vocab_size};  // next multiple of 256

// Attention
static constexpr int NUM_Q_HEADS = {cfg.num_q_heads};
static constexpr int NUM_KV_HEADS = {cfg.num_kv_heads};
static constexpr int HEAD_DIM = {cfg.head_dim};
static constexpr int NUM_Q_PER_KV = {cfg.q_per_kv};  // {cfg.num_q_heads} / {cfg.num_kv_heads}

// QKV GEMM output: Q({cfg.num_q_heads * cfg.head_dim}) + K({cfg.num_kv_heads * cfg.head_dim}) + V({cfg.num_kv_heads * cfg.head_dim}) = {cfg.qkv_output_size}
static constexpr int QKV_OUTPUT_SIZE = NUM_Q_HEADS * HEAD_DIM
                                     + 2 * NUM_KV_HEADS * HEAD_DIM;  // {cfg.qkv_output_size}

// GPU layout
static constexpr int NUM_XCDS = {cfg.num_xcds};
static constexpr int WORKERS_PER_XCD = {cfg.workers_per_xcd};
static constexpr int TOTAL_WORKERS = NUM_XCDS * WORKERS_PER_XCD;

// All GEMMs use OPW={cfg.output_per_wg}
static constexpr int OUTPUT_PER_WG = {cfg.output_per_wg};

// QKV GEMM
static constexpr int QKV_N_WGS = QKV_OUTPUT_SIZE / OUTPUT_PER_WG;     // {cfg.qkv_n_wgs}
static constexpr int QKV_N_WGS_PER_XCD = QKV_N_WGS / NUM_XCDS;       // {cfg.qkv_n_wgs_per_xcd}

// O-proj GEMM (reduction = num_q_heads * head_dim = {cfg.oproj_reduction})
static constexpr int OPROJ_REDUCTION = NUM_Q_HEADS * HEAD_DIM;         // {cfg.oproj_reduction}
static constexpr int OPROJ_OPW = {cfg.oproj_opw};
static constexpr int OPROJ_N_WGS = HIDDEN_SIZE / OPROJ_OPW;            // {cfg.oproj_n_wgs}
static constexpr int OPROJ_N_WGS_PER_XCD = OPROJ_N_WGS / NUM_XCDS;   // {cfg.oproj_n_wgs_per_xcd}

// GateUp GEMM with fused SwiGLU (OPW={cfg.gateup_opw})
static constexpr int GATEUP_OPW = {cfg.gateup_opw};
static constexpr int GATEUP_OUTPUT_SIZE = 2 * INTERMEDIATE_SIZE;       // {cfg.gateup_output_size}
static constexpr int GATEUP_N_WGS = GATEUP_OUTPUT_SIZE / GATEUP_OPW;  // {cfg.gateup_n_wgs}
static constexpr int GATEUP_N_WGS_PER_XCD = GATEUP_N_WGS / NUM_XCDS; // {cfg.gateup_n_wgs_per_xcd}

// Down GEMM (reduction = intermediate_size)
static constexpr int DOWN_N_WGS = HIDDEN_SIZE / OUTPUT_PER_WG;        // {cfg.down_n_wgs}
static constexpr int DOWN_N_WGS_PER_XCD = DOWN_N_WGS / NUM_XCDS;     // {cfg.down_n_wgs_per_xcd}

// LM head
static constexpr int LM_N_WGS = PADDED_VOCAB_SIZE / OUTPUT_PER_WG;    // {cfg.lm_n_wgs}
static constexpr int LM_N_WGS_PER_XCD = LM_N_WGS / NUM_XCDS;         // {cfg.lm_n_wgs_per_xcd}

// MXFP4 weight byte sizes per workgroup
static constexpr int QKV_WG_BYTES = OUTPUT_PER_WG * (HIDDEN_SIZE / 2 + HIDDEN_SIZE / 32);
static constexpr int OPROJ_WG_BYTES = OPROJ_OPW * (OPROJ_REDUCTION / 2 + OPROJ_REDUCTION / 32);
static constexpr int GATEUP_WG_BYTES = GATEUP_OPW * (HIDDEN_SIZE / 2 + HIDDEN_SIZE / 32);
static constexpr int DOWN_WG_BYTES = OUTPUT_PER_WG * (INTERMEDIATE_SIZE / 2 + INTERMEDIATE_SIZE / 32);
static constexpr int LM_WG_BYTES = OUTPUT_PER_WG * (HIDDEN_SIZE / 2 + HIDDEN_SIZE / 32);

// Attention
static constexpr int PAGE_SIZE = {cfg.page_size};
static constexpr int NUM_KV_CHUNKS = {cfg.num_kv_chunks};
static constexpr int MAX_SEQ_LEN = {cfg.max_seq_len};
static constexpr int KV_CACHE_STRIDE = NUM_KV_HEADS * HEAD_DIM;       // {cfg.kv_cache_stride}
static constexpr int Q_WORKSPACE_STRIDE = NUM_Q_PER_KV * HEAD_DIM;    // {cfg.q_workspace_stride}

// Pointer table layout
static constexpr int PTRS_IN = {cfg.ptrs_in};
static constexpr int PTRS_OUT = {cfg.ptrs_out};
static constexpr int PTRS_PER_LAYER = PTRS_IN + PTRS_OUT;

}} // namespace {ns}
''')

    # ── Pointer namespace ──
    qk_norm_entries = ""
    if cfg.has_qk_norm:
        qk_norm_entries = f"""\
    static constexpr int Q_NORM_WEIGHT     = 16;  // [HEAD_DIM] bf16 (per-head QK norm)
    static constexpr int K_NORM_WEIGHT     = 17;  // [HEAD_DIM] bf16 (per-head QK norm)
"""
    parts.append(f'''\
// ============================================================================
// Pointer table indices (Dense)
// ============================================================================
namespace {ns}_ptr {{
    // Input pointers
    static constexpr int RESIDUAL          = 0;   // [bs, hidden] bf16 (layer input)
    static constexpr int NORM_W1           = 1;   // [hidden] bf16 (pre-attn RMSNorm weight)
    static constexpr int NORM_SCRATCH1     = 2;   // [bs, hidden] bf16 (pre-attn norm output)
    static constexpr int QKV_WEIGHT        = 3;   // MXFP4 packed (per-XCD, OPW={cfg.output_per_wg})
    static constexpr int QKV_BIAS          = 4;   // [qkv_output] bf16
    static constexpr int LSE_ACC           = 5;   // [bs, NUM_KV_CHUNKS] f32
    static constexpr int O_ACC_F32         = 6;   // [bs, H, NUM_KV_CHUNKS] f32
    static constexpr int OPROJ_WEIGHT      = 7;   // MXFP4 packed (per-XCD)
    static constexpr int OPROJ_BIAS        = 8;   // [hidden] bf16
    static constexpr int NORM_W2           = 9;   // [hidden] bf16 (pre-FFN RMSNorm weight)
    static constexpr int NORM_SCRATCH2     = 10;  // [bs, hidden] bf16 (pre-FFN norm output)
    static constexpr int GATEUP_WEIGHT     = 11;  // MXFP4 packed (per-XCD)
    static constexpr int GATEUP_BIAS       = 12;  // [2*intermediate] bf16
    static constexpr int DOWN_WEIGHT       = 13;  // MXFP4 packed (per-XCD)
    static constexpr int DOWN_BIAS         = 14;  // [hidden] bf16
    static constexpr int COUNTER_BUF       = 15;  // per-layer counter buffer
{qk_norm_entries}
    // Output pointers
    static constexpr int QKV_OUTPUT        = 0;   // [bs, {cfg.qkv_output_size}] bf16 (raw QKV GEMM output)
    static constexpr int K_CACHE           = 1;   // paged K cache
    static constexpr int V_CACHE           = 2;   // paged V cache
    static constexpr int Q_WORKSPACE       = 3;   // [bs, num_q_heads * head_dim] bf16
    static constexpr int ATTN_OUT          = 4;   // [bs, num_q_heads * head_dim] bf16
    static constexpr int OPROJ_OUT         = 5;   // [bs, hidden] bf16 (O-proj + ResAdd)
    static constexpr int GATEUP_SCRATCH    = 6;   // [bs, {cfg.gateup_output_size}] bf16
    static constexpr int SWIGLU_OUT        = 7;   // [bs, {cfg.padded_intermediate_size}] bf16
    static constexpr int LAYER_OUTPUT      = 8;   // [bs, hidden] bf16 (Down + ResAdd = next layer input)
}} // namespace {ns}_ptr
''')

    # ── Config struct ──
    parts.append(f'''\

// ============================================================================
// {cfg.name} Runtime Config
// ============================================================================
struct {nt}Config {{
    int num_active_tokens;
    float attn_scale;           // 1/sqrt(head_dim) * log2(e)

    // RoPE tables
    void const *cos_ptr;
    void const *sin_ptr;

    // Paged attention
    int const *qo_indptr;
    int const *kv_indptr;
    int const *kv_indices;
    int const *kv_last_page_len;

    // LM head tail
    void const *lm_norm_weight;
    void       *lm_norm_scratch;
    void const *lm_mxfp4_weight;
    void const *lm_bias;
    void       *argmax_output;

    // Subphase timing (optional, enabled via FLEET_MK_SUBPHASE_TIMING)
    unsigned long long *timing_buf;  // [NUM_LAYERS * TIMING_SLOTS_PER_LAYER] u64
}};

// Timing: 12 timestamps per layer + 4 for tail
// 0=layer_start, 1=after_rmsnorm1, 2=after_qkv_barrier, 3=after_rope,
// 4=after_attn, 5=after_oproj_barrier, 6=after_oproj_gemm,
// 7=after_ffn_barrier, 8=after_gateup, 9=after_gateup_barrier,
// 10=after_down, 11=after_layer_barrier
static constexpr int TIMING_SLOTS_PER_LAYER = 12;
static constexpr int TIMING_TAIL_SLOTS = 4;  // tail rmsnorm, lmhead, argmax, end

#ifdef FLEET_MK_SUBPHASE_TIMING
#define FLEET_MK_TIMESTAMP(buf, slot) \\
    if (xcd_id == 0 && xcd_rank == 0 && tid == 0) {{ \\
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory"); \\
        (buf)[(slot)] = get_gpu_time(); \\
    }}
#else
#define FLEET_MK_TIMESTAMP(buf, slot) ((void)0)
#endif
''')

    # ── Kernel function ──
    # RoPE call depends on has_qk_norm
    if cfg.has_qk_norm:
        rope_call = f"""\
            fleet_mk::rope_kv_update<HEAD_DIM, NUM_Q_PER_KV, NUM_KV_HEADS, PAGE_SIZE, true>(
                (const unsigned short *)s_output_ptrs[ptr::QKV_OUTPUT],
                (const unsigned short *)config.cos_ptr,
                (const unsigned short *)config.sin_ptr,
                (unsigned short *)s_output_ptrs[ptr::Q_WORKSPACE],
                (unsigned short *)s_output_ptrs[ptr::K_CACHE],
                (unsigned short *)s_output_ptrs[ptr::V_CACHE],
                config.qo_indptr, config.kv_indptr,
                config.kv_indices, config.kv_last_page_len,
                xcd_id,  // kv_head_idx = xcd_id
                KV_CACHE_STRIDE,
                Q_WORKSPACE_STRIDE,
                (const unsigned short *)s_input_ptrs[ptr::Q_NORM_WEIGHT],
                (const unsigned short *)s_input_ptrs[ptr::K_NORM_WEIGHT]);"""
    else:
        rope_call = f"""\
            fleet_mk::rope_kv_update<HEAD_DIM, NUM_Q_PER_KV, NUM_KV_HEADS, PAGE_SIZE, false>(
                (const unsigned short *)s_output_ptrs[ptr::QKV_OUTPUT],
                (const unsigned short *)config.cos_ptr,
                (const unsigned short *)config.sin_ptr,
                (unsigned short *)s_output_ptrs[ptr::Q_WORKSPACE],
                (unsigned short *)s_output_ptrs[ptr::K_CACHE],
                (unsigned short *)s_output_ptrs[ptr::V_CACHE],
                config.qo_indptr, config.kv_indptr,
                config.kv_indices, config.kv_last_page_len,
                xcd_id,  // kv_head_idx = xcd_id
                KV_CACHE_STRIDE,
                Q_WORKSPACE_STRIDE,
                nullptr,
                nullptr);"""

    parts.append(f'''\

// ============================================================================
// The {cfg.name} persistent kernel -- built entirely from device functions
// ============================================================================
__global__ void __launch_bounds__(256)
{nc}_kernel(
    {nt}Config config,
    void **ptr_table,       // [NUM_XCDS * NUM_LAYERS * PTRS_PER_LAYER]
    int  *counter_buf)
{{
    using namespace {ns};
    namespace ptr = {ns}_ptr;

    // -- Worker identification --
    int xcd_id = get_xcd_id();
    int tid = threadIdx.x;

    // XCD-local rank via atomic counter
    int *rank_counters = counter_buf + NUM_LAYERS * fleet_mk::COUNTERS_PER_LAYER;

    __shared__ int s_xcd_rank;
    if (tid == 0) {{
        s_xcd_rank = atomicAdd(&rank_counters[xcd_id * 16], 1);
    }}
    __syncthreads();
    int xcd_rank = s_xcd_rank;

    // -- Per-layer pointer table in shared memory --
    __shared__ void *s_input_ptrs[PTRS_IN];
    __shared__ void *s_output_ptrs[PTRS_OUT];

    // Invalidate L2 at kernel entry
    asm volatile("buffer_inv" ::: "memory");
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");

    // Pointer table base for this XCD
    int xcd_table_base = xcd_id * NUM_LAYERS * PTRS_PER_LAYER;

    // ================================================================
    // Layer loop
    // ================================================================
#ifdef FLEET_MK_SINGLE_LAYER_DEBUG
    for (int layer = 0; layer < FLEET_MK_SINGLE_LAYER_DEBUG; layer++) {{
#else
    for (int layer = 0; layer < NUM_LAYERS; layer++) {{
#endif

        // Cooperative pointer loading
        int layer_base = xcd_table_base + layer * PTRS_PER_LAYER;
        for (int i = tid; i < PTRS_PER_LAYER; i += 256) {{
            if (i < PTRS_IN) {{
                s_input_ptrs[i] = ptr_table[layer_base + i];
            }} else {{
                s_output_ptrs[i - PTRS_IN] = ptr_table[layer_base + i];
            }}
        }}
        __syncthreads();

        // Counter buffer for this layer
        int *layer_counters = static_cast<int *>(s_input_ptrs[ptr::COUNTER_BUF]);

#ifdef FLEET_MK_SUBPHASE_TIMING
        unsigned long long *layer_timing = config.timing_buf + layer * TIMING_SLOTS_PER_LAYER;
#endif
        FLEET_MK_TIMESTAMP(layer_timing, 0);  // layer_start

        // ────────────────────────────────────────────────────────────────
        // Phase 1: RMSNorm -> QKV GEMM -> RoPE + KV update
        // ────────────────────────────────────────────────────────────────

        // Phase 1a: RMSNorm (all workers redundantly compute)
        fleet_mk::rmsnorm<HIDDEN_SIZE, ACTUAL_HIDDEN_DIM>(
            s_input_ptrs[ptr::RESIDUAL],
            s_input_ptrs[ptr::NORM_W1],
            s_input_ptrs[ptr::NORM_SCRATCH1]);

        FLEET_MK_TIMESTAMP(layer_timing, 1);  // after_rmsnorm1

        // Phase 1b: QKV GEMM with EpilogueStore (OPW={cfg.output_per_wg})
        {{
            int qkv_col_offset = xcd_id * QKV_N_WGS_PER_XCD * OUTPUT_PER_WG;
            unsigned short *qkv_out_base = (unsigned short *)s_output_ptrs[ptr::QKV_OUTPUT]
                + qkv_col_offset;
            const unsigned short *qkv_bias_base = (const unsigned short *)s_input_ptrs[ptr::QKV_BIAS]
                + qkv_col_offset;

            int qkv_total_tiles = QKV_N_WGS_PER_XCD * config.num_active_tokens;
            for (int t = xcd_rank; t < qkv_total_tiles; t += WORKERS_PER_XCD) {{
                int wg_idx = t % QKV_N_WGS_PER_XCD;

                fleet_mk::EpilogueStore store_ep;
                store_ep.output = qkv_out_base;
                store_ep.output_stride = QKV_OUTPUT_SIZE;
                store_ep.output_size = QKV_N_WGS_PER_XCD * OUTPUT_PER_WG;  // local size

                fleet_mk::gemm_mxfp4<fleet_mk::EpilogueStore, 1, HIDDEN_SIZE, OUTPUT_PER_WG>(
                    s_input_ptrs[ptr::NORM_SCRATCH1],  // input: norm output
                    s_input_ptrs[ptr::QKV_WEIGHT],
                    s_output_ptrs[ptr::QKV_OUTPUT],
                    config.num_active_tokens,
                    QKV_N_WGS_PER_XCD,
                    wg_idx,
                    (void *)qkv_bias_base,
                    QKV_OUTPUT_SIZE,
                    QKV_OUTPUT_SIZE,
                    store_ep);
            }}
        }}

        // QKV barrier (local)
        {{
            int *qkv_done = layer_counters + fleet_mk::SLOT_QKV_DONE;
            __shared__ int s_qkv_expected;
            if (tid == 0) {{
                int cur = __atomic_load_n(qkv_done, __ATOMIC_RELAXED);
                s_qkv_expected = ((cur / TOTAL_WORKERS) + 1) * TOTAL_WORKERS;
            }}
            __syncthreads();
            fleet_mk::barrier_global_local(qkv_done, s_qkv_expected, TOTAL_WORKERS);
        }}

        FLEET_MK_TIMESTAMP(layer_timing, 2);  // after_qkv_barrier

        // Phase 1c: RoPE + KV cache update (one worker per KV head)
        if (xcd_rank == 0) {{
{rope_call}
        }}

        FLEET_MK_TIMESTAMP(layer_timing, 3);  // after_rope (no barrier)

        // ────────────────────────────────────────────────────────────────
        // Phase 2: Attention (scalar decode, no split-KV merge needed)
        // ────────────────────────────────────────────────────────────────
        {{
            if (xcd_rank == 0) {{
                int kv_head_idx = xcd_id;  // one KV head per XCD

                using bf16_t = __hip_bfloat16;
                const void *offset_k = reinterpret_cast<const bf16_t *>(s_output_ptrs[ptr::K_CACHE])
                    + static_cast<size_t>(xcd_id) * HEAD_DIM;
                const void *offset_v = reinterpret_cast<const bf16_t *>(s_output_ptrs[ptr::V_CACHE])
                    + static_cast<size_t>(xcd_id) * HEAD_DIM;

                kernel::paged_attention_ck_fmha_decode<
                    kernel::bfloat16, NUM_Q_PER_KV, HEAD_DIM, PAGE_SIZE,
                    MAX_SEQ_LEN, NUM_KV_CHUNKS, Q_WORKSPACE_STRIDE,
                    KV_CACHE_STRIDE, NUM_KV_HEADS>(
                    s_output_ptrs[ptr::Q_WORKSPACE],
                    const_cast<void *>(offset_k),
                    const_cast<void *>(offset_v),
                    s_output_ptrs[ptr::ATTN_OUT],    // write bf16 directly here
                    s_input_ptrs[ptr::LSE_ACC],       // LSE scratch (unused by merge)
                    config.qo_indptr, config.kv_indptr,
                    config.kv_indices, config.kv_last_page_len,
                    /*request_id=*/(int16_t)0,
                    kv_head_idx,
                    /*kv_chunk_idx=*/0,
                    config.attn_scale,
                    /*sliding_window=*/0);

                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            }}

            FLEET_MK_TIMESTAMP(layer_timing, 4);  // after_attn_compute (only xcd_rank==0)

            // Attention barrier (selective wbl2)
            {{
                int *oproj_done = layer_counters + fleet_mk::SLOT_OPROJ_DONE;
                __shared__ int s_oproj_expected;
                if (tid == 0) {{
                    int cur = __atomic_load_n(oproj_done, __ATOMIC_RELAXED);
                    s_oproj_expected = ((cur / TOTAL_WORKERS) + 1) * TOTAL_WORKERS;
                }}
                __syncthreads();
                fleet_mk::barrier_global_selective_wbl2(
                    oproj_done, s_oproj_expected, TOTAL_WORKERS,
                    /*is_writer=*/xcd_rank == 0);
            }}
        }}

        FLEET_MK_TIMESTAMP(layer_timing, 5);  // after_oproj_barrier

        // ────────────────────────────────────────────────────────────────
        // Phase 3: O-proj GEMM + ResAdd + RMSNorm
        // ────────────────────────────────────────────────────────────────
        {{
            int oproj_col_offset = xcd_id * OPROJ_N_WGS_PER_XCD * OPROJ_OPW;

            int oproj_total_tiles = OPROJ_N_WGS_PER_XCD * config.num_active_tokens;
            for (int t = xcd_rank; t < oproj_total_tiles; t += WORKERS_PER_XCD) {{
                int wg_idx = t % OPROJ_N_WGS_PER_XCD;

                fleet_mk::EpilogueResAdd resadd_ep;
                resadd_ep.output = (unsigned short *)s_output_ptrs[ptr::OPROJ_OUT]
                    + oproj_col_offset;
                resadd_ep.residual = (const unsigned short *)s_input_ptrs[ptr::RESIDUAL]
                    + oproj_col_offset;
                resadd_ep.output_stride = HIDDEN_SIZE;
                resadd_ep.output_size = OPROJ_N_WGS_PER_XCD * OPROJ_OPW;

                fleet_mk::gemm_mxfp4<fleet_mk::EpilogueResAdd, 1, OPROJ_REDUCTION, OPROJ_OPW>(
                    s_output_ptrs[ptr::ATTN_OUT],    // input: attention output
                    s_input_ptrs[ptr::OPROJ_WEIGHT],
                    s_output_ptrs[ptr::OPROJ_OUT],
                    config.num_active_tokens,
                    OPROJ_N_WGS_PER_XCD,
                    wg_idx,
                    (void *)((const unsigned short *)s_input_ptrs[ptr::OPROJ_BIAS]
                        + oproj_col_offset),
                    HIDDEN_SIZE,
                    HIDDEN_SIZE,
                    resadd_ep);
            }}

            FLEET_MK_TIMESTAMP(layer_timing, 6);  // after_oproj_gemm

            // O-proj hierarchical barrier
            {{
                int *ffn_mid_done = layer_counters + fleet_mk::SLOT_FFN_MID_DONE;
                int *ffn_local = layer_counters + fleet_mk::SLOT_FFN_LOCAL;
                __shared__ int s_ffn_mid_expected;
                if (tid == 0) {{
                    int cur = __atomic_load_n(ffn_mid_done, __ATOMIC_RELAXED);
                    s_ffn_mid_expected = ((cur / NUM_XCDS) + 1) * NUM_XCDS;
                }}
                __syncthreads();
                fleet_mk::barrier_global(ffn_mid_done, s_ffn_mid_expected, TOTAL_WORKERS,
                                      ffn_local, xcd_id, WORKERS_PER_XCD);
            }}

            FLEET_MK_TIMESTAMP(layer_timing, 7);  // after_ffn_mid_barrier

            // RMSNorm on O-proj output (all workers redundantly)
            fleet_mk::rmsnorm<HIDDEN_SIZE, ACTUAL_HIDDEN_DIM>(
                s_output_ptrs[ptr::OPROJ_OUT],
                s_input_ptrs[ptr::NORM_W2],
                s_input_ptrs[ptr::NORM_SCRATCH2]);
        }}

{'        // ────────────────────────────────────────────────────────────────' + chr(10) + '        // Phase 4: Fused GateUp+SwiGLU GEMM -> Down GEMM + ResAdd' + chr(10) + '        // ────────────────────────────────────────────────────────────────' + chr(10) + """        {{
            // Step 4a: Fused GateUp+SwiGLU GEMM (OPW={gateup_opw}, SwiGLU in epilogue)
            {{
                int swiglu_col_offset = xcd_id * GATEUP_N_WGS_PER_XCD * (GATEUP_OPW / 2);
                int gateup_bias_offset = xcd_id * GATEUP_N_WGS_PER_XCD * GATEUP_OPW;

                int gateup_total_tiles = GATEUP_N_WGS_PER_XCD * config.num_active_tokens;
                for (int t = xcd_rank; t < gateup_total_tiles; t += WORKERS_PER_XCD) {{
                    int wg_idx = t % GATEUP_N_WGS_PER_XCD;

                    fleet_mk::EpilogueSwiGLU swiglu_ep;
                    swiglu_ep.output = (unsigned short *)s_output_ptrs[ptr::SWIGLU_OUT]
                        + swiglu_col_offset;
                    swiglu_ep.output_stride = INTERMEDIATE_SIZE;
                    swiglu_ep.output_size = GATEUP_N_WGS_PER_XCD * (GATEUP_OPW / 2);

                    fleet_mk::gemm_mxfp4<fleet_mk::EpilogueSwiGLU, 1, HIDDEN_SIZE, GATEUP_OPW>(
                        s_input_ptrs[ptr::NORM_SCRATCH2],  // input: pre-FFN norm output
                        s_input_ptrs[ptr::GATEUP_WEIGHT],
                        s_output_ptrs[ptr::SWIGLU_OUT],    // output buffer
                        config.num_active_tokens,
                        GATEUP_N_WGS_PER_XCD,
                        wg_idx,
                        (void *)((const unsigned short *)s_input_ptrs[ptr::GATEUP_BIAS]
                            + gateup_bias_offset),
                        INTERMEDIATE_SIZE,
                        INTERMEDIATE_SIZE,
                        swiglu_ep);
                }}
            }}

            FLEET_MK_TIMESTAMP(layer_timing, 8);  // after_gateup_gemm

            // GateUp->Down hierarchical barrier
            {{
                int *down_done = layer_counters + fleet_mk::SLOT_DOWN_DONE;
                int *down_local = layer_counters + fleet_mk::SLOT_DOWN_LOCAL;
                __shared__ int s_down_expected;
                if (tid == 0) {{
                    int cur = __atomic_load_n(down_done, __ATOMIC_RELAXED);
                    s_down_expected = ((cur / NUM_XCDS) + 1) * NUM_XCDS;
                }}
                __syncthreads();
                fleet_mk::barrier_global(down_done, s_down_expected, TOTAL_WORKERS,
                                      down_local, xcd_id, WORKERS_PER_XCD);
            }}

            FLEET_MK_TIMESTAMP(layer_timing, 9);  // after_gateup_barrier

            // Step 4b: Down GEMM with ResAdd
            {{
                int down_col_offset = xcd_id * DOWN_N_WGS_PER_XCD * OUTPUT_PER_WG;

                int down_total_tiles = DOWN_N_WGS_PER_XCD * config.num_active_tokens;
                for (int t = xcd_rank; t < down_total_tiles; t += WORKERS_PER_XCD) {{
                    int wg_idx = t % DOWN_N_WGS_PER_XCD;

                    fleet_mk::EpilogueResAdd resadd_ep;
                    resadd_ep.output = (unsigned short *)s_output_ptrs[ptr::LAYER_OUTPUT]
                        + down_col_offset;
                    resadd_ep.residual = (const unsigned short *)s_output_ptrs[ptr::OPROJ_OUT]
                        + down_col_offset;
                    resadd_ep.output_stride = HIDDEN_SIZE;
                    resadd_ep.output_size = DOWN_N_WGS_PER_XCD * OUTPUT_PER_WG;

                    fleet_mk::gemm_mxfp4<fleet_mk::EpilogueResAdd, 1, INTERMEDIATE_SIZE, OUTPUT_PER_WG>(
                        s_output_ptrs[ptr::SWIGLU_OUT],
                        s_input_ptrs[ptr::DOWN_WEIGHT],
                        s_output_ptrs[ptr::LAYER_OUTPUT],
                        config.num_active_tokens,
                        DOWN_N_WGS_PER_XCD,
                        wg_idx,
                        (void *)((const unsigned short *)s_input_ptrs[ptr::DOWN_BIAS]
                            + down_col_offset),
                        HIDDEN_SIZE,
                        HIDDEN_SIZE,
                        resadd_ep);
                }}
            }}
        }}

        FLEET_MK_TIMESTAMP(layer_timing, 10);  // after_down_gemm""".format(gateup_opw=cfg.gateup_opw)}

        // -- End-of-layer hierarchical barrier --
        {{
            int *layer_done = layer_counters + fleet_mk::SLOT_LAYER_DONE;
            int *layer_local = layer_counters + fleet_mk::SLOT_LAYER_LOCAL;
            __shared__ int s_layer_expected;
            if (tid == 0) {{
                int cur = __atomic_load_n(layer_done, __ATOMIC_RELAXED);
                s_layer_expected = ((cur / NUM_XCDS) + 1) * NUM_XCDS;
            }}
            __syncthreads();
            fleet_mk::barrier_global(layer_done, s_layer_expected, TOTAL_WORKERS,
                                  layer_local, xcd_id, WORKERS_PER_XCD);
        }}
        FLEET_MK_TIMESTAMP(layer_timing, 11);  // after_layer_barrier
    }} // end layer loop

    // ================================================================
    // Fused tail: RMSNorm + LM head GEMM + argmax
    // ================================================================

    // Use tail counter slots (after last layer)
    int *tail_counters = counter_buf + (NUM_LAYERS - 1) * fleet_mk::COUNTERS_PER_LAYER;
    int *lmhead_done   = tail_counters + fleet_mk::SLOT_TAIL_LMHEAD;
    float *argmax_packed_base = reinterpret_cast<float *>(
        tail_counters + fleet_mk::SLOT_TAIL_ARGMAX);

    __shared__ int s_lmhead_expected;
    if (tid == 0) {{
        int cur_lmhead = __atomic_load_n(lmhead_done, __ATOMIC_RELAXED);
        s_lmhead_expected = ((cur_lmhead / TOTAL_WORKERS) + 1) * TOTAL_WORKERS;
    }}
    __syncthreads();

    // Wait for last layer to complete
    {{
        int *tail_sync = tail_counters + fleet_mk::SLOT_QKV_DONE;
        __shared__ int s_tail_expected;
        if (tid == 0) {{
            int cur = __atomic_load_n(tail_sync, __ATOMIC_RELAXED);
            s_tail_expected = ((cur / TOTAL_WORKERS) + 1) * TOTAL_WORKERS;
        }}
        __syncthreads();
        // LAYER_DONE barrier already flushed L2 for last layer output
        fleet_mk::barrier_global_local(tail_sync, s_tail_expected, TOTAL_WORKERS);
    }}

    // Initialize per-XCD argmax slots
    if (xcd_rank == 0 && tid == 0) {{
        float neg_inf = -1e30f;
        int neg_inf_bits;
        __builtin_memcpy(&neg_inf_bits, &neg_inf, 4);
        unsigned long long init_packed =
            (static_cast<unsigned long long>(0xFFFFFFFFu) << 32)
            | static_cast<unsigned int>(neg_inf_bits);
        reinterpret_cast<unsigned long long *>(&argmax_packed_base[xcd_id * 4])[0] = init_packed;
    }}

    // RMSNorm on last layer output
    fleet_mk::rmsnorm<HIDDEN_SIZE, ACTUAL_HIDDEN_DIM>(
        s_output_ptrs[ptr::LAYER_OUTPUT],
        config.lm_norm_weight,
        config.lm_norm_scratch);

    // LM head GEMM with inline argmax
    {{
        constexpr int LM_NUM_BLOCKS_32 = HIDDEN_SIZE / 32;
        constexpr int LM_WG_DATA_BYTES = OUTPUT_PER_WG * (HIDDEN_SIZE / 2);
        constexpr int LM_WG_SCALE_BYTES = OUTPUT_PER_WG * LM_NUM_BLOCKS_32;
        constexpr int LM_WG_BYTES_TOTAL = LM_WG_DATA_BYTES + LM_WG_SCALE_BYTES;
        constexpr int K_PER_MFMA = 128;
        constexpr int MFMA_ITERS = HIDDEN_SIZE / K_PER_MFMA;
        constexpr int NUM_WAVES = 4;
        constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;

        extern __shared__ char _lm_smem[];
        uint8_t *s_tok_fp8 = (uint8_t *)_lm_smem;
        uint8_t *s_tok_scales = s_tok_fp8 + HIDDEN_SIZE;

        // FP8 quantize the normalized input
        kernel::_gang_wave_parallel_fp8_quant<HIDDEN_SIZE>(
            (const unsigned short *)config.lm_norm_scratch,
            s_tok_fp8, s_tok_scales);

        const uint8_t *lm_W = (const uint8_t *)config.lm_mxfp4_weight
            + static_cast<int64_t>(xcd_id) * LM_N_WGS_PER_XCD * LM_WG_BYTES_TOTAL;
        const unsigned short *lm_bias = (const unsigned short *)config.lm_bias
            + xcd_id * LM_N_WGS_PER_XCD * OUTPUT_PER_WG;

        int const warp_id = tid >> 6;
        int const lane_id = tid & 63;
        int const col = lane_id & 15;
        int const g = lane_id >> 4;

        float thread_max = -1e30f;
        long long thread_max_idx = -1;
        int partition_start = xcd_id * LM_N_WGS_PER_XCD * OUTPUT_PER_WG;

        int lm_total_tiles = LM_N_WGS_PER_XCD * config.num_active_tokens;
        for (int lm_t = xcd_rank; lm_t < lm_total_tiles; lm_t += WORKERS_PER_XCD) {{
            int wg_idx = lm_t % LM_N_WGS_PER_XCD;
            uint8_t const *wg_data = lm_W + static_cast<int64_t>(wg_idx) * LM_WG_BYTES_TOTAL;
            uint8_t const *wg_scales = wg_data + LM_WG_DATA_BYTES;

            for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {{
                int wave_tile = warp_id + tile_iter * NUM_WAVES;
                int w_row = wave_tile * 16 + col;
                int const row_data_base = w_row * (HIDDEN_SIZE / 2);
                int const row_scale_base = w_row * LM_NUM_BLOCKS_32;

                kernel::f32x4_t acc = {{0.0f, 0.0f, 0.0f, 0.0f}};
                kernel::i32x8_t a0 = *(const kernel::i32x8_t *)(wg_data + row_data_base + 0 * 64 + g * 16);
                int sa0 = (int)wg_scales[row_scale_base + 0 * 4 + g];
                kernel::i32x8_t a1 = *(const kernel::i32x8_t *)(wg_data + row_data_base + 1 * 64 + g * 16);
                int sa1 = (int)wg_scales[row_scale_base + 1 * 4 + g];
                kernel::i32x8_t a2 = *(const kernel::i32x8_t *)(wg_data + row_data_base + 2 * 64 + g * 16);
                int sa2 = (int)wg_scales[row_scale_base + 2 * 4 + g];
                kernel::i32x8_t a3 = *(const kernel::i32x8_t *)(wg_data + row_data_base + 3 * 64 + g * 16);
                int sa3 = (int)wg_scales[row_scale_base + 3 * 4 + g];

                #pragma unroll 1
                for (int ki = 0; ki < MFMA_ITERS; ki += 4) {{
                    {{
                        kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
                        int sb = (int)s_tok_scales[ki];
                        acc = kernel::_gang_mfma_f4xf8(a0, b, acc, sa0, sb);
                    }}
                    if (ki + 4 < MFMA_ITERS) {{
                        int kt = (ki + 4) * K_PER_MFMA;
                        a0 = *(const kernel::i32x8_t *)(wg_data + row_data_base + kt / 2 + g * 16);
                        sa0 = (int)wg_scales[row_scale_base + kt / 32 + g];
                    }}
                    {{
                        kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
                        int sb = (int)s_tok_scales[ki + 1];
                        acc = kernel::_gang_mfma_f4xf8(a1, b, acc, sa1, sb);
                    }}
                    if (ki + 5 < MFMA_ITERS) {{
                        int kt = (ki + 5) * K_PER_MFMA;
                        a1 = *(const kernel::i32x8_t *)(wg_data + row_data_base + kt / 2 + g * 16);
                        sa1 = (int)wg_scales[row_scale_base + kt / 32 + g];
                    }}
                    {{
                        kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
                        int sb = (int)s_tok_scales[ki + 2];
                        acc = kernel::_gang_mfma_f4xf8(a2, b, acc, sa2, sb);
                    }}
                    if (ki + 6 < MFMA_ITERS) {{
                        int kt = (ki + 6) * K_PER_MFMA;
                        a2 = *(const kernel::i32x8_t *)(wg_data + row_data_base + kt / 2 + g * 16);
                        sa2 = (int)wg_scales[row_scale_base + kt / 32 + g];
                    }}
                    if (ki + 3 < MFMA_ITERS) {{
                        kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
                        int sb = (int)s_tok_scales[ki + 3];
                        acc = kernel::_gang_mfma_f4xf8(a3, b, acc, sa3, sb);
                    }}
                    if (ki + 7 < MFMA_ITERS) {{
                        int kt = (ki + 7) * K_PER_MFMA;
                        a3 = *(const kernel::i32x8_t *)(wg_data + row_data_base + kt / 2 + g * 16);
                        sa3 = (int)wg_scales[row_scale_base + kt / 32 + g];
                    }}
                }}

                // Argmax epilogue
                if (col == 0) {{
                    for (int i = 0; i < 4; i++) {{
                        int out_n = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
                        float sum = acc[i];
                        unsigned bt = (unsigned)lm_bias[out_n] << 16;
                        float bv;
                        __builtin_memcpy(&bv, &bt, 4);
                        float val = sum + bv;
                        long long abs_idx = (long long)(partition_start + out_n);
                        if (val > thread_max) {{
                            thread_max = val;
                            thread_max_idx = abs_idx;
                        }}
                    }}
                }}
            }}
        }}

        // Per-worker warp reduce
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1) {{
            float other_val = __shfl_xor(thread_max, offset, 64);
            unsigned int idx_lo = static_cast<unsigned int>(thread_max_idx & 0xFFFFFFFF);
            unsigned int idx_hi = static_cast<unsigned int>((thread_max_idx >> 32) & 0xFFFFFFFF);
            unsigned int other_lo = __shfl_xor(idx_lo, offset, 64);
            unsigned int other_hi = __shfl_xor(idx_hi, offset, 64);
            long long other_idx = (static_cast<long long>(other_hi) << 32) | other_lo;
            if (other_val > thread_max) {{
                thread_max = other_val;
                thread_max_idx = other_idx;
            }}
        }}

        // Cross-warp reduce via shared memory
        __shared__ float s_max_vals[4];
        __shared__ long long s_max_idxs[4];
        if (lane_id == 0) {{
            s_max_vals[warp_id] = thread_max;
            s_max_idxs[warp_id] = thread_max_idx;
        }}
        __syncthreads();

        if (tid == 0) {{
            float best_val = -1e30f;
            long long best_idx = -1;
            for (int w = 0; w < 4; w++) {{
                if (s_max_vals[w] > best_val) {{
                    best_val = s_max_vals[w];
                    best_idx = s_max_idxs[w];
                }}
            }}
            // Atomic CAS to per-XCD max
            unsigned long long *argmax_packed =
                reinterpret_cast<unsigned long long *>(&argmax_packed_base[xcd_id * 4]);
            unsigned long long old_packed = __atomic_load_n(argmax_packed, __ATOMIC_RELAXED);
            while (true) {{
                int old_val_bits = static_cast<int>(old_packed & 0xFFFFFFFF);
                float old_val;
                __builtin_memcpy(&old_val, &old_val_bits, 4);
                if (best_val <= old_val) break;
                int new_val_bits;
                __builtin_memcpy(&new_val_bits, &best_val, 4);
                int new_idx_bits = static_cast<int>(best_idx);
                unsigned long long new_packed =
                    (static_cast<unsigned long long>(static_cast<unsigned int>(new_idx_bits)) << 32)
                    | static_cast<unsigned int>(new_val_bits);
                if (__atomic_compare_exchange_n(argmax_packed, &old_packed, new_packed,
                                                 true, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                    break;
            }}
        }}
    }}

    // LM head global barrier
    fleet_mk::barrier_global_local(lmhead_done, s_lmhead_expected, TOTAL_WORKERS);

    // Cross-XCD argmax reduce (worker 0 on XCD 0)
    if (xcd_id == 0 && xcd_rank == 0 && tid == 0) {{
        float best_val = -1e30f;
        long long best_idx = -1;
        for (int x = 0; x < NUM_XCDS; x++) {{
            unsigned long long packed = reinterpret_cast<unsigned long long *>(
                &argmax_packed_base[x * 4])[0];
            int val_bits = static_cast<int>(packed & 0xFFFFFFFF);
            float v;
            __builtin_memcpy(&v, &val_bits, 4);
            int idx_bits = static_cast<int>(packed >> 32);
            if (v > best_val) {{
                best_val = v;
                best_idx = static_cast<long long>(idx_bits);
            }}
        }}
        static_cast<long long *>(config.argmax_output)[0] = best_idx;
    }}

    // Final L2 writeback
    fleet_mk::final_writeback();
}}
''')

    return "\n".join(parts)


# ============================================================================
# generate_driver
# ============================================================================

def generate_driver(cfg: ModelConfig) -> str:
    """Emit the Python demo driver for `cfg`. Dispatches on arch -- see
    generate_kernel for why the two arms are separate functions."""
    if cfg.arch == "dense":
        return generate_driver_dense(cfg)
    if cfg.arch == "moe":
        return generate_driver_fused_moe(cfg)
    raise ValueError(f"Unknown arch '{cfg.arch}'")


def generate_driver_fused_moe(cfg: ModelConfig) -> str:
    """MoE (GPT-OSS 120B) Python driver.

    THIS IS THE MAINTAINED SOURCE of demo_gpt_oss_120b.py, the driver that runs
    the 2.520 ms/tok kernel. Edit here, never the demo -- the gate is
    byte-identity, checked by `check_roundtrip.py --strict`.

    Built backwards, same as generate_kernel_fused_moe: the hand-written driver
    was pasted in verbatim (brace- and backslash-escaped) and is being
    parameterized one literal at a time with the gate re-run after each.

    Four decode strategies live in here, selected by env var. Their ctypes
    argtypes lists and the C signatures in generate_launch_fused_moe are two
    views of one ABI -- a mismatch is stack corruption reproducible only under
    whichever of the four paths the env happens to select.
    """
    return f'''\
#!/usr/bin/env python3
"""Auto-generated by fleet_mk_generate.py
Fleet MK demo for {cfg.name} inference using two-level fusion.

Loads {cfg.name} weights from HuggingFace, quantizes to MXFP4, builds the
pointer table for the {cfg.name} persistent kernel, runs prefill via PyTorch,
then decode via the fleet_mk kernel.

Usage:
    HIP_VISIBLE_DEVICES=0 python3 demo_{cfg.name_clean}.py \\
        --model Gpt/GPT-OSS-120B \\
        --prompt "Tell me the history of america" --max-seq-length {cfg.max_seq_len}
"""

import argparse
import ctypes
import gc
import json
import math
import os
import sys
import time

import torch
import yaml
from safetensors import safe_open

# Weight packing primitives. These live in fleet_mk now (fleet_megakernel_vllm/mxfp4_pack.py,
# a verified copy of mirage's demo.py originals) rather than being imported out
# of the mirage checkout at runtime -- the vLLM plugin already depends on the
# local copy, and the two packers must agree byte-for-byte or the two entry
# points build different slabs for the same kernel.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fleet_megakernel_vllm.mxfp4_pack import (  # noqa: E402
    pad_weight_1d, pad_weight_2d, quantize_bf16_to_mxfp4, pack_mxfp4_workgroup{emit_oproj_kmajor_import(cfg)}{emit_w13_kmajor_import(cfg)}{emit_lm_head_kmajor_import(cfg)})

from transformers import AutoTokenizer, AutoModelForCausalLM

# -- {cfg.name} constants --
HIDDEN_SIZE = {cfg.padded_hidden_size}
ACTUAL_HIDDEN_DIM = {cfg.hidden_size}  # for RMSNorm mean
INTERMEDIATE_SIZE = {cfg.padded_intermediate_size}
VOCAB_SIZE = {cfg.vocab_size}
PADDED_VOCAB_SIZE = {cfg.padded_vocab_size}  # next multiple for LM head split
NUM_LAYERS = {cfg.num_layers}
NUM_Q_HEADS = {cfg.num_q_heads}
NUM_KV_HEADS = {cfg.num_kv_heads}
HEAD_DIM = {cfg.head_dim}
Q_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS  # {cfg.q_per_kv}
QKV_OUTPUT_SIZE = NUM_Q_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM  # {cfg.qkv_output_size}

# All GEMMs use OPW={cfg.output_per_wg}
OUTPUT_PER_WG = {cfg.output_per_wg}
NUM_XCDS = {cfg.num_xcds}
WORKERS_PER_XCD = {cfg.workers_per_xcd}
TOTAL_WORKERS = NUM_XCDS * WORKERS_PER_XCD
NUM_KV_CHUNKS = {cfg.num_kv_chunks}

# Per-XCD workgroup counts
QKV_N_WGS_PER_XCD = (QKV_OUTPUT_SIZE // OUTPUT_PER_WG) // NUM_XCDS  # {cfg.qkv_n_wgs_per_xcd}
OPROJ_REDUCTION = NUM_Q_HEADS * HEAD_DIM  # {cfg.oproj_reduction}
OPROJ_OPW = {cfg.oproj_opw}
OPROJ_N_WGS = HIDDEN_SIZE // OPROJ_OPW  # {cfg.oproj_n_wgs} (global distribution)
OPROJ_N_WGS_PER_XCD = OPROJ_N_WGS // NUM_XCDS  # {cfg.oproj_n_wgs_per_xcd}
# MoE expert dimensions
NUM_EXPERTS = {cfg.num_experts}
NUM_TOPK = {cfg.num_experts_per_tok}
MOE_INTERMEDIATE_SIZE = {cfg.padded_moe_intermediate_size}

# ── MoE weight row stride (expert weights only) ──────────────────────────────
# How far apart consecutive expert weight ROWS are stored, in K values. This is
# NOT the reduction length: the MFMA always sums HIDDEN_SIZE ({cfg.padded_hidden_size}) columns, {cfg.moe_k_mfma_iters}
# iterations of 16x16x128, whatever this is set to. The two are separate because
# the row pitch belongs to whoever wrote the buffer, and the reduction belongs to
# the MFMA -- see MPK_MOE_K_STRIDE in gang_moe_fused_mxfp4_mi300.cuh.
#
# It is only ever non-default when fleet_mk reads expert weights out of an
# allocation someone else padded. vLLM rounds GPT-OSS's K to 3072, so its rows
# sit 1536 B apart; to alias that memory rather than copy it, fleet_mk has to walk
# rows at ITS pitch while still reducing {cfg.padded_hidden_size}. Setting this here packs fleet_mk's
# own slab the same way, which exercises the kernel's stride path without vLLM
# in the picture.
#
# MUST equal the -DMPK_MOE_K_STRIDE the .so was built with (0/unset means "same
# as the reduction"). A mismatch does not fail the build -- it reads every
# expert row at the wrong offset and produces fluent-looking garbage.
MOE_K_STRIDE = int(os.environ.get("FLEET_MK_MOE_K_STRIDE", "0")) or HIDDEN_SIZE
assert MOE_K_STRIDE >= HIDDEN_SIZE and MOE_K_STRIDE % 32 == 0, (
    f"MOE_K_STRIDE={{MOE_K_STRIDE}} must be a multiple of 32 and at least the "
    f"{{HIDDEN_SIZE}}-wide reduction")
MOE_K_STRIDE_BLOCKS = MOE_K_STRIDE // 32

# ── MoE scale placement ──────────────────────────────────────────────────────
# 0: scales interleaved per workgroup -- [data][scales][data][scales]... This is
#    what fleet_mk has always packed, and the kernel finds a workgroup's scales by
#    walking WG_DATA bytes past its own data.
# 1: split -- [all data][all scales], two contiguous sections addressed with
#    independent bases and pitches.
#
# The split mode is the shape vLLM's memory already has: w13_weight and
# w13_weight_scale are two separate tensors, and no offset from one reaches the
# other. Keeping them adjacent in ONE allocation here exercises the whole
# split-addressing path in the kernel without yet having to widen the pointer
# table -- the kernel derives the scale base from the data base. The later step
# that aliases vLLM's memory only changes where that base comes from.
#
# MUST equal the -DMPK_MOE_SPLIT_SCALES the .so was built with. Like the stride
# knob, a mismatch is silent: it reads real scale bytes for the wrong rows.
MOE_SPLIT_SCALES = os.environ.get("FLEET_MK_MOE_SPLIT_SCALES", "0") == "1"

# ── MoE data/scale allocation split ──────────────────────────────────────────
# Split mode puts the two sections in ONE allocation, adjacent, so the scale base
# is data_base + NUM_EXPERTS*EXPERT_BYTES. That is a convenience, not something
# the kernel requires: since the pointer table carries the two bases separately,
# the sections may live in unrelated allocations.
#
# This knob makes them so. It buys nothing on its own -- same bytes, same
# addresses modulo the allocator -- and that is exactly the point: it is the last
# structural difference between fleet_mk's slab and the arrangement where the DATA
# pointer belongs to vLLM and only the SCALE pointer is fleet_mk's. Exercising it on
# fleet_mk's own bytes first isolates "two allocations" from "someone else's bytes",
# so that if the aliased run misbehaves, this is already ruled out.
#
# Costs no .so flag: the kernel takes both bases as arguments and never assumes a
# relationship between them.
MOE_SPLIT_BUFFERS = os.environ.get("FLEET_MK_MOE_SPLIT_BUFFERS", "0") == "1"

# ── Fleet QKV weight prefetch hand-off (input_ptrs[24]/[25]) ─────────────────
# ON (the default) puts the QKV weight pointers fleet actually reads in those
# slots. OFF restores fleet_mk's MoE scale bases, which is what shipped before and
# what every -DMPK_PREFETCH_NEXT_QKV build ran with. See the long note at the
# [24]/[25] entries in the pointer table for the mechanism.
#
# Kept as a knob rather than a straight edit because the two arms differ in
# exactly two host-side integers -- same .so, same defines, same everything --
# which makes it the cleanest A/B available and the only way to get a control
# run without a rebuild. Both arms are correct; they differ in cost.
FLEET_MK_QKV_PF = os.environ.get("FLEET_MK_QKV_PF", "1") == "1"

# ── MoE slab tail pad (DIAGNOSTIC) ───────────────────────────────────────────
# Bytes of dead space appended to the END of each packed expert slab. Purely a
# probe for reads that run off the end: it moves no base and no section offset
# (data still starts at byte 0, scales still at NUM_EXPERTS*EXPERT_BYTES), it
# only makes the addresses just past the last scale byte legal instead of
# faulting. So a run that faults at pad=0 and survives at pad=1MiB is reading
# past the buffer -- the pad does not FIX that, it hides it, which is exactly
# what makes it a usable bisection signal.
# Set to -1 for a PURE RELOCATION: copy each slab to a fresh allocation with
# zero extra bytes. This is the control for the pad experiment. A pad does two
# things at once -- it adds legal bytes past the end AND it moves the buffer to
# a different address -- and only the first is evidence of an overrun. If -1
# also survives, the pad proved nothing about reads running off the end and the
# fault is address- or allocator-dependent instead.
MOE_TAIL_PAD = int(os.environ.get("FLEET_MK_MOE_TAIL_PAD", "0"))

# Which expert slabs the pad applies to: "both" (default), "w13", or "w2".
# Padding one at a time says WHICH tensor is overrun, which the combined run
# cannot: both slabs are rebuilt together otherwise.
MOE_TAIL_PAD_SLOT = os.environ.get("FLEET_MK_MOE_TAIL_PAD_SLOT", "both")
assert MOE_TAIL_PAD_SLOT in ("both", "w13", "w2")

assert not MOE_SPLIT_BUFFERS or MOE_SPLIT_SCALES, (
    "FLEET_MK_MOE_SPLIT_BUFFERS requires FLEET_MK_MOE_SPLIT_SCALES=1 -- there are no "
    "separable sections to allocate apart while they are interleaved per "
    "workgroup")

# ── MoE expert row pitch (N axis) ────────────────────────────────────────────
# How many ROWS one expert occupies in the data section, against how many rows
# the kernel computes. Same stride-vs-extent split as MOE_K_STRIDE, one axis
# over: the pitch belongs to whoever wrote the buffer, the row count belongs to
# the tiles.
#
# vLLM pads BOTH axes to a multiple of 256, so its experts sit 6144/3072 rows
# apart while fleet_mk computes 5888/2944. Unlike the K pad, these extra rows are
# never addressed by any tile -- no MFMA sums them -- so their content is
# irrelevant rather than required-zero.
#
# Set in units of the INTERMEDIATE/HIDDEN row count (3072 for vLLM), not of
# W13's doubled axis; the packer's W13 call doubles it, matching the kernel's
# W13_N_STRIDE = 2 * MPK_MOE_N_STRIDE.
#
# MUST equal the -DMPK_MOE_N_STRIDE the .so was built with. Requires split mode
# on both sides: a foreign expert pitch implies a foreign buffer, which cannot
# also carry fleet_mk's per-workgroup interleaved scales.
MOE_N_STRIDE = int(os.environ.get("FLEET_MK_MOE_N_STRIDE", "0"))
assert not MOE_N_STRIDE or MOE_SPLIT_SCALES, (
    "FLEET_MK_MOE_N_STRIDE requires FLEET_MK_MOE_SPLIT_SCALES=1")
assert not MOE_N_STRIDE or MOE_N_STRIDE >= HIDDEN_SIZE, (
    f"MOE_N_STRIDE={{MOE_N_STRIDE}} is shorter than the {{HIDDEN_SIZE}} rows W2 "
    f"computes -- experts would overlap")
# One knob covers both axes only because GPT-OSS pads hidden and intermediate
# alike (fleet_mk 2944/2944, vLLM 3072/3072). The kernel static_asserts this too.
assert not MOE_N_STRIDE or HIDDEN_SIZE == MOE_INTERMEDIATE_SIZE, (
    "one N-stride knob assumes hidden and intermediate pad alike")

# Must match W13_OPW in generated/{cfg.name_clean}_kernel.cuh (weight packing layout).
# MEASURED: W13_OPW=64 (with MOE_TOTAL_TILES_PER_XCD=83) removes the padding
# waste and drops W13_TILES_PER_WAVE 2->1, but is 280us/tok WORSE (2.818 vs
# 2.537ms). Note this is the OPPOSITE sign to W2_OPW below: W13 wants fat tiles
# (it reuses one LDS weight load across two MFMA rounds), W2 wants thin ones.
# Do not "symmetrize" these two constants.
W13_OPW = {cfg.w13_output_per_wg}
W13_OUTPUT_SIZE = 2 * MOE_INTERMEDIATE_SIZE
W13_N_WGS = W13_OUTPUT_SIZE // W13_OPW  # {cfg.w13_n_wgs}
# Rows one expert occupies in the DATA section. W13 is gate+up interleaved, so a
# foreign N stride counts intermediate rows and this axis holds two of them --
# the kernel's W13_N_STRIDE uses the identical expression.
W13_N_STRIDE = 2 * MOE_N_STRIDE if MOE_N_STRIDE else W13_OUTPUT_SIZE
assert W13_N_STRIDE % W13_OPW == 0, "W13 expert pitch is not a whole number of workgroups"
W13_DATA_WGS = W13_N_STRIDE // W13_OPW
# Data bytes stride, scale bytes reduce -- the kernel derives row_data_base from
# K_STRIDE and row_scale_base from K_REDUCE independently. In split mode the
# workgroup pitch covers DATA ONLY, exactly as W13_WG_BYTES does in the kernel;
# these must stay the same expression on both sides.
W13_WG_DATA = W13_OPW * (MOE_K_STRIDE // 2)
W13_WG_SCALE = W13_OPW * (HIDDEN_SIZE // 32)
W13_WG_BYTES = W13_WG_DATA if MOE_SPLIT_SCALES else (W13_WG_DATA + W13_WG_SCALE)
# Pitch from one expert's data to the next: keyed on the STORED row count, not
# the computed one. Identical to W13_N_WGS whenever MOE_N_STRIDE is unset.
W13_EXPERT_BYTES = W13_DATA_WGS * W13_WG_BYTES
# Must match W2_OPW in generated/{cfg.name_clean}_kernel.cuh — this controls the
# weight packing layout, so a mismatch silently produces garbage output.
# MEASURED: W2_OPW=128 (with MOE_TOTAL_TILES_PER_XCD=42) halves the W2 tile
# count and enables W2_TILES_PER_WAVE=2 cross-tile pipelining, but is 142us/tok
# WORSE (2517 vs 2375us device time). Fewer, fatter W2 tiles cut the parallelism
# that was hiding W2's HBM latency. Do not "fix" this back to 128.
W2_OPW = {cfg.w2_output_per_wg}
W2_N_WGS = HIDDEN_SIZE // W2_OPW  # {cfg.w2_n_wgs}
# W2's output axis is hidden, so the foreign N stride applies once here against
# W13's twice.
W2_N_STRIDE = MOE_N_STRIDE if MOE_N_STRIDE else HIDDEN_SIZE
assert W2_N_STRIDE % W2_OPW == 0, "W2 expert pitch is not a whole number of workgroups"
W2_DATA_WGS = W2_N_STRIDE // W2_OPW
W2_WG_DATA = W2_OPW * (MOE_K_STRIDE // 2)
W2_WG_SCALE = W2_OPW * (MOE_INTERMEDIATE_SIZE // 32)
W2_WG_BYTES = W2_WG_DATA if MOE_SPLIT_SCALES else (W2_WG_DATA + W2_WG_SCALE)
W2_EXPERT_BYTES = W2_DATA_WGS * W2_WG_BYTES
ROUTER_OUTPUT_SIZE = {cfg.num_experts}
ROUTER_N_WGS = ROUTER_OUTPUT_SIZE // OUTPUT_PER_WG  # {cfg.router_n_wgs}
LM_N_WGS_PER_XCD = (PADDED_VOCAB_SIZE // OUTPUT_PER_WG) // NUM_XCDS  # {cfg.lm_n_wgs_per_xcd}

# MXFP4 workgroup byte sizes
QKV_WG_BYTES = OUTPUT_PER_WG * (HIDDEN_SIZE // 2 + HIDDEN_SIZE // 32)
OPROJ_WG_BYTES = OPROJ_OPW * (OPROJ_REDUCTION // 2 + OPROJ_REDUCTION // 32)  # {cfg.oproj_wg_bytes} for OPW={cfg.oproj_opw}
ROUTER_WG_BYTES = OUTPUT_PER_WG * (HIDDEN_SIZE // 2 + HIDDEN_SIZE // 32)
# W13/W2 WG_BYTES defined above
LM_WG_BYTES = OUTPUT_PER_WG * (HIDDEN_SIZE // 2 + HIDDEN_SIZE // 32)

# Pointer table layout: pre-computed mirage format
# {len(MIRAGE_IN)} mirage_in + {len(MIRAGE_OUT)} mirage_out + 1 layer_output = {MIRAGE_PTRS_PER_LAYER} per layer per XCD
MIRAGE_IN = {len(MIRAGE_IN)}
MIRAGE_OUT = {len(MIRAGE_OUT)}
PTRS_PER_LAYER = MIRAGE_IN + MIRAGE_OUT + 1  # {MIRAGE_PTRS_PER_LAYER}

# Counter slot offsets (must match kernel .cuh)
{emit_driver_qkv_slot_note(cfg)}SLOT_QKV_BARRIER = {counter_slots()['SLOT_QKV_BARRIER_NEW'][0]} * 16  # per-XCD QKV barrier arrival

# Counter buffer
COUNTERS_PER_LAYER = {cfg.counters_per_layer // 16} * 16  # 84 base + 10 fused oproj + 9 routing ready


def parse_args():
    parser = argparse.ArgumentParser(description="Fleet MK {cfg.name} demo")
    parser.add_argument("--model", default="Gpt/GPT-OSS-120B",
                        help="HuggingFace model name or path")
    parser.add_argument("--model-path", default=None,
                        help="Local model path (overrides --model)")
    parser.add_argument("--prompt", default="Tell me the history of america")
    parser.add_argument("--max-seq-length", type=int, default={cfg.max_seq_len})
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--fleet-mk-so", default=None,
                        help="Path to compiled {cfg.name_clean}.so")
    parser.add_argument("--pytorch-only", action="store_true",
                        help="Use PyTorch for decode (skip fleet_mk kernel)")
    return parser.parse_args()


def load_{cfg.name_clean}_kernel(so_path: str):
    """Load the compiled Gpt fleet_mk .so."""
    lib = ctypes.CDLL(so_path)

    lib.{cfg.name_clean}_init.restype = None
    lib.{cfg.name_clean}_init.argtypes = []

    lib.{cfg.name_clean}_launch.restype = None
    lib.{cfg.name_clean}_launch.argtypes = [
{emit_ctypes_argtypes('launch')}
    ]

    lib.{cfg.name_clean}_finalize.restype = None
    lib.{cfg.name_clean}_finalize.argtypes = []

    # hipGraph-based launch
    lib.{cfg.name_clean}_graph_capture.restype = None
    lib.{cfg.name_clean}_graph_capture.argtypes = [
{emit_ctypes_argtypes('graph_capture')}
    ]

    lib.{cfg.name_clean}_graph_launch.restype = None
    lib.{cfg.name_clean}_graph_launch.argtypes = [
        ctypes.c_int,       # cur_token_id
        ctypes.c_void_p,    # stream
    ]

    lib.{cfg.name_clean}_graph_destroy.restype = None
    lib.{cfg.name_clean}_graph_destroy.argtypes = []

    # Pipelined graph (bridge + main kernel, no host sync between tokens)
    lib.{cfg.name_clean}_pipe_capture.restype = None
    lib.{cfg.name_clean}_pipe_capture.argtypes = [
{emit_ctypes_argtypes('pipe_capture')}
    ]

    lib.{cfg.name_clean}_pipe_launch_first.restype = None
    lib.{cfg.name_clean}_pipe_launch_first.argtypes = [ctypes.c_void_p]

    lib.{cfg.name_clean}_pipe_launch_step.restype = None
    lib.{cfg.name_clean}_pipe_launch_step.argtypes = [ctypes.c_void_p]

    lib.{cfg.name_clean}_pipe_launch_all.restype = None
    lib.{cfg.name_clean}_pipe_launch_all.argtypes = [ctypes.c_void_p, ctypes.c_int]

    lib.{cfg.name_clean}_pipe_destroy.restype = None
    lib.{cfg.name_clean}_pipe_destroy.argtypes = []

    # C-side decode loop (eliminates Python per-token overhead)
    lib.{cfg.name_clean}_decode_loop.restype = None
    lib.{cfg.name_clean}_decode_loop.argtypes = [
{emit_ctypes_argtypes('decode_loop')}
    ]

    return lib


def main():
    args = parse_args()
    torch.set_default_dtype(torch.bfloat16)

    # -- Load model --
    model_path = args.model_path or args.model
    print(f"Loading {cfg.name} from: {{model_path}}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    config = model.config
    assert config.hidden_size == ACTUAL_HIDDEN_DIM or config.hidden_size == HIDDEN_SIZE
    assert config.num_attention_heads == NUM_Q_HEADS
    assert config.num_key_value_heads == NUM_KV_HEADS
    assert getattr(config, "head_dim", HEAD_DIM) == HEAD_DIM

    print(f"{cfg.name}: {{NUM_LAYERS}} layers, hidden={{HIDDEN_SIZE}}, "
          f"moe_intermediate={{MOE_INTERMEDIATE_SIZE}}, experts={{NUM_EXPERTS}}, topk={{NUM_TOPK}}, "
          f"heads={{NUM_Q_HEADS}}Q/{{NUM_KV_HEADS}}KV, head_dim={{HEAD_DIM}}")
    # Printed unconditionally: a packer/.so stride mismatch is silent at runtime,
    # so this line is the only place the packed pitch is visible next to the
    # -DMPK_MOE_K_STRIDE the kernel was built with.
    print(f"MoE expert rows: stride={{MOE_K_STRIDE}} values "
          f"({{MOE_K_STRIDE // 2}} B), reduction={{HIDDEN_SIZE}} values "
          f"({{HIDDEN_SIZE // 128}} MFMA iters)"
          + ("" if MOE_K_STRIDE == HIDDEN_SIZE else
             "  <-- .so MUST be built -DMPK_MOE_K_STRIDE=%d" % MOE_K_STRIDE))
    print(f"MoE scales: {{'SPLIT [all data][all scales]' if MOE_SPLIT_SCALES else 'interleaved per workgroup'}}"
          + ("  <-- .so MUST be built -DMPK_MOE_SPLIT_SCALES=1" if MOE_SPLIT_SCALES else ""))
    print(f"MoE expert pitch: W13 {{W13_N_STRIDE}} rows stored / {{W13_OUTPUT_SIZE}} "
          f"computed, W2 {{W2_N_STRIDE}} / {{HIDDEN_SIZE}}"
          + ("" if not MOE_N_STRIDE else
             "  <-- .so MUST be built -DMPK_MOE_N_STRIDE=%d" % MOE_N_STRIDE))
    # No .so flag for this one: the kernel already takes the two bases as
    # separate arguments and never assumes a relationship between them.
    print("MoE buffers: "
          + ("SEPARATE allocations for data and scales" if MOE_SPLIT_BUFFERS
             else "one allocation"))

    bs = 1  # decode batch size
    page_size = {cfg.page_size}  # == fleet_mk PAGE_SIZE (vLLM's ROCm default block size)
    # Enough pages to hold the whole sequence. This used to be a bare 16, which
    # only worked because page_size was 128 (16 x 128 = 2048 entries, far past
    # any --max-seq-length). At page_size 16 a fixed 16 pages is 256 entries and
    # silently truncates the default 512-token run, so derive it instead.
    max_num_pages = (args.max_seq_length + page_size - 1) // page_size

    # -- Tokenize --
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{{"role": "user", "content": args.prompt}}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([formatted], return_tensors="pt",
                                  add_special_tokens=False).to("cuda")
    else:
        model_inputs = tokenizer([args.prompt], return_tensors="pt").to("cuda")

    tokens = torch.zeros(args.max_seq_length, dtype=torch.long, device="cuda")
    prompt_ids = model_inputs.input_ids[0]
    prompt_len = prompt_ids.shape[0]
    tokens[:prompt_len] = prompt_ids
    print(f"Prompt: {{prompt_len}} tokens, max_seq_length: {{args.max_seq_length}}")

    # -- RoPE position embeddings --
    rotary_emb = model.model.rotary_emb
    positions = torch.arange(args.max_seq_length, device="cuda").unsqueeze(0)
    cos_raw, sin_raw = rotary_emb(
        torch.ones(1, 1, device="cuda"), positions)
    cos_raw = cos_raw[0]  # [max_seq, head_dim]
    sin_raw = sin_raw[0]
    if cos_raw.shape[-1] < HEAD_DIM:
        cos_raw = torch.nn.functional.pad(cos_raw, (0, HEAD_DIM - cos_raw.shape[-1]))
        sin_raw = torch.nn.functional.pad(sin_raw, (0, HEAD_DIM - sin_raw.shape[-1]))
    cos_padded = cos_raw.contiguous()
    sin_padded = sin_raw.contiguous()

    # -- Pack weights --
    print("Packing MXFP4 weights...")
    weight_tensors = []  # keep refs to prevent GC
    # MoE scale sections, one entry per layer, when they live in their own
    # allocation. Kept out of weight_tensors because that list is indexed by a
    # fixed WEIGHTS_PER_LAYER stride the pointer table depends on; these are
    # optional and would shift every slot after them. None when unused.
    moe_w13_scales = []
    moe_w2_scales = []

    # Load safetensors index for direct MXFP4 expert weight access
    # (HuggingFace dequantizes to bf16 without Triton -- we bypass that)
    _st_index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(_st_index_path) as _f:
        _st_index = json.load(_f)["weight_map"]
    _st_file_cache = {{}}  # cache open safetensor file handles
    def _load_st_tensor(name):
        fname = _st_index[name]
        fpath = os.path.join(model_path, fname)
        if fpath not in _st_file_cache:
            _st_file_cache[fpath] = safe_open(fpath, framework="pt")
        return _st_file_cache[fpath].get_tensor(name)

    for li in range(NUM_LAYERS):
        layer = model.model.layers[li]

        # -- QKV weight: interleave Q/K/V by KV groups --
        w_q = pad_weight_2d(layer.self_attn.q_proj.weight.data.to("cuda"),
                            target_cols=HIDDEN_SIZE)
        w_k = pad_weight_2d(layer.self_attn.k_proj.weight.data.to("cuda"),
                            target_cols=HIDDEN_SIZE)
        w_v = pad_weight_2d(layer.self_attn.v_proj.weight.data.to("cuda"),
                            target_cols=HIDDEN_SIZE)

        qkv_chunks = []
        for g in range(NUM_KV_HEADS):
            qkv_chunks.append(w_q[g * Q_PER_KV * HEAD_DIM:(g + 1) * Q_PER_KV * HEAD_DIM])
            qkv_chunks.append(w_k[g * HEAD_DIM:(g + 1) * HEAD_DIM])
            qkv_chunks.append(w_v[g * HEAD_DIM:(g + 1) * HEAD_DIM])
        w_qkv = torch.cat(qkv_chunks, dim=0).contiguous()
        assert w_qkv.shape == (QKV_OUTPUT_SIZE, HIDDEN_SIZE), \\
            f"QKV shape mismatch: {{w_qkv.shape}} vs ({{QKV_OUTPUT_SIZE}}, {{HIDDEN_SIZE}})"

        qkv_blocks, qkv_scales = quantize_bf16_to_mxfp4(w_qkv)
        qkv_packed = pack_mxfp4_workgroup(
            qkv_blocks, qkv_scales, output_per_wg=OUTPUT_PER_WG,
            target_out_dim=QKV_OUTPUT_SIZE,
            target_num_blocks=HIDDEN_SIZE // 32)
{emit_weight_append(cfg, 'qkv_weight_base')}

        # QKV bias
        qkv_bias = torch.zeros(QKV_OUTPUT_SIZE, dtype=torch.bfloat16, device="cuda")
        if layer.self_attn.q_proj.bias is not None:
            q_bias = layer.self_attn.q_proj.bias.data.to("cuda")
            k_bias = layer.self_attn.k_proj.bias.data.to("cuda")
            v_bias = layer.self_attn.v_proj.bias.data.to("cuda")
            bias_chunks = []
            for g in range(NUM_KV_HEADS):
                bias_chunks.append(q_bias[g * Q_PER_KV * HEAD_DIM:(g + 1) * Q_PER_KV * HEAD_DIM])
                bias_chunks.append(k_bias[g * HEAD_DIM:(g + 1) * HEAD_DIM])
                bias_chunks.append(v_bias[g * HEAD_DIM:(g + 1) * HEAD_DIM])
            qkv_bias = torch.cat(bias_chunks, dim=0).contiguous()
{emit_weight_append(cfg, 'qkv_bias')}

        # -- O-proj weight (OPW=16, per-XCD partitioned) --
        w_o = pad_weight_2d(layer.self_attn.o_proj.weight.data.to("cuda"),
                            target_rows=HIDDEN_SIZE)
        o_blocks, o_scales = quantize_bf16_to_mxfp4(w_o)
        o_packed = pack_mxfp4_workgroup(
            o_blocks, o_scales, output_per_wg=OPROJ_OPW,
            target_out_dim=HIDDEN_SIZE,
            target_num_blocks=OPROJ_REDUCTION // 32)
{emit_oproj_kmajor_shuffle(cfg)}\
{emit_weight_append(cfg, 'oproj_weight_base')}

        # O-proj bias (zeros)
        o_bias = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        if layer.self_attn.o_proj.bias is not None:
            o_raw = layer.self_attn.o_proj.bias.data.to("cuda")
            o_bias[:o_raw.shape[0]] = o_raw
{emit_weight_append(cfg, 'oproj_bias')}

        # -- RMSNorm weights (pad to HIDDEN_SIZE if needed) --
        norm_w1 = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        norm_w1_raw = layer.input_layernorm.weight.data.to("cuda")
        norm_w1[:norm_w1_raw.shape[0]] = norm_w1_raw
{emit_weight_append(cfg, 'norm_w1')}

        norm_w2 = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        norm_w2_raw = layer.post_attention_layernorm.weight.data.to("cuda")
        norm_w2[:norm_w2_raw.shape[0]] = norm_w2_raw
{emit_weight_append(cfg, 'norm_w2')}

        # -- Router weight --
        w_router = pad_weight_2d(layer.mlp.router.weight.data.to("cuda"),
                                 target_cols=HIDDEN_SIZE)
        r_blocks, r_scales = quantize_bf16_to_mxfp4(w_router)
        r_packed = pack_mxfp4_workgroup(
            r_blocks, r_scales, output_per_wg=OUTPUT_PER_WG,
            target_out_dim=ROUTER_OUTPUT_SIZE,
            target_num_blocks=HIDDEN_SIZE // 32)
{emit_weight_append(cfg, 'router_weight_base')}

        # Router bias (padded to ROUTER_OUTPUT_SIZE)
        r_bias = torch.zeros(ROUTER_OUTPUT_SIZE, dtype=torch.bfloat16, device="cuda")
        r_bias_raw = layer.mlp.router.bias.data.to("cuda")
        r_bias[:r_bias_raw.shape[0]] = r_bias_raw
{emit_weight_append(cfg, 'router_bias')}

        # BF16 router weight for fused GEMV (padded to HIDDEN_SIZE columns)
        w_router_bf16 = pad_weight_2d(layer.mlp.router.weight.data.to("cuda"),
                                       target_cols=HIDDEN_SIZE)
{emit_weight_append(cfg, 'router_weight_bf16')}

        # -- Per-expert W13 (gate+up) weights --
        # Load MXFP4 blocks+scales directly from safetensors (bypasses Triton dequant)
        gu_blocks = _load_st_tensor(f"model.layers.{{li}}.mlp.experts.gate_up_proj_blocks").to("cuda")
        gu_scales = _load_st_tensor(f"model.layers.{{li}}.mlp.experts.gate_up_proj_scales").to("cuda")
        gu_common = dict(
            output_per_wg=W13_OPW,
            target_out_dim=W13_OUTPUT_SIZE,
            target_num_blocks=HIDDEN_SIZE // 32,
            row_stride_blocks=MOE_K_STRIDE_BLOCKS,
            out_stride_rows=W13_N_STRIDE if MOE_N_STRIDE else None,
            split_scales=MOE_SPLIT_SCALES)
        # In split-buffer mode the two sections are packed as two independent
        # allocations rather than one. Same bytes in the same within-section
        # order either way -- `section` partitions the buffer, it does not
        # recompute it (tools/check_pack_section.py property 2).
        gu_packed = pack_mxfp4_workgroup(
            gu_blocks, gu_scales,
            section="data" if MOE_SPLIT_BUFFERS else "both", **gu_common)
{emit_w13_kmajor_shuffle(cfg)}\
{emit_weight_append(cfg, 'w13_weight_base')}
        moe_w13_scales.append(pack_mxfp4_workgroup(
            gu_blocks, gu_scales, section="scales", **gu_common)
            if MOE_SPLIT_BUFFERS else None)
        del gu_blocks, gu_scales

        # W13 bias [E, 2*intermediate]
        w13_bias = pad_weight_2d(layer.mlp.experts.gate_up_proj_bias.data.to("cuda"),
                                 target_cols=W13_OUTPUT_SIZE)
{emit_weight_append(cfg, 'w13_bias')}

        # -- Per-expert W2 (down) weights --
        dp_blocks = _load_st_tensor(f"model.layers.{{li}}.mlp.experts.down_proj_blocks").to("cuda")
        dp_scales = _load_st_tensor(f"model.layers.{{li}}.mlp.experts.down_proj_scales").to("cuda")
        dp_common = dict(
            output_per_wg=W2_OPW,
            target_out_dim=HIDDEN_SIZE,
            target_num_blocks=MOE_INTERMEDIATE_SIZE // 32,
            row_stride_blocks=MOE_K_STRIDE_BLOCKS,
            out_stride_rows=W2_N_STRIDE if MOE_N_STRIDE else None,
            split_scales=MOE_SPLIT_SCALES)
        dp_packed = pack_mxfp4_workgroup(
            dp_blocks, dp_scales,
            section="data" if MOE_SPLIT_BUFFERS else "both", **dp_common)
{emit_weight_append(cfg, 'w2_weight_base')}
        moe_w2_scales.append(pack_mxfp4_workgroup(
            dp_blocks, dp_scales, section="scales", **dp_common)
            if MOE_SPLIT_BUFFERS else None)
        del dp_blocks, dp_scales

        # W2 bias [E, hidden]
        w2_bias = pad_weight_2d(layer.mlp.experts.down_proj_bias.data.to("cuda"),
                                target_cols=HIDDEN_SIZE)
{emit_weight_append(cfg, 'w2_bias')}

        # -- Per-head attention sinks [num_q_heads] bf16 (GPT-OSS specific) --
        # merge_splitkv indexes as sinks[kv_head_idx * NUM_Q_PER_KV + head_idx],
        # which matches the natural [num_q_heads] ordering.
        w_sinks = layer.self_attn.sinks.data.to("cuda").to(torch.bfloat16).contiguous()
        assert w_sinks.shape == (NUM_Q_HEADS,), f"sinks shape {{w_sinks.shape}}"
{emit_weight_append(cfg, 'attn_sinks')}

        # Free layer refs
        del w_q, w_k, w_v, w_qkv, w_o
        del layer

        gc.collect()
        torch.cuda.empty_cache()

        if (li + 1) % 6 == 0 or li == NUM_LAYERS - 1:
            print(f"  Packed layer {{li + 1}}/{{NUM_LAYERS}}")

    WEIGHTS_PER_LAYER = {len(WEIGHT_SLOTS)}
    assert len(weight_tensors) == NUM_LAYERS * WEIGHTS_PER_LAYER
    _st_file_cache.clear()  # close safetensor file handles

    weight_ptrs_host = [t.data_ptr() for t in weight_tensors]

    # Diagnostic tail pad: see FLEET_MK_MOE_TAIL_PAD. Appended AFTER data_ptr() is
    # taken, so every base and every section offset is byte-for-byte what it was
    # -- the pad exists only to make the addresses past the last scale byte
    # legal. Held in weight_tensors' shadow by this list so it is not collected.
    _moe_tail_pads = []
    if MOE_TAIL_PAD:
        _extra = 0 if MOE_TAIL_PAD < 0 else MOE_TAIL_PAD
        for li in range(NUM_LAYERS):
            for slot in (9, 11):  # w13_weight, w2_weight
                if MOE_TAIL_PAD_SLOT == "w13" and slot != 9:
                    continue
                if MOE_TAIL_PAD_SLOT == "w2" and slot != 11:
                    continue
                t = weight_tensors[li * WEIGHTS_PER_LAYER + slot]
                padded = torch.empty(t.numel() + _extra,
                                     dtype=t.dtype, device=t.device)
                padded[:t.numel()] = t.reshape(-1)
                padded[t.numel():] = 0
                _moe_tail_pads.append(padded)
                weight_ptrs_host[li * WEIGHTS_PER_LAYER + slot] = padded.data_ptr()
        print(f"MoE slab tail pad: {{MOE_TAIL_PAD}} B appended per expert slab "
              f"(DIAGNOSTIC -- hides overruns, does not fix them)")

    # -- Embedding + LM head --
    embed_weight = model.model.embed_tokens.weight.data.to("cuda")
    final_norm_w_raw = model.model.norm.weight.data.to("cuda")
    final_norm_w = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    final_norm_w[:final_norm_w_raw.shape[0]] = final_norm_w_raw
    final_norm_w = final_norm_w.contiguous()

    lm_head_weight = pad_weight_2d(
        model.lm_head.weight.data.to("cuda"),
        target_rows=PADDED_VOCAB_SIZE,
        target_cols=HIDDEN_SIZE)
    lm_blocks, lm_scales = quantize_bf16_to_mxfp4(lm_head_weight)
    lm_head_packed = pack_mxfp4_workgroup(
        lm_blocks, lm_scales, output_per_wg=OUTPUT_PER_WG,
    ).squeeze(0)
{emit_lm_head_kmajor_shuffle(cfg)}\
    print(f"LM head MXFP4: {{lm_head_weight.shape}} -> {{lm_head_packed.shape}}")
    lm_head_zero_bias = torch.zeros(1, PADDED_VOCAB_SIZE, dtype=torch.bfloat16, device="cuda")

    # -- Allocate workspace buffers --
    print("Allocating buffers...")
    q_ws_stride = Q_PER_KV * HEAD_DIM  # {cfg.q_workspace_stride}
    kv_cache_stride = NUM_KV_HEADS * HEAD_DIM  # {cfg.kv_cache_stride}

    # Per-layer KV cache (paged)
    num_kv_entries = max_num_pages * page_size
    buf_k_cache = [torch.zeros(num_kv_entries, kv_cache_stride,
                                dtype=torch.bfloat16, device="cuda") for _ in range(NUM_LAYERS)]
    buf_v_cache = [torch.zeros(num_kv_entries, kv_cache_stride,
                                dtype=torch.bfloat16, device="cuda") for _ in range(NUM_LAYERS)]

    # Residual buffer (embedding goes here, then oproj_out serves as residual for layers 1+)
    buf_residual = torch.zeros(bs, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    # x_output: QKV writes ResAdd+RMSNorm result here (separate from residual, like mirage's mlp_weighted_sum_out)
    buf_x_output = torch.zeros(bs, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    # Norm scratch buffers
    buf_norm_scratch1 = torch.zeros(bs, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    buf_norm_scratch2 = torch.zeros(bs, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    # QKV output
    buf_qkv_output = torch.zeros(bs, QKV_OUTPUT_SIZE, dtype=torch.bfloat16, device="cuda")

    # Attention buffers
    buf_q_workspace = torch.zeros(bs, NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    buf_attn_out = torch.zeros(bs, NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")

    # O-proj output
    buf_oproj_out = torch.zeros(bs, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    # MoE routing buffers
    buf_routing_indices = torch.zeros(NUM_EXPERTS, bs, dtype=torch.int32, device="cuda")
    buf_topk_weight = torch.zeros(bs, NUM_TOPK, dtype=torch.float32, device="cuda")
    buf_active_expert_ids = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device="cuda")

    # MoE intermediate buffers
    buf_swiglu_out = torch.zeros(bs, NUM_TOPK, MOE_INTERMEDIATE_SIZE, dtype=torch.bfloat16, device="cuda")
    # MOE_WS_SLOTS slabs per token, not one. Fleet's MoE W2 epilogue writes to
    # (b * MOE_WS_SLOTS + slot) * HIDDEN_SIZE: a token routes to exactly
    # NUM_TOPK experts, topk_slot is known per tile, and wg_idx partitions the
    # hidden axis, so (token, slot, hidden range) has exactly one writer and the
    # float atomicAdd becomes a plain write-through store. That removes the
    # accumulation-order dependence that made decode non-deterministic.
    # The kernel writes 4x this stride whether or not this line is widened, so a
    # narrow buffer here corrupts silently -- it still compiles and links.
    MOE_WS_SLOTS = NUM_TOPK  # must match kernel::MOE_WS_SLOTS (moe_ws_layout.cuh)
    buf_moe_workspace_f32 = torch.zeros(bs, MOE_WS_SLOTS * HIDDEN_SIZE,
                                        dtype=torch.float32, device="cuda")

    # MoE per-expert barrier. 160 ints (10 cache lines) per expert, NOT 16.
    #
    # Fleet's kernel indexes this as MOE_BAR_STRIDE = MOE_BAR_SLOTS(10) *
    # MOE_BAR_LINE(16) ints per expert: 8 per-XCD release flags, then the global
    # arrival counter on line 8. At 16 ints/expert every expert past the first
    # aliases the next one's flags, and the arrival counter at line 8 lands
    # outside the allocation entirely.
    #
    # The one-line-per-slot spacing is a correctness requirement, not padding.
    # Fleet's own note (gang_moe_fused_mxfp4_mi300.cuh, above the atomic at
    # :2412): the release fan-out uses st_wt (sc0 sc1), which bypasses L2 and
    # goes straight to HBM, while the arrival counter is an L2-resident atomic
    # RMW. Sharing a 64-byte line lets the stale L2 copy be written back over
    # fresh write-through release values, silently reverting releases that
    # already fired.
    #
    # NOT zeroed between decode iterations, and the in-kernel zeroing loop is
    # compiled out under the fleet layer body. The W13->W2 release value is
    # `layer_idx + 1` on a counter fleet never resets, so wiping it destroys the
    # ordering it encodes. Measured: with the per-layer counter block already
    # shared, zeroing this left every XCD parked at
    #   phase=80 layer=1 barrier=800..803 observed=0 expected=2
    # -- fleet's MoE W13->W2 barrier at layer 1, waiting for the 2 that layer
    # 0's release wrote and the zeroing erased.
    MOE_BAR_STRIDE = 160  # must equal kernel::MOE_BAR_SLOTS * kernel::MOE_BAR_LINE
    buf_moe_barrier = torch.zeros(MOE_BAR_STRIDE * NUM_EXPERTS,
                                  dtype=torch.int32, device="cuda")

    # Counter buffer — monotonic counters, NO need to zero between iterations
    # Layout: [per-layer counters | rank counters | decode iter counter | embed barrier]
{emit_trailing_counters(cfg)}
    buf_counter = torch.zeros(counter_total_ints, dtype=torch.int32, device="cuda")

    # Logits scratch for fused OProj+Router+TopK kernel
    buf_logits_scratch = torch.zeros(bs, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")

    # Attention split-KV buffers
    o_acc_dim = NUM_KV_HEADS * NUM_KV_CHUNKS * Q_PER_KV * HEAD_DIM
    buf_o_acc_f32 = torch.zeros(bs, o_acc_dim, dtype=torch.float32, device="cuda")
    lse_dim = NUM_KV_HEADS * NUM_KV_CHUNKS * Q_PER_KV
    buf_lse_acc = torch.zeros(bs, lse_dim, dtype=torch.float32, device="cuda")

    # LM head buffers
    buf_argmax_out = torch.zeros(1, dtype=torch.int64, device="cuda")
    buf_lm_norm_scratch = torch.zeros(bs, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    # Counter buffer is a view into buf_merged_counters (allocated above with MoE barrier)

    # Subphase timing buffer (12 slots per layer + 4 tail slots)
    TIMING_SLOTS_PER_LAYER = 14
    TIMING_TAIL_SLOTS = 4
    timing_total = NUM_LAYERS * TIMING_SLOTS_PER_LAYER + TIMING_TAIL_SLOTS
    buf_timing = torch.zeros(timing_total, dtype=torch.int64, device="cuda")

    # Debug: workspace snapshot buffer for layer 0

    # -- Build pointer table --
    # Layout: ptr_table[xcd * NUM_LAYERS * PTRS_PER_LAYER + layer * PTRS_PER_LAYER + slot]
    print("Building pointer table...")

    ptr_table_host = []
    for xcd in range(NUM_XCDS):
        for li in range(NUM_LAYERS):
            base = li * WEIGHTS_PER_LAYER
{emit_weight_unpack()}

            # Per-XCD weight offsets
            qkv_weight_xcd = qkv_weight_base + xcd * QKV_N_WGS_PER_XCD * QKV_WG_BYTES

            # Fleet's QKV weight prefetch hand-off (input_ptrs[24]/[25]).
            #
            # Fleet stages the NEXT layer's QKV weight tile from HBM into LDS
            # during the cross-XCD barrier spin at the end of this layer, when
            # the memory system is otherwise idle. Two pointers drive it:
            #
            #   [24] what to stage = next layer's per-XCD QKV weight. Null on
            #        the last layer: fleet's own comment at :1690 says nothing
            #        may be staged across the iteration boundary, because the
            #        buffer the next iteration reads is not yet published.
            #   [25] whether this layer's tile was already staged. Fleet gates
            #        on `input_ptrs[25] == input_ptrs[4]` (:458) and passes the
            #        result as weights_preloaded, which tells the QKV GEMM to
            #        skip its own DMA and just drain. So [25] must be THIS
            #        layer's own qkv_weight_xcd when the previous layer staged
            #        it, and anything unequal otherwise -- null on layer 0,
            #        which has no previous layer.
            #
            # The per-XCD offset matters: the prefetching worker passes its own
            # xcd_rank as tile_idx, and xcd_rank is stable across layers, so it
            # stages exactly the tile it will itself execute next layer. Handing
            # it the unoffset base would stage XCD 0's tile for all eight.
            next_qkv_weight_xcd = 0
            if li + 1 < NUM_LAYERS:
                next_qkv_weight_xcd = (
                    weight_ptrs_host[(li + 1) * WEIGHTS_PER_LAYER + 0]
                    + xcd * QKV_N_WGS_PER_XCD * QKV_WG_BYTES)
            prefetched_qkv_weight_xcd = qkv_weight_xcd if li > 0 else 0
            # QKV bias per-XCD (10 tiles * 64 = 640 bf16 per XCD)
            qkv_bias_xcd = qkv_bias + xcd * QKV_N_WGS_PER_XCD * OUTPUT_PER_WG * 2  # bf16 = 2 bytes
            # OProj weight per-XCD partitioned (OPW=16, 23 tiles/XCD)
            oproj_weight_xcd = oproj_weight_base + xcd * OPROJ_N_WGS_PER_XCD * OPROJ_WG_BYTES
            # OProj bias per-XCD (23 tiles * 16 = 368 bf16 per XCD)
            oproj_bias_xcd = oproj_bias + xcd * OPROJ_N_WGS_PER_XCD * OPROJ_OPW * 2  # bf16 = 2 bytes
            # Note: OProj residual per-XCD offset computed at runtime in the kernel
            # Router BF16 weight per-XCD (16 experts per XCD, each row = HIDDEN_SIZE bf16)
            router_tile_n = NUM_EXPERTS // NUM_XCDS  # {cfg.num_experts // cfg.num_xcds}
            router_bf16_xcd = router_weight_bf16 + xcd * router_tile_n * HIDDEN_SIZE * 2  # bf16 = 2 bytes
            # Router bias per-XCD (16 experts per XCD)
            router_bias_xcd = router_bias + xcd * router_tile_n * 2  # bf16 = 2 bytes
            # Logits scratch per-XCD (16 experts per XCD)
            logits_scratch_xcd = buf_logits_scratch.data_ptr() + xcd * router_tile_n * 2  # bf16

            # MoE scale bases. NOT per-XCD: every XCD runs every expert, so
            # unlike qkv/oproj there is nothing to partition here.
            #
            # In split mode the slab is [all data][all scales] in one
            # allocation, so the scale section starts one full data region past
            # the base -- the same arithmetic the kernel used to do internally,
            # moved to the host now that the kernel takes the base as an
            # argument. In interleaved mode there IS no scale section, and these
            # resolve to the data base: a mapped address the kernel never
            # dereferences, rather than a null or a stale value that would fault
            # confusingly if the knobs ever disagreed.
            #
            # Under split BUFFERS the scale section is its own allocation and
            # there is no arithmetic at all -- which is the arrangement that lets
            # the data pointer belong to someone else entirely.
            if MOE_SPLIT_BUFFERS:
                w13_scale_base = moe_w13_scales[li].data_ptr()
                w2_scale_base = moe_w2_scales[li].data_ptr()
            else:
                w13_scale_base = w13_weight_base + (
                    NUM_EXPERTS * W13_EXPERT_BYTES if MOE_SPLIT_SCALES else 0)
                w2_scale_base = w2_weight_base + (
                    NUM_EXPERTS * W2_EXPERT_BYTES if MOE_SPLIT_SCALES else 0)

            # Counter buffer. ONE block shared by every layer -- NOT `+ li *
            # COUNTERS_PER_LAYER`, which is what this used to be.
            #
            # Fleet's layer body derives every barrier's release value from the
            # layer index rather than snapshotting a shared counter:
            #     layer_counter = task_layer_idx
            #     qkv_epoch_expected = attn_release_expected = layer_counter + 1
            # and its counters are monotonic, never reset (fleet allocates a
            # single `oproj_topk_counters` tensor and passes that same tensor at
            # every call site; see fleet demo/gpt_oss/demo.py:1235,1615,1685).
            # Layer L therefore waits for its epoch counter to reach L+1, having
            # been left at L by layer L-1 *in the same memory*.
            #
            # Giving each layer its own zeroed block breaks exactly that. Layer
            # 2's counter starts at 0, one arrival takes it to 1, and every
            # worker waits forever for 2. That is not a hypothesis: with
            # -DFLEET_MK_WORKER_STATE the mid-hang dump reads
            #   xcd=* phase=20 layer=2 barrier=20 observed=1 expected=2 -> 10 workers
            #   xcd=* phase=60 layer=2 barrier=60 observed=0 expected=2 -> 20 workers
            # on all 8 XCDs -- fleet's Phase 2 QKV epoch and Phase 6 attn
            # release, short by precisely the arrivals the previous layer would
            # have contributed had it shared the block.
            #
            # The per-layer stride was correct for fleet_mk's OWN barriers, which
            # zero and reuse their slots; it is incompatible with fleet's
            # monotonic-counter contract. Sharing one block is what fleet does.
            counter_ptr = buf_counter.data_ptr()

            # Mirage buffer layout (no ping-pong):
            #   output_ptrs[0] = buf_x_output (separate QKV intermediate, like mirage's mlp_weighted_sum_out)
            #   output_ptrs[5] = buf_oproj_out (residual after OProj, shared across all layers)
            #   input_ptrs[1]  = buf_residual for layer 0, buf_oproj_out for layers 1+

            # Pre-computed mirage format: {len(MIRAGE_IN)} mirage_in + {len(MIRAGE_OUT)} mirage_out + 1 counter = {MIRAGE_PTRS_PER_LAYER} ptrs
            # Eliminates all conditional remapping in the kernel.
            qkv_barrier_ptr = counter_ptr + SLOT_QKV_BARRIER * 4  # byte offset

            mirage_in = [
{emit_mirage_ptr_list('in')}
            ]

            mirage_out = [
{emit_mirage_ptr_list('out')}
            ]

            # Last entry: layer_output for last-layer ResAdd
            layer_output_entry = [buf_oproj_out.data_ptr()]

            assert len(mirage_in) == {len(MIRAGE_IN)}
            assert len(mirage_out) == {len(MIRAGE_OUT)}
            ptr_table_host.extend(mirage_in)
            ptr_table_host.extend(mirage_out)
            ptr_table_host.extend(layer_output_entry)

    assert len(ptr_table_host) == NUM_XCDS * NUM_LAYERS * PTRS_PER_LAYER
    ptr_table_tensor = torch.tensor(ptr_table_host, dtype=torch.long, device="cuda")
    print(f"  Pointer table: {{len(ptr_table_host)}} entries "
          f"({{NUM_XCDS}}x{{NUM_LAYERS}}x{{PTRS_PER_LAYER}})")

    rank_counter_offset = NUM_LAYERS * COUNTERS_PER_LAYER
    rank_counter_slice = buf_counter[rank_counter_offset:]

    # -- Paged attention index arrays --
    qo_indptr = torch.zeros(2, dtype=torch.int32, device="cuda")
    qo_indptr[1] = 1
    kv_indptr = torch.zeros(2, dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(max_num_pages, dtype=torch.int32, device="cuda")
    kv_last_page_len = torch.zeros(1, dtype=torch.int32, device="cuda")

    # -- Load fleet_mk kernel --
    fleet_mk_so = args.fleet_mk_so or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "generated", "{cfg.name_clean}.so")
    print(f"Loading fleet_mk kernel from: {{fleet_mk_so}}")
    fleet_mk = load_{cfg.name_clean}_kernel(fleet_mk_so)
    fleet_mk.{cfg.name_clean}_init()

    # -- Inference loop --
    stream = torch.cuda.current_stream()
    attn_scale = (1.0 / math.sqrt(HEAD_DIM)) * 1.44269504088896340736

    # Prefill via PyTorch
    _model_device = next(model.parameters()).device
    print(f"\\n=== Prefill ({{prompt_len}} tokens) via PyTorch (device={{_model_device}}) ===")
    past_key_values = None
    with torch.no_grad():
        input_ids = tokens[:prompt_len].unsqueeze(0).to(_model_device)
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits

    next_token = logits[:, -1, :VOCAB_SIZE].argmax(dim=-1).item()
    tokens[prompt_len] = next_token
    print(f"Prefill complete. First token: {{next_token}} "
          f"('{{tokenizer.decode([next_token])}}')")

    # Copy PyTorch KV cache into paged KV cache buffers
    if not args.pytorch_only:
        print("Converting KV cache to paged format...")
        for li in range(NUM_LAYERS):
            k_pt = past_key_values[li][0]  # [1, num_kv_heads, seq_len, head_dim]
            v_pt = past_key_values[li][1]
            seq_len_cached = k_pt.shape[2]
            k_flat = k_pt[0].permute(1, 0, 2).reshape(seq_len_cached, -1)
            v_flat = v_pt[0].permute(1, 0, 2).reshape(seq_len_cached, -1)
            buf_k_cache[li][:seq_len_cached] = k_flat.to(torch.bfloat16)
            buf_v_cache[li][:seq_len_cached] = v_flat.to(torch.bfloat16)
        print(f"  Copied {{seq_len_cached}} positions into paged KV cache")

    # -- Decode --
    output_len = args.max_new_tokens if args.max_new_tokens else (
        args.max_seq_length - prompt_len - 1)
    output_len = min(output_len, args.max_seq_length - prompt_len - 1)

    print(f"\\n=== Decode ({{output_len}} tokens) ===")

    if args.pytorch_only:
        # PyTorch path: per-token loop
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        decode_times = []

        for di in range(output_len):
            cur_pos = prompt_len + di
            cur_token = tokens[cur_pos].item()

            if cur_token == tokenizer.eos_token_id and not args.ignore_eos:
                print(f"EOS at position {{cur_pos}}")
                break

            if di >= args.warmup:
                starter.record()

            input_ids_t = torch.tensor([[cur_token]], device=_model_device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids_t,
                              past_key_values=past_key_values,
                              use_cache=True)
                past_key_values = outputs.past_key_values
                logits = outputs.logits
            next_token = logits[:, -1, :VOCAB_SIZE].argmax(dim=-1).item()

            if di >= args.warmup:
                ender.record()
                torch.cuda.synchronize()
                elapsed = starter.elapsed_time(ender)
                decode_times.append(elapsed)
            else:
                torch.cuda.synchronize()

            if di < 5:
                print(f"  iter {{di + 1}}: next_token={{next_token}} "
                      f"('{{tokenizer.decode([next_token])}}')")
            tokens[cur_pos + 1] = next_token

        end_pos = prompt_len + min(di + 1, output_len)
        generated_ids = tokens[:end_pos + 1]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"\\n{{'=' * 60}}")
        print(response)
        print(f"{{'=' * 60}}")

        gen_tokens = len(decode_times)
        if gen_tokens > 0:
            avg_ms = sum(decode_times) / gen_tokens
            print(f"\\nPrompt: {{prompt_len}} tokens")
            print(f"Generated: {{gen_tokens}} tokens (after {{args.warmup}} warmup)")
            print(f"Decode avg: {{avg_ms:.3f}} ms/token")
            print(f"Decode min: {{min(decode_times):.3f}} ms/token")
            print(f"Decode max: {{max(decode_times):.3f}} ms/token")
    else:
        # Fleet MK kernel path: per-token launch loop with hipGraph
        import struct as st

        import time as _time

        cur_token = tokens[prompt_len].item()
        cur_pos = prompt_len

        use_decode_loop = os.environ.get("FLEET_MK_DECODE_LOOP", "0") == "1"

        if use_decode_loop:
            # ============================================================
            # C-SIDE DECODE LOOP: kernel launch + argmax readback in C
            # Eliminates Python per-token overhead entirely.
            # ============================================================
            # Host-pinned buffer for output tokens (C side writes directly)
            token_output_host = torch.zeros(output_len, dtype=torch.int32).pin_memory()

            buf_decode_ctrl = torch.zeros(48, dtype=torch.uint8, device="cuda")

            # Warmup with single-step launches
            for wi in range(args.warmup):
                num_pages_used = (cur_pos + page_size) // page_size
                kv_indptr[1] = num_pages_used
                kv_last_page_len[0] = (cur_pos % page_size) + 1
                fleet_mk.{cfg.name_clean}_launch(
                    1, attn_scale,
                    cos_padded.data_ptr(), sin_padded.data_ptr(),
                    qo_indptr.data_ptr(), kv_indptr.data_ptr(),
                    kv_indices.data_ptr(), kv_last_page_len.data_ptr(),
                    ptr_table_tensor.data_ptr(), buf_counter.data_ptr(),
                    final_norm_w.data_ptr(), buf_lm_norm_scratch.data_ptr(),
                    lm_head_packed.data_ptr(), lm_head_zero_bias.data_ptr(),
                    buf_argmax_out.data_ptr(),
                    None,  # logits_output (argmax-only; vLLM passes a buffer)
                    buf_timing.data_ptr(),
                    embed_weight.data_ptr(), cur_token,
                    buf_decode_ctrl.data_ptr(),
                    stream.cuda_stream,
                )
                torch.cuda.synchronize()
                next_token = buf_argmax_out[0].item()
                print(f"  warmup {{wi+1}}: next_token={{next_token}} "
                      f"('{{tokenizer.decode([next_token])}}')")
                tokens[cur_pos + 1] = next_token
                cur_token = next_token
                cur_pos += 1

            remaining = output_len - args.warmup

            wall_t0 = _time.perf_counter()

            # C-side decode loop: launches kernel repeatedly from C
            fleet_mk.{cfg.name_clean}_decode_loop(
                1, attn_scale,
                cos_padded.data_ptr(), sin_padded.data_ptr(),
                qo_indptr.data_ptr(), kv_indptr.data_ptr(),
                kv_indices.data_ptr(), kv_last_page_len.data_ptr(),
                ptr_table_tensor.data_ptr(), buf_counter.data_ptr(),
                final_norm_w.data_ptr(), buf_lm_norm_scratch.data_ptr(),
                lm_head_packed.data_ptr(), lm_head_zero_bias.data_ptr(),
                buf_argmax_out.data_ptr(),
                None,  # logits_output (argmax-only; vLLM passes a buffer)
                buf_timing.data_ptr(),
                embed_weight.data_ptr(), cur_token,
                buf_decode_ctrl.data_ptr(),
                stream.cuda_stream,
                # Decode loop params
                remaining,
                cur_pos,
                page_size,
                ctypes.cast(token_output_host.data_ptr(), ctypes.c_void_p),
            )

            wall_t1 = _time.perf_counter()
            wall_elapsed = (wall_t1 - wall_t0) * 1000

            # Read generated tokens from host-pinned buffer
            gen_tokens = token_output_host[:remaining].tolist()
            actual_gen = len(gen_tokens)
            print(f"\\n  C-side decode loop generated {{actual_gen}} tokens")
            for gi, gtok in enumerate(gen_tokens[:5]):
                print(f"  token {{gi}}: {{gtok}} ('{{tokenizer.decode([gtok])}}')")

            # Reconstruct token sequence
            for gi, gtok in enumerate(gen_tokens):
                tokens[cur_pos + 1 + gi] = gtok
            end_pos = cur_pos + actual_gen
            generated_ids = tokens[:end_pos + 1]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"\\n{{'=' * 60}}")
            print(response)
            print(f"{{'=' * 60}}")

            wall_avg = wall_elapsed / actual_gen if actual_gen > 0 else 0
            print(f"\\nPrompt: {{prompt_len}} tokens")
            print(f"Generated: {{actual_gen}} tokens (C-side loop, {{args.warmup}} warmup)")
            print(f"Wall total: {{wall_elapsed:.3f}} ms")
            print(f"Wall avg: {{wall_avg:.3f}} ms/token")

        elif os.environ.get("FLEET_MK_PIPE", "0") == "1":
            # ============================================================
            # PIPELINED GRAPH: queue ALL decode steps, no host sync between
            # Bridge kernel handles argmax→token + KV metadata between steps
            # ============================================================
            buf_decode_ctrl = torch.zeros(48, dtype=torch.uint8, device="cuda")

            # GPU-side token pointer and step counter
            buf_cur_token = torch.zeros(1, dtype=torch.int32, device="cuda")
            buf_step_counter = torch.zeros(1, dtype=torch.int32, device="cuda")
            token_output_buf = torch.zeros(output_len, dtype=torch.int32, device="cuda")

            moe_barrier_bytes = buf_moe_barrier.nelement() * buf_moe_barrier.element_size()
            workspace_bytes = buf_moe_workspace_f32.nelement() * buf_moe_workspace_f32.element_size()

            # Warmup with regular launches to establish correct state
            for wi in range(args.warmup):
                num_pages_used = (cur_pos + page_size) // page_size
                kv_indptr[1] = num_pages_used
                kv_last_page_len[0] = (cur_pos % page_size) + 1
                fleet_mk.{cfg.name_clean}_launch(
                    1, attn_scale,
                    cos_padded.data_ptr(), sin_padded.data_ptr(),
                    qo_indptr.data_ptr(), kv_indptr.data_ptr(),
                    kv_indices.data_ptr(), kv_last_page_len.data_ptr(),
                    ptr_table_tensor.data_ptr(), buf_counter.data_ptr(),
                    final_norm_w.data_ptr(), buf_lm_norm_scratch.data_ptr(),
                    lm_head_packed.data_ptr(), lm_head_zero_bias.data_ptr(),
                    buf_argmax_out.data_ptr(),
                    None,  # logits_output (argmax-only; vLLM passes a buffer)
                    buf_timing.data_ptr(),
                    embed_weight.data_ptr(), cur_token,
                    buf_decode_ctrl.data_ptr(),
                    stream.cuda_stream,
                )
                torch.cuda.synchronize()
                next_token = buf_argmax_out[0].item()
                print(f"  warmup {{wi+1}}: next_token={{next_token}} "
                      f"('{{tokenizer.decode([next_token])}}')")
                tokens[cur_pos + 1] = next_token
                cur_token = next_token
                cur_pos += 1

            remaining = output_len - args.warmup

            # Set initial state for pipe: write cur_token to GPU buffer
            buf_cur_token[0] = cur_token
            # Update KV metadata for first pipe step
            num_pages_used = (cur_pos + page_size) // page_size
            kv_indptr[1] = num_pages_used
            kv_last_page_len[0] = (cur_pos % page_size) + 1
            torch.cuda.synchronize()

            # Capture pipelined graph
            fleet_mk.{cfg.name_clean}_pipe_capture(
                1, attn_scale,
                cos_padded.data_ptr(), sin_padded.data_ptr(),
                qo_indptr.data_ptr(), kv_indptr.data_ptr(),
                kv_indices.data_ptr(), kv_last_page_len.data_ptr(),
                ptr_table_tensor.data_ptr(), buf_counter.data_ptr(),
                final_norm_w.data_ptr(), buf_lm_norm_scratch.data_ptr(),
                lm_head_packed.data_ptr(), lm_head_zero_bias.data_ptr(),
                buf_argmax_out.data_ptr(),
                None,  # logits_output (argmax-only; vLLM passes a buffer)
                buf_timing.data_ptr(),
                embed_weight.data_ptr(), cur_token,
                buf_decode_ctrl.data_ptr(),
                stream.cuda_stream,
                # Bridge kernel args
                buf_moe_barrier.data_ptr(),
                buf_moe_barrier.nelement(),
                buf_cur_token.data_ptr(),
                token_output_buf.data_ptr(),
                buf_step_counter.data_ptr(),
                cur_pos,
                page_size,
                # Workspace
                buf_moe_workspace_f32.data_ptr(),
                workspace_bytes,
            )

            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            # Queue ALL decode steps at once — no host sync between!
            starter.record()
            fleet_mk.{cfg.name_clean}_pipe_launch_all(
                stream.cuda_stream,
                remaining,
            )
            ender.record()
            torch.cuda.synchronize()

            gpu_elapsed = starter.elapsed_time(ender)
            gpu_avg = gpu_elapsed / remaining if remaining > 0 else 0

            # Read generated tokens from GPU buffer
            gen_tokens = token_output_buf[:remaining].cpu().tolist()
            for gi, gtok in enumerate(gen_tokens[:5]):
                print(f"  token {{gi}}: {{gtok}} ('{{tokenizer.decode([gtok])}}')")

            # Reconstruct token sequence
            for gi, gtok in enumerate(gen_tokens):
                tokens[cur_pos + 1 + gi] = gtok
            end_pos = cur_pos + remaining
            generated_ids = tokens[:end_pos + 1]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"\\n{{'=' * 60}}")
            print(response)
            print(f"{{'=' * 60}}")

            print(f"\\nPrompt: {{prompt_len}} tokens")
            print(f"Generated: {{remaining}} tokens (pipelined graph, {{args.warmup}} warmup)")
            print(f"GPU total: {{gpu_elapsed:.3f}} ms")
            print(f"GPU avg: {{gpu_avg:.3f}} ms/token (zero host overhead)")

            fleet_mk.{cfg.name_clean}_pipe_destroy()

        else:
            # ============================================================
            # PER-TOKEN LAUNCH LOOP (legacy path)
            # ============================================================
            # DecodeControl. Zeroed, so iter_base_p1 == 0 and the kernel derives
            # its decode epoch from the rank counter exactly as it always has.
            buf_decode_ctrl = torch.zeros(48, dtype=torch.uint8, device="cuda")
            # int32 view for iter_base_p1, field 3 (after max_decode_tokens,
            # start_pos, eos_token_id).
            decode_ctrl_i32 = buf_decode_ctrl.view(torch.int32)

            # FLEET_MK_HOST_EPOCH=1 hands the kernel the decode epoch from here
            # instead of letting it divide the rank counter.
            #
            # This exists to be an EQUIVALENCE TEST, not a feature: the two
            # derivations must agree token for token, and the whole point of
            # the field is that the atomic's quotient stops being correct once
            # one launch covers several decode steps. There is no way to trust
            # it in that setting without first exercising it in this one, where
            # a known-good answer is available to disagree with.
            #
            # The value is the loop index because prefill runs on PyTorch and
            # never launches this kernel, so decode step di is the di'th launch
            # and the counter's quotient is exactly di. Plus one for the bias.
            host_epoch = os.environ.get("FLEET_MK_HOST_EPOCH", "0") == "1"
            if host_epoch:
                print("  [host epoch] iter_base_p1 supplied by driver")

            # We need a GPU-side token ID buffer so we can update it between graph launches
            buf_cur_token = torch.zeros(1, dtype=torch.int32, device="cuda")

            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            decode_times = []
            wall_times = []

            # Helper to launch one decode step
            def launch_one_step(cur_tok_id):
                fleet_mk.{cfg.name_clean}_launch(
                    1, attn_scale,
                    cos_padded.data_ptr(), sin_padded.data_ptr(),
                    qo_indptr.data_ptr(), kv_indptr.data_ptr(),
                    kv_indices.data_ptr(), kv_last_page_len.data_ptr(),
                    ptr_table_tensor.data_ptr(), buf_counter.data_ptr(),
                    final_norm_w.data_ptr(), buf_lm_norm_scratch.data_ptr(),
                    lm_head_packed.data_ptr(), lm_head_zero_bias.data_ptr(),
                    buf_argmax_out.data_ptr(),
                    None,  # logits_output (argmax-only; vLLM passes a buffer)
                    buf_timing.data_ptr(),
                    embed_weight.data_ptr(), cur_tok_id,
                    buf_decode_ctrl.data_ptr(),
                    stream.cuda_stream,
                )

            use_graph = os.environ.get("FLEET_MK_GRAPH", "0") == "1"
            graph_captured = False

            # Buffer sizes for graph capture
            moe_barrier_bytes = buf_moe_barrier.nelement() * buf_moe_barrier.element_size()
            workspace_bytes = buf_moe_workspace_f32.nelement() * buf_moe_workspace_f32.element_size()

            # ============================================================
            # CHUNKED PERSISTENT LOOP (FLEET_MK_PERSIST=N)
            # ============================================================
            # One launch covers N decode steps. A rocprofv3 kernel+HIP-runtime
            # trace of the per-token path measured, on one clock, 2024.9 us
            # inside the dispatch against a 2213.6 us token cadence -- 188.0 us
            # of dead time between dispatches, plus ~132 us inside the dispatch
            # but outside s_memrealtime (wave ramp-up and drain for 240
            # workgroups, which the device timer cannot see). That ~320 us/token
            # is paid per LAUNCH, so covering N steps per launch should amortise
            # it to ~320/N.
            #
            # N == 1 falls through to the per-token loop below untouched, so the
            # two arms are the same binary and the A/B is clean.
            persist_n = int(os.environ.get("FLEET_MK_PERSIST", "1"))
            if persist_n > 1:
                print(f"  [persist] {{persist_n}} decode steps per launch")
                # The kernel writes every token but the last one here; the last
                # is still in argmax_output, which is where the next chunk's
                # first embedding also reads it from.
                buf_tokens_out = torch.zeros(persist_n, dtype=torch.int32,
                                             device="cuda")
                # Pointer half of DecodeControl: 4 int32 (16 B) then 4 pointers,
                # so as int64 the pointers start at index 2.
                decode_ctrl_i64 = buf_decode_ctrl.view(torch.int64)
                decode_ctrl_i64[2] = kv_indptr.data_ptr()
                decode_ctrl_i64[3] = kv_last_page_len.data_ptr()
                decode_ctrl_i64[4] = buf_tokens_out.data_ptr()
                decode_ctrl_i64[5] = 0   # num_generated: kernel does not write it
                decode_ctrl_i32[2] = -1  # eos_token_id: the kernel does not stop

                di = 0
                while di < output_len:
                    if cur_token == tokenizer.eos_token_id and not args.ignore_eos:
                        print(f"EOS at position {{cur_pos}}")
                        break
                    n_this = min(persist_n, output_len - di)

                    wall_t0 = _time.perf_counter()

                    # Step 0's KV metadata is the host's job -- the kernel only
                    # advances it for iter > 0. Same two writes as the per-token
                    # path, once per chunk instead of once per token.
                    kv_indptr[1] = (cur_pos + page_size) // page_size
                    kv_last_page_len[0] = (cur_pos % page_size) + 1

                    decode_ctrl_i32[0] = n_this   # max_decode_tokens
                    decode_ctrl_i32[1] = cur_pos  # start_pos
                    # The epoch the kernel cannot derive itself: rank/30 counts
                    # LAUNCHES, and a launch is now n_this steps.
                    decode_ctrl_i32[3] = di + 1   # iter_base_p1

                    timed = di >= args.warmup
                    if timed:
                        starter.record()
                    launch_one_step(cur_token)
                    if timed:
                        ender.record()
                    torch.cuda.synchronize()

                    toks = buf_tokens_out[:n_this - 1].tolist() if n_this > 1 else []
                    toks.append(buf_argmax_out[0].item())

                    if timed:
                        # Charge the launch evenly across the tokens it produced.
                        # Per-token GPU time is not separable inside one launch,
                        # which is the entire point of the change.
                        per_tok = starter.elapsed_time(ender) / n_this
                        wall_t1 = _time.perf_counter()
                        per_tok_wall = (wall_t1 - wall_t0) * 1000 / n_this
                        for _ in range(n_this):
                            decode_times.append(per_tok)
                            wall_times.append(per_tok_wall)

                    for k, nt in enumerate(toks):
                        if di + k < 5:
                            print(f"  iter {{di + k + 1}}: next_token={{nt}} "
                                  f"('{{tokenizer.decode([nt])}}')")
                        tokens[cur_pos + 1] = nt
                        cur_token = nt
                        cur_pos += 1
                    di += n_this

            # persist_n > 1 already generated everything; range(0) skips this
            # loop without re-indenting it, so the legacy path stays as written.
            for di in range(0 if persist_n > 1 else output_len):
                if cur_token == tokenizer.eos_token_id and not args.ignore_eos:
                    print(f"EOS at position {{cur_pos}}")
                    break

                wall_t0 = _time.perf_counter()

                # Update KV metadata
                num_pages_used = (cur_pos + page_size) // page_size
                kv_indptr[1] = num_pages_used
                kv_last_page_len[0] = (cur_pos % page_size) + 1

                if host_epoch:
                    decode_ctrl_i32[3] = di + 1

                if use_graph and graph_captured:
                    # Graph replay: zero + kernel in one submission
                    if di >= args.warmup:
                        starter.record()
                    fleet_mk.{cfg.name_clean}_graph_launch(
                        cur_token,
                        stream.cuda_stream,
                    )
                elif use_graph and di == args.warmup - 1:
                    # Last warmup: capture graph
                    fleet_mk.{cfg.name_clean}_graph_capture(
                        1, attn_scale,
                        cos_padded.data_ptr(), sin_padded.data_ptr(),
                        qo_indptr.data_ptr(), kv_indptr.data_ptr(),
                        kv_indices.data_ptr(), kv_last_page_len.data_ptr(),
                        ptr_table_tensor.data_ptr(), buf_counter.data_ptr(),
                        final_norm_w.data_ptr(), buf_lm_norm_scratch.data_ptr(),
                        lm_head_packed.data_ptr(), lm_head_zero_bias.data_ptr(),
                        buf_argmax_out.data_ptr(),
                        None,  # logits_output (argmax-only; vLLM passes a buffer)
                        buf_timing.data_ptr(),
                        embed_weight.data_ptr(), cur_token,
                        buf_decode_ctrl.data_ptr(),
                        stream.cuda_stream,
                        buf_moe_barrier.data_ptr(), moe_barrier_bytes,
                        buf_moe_workspace_f32.data_ptr(), workspace_bytes,
                        0, 0,  # counter buf: monotonic, no zeroing needed
                    )
                    graph_captured = True
                    print(f"  [hipGraph captured at warmup iter {{di + 1}}]")
                    # Now replay the captured graph for this iteration
                    fleet_mk.{cfg.name_clean}_graph_launch(
                        cur_token,
                        stream.cuda_stream,
                    )
                else:
                    # Regular launch (warmup or non-graph mode)
                    # MoE barrier: zeroed by kernel at end of each step (first iter uses pre-zeroed buf)
                    # Counter buf: monotonic counters, no zeroing needed
                    # buf_moe_workspace_f32: zeroed by kernel's ResAdd (last layer)
                    if di >= args.warmup:
                        starter.record()
                    launch_one_step(cur_token)

                if di >= args.warmup:
                    ender.record()
                    torch.cuda.synchronize()
                    elapsed = starter.elapsed_time(ender)
                    decode_times.append(elapsed)
                    wall_t1 = _time.perf_counter()
                    wall_times.append((wall_t1 - wall_t0) * 1000)
                else:
                    torch.cuda.synchronize()

                next_token = buf_argmax_out[0].item()
                if di < 5:
                    print(f"  iter {{di + 1}}: next_token={{next_token}} "
                          f"('{{tokenizer.decode([next_token])}}')")

                tokens[cur_pos + 1] = next_token
                cur_token = next_token
                cur_pos += 1

            end_pos = cur_pos
            generated_ids = tokens[:end_pos + 1]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"\\n{{'=' * 60}}")
            print(response)
            print(f"{{'=' * 60}}")

            gen_tokens = len(decode_times)
            if gen_tokens > 0:
                avg_ms = sum(decode_times) / gen_tokens
                # Count outliers (>5ms)
                outliers = [t for t in decode_times if t > 5.0]
                non_outlier = [t for t in decode_times if t <= 5.0]
                print(f"\\nPrompt: {{prompt_len}} tokens")
                print(f"Generated: {{gen_tokens}} tokens (after {{args.warmup}} warmup)")
                print(f"Decode avg: {{avg_ms:.3f}} ms/token")
                print(f"Decode min: {{min(decode_times):.3f}} ms/token")
                print(f"Decode max: {{max(decode_times):.3f}} ms/token")
                if outliers:
                    print(f"Outliers (>5ms): {{len(outliers)}} tokens, "
                          f"avg={{sum(outliers)/len(outliers):.1f}}ms")
                if non_outlier:
                    print(f"Non-outlier avg: {{sum(non_outlier)/len(non_outlier):.3f}} ms/token")
                if wall_times:
                    wall_non_outlier = [w for w in wall_times if w <= 5.0]
                    if wall_non_outlier:
                        print(f"Wall avg (non-outlier): {{sum(wall_non_outlier)/len(wall_non_outlier):.3f}} ms/token")
                        gpu_non_outlier = [t for t, w in zip(decode_times, wall_times) if w <= 5.0]
                        if gpu_non_outlier:
                            overhead = sum(wall_non_outlier)/len(wall_non_outlier) - sum(gpu_non_outlier)/len(gpu_non_outlier)
                            print(f"Python overhead: {{overhead*1000:.0f}} us/token")

    # -- Subphase timing readback --
    if os.environ.get("FLEET_MK_SUBPHASE_TIMING"):
        timing_data = buf_timing.cpu().numpy()
        # GPU clock is 100MHz on MI350 (10ns ticks)
        GPU_CLOCK_NS = 10.0
        print(f"\\n{{'=' * 60}}")
        print("SUBPHASE TIMING (last decode step, XCD0/worker0)")
        print(f"{{'=' * 60}}")
        phase_names = [
            "layer_start", "after_rmsnorm1", "after_qkv_barrier",
            "after_rope", "after_attn_compute", "after_attn_barrier",
            "after_oproj_gemm", "after_ffn_mid_barrier", "after_router_gemm",
            "after_topk", "after_w13", "after_w13_barrier",
            "after_w2", "after_w2_barrier"
        ]
        # Print per-layer breakdown for layers 0, 1, last
        for li in [0, 1, NUM_LAYERS - 1]:
            base = li * TIMING_SLOTS_PER_LAYER
            timestamps = timing_data[base:base + TIMING_SLOTS_PER_LAYER]
            if timestamps[0] == 0:
                print(f"  Layer {{li}}: no timing data")
                continue
            print(f"\\n  Layer {{li}}:")
            for pi in range(1, TIMING_SLOTS_PER_LAYER):
                if timestamps[pi] == 0:
                    break
                delta_ns = (timestamps[pi] - timestamps[pi-1]) * GPU_CLOCK_NS
                delta_us = delta_ns / 1000.0
                print(f"    {{phase_names[pi-1]:30s}} -> {{phase_names[pi]:30s}}: {{delta_us:8.1f}} us")
            total_ns = (timestamps[TIMING_SLOTS_PER_LAYER-1] - timestamps[0]) * GPU_CLOCK_NS
            print(f"    {{'TOTAL':62s}}: {{total_ns/1000.0:8.1f}} us")

        # Print average across all layers
        print(f"\\n  Average across {{NUM_LAYERS}} layers:")
        avg_deltas = [0.0] * (TIMING_SLOTS_PER_LAYER - 1)
        valid_layers = 0
        for li in range(NUM_LAYERS):
            base = li * TIMING_SLOTS_PER_LAYER
            timestamps = timing_data[base:base + TIMING_SLOTS_PER_LAYER]
            if timestamps[0] == 0 or timestamps[-1] == 0:
                continue
            valid_layers += 1
            for pi in range(1, TIMING_SLOTS_PER_LAYER):
                avg_deltas[pi-1] += (timestamps[pi] - timestamps[pi-1]) * GPU_CLOCK_NS / 1000.0
        if valid_layers > 0:
            total_avg = 0.0
            for pi in range(TIMING_SLOTS_PER_LAYER - 1):
                avg_deltas[pi] /= valid_layers
                total_avg += avg_deltas[pi]
                print(f"    {{phase_names[pi]:30s}} -> {{phase_names[pi+1]:30s}}: {{avg_deltas[pi]:8.1f}} us")
            print(f"    {{'TOTAL':62s}}: {{total_avg:8.1f}} us")
            print(f"    {{'TOTAL x {cfg.num_layers} layers':62s}}: {{total_avg * {cfg.num_layers} / 1000.0:8.3f}} ms")

    fleet_mk.{cfg.name_clean}_finalize()
    print("\\nDone.")


if __name__ == "__main__":
    main()
'''


def generate_driver_dense(cfg: ModelConfig) -> str:
    """Dense driver (Qwen3, Llama3)."""
    nc = cfg.name_clean
    ns = nc  # namespace name (same as name_clean)
    nt = cfg.name_title

    # Build QK norm weight packing + pointer entries conditionally
    if cfg.has_qk_norm:
        qk_norm_pack_code = f"""\
        # -- QK normalization weights --
        q_norm_w = layer.self_attn.q_norm.weight.data.to("cuda").contiguous()
        weight_tensors.append(q_norm_w)  # [{10}] q_norm_weight
        k_norm_w = layer.self_attn.k_norm.weight.data.to("cuda").contiguous()
        weight_tensors.append(k_norm_w)  # [{11}] k_norm_weight
"""
        weights_per_layer = 12
        qk_norm_ptr_lines = f"""\
            q_norm_w = weight_ptrs_host[base + 10]
            k_norm_w = weight_ptrs_host[base + 11]
"""
        qk_norm_input_entries = f"""\
                q_norm_w,                        # [16] Q_NORM_WEIGHT
                k_norm_w,                        # [17] K_NORM_WEIGHT
"""
    else:
        qk_norm_pack_code = ""
        weights_per_layer = 10
        qk_norm_ptr_lines = ""
        qk_norm_input_entries = ""

    # Layer attribute paths for weight loading
    # Standard transformer layer structure
    qkv_bias_check = """\
        qkv_bias = torch.zeros(QKV_OUTPUT_SIZE, dtype=torch.bfloat16, device="cuda")
        if layer.self_attn.q_proj.bias is not None:
            q_bias = layer.self_attn.q_proj.bias.data.to("cuda")
            k_bias = layer.self_attn.k_proj.bias.data.to("cuda")
            v_bias = layer.self_attn.v_proj.bias.data.to("cuda")
            bias_chunks = []
            for g in range(NUM_KV_HEADS):
                bias_chunks.append(q_bias[g * Q_PER_KV * HEAD_DIM:(g + 1) * Q_PER_KV * HEAD_DIM])
                bias_chunks.append(k_bias[g * HEAD_DIM:(g + 1) * HEAD_DIM])
                bias_chunks.append(v_bias[g * HEAD_DIM:(g + 1) * HEAD_DIM])
            qkv_bias = torch.cat(bias_chunks, dim=0).contiguous()
        weight_tensors.append(qkv_bias)  # [1] qkv_bias"""

    return f'''\
#!/usr/bin/env python3
"""Auto-generated by fleet_mk_generate.py
Fleet MK demo for {cfg.name} inference using two-level fusion.

Loads {cfg.name} weights from HuggingFace, quantizes to MXFP4, builds the
pointer table for the {cfg.name} persistent kernel, runs prefill via PyTorch,
then decode via the fleet_mk kernel.

Usage:
    HIP_VISIBLE_DEVICES=0 python3 demo_{nc}.py \\
        --model {nt}/{cfg.name.upper().replace("-", "-")} \\
        --prompt "Tell me the history of america" --max-seq-length {cfg.max_seq_len}
"""

import argparse
import ctypes
import gc
import json
import math
import os
import sys
import time

import torch
import yaml
from safetensors import safe_open

# Add mirage demo directory for weight packing utilities
MIRAGE_DIR = os.environ.get("MIRAGE_DIR", "/home/claudeuser/mirage")
sys.path.insert(0, os.path.join(MIRAGE_DIR, "demo/gpt_oss"))

# Import weight packing from mirage's GPT-OSS demo
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "gptoss_demo", os.path.join(MIRAGE_DIR, "demo/gpt_oss/demo.py"))
_gptoss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gptoss)
pad_weight_1d = _gptoss.pad_weight_1d
pad_weight_2d = _gptoss.pad_weight_2d
quantize_bf16_to_mxfp4 = _gptoss.quantize_bf16_to_mxfp4
pack_mxfp4_workgroup = _gptoss.pack_mxfp4_workgroup

from transformers import AutoTokenizer, AutoModelForCausalLM

# -- {cfg.name} constants --
HIDDEN_SIZE = {cfg.padded_hidden_size}
ACTUAL_HIDDEN_DIM = {cfg.hidden_size}  # for RMSNorm mean
INTERMEDIATE_SIZE = {cfg.padded_intermediate_size}
VOCAB_SIZE = {cfg.vocab_size}
PADDED_VOCAB_SIZE = {cfg.padded_vocab_size}  # next multiple for LM head split
NUM_LAYERS = {cfg.num_layers}
NUM_Q_HEADS = {cfg.num_q_heads}
NUM_KV_HEADS = {cfg.num_kv_heads}
HEAD_DIM = {cfg.head_dim}
Q_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS  # {cfg.q_per_kv}
QKV_OUTPUT_SIZE = NUM_Q_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM  # {cfg.qkv_output_size}

# All GEMMs use OPW={cfg.output_per_wg}
OUTPUT_PER_WG = {cfg.output_per_wg}
NUM_XCDS = {cfg.num_xcds}
WORKERS_PER_XCD = {cfg.workers_per_xcd}
TOTAL_WORKERS = NUM_XCDS * WORKERS_PER_XCD
NUM_KV_CHUNKS = {cfg.num_kv_chunks}

# Per-XCD workgroup counts
QKV_N_WGS_PER_XCD = (QKV_OUTPUT_SIZE // OUTPUT_PER_WG) // NUM_XCDS  # {cfg.qkv_n_wgs_per_xcd}
OPROJ_REDUCTION = NUM_Q_HEADS * HEAD_DIM  # {cfg.oproj_reduction}
OPROJ_OPW = {cfg.oproj_opw}
OPROJ_N_WGS_PER_XCD = (HIDDEN_SIZE // OPROJ_OPW) // NUM_XCDS  # {cfg.oproj_n_wgs_per_xcd}
{f"""GATEUP_OUTPUT_SIZE = 2 * INTERMEDIATE_SIZE  # {cfg.gateup_output_size}
GATEUP_OPW = {cfg.gateup_opw}
GATEUP_N_WGS_PER_XCD = (GATEUP_OUTPUT_SIZE // GATEUP_OPW) // NUM_XCDS  # {cfg.gateup_n_wgs_per_xcd}
DOWN_N_WGS_PER_XCD = (HIDDEN_SIZE // OUTPUT_PER_WG) // NUM_XCDS  # {cfg.down_n_wgs_per_xcd}"""}
LM_N_WGS_PER_XCD = (PADDED_VOCAB_SIZE // OUTPUT_PER_WG) // NUM_XCDS  # {cfg.lm_n_wgs_per_xcd}

# MXFP4 workgroup byte sizes
QKV_WG_BYTES = OUTPUT_PER_WG * (HIDDEN_SIZE // 2 + HIDDEN_SIZE // 32)
OPROJ_WG_BYTES = OPROJ_OPW * (OPROJ_REDUCTION // 2 + OPROJ_REDUCTION // 32)
GATEUP_WG_BYTES = GATEUP_OPW * (HIDDEN_SIZE // 2 + HIDDEN_SIZE // 32)
DOWN_WG_BYTES = OUTPUT_PER_WG * (INTERMEDIATE_SIZE // 2 + INTERMEDIATE_SIZE // 32)
LM_WG_BYTES = OUTPUT_PER_WG * (HIDDEN_SIZE // 2 + HIDDEN_SIZE // 32)

# Pointer table layout (matches {nc}_kernel.cuh)
PTRS_IN = {cfg.ptrs_in}
PTRS_OUT = {cfg.ptrs_out}
PTRS_PER_LAYER = PTRS_IN + PTRS_OUT

# Counter buffer -- must equal COUNTERS_PER_LAYER in device_functions.cuh.
# The kernel derives rank_counters from the HEADER's value; this demo sizes the
# allocation. A stale literal here is an out-of-bounds atomicAdd, not an error.
COUNTERS_PER_LAYER = {cfg.counters_per_layer // 16} * 16


def parse_args():
    parser = argparse.ArgumentParser(description="Fleet MK {cfg.name} demo")
    parser.add_argument("--model", default="{nt}/{cfg.name.upper().replace('-', '-')}",
                        help="HuggingFace model name or path")
    parser.add_argument("--model-path", default=None,
                        help="Local model path (overrides --model)")
    parser.add_argument("--prompt", default="Tell me the history of america")
    parser.add_argument("--max-seq-length", type=int, default={cfg.max_seq_len})
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--fleet-mk-so", default=None,
                        help="Path to compiled {nc}.so")
    parser.add_argument("--pytorch-only", action="store_true",
                        help="Use PyTorch for decode (skip fleet_mk kernel)")
    return parser.parse_args()


def load_{nc}_kernel(so_path: str):
    """Load the compiled {nt} fleet_mk .so."""
    lib = ctypes.CDLL(so_path)

    lib.{nc}_init.restype = None
    lib.{nc}_init.argtypes = []

    lib.{nc}_launch.restype = None
    lib.{nc}_launch.argtypes = [
        ctypes.c_int,       # num_active_tokens
        ctypes.c_float,     # attn_scale
        ctypes.c_void_p,    # cos_ptr
        ctypes.c_void_p,    # sin_ptr
        ctypes.c_void_p,    # qo_indptr
        ctypes.c_void_p,    # kv_indptr
        ctypes.c_void_p,    # kv_indices
        ctypes.c_void_p,    # kv_last_page_len
        ctypes.c_void_p,    # ptr_table
        ctypes.c_void_p,    # counter_buf
        ctypes.c_void_p,    # lm_norm_weight
        ctypes.c_void_p,    # lm_norm_scratch
        ctypes.c_void_p,    # lm_mxfp4_weight
        ctypes.c_void_p,    # lm_bias
        ctypes.c_void_p,    # argmax_output
        ctypes.c_void_p,    # timing_buf
        ctypes.c_void_p,    # stream
    ]

    lib.{nc}_finalize.restype = None
    lib.{nc}_finalize.argtypes = []

    return lib


def main():
    args = parse_args()
    torch.set_default_dtype(torch.bfloat16)

    # -- Load model --
    model_path = args.model_path or args.model
    print(f"Loading {cfg.name} from: {{model_path}}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    config = model.config
    assert config.hidden_size == ACTUAL_HIDDEN_DIM or config.hidden_size == HIDDEN_SIZE
    assert config.num_attention_heads == NUM_Q_HEADS
    assert config.num_key_value_heads == NUM_KV_HEADS
    assert config.head_dim == HEAD_DIM

    print(f"{cfg.name}: {{NUM_LAYERS}} layers, hidden={{HIDDEN_SIZE}}, "
          f"intermediate={{INTERMEDIATE_SIZE}}, "
          f"heads={{NUM_Q_HEADS}}Q/{{NUM_KV_HEADS}}KV, head_dim={{HEAD_DIM}}")

    bs = 1  # decode batch size
    page_size = {cfg.page_size}  # == fleet_mk PAGE_SIZE (vLLM's ROCm default block size)
    # Enough pages to hold the whole sequence. This used to be a bare 16, which
    # only worked because page_size was 128 (16 x 128 = 2048 entries, far past
    # any --max-seq-length). At page_size 16 a fixed 16 pages is 256 entries and
    # silently truncates the default 512-token run, so derive it instead.
    max_num_pages = (args.max_seq_length + page_size - 1) // page_size

    # -- Tokenize --
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{{"role": "user", "content": args.prompt}}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([formatted], return_tensors="pt",
                                  add_special_tokens=False).to("cuda")
    else:
        model_inputs = tokenizer([args.prompt], return_tensors="pt").to("cuda")

    tokens = torch.zeros(args.max_seq_length, dtype=torch.long, device="cuda")
    prompt_ids = model_inputs.input_ids[0]
    prompt_len = prompt_ids.shape[0]
    tokens[:prompt_len] = prompt_ids
    print(f"Prompt: {{prompt_len}} tokens, max_seq_length: {{args.max_seq_length}}")

    # -- RoPE position embeddings --
    rotary_emb = model.model.rotary_emb
    positions = torch.arange(args.max_seq_length, device="cuda").unsqueeze(0)
    cos_raw, sin_raw = rotary_emb(
        torch.ones(1, 1, device="cuda"), positions)
    cos_raw = cos_raw[0]  # [max_seq, head_dim]
    sin_raw = sin_raw[0]
    if cos_raw.shape[-1] < HEAD_DIM:
        cos_raw = torch.nn.functional.pad(cos_raw, (0, HEAD_DIM - cos_raw.shape[-1]))
        sin_raw = torch.nn.functional.pad(sin_raw, (0, HEAD_DIM - sin_raw.shape[-1]))
    cos_padded = cos_raw.contiguous()
    sin_padded = sin_raw.contiguous()

    # -- Pack weights --
    print("Packing MXFP4 weights...")
    weight_tensors = []  # keep refs to prevent GC

    for li in range(NUM_LAYERS):
        layer = model.model.layers[li]

        # -- QKV weight: interleave Q/K/V by KV groups --
        w_q = pad_weight_2d(layer.self_attn.q_proj.weight.data.to("cuda"),
                            target_cols=HIDDEN_SIZE)
        w_k = pad_weight_2d(layer.self_attn.k_proj.weight.data.to("cuda"),
                            target_cols=HIDDEN_SIZE)
        w_v = pad_weight_2d(layer.self_attn.v_proj.weight.data.to("cuda"),
                            target_cols=HIDDEN_SIZE)

        qkv_chunks = []
        for g in range(NUM_KV_HEADS):
            qkv_chunks.append(w_q[g * Q_PER_KV * HEAD_DIM:(g + 1) * Q_PER_KV * HEAD_DIM])
            qkv_chunks.append(w_k[g * HEAD_DIM:(g + 1) * HEAD_DIM])
            qkv_chunks.append(w_v[g * HEAD_DIM:(g + 1) * HEAD_DIM])
        w_qkv = torch.cat(qkv_chunks, dim=0).contiguous()
        assert w_qkv.shape == (QKV_OUTPUT_SIZE, HIDDEN_SIZE), \\
            f"QKV shape mismatch: {{w_qkv.shape}} vs ({{QKV_OUTPUT_SIZE}}, {{HIDDEN_SIZE}})"

        qkv_blocks, qkv_scales = quantize_bf16_to_mxfp4(w_qkv)
        qkv_packed = pack_mxfp4_workgroup(
            qkv_blocks, qkv_scales, output_per_wg=OUTPUT_PER_WG,
            target_out_dim=QKV_OUTPUT_SIZE,
            target_num_blocks=HIDDEN_SIZE // 32)
        weight_tensors.append(qkv_packed)  # [0] qkv_weight

        # QKV bias
{qkv_bias_check}

        # -- O-proj weight --
        w_o = pad_weight_2d(layer.self_attn.o_proj.weight.data.to("cuda"),
                            target_rows=HIDDEN_SIZE)
        o_blocks, o_scales = quantize_bf16_to_mxfp4(w_o)
        o_packed = pack_mxfp4_workgroup(
            o_blocks, o_scales, output_per_wg=OUTPUT_PER_WG,
            target_out_dim=HIDDEN_SIZE,
            target_num_blocks=OPROJ_REDUCTION // 32)
        weight_tensors.append(o_packed)  # [2] oproj_weight

        # O-proj bias (zeros)
        o_bias = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        if layer.self_attn.o_proj.bias is not None:
            o_raw = layer.self_attn.o_proj.bias.data.to("cuda")
            o_bias[:o_raw.shape[0]] = o_raw
        weight_tensors.append(o_bias)  # [3] oproj_bias

        # -- RMSNorm weights (pad to HIDDEN_SIZE if needed) --
        norm_w1 = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        norm_w1_raw = layer.input_layernorm.weight.data.to("cuda")
        norm_w1[:norm_w1_raw.shape[0]] = norm_w1_raw
        weight_tensors.append(norm_w1.contiguous())  # [4] norm_weight_pre

        norm_w2 = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        norm_w2_raw = layer.post_attention_layernorm.weight.data.to("cuda")
        norm_w2[:norm_w2_raw.shape[0]] = norm_w2_raw
        weight_tensors.append(norm_w2.contiguous())  # [5] norm_weight_post

{f"""        # -- GateUp weight (interleaved gate/up for fused SwiGLU, OPW={cfg.gateup_opw}) --
        w_gate = layer.mlp.gate_proj.weight.data.to("cuda")
        w_up = layer.mlp.up_proj.weight.data.to("cuda")
        n_wgs = INTERMEDIATE_SIZE // (GATEUP_OPW // 2)
        w_gateup_interleaved = torch.empty(GATEUP_OUTPUT_SIZE, HIDDEN_SIZE,
                                           dtype=torch.bfloat16, device="cuda")
        for wg in range(n_wgs):
            start = wg * (GATEUP_OPW // 2)
            end = start + (GATEUP_OPW // 2)
            dst_start = wg * GATEUP_OPW
            w_gateup_interleaved[dst_start:dst_start + GATEUP_OPW // 2] = w_gate[start:end]
            w_gateup_interleaved[dst_start + GATEUP_OPW // 2:dst_start + GATEUP_OPW] = w_up[start:end]
        w_gateup_interleaved = w_gateup_interleaved.contiguous()

        gu_blocks, gu_scales = quantize_bf16_to_mxfp4(w_gateup_interleaved)
        gu_packed = pack_mxfp4_workgroup(
            gu_blocks, gu_scales, output_per_wg=GATEUP_OPW,
            target_out_dim=GATEUP_OUTPUT_SIZE,
            target_num_blocks=HIDDEN_SIZE // 32)
        weight_tensors.append(gu_packed)  # [6] gateup_weight

        # GateUp bias (zeros -- interleaved same way)
        gu_bias = torch.zeros(GATEUP_OUTPUT_SIZE, dtype=torch.bfloat16, device="cuda")
        weight_tensors.append(gu_bias)  # [7] gateup_bias

        # -- Down weight --
        w_down = layer.mlp.down_proj.weight.data.to("cuda")
        d_blocks, d_scales = quantize_bf16_to_mxfp4(w_down)
        d_packed = pack_mxfp4_workgroup(
            d_blocks, d_scales, output_per_wg=OUTPUT_PER_WG,
            target_out_dim=HIDDEN_SIZE,
            target_num_blocks=INTERMEDIATE_SIZE // 32)
        weight_tensors.append(d_packed)  # [8] down_weight

        # Down bias (zeros)
        d_bias = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        weight_tensors.append(d_bias)  # [9] down_bias

{qk_norm_pack_code}        # Free layer refs
        del w_q, w_k, w_v, w_qkv, w_o, w_gate, w_up, w_gateup_interleaved, w_down
        del layer"""}

        gc.collect()
        torch.cuda.empty_cache()

        if (li + 1) % 6 == 0 or li == NUM_LAYERS - 1:
            print(f"  Packed layer {{li + 1}}/{{NUM_LAYERS}}")

    WEIGHTS_PER_LAYER = {weights_per_layer}
    assert len(weight_tensors) == NUM_LAYERS * WEIGHTS_PER_LAYER

    weight_ptrs_host = [t.data_ptr() for t in weight_tensors]

    # -- Embedding + LM head --
    embed_weight = model.model.embed_tokens.weight.data.to("cuda")
    final_norm_w_raw = model.model.norm.weight.data.to("cuda")
    final_norm_w = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    final_norm_w[:final_norm_w_raw.shape[0]] = final_norm_w_raw
    final_norm_w = final_norm_w.contiguous()

    lm_head_weight = pad_weight_2d(
        model.lm_head.weight.data.to("cuda"),
        target_rows=PADDED_VOCAB_SIZE,
        target_cols=HIDDEN_SIZE)
    lm_blocks, lm_scales = quantize_bf16_to_mxfp4(lm_head_weight)
    lm_head_packed = pack_mxfp4_workgroup(
        lm_blocks, lm_scales, output_per_wg=OUTPUT_PER_WG,
    ).squeeze(0)
    print(f"LM head MXFP4: {{lm_head_weight.shape}} -> {{lm_head_packed.shape}}")
    lm_head_zero_bias = torch.zeros(1, PADDED_VOCAB_SIZE, dtype=torch.bfloat16, device="cuda")

    # -- Allocate workspace buffers --
    print("Allocating buffers...")
    q_ws_stride = Q_PER_KV * HEAD_DIM  # {cfg.q_workspace_stride}
    kv_cache_stride = NUM_KV_HEADS * HEAD_DIM  # {cfg.kv_cache_stride}

    # Per-layer KV cache (paged)
    num_kv_entries = max_num_pages * page_size
    buf_k_cache = [torch.zeros(num_kv_entries, kv_cache_stride,
                                dtype=torch.bfloat16, device="cuda") for _ in range(NUM_LAYERS)]
    buf_v_cache = [torch.zeros(num_kv_entries, kv_cache_stride,
                                dtype=torch.bfloat16, device="cuda") for _ in range(NUM_LAYERS)]

    # Workspace buffers (ping-pong residual)
    buf_residual_a = torch.zeros(bs, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    buf_residual_b = torch.zeros(bs, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    # Norm scratch buffers
    buf_norm_scratch1 = torch.zeros(bs, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    buf_norm_scratch2 = torch.zeros(bs, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    # QKV output
    buf_qkv_output = torch.zeros(bs, QKV_OUTPUT_SIZE, dtype=torch.bfloat16, device="cuda")

    # Attention buffers
    buf_q_workspace = torch.zeros(bs, NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    buf_attn_out = torch.zeros(bs, NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")

    # O-proj output
    buf_oproj_out = torch.zeros(bs, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

{f"""    # GateUp scratch
    buf_gateup_scratch = torch.zeros(bs, GATEUP_OUTPUT_SIZE, dtype=torch.bfloat16, device="cuda")

    # SwiGLU output
    buf_swiglu_out = torch.zeros(bs, INTERMEDIATE_SIZE, dtype=torch.bfloat16, device="cuda")"""}

    # Attention split-KV buffers
    o_acc_dim = NUM_KV_HEADS * NUM_KV_CHUNKS * Q_PER_KV * HEAD_DIM
    buf_o_acc_f32 = torch.zeros(bs, o_acc_dim, dtype=torch.float32, device="cuda")
    lse_dim = NUM_KV_HEADS * NUM_KV_CHUNKS * Q_PER_KV
    buf_lse_acc = torch.zeros(bs, lse_dim, dtype=torch.float32, device="cuda")

    # LM head buffers
    buf_argmax_out = torch.zeros(1, dtype=torch.int64, device="cuda")
    buf_lm_norm_scratch = torch.zeros(bs, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    # Counter buffer
    RANK_COUNTER_INTS = NUM_XCDS * 16
    counter_total_ints = NUM_LAYERS * COUNTERS_PER_LAYER + RANK_COUNTER_INTS
    buf_counter = torch.zeros(counter_total_ints, dtype=torch.int32, device="cuda")

    # Subphase timing buffer (12 slots per layer + 4 tail slots)
    TIMING_SLOTS_PER_LAYER = 12
    TIMING_TAIL_SLOTS = 4
    timing_total = NUM_LAYERS * TIMING_SLOTS_PER_LAYER + TIMING_TAIL_SLOTS
    buf_timing = torch.zeros(timing_total, dtype=torch.int64, device="cuda")

    # -- Build pointer table --
    # Layout: ptr_table[xcd * NUM_LAYERS * PTRS_PER_LAYER + layer * PTRS_PER_LAYER + slot]
    print("Building pointer table...")

    ptr_table_host = []
    for xcd in range(NUM_XCDS):
        for li in range(NUM_LAYERS):
            base = li * WEIGHTS_PER_LAYER
            qkv_weight_base = weight_ptrs_host[base + 0]
            qkv_bias = weight_ptrs_host[base + 1]
            oproj_weight_base = weight_ptrs_host[base + 2]
            oproj_bias = weight_ptrs_host[base + 3]
            norm_w1 = weight_ptrs_host[base + 4]
            norm_w2 = weight_ptrs_host[base + 5]
{f"""            gateup_weight_base = weight_ptrs_host[base + 6]
            gateup_bias = weight_ptrs_host[base + 7]
            down_weight_base = weight_ptrs_host[base + 8]
            down_bias = weight_ptrs_host[base + 9]
{qk_norm_ptr_lines}"""}
            # Per-XCD weight offsets
            qkv_weight_xcd = qkv_weight_base + xcd * QKV_N_WGS_PER_XCD * QKV_WG_BYTES
            oproj_weight_xcd = oproj_weight_base + xcd * OPROJ_N_WGS_PER_XCD * OPROJ_WG_BYTES
            gateup_weight_xcd = gateup_weight_base + xcd * GATEUP_N_WGS_PER_XCD * GATEUP_WG_BYTES
            down_weight_xcd = down_weight_base + xcd * DOWN_N_WGS_PER_XCD * DOWN_WG_BYTES
            # Counter buffer for this layer
            counter_ptr = buf_counter.data_ptr() + li * COUNTERS_PER_LAYER * 4

            # Ping-pong residual: even layers read from A, write to B; odd layers reverse
            if li % 2 == 0:
                residual_ptr = buf_residual_a.data_ptr()
                layer_output_ptr = buf_residual_b.data_ptr()
            else:
                residual_ptr = buf_residual_b.data_ptr()
                layer_output_ptr = buf_residual_a.data_ptr()

{f"""            # input_ptrs[{cfg.ptrs_in}] - matches {ns}_ptr namespace (Dense)
            input_ptrs = [
                residual_ptr,                    # [0]  RESIDUAL
                norm_w1,                         # [1]  NORM_W1
                buf_norm_scratch1.data_ptr(),    # [2]  NORM_SCRATCH1
                qkv_weight_xcd,                  # [3]  QKV_WEIGHT (per-XCD)
                qkv_bias,                        # [4]  QKV_BIAS
                buf_lse_acc.data_ptr(),           # [5]  LSE_ACC
                buf_o_acc_f32.data_ptr(),         # [6]  O_ACC_F32
                oproj_weight_xcd,                # [7]  OPROJ_WEIGHT (per-XCD)
                oproj_bias,                      # [8]  OPROJ_BIAS
                norm_w2,                         # [9]  NORM_W2
                buf_norm_scratch2.data_ptr(),    # [10] NORM_SCRATCH2
                gateup_weight_xcd,               # [11] GATEUP_WEIGHT (per-XCD)
                gateup_bias,                     # [12] GATEUP_BIAS
                down_weight_xcd,                 # [13] DOWN_WEIGHT (per-XCD)
                down_bias,                       # [14] DOWN_BIAS
                counter_ptr,                     # [15] COUNTER_BUF
{qk_norm_input_entries}            ]

            # output_ptrs[{cfg.ptrs_out}] - matches {ns}_ptr namespace (Dense)
            output_ptrs = [
                buf_qkv_output.data_ptr(),       # [0]  QKV_OUTPUT
                buf_k_cache[li].data_ptr(),      # [1]  K_CACHE
                buf_v_cache[li].data_ptr(),      # [2]  V_CACHE
                buf_q_workspace.data_ptr(),      # [3]  Q_WORKSPACE
                buf_attn_out.data_ptr(),         # [4]  ATTN_OUT
                buf_oproj_out.data_ptr(),        # [5]  OPROJ_OUT
                buf_gateup_scratch.data_ptr(),   # [6]  GATEUP_SCRATCH
                buf_swiglu_out.data_ptr(),       # [7]  SWIGLU_OUT
                layer_output_ptr,                # [8]  LAYER_OUTPUT (ping-pong)
            ]"""}

            assert len(input_ptrs) == PTRS_IN
            assert len(output_ptrs) == PTRS_OUT
            ptr_table_host.extend(input_ptrs)
            ptr_table_host.extend(output_ptrs)

    assert len(ptr_table_host) == NUM_XCDS * NUM_LAYERS * PTRS_PER_LAYER
    ptr_table_tensor = torch.tensor(ptr_table_host, dtype=torch.long, device="cuda")
    print(f"  Pointer table: {{len(ptr_table_host)}} entries "
          f"({{NUM_XCDS}}x{{NUM_LAYERS}}x{{PTRS_PER_LAYER}})")

    rank_counter_offset = NUM_LAYERS * COUNTERS_PER_LAYER
    rank_counter_slice = buf_counter[rank_counter_offset:]

    # -- Paged attention index arrays --
    qo_indptr = torch.zeros(2, dtype=torch.int32, device="cuda")
    qo_indptr[1] = 1
    kv_indptr = torch.zeros(2, dtype=torch.int32, device="cuda")
    kv_indices = torch.zeros(max_num_pages, dtype=torch.int32, device="cuda")
    kv_last_page_len = torch.zeros(1, dtype=torch.int32, device="cuda")

    # -- Load fleet_mk kernel --
    fleet_mk_so = args.fleet_mk_so or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "generated", "{nc}.so")
    print(f"Loading fleet_mk kernel from: {{fleet_mk_so}}")
    fleet_mk = load_{nc}_kernel(fleet_mk_so)
    fleet_mk.{nc}_init()

    # -- Inference loop --
    stream = torch.cuda.current_stream()
    attn_scale = (1.0 / math.sqrt(HEAD_DIM)) * 1.44269504088896340736

    # Prefill via PyTorch
    _model_device = next(model.parameters()).device
    print(f"\\n=== Prefill ({{prompt_len}} tokens) via PyTorch (device={{_model_device}}) ===")
    past_key_values = None
    with torch.no_grad():
        input_ids = tokens[:prompt_len].unsqueeze(0).to(_model_device)
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits

    next_token = logits[:, -1, :VOCAB_SIZE].argmax(dim=-1).item()
    tokens[prompt_len] = next_token
    print(f"Prefill complete. First token: {{next_token}} "
          f"('{{tokenizer.decode([next_token])}}')")

    # Copy PyTorch KV cache into paged KV cache buffers
    if not args.pytorch_only:
        print("Converting KV cache to paged format...")
        for li in range(NUM_LAYERS):
            k_pt = past_key_values[li][0]  # [1, num_kv_heads, seq_len, head_dim]
            v_pt = past_key_values[li][1]
            seq_len_cached = k_pt.shape[2]
            k_flat = k_pt[0].permute(1, 0, 2).reshape(seq_len_cached, -1)
            v_flat = v_pt[0].permute(1, 0, 2).reshape(seq_len_cached, -1)
            buf_k_cache[li][:seq_len_cached] = k_flat.to(torch.bfloat16)
            buf_v_cache[li][:seq_len_cached] = v_flat.to(torch.bfloat16)
        print(f"  Copied {{seq_len_cached}} positions into paged KV cache")

    # -- Decode loop --
    output_len = args.max_new_tokens if args.max_new_tokens else (
        args.max_seq_length - prompt_len - 1)
    output_len = min(output_len, args.max_seq_length - prompt_len - 1)

    print(f"\\n=== Decode ({{output_len}} tokens) ===")

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    decode_times = []

    for di in range(output_len):
        cur_pos = prompt_len + di
        cur_token = tokens[cur_pos].item()

        if cur_token == tokenizer.eos_token_id and not args.ignore_eos:
            print(f"EOS at position {{cur_pos}}")
            break

        if di >= args.warmup:
            starter.record()

        if args.pytorch_only:
            input_ids_t = torch.tensor([[cur_token]], device=_model_device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids_t,
                              past_key_values=past_key_values,
                              use_cache=True)
                past_key_values = outputs.past_key_values
                logits = outputs.logits
            next_token = logits[:, -1, :VOCAB_SIZE].argmax(dim=-1).item()
        else:
            # Fleet MK kernel path
            embed_out = model.model.embed_tokens(
                torch.tensor([[cur_token]], device=_model_device)).to("cuda")
            # Layer 0 reads from buf_residual_a (even layer -> RESIDUAL = A)
            buf_residual_a.zero_()
            buf_residual_a[:, :ACTUAL_HIDDEN_DIM] = embed_out[:, 0, :ACTUAL_HIDDEN_DIM]
            rank_counter_slice.zero_()


            num_pages_used = (cur_pos + page_size) // page_size
            kv_indptr[0] = 0
            kv_indptr[1] = num_pages_used
            for p in range(num_pages_used):
                kv_indices[p] = p
            kv_last_page_len[0] = (cur_pos % page_size) + 1

            fleet_mk.{nc}_launch(
                1, attn_scale,
                cos_padded.data_ptr(), sin_padded.data_ptr(),
                qo_indptr.data_ptr(), kv_indptr.data_ptr(),
                kv_indices.data_ptr(), kv_last_page_len.data_ptr(),
                ptr_table_tensor.data_ptr(), buf_counter.data_ptr(),
                final_norm_w.data_ptr(), buf_lm_norm_scratch.data_ptr(),
                lm_head_packed.data_ptr(), lm_head_zero_bias.data_ptr(),
                buf_argmax_out.data_ptr(),
                buf_timing.data_ptr(),
                stream.cuda_stream,
            )
            torch.cuda.synchronize()
            next_token = buf_argmax_out[0].item()

        if di >= args.warmup:
            ender.record()
            torch.cuda.synchronize()
            elapsed = starter.elapsed_time(ender)
            decode_times.append(elapsed)
        else:
            torch.cuda.synchronize()

        if di < 5:
            print(f"  iter {{di + 1}}: next_token={{next_token}} "
                  f"('{{tokenizer.decode([next_token])}}')")

        tokens[cur_pos + 1] = next_token

    # -- Results --
    end_pos = prompt_len + min(di + 1, output_len)
    generated_ids = tokens[:end_pos + 1]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\\n{{'=' * 60}}")
    print(response)
    print(f"{{'=' * 60}}")

    gen_tokens = len(decode_times)
    if gen_tokens > 0:
        avg_ms = sum(decode_times) / gen_tokens
        print(f"\\nPrompt: {{prompt_len}} tokens")
        print(f"Generated: {{gen_tokens}} tokens (after {{args.warmup}} warmup)")
        print(f"Decode avg: {{avg_ms:.3f}} ms/token")
        print(f"Decode min: {{min(decode_times):.3f}} ms/token")
        print(f"Decode max: {{max(decode_times):.3f}} ms/token")
    else:
        print("\\nNo decode tokens generated (warmup >= output_len?)")

    # -- Subphase timing readback --
    if os.environ.get("FLEET_MK_SUBPHASE_TIMING"):
        timing_data = buf_timing.cpu().numpy()
        # GPU clock is 100MHz on MI350 (10ns ticks)
        GPU_CLOCK_NS = 10.0
        print(f"\\n{{'=' * 60}}")
        print("SUBPHASE TIMING (last decode step, XCD0/worker0)")
        print(f"{{'=' * 60}}")
        phase_names = [
            "layer_start", "after_rmsnorm1", "after_qkv_barrier",
            "after_rope", "after_attn_compute", "after_oproj_barrier",
            "after_oproj_gemm", "after_ffn_barrier", "after_gateup_gemm",
            "after_gateup_barrier", "after_down_gemm", "after_layer_barrier"
        ]
        # Print per-layer breakdown for layers 0, 1, last
        for li in [0, 1, NUM_LAYERS - 1]:
            base = li * TIMING_SLOTS_PER_LAYER
            timestamps = timing_data[base:base + TIMING_SLOTS_PER_LAYER]
            if timestamps[0] == 0:
                print(f"  Layer {{li}}: no timing data")
                continue
            print(f"\\n  Layer {{li}}:")
            for pi in range(1, TIMING_SLOTS_PER_LAYER):
                if timestamps[pi] == 0:
                    break
                delta_ns = (timestamps[pi] - timestamps[pi-1]) * GPU_CLOCK_NS
                delta_us = delta_ns / 1000.0
                print(f"    {{phase_names[pi-1]:30s}} -> {{phase_names[pi]:30s}}: {{delta_us:8.1f}} us")
            total_ns = (timestamps[TIMING_SLOTS_PER_LAYER-1] - timestamps[0]) * GPU_CLOCK_NS
            print(f"    {{'TOTAL':62s}}: {{total_ns/1000.0:8.1f}} us")

        # Print average across all layers
        print(f"\\n  Average across {{NUM_LAYERS}} layers:")
        avg_deltas = [0.0] * (TIMING_SLOTS_PER_LAYER - 1)
        valid_layers = 0
        for li in range(NUM_LAYERS):
            base = li * TIMING_SLOTS_PER_LAYER
            timestamps = timing_data[base:base + TIMING_SLOTS_PER_LAYER]
            if timestamps[0] == 0 or timestamps[-1] == 0:
                continue
            valid_layers += 1
            for pi in range(1, TIMING_SLOTS_PER_LAYER):
                avg_deltas[pi-1] += (timestamps[pi] - timestamps[pi-1]) * GPU_CLOCK_NS / 1000.0
        if valid_layers > 0:
            total_avg = 0.0
            for pi in range(TIMING_SLOTS_PER_LAYER - 1):
                avg_deltas[pi] /= valid_layers
                total_avg += avg_deltas[pi]
                print(f"    {{phase_names[pi]:30s}} -> {{phase_names[pi+1]:30s}}: {{avg_deltas[pi]:8.1f}} us")
            print(f"    {{'TOTAL':62s}}: {{total_avg:8.1f}} us")
            print(f"    {{'TOTAL x {cfg.num_layers} layers':62s}}: {{total_avg * {cfg.num_layers} / 1000.0:8.3f}} ms")

    fleet_mk.{nc}_finalize()
    print("\\nDone.")


if __name__ == "__main__":
    main()
'''


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fleet MK code generator: YAML config -> kernel + launch + driver + build")
    parser.add_argument("config", help="Path to YAML model config file")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as config parent)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be generated without writing")
    args = parser.parse_args()

    cfg = load_and_validate(args.config)

    # Determine output directory
    config_dir = os.path.dirname(os.path.abspath(args.config))
    base_dir = args.output_dir or os.path.dirname(config_dir)
    gen_dir = os.path.join(base_dir, "generated")

    nc = cfg.name_clean

    files = {
        os.path.join(gen_dir, f"{nc}_kernel.cuh"): generate_kernel(cfg),
        os.path.join(gen_dir, f"{nc}_launch.hip"): generate_launch(cfg),
        os.path.join(base_dir, f"demo_{nc}.py"): generate_driver(cfg),
        os.path.join(base_dir, f"build_{nc}.sh"): generate_build(cfg),
    }

    if args.dry_run:
        for path, content in files.items():
            lines = content.count('\n')
            print(f"  Would write: {path} ({lines} lines)")
        return

    os.makedirs(gen_dir, exist_ok=True)

    for path, content in files.items():
        # Explicit utf-8: the GPT-OSS targets contain em-dash and arrow
        # characters, so a C-locale environment would encode them differently
        # and break byte-identity against the on-disk artifact. Keep this in
        # sync with the read side in check_roundtrip.py.
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        lines = content.count('\n')
        print(f"  Wrote: {path} ({lines} lines)")

    # Make build script executable
    build_path = os.path.join(base_dir, f"build_{nc}.sh")
    os.chmod(build_path, 0o755)

    print(f"\nGeneration complete for {cfg.name}.")
    print(f"  Build:  bash {build_path}")
    print(f"  Run:    python3 {os.path.join(base_dir, f'demo_{nc}.py')} --model-path <path>")


if __name__ == "__main__":
    main()
