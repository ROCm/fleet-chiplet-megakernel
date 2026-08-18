#!/usr/bin/env python3
"""Reduce Fleet's [PSLOTW] phase-slot dump into a per-phase table.

Consumes the output of a run with MPK_PHASE_SLOTS=1 (see the slot map in
persistent_kernel.cuh). Each [PSLOTW] line is one worker: `w=<id> n=<layers>`
followed by 11 numbers, the mean ns of spans 0->1, 1->2, ... 10->11 over every
recorded layer.

Two things this deliberately does NOT do.

It does not average all 248 workers into one row. Rank is role here: ranks
0..total_qkv_tiles_per_xcd-1 run QKV and attention, the rest skip straight to
Phase 6 and only ever run MoE tiles. Their slot 1 and slot 3 mean different
things -- for a MoE-only worker those are near-zero passthroughs, and folding
them in halves the apparent QKV cost. So the bands are reported separately and
the split point is detected from the data (the last rank whose QKV span is
above a floor), not assumed.

It does not sum the per-worker means and call that the layer. Workers run
concurrently; the layer's cost is the critical path, which is the MAX over
workers of the cumulative time to each slot, not the sum of any one worker's
spans. Both are printed, because their difference is exactly the skew.

Usage: summarize_phase_slots.py <run.log> [--json out.json]
"""
import json
import re
import sys

SLOT_NAMES = [
    # Slot 0 is the inter-layer span: previous layer's slot 11 to this layer's
    # slot 0. Older logs (before the recorder emitted it) have 11 columns
    # instead of 12 and are still parsed -- see the width check below.
    "0 inter_layer",
    "1 qkv_gemm",
    "2 qkv_epoch_barrier",
    "3 attention+merge",
    "4 (marker)",
    "5 attn_release_wait",
    "6 oproj+rmsnorm+router",
    "7 topk_wait",
    "8 moe(w13+swiglu+w2)",
    "9 layer_arrive+fanout+pf",
    "10 layer_gate_poll",
    "11 layer_exit",
]


# Strict, because the host's stdout interleaves with the device printf stream:
# the prompt echo has been observed splicing itself into the middle of a
# [PSLOTW] line. A truncated row must be dropped and counted, never parsed as
# if its missing slots were zero.
ROW_RE = re.compile(r"^\[PSLOTW\] w=(\d+) n=(\d+)((?: \d+)+)\s*$")


def parse(path):
    hdr = None
    rows = []
    corrupt = 0
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("[PSLOT] "):
                hdr = dict(
                    kv.split("=", 1) for kv in line.split()[1:] if "=" in kv
                )
            elif line.startswith("[PSLOTW] "):
                m = ROW_RE.match(line)
                if not m:
                    corrupt += 1
                    continue
                rows.append(
                    (int(m.group(1)), int(m.group(2)),
                     [int(x) for x in m.group(3).split()])
                )
    if corrupt:
        print(f"WARNING: dropped {corrupt} interleaved/truncated [PSLOTW] rows")
    widths = {len(s) for _, _, s in rows}
    if len(widths) > 1:
        sys.exit(f"[PSLOTW] rows have inconsistent slot counts: {widths}")
    return hdr, rows


def main():
    path = sys.argv[1]
    out_json = None
    if "--json" in sys.argv:
        out_json = sys.argv[sys.argv.index("--json") + 1]

    hdr, rows = parse(path)
    if hdr is None:
        sys.exit(f"no [PSLOT] header in {path} -- was MPK_PHASE_SLOTS=1 set?")
    if int(hdr.get("arm", "0")) == 0:
        sys.exit(
            f"[PSLOT] arm=0 in {path}: the recorder never started, so every "
            f"span would be zero. total_iters={hdr.get('total_iters')} "
            f"start_iter={hdr.get('start_iter')}"
        )
    if not rows:
        sys.exit(f"[PSLOT] armed but no [PSLOTW] rows in {path}")

    nslots = len(rows[0][2])
    # Logs written before the recorder emitted slot 0 have 11 columns starting
    # at "1 qkv_gemm". Dropping the slot-0 name for those keeps every other
    # column labelled correctly instead of shifting the whole table by one.
    names = SLOT_NAMES if nslots == len(SLOT_NAMES) else SLOT_NAMES[1:]
    if nslots != len(names):
        sys.exit(f"{path}: {nslots} slots but {len(names)} slot names")
    # Expected layer count, straight from the header: 36 layers per forward
    # pass over the iterations past the arm. A worker short of this recorded
    # partial layers and its means are over the wrong denominator.
    n_layers = sorted({n for _, n, _ in rows})
    print(f"workers={len(rows)} slots={nslots} "
          f"layers_per_worker={n_layers if len(n_layers) < 5 else f'{min(n_layers)}..{max(n_layers)}'} "
          f"total_iters={hdr.get('total_iters')} start_iter={hdr.get('start_iter')}")

    # Split QKV-running ranks from MoE-only ranks by the data, not by an
    # assumed geometry: a MoE-only worker's QKV span is a passthrough and
    # orders of magnitude below a real GEMM.
    #
    # `qkv_col` is where that span lives, which moved when slot 0 started being
    # emitted -- keying on column 0 unconditionally would classify every worker
    # by its inter-layer time instead, and that is large for both bands.
    qkv_col = 1 if nslots == len(SLOT_NAMES) else 0
    qkv = [r for r in rows if r[2][qkv_col] > 1000]
    moe = [r for r in rows if r[2][qkv_col] <= 1000]

    def band(name, rs):
        if not rs:
            return None
        means = [
            sum(r[2][s] for r in rs) / len(rs) for s in range(nslots)
        ]
        print(f"\n--- {name}: {len(rs)} workers, mean ns/layer")
        tot = 0.0
        for s in range(nslots):
            tot += means[s]
            print(f"  {names[s]:<28} {means[s]:8.0f}  (cum {tot:8.0f})")
        print(f"  {'TOTAL':<28} {tot:8.0f} ns/layer = {tot * 36 / 1e6:.3f} ms/token")
        return means

    qkv_means = band("QKV+attn ranks", qkv)
    moe_means = band("MoE-only ranks", moe)

    # The layer's real cost is the slowest worker's, per slot boundary --
    # summing one worker's spans understates it whenever the skew is what
    # costs, which is the whole reason this instrument exists.
    print("\n--- critical path: max over workers of each cumulative boundary")
    cum_max = []
    for s in range(nslots):
        cum_max.append(max(sum(r[2][: s + 1]) for r in rows))
    prev = 0
    for s in range(nslots):
        print(f"  {names[s]:<28} {cum_max[s] - prev:8.0f}  (cum {cum_max[s]:8.0f})")
        prev = cum_max[s]
    print(f"  {'TOTAL (max cum)':<28} {cum_max[-1]:8.0f} ns/layer "
          f"= {cum_max[-1] * 36 / 1e6:.3f} ms/token")

    if out_json:
        with open(out_json, "w") as fh:
            json.dump(
                {
                    "header": hdr,
                    "slot_names": SLOT_NAMES,
                    "qkv_band_mean_ns": qkv_means,
                    "moe_band_mean_ns": moe_means,
                    "critical_cum_ns": cum_max,
                    "rows": [{"w": w, "n": n, "spans": s} for w, n, s in rows],
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
