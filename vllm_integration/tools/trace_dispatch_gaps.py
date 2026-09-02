#!/usr/bin/env python3
"""Measure per-token launch overhead from a rocprofv3 trace, instead of inferring it.

Why this exists
---------------
The device-side `[FLEET_MK_TIME]` timer (tools/layers_timer_stats.py) reads
`s_memrealtime` from INSIDE the kernel, so it starts after the first wave is
already resident and stops before the last one retires. The driver's CUDA-event
`Decode avg` brackets the launch from the HOST. The difference between the two
was being reported as "launch and teardown overhead" -- but a difference of two
instruments is an inference, not a measurement. It attributes to launch whatever
either instrument happens to miss, including any bias in the instruments
themselves.

`rocprofv3 --kernel-trace --hip-runtime-trace` closes that gap directly. It
timestamps, on one clock:

  * KERNEL_DISPATCH begin/end   -- what the GPU actually spent on the kernel,
                                   wave-launch and drain included
  * hipLaunchKernel begin/end   -- what the host spent enqueueing it

so the per-token cost outside the kernel becomes an OBSERVED end-to-begin gap
between consecutive dispatches, not a residue.

The same trace settles the fleet comparison structurally. fleet is a PERSISTENT
kernel: one dispatch spans every decode iteration, so its FWD_PASS number cannot
contain per-token launch cost -- there is no per-token launch. Point this at
fleet's trace and it reports a dispatch count of 1. That is the evidence, and it
is a count, not a timing argument.

Reading the output
------------------
  in-kernel (dispatch)   what the GPU charged for the kernel
  gap (end -> begin)     dead time between consecutive dispatches: teardown,
                         host turnaround, next enqueue. THIS is launch overhead.
  hip api span           how long hipLaunchKernel itself blocked the host

Compare `in-kernel` here against `[FLEET_MK_TIME] total=` from the same run:
the difference is the kernel's own ramp-up/drain, which the device timer cannot
see because it runs on the device.

Both CSVs are streamed line by line. fleet's kernel trace is several GB -- do
not load it into memory.

Usage:
    python3 tools/trace_dispatch_gaps.py --arm mk    /tmp/trace_mk \
                                         --arm fleet /tmp/trace_fleet

    # single arm, and pick the kernel explicitly
    python3 tools/trace_dispatch_gaps.py --arm mk /tmp/trace_mk \
                                         --kernel gpt_oss_120b_kernel
"""

import argparse
import csv
import glob
import os
import statistics as st
import sys

NS_PER_US = 1000.0

# rocprofv3 writes <prefix>_kernel_trace.csv / <prefix>_hip_api_trace.csv into
# the -d directory; the prefix is the -o value and is not knowable up front.
KERNEL_GLOB = "*_kernel_trace.csv"
HIP_GLOB = "*_hip_api_trace.csv"

# Substrings identifying a megakernel dispatch. Everything else in the trace is
# PyTorch's own elementwise/copy kernels from tokenization and sampling, which
# are not what is being measured.
DEFAULT_KERNEL_HINTS = ("gpt_oss_120b_kernel", "persistent_kernel", "_megakernel")

# The host enqueue call. HIP offers several launch entry points and which one is
# used depends on whether the module was loaded via the runtime or the driver
# API, so match on any of them.
LAUNCH_FNS = ("hipLaunchKernel", "hipModuleLaunchKernel",
              "hipExtModuleLaunchKernel", "hipGraphLaunch")


def one_file(directory, pattern):
    hits = sorted(glob.glob(os.path.join(directory, pattern)))
    if not hits:
        return None
    if len(hits) > 1:
        print(f"  note: {len(hits)} files match {pattern}, using {os.path.basename(hits[0])}")
    return hits[0]


def dispatches(path, hints):
    """Stream a kernel trace, yielding (name, start_ns, end_ns) for matching rows.

    csv.reader rather than a split on commas: kernel names are C++ signatures
    full of commas, quoted by rocprofv3. Splitting would shred them.
    """
    with open(path, newline="", errors="replace") as fh:
        rd = csv.reader(fh)
        header = next(rd, None)
        if header is None:
            return
        try:
            i_name = header.index("Kernel_Name")
            i_beg = header.index("Start_Timestamp")
            i_end = header.index("End_Timestamp")
        except ValueError:
            sys.exit(f"{path}: unexpected columns {header}")
        for row in rd:
            if len(row) <= i_end:
                continue
            name = row[i_name]
            if any(h in name for h in hints):
                yield name, int(row[i_beg]), int(row[i_end])


def launch_spans(path):
    """Stream a HIP API trace, yielding (fn, start_ns, end_ns) for launch calls."""
    if path is None:
        return
    with open(path, newline="", errors="replace") as fh:
        rd = csv.reader(fh)
        header = next(rd, None)
        if header is None:
            return
        try:
            i_fn = header.index("Function")
            i_beg = header.index("Start_Timestamp")
            i_end = header.index("End_Timestamp")
        except ValueError:
            return
        for row in rd:
            if len(row) <= i_end:
                continue
            if row[i_fn] in LAUNCH_FNS:
                yield row[i_fn], int(row[i_beg]), int(row[i_end])


def describe(label, vals_us, indent="  "):
    if not vals_us:
        print(f"{indent}{label:22} -- none")
        return None
    s = sorted(vals_us)
    n = len(s)
    print(f"{indent}{label:22} n={n:<6} median={st.median(s):9.1f}  "
          f"mean={st.mean(s):9.1f}  p10={s[n//10]:9.1f}  p90={s[(9*n)//10]:9.1f}  "
          f"min={s[0]:9.1f}  max={s[-1]:9.1f}")
    return st.median(s)


def analyse(name, directory, hints, warmup, outlier_us):
    print(f"\n=== arm {name} : {directory} ===")
    kpath = one_file(directory, KERNEL_GLOB)
    if kpath is None:
        print(f"  no {KERNEL_GLOB} in {directory} -- trace incomplete or still flushing")
        return None
    hpath = one_file(directory, HIP_GLOB)

    rows = sorted(dispatches(kpath, hints), key=lambda r: r[1])
    if not rows:
        print(f"  no dispatch matching {hints} in {os.path.basename(kpath)}")
        return None

    names = sorted({r[0] for r in rows})
    for nm in names:
        print(f"  kernel: {nm[:110]}")
    print(f"  dispatches: {len(rows)}")

    # A persistent kernel is a structural fact, not a slow measurement: one
    # dispatch spanning the whole decode means there is no per-token launch to
    # charge anything to.
    if len(rows) == 1:
        span_us = (rows[0][2] - rows[0][1]) / NS_PER_US
        print(f"\n  PERSISTENT: a single dispatch covering {span_us:,.1f} us "
              f"({span_us/1000:.1f} ms).")
        print(f"  Every decode iteration ran inside it, so there is no per-token")
        print(f"  launch or teardown in this arm -- not a small one, none.")
        return {"persistent": True, "dispatches": 1, "span_us": span_us}

    # Skip warmup: the first dispatches carry code load, first-touch page
    # faults, and allocator growth that no steady-state token pays.
    use = rows[warmup:] if len(rows) > warmup else rows
    print(f"  dropping first {len(rows)-len(use)} as warmup, {len(use)} remain")

    in_kernel = [(e - b) / NS_PER_US for _, b, e in use]
    gaps = [(use[i + 1][1] - use[i][2]) / NS_PER_US for i in range(len(use) - 1)]
    cadence = [(use[i + 1][1] - use[i][1]) / NS_PER_US for i in range(len(use) - 1)]

    kept_gaps = [g for g in gaps if g < outlier_us]
    dropped = len(gaps) - len(kept_gaps)

    print()
    med_k = describe("in-kernel (dispatch)", in_kernel)
    med_g = describe(f"gap (end -> begin)", kept_gaps)
    if dropped:
        big = sorted(g for g in gaps if g >= outlier_us)
        print(f"  {'':22} {dropped} gap(s) >= {outlier_us:g} us dropped "
              f"(max {big[-1]:,.0f} us) -- host stalls, not steady state")
    describe("token cadence", [c for c in cadence if c < outlier_us])

    api = [(e - b) / NS_PER_US for _, b, e in launch_spans(hpath)]
    med_a = describe("hip launch api span", api) if api else None

    if med_k is not None and med_g is not None:
        tot = med_k + med_g
        print(f"\n  per token (median): {med_k:.1f} us in kernel + {med_g:.1f} us "
              f"outside = {tot:.1f} us")
        print(f"                      outside-kernel share: {100*med_g/tot:.1f}%")
    return {"persistent": False, "dispatches": len(rows),
            "in_kernel_us": med_k, "gap_us": med_g, "api_us": med_a}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs=2, action="append", metavar=("NAME", "DIR"),
                    required=True, help="arm name and its rocprofv3 output dir")
    ap.add_argument("--kernel", action="append", default=None,
                    help="substring identifying the megakernel (repeatable); "
                         f"default {DEFAULT_KERNEL_HINTS}")
    ap.add_argument("--warmup", type=int, default=8,
                    help="dispatches to drop from the front (default 8)")
    ap.add_argument("--outlier-us", type=float, default=5000.0,
                    help="drop gaps at or above this (default 5000)")
    args = ap.parse_args()

    hints = tuple(args.kernel) if args.kernel else DEFAULT_KERNEL_HINTS

    results = {}
    for name, directory in args.arm:
        results[name] = analyse(name, directory, hints, args.warmup, args.outlier_us)

    per = [(n, r) for n, r in results.items() if r and not r["persistent"]]
    pers = [(n, r) for n, r in results.items() if r and r["persistent"]]

    if per and pers:
        print("\n=== comparison ===")
        for n, r in per:
            print(f"  {n:8} launches per token: {r['in_kernel_us']:.1f} us in kernel, "
                  f"{r['gap_us']:.1f} us between dispatches")
        for n, r in pers:
            print(f"  {n:8} does not launch per token: 1 dispatch, "
                  f"{r['span_us']/1000:.1f} ms, all iterations inside it")
        print("\n  The two arms' per-token numbers are therefore NOT the same")
        print("  measurement. Compare in-kernel time to in-kernel time; the")
        print("  between-dispatch gap has no counterpart on the persistent side.")
    elif len(per) == 2:
        (na, ra), (nb, rb) = per
        print("\n=== comparison ===")
        print(f"  in-kernel : {ra['in_kernel_us']:9.1f} -> {rb['in_kernel_us']:9.1f}  "
              f"{rb['in_kernel_us']-ra['in_kernel_us']:+.1f} us")
        print(f"  gap       : {ra['gap_us']:9.1f} -> {rb['gap_us']:9.1f}  "
              f"{rb['gap_us']-ra['gap_us']:+.1f} us")


if __name__ == "__main__":
    main()
