#!/usr/bin/env python3
"""Compare arms on the device-side `layers` timer instead of the summary line.

Why this exists alongside compare_runs.py
-----------------------------------------
`compare_runs.py` reads the driver's "Non-outlier avg", which is wall clock per
token: megakernel + Python + whatever else the host did. That is the right
number to quote, because it is what a caller experiences, and it is the number
comparable to fleet's own.

It is the wrong number to *attribute* with. A kernel-side change of a few
microseconds per token sits under the host's own variance, and the outlier
filter -- a threshold on total time -- can differ between arms for reasons that
have nothing to do with the kernel, which moves the average of what remains.

`[FLEET_MK_TIME] layers=` is emitted by the megakernel itself, once per token,
and covers only the 36-layer body. No Python, no launch overhead, no tokenizer.
Several hundred samples per run rather than one summary number, so the median is
stable and the distribution is visible.

Read BOTH. A change that moves `layers` and not the wall clock is real but
hidden by host overhead; one that moves the wall clock and not `layers` is not
in the kernel at all.

Paired deltas, not pooled means
-------------------------------
Runs drift over a session. `tools/ab_interleave.sh` alternates arms so the
drift has no preferred arm, and this reads the result the way the interleave
intends: round N's arm-B minus round N's arm-A, one delta per round. If the
sign flips round to round the effect is noise however tidy the means look. A
sign that survives swapping which arm occupies the cold slot is the strongest
thing two rounds can tell you.

Usage:
    python3 tools/layers_timer_stats.py --arm base /tmp/base_run*.log \
                                        --arm km   /tmp/km_run*.log

Logs are paired by their run number (`_run<N>.log`), which is what
ab_interleave.sh writes, so round K of one arm lines up with round K of the
other.
"""

import argparse
import os
import re
import statistics as st
import sys

# The megakernel prints one of these per decode token.
LINE = re.compile(r"\[FLEET_MK_TIME\][^\n]*?\blayers=([\d.]+)us")
RUN_N = re.compile(r"_run(\d+)\.log$")

# Scheduler hiccups show up as a single ~10 000 us token. They are real, they
# are counted in the wall-clock average where they belong, and including them
# here would swamp a few-microsecond effect with one sample. This threshold only
# governs THIS view; it is not a claim that they do not matter.
OUTLIER_US = 5000.0


def layers_us(path):
    """Every per-token `layers=` sample in one run log."""
    out = []
    with open(path, errors="replace") as fh:
        for line in fh:
            # One log line can carry several records: the driver's own prints
            # interleave with the kernel's, so a record is not always alone.
            out.extend(float(m.group(1)) for m in LINE.finditer(line))
    return out


def run_index(path):
    m = RUN_N.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs="+", action="append", metavar=("NAME", "LOG"),
                    required=True, help="arm name followed by its run logs")
    ap.add_argument("--outlier-us", type=float, default=OUTLIER_US,
                    help=f"drop samples above this (default {OUTLIER_US:g})")
    args = ap.parse_args()

    if len(args.arm) != 2:
        sys.exit("need exactly two --arm groups to compare")

    arms = {}
    for group in args.arm:
        name, logs = group[0], group[1:]
        if not logs:
            sys.exit(f"arm {name}: no logs given")
        arms[name] = logs

    print(f"device-side `layers` timer, us per token (36 layers)")
    print(f"samples above {args.outlier_us:g} us dropped as scheduler hiccups\n")
    print(f"{'arm':10} {'run':>4} {'n':>5} {'drop':>5} {'median':>9} {'mean':>9} "
          f"{'p10':>9} {'min':>9}")

    per_run = {}          # arm -> {run index: median}
    for name, logs in arms.items():
        per_run[name] = {}
        for log in sorted(logs, key=lambda p: (run_index(p) or 0, p)):
            raw = layers_us(log)
            if not raw:
                print(f"{name:10} {'?':>4} -- no [FLEET_MK_TIME] records in {log}")
                continue
            v = [x for x in raw if x < args.outlier_us]
            if not v:
                print(f"{name:10} {'?':>4} -- every sample above the threshold in {log}")
                continue
            idx = run_index(log)
            per_run[name][idx] = st.median(v)
            sv = sorted(v)
            print(f"{name:10} {str(idx):>4} {len(v):5} {len(raw)-len(v):5} "
                  f"{st.median(v):9.1f} {st.mean(v):9.1f} "
                  f"{sv[len(sv)//10]:9.1f} {sv[0]:9.1f}")

    a, b = list(arms)
    shared = sorted(set(per_run[a]) & set(per_run[b]))
    if not shared:
        sys.exit("\nno run numbers in common -- cannot pair. Name logs _run<N>.log.")

    print(f"\npaired per-round delta ({b} - {a}):")
    deltas = []
    for i in shared:
        d = per_run[b][i] - per_run[a][i]
        deltas.append(d)
        print(f"  round {i}: {per_run[a][i]:8.1f} -> {per_run[b][i]:8.1f}   {d:+7.1f} us")

    med = st.median(deltas)
    signs = {d > 0 for d in deltas}
    print(f"\nmedian paired delta : {med:+.1f} us over 36 layers "
          f"= {med/36:+.3f} us/layer")
    if len(deltas) > 1:
        print(f"delta range         : {min(deltas):+.1f} .. {max(deltas):+.1f} us")

    if len(signs) > 1:
        print(f"\nVERDICT: sign flips across rounds -- noise, not an effect.")
    elif len(deltas) < 2:
        print(f"\nVERDICT: one round only -- no sign check possible. Run more rounds.")
    else:
        direction = "SLOWER" if med > 0 else "FASTER"
        print(f"\nVERDICT: {b} is consistently {direction} than {a} across all "
              f"{len(deltas)} rounds\n         ({abs(med)/36:.3f} us/layer, "
              f"{abs(med)/st.median([per_run[a][i] for i in shared])*100:.2f}%). "
              f"Consistent sign is evidence;\n         magnitude this small still "
              f"wants the wall-clock number before shipping.")


if __name__ == "__main__":
    main()
