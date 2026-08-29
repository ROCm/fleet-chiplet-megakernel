#!/usr/bin/env python3
"""Run the fleet_mk/vLLM harness N times and classify each run's failure mode.

Why this exists
---------------
The MoE fault under SPLIT_SCALES + K_STRIDE=3072 is PROBABILISTIC -- it hits
roughly one run in three. Three separate investigations were derailed by
judging an arm on a single run: a passing run was read as "fixed" and a
crashing run as "always crashes". Neither conclusion is supportable at this
rate. Six runs is the minimum that distinguishes 0/6 from 2/6.

The second trap this encodes: the harness log's TAIL is only the shutdown
sequence. The real GPU error sits ~9 KB earlier, in the middle of the log, and
`rc` alone does not identify it -- the same underlying fault surfaces as at
least three different exit signatures:

  rc=134 + HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION  (queue aborts, then throws)
  rc=124 + "Memory access fault by GPU node-N ... address (nil)"  (driver kills
           the queue, the process then STALLS in shutdown and hits the timeout)
  rc=124 with no fault line at all                     (a genuine hang)

Reading only `rc` conflates the last two, which is exactly the mistake that
recorded this fault as "a timeout" for three windows. Classification here is by
log CONTENT, not by exit code.

Usage
-----
  python3 tools/repeat_harness.py --runs 6 --so /tmp/gpt_oss_120b.so.split_k \\
      --env FLEET_MK_MOE_SPLIT_SCALES=1 FLEET_MK_MOE_SPLIT_BUFFERS=1 \\
            FLEET_MK_MOE_K_STRIDE=3072

Runs are sequential by construction: never run two MPK arms concurrently.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

FLEET_MK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = "/home/claudeuser/venv-vllm027/bin/python3"
LIVE_SO = os.path.join(FLEET_MK_DIR, "generated", "gpt_oss_120b.so")

# Ordered most-specific first: the aperture violation and the node fault are
# both real GPU memory faults; a bare timeout with neither is a hang.
FAULT_MARKERS = [
    ("aperture", "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION"),
    ("memfault", "Memory access fault by GPU node"),
    ("illegal", "an illegal memory access was encountered"),
]


def classify(log_text: str, rc: int) -> tuple[str, str]:
    """Return (verdict, detail). Content first, exit code only as a fallback."""
    for name, marker in FAULT_MARKERS:
        if marker in log_text:
            line = next(l for l in log_text.split("\n") if marker in l)
            return "CRASH", f"{name}: {line.strip()[:160]}"
    if rc == 124:
        return "HANG", "timed out with no GPU fault line in the log"
    if rc != 0:
        return "FAIL", f"rc={rc}, no known fault marker"
    return "PASS", ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6,
                    help="repeat count; below 6 cannot separate 0/6 from 2/6")
    ap.add_argument("--so", help="path to a .so to install before running")
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=200)
    ap.add_argument("--logdir", default="/tmp/repeat_harness")
    ap.add_argument("--env", nargs="*", default=[],
                    help="extra KEY=VALUE pairs passed to the harness")
    args = ap.parse_args()

    if args.so:
        shutil.copy(args.so, LIVE_SO)
        os.chmod(LIVE_SO, 0o755)
        subprocess.run(["sync"], check=False)
    md5 = subprocess.run(["md5sum", LIVE_SO], capture_output=True,
                         text=True).stdout.split()[0]
    print(f"[repeat_harness] .so md5 {md5}")

    os.makedirs(args.logdir, exist_ok=True)

    env = dict(os.environ)
    env.update({
        "VLLM_PLUGINS": "fleet_mk",
        "FLEET_MK_MODEL": "/home/claudeuser/models/gpt-oss-120b",
        "HIP_VISIBLE_DEVICES": "0",
        "FLEET_MK_MAX_TOKENS": str(args.tokens),
    })
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    tally: dict[str, int] = {}
    for i in range(1, args.runs + 1):
        log_path = os.path.join(args.logdir, f"run{i}.log")
        t0 = time.time()
        with open(log_path, "w") as fh:
            # ulimit -c 0: a GPU core dump is ~30 MB and the disk sits at 94%.
            proc = subprocess.run(
                ["bash", "-c",
                 f"ulimit -c 0; exec timeout {args.timeout} {PYTHON} "
                 f"-m fleet_megakernel_vllm.harness"],
                cwd=FLEET_MK_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT)
        text = open(log_path, errors="replace").read()
        verdict, detail = classify(text, proc.returncode)
        tally[verdict] = tally.get(verdict, 0) + 1
        print(f"[run {i}/{args.runs}] {verdict:5s} rc={proc.returncode} "
              f"{time.time() - t0:.0f}s  {log_path}")
        if detail:
            print(f"           {detail}")
        # Cores are disabled above, but a prior arm may have left some behind.
        for pat in ("/tmp/gpucore*", f"{FLEET_MK_DIR}/gpucore*"):
            subprocess.run(["bash", "-c", f"rm -f {pat}"], check=False)

    print(f"\n[repeat_harness] {args.runs} runs: " +
          ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    crashes = tally.get("CRASH", 0) + tally.get("HANG", 0)
    print(f"[repeat_harness] failure rate {crashes}/{args.runs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
