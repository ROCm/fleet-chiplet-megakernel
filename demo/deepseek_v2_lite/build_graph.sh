#!/usr/bin/env bash
# Generate task_graph.json + test.cu for the DeepSeek-V2-Lite decode graph on the MI300X box.
# Mirrors demo/qwen3/demo_30B_A3B.py --use-mirage ("Step 1/2" of the top-level README), with our
# fleet_ds/mirage_graph.py in place of the demo. Cannot run on the laptop (needs ROCm clang++).
set -euo pipefail
cd "$(dirname "$0")"
FLEET=${FLEET:-../..}   # the fleet-chiplet-megakernel checkout
OUT=${OUT:-build/graph}

# 0. one-time: Fleet toolchain (Mirage C++ core + Cython bridge) = `AMDGPU_TARGETS=gfx942 pip install -e .` at the repo root
#    (see README.md here). Not repeated here.
PY="${PYTHON:-$PWD/.venv/bin/python}"; [ -x "$PY" ] || PY=python3
"$PY" -c "import mirage" 2>/dev/null || { echo "import mirage failed: pip install -e . at the repo root first (same venv on PATH)"; exit 1; }
export MIRAGE_HOME="$(cd "$FLEET" && pwd)"
export AMDGPU_TARGETS=gfx942

# 1. graph only (no kernels needed): writes $OUT/task_graph.json and $OUT/test.cu via kn_graph.generate_task_graph,
#    exactly what demo_30B_A3B.py does before mpk.compile(). register_task() asserts that every name in
#    fleet_ds/mirage_graph.py:TASK is registered (Fleet's task_register.cc or src/kernel/ds_task_register.cc), so this
#    fails if src/kernel/ds_task_register.cc was not part of the runtime library build.
rm -rf permanent_output_dir            # README: stale .so silently reused otherwise
if [ -z "${MODEL:-}" ]; then MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Base; [ -d models/DeepSeek-Coder-V2-Lite-Base ] && MODEL=models/DeepSeek-Coder-V2-Lite-Base; fi
"$PY" -u -m fleet_ds.mirage_graph --model "$MODEL" \
  --max-seq-length "${T_MAX:-1056}" --output-dir "$OUT" ${COMPILE:+--compile} ${NO_GPU:+--no-gpu}

# 2. cross-check against the local builder (decision 4: graph.py is the reference count)
"$PY" -u -m fleet_ds.graph "${T_MAX:-1056}" > "$OUT/graph_local.json"
"$PY" - "$OUT" <<'PY'
import json, sys
m = json.load(open(f"{sys.argv[1]}/task_graph.json")); g = json.load(open(f"{sys.argv[1]}/graph_local.json"))
print("mirage tasks/events:", len(m["all_tasks"]), len(m["all_events"]), " local:", len(g["tasks"]), len(g["events"]))
# decision 12a: register_mugraph's events (src/kernel/runtime.cc) == graph.py's derived producers, for every op; every
# event's num_triggers == the tasks that trigger it (Mirage splits some boundaries by gcd, so events are compared per op)
mt, gt, M = m["all_tasks"], g["tasks"], 0xFFFFFFFF
assert len(mt) == len(gt), (len(mt), len(gt))
op = [t["op"] for t in gt]
trig, got = {}, {}
for i in range(1, len(mt)):
    trig.setdefault(mt[i]["trigger_event"] & M, []).append(i)
for e, ts in trig.items():
    assert m["all_events"][e]["num_triggers"] == len(ts), ("num_triggers", e, m["all_events"][e]["num_triggers"], len(ts))
for i in range(2, len(mt)):
    got.setdefault(op[i], set()).update(op[j] for j in trig.get(mt[i]["dependent_event"] & M, ()) if j >= 2)
bad = [(o["id"], o["layer"], o["type"], sorted(got.get(o["id"], ())), o["producers"]) for o in g["ops"]
       if sorted(got.get(o["id"], ())) != sorted(o["producers"])]
assert not bad, ("register_mugraph producers != graph.py", bad[:5])
print(f"events ok: {len(trig)} triggered events recounted, producers of all {len(g['ops'])} ops == graph.py", flush=True)
PY
# 3. COMPILE=1 to also hipcc the megakernel into permanent_output_dir/ (needs every task kernel implemented).
#    NO_GPU=1: shapes-only weights, CPU torch, and COMPILE=1 runs the same hipcc line
#    without importing the .so -> $OUT/test_rank0.so (fleet_ds/mirage_graph.py --no-gpu).
