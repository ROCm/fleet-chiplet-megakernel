"""Every script in demo/deepseek_v2_lite parses; run_layer/run_e2e --dry-run build the plan + arena on CPU; the
rocprof parser runs on a fixture. No ROCm, no model, no Mirage needed."""
import glob
import json
import os
import py_compile
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = ROOT  # the scripts sit next to fleet_ds/
PY = sys.executable
sys.path.insert(0, SCRIPTS)


@pytest.mark.parametrize("sh", sorted(os.path.basename(p) for p in glob.glob(os.path.join(SCRIPTS, "*.sh"))))
def test_bash_n(sh):
    subprocess.check_call(["bash", "-n", os.path.join(SCRIPTS, sh)])


@pytest.mark.parametrize("py", sorted(os.path.basename(p) for p in glob.glob(os.path.join(SCRIPTS, "*.py"))))
def test_py_compile(py):
    py_compile.compile(os.path.join(SCRIPTS, py), doraise=True)


def run(*args, knobs=""):
    return subprocess.run([PY, *args], capture_output=True, text=True, cwd=ROOT, check=True, env={**os.environ, "FLEET_KNOBS": knobs}).stdout


KNOBS = pytest.mark.parametrize("knobs, layer13, token", [("", "total ops 37, tasks 2543", "total ops 298, tasks 20492"),  # default: DAG + folds
                                                          ("dag=0,fold=0", "total ops 38, tasks 2598", "total ops 301, tasks 20979")],  # the chain
                                ids=["default", "chain"])


@KNOBS
def test_run_layer_dry_run(knobs, layer13, token):
    out = run(os.path.join(SCRIPTS, "run_layer.py"), "--layers", "1-3", "--dry-run", knobs=knobs)
    assert layer13 in out and "EXPERT_W13_SILU" in out and "GATE_UP_SILU" not in out
    assert "L0.K                         (16, 1056, 192)" in out and "logits                       (1, 102400)" in out
    assert "max_seq_length 1026 -> 1 iteration" in out


@KNOBS
def test_run_e2e_dry_run(knobs, layer13, token):
    out = run(os.path.join(SCRIPTS, "run_e2e.py"), "--dry-run", knobs=knobs)
    assert token in out and "first_k_dense=1" in out and "GATE_UP_SILU" in out
    assert "max_seq_length 1057 -> 32 iteration" in out


def test_rocprof_parser(tmp_path):
    import bench
    csv = tmp_path / "run_kernel_trace.csv"
    csv.write_text("Kernel_Id,Kernel_Name,Start_Timestamp,End_Timestamp,LDS_Block_Size,Scratch_Size,VGPR_Count,Accum_VGPR_Count,SGPR_Count,Workgroup_Size_X,Grid_Size_X\n"
                   "1,prepare_kernel(RuntimeConfig),1000,3000,0,0,32,0,48,128,37888\n"
                   "2,worker_kernel(RuntimeConfig),3000,103003000,58368,0,128,64,96,256,75776\n"
                   "3,scheduler_kernel(RuntimeConfig),3000,103002000,0,0,40,0,64,128,1024\n")
    r = bench.parse_kernel_trace([str(csv)])
    assert r["total_dispatches"] == 3 and r["kernels"]["worker_kernel(RuntimeConfig)"]["us_p50"] == 103000.0
    occ = r["kernels"]["worker_kernel(RuntimeConfig)"]["occupancy"]
    assert occ["waves_per_simd_vgpr_bound"] == 2 and occ["workgroups_per_cu_lds_bound"] == 1 and not occ["spills"]

