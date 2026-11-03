"""The runtime glue for the DeepSeek-V2-Lite tasks, checked without a GPU, against the files in this checkout:
include/mirage/persistent_kernel/runtime_header.h (TaskType enum), src/kernel/graph.cc (registration hook),
src/kernel/runtime.cc (bid packing, codegen), src/kernel/ds_task_register.cc (per-name register functions) and
include/mirage/persistent_kernel/kernels/fleet_dispatch.cuh (the dispatch glue the generated _execute_task calls).
"""
import os
import re
import shutil
import subprocess
import sysconfig
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fleet_ds.mirage_graph import TASK, layer_plan  # noqa: E402
from fleet_ds.sim import CPU_FLAGS, KERNELS  # noqa: E402

REPO = os.path.dirname(os.path.dirname(ROOT))
HEADER = os.path.join(REPO, "include", "mirage", "persistent_kernel", "runtime_header.h")
GRAPH_CC = os.path.join(REPO, "src", "kernel", "graph.cc")
RUNTIME_CC = os.path.join(REPO, "src", "kernel", "runtime.cc")
REGISTER_CC = os.path.join(REPO, "src", "kernel", "ds_task_register.cc")
# register name -> (enum, num_inputs, num_outputs); gemv_bf16 has the 4-operand KV_B_PROJ variant.
# The norm weight is the gemv/gate prologue's extra input (fold); ATTN_CHUNK/KV_B get qkva + cos/sin.
CONTRACT = {"gemv_bf16": ("TASK_GEMV_BF16", 3, 1), "gemv_bf16_silu": ("TASK_GEMV_BF16_SILU", 3, 1),
            "gemv_bf16_res": ("TASK_GEMV_BF16_RES", 4, 1), "gemv_splitk": ("TASK_GEMV_SPLITK", 2, 1),
            "attn_chunk_192_128": ("TASK_ATTN_CHUNK_192_128", 5, 2), "moe_gate_topk": ("TASK_MOE_TOPK", 4, 3),
            "moe_expert_w13_silu": ("TASK_MOE_EXPERT_W13_SILU", 4, 1), "moe_expert_w2": ("TASK_MOE_EXPERT_W2", 4, 1),
            "moe_combine": ("TASK_MOE_COMBINE", 4, 1)}
BID_READERS = {"TASK_GEMV_BF16", "TASK_ATTN_CHUNK_192_128", "TASK_ATTN_MERGE", "TASK_MOE_EXPERT_W13_SILU",
               "TASK_MOE_EXPERT_W2"}
pytestmark = pytest.mark.skipif(not os.path.isfile(REGISTER_CC), reason="not inside a fleet-chiplet-megakernel checkout")


def read(*p):
    with open(os.path.join(*p)) as f:
        return f.read()


def test_enum_mirrors_task_types_h():
    ours = dict(re.findall(r"^\s*(TASK_\w+) = (\d+),", read(KERNELS, "task_types.h"), re.M))
    fleet = dict(re.findall(r"^\s*(TASK_\w+) = (\d+),", read(HEADER), re.M))
    by_value = {v: k for k, v in fleet.items()}
    assert len(by_value) == len(fleet), "duplicate TaskType values in Fleet's enum"
    for name, v in ours.items():  # every value 300-315 present; graph-emitted names identical (fallbacks get _DS)
        assert v in by_value, (name, v)
        assert by_value[v] == name or by_value[v] == name + "_DS", (name, by_value[v])
    for name, _, _ in CONTRACT.values():
        assert fleet[name] == ours[name]
    assert max(int(v) for k, v in fleet.items() if k not in ours and not k.endswith("_DS")
               and k != "TASK_DS_END") < 300  # no clash with Fleet's own values
    assert not any(k.startswith("EVENT_") for k in fleet)


def test_every_graph_name_registers():
    fleet_names = set(re.findall(r'name == "(\w+)"', read(GRAPH_CC)))
    ds_names = set(re.findall(r'name == "(\w+)"', read(REGISTER_CC)))
    assert ds_names == set(CONTRACT) | {"mla_prep", "attn_merge", "moe_topk"}  # legacy chain branches; moe_topk = gate_tiles knob
    assert "ds_register_task(name, customized->bgraph, params, task_config[op])" in read(GRAPH_CC)
    assert set(TASK.values()) <= fleet_names | ds_names
    assert {t for kw in ({}, dict(dag=0, fold=0)) for _, t, _ in layer_plan(**kw)} <= set(TASK)  # default (DAG + folds) and the chain


def test_register_functions_match_contract():
    cc = read(REGISTER_CC).split("// Legacy: pre-D11b")[0]  # current graph shapes only
    for name, (enum, n_in, n_out) in CONTRACT.items():
        m = re.search(rf"gemv\(runtime::{enum}, (\d+),", cc) or re.search(rf"reg\(runtime::{enum}, (\d+), (\d+),", cc)
        assert m, name
        got = (int(m.group(1)), int(m.group(2)) if m.lastindex == 2 else 1)
        assert got == (n_in, n_out), (name, got)
    assert re.search(r"reg\(runtime::TASK_GEMV_BF16, 5, 2,", cc)  # KV_B_PROJ variant: K, V declared as outputs


def test_dispatch_arity_and_names():
    cuh, cc, cpu = read(KERNELS, "fleet_dispatch.cuh"), read(REGISTER_CC), read(ROOT, "tests", "dispatch_cpu.cc")
    fns = {}
    for tpl, fn in re.findall(r"template <([^>]*)>\s*__device__ __forceinline__ void (ds_\w+)\(", cuh):
        fns[fn] = sum(1 for p in tpl.split(",") if not p.strip().startswith("class"))
    assert set(fns) == {"ds_gemv", "ds_kv_b_proj", "ds_attn_chunk", "ds_attn_merge", "ds_moe_gate_topk",
                        "ds_expert_w13", "ds_expert_w2", "ds_moe_combine", "ds_gemv_norm", "ds_moe_topk", "ds_o_proj_merge",
                        "ds_moe_gate_topk_norm"}
    for fn, n_tpl in fns.items():
        assert f"dsk::{fn}" in cc, fn
        calls = re.findall(rf"dsk::{fn}<([^>]*)>\(task_desc, runtime_config\)", cpu)
        assert calls, fn
        assert all(len(c.split(",")) == n_tpl for c in calls), (fn, n_tpl, calls)
    assert cpu.count("task_desc->task_type == TASK_") == 18  # 10 names + KV_B_PROJ's and LM_HEAD's gemv_bf16 variants + the DAG's shared-expert GEMVs


def test_runtime_cc_branches():
    rt = read(RUNTIME_CC)
    m = re.search(r"DeepSeek-V2-Lite tasks that read bid.*?\{", rt, re.S)
    assert set(re.findall(r"task_type == (TASK_\w+)", m.group(0))) == BID_READERS
    named = set(re.findall(r"task_type_to_name\[(TASK_\w+)\] = \"\1\"", rt))
    assert {e for e, _, _ in CONTRACT.values()} <= named
    assert 'code.e("#include \\"kernels/fleet_dispatch.cuh\\"")' in rt
    assert rt.index('#include \\"persistent_kernel.cuh\\"') < rt.index("kernels/fleet_dispatch.cuh")


def test_dispatch_cpu_compiles_and_mocks_real_fields():
    subprocess.run(["clang++", "-std=c++17", *CPU_FLAGS, "-fsyntax-only", "-Wall", "-Werror",
                    os.path.join(ROOT, "tests", "dispatch_cpu.cc")], check=True)
    header = read(HEADER)
    cpu = read(ROOT, "tests", "dispatch_cpu.cc")
    mock = re.search(r"namespace runtime \{(.*?)\}  // namespace runtime", cpu, re.S).group(1)
    for line in mock.splitlines():  # every mocked field/constant is verbatim a line of Fleet's header
        line = line.strip()
        if re.match(r"(int|unsigned|void|struct|union|enum|TaskType|TASK_|FullTaskDesc)", line) and not line.startswith("struct {"):
            assert line in header, line
    used = set(re.findall(r"(?:t->|task_desc->|rc\.|runtime_config\.)([\w.]+)", read(KERNELS, "fleet_dispatch.cuh")))
    assert used == {"input_ptrs", "output_ptrs", "task_metadata.expert_offset", "step"}


def test_softmax_scale_literal():
    from tests.test_reference import COMMON, REAL, DeepseekV2Config  # ref/hf remote code
    from fleet_ds import reference as R
    lit = float(re.search(r"DS_SOFTMAX_SCALE = ([0-9.]+)f", read(KERNELS, "fleet_dispatch.cuh")).group(1))
    assert np.float32(lit) == np.float32(R.softmax_scale(DeepseekV2Config(**COMMON, **REAL)))


def _json_include():
    for d in [os.environ.get("NLOHMANN_JSON_INCLUDE", ""), "/usr/include", "/usr/local/include",
              os.path.join(REPO, "deps", "json", "include")]:
        if d and os.path.isfile(os.path.join(d, "nlohmann", "json.hpp")):
            return d
    return None


@pytest.mark.skipif(_json_include() is None, reason="nlohmann/json.hpp not found (set NLOHMANN_JSON_INCLUDE)")
def test_host_glue_syntax_against_fleet_headers(tmp_path):
    """ds_task_register.cc + graph.cc + runtime.cc against Fleet's real host headers (no HIP: the NKI backend
    branch of mirage/vector_types.h defines dim3/int3; cuda_runtime.h is stubbed to the four tokens it needs)."""
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "cuda_runtime.h").write_text("#pragma once\n#define __host__\n#define __device__\n#define __global__\n"
                                         "#define __forceinline__ inline\ntypedef void *cudaStream_t; "
                                         "typedef void *cudaEvent_t; typedef int cudaError_t;\n")
    (stub / "vector_types.h").write_text("")
    inc = os.path.join(REPO, "include")
    for src in (REGISTER_CC, GRAPH_CC, RUNTIME_CC):
        subprocess.run(["clang++", "-std=c++17", "-fsyntax-only", "-w", "-DMIRAGE_BACKEND_USE_NKI", f"-I{stub}",
                        f"-I{inc}", f"-I{_json_include()}", src], check=True)


def test_mpk_debug_c_syntax(tmp_path):
    """mpk_debug.c (spliced into Mirage's HARD_CODE by run_layer.splice_hardcode) compiles against a stub of
    the runtime symbols it touches; the splice anchor exists exactly once in Mirage's HARD_CODE."""
    funcs, entries = open(os.path.join(ROOT, "mpk_debug.c")).read().split("// ---- ModuleMethods entries ----\n")
    pk = open(os.path.join(REPO, "python", "mirage", "mpk", "persistent_kernel.py")).read()
    assert pk.count("static PyMethodDef ModuleMethods[] = {\n") == 1
    stub = """#include <Python.h>
#include <cstddef>
typedef struct ihipStream_t *hipStream_t; typedef hipStream_t cudaStream_t;
enum hipError_t { hipSuccess = 0, hipErrorUnknown = 999 };
enum hipMemcpyKind { hipMemcpyDeviceToHost = 2 };
enum { hipStreamNonBlocking = 1, hipHostMallocDefault = 0 };
hipError_t hipMemset(void *, int, size_t); hipError_t hipMemcpyAsync(void *, void const *, size_t, hipMemcpyKind, hipStream_t);
hipError_t hipStreamQuery(hipStream_t); hipError_t hipHostMalloc(void **, size_t, unsigned); hipError_t hipStreamCreateWithFlags(hipStream_t *, unsigned);
#define MPK_MAX_NUM_BATCHED_REQUESTS 1
struct RuntimeConfig {
  int num_workers, num_local_schedulers, num_remote_schedulers, num_events, num_xcds, max_seq_length, total_num_requests, event_timing_max_entries;
  long long eos_token_id;
  unsigned long long *worker_queue_last_ready_task_id, *sched_queue_last_ready_event_id, *sched_queue_next_free_event_id,
      *all_event_counters, *xcd_local_event_counters, *event_timing_buffer;
  int *all_event_num_triggers, *xcd_event_num_tasks, *worker_xcd_ready_count, *step, *request_ids, *next_request_id,
      *page_queue_head, *page_queue_tail, *qo_indptr_buffer, *event_timing_count;
  long long *input_tokens, *output_tokens;
};
static RuntimeConfig global_runtime_config;
void launch_persistent_kernel(cudaStream_t);
"""
    (tmp_path / "t.cc").write_text(stub + funcs + "static PyMethodDef ModuleMethods[] = {\n" + entries + "  {NULL, NULL, 0, NULL}};\n"
                                   "void *use_table = (void *)ModuleMethods;\n")
    subprocess.run(["clang++", "-std=c++17", "-fsyntax-only", "-Wall", "-Werror", f"-I{sysconfig.get_paths()['include']}",
                    str(tmp_path / "t.cc")], check=True)
