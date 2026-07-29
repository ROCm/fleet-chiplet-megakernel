import torch
import os
import tempfile
import subprocess
import shutil
import sys
import sysconfig

from ..core import *
from ..kernel import get_key_paths, KNGraph, TBGraph
from .speculative import (
    SpecDecodeConfig,
    PromptLookupConfig,
)
from typing import Optional

HARD_CODE = """
#include <Python.h>

// HIP/CUDA runtime abstraction
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300) || defined(MIRAGE_BACKEND_USE_ROCM)
#include <hip/hip_runtime.h>
typedef hipStream_t cudaStream_t;
#else
#include <cuda_runtime.h>
#endif

static PyObject *init_func(PyObject *self, PyObject *args) {
  PyObject *meta_list, *py_profiler_buffer;
  std::vector<void*> meta_tensors;
  int my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers, max_seq_length, total_num_requests;
  long long eos_token_id;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, "OOiiiiiiL", &meta_list, &py_profiler_buffer, &my_mpi_rank, &num_workers, &num_local_schedulers, &num_remote_schedulers, &max_seq_length, &total_num_requests, &eos_token_id)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }

  if(!PyList_Check(meta_list)) {
    PyErr_SetString(PyExc_TypeError, "arg1 must be a list.");
    return NULL;
  }

  Py_ssize_t meta_size = PyList_Size(meta_list);

  for(Py_ssize_t i = 0; i < meta_size; i++) {
    PyObject *item = PyList_GetItem(meta_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (meta) to void pointer", i);
      return NULL;
    }
    meta_tensors.push_back(PyLong_AsVoidPtr(item));
  }
  profiler_buffer = PyLong_AsVoidPtr(py_profiler_buffer);

  init_persistent_kernel(meta_tensors, profiler_buffer, my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers, max_seq_length, total_num_requests, eos_token_id);

  Py_RETURN_NONE;
}

static PyObject *init_request_func(PyObject *self, PyObject *args) {
  Py_BEGIN_ALLOW_THREADS
  init_request_resources();
  Py_END_ALLOW_THREADS
  Py_RETURN_NONE;
}

static PyObject *launch_func(PyObject *self, PyObject *args) {
  PyObject *py_stream;
  cudaStream_t stream;
  if (!PyArg_ParseTuple(args, "O", &py_stream)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }
  stream = (cudaStream_t)PyLong_AsVoidPtr(py_stream);
  launch_persistent_kernel(stream);

  Py_RETURN_NONE;
}

static PyObject *finalize_func(PyObject *self, PyObject *args) {
  finalize_persistent_kernel();

  Py_RETURN_NONE;
}

static PyObject *set_rope_tables_func(PyObject *self, PyObject *args) {
  PyObject *py_cos, *py_sin;
  if (!PyArg_ParseTuple(args, "OO", &py_cos, &py_sin)) {
    PyErr_SetString(PyExc_TypeError, "Expected (cos_ptr, sin_ptr)");
    return NULL;
  }
  void *cos_ptr = PyLong_AsVoidPtr(py_cos);
  void *sin_ptr = PyLong_AsVoidPtr(py_sin);
  set_rope_tables(cos_ptr, sin_ptr);
  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"init_func", init_func, METH_VARARGS, "initialize persistent kernel"},
  {"init_request_func", init_request_func, METH_VARARGS, "initialize request resources"},
  {"launch_func", launch_func, METH_VARARGS, "launch persistent kernel"},
  {"finalize_func", finalize_func, METH_VARARGS, "finalize persistent kernel"},
  {"set_rope_tables_func", set_rope_tables_func, METH_VARARGS, "set RoPE cos/sin tables"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__mirage_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods,
  NULL, // m_slots
  NULL, // m_traverse
  NULL, // m_clear
  NULL  // m_free
};

PyMODINIT_FUNC PyInit___mirage_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
"""

valid_persistent_kernel_modes = {"offline", "online", "online_notoken", "onepass", "online_multi_turn"}

def get_compile_command(
    mpk,
    target_cc,
    cc,
    file_name,
    py_include_dir,
    mirage_home_path,
    mirage_inc_path,
    mirage_deps_path,
    nvshmem_inc_path,
    nvshmem_lib_path,
    mpi_inc_path,
    mpi_lib_path,
    py_so_path,
    profiling,
    use_nvshmem,
    num_workers=None,
    num_local_schedulers=None,
    num_remote_schedulers=None,
    use_cutlass_kernel=True,
):
    max_worker_per_scheduler = 128
    if num_workers != None and num_local_schedulers != None and num_remote_schedulers != None:
        min_schedulers = 0
        if num_remote_schedulers == 0:
            min_schedulers = num_local_schedulers
        else:
            min_schedulers = min(num_local_schedulers, num_remote_schedulers)
        # advance by 1 for the scheduler who are handling the not divisiable num_worker.
        max_worker_per_scheduler = (num_workers // min_schedulers) + 1

    common_cmd = [
        cc,
        # "--default-stream per-thread" is used to create new stream for 
        # each host thread as default stream instead of using the same 
        # legacy stream for all host threads
        # This is important in multi-threaded environment.
        # "--default-stream",
        # "per-thread",
        file_name,
        "-O3",
        # Use following flags when debugging
        # "-O0",
        # "-g",
        # "-G",
        # "--ptxas-options=-v",
        # "-Xptxas=-v",
        "-lineinfo",
        f"-I{py_include_dir}",
        f"-I{mirage_inc_path}",
        f"-I{os.path.join(mirage_inc_path, 'mirage/persistent_kernel')}",
        f"-I{os.path.join(mirage_deps_path, 'cutlass/include')}",
        f"-I{os.path.join(mirage_deps_path, 'cutlass/tools/util/include')}",
        f"-I{os.path.join(mirage_home_path, 'deps/json/include')}",
        f"-DMAX_WORKER_PER_SCHEDULER={max_worker_per_scheduler}",
        f"-DMIRAGE_USE_CUTLASS_KERNEL={'1' if use_cutlass_kernel else '0'}",
    ]

    flags = [
        "-shared",
        "-std=c++17",
        "-rdc=false" if not use_nvshmem else "-rdc=true",
        "-use_fast_math",
        "-lcuda",
        "-Xcompiler=-fPIC",
        "--expt-relaxed-constexpr",
        "-o",
        py_so_path,
    ]
    flags = flags + [f"-DMPK_TARGET_CC={target_cc}", "-DMIRAGE_BACKEND_USE_CUDA"]

    if mpk.mode == "offline":
        flags = flags + ["-DMODE_OFFLINE"]
    elif mpk.mode == "online":
        flags = flags + ["-DMODE_ONLINE"]
    elif mpk.mode == "online_notoken":
        flags = flags + ["-DMODE_ONLINE_NOTOKEN"]
    elif mpk.mode == "onepass":
        flags = flags + ["-DMODE_ONEPASS"]
    elif mpk.mode == "online_multi_turn":
        flags = flags + ["-DMODE_MULTI_TURN"]
    else:
        raise ValueError(f"Invalid persistent kernel mode: {mpk.mode}")

    flags = flags + [f"-DMPK_MAX_NUM_BATCHED_REQUESTS={mpk.max_num_batched_requests}"]
    flags = flags + [f"-DMPK_MAX_NUM_BATCHED_TOKENS={mpk.max_num_batched_tokens}"]
    flags = flags + [f"-DMPK_MAX_NUM_PAGES={mpk.max_num_pages}"]
    flags = flags + [f"-DMPK_PAGE_SIZE={mpk.page_size}"]
    flags = flags + [f"-DMPK_MAX_SEQ_LENGTH={mpk.max_seq_length}"]
    # Enable gang task support (adds extra code paths in persistent kernel)
    if int(os.environ.get("USE_GANG", "0")) == 1:
        flags = flags + ["-DMPK_ENABLE_GANG_TASKS"]
    if int(os.environ.get("USE_NT_WEIGHTS", "0")) == 1:
        flags = flags + ["-DMPK_NT_WEIGHT_LOADS"]
    if int(os.environ.get("W13_LDS_WEIGHTS", "0")) == 1:
        flags = flags + ["-DMPK_W13_LDS_WEIGHTS"]
    if int(os.environ.get("W13_LDS_PREFETCH", "1")) == 1:
        flags = flags + ["-DMPK_W13_LDS_PREFETCH"]
    # Use when debugging
    # flags = flags + [f"-DMPK_ENABLE_VERBOSE"]
    if int(os.environ.get("PRECOMPUTED_DISPATCH", "1")) == 1:
        flags = flags + ["-DMPK_PRECOMPUTED_DISPATCH"]
    if int(os.environ.get("TRACE_MOE", "0")) == 1:
        flags = flags + ["-DMPK_TRACE_MOE_DISPATCH"]
    if int(os.environ.get("EMBED_DEBUG", "0")) == 1:
        flags = flags + ["-DEMBED_DEBUG"]
    if int(os.environ.get("MPK_DEBUG_LAYERS", "0")) == 1:
        flags = flags + ["-DMPK_DEBUG_RMSNORM", "-DMPK_DEBUG_MOE_MUL_SUM"]
    # Enable debug output for HIP builds
    # if target_cc == 94:
    #    flags = flags + ["-DMPK_ENABLE_VERBOSE"]

    if use_nvshmem:
        nvshmem_cmd = [
            f"-I{nvshmem_inc_path}",
            f"-I{mpi_inc_path}",
            f"-L{nvshmem_lib_path}",
            f"-L{mpi_lib_path}",
        ]
        nvshmem_flags = ["-DUSE_NVSHMEM", "-ccbin=mpic++", "-lnvshmem_host", "-lnvshmem_device", "-lmpi"]
        common_cmd = common_cmd + nvshmem_cmd
        flags = flags + nvshmem_flags

    if target_cc == 90:
        specific_cmd = [
            "-arch=sm_90a",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-DMPK_ENABLE_TMA",
            "-DMIRAGE_GRACE_HOPPER",
            "-DNDEBUG",
        ] + (["-DMIRAGE_ENABLE_PROFILER"] if profiling else [])
    elif target_cc == 100:
        specific_cmd = [
            "-arch=sm_100a",
            "-gencode=arch=compute_100a,code=sm_100a",
            "-DMPK_ENABLE_TMA",
            "-DMIRAGE_GRACE_BLACKWELL",
        ]
    elif target_cc in (94, 95):
        # MI300/MI350 ROCm: use HIP. specific_cmd set below with ROCm-specific args.
        specific_cmd = []
    else:
        specific_cmd = [
            "-arch=native",
        ]

    if target_cc in (94, 95):
        # ROCm/MI300/MI350 compile path: hipcc, ROCm includes/libs
        rocm_home = os.environ.get("ROCM_PATH", "/opt/rocm")
        rocm_include = os.path.join(rocm_home, "include")
        rocm_lib = os.path.join(rocm_home, "lib")
        rocm_lib64 = os.path.join(rocm_home, "lib64") if os.path.exists(os.path.join(rocm_home, "lib64")) else rocm_lib
        rocblas_inc = os.path.join(mirage_deps_path, "rocblas", "include")
        # HIP compatibility headers directory (must come before CUTLASS includes)
        hip_compat_inc = os.path.join(mirage_inc_path, 'mirage/hip_compat')
        common_cmd = [
            cc,
            "-x", "hip",
            file_name,
            "-O2",  # -O3 causes LLVM AMDGPU register allocator to hang on large fused kernels
            "--save-temps",  # TEMP: dump assembly for v_mov analysis
            # Omit -lineinfo for ROCm: hipcc forwards it to ld.lld which treats it as -l lineinfo
            f"-I{py_include_dir}",
            f"-I{mirage_inc_path}",
            f"-I{hip_compat_inc}",  # HIP compatibility headers (before CUTLASS, includes cuda/std/ compatibility)
            f"-I{os.path.join(mirage_inc_path, 'mirage/persistent_kernel')}",
            f"-I{rocblas_inc}",  # ROCm-compatible CUTLASS headers (before main CUTLASS)
            f"-I{os.path.join(mirage_deps_path, 'cutlass/include')}",  # Main CUTLASS (fallback)
            f"-I{os.path.join(mirage_deps_path, 'cutlass/tools/util/include')}",
            # CK submodule for PagedKV FMHA (must come before system CK)
            f"-I{os.path.join(mirage_deps_path, 'composable_kernel/include')}",  # CK-tile headers
            f"-I{rocm_include}",
            f"-I{os.path.join(mirage_home_path, 'deps/json/include')}",
            f"-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
            f"-DMAX_WORKER_PER_SCHEDULER={max_worker_per_scheduler}",
            f"-DMIRAGE_USE_CUTLASS_KERNEL={'1' if use_cutlass_kernel else '0'}",
        ]
        flags = [
            "-shared",
            "-std=c++17",
            "-fPIC",
            "-D__HIP_PLATFORM_AMD__=1",
            "-DMIRAGE_AMD_MI300",
            "-DMIRAGE_BACKEND_USE_ROCM",
            f"-DMPK_TARGET_CC={target_cc}",
            f"-DMPK_MAX_NUM_BATCHED_REQUESTS={mpk.max_num_batched_requests}",
            f"-DMPK_MAX_NUM_BATCHED_TOKENS={mpk.max_num_batched_tokens}",
            f"-DMPK_MAX_NUM_PAGES={mpk.max_num_pages}",
            f"-DMPK_PAGE_SIZE={mpk.page_size}",
            f"-DMPK_MAX_SEQ_LENGTH={mpk.max_seq_length}",
            "-o",
            py_so_path,
        ]
        if mpk.mode == "offline":
            flags = flags + ["-DMODE_OFFLINE"]
        elif mpk.mode == "online":
            flags = flags + ["-DMODE_ONLINE"]
        elif mpk.mode == "online_notoken":
            flags = flags + ["-DMODE_ONLINE_NOTOKEN"]
        elif mpk.mode == "onepass":
            flags = flags + ["-DMODE_ONEPASS"]
        elif mpk.mode == "online_multi_turn":
            flags = flags + ["-DMODE_MULTI_TURN"]
        if profiling:
            flags = flags + ["-DMPK_ENABLE_PROFILING"]
            flags = flags + ["-DMPK_ENABLE_TIMING"]
            profiling_iters = int(os.environ.get("MPK_PROFILING_ITERS", "1"))
            flags = flags + [f"-DMPK_PROFILING_NUM_ITERS={profiling_iters}"]
        else:
            flags = flags + ["-DMPK_PROFILING_NUM_ITERS=0"]
        if int(os.environ.get("MPK_TIMING", "0")) == 1:
            flags = flags + ["-DMPK_ENABLE_TIMING"]
        if int(os.environ.get("MPK_DEVICE_TIMING", "0")) == 1:
            flags = flags + ["-DMPK_ENABLE_DEVICE_TASK_TIMING"]
        if int(os.environ.get("MPK_DEVICE_ACCUM", "0")) == 1:
            flags = flags + ["-DMPK_ENABLE_DEVICE_TASK_ACCUM"]
        if int(os.environ.get("MPK_SUBPHASE_TIMING", "0")) == 1:
            flags = flags + ["-DMPK_ENABLE_SUBPHASE_TIMING"]
        if int(os.environ.get("MPK_MOE_SUBPHASE", "0")) == 1:
            flags = flags + ["-DMPK_ENABLE_MOE_SUBPHASE"]
        if int(os.environ.get("MPK_FUSED_PHASE_TIMING", "0")) == 1:
            flags = flags + ["-DMPK_FUSED_PHASE_TIMING"]
        if int(os.environ.get("MPK_SPAN_TIMING", "0")) == 1:
            flags = flags + ["-DMPK_ENABLE_SPAN_TIMING"]
        if int(os.environ.get("USE_GANG", "0")) == 1:
            flags = flags + ["-DMPK_ENABLE_GANG_TASKS"]
        if int(os.environ.get("USE_NT_WEIGHTS", "0")) == 1:
            flags = flags + ["-DMPK_NT_WEIGHT_LOADS"]
        if int(os.environ.get("W13_LDS_WEIGHTS", "0")) == 1:
            flags = flags + ["-DMPK_W13_LDS_WEIGHTS"]
        if int(os.environ.get("W13_LDS_PREFETCH", "1")) == 1:
            flags = flags + ["-DMPK_W13_LDS_PREFETCH"]
        if int(os.environ.get("MPK_GAP_TIMING", "0")) == 1:
            flags = flags + ["-DMPK_ENABLE_GAP_TIMING"]
            flags = flags + ["-DMPK_ENABLE_DEVICE_TASK_ACCUM"]
        if int(os.environ.get("MPK_HBM_LATENCY", "0")) == 1:
            flags = flags + ["-DMPK_HBM_LATENCY"]
        if int(os.environ.get("MPK_MOE_SINGLE_EXPERT", "0")) == 1:
            flags = flags + ["-DMPK_MOE_SINGLE_EXPERT"]
        if int(os.environ.get("MPK_FUSED_TAIL_TIMING", "0")) == 1:
            flags = flags + ["-DMPK_FUSED_TAIL_TIMING"]
        if int(os.environ.get("MPK_K2944_DEBUG", "0")) == 1:
            flags = flags + ["-DMPK_K2944_DEBUG"]
        if int(os.environ.get("PRECOMPUTED_DISPATCH", "1")) == 1:
            flags = flags + ["-DMPK_PRECOMPUTED_DISPATCH"]
            flags = flags + ["-DMPK_FUSED_LAYER_BATCHING"]
        if int(os.environ.get("TRACE_MOE", "0")) == 1:
            flags = flags + ["-DMPK_TRACE_MOE_DISPATCH"]
        if int(os.environ.get("EMBED_DEBUG", "0")) == 1:
            flags = flags + ["-DEMBED_DEBUG"]
        if int(os.environ.get("MPK_DEBUG_LAYERS", "0")) == 1:
            flags = flags + ["-DMPK_DEBUG_RMSNORM", "-DMPK_DEBUG_MOE_MUL_SUM"]
        if int(os.environ.get("CK_FMHA_1TOK", "0")) == 1:
            # Force seqlen_q=1: uses merge path only (faster decode, slower prefill)
            flags = flags + ["-DMPK_MAX_TOKENS_PER_REQUEST=1"]
        amdgpu_target = os.environ.get("AMDGPU_TARGETS", "gfx950")
        specific_cmd = [
            f"--offload-arch={amdgpu_target}",
            f"-L{rocm_lib}",
            f"-L{rocm_lib64}",
            "-lamdhip64",
            "-lrocblas",
            "-lhipblas",
        ]
        return common_cmd + specific_cmd + flags

    if profiling:
        flags = flags + ["-DMPK_ENABLE_PROFILING"]

    return common_cmd + specific_cmd + flags


class PersistentKernel:
    def __init__(
        self,
        mode: str,
        world_size: int,
        mpi_rank: int,
        num_workers: int,
        num_local_schedulers: int,
        num_remote_schedulers: int,
        max_seq_length: int,
        max_num_batched_requests: int,
        max_num_batched_tokens: int,
        max_num_pages: int,
        page_size: int,
        meta_tensors: dict,
        profiler_tensor: torch.Tensor,
        trace_name: str,
        spec_decode_config: SpecDecodeConfig,
        use_cutlass_kernel: bool,
        eos_token_id: int64 = -1,
    ):
        self.__finalized__ = False
        self._is_compiled = False
        self._dummy_counter = 0
        self._dummy_tensor_refs = []  # prevent GC of dummy tensors (pointer reuse)
        if mode not in valid_persistent_kernel_modes:
            raise ValueError(f"Invalid persistent kernel mode: {mode}")
        self.mode = mode
        self.world_size = world_size
        self.mpi_rank = mpi_rank
        self.num_workers = num_workers
        self.num_local_schedulers = num_local_schedulers
        self.num_remote_schedulers = num_remote_schedulers
        self.max_seq_length = max_seq_length
        self.max_num_batched_requests = max_num_batched_requests
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_pages = max_num_pages
        self.page_size = page_size
        self.eos_token_id = eos_token_id
        self.kn_graph = KNGraph(CyKNGraph(disable_fingerprint=True))
        self.meta_tensors = meta_tensors
        self.profiler_tensor = profiler_tensor
        self.trace_name = trace_name
        self.use_nvshmem = True if world_size > 1 else False
        self.spec_decode_config = spec_decode_config
        self._spec_decode_handlers = {
            "promptlookup": self.prompt_lookup_spec_handler,
        }
        self._spec_verify_handlers = {
            "promptlookup": self.prompt_lookup_verify_handler,
        }
        # determine total number of requests for offline serving
        self.total_num_requests = meta_tensors["tokens"].shape[0]
        assert self.max_seq_length == meta_tensors["tokens"].shape[1]
        self.is_rocm = bool(getattr(torch.version, "hip", None))
        # Force CUTLASS off on ROCm - CUTLASS requires CUDA tensor cores
        if self.is_rocm:
            self.use_cutlass_kernel = False
        else:
            self.use_cutlass_kernel = use_cutlass_kernel
        if self.is_rocm:
            # Detect AMD GPU generation from offload-arch target
            amdgpu_target = os.environ.get("AMDGPU_TARGETS", "gfx950")
            if "gfx950" in amdgpu_target:
                self.target_cc = 95  # MI350 (gfx950): 160KB LDS
            else:
                self.target_cc = 94  # MI300 (gfx942): 64KB LDS
        else:
            self.target_cc = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
        # Check tensor shapes
        qo_indptr_buffer = self.meta_tensors["qo_indptr_buffer"]
        # Asserts "==" below is not guaranteed by vllm, because the shape is changed depending on real situation. But the mem space won't change.
        assert qo_indptr_buffer.shape[0] <= self.max_num_batched_requests+1, f"qo_indptr_buffer.shape: {qo_indptr_buffer.shape}, max_num_batched_requests: {self.max_num_batched_requests}"
        paged_kv_indptr_buffer = self.meta_tensors["paged_kv_indptr_buffer"]
        assert paged_kv_indptr_buffer.shape[0] <= self.max_num_batched_requests+1, f"paged_kv_indptr_buffer.shape: {paged_kv_indptr_buffer.shape}, max_num_batched_requests: {self.max_num_batched_requests}"
        paged_kv_indices_buffer = self.meta_tensors["paged_kv_indices_buffer"]
        # assert paged_kv_indices_buffer.shape == (self.max_num_pages,), f"paged_kv_indices_buffer.shape: {paged_kv_indices_buffer.shape}, max_num_pages: {self.max_num_pages}"
        # TODO: This is because the paged_kv_indices_buffer can be limited by max len on vllm side
        assert paged_kv_indices_buffer.shape[0] <= self.max_num_pages, f"paged_kv_indices_buffer.shape: {paged_kv_indices_buffer.shape}, max_num_pages: {self.max_num_pages}"
        paged_kv_last_page_len_buffer = self.meta_tensors["paged_kv_last_page_len_buffer"]
        assert paged_kv_last_page_len_buffer.shape[0] <= self.max_num_batched_requests, f"paged_kv_last_page_len_buffer.shape: {paged_kv_last_page_len_buffer.shape}, max_num_batched_requests: {self.max_num_batched_requests}"
        
        # check type of meta_tensors
        assert self.meta_tensors["tokens"].dtype == torch.int64, f"tokens.dtype: {self.meta_tensors['tokens'].dtype}"
        assert self.meta_tensors["input_tokens"].dtype == torch.int64, f"input_tokens.dtype: {self.meta_tensors['input_tokens'].dtype}"
        assert self.meta_tensors["output_tokens"].dtype == torch.int64, f"output_tokens.dtype: {self.meta_tensors['output_tokens'].dtype}"
        assert self.meta_tensors["num_new_tokens"].dtype == torch.int32, f"num_new_tokens.dtype: {self.meta_tensors['num_new_tokens'].dtype}"
        assert self.meta_tensors["prompt_lengths"].dtype == torch.int32, f"prompt_lengths.dtype: {self.meta_tensors['prompt_lengths'].dtype}"
        assert qo_indptr_buffer.dtype == torch.int32, f"qo_indptr_buffer.dtype: {qo_indptr_buffer.dtype}"
        assert paged_kv_indptr_buffer.dtype == torch.int32, f"paged_kv_indptr_buffer.dtype: {paged_kv_indptr_buffer.dtype}"
        assert paged_kv_indices_buffer.dtype == torch.int32, f"paged_kv_indices_buffer.dtype: {paged_kv_indices_buffer.dtype}"
        assert paged_kv_last_page_len_buffer.dtype == torch.int32, f"paged_kv_last_page_len_buffer.dtype: {paged_kv_last_page_len_buffer.dtype}"

    def _raise_unsupported_target_cc(self, layer: str, supported_tasks: list) -> None:
        """Raise NotImplementedError for ROCm/AMD or other unsupported target_cc."""
        hip = getattr(torch.version, "hip", None)
        if hip:
            raise NotImplementedError(
                f"MPK {layer} is not supported on ROCm/AMD GPUs. "
                "The persistent kernel layer compiles CUDA code (nvcc) and currently supports "
                "NVIDIA sm_80, sm_90, sm_100 only. Use a non-MPK mirage path or an NVIDIA GPU."
            )
        raise NotImplementedError(
            f"MPK {layer} does not support target_cc={self.target_cc}. "
            f"Supported: sm_80 (80), sm_90 (90), sm_100 (100). "
            f"Tasks for this layer: {supported_tasks}."
        )

    def attach_input(self, torch_tensor: torch.Tensor, name: str = None) -> DTensor:
        dims = tuple([d for d in torch_tensor.shape])
        strides = tuple([s for s in torch_tensor.stride()])
        # Assert a row-major layout
        for d in range(len(dims) - 1):
            assert strides[d] == strides[d + 1] * dims[d + 1]
        dtype = convert_torch_type_to_dtype(torch_tensor.dtype)
        t = self.kn_graph.new_input(dims=dims, strides=strides, dtype=dtype)
        # FIXME: currently assert that name is not None
        assert name is not None
        self.kn_graph.attach_torch_tensor(t, torch_tensor, name)
        return t

    def new_tensor(
        self,
        dims: tuple,
        strides: tuple = None,
        dtype: dtype = bfloat16,
        name: str = None,
        io_category: str = "cuda_tensor",
    ) -> DTensor:
        # Assert a row-major layout
        # if strides is not None:
        #     for d in range(len(dims) - 1):
        #         assert strides[d] == strides[d + 1] * dims[d + 1]
        t = self.kn_graph.new_input(dims=dims, strides=strides, dtype=dtype)
        # FIXME: currently assert that name is not None
        assert name is not None
        if io_category == "cuda_tensor":
            self.kn_graph.attach_cuda_tensor(t, name)
        elif io_category == "nvshmem_tensor":
            self.kn_graph.attach_nvshmem_tensor(t, name)
        else:
            raise RuntimeError(f"Invalid io_category: {io_category}")
        return t

    def fuse_tensors(
        self, inputs: list[DTensor], fused_dim: int, num_groups: int, name: str = None
    ) -> DTensor:
        # Currently only support fusing the 0-th dimension
        assert fused_dim == 0
        t = self.kn_graph.fuse_tensors(inputs, fused_dim, num_groups, name)
        return t

    def shuffle_tensors(
        self, inputs: list[DTensor], shuffled_dim: int, num_groups: int, name: str = None
    ) -> DTensor:
        # Currently only support shuffling the 0-th dimension
        assert shuffled_dim == 0
        t = self.kn_graph.shuffle_tensors(inputs, shuffled_dim, num_groups, name)
        return t

    def embed_layer(
        self,
        input: DTensor, # [batch_size, num_spec_tokens]
        weight: DTensor, # [vocab_size, hidden_size]
        output: DTensor, # [batch_size, hidden_size]
        grid_dim: tuple,
        block_dim: tuple,
        input_source: int = 0, # 0: all_tokens, 1: input_token
    ):
        # TODO: Support batch size > 1
        # tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # tb_graph.new_input(input, (-1, -1, -1), -1, True)
        # tb_graph.new_input(weight, (-1, -1, -1), -1, True)
        # tb_graph.new_input(output, (-1, -1, -1), -1, True)
        # self.kn_graph.customized([input, weight, output], tb_graph)
        # self.kn_graph.register_task(tb_graph, "embedding")
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, 1, -1), -1, True)
        tb_graph.new_input(weight, (1, -1, -1), -1, True)
        tb_graph.new_input(output, (1, 0, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "embedding" if self.target_cc == 90 else "embedding", [input_source])

    def rmsnorm_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        actual_hidden_dim: int = 0,
    ):
        # Currently assume that the input/output are 2D tensors
        assert input.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(output, (0, -1, -1), 1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        # 94 (MI300): use base "rmsnorm"; 90/100 use hopper
        # actual_hidden_dim: if > 0, divide by this instead of HIDDEN_DIM in RMS computation
        # (used when padding hidden dim to avoid bf16 rounding errors from scale factor)
        params = [actual_hidden_dim] if actual_hidden_dim > 0 else []
        self.kn_graph.register_task(
            tb_graph,
            "rmsnorm_hopper" if (self.target_cc == 90 or self.target_cc == 100) else "rmsnorm",
            params,
        )

    def rmsnorm_linear_layer(
        self,
        input: DTensor,
        weight_norm: DTensor,
        weight_linear: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that the input/weight_linear/output are 2D tensors
        assert input.num_dims == 2
        assert weight_linear.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight_norm, (-1, -1, -1), 0, True)
        tb_graph.new_input(weight_linear, (0, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight_norm, weight_linear, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "rmsnorm_linear")

    def attention_layer(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, fused_outdim / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert k_cache.num_dims == 4  # (batch_size, seq_len, kv_heads, head_dim)
        assert v_cache.num_dims == 4  # (batch_size, seq_len, kv_heads, head_dim)
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = output.dim(1) // head_dim
        rotary_embed = 0
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert sin_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert cos_pos_embed.dim(1) == head_dim
            assert sin_pos_embed.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1  # (head_dim)
            assert k_norm.num_dims == 1  # (head_dim)
            qk_norm = 1
            assert q_norm.dim(0) == head_dim
            assert k_norm.dim(0) == head_dim

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized(
            [
                input,
                k_cache,
                v_cache,
                q_norm,
                k_norm,
                cos_pos_embed,
                sin_pos_embed,
                output,
            ],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "attention", params)

    def single_batch_extend_attention_layer(
        self,
        input: DTensor, # [6, 6144]
        k_cache: DTensor, 
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        output: DTensor,
        grid_dim: tuple, # (6, 8, 1)
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, fused_outdim / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert k_cache.num_dims == 4  # (batch_size, seq_len, kv_heads, head_dim)
        assert v_cache.num_dims == 4  # (batch_size, seq_len, kv_heads, head_dim)
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = output.dim(1) // head_dim # 32
        rotary_embed = 0
        output_stride = output.dim(1)

        extend_num = input.dim(0) - 1
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert sin_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert cos_pos_embed.dim(1) == head_dim
            assert sin_pos_embed.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1  # (head_dim)
            assert k_norm.num_dims == 1  # (head_dim)
            qk_norm = 1
            assert q_norm.dim(0) == head_dim
            assert k_norm.dim(0) == head_dim

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        # params[4]: extend_num
        # params[5]: output_stride
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed, extend_num, output_stride]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized(
            [
                input,
                k_cache,
                v_cache,
                q_norm,
                k_norm,
                cos_pos_embed,
                sin_pos_embed,
                output,
            ],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "single_batch_extend_attention", params)

    def paged_attention_layer(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (num_tokens, fused_outdim / world_size)
        assert output.num_dims == 2  # (num_tokens, hidden_size / world_size)
        assert k_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert v_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert k_cache.dim(0) == self.max_num_pages
        assert v_cache.dim(0) == self.max_num_pages
        assert k_cache.dim(1) == self.page_size
        assert v_cache.dim(1) == self.page_size
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = output.dim(1) // head_dim
        rotary_embed = 0
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert sin_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert cos_pos_embed.dim(1) == head_dim
            assert sin_pos_embed.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1  # (head_dim)
            assert k_norm.num_dims == 1  # (head_dim)
            qk_norm = 1
            assert q_norm.dim(0) == head_dim
            assert k_norm.dim(0) == head_dim

        # If q_norm/k_norm are None, create dummy tensors (kernel still expects 8 inputs)
        if q_norm is None:
            import torch
            dummy = torch.ones(head_dim, dtype=torch.bfloat16, device="cuda")
            self._dummy_tensor_refs.append(dummy)
            q_norm = self.attach_input(torch_tensor=dummy, name=f"_dummy_q_norm_pa_{self._dummy_counter}")
            self._dummy_counter += 1
        if k_norm is None:
            import torch
            dummy = torch.ones(head_dim, dtype=torch.bfloat16, device="cuda")
            self._dummy_tensor_refs.append(dummy)
            k_norm = self.attach_input(torch_tensor=dummy, name=f"_dummy_k_norm_pa_{self._dummy_counter}")
            self._dummy_counter += 1

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        # params[4]: max_seq_len
        # params[5]: page_size
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed, self.max_seq_length, self.page_size]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        assert grid_dim[0] == self.max_num_batched_requests
        assert grid_dim[1] == num_kv_heads
        tb_graph.new_input(input, (-1, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 1, -1), -1, True)
        self.kn_graph.customized(
            [
                input,
                k_cache,
                v_cache,
                q_norm,
                k_norm,
                cos_pos_embed,
                sin_pos_embed,
                output,
            ],
            tb_graph,
        )
        if self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "paged_attention_hopper", params)
        elif self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "paged_attention_sm100", params)
        else:
            self.kn_graph.register_task(tb_graph, "paged_attention", params)

    
    def paged_attention_split_kv_layer(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        lse: DTensor,
        output: DTensor,
        attention_params: tuple,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (num_tokens, fused_outdim / world_size)
        assert k_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert v_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert k_cache.dim(0) == self.max_num_pages
        assert v_cache.dim(0) == self.max_num_pages
        assert k_cache.dim(1) == self.page_size
        assert v_cache.dim(1) == self.page_size
        assert output.num_dims == 3  # (num_tokens, num_kv_chunks * num_qo_per_kv * head_dim / world_size, num_kv_heads)
        assert lse.num_dims == 3  # (num_tokens, num_kv_chunks * num_qo_per_kv / world_size, num_kv_heads)

        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = attention_params[0]
        num_kv_chunks = attention_params[1]
        
        rotary_embed = 0
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert sin_pos_embed.num_dims == 2  # (seq_len, head_dim)
            assert cos_pos_embed.dim(1) == head_dim
            assert sin_pos_embed.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1  # (head_dim)
            assert k_norm.num_dims == 1  # (head_dim)
            qk_norm = 1
            assert q_norm.dim(0) == head_dim
            assert k_norm.dim(0) == head_dim

        # params[0]: num_q_heads
        # params[1]: num_kv_heads
        # params[2]: qk_norm
        # params[3]: rotary_embed
        # params[4]: max_seq_len
        # params[5]: page_size
        # params[6]: num_kv_chunks
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed, self.max_seq_length, self.page_size, num_kv_chunks]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        assert grid_dim[0] == self.max_num_batched_requests
        assert grid_dim[1] == num_kv_heads
        tb_graph.new_input(input, (-1, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(lse, (-1, 2, 1), -1, True)
        tb_graph.new_input(output, (-1, 2, 1), -1, True)
        self.kn_graph.customized(
            [
                input,
                k_cache,
                v_cache,
                q_norm,
                k_norm,
                cos_pos_embed,
                sin_pos_embed,
                lse,
                output,
            ],
            tb_graph,
        )
        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "paged_attention_split_kv_sm100", params)
        elif self.target_cc in (94, 95):
            self.kn_graph.register_task(tb_graph, "paged_attention_split_kv_mi300", params)
        elif self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "paged_attention_split_kv_hopper", params)
        else:
            raise ValueError(f"Unsupported target CC: {self.target_cc}")

    def paged_attention_split_kv_merge_layer(
        self,
        lse: DTensor,
        output_tmp: DTensor,
        output: DTensor,
        attention_params: tuple,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        assert lse.num_dims == 3  # (num_tokens, num_kv_chunks * num_qo_per_kv / world_size, num_kv_heads)
        assert output_tmp.num_dims == 3  # (num_tokens, num_chunks, hidden_size / world_size)
        assert output.num_dims == 2  # (num_tokens, hidden_size / world_size)

        num_q_heads = attention_params[0]
        head_dim = attention_params[1]
        num_qo_heads_per_kv = num_q_heads / grid_dim[1]
        num_kv_heads = grid_dim[1]
        # params[0]: num_qo_heads_per_kv
        # params[1]: head_dim
        # params[2]: max_seq_len
        # params[3]: page_size
        # params[4]: num_kv_heads
        params = [num_qo_heads_per_kv, head_dim, self.max_seq_length, self.page_size, num_kv_heads]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(lse, (-1, 2, -1), -1, True)
        tb_graph.new_input(output_tmp, (-1, 2, -1), -1, True)
        tb_graph.new_input(output, (-1, 1, -1), -1, True)
        self.kn_graph.customized(
            [
                lse,
                output_tmp,
                output,
            ],
            tb_graph,
        )
        if self.target_cc == 100 or self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "paged_attention_split_kv_merge_sm100", params)
        elif self.target_cc in (94, 95):
            self.kn_graph.register_task(tb_graph, "paged_attention_split_kv_merge_mi300", params)
        else:
            raise ValueError(f"Unsupported target CC: {self.target_cc}")

    def kv_cache_update_layer(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        q_workspace: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        assert input.num_dims == 2  # (num_tokens, fused_outdim / world_size)
        assert k_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert v_cache.num_dims == 4
        assert q_workspace.num_dims == 2  # (num_tokens, q_workspace_stride)
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = q_workspace.dim(1) // head_dim
        q_workspace_stride = q_workspace.dim(1)
        rotary_embed = 1 if cos_pos_embed is not None else 0
        qk_norm = 1 if q_norm is not None else 0

        # If q_norm/k_norm are None, create dummy tensors (kernel still expects 8 inputs)
        if q_norm is None:
            import torch
            dummy = torch.ones(head_dim, dtype=torch.bfloat16, device="cuda")
            self._dummy_tensor_refs.append(dummy)
            q_norm = self.attach_input(torch_tensor=dummy, name=f"_dummy_q_norm_{self._dummy_counter}")
            self._dummy_counter += 1
        if k_norm is None:
            import torch
            dummy = torch.ones(head_dim, dtype=torch.bfloat16, device="cuda")
            self._dummy_tensor_refs.append(dummy)
            k_norm = self.attach_input(torch_tensor=dummy, name=f"_dummy_k_norm_{self._dummy_counter}")
            self._dummy_counter += 1

        # params: num_q_heads, num_kv_heads, qk_norm, rotary_embed, max_seq_len, page_size, q_workspace_stride
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed,
                  self.max_seq_length, self.page_size, q_workspace_stride]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_workspace, (-1, 1, -1), -1, True)
        self.kn_graph.customized(
            [input, k_cache, v_cache, q_norm, k_norm,
             cos_pos_embed, sin_pos_embed, q_workspace],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "kv_cache_update_mi300", params)

    def paged_attention_ck_fmha_layer(
        self,
        q_workspace: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        o_acc: DTensor,
        lse_acc: DTensor,
        attention_params: tuple,
        grid_dim: tuple,
        block_dim: tuple,
        sinks: DTensor = None,
        sliding_window: int = 0,
    ):
        assert q_workspace.num_dims == 2  # (num_tokens, q_workspace_stride)
        assert k_cache.num_dims == 4  # (num_pages, page_size, kv_heads, head_dim)
        assert v_cache.num_dims == 4
        assert o_acc.num_dims == 2   # (num_tokens, kv_heads*chunks*qo_per_kv*head_dim)
        assert lse_acc.num_dims == 2 # (num_tokens, kv_heads*chunks*qo_per_kv)

        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = attention_params[0]
        num_kv_chunks = attention_params[1]
        max_num_requests = attention_params[2]
        q_workspace_stride = q_workspace.dim(1)
        kv_cache_stride = num_kv_heads * head_dim

        # params: num_q_heads, num_kv_heads, head_dim, page_size, max_seq_len,
        #         num_kv_chunks, q_workspace_stride, kv_cache_stride,
        #         max_num_requests, sliding_window
        params = [num_q_heads, num_kv_heads, head_dim, self.page_size,
                  self.max_seq_length, num_kv_chunks, q_workspace_stride,
                  kv_cache_stride, max_num_requests, sliding_window]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        assert grid_dim[0] == max_num_requests
        assert grid_dim[1] == num_kv_heads
        assert grid_dim[2] == num_kv_chunks
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (-1, 2, -1), 1, True)
        # Optional sinks input (GPT-OSS per-head attention sinks).
        # When provided, sink correction is fused into the attention epilogue,
        # eliminating the standalone attention_sink_layer task.
        if sinks is not None:
            assert sinks.num_dims == 1  # (num_q_heads,)
            tb_graph.new_input(sinks, (-1, -1, -1), -1, True)
        # o_acc/lse_acc: 2D flat, no partitioning — kernel offsets by kv_head internally
        tb_graph.new_input(o_acc, (-1, -1, -1), -1, True)
        tb_graph.new_input(lse_acc, (-1, -1, -1), -1, True)
        if sinks is not None:
            self.kn_graph.customized(
                [q_workspace, k_cache, v_cache, sinks, o_acc, lse_acc],
                tb_graph,
            )
        else:
            self.kn_graph.customized(
                [q_workspace, k_cache, v_cache, o_acc, lse_acc],
                tb_graph,
            )
        self.kn_graph.register_task(tb_graph, "paged_attention_ck_fmha_split_kv_mi300", params)

    def paged_attention_ck_fmha_merge_layer(
        self,
        lse: DTensor,
        output_tmp: DTensor,
        output: DTensor,
        attention_params: tuple,
        grid_dim: tuple,
        block_dim: tuple,
        sinks: DTensor = None,
    ):
        # lse: 2D (num_tokens, num_kv_heads * chunks * qo_per_kv)
        # output_tmp: 2D (num_tokens, num_kv_heads * chunks * qo_per_kv * head_dim)
        # output: 2D (num_tokens, num_q_heads * head_dim)
        # sinks (optional): 1D (num_q_heads,) — applies sigmoid(LSE - sink) per
        # q-head to the merged output (GPT-OSS attention sinks).
        assert lse.num_dims == 2
        assert output_tmp.num_dims == 2
        assert output.num_dims == 2
        if sinks is not None:
            assert sinks.num_dims == 1

        num_q_heads = attention_params[0]
        head_dim = attention_params[1]
        num_kv_chunks = attention_params[2]
        num_kv_heads = attention_params[3]
        num_qo_heads_per_kv = num_q_heads // num_kv_heads
        # params: num_qo_heads_per_kv, head_dim, max_seq_len, page_size, num_kv_heads, num_kv_chunks
        params = [num_qo_heads_per_kv, head_dim, self.max_seq_length,
                  self.page_size, num_kv_heads, num_kv_chunks]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # No partitioning — merge kernel handles all offsets internally
        tb_graph.new_input(lse, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_tmp, (-1, -1, -1), -1, True)
        if sinks is not None:
            tb_graph.new_input(sinks, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        if sinks is not None:
            self.kn_graph.customized(
                [lse, output_tmp, sinks, output],
                tb_graph,
            )
        else:
            self.kn_graph.customized(
                [lse, output_tmp, output],
                tb_graph,
            )
        self.kn_graph.register_task(tb_graph, "paged_attention_ck_fmha_merge_mi300", params)

    def attention_sink_layer(
        self,
        attn_out: DTensor,
        lse_acc: DTensor,
        sinks: DTensor,
        num_q_heads: int,
        head_dim: int,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Post-attention sink correction.
        Multiplies attention output by sigmoid(LSE - sink) per head.
        attn_out is modified in-place (same tensor for input and output).
        """
        assert attn_out.num_dims == 2  # (max_tokens, num_q_heads * head_dim)
        assert lse_acc.num_dims == 2   # (max_tokens, num_q_heads)
        assert sinks.num_dims == 1     # (num_q_heads,)

        params = [num_q_heads, head_dim]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(attn_out, (-1, 1, -1), -1, True)  # in-place: both input and output
        tb_graph.new_input(lse_acc, (-1, -1, -1), -1, True)
        tb_graph.new_input(sinks, (-1, -1, -1), -1, True)
        tb_graph.new_input(attn_out, (-1, 1, -1), -1, True)  # output = same tensor
        self.kn_graph.customized(
            [attn_out, lse_acc, sinks, attn_out],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "attention_sink_mi300", params)

    def gang_paged_attention_split_kv_layer(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_norm: DTensor,
        k_norm: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        lse: DTensor,
        output: DTensor,
        q_workspace: DTensor,
        attention_params: tuple,
        block_dim: tuple,
    ):
        """Gang CK FMHA attention: 8 tasks (1 per XCD), broadcast to workers.
        Fuses KV cache update + CK FMHA attention into one gang task.
        Each worker decodes tile_idx → (request_id, kv_head).
        """
        assert input.num_dims == 2
        assert k_cache.num_dims == 4
        assert v_cache.num_dims == 4
        assert self.target_cc in (94, 95), "Gang attention only supported on MI300X"

        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = attention_params[0]
        num_kv_chunks = attention_params[1]
        q_workspace_stride = attention_params[2]

        rotary_embed = 0
        if cos_pos_embed is not None or sin_pos_embed is not None:
            assert cos_pos_embed.num_dims == 2
            assert sin_pos_embed.num_dims == 2
            rotary_embed = 1
        qk_norm = 0
        if q_norm is not None or k_norm is not None:
            assert q_norm.num_dims == 1
            assert k_norm.num_dims == 1
            qk_norm = 1

        # Total work items = max_requests * num_kv_heads
        total_work_items = self.max_num_batched_requests * num_kv_heads
        import math
        total_work_items_per_xcd = math.ceil(total_work_items / 8)

        # params: [num_q_heads, num_kv_heads, qk_norm, rotary_embed,
        #          max_seq_len, page_size, num_kv_chunks,
        #          total_work_items_per_xcd, total_work_items,
        #          q_workspace_stride]
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed,
                  self.max_seq_length, self.page_size, num_kv_chunks,
                  total_work_items_per_xcd, total_work_items,
                  q_workspace_stride]

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 7 inputs: qkv, k_cache, v_cache, q_norm, k_norm, cos, sin
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(k_cache, (-1, -1, -1), 1, True)
        tb_graph.new_input(v_cache, (-1, -1, -1), 1, True)
        tb_graph.new_input(q_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        # 3 outputs: lse, output, q_workspace
        tb_graph.new_input(lse, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input, k_cache, v_cache, q_norm, k_norm,
             cos_pos_embed, sin_pos_embed, lse, output, q_workspace],
            tb_graph,
        )
        self.kn_graph.register_task(tb_graph, "gang_attn_split_kv_mi300", params)

    def gang_paged_attention_split_kv_merge_layer(
        self,
        lse: DTensor,
        output_tmp: DTensor,
        output: DTensor,
        attention_params: tuple,
        block_dim: tuple,
    ):
        """Gang merge split-KV: 8 tasks (1 per XCD), broadcast to workers."""
        assert lse.num_dims == 3
        assert output_tmp.num_dims == 3
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Gang attention only supported on MI300X"

        num_q_heads = attention_params[0]
        head_dim = attention_params[1]
        num_kv_heads = attention_params[2]
        num_qo_heads_per_kv = num_q_heads // num_kv_heads

        total_work_items = self.max_num_batched_requests * num_kv_heads
        import math
        total_work_items_per_xcd = math.ceil(total_work_items / 8)

        # params: [num_qo_heads_per_kv, head_dim, max_seq_len, page_size,
        #          num_kv_heads, total_work_items_per_xcd, total_work_items]
        params = [num_qo_heads_per_kv, head_dim, self.max_seq_length,
                  self.page_size, num_kv_heads, total_work_items_per_xcd,
                  total_work_items]

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # No bid.y partitioning — full tensors
        tb_graph.new_input(lse, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_tmp, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([lse, output_tmp, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "gang_attn_merge_mi300", params)

    # MoE Layers
    def tensor_init_layer(
        self,
        input: DTensor,
        dummy_input: DTensor,
        dummy_output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that output
        assert input.num_dims == 2  # (batch_size, output_size)
        assert dummy_input.num_dims == 2 # (batch_size, hidden_size)
        assert dummy_output.num_dims == 2 # (batch_size, output_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, -1, -1), -1, True)
        tb_graph.new_input(dummy_input, (0, -1, -1), -1, True)
        tb_graph.new_input(dummy_output, (0, -1, -1), -1, True)
        self.kn_graph.customized([input, dummy_input, dummy_output], tb_graph)

        self.kn_graph.register_task(tb_graph, "tensor_init")
    
    def moe_topk_softmax_routing_layer(
        self,
        input: DTensor,
        output: tuple[DTensor, DTensor, DTensor],
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, num_experts)
        assert len(output) == 3
        moe_topk_weight, moe_routing_indices, moe_masks = output
        assert moe_topk_weight.num_dims == 2  # (batch_size, num_experts_per_tok)
        assert moe_routing_indices.num_dims == 2  # (num_experts, batch_size)
        assert moe_masks.num_dims == 1  # (num_experts + 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, -1, -1), -1, True)
        tb_graph.new_input(moe_topk_weight, (0, -1, -1), -1, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_masks, (-1, -1, -1), -1, True)
        self.kn_graph.customized([input, moe_topk_weight, moe_routing_indices, moe_masks], tb_graph)

        if self.target_cc in (94, 95):
            self.kn_graph.register_task(tb_graph, "moe_topk_softmax_mi300")
        else:
            self.kn_graph.register_task(tb_graph, "moe_topk_softmax_sm100")
        
    def moe_w13_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 3  # (num_experts, 2*intermediate_size, hidden_size)
        assert moe_routing_indices.num_dims == 2  # (num_experts_per_tok, batch_size)
        assert moe_mask.num_dims == 1  # (num_experts + 1)
        assert bias.num_dims == 2  # (num_experts, output_stride)
        assert output.num_dims == 3  # (batch_size, num_expert_per_tok, 2*intermediate_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph)

        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "moe_w13_linear_sm100")
        elif self.target_cc in (94, 95):
            self.kn_graph.register_task(tb_graph, "moe_w13_linear_mi300")
        elif self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "moe_w13_linear_sm90")
        else:
            assert False
            
    def moe_silu_mul_layer(
        self,
        input: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 3 # (batch_size, num_expert_per_tok, 2 * intermediate_size)
        assert output.num_dims == 3 # (batch_size, num_expert_per_tok, intermediate_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "moe_silu_mul")

    def moe_swigluoai_layer(
        self,
        input: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # SwigluOAI activation for GPT-OSS
        # Input: (batch_size, num_expert_per_tok, 2 * intermediate_size) - interleaved gate/up
        # Output: (batch_size, num_expert_per_tok, intermediate_size)
        assert input.num_dims == 3
        assert output.num_dims == 3
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "moe_swigluoai")

    def moe_w2_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 3  # (batch_size, num_expert_per_tok, intermediate_size)
        assert weight.num_dims == 3  # (num_experts, hidden_size, intermediate_size)
        assert moe_routing_indices.num_dims == 2  # (num_experts_per_tok, batch_size)
        assert moe_mask.num_dims == 1  # (num_experts + 1)
        assert bias.num_dims == 2  # (num_experts, output_stride)
        assert output.num_dims == 3  # (batch_size, num_expert_per_tok, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 2, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph)

        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "moe_w2_linear_sm100")
        elif self.target_cc in (94, 95):
            self.kn_graph.register_task(tb_graph, "moe_w2_linear_mi300")
        elif self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "moe_w2_linear_sm90")
        else:
            assert False
        
    def moe_w13_linear_mxfp4_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 16,
        grid_dim: tuple = (1, 1, 1),
        block_dim: tuple = (256, 1, 1),
    ):
        """MoE W13 linear with MXFP4 weights + bias (native FP4 format, no dequant).
        Weight shape: [num_experts, expert_wgs, wg_bytes] as uint8.
        Bias shape: [num_experts, 2*intermediate_size] as bfloat16.
        """
        assert input.num_dims == 2   # [batch, hidden_size]
        assert weight.num_dims == 3  # [num_experts, expert_wgs, wg_bytes] uint8
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 3    # [num_experts, expert_wgs, output_per_wg]
        assert output.num_dims == 3  # [batch, topk, 2*intermediate]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, 1, -1), 2, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "moe_w13_linear_mxfp4_mi300", [output_per_wg])

    def moe_w2_linear_mxfp4_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 16,
        grid_dim: tuple = (1, 1, 1),
        block_dim: tuple = (256, 1, 1),
    ):
        """MoE W2 linear with MXFP4 weights + bias (native FP4 format, no dequant).
        Weight shape: [num_experts, expert_wgs, wg_bytes] as uint8.
        Bias shape: [num_experts, hidden_size] as bfloat16.
        """
        assert input.num_dims == 3   # [batch, topk, intermediate]
        assert weight.num_dims == 3  # [num_experts, expert_wgs, wg_bytes] uint8
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 3    # [num_experts, expert_wgs, output_per_wg]
        assert output.num_dims == 3  # [batch, topk, hidden]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 2, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, 1, -1), 2, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "moe_w2_linear_mxfp4_mi300", [output_per_wg])

    def moe_w13_linear_mxfp4_ck_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 16,
        grid_dim: tuple = (1, 1, 1),
        block_dim: tuple = (256, 1, 1),
    ):
        """MoE W13 linear with MXFP4 weights + bias (MFMA-based, replaces scalar GEMV).
        Same weight format as moe_w13_linear_mxfp4_layer.
        """
        assert input.num_dims == 2
        assert weight.num_dims == 3
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 3
        assert output.num_dims == 3
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, 1, -1), 2, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "moe_w13_linear_mxfp4_ck_mi300", [output_per_wg])

    def moe_w2_linear_mxfp4_ck_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 16,
        grid_dim: tuple = (1, 1, 1),
        block_dim: tuple = (256, 1, 1),
    ):
        """MoE W2 linear with MXFP4 weights + bias (MFMA-based, replaces scalar GEMV).
        Same weight format as moe_w2_linear_mxfp4_layer.
        """
        assert input.num_dims == 3
        assert weight.num_dims == 3
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 3
        assert output.num_dims == 3
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 2, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, 1, -1), 2, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized([input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "moe_w2_linear_mxfp4_ck_mi300", [output_per_wg])

    def gang_moe_w13_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang MoE W13 linear: 8 tasks (1/XCD), workers cooperate per expert.
        All workers on an XCD process the same expert's GEMM tiles concurrently,
        eliminating L2 thrashing from concurrent expert weight loads.
        """
        assert input.num_dims == 2   # [batch, hidden_size]
        assert weight.num_dims == 3  # [num_experts, 2*intermediate, hidden_size]
        assert moe_routing_indices.num_dims == 2  # [num_experts, batch_size]
        assert moe_mask.num_dims == 1  # [num_experts + 1]
        assert bias.num_dims == 2  # [num_experts, output_stride]
        assert output.num_dims == 3  # [batch, topk, 2*intermediate]
        assert self.target_cc in (94, 95), "Gang MoE linear only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        num_experts = weight.dim(0)
        output_size = weight.dim(1)
        reduction_size = weight.dim(2)
        tile_n = 64
        assert output_size % tile_n == 0, f"output_size {output_size} not divisible by {tile_n}"
        assert reduction_size % 256 == 0, f"W13 K={reduction_size} not divisible by 256"
        n_tiles = output_size // tile_n
        m_tiles = max(1, batch_size // 16)
        tiles_per_expert = m_tiles * n_tiles
        total_tiles_all = num_experts * tiles_per_expert
        total_tiles_per_xcd = (total_tiles_all + 7) // 8
        assert total_tiles_per_xcd <= 65535, \
            f"total_tiles_per_xcd={total_tiles_per_xcd} exceeds uint16_t (bs={batch_size})"

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized(
            [input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "gang_moe_w13_linear_mi300",
            [tiles_per_expert, 0, total_tiles_per_xcd],
        )

    def gang_moe_w2_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang MoE W2 linear: 8 tasks (1/XCD), workers cooperate per expert.
        W2 processes tokens one at a time (per-token topk_slot offsets differ).
        tiles_per_expert = n_tiles * batch_size.
        """
        assert input.num_dims == 3   # [batch, topk, intermediate]
        assert weight.num_dims == 3  # [num_experts, hidden_size, intermediate]
        assert moe_routing_indices.num_dims == 2  # [num_experts, batch_size]
        assert moe_mask.num_dims == 1  # [num_experts + 1]
        assert bias.num_dims == 2  # [num_experts, output_stride]
        assert output.num_dims == 3  # [batch, topk, hidden_size]
        assert self.target_cc in (94, 95), "Gang MoE linear only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        num_experts = weight.dim(0)
        output_size = weight.dim(1)
        reduction_size = weight.dim(2)
        tile_n = 64
        assert output_size % tile_n == 0, f"output_size {output_size} not divisible by {tile_n}"
        assert reduction_size % 128 == 0, f"W2 K={reduction_size} not divisible by 128"
        n_tiles = output_size // tile_n
        # W2: one token at a time, so tiles_per_expert = n_tiles * batch_size
        tiles_per_expert = n_tiles * batch_size
        total_tiles_all = num_experts * tiles_per_expert
        total_tiles_per_xcd = (total_tiles_all + 7) // 8
        assert total_tiles_per_xcd <= 65535, \
            f"total_tiles_per_xcd={total_tiles_per_xcd} exceeds uint16_t (bs={batch_size})"

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 2, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized(
            [input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "gang_moe_w2_linear_mi300",
            [tiles_per_expert, 0, total_tiles_per_xcd],
        )

    def gang_moe_w13_linear_mxfp4_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang MoE W13 MXFP4 linear: 8 tasks (1/XCD), MFMA-based MXFP4 dequant.
        Weight format: [E, expert_wgs, wg_bytes] (MXFP4 packed per workgroup).
        Bias format: [E, output_stride] (2D flat).
        """
        assert input.num_dims == 2   # [batch, hidden_size]
        assert weight.num_dims == 3  # [E, expert_wgs, wg_bytes]
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 2    # [E, output_stride]
        assert output.num_dims == 3  # [batch, topk, output_size]
        assert self.target_cc in (94, 95), "Gang MoE MXFP4 only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        num_experts = weight.dim(0)
        expert_wgs = weight.dim(1)
        output_size = expert_wgs * output_per_wg

        # tiles_per_expert = batch_size * expert_wgs (one tile per token per wg)
        tiles_per_expert = batch_size * expert_wgs
        # Spread all expert tiles across 8 XCDs (flat round-robin).
        # Old: each expert assigned to 1 XCD → only top-k XCDs active.
        # New: all tiles pooled and distributed → all 8 XCDs active.
        num_topk = output.dim(1)  # topk dimension from output shape
        max_activated = min(num_topk * batch_size, num_experts)
        total_tiles_all = max_activated * tiles_per_expert
        total_tiles_per_xcd = (total_tiles_all + 7) // 8
        assert total_tiles_per_xcd <= 65535, \
            f"total_tiles_per_xcd={total_tiles_per_xcd} exceeds uint16_t"

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized(
            [input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "gang_moe_w13_linear_mxfp4_mi300",
            [tiles_per_expert, 0, total_tiles_per_xcd, output_per_wg],
        )

    def gang_moe_w13_swiglu_mxfp4_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang MoE W13 MXFP4 with SwiGLU fused into epilogue.
        Same MFMA as W13 but applies SwiGLU(gate+bias, up+bias) in the
        epilogue and writes half-sized output.
        Input: [batch, hidden] (2D, same as W13).
        Weight: [E, expert_wgs, wg_bytes] (interleaved gate/up, MXFP4).
        Bias: [E, 2*intermediate] (interleaved gate/up bias).
        Output: [batch, topk, intermediate] (activated, half of W13 output).
        """
        assert input.num_dims == 2   # [batch, hidden_size]
        assert weight.num_dims == 3  # [E, expert_wgs, wg_bytes]
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 2    # [E, 2*intermediate]
        assert output.num_dims == 3  # [batch, topk, intermediate]
        assert self.target_cc in (94, 95), "Gang MoE MXFP4 only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        num_experts = weight.dim(0)
        expert_wgs = weight.dim(1)

        tiles_per_expert = batch_size * expert_wgs
        num_topk = output.dim(1)
        max_activated = min(num_topk * batch_size, num_experts)
        total_tiles_all = max_activated * tiles_per_expert
        total_tiles_per_xcd = (total_tiles_all + 7) // 8
        assert total_tiles_per_xcd <= 65535, \
            f"total_tiles_per_xcd={total_tiles_per_xcd} exceeds uint16_t"

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized(
            [input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "gang_moe_w13_swiglu_mxfp4_mi300",
            [tiles_per_expert, 0, total_tiles_per_xcd, output_per_wg],
        )

    def gang_moe_w2_linear_mxfp4_layer(
        self,
        input: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang MoE W2 MXFP4 linear: 8 tasks (1/XCD), MFMA-based MXFP4 dequant.
        Weight format: [E, expert_wgs, wg_bytes] (MXFP4 packed per workgroup).
        Bias format: [E, output_stride] (2D flat).
        """
        assert input.num_dims == 3   # [batch, topk, intermediate]
        assert weight.num_dims == 3  # [E, expert_wgs, wg_bytes]
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 2    # [E, output_stride]
        assert output.num_dims == 3  # [batch, topk, hidden_size]
        assert self.target_cc in (94, 95), "Gang MoE MXFP4 only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        num_experts = weight.dim(0)
        expert_wgs = weight.dim(1)

        tiles_per_expert = batch_size * expert_wgs
        num_topk = input.dim(1)  # topk dimension from input shape
        max_activated = min(num_topk * batch_size, num_experts)
        total_tiles_all = max_activated * tiles_per_expert
        total_tiles_per_xcd = (total_tiles_all + 7) // 8
        assert total_tiles_per_xcd <= 65535, \
            f"total_tiles_per_xcd={total_tiles_per_xcd} exceeds uint16_t"

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 2, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized(
            [input, weight, moe_routing_indices, moe_mask, bias, output], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "gang_moe_w2_linear_mxfp4_mi300",
            [tiles_per_expert, 0, total_tiles_per_xcd, output_per_wg],
        )

    def gang_moe_fused_mxfp4_layer(
        self,
        input: DTensor,
        gate_up_weight: DTensor,
        down_weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        w13_bias: DTensor,
        w2_bias: DTensor,
        routing_weight: DTensor,
        swiglu_out: DTensor,
        workspace_f32: DTensor,
        barrier: DTensor,
        w13_output_per_wg: int = 128,
        w2_output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused W13+SwiGLU+W2 MoE gang kernel with per-expert pipelining.
        Single gang task replaces separate W13 and W2 tasks. Phase-ordered
        tile encoding (all W13 before all W2) with in-kernel atomicAdd
        barrier per expert. Supports different OPW for W13 and W2.

        W2 epilogue does atomicAdd(workspace_f32, (result+bias)*routing_weight)
        instead of writing bf16 to mlp_out. This eliminates the standalone
        MulSumAdd task (~10.2us/layer savings).

        8 inputs: input, gate_up_weight, down_weight, routing, mask, w13_bias, w2_bias, routing_weight
        3 outputs: swiglu_out (intermediate), workspace_f32 (f32 accumulator), barrier
        """
        assert input.num_dims == 2           # [batch, hidden_size]
        assert gate_up_weight.num_dims == 3  # [E, W13_WGS, wg_bytes]
        assert down_weight.num_dims == 3     # [E, W2_WGS, wg_bytes]
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert w13_bias.num_dims == 2        # [E, 2*intermediate]
        assert w2_bias.num_dims == 2         # [E, hidden]
        assert routing_weight.num_dims == 2  # [batch, topk] f32
        assert swiglu_out.num_dims == 3      # [batch, topk, intermediate]
        assert workspace_f32.num_dims == 2   # [batch, hidden] f32
        assert barrier.num_dims == 1         # [2*E]
        assert self.target_cc in (94, 95), "Fused MoE MXFP4 only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        num_experts = gate_up_weight.dim(0)
        w13_wgs = gate_up_weight.dim(1)  # 2*intermediate/W13_OPW
        w2_wgs = down_weight.dim(1)      # hidden/W2_OPW

        # Combined tile count: W13 tiles + W2 tiles (phase-ordered in kernel)
        w13_tiles = batch_size * w13_wgs
        w2_tiles = batch_size * w2_wgs
        tiles_per_expert = w13_tiles + w2_tiles

        num_topk = swiglu_out.dim(1)
        max_activated = min(num_topk * batch_size, num_experts)
        # Pad W13 tile space to next multiple of 240 (30 workers/XCD × 8 XCDs)
        # so every worker's first tile is W13, eliminating compute imbalance.
        # Must match PAD_MULTIPLE in gang_moe_fused_mxfp4_mi300.cuh.
        PAD_MULTIPLE = 240
        total_w13_real = max_activated * w13_tiles
        total_w13_padded = ((total_w13_real + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        total_w2 = max_activated * w2_tiles
        total_tiles_all = total_w13_padded + total_w2
        total_tiles_per_xcd = (total_tiles_all + 7) // 8
        assert total_tiles_per_xcd <= 65535, \
            f"total_tiles_per_xcd={total_tiles_per_xcd} exceeds uint16_t"

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 8 inputs
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(gate_up_weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(down_weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(w13_bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(w2_bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(routing_weight, (-1, -1, -1), -1, True)
        # 3 outputs
        tb_graph.new_input(swiglu_out, (-1, 2, -1), -1, True)
        tb_graph.new_input(workspace_f32, (-1, -1, -1), -1, True)
        tb_graph.new_input(barrier, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input, gate_up_weight, down_weight,
             moe_routing_indices, moe_mask, w13_bias, w2_bias,
             routing_weight,
             swiglu_out, workspace_f32, barrier], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "gang_moe_fused_mxfp4_mi300",
            [tiles_per_expert, w13_output_per_wg, total_tiles_per_xcd, w2_output_per_wg],
        )

    def gang_moe_swiglu_w2_mxfp4_layer(
        self,
        w13_output: DTensor,
        weight: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang fused SwiGLU+W2 MXFP4: reads interleaved gate/up from W13,
        applies SwiGLU during FP8 quantization, feeds into W2 MFMA.
        No cross-WG barrier needed (croc-style activation fusion).
        Input: [batch, topk, 2*intermediate] (interleaved gate/up from W13).
        Weight: [E, expert_wgs, wg_bytes] (W2 down weights, MXFP4 packed).
        Bias: [E, output_stride] (W2 bias).
        Output: [batch, topk, hidden_size] BF16.
        """
        assert w13_output.num_dims == 3   # [batch, topk, 2*intermediate]
        assert weight.num_dims == 3       # [E, expert_wgs, wg_bytes]
        assert moe_routing_indices.num_dims == 2
        assert moe_mask.num_dims == 1
        assert bias.num_dims == 2         # [E, output_stride]
        assert output.num_dims == 3       # [batch, topk, hidden_size]
        assert self.target_cc in (94, 95), "Gang MoE MXFP4 only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        num_experts = weight.dim(0)
        expert_wgs = weight.dim(1)

        tiles_per_expert = batch_size * expert_wgs
        num_topk = w13_output.dim(1)
        max_activated = min(num_topk * batch_size, num_experts)
        total_tiles_all = max_activated * tiles_per_expert
        total_tiles_per_xcd = (total_tiles_all + 7) // 8
        assert total_tiles_per_xcd <= 65535, \
            f"total_tiles_per_xcd={total_tiles_per_xcd} exceeds uint16_t"

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(w13_output, (-1, -1, -1), 2, True)
        tb_graph.new_input(weight, (-1, 1, -1), 2, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        tb_graph.new_input(bias, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, 2, -1), -1, True)
        self.kn_graph.customized(
            [w13_output, weight, moe_routing_indices, moe_mask, bias, output], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "gang_moe_swiglu_w2_mxfp4_mi300",
            [tiles_per_expert, 0, total_tiles_per_xcd, output_per_wg],
        )

    def moe_mul_sum_add_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 3  # (batch_size, num_experts_per_tok, hidden_size)
        assert weight.num_dims == 2  # (batch_size, num_experts_per_tok)
        assert residual.num_dims == 2  # (batch_size, hidden_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 2, -1), -1, True)
        tb_graph.new_input(weight, (0, -1, -1), -1, True)
        tb_graph.new_input(residual, (0, 1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized([input, weight, residual, output], tb_graph)

        if self.target_cc in (94, 95):
            self.kn_graph.register_task(tb_graph, "moe_mul_sum_add_mi300")
        else:
            self.kn_graph.register_task(tb_graph, "moe_mul_sum_add_sm100")

    def splitk_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, 1, -1), 1, True)
        tb_graph.new_input(weight, (0, 1, -1), 1, True)
        if self.target_cc in (94, 95):
            # MI300X: workspace output partitioned by grid_dim.x (N) and grid_dim.y (K-splits)
            tb_graph.new_input(output, (1, 0, -1), -1, True)
        else:
            tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)

        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "splitk_linear_sm100")
        elif self.target_cc in (94, 95):
            self.kn_graph.register_task(tb_graph, "splitk_linear_mi300")
        elif self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "splitk_linear_swapAB_hopper")
        else:
            assert False

    def gang_ksplit_linear_with_residual_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        workspace: DTensor,
        output: DTensor,
        tile_n: int,
        output_stride: int,
        k_splits: int = 8,
        block_dim: tuple = (256, 1, 1),
    ):
        """Cross-XCD K-split linear with residual (SKXCCM-style).
        Phase 1: 8 gang tasks, each XCD handles K/8 for ALL N-tiles.
        Phase 2: 8 gang tasks, each XCD finalizes its N-partition."""
        assert self.target_cc in (94, 95)
        batch_size = self.max_num_batched_tokens
        output_size = weight.dim(0)
        reduction_size = weight.dim(1) if weight.num_dims == 2 else input.dim(1)
        assert reduction_size % k_splits == 0
        n_tiles = output_size // tile_n
        n_cols_per_xcd = output_size // 8

        # Phase 1: K-split GEMM — each XCD reads full input+weight, writes to workspace
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)     # full input
        tb_graph.new_input(weight, (-1, -1, -1), 1, True)    # full weight
        tb_graph.new_input(workspace, (-1, -1, -1), -1, True) # full workspace (atomic target)
        self.kn_graph.customized([input, weight, workspace], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "gang_ksplit_gemm_mi300",
            [output_stride, tile_n, n_tiles, k_splits]
        )

        # Phase 2: Finalize — each XCD reads its workspace partition + residual
        finalize_tiles = max(1, (batch_size * n_cols_per_xcd + 511) // 512)
        tb_graph2 = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph2.new_input(workspace, (-1, -1, -1), 1, True)  # full workspace
        tb_graph2.new_input(residual, (1, -1, -1), 1, True)    # XCD's residual partition
        tb_graph2.new_input(output, (1, -1, -1), -1, True)     # XCD's output partition
        self.kn_graph.customized([workspace, residual, output], tb_graph2)
        self.kn_graph.register_task(
            tb_graph2, "gang_ksplit_finalize_mi300",
            [output_stride, n_cols_per_xcd, finalize_tiles]
        )

    def gang_splitk_linear_with_residual_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        workspace: DTensor,  # [batch, hidden_size] float32
        output: DTensor,
        tile_n: int,
        output_stride: int,
        k_splits: int = 4,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang split-K linear with residual: splits K within XCD for better utilization.
        8 tasks (1 per XCD), each with n_tiles × k_splits total tiles.
        Uses XCD-local atomics for merge (cheaper than GPU-scope)."""
        assert self.target_cc in (94, 95)
        batch_size = self.max_num_batched_tokens
        output_size = weight.dim(0)
        assert output_size % 8 == 0
        chunk_n = output_size // 8
        assert chunk_n % tile_n == 0
        n_tiles_per_xcd = chunk_n // tile_n
        reduction_size = weight.dim(1) if weight.num_dims == 2 else input.dim(1)
        assert reduction_size % k_splits == 0, f"K={reduction_size} not divisible by k_splits={k_splits}"
        total_tiles = n_tiles_per_xcd * k_splits
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)        # input_ptrs[0]
        tb_graph.new_input(weight, (0, -1, -1), 1, True)        # input_ptrs[1]
        tb_graph.new_input(residual, (1, -1, -1), 1, True)      # input_ptrs[2]
        tb_graph.new_input(workspace, (1, -1, -1), 1, True)     # input_ptrs[3] float32 workspace
        tb_graph.new_input(output, (1, -1, -1), -1, True)       # output_ptrs[0]
        self.kn_graph.customized([input, weight, residual, workspace, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "gang_splitk_linear_res_mi300",
            [output_stride, tile_n, n_tiles_per_xcd, k_splits]
        )

    def gang_rmsnorm_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        block_dim: tuple = (128, 1, 1),
    ):
        """Gang RMSNorm: 8 tasks (1 per XCD), each computes same RMSNorm.
        Enables XCD-local event counting to avoid cross-XCD barrier."""
        assert self.target_cc in (94, 95), "Gang RMSNorm only supported on MI300X"
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "gang_rmsnorm_mi300", [])

    def gang_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        tile_n: int,
        output_stride: int,
        m_tiles: int = 1,
        wgm: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang linear with HipKittens Algorithm 1 windowed traversal.
        8 tasks (1 per XCD), each broadcast to m_tiles * n_tiles_per_xcd workers.

        Args:
            m_tiles: number of M-tiles to split batch across (1=no M-split)
            wgm: window height W for Algorithm 1 (0 = full M-major, default)
        """
        assert input.num_dims == 2
        assert weight.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Gang linear only supported on MI300X"
        batch_size = self.max_num_batched_tokens
        output_size = weight.dim(0)
        assert output_size % 8 == 0, f"Output size {output_size} must be divisible by 8"
        chunk_n = output_size // 8
        assert chunk_n % tile_n == 0, f"Chunk {chunk_n} must be divisible by tile_n {tile_n}"
        n_tiles_per_xcd = chunk_n // tile_n
        assert batch_size % m_tiles == 0, f"batch {batch_size} must be divisible by m_tiles {m_tiles}"
        m_per_tile = batch_size // m_tiles
        total_tiles_per_xcd = n_tiles_per_xcd * m_tiles
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        # weight: partition dim 0 (rows) by bid.x → each XCD gets chunk of weight rows
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        # output: partition dim 1 (columns) by bid.x → each XCD writes to its column range
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        # params: [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
        #          n_tiles_per_xcd, wgm]
        self.kn_graph.register_task(
            tb_graph, "gang_linear_mi300",
            [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
             n_tiles_per_xcd, wgm]
        )

    def gang_linear_with_residual_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        tile_n: int,
        output_stride: int,
        m_tiles: int = 1,
        wgm: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang linear with residual + HipKittens Algorithm 1 windowed traversal."""
        assert input.num_dims == 2
        assert weight.num_dims == 2
        assert residual.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Gang linear only supported on MI300X"
        batch_size = self.max_num_batched_tokens
        output_size = weight.dim(0)
        assert output_size % 8 == 0
        chunk_n = output_size // 8
        assert chunk_n % tile_n == 0
        n_tiles_per_xcd = chunk_n // tile_n
        assert batch_size % m_tiles == 0
        m_per_tile = batch_size // m_tiles
        total_tiles_per_xcd = n_tiles_per_xcd * m_tiles
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        # weight: partition dim 0 (rows) by bid.x
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        # residual: partition dim 1 (columns) by bid.x
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        # output: partition dim 1 (columns) by bid.x
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, residual, output], tb_graph)
        # params: [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
        #          n_tiles_per_xcd, wgm]
        self.kn_graph.register_task(
            tb_graph, "gang_linear_res_mi300",
            [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
             n_tiles_per_xcd, wgm]
        )


    def gang_linear_bias_layer(
        self,
        input: DTensor,
        weight: DTensor,
        bias: DTensor,
        output: DTensor,
        tile_n: int,
        output_stride: int,
        m_tiles: int = 1,
        wgm: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang linear with fused bias_add in epilogue.
        3 inputs (activation, weight, bias), 1 output."""
        assert input.num_dims == 2
        assert weight.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Gang linear only supported on MI300X"
        batch_size = self.max_num_batched_tokens
        output_size = weight.dim(0)
        assert output_size % 8 == 0
        chunk_n = output_size // 8
        assert chunk_n % tile_n == 0
        n_tiles_per_xcd = chunk_n // tile_n
        assert batch_size % m_tiles == 0
        m_per_tile = batch_size // m_tiles
        total_tiles_per_xcd = n_tiles_per_xcd * m_tiles
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)  # bias: partition dim 1 (columns) by bid.x
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, bias, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "gang_linear_bias_mi300",
            [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
             n_tiles_per_xcd, wgm]
        )

    def gang_rmsnorm_linear_bias_layer(
        self,
        norm_input: DTensor,
        norm_weight: DTensor,
        norm_output: DTensor,
        linear_weight: DTensor,
        bias: DTensor,
        output: DTensor,
        actual_hidden_dim: int,
        tile_n: int,
        output_stride: int,
        m_tiles: int = 1,
        wgm: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused RMSNorm + Gang Linear + Bias.

        Eliminates the dispatch barrier between rmsnorm and a downstream
        gang_linear_bias by having every gang-linear worker compute the
        RMSNorm prologue locally before its MFMA. All workers write the
        same normalized values to ``norm_output`` (idempotent), then read
        from it for their linear tile.

        Inputs: norm_input, norm_weight, norm_output (writable scratch),
                linear_weight, bias.
        Output: linear output.

        ``actual_hidden_dim`` is the unpadded hidden size used for the RMS
        denominator (e.g. 2880 for GPT-OSS, with norm_input padded to 3072).
        """
        assert norm_input.num_dims == 2
        assert linear_weight.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"
        batch_size = self.max_num_batched_tokens
        output_size = linear_weight.dim(0)
        assert output_size % 8 == 0
        chunk_n = output_size // 8
        assert chunk_n % tile_n == 0
        n_tiles_per_xcd = chunk_n // tile_n
        assert batch_size % m_tiles == 0
        m_per_tile = batch_size // m_tiles
        total_tiles_per_xcd = n_tiles_per_xcd * m_tiles
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # forloop_dim must reference an existing dim for each tensor
        tb_graph.new_input(norm_input, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)  # 1D tensor
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(linear_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [norm_input, norm_weight, norm_output, linear_weight, bias, output],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_rmsnorm_linear_bias_mi300",
            [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
             n_tiles_per_xcd, wgm, actual_hidden_dim]
        )

    def gang_rmsnorm_linear_bias_topk_layer(
        self,
        norm_input: DTensor,
        norm_weight: DTensor,
        norm_output: DTensor,
        linear_weight: DTensor,
        bias: DTensor,
        logits_scratch: DTensor,
        gang_counter: DTensor,
        topk_weight: DTensor,
        routing_indices: DTensor,
        active_expert_ids: DTensor,
        actual_hidden_dim: int,
        tile_n: int,
        output_stride: int,
        num_experts_per_tok: int = 4,
        m_tiles: int = 1,
        wgm: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused RMSNorm + Gang Linear + Bias + TopK Softmax.

        Eliminates the scheduler gaps between the router gang linear and the
        TopK softmax routing task. The last gang worker across all 8 XCDs
        detects completion via an atomic counter and computes TopK inline.

        Inputs (7): norm_input, norm_weight, norm_output (scratch),
                    linear_weight, bias, logits_scratch (scratch),
                    gang_counter (scratch).
        Outputs (3): topk_weight, routing_indices, active_expert_ids.
        """
        assert norm_input.num_dims == 2
        assert linear_weight.num_dims == 2
        assert logits_scratch.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"
        batch_size = self.max_num_batched_tokens
        num_experts = output_stride  # router output width = num_experts
        output_size = linear_weight.dim(0)
        assert output_size % 8 == 0
        chunk_n = output_size // 8
        assert chunk_n % tile_n == 0
        n_tiles_per_xcd = chunk_n // tile_n
        assert batch_size % m_tiles == 0
        m_per_tile = batch_size // m_tiles
        total_tiles_per_xcd = n_tiles_per_xcd * m_tiles
        total_gang_tiles = total_tiles_per_xcd * 8  # 8 XCDs on MI300X
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 7 inputs
        tb_graph.new_input(norm_input, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(linear_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        tb_graph.new_input(logits_scratch, (1, -1, -1), 1, True)
        tb_graph.new_input(gang_counter, (-1, -1, -1), 0, True)
        # 3 outputs
        tb_graph.new_input(topk_weight, (0, -1, -1), -1, True)
        tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(active_expert_ids, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [norm_input, norm_weight, norm_output, linear_weight, bias,
             logits_scratch, gang_counter,
             topk_weight, routing_indices, active_expert_ids],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_rmsnorm_linear_bias_topk_mi300",
            [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
             n_tiles_per_xcd, wgm, actual_hidden_dim, num_experts,
             num_experts_per_tok, total_gang_tiles]
        )

    def gang_rmsnorm_linear_mxfp4_bias_layer(
        self,
        norm_input: DTensor,
        norm_weight: DTensor,
        norm_output: DTensor,
        mxfp4_weight: DTensor,
        bias: DTensor,
        output: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        output_stride: int,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused RMSNorm + MXFP4 Gang Linear + Bias.

        Same as gang_rmsnorm_linear_bias_layer but uses MXFP4 weights with
        hardware FP4xFP8 MFMA instead of BF16 CK GEMM. Weight is packed in
        workgroup layout: [n_wgs, wg_bytes].

        ``actual_hidden_dim`` is the unpadded hidden size used for the RMS
        denominator (e.g. 2880 for GPT-OSS, with norm_input padded to 3072).
        """
        assert norm_input.num_dims == 2
        assert mxfp4_weight.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"
        batch_size = self.max_num_batched_tokens
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        n_wgs_per_xcd = n_wgs // 8
        total_tiles_per_xcd = batch_size * n_wgs_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(norm_input, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [norm_input, norm_weight, norm_output, mxfp4_weight, bias, output],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_rmsnorm_linear_mxfp4_bias_mi300",
            [output_stride, output_per_wg, n_wgs_per_xcd,
             total_tiles_per_xcd, actual_hidden_dim]
        )

    def gang_rmsnorm_linear_mxfp4_bias_argmax_layer(
        self,
        norm_input: DTensor,
        norm_weight: DTensor,
        norm_output: DTensor,
        mxfp4_weight: DTensor,
        bias: DTensor,
        argmax_part_value: DTensor,
        argmax_part_index: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        output_stride: int,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused RMSNorm + MXFP4 Gang Linear + Bias + Argmax (norm-once).

        CROC-style: each worker enters once, does RMSNorm+FP8 quant once,
        then loops internally over all its assigned WGs. Argmax accumulated
        in registers across ALL tiles. No logits written to HBM.

        total_tiles_per_xcd = workers_per_xcd (each worker enters once).
        Output: one (bf16 max, int64 abs_idx) per worker.
        Follow with argmax_reduce_layer(CHUNK_SIZE=0) for final token.
        """
        assert norm_input.num_dims == 2
        assert mxfp4_weight.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        n_wgs_per_xcd = n_wgs // 8
        workers_per_xcd = self.num_workers // 8
        total_tiles_per_xcd = workers_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(norm_input, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        tb_graph.new_input(argmax_part_value, (1, -1, -1), -1, True)
        tb_graph.new_input(argmax_part_index, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [norm_input, norm_weight, norm_output, mxfp4_weight, bias,
             argmax_part_value, argmax_part_index],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_rmsnorm_linear_mxfp4_bias_argmax_mi300",
            [output_stride, output_per_wg, n_wgs_per_xcd,
             workers_per_xcd, actual_hidden_dim]
        )
        # Absolute index — CHUNK_SIZE=0 in argmax_reduce skips chunk math
        self.argmax_partial_output_size = 0

    def gang_mulsumradd_rmsnorm_linear_mxfp4_bias_layer(
        self,
        mlp_out: DTensor,
        routing_weight: DTensor,
        residual: DTensor,
        x_output: DTensor,
        norm_weight: DTensor,
        norm_scratch: DTensor,
        mxfp4_weight: DTensor,
        bias: DTensor,
        qkv_output: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        output_stride: int,
        num_topk: int = 4,
        input_stride: int = -1,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused MulSumAdd + RMSNorm + MXFP4 Gang Linear + Bias.

        Merges the MulSumAdd from the previous layer's MoE block into the
        QKV kernel's prologue, eliminating one task dispatch.

        7 inputs (mlp_out, routing_weight, residual, norm_weight, norm_scratch,
        mxfp4_weight, bias) + 2 outputs (x_output, qkv_output).
        """
        assert mlp_out.num_dims == 3   # (batch, topk, hidden)
        assert routing_weight.num_dims == 2  # (batch, topk)
        assert residual.num_dims == 2  # (batch, hidden)
        assert mxfp4_weight.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        if input_stride < 0:
            input_stride = residual.dim(1)

        batch_size = self.max_num_batched_tokens
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        n_wgs_per_xcd = n_wgs // 8
        total_tiles_per_xcd = batch_size * n_wgs_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 7 inputs
        tb_graph.new_input(mlp_out, (-1, -1, -1), 1, True)
        tb_graph.new_input(routing_weight, (-1, -1, -1), 1, True)
        tb_graph.new_input(residual, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_scratch, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        # 2 outputs
        tb_graph.new_input(x_output, (-1, -1, -1), -1, True)
        tb_graph.new_input(qkv_output, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [mlp_out, routing_weight, residual, norm_weight, norm_scratch,
             mxfp4_weight, bias, x_output, qkv_output],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_mulsumradd_rmsnorm_linear_mxfp4_bias_mi300",
            [output_stride, output_per_wg, n_wgs_per_xcd,
             total_tiles_per_xcd, actual_hidden_dim, num_topk, input_stride]
        )

    def gang_rmsnorm_linear_mxfp4_bias_kvupd_layer(
        self,
        norm_input: DTensor,
        norm_weight: DTensor,
        norm_output: DTensor,
        mxfp4_weight: DTensor,
        bias: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_workspace: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        head_dim: int,
        num_q_per_kv: int,
        kv_stride: int,
        q_ws_stride: int,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused RMSNorm + MXFP4 Gang Linear + KV Cache Update (layer 0).

        Combines QKV MFMA with KV cache update: the epilogue applies RoPE and
        writes Q to q_workspace, K/V to paged caches directly.
        5 inputs + 3 outputs.
        """
        assert norm_input.num_dims == 2
        assert mxfp4_weight.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0
        n_wgs_per_xcd = n_wgs // 8
        total_tiles_per_xcd = batch_size * n_wgs_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 5 inputs
        tb_graph.new_input(norm_input, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        # 3 outputs (un-partitioned caches + workspace)
        tb_graph.new_input(k_cache, (-1, -1, -1), -1, True)
        tb_graph.new_input(v_cache, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [norm_input, norm_weight, norm_output, mxfp4_weight, bias,
             k_cache, v_cache, q_workspace],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_rmsnorm_linear_mxfp4_bias_kvupd_mi300",
            [output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
             actual_hidden_dim, head_dim, num_q_per_kv, self.page_size,
             kv_stride, q_ws_stride]
        )

    def gang_mulsumradd_rmsnorm_linear_mxfp4_bias_kvupd_layer(
        self,
        mlp_out: DTensor,
        routing_weight: DTensor,
        residual: DTensor,
        x_output: DTensor,
        norm_weight: DTensor,
        norm_scratch: DTensor,
        mxfp4_weight: DTensor,
        bias: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_workspace: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        num_topk: int,
        input_stride: int,
        head_dim: int,
        num_q_per_kv: int,
        kv_stride: int,
        q_ws_stride: int,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused MulSumAdd + RMSNorm + MXFP4 Gang Linear + KV Cache Update (layers 1+).

        7 inputs + 4 outputs.
        """
        assert mlp_out.num_dims == 3
        assert routing_weight.num_dims == 2
        assert residual.num_dims == 2
        assert mxfp4_weight.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        if input_stride < 0:
            input_stride = residual.dim(1)

        batch_size = self.max_num_batched_tokens
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0
        n_wgs_per_xcd = n_wgs // 8
        total_tiles_per_xcd = batch_size * n_wgs_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 7 inputs
        tb_graph.new_input(mlp_out, (-1, -1, -1), 1, True)
        tb_graph.new_input(routing_weight, (-1, -1, -1), 1, True)
        tb_graph.new_input(residual, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_scratch, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        # 4 outputs
        tb_graph.new_input(x_output, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, -1, -1), -1, True)
        tb_graph.new_input(v_cache, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [mlp_out, routing_weight, residual, norm_weight, norm_scratch,
             mxfp4_weight, bias, x_output, k_cache, v_cache, q_workspace],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_mulsumradd_rmsnorm_linear_mxfp4_bias_kvupd_mi300",
            [output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
             actual_hidden_dim, num_topk, input_stride,
             head_dim, num_q_per_kv, self.page_size,
             kv_stride, q_ws_stride]
        )

    def gang_resaddf32_rmsnorm_linear_mxfp4_bias_layer(
        self,
        workspace_f32: DTensor,
        residual: DTensor,
        x_output: DTensor,
        norm_weight: DTensor,
        norm_scratch: DTensor,
        mxfp4_weight: DTensor,
        bias: DTensor,
        qkv_output: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        output_stride: int,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused ResAddF32 + RMSNorm + MXFP4 Gang Linear + Bias.

        Reads from f32 workspace (pre-accumulated by W2 atomicAdd) instead of
        doing MulSumAdd from 4 expert bf16 slots. Zeros workspace after read.

        6 inputs (workspace_f32, residual, norm_weight, norm_scratch,
        mxfp4_weight, bias) + 2 outputs (x_output, qkv_output).
        """
        assert workspace_f32.num_dims == 2   # (batch, hidden) f32
        assert residual.num_dims == 2        # (batch, hidden) bf16
        assert mxfp4_weight.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        n_wgs_per_xcd = n_wgs // 8
        total_tiles_per_xcd = batch_size * n_wgs_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 6 inputs
        tb_graph.new_input(workspace_f32, (-1, -1, -1), 1, True)
        tb_graph.new_input(residual, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_scratch, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        # 2 outputs
        tb_graph.new_input(x_output, (-1, -1, -1), -1, True)
        tb_graph.new_input(qkv_output, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [workspace_f32, residual, norm_weight, norm_scratch,
             mxfp4_weight, bias, x_output, qkv_output],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_resaddf32_rmsnorm_linear_mxfp4_bias_mi300",
            [output_stride, output_per_wg, n_wgs_per_xcd,
             total_tiles_per_xcd, actual_hidden_dim]
        )

    def gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_layer(
        self,
        workspace_f32: DTensor,
        residual: DTensor,
        x_output: DTensor,
        norm_weight: DTensor,
        norm_scratch: DTensor,
        mxfp4_weight: DTensor,
        bias: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_workspace: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        head_dim: int,
        num_q_per_kv: int,
        kv_stride: int,
        q_ws_stride: int,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused ResAddF32 + RMSNorm + MXFP4 Gang Linear + KV Cache Update (layers 1+).

        6 inputs + 4 outputs.
        """
        assert workspace_f32.num_dims == 2
        assert residual.num_dims == 2
        assert mxfp4_weight.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0
        n_wgs_per_xcd = n_wgs // 8
        total_tiles_per_xcd = batch_size * n_wgs_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 6 inputs
        tb_graph.new_input(workspace_f32, (-1, -1, -1), 1, True)
        tb_graph.new_input(residual, (-1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_scratch, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        # 4 outputs
        tb_graph.new_input(x_output, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, -1, -1), -1, True)
        tb_graph.new_input(v_cache, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [workspace_f32, residual, norm_weight, norm_scratch,
             mxfp4_weight, bias, x_output, k_cache, v_cache, q_workspace],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_mi300",
            [output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
             actual_hidden_dim, head_dim, num_q_per_kv, self.page_size,
             kv_stride, q_ws_stride]
        )

    def gang_qkv_attn_fused_layer(
        self,
        workspace_f32: DTensor,
        residual: DTensor,
        x_output: DTensor,
        norm_weight: DTensor,
        norm_scratch: DTensor,
        mxfp4_weight: DTensor,
        bias: DTensor,
        sinks: DTensor,
        barrier: DTensor,
        lse_acc: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_workspace: DTensor,
        o_acc: DTensor,
        actual_hidden_dim: int,
        output_per_wg: int,
        head_dim: int,
        num_q_per_kv: int,
        kv_stride: int,
        q_ws_stride: int,
        num_kv_chunks: int,
        num_kv_heads: int,
        sliding_window: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused QKV + Attention gang task (layers 1+).

        Phase 1: ResAddF32+RMSNorm+QKV+KVUpdate (all workers, gang tiles)
        Phase 2: CK FMHA attention (1 worker per XCD, after hierarchical barrier)

        9 inputs + 5 outputs.
        """
        assert workspace_f32.num_dims == 2
        assert residual.num_dims == 2
        assert mxfp4_weight.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0
        n_wgs_per_xcd = n_wgs // 8
        total_qkv_tiles_per_xcd = batch_size * n_wgs_per_xcd

        has_sinks = 1 if sinks is not None else 0
        q_workspace_stride = q_workspace.dim(1)
        kv_cache_stride = num_kv_heads * head_dim

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 9 inputs: slot[6] is sinks (or barrier as placeholder when no sinks)
        tb_graph.new_input(workspace_f32, (-1, -1, -1), 1, True)   # [0]
        tb_graph.new_input(residual, (-1, -1, -1), 1, True)        # [1]
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)     # [2]
        tb_graph.new_input(norm_scratch, (-1, -1, -1), 1, True)    # [3]
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)     # [4]
        tb_graph.new_input(bias, (1, -1, -1), 1, True)             # [5]
        sinks_or_placeholder = sinks if sinks is not None else barrier
        tb_graph.new_input(sinks_or_placeholder, (-1, -1, -1), -1, True)  # [6]
        tb_graph.new_input(barrier, (-1, -1, -1), -1, True)        # [7]
        tb_graph.new_input(lse_acc, (-1, -1, -1), -1, True)        # [8]
        # 5 outputs
        tb_graph.new_input(x_output, (-1, -1, -1), -1, True)       # [0]
        tb_graph.new_input(k_cache, (-1, -1, -1), -1, True)        # [1]
        tb_graph.new_input(v_cache, (-1, -1, -1), -1, True)        # [2]
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)    # [3]
        tb_graph.new_input(o_acc, (-1, -1, -1), -1, True)          # [4]
        self.kn_graph.customized(
            [workspace_f32, residual, norm_weight, norm_scratch,
             mxfp4_weight, bias, sinks_or_placeholder, barrier, lse_acc,
             x_output, k_cache, v_cache, q_workspace, o_acc],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_qkv_attn_fused_mi300",
            [output_per_wg, n_wgs_per_xcd, total_qkv_tiles_per_xcd,
             actual_hidden_dim, head_dim, num_q_per_kv, self.page_size,
             kv_stride, q_ws_stride,
             self.max_seq_length, num_kv_chunks, q_workspace_stride,
             kv_cache_stride, num_kv_heads, sliding_window, has_sinks]
        )

    def moe_residual_add_f32_layer(
        self,
        workspace_f32: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """MoE residual add from f32 workspace (last layer).

        output[b,h] = bf16(workspace_f32[b,h] + residual_bf16[b,h])
        workspace_f32[b,h] = 0 (zero for next iteration)
        """
        assert workspace_f32.num_dims == 2  # (batch, hidden) f32
        assert residual.num_dims == 2       # (batch, hidden) bf16
        assert output.num_dims == 2         # (batch, hidden) bf16
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(workspace_f32, (0, 1, -1), -1, True)
        tb_graph.new_input(residual, (0, 1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        self.kn_graph.customized(
            [workspace_f32, residual, output], tb_graph
        )
        self.kn_graph.register_task(
            tb_graph, "moe_residual_add_f32_mi300",
            [residual.dim(1)]  # output_stride = hidden_size
        )

    def gang_linear_mxfp4_res_bias_layer(
        self,
        input: DTensor,
        mxfp4_weight: DTensor,
        residual: DTensor,
        bias: DTensor,
        output: DTensor,
        output_per_wg: int,
        output_stride: int,
        block_dim: tuple = (256, 1, 1),
    ):
        """MXFP4 Gang Linear with Residual + Bias.

        Replaces gang_splitk_linear_res_bias_layer when weights are MXFP4.
        No split-K needed since MXFP4 tiles are fast enough.

        4 inputs (input, mxfp4_weight, residual, bias), 1 output.
        """
        assert input.num_dims == 2
        assert mxfp4_weight.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"
        batch_size = self.max_num_batched_tokens
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        n_wgs_per_xcd = n_wgs // 8
        total_tiles_per_xcd = batch_size * n_wgs_per_xcd
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input, mxfp4_weight, residual, bias, output],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_linear_mxfp4_res_bias_mi300",
            [output_stride, output_per_wg, n_wgs_per_xcd,
             total_tiles_per_xcd]
        )

    def gang_linear_mxfp4_res_bias_rmsnorm_topk_layer(
        self,
        # O-PROJ inputs
        input: DTensor,
        mxfp4_weight: DTensor,
        residual: DTensor,
        oproj_bias: DTensor,
        # TopK inputs
        norm_weight: DTensor,
        norm_output: DTensor,
        router_weight: DTensor,
        router_bias: DTensor,
        logits_scratch: DTensor,
        counters: DTensor,
        # Outputs
        output: DTensor,
        topk_weight: DTensor,
        routing_indices: DTensor,
        active_expert_ids: DTensor,
        # Parameters
        output_per_wg: int,
        output_stride: int,
        actual_hidden_dim: int,
        num_experts: int,
        topk_k: int,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused O-PROJ + RMSNorm + Router Linear + TopK Softmax.

        Combines gang_linear_mxfp4_res_bias and gang_rmsnorm_linear_bias_topk
        into a single gang task, eliminating one event barrier per layer.

        10 inputs, 4 outputs.
        """
        assert input.num_dims == 2
        assert mxfp4_weight.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"
        batch_size = self.max_num_batched_tokens

        # O-PROJ tiling
        n_wgs = mxfp4_weight.dim(0)
        assert n_wgs % 8 == 0, f"n_wgs {n_wgs} must be divisible by 8"
        n_wgs_per_xcd = n_wgs // 8
        oproj_tiles_per_xcd = batch_size * n_wgs_per_xcd

        # TopK tiling (one expert per worker)
        router_output_size = router_weight.dim(0)
        assert router_output_size % 8 == 0
        router_tile_n = router_output_size // 8  # chunk_N per XCD
        topk_tiles_per_xcd = router_tile_n  # 1 tile per expert
        total_topk_tiles = topk_tiles_per_xcd * 8

        # Gang dispatch uses max of both tile counts
        total_tiles_per_xcd = max(oproj_tiles_per_xcd, topk_tiles_per_xcd)
        # All dispatched workers enter the oproj barrier, not just oproj workers
        total_oproj_tiles = total_tiles_per_xcd * 8

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 10 inputs
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(mxfp4_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        tb_graph.new_input(oproj_bias, (1, -1, -1), 1, True)
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)
        tb_graph.new_input(router_weight, (0, -1, -1), 1, True)
        tb_graph.new_input(router_bias, (1, -1, -1), 1, True)
        tb_graph.new_input(logits_scratch, (1, -1, -1), 1, True)
        tb_graph.new_input(counters, (-1, -1, -1), 0, True)
        # 4 outputs
        # output is replicated so RMSNorm can read full hidden dim;
        # O-PROJ epilogue uses xcd_output_col_offset for correct writes
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        tb_graph.new_input(topk_weight, (0, -1, -1), -1, True)
        tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(active_expert_ids, (-1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input, mxfp4_weight, residual, oproj_bias,
             norm_weight, norm_output, router_weight, router_bias,
             logits_scratch, counters,
             output, topk_weight, routing_indices, active_expert_ids],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300",
            [output_stride, output_per_wg, n_wgs_per_xcd, total_oproj_tiles,
             actual_hidden_dim, num_experts, topk_k, router_tile_n,
             total_topk_tiles, total_tiles_per_xcd]
        )

    def gang_oproj_topk_moe_fused_layer(
        self,
        # O-PROJ inputs
        input: DTensor,
        oproj_weight: DTensor,
        residual: DTensor,
        oproj_bias: DTensor,
        norm_weight: DTensor,
        norm_output: DTensor,
        router_weight: DTensor,
        router_bias: DTensor,
        logits_scratch: DTensor,
        counters: DTensor,
        # MoE inputs
        gate_up_weight: DTensor,
        down_weight: DTensor,
        w13_bias: DTensor,
        w2_bias: DTensor,
        moe_barrier: DTensor,
        swiglu_out: DTensor,
        # Outputs
        oproj_output: DTensor,
        topk_weight: DTensor,
        routing_indices: DTensor,
        active_expert_ids: DTensor,
        routing_weight_moe: DTensor,
        workspace_f32: DTensor,
        # Parameters
        output_per_wg: int,
        output_stride: int,
        actual_hidden_dim: int,
        num_experts: int,
        topk_k: int,
        w13_output_per_wg: int = 128,
        w2_output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
    ):
        """Fused O-PROJ+TopK+MoE: combines task 213 and 187 into one gang task.
        Eliminates one inter-task event barrier per layer.

        16 inputs, 6 outputs.
        """
        assert input.num_dims == 2
        assert oproj_weight.num_dims == 2
        assert gate_up_weight.num_dims == 3
        assert down_weight.num_dims == 3
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens

        # O-PROJ tiling (same as task 213)
        n_wgs = oproj_weight.dim(0)
        assert n_wgs % 8 == 0
        n_wgs_per_xcd = n_wgs // 8
        oproj_tiles_per_xcd = batch_size * n_wgs_per_xcd

        # TopK tiling (same as task 213)
        router_output_size = router_weight.dim(0)
        assert router_output_size % 8 == 0
        router_tile_n = router_output_size // 8
        topk_tiles_per_xcd = router_tile_n
        total_topk_tiles = topk_tiles_per_xcd * 8
        total_oproj_tiles = max(oproj_tiles_per_xcd, topk_tiles_per_xcd) * 8

        # MoE dimensions from tensors
        # hidden_size for MoE = norm_output dimension (PADDED_HIDDEN_SIZE),
        # NOT input (attn_out) dimension which is num_heads * head_dim.
        hidden_size = norm_output.dim(1)
        intermediate_size = swiglu_out.dim(2)

        # MoE tiling (same as task 187)
        moe_num_experts = gate_up_weight.dim(0)
        w13_wgs = gate_up_weight.dim(1)
        w2_wgs = down_weight.dim(1)
        num_topk = swiglu_out.dim(1)
        max_activated = min(num_topk * batch_size, moe_num_experts)
        PAD_MULTIPLE = 240

        w13_tiles = batch_size * w13_wgs
        w2_tiles = batch_size * w2_wgs
        total_w13_real = max_activated * w13_tiles
        total_w13_padded = ((total_w13_real + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        total_w2 = max_activated * w2_tiles
        total_tiles_all = total_w13_padded + total_w2
        moe_total_tiles_per_xcd = (total_tiles_all + 7) // 8

        # Match standalone MoE worker count (30 = 240 workers / 8 XCDs)
        workers_per_xcd = self.num_workers // 8  # 30

        reduction_size = input.dim(1)

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 16 inputs
        tb_graph.new_input(input, (-1, -1, -1), 1, True)            # [0] attn_out
        tb_graph.new_input(oproj_weight, (0, -1, -1), 1, True)      # [1] O-proj weight
        tb_graph.new_input(residual, (1, -1, -1), 1, True)          # [2] residual
        tb_graph.new_input(oproj_bias, (1, -1, -1), 1, True)        # [3] O-proj bias
        tb_graph.new_input(norm_weight, (-1, -1, -1), 0, True)      # [4] RMSNorm weight
        tb_graph.new_input(norm_output, (-1, -1, -1), 1, True)      # [5] norm output
        tb_graph.new_input(router_weight, (0, -1, -1), 1, True)     # [6] router weight
        tb_graph.new_input(router_bias, (1, -1, -1), 1, True)       # [7] router bias
        tb_graph.new_input(logits_scratch, (1, -1, -1), 1, True)    # [8] logits scratch
        tb_graph.new_input(counters, (-1, -1, -1), 0, True)         # [9] hier barrier
        tb_graph.new_input(gate_up_weight, (-1, 1, -1), 2, True)    # [10] W13 weight
        tb_graph.new_input(down_weight, (-1, 1, -1), 2, True)       # [11] W2 weight
        tb_graph.new_input(w13_bias, (-1, -1, -1), -1, True)        # [12] W13 bias
        tb_graph.new_input(w2_bias, (-1, -1, -1), -1, True)         # [13] W2 bias
        tb_graph.new_input(moe_barrier, (-1, -1, -1), -1, True)     # [14] MoE barrier
        tb_graph.new_input(swiglu_out, (-1, 2, -1), -1, True)       # [15] SwiGLU scratch
        # 6 outputs
        tb_graph.new_input(oproj_output, (-1, -1, -1), -1, True)    # [0] O-proj output
        tb_graph.new_input(topk_weight, (0, -1, -1), -1, True)      # [1] topk weight
        tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)  # [2] routing indices
        tb_graph.new_input(active_expert_ids, (-1, -1, -1), -1, True)  # [3] expert mask
        tb_graph.new_input(routing_weight_moe, (-1, -1, -1), -1, True)  # [4] routing weight (MoE)
        tb_graph.new_input(workspace_f32, (-1, -1, -1), -1, True)   # [5] MoE accumulator

        self.kn_graph.customized(
            [input, oproj_weight, residual, oproj_bias,
             norm_weight, norm_output, router_weight, router_bias,
             logits_scratch, counters,
             gate_up_weight, down_weight, w13_bias, w2_bias,
             moe_barrier, swiglu_out,
             oproj_output, topk_weight, routing_indices,
             active_expert_ids, routing_weight_moe, workspace_f32],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_oproj_topk_moe_fused_mi300",
            [output_stride, output_per_wg, n_wgs_per_xcd, total_oproj_tiles,
             actual_hidden_dim, num_experts, topk_k, router_tile_n,
             total_topk_tiles, oproj_tiles_per_xcd,
             intermediate_size, hidden_size,
             w13_output_per_wg, w2_output_per_wg,
             moe_total_tiles_per_xcd, workers_per_xcd]
        )

    def gang_full_layer_fused_layer(
        self,
        # QKV+Attn inputs (from type 214)
        workspace_f32: DTensor,
        residual: DTensor,
        norm_weight_pre: DTensor,
        norm_scratch_pre: DTensor,
        qkv_weight: DTensor,
        qkv_bias: DTensor,
        sinks: DTensor,
        qkv_barrier: DTensor,
        lse_acc: DTensor,
        # O-proj+TopK inputs (from type 215)
        oproj_weight: DTensor,
        oproj_bias: DTensor,
        norm_weight_post: DTensor,
        norm_scratch_post: DTensor,
        router_weight: DTensor,
        router_bias: DTensor,
        logits_scratch: DTensor,
        oproj_counters: DTensor,
        # MoE inputs
        gate_up_weight: DTensor,
        down_weight: DTensor,
        w13_bias: DTensor,
        w2_bias: DTensor,
        moe_barrier: DTensor,
        swiglu_out: DTensor,
        o_acc_f32: DTensor,
        # Outputs (QKV+Attn)
        x_output: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_workspace: DTensor,
        o_acc: DTensor,
        # Outputs (O-proj+TopK+MoE)
        attn_proj_out: DTensor,
        topk_weight: DTensor,
        routing_indices: DTensor,
        active_expert_ids: DTensor,
        routing_weight_moe: DTensor,
        moe_workspace_f32: DTensor,
        # Parameters
        actual_hidden_dim: int,
        qkv_output_per_wg: int,
        oproj_output_per_wg: int,
        head_dim: int,
        num_q_per_kv: int,
        kv_stride: int,
        q_ws_stride: int,
        num_kv_chunks: int,
        num_kv_heads: int,
        num_experts: int,
        topk_k: int,
        sliding_window: int = 0,
        w13_output_per_wg: int = 128,
        w2_output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
    ):
        """Full-layer fused gang task: QKV+Attn+O-proj+TopK+MoE.
        Combines task 214 and 215 into one gang task per layer.
        24 inputs, 11 outputs.
        """
        assert residual.num_dims == 2
        assert qkv_weight.num_dims == 2
        assert oproj_weight.num_dims == 2
        assert gate_up_weight.num_dims == 3
        assert down_weight.num_dims == 3
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens

        # QKV tiling (from type 214)
        qkv_n_wgs = qkv_weight.dim(0)
        assert qkv_n_wgs % 8 == 0
        qkv_n_wgs_per_xcd = qkv_n_wgs // 8
        total_qkv_tiles_per_xcd = batch_size * qkv_n_wgs_per_xcd

        has_sinks = 1 if sinks is not None else 0
        q_workspace_stride = q_workspace.dim(1)
        kv_cache_stride = num_kv_heads * head_dim

        # O-PROJ tiling (from type 215)
        oproj_n_wgs = oproj_weight.dim(0)
        assert oproj_n_wgs % 8 == 0
        oproj_n_wgs_per_xcd = oproj_n_wgs // 8
        oproj_tiles_per_xcd = batch_size * oproj_n_wgs_per_xcd
        oproj_output_stride = norm_scratch_post.dim(1)

        # TopK tiling
        router_output_size = router_weight.dim(0)
        assert router_output_size % 8 == 0
        router_tile_n = router_output_size // 8
        total_topk_tiles = router_tile_n * 8
        total_oproj_tiles = max(oproj_tiles_per_xcd, router_tile_n) * 8

        # MoE tiling (from type 187/215)
        moe_num_experts = gate_up_weight.dim(0)
        w13_wgs = gate_up_weight.dim(1)
        w2_wgs = down_weight.dim(1)
        num_topk = swiglu_out.dim(1)
        max_activated = min(num_topk * batch_size, moe_num_experts)
        PAD_MULTIPLE = 240

        intermediate_size = swiglu_out.dim(2)

        w13_tiles = batch_size * w13_wgs
        w2_tiles = batch_size * w2_wgs
        total_w13_real = max_activated * w13_tiles
        total_w13_padded = ((total_w13_real + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        total_w2 = max_activated * w2_tiles
        total_tiles_all = total_w13_padded + total_w2
        moe_total_tiles_per_xcd = (total_tiles_all + 7) // 8

        workers_per_xcd = self.num_workers // 8  # 30

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 24 inputs
        tb_graph.new_input(workspace_f32, (-1, -1, -1), 1, True)         # [0]
        tb_graph.new_input(residual, (-1, -1, -1), 1, True)              # [1]
        tb_graph.new_input(norm_weight_pre, (-1, -1, -1), 0, True)       # [2]
        tb_graph.new_input(norm_scratch_pre, (-1, -1, -1), 1, True)      # [3]
        tb_graph.new_input(qkv_weight, (0, -1, -1), 1, True)            # [4]
        tb_graph.new_input(qkv_bias, (1, -1, -1), 1, True)              # [5]
        sinks_or_placeholder = sinks if sinks is not None else qkv_barrier
        tb_graph.new_input(sinks_or_placeholder, (-1, -1, -1), -1, True) # [6]
        tb_graph.new_input(qkv_barrier, (-1, -1, -1), -1, True)         # [7]
        tb_graph.new_input(lse_acc, (-1, -1, -1), -1, True)             # [8]
        tb_graph.new_input(oproj_weight, (0, -1, -1), 1, True)          # [9]
        tb_graph.new_input(oproj_bias, (1, -1, -1), 1, True)            # [10]
        tb_graph.new_input(norm_weight_post, (-1, -1, -1), 0, True)     # [11]
        tb_graph.new_input(norm_scratch_post, (-1, -1, -1), 1, True)    # [12]
        tb_graph.new_input(router_weight, (0, -1, -1), 1, True)         # [13]
        tb_graph.new_input(router_bias, (1, -1, -1), 1, True)           # [14]
        tb_graph.new_input(logits_scratch, (1, -1, -1), 1, True)        # [15]
        tb_graph.new_input(oproj_counters, (-1, -1, -1), 0, True)       # [16]
        tb_graph.new_input(gate_up_weight, (-1, 1, -1), 2, True)        # [17]
        tb_graph.new_input(down_weight, (-1, 1, -1), 2, True)           # [18]
        tb_graph.new_input(w13_bias, (-1, -1, -1), -1, True)            # [19]
        tb_graph.new_input(w2_bias, (-1, -1, -1), -1, True)             # [20]
        tb_graph.new_input(moe_barrier, (-1, -1, -1), -1, True)         # [21]
        tb_graph.new_input(swiglu_out, (-1, 2, -1), -1, True)           # [22]
        tb_graph.new_input(o_acc_f32, (-1, -1, -1), -1, True)           # [23]
        # 11 outputs
        tb_graph.new_input(x_output, (-1, -1, -1), -1, True)            # [0]
        tb_graph.new_input(k_cache, (-1, -1, -1), -1, True)             # [1]
        tb_graph.new_input(v_cache, (-1, -1, -1), -1, True)             # [2]
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)         # [3]
        tb_graph.new_input(o_acc, (-1, -1, -1), -1, True)               # [4]
        tb_graph.new_input(attn_proj_out, (-1, -1, -1), -1, True)       # [5]
        tb_graph.new_input(topk_weight, (0, -1, -1), -1, True)          # [6]
        tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)     # [7]
        tb_graph.new_input(active_expert_ids, (-1, -1, -1), -1, True)   # [8]
        tb_graph.new_input(routing_weight_moe, (-1, -1, -1), -1, True)  # [9]
        tb_graph.new_input(moe_workspace_f32, (-1, -1, -1), -1, True)   # [10]

        self.kn_graph.customized(
            [workspace_f32, residual, norm_weight_pre, norm_scratch_pre,
             qkv_weight, qkv_bias, sinks_or_placeholder, qkv_barrier, lse_acc,
             oproj_weight, oproj_bias, norm_weight_post, norm_scratch_post,
             router_weight, router_bias, logits_scratch, oproj_counters,
             gate_up_weight, down_weight, w13_bias, w2_bias,
             moe_barrier, swiglu_out, o_acc_f32,
             x_output, k_cache, v_cache, q_workspace, o_acc,
             attn_proj_out, topk_weight, routing_indices,
             active_expert_ids, routing_weight_moe, moe_workspace_f32],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_full_layer_fused_mi300",
            [qkv_output_per_wg, qkv_n_wgs_per_xcd, total_qkv_tiles_per_xcd,
             actual_hidden_dim, head_dim, num_q_per_kv, self.page_size,
             kv_stride, q_ws_stride,
             self.max_seq_length, num_kv_chunks, q_workspace_stride,
             kv_cache_stride, num_kv_heads, sliding_window, has_sinks,
             oproj_output_per_wg, oproj_output_stride, total_oproj_tiles,
             num_experts, topk_k, router_tile_n, total_topk_tiles,
             oproj_tiles_per_xcd, moe_total_tiles_per_xcd,
             w13_output_per_wg, w2_output_per_wg,
             intermediate_size, workers_per_xcd]
        )

    def gang_full_layer_with_lmhead_fused_layer(
        self,
        # QKV+Attn inputs (same as type 216)
        workspace_f32: DTensor,
        residual: DTensor,
        norm_weight_pre: DTensor,
        norm_scratch_pre: DTensor,
        qkv_weight: DTensor,
        qkv_bias: DTensor,
        sinks: DTensor,
        qkv_barrier: DTensor,
        lse_acc: DTensor,
        # O-proj+TopK inputs
        oproj_weight: DTensor,
        oproj_bias: DTensor,
        norm_weight_post: DTensor,
        norm_scratch_post: DTensor,
        router_weight: DTensor,
        router_bias: DTensor,
        logits_scratch: DTensor,
        oproj_counters: DTensor,
        # MoE inputs
        gate_up_weight: DTensor,
        down_weight: DTensor,
        w13_bias: DTensor,
        w2_bias: DTensor,
        moe_barrier: DTensor,
        swiglu_out: DTensor,
        o_acc_f32: DTensor,
        # LM head inputs (4 extra)
        lm_norm_weight: DTensor,
        lm_norm_scratch: DTensor,
        lm_mxfp4_weight: DTensor,
        lm_bias: DTensor,
        # Outputs (QKV+Attn)
        x_output: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        q_workspace: DTensor,
        o_acc: DTensor,
        # Outputs (O-proj+TopK+MoE)
        attn_proj_out: DTensor,
        topk_weight: DTensor,
        routing_indices: DTensor,
        active_expert_ids: DTensor,
        routing_weight_moe: DTensor,
        moe_workspace_f32: DTensor,
        # LM head outputs (2 extra)
        lm_logits: DTensor,
        argmax_output: DTensor,
        # Parameters
        actual_hidden_dim: int,
        qkv_output_per_wg: int,
        oproj_output_per_wg: int,
        head_dim: int,
        num_q_per_kv: int,
        kv_stride: int,
        q_ws_stride: int,
        num_kv_chunks: int,
        num_kv_heads: int,
        num_experts: int,
        topk_k: int,
        lm_output_per_wg: int,
        lm_output_stride: int,
        sliding_window: int = 0,
        w13_output_per_wg: int = 128,
        w2_output_per_wg: int = 64,
        block_dim: tuple = (256, 1, 1),
    ):
        """Full-layer + LM head + argmax fused gang task (type 217).
        28 inputs, 13 outputs, 33 params.
        """
        assert residual.num_dims == 2
        assert self.target_cc in (94, 95), "Only supported on MI300/MI350"

        batch_size = self.max_num_batched_tokens

        # QKV tiling
        qkv_n_wgs = qkv_weight.dim(0)
        assert qkv_n_wgs % 8 == 0
        qkv_n_wgs_per_xcd = qkv_n_wgs // 8
        total_qkv_tiles_per_xcd = batch_size * qkv_n_wgs_per_xcd

        has_sinks = 1 if sinks is not None else 0
        q_workspace_stride = q_workspace.dim(1)
        kv_cache_stride = num_kv_heads * head_dim

        # O-PROJ tiling
        oproj_n_wgs = oproj_weight.dim(0)
        assert oproj_n_wgs % 8 == 0
        oproj_tiles_per_xcd = batch_size * (oproj_n_wgs // 8)
        oproj_output_stride = norm_scratch_post.dim(1)

        # TopK tiling
        router_output_size = router_weight.dim(0)
        assert router_output_size % 8 == 0
        router_tile_n = router_output_size // 8
        total_topk_tiles = router_tile_n * 8
        total_oproj_tiles = max(oproj_tiles_per_xcd, router_tile_n) * 8

        # MoE tiling
        moe_num_experts = gate_up_weight.dim(0)
        w13_wgs = gate_up_weight.dim(1)
        w2_wgs = down_weight.dim(1)
        num_topk = swiglu_out.dim(1)
        max_activated = min(num_topk * batch_size, moe_num_experts)
        PAD_MULTIPLE = 240

        intermediate_size = swiglu_out.dim(2)

        w13_tiles = batch_size * w13_wgs
        w2_tiles = batch_size * w2_wgs
        total_w13_real = max_activated * w13_tiles
        total_w13_padded = ((total_w13_real + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        total_w2 = max_activated * w2_tiles
        total_tiles_all = total_w13_padded + total_w2
        moe_total_tiles_per_xcd = (total_tiles_all + 7) // 8

        workers_per_xcd = self.num_workers // 8  # 30

        # LM head tiling
        lm_n_wgs = lm_mxfp4_weight.dim(0)
        assert lm_n_wgs % 8 == 0
        lm_n_wgs_per_xcd = lm_n_wgs // 8

        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # 24 base inputs (same as type 216)
        tb_graph.new_input(workspace_f32, (-1, -1, -1), 1, True)         # [0]
        tb_graph.new_input(residual, (-1, -1, -1), 1, True)              # [1]
        tb_graph.new_input(norm_weight_pre, (-1, -1, -1), 0, True)       # [2]
        tb_graph.new_input(norm_scratch_pre, (-1, -1, -1), 1, True)      # [3]
        tb_graph.new_input(qkv_weight, (0, -1, -1), 1, True)            # [4]
        tb_graph.new_input(qkv_bias, (1, -1, -1), 1, True)              # [5]
        sinks_or_placeholder = sinks if sinks is not None else qkv_barrier
        tb_graph.new_input(sinks_or_placeholder, (-1, -1, -1), -1, True) # [6]
        tb_graph.new_input(qkv_barrier, (-1, -1, -1), -1, True)         # [7]
        tb_graph.new_input(lse_acc, (-1, -1, -1), -1, True)             # [8]
        tb_graph.new_input(oproj_weight, (0, -1, -1), 1, True)          # [9]
        tb_graph.new_input(oproj_bias, (1, -1, -1), 1, True)            # [10]
        tb_graph.new_input(norm_weight_post, (-1, -1, -1), 0, True)     # [11]
        tb_graph.new_input(norm_scratch_post, (-1, -1, -1), 1, True)    # [12]
        tb_graph.new_input(router_weight, (0, -1, -1), 1, True)         # [13]
        tb_graph.new_input(router_bias, (1, -1, -1), 1, True)           # [14]
        tb_graph.new_input(logits_scratch, (1, -1, -1), 1, True)        # [15]
        tb_graph.new_input(oproj_counters, (-1, -1, -1), 0, True)       # [16]
        tb_graph.new_input(gate_up_weight, (-1, 1, -1), 2, True)        # [17]
        tb_graph.new_input(down_weight, (-1, 1, -1), 2, True)           # [18]
        tb_graph.new_input(w13_bias, (-1, -1, -1), -1, True)            # [19]
        tb_graph.new_input(w2_bias, (-1, -1, -1), -1, True)             # [20]
        tb_graph.new_input(moe_barrier, (-1, -1, -1), -1, True)         # [21]
        tb_graph.new_input(swiglu_out, (-1, 2, -1), -1, True)           # [22]
        tb_graph.new_input(o_acc_f32, (-1, -1, -1), -1, True)           # [23]
        # 4 extra LM head inputs
        tb_graph.new_input(lm_norm_weight, (-1, -1, -1), 0, True)       # [24]
        tb_graph.new_input(lm_norm_scratch, (-1, -1, -1), 1, True)      # [25]
        tb_graph.new_input(lm_mxfp4_weight, (0, -1, -1), 1, True)       # [26]
        tb_graph.new_input(lm_bias, (1, -1, -1), 1, True)               # [27]
        # 11 base outputs (same as type 216)
        tb_graph.new_input(x_output, (-1, -1, -1), -1, True)            # [0]
        tb_graph.new_input(k_cache, (-1, -1, -1), -1, True)             # [1]
        tb_graph.new_input(v_cache, (-1, -1, -1), -1, True)             # [2]
        tb_graph.new_input(q_workspace, (-1, -1, -1), -1, True)         # [3]
        tb_graph.new_input(o_acc, (-1, -1, -1), -1, True)               # [4]
        tb_graph.new_input(attn_proj_out, (-1, -1, -1), -1, True)       # [5]
        tb_graph.new_input(topk_weight, (0, -1, -1), -1, True)          # [6]
        tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)     # [7]
        tb_graph.new_input(active_expert_ids, (-1, -1, -1), -1, True)   # [8]
        tb_graph.new_input(routing_weight_moe, (-1, -1, -1), -1, True)  # [9]
        tb_graph.new_input(moe_workspace_f32, (-1, -1, -1), -1, True)   # [10]
        # 2 extra LM head outputs
        tb_graph.new_input(lm_logits, (1, -1, -1), -1, True)            # [11]
        tb_graph.new_input(argmax_output, (-1, -1, -1), -1, True)       # [12]

        self.kn_graph.customized(
            [workspace_f32, residual, norm_weight_pre, norm_scratch_pre,
             qkv_weight, qkv_bias, sinks_or_placeholder, qkv_barrier, lse_acc,
             oproj_weight, oproj_bias, norm_weight_post, norm_scratch_post,
             router_weight, router_bias, logits_scratch, oproj_counters,
             gate_up_weight, down_weight, w13_bias, w2_bias,
             moe_barrier, swiglu_out, o_acc_f32,
             lm_norm_weight, lm_norm_scratch, lm_mxfp4_weight, lm_bias,
             x_output, k_cache, v_cache, q_workspace, o_acc,
             attn_proj_out, topk_weight, routing_indices,
             active_expert_ids, routing_weight_moe, moe_workspace_f32,
             lm_logits, argmax_output],
            tb_graph,
        )
        self.kn_graph.register_task(
            tb_graph, "gang_full_layer_with_lmhead_fused_mi300",
            [qkv_output_per_wg, qkv_n_wgs_per_xcd, total_qkv_tiles_per_xcd,
             actual_hidden_dim, head_dim, num_q_per_kv, self.page_size,
             kv_stride, q_ws_stride,
             self.max_seq_length, num_kv_chunks, q_workspace_stride,
             kv_cache_stride, num_kv_heads, sliding_window, has_sinks,
             oproj_output_per_wg, oproj_output_stride, total_oproj_tiles,
             num_experts, topk_k, router_tile_n, total_topk_tiles,
             oproj_tiles_per_xcd, moe_total_tiles_per_xcd,
             w13_output_per_wg, w2_output_per_wg,
             intermediate_size, workers_per_xcd,
             lm_output_per_wg, lm_n_wgs_per_xcd,
             lm_output_stride, actual_hidden_dim]
        )

    def gang_splitk_linear_res_bias_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        workspace: DTensor,
        bias: DTensor,
        output: DTensor,
        tile_n: int,
        output_stride: int,
        k_splits: int = 4,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang split-K linear with residual + fused bias_add in epilogue.
        5 inputs (input, weight, residual, workspace, bias), 1 output."""
        assert self.target_cc in (94, 95)
        batch_size = self.max_num_batched_tokens
        output_size = weight.dim(0)
        assert output_size % 8 == 0
        chunk_n = output_size // 8
        assert chunk_n % tile_n == 0
        n_tiles_per_xcd = chunk_n // tile_n
        reduction_size = weight.dim(1) if weight.num_dims == 2 else input.dim(1)
        assert reduction_size % k_splits == 0
        total_tiles = n_tiles_per_xcd * k_splits
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        tb_graph.new_input(workspace, (1, -1, -1), 1, True)
        tb_graph.new_input(bias, (1, -1, -1), 1, True)  # bias: partition dim 1 (columns) by bid.x
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, residual, workspace, bias, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "gang_splitk_linear_res_bias_mi300",
            [output_stride, tile_n, n_tiles_per_xcd, k_splits]
        )

    def linear_silu_layer(
        self, input, weight, output, output_stride,
        tile_n=64, m_tiles=1, wgm=0, block_dim=(256, 1, 1)):
        """CU-task linear with fused SiLU+mul. Same kernel as gang version."""
        assert input.num_dims == 2 and weight.num_dims == 2 and output.num_dims == 2
        assert self.target_cc in (94, 95)
        batch_size = self.max_num_batched_tokens
        gate_up_size = weight.dim(0)
        n_weight_tiles = gate_up_size // tile_n
        n_output_tiles = n_weight_tiles // 2
        m_per_tile = batch_size // m_tiles if m_tiles > 0 else batch_size
        total_tiles = n_output_tiles * m_tiles
        grid_dim = (total_tiles, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, -1, -1), 1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "linear_silu_mi300",
            [output_stride, tile_n, m_tiles, m_per_tile, total_tiles, n_output_tiles, wgm])

    def gang_linear_silu_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        tile_n: int,
        output_stride: int,
        m_tiles: int = 1,
        wgm: int = 0,
        block_dim: tuple = (256, 1, 1),
    ):
        """Gang linear with fused SiLU+mul.
        Weight is interleaved gate+up from shuffle_tensors(num_groups=G).
        Output is [bs, inter_size] (half of gate_up_size).

        Weight layout per XCD chunk: [gate_0(128), up_0(128), gate_1(128), up_1(128), ...]
        With tile_n=64: each 128-row block = 2 tiles, each group = 4 tiles.
        n_tiles_per_xcd counts OUTPUT tiles (= half of weight tiles per XCD).
        """
        assert input.num_dims == 2
        assert weight.num_dims == 2
        assert output.num_dims == 2
        assert self.target_cc in (94, 95), "Gang linear SiLU only supported on MI300X"
        batch_size = self.max_num_batched_tokens
        gate_up_size = weight.dim(0)
        assert gate_up_size % 8 == 0
        chunk_n_gateup = gate_up_size // 8  # weight rows per XCD
        # Each group = 4 tiles (2 gate + 2 up). Output tiles = half of weight tiles.
        n_weight_tiles = chunk_n_gateup // tile_n
        n_tiles_per_xcd = n_weight_tiles // 2  # output tiles (one per gate+up pair)
        assert batch_size % m_tiles == 0
        m_per_tile = batch_size // m_tiles
        total_tiles_per_xcd = n_tiles_per_xcd * m_tiles
        grid_dim = (8, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        # weight: partition dim 0 (rows) by bid.x → each XCD gets its interleaved chunk
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        # output: partition dim 1 (columns) by bid.x → each XCD writes to its column range
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        # params: [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
        #          n_tiles_per_xcd, wgm]
        self.kn_graph.register_task(
            tb_graph, "gang_linear_silu_mi300",
            [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
             n_tiles_per_xcd, wgm]
        )

    def splitk_reduce_layer(
        self,
        workspace: DTensor,
        residual: DTensor,
        output: DTensor,
        k_splits: int,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        assert workspace.num_dims == 2  # (k_splits * batch_size, hidden_size)
        assert residual.num_dims == 2   # (batch_size, hidden_size)
        assert output.num_dims == 2     # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(workspace, (1, -1, -1), 1, True)  # full K-splits visible
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([workspace, residual, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "splitk_reduce_mi300", [k_splits])

    def splitk_linear_res_atomic_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        workspace: DTensor,
        done_counter: DTensor,
        output: DTensor,
        k_splits: int,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Single-task split-K with float32 atomicAdd.
        grid_dim = (N_blocks, K_splits, 1)
        """
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # input [batch, K]: grid_dim.y partitions dim 1 (K)
        tb_graph.new_input(input, (-1, 1, -1), 1, True)
        # weight [N, K]: grid_dim.x partitions dim 0 (N), grid_dim.y partitions dim 1 (K)
        tb_graph.new_input(weight, (0, 1, -1), 1, True)
        # residual [batch, hidden]: grid_dim.x partitions dim 1 (N portion)
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        # workspace [batch, hidden] float32: grid_dim.x partitions dim 1, shared across K-splits
        tb_graph.new_input(workspace, (1, -1, -1), 1, True)
        # done_counter [n_blocks] int32: grid_dim.x partitions dim 0
        tb_graph.new_input(done_counter, (0, -1, -1), 1, True)
        # output [batch, hidden] bf16: grid_dim.x partitions dim 1
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized(
            [input, weight, residual, workspace, done_counter, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "splitk_linear_res_atomic_mi300", [k_splits])

    def linear_m_parallel_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Linear with M-dimension parallelism for L2 cache sharing.
        grid_dim = (N_tiles, M_tiles, 1): partitions both output columns and input rows.
        Workers on same XCD with different M-tiles share weight columns in L2.
        """
        assert input.num_dims == 2
        assert weight.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # input[batch, K]: dim 0 (batch) partitioned by grid_dim.y (M_tiles)
        tb_graph.new_input(input, (-1, 0, -1), 1, True)
        # weight[N, K]: dim 0 (N) partitioned by grid_dim.x (N_tiles)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        # output[batch, N]: dim 0 (batch) by grid_dim.y, dim 1 (N) by grid_dim.x
        tb_graph.new_input(output, (1, 0, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "linear")

    def linear_with_residual_m_parallel_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Linear + residual with M-dimension parallelism.
        grid_dim = (N_tiles, M_tiles, 1).
        """
        assert input.num_dims == 2
        assert weight.num_dims == 2
        assert residual.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, 0, -1), 1, True)      # batch by grid_dim.y
        tb_graph.new_input(weight, (0, -1, -1), 1, True)      # N by grid_dim.x
        tb_graph.new_input(residual, (1, 0, -1), 1, True)     # batch by grid_dim.y, N by grid_dim.x
        tb_graph.new_input(output, (1, 0, -1), -1, True)      # batch by grid_dim.y, N by grid_dim.x
        self.kn_graph.customized([input, weight, residual, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "linear_with_residual")

    def splitk_linear_res_atomic_m_parallel_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        workspace: DTensor,
        done_counter: DTensor,
        output: DTensor,
        k_splits: int,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Split-K atomic linear + residual with M-dimension parallelism.
        grid_dim = (N_tiles, K_splits, M_tiles).
        M-tiling via grid_dim.z enables L2 weight sharing between M-tiles.
        """
        assert input.num_dims == 2
        assert weight.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # input[batch, K]: dim 0 (batch) by grid_dim.z, dim 1 (K) by grid_dim.y
        tb_graph.new_input(input, (-1, 1, 0), 1, True)
        # weight[N, K]: dim 0 (N) by grid_dim.x, dim 1 (K) by grid_dim.y
        tb_graph.new_input(weight, (0, 1, -1), 1, True)
        # residual[batch, N]: dim 0 (batch) by grid_dim.z, dim 1 (N) by grid_dim.x
        tb_graph.new_input(residual, (1, -1, 0), 1, True)
        # workspace[batch, N] fp32: dim 0 (batch) by grid_dim.z, dim 1 (N) by grid_dim.x
        tb_graph.new_input(workspace, (1, -1, 0), 1, True)
        # done_counter[N_blocks * M_tiles]: dim 0 by grid_dim.x, dim 1(?) by grid_dim.z
        # Actually done_counter is indexed by bid.x only in the kernel.
        # With M-tiles, each M-tile needs its own counter set.
        # Use (0, -1, 2) to partition dim 0 by grid_dim.x and dim 2 by grid_dim.z
        # But done_counter is 2D: (n_blocks, m_tiles) or (n_blocks * m_tiles, 1)
        # Simplest: make done_counter (n_blocks * m_tiles, 1), partition by (0, -1, -1)
        # and let the kernel compute the right index.
        # done_counter[n_blocks, m_tiles]: dim 0 by grid_dim.x, dim 1 by grid_dim.z
        tb_graph.new_input(done_counter, (0, -1, 1), 1, True)
        # output[batch, N]: dim 0 by grid_dim.z, dim 1 by grid_dim.x
        tb_graph.new_input(output, (1, -1, 0), -1, True)
        self.kn_graph.customized(
            [input, weight, residual, workspace, done_counter, output], tb_graph)
        self.kn_graph.register_task(
            tb_graph, "splitk_linear_res_atomic_mi300", [k_splits])

    def linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, output], tb_graph)

        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "linear_sm100")
        elif self.target_cc == 90:
            if weight.dim(0) // grid_dim[0] <= 64:
                self.kn_graph.register_task(tb_graph, "linear_swapAB_hopper")
                # self.kn_graph.register_task(tb_graph, "linear_cutlass_hopper")
            else:
                self.kn_graph.register_task(tb_graph, "linear_swapAB_hopper")
        elif self.target_cc == 80 or self.target_cc in (94, 95):
            # 94: MI300/ROCm – use sm_80-style "linear" (base PTX path)
            self.kn_graph.register_task(tb_graph, "linear")
        else:
            self._raise_unsupported_target_cc("linear_layer", ["linear_sm100", "linear_swapAB_hopper", "linear"])

    def linear_with_residual_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert residual.num_dims == 2  # (batch_size, hidden_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), -1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, weight, residual, output], tb_graph)
        
        if self.target_cc == 100:
            self.kn_graph.register_task(tb_graph, "linear_with_residual_sm100")
        elif self.target_cc == 90:
            if weight.dim(0) // grid_dim[0] <= 64:
                # self.kn_graph.register_task(tb_graph, "linear_cutlass_with_residual_hopper")
                self.kn_graph.register_task(tb_graph, "linear_swapAB_with_residual_hopper")
            else:
                self.kn_graph.register_task(tb_graph, "linear_swapAB_with_residual_hopper")
        elif self.target_cc == 80 or self.target_cc in (94, 95):
            # 94: MI300/ROCm – use sm_80-style "linear_with_residual"
            self.kn_graph.register_task(tb_graph, "linear_with_residual")
        else:
            self._raise_unsupported_target_cc(
                "linear_with_residual_layer",
                ["linear_with_residual_sm100", "linear_swapAB_with_residual_hopper", "linear_with_residual"],
            )

    def allreduce_layer(
        self,
        input: DTensor,
        buffer: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size)
        assert buffer.num_dims == 3  # (world_size, batch_size, hidden_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        # params[0]: num_gpus
        # params[1]: my_gpu_id
        params = [self.world_size, self.mpi_rank]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (1, -1, -1), -1, True)
        tb_graph.new_input(buffer, (2, -1, -1), -1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, buffer, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "allreduce", params)

    def bias_add_layer(
        self,
        input: DTensor,
        bias: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        """Element-wise bias add: output = input + bias (broadcast across batch)."""
        assert input.num_dims == 2  # (batch_size, size)
        assert bias.num_dims == 2   # (1, size) - pre-unsqueezed
        assert output.num_dims == 2 # (batch_size, size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (1, -1, -1), 1, True)
        tb_graph.new_input(bias, (-1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), 1, True)
        self.kn_graph.customized([input, bias, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "bias_add_mi300")

    def silu_mul_layer(
        self,
        input: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2 # (batch_size, 2 * intermediate_size)
        assert output.num_dims == 2 # (batch_size, intermediate_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), 1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "silu_mul" if self.target_cc == 90 else "silu_mul")

    def identity_layer(
        self,
        input: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
        dependent_tensor: DTensor = None,
    ):
        # TODO: Add support from kn_graph
        last_dim = 0
        assert input.num_dims == output.num_dims
        for i in range(input.num_dims):
            assert input.dim(i) == output.dim(i)
            last_dim = i
        assert last_dim == 1 or last_dim == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (last_dim, -1, -1), 1, True)
        tb_graph.new_input(output, (last_dim, -1, -1), 1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "identity")

    def silu_mul_linear_with_residual_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, 2*intermediate_size)
        assert weight.num_dims == 2  # (hidden_size, intermediate_size)
        assert residual.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), 1, True)
        self.kn_graph.customized([input, weight, residual, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "silu_mul_linear_with_residual")

    def argmax_layer(
        self, input: DTensor, output: DTensor, grid_dim: tuple, block_dim: tuple
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, vocab_size)
        assert output.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "argmax")

    def argmax_partial_layer(
        self,
        input: DTensor,
        output: tuple[DTensor, DTensor],
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, vocab_size)
        assert len(output) == 2
        output_value, output_index = output
        assert output_value.num_dims == 2  # (batch_size, num_tasks)
        assert output_index.num_dims == 2  # (batch_size, num_tasks)
        num_tasks = grid_dim[0]
        self.argmax_partial_output_size = input.dim(1) // num_tasks
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (1, 0, -1), -1, True)
        tb_graph.new_input(output_value, (1, 0, -1), -1, True)
        tb_graph.new_input(output_index, (1, 0, -1), -1, True)
        self.kn_graph.customized([input, output_value, output_index], tb_graph)
        if self.target_cc == 100 or self.target_cc == 90:
            self.kn_graph.register_task(tb_graph, "argmax_partial_sm100", [num_tasks])
        else:
            self.kn_graph.register_task(tb_graph, "argmax_partial", [num_tasks])

    def argmax_reduce_layer(
        self,
        input: tuple[DTensor, DTensor],
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert len(input) == 2
        input_value, input_index = input
        assert input_value.num_dims == 2  # (batch_size, num_tasks)
        assert input_index.num_dims == 2  # (batch_size, num_tasks)
        assert output.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_value, (1, 0, -1), -1, True)
        tb_graph.new_input(input_index, (1, 0, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True) #TODO: Make sure the output map is expected
        self.kn_graph.customized([input_value, input_index, output], tb_graph)
        if self.target_cc == 100:
            self.kn_graph.register_task(
                tb_graph, "argmax_reduce_sm100", [self.argmax_partial_output_size]
            )
        else:
            self.kn_graph.register_task(
                tb_graph, "argmax_reduce", [self.argmax_partial_output_size]
            )

    def sampling_sm100_layer(
        self,
        logits: DTensor,      # [batch_size, vocab_size]
        output: DTensor,      # [batch_size, 1]
        grid_dim: tuple,
        block_dim: tuple,
        seed: int = 42,
    ):
        """Sampling from logits using Gumbel-Max trick for stochastic token generation."""
        assert logits.num_dims == 2      # (batch_size, vocab_size)
        assert output.num_dims == 2      # (batch_size, 1)

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(logits, (0, -1, -1), -1, True)
        tb_graph.new_input(output, (0, -1, -1), -1, True)
        self.kn_graph.customized([logits, output], tb_graph)

        # Register task with seed parameter
        self.kn_graph.register_task(tb_graph, "sampling_sm100", [seed])

    def find_ngram_partial_layer(
        self, input: DTensor, output: DTensor, grid_dim: tuple, block_dim: tuple, ngram_size: int = 3):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, seq_len)
        assert output.num_dims == 2  # (batch_size, num_tasks)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "find_ngram_partial", [ngram_size])
        
    def find_ngram_global_layer(
        self, input: tuple[DTensor, DTensor], output: DTensor, grid_dim: tuple, block_dim: tuple, ngram_size: int = 3, spec_length: int = 5):
        # Currently assume that input/output
        assert len(input) == 2
        partial_results, tokens = input
        assert partial_results.num_dims == 2  # (batch_size, num_tasks)
        assert tokens.num_dims == 2  # (batch_size, vocab_size)
        assert output.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(partial_results, (-1, -1, -1), -1, True)
        tb_graph.new_input(tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([partial_results, tokens, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "find_ngram_global", [ngram_size, spec_length])

    def prompt_lookup_spec_handler(
        self, 
        spec_decode_config: PromptLookupConfig,
        tokens: DTensor,
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
    ):
        partial_ngram_output = self.new_tensor(
            dims=(tokens.dim(0), 96),
            dtype=int64,
            name="partial_ngram_output",
            io_category="cuda_tensor",
        )
        self.find_ngram_partial_layer(
            input=tokens, 
            output=partial_ngram_output, 
            grid_dim=grid_dim, 
            block_dim=block_dim, 
            ngram_size=spec_decode_config.ngram_size
        )
        spec_tokens = self.new_tensor(
            dims=(tokens.dim(0), spec_decode_config.spec_length + 1),
            dtype=int64,
            name="spec_tokens",
            io_category="cuda_tensor",
        )   
        self.find_ngram_global_layer(
            input=(partial_ngram_output, tokens), 
            output=spec_tokens, 
            grid_dim=(1, 1, 1), 
            block_dim=(128, 1, 1), 
            ngram_size=spec_decode_config.ngram_size,
            spec_length=spec_decode_config.spec_length
        )
        return spec_tokens
    
    def draft_forward_layer_dispatcher(
        self,
        spec_decode_config: SpecDecodeConfig,
        tokens: DTensor,
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
    ):
        method = spec_decode_config.method
        handler = self._spec_decode_handlers[method]
        if handler is None:
            raise ValueError(f"Invalid spec decode method: {method}")
        return handler(spec_decode_config, tokens, grid_dim, block_dim)
    
    def target_verify_greedy_layer(
        self, input: tuple[DTensor, DTensor], output: DTensor, grid_dim: tuple, block_dim: tuple):
        # Currently assume that input/output
        # This tensor is not realy used
        assert len(input) == 2
        spec_tokens, target_tokens = input
        assert spec_tokens.num_dims == 2  # (batch_size, vocab_size)
        assert target_tokens.num_dims == 2  # (batch_size, vocab_size)
        assert output.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(spec_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(target_tokens, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        self.kn_graph.customized([spec_tokens, target_tokens, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "target_verify_greedy")
        
    def prompt_lookup_verify_handler(
        self,
        spec_decode_config: SpecDecodeConfig,
        spec_tokens: DTensor,
        target_output: DTensor,
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
    ):
        # This tensor is not realy used
        verify_out = self.new_tensor(
            dims=(1, 1),
            dtype=int64,
            name="verify_out",
            io_category="cuda_tensor",
        )
        self.target_verify_greedy_layer(
            input=(spec_tokens, target_output), output=verify_out, grid_dim=grid_dim, block_dim=block_dim
        )
        return verify_out
    
    def verify_layer_dispatcher(
        self,
        spec_decode_config: SpecDecodeConfig,
        spec_tokens: DTensor,
        target_output: DTensor,
        grid_dim: tuple[int, int, int] = (1, 1, 1),
        block_dim: tuple[int, int, int] = (128, 1, 1),
    ):
        method = spec_decode_config.method
        handler = self._spec_verify_handlers[method]
        if handler is None:
            raise ValueError(f"Invalid spec decode method: {method}")
        return handler(spec_decode_config, spec_tokens, target_output, grid_dim, block_dim)

    def compile(
        self,
        **kwargs,
    ):
        assert not self._is_compiled
        
        output_dir = kwargs.get("output_dir", None)

        MIRAGE_ROOT, INCLUDE_PATH, DEPS_PATH = get_key_paths()
        if self.mode == "online_notoken" or self.mode == "online" or self.mode == "multi_turn":
            # We will init for multiple times so the output directory should be permanent
            tempdir = "./permanent_output_dir/"
        else:
            tempdir_obj = tempfile.TemporaryDirectory()
            #tempdir = tempdir_obj.name
            tempdir = "./permanent_output_dir/"
        os.makedirs(tempdir, exist_ok=True)
        results = self.kn_graph.generate_task_graph(num_gpus=self.world_size, my_gpu_id=self.mpi_rank)

        cuda_code_path = os.path.join(tempdir, "test.cu")
        so_path = os.path.join(tempdir, "test.cpython-38-x86_64-linux-gnu.so")
        # check json file
        json_file_path = os.path.join(tempdir, "task_graph.json")
        # build if files are not exist
            
        with open(json_file_path, "w") as f:
            f.write(results["json_file"])

        # Event fusion DISABLED to match NVIDIA implementation
        # Original event fusion code (AMD only) reduced events but didn't improve performance
        # if self.target_cc in (94, 95):  # AMD MI300
        #     from .event_fusion import fuse_events
        #     import json
        #     with open(json_file_path, "r") as f:
        #         task_graph = json.load(f)
        #     original_events = len(task_graph['all_events'])
        #     fused_graph = fuse_events(task_graph)
        #     fused_events = len(fused_graph['all_events'])
        #     print(f"Event fusion: {original_events} -> {fused_events} events ({(1-fused_events/original_events)*100:.1f}% reduction)")
        #     with open(json_file_path, "w") as f:
        #         json.dump(fused_graph, f)

        with open(cuda_code_path, "w") as f:
            f.write(results["cuda_code"] + HARD_CODE)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(cuda_code_path, os.path.join(output_dir, f"test_rank{self.mpi_rank}.cu"))
            shutil.copy(json_file_path, os.path.join(output_dir, f"task_graph_rank{self.mpi_rank}.json"))

        if self.target_cc in (94, 95):
            rocm_home = os.environ.get("ROCM_PATH", "/opt/rocm")
            cc = shutil.which("hipcc") or os.path.join(rocm_home, "bin", "hipcc")
            if not cc or not os.path.isfile(cc):
                raise RuntimeError(
                    "hipcc not found. For MI300/ROCm builds set ROCM_PATH or ensure hipcc is on PATH."
                )
        else:
            cc = shutil.which("nvcc")
            if cc is None:
                raise RuntimeError(
                    "nvcc not found. Please make sure you have installed CUDA."
                )
        # This function was renamed and made public in Python 3.10
        if hasattr(sysconfig, "get_default_scheme"):
            scheme = sysconfig.get_default_scheme()
        else:
            scheme = sysconfig._get_default_scheme()
        # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
        # path changes to include 'local'. This change is required to use triton with system-wide python.
        if scheme == "posix_local":
            scheme = "posix_prefix"
        py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

        # find mirage home
        if "MIRAGE_HOME" in os.environ:
            MIRAGE_HOME_PATH = os.environ.get("MIRAGE_HOME")
        else:
            raise RuntimeError(
                "MIRAGE_HOME unspecified; Please set MIRAGE_HOME to be the root of the Mirage folder"
            )

        NVSHMEM_INC_PATH = None
        NVSHMEM_LIB_PATH = None
        MPI_INC_PATH = None
        MPI_LIB_PATH = None
        if self.use_nvshmem:
            # find nvshmem include folder and library folder
            if "NVSHMEM_INC_PATH" in os.environ:
                NVSHMEM_INC_PATH = os.environ.get("NVSHMEM_INC_PATH")
                header_file_path = os.path.join(NVSHMEM_INC_PATH, "nvshmem.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        "Environment variable NVSHMEM_INC_PATH is set but cannot find nvshmem.h at {header_file_path}"
                    )
            else:
                NVSHMEM_INC_PATH = "/usr/include/nvshmem_12/"
                header_file_path = os.path.join(NVSHMEM_INC_PATH, "nvshmem.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        "Cannot find nvshmem.h, please set environment variable NVSHMEM_INC_PATH"
                    )
            # find nvshmem shared library
            if "NVSHMEM_LIB_PATH" in os.environ:
                NVSHMEM_LIB_PATH = os.environ.get("NVSHMEM_LIB_PATH")
                lib_file_path = os.path.join(NVSHMEM_LIB_PATH, "libnvshmem.a")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        "Environment variable NVSHMEM_LIB_PATH is set but cannot find libnvshmem.a at {lib_file_path}"
                    )
            else:
                NVSHMEM_LIB_PATH = "/usr/lib/x86_64-linux-gnu/"
                lib_file_path = os.path.join(NVSHMEM_LIB_PATH, "libnvshmem.a")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        "Cannot find libnvshmem.a, please set environment variable NVSHMEM_LIB_PATH"
                    )
            # find mpi include foler
            if "MPI_INC_PATH" in os.environ:
                MPI_INC_PATH = os.environ.get("MPI_INC_PATH")
                header_file_path = os.path.join(MPI_INC_PATH, "mpi.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        f"Environment variable MPI_INC_PATH is set but cannot find mpi.h at {header_file_path}"
                    )
            else:
                MPI_INC_PATH = "/usr/include/"
                header_file_path = os.path.join(MPI_INC_PATH, "mpi.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        f"Cannot find mpi.h, please set environment variable MPI_INC_PATH"
                    )
            # find mpi shared library
            if "MPI_LIB_PATH" in os.environ:
                MPI_LIB_PATH = os.environ.get("MPI_LIB_PATH")
                lib_file_path = os.path.join(MPI_LIB_PATH, "libmpi.so")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        f"Environment variable MPI_LIB_PATH is set but cannot find libmpi.so at {lib_file_path}"
                    )
            else:
                NVSHMEM_LIB_PATH = "/usr/lib/"
                lib_file_path = os.path.join(MPI_LIB_PATH, "libmpi.so")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        f"Cannot find libmpi.so, please set environment variable MPI_LIB_PATH"
                    )

        cc_cmd = get_compile_command(
            mpk=self,
            target_cc=self.target_cc,
            cc=cc,
            file_name=cuda_code_path,
            py_include_dir=py_include_dir,
            mirage_home_path=MIRAGE_HOME_PATH,
            mirage_inc_path=INCLUDE_PATH,
            mirage_deps_path=DEPS_PATH,
            nvshmem_inc_path=NVSHMEM_INC_PATH,
            nvshmem_lib_path=NVSHMEM_LIB_PATH,
            mpi_inc_path=MPI_INC_PATH,
            mpi_lib_path=MPI_LIB_PATH,
            py_so_path=so_path,
            profiling=True if self.profiler_tensor is not None else False,
            use_nvshmem=self.use_nvshmem,
            num_workers=self.num_workers,
            num_local_schedulers=self.num_local_schedulers, 
            num_remote_schedulers=self.num_remote_schedulers,
            use_cutlass_kernel=self.use_cutlass_kernel,
        )
        print("Compiling megakernel using the following command line:")
        print(cc_cmd)
        subprocess.check_call(cc_cmd)

        import importlib.util

        spec = importlib.util.spec_from_file_location("__mirage_launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.init_func = getattr(mod, "init_func")
        self.launch_func = getattr(mod, "launch_func")
        self.init_request_func = getattr(mod, "init_request_func")
        self.finalize_func = getattr(mod, "finalize_func")
        self._set_rope_tables_func = getattr(mod, "set_rope_tables_func", None)
        print("Finished megakernel compilation...")

        #meta_tensors_ptr = [tensor.data_ptr() for tensor in self.meta_tensors]
        meta_tensors = list()
        meta_tensors.append(self.meta_tensors["step"])
        meta_tensors.append(self.meta_tensors["tokens"])
        meta_tensors.append(self.meta_tensors["input_tokens"])
        meta_tensors.append(self.meta_tensors["output_tokens"])
        meta_tensors.append(self.meta_tensors["num_new_tokens"])
        meta_tensors.append(self.meta_tensors["prompt_lengths"])
        meta_tensors.append(self.meta_tensors["qo_indptr_buffer"])
        meta_tensors.append(self.meta_tensors["paged_kv_indptr_buffer"])
        meta_tensors.append(self.meta_tensors["paged_kv_indices_buffer"])
        meta_tensors.append(self.meta_tensors["paged_kv_last_page_len_buffer"])
        meta_tensors_ptr = [tensor.data_ptr() for tensor in meta_tensors]
        profiler_buffer_ptr = (
            self.profiler_tensor.data_ptr() if self.profiler_tensor is not None else 0
        )
        self.eos_token_id = kwargs.get("eos_token_id", self.eos_token_id)
        self.init_func(
            meta_tensors_ptr,
            profiler_buffer_ptr,
            self.mpi_rank,
            self.num_workers,
            self.num_local_schedulers,
            self.num_remote_schedulers,
            self.max_seq_length,
            self.total_num_requests,
            self.eos_token_id,
        )

        self._is_compiled = True

        # self.call_func = getattr(mod, "call_func")

    def set_rope_tables(self, cos_tensor: "torch.Tensor", sin_tensor: "torch.Tensor"):
        """Set RoPE cos/sin tables in RuntimeConfig (call after compile)."""
        assert self._is_compiled, "Must call compile() before set_rope_tables()"
        assert self._set_rope_tables_func is not None
        self._set_rope_tables_func(cos_tensor.data_ptr(), sin_tensor.data_ptr())

    def __call__(self, **kwargs):
        stream = kwargs.get("default_stream", None)
        if stream is None:
           stream = torch.cuda.current_stream()
        # Convert torch.cuda.Stream to raw pointer (integer) for the C launcher
        stream_ptr = 0
        if hasattr(stream, "cuda_stream"):
            try:
                stream_ptr = int(stream.cuda_stream)
            except Exception:
                try:
                    stream_ptr = int(stream.cuda_stream.value)
                except Exception as e:
                    raise ValueError(f"Invalid stream object: {stream} is of type {type(stream)}: {e}")
        elif isinstance(stream, int):
            stream_ptr = stream
        else:
            raise ValueError("Invalid stream object")
        self.launch_func(stream_ptr)
        if self.profiler_tensor is not None:
            from .profiler_persistent import export_to_perfetto_trace
            
            if self.trace_name:
                trace_name = self.trace_name + ".perfetto-trace"
            else:
                trace_name = f"mirage_{self.mpi_rank}.perfetto-trace"

            export_to_perfetto_trace(
                self.profiler_tensor, trace_name
            )
            # Also save raw profiler tensor for programmatic analysis
            raw_path = trace_name.replace(".perfetto-trace", ".pt")
            torch.save(self.profiler_tensor.cpu(), raw_path)
            print(f"Saved raw profiler tensor to {raw_path}")

    def __del__(self):
        if not self.__finalized__:
            self.finalize()

    def finalize(self):
        assert not self.__finalized__
        if self._is_compiled:
            self.finalize_func()
        self.__finalized__ = True
