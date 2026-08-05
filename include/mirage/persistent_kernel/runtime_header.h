/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "mirage/config.h"
// Include platform definition first - MUST be before any HIP headers
#include "mirage/hip_platform.h"
// Don't redefine MIRAGE_BACKEND_USE_ROCM if already defined via compiler flags
#ifdef MIRAGE_BACKEND_USE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#else
#include <cuda_runtime.h>
#endif

namespace mirage {
namespace runtime {

#if defined(MIRAGE_GRACE_HOPPER) || defined(MIRAGE_GRACE_BLACKWELL)
constexpr int WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE = 6 * 1024;
#else
constexpr int WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE = 3 * 1024;
#endif

// Layer index at fixed shared memory offset (last 4 bytes of extern
// __shared__). Used by MoE barriers for monotonically increasing expected
// values.
constexpr int LAYER_IDX_SMEM_OFFSET_FROM_END = 4;

// AMD LDS limits per GPU generation:
// - MI300 (gfx942): 64KB LDS per workgroup, use conservative 60KB
// - MI350 (gfx950): 160KB LDS per workgroup, use 155KB for LDS-resident weights
#if (MPK_TARGET_CC == 95)
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    155 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#elif defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300) ||            \
    (MPK_TARGET_CC == 94)
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    60 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#elif defined(MODE_ONLINE_NOTOKEN) || defined(MODE_MULTI_TURN)
// Have to be smaller for vllm compatibility, or program will stuck
#if MPK_TARGET_CC >= 90
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    220 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#elif MPK_TARGET_CC >= 86
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    99 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#elif MPK_TARGET_CC >= 80
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    160 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#else
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    163 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#endif
#else
#if MPK_TARGET_CC >= 90
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    225 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#elif MPK_TARGET_CC >= 86
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    99 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#elif MPK_TARGET_CC >= 80
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    163 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#else
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    163 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#endif
#endif

typedef unsigned long long int TaskId;
unsigned long long int const TASK_INVALID_ID = 0x7fffffffffffffff;
// Task IDs are 64-bit values encoding both the current iteration of the task
// and its index TASK: iteration id: 32, task index: 32
typedef unsigned long long int EventId;
// Event IDs are 64-bit values encoding both the owner of the event and its
// index EVENT: nvshmem_tag: 16, owner_node: 16, event_idx: 32
unsigned long long int const EVENT_NVSHMEM_TAG = 0x1e00000000000000;
unsigned long long int const EVENT_INVALID_ID = 0x7ffffffffffffffe;
typedef unsigned long long int EventCounter;

int const MAX_INPUTS_PER_TASK = 28;
int const MAX_OUTPUTS_PER_TASK = 13;

// Nil-address tripwire buffer geometry (see MPK_NIL_TRIPWIRE).
// Per worker: [0] layer  [1] outer phase  [2] tile  [3] input_ptrs[0]
//             [4] fused-kernel sub-phase  [5] sub-phase aux value
#define MPK_TW_HDR 4
#define MPK_TW_PER_WORKER 8
#define MPK_TW_SLOTS (MPK_TW_HDR + 256 * MPK_TW_PER_WORKER)
// Increased to 304 to support full CU utilization on AMD MI300X (304 CUs)
// and NVIDIA Blackwell (160+ SMs which uses 144 workers)
int const MAX_NUM_WORKERS = 304;

enum TaskType {
  TASK_TERMINATE = 0,
  TASK_BEGIN_TASK_GRAPH = 10,
  // compute task starts from 100
  TASK_EMBEDDING = 101,
  TASK_RMS_NORM_LINEAR = 102,
  TASK_ATTENTION_1 = 103,
  TASK_ATTENTION_2 = 104,
  TASK_SILU_MUL_LINEAR_WITH_RESIDUAL = 105,
  TASK_ALLREDUCE = 106,
  TASK_REDUCE = 107,
  TASK_LINEAR_WITH_RESIDUAL = 108,
  TASK_ARGMAX = 109,
  TASK_ARGMAX_PARTIAL = 110,
  TASK_ARGMAX_REDUCE = 111,
  TASK_FIND_NGRAM_PARTIAL = 112,
  TASK_FIND_NGRAM_GLOBAL = 113,
  TASK_TARGET_VERIFY_GREEDY = 114,
  TASK_SINGLE_BATCH_EXTEND_ATTENTION = 115,
  TASK_PAGED_ATTENTION_1 = 116,
  TASK_PAGED_ATTENTION_2 = 117,
  TASK_SILU_MUL = 118,
  TASK_RMS_NORM = 119,
  TASK_LINEAR = 120,
  TASK_IDENTITY = 121,
  // MI300 Tasks
  TASK_SPLITK_LINEAR_MI300 = 129,
  TASK_PAGED_ATTENTION_SPLIT_KV_MI300 = 130,
  TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_MI300 = 131,
  TASK_SPLITK_REDUCE_MI300 = 132,
  TASK_SPLITK_LINEAR_RES_ATOMIC_MI300 = 133,
  TASK_KV_PREP_MI300 = 134,
  TASK_PAGED_ATTENTION_CK_MI300 = 135,
  TASK_GANG_LINEAR_MI300 = 136,
  TASK_GANG_LINEAR_RES_MI300 = 137,
  TASK_GANG_ATTN_SPLIT_KV_MI300 = 138,
  TASK_GANG_ATTN_MERGE_MI300 = 139,
  TASK_KV_CACHE_UPDATE_MI300 = 140,
  TASK_PAGED_ATTENTION_CK_FMHA_SPLIT_KV_MI300 = 141,
  TASK_GANG_LINEAR_SILU_MI300 = 142,
  TASK_LINEAR_SILU_MI300 = 182,
  TASK_GANG_RMS_NORM_MI300 = 143,
  TASK_GANG_SPLITK_LINEAR_RES_MI300 = 144,
  TASK_GANG_KSPLIT_GEMM_MI300 = 145,
  TASK_GANG_KSPLIT_FINALIZE_MI300 = 146,
  // MI300/MI350 MoE Tasks
  TASK_MOE_W13_LINEAR_MI300 = 170,
  TASK_MOE_W2_LINEAR_MI300 = 171,
  TASK_MOE_TOPK_SOFTMAX_MI300 = 172,
  TASK_MOE_MUL_SUM_ADD_MI300 = 173,
  TASK_GANG_MOE_W13_LINEAR_MI300 = 174,
  TASK_GANG_MOE_W2_LINEAR_MI300 = 175,
  TASK_SWIGLUOAI_MI300 = 176,
  TASK_MOE_W13_LINEAR_MXFP4_MI300 = 177,
  TASK_MOE_W2_LINEAR_MXFP4_MI300 = 178,
  TASK_ATTENTION_SINK_MI300 = 179,
  TASK_BIAS_ADD_MI300 = 180,
  TASK_MOE_W13_LINEAR_MXFP4_CK_MI300 = 183,
  TASK_MOE_W2_LINEAR_MXFP4_CK_MI300 = 184,
  TASK_GANG_MOE_W13_LINEAR_MXFP4_MI300 = 185,
  TASK_GANG_MOE_W2_LINEAR_MXFP4_MI300 = 186,
  TASK_GANG_MOE_FUSED_MXFP4_MI300 = 187,
  TASK_GANG_MOE_SWIGLU_W2_MXFP4_MI300 = 188,
  TASK_GANG_MOE_W13_SWIGLU_MXFP4_MI300 = 189,
  TASK_GANG_LINEAR_BIAS_MI300 = 190,
  TASK_GANG_SPLITK_LINEAR_RES_BIAS_MI300 = 191,
  TASK_GANG_RMSNORM_LINEAR_BIAS_MI300 = 192,
  TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_MI300 = 193,
  TASK_GANG_LINEAR_MXFP4_RES_BIAS_MI300 = 194,
  TASK_GANG_MULSUMRADD_RMSNORM_LINEAR_MXFP4_BIAS_MI300 = 195,
  TASK_GANG_RMSNORM_LINEAR_BIAS_TOPK_MI300 = 196,
  TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300 = 197,
  TASK_GANG_MULSUMRADD_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300 = 198,
  TASK_GANG_RESADDF32_RMSNORM_LINEAR_MXFP4_BIAS_MI300 = 210,
  TASK_GANG_RESADDF32_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300 = 211,
  TASK_MOE_RESIDUAL_ADD_F32_MI300 = 212,
  TASK_GANG_LINEAR_MXFP4_RES_BIAS_RMSNORM_TOPK_MI300 = 213,
  TASK_GANG_QKV_ATTN_FUSED_MI300 = 214,
  TASK_GANG_OPROJ_TOPK_MOE_FUSED_MI300 = 215,
  TASK_GANG_FULL_LAYER_FUSED_MI300 = 216,
  TASK_GANG_FULL_LAYER_WITH_LMHEAD_FUSED_MI300 = 217,
  TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_ARGMAX_MI300 = 218,
  // Hopper Tasks
  TASK_HOPPER_TASK_BEGIN = 150, // Hopper start placeholder, not a real task
  TASK_LINEAR_WITH_RESIDUAL_HOPPER = 151,
  TASK_LINEAR_HOPPER = 152,
  TASK_PAGED_ATTENTION_HOPPER = 153,
  TASK_RMS_NORM_HOPPER = 154,
  TASK_LINEAR_SWAPAB_HOPPER = 155,
  TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER = 156,
  TASK_LINEAR_CUTLASS_HOPPER = 157,
  TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER = 158,
  TASK_SILU_MUL_HOPPER = 159,
  TASK_EMBEDDING_HOPPER = 160,
  TASK_MOE_W13_LINEAR_SM90 = 161,
  TASK_MOE_W2_LINEAR_SM90 = 162,
  TASK_SPLITK_LINEAR_SWAPAB_HOPPER = 163,
  TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER = 164,
  TASK_HOPPER_TASK_END = 198, // Hopper end placeholder, not a real task
  // SM100 Tasks
  TASK_SM100_TASK_BEGIN = 230, // SM100 start placeholder, not a real task
  TASK_SM100_TMA_START_TASK = 231,
  TASK_SPLITK_LINEAR_SM100 = 251,
  TASK_LINEAR_WITH_RESIDUAL_SM100 = 252,
  TASK_LINEAR_SM100 = 253,
  TASK_MOE_W13_LINEAR_SM100 = 254,
  TASK_MOE_W2_LINEAR_SM100 = 255,
  TASK_SM100_TMA_END_TASK = 256,
  TASK_ATTN_SM100 = 257,
  TASK_ARGMAX_REDUCE_SM100 = 258,
  TASK_ARGMAX_PARTIAL_SM100 = 259,
  TASK_MOE_TOPK_SOFTMAX_SM100 = 260,
  TASK_MOE_MUL_SUM_ADD_SM100 = 261,
  TASK_TENSOR_INIT = 262,
  TASK_PAGED_ATTENTION_SPLIT_KV_SM100 = 263,
  TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100 = 264,
  TASK_SAMPLING_SM100 = 265,
  TASK_SM100_TASK_END = 298, // SM100 end placeholder, not a real task
  TASK_NVSHMEM_COPY = 199,
  TASK_SCHD_TASKS = 200,
  TASK_SCHD_EVENTS = 201,
  TASK_GET_EVENT = 202,
  TASK_GET_NEXT_TASK = 203,
};

enum EventType {
  EVENT_EMPTY = 900,
  EVENT_LAUNCH_TASKS = 901,
  EVENT_LAUNCH_MASSIVE_TASKS = 902,
  EVENT_LAUNCH_DEPENDENT_TASKS = 903,
  EVENT_END_OF_TASK_GRAPH = 910,
  EVENT_TERMINATION = 911,
  EVENT_INVALID = 999,
};

struct TensorDesc {
  int num_dims;
  void *base_ptr;
#ifdef MPK_ENABLE_TMA
  void *tma_desc_ptrs[mirage::config::MAX_TMA_DESC_PER_TENSOR];
#endif
  int data_type;
  int dim[mirage::config::MAX_TENSOR_DIMS];
  int stride[mirage::config::MAX_TENSOR_DIMS];
};

struct EventDesc {
  EventDesc(void)
      : event_type(EVENT_INVALID), num_triggers(0),
        first_task_id(TASK_INVALID_ID), last_task_id(TASK_INVALID_ID) {}
  EventDesc(EventType type, int nt, TaskId f, TaskId l)
      : event_type(type), num_triggers(nt), first_task_id(f), last_task_id(l) {}
  EventType event_type;
  int num_triggers;
  TaskId first_task_id, last_task_id;
};

struct FullTaskDesc {
  FullTaskDesc(TaskType t, int _variant_id)
      : task_type(t), variant_id(_variant_id), num_inputs(0), num_outputs(0),
        trigger_event(EVENT_INVALID_ID), dependent_event(EVENT_INVALID_ID) {
    task_metadata.raw_payload = ~0ull;
  }
  FullTaskDesc() {
    task_metadata.raw_payload = ~0ull;
  }
  TaskType task_type;
  unsigned variant_id;
  int num_inputs, num_outputs;
  EventId trigger_event;
  EventId dependent_event;
  TensorDesc inputs[MAX_INPUTS_PER_TASK];
  TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
  union TaskMetadata {
    struct {
      int expert_offset; // Used for MoE
    };
    struct {
      int16_t request_id;    // Used for paged attention
      uint16_t kv_idx;       // Used for paged attention split kv
      int merge_task_offset; // Used for paged attention split kv merge
    };
    struct {
      size_t xfer_size_in_bytes; // Used for nvshmem
    };
    struct {
      uint16_t n_tile_start; // Used for LINEAR N-tile parallelization
      uint16_t n_tile_count; // 0 means all remaining tiles
      int32_t _linear_reserved;
    };
    unsigned long long raw_payload;
  } task_metadata;
};

static_assert(
    sizeof(FullTaskDesc::TaskMetadata) == sizeof(unsigned long long),
    "FullTaskDesc::TaskMetadata layout changed; update raw_payload type.");

struct alignas(16) TaskDesc {
  TaskDesc(FullTaskDesc t)
      : task_type(t.task_type), variant_id(t.variant_id),
        trigger_event(t.trigger_event),
        dependent_event(t.dependent_event), input_ptrs{}, output_ptrs{},
#ifdef MPK_ENABLE_TMA
        input_tma_desc_ptrs{}, output_tma_desc_ptrs{},
#endif
        task_metadata(t.task_metadata) {
    // The pointer arrays are value-initialized above, not just the first
    // num_inputs/num_outputs entries. The whole TaskDesc is memcpy'd to the
    // GPU and, for the fused layer path, copied into a *reused* shared-memory
    // slot -- so any slot this constructor skips carried indeterminate host
    // stack bytes into device memory, and on the device it aliases whatever
    // the previous task left behind. Consumers index by slot, not by count
    // (see the ml pointer-table build in persistent_kernel.cuh, which snapshots
    // all MAX_* slots), so "past num_outputs" is not the same as "never read".
    for (int i = 0; i < t.num_inputs; i++) {
      input_ptrs[i] = t.inputs[i].base_ptr;
    }
    for (int i = 0; i < t.num_outputs; i++) {
      output_ptrs[i] = t.outputs[i].base_ptr;
    }
#ifdef MPK_ENABLE_TMA
    for (int i = 0; i < t.num_inputs; i++) {
      for (int k = 0; k < mirage::config::MAX_TMA_DESC_PER_TENSOR; k++) {
        input_tma_desc_ptrs[i][k] = t.inputs[i].tma_desc_ptrs[k];
      }
    }
    for (int i = 0; i < t.num_outputs; i++) {
      for (int k = 0; k < mirage::config::MAX_TMA_DESC_PER_TENSOR; k++) {
        output_tma_desc_ptrs[i][k] = t.outputs[i].tma_desc_ptrs[k];
      }
    }
#endif
  }
  __host__ __device__ TaskDesc()
      : input_ptrs{}, output_ptrs{}
#ifdef MPK_ENABLE_TMA
        ,
        input_tma_desc_ptrs{}, output_tma_desc_ptrs{}
#endif
  {
    task_metadata.raw_payload = ~0ull;
  }
  TaskType task_type;
  unsigned variant_id;
  EventId trigger_event;
  EventId dependent_event;
  void *input_ptrs[MAX_INPUTS_PER_TASK];
  void *output_ptrs[MAX_OUTPUTS_PER_TASK];
#ifdef MPK_ENABLE_TMA
  void *input_tma_desc_ptrs[MAX_INPUTS_PER_TASK]
                           [mirage::config::MAX_TMA_DESC_PER_TENSOR];
  void *output_tma_desc_ptrs[MAX_OUTPUTS_PER_TASK]
                            [mirage::config::MAX_TMA_DESC_PER_TENSOR];
#endif
  FullTaskDesc::TaskMetadata task_metadata;
};

struct RuntimeConfig {
  int num_workers, num_local_schedulers, num_remote_schedulers, num_graphs;
  int num_gpus, my_gpu_id;
  int num_events;
  unsigned long long int per_worker_queue_len, per_sched_queue_len;
  unsigned long long int *worker_queue_last_ready_task_id;
  unsigned long long int *sched_queue_last_ready_event_id;
  unsigned long long int *sched_queue_next_free_event_id;
  EventCounter *all_event_counters;
  int *all_event_num_triggers;
  TaskDesc *all_tasks;
  EventDesc *all_events;
  TaskId **worker_queues;
  EventId **sched_queues;
  TaskId *first_tasks;
  int *step;                    // Metadata for LLM serving
  long long *tokens;            // Metadata for LLM serving
  long long *input_tokens;      // Metadata for LLM serving
  long long *output_tokens;     // Metadata for LLM serving
  long long eos_token_id;       // Metadata for LLM serving
  int max_seq_length;           // Metadata for LLM serving
  int *new_token_nums;          // Metadata for LLM serving
  int *qo_indptr_buffer;        // Metadata for LLM serving (paged attention)
  int *paged_kv_indptr_buffer;  // Metadata for LLM serving (paged attention)
  int *paged_kv_indices_buffer; // Metadata for LLM serving (paged attention)
  int *paged_kv_last_page_len_buffer; // Metadata for LLM serving
  void *rope_cos_ptr; // [max_seq_len, head_dim] bf16 cosine table for RoPE
  void *rope_sin_ptr; // [max_seq_len, head_dim] bf16 sine table for RoPE
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE) ||                           \
    defined(MODE_ONLINE_NOTOKEN)
  int *prompt_length;     // Metadata for online/offline serving
  int *request_ids;       // Metadata for online/offline serving
  int *page_queue;        // Metadata for online/offline serving
  int *page_queue_head;   // Metadata for online/offline serving
  int *page_queue_tail;   // Metadata for oneline/offline serving
  int *next_request_id;   // Metadata for LLM serving
  int total_num_requests; // Metadata for LLM serving
#endif
  void *profiler_buffer;
  int profiling_num_iters; // Number of iterations to run in profiling mode (0 =
                           // unlimited)
  bool split_worker_scheduler;
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  void *worker_log_buffer; // device buffer: WorkerLogEntry[] for execute_worker
                           // debug
  int *worker_log_count;   // device counter for log entries
  // Per-XCD event counter replication for reduced cross-XCD polling contention
  // Each XCD has a local copy of event counters, updated by a leader worker
  EventCounter *xcd_local_event_counters; // [NUM_XCDS * num_events] - per-XCD
                                          // counter copies
  int *xcd_leader_worker; // [NUM_XCDS] - worker ID that is leader for each XCD
                          // (-1 if none)
  int num_xcds;           // Number of XCDs (8 for MI300X)
  // Runtime XCD mapping: workers write their hardware XCD ID at startup,
  // schedulers read it to build XCD-aligned worker lists
  int *worker_xcd_map; // [num_workers] — worker_id → actual hardware XCD ID
  int *worker_xcd_ready_count; // atomic counter: workers increment after
                               // writing xcd_map
  int *xcd_event_num_tasks; // [num_xcds * num_events] — per-XCD per-event task
                            // count (set by scheduler)
  // Combined kernel: dynamic role election — one scheduler per XCD
  int *xcd_scheduler_claimed;     // [num_xcds] — atomicCAS to claim XCD as
                                  // scheduler (-1 = unclaimed)
  int *dynamic_worker_id_counter; // atomic counter for assigning worker IDs in
                                  // combined kernel
#endif
#ifdef MPK_PRECOMPUTED_DISPATCH
  // Per-worker task queues (filled at runtime from XCD templates)
  size_t
      *precomp_queue; // [num_workers * precomp_max_tpw] — task position indices
  int *precomp_queue_len;                 // [num_workers] — tasks per worker
  int precomp_max_tpw;                    // queue stride (max tasks per worker)
  unsigned long long *precomp_iter_ready; // device counter: scheduler bumps
                                          // after prepare_next_batch
  int *precomp_iter_xcd_release; // [8*16] per-XCD release flags (st_wt/ld_nt
                                 // for fast cross-XCD)
  int *precomp_terminate;        // device flag: 0=run, 1=exit
  unsigned long long *precomp_dbg_tasks_done; // debug: host-mapped task counter
  int *precomp_dbg_worker_state; // debug: [num_workers*4] = {task_pos,
                                 // dep_event, tasks_done, stuck_count}
  // Nil-address fault tripwire: pinned host memory mapped for device access,
  // so breadcrumbs survive the abort that a memory fault triggers. Null
  // unless built with MPK_NIL_TRIPWIRE. See persistent_kernel.cuh.
  unsigned long long *tripwire;
  // Cross-XCD gang barrier: workers sync before executing gang tasks with
  // internal barriers
  unsigned long long
      *precomp_gang_barrier; // [num_events] — per-trigger-event atomic counter
  int *precomp_gang_dispatch_count; // [num_events] — total workers dispatched
                                    // per gang event
  // XCD-template queues: per-XCD-slot, per-rank task lists built on host.
  // Workers discover hardware XCD at runtime and copy their template.
  size_t *precomp_xcd_template;  // [8 * workers_per_xcd * precomp_max_tpw]
  int *precomp_xcd_template_len; // [8 * workers_per_xcd]
  int *precomp_xcd_rank_counter; // [8] — per-XCD atomic rank counter (device
                                 // memory)
  int precomp_workers_per_xcd;   // num_workers / 8
  int *precomp_dbg_gang_phase;   // [num_workers] host-mapped: gang kernel phase
                                 // tracking
  // Multi-layer all-fused execution: one task processes all transformer layers
  void **ml_input_table;      // [ml_num_layers * 24] per-layer input pointers
  void **ml_output_table;     // [ml_num_layers * 11] per-layer output pointers
  EventId *ml_trigger_events; // [ml_num_layers] per-layer trigger events
  size_t *ml_task_positions;  // [ml_num_layers] per-layer task position indices
  unsigned *ml_variant_ids;   // [ml_num_layers] per-layer variant_id
  int *ml_barrier_arrive;     // [8 * 16] per-XCD arrival counters (cache-line
                              // padded)
  int *ml_barrier_global;     // [16] global arrival counter
  int *ml_barrier_release;    // [8 * 16] per-XCD release flags
  int ml_num_layers;          // 0 = disabled, 36 = enabled
  int ml_workers_per_xcd;     // workers per XCD for barrier threshold
#endif
#ifdef MIRAGE_BACKEND_USE_ROCM
  hipStream_t worker_stream, scheduler_stream;
  hipEvent_t prepare_done_event;
  hipEvent_t worker_done_event, scheduler_done_event;
#else
  cudaStream_t worker_stream, scheduler_stream;
  cudaEvent_t prepare_done_event;
  cudaEvent_t worker_done_event, scheduler_done_event;
#endif
};

} // namespace runtime
} // namespace mirage
