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

#include "profiler.h"
#include "tasks/common/copy_sm80.cuh"
#ifdef MPK_ENABLE_TMA
#include "tma.cuh"
#endif
#include "mpk_atoms.cuh"
#include "runtime_header.h"
#ifdef USE_NVSHMEM
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#endif
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(MIRAGE_GRACE_HOPPER)
#include "tasks/hopper/task_header.cuh"
#elif defined(MIRAGE_GRACE_BLACKWELL)
#include "tasks/blackwell/task_header.cuh"
#elif defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
#include "tasks/mi300/task_header.cuh"
#else
#include "tasks/ampere/task_header.cuh"
#endif

// HIP/ROCm compatibility macros and type aliases
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  #include <hip/hip_runtime.h>
  // CUDA runtime API -> HIP equivalents
  #define cudaMalloc hipMalloc
  #define cudaFree hipFree
  #define cudaStreamSynchronize hipStreamSynchronize
  #define cudaDeviceSynchronize hipDeviceSynchronize
  #define cudaSetDevice hipSetDevice
  #define cudaMemcpy hipMemcpy
  #define cudaMemset hipMemset
  #define cudaMemcpy2DAsync hipMemcpy2DAsync
  #define cudaMemcpyHostToDevice hipMemcpyHostToDevice
  #define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
  #define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
  #define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
  #define cudaStreamNonBlocking hipStreamNonBlocking
  #define cudaEventDisableTiming hipEventDisableTiming
  // HIP's hipFuncSetAttribute requires const void*; wrap to cast function pointers
  #define cudaFuncSetAttribute(func, attr, value) hipFuncSetAttribute(reinterpret_cast<const void*>(func), attr, value)
  #define cudaStreamCreateWithFlags hipStreamCreateWithFlags
  #define cudaEventCreateWithFlags hipEventCreateWithFlags
  #define cudaEventRecord hipEventRecord
  #define cudaEventDestroy hipEventDestroy
  #define cudaStreamWaitEvent hipStreamWaitEvent
  #define cudaStreamDestroy hipStreamDestroy
  #define cudaError_t hipError_t
  #define cudaSuccess hipSuccess
  #define cudaGetErrorString hipGetErrorString
  #define cudaGetLastError hipGetLastError
  // cudaStream_t type alias
  typedef hipStream_t cudaStream_t;
  // __nanosleep is CUDA-specific; use AMD's s_sleep intrinsic for HIP
  __device__ __forceinline__ void __nanosleep(unsigned int ns) {
    // AMD GPU s_sleep: each unit is approximately 64 clock cycles
    // For MI300 at ~1.5GHz, 64 cycles ~= 43ns, so 10ns -> 1 s_sleep unit
    // Use at least 1 to ensure we yield
    unsigned int sleep_units = (ns > 0) ? ((ns + 63) / 64) : 1;
    // s_sleep argument must be a compile-time constant, so we use a small fixed value
    // and loop if needed. This is still much better than a busy loop.
    for (unsigned int i = 0; i < sleep_units; i++) {
      __builtin_amdgcn_s_sleep(1);  // Sleep for ~64 cycles
    }
  }
#endif

// Returns wall-clock time in nanoseconds using a fixed-frequency real-time
// clock. Unlike clock64() whose frequency varies by GPU model, this gives
// consistent nanosecond timestamps across all platforms.
__device__ __forceinline__ unsigned long long get_wallclock_ns() {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  // s_memrealtime runs at 100 MHz (10 ns/tick) on MI300X
  return __builtin_amdgcn_s_memrealtime() * 10;
#else
  unsigned long long ret;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret));
  return ret;  // globaltimer runs at 1 GHz (1 ns/tick)
#endif
}

using bfloat16 = type::bfloat16_t;
using namespace mirage::runtime;
using namespace kernel;
// Configurations for the MPK runtime
// #define MPK_MAX_NUM_BATCHED_REQUESTS 16
// #define MPK_MAX_NUM_BATCHED_TOKENS 64
// #define MPK_MAX_NUM_PAGES 1024
// #define MPK_PAGE_SIZE 64

#ifndef MPK_PROFILING_NUM_ITERS
#define MPK_PROFILING_NUM_ITERS 0
#endif

// Max tokens per request per iteration during prefill.
// Capped by attention kernel LDS usage on MI300 (64KB - reserved).
// If not explicitly defined, use full batch size (no per-request cap).
#ifndef MPK_MAX_TOKENS_PER_REQUEST
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
#define MPK_MAX_TOKENS_PER_REQUEST 8
#else
#define MPK_MAX_TOKENS_PER_REQUEST MPK_MAX_NUM_BATCHED_TOKENS
#endif
#endif

#if defined(MIRAGE_GRACE_HOPPER)
#define WORKER_NUM_THREADS 256
#define SINGLE_KERNEL_NUM_THREADS 256
#elif defined(MIRAGE_GRACE_BLACKWELL)
#define WORKER_NUM_THREADS 256
#define SINGLE_KERNEL_NUM_THREADS 256
#elif defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
// AMD MI300: Use 256 threads for better memory-level parallelism
// Benchmarks showed 1.44x speedup vs 128 threads due to more outstanding loads
#define WORKER_NUM_THREADS 256
#define SINGLE_KERNEL_NUM_THREADS 256
#else
#define WORKER_NUM_THREADS 128
#define SINGLE_KERNEL_NUM_THREADS 128
#endif
#define INIT_NUM_THREADS 128

#ifndef CUDA_CHECK
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    hipError_t err = call;                                                    \
    if (err != hipSuccess) {                                                  \
      fprintf(stderr,                                                          \
              "HIP error at %s:%d: %s\n",                                     \
              __FILE__,                                                        \
              __LINE__,                                                        \
              hipGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#else
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr,                                                          \
              "CUDA error at %s:%d: %s\n",                                     \
              __FILE__,                                                        \
              __LINE__,                                                        \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#endif
#endif

// Verbose output for debugging (disabled by default for performance)
// Uncomment the line below to enable verbose debugging for AMD
// #define MPK_ENABLE_VERBOSE

// =============================================================================
// AMD MI300X XCD-aware synchronization for reduced cross-XCD polling contention
// =============================================================================
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)

// Number of XCDs on MI300X
constexpr int MI300X_NUM_XCDS = 8;

// Get the XCD (XCC) ID for the current thread
__device__ __forceinline__ int get_current_xcd_id() {
  int xcd_id;
  asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
}

// Per-block shared state for XCD info (computed once at block start)
struct XCDBlockInfo {
  int xcd_id;           // XCD this block is running on
  int is_xcd_leader;    // 1 if this worker is leader for its XCD, 0 otherwise
};

// Elect XCD leaders - first worker on each XCD becomes leader
// Returns true if this worker should be leader
__device__ __forceinline__ bool elect_xcd_leader(
    RuntimeConfig const &config,
    int worker_id,
    int xcd_id) {
  // Use atomicCAS to elect first worker on each XCD as leader
  // -1 means no leader yet
  int old = atomicCAS(&config.xcd_leader_worker[xcd_id], -1, worker_id);
  return (old == -1 || old == worker_id);
}

#else
// Stubs for NVIDIA - no XCD concept
__device__ __forceinline__ int get_current_xcd_id() { return 0; }
#endif

__device__ __forceinline__ void
    _execute_task(TaskDesc const *task_desc,
                  RuntimeConfig const &runtime_config);

// Gang task execution: same as _execute_task but with tile_idx for gang dispatch
__device__ __forceinline__ void
    _execute_gang_task(TaskDesc const *task_desc,
                       RuntimeConfig const &runtime_config,
                       int tile_idx);

// Helper: check if a task type is a gang task
__device__ __host__ __forceinline__ bool is_gang_task_type(TaskType t) {
  return t == TASK_GANG_LINEAR_MI300 || t == TASK_GANG_LINEAR_N_TILING_MI300 ||
         t == TASK_GANG_LINEAR_RES_MI300 ||
         t == TASK_GANG_LINEAR_RES_N_TILING_MI300 ||
         t == TASK_GANG_LINEAR_SILU_MI300 || t == TASK_GANG_RMS_NORM_MI300 ||
         t == TASK_GANG_SPLITK_LINEAR_RES_MI300 ||
         t == TASK_GANG_KSPLIT_GEMM_MI300 || t == TASK_GANG_KSPLIT_FINALIZE_MI300 ||
         t == TASK_GANG_ATTN_SPLIT_KV_MI300 || t == TASK_GANG_ATTN_MERGE_MI300 ||
         t == TASK_GANG_MOE_W13_LINEAR_MI300 || t == TASK_GANG_MOE_W2_LINEAR_MI300 ||
         t == TASK_GANG_LINEAR_MSPLIT_MI300 || t == TASK_GANG_LINEAR_RES_MSPLIT_MI300;
}

__device__ __forceinline__ bool is_termination_event(size_t event_loc,
                                                     EventDesc e) {
  return (event_loc == 0);
}

__device__ __forceinline__ bool is_nvshmem_event(EventId event_id) {
  return (event_id & EVENT_NVSHMEM_TAG) > 0;
}

__device__ __forceinline__ size_t get_event_gpu_id(EventId event_id) {
  return ((event_id >> 32) & 0xffff);
}

__device__ __forceinline__ size_t get_event_position_index(EventId event_id) {
  return (event_id & 0xffffffff);
}

__device__ __forceinline__ size_t get_task_iteration_num(TaskId task_id) {
  return (task_id >> 32);
}

__device__ __forceinline__ size_t get_task_position_index(TaskId task_id) {
  return (task_id & 0xffffffff);
}

__device__ __forceinline__ TaskId compute_task_id(size_t iteration_num,
                                                  size_t position_index) {
  return ((iteration_num << 32) | position_index);
}

__global__ void init_kernel(RuntimeConfig config) {
  assert(gridDim.x == 1);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Only a single thread that initializes everything
  if (threadIdx.x == 0) {
    // initialize metadata
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
    for (int i = 0; i < config.total_num_requests; i++) {
      config.step[i] = 0;
    }
    *config.next_request_id = 0;
    for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
      config.request_ids[i] = -1;
    }
    for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS + 1; i++) {
      config.qo_indptr_buffer[i] = 0;
      config.paged_kv_indptr_buffer[i] = 0;
    }
    // Page manager
    *config.page_queue_head = 0;
    *config.page_queue_tail = MPK_MAX_NUM_PAGES;
    for (int i = 0; i < MPK_MAX_NUM_PAGES; i++) {
      config.page_queue[i] = i;
    }
#endif
  }
}

__global__ void prepare_kernel(RuntimeConfig config,
                               int end_of_task_graph_event_pos) {
  // Initialize worker queue last task id
  // Each worker now maintains a local and a remote worker queue
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < 2 * config.num_workers;
       i += blockDim.x * gridDim.x) {
    config.worker_queue_last_ready_task_id[i] = 0;
  }
  // Initialize scheduler queue last event id
  // We maintain one extra scheduler queue for the global scheduler
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_schedulers + 1;
       i += blockDim.x * gridDim.x) {
    config.sched_queue_last_ready_event_id[i] = 0;
    config.sched_queue_next_free_event_id[i] = 0;
  }
  // Initialize all event counters
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < config.num_events;
       i += blockDim.x * gridDim.x) {
    config.all_event_counters[i] = 0;
  }

#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // Reset per-XCD local event counters and leaders
  if (config.xcd_local_event_counters != nullptr) {
    int total_local_counters = config.num_xcds * config.num_events;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_local_counters;
         i += blockDim.x * gridDim.x) {
      config.xcd_local_event_counters[i] = 0;
    }
  }
  // Reset XCD leaders (-1 = no leader)
  if (config.xcd_leader_worker != nullptr) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < config.num_xcds;
         i += blockDim.x * gridDim.x) {
      config.xcd_leader_worker[i] = -1;
    }
  }
  // Reset worker XCD map and ready counter for next kernel launch
  if (config.worker_xcd_map != nullptr) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < config.num_workers;
         i += blockDim.x * gridDim.x) {
      config.worker_xcd_map[i] = -1;
    }
  }
  if (config.worker_xcd_ready_count != nullptr &&
      blockIdx.x == 0 && threadIdx.x == 0) {
    *config.worker_xcd_ready_count = 0;
  }
  // Reset per-XCD per-event task thresholds (scheduler will recompute)
  if (config.xcd_event_num_tasks != nullptr) {
    int total_xcd_events = config.num_xcds * config.num_events;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_xcd_events;
         i += blockDim.x * gridDim.x) {
      config.xcd_event_num_tasks[i] = 0;
    }
  }
  // Reset combined kernel election state
  if (config.xcd_scheduler_claimed != nullptr && blockIdx.x == 0) {
    for (int i = threadIdx.x; i < config.num_xcds; i += blockDim.x) {
      config.xcd_scheduler_claimed[i] = -1;
    }
  }
  if (config.dynamic_worker_id_counter != nullptr &&
      blockIdx.x == 0 && threadIdx.x == 0) {
    *config.dynamic_worker_id_counter = 0;
  }
#endif

  // Send event to scheduler[0]
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    assert(config.all_events[end_of_task_graph_event_pos].event_type ==
           EVENT_END_OF_TASK_GRAPH);
    config.sched_queue_next_free_event_id[0] = 1;
    config.sched_queues[0][0] = end_of_task_graph_event_pos;
    config.sched_queue_last_ready_event_id[0] = 1;
  }
}

#ifdef MODE_OFFLINE
// TODO: parallelize this processing
__device__ __forceinline__ bool
    prepare_next_batch(RuntimeConfig const &config) {
  __shared__ int smem_kv_indices[MPK_MAX_NUM_PAGES];
  int page_queue_head = *config.page_queue_head;
  int page_queue_tail = *config.page_queue_tail;
  // Step 1: finalize previous batch
  for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    int16_t request_id = config.request_ids[i];
    if (request_id != -1) {
      // Step 1.1: move output_tokens to tokens
      int step = config.step[request_id];
      int qo_indptr = config.qo_indptr_buffer[i];
      int num_tokens = config.qo_indptr_buffer[i + 1] - qo_indptr;
      int prompt_len = config.prompt_length[request_id];
      for (int j = 0; j < num_tokens; j++) {
        if (step + j + 1 >= prompt_len &&
            step + j + 1 < config.max_seq_length) {
          config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step + j + 1] =
              config.output_tokens[qo_indptr + j];
        }
      }
      config.step[request_id] = step + num_tokens;
#ifdef MPK_ENABLE_PROFILING
      if (config.profiling_num_iters > 0 && step + num_tokens >= config.profiling_num_iters)
#else
      if ((step + num_tokens + 1 >= config.max_seq_length) ||
          ((config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step +
                          num_tokens] == config.eos_token_id) &&
           (step + num_tokens >= prompt_len)))
#endif
      {
        // Request is done
        config.request_ids[i] = -1;
        // Free pages
        int kv_indptr = config.paged_kv_indptr_buffer[i];
        int num_pages = config.paged_kv_indptr_buffer[i + 1] - kv_indptr;
        for (int j = 0; j < num_pages; j++) {
          config.page_queue[page_queue_tail % MPK_MAX_NUM_PAGES] =
              config.paged_kv_indices_buffer[kv_indptr + j];
          page_queue_tail++;
        }
      }
    }
  }

  // Step 2: copy kv_indices to shared mem
  int num_pages = config.paged_kv_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
  for (int i = 0; i < num_pages; i++) {
    smem_kv_indices[i] = config.paged_kv_indices_buffer[i];
  }

  // Step 3: prepare next batch
  int num_reqs = 0, num_tokens = 0;
  num_pages = 0;
  for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    int16_t request_id = config.request_ids[i];
    if (request_id != -1) {
      int kv_indptr = config.paged_kv_indptr_buffer[i];
      int num_old_pages = config.paged_kv_indptr_buffer[i + 1] - kv_indptr;
      config.request_ids[num_reqs] = request_id;
      config.qo_indptr_buffer[num_reqs] = num_tokens;
      config.paged_kv_indptr_buffer[num_reqs] = num_pages;
      int step = config.step[request_id];
      int num_new_tokens = config.prompt_length[request_id] - step;
      if (num_new_tokens > 0) {
        // Prefill requests: cap per-request tokens to avoid attention LDS overflow
        num_new_tokens =
            min(num_new_tokens, min(MPK_MAX_TOKENS_PER_REQUEST,
                                    MPK_MAX_NUM_BATCHED_TOKENS - num_tokens));
      } else {
        // Decode requests
        num_new_tokens = min(1, MPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
      }
      // Move tokens to input_tokens
      for (int j = 0; j < num_new_tokens; j++) {
        config.input_tokens[num_tokens + j] =
            config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step + j];
      }
      // Prepare page indptrs
      int num_new_pages =
          (step + num_new_tokens + MPK_PAGE_SIZE - 1) / MPK_PAGE_SIZE;
      {
        int last_len = (step + num_new_tokens) % MPK_PAGE_SIZE;
        config.paged_kv_last_page_len_buffer[num_reqs] =
            last_len == 0 ? MPK_PAGE_SIZE : last_len;
      }
      for (int j = 0; j < num_old_pages; j++) {
        config.paged_kv_indices_buffer[num_pages + j] =
            smem_kv_indices[kv_indptr + j];
      }
      for (int j = num_old_pages; j < num_new_pages; j++) {
        config.paged_kv_indices_buffer[num_pages + j] =
            config.page_queue[page_queue_head % MPK_MAX_NUM_PAGES];
        page_queue_head++;
      }
      num_pages += num_new_pages;
      num_tokens += num_new_tokens;
      num_reqs++;
    }
  }

  // Add new prefill requests until we reach capacity
  while (num_reqs < MPK_MAX_NUM_BATCHED_REQUESTS &&
         num_tokens < MPK_MAX_NUM_BATCHED_TOKENS) {
    int next_request_id = *config.next_request_id;
    if (next_request_id >= config.total_num_requests) {
      break;
    }
    config.request_ids[num_reqs] = next_request_id;
    config.qo_indptr_buffer[num_reqs] = num_tokens;
    config.paged_kv_indptr_buffer[num_reqs] = num_pages;
    // Prefill request: cap per-request tokens to avoid attention LDS overflow
    int num_new_tokens = min(config.prompt_length[next_request_id],
                             min(MPK_MAX_TOKENS_PER_REQUEST,
                                 MPK_MAX_NUM_BATCHED_TOKENS - num_tokens));
    // Move tokens to input tokens
    for (int j = 0; j < num_new_tokens; j++) {
      config.input_tokens[num_tokens + j] =
          config.tokens[next_request_id * MPK_MAX_SEQ_LENGTH + j];
    }
    int num_new_pages = (num_new_tokens + MPK_PAGE_SIZE - 1) / MPK_PAGE_SIZE;
    {
      int last_len = num_new_tokens % MPK_PAGE_SIZE;
      config.paged_kv_last_page_len_buffer[num_reqs] =
          last_len == 0 ? MPK_PAGE_SIZE : last_len;
    }
    for (int j = 0; j < num_new_pages; j++) {
      config.paged_kv_indices_buffer[num_pages + j] =
          config.page_queue[page_queue_head % MPK_MAX_NUM_PAGES];
      page_queue_head++;
    }
    num_tokens += num_new_tokens;
    num_pages += num_new_pages;
    num_reqs++;
    *config.next_request_id = next_request_id + 1;
  }

  // Step 4: Update all unused requests slots
  for (int i = num_reqs; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    config.request_ids[i] = -1;
  }
  for (int i = num_reqs; i <= MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    config.qo_indptr_buffer[i] = num_tokens;
    config.paged_kv_indptr_buffer[i] = num_pages;
  }

  // Step 5: update page head tail
  *config.page_queue_head = page_queue_head;
  *config.page_queue_tail = page_queue_tail;

  if (num_tokens == 0) {
    return false;
  } else {
    return true;
  }
}
#endif

#ifdef MODE_ONLINE
__device__ __forceinline__ bool
    prepare_next_batch(RuntimeConfig const &config) {
  int step = config.step[0];
#ifdef MPK_ENABLE_VERBOSE
  printf("step: %d, new_token_num(%p): %d, new_token_ids:\n",
         step,
         config.new_token_nums,
         config.new_token_nums[0]);
  for (int i = 0; i < config.new_token_nums[0]; i++) {
    printf("%lld ", config.tokens[step + 1 + i]);
  }
  printf("\n");
#endif
  config.step[0] = step + config.new_token_nums[0];

#ifdef MPK_ENABLE_PROFILING
  if (config.profiling_num_iters > 0 && step + 1 >= config.profiling_num_iters) {
    return false;
  }
  return true;
#else
  if ((step + 2 >= config.max_seq_length) ||
      (config.tokens[step + 1] == config.eos_token_id)) {
    return false;
  } else {
    return true;
  }
#endif
}
#endif

#ifdef MODE_ONLINE_NOTOKEN
__device__ __forceinline__ bool prepare_next_batch(RuntimeConfig const &config,
                                                   size_t iteration_num = 0) {
  // TODO: iteration_num is a current workaround
  // We may consider split EVENT_END_OF_TASK_GRAPH into
  // EVENT_END_OF_TASK_GRAPH and EVENT_START_OF_TASK_GRAPH
  if (iteration_num > 0) {
    return false;
  } else { // iteration_num == 0
    return true;
  }
}
#endif

__device__ __forceinline__ int get_rand_sched_id(size_t event_index,
                                                 int worker_id,
                                                 int num_workers,
                                                 int num_schedulers,
                                                 int xcd_id = 0) {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // XCD-aligned: worker signals the scheduler on its own XCD.
  // scheduler_kernel block k runs on XCD k, so sched_id == XCD ID.
  return xcd_id;
#else
  size_t x = worker_id;
  return x / ((num_workers + num_schedulers - 1) / num_schedulers);
#endif
}

__device__ __forceinline__ void
    get_first_last_ids(unsigned long long int num_elements,
                       unsigned long long int num_workers,
                       unsigned long long int my_id,
                       unsigned long long int *my_first_element,
                       unsigned long long int *my_last_element) {
  unsigned long long int num_elements_per_worker = num_elements / num_workers;
  unsigned long long int reminder = num_elements % num_workers;
  if (my_id < reminder) {
    *my_first_element = (num_elements_per_worker + 1) * my_id;
    *my_last_element = *my_first_element + num_elements_per_worker + 1;
  } else {
    *my_first_element = num_elements_per_worker * my_id + reminder;
    *my_last_element = *my_first_element + num_elements_per_worker;
  }
}

__device__ __forceinline__ void terminate_schedulers(RuntimeConfig config) {
  // Event ID 0 is the termination event
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  for (int i = 0; i < num_schedulers; i++) {
    // size_t last_event_id =
    //     atomicAdd(&config.sched_queue_next_free_event_id[i], 1);
    size_t last_event_id =
        atom_add_release_gpu_u64(&config.sched_queue_next_free_event_id[i], 1);
    st_relaxed_gpu_u64(
        &config.sched_queues[i][last_event_id % config.per_sched_queue_len], 0);
    // Memory fence to ensure event store is visible before signaling
    threadfence_gpu();
    // CAS loop to update last_ready_event_id in order
    size_t old;
    do {
      old = atom_cas_release_gpu_u64(&config.sched_queue_last_ready_event_id[i],
                                     last_event_id,
                                     last_event_id + 1);
    } while (old != last_event_id);
  }
}

__device__ __forceinline__ void worker_checker(RuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  // int num_schedulers =
  //    config.num_local_schedulers + config.num_remote_schedulers;

  assert(gridDim.x == config.num_workers);
  assert(config.num_workers <= MAX_NUM_WORKERS);
  // We will reinterpret TaskDesc as an array of integers to
  // collectively load it from device to shared memory
  static_assert(sizeof(TaskDesc) % sizeof(int) == 0);
}

__device__ __forceinline__ void scheduler_checker(RuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  // int num_schedulers =
  //    config.num_local_schedulers + config.num_remote_schedulers;

  assert(config.num_workers <= MAX_NUM_WORKERS);
}

__device__ __forceinline__ void persistent_checker(RuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  int const num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  int const num_schedulers_per_sm = std::min((int)blockDim.x / 32, 4);
  assert(num_schedulers % num_schedulers_per_sm == 0);
  assert(gridDim.x ==
         config.num_workers + num_schedulers / num_schedulers_per_sm);
  assert(config.num_workers <= MAX_NUM_WORKERS);
  // We will reinterpret TaskDesc as an array of integers to
  // collectively load it from device to shared memory
  static_assert(sizeof(TaskDesc) % sizeof(int) == 0);
  // assert(blockDim.x >= 128);
}

// ── Device-global task-time accumulators (all workers atomicAdd) ──
#ifdef MPK_ENABLE_DEVICE_TASK_ACCUM
// Slots: 0=QKV 1=GATE_UP 2=O_PROJ 3=DOWN_PROJ 4=RMS 5=SILU 6=ATTN_SPLIT 7=ATTN_MERGE 8=OTHER
#define DACCUM_SLOTS 9
__device__ unsigned long long g_daccum_ns[DACCUM_SLOTS];
__device__ unsigned long long g_daccum_cnt[DACCUM_SLOTS];
__device__ int g_daccum_active;       // 0=prefill, 1=decode
__device__ int g_daccum_decode_iters; // count of decode iterations
#endif

__device__ __forceinline__ void execute_worker(RuntimeConfig config,
                                               int assigned_worker_id = -1) {
  // Make sure overall smem usage here do not exceed 3KB
  // last_task_pos: 2 * 8 = 16 B
  // next_task_pos: 2 * 8 = 16 B
  // worker_queue_ids: 2 * 4 = 8 B
  // worker_queues: 2 * 8 = 16 B
  // remaining: 3016 B

  constexpr int TASK_DESCS_BUFFER_LENGTH = std::min(
      (mirage::runtime::WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE - 56) /
          (int)(sizeof(TaskDesc) + sizeof(TaskId)),
      16);
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // HIP: Use char arrays and cast to avoid constructor calls on __shared__ arrays
  // HIP doesn't support default construction of __shared__ array elements
  // HIP doesn't support alignas with __shared__, use __align__ instead
  static_assert(TASK_DESCS_BUFFER_LENGTH <= 16, "Buffer length exceeds 16");
  __shared__ __align__(alignof(TaskDesc)) char task_descs_storage[16 * sizeof(TaskDesc)];
  __shared__ __align__(alignof(TaskId)) char task_ids_storage[16 * sizeof(TaskId)];
  TaskDesc* task_descs = reinterpret_cast<TaskDesc*>(task_descs_storage);
  TaskId* task_ids = reinterpret_cast<TaskId*>(task_ids_storage);
#else
  __shared__ TaskDesc task_descs[TASK_DESCS_BUFFER_LENGTH];
  __shared__ TaskId task_ids[TASK_DESCS_BUFFER_LENGTH];
#endif
  __shared__ TaskId *worker_queues[2];
  __shared__ int worker_queue_ids[2];
  __shared__ size_t next_task_pos[2];
  __shared__ size_t last_task_pos[2];

#ifdef MPK_ENABLE_PROFILING
  PROFILER_CLOSURE_PARAMS_DECL;
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),
                0,
                1,
                (threadIdx.x % WORKER_NUM_THREADS == 0));

#endif
  int const worker_id = (assigned_worker_id >= 0) ? assigned_worker_id : blockIdx.x;
  worker_queues[0] = config.worker_queues[worker_id];
  worker_queue_ids[0] = worker_id;
  int num_worker_queues = 1;
  if (config.num_gpus > 1) {
    worker_queues[num_worker_queues] =
        config.worker_queues[worker_id + config.num_workers];
    worker_queue_ids[num_worker_queues] = worker_id + config.num_workers;
    num_worker_queues++;
  }

#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // XCD-aware polling: detect XCD and elect leader
  __shared__ int block_xcd_id;
  __shared__ int block_is_xcd_leader;

  if (threadIdx.x == 0) {
    block_xcd_id = get_current_xcd_id();
    // Write this worker's XCD ID to shared map for scheduler to read
    if (config.worker_xcd_map != nullptr) {
      config.worker_xcd_map[worker_id] = block_xcd_id;
      if (worker_id < 8) {
        printf("[WORKER_XCD] worker_id=%d block=%d xcd=%d\n",
               worker_id, (int)blockIdx.x, block_xcd_id);
      }
      // Fence ensures map write is visible before incrementing ready counter
      threadfence_gpu();
      atomicAdd(config.worker_xcd_ready_count, 1);
    }
    // Elect leader: first worker on each XCD becomes leader
    // Leader polls global counter, others poll local counter
    if (config.xcd_leader_worker != nullptr) {
      block_is_xcd_leader = elect_xcd_leader(config, worker_id, block_xcd_id) ? 1 : 0;
    } else {
      block_is_xcd_leader = 0;  // No hierarchical polling if not configured
    }
  }
  __syncthreads();

  int const xcd_id = block_xcd_id;
  int const is_xcd_leader = block_is_xcd_leader;

#ifdef MPK_ENABLE_GANG_TASKS
  // XCD-local rank for gang tasks: computed lazily on first gang task.
  // -1 means not yet computed. Safe to compute after scheduler starts
  // dispatching (all worker_xcd_map entries are valid by then).
  __shared__ int block_xcd_local_rank;
  __shared__ int block_workers_on_xcd;
  if (threadIdx.x == 0) {
    block_xcd_local_rank = -1;
    block_workers_on_xcd = 0;
  }
#endif
#endif
#if !defined(__HIP_PLATFORM_AMD__) && !defined(MIRAGE_AMD_MI300)
  int const xcd_id = 0;  // NVIDIA: no XCD concept
#endif

  if (threadIdx.x == 0) {
    for (int i = 0; i < 2; i++) {
      next_task_pos[i] = 0;
    }
    for (int i = 0; i < 2; i++) {
      last_task_pos[i] = 0;
    }
    // num_loaded_tasks = 0;
  }

  int queue_pos = 0, queue_len = 0;
  unsigned long long prev_begin_clk = 0;
  int fwd_pass_count = 0;
#ifdef MPK_ENABLE_PROFILING
  size_t task_counter = 0;
#endif
  __syncthreads();

  // Timing instrumentation
#ifdef MPK_ENABLE_TIMING
  unsigned long long total_poll_cycles = 0;
  unsigned long long total_dep_wait_cycles = 0;
  unsigned long long total_exec_cycles = 0;
  unsigned long long total_signal_cycles = 0;
  int total_poll_iters = 0;
  int total_dep_wait_iters = 0;
  int total_tasks_executed = 0;
  // Per-task-type timing (major types only)
  unsigned long long linear_cycles = 0, linear_res_cycles = 0;
  unsigned long long attention_cycles = 0, rms_cycles = 0, silu_cycles = 0;
  unsigned long long fused_cycles = 0;
  int linear_count = 0, linear_res_count = 0;
  int attention_count = 0, rms_count = 0, silu_count = 0;
  int fused_count = 0;
#endif

  while (true) {
    // fetch next task from a task queue if task_descs is empty
    if (queue_pos == queue_len) {
      int queue_idx = 0;
      if (threadIdx.x == 0) {
        int poll_count = 0;
#ifdef MPK_ENABLE_TIMING
        unsigned long long poll_start = clock64();
#endif
        while (next_task_pos[queue_idx] == last_task_pos[queue_idx]) {
          // Scheduler is on same XCD — use L2-local load instead of NT/HBM
          last_task_pos[queue_idx] =
              ld_local_u64(&config.worker_queue_last_ready_task_id
                                [worker_queue_ids[queue_idx]]);
          if (next_task_pos[queue_idx] < last_task_pos[queue_idx]) {
            break;
          } else {
            queue_idx =
                (queue_idx == num_worker_queues - 1) ? 0 : queue_idx + 1;
          }
          // nanosleep to avoid overwhelming I/O
          __nanosleep(10);
#ifdef MPK_ENABLE_TIMING
          total_poll_iters++;
#endif
        }
#ifdef MPK_ENABLE_TIMING
        total_poll_cycles += clock64() - poll_start;
#endif
        assert(next_task_pos[queue_idx] + config.per_worker_queue_len >
               last_task_pos[queue_idx]);
        // Compiler fence for ordering — scheduler is on same XCD, L2 coherent
        fence_local();
      }
      __syncthreads();
      int num_loaded_tasks =
          min((int)(last_task_pos[queue_idx] - next_task_pos[queue_idx]),
              TASK_DESCS_BUFFER_LENGTH);
      // Load task ids
      if (threadIdx.x < num_loaded_tasks) {
        // Scheduler is on same XCD — task data visible via L2
        task_ids[threadIdx.x] = ld_local_u64(
            &worker_queues[queue_idx][(next_task_pos[queue_idx] + threadIdx.x) %
                                      config.per_worker_queue_len]);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
#ifdef MPK_ENABLE_VERBOSE
        for (int i = 0; i < num_loaded_tasks; i++) {
          printf(
              "[%d][FTCH] worker_id(%d) queue_idx(%d) next_task_pos(%llu, "
              "%llu) last_task_pos(%llu, %llu) "
              "task_id(%llu) task_type(%d) event_id(%llx) \n",
              config.my_gpu_id,
              worker_id,
              queue_idx,
              next_task_pos[0],
              next_task_pos[1],
              last_task_pos[0],
              last_task_pos[1],
              get_task_position_index(task_ids[i]),
              config.all_tasks[get_task_position_index(task_ids[i])].task_type,
              config.all_tasks[get_task_position_index(task_ids[i])]
                  .trigger_event);
        }
#endif
        next_task_pos[queue_idx] += num_loaded_tasks;
      }
      // Load task descs
      static_assert(sizeof(TaskDesc) % 16 == 0);
      constexpr int TASK_SIZE = sizeof(TaskDesc) / 16; // 128b copy-async
      for (int i = threadIdx.x; i < num_loaded_tasks * TASK_SIZE;
           i += blockDim.x) {
        int task_idx = i / TASK_SIZE;
        int offset = i % TASK_SIZE;
        load_smem(reinterpret_cast<char *>(task_descs) + i * 16,
                  reinterpret_cast<char *>(
                      config.all_tasks +
                      get_task_position_index(task_ids[task_idx])) +
                      offset * 16);
      }
      kernel::cp_async_fence();
      kernel::cp_async_wait<0>();
      __syncthreads();
      queue_pos = 0;
      queue_len = num_loaded_tasks;
    }
    TaskDesc *task_desc = task_descs + queue_pos;
    // Dependency check: The scheduler only dispatches tasks after their
    // dependency event has fired. The release-acquire chain guarantees:
    //   completing_worker RELEASE → scheduler ACQUIRE → scheduler RELEASE → this_worker ACQUIRE
    // So by the time we read the task from our queue, the dependency is satisfied.
    if (threadIdx.x == 0) {
      if (task_desc->dependent_event != EVENT_INVALID_ID) {
        EventId event_id = task_desc->dependent_event;
        assert(get_event_gpu_id(event_id) == config.my_gpu_id);
        size_t event_index = get_event_position_index(event_id);
        EventCounter needed_counts =
            static_cast<EventCounter>(
                config.all_event_num_triggers[event_index]) *
            get_task_iteration_num(task_ids[queue_pos]);
        EventCounter actual_counts = 0;
        if (is_nvshmem_event(event_id)) {
#ifdef USE_NVSHMEM
          nvshmem_signal_wait_until(
              reinterpret_cast<uint64_t *>(
                  &config.all_event_counters[event_index]),
              NVSHMEM_CMP_EQ,
              needed_counts);
#endif
        } else {
#ifdef MPK_ENABLE_TIMING
          unsigned long long dep_start = clock64();
#endif
          // Single ACQUIRE load — dependency guaranteed satisfied by scheduler dispatch chain.
          // The scheduler dispatches this task only after the dependency event fired,
          // and the release-acquire ordering through the scheduler ensures visibility.
          actual_counts = __atomic_load_n(
              reinterpret_cast<unsigned long long*>(&config.all_event_counters[event_index]),
              __ATOMIC_RELAXED);
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
          // Agent-scope acquire fence (GPU-only, not system scope)
          __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#else
          __atomic_thread_fence(__ATOMIC_ACQUIRE);
#endif
          if (actual_counts < needed_counts) {
            // Dependency not yet satisfied — poll until ready
            while (actual_counts < needed_counts) {
              actual_counts = __atomic_load_n(
                  reinterpret_cast<unsigned long long*>(&config.all_event_counters[event_index]),
                  __ATOMIC_RELAXED);
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
              __builtin_amdgcn_s_sleep(1);
#endif
#ifdef MPK_ENABLE_TIMING
              total_dep_wait_iters++;
#endif
            }
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#else
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
#endif
          }
#ifdef MPK_ENABLE_TIMING
          total_dep_wait_cycles += clock64() - dep_start;
#endif
        }
      }
    }
    __syncthreads();

#ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_START(task_desc->task_type, task_counter);
    }
#endif

#ifdef MPK_ENABLE_GANG_TASKS
    // Gang tasks: tiles executed per worker for event counting (default 1)
    __shared__ int gang_tiles_executed;
    if (threadIdx.x == 0) gang_tiles_executed = 1;
#endif
#ifdef MPK_ENABLE_TIMING
    unsigned long long exec_start = clock64();
#endif
#ifdef MPK_ENABLE_DEVICE_TASK_ACCUM
    unsigned long long _accum_t0 = __builtin_amdgcn_s_memrealtime();
#endif
    // Successfully fetched a new task
    if (task_desc->task_type == TASK_TERMINATE) {
      // Terminate
#ifdef MPK_ENABLE_TIMING
      // Print timing stats before returning
      if (threadIdx.x == 0 && worker_id < 8) {
        printf("[TIMING] worker=%d tasks=%d poll_iters=%d dep_iters=%d "
               "poll_cycles=%llu dep_cycles=%llu exec_cycles=%llu signal_cycles=%llu\n",
               worker_id, total_tasks_executed, total_poll_iters, total_dep_wait_iters,
               total_poll_cycles, total_dep_wait_cycles, total_exec_cycles, total_signal_cycles);
        printf("[TASK_TIME] worker=%d linear=%llu/%d linear_res=%llu/%d attn=%llu/%d rms=%llu/%d silu=%llu/%d fused=%llu/%d\n",
               worker_id, linear_cycles, linear_count, linear_res_cycles, linear_res_count,
               attention_cycles, attention_count, rms_cycles, rms_count, silu_cycles, silu_count,
               fused_cycles, fused_count);
      }
#endif
#ifdef MPK_ENABLE_DEVICE_TASK_ACCUM
      if (threadIdx.x == 0 && worker_id == 0) {
        // g_daccum_decode_iters is incremented by ALL workers that see BEGIN
        // during decode, so divide by approx workers-per-BEGIN to get actual iters.
        // Use a simpler approach: just report raw totals.
        int ni = g_daccum_decode_iters > 0 ? g_daccum_decode_iters : 1;
        int nw = config.num_workers;
        printf("[DACCUM] iters=%d workers=%d\n", ni, nw);
        printf("[DACCUM] QKV        total_us=%llu  count=%llu  per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[0]/1000, g_daccum_cnt[0], (double)g_daccum_ns[0]/1000.0/ni, (double)g_daccum_cnt[0]/ni);
        printf("[DACCUM] GATE_UP    total_us=%llu  count=%llu  per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[1]/1000, g_daccum_cnt[1], (double)g_daccum_ns[1]/1000.0/ni, (double)g_daccum_cnt[1]/ni);
        printf("[DACCUM] O_PROJ     total_us=%llu  count=%llu  per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[2]/1000, g_daccum_cnt[2], (double)g_daccum_ns[2]/1000.0/ni, (double)g_daccum_cnt[2]/ni);
        printf("[DACCUM] DOWN_PROJ  total_us=%llu  count=%llu  per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[3]/1000, g_daccum_cnt[3], (double)g_daccum_ns[3]/1000.0/ni, (double)g_daccum_cnt[3]/ni);
        printf("[DACCUM] RMS_NORM   total_us=%llu  count=%llu  per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[4]/1000, g_daccum_cnt[4], (double)g_daccum_ns[4]/1000.0/ni, (double)g_daccum_cnt[4]/ni);
        printf("[DACCUM] SILU_MUL   total_us=%llu  count=%llu  per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[5]/1000, g_daccum_cnt[5], (double)g_daccum_ns[5]/1000.0/ni, (double)g_daccum_cnt[5]/ni);
        printf("[DACCUM] ATTN_SPLIT total_us=%llu  count=%llu  per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[6]/1000, g_daccum_cnt[6], (double)g_daccum_ns[6]/1000.0/ni, (double)g_daccum_cnt[6]/ni);
        printf("[DACCUM] ATTN_MERGE total_us=%llu  count=%llu  per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[7]/1000, g_daccum_cnt[7], (double)g_daccum_ns[7]/1000.0/ni, (double)g_daccum_cnt[7]/ni);
        printf("[DACCUM] OTHER      total_us=%llu  count=%llu  per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[8]/1000, g_daccum_cnt[8], (double)g_daccum_ns[8]/1000.0/ni, (double)g_daccum_cnt[8]/ni);
      }
#endif
      return;
    } else if (task_desc->task_type == TASK_BEGIN_TASK_GRAPH) {
      // Track per-iteration wall-clock time on worker 0
      if (threadIdx.x == 0 && worker_id == 0) {
        unsigned long long cur_clk = get_wallclock_ns();
#ifndef MPK_ENABLE_PROFILING
        int num_active_tokens = config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
        if (prev_begin_clk != 0 && (fwd_pass_count < 10 || fwd_pass_count % 50 == 0)) {
          printf("[FWD_PASS] iter=%d time_ms=%.3f num_active_tokens=%d\n",
                 fwd_pass_count,
                 (double)(cur_clk - prev_begin_clk) / 1000000.0,
                 num_active_tokens);
        }
#endif
        prev_begin_clk = cur_clk;
        fwd_pass_count++;
      }
#ifdef MPK_ENABLE_DEVICE_TASK_ACCUM
      // Decode-only accumulation: activate on first decode iter (nat==1)
      if (threadIdx.x == 0) {
        int nat = config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
        if (nat <= 1) {
          // Transition: atomicCAS ensures exactly one worker resets
          int old = atomicCAS(&g_daccum_active, 0, 1);
          if (old == 0) {
            for (int s = 0; s < DACCUM_SLOTS; s++) {
              g_daccum_ns[s] = 0;
              g_daccum_cnt[s] = 0;
            }
            g_daccum_decode_iters = 0;
            __threadfence();
          }
          // Count decode iters: only worker_id 0 increments
          if (worker_id == 0) {
            g_daccum_decode_iters++;
          }
        }
      }
#endif
#ifdef MPK_ENABLE_GANG_TASKS
    } else if (is_gang_task_type(task_desc->task_type)) {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
      // Gang: compute XCD-local rank + workers_on_xcd lazily on first encounter
      if (threadIdx.x == 0 && block_xcd_local_rank == -1) {
        // All worker_xcd_map entries are valid by now (scheduler waited for ready_count)
        int rank = 0;
        int total_on_xcd = 0;
        for (int w = 0; w < config.num_workers; w++) {
          if (config.worker_xcd_map[w] == xcd_id) {
            if (w < worker_id) rank++;
            total_on_xcd++;
          }
        }
        block_xcd_local_rank = rank;
        block_workers_on_xcd = total_on_xcd;
      }
      __syncthreads();

      // Workers loop over multiple tiles when n_tile_count > workers_per_xcd
      {
        int n_tile_start = (int)task_desc->task_metadata.n_tile_start;
        int n_tile_count = (int)task_desc->task_metadata.n_tile_count;
        int my_tiles = 0;
        for (int t = block_xcd_local_rank; t < n_tile_count; t += block_workers_on_xcd) {
          int tile_idx = n_tile_start + t;
          _execute_gang_task(task_desc, config, tile_idx);
          my_tiles++;
        }
        if (threadIdx.x == 0) gang_tiles_executed = my_tiles;
      }
#else
      // Fallback for non-AMD: just execute as regular task
      _execute_task(task_desc, config);
#endif
#endif // MPK_ENABLE_GANG_TASKS
    } else {
#ifdef MPK_ENABLE_VERBOSE
      if (threadIdx.x == 0) {
        printf("[worker] _execute_task EXECUTE_TASK %d\n",
               task_desc->task_type);
      }
#endif
      _execute_task(task_desc, config);
    }
    __syncthreads();

#ifdef MPK_ENABLE_DEVICE_TASK_ACCUM
    if (threadIdx.x == 0 && g_daccum_active
        && task_desc->task_type != TASK_BEGIN_TASK_GRAPH) {
      unsigned long long _accum_dur = (__builtin_amdgcn_s_memrealtime() - _accum_t0) * 10; // ns
      int slot = 8; // OTHER
      unsigned vid = task_desc->variant_id;
      switch (task_desc->task_type) {
        case TASK_LINEAR:
        case TASK_GANG_LINEAR_MI300:
        case TASK_GANG_LINEAR_N_TILING_MI300:
        case TASK_GANG_LINEAR_MSPLIT_MI300:
          slot = (vid == 0) ? 0 : 1; break;  // QKV vs gate_up
        case TASK_LINEAR_WITH_RESIDUAL:
        case TASK_GANG_LINEAR_RES_MI300:
        case TASK_GANG_LINEAR_RES_N_TILING_MI300:
        case TASK_GANG_LINEAR_RES_MSPLIT_MI300:
        case TASK_SPLITK_LINEAR_RES_ATOMIC_MI300:
          slot = (vid == 0) ? 2 : 3; break;  // o_proj vs down_proj
        case TASK_RMS_NORM: slot = 4; break;
        case TASK_SILU_MUL: slot = 5; break;
        case TASK_PAGED_ATTENTION_SPLIT_KV_MI300:
        case TASK_PAGED_ATTENTION_CK_FMHA_SPLIT_KV_MI300:
        case TASK_GANG_ATTN_SPLIT_KV_MI300:
          slot = 6; break;
        case TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_MI300:
        case TASK_GANG_ATTN_MERGE_MI300:
          slot = 7; break;
        default: slot = 8; break;
      }
      atomicAdd(&g_daccum_ns[slot], _accum_dur);
      atomicAdd(&g_daccum_cnt[slot], 1ULL);
    }
#endif

#ifdef MPK_ENABLE_TIMING
    if (threadIdx.x == 0) {
      unsigned long long task_time = clock64() - exec_start;
      total_exec_cycles += task_time;
      total_tasks_executed++;
      // Track per-task-type timing
      switch (task_desc->task_type) {
        case TASK_LINEAR:
        case TASK_SPLITK_LINEAR_MI300:
        case TASK_SPLITK_REDUCE_MI300:
        case TASK_GANG_LINEAR_MI300:
        case TASK_GANG_LINEAR_N_TILING_MI300:
        case TASK_GANG_LINEAR_MSPLIT_MI300:
        case TASK_GANG_LINEAR_RES_MI300:
        case TASK_GANG_LINEAR_RES_N_TILING_MI300:
        case TASK_GANG_LINEAR_RES_MSPLIT_MI300:
        case TASK_GANG_LINEAR_SILU_MI300: linear_cycles += task_time; linear_count++; break;
        case TASK_LINEAR_WITH_RESIDUAL:
        case TASK_SPLITK_LINEAR_RES_ATOMIC_MI300: linear_res_cycles += task_time; linear_res_count++; break;
        case TASK_PAGED_ATTENTION_1:
        case TASK_PAGED_ATTENTION_2:
        case TASK_PAGED_ATTENTION_SPLIT_KV_MI300:
        case TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_MI300:
        case TASK_KV_CACHE_UPDATE_MI300:
        case TASK_PAGED_ATTENTION_CK_FMHA_SPLIT_KV_MI300:
        case TASK_GANG_ATTN_SPLIT_KV_MI300:
        case TASK_GANG_ATTN_MERGE_MI300: attention_cycles += task_time; attention_count++; break;
        case TASK_RMS_NORM: rms_cycles += task_time; rms_count++; break;
        case TASK_SILU_MUL: silu_cycles += task_time; silu_count++; break;
        case TASK_SILU_MUL_LINEAR_WITH_RESIDUAL: fused_cycles += task_time; fused_count++; break;
        default: break;
      }
    }
    unsigned long long signal_start = clock64();
#endif

#ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_END(task_desc->task_type, task_counter++);
    }
#endif

    // Trigger event
    if (threadIdx.x == 0) {
      EventId event_id = task_desc->trigger_event;
      size_t event_index = get_event_position_index(event_id);
      if (!is_nvshmem_event(event_id)) {
        size_t gpu_id = get_event_gpu_id(event_id);
        assert(gpu_id == config.my_gpu_id);
        // Case 1: Trigger a local non-nvshmem event
        int num_triggers = config.all_event_num_triggers[event_index];
        EventCounter count = 0;
        bool event_fired = false;

#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
        // Two-level event counting: XCD-local first, batch flush when
        // this XCD's share is done. Reduces buffer_wbl2 from ~24K to ~300.
        int xcd_slot = xcd_id * config.num_events + (int)event_index;
        int xcd_threshold = config.xcd_event_num_tasks != nullptr
                                ? config.xcd_event_num_tasks[xcd_slot]
                                : 0;
        if (xcd_threshold > 0) {
#ifdef MPK_ENABLE_GANG_TASKS
          // Gang tasks: workers accumulate tiles locally, last worker
          // on this XCD flushes 1 to global (not xcd_threshold).
          // Workers with 0 tiles must not touch the counter.
          if (gang_tiles_executed == 0 && is_gang_task_type(task_desc->task_type)) {
            // No work done — skip event signaling entirely
          } else {
            int event_increment = is_gang_task_type(task_desc->task_type)
                                      ? gang_tiles_executed : 1;
            EventCounter local_count = atom_add_local_u64(
                reinterpret_cast<unsigned long long int *>(
                    &config.xcd_local_event_counters[xcd_slot]),
                event_increment) + event_increment;
            EventCounter needed_local =
                static_cast<EventCounter>(xcd_threshold) *
                get_task_iteration_num(task_ids[queue_pos]);

            if (local_count == needed_local) {
              // Last worker/task on this XCD — flush + signal global
              threadfence_gpu();
              // Gang: flush 1 (one task done). Non-gang: flush xcd_threshold.
              int flush_amount = is_gang_task_type(task_desc->task_type)
                                     ? 1 : xcd_threshold;
              count = atom_add_release_gpu_u64(
                  &config.all_event_counters[event_index], flush_amount);
              event_fired =
                  (count + flush_amount) ==
                  static_cast<EventCounter>(num_triggers) *
                      get_task_iteration_num(task_ids[queue_pos]);
            }
          }
#else
          // Non-gang: each task increments by 1
          EventCounter local_count = atom_add_local_u64(
              reinterpret_cast<unsigned long long int *>(
                  &config.xcd_local_event_counters[xcd_slot]),
              1) + 1;
          EventCounter needed_local =
              static_cast<EventCounter>(xcd_threshold) *
              get_task_iteration_num(task_ids[queue_pos]);

          if (local_count == needed_local) {
            threadfence_gpu();
            count = atom_add_release_gpu_u64(
                &config.all_event_counters[event_index], xcd_threshold);
            event_fired =
                (count + xcd_threshold) ==
                static_cast<EventCounter>(num_triggers) *
                    get_task_iteration_num(task_ids[queue_pos]);
          }
#endif
          // else: not the last on this XCD — skip fence + global atomic
        } else
#endif
        {
          // Fallback: per-task fence + global atomic (NVIDIA, or uncounted events)
          threadfence_gpu();
          count = atom_add_release_gpu_u64(
              &config.all_event_counters[event_index], 1);
          event_fired =
              (count + 1) == static_cast<EventCounter>(num_triggers) *
                                 get_task_iteration_num(task_ids[queue_pos]);
        }

#ifdef MPK_ENABLE_VERBOSE
        printf("[%d][DONE] worker_id(%d) iter_num(%llu) task_idx(%llu) "
               "event_id(%llu) "
               "event_type(local) count(%llu)\n",
               config.my_gpu_id,
               worker_id,
               get_task_iteration_num(task_ids[queue_pos]),
               get_task_position_index(task_ids[queue_pos]),
               event_id,
               count);
#endif

        if (event_fired) {
#ifdef MPK_ENABLE_EVENT_TIMING
          if (config.event_timing_buffer != nullptr) {
            int pos = atomicAdd(config.event_timing_count, 1);
            if (pos < config.event_timing_max_entries) {
              config.event_timing_buffer[pos * 2] = (unsigned long long)event_index;
              config.event_timing_buffer[pos * 2 + 1] = __builtin_amdgcn_s_memrealtime();
            }
          }
#endif
#ifdef MPK_ENABLE_PROFILING
          PROFILER_EVENT_START(TASK_SCHD_EVENTS, task_counter);
#endif
          EventDesc event_desc = config.all_events[event_index];
          if (event_desc.event_type == EVENT_EMPTY) {
            // Do nothing for empty event
          } else {
            bool use_bcast_queue = false;
            if (event_desc.event_type == EVENT_LAUNCH_MASSIVE_TASKS ||
                event_desc.event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
              use_bcast_queue = true;
            }
            int sched_id =
                use_bcast_queue
                    ? config.num_local_schedulers + config.num_remote_schedulers
                    : get_rand_sched_id(event_index,
                                        worker_id,
                                        config.num_workers,
                                        config.num_local_schedulers,
                                        xcd_id);
            if (use_bcast_queue) {
              // Cross-XCD broadcast queue — need global ops
              size_t last_event_pos = atom_add_release_gpu_u64(
                  &config.sched_queue_next_free_event_id[sched_id], 1);
              st_relaxed_gpu_u64(
                  &config.sched_queues[sched_id][last_event_pos %
                                                 config.per_sched_queue_len],
                  event_index);
              threadfence_gpu();
              size_t old;
              do {
                old = atom_cas_release_gpu_u64(
                    &config.sched_queue_last_ready_event_id[sched_id],
                    last_event_pos,
                    last_event_pos + 1);
              } while (old != last_event_pos);
            } else {
              // Same-XCD scheduler queue — use L2-local ops
              size_t last_event_pos = atom_add_local_u64(
                  &config.sched_queue_next_free_event_id[sched_id], 1);
              st_local_u64(
                  &config.sched_queues[sched_id][last_event_pos %
                                                 config.per_sched_queue_len],
                  event_index);
              fence_local();
              size_t old;
              do {
                old = atom_cas_local_u64(
                    &config.sched_queue_last_ready_event_id[sched_id],
                    last_event_pos,
                    last_event_pos + 1);
              } while (old != last_event_pos);
            }
          }
#ifdef MPK_ENABLE_PROFILING
          PROFILER_EVENT_END(TASK_SCHD_EVENTS, task_counter++);
#endif
        }
      } else {
        // Case 2: trigger a nvshmem event
        assert(task_desc->task_type == TASK_NVSHMEM_COPY);
        // Note that nvshmem copy task signal counter during data copy
        // we don't need to do anything here is the task type is NVSHMEM_COPY
#ifdef MPK_ENABLE_VERBOSE
        printf("[%d][DONE] worker_id(%d) task_id(%llu) event_id(%llx) "
               "event_type(remote)\n",
               config.my_gpu_id,
               worker_id,
               get_task_position_index(task_ids[queue_pos]),
               event_id);
#endif
      }
    }
#ifdef MPK_ENABLE_TIMING
    if (threadIdx.x == 0) {
      total_signal_cycles += clock64() - signal_start;
    }
#endif
    queue_pos += 1;
  }
}

// Single-warp scheduler: lane 0 handles all dispatch for this XCD.
__device__ __forceinline__ void execute_scheduler(RuntimeConfig config,
                                                  int offset,
                                                  int assigned_sched_id = -1) {
  int const num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  // CANNOT use syncthreads below — only thread 0 runs the scheduler
  if (threadIdx.x == 0) {
    int const sched_id = (assigned_sched_id >= 0) ? assigned_sched_id : (blockIdx.x + offset);
    int num_sched_queues = 1;
    size_t iteration_num = 0;
    unsigned long long prev_end_of_graph_clk = 0;
    int end_of_graph_count = 0;
    EventId *sched_queues[2];
    int sched_queue_ids[2];
    sched_queues[0] = config.sched_queues[sched_id];
    sched_queue_ids[0] = sched_id;

    // Build worker list for this scheduler (all workers on this XCD)
    int my_workers[MAX_WORKER_PER_SCHEDULER];
    int my_num_workers = 0;
    int my_xcd = 0;  // Scheduler's hardware XCD ID (0 for NVIDIA)

    if (sched_id < config.num_local_schedulers) {
      // local schedulers also (collectively) process events from
      // the global queue
      sched_queues[num_sched_queues] = config.sched_queues[num_schedulers];
      sched_queue_ids[num_sched_queues] = num_schedulers;
      num_sched_queues++;

#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
      // Wait for all workers to report their XCD IDs
      if (config.worker_xcd_map != nullptr) {
        while (atomicAdd(config.worker_xcd_ready_count, 0) <
               config.num_workers) {
          __nanosleep(100);
        }
        // Fence ensures we see all worker_xcd_map writes
        threadfence_gpu();
        // Read this scheduler's XCD and collect ALL matching workers
        my_xcd = get_current_xcd_id();
        for (int w = 0; w < config.num_workers; w++) {
          if (config.worker_xcd_map[w] == my_xcd) {
            my_workers[my_num_workers++] = w;
          }
        }
        printf("[SCHED_XCD] sched_id=%d xcd=%d workers_on_xcd=%d block=%d\n",
               sched_id, my_xcd, my_num_workers, (int)blockIdx.x);
      } else
#endif
      {
        // Fallback: stride-based mapping (NVIDIA or if XCD map not available)
        for (int w = sched_id; w < config.num_workers;
             w += config.num_local_schedulers) {
          my_workers[my_num_workers++] = w;
        }
      }
    } else {
      // Remote schedulers: stride across remote worker IDs
      int remote_sched = sched_id - config.num_local_schedulers;
      for (int w = remote_sched; w < config.num_workers;
           w += config.num_remote_schedulers) {
        my_workers[my_num_workers++] = w + config.num_workers;
      }
    }
    size_t cur_event_pos[2], last_event_pos[2];
    for (int i = 0; i < 2; i++) {
      cur_event_pos[i] = 0;
      last_event_pos[i] = 0;
    }

    size_t worker_queue_next_free_task_pos[MAX_WORKER_PER_SCHEDULER];
    for (int i = 0; i < my_num_workers; i++) {
      worker_queue_next_free_task_pos[i] = 0;
    }

    int next_worker_idx = 0;
    int queue_idx = 0;
#ifdef MPK_ENABLE_DISPATCH_TIMING
    unsigned long long last_dep_dispatch_end_ns = 0;
#endif

    while (true) {
      int poll_count = 0;
      while (cur_event_pos[queue_idx] == last_event_pos[queue_idx]) {
        // queue_idx 0 = own scheduler queue (XCD-local), 1 = broadcast (cross-XCD)
        if (queue_idx == 0) {
          last_event_pos[0] = ld_local_u64(
              &config.sched_queue_last_ready_event_id[sched_queue_ids[0]]);
        } else {
          last_event_pos[queue_idx] = ld_acquire_gpu_u64(
              &config.sched_queue_last_ready_event_id[sched_queue_ids[queue_idx]]);
        }

        if (cur_event_pos[queue_idx] < last_event_pos[queue_idx]) {
          break;
        } else {
          queue_idx = (queue_idx == num_sched_queues - 1) ? 0 : queue_idx + 1;
        }
        // nanosleep to avoid overwhelming I/O
        __nanosleep(10);
      }
      // Make sure the schedule queue is not overflow
      assert(cur_event_pos[queue_idx] + config.per_sched_queue_len >
             last_event_pos[queue_idx]);
      // Read event from queue — use local ops for own queue, global for broadcast
      EventId event_id;
      if (queue_idx == 0) {
        fence_local();
        event_id = ld_local_u64(
            &sched_queues[0]
                         [cur_event_pos[0] % config.per_sched_queue_len]);
      } else {
        threadfence_gpu();
        event_id = ld_relaxed_gpu_u64(
            &sched_queues[queue_idx]
                         [cur_event_pos[queue_idx] % config.per_sched_queue_len]);
      }
      EventDesc e = config.all_events[event_id];
      if (is_termination_event(event_id, e)) {
        // terminate all workers (same XCD — use local ops)
        if (sched_id < config.num_local_schedulers) {
          for (int widx = 0; widx < my_num_workers; widx++) {
            int w = my_workers[widx];
            size_t last_task_id =
                worker_queue_next_free_task_pos[widx]++;
            st_local_u64(
                &config.worker_queues[w][last_task_id %
                                         config.per_worker_queue_len],
                0);
            fence_local();
            atom_add_local_u64(&config.worker_queue_last_ready_task_id[w],
                               1);
          }
        }
        return;
      }
      // This is the ending task of the current task graph
      if (e.event_type == EVENT_END_OF_TASK_GRAPH) {
#ifdef MPK_ENABLE_VERBOSE
          printf("[SCHD] END_OF_TASK_GRAPH\n");
#endif
          unsigned long long iter_end_clk = get_wallclock_ns();
          int num_active_tokens = config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
#ifndef MPK_ENABLE_PROFILING
          if (prev_end_of_graph_clk != 0 &&
              (end_of_graph_count < 10 || end_of_graph_count % 100 == 0)) {
#ifdef MPK_ENABLE_DISPATCH_TIMING
            double compute_after_dispatch_us = last_dep_dispatch_end_ns > 0
                ? (double)(iter_end_clk - last_dep_dispatch_end_ns) / 1000.0
                : 0.0;
            printf("[FWD_PASS] iter=%d time_ms=%.3f num_active_tokens=%d compute_after_dispatch_us=%.1f\n",
                   end_of_graph_count,
                   (double)(iter_end_clk - prev_end_of_graph_clk) / 1000000.0,
                   num_active_tokens,
                   compute_after_dispatch_us);
#else
            printf("[FWD_PASS] iter=%d time_ms=%.3f num_active_tokens=%d\n",
                   end_of_graph_count,
                   (double)(iter_end_clk - prev_end_of_graph_clk) / 1000000.0,
                   num_active_tokens);
#endif
          }
#endif
          prev_end_of_graph_clk = iter_end_clk;
          end_of_graph_count++;
          // Check if we want to continue
#ifdef MODE_ONLINE_NOTOKEN
          if (!prepare_next_batch(config, iteration_num))
#else
          if (!prepare_next_batch(config))
#endif
          {
            unsigned long long prep_done_clk = get_wallclock_ns();
#if 0 // disabled for large BS benchmarks
            printf("[ITER_TIME] sched=%d iter=%llu prep_us=%.1f DONE\n",
                   sched_id,
                   iteration_num,
                   (double)(prep_done_clk - iter_end_clk) / 1000.0);
#endif
            terminate_schedulers(config);
          } else {
            unsigned long long prep_done_clk = get_wallclock_ns();
            // Launch task 1 (begin_task_graph) for the next iteration (same XCD)
            int nw = my_workers[next_worker_idx];
            size_t last_task_id =
                worker_queue_next_free_task_pos[next_worker_idx]++;
            st_local_u64(
                &config.worker_queues[nw]
                                     [last_task_id % config.per_worker_queue_len],
                compute_task_id(iteration_num + 1, 1 /*begin_task_graph*/));
            fence_local();
            atom_add_local_u64(
                &config.worker_queue_last_ready_task_id[nw], 1);
#ifdef MPK_ENABLE_VERBOSE
            printf("[%d][SCHD]EVENT_END_OF_TASK_GRAPH schd_id(%d) "
                   "iter_num(%llu) task_idx(1) "
                   "worker_id(%d) "
                   "worker_last_ready_pos(%llu)\n",
                   config.my_gpu_id,
                   sched_id,
                   iteration_num + 1,
                   nw,
                   last_task_id + 1);
#endif
#if 0 // disabled for large BS benchmarks
            printf("[ITER_TIME] sched=%d iter=%llu prep_us=%.1f dispatch_us=%.1f\n",
                   sched_id,
                   iteration_num,
                   (double)(prep_done_clk - iter_end_clk) / 1000.0,
                   (double)(get_wallclock_ns() - prep_done_clk) / 1000.0);
#endif
            next_worker_idx = (next_worker_idx + 1) % my_num_workers;
          }
      } else if (e.event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
        iteration_num = iteration_num + 1;
        // assign event in a round-robin fashion
        // Split event across local schedulers (stride-based workers)
        assert(sched_id < config.num_local_schedulers);

#ifdef MPK_ENABLE_GANG_TASKS
        // Check if this event contains gang tasks
        bool is_gang_event = false;
        int gang_task_count = (int)(e.last_task_id - e.first_task_id);
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
        if (gang_task_count > 0 &&
            gang_task_count == config.num_xcds &&
            is_gang_task_type(config.all_tasks[e.first_task_id].task_type)) {
          is_gang_event = true;
        }
#endif

        if (is_gang_event) {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
          // Gang dispatch: each scheduler takes its XCD's task and broadcasts
          // to all workers on that XCD
          size_t my_gang_task = e.first_task_id + sched_id;
          assert(my_gang_task < e.last_task_id);

          int tiles_per_xcd = (int)config.all_tasks[my_gang_task].task_metadata.n_tile_count;
          int dispatch_count = (tiles_per_xcd < my_num_workers) ? tiles_per_xcd : my_num_workers;

          assert(dispatch_count > 0);
          assert(dispatch_count <= my_num_workers);

          // Pass 1 (first iteration): set xcd_event_num_tasks for two-level counting
          if (iteration_num == 1 && config.xcd_event_num_tasks != nullptr) {
            int xcd_base = my_xcd * config.num_events;
            EventId trigger_ev = config.all_tasks[my_gang_task].trigger_event;
            int tev_idx = (int)get_event_position_index(trigger_ev);
            config.xcd_event_num_tasks[xcd_base + tev_idx] = tiles_per_xcd;
            fence_local();
          }

          // Pass 2: Broadcast the gang task to dispatch_count workers
          for (int widx = 0; widx < dispatch_count; widx++) {
            int nw = my_workers[widx];
            size_t last_task_id =
                worker_queue_next_free_task_pos[widx]++;
            st_local_u64(
                &config.worker_queues[nw][last_task_id %
                                          config.per_worker_queue_len],
                compute_task_id(iteration_num, my_gang_task));
            fence_local();
            atom_add_local_u64(
                &config.worker_queue_last_ready_task_id[nw], 1);
          }
#endif
        } else
#endif // MPK_ENABLE_GANG_TASKS
        {
          // Normal (non-gang) dependent task dispatch

#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
        // Pass 1: Count per-event tasks for two-level event counting
        // Only needed in first iteration (task graph is identical every iteration)
        if (iteration_num == 1 && config.xcd_event_num_tasks != nullptr) {
          int xcd_base = my_xcd * config.num_events;
          size_t num_rounds =
              (e.last_task_id - e.first_task_id + config.num_workers - 1) /
              config.num_workers;
          for (size_t ri = 0; ri < num_rounds; ri++) {
            for (int widx = 0; widx < my_num_workers; widx++) {
              size_t j = my_workers[widx];
              size_t pos = e.first_task_id + ri * config.num_workers + j;
              if (pos < e.last_task_id) {
                EventId trigger_ev = config.all_tasks[pos].trigger_event;
                int tev_idx = (int)get_event_position_index(trigger_ev);
#ifdef MPK_ENABLE_GANG_TASKS
                // Gang tasks: each worker fires per tile processed
                if (is_gang_task_type(config.all_tasks[pos].task_type)) {
                  int gang_tiles = (int)config.all_tasks[pos].task_metadata.n_tile_count;
                  config.xcd_event_num_tasks[xcd_base + tev_idx] += gang_tiles;
                } else
#endif
                {
                  config.xcd_event_num_tasks[xcd_base + tev_idx] += 1;
                }
              }
            }
          }
          fence_local(); // Ensure thresholds visible to workers on same XCD
        }
#endif

        // Pass 2: Dispatch tasks to workers
        {
          size_t num_rounds =
              (e.last_task_id - e.first_task_id + config.num_workers - 1) /
              config.num_workers;

          for (size_t i = 0; i < num_rounds; i++) {
            for (int widx = 0; widx < my_num_workers; widx++) {
              size_t j = my_workers[widx];
              size_t position_index =
                  e.first_task_id + i * config.num_workers + j;
              if (position_index < e.last_task_id) {
#if defined(MPK_ENABLE_GANG_TASKS) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
                // Gang task: broadcast to workers instead of 1-to-1
                if (is_gang_task_type(config.all_tasks[position_index].task_type)) {
                  int gang_tiles = (int)config.all_tasks[position_index].task_metadata.n_tile_count;
                  int gang_dispatch = (gang_tiles < my_num_workers) ? gang_tiles : my_num_workers;
                  for (int gw = 0; gw < gang_dispatch; gw++) {
                    int gnw = my_workers[gw];
                    size_t gslot = worker_queue_next_free_task_pos[gw]++;
                    st_local_u64(
                        &config.worker_queues[gnw][gslot % config.per_worker_queue_len],
                        compute_task_id(iteration_num, position_index));
                    fence_local();
                    atom_add_local_u64(
                        &config.worker_queue_last_ready_task_id[gnw], 1);
                  }
                } else
#endif
                {
                  int nw = my_workers[next_worker_idx];
                  size_t last_task_id =
                      worker_queue_next_free_task_pos[next_worker_idx]++;
                  st_local_u64(
                      &config
                           .worker_queues[nw][last_task_id %
                                              config.per_worker_queue_len],
                      compute_task_id(iteration_num, position_index));
                  fence_local();
                  atom_add_local_u64(
                      &config.worker_queue_last_ready_task_id[nw], 1);
#ifdef MPK_TRACE_MOE_DISPATCH
                  // Trace MoE task placement (first iteration only)
                  if (iteration_num == 1) {
                    int tt = config.all_tasks[position_index].task_type;
                    if (tt == TASK_MOE_W13_LINEAR_MI300) {
                      int eo = config.all_tasks[position_index].task_metadata.expert_offset;
                      printf("[MOE_DISPATCH] sched=%d xcd=%d worker=%d pos=%d type=W13 expert_offset=%d\n",
                             sched_id, (int)my_xcd, (int)nw, (int)position_index, eo);
                    }
                  }
#endif
                  next_worker_idx = (next_worker_idx + 1) % my_num_workers;
                }
              }
            }
          }
        }
        } // end else (non-gang)
      } else {
        TaskId my_first_task = e.first_task_id, my_last_task = e.last_task_id;
        if (e.event_type == EVENT_LAUNCH_MASSIVE_TASKS) {
          // Split event across local schedulers (by sched_id)
          assert(sched_id < config.num_local_schedulers);
          get_first_last_ids(e.last_task_id - e.first_task_id,
                             config.num_local_schedulers,
                             sched_id,
                             &my_first_task,
                             &my_last_task);
          my_first_task += e.first_task_id;
          my_last_task += e.first_task_id;
        }

#if defined(MPK_ENABLE_GANG_TASKS) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
        // Check if this is a gang event (MASSIVE or LAUNCH_TASKS with gang task type)
        bool is_gang_in_else = false;
        if (my_first_task < my_last_task &&
            is_gang_task_type(config.all_tasks[my_first_task].task_type)) {
          is_gang_in_else = true;
        }

        if (is_gang_in_else) {
          // Gang dispatch: each scheduler has 1 gang task, broadcast to workers
          assert(my_last_task - my_first_task == 1 && "Expected exactly 1 gang task per scheduler");
          size_t my_gang_task = my_first_task;
          int tiles_per_xcd = (int)config.all_tasks[my_gang_task].task_metadata.n_tile_count;
          int dispatch_count = (tiles_per_xcd < my_num_workers) ? tiles_per_xcd : my_num_workers;

          // Pass 1: xcd_event_num_tasks
          if (iteration_num == 1 && config.xcd_event_num_tasks != nullptr) {
            int xcd_base = my_xcd * config.num_events;
            EventId trigger_ev = config.all_tasks[my_gang_task].trigger_event;
            int tev_idx = (int)get_event_position_index(trigger_ev);
            config.xcd_event_num_tasks[xcd_base + tev_idx] = tiles_per_xcd;
            fence_local();
          }

          // Pass 2: broadcast to dispatch_count workers
          for (int widx = 0; widx < dispatch_count; widx++) {
            int nw = my_workers[widx];
            size_t last_task_id =
                worker_queue_next_free_task_pos[widx]++;
            st_local_u64(
                &config.worker_queues[nw][last_task_id %
                                          config.per_worker_queue_len],
                compute_task_id(iteration_num, my_gang_task));
            fence_local();
            atom_add_local_u64(
                &config.worker_queue_last_ready_task_id[nw], 1);
          }
        } else
#endif
        {
        // Normal (non-gang) dispatch
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
        // Count per-event tasks for two-level event counting (first iteration)
        if (iteration_num == 1 && config.xcd_event_num_tasks != nullptr) {
          int xcd_base = my_xcd * config.num_events;
          for (size_t ti = my_first_task; ti < my_last_task; ti++) {
            EventId trigger_ev = config.all_tasks[ti].trigger_event;
            int tev_idx = (int)get_event_position_index(trigger_ev);
            config.xcd_event_num_tasks[xcd_base + tev_idx] += 1;
          }
          fence_local();
        }
#endif

        for (size_t i = my_first_task; i < my_last_task; i++) {
          int nw = my_workers[next_worker_idx];
          size_t last_task_id =
              worker_queue_next_free_task_pos[next_worker_idx]++;
          st_local_u64(
              &config.worker_queues[nw]
                                   [last_task_id % config.per_worker_queue_len],
              compute_task_id(iteration_num, i));
          fence_local();
          atom_add_local_u64(
              &config.worker_queue_last_ready_task_id[nw], 1);

#ifdef MPK_ENABLE_VERBOSE
          printf("[%d][SCHD] EXECUTE_TASK schd_id(%d) iter_num(%llu) "
                 "task_idx(%llu) "
                 "worker_id(%d) "
                 "worker_last_ready_pos(%llu)\n",
                 config.my_gpu_id,
                 sched_id,
                 iteration_num,
                 i,
                 nw,
                 last_task_id + 1);
#endif

          next_worker_idx = (next_worker_idx + 1) % my_num_workers;
        }
        } // end non-gang dispatch in else branch
      }
      cur_event_pos[queue_idx] += 1;
    }
  }
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
__global__ __launch_bounds__(WORKER_NUM_THREADS,
                             1) void persistent_kernel(RuntimeConfig config) {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // Dynamic role election: each block discovers its XCD, then
  // the first block on each XCD becomes the scheduler for that XCD.
  // All other blocks become workers. This guarantees exactly 1
  // scheduler per XCD regardless of hardware block placement.
  __shared__ int my_role;   // 0 = worker, 1 = scheduler
  __shared__ int my_id;     // worker_id or sched_id

  if (threadIdx.x == 0) {
    int my_xcd = get_current_xcd_id();
    // Try to claim this XCD as scheduler
    int old = atomicCAS(&config.xcd_scheduler_claimed[my_xcd], -1, (int)blockIdx.x);
    if (old == -1) {
      // Successfully claimed — become scheduler for this XCD
      my_role = 1;
      my_id = my_xcd;  // sched_id = XCD ID
    } else {
      // XCD already has a scheduler — become a worker
      my_role = 0;
      my_id = atomicAdd(config.dynamic_worker_id_counter, 1);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int my_xcd = get_current_xcd_id();
    if (my_role == 1) {
      printf("[ELECTION] block=%d xcd=%d -> SCHEDULER (sched_id=%d)\n",
             (int)blockIdx.x, my_xcd, my_id);
    }
  }
  __syncthreads();

  if (my_role == 0) {
    execute_worker(config, my_id);
  } else {
    execute_scheduler(config, 0, my_id);
  }
#else
  // NVIDIA: static role assignment (no XCD concept)
  persistent_checker(config);
  if (blockIdx.x < config.num_workers) {
    execute_worker(config);
  } else {
    execute_scheduler(config, -config.num_workers);
  }
#endif
}

__global__ __launch_bounds__(WORKER_NUM_THREADS,
                             1) void worker_kernel(RuntimeConfig config) {
  worker_checker(config);
  execute_worker(config);
}

__global__ void scheduler_kernel(RuntimeConfig config) {
  scheduler_checker(config);
  execute_scheduler(config, 0);
}
#pragma clang diagnostic pop

template <typename DT>
DT *gpu_malloc(size_t size) {
  void *dst_ptr;
#ifdef USE_NVSHMEM
  dst_ptr = nvshmem_malloc(size);
#else
  (void)cudaMalloc(&dst_ptr, size);
#endif
  return static_cast<DT *>(dst_ptr);
}

void gpu_free(void *ptr) {
#ifdef USE_NVSHMEM
  nvshmem_free(ptr);
#else
  (void)cudaFree(ptr);
#endif
}

// The following function will be generated by the transpiler
static void _init_persistent_kernel(std::vector<FullTaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                    std::vector<TaskId> &first_tasks,
                                    int num_gpus,
                                    int my_gpu_id);

static RuntimeConfig global_runtime_config;

// meta_tensors[0]: seq_length
// meta_tensors[1]: tokens
// meta_tensors[2]: input_tokens
// meta_tensors[3]: output_tokens
// meta_tensors[4]: new_tokens_nums
// meta_tensors[5]: prompt_length
// meta_tensors[6]: qo_indptr_buffer
// meta_tensors[7]: paged_kv_indptr_buffer
// meta_tensors[8]: paged_kv_indices_buffer
// meta_tensors[9]: paged_kv_last_page_len_buffer

extern "C" void init_request_resources() {
  init_kernel<<<dim3(1, 1, 1), dim3(INIT_NUM_THREADS, 1, 1)>>>(
      global_runtime_config);
  (void)cudaStreamSynchronize(NULL);
}

extern "C" void init_persistent_kernel(std::vector<void *> meta_tensors,
                                       void *profiler_buffer,
                                       int my_rank,
                                       int num_workers,
                                       int num_local_schedulers,
                                       int num_remote_schedulers,
                                       int max_seq_length,
                                       int total_num_requests,
                                       long long eos_token_id) {
  assert(meta_tensors.size() == 10);
  global_runtime_config.step = static_cast<int *>(meta_tensors[0]);
  global_runtime_config.tokens = static_cast<long long *>(meta_tensors[1]);
  global_runtime_config.input_tokens =
      static_cast<long long *>(meta_tensors[2]);
  global_runtime_config.output_tokens =
      static_cast<long long *>(meta_tensors[3]);
  global_runtime_config.new_token_nums = static_cast<int *>(meta_tensors[4]);
  global_runtime_config.prompt_length = static_cast<int *>(meta_tensors[5]);
  global_runtime_config.qo_indptr_buffer = static_cast<int *>(meta_tensors[6]);
  global_runtime_config.paged_kv_indptr_buffer =
      static_cast<int *>(meta_tensors[7]);
  global_runtime_config.paged_kv_indices_buffer =
      static_cast<int *>(meta_tensors[8]);
  global_runtime_config.paged_kv_last_page_len_buffer =
      static_cast<int *>(meta_tensors[9]);
  global_runtime_config.num_workers = num_workers;
  global_runtime_config.num_local_schedulers = num_local_schedulers;
  global_runtime_config.num_remote_schedulers = num_remote_schedulers;
  global_runtime_config.max_seq_length = max_seq_length;
  global_runtime_config.eos_token_id = eos_token_id;
  global_runtime_config.profiler_buffer = profiler_buffer;
  int num_schedulers = num_local_schedulers + num_remote_schedulers;

  // Initialize nvshmem
  (void)cudaSetDevice(my_rank);

#ifdef USE_NVSHMEM
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  nvshmem_barrier_all();
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  printf("mype(%d) npes(%d) mype_node(%d)\n", mype, npes, mype_node);
#else
  int mype = 0;
  int npes = 1;
#endif

#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
  global_runtime_config.request_ids =
      gpu_malloc<int>(sizeof(int) * (MPK_MAX_NUM_BATCHED_REQUESTS + 1));
  global_runtime_config.next_request_id = gpu_malloc<int>(sizeof(int));
  global_runtime_config.page_queue =
      gpu_malloc<int>(MPK_MAX_NUM_PAGES * sizeof(int));
  global_runtime_config.page_queue_head = gpu_malloc<int>(sizeof(int));
  global_runtime_config.page_queue_tail = gpu_malloc<int>(sizeof(int));
  global_runtime_config.total_num_requests = total_num_requests;
#endif
  global_runtime_config.per_worker_queue_len = 1024;
  global_runtime_config.per_sched_queue_len = 1024;
  global_runtime_config.num_gpus = npes;
  global_runtime_config.my_gpu_id = mype;
  global_runtime_config.num_graphs = 1;
  global_runtime_config.profiling_num_iters = MPK_PROFILING_NUM_ITERS;
  // Split mode: workers and schedulers in separate kernel launches.
  // Note: rocprofv3 --pmc serializes kernel dispatches and deadlocks split mode,
  // but --profiling (Perfetto trace) works fine since profiler writes are worker-only.
  global_runtime_config.split_worker_scheduler = true;

  std::vector<FullTaskDesc> all_fulltasks;
  std::vector<EventDesc> all_events;
  std::vector<TaskId> first_tasks;
  _init_persistent_kernel(all_fulltasks, all_events, first_tasks, npes, mype);
  std::vector<TaskDesc> all_tasks;
  for (auto const &ft : all_fulltasks) {
    TaskDesc task_desc(ft);
    // if (ft.task_type == TASK_PAGED_ATTENTION_SPLIT_KV_SM100 || ft.task_type
    // == TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100) {
    //   printf("ft.kv_idx %d\n", ft.kv_idx);
    //   printf("ft.merge_task_offset %d\n", ft.merge_task_offset);
    // }
    // Reinterpret part of TaskDesc to save xfer_size information
    if (ft.task_type == TASK_NVSHMEM_COPY) {
      int size_in_bytes = 2;
      for (int i = 0; i < ft.inputs[0].num_dims; i++) {
        size_in_bytes *= ft.inputs[0].dim[i];
      }
      task_desc.task_metadata.xfer_size_in_bytes = size_in_bytes;
    }
    all_tasks.push_back(task_desc);
  }

  // Initialize worker queue last task id
  // Each worker now maintains a local and a remote worker queue
  global_runtime_config.worker_queue_last_ready_task_id =
      gpu_malloc<unsigned long long int>((num_workers * 2) *
                                         sizeof(unsigned long long int));
  // std::vector<unsigned long long int> host_worker_queue_last_task_id;
  // for (int i = 0; i < 2 * num_workers; i++) {
  //   host_worker_queue_last_task_id.push_back(0);
  // }
  // cudaMemcpy(global_runtime_config.worker_queue_last_ready_task_id,
  //            host_worker_queue_last_task_id.data(),
  //            (num_workers * 2) * sizeof(unsigned long long int),
  //            cudaMemcpyHostToDevice);
  //  Initialize scheduler queue last event id
  //  We maintain one extra scheduler queue for the global scheduler
  global_runtime_config.sched_queue_last_ready_event_id =
      gpu_malloc<unsigned long long int>((num_schedulers + 1) *
                                         sizeof(unsigned long long int));
  global_runtime_config.sched_queue_next_free_event_id =
      gpu_malloc<unsigned long long int>((num_schedulers + 1) *
                                         sizeof(unsigned long long int));

  // std::vector<unsigned long long int> host_sched_queue_last_event_id;
  // for (int i = 0; i < (num_schedulers + 1); i++) {
  //   host_sched_queue_last_event_id.push_back(0);
  // }
  // cudaMemcpy(global_runtime_config.sched_queue_last_ready_event_id,
  //            host_sched_queue_last_event_id.data(),
  //            (num_schedulers + 1) * sizeof(unsigned long long int),
  //            cudaMemcpyHostToDevice);
  // cudaMemcpy(global_runtime_config.sched_queue_next_free_event_id,
  //            host_sched_queue_last_event_id.data(),
  //            (num_schedulers + 1) * sizeof(unsigned long long int),
  //            cudaMemcpyHostToDevice);
  //  Initialize all event counters
  global_runtime_config.all_event_counters =
      gpu_malloc<EventCounter>(all_events.size() * sizeof(EventCounter));
  global_runtime_config.all_event_num_triggers =
      gpu_malloc<int>(all_events.size() * sizeof(int));
  std::vector<int> host_all_event_counters;
  for (size_t i = 0; i < all_events.size(); i++) {
    host_all_event_counters.push_back(all_events.at(i).num_triggers);
  }
  (void)cudaMemcpy(global_runtime_config.all_event_num_triggers,
                   host_all_event_counters.data(),
                   all_events.size() * sizeof(int),
                   cudaMemcpyHostToDevice);
  // cudaMemset(global_runtime_config.all_event_counters,
  //            0,
  //            all_events.size() * sizeof(EventCounter));

#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300) || defined(MIRAGE_BACKEND_USE_ROCM)
  // Initialize per-XCD local event counters for hierarchical polling
  // This reduces cross-XCD atomic contention by having leaders poll global
  // and other workers poll XCD-local counters
  constexpr int NUM_XCDS_INIT = 8;  // MI300X has 8 XCDs
  global_runtime_config.num_xcds = NUM_XCDS_INIT;
  global_runtime_config.xcd_local_event_counters =
      gpu_malloc<EventCounter>(NUM_XCDS_INIT * all_events.size() * sizeof(EventCounter));
  // Initialize to 0 (same initial state as global counters)
  (void)cudaMemset(global_runtime_config.xcd_local_event_counters,
                   0,
                   NUM_XCDS_INIT * all_events.size() * sizeof(EventCounter));

  // Initialize XCD leader array (-1 means no leader yet)
  global_runtime_config.xcd_leader_worker =
      gpu_malloc<int>(NUM_XCDS_INIT * sizeof(int));
  std::vector<int> initial_leaders(NUM_XCDS_INIT, -1);
  (void)cudaMemcpy(global_runtime_config.xcd_leader_worker,
                   initial_leaders.data(),
                   NUM_XCDS_INIT * sizeof(int),
                   cudaMemcpyHostToDevice);

  // Worker XCD map: workers write their hardware XCD ID at kernel startup,
  // schedulers read it to build XCD-aligned worker lists
  global_runtime_config.worker_xcd_map =
      gpu_malloc<int>(num_workers * sizeof(int));
  (void)cudaMemset(global_runtime_config.worker_xcd_map, -1,
                   num_workers * sizeof(int));
  global_runtime_config.worker_xcd_ready_count = gpu_malloc<int>(sizeof(int));
  (void)cudaMemset(global_runtime_config.worker_xcd_ready_count, 0, sizeof(int));
  // Per-XCD per-event task thresholds for two-level event counting
  global_runtime_config.xcd_event_num_tasks =
      gpu_malloc<int>(NUM_XCDS_INIT * all_events.size() * sizeof(int));
  (void)cudaMemset(global_runtime_config.xcd_event_num_tasks, 0,
                   NUM_XCDS_INIT * all_events.size() * sizeof(int));
  // Combined kernel: XCD-based scheduler election
  global_runtime_config.xcd_scheduler_claimed =
      gpu_malloc<int>(NUM_XCDS_INIT * sizeof(int));
  (void)cudaMemset(global_runtime_config.xcd_scheduler_claimed, -1,
                   NUM_XCDS_INIT * sizeof(int));
  global_runtime_config.dynamic_worker_id_counter = gpu_malloc<int>(sizeof(int));
  (void)cudaMemset(global_runtime_config.dynamic_worker_id_counter, 0, sizeof(int));
#endif

  // Initialize event timing buffer (lightweight per-phase wall-clock timing)
#ifdef MPK_ENABLE_EVENT_TIMING
  {
    int max_entries = (int)all_events.size() * 100; // 100 iterations worth
    global_runtime_config.event_timing_max_entries = max_entries;
    global_runtime_config.event_timing_buffer =
        gpu_malloc<unsigned long long>(max_entries * 2 * sizeof(unsigned long long));
    global_runtime_config.event_timing_count = gpu_malloc<int>(sizeof(int));
    (void)cudaMemset(global_runtime_config.event_timing_count, 0, sizeof(int));
    printf("[EVENT_TIMING] Allocated buffer for %d entries (%d events x 100 iters)\n",
           max_entries, (int)all_events.size());
  }
#else
  global_runtime_config.event_timing_buffer = nullptr;
  global_runtime_config.event_timing_count = nullptr;
  global_runtime_config.event_timing_max_entries = 0;
#endif

  //  Initialize all tasks
  global_runtime_config.all_tasks =
      gpu_malloc<TaskDesc>(all_tasks.size() * sizeof(TaskDesc));
  (void)cudaMemcpy(global_runtime_config.all_tasks,
                   all_tasks.data(),
                   all_tasks.size() * sizeof(TaskDesc),
                   cudaMemcpyHostToDevice);
  // Initialize all events
  global_runtime_config.num_events = (int)all_events.size();
  global_runtime_config.all_events =
      gpu_malloc<EventDesc>(all_events.size() * sizeof(EventDesc));
  (void)cudaMemcpy(global_runtime_config.all_events,
                   all_events.data(),
                   all_events.size() * sizeof(EventDesc),
                   cudaMemcpyHostToDevice);
  // Initialize worker queues
  {
    std::vector<TaskId *> host_worker_queues;
    for (int i = 0; i < (num_workers * 2); i++) {
      TaskId *worker_queue = gpu_malloc<TaskId>(
          global_runtime_config.per_worker_queue_len * sizeof(TaskId));
      host_worker_queues.push_back(worker_queue);
    }
    global_runtime_config.worker_queues =
        gpu_malloc<TaskId *>((num_workers * 2) * sizeof(TaskId *));
    (void)cudaMemcpy(global_runtime_config.worker_queues,
                     host_worker_queues.data(),
                     (num_workers * 2) * sizeof(TaskId *),
                     cudaMemcpyHostToDevice);
  }
  // Initialize scheduler queues
  {
    std::vector<EventId *> host_sched_queues;
    for (int i = 0; i < (num_schedulers + 1); i++) {
      EventId *sched_queue = gpu_malloc<EventId>(
          global_runtime_config.per_sched_queue_len * sizeof(EventId));
      host_sched_queues.push_back(sched_queue);
    }
    global_runtime_config.sched_queues =
        gpu_malloc<EventId *>((num_schedulers + 1) * sizeof(EventId *));
    (void)cudaMemcpy(global_runtime_config.sched_queues,
                     host_sched_queues.data(),
                     (num_schedulers + 1) * sizeof(EventId *),
                     cudaMemcpyHostToDevice);
  }
  // Initialize first tasks
  {
    global_runtime_config.first_tasks =
        gpu_malloc<TaskId>(first_tasks.size() * sizeof(TaskId));
    (void)cudaMemcpy(global_runtime_config.first_tasks,
                     first_tasks.data(),
                     first_tasks.size() * sizeof(TaskId),
                     cudaMemcpyHostToDevice);
  }

  // Set configuration for kernels
  (void)cudaFuncSetAttribute(worker_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  (void)cudaFuncSetAttribute(
      scheduler_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024);
  (void)cudaFuncSetAttribute(persistent_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  // Create worker and scheduler streams
  (void)cudaStreamCreateWithFlags(&global_runtime_config.worker_stream,
                                  cudaStreamNonBlocking);
  (void)cudaStreamCreateWithFlags(&global_runtime_config.scheduler_stream,
                                  cudaStreamNonBlocking);
  // Create events
  (void)cudaEventCreateWithFlags(&global_runtime_config.prepare_done_event,
                                 cudaEventDisableTiming);
  (void)cudaEventCreateWithFlags(&global_runtime_config.worker_done_event,
                                 cudaEventDisableTiming);
  (void)cudaEventCreateWithFlags(&global_runtime_config.scheduler_done_event,
                                 cudaEventDisableTiming);

  init_request_resources();
#ifdef USE_NVSHMEM
  // Add a global barrier for all init_kernel to complete
  nvshmem_barrier_all();
#endif
}

// Entry point for C/C++
// TODO: change launch config
extern "C" void launch_persistent_kernel(cudaStream_t default_stream) {
  // Prepare next persistent kernel by resetting queue pointers
  {
    int end_of_task_graph_event_pos = global_runtime_config.num_events - 1;
    prepare_kernel<<<dim3(global_runtime_config.num_workers, 1, 1),
                     dim3(128, 1, 1),
                     0 /*smem*/,
                     default_stream>>>(global_runtime_config,
                                       end_of_task_graph_event_pos);
    (void)cudaEventRecord(global_runtime_config.prepare_done_event, default_stream);
#ifdef USE_NVSHMEM
    nvshmem_barrier_all();
#endif
  }
  int num_schedulers = global_runtime_config.num_local_schedulers +
                       global_runtime_config.num_remote_schedulers;
  if (global_runtime_config.split_worker_scheduler) {
    (void)cudaStreamWaitEvent(global_runtime_config.worker_stream,
                              global_runtime_config.prepare_done_event,
                              0);
    (void)cudaStreamWaitEvent(global_runtime_config.scheduler_stream,
                              global_runtime_config.prepare_done_event,
                              0);

    // The split kernel does not support NVSHMEM because
    // nvshmemx_collective_launch launches kernels sequentially, which blocks
    // the interaction between the worker kernel and the scheduler kernel
    worker_kernel<<<dim3(global_runtime_config.num_workers, 1, 1),
                    dim3(WORKER_NUM_THREADS, 1, 1),
                    MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/,
                    global_runtime_config.worker_stream>>>(
        global_runtime_config);

    scheduler_kernel<<<dim3(global_runtime_config.num_local_schedulers, 1, 1),
                       dim3(128, 1, 1),
                       0 /*smem*/,
                       global_runtime_config.scheduler_stream>>>(
        global_runtime_config);

    (void)cudaEventRecord(global_runtime_config.worker_done_event,
                          global_runtime_config.worker_stream);
    (void)cudaEventRecord(global_runtime_config.scheduler_done_event,
                          global_runtime_config.scheduler_stream);

    (void)cudaStreamWaitEvent(
        default_stream, global_runtime_config.worker_done_event, 0);
    (void)cudaStreamWaitEvent(
        default_stream, global_runtime_config.scheduler_done_event, 0);

    (void)cudaStreamSynchronize(global_runtime_config.worker_stream);
    (void)cudaStreamSynchronize(global_runtime_config.scheduler_stream);
  } else {
    int num_sms_to_use = global_runtime_config.num_workers + num_schedulers;
#ifdef USE_NVSHMEM
    void *args[] = {&global_runtime_config};
    nvshmemx_collective_launch((void const *)persistent_kernel,
                               dim3(num_sms_to_use, 1, 1),
                               dim3(SINGLE_KERNEL_NUM_THREADS, 1, 1),
                               args,
                               MAX_DYNAMIC_SHARED_MEMORY_SIZE /*sharedmem*/,
                               0 /*stream*/);
#else
    persistent_kernel<<<dim3(num_sms_to_use, 1, 1),
                        dim3(SINGLE_KERNEL_NUM_THREADS, 1, 1),
                        MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/>>>(
        global_runtime_config);
#endif
    (void)cudaDeviceSynchronize();
  }
}

extern "C" void finalize_persistent_kernel() {
  gpu_free(global_runtime_config.worker_queue_last_ready_task_id);
  gpu_free(global_runtime_config.sched_queue_last_ready_event_id);
  gpu_free(global_runtime_config.sched_queue_next_free_event_id);
  gpu_free(global_runtime_config.all_event_counters);
  gpu_free(global_runtime_config.all_event_num_triggers);
  gpu_free(global_runtime_config.all_tasks);
  gpu_free(global_runtime_config.all_events);
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
  gpu_free(global_runtime_config.next_request_id);
  gpu_free(global_runtime_config.page_queue);
  gpu_free(global_runtime_config.page_queue_head);
  gpu_free(global_runtime_config.page_queue_tail);
#endif
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300) || defined(MIRAGE_BACKEND_USE_ROCM)
  // Free per-XCD local event counters
  if (global_runtime_config.xcd_local_event_counters != nullptr) {
    gpu_free(global_runtime_config.xcd_local_event_counters);
  }
  if (global_runtime_config.xcd_leader_worker != nullptr) {
    gpu_free(global_runtime_config.xcd_leader_worker);
  }
  if (global_runtime_config.worker_xcd_map != nullptr) {
    gpu_free(global_runtime_config.worker_xcd_map);
  }
  if (global_runtime_config.worker_xcd_ready_count != nullptr) {
    gpu_free(global_runtime_config.worker_xcd_ready_count);
  }
  if (global_runtime_config.xcd_event_num_tasks != nullptr) {
    gpu_free(global_runtime_config.xcd_event_num_tasks);
  }
  if (global_runtime_config.xcd_scheduler_claimed != nullptr) {
    gpu_free(global_runtime_config.xcd_scheduler_claimed);
  }
  if (global_runtime_config.dynamic_worker_id_counter != nullptr) {
    gpu_free(global_runtime_config.dynamic_worker_id_counter);
  }
#endif
  int num_workers = global_runtime_config.num_workers;
  std::vector<TaskId *> host_worker_queues(num_workers * 2);
  (void)cudaMemcpy(host_worker_queues.data(),
                   global_runtime_config.worker_queues,
                   (num_workers * 2) * sizeof(TaskId *),
                   cudaMemcpyDeviceToHost);
  for (int i = 0; i < 2 * num_workers; i++) {
    gpu_free(host_worker_queues[i]);
  }
  gpu_free(global_runtime_config.worker_queues);
  int num_schedulers = global_runtime_config.num_local_schedulers +
                       global_runtime_config.num_remote_schedulers;
  std::vector<EventId *> host_sched_queues(num_schedulers + 1);
  (void)cudaMemcpy(host_sched_queues.data(),
                   global_runtime_config.sched_queues,
                   (num_schedulers + 1) * sizeof(EventId *),
                   cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_schedulers + 1; i++) {
    gpu_free(host_sched_queues[i]);
  }
  gpu_free(global_runtime_config.sched_queues);
  gpu_free(global_runtime_config.first_tasks);
#ifdef USE_NVSHMEM
  nvshmem_barrier_all();
  nvshmem_finalize();
#endif
  // Free event timing buffer
  if (global_runtime_config.event_timing_buffer != nullptr) {
    gpu_free(global_runtime_config.event_timing_buffer);
  }
  if (global_runtime_config.event_timing_count != nullptr) {
    gpu_free(global_runtime_config.event_timing_count);
  }
  // Free worker and scheduler streams
  (void)cudaEventDestroy(global_runtime_config.prepare_done_event);
  (void)cudaEventDestroy(global_runtime_config.worker_done_event);
  (void)cudaEventDestroy(global_runtime_config.scheduler_done_event);
  (void)cudaStreamDestroy(global_runtime_config.worker_stream);
  (void)cudaStreamDestroy(global_runtime_config.scheduler_stream);
}

// Copy event timing data to a host buffer provided by Python
// Returns the number of entries written
extern "C" int get_event_timing(unsigned long long *host_buffer, int max_entries) {
  if (global_runtime_config.event_timing_buffer == nullptr ||
      global_runtime_config.event_timing_count == nullptr) {
    return 0;
  }
  int count = 0;
  (void)cudaMemcpy(&count, global_runtime_config.event_timing_count,
                   sizeof(int), cudaMemcpyDeviceToHost);
  if (count > max_entries) count = max_entries;
  if (count > global_runtime_config.event_timing_max_entries)
    count = global_runtime_config.event_timing_max_entries;
  if (count > 0) {
    (void)cudaMemcpy(host_buffer, global_runtime_config.event_timing_buffer,
                     count * 2 * sizeof(unsigned long long),
                     cudaMemcpyDeviceToHost);
  }
  return count;
}
