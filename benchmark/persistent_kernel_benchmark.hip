/* Persistent Kernel Communication Benchmark
 *
 * Demonstrates that Mirage's worker-scheduler synchronization pattern works
 * reliably at full scale on AMD MI300X using RELAXED atomics + threadfence_gpu().
 *
 * Architecture (matches Mirage exactly):
 *   - 240 worker blocks (256 threads each) - execute tasks
 *   - 64 scheduler blocks (32 threads, 4 schedulers each) - dispatch tasks
 *   - Total: 304 blocks = 304 CUs on MI300X
 *
 * Synchronization Pattern:
 *   - Scheduler → Worker: st_release (task data) + threadfence_gpu + atom_add_relaxed (signal)
 *   - Worker → Scheduler: threadfence_gpu + atom_add_relaxed (completion) + broadcast
 *   - Polling: atomic_load_relaxed + nanosleep (no L2 invalidation penalty)
 *   - Visibility: threadfence_gpu() after polling loop exits (SEQ_CST agent-scope fence)
 *
 * Why this matters:
 *   - SEQ_CST atomics on MI300X generate `buffer_inv sc0 sc1` (L2 cache invalidation)
 *   - L2 inv is expensive: ~36ms for inference iteration
 *   - RELAXED atomics skip L2 inv; threadfence_gpu() runs once per task completion
 *   - Result: ~17.6ms per inference iteration (2x speedup)
 *
 * Compile: hipcc -O3 -std=c++17 persistent_kernel_benchmark.hip -o pk_benchmark
 * Run: ./pk_benchmark
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <chrono>

// Configuration - matches Mirage on MI300X
#define NUM_WORKERS 240
#define NUM_SCHEDULER_BLOCKS 64
#define SCHEDULERS_PER_BLOCK 4
#define NUM_SCHEDULERS (NUM_SCHEDULER_BLOCKS * SCHEDULERS_PER_BLOCK)
#define TOTAL_BLOCKS (NUM_WORKERS + NUM_SCHEDULER_BLOCKS)
#define NUM_ITERATIONS 100
#define QUEUE_SIZE 256

// ============================================================================
// Atomic primitives - copied directly from Mirage's mpk_atoms.cuh
// ============================================================================

// Relaxed load for polling
__device__ __forceinline__ uint64_t ld_relaxed_gpu_u64(uint64_t* addr) {
    return __atomic_load_n(addr, __ATOMIC_RELAXED);
}

// Release store for data
__device__ __forceinline__ void st_release_gpu_u64(uint64_t* addr, uint64_t val) {
    __atomic_store_n(addr, val, __ATOMIC_RELEASE);
}

// Relaxed atomic add for signaling
__device__ __forceinline__ uint64_t atom_add_relaxed_gpu_u64(uint64_t* addr, uint64_t val) {
    return __atomic_fetch_add(addr, val, __ATOMIC_RELAXED);
}

// Memory fence for GPU scope
__device__ __forceinline__ void threadfence_gpu() {
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
}

// Sleep/yield - AMD specific
__device__ __forceinline__ void nanosleep_gpu(int ns) {
    // s_sleep takes a 4-bit immediate (0-15), each unit ~100 clocks
    // For ~10ns at 1.5GHz, use s_sleep(1)
    __builtin_amdgcn_s_sleep(1);
}

// ============================================================================
// Runtime config
// ============================================================================
struct Config {
    uint64_t* worker_queues;              // [NUM_WORKERS][QUEUE_SIZE]
    uint64_t* worker_queue_last_ready_id; // [NUM_WORKERS]
    uint64_t* iter_counter;               // [NUM_ITERATIONS]
    uint64_t* bcast_tail;                 // [1]
    uint64_t* done;                       // [NUM_WORKERS]
    int per_worker_queue_len;
};

// ============================================================================
// get_first_last_ids - copied from Mirage
// ============================================================================
__device__ void get_first_last_ids(int num_workers, int num_schedulers,
                                   int sched_id, int* first, int* last) {
    int per = num_workers / num_schedulers;
    int rem = num_workers % num_schedulers;
    if (sched_id < rem) {
        *first = sched_id * (per + 1);
        *last = *first + per + 1;
    } else {
        *first = rem * (per + 1) + (sched_id - rem) * per;
        *last = *first + per;
    }
}

// ============================================================================
// Worker - polls queue, executes task, signals completion
// ============================================================================
__device__ void worker_main(Config cfg, int worker_id) {
    __shared__ uint64_t task_ids[32];  // Buffer for tasks
    __shared__ uint64_t next_task_pos;
    __shared__ uint64_t last_task_pos;

    uint64_t* my_queue = cfg.worker_queues + worker_id * cfg.per_worker_queue_len;

    if (threadIdx.x == 0) {
        next_task_pos = 0;
        last_task_pos = 0;
    }
    __syncthreads();

    uint64_t count = 0;
    int queue_pos = 0;
    int queue_len = 0;

    while (true) {
        // Fetch next task batch if buffer empty
        if (queue_pos == queue_len) {
            if (threadIdx.x == 0) {
                // Poll for tasks
                while (next_task_pos == last_task_pos) {
                    last_task_pos = ld_relaxed_gpu_u64(&cfg.worker_queue_last_ready_id[worker_id]);
                    if (next_task_pos < last_task_pos) break;
                    nanosleep_gpu(10);
                }
                threadfence_gpu();
            }
            __syncthreads();

            // Load task IDs (multiple threads can help)
            int num_to_load = min((int)(last_task_pos - next_task_pos), 32);
            if (threadIdx.x < num_to_load) {
                task_ids[threadIdx.x] = ld_relaxed_gpu_u64(
                    &my_queue[(next_task_pos + threadIdx.x) % cfg.per_worker_queue_len]);
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                next_task_pos += num_to_load;
            }
            queue_pos = 0;
            queue_len = num_to_load;
            __syncthreads();
        }

        // Get task from buffer
        uint64_t task = task_ids[queue_pos++];
        if (task == 0) break;

        // Execute task (minimal work for benchmark)
        if (threadIdx.x == 0) {
            count++;
            uint64_t iter = task;  // Task ID = iteration number

            // Signal completion for this iteration
            threadfence_gpu();
            uint64_t prev = atom_add_relaxed_gpu_u64(&cfg.iter_counter[iter - 1], 1);

            // Last worker for this iteration broadcasts to schedulers
            if (prev + 1 == NUM_WORKERS) {
                atom_add_relaxed_gpu_u64(cfg.bcast_tail, 1);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        cfg.done[worker_id] = count;
    }
}

// ============================================================================
// Scheduler - dispatches tasks to assigned workers
// ============================================================================
__device__ void scheduler_main(Config cfg, int sched_id) {
    int first_worker, last_worker;
    get_first_last_ids(NUM_WORKERS, NUM_SCHEDULERS, sched_id, &first_worker, &last_worker);
    if (first_worker >= last_worker) return;

    int num_my_workers = last_worker - first_worker;
    uint64_t worker_pos[4] = {0, 0, 0, 0};  // Next free slot per worker
    uint64_t my_pos = 0;
    uint64_t iteration = 1;

    // Dispatch initial iteration
    for (int i = 0; i < num_my_workers; i++) {
        int w = first_worker + i;
        uint64_t pos = worker_pos[i]++;
        st_release_gpu_u64(&cfg.worker_queues[w * cfg.per_worker_queue_len + pos % cfg.per_worker_queue_len], iteration);
    }
    threadfence_gpu();
    for (int i = 0; i < num_my_workers; i++) {
        atom_add_relaxed_gpu_u64(&cfg.worker_queue_last_ready_id[first_worker + i], 1);
    }

    while (true) {
        // Poll for iteration completion (broadcast from workers)
        while (ld_relaxed_gpu_u64(cfg.bcast_tail) <= my_pos) {
            nanosleep_gpu(10);
        }
        threadfence_gpu();
        my_pos++;

        iteration++;
        if (iteration > NUM_ITERATIONS) {
            // Terminate workers
            for (int i = 0; i < num_my_workers; i++) {
                int w = first_worker + i;
                uint64_t pos = worker_pos[i]++;
                st_release_gpu_u64(&cfg.worker_queues[w * cfg.per_worker_queue_len + pos % cfg.per_worker_queue_len], 0);
            }
            threadfence_gpu();
            for (int i = 0; i < num_my_workers; i++) {
                atom_add_relaxed_gpu_u64(&cfg.worker_queue_last_ready_id[first_worker + i], 1);
            }
            return;
        }

        // Dispatch next iteration
        for (int i = 0; i < num_my_workers; i++) {
            int w = first_worker + i;
            uint64_t pos = worker_pos[i]++;
            st_release_gpu_u64(&cfg.worker_queues[w * cfg.per_worker_queue_len + pos % cfg.per_worker_queue_len], iteration);
        }
        threadfence_gpu();
        for (int i = 0; i < num_my_workers; i++) {
            atom_add_relaxed_gpu_u64(&cfg.worker_queue_last_ready_id[first_worker + i], 1);
        }
    }
}

// ============================================================================
// Main kernel - dispatches to worker or scheduler
// ============================================================================
__global__ void persistent_kernel(Config cfg) {
    if (blockIdx.x < NUM_WORKERS) {
        worker_main(cfg, blockIdx.x);
    } else {
        // Scheduler blocks: 4 schedulers per block (one per warp)
        int warp_id = threadIdx.x / 32;
        if (threadIdx.x % 32 == 0 && warp_id < SCHEDULERS_PER_BLOCK) {
            int sched_id = (blockIdx.x - NUM_WORKERS) * SCHEDULERS_PER_BLOCK + warp_id;
            if (sched_id < NUM_SCHEDULERS) {
                scheduler_main(cfg, sched_id);
            }
        }
    }
}

// ============================================================================
// Host
// ============================================================================
int main() {
    printf("=== Persistent Kernel Benchmark (Mirage Pattern) ===\n");
    printf("Workers: %d, Schedulers: %d, Iterations: %d\n",
           NUM_WORKERS, NUM_SCHEDULERS, NUM_ITERATIONS);
    printf("Sync: RELAXED atomics + threadfence_gpu()\n\n");

    Config cfg;
    cfg.per_worker_queue_len = QUEUE_SIZE;

    hipMalloc(&cfg.worker_queues, NUM_WORKERS * QUEUE_SIZE * sizeof(uint64_t));
    hipMalloc(&cfg.worker_queue_last_ready_id, NUM_WORKERS * sizeof(uint64_t));
    hipMalloc(&cfg.iter_counter, NUM_ITERATIONS * sizeof(uint64_t));
    hipMalloc(&cfg.bcast_tail, sizeof(uint64_t));
    hipMalloc(&cfg.done, NUM_WORKERS * sizeof(uint64_t));

    hipMemset(cfg.worker_queues, 0, NUM_WORKERS * QUEUE_SIZE * sizeof(uint64_t));
    hipMemset(cfg.worker_queue_last_ready_id, 0, NUM_WORKERS * sizeof(uint64_t));
    hipMemset(cfg.iter_counter, 0, NUM_ITERATIONS * sizeof(uint64_t));
    hipMemset(cfg.bcast_tail, 0, sizeof(uint64_t));
    hipMemset(cfg.done, 0, NUM_WORKERS * sizeof(uint64_t));

    printf("Launching %d blocks (%d workers + %d scheduler blocks)...\n",
           TOTAL_BLOCKS, NUM_WORKERS, NUM_SCHEDULER_BLOCKS);

    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(persistent_kernel, dim3(TOTAL_BLOCKS), dim3(256), 0, 0, cfg);
    hipError_t err = hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    if (err != hipSuccess) {
        printf("Error: %s\n", hipGetErrorString(err));
        return 1;
    }

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("\nFinished in %.2f ms\n", ms);
    printf("Time per iteration: %.2f ms\n", ms / NUM_ITERATIONS);

    // Verify
    uint64_t* done = new uint64_t[NUM_WORKERS];
    hipMemcpy(done, cfg.done, NUM_WORKERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

    uint64_t total = 0;
    int failed = 0;
    for (int i = 0; i < NUM_WORKERS; i++) {
        total += done[i];
        if (done[i] != NUM_ITERATIONS) {
            if (failed < 5) {
                printf("Worker %d: %lu (expected %d)\n", i, done[i], NUM_ITERATIONS);
            }
            failed++;
        }
    }

    printf("\nTotal tasks: %lu (expected %d)\n", total, NUM_WORKERS * NUM_ITERATIONS);
    printf("Result: %s\n", failed == 0 ? "PASSED" : "FAILED");

    delete[] done;
    hipFree(cfg.worker_queues);
    hipFree(cfg.worker_queue_last_ready_id);
    hipFree(cfg.iter_counter);
    hipFree(cfg.bcast_tail);
    hipFree(cfg.done);

    return failed == 0 ? 0 : 1;
}
