/* Titan: real implementations of fleet's worker-state tracing macros.
 *
 * Fleet's task headers are already instrumented for exactly the failure titan
 * hits when it drives them directly: a worker parked forever in a
 * "poll until counter >= expected" loop. Fleet's own comment
 * (persistent_kernel.cuh, above mpk_ws_wait_begin) states the discrimination
 * this buys:
 *
 *   "either the producer never arrived (observed < expected, and the shortfall
 *    says how many arrivals are missing), or the waiter computed an expected
 *    value the producer will never reach (observed >= expected but the load is
 *    reading a stale cache line, or expected ran ahead by a full epoch)."
 *
 * That is the entire question for titan's layer-0 hang, and answering it by
 * bisecting layers costs a 4-minute build-and-run per bit. Fleet defines these
 * macros in persistent_kernel.cuh, which titan does not compile; titan's
 * kernels/mpk_ws_stubs.cuh no-ops them. Every stub there is #ifndef-guarded
 * specifically so a real definition can win, so this file is included ahead of
 * it and NEITHER the stub file NOR any fleet header is edited.
 *
 * The buffer is pinned HOST memory. That is the load-bearing detail: a hung
 * kernel never returns, so anything written to device memory is unreadable
 * without a device sync that will never complete. Pinned host memory is
 * coherent with the device while the kernel is still running, so the watchdog
 * thread in the launch wrapper can read a live snapshot of all 240 workers
 * mid-hang.
 *
 * Layout matches fleet's indexing exactly -- these offsets are not a choice,
 * they are what fleet's own accessors compute:
 *   [w*4 + 3]                 phase code   (mpk_ws_phase)
 *   [nworkers*4 + w*4 + 0..3] barrier id, observed, expected, spins
 *   [nworkers*8 + w*4 + 0..3] aux0, aux1, sync mask, exit mask
 * 16 ints per worker are allocated (12 used) to keep the quarters aligned.
 *
 * COST: this is debug-only and must never be enabled in a build whose latency
 * is quoted. Fleet measured the four-store variant at 2.3x (6.99 vs 3.01
 * ms/token) and 2.386 -> 2.321 ms/iter merely from compiling the runtime null
 * checks out. Enabled only by -DTITAN_WORKER_STATE.
 */
#pragma once

#ifdef TITAN_WORKER_STATE

#include <hip/hip_runtime.h>

// Pointer to pinned host memory, published once by the host before launch.
// Every block reads it; nobody writes it from the device.
__device__ int *g_ws_dev = nullptr;
__device__ int g_ws_nworkers = 0;

#define MPK_WS_WAIT_REFRESH 4096

__device__ __forceinline__ void
    titan_ws_phase(int phase, int layer, int xcd, int tid) {
  if (tid == 0 && g_ws_dev != nullptr) {
    __atomic_store_n(&g_ws_dev[blockIdx.x * 4 + 3],
                     50000000 + xcd * 100000 + phase * 1000 + (layer % 1000),
                     __ATOMIC_RELAXED);
  }
}

__device__ __forceinline__ void
    titan_ws_wait_begin(int barrier_id, int expected, int tid) {
  if (tid == 0 && g_ws_dev != nullptr) {
    int *b = g_ws_dev + g_ws_nworkers * 4 + blockIdx.x * 4;
    __atomic_store_n(&b[0], barrier_id, __ATOMIC_RELAXED);
    __atomic_store_n(&b[1], -1, __ATOMIC_RELAXED);
    __atomic_store_n(&b[2], expected, __ATOMIC_RELAXED);
    __atomic_store_n(&b[3], 0, __ATOMIC_RELAXED);
  }
}

__device__ __forceinline__ void
    titan_ws_wait_tick(int observed, int spins, int tid) {
  if (tid == 0 && g_ws_dev != nullptr) {
    int *b = g_ws_dev + g_ws_nworkers * 4 + blockIdx.x * 4;
    __atomic_store_n(&b[1], observed, __ATOMIC_RELAXED);
    __atomic_store_n(&b[3], spins, __ATOMIC_RELAXED);
  }
}

__device__ __forceinline__ void
    titan_ws_wait_aux(int a0, int a1, int tid) {
  if (tid == 0 && g_ws_dev != nullptr) {
    int *b = g_ws_dev + g_ws_nworkers * 8 + blockIdx.x * 4;
    __atomic_store_n(&b[0], a0, __ATOMIC_RELAXED);
    __atomic_store_n(&b[1], a1, __ATOMIC_RELAXED);
  }
}

// Straight-line progress mark for code that is not a spin loop. Written into
// the spins slot as a NEGATIVE so the host can tell a mark from a poll count.
__device__ __forceinline__ void titan_ws_mark(int code, int aux, int tid) {
  if (tid == 0 && g_ws_dev != nullptr) {
    int *b = g_ws_dev + g_ws_nworkers * 4 + blockIdx.x * 4;
    __atomic_store_n(
        &b[3], -(code * 100000 + (aux % 100000)), __ATOMIC_RELAXED);
  }
}

// Per-wave masks, for the thread-divergent MoE W13->W2 poll where tid 0 can
// run on while other waves of the same block are still spinning.
__device__ __forceinline__ void titan_ws_wave_exit(int wave, int tid) {
  if ((tid & 63) == 0 && g_ws_dev != nullptr) {
    int *b = g_ws_dev + g_ws_nworkers * 8 + blockIdx.x * 4;
    __atomic_fetch_or(&b[3], 1 << (wave & 31), __ATOMIC_RELAXED);
  }
}

__device__ __forceinline__ void titan_ws_wave_sync(int wave, int tid) {
  if ((tid & 63) == 0 && g_ws_dev != nullptr) {
    int *b = g_ws_dev + g_ws_nworkers * 8 + blockIdx.x * 4;
    __atomic_fetch_or(&b[2], 1 << (wave & 31), __ATOMIC_RELAXED);
  }
}

// Each wave clears its OWN bit. Not "tid 0 zeroes then __syncthreads": adding
// a __syncthreads to a region whose whole point is that it is thread-divergent
// changes the control flow being measured (fleet's note on the ix_1 capture).
__device__ __forceinline__ void titan_ws_wave_clear(int wave, int tid) {
  if ((tid & 63) == 0 && g_ws_dev != nullptr) {
    int *b = g_ws_dev + g_ws_nworkers * 8 + blockIdx.x * 4;
    __atomic_fetch_and(&b[2], ~(1 << (wave & 31)), __ATOMIC_RELAXED);
    __atomic_fetch_and(&b[3], ~(1 << (wave & 31)), __ATOMIC_RELAXED);
  }
}

// These names are what fleet's call sites use. Defined here, ahead of
// mpk_ws_stubs.cuh, so its #ifndef guards stand down.
#define MPK_WS_PHASE(phase, layer, xcd) titan_ws_phase((phase), (layer), (xcd), tid)
#define MPK_WS_WAIT_BEGIN(barrier_id, expected)                                \
  titan_ws_wait_begin((barrier_id), (expected), tid)
#define MPK_WS_WAIT_TICK(observed, spins)                                      \
  do {                                                                         \
    if (((spins) & (MPK_WS_WAIT_REFRESH - 1)) == 0) {                          \
      titan_ws_wait_tick((observed), (spins), tid);                            \
    }                                                                          \
  } while (0)
#define MPK_WS_WAIT_AUX(a0, a1, a2, a3) titan_ws_wait_aux((a0), (a1), tid)
#define MPK_WS_MARK(code, aux) titan_ws_mark((code), (aux), tid)
#define MPK_WS_WAVE_EXIT(wave) titan_ws_wave_exit((wave), tid)
#define MPK_WS_WAVE_SYNC(wave) titan_ws_wave_sync((wave), tid)
#define MPK_WS_WAVE_CLEAR(wave) titan_ws_wave_clear((wave), tid)

// MPK_WS_ON gates fleet's *inline dump* sites, which read a RuntimeConfig field
// (precomp_dbg_worker_state) that titan's GptConfig does not have. Leave it
// false; the watchdog thread on the host reads the same buffer without needing
// the kernel to cooperate.
#define MPK_WS_ON(cfg) (false)

// Ints per worker. 12 used, 16 allocated to keep the quarters aligned.
#define TITAN_WS_INTS_PER_WORKER 16

#endif // TITAN_WORKER_STATE
