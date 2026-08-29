/* Fleet MK: Barrier primitives for 8-phase fused transformer layer
 *
 * Extracted from mirage's gang_full_layer_fused_mi300.cuh and adapted
 * as standalone reusable barrier functions.
 *
 * Three barrier types:
 *   1. Per-XCD epoch barrier (QKV completion within one XCD)
 *   2. Per-XCD chunk barrier (attention chunk completion within one XCD)
 *   3. Cross-XCD global barrier (attention completion across all XCDs)
 *
 * Counter buffer layout (per layer):
 *   counter_buf[layer * COUNTERS_PER_LAYER + 0..17]     : oproj counters (18 slots x 16 bytes)
 *   counter_buf[layer * COUNTERS_PER_LAYER + 19*16]     : attn_global_counter (cross-XCD)
 *   counter_buf[layer * COUNTERS_PER_LAYER + 20*16 + xcd*16] : qkv_epoch[xcd] (per-XCD)
 *   counter_buf[layer * COUNTERS_PER_LAYER + 28*16 + xcd*16] : chunk_barrier[xcd] (per-XCD)
 *   counter_buf[layer * COUNTERS_PER_LAYER + 10*16 + xcd*16] : routing_ready[xcd] (per-XCD)
 */
#pragma once
#include "common.cuh"

namespace fleet_mk {

// Counter buffer slot offsets (matching mirage's gang_full_layer_fused_mi300.cuh)
// Slots 0..17: oproj HierBarrier (xcd_arrive[0..7], global_arrive, xcd_release[0..7], topk_counter)
// Slots 20..27: qkv_epoch[0..7] per-XCD epoch flags
// Slots 28..35: chunk_barrier[0..7] per-XCD chunk arrival
// Slots 36..43: routing_ready[0..8] per-XCD routing release
// Slots 48..64: HierBarrier for cross-XCD attention sync (272 int32 = 17 slots)
// Slot 66: grid barrier (inter-layer sync)
static constexpr int QKV_EPOCH_SLOT             = 20 * 16;
static constexpr int CHUNK_BARRIER_SLOT         = 28 * 16;
static constexpr int ATTN_XCD_RELEASE_SLOT      = 36 * 16;  // per-XCD release flags for cross-XCD attn barrier
static constexpr int HIER_BARRIER_SLOT          = 48 * 16;

// Total counter slots per layer
// HierBarrier at slot 48 needs 17 cachelines = 17*16 ints, ending at slot 65.
// Grid barrier at slot 66. Phase B→C barrier at slot 67. Total: 69 slots.
static constexpr int GRID_BARRIER_SLOT          = 66 * 16;
static constexpr int PHASE_BC_BARRIER_SLOT      = 67 * 16;
static constexpr int COUNTERS_PER_LAYER         = 69 * 16;

// ============================================================================
// Barrier 1: Per-XCD QKV epoch barrier
// ============================================================================
// Used after Phase 1 (QKV GEMM) to synchronize all QKV workers within one XCD.
// Pattern:
//   - Each QKV worker atomically increments the per-XCD arrival counter
//   - Last worker resets the counter and bumps the epoch
//   - All workers poll the epoch counter until it reaches expected value
//
// This is L2-coherent within an XCD, so no cross-XCD traffic.

struct QkvBarrierState {
  int expected;  // expected epoch value (computed once at layer start)
};

__device__ __forceinline__ QkvBarrierState
qkv_barrier_init(int *qkv_epoch, int xcd_id) {
  QkvBarrierState state;
  state.expected = __atomic_load_n(&qkv_epoch[xcd_id * 16], __ATOMIC_RELAXED) + 1;
  return state;
}

__device__ __forceinline__ void
qkv_barrier_arrive_and_wait(
    int *arrival_counter,  // input_ptrs[7] (qkv_barrier), indexed by xcd_id
    int *qkv_epoch,        // counter_buf + QKV_EPOCH_SLOT
    int xcd_id,
    int total_qkv_tiles_per_xcd,
    int expected_epoch,
    int tid)
{
  // Arrive: atomically increment per-XCD arrival counter
  __shared__ int s_prev;
  if (tid == 0) {
    s_prev = atom_add_release_gpu_s32(&arrival_counter[xcd_id], 1);
  }
  __syncthreads();

  if (s_prev == total_qkv_tiles_per_xcd - 1) {
    // Last QKV worker: reset arrival counter, bump epoch
    if (tid == 0) {
      st_wt_u32((void *)&arrival_counter[xcd_id], 0u);
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      atom_add_release_gpu_s32(&qkv_epoch[xcd_id * 16], 1);
    }
  }

  // Wait: poll epoch counter until it reaches expected value
  if (tid == 0) {
    while (__atomic_load_n(&qkv_epoch[xcd_id * 16], __ATOMIC_RELAXED) < expected_epoch) {
      __builtin_amdgcn_s_sleep(1);
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
  __syncthreads();
  asm volatile("buffer_inv" ::: "memory");
}

// ============================================================================
// Barrier 2: Per-XCD chunk barrier (attention chunks)
// ============================================================================
// Used after Phase 3 (parallel attention chunks) to determine when all
// chunks within an XCD have completed. Last chunk worker runs merge.

__device__ __forceinline__ bool
chunk_barrier_arrive(
    int *chunk_barrier,  // counter_buf + CHUNK_BARRIER_SLOT
    int xcd_id,
    int num_kv_chunks,
    int tid)
{
  __syncthreads();
  __shared__ int s_chunk_prev;
  if (tid == 0) {
    s_chunk_prev = atom_add_release_gpu_s32(
        &chunk_barrier[xcd_id * 16], 1);
  }
  __syncthreads();

  if (s_chunk_prev == num_kv_chunks - 1) {
    // Last chunk: reset barrier
    if (tid == 0) {
      st_wt_u32((void *)&chunk_barrier[xcd_id * 16], 0u);
    }
    return true;  // caller should run merge
  }
  return false;
}

// ============================================================================
// Barrier 3: Cross-XCD global barrier (attention completion)
// ============================================================================
// Used after Phase 5 (merge) to synchronize all 8 XCDs. Each XCD's merge
// worker signals completion; all O-proj workers poll until all XCDs done.

__device__ __forceinline__ void
cross_xcd_barrier_signal(int *attn_global) {
  atom_add_release_gpu_s32(attn_global, 1);
}

__device__ __forceinline__ void
cross_xcd_barrier_wait(int *attn_global, int expected, int tid) {
  if (tid == 0) {
    while (__atomic_load_n(attn_global, __ATOMIC_RELAXED) < expected) {
      __builtin_amdgcn_s_sleep(1);
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
  __syncthreads();
  asm volatile("buffer_inv" ::: "memory");
}

// ============================================================================
// Barrier 4: Routing-ready barrier (TopK completion)
// ============================================================================
// Per-XCD: O-proj+TopK signals when routing indices are ready for MoE.

__device__ __forceinline__ void
routing_barrier_wait(int *routing_ready, int xcd_id, int expected, int tid) {
  if (tid == 0) {
    int *my_release = &routing_ready[(1 + xcd_id) * 16];
    while (ld_nt_s32(my_release) < expected) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  asm volatile("buffer_inv" ::: "memory");
}

// ============================================================================
// Inter-layer grid-wide barrier
// ============================================================================
// Between transformer layers, ensures ALL 240 threadblocks have finished
// layer N before any can start layer N+1. Uses a per-layer atomic counter
// in the counter buffer.
//
// Pattern: sense-reversing barrier.
// - Each layer has its own arrival counter at a fixed slot.
// - Thread 0 of each block atomically increments the counter.
// - All blocks spin-wait until the counter reaches total_workers.
// - The counter is NOT reset (monotonic), so layer N+1's barrier
//   uses a different counter slot.


__device__ __forceinline__ void
inter_layer_barrier(int tid, int *grid_barrier_counter, int total_workers) {
  // Ensure all in-flight stores (MoE atomicAdd, KV cache st_wt, etc.) complete.
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");

  // Agent-scope fence: makes all prior writes visible to other CUs/XCDs.
  // NOT system-scope — we don't need CPU visibility, just GPU-wide.
  // MoE atomicAdd already writes through to L2/HBM (atomic ops are coherent).
  // O-proj and QKV use write-through stores (st_wt) that bypass L2.
  // So no buffer_wbl2 needed — data is already in HBM.
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();

  // Arrive: atomically increment global arrival counter (cross-XCD coherent)
  if (tid == 0) {
    atom_add_release_gpu_s32(grid_barrier_counter, 1);
  }

  // Wait: poll until all workers arrive
  if (tid == 0) {
    while (__atomic_load_n(grid_barrier_counter, __ATOMIC_ACQUIRE) < total_workers) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();

  // Invalidate L2 so reads see fresh HBM values.
  // No buffer_wbl2 needed: atomicAdd and st_wt stores are already in HBM.
  asm volatile("buffer_inv" ::: "memory");
  __syncthreads();
}

// Lightweight version: no grid sync, just memory coherence.
// Only safe when ALL shared buffers self-reset within each layer.
__device__ __forceinline__ void
inter_layer_fence(int tid) {
  if (tid == 0) {
    threadfence_gpu();
  }
  __syncthreads();
  asm volatile("buffer_inv" ::: "memory");
}

} // namespace fleet_mk
