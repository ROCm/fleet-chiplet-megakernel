---
name: phase-profiling
description: >-
    Measure which phase/stage of a fused HIP kernel is the bottleneck using
    s_memrealtime cycle timestamps at phase boundaries, then diff per-phase
    cycles on the host. Use when a fused or multi-stage GPU kernel has several
    logical phases and you need per-phase timing to decide which stage to
    optimize — i.e. when total kernel time from hipEvent is not enough to
    localize the hotspot.
---

# Phase Profiling with `s_memrealtime`

## When to use

You need to know which *phase* of a kernel is the bottleneck — not just the total kernel time from hipEvent. Common scenario: a fused kernel has 4-6 logical stages and you need to find which one to optimize.

## How it works

`__builtin_amdgcn_s_memrealtime()` reads the GPU's real-time clock as a `uint64_t` cycle count. It doesn't stall the pipeline and has negligible overhead (~1 cycle). Insert it at phase boundaries to get per-phase wall-clock cycles.

## Pattern

### 1. Define phase enum

```cpp
enum Phase {
    PHASE_START = 0,
    PHASE_A_DONE,
    PHASE_B_DONE,
    PHASE_C_DONE,
    PHASE_COUNT
};
```

### 2. Allocate timestamp buffer on device

```cpp
uint64_t* d_timestamps;
hipMalloc(&d_timestamps, PHASE_COUNT * sizeof(uint64_t));
```

Pass `d_timestamps` as an extra kernel argument.

### 3. Record timestamps at phase boundaries

Only one thread records (typically thread 0 of block 0). Place timestamps after barriers/syncs so you capture the full phase including memory latency.

```cpp
__global__ void my_kernel(..., uint64_t* timestamps) {
    int thread = threadIdx.x + threadIdx.y * blockDim.x;

    if (thread == 0) timestamps[PHASE_START] = __builtin_amdgcn_s_memrealtime();

    // Phase A: some work
    phase_a();
    __syncthreads();  // or smem_barrier() — sync before timestamp
    if (thread == 0) timestamps[PHASE_A_DONE] = __builtin_amdgcn_s_memrealtime();

    // Phase B: more work
    phase_b();
    __syncthreads();
    if (thread == 0) timestamps[PHASE_B_DONE] = __builtin_amdgcn_s_memrealtime();

    // ...
}
```

### 4. Read back and diff on host

```cpp
uint64_t h_ts[PHASE_COUNT];
hipMemcpy(h_ts, d_timestamps, sizeof(h_ts), hipMemcpyDeviceToHost);

printf("Phase A: %lu cycles\n", h_ts[PHASE_A_DONE] - h_ts[PHASE_START]);
printf("Phase B: %lu cycles\n", h_ts[PHASE_B_DONE] - h_ts[PHASE_A_DONE]);
```

### 5. Run many iterations, report median

GPU cycle counts are noisy. Run 100-1000 iterations and report the median, not the mean. Discard the first few iterations as warmup.

```cpp
std::vector<uint64_t> samples(NUM_ITERS);
for (int i = 0; i < NUM_ITERS; i++) {
    my_kernel<<<grid, block>>>(..., d_timestamps);
    hipDeviceSynchronize();
    hipMemcpy(h_ts, d_timestamps, sizeof(h_ts), hipMemcpyDeviceToHost);
    samples[i] = h_ts[PHASE_A_DONE] - h_ts[PHASE_START];
}
std::sort(samples.begin(), samples.end());
printf("Phase A median: %lu cycles\n", samples[NUM_ITERS / 2]);
```

## Practical tips

- **Don't instrument the production kernel.** Write a standalone `.hip` file that duplicates the kernel with timestamps. This avoids perturbing register allocation and keeps the production code clean.
- **`#define private public` trick**: If the kernel calls private methods on a class, you can expose them for instrumentation: `#define private public` before the include, `#undef private` after. Ugly but effective for one-off profiling.
- **Subclass for deep instrumentation**: If the kernel uses a class with many internal phases, subclass it and override the method to insert timestamps between each internal call.
- **Phase overhead**: Each `s_memrealtime` + conditional store adds ~2-4 cycles. With 10 phases that's 20-40 cycles — negligible for kernels taking 1000+ cycles, but be aware for micro-kernels.
- **Multi-block kernels**: Only instrument block 0. Other blocks may execute at different times due to scheduling.
- **Clock frequency**: `s_memrealtime` ticks at the GPU engine clock. On MI300X this is ~2.1 GHz. Convert cycles to microseconds with `cycles / 2100.0`.

## Example

See `utils/bench/router_phase_profile.hip` for a complete example that profiles a fused MoE router kernel with 10+ phases.
