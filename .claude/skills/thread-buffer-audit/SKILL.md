---
name: thread-buffer-audit
description: "Audit global buffers indexed by threadIdx (or a linear thread index) that may have fewer entries than the block size — a common cause of intermittent, nondeterministic GPU memory faults. Use proactively when reviewing thread-indexed global memory reads/writes, AND reactively when debugging a nondeterministic memory access fault, a fault address far from any allocated buffer, or a misleading 'Write access to a read-only page' error (especially involving prefix sums, reductions, or scatter/gather)."
---
# Thread-Indexed Buffer Audit

## When to use

A kernel indexes a global memory buffer by `threadIdx` (or a linear thread index), and the buffer might have fewer entries than the block size. This is a common source of memory faults that only appear under specific runtime conditions.

## The bug pattern

```cpp
// Buffer allocated as: hipMalloc(&buf, num_experts * sizeof(int))  // e.g., 128 ints
// Block size: 256 threads

__global__ void kernel(int* buf, int num_experts) {
    int thread = threadIdx.x + threadIdx.y * blockDim.x;
    int val = buf[thread];        // BUG: threads 128-255 read OOB
    // ... use val in prefix sum, reduction, scatter ...
    buf[thread] = 0;              // BUG: threads 128-255 write OOB
}
```

## Why it's hard to catch

- **Works most of the time**: OOB reads often land in adjacent allocations that happen to be zero. The garbage values don't affect the output because they contribute zero to prefix sums/reductions.
- **Fails nondeterministically**: When GPU memory layout changes (different allocation sizes between kernel launches, different order of `hipMalloc`/`hipFree`), the OOB reads pick up non-zero garbage. This garbage flows through prefix sums and produces wild offsets in downstream scatter/gather operations.
- **Symptom is far from cause**: The fault address is unrelated to any allocated buffer — it's the result of a prefix sum of garbage values producing an enormous offset into a different buffer.
- **Error message is misleading**: "Write access to a read-only page" makes you look at write operations, but the root cause is an OOB *read* that happened many instructions earlier.

## How to audit

For every `buffer[thread]` or `buffer[threadIdx.x]` access in a kernel:

1. **Check the buffer allocation size** — is it `block_size` entries or fewer?
2. **Check if all threads access it** — is there a `thread < N` guard?
3. **Check both reads AND writes** — a guarded read with an unguarded write (or vice versa) is still a bug.

## The fix

Guard the access with a bounds check. Use 0 (identity for addition) or the appropriate identity element for the operation:

```cpp
int val = (thread < num_experts) ? buf[thread] : 0;  // 0 = identity for add
// ... prefix sum / reduction ...
if (thread < num_experts) {
    buf[thread] = 0;  // reset only valid entries
}
```

## Debugging when you hit this class of bug

When you see a memory fault with an address far from any allocated buffer:

1. **Bisect by disabling kernel phases.** Start with an empty kernel, add phases back one at a time. The crashing phase contains the OOB access or consumes its output.
2. **Check the fault pattern**: Does it only happen when kernel parameters change between launches? That's a strong signal for OOB reads picking up stale data from a previous allocation layout.
3. **Log buffer addresses**: Print all `hipMalloc` pointers and compare to the faulting address. If the fault is in a completely different region, you're chasing a wild pointer from an OOB-corrupted index.

## Related skills

If you only have a fault address and no candidate kernel yet, start with the `gpu-fault-debugging` skill to localize the exact faulting source line first, then return here for root-cause analysis of thread-indexed OOB access.
