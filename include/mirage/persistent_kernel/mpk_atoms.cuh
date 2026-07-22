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

// PERF TEST flags (uncomment to test):
// #define MPK_DISABLE_THREADFENCE  // Disable threadfence_gpu()
#define MPK_USE_RELAXED_ATOMICS
#define MPK_USE_NT_MEMORY  // Use non-temporal loads/stores (bypass cache, fence for ordering)
// #define MPK_ENABLE_TIMING  // Enable timing instrumentation
// #define MPK_DISABLE_LINEAR  // Disable LINEAR kernel entirely

// LINEAR kernel section flags (uncomment one at a time to find bottleneck):
// #define MPK_SKIP_LOAD_INPUT   // Skip loading input from global memory
// #define MPK_SKIP_LOAD_WEIGHT  // Skip loading weight from global memory
// #define MPK_SKIP_MFMA         // Skip MFMA computation
// #define MPK_SKIP_STORE_OUTPUT // Skip storing output to global memory

#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
#include <atomic>
#endif

// =============================================================================
// Non-Temporal (NT) Load/Store for MI300X
// =============================================================================
//
// NT loads/stores bypass both L1 and L2 caches, reading/writing directly
// to HBM memory. This avoids relying on cache coherence protocol.
//
// Assembly flags:
//   global_load_dwordx2 %0, %1, off nt
//   global_store_dwordx2 %0, %1, off nt
//
// On MI300X (gfx942), the 'nt' (non-temporal) flag sets:
//   - L1: sc1=1 or NT=1 → MISS EVICT (bypasses L1)
//   - L2: sc0=1 or NT=1 → HIT STREAM / Cache Bypass
//
// Why fence is still needed with NT:
//   NT provides cache bypass but NOT memory ordering. Without fence:
//   1. Hardware can reorder NT loads (read data before flag)
//   2. Compiler can reorder code (asm volatile helps but isn't enough)
//
// The pattern is:
//   Producer: st_nt(data) → fence → st_nt(flag)
//   Consumer: ld_nt(flag) → fence → ld_nt(data)
//
// The fence (SEQ_CST) ensures:
//   - Producer's data write completes before flag write
//   - Consumer's flag read completes before data read
//
// Performance: NT + fence ≈ RELAXED atomics + fence (~17.4ms per token)
// Both are valid approaches; NT is more explicit about bypassing cache.
// =============================================================================

// Non-temporal load (bypasses cache, reads from memory)
__device__ __forceinline__ unsigned long long int
    ld_nt_u64(unsigned long long int *addr) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  unsigned long long int val;
  asm volatile(
      "global_load_dwordx2 %0, %1, off nt\n"
      "s_waitcnt vmcnt(0)"
      : "=v"(val) : "v"(addr) : "memory"
  );
  return val;
#else
  // NVIDIA: use volatile for non-cached access
  return *reinterpret_cast<volatile unsigned long long int*>(addr);
#endif
}

// Non-temporal store (bypasses cache, writes to memory)
__device__ __forceinline__ void
    st_nt_u64(unsigned long long int *addr, unsigned long long int val) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  asm volatile(
      "global_store_dwordx2 %0, %1, off nt\n"
      "s_waitcnt vmcnt(0)"
      : : "v"(addr), "v"(val) : "memory"
  );
#else
  // NVIDIA: use volatile for non-cached access
  *reinterpret_cast<volatile unsigned long long int*>(addr) = val;
#endif
}

// =============================================================================
// Write-Through (WT) Store for MI300X
// =============================================================================
// sc0=1, sc1=1 on stores → "Coherent_Cache_Bypass" in L2
// Data bypasses L2 entirely, writes directly to HBM/MALL.
// No buffer_wbl2 needed after WT stores — data is already in memory.
// Use s_waitcnt vmcnt(0) to ensure stores complete before signaling.

// Write-through 64-bit store (4x bf16 or 2x float)
__device__ __forceinline__ void
    st_wt_u64(void *addr, unsigned long long int val) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  asm volatile(
      "global_store_dwordx2 %0, %1, off sc0 sc1"
      : : "v"(addr), "v"(val) : "memory"
  );
#else
  *reinterpret_cast<volatile unsigned long long int*>(addr) = val;
#endif
}

// Write-through 32-bit store (1x float or 2x bf16)
__device__ __forceinline__ void
    st_wt_u32(void *addr, unsigned int val) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  asm volatile(
      "global_store_dword %0, %1, off sc0 sc1"
      : : "v"(addr), "v"(val) : "memory"
  );
#else
  *reinterpret_cast<volatile unsigned int*>(addr) = val;
#endif
}

// Write-through 16-bit store (1x bf16)
__device__ __forceinline__ void
    st_wt_u16(void *addr, unsigned short val) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  asm volatile(
      "global_store_short %0, %1, off sc0 sc1"
      : : "v"(addr), "v"(val) : "memory"
  );
#else
  *reinterpret_cast<volatile unsigned short*>(addr) = val;
#endif
}

__device__ __forceinline__ int atom_add_release_gpu_s32(int *addr, int val) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  // Inline asm: no compiler-generated buffer_wbl2/buffer_inv around atomic.
  // Ordering provided by explicit threadfence_gpu() before this call.
  // sc0 sc1 required on GFX942 for cross-CU atomic visibility.
  int old_val;
  asm volatile(
      "flat_atomic_add %0, %1, %2 sc0 sc1\n"
      "s_waitcnt vmcnt(0) lgkmcnt(0)"
      : "=v"(old_val) : "v"(addr), "v"(val) : "memory"
  );
  return old_val;
#else
  int old_val;
  asm volatile("atom.add.release.gpu.s32 %0,[%1],%2;"
               : "=r"(old_val)
               : "l"(addr), "r"(val)
               : "memory");
  return old_val;
#endif
}

__device__ __forceinline__ unsigned long long int
    atom_add_release_gpu_u64(unsigned long long int *addr,
                             unsigned long long int val) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  // Inline asm: no compiler-generated buffer_wbl2/buffer_inv around atomic.
  // Ordering provided by explicit threadfence_gpu() before this call.
  // sc0 sc1 required on GFX942 for cross-CU atomic visibility.
  unsigned long long int old_val;
  asm volatile(
      "flat_atomic_add_x2 %0, %1, %2 sc0 sc1\n"
      "s_waitcnt vmcnt(0) lgkmcnt(0)"
      : "=v"(old_val) : "v"(addr), "v"(val) : "memory"
  );
  return old_val;
#else
  unsigned long long int old_val;
  asm volatile("atom.add.release.gpu.u64 %0,[%1],%2;"
               : "=l"(old_val)
               : "l"(addr), "l"(val)
               : "memory");
  return old_val;
#endif
}

__device__ __forceinline__ unsigned long long int
    atom_cas_release_gpu_u64(unsigned long long int *addr,
                             unsigned long long int cmp,
                             unsigned long long int val) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  // Inline asm: no compiler-generated buffer_wbl2/buffer_inv around atomic.
  // Ordering provided by explicit threadfence_gpu() before this call.
  // sc0 sc1 required on GFX942 for cross-CU atomic visibility.
  // flat_atomic_cmpswap_x2 layout: {swap[63:0], compare[63:0]} in 4 VGPRs.
  typedef unsigned long long v2u64 __attribute__((ext_vector_type(2)));
  v2u64 cmp_swap;
  cmp_swap[0] = val;  // swap value (low 64 bits) — new value to write
  cmp_swap[1] = cmp;  // compare value (high 64 bits) — expected old value
  unsigned long long int old_val;
  asm volatile(
      "flat_atomic_cmpswap_x2 %0, %1, %2 sc0 sc1\n"
      "s_waitcnt vmcnt(0) lgkmcnt(0)"
      : "=v"(old_val) : "v"(addr), "v"(cmp_swap) : "memory"
  );
  return old_val;
#else
  unsigned long long int old_val;
  asm volatile("atom.cas.release.gpu.b64 %0,[%1],%2,%3;"
               : "=l"(old_val)
               : "l"(addr), "l"(cmp), "l"(val)
               : "memory");
  return old_val;
#endif
}

__device__ __forceinline__ unsigned long long int
    ld_acquire_gpu_u64(unsigned long long int *addr) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
#ifdef MPK_USE_NT_MEMORY
  // Non-temporal load bypasses cache, reads from memory
  return ld_nt_u64(addr);
#elif defined(MPK_USE_RELAXED_ATOMICS)
  return __atomic_load_n(addr, __ATOMIC_RELAXED);
#else
  // HIP/AMD: Use SEQ_CST for cross-XCD visibility on MI300
  return __atomic_load_n(addr, __ATOMIC_SEQ_CST);
#endif
#else
  unsigned long long int val;
  asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(val) : "l"(addr));
  return val;
#endif
}

__device__ __forceinline__ unsigned long long int
    ld_acquire_sys_u64(unsigned long long int *addr) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
#ifdef MPK_USE_RELAXED_ATOMICS
  return __atomic_load_n(addr, __ATOMIC_RELAXED);
#else
  // HIP/AMD: Use SEQ_CST for cross-XCD visibility on MI300
  return __atomic_load_n(addr, __ATOMIC_SEQ_CST);
#endif
#else
  unsigned long long int val;
  asm volatile("ld.acquire.sys.u64 %0, [%1];"
               : "=l"(val)
               : "l"(addr)
               : "memory");
  return val;
#endif
}

__device__ __forceinline__ unsigned long long int
    ld_relaxed_gpu_u64(unsigned long long int *addr) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
#ifdef MPK_USE_NT_MEMORY
  // Non-temporal load bypasses cache, reads from memory
  return ld_nt_u64(addr);
#else
  // HIP/AMD: use __atomic_load_n with relaxed ordering
  return __atomic_load_n(addr, __ATOMIC_RELAXED);
#endif
#else
  unsigned long long int val;
  asm volatile("ld.relaxed.gpu.u64 %0, [%1];" : "=l"(val) : "l"(addr));
  return val;
#endif
}

__device__ __forceinline__ void st_relaxed_gpu_u64(unsigned long long int *addr,
                                                   unsigned long long int val) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
#ifdef MPK_USE_NT_MEMORY
  // Non-temporal store bypasses cache, writes to memory
  st_nt_u64(addr, val);
#else
  // Use RELAXED for stores - threadfence_gpu() after provides ordering
  __atomic_store_n(addr, val, __ATOMIC_RELAXED);
#endif
#else
  asm volatile("st.relaxed.gpu.u64 [%0], %1;" : : "l"(addr), "l"(val));
#endif
}

// Memory fence for GPU scope - ensures all previous memory operations are visible
// Define MPK_DISABLE_THREADFENCE to disable for performance testing
__device__ __forceinline__ void threadfence_gpu() {
#ifdef MPK_DISABLE_THREADFENCE
  // Disabled for performance testing - results may be incorrect
  (void)0;
#elif defined(__HIP_DEVICE_COMPILE__) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  // MI300X: L2 NOT coherent across XCDs — buffer_wbl2 required.
  // Agent scope (sc1) sufficient — no need for system scope (sc0 sc1).
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
#else
  __threadfence();
#endif
}

// =========================================================================
// Intra-XCD (same L2 partition) memory operations.
// On MI300X, all CUs within an XCD share the same 32MB L2 cache.
// Regular stores are immediately visible to regular loads on the same XCD.
// No buffer_wbl2 (L2→HBM writeback) or NT loads needed.
// =========================================================================

// Regular volatile store through L2 — visible to same-XCD CUs immediately
__device__ __forceinline__ void st_local_u64(unsigned long long int *addr,
                                              unsigned long long int val) {
  *((volatile unsigned long long int *)addr) = val;
}

// Regular volatile load from L2 — sees same-XCD stores immediately
__device__ __forceinline__ unsigned long long int
    ld_local_u64(unsigned long long int const *addr) {
  return *((volatile unsigned long long int const *)addr);
}

// Compiler fence — ensures store ordering without buffer_wbl2.
// L2 coherency within an XCD means no hardware fence is needed.
__device__ __forceinline__ void fence_local() {
  asm volatile("" ::: "memory");
}

// Device-scope atomicAdd — sufficient for intra-XCD communication.
// No sc0/sc1 system-scope qualifiers needed.
__device__ __forceinline__ unsigned long long int
    atom_add_local_u64(unsigned long long int *addr,
                       unsigned long long int val) {
  return atomicAdd(addr, val);
}

// Device-scope atomicCAS — sufficient for intra-XCD communication.
__device__ __forceinline__ unsigned long long int
    atom_cas_local_u64(unsigned long long int *addr,
                       unsigned long long int cmp,
                       unsigned long long int val) {
  return atomicCAS(addr, cmp, val);
}
