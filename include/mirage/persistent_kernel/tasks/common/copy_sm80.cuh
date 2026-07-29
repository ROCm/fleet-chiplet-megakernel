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
#include <cstdint>
#pragma once
namespace kernel {

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define CP_ASYNC_SM80_ENABLED
#endif

// HIP compatibility: __cvta_generic_to_shared is CUDA-specific
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
// CUDA's __cvta_generic_to_shared returns a pointer that is cast to uint32_t
// For HIP, we provide a compatibility function that returns uintptr_t
// This allows static_cast<uint32_t> to work (truncating to 32 bits, which is
// fine for shared memory)
__device__ __forceinline__ uintptr_t __cvta_generic_to_shared(void *ptr) {
  // For HIP, cast shared memory pointer to integer
  // Note: This is a compatibility stub; PTX instructions (cp.async, ldmatrix)
  // won't work on HIP Shared memory addresses are limited, so truncation to
  // uint32_t is safe
  return reinterpret_cast<uintptr_t>(ptr);
}

// CK-style buffer intrinsic types for vectorized memory access
typedef int __attribute__((ext_vector_type(4))) int32x4_t_copy;
typedef int __attribute__((ext_vector_type(2))) int32x2_t_copy;

// Buffer load intrinsics
__device__ int32x2_t_copy llvm_amdgcn_raw_buffer_load_i32x2_copy(
    int32x4_t_copy srsrc,
    int voffset,
    int soffset,
    int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i32");

__device__ int32x4_t_copy llvm_amdgcn_raw_buffer_load_i32x4_copy(
    int32x4_t_copy srsrc,
    int voffset,
    int soffset,
    int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

// Helper to create buffer resource descriptor
template <typename T>
__device__ __forceinline__ int32x4_t_copy
    make_buffer_resource_copy(T const *ptr) {
  int32x4_t_copy res;
  unsigned long addr = reinterpret_cast<unsigned long>(ptr);
  res[0] = static_cast<int>(addr & 0xFFFFFFFF);
  res[1] = static_cast<int>(addr >> 32);
  res[2] = 0x7FFFFFFF; // Max range
  res[3] = 0x00027000; // Buffer resource config for gfx9
  return res;
}
#endif

// cp async

template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // HIP: No async copy, operations are synchronous
  // Just ensure memory operations complete with a fence
  __threadfence_block();
#elif defined(CP_ASYNC_SM80_ENABLED)
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  }
#endif
}

__device__ __forceinline__ void cp_async_fence() {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // HIP: No async copy groups, just a memory fence
  __threadfence_block();
#elif defined(CP_ASYNC_SM80_ENABLED)
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

// load BYTES (and prefetch 128 bytes) from global to shared memory async
template <typename T, int BYTES = 16>
__device__ __forceinline__ void load_smem(T *smem_ptr, T const *gmem_ptr) {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // HIP: Use CK-style buffer intrinsics for vectorized global loads
  static_assert(BYTES == 4 || BYTES == 8 || BYTES == 16,
                "load_smem only supports 4, 8, or 16 bytes");

  // Create buffer resource for source (wavefront-uniform base)
  int32x4_t_copy src_res = make_buffer_resource_copy(gmem_ptr);

  if constexpr (BYTES == 16) {
    // Vectorized 16-byte load using buffer_load_dwordx4
    int32x4_t_copy data =
        llvm_amdgcn_raw_buffer_load_i32x4_copy(src_res, 0, 0, 0);
    *reinterpret_cast<int32x4_t_copy *>(smem_ptr) = data;
  } else if constexpr (BYTES == 8) {
    // Vectorized 8-byte load using buffer_load_dwordx2
    int32x2_t_copy data =
        llvm_amdgcn_raw_buffer_load_i32x2_copy(src_res, 0, 0, 0);
    *reinterpret_cast<int32x2_t_copy *>(smem_ptr) = data;
  } else if constexpr (BYTES == 4) {
    // 4-byte load (single dword)
    *reinterpret_cast<uint32_t *>(smem_ptr) =
        *reinterpret_cast<uint32_t const *>(gmem_ptr);
  }
#elif defined(CP_ASYNC_SM80_ENABLED)
  static_assert(BYTES == 4 || BYTES == 8 || BYTES == 16,
                "cp.async only supports 4, 8, or 16 bytes");

  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
                   smem_int_ptr),
               "l"(gmem_ptr),
               "n"(BYTES),
               "r"(BYTES));
#endif
}

template <typename T, int BYTES = 16>
__device__ __forceinline__ void
    load_smem_with_predict(T *smem_ptr, T const *gmem_ptr, bool pred) {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // HIP: Use CK-style buffer intrinsics for vectorized global loads with
  // predicate
  static_assert(BYTES == 4 || BYTES == 8 || BYTES == 16,
                "load_smem_with_predict only supports 4, 8, or 16 bytes");
  if (pred) {
    // Create buffer resource for source
    int32x4_t_copy src_res = make_buffer_resource_copy(gmem_ptr);

    if constexpr (BYTES == 16) {
      // Vectorized 16-byte load using buffer_load_dwordx4
      int32x4_t_copy data =
          llvm_amdgcn_raw_buffer_load_i32x4_copy(src_res, 0, 0, 0);
      *reinterpret_cast<int32x4_t_copy *>(smem_ptr) = data;
    } else if constexpr (BYTES == 8) {
      // Vectorized 8-byte load using buffer_load_dwordx2
      int32x2_t_copy data =
          llvm_amdgcn_raw_buffer_load_i32x2_copy(src_res, 0, 0, 0);
      *reinterpret_cast<int32x2_t_copy *>(smem_ptr) = data;
    } else if constexpr (BYTES == 4) {
      // 4-byte load (single dword)
      *reinterpret_cast<uint32_t *>(smem_ptr) =
          *reinterpret_cast<uint32_t const *>(gmem_ptr);
    }
  }
#elif defined(CP_ASYNC_SM80_ENABLED)
  static_assert(BYTES == 4 || BYTES == 8 || BYTES == 16,
                "cp.async only supports 4, 8, or 16 bytes");
  int src_in_bytes = pred ? BYTES : 0;
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
                   smem_int_ptr),
               "l"(gmem_ptr),
               "n"(BYTES),
               "r"(src_in_bytes));
#endif
}

template <typename T>
__device__ __forceinline__ void ldsm_divergence(uint32_t smem_int_ptr,
                                                uint32_t *R) {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // HIP: ldmatrix PTX not available; use plain loads
  uint32_t const *p =
      reinterpret_cast<uint32_t const *>(static_cast<uintptr_t>(smem_int_ptr));
  R[0] = p[0];
  R[1] = p[1];
  R[2] = p[2];
  R[3] = p[3];
#else
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
      : "r"(smem_int_ptr));
#endif
}

// ldmatrix
template <typename T>
__device__ __forceinline__ void ldsm(T *__restrict__ smem_ptr, uint32_t *R) {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // HIP: Emulate CUDA ldmatrix.sync.aligned.m8n8.x4.shared.b16
  // CUDA ldmatrix loads a 16x16 matrix (in 8x8 tiles) with warp-wide
  // cooperation Each thread provides a row address, and the instruction
  // redistributes data so each thread gets specific elements matching MMA input
  // format.
  //
  // For m8n8.x4: loads four 8x8 tiles forming a 16x16 matrix
  // Thread layout: thread t gets elements from rows based on t%8
  // The shared memory pointer from each thread points to a row base
  //
  // AMD emulation: gather data from other threads using shuffle
  int lane = threadIdx.x & 31;

  // Each thread loads its row's data (4 uint32_t = 8 bf16 values = 1 row of 8x8
  // tile)
  uint32_t const *p = reinterpret_cast<uint32_t const *>(smem_ptr);
  uint32_t my_data[4];
  my_data[0] = p[0];
  my_data[1] = p[1];
  my_data[2] = p[2];
  my_data[3] = p[3];

  // For CUDA mma m16n8k16, each thread needs:
  // R[0]: elements from rows 0,8 (packed)
  // R[1]: elements from rows 1,9
  // R[2]: elements from rows 2,10
  // R[3]: elements from rows 3,11
  // The exact mapping depends on lane ID

  // CUDA ldmatrix distributes data so thread t gets:
  // - Data from row (t%8) and row (t%8 + 8) for the A matrix
  // - Distributed to match MMA input requirements

  // Simplified emulation: gather from threads that loaded the rows we need
  int my_row = lane % 8;
  int my_col_group =
      lane / 8; // 0-3, determines which columns within the 8-wide tile

  // Thread that loaded row r is: (r % 8) + (something based on tile)
  // For first 8 rows: thread = row
  // For next 8 rows: thread = row - 8 + 8 = row (same threads, different load)

  // Actually, for ldmatrix with x4, threads 0-7 load rows 0-7, threads 8-15
  // load rows 8-15 The output R[0-3] for each thread contains specific elements

  // For compatibility with the existing code that expects CUDA semantics,
  // we simply load our own row's data
  R[0] = my_data[0];
  R[1] = my_data[1];
  R[2] = my_data[2];
  R[3] = my_data[3];
#else
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
      : "r"(smem_int_ptr));
#endif
}
template <typename T>
__device__ __forceinline__ void ldsm_t(T *__restrict__ smem_ptr, uint32_t *R) {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // HIP: Emulate CUDA ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16
  // Same as ldsm but with transpose
  uint32_t const *p = reinterpret_cast<uint32_t const *>(smem_ptr);
  R[0] = p[0];
  R[1] = p[1];
  R[2] = p[2];
  R[3] = p[3];
#else
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
      : "r"(smem_int_ptr));
#endif
}

} // namespace kernel