/* Copyright 2023-2024 CMU
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

// Central definition of device function macros
// These macros are used throughout Mirage and are compatible with both CUDA and HIP
// They do NOT require CUTLASS - the "CUTLASS_" prefix is just for compatibility
// with existing code that uses CUTLASS naming conventions

// Suppress macro redefinition warnings - CUTLASS may define these for host code,
// and we intentionally redefine them for device code contexts
#if defined(__GNUC__) || defined(__clang__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wmacro-redefined"
#endif

#ifdef MIRAGE_BACKEND_USE_CUDA
  // For CUDA builds, CUTLASS may define these, but we ensure they're defined
  #ifndef CUTLASS_HOST_DEVICE
    #define CUTLASS_HOST_DEVICE __host__ __device__
  #endif
  #ifndef CUTLASS_DEVICE
    #define CUTLASS_DEVICE __device__
  #endif
  #ifndef CUTLASS_HOST
    #define CUTLASS_HOST __host__
  #endif
#elif defined(MIRAGE_BACKEND_USE_ROCM) || defined(MIRAGE_BACKEND_USE_HIP)
  // For HIP/ROCm builds, define these macros (CUTLASS is not available)
  // Ensure __host__ and __device__ are defined for host code compilation (e.g. Cython)
  #ifndef __HIP__
    #ifndef __host__
      #define __host__
    #endif
    #ifndef __device__
      #define __device__
    #endif
  #endif
  #ifndef CUTLASS_HOST_DEVICE
    #define CUTLASS_HOST_DEVICE __host__ __device__
  #endif
  #ifndef CUTLASS_DEVICE
    #define CUTLASS_DEVICE __device__
  #endif
  #ifndef CUTLASS_HOST
    #define CUTLASS_HOST __host__
  #endif
#else
  // For non-GPU builds (CPU-only), define as empty or inline
  #ifndef CUTLASS_HOST_DEVICE
    #define CUTLASS_HOST_DEVICE inline
  #endif
  #ifndef CUTLASS_DEVICE
    #define CUTLASS_DEVICE inline
  #endif
  #ifndef CUTLASS_HOST
    #define CUTLASS_HOST
  #endif
#endif

// Alternative generic macros (recommended for new code)
// These don't have the confusing "CUTLASS" prefix
#ifndef MIRAGE_HOST_DEVICE
  #define MIRAGE_HOST_DEVICE CUTLASS_HOST_DEVICE
#endif
#ifndef MIRAGE_DEVICE
  #define MIRAGE_DEVICE CUTLASS_DEVICE
#endif
#ifndef MIRAGE_HOST
  #define MIRAGE_HOST CUTLASS_HOST
#endif
