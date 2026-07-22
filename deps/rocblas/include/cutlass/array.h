/*
 * ROCm cutlass-compat: Array and related types.
 * For ROCm builds, cutlass::Array, half_t, fast_exp_op, etc. are provided
 * by mirage/utils/rocblas_helper.h (cutlass namespace aliases).
 */

#pragma once

#include "cutlass/cutlass.h"

#if defined(MIRAGE_BACKEND_USE_ROCM) || defined(MIRAGE_BACKEND_USE_HIP)
#include "mirage/utils/rocblas_helper.h"
#endif
