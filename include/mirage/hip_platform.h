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

// Define HIP platform before any HIP headers are included
// This MUST be defined before any HIP headers are included
// Support both MIRAGE_BACKEND_USE_ROCM and MIRAGE_BACKEND_USE_HIP for
// compatibility
#if defined(MIRAGE_BACKEND_USE_ROCM) || defined(MIRAGE_BACKEND_USE_HIP)
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif
// For host code compilation (e.g. Cython with regular g++), ensure HIP types
// are available HIP headers need __HIP_PLATFORM_AMD__ to be defined, which
// we've done above Some HIP types might need additional defines - ensure
// they're available
#ifndef __HIP__
// When not using HIP compiler, ensure basic HIP types are available
// HIP headers should still work for host code, but we ensure platform is set
#endif
#endif

// Also check if __HIP_PLATFORM_AMD__ was defined via compiler flag
// If it was, ensure MIRAGE_BACKEND_USE_ROCM is defined for consistency
// But only if it's not already defined (to avoid redefinition warnings)
#ifdef __HIP_PLATFORM_AMD__
#ifndef MIRAGE_BACKEND_USE_ROCM
#define MIRAGE_BACKEND_USE_ROCM
#endif
#endif
