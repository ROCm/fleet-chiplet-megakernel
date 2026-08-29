/* Fleet MK: Common utilities and type definitions for MI350 (gfx950)
 *
 * The memory atoms used to be TRANSCRIBED here from mirage's
 * mpk_atoms.cuh. That copy has been deleted: fleet_mk now includes fleet's own
 * header (ROCm/fleet-chiplet-megakernel, branch amd_mi355_gpt_oss120b), which
 * arrives on the include path as <mirage/persistent_kernel/mpk_atoms.cuh>.
 *
 * The transcription had to go for two reasons, in order of importance:
 *
 *   1. It COLLIDED. Fleet's task headers pull mpk_atoms.cuh in transitively
 *      (device_functions.cuh -> gang_rmsnorm_linear_mxfp4_bias_mi300.cuh ->
 *      ... -> moe_topk_softmax_mi300.cuh:25), so building against fleet gave
 *      redefinition errors on ld_local_u64 / fence_local / atom_add_local_u64.
 *   2. It DRIFTED. A hand copy of someone else's memory-ordering primitives is
 *      a silent-corruption hazard: fleet can change an sc0/sc1 bit or a
 *      waitcnt and fleet_mk would keep the old semantics with no build error.
 *
 * Only the three atoms with NO fleet counterpart are kept below -- verified
 * 0 occurrences each in fleet's tree: get_xcd_id, kernel::_gang_moe_get_xcd_id,
 * get_gpu_time. Everything else comes from fleet, out of the box.
 */
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_bf16.h>       // provides __hip_bfloat16 struct (different from hip_bfloat16)
#include <cstdint>

// Force AMD platform for all fleet_mk code.
// Must precede mpk_atoms.cuh: every atom there selects its AMD inline-asm body
// with `defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)`, and the
// #else branch is a plain volatile access with NO cache-bypass bits. Getting
// that fallback would compile and run, and quietly drop the nt / sc0 sc1
// qualifiers the cross-XCD protocol depends on.
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif

// Fleet's memory atoms: ld_nt_*, st_nt_*, st_wt_*, atom_add_release_gpu_*,
// threadfence_gpu, and the intra-XCD st_local_u64 / ld_local_u64 /
// fence_local / atom_add_local_u64.
#include <mirage/persistent_kernel/mpk_atoms.cuh>

// No-op stubs for fleet's worker-state tracing macros (MPK_WS_MARK et al).
// Fleet defines these in persistent_kernel.cuh, which fleet_mk does not compile;
// without the stubs, fleet's MoE header fails on undeclared MPK_WS_MARK at
// gang_moe_fused_mxfp4_mi300.cuh:289. Each stub is #ifndef-guarded, so if
// persistent_kernel.cuh ever does enter the build its definitions win.
#include "mpk_ws_stubs.cuh"

// Use relaxed atomics + non-temporal memory (matches fleet's tuned config)
#define FLEET_MK_USE_RELAXED_ATOMICS
#define FLEET_MK_USE_NT_MEMORY

// =============================================================================
// XCD identification
// =============================================================================

// Get current XCD ID at runtime (MI350: 0-7)
__device__ __forceinline__ int get_xcd_id() {
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
}

// Shim for mirage MoE sub-kernels that call _gang_moe_get_xcd_id()
// Must be in kernel namespace since the callers are in namespace kernel.
namespace kernel {
__device__ __forceinline__ int _gang_moe_get_xcd_id() {
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
}
} // namespace kernel

// High-resolution GPU timer (10ns resolution on MI350)
__device__ __forceinline__ unsigned long long get_gpu_time() {
  return __builtin_amdgcn_s_memrealtime();
}
