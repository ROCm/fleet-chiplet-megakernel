/* Titan: no-op stubs for the megakernel's worker-state / phase tracing macros.
 *
 * Fleet's task headers call a family of debug-instrumentation macros
 * (MPK_WS_*, MPK_TW_SUB, MPK_PHASE_MARK) that are defined in
 * persistent_kernel.cuh. Titan does not compile persistent_kernel.cuh -- it
 * drives the task headers from its own generated kernel -- so those macros
 * arrive undefined and every call site is a hard error.
 *
 * These are the disabled branches from fleet's own #ifdef, transcribed with
 * matching arity so a call site that compiles here compiles there too:
 *   MPK_PHASE_MARK    persistent_kernel.cuh:207   (guard MPK_FUSED_PHASE_TIMING)
 *   MPK_TW_SUB        :262
 *   MPK_WS_PHASE      :293
 *   MPK_WS_WAIT_REFRESH :316  <- object-like constant, NOT function-like
 *   MPK_WS_WAIT_AUX   :365
 *   MPK_WS_WAVE_EXIT  :406
 *   MPK_WS_WAVE_CLEAR :407
 *   MPK_WS_WAVE_SYNC  :425
 *   MPK_WS_ON         :455
 *   MPK_WS_MARK       :461
 *   MPK_WS_WAIT_BEGIN :474
 *   MPK_WS_WAIT_TICK  :475
 * all under #ifdef MPK_WORKER_STATE, which titan never defines.
 *
 * Stubbing rather than enabling is the correct call on performance grounds,
 * not just convenience: fleet's own comment records the four-store tracing
 * version costing 2.3x (6.99 vs 3.01 ms/token), and 2.386 -> 2.321 ms/iter
 * when compiled out.
 *
 * Each is #ifndef-guarded so that if persistent_kernel.cuh ever does enter the
 * build, its definitions win and these silently stand down instead of
 * colliding.
 */
#pragma once

#ifndef MPK_PHASE_MARK
#define MPK_PHASE_MARK(worker, slot)                                           \
  do {                                                                         \
  } while (0)
#endif

#ifndef MPK_TW_SUB
#define MPK_TW_SUB(sub, aux) ((void)0)
#endif

#ifndef MPK_WS_PHASE
#define MPK_WS_PHASE(phase, layer, xcd) ((void)0)
#endif

// Spin count between observed-value refreshes. Object-like: it is compared
// against a loop counter, not invoked.
#ifndef MPK_WS_WAIT_REFRESH
#define MPK_WS_WAIT_REFRESH 4096
#endif

#ifndef MPK_WS_WAIT_AUX
#define MPK_WS_WAIT_AUX(a0, a1, a2, a3) ((void)0)
#endif

#ifndef MPK_WS_WAVE_EXIT
#define MPK_WS_WAVE_EXIT(wave) ((void)0)
#endif

#ifndef MPK_WS_WAVE_CLEAR
#define MPK_WS_WAVE_CLEAR(wave) ((void)0)
#endif

#ifndef MPK_WS_WAVE_SYNC
#define MPK_WS_WAVE_SYNC(wave) ((void)0)
#endif

// Used in value context (`if (MPK_WS_ON(config))`), so it must expand to an
// expression, not to a statement.
#ifndef MPK_WS_ON
#define MPK_WS_ON(cfg) (false)
#endif

#ifndef MPK_WS_MARK
#define MPK_WS_MARK(code, aux) ((void)0)
#endif

#ifndef MPK_WS_WAIT_BEGIN
#define MPK_WS_WAIT_BEGIN(barrier_id, expected) ((void)0)
#endif

#ifndef MPK_WS_WAIT_TICK
#define MPK_WS_WAIT_TICK(observed, spins) ((void)0)
#endif
