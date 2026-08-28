// Codegen-only compile check for the fused MoE gang kernel.
//
// The megakernel is JIT-compiled at run time from a generated translation
// unit, so a source edit to gang_moe_fused_mxfp4_mi300.cuh is not validated by
// any build step -- the first thing that sees it is a run that needs the
// weights. This file instantiates the kernel at the gpt-oss-120b geometry so
// `hipcc --offload-arch=gfx950 -c` exercises both the C++ front end and the
// assembler, which is what catches a malformed inline-asm operand list or a
// tripped static_assert in the MFMA schedules.
//
// Build (see tests/standalone/README or the command in the porting notes):
//   hipcc -x hip --offload-arch=gfx950 -O2 -c compile_check_moe_fused.hip \
//     -I<fleet>/include -I<fleet>/include/mirage/persistent_kernel ...
//
// The geometry below must track the weights the demo actually packs:
//   hidden 2880 and intermediate 2880 both pad up to 2944 = 23 * 128, which is
//   where W13_MFMA_ITERS == W2_MFMA_ITERS == 23 comes from. 23 is odd (the
//   ping-pong precondition) and 23 % 4 == 3 (the quad-chain precondition), so
//   this instantiation is the one that proves both static_asserts hold.
// Mirrors the preamble of the JIT-generated translation unit: the standard
// headers come first so HIP's CUDA wrapper headers do not collide with them,
// and persistent_kernel.cuh has to be the thing that pulls in the task headers
// because it is what defines MPK_WS_MARK, _gang_moe_get_xcd_id, st_wt_u32 and
// the rest of the runtime vocabulary the MoE kernel is written against.
#include <thread>
#include <unistd.h>
#include <vector>

#define MPK_USE_CK_FMHA 1
#include "persistent_kernel.cuh"

#include "tasks/mi300/gang_moe_fused_mxfp4_mi300.cuh"

// persistent_kernel.cuh forward-declares these and the generated TU defines
// them; the worker loop calls them, so the device link fails without a body.
// Stubs are enough -- nothing here launches the persistent kernel.
__device__ __forceinline__ void
    _execute_task(mirage::runtime::TaskDesc const *,
                  mirage::runtime::RuntimeConfig const &) {}
__device__ __forceinline__ void
    _execute_gang_task(mirage::runtime::TaskDesc const *,
                       mirage::runtime::RuntimeConfig const &,
                       int) {}

namespace {
constexpr int kBatch = 1;
constexpr int kIntermediate = 2944;
constexpr int kHidden = 2944;
constexpr int kExperts = 128;
constexpr int kTopk = 4;
constexpr int kW13Opw = 128;
constexpr int kW2Opw = 64;
} // namespace

__global__ void compile_check_moe_fused(void const *input,
                                        void const *gate_up_w,
                                        void const *down_w,
                                        void const *routing,
                                        void const *mask,
                                        void const *w13_bias,
                                        void const *w2_bias,
                                        void const *routing_weight,
                                        void *swiglu_out,
                                        void *workspace_f32,
                                        void *barrier,
                                        int tile_idx) {
  kernel::gang_moe_fused_mxfp4_kernel_mi300<kBatch,
                                            kIntermediate,
                                            kHidden,
                                            kExperts,
                                            kTopk,
                                            kW13Opw,
                                            kW2Opw>(input,
                                                    gate_up_w,
                                                    down_w,
                                                    routing,
                                                    mask,
                                                    w13_bias,
                                                    w2_bias,
                                                    routing_weight,
                                                    swiglu_out,
                                                    workspace_f32,
                                                    barrier,
                                                    tile_idx);
}
