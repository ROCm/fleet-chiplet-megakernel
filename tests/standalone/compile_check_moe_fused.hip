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
constexpr int kLmOpw = 64;
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
                                                    tile_idx
#ifdef MPK_MOE_XCD_STRIPE_LAYER
                                                    ,
                                                    /*layer_idx=*/0
#endif
    );
}

// Instantiate the QKV kvupd kernel so K-split / MFMA-loop edits are codegen
// gated. The MoE body above does not pull this template in.
__global__ void compile_check_qkv_kvupd(void *workspace_f32,
                                        void const *residual,
                                        void const *norm_weight,
                                        void *norm_scratch,
                                        void const *weight,
                                        void const *bias,
                                        void *x_output,
                                        void *k_cache,
                                        void *v_cache,
                                        void *q_workspace,
                                        void const *cos,
                                        void const *sin,
                                        int const *qo_indptr,
                                        int const *kv_indptr,
                                        int const *kv_indices,
                                        int const *kv_last_page_len) {
  kernel::gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_kernel<1,
                                                                64,
                                                                2944,
                                                                2880,
                                                                64,
                                                                8,
                                                                64>(workspace_f32,
                                                                    residual,
                                                                    norm_weight,
                                                                    norm_scratch,
                                                                    weight,
                                                                    bias,
                                                                    x_output,
                                                                    k_cache,
                                                                    v_cache,
                                                                    q_workspace,
                                                                    cos,
                                                                    sin,
                                                                    qo_indptr,
                                                                    kv_indptr,
                                                                    kv_indices,
                                                                    kv_last_page_len,
                                                                    /*num_active_tokens=*/1,
                                                                    /*n_wgs_per_xcd=*/10,
                                                                    /*kv_stride=*/64,
                                                                    /*q_ws_stride=*/64,
                                                                    /*tile_idx=*/0,
                                                                    /*weights_preloaded=*/false
#ifdef MPK_QKV_KSPLIT
                                                                    ,
                                                                    /*k_part=*/1
#endif
  );
}

// Keep the LM-head specialization in the same codegen gate. It has the same
// JIT-only failure mode as the MoE body and carries inline gfx950 M0/D2LDS
// sequences when MPK_LM_HEAD_GROUP_PIPELINE is enabled.
__global__ void compile_check_lm_head(void const *norm_input,
                                      void const *norm_weight,
                                      void *norm_output,
                                      void const *lm_weight,
                                      void const *bias,
                                      void *argmax_value,
                                      void *argmax_index,
                                      int tile_idx) {
  kernel::gang_rmsnorm_linear_mxfp4_bias_argmax_kernel<kBatch,
                                                       kLmOpw,
                                                       kHidden,
                                                       2880>(norm_input,
                                                             norm_weight,
                                                             norm_output,
                                                             lm_weight,
                                                             bias,
                                                             argmax_value,
                                                             argmax_index,
                                                             nullptr,
                                                             0,
                                                             kBatch,
                                                             53,
                                                             31,
                                                             201216,
                                                             tile_idx);
}

// Instantiate the fused full-layer dispatch so O-proj rank remaps (and any
// other fused-only control flow) are codegen-gated. The kernels above do
// not pull this template in.
__global__ void compile_check_fused_layer(void *const *input_ptrs,
                                           void *const *output_ptrs,
                                           void const *cos,
                                           void const *sin,
                                           int const *qo,
                                           int const *kvi,
                                           int const *kvix,
                                           int const *kvlp) {
  kernel::gang_full_layer_fused_kernel_mi300<1,
                                             64,
                                             2944,
                                             2880,
                                             64,
                                             8,
                                             4096,
                                             136,
                                             8,
                                             4096,
                                             512,
                                             8,
                                             128,
                                             1,
                                             16,
                                             4096,
                                             128,
                                             4,
                                             2944,
                                             2944,
                                             128,
                                             64,
                                             true,
                                             MPK_MAX_NUM_BATCHED_REQUESTS>(
      input_ptrs,
      output_ptrs,
      cos,
      sin,
      qo,
      kvi,
      kvix,
      kvlp,
      /*num_active_tokens=*/1,
      /*qkv_n_wgs_per_xcd=*/10,
      /*kv_stride=*/512,
      /*q_ws_stride=*/4096,
      /*attn_scale=*/0.180337f,
      /*total_qkv_tiles_per_xcd=*/10,
      /*oproj_n_wgs_per_xcd=*/23,
      /*oproj_output_stride=*/2944,
      /*router_tile_n=*/16,
      /*total_oproj_tiles=*/184,
      /*total_topk_tiles=*/128,
      /*oproj_tiles_per_xcd=*/23,
      /*moe_total_tiles_per_xcd=*/53,
      /*workers_per_xcd=*/31,
      /*tile_idx=*/0,
      /*task_layer_idx=*/0);
}
