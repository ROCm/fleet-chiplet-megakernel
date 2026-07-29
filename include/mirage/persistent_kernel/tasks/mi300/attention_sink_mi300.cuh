/* Attention sink correction for GPT-OSS style per-head sinks.
 *
 * After standard attention: out_h = sum_i(softmax(s_i) * v_i)
 *   where softmax(s_i) = exp(s_i) / sum_j(exp(s_j)) = exp(s_i - LSE)
 *
 * With sinks, the denominator gains exp(sink_h):
 *   softmax_sink(s_i) = exp(s_i) / (sum_j(exp(s_j)) + exp(sink_h))
 *                     = softmax(s_i) * sum_j(exp(s_j)) / (sum_j(exp(s_j)) +
 * exp(sink_h)) = softmax(s_i) * exp(LSE) / (exp(LSE) + exp(sink_h)) =
 * softmax(s_i) * sigmoid(LSE - sink_h)
 *
 * So: out_with_sink_h = out_h * sigmoid(LSE_h - sink_h)
 *
 * Grid: (max_requests, 1, 1) with 256 threads
 * Inputs:
 *   input_ptrs[0]: attn_out [max_tokens, num_q_heads * HEAD_DIM] bf16
 * (in-place) input_ptrs[1]: lse_acc  [max_tokens, num_q_heads] float32
 *   input_ptrs[2]: sinks    [num_q_heads] bf16
 * Outputs:
 *   output_ptrs[0]: attn_out [max_tokens, num_q_heads * HEAD_DIM] bf16 (same as
 * input)
 */

#pragma once

namespace kernel {

template <int NUM_Q_HEADS, int HEAD_DIM>
__device__ __forceinline__ void attention_sink_correction_impl(
    void *attn_out_ptr,  // [max_tokens, num_q_heads * HEAD_DIM] bf16, modified
                         // in-place
    void const *lse_ptr, // [max_tokens, num_q_heads] float32
    void const *sinks_ptr, // [num_q_heads] bf16
    int16_t request_id,
    int const *qo_indptr_buffer_ptr) {

  using bf16 = __hip_bfloat16;
  constexpr int NUM_THREADS = 256;
  constexpr int TOTAL_OUT_DIM = NUM_Q_HEADS * HEAD_DIM;

  int first_token = qo_indptr_buffer_ptr[request_id];
  int last_token = qo_indptr_buffer_ptr[request_id + 1];
  if (first_token == last_token) {
    return;
  }
  int num_tokens = last_token - first_token;

  bf16 *d_attn_out = reinterpret_cast<bf16 *>(attn_out_ptr);
  float const *d_lse = reinterpret_cast<float const *>(lse_ptr);
  bf16 const *d_sinks = reinterpret_cast<bf16 const *>(sinks_ptr);

  // Load sink values to registers (shared across all tokens)
  // With 64 heads, each thread handles at most 1 sink value
  float sink_val[1];
  if (threadIdx.x < NUM_Q_HEADS) {
    sink_val[0] = __bfloat162float(d_sinks[threadIdx.x]);
  }

  for (int tok = 0; tok < num_tokens; tok++) {
    int token_idx = first_token + tok;
    float const *lse_row = d_lse + token_idx * NUM_Q_HEADS;
    bf16 *out_row = d_attn_out + token_idx * TOTAL_OUT_DIM;

    // For each Q head, compute correction = sigmoid(LSE - sink)
    // Then multiply all HEAD_DIM elements of that head's output
    for (int h = threadIdx.x; h < NUM_Q_HEADS; h += NUM_THREADS) {
      float lse_h = lse_row[h];
      float s_h = __bfloat162float(d_sinks[h]);
      float correction = 1.0f / (1.0f + expf(s_h - lse_h));

      // Multiply output for this head by correction
      bf16 *head_out = out_row + h * HEAD_DIM;
      for (int d = 0; d < HEAD_DIM; d++) {
        float val = __bfloat162float(head_out[d]);
        head_out[d] = __float2bfloat16(val * correction);
      }
    }
  }
}

} // namespace kernel
