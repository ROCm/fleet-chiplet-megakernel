"""Greedy-only handoff from Fleet MK argmax to vLLM.

Fleet already computes argmax in its LM-head epilogue.  vLLM normally converts
Fleet's bf16 logits to fp32, applies processors, and launches another argmax.
For an unmodified greedy request this module returns Fleet's token directly.
Unsupported sampling features fail loudly rather than silently bypassing vLLM.
"""

import torch


def _unsupported_reason(md):
    if not md.all_greedy:
        return "request is not all-greedy"
    if md.max_num_logprobs is not None or md.logprob_token_ids:
        return "logprobs were requested"
    if not md.no_penalties:
        return "frequency/presence/repetition penalties are active"
    if md.allowed_token_ids_mask is not None:
        return "allowed_token_ids is active"
    if md.bad_words_token_ids:
        return "bad_words is active"
    if md.spec_token_ids and any(md.spec_token_ids):
        return "speculative decoding is active"
    holder = md.thinking_budget_state_holder
    if holder is not None and holder.has_tracked_requests():
        return "thinking-budget processing is active"

    # vLLM 0.27 always constructs MinTokens and LogitBias processors.  They are
    # no-ops while these dictionaries are empty.  Reject any active instance,
    # and reject unknown non-argmax-invariant processors conservatively.
    for proc in md.logitsprocs.non_argmax_invariant:
        name = type(proc).__name__
        if name == "MinTokensLogitsProcessor":
            if proc.min_toks:
                return "min_tokens is active"
        elif name == "LogitBiasLogitsProcessor":
            if proc.biases:
                return "logit_bias is active"
        else:
            return f"argmax-changing logits processor {name} is installed"
    return None


def install_greedy_argmax_fastpath():
    """Patch vLLM 0.27's sampler once; inactive unless logits carry our tag."""
    from vllm.v1.outputs import SamplerOutput
    from vllm.v1.sample.sampler import Sampler

    if getattr(Sampler, "_fleet_mk_argmax_installed", False):
        return
    original = Sampler.forward

    def forward(self, logits, sampling_metadata, predict_bonus_token=False,
                logprobs_mode_override=None):
        argmax = getattr(logits, "_fleet_mk_argmax", None)
        if argmax is None:
            return original(self, logits, sampling_metadata,
                            predict_bonus_token, logprobs_mode_override)
        reason = _unsupported_reason(sampling_metadata)
        if reason is not None:
            raise RuntimeError(
                "FLEET_MK_GREEDY_ARGMAX=1 requires plain greedy sampling; "
                f"{reason}. Unset the env var to use vLLM's logits sampler.")
        if logits.shape[0] != 1 or argmax.numel() != 1:
            raise RuntimeError(
                "Fleet MK argmax fast path currently supports batch size 1 only")

        # For a persistent chunk Fleet writes the first N-1 tokens into the
        # prefix buffer and leaves the final token in argmax_output.
        count = int(getattr(logits, "_fleet_mk_token_count", 1))
        prefix = getattr(logits, "_fleet_mk_token_prefix", None)
        last = argmax.view(torch.int32)[:1]
        if count > 1:
            if prefix is None or prefix.numel() < count - 1:
                raise RuntimeError("Fleet MK persistent token buffer is missing")
            sampled = torch.cat((prefix[:count - 1], last)).reshape(1, count)
        else:
            sampled = last.reshape(1, 1)
        return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None)

    Sampler.forward = forward
    Sampler._fleet_mk_argmax_installed = True
