"""vLLM scheduler hooks for Fleet's multi-token persistent decode chunks."""

import os


def install_persistent_scheduler_hooks():
    persist_n = max(1, int(os.environ.get("FLEET_MK_PERSIST", "1")))
    if persist_n == 1:
        return

    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_fleet_mk_persist_installed", False):
        return

    original_init = Scheduler.__init__
    original_update = Scheduler._update_request_with_output

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Reserve KV for every position the one-query Fleet launch will write.
        # This does not schedule N query tokens or enable speculative decoding.
        self.num_lookahead_tokens = max(
            self.num_lookahead_tokens, persist_n - 1)

    def update_request_with_output(
            self, request, new_token_ids, is_stale=False):
        accepted, stopped = original_update(
            self, request, new_token_ids, is_stale=is_stale)
        if not is_stale and len(accepted) > 1:
            # schedule() already counted the one query token. Fleet also wrote
            # KV for the first N-1 generated tokens inside the same launch.
            request.num_computed_tokens += len(accepted) - 1
        return accepted, stopped

    Scheduler.__init__ = init
    Scheduler._update_request_with_output = update_request_with_output
    Scheduler._fleet_mk_persist_installed = True
