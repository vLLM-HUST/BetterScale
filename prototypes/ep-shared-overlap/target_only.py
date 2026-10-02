"""Diagnostic target-only controls, preserving native speculative accounting.

install() alone suppresses future proposals but retains draft compute.
install_worker() additionally skips draft forward and preserves sample feedback.
Neither releases draft buffers or qualifies general no-MTP production serving.
"""
def install(scheduler_class):
    original=scheduler_class._update_after_schedule
    def update(self,output):
        original(self,output)
        for rid in output.num_scheduled_tokens:
            # Native accounting above used the actually scheduled proposal count.
            # Suppress only the next grant's placeholders; keep lookahead capacity.
            self.requests[rid].spec_token_ids=[]
    scheduler_class._update_after_schedule=update


def install_worker():
    """Skip draft forward while preserving native asynchronous sample feedback.

    Still allocates/captures the baseline drafter at startup. This is an initial
    all-target runner mode, not per-request reactivation or memory reclamation.
    """
    import torch
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner as Runner
    def propose(runner,sampled,sampling,schedule,*args,**kwargs):
        if any(schedule.scheduled_spec_decode_tokens.values()):
            raise RuntimeError('Target-only runner received speculative work')
        if not runner.use_async_spec_decode or not isinstance(sampled,torch.Tensor):
            raise RuntimeError('Target-only cut requires pinned async padded sampling')
        if sampled.ndim!=2 or sampled.shape[1]!=1:
            raise RuntimeError('Target-only cut must sample one token per row')
        next_ids,counts=runner.drafter.prepare_next_token_ids_padded(
            sampled,runner.requests,runner.input_batch,
            runner.discard_request_indices.gpu,runner.num_discarded_requests)
        runner._copy_valid_sampled_token_count(next_ids,counts)
        return None
    Runner.propose_draft_token_ids=propose
