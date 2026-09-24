"""Separate asynchronous Mamba feedback from the mutable input-batch row table.

The pinned base runner D2H-writes into GPUInputBatch's accepted-count tensor.
InputBatch.add_request/condense/swap_states mutate that same backing. Ascend
then remaps it AGAIN from previous request order in _prepare_inputs. Depending
on D2H completion and batch changes, a previous request's count can be replaced
by another request's count (or a new request's neutral1).

Keep the native postprocess/reset semantics and existing event. Only its D2H
mailbox becomes private; publish that old-order receipt after all batch edits,
just before the native previous-position remap. No new sampling or state copy.
"""
from functools import wraps


def install(runner):
    if getattr(runner, '_bs_mtp_feedback', None) is not None:
        return
    if not (runner.speculative_config and runner.model_config.is_hybrid and runner.use_async_scheduling):
        return
    import torch
    original_post = runner._update_states_after_model_execute
    original_prepare = runner._prepare_inputs
    mailbox = {'tensor': None}

    @wraps(original_post)
    def post(*args, **kwargs):
        batch = runner.input_batch
        destination = batch.num_accepted_tokens_cpu_tensor
        if mailbox['tensor'] is None:
            mailbox['tensor'] = torch.ones_like(destination, pin_memory=destination.is_pinned())
        raw = mailbox['tensor']
        if raw.shape != destination.shape or raw.dtype != destination.dtype:
            raise RuntimeError('MTP feedback capacity changed; start a fresh runner')
        # The NumPy batch-row view stays attached to its ORIGINAL tensor. Only
        # the native D2H destination is redirected, until native has enqueued it.
        batch.num_accepted_tokens_cpu_tensor = raw
        try:
            return original_post(*args, **kwargs)
        finally:
            batch.num_accepted_tokens_cpu_tensor = destination

    @wraps(original_prepare)
    def prepare(*args, **kwargs):
        raw = mailbox['tensor']
        if raw is not None:
            event = runner.num_accepted_tokens_event
            if event is None:
                raise RuntimeError('Missing native MTP feedback completion fence')
            event.synchronize()
            # condense/add/swap have finished. The pinned _prepare_inputs does
            # not mutate request order before its native previous-index remap.
            batch = runner.input_batch
            if batch.prev_req_id_to_index:
                batch.num_accepted_tokens_cpu[:] = raw.numpy()
            else:
                # A fresh batch has no previous-row owner. Publishing the last
                # finished request's receipt here would undo add_request's
                # neutral count and select a stale cached candidate state.
                batch.num_accepted_tokens_cpu.fill(1)
        return original_prepare(*args, **kwargs)

    runner._update_states_after_model_execute = post
    runner._prepare_inputs = prepare
    runner._bs_mtp_feedback = mailbox
