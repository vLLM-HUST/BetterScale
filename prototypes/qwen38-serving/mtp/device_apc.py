"""Experimental device-authoritative APC progression at the existing runner seam.

The runner still owns admission and raw token-progress correction. This leaf owns
state-selection counts and running columns; neither returns through CPU. Host
request identity determines remapping, never an accepted count or actual length.
The donor's corrected-length FIA fence remains until its consumer is replaced.
"""
import torch


def prepare(runner, schedule):
    ctx = runner._get_mamba_bufs().postprocess_align
    assert ctx is not None and runner.cache_config.mamba_cache_mode == 'align'
    batch = runner.input_batch
    n = batch.num_reqs
    if not ctx.is_initialized:
        ctx.initialize_from_forward_context(
            runner.kv_cache_config, runner.compilation_config.static_forward_context,
            runner.model.get_mamba_state_copy_func(),
            [batch.block_table[g].get_device_tensor(n) for g in ctx.mamba_group_ids])
    state = getattr(runner, '_mtp_apc_state', None)
    if state is None:
        capacity = runner.scheduler_config.max_num_seqs
        state = runner._mtp_apc_state = dict(
            requests={}, columns=torch.zeros(capacity, dtype=torch.int32, device=runner.device),
            selection=torch.ones(capacity, dtype=torch.int32, device=runner.device))
        # Native synchronize_input_prep protects reuse of these pinned sources.
        # torch.tensor(host_list, device=...) would perform a blocking transfer
        # and silently recreate the very host/device barrier being removed.
        pin = str(runner.device).split(':')[0] != 'cpu'
        for name,dtype in (('seats',torch.int64),('fresh',torch.bool)):
            state['h_'+name] = torch.empty(capacity,dtype=dtype,pin_memory=pin)
            state['np_'+name] = state['h_'+name].numpy()
            state[name] = torch.empty(capacity,dtype=dtype,device=runner.device)
    reset = (set(schedule.finished_req_ids) | set(schedule.preempted_req_ids or ())
             | set(schedule.scheduled_cached_reqs.resumed_req_ids))
    for req in reset:
        state['requests'].pop(req, None)
    free = iter(sorted(set(range(state['columns'].numel())) - set(state['requests'].values())))
    fresh = []
    for req in batch.req_ids:
        fresh.append(req not in state['requests'])
        if fresh[-1]:
            state['requests'][req] = next(free)
    # Stable seats retain selection for temporarily unscheduled requests.
    state['np_seats'][:n] = [state['requests'][req] for req in batch.req_ids]
    state['np_fresh'][:n] = fresh
    seats, fresh = state['seats'][:n], state['fresh'][:n]
    seats.copy_(state['h_seats'][:n],non_blocking=True)
    fresh.copy_(state['h_fresh'][:n],non_blocking=True)
    state['active_seats'] = seats
    computed = runner.num_computed_tokens[:n]
    scheduled = runner.num_scheduled_tokens.gpu[:n]
    source = torch.where(fresh, (computed - 1) // ctx.block_size, state['columns'][seats])
    selection = torch.where(fresh, 1, state['selection'][seats])
    destination = (computed + scheduled + ctx.block_size - 1) // ctx.block_size - 1
    migrate = (source >= 0) & (source != destination)
    # The SD copy primitive selects exactly destination and selection-1 using
    # an artificial boundary. No-copy rows cannot reach a boundary. Synthetic
    # inputs are NEVER installed as the actual postprocess token positions.
    synthetic = torch.where(migrate,
        (destination + 1) * ctx.block_size - selection, 0).int()
    ones = torch.ones_like(selection)
    zeros = torch.zeros_like(selection)
    ctx.run_fused_postprocess(n, selection, source, ones, synthetic, zeros)
    runner.num_accepted_tokens.gpu[:n].copy_(torch.where(migrate, 1, selection))
    runner.num_accepted_tokens.gpu[n:].fill_(1)
    ctx.mamba_state_idx_buf.gpu[:n].copy_(destination)
    ctx.num_computed_tokens_buf.gpu[:n].copy_(computed)
    ctx.num_scheduled_tokens_buf.gpu[:n].copy_(scheduled)
    drafts = ctx.num_draft_tokens_buf
    for i, req in enumerate(batch.req_ids):
        drafts.np[i] = len(schedule.scheduled_spec_decode_tokens.get(req, ()))
    drafts.copy_to_gpu(n)
    state['columns'].index_copy_(0, seats, destination)


def postprocess(runner, output_token_ids, schedule):
    n = output_token_ids.shape[0]
    accepted = runner.num_accepted_tokens.gpu
    accepted[:n].copy_((output_token_ids != -1).sum(dim=1))
    ctx = runner._get_mamba_bufs().postprocess_align
    ctx.run_fused_postprocess(n, accepted, ctx.mamba_state_idx_buf.gpu,
        ctx.num_scheduled_tokens_buf.gpu, ctx.num_computed_tokens_buf.gpu,
        ctx.num_draft_tokens_buf.gpu)
    state = runner._mtp_apc_state
    state['selection'].index_copy_(0, state['active_seats'], ctx.num_accepted_tokens_out[:n])
    if not hasattr(runner, '_mtp_apc_done'):
        runner._mtp_apc_done = torch.npu.Event()
    runner._mtp_apc_done.record()
    # ctx.num_accepted_tokens_out retains APC-reset selection for the next wave.
    # The runner's separate valid_sampled_token_count_gpu remains raw progress.


def wait_for_previous(runner, schedule):
    # The native postprocess uses global_stream(), not necessarily input prep's
    # stream. Order all shared block-table/count writes without blocking host.
    state = getattr(runner, '_mtp_apc_state', None)
    if state is not None:
        reset = (set(schedule.finished_req_ids) | set(schedule.preempted_req_ids or ())
                 | set(schedule.scheduled_cached_reqs.resumed_req_ids))
        for req in reset:
            state['requests'].pop(req, None)
    event = getattr(runner, '_mtp_apc_done', None)
    if event is not None:
        torch.npu.current_stream().wait_event(event)


def install():
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    NPUModelRunner._update_states_after_model_execute = postprocess
