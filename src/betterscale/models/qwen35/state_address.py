"""State-only ingress and commit around the unchanged baseline model/proposer."""

from .execution_capacity import EXECUTION

from dataclasses import replace
from types import SimpleNamespace

import torch


def wait_for_previous(runner, schedule):
    event = getattr(runner, "_mtp_apc_done", None)
    if event is not None:
        torch.npu.current_stream().wait_event(event)
    root = runner._live_state_root
    leases = schedule.resident_leases
    runner._live_schedule_leases = leases
    columns = (runner.max_model_len + runner.block_size - 1) // runner.block_size + 2
    group = runner._live_gdn_group
    if group != 1:
        raise ValueError("live State expects FA group0 and one GDN metadata group")
    # Extend only worker-private descriptors. No GDN IDs enter the scheduler's
    # shared token pool or its allocation/refcount decisions.
    schedule.scheduled_new_reqs = [
        replace(req, block_ids=(*req.block_ids, [leases[req.req_id][0]] * columns))
        for req in schedule.scheduled_new_reqs
    ]
    cached = schedule.scheduled_cached_reqs
    blocks = []
    for rid, new in zip(cached.req_ids, cached.new_block_ids, strict=True):
        if new is None:
            blocks.append(None)
        else:
            resident = (
                [leases[rid][0]] * columns if rid in cached.resumed_req_ids else []
            )
            blocks.append((*new, resident))
    schedule.scheduled_cached_reqs = replace(cached, new_block_ids=blocks)
    schedule.num_common_prefix_blocks = [*schedule.num_common_prefix_blocks, 0]
    fresh = [
        (seat, epoch)
        for seat, epoch in leases.values()
        if runner._live_resident_epochs[seat] != epoch
    ]
    if fresh:
        root.clear_state_blocks(tuple(seat for seat, _ in fresh), domain=root.residents)
        for seat, epoch in fresh:
            root.continuation.selection.tensor[seat] = 1
            root.conv_selection.tensor[seat] = 1
            root.continuation.resident_epoch.tensor[seat] = epoch
            runner._live_resident_epochs[seat] = epoch
        runner._live_previous_verify.difference_update(seat for seat, _ in fresh)
    for rid, remaining in schedule.generation_limits.items():
        root.remaining_outputs.tensor[leases[rid][0]] = remaining


def initialize(runner):
    state = runner._live_ingress = SimpleNamespace()
    for name in ("seats", "drafts", "sampling"):
        host = torch.zeros(EXECUTION, dtype=torch.int64, pin_memory=True)
        setattr(state, "h_" + name, host)
        setattr(state, name, torch.zeros(EXECUTION, dtype=torch.int64, device=runner.device))
    state.h_seats.copy_(torch.arange(EXECUTION))
    state.seats.copy_(state.h_seats)
    state.history = torch.arange(3, device=runner.device)
    runner._live_previous_verify = set()


def prepare(runner, schedule):
    root, state = runner._live_state_root, runner._live_ingress
    n = runner.input_batch.num_reqs
    ids = runner.input_batch.req_ids
    state.h_seats.numpy()[:n] = [schedule.resident_leases[rid][0] for rid in ids]
    state.h_drafts.numpy()[:n] = [
        len(schedule.scheduled_spec_decode_tokens.get(rid, ())) for rid in ids
    ]
    state.h_sampling.numpy()[:n] = [rid in schedule.sampling_requests for rid in ids]
    for name in ("seats", "drafts", "sampling"):
        getattr(state, name)[:n].copy_(
            getattr(state, "h_" + name)[:n], non_blocking=True
        )
    runner.num_accepted_tokens.gpu[:n].copy_(
        root.conv_selection.tensor[state.seats[:n]]
    )
    runner.num_accepted_tokens.gpu[n:].fill_(1)


def publish_slots(meta):
    from .state_slots import publish

    runner = meta.resident_runner
    root, ingress = runner._live_state_root, runner._live_ingress
    _, seq, _, pre, ver = meta.device_slot_source
    previous = runner._live_previous_verify
    host_seats = ingress.h_seats.numpy()
    # Only a verify -> bulk-prefill transition normalizes the small history.
    # The recurrent matrix stays in its selected candidate, without copying it.
    normalize = [
        int(host_seats[row]) for row in pre if int(host_seats[row]) in previous
    ]
    if normalize:
        seats = torch.tensor(normalize, device=runner.device, dtype=torch.int64)
        offsets = (
            root.conv_selection.tensor[seats].long()[:, None]
            - 1
            + ingress.history[None, :]
        )
        for name in runner._live_gdn_specs:
            conv = root.layer_states[name].conv.tensor
            history = conv[seats[:, None], offsets].clone()
            conv[seats[:, None], ingress.history[None, :]] = history
        root.conv_selection.tensor[seats] = 1
    for row in pre:
        previous.discard(int(host_seats[row]))
    for row in ver:
        previous.add(int(host_seats[row]))
    publish[(1,)](
        ingress.seats,
        root.continuation.selection.tensor,
        root.remaining_outputs.tensor,
        seq,
        meta.cu,
        meta.prefill_ids,
        meta.verify_ids,
        meta.initial,
        meta.prefill_conv,
        meta.verify_conv,
        meta.prefill.state,
        meta.verify.slots,
        meta.verify.accepted,
        meta.live,
        len(pre),
        meta.live if meta.decode else len(ver),
        DECODE=meta.decode,
        ROWS=1 << (EXECUTION - 1).bit_length(),
        num_warps=4,
    )
    runner._live_verify_roles = meta.verify_roles


def postprocess(runner, output_token_ids, schedule):
    n = output_token_ids.shape[0]
    ingress, root = runner._live_ingress, runner._live_state_root
    seats = ingress.seats[:n]
    accepted = (output_token_ids != -1).sum(dim=1).to(torch.int32)
    remaining = root.remaining_outputs.tensor[seats]
    writable = remaining > 0
    committed = torch.minimum(accepted, remaining)
    selected = (
        runner.num_scheduled_tokens.gpu[:n] - ingress.drafts[:n] + committed - 1
    ).int()
    verify = runner._live_verify_roles[:n]
    recurrent = root.continuation.selection.tensor
    recurrent.index_copy_(
        0, seats, torch.where(verify & writable, selected, recurrent[seats])
    )
    conv = root.conv_selection.tensor
    conv.index_copy_(
        0, seats, torch.where(writable, torch.where(verify, selected, 1), conv[seats])
    )
    root.remaining_outputs.tensor.index_copy_(
        0, seats, remaining - torch.where(ingress.sampling[:n].bool(), committed, 0)
    )
    runner.num_accepted_tokens.gpu[:n].copy_(accepted)
    if not hasattr(runner, "_mtp_apc_done"):
        runner._mtp_apc_done = torch.npu.Event()
    runner._mtp_apc_done.record()


def install():
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner as Runner
    from . import device_apc, device_metadata
    from .service_metadata import Core, MTPFrame

    device_apc.wait_for_previous = wait_for_previous
    device_apc.prepare = prepare
    Runner._update_states_after_model_execute = postprocess
    device_metadata.publish_slots = publish_slots
    original_core, original_fill = Core.__init__, MTPFrame.fill_mtp

    def core_init(core, tokens, device):
        original_core(core, tokens, device)
        core.verify_roles = torch.zeros(EXECUTION + 1, dtype=torch.bool, device=device)

    def fill(frame, key, m, lengths, table, builder, accepted, drafts):
        meta, roles = original_fill(
            frame, key, m, lengths, table, builder, accepted, drafts
        )
        meta.resident_runner = builder._live_state_runner
        host = meta.host_view.verify_roles.numpy()
        host.fill(False)
        host[: len(roles)] = [True] * len(roles) if meta.decode else roles
        return meta, roles

    Core.__init__ = core_init
    MTPFrame.fill_mtp = fill
