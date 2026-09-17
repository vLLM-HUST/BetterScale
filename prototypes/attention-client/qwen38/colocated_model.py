"""TP2 attention with one globally synchronous, native EP8 expert phase."""

import torch
from attention import RemoteMoE, RemoteAscend
from colocated_ep import ColocatedEP
from livemodule.arch.binding import ArchBindings
from livemodule.llm.qwen38.moe import ArchQwen38MoE
from livemodule.llm.distributed import get_tp_group
from livemodule.llm.forward_context import current_forward_context
from livemodule.arch.ascend.vllm.moe_runtime.experts_selector import select_experts


def bootstrap_groups(tp_size=2):
    import livemodule.llm.distributed as parallel

    rank = torch.distributed.get_rank()
    if torch.distributed.get_world_size() != 8 or not hasattr(
        parallel, "_BETTERSCALE_GROUPS"
    ):
        raise RuntimeError("Use the dedicated colocated overlay and WORLD8")
    # Identical group-creation order on every process, including non-members.
    for kind, memberships in (
        ("tp", [tuple(range(i, i + tp_size)) for i in range(0, 8, tp_size)]),
        ("dp", [tuple(range(i, 8, tp_size)) for i in range(tp_size)]),
    ):
        for ranks in memberships:
            group = torch.distributed.new_group(ranks=list(ranks), backend="hccl")
            if rank in ranks:
                parallel._BETTERSCALE_GROUPS[kind] = parallel.GroupCoordinator(
                    ranks=ranks,
                    rank=rank,
                    local_rank=0,
                    rank_in_group=ranks.index(rank),
                    device_group=group,
                )
    # Initialize actual TP communicators before FULL capture.
    probe = torch.ones(1, device="npu")
    torch.distributed.all_reduce(probe, group=get_tp_group().device_group)
    torch.npu.synchronize()
    assert probe.item() == tp_size
    # QSA islands are exactly TP2 here. The legacy helper creates new_group
    # from each local TP membership, which is NOT a consistent WORLD8 creation
    # order across four DP sources. Reuse the already-created, warmed TP pair.
    from livemodule.llm.qwen38.parallel import _QSA_GROUP_CACHE

    tp = get_tp_group()
    assert tp.world_size == tp_size
    _QSA_GROUP_CACHE[(id(tp.device_group), tuple(tp.ranks))] = tp


class ColocatedMoE(RemoteMoE):
    def forward(self, hidden):
        group = get_tp_group()
        flat = hidden.reshape(-1, hidden.shape[-1])
        rows = flat.shape[0]
        size = group.world_size
        count = (rows + size - 1) // size
        # TP replicas partition their *token rows* only at EP ingress. Each
        # expert lives on one of8 ranks, never on both ranks of a TP pair.
        local = flat[group.rank_in_group :: size].contiguous()
        active = torch.arange(count, device=flat.device) < local.shape[0]
        context = current_forward_context()
        topology = getattr(context, "batch_topology", None)
        if topology is not None and hasattr(topology, "query_valid"):
            valid = topology.query_valid.reshape(-1)[group.rank_in_group :: size]
            if valid.numel() != local.shape[0]:
                raise ValueError("EP token rows disagree with attention validity")
            if valid.numel() < count:
                valid = torch.nn.functional.pad(
                    valid, (0, count - valid.numel()), value=False
                )
            active = active & valid
        if local.shape[0] < count:
            local = torch.cat(
                (
                    local,
                    torch.zeros(1, flat.shape[1], dtype=flat.dtype, device=flat.device),
                )
            )
        logits = torch.nn.functional.linear(local.float(), self.gate.weight).to(
            local.dtype
        )
        probs, ids = select_experts(local, logits, 10, False, True, num_experts=512)
        result = self.runtime_config.colocated_experts.routed(
            self.layer,
            local,
            ids,
            probs,
            active=active,
            shared=lambda: self.shared_expert(flat),
        )
        routed, shared = result
        if size == 1:
            return (routed + shared).reshape_as(hidden)
        gathered = torch.empty(
            size * count, flat.shape[1], dtype=flat.dtype, device=flat.device
        )
        torch.distributed.all_gather_into_tensor(
            gathered, routed, group=group.device_group
        )
        ordered = (
            gathered.view(size, count, -1)
            .transpose(0, 1)
            .reshape(size * count, -1)[:rows]
        )
        return (ordered + shared).reshape_as(hidden)


class ColocatedAscend(RemoteAscend):
    bindings = ArchBindings(
        {**RemoteAscend.bindings.classes, ArchQwen38MoE: ColocatedMoE}
    )
