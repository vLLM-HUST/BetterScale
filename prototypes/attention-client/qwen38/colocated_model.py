"""TP2 attention with one globally synchronous, native EP8 expert phase."""

import torch
from attention import RemoteMoE, RemoteAscend
from colocated_ep import ColocatedEP
from livemodule.arch.binding import ArchBindings
from livemodule.llm.qwen38.moe import ArchQwen38MoE
from livemodule.llm.distributed import get_tp_group
from livemodule.arch.ascend.vllm.moe_runtime.experts_selector import select_experts


def bootstrap_groups():
    import livemodule.llm.distributed as parallel

    rank = torch.distributed.get_rank()
    if torch.distributed.get_world_size() != 8 or not hasattr(
        parallel, "_BETTERSCALE_GROUPS"
    ):
        raise RuntimeError("Use the dedicated colocated overlay and WORLD8")
    # Identical group-creation order on every process, including non-members.
    for kind, memberships in (
        ("tp", [tuple(range(i, i + 2)) for i in range(0, 8, 2)]),
        ("dp", [tuple(range(i, 8, 2)) for i in range(2)]),
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
    assert probe.item() == 2


class ColocatedMoE(RemoteMoE):
    def forward(self, hidden):
        group = get_tp_group()
        flat = hidden.reshape(-1, hidden.shape[-1])
        rows = flat.shape[0]
        count = (rows + 1) // 2
        # TP replicas partition their *token rows* only at EP ingress. Each
        # expert lives on one of8 ranks, never on both ranks of a TP pair.
        local = flat[group.rank_in_group :: 2].contiguous()
        active = torch.arange(count, device=flat.device) < local.shape[0]
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
        gathered = torch.empty(
            2 * count, flat.shape[1], dtype=flat.dtype, device=flat.device
        )
        torch.distributed.all_gather_into_tensor(
            gathered, routed, group=group.device_group
        )
        ordered = (
            gathered.view(2, count, -1).transpose(0, 1).reshape(2 * count, -1)[:rows]
        )
        return (ordered + shared).reshape_as(hidden)


class ColocatedAscend(RemoteAscend):
    bindings = ArchBindings(
        {**RemoteAscend.bindings.classes, ArchQwen38MoE: ColocatedMoE}
    )
