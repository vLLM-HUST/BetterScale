"""Synchronous EP8 control using native Ascend MC2 and NZ grouped GEMMs.

Mechanism follows pinned vLLM-Ascend token_dispatcher.py / moe_mlp.py (Apache2).
This is an owned-model topology control, not an unmodified vLLM engine. All EP
ranks MUST execute the same layer/phase; idle attention groups still participate
with masked rows. Independent layer schedules belong to the separated server.
"""

import torch
import torch_npu


def compact_prefix(hidden, ids, probabilities, active):
    """Stable device-only row permutation for MC2's true-prefix mask contract."""
    prefix = active.to(torch.int32).cumsum(0)
    rows = torch.arange(active.numel(), device=active.device)
    total = prefix[-1]
    destination = torch.where(active, prefix - 1, total + rows - prefix).long()
    packed = [
        torch.empty_like(x).index_copy_(0, destination, x)
        for x in (hidden, ids, probabilities)
    ]
    return (*packed, rows < total, destination)


class ColocatedEP:
    def __init__(self, group, catalog):
        self.group = group
        self.rank = torch.distributed.get_rank(group)
        self.size = torch.distributed.get_world_size(group)
        if self.size != 8:
            raise ValueError("the colocated control requires EP8")
        self.name = group._get_backend(torch.device("npu")).get_hccl_comm_name(
            self.rank
        )
        self.catalog = catalog
        # Native BF16-output per-token GMM requires BF16 channel scales.
        # Convert once, exactly as the donor path; never in the replay.
        self.down_scales = [
            sd.to(torch.bfloat16) if sd is not None else None for _, _, _, sd in catalog
        ]
        for up, down, su, sd in catalog:
            if up.shape != (64, 2560, 1280) or down.shape != (64, 640, 2560):
                raise ValueError("EP8 owns exactly64 of512 experts per rank")
            if (
                torch_npu.get_npu_format(up) != 29
                or torch_npu.get_npu_format(down) != 29
            ):
                raise ValueError("native fused expert GEMMs require NZ weights")

    def routed(self, layer, hidden, ids, probabilities, *, active=None, shared=None):
        """Input rows are UNIQUE owner rows, not both TP replicas.

        Counts, activation scales and dispatch-return metadata stay on device.
        No .item()/CPU route count is used in the forward path.
        """
        if (
            hidden.ndim != 2
            or hidden.shape[1] != 2560
            or not 1 <= hidden.shape[0] <= 1024
        ):
            raise ValueError("native EP wrapper admits1..1024 H2560 source rows")
        if ids.shape != probabilities.shape or ids.shape != (hidden.shape[0], 10):
            raise ValueError("native EP routes must match the source rows and topk10")
        if active is not None and active.shape != (hidden.shape[0],):
            raise ValueError("native EP validity must match the source rows")
        # A2 MC2 admits at most256 source rows per dispatch, unlike the
        # A3 512-row lane. Keep larger attention buckets: every EP rank
        # visits the same fixed row chunks, including inactive masked rows.
        # Shared TP computation runs once, overlapped with the first dispatch.
        if hidden.shape[0] > 256:
            outputs = []
            shared_output = None
            for start in range(0, hidden.shape[0], 256):
                stop = start + 256
                callback = shared if start == 0 else None
                value = self.routed(
                    layer,
                    hidden[start:stop],
                    ids[start:stop],
                    probabilities[start:stop],
                    active=None if active is None else active[start:stop],
                    shared=callback,
                )
                if callback is not None:
                    value, shared_output = value
                outputs.append(value)
            output = torch.cat(outputs)
            return (output, shared_output) if shared is not None else output
        up, down, su, sd = self.catalog[layer]
        quantized = up.dtype == torch.int8
        if (
            hidden.ndim != 2
            or hidden.shape[1] != 2560
            or not 1 <= hidden.shape[0] <= 256
        ):
            raise ValueError("A2 MC2 leaf admits1..256 source rows/rank")
        if active is None:
            active = torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device)
        # CANN Dispatch/Combine's 1D mask must be true-prefix, not arbitrary
        # holes. Request-padded prefill and finished decode seats violate that
        # unless compacted. Counts and the inverse mapping stay on device.
        original_active = active
        hidden, ids, probabilities, active, destination = compact_prefix(
            hidden, ids, probabilities, active
        )
        communication = dict(
            group_ep=self.name,
            ep_world_size=8,
            ep_rank_id=self.rank,
            moe_expert_num=512,
            expert_shard_type=0,
            shared_expert_rank_num=0,
            global_bs=0,
            group_tp=self.name,
            tp_world_size=1,
            tp_rank_id=0,
            x_active_mask=active,
        )
        x, scale, assist, ends, ep_counts, tp_counts, expand_scales = (
            torch_npu.npu_moe_distribute_dispatch_v2(
                x=hidden,
                expert_ids=ids,
                scales=None,
                quant_mode=2 if quantized else 0,
                expert_token_nums_type=0,
                **communication,
            )
        )
        shared_output = shared() if shared is not None else None
        if quantized:
            z, zscale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                x=x, weight=up, group_list=ends, weight_scale=su, x_scale=scale
            )
            y = torch_npu.npu_grouped_matmul(
                x=[z],
                weight=[down],
                scale=[self.down_scales[layer]],
                per_token_scale=[zscale],
                group_list=ends,
                group_list_type=0,
                split_item=2,
                group_type=0,
                output_dtype=torch.bfloat16,
            )[0]
        else:
            u = torch_npu.npu_grouped_matmul(
                x=[x],
                weight=[up],
                group_list=ends,
                group_list_type=0,
                split_item=2,
                group_type=0,
                output_dtype=torch.bfloat16,
            )[0]
            z = torch_npu.npu_swiglu(u)
            y = torch_npu.npu_grouped_matmul(
                x=[z],
                weight=[down],
                group_list=ends,
                group_list_type=0,
                split_item=2,
                group_type=0,
                output_dtype=torch.bfloat16,
            )[0]
        output = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=y,
            expert_ids=ids,
            assist_info_for_combine=assist,
            ep_send_counts=ep_counts,
            tp_send_counts=tp_counts,
            expert_scales=probabilities.float(),
            expand_scales=expand_scales,
            comm_quant_mode=0,
            **communication,
        )

        output = output.index_select(0, destination)
        output = torch.where(original_active[:, None], output, 0)
        return (output, shared_output) if shared is not None else output

    def close(self):
        # Native synchronous collectives have no persistent mailbox to retire.
        torch.npu.synchronize()
        return None
