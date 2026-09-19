"""Materialize initialized HCCL groups' buffers before any graph capture.

Pinned TP8 scope: the process groups already created by the pinned TP8 engine,
not newly constructed communicators. Exercise actual collective families at
small and maximum admitted row counts outside graph pools. Memory profiling
must run afterwards so persistent transport residency is included in KV sizing.
"""

from . import snapshot


def prepare(worker):
    import torch
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import _world

    assert not torch.npu.is_current_stream_capturing()
    rows = worker.vllm_config.scheduler_config.max_num_batched_tokens
    hidden = worker.model_config.hf_text_config.hidden_size
    groups = sorted(
        (
            pg
            for pg, (backend, _) in _world.pg_map.items()
            if str(backend) == "hccl" and dist.get_world_size(pg) > 1
        ),
        key=lambda pg: int(pg.group_name),
    )
    assert groups
    with torch.inference_mode():
        for pg in groups:
            size = dist.get_world_size(pg)
            for n in (1, rows):
                local = torch.zeros(
                    (n, hidden), device=worker.device, dtype=torch.bfloat16
                )
                full = torch.empty(
                    (size * n, hidden), device=worker.device, dtype=local.dtype
                )
                dist.all_reduce(local, group=pg)
                dist.all_gather_into_tensor(full, local, group=pg)
                dist.reduce_scatter_tensor(local, full, group=pg)
                dist.all_to_all_single(full, torch.zeros_like(full), group=pg)
                torch.npu.synchronize()
                del local, full
            print(
                f"HCCL_PRECAPTURE_PRIMED rank={worker.rank} group={pg.group_name} size={size} rows={rows}",
                flush=True,
            )
    snapshot(worker, "communication_buffers_prepared", groups=len(groups))
