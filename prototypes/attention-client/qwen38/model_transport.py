"""Install one explicitly selected model data plane before warmup/capture."""

from client import Session


def install_transport(cfg, args, rank):
    if getattr(args, "colocated", False):
        import torch
        from catalog import load
        from colocated_ep import ColocatedEP

        physical_rank = torch.distributed.get_rank()
        # Model construction runs under an NPU default-device context.
        # Safetensors slices and ND assembly must stay on CPU until the
        # explicit NZ upload; do not reinterpret mapped host storage as NPU.
        with torch.device("cpu"):
            catalog = load(physical_rank, owners=8, mtp=bool(args.mtp_tokens))
        cfg.colocated_experts = ColocatedEP(torch.distributed.group.WORLD, catalog)
        cfg.remote_expert_transport = cfg.colocated_experts
    elif rank == 0:
        cfg.remote_expert_transport = Session(
            args.directory, args.build, source=args.source
        )
