"""Fuse owned prefill/mixed row projections; keep native decode collectives."""
import types


def install(model):
    import torch
    import torch_npu
    import torch.distributed as dist
    from vllm.distributed import get_tp_group
    from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod
    from vllm_ascend.distributed.parallel_state import get_mc2_group

    tp = get_tp_group()
    pg = get_mc2_group().device_group
    if (tp.world_size != 2 or dist.get_world_size(pg) != 2
            or dist.get_process_group_ranks(pg)
            != dist.get_process_group_ranks(tp.device_group)):
        raise ValueError("MC2 requires the same two ranks as the TP group")
    targets = [(name, layer) for name, layer in model.named_modules()
               if isinstance(layer, RowParallelLinear)]
    if len(targets) != 128:
        raise ValueError(f"expected128 Qwen27 row projections, found{len(targets)}")
    shapes = []
    for name, layer in targets:
        if (layer.custom_op is not None
                or not isinstance(layer.quant_method, UnquantizedLinearMethod)
                or layer.tp_size != 2 or not layer.input_is_parallel
                or not layer.reduce_results or layer.bias is not None
                or layer.weight.dtype != torch.bfloat16
                or tuple(layer.weight.shape) not in ((5120, 3072), (5120, 8704))):
            raise ValueError(f"unqualified MC2 row projection: {name}")
        shapes.append(tuple(layer.weight.shape))
    if shapes.count((5120, 3072)) != 64 or shapes.count((5120, 8704)) != 64:
        raise ValueError("MC2 requires64 attention and64 MLP row projections")
    # Initialize resources before MC2's first eager warmup/capture, not per forward.
    bootstrap = torch.ones(1, device=targets[0][1].weight.device)
    dist.all_reduce(bootstrap, group=pg)
    torch.npu.synchronize()
    dist.barrier(group=tp.cpu_group)
    hcom = pg._get_backend(torch.device('npu')).get_hccl_comm_name(tp.rank_in_group)

    from vllm.forward_context import get_forward_context

    originals = {name: layer.forward for name, layer in targets}

    # Dynamo must never specialize the wave-kind decision: one compiled graph
    # spans decode and prefill sizes. The opaque leaf runs during eager/capture;
    # ACLGraph replay then contains only the chosen device implementation.
    @torch.library.custom_op("betterscale::qwen_mc2_row", mutates_args=())
    def row(x: torch.Tensor, weight: torch.Tensor, name: str) -> torch.Tensor:
        context = get_forward_context()
        metadata = context.attn_metadata
        owned = next(
            (m.owned for m in metadata.values() if hasattr(m, "owned")), None
        ) if metadata else None
        if owned is None or owned.decode:
            result = originals[name](x)
            return result[0] if isinstance(result, tuple) else result
        if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError("MC2 requires a contiguous BF16 matrix")
        return torch_npu.npu_mm_all_reduce_base(x, weight.T, hcom, reduce_op="sum")

    @row.register_fake
    def fake(x, weight, name):
        return x.new_empty((x.shape[0], weight.shape[0]))

    def bind(name, layer):
        def forward(self, x):
            out = row(x, self.weight, name)
            return (out, None) if self.return_bias else out
        layer.forward = types.MethodType(forward, layer)
    for name, layer in targets:
        bind(name, layer)
    return len(targets)
