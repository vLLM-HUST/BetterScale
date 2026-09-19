"""Prototype leaf override: opaque, wave-metadata-dispatched TP2 row projections."""
import types


def apply(model):
    import torch
    import torch_npu
    import torch.distributed as dist
    from vllm.distributed import get_tp_group
    from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod
    from vllm_ascend.distributed.parallel_state import get_mc2_group

    tp = get_tp_group()
    pg = get_mc2_group().device_group
    assert tp.world_size == dist.get_world_size(pg) == 2
    assert dist.get_process_group_ranks(pg) == dist.get_process_group_ranks(tp.device_group)
    targets = [(name, layer) for name, layer in model.named_modules()
               if isinstance(layer, RowParallelLinear)]
    assert len(targets) == 128, [name for name, _ in targets]
    shapes = []
    for name, layer in targets:
        assert layer.custom_op is None, name
        assert isinstance(layer.quant_method, UnquantizedLinearMethod), name
        assert layer.tp_size == 2 and layer.input_is_parallel and layer.reduce_results, name
        assert layer.bias is None and layer.weight.dtype == torch.bfloat16, name
        assert tuple(layer.weight.shape) in ((5120,3072),(5120,8704)), (name,layer.weight.shape)
        shapes.append(tuple(layer.weight.shape))
    assert shapes.count((5120,3072)) == shapes.count((5120,8704)) == 64
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
        owned = next((m.owned for m in metadata.values() if hasattr(m, "owned")), None) if metadata else None
        if owned is None or owned.decode:
            result = originals[name](x)
            return result[0] if isinstance(result, tuple) else result
        assert x.ndim == 2 and x.dtype == torch.bfloat16 and x.is_contiguous()
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
    print("PROTOTYPE_MC2_ROWS PASS", len(targets), "owned-wave-kind; all prefill/mixed", flush=True)
