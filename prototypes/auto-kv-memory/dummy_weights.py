"""Probe-only native DSV4 dummy layout repair; never reads checkpoint tensors.

The pinned AscendColumnParallelLinear.weight_loader normally reshapes wo_a
from [groups * rank, input] to [groups, input, rank]. DummyModelLoader skips
weight_loader, so its random values need that same layout transform before
native post-load attention setup observes the parameter. Target and draft both
use this loader; do not patch their forward or substitute different arithmetic.
"""


def install():
    import torch
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    from vllm_ascend.ops.linear import AscendColumnParallelLinear

    original = DummyModelLoader.load_weights
    if getattr(original, "_betterscale_dummy_layout", False):
        return

    def load_weights(self, model, model_config):
        original(self, model, model_config)
        count = 0
        with torch.no_grad():
            for layer in model.modules():
                if not isinstance(layer, AscendColumnParallelLinear):
                    continue
                if "wo_a" not in layer.prefix:
                    continue
                weight = layer.weight
                assert weight.ndim == 2, (layer.prefix, weight.shape)
                assert weight.dtype == torch.bfloat16, (layer.prefix, weight.dtype)
                assert weight.shape[0] == layer.n_local_groups * layer.o_lora_rank
                weight.data = (
                    weight.data.view(layer.n_local_groups, layer.o_lora_rank, -1)
                    .transpose(2, 1)
                    .contiguous()
                )
                count += 1
        assert count, type(model).__name__
        print(
            f"DUMMY_WO_A_LAYOUT model={type(model).__name__} layers={count}", flush=True
        )

    load_weights._betterscale_dummy_layout = True
    DummyModelLoader.load_weights = load_weights
