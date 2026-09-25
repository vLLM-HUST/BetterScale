"""Coverage receipts must follow actual expert ownership, not skip MoE weights."""

import torch
from betterscale.live.llm.qwen35.loader import loaded_parameter_names


class Runner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.routed_experts = torch.nn.Linear(2, 2, bias=False)

    def load_weights(self, weights):
        return {"weight"}


def test_flattened_moe_receipt_maps_only_real_loaded_parameters():
    model = torch.nn.Module()
    model.experts = Runner()
    names = loaded_parameter_names(model, {"experts.weight", "experts.fake"})
    assert "experts.routed_experts.weight" in names
    assert "experts.routed_experts.fake" not in names
    assert "experts.routed_experts.weight" not in loaded_parameter_names(model, set())


def test_moe_router_keeps_fp32_weight_without_native_worker():
    import torch
    from torch import nn
    from types import SimpleNamespace
    from betterscale.live.llm.qwen35.loader import prepare_moe_routing

    gate = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)

    class Experts:
        @property
        def is_internal_router(self):
            return hasattr(gate, "weight_fp32")

    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(mlp=SimpleNamespace(gate=gate, experts=Experts()))]
        )
    )
    prepare_moe_routing(model)
    assert gate.weight_fp32.dtype == torch.float32
    assert not gate.weight_fp32.requires_grad
    assert "weight_fp32" not in gate.state_dict()
    torch.testing.assert_close(gate.weight_fp32, gate.weight.float())
