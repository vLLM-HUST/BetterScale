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
