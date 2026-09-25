"""First owned execution vertical: two declared GDN graph banks, one generation."""

from dataclasses import dataclass

import torch
from betterscale.live import GraphCallSchema, MetaTensor, construct_meta_tensors
from .root import QwenStateRoot


@dataclass(frozen=True, slots=True, kw_only=True)
class GDNCallSchema(GraphCallSchema):
    capture_state_blocks: dict


class GDNGraphRoot(QwenStateRoot):
    def __init__(self, geometry, capacity, weight):
        if (
            (
                geometry.gdn_key_heads,
                geometry.gdn_value_heads,
                geometry.gdn_key_dim,
                geometry.gdn_value_dim,
                geometry.conv_kernel,
            )
            != (16, 16, 128, 128, 4)
            or geometry.layer_types[0] != "linear_attention"
            or capacity.execution_seats != 4
            or capacity.resident_seats < 5
        ):
            raise ValueError(
                "this bounded graph probe requires Qwen0.8B geometry, E4/R>=5"
            )
        super().__init__(geometry, capacity)
        # Weights are ordinary model storage; State and metadata remain unbound.
        self.register_buffer("conv_weight", weight.to(self.live_device))
        for bank in range(2):
            self.register_meta_tensor(
                f"conv_output_{bank}", MetaTensor((12, 6144), dtype=torch.bfloat16)
            )
            self.register_meta_tensor(
                f"recurrent_output_{bank}",
                MetaTensor((1, 12, 16, 128), dtype=torch.bfloat16),
            )
            schema = GDNCallSchema(
                capture_state_blocks={self.residents: (0, 1, 2, 3), self.pages: ()},
                args=(
                    torch.zeros(12, 6144, dtype=torch.bfloat16),
                    torch.full((1, 12, 16), -0.1),
                    torch.full((1, 12, 16), 0.5),
                    torch.tensor([0, 3, 6, 9, 12, 12], dtype=torch.int32),
                    torch.tensor(
                        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [-1, -1, -1]]
                    ),
                    torch.tensor([[0], [1], [2], [3], [-1]], dtype=torch.int32),
                    torch.ones(5, dtype=torch.int32),
                    bank,
                ),
                kwargs={},
            )
            self.register_graph(f"bank{bank}", entry=self.step, schema=schema)

    def _initialize_live_generation(self):
        # Probe seed: capture mutates these nonzero values and must restore them.
        self.target["0"].recurrent.tensor.fill_(0.125)
        self.target["0"].conv.tensor.fill_(0.25)

    def step(self, x, g, beta, cu, slots, conv_slots, accepted, bank):
        from .gdn_candidates import fused_recurrent_gated_delta_rule_fwd

        state = self.target["0"]
        y, output = construct_meta_tensors(
            self._metadata0 if bank == 0 else self._metadata1, context=None
        )
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            y,
            x,
            self.conv_weight,
            conv_state=state.conv.tensor,
            bias_opt=None,
            query_start_loc_opt=cu,
            cache_indices_opt=conv_slots,
            initial_state_mode_opt=None,
            num_accepted_tokens_opt=accepted,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
        )
        q, k, v = (
            part.reshape(1, 12, 16, 128).contiguous()
            for part in y.split([2048, 2048, 2048], -1)
        )
        out, _ = fused_recurrent_gated_delta_rule_fwd(
            q,
            k,
            v,
            g,
            beta,
            128**-0.5,
            state.recurrent.tensor,
            cu_seqlens=cu,
            ssm_state_indices=slots,
            num_accepted_tokens=accepted,
            use_qk_l2norm_in_kernel=True,
        )
        output.copy_(out)

    def _metadata0(self, context):
        return self.outputs(0)

    def _metadata1(self, context):
        return self.outputs(1)

    def outputs(self, bank):
        return (
            getattr(self, f"conv_output_{bank}").tensor,
            getattr(self, f"recurrent_output_{bank}").tensor,
        )
