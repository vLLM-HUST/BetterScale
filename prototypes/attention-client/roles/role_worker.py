"""Native worker with an ordinary MoE backend, not a model-forward transplant."""

import os
import re
import torch
from vllm_ascend.worker.worker import NPUWorker

SESSION = None


class RemoteExperts(torch.nn.Module):
    is_internal_router = True

    def __init__(self, *, gate, prefix, shared_experts=None, **kwargs):
        super().__init__()
        assert shared_experts is None, "this gate qualifies ordinary Qwen3-MoE only"
        assert kwargs["quant_config"] is None and not kwargs["enable_eplb"]
        self.layer_id = int(re.search(r"layers\.(\d+)\.", prefix)[1])
        object.__setattr__(self, "gate", gate)  # native MLP already owns the gate

    def forward(self, hidden_states, router_logits):
        assert SESSION is not None, "expert session must precede native warmup"
        logits, _ = self.gate(hidden_states)
        return SESSION.forward(self.layer_id, hidden_states, logits)


class Worker(NPUWorker):
    def load_model(self, *args, **kwargs):
        config = self.vllm_config.model_config.hf_text_config
        parallel = self.vllm_config.parallel_config
        assert config.num_hidden_layers == 2 and config.hidden_size == 2048
        assert config.num_experts == 128 and config.moe_intermediate_size == 768
        assert parallel.tensor_parallel_size == parallel.pipeline_parallel_size == 1
        assert (
            self.vllm_config.load_config.load_format == "dummy"
        ), "role loader: dummy gate only"
        from vllm.model_executor.models import qwen3_moe

        original = qwen3_moe.FusedMoE
        qwen3_moe.FusedMoE = RemoteExperts
        try:
            result = super().load_model(*args, **kwargs)
        finally:
            qwen3_moe.FusedMoE = original
        assert not any(
            ".mlp.experts." in name
            for name, _ in self.model_runner.model.named_parameters()
        )
        # Native memory profiling executes real model forwards before
        # compile_or_warm_up_model. Register/capture before that first consumer.
        global SESSION
        from remote import Session

        SESSION = Session()
        return result

    def audit_experts(self):
        # Explicit diagnostic RPC, never called by model.forward. Reconstruct
        # the independently seeded dummy shards only for this numerical oracle.
        import torch_npu
        from server import weights
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts

        checks = []
        for layer, rows in ((0, 3), (1, 5)):
            torch.manual_seed(913 + layer)
            x = (torch.randn(rows, 2048) * 0.1).bfloat16().npu()
            ids = (
                (
                    (torch.arange(rows)[:, None] * 9 + torch.arange(8)[None, :] * 17)
                    % 128
                )
                .int()
                .npu()
            )
            logits = torch.full((rows, 128), -100.0, device="npu")
            logits.scatter_(
                1, ids.long(), torch.arange(1, 9, device="npu").float().expand(rows, 8)
            )
            probs, selected = select_experts(x, logits, 8, False, True, num_experts=128)
            actual = SESSION.forward(layer, x, logits).clone()
            contributions = torch.zeros(
                (rows * 8, 2048), dtype=torch.bfloat16, device="npu"
            )
            flat_ids = selected.flatten()
            for owner in range(2):
                up, down = weights(owner, nz=False)
                for expert in flat_ids.unique().cpu().tolist():
                    if expert // 64 != owner:
                        continue
                    positions = (flat_ids == expert).nonzero().flatten()
                    group = layer * 64 + expert % 64
                    contributions[positions] = (
                        torch_npu.npu_swiglu(x[positions // 8] @ up[group])
                        @ down[group]
                    )
                del up, down
            expected = (
                (
                    contributions.reshape(rows, 8, 2048).float()
                    * probs.bfloat16().float().unsqueeze(-1)
                )
                .sum(1)
                .bfloat16()
            )
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=2e-5)
            relative = (
                torch.linalg.vector_norm(actual.float() - expected.float())
                / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
            ).item()
            assert relative < 0.01
            checks.append(dict(layer=layer, rows=rows, relative_l2=relative))
        return checks

    def drain_experts(self):
        global SESSION
        count = SESSION.close()
        SESSION = None
        return dict(
            parameter_bytes=sum(
                p.numel() * p.element_size()
                for p in self.model_runner.model.parameters()
            ),
            npu_peak_bytes=torch.npu.max_memory_allocated(),
            completed=count,
            local_routed_expert_parameters=0,
            native_model_forward=True,
            shadow=False,
        )

    def shutdown(self):
        if SESSION is not None:
            self.drain_experts()
        return super().shutdown()
