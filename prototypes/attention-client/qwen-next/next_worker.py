"""Native hybrid model; only routed-expert construction and loading are replaced."""

import re
import os
import torch
from vllm_ascend.worker.worker import NPUWorker
from settings import LAYERS, REAL

SESSION = None


class RemoteExperts(torch.nn.Module):
    is_internal_router = True

    def __init__(self, *, gate, shared_experts, prefix, **kwargs):
        super().__init__()
        assert shared_experts is not None
        assert kwargs["quant_config"] is None and not kwargs["enable_eplb"]
        assert kwargs["num_experts"] == 512 and kwargs["top_k"] == 10
        self.layer_id = int(re.search(r"layers\.(\d+)\.", prefix)[1])
        # Native SparseMoeBlock owns these parameters and their checkpoint names.
        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "shared", shared_experts)

    def forward(self, hidden_states, router_logits):
        assert SESSION is not None
        logits, _ = self.gate(hidden_states)
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        priority = 0
        if is_forward_context_available():
            metadata = get_forward_context().attn_metadata
            if isinstance(metadata, dict) and metadata:
                # Current fixture has one sequence: this frame is homogeneous.
                # Decode priority must not be guessed from row count (K+1 and
                # short prefills overlap). Use native scheduling metadata.
                priority = int(
                    any(getattr(m, "num_prefills", 0) > 0 for m in metadata.values())
                )
        return SESSION.forward(
            self.layer_id, hidden_states, logits, self.shared, priority=priority
        )


class Worker(NPUWorker):
    def load_model(self, *args, **kwargs):
        config = self.vllm_config.model_config.hf_text_config
        parallel = self.vllm_config.parallel_config
        assert config.num_hidden_layers == LAYERS and config.hidden_size == 2048
        assert config.num_experts == 512 and config.num_experts_per_tok == 10
        assert parallel.tensor_parallel_size == parallel.pipeline_parallel_size == 1
        assert (self.vllm_config.load_config.load_format != "dummy") == REAL
        from vllm.model_executor.models import qwen3_next

        original = qwen3_next.FusedMoE
        original_loader = qwen3_next.Qwen3NextForCausalLM.load_weights

        def load_attention(model, weights):
            return original_loader(
                model,
                (
                    (n, w)
                    for n, w in weights
                    if ".experts." not in n and not n.startswith("mtp.")
                ),
            )

        qwen3_next.FusedMoE = RemoteExperts
        qwen3_next.Qwen3NextForCausalLM.load_weights = load_attention
        try:
            result = super().load_model(*args, **kwargs)
        finally:
            qwen3_next.FusedMoE = original
            qwen3_next.Qwen3NextForCausalLM.load_weights = original_loader
        assert not any(
            ".mlp.experts." in name
            for name, _ in self.model_runner.model.named_parameters()
        )
        global SESSION
        from next_remote import Session

        self.audit_cases = []
        if os.environ.get("EXPERT_ROLE_AUDIT") == "1":
            self.prepare_audit()
        SESSION = Session()
        if os.environ.get("NEXT_FULL_GRAPH") == "1":
            # Native runner gates metadata initialization/updates on Dynamo mode,
            # although ACL graph capture itself does not require torch.compile.
            # Our ctypes remote kernels are captured directly by ACL, not Dynamo.
            assert self.vllm_config.compilation_config.mode == 0
            self.model_runner.use_aclgraph = True
        return result

    @torch.inference_mode()
    def prepare_audit(self):
        from next_weights import weights
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts

        for layer, rows in ((0, 3), (3, 5)):
            torch.manual_seed(931 + layer)
            x = (torch.randn(rows, 2048) * 0.1).bfloat16().npu()
            ids = (
                (
                    (torch.arange(rows)[:, None] * 13 + torch.arange(10)[None, :] * 53)
                    % 512
                )
                .int()
                .npu()
            )
            logits = torch.full((rows, 512), -100.0, device="npu")
            logits.scatter_(
                1,
                ids.long(),
                torch.arange(1, 11, device="npu").float().expand(rows, 10),
            )
            probs, ids = select_experts(x, logits, 10, False, True, num_experts=512)
            shared = self.model_runner.model.model.layers[layer].mlp.shared_expert
            contributions = torch.zeros(
                (rows * 10, 2048), dtype=torch.bfloat16, device="npu"
            )
            flat = ids.flatten()
            for owner in range(4):
                up, down = weights(owner, layers=[layer])[0]
                # ND view for an independent torch matmul oracle.
                import torch_npu

                up = torch_npu.npu_format_cast(up, 2)
                down = torch_npu.npu_format_cast(down, 2)
                for expert in flat.unique().cpu().tolist():
                    if expert // 128 != owner:
                        continue
                    positions = (flat == expert).nonzero().flatten()
                    z = x[positions // 10] @ up[expert % 128]
                    contributions[positions] = (
                        torch_npu.npu_swiglu(z) @ down[expert % 128]
                    )
                del up, down
            expected = (
                contributions.view(rows, 10, 2048).float()
                * probs.bfloat16().float().unsqueeze(-1)
            ).sum(1).bfloat16() + shared(x)
            self.audit_cases.append((layer, rows, x, logits, expected))
        torch.npu.synchronize()
        # Reclaimed reference weights must not inflate service-residency reporting.
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()

    @torch.inference_mode()
    def audit_experts(self):
        checks = []
        for layer, rows, x, logits, expected in self.audit_cases:
            shared = self.model_runner.model.model.layers[layer].mlp.shared_expert
            actual = SESSION.forward(layer, x, logits, shared).clone()
            torch.npu.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=2e-5)
            rel = (
                torch.linalg.vector_norm(actual.float() - expected.float())
                / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
            ).item()
            assert rel < 0.01
            checks.append(dict(layer=layer, rows=rows, relative_l2=rel))
        return checks

    def drain_experts(self):
        global SESSION
        count = SESSION.close()
        SESSION = None
        return dict(
            completed=count,
            local_routed_expert_parameters=0,
            parameter_bytes=sum(
                p.numel() * p.element_size()
                for p in self.model_runner.model.parameters()
            ),
            npu_peak_bytes=torch.npu.max_memory_allocated(),
            native_model_forward=True,
        )

    def shutdown(self):
        if SESSION is not None:
            self.drain_experts()
        return super().shutdown()
