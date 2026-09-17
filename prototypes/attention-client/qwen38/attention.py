"""Owned LiveInfer Qwen4Exp root with external routed experts and W8A8 QSA.

No global donor patching: construct a distinct architecture binding and root.
GDN, HC, PLE, QSA State and the native TP shared expert stay in LiveInfer.
"""

import json

import torch
import torch_npu
from torch import nn

from livemodule.arch.ascend.binding import AscendArchitectureBinding
from livemodule.arch.binding import ArchBindings
from livemodule.arch.ascend.llm.qwen38.moe import _Qwen38SharedExpert
from livemodule.llm.distributed import get_tp_group
from livemodule.llm.layers.linear import ReplicatedLinear
from livemodule.llm.qwen38.causal_lm import Qwen38ForCausalLM
from livemodule.llm.qwen38.moe import ArchQwen38MoE
from weights import MODEL


class QuantLinear(nn.Module):
    """Preserve the QSA island's existing row/column weight ownership."""

    def __init__(self, original, *, column_sharded=False):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = nn.Parameter(
            torch.empty_like(original.weight, dtype=torch.int8), requires_grad=False
        )
        loader = getattr(original.weight, "weight_loader", None)
        if loader is not None:
            self.weight.weight_loader = loader
        for name in ("weight_scale", "weight_offset"):
            parameter = nn.Parameter(
                torch.empty(
                    self.out_features,
                    1,
                    dtype=torch.float32,
                    device=original.weight.device,
                ),
                requires_grad=False,
            )
            # o_proj's input slice does not slice its per-output-channel scales.
            if loader is not None and not column_sharded:
                parameter.weight_loader = loader
            self.register_parameter(name, parameter)
        self.ready = False

    def finish(self):
        if self.ready:
            raise RuntimeError("QSA quantized weights already finalized")
        if torch.count_nonzero(self.weight_offset):
            raise ValueError("QSA asymmetric weight offset is unsupported")
        assert torch.isfinite(self.weight_scale).all() and (self.weight_scale > 0).all()
        self.weight.data = torch_npu.npu_format_cast(self.weight.T.contiguous(), 29)
        self.weight_scale.data = self.weight_scale.flatten()
        self.ready = True

    def forward(self, x):
        if not self.ready:
            raise RuntimeError("QSA quantized projection used before loading")
        shape = x.shape[:-1]
        q, s = torch_npu.npu_dynamic_quant(x.reshape(-1, self.in_features))
        y = torch_npu.npu_quant_matmul(
            q, self.weight, self.weight_scale, pertoken_scale=s, output_dtype=x.dtype
        )
        return y.reshape(*shape, self.out_features)


class RemoteMoE(ArchQwen38MoE):
    """Only a TP-group leader submits; both ranks participate in shared TP.

    The transport is installed after weight loading and before any forward.
    Publishing precedes shared computation; collect/retire precedes the pair
    broadcast. No routed expert Parameter is allocated on either attention rank.
    """

    def __init__(self, *, vllm_config, prefix):
        nn.Module.__init__(self)
        cfg = vllm_config.model_config.hf_text_config
        self.layer = (
            48
            if prefix.startswith("mtp.")
            else int(prefix.split("layers.")[1].split(".")[0])
        )
        self.runtime_config = vllm_config
        self.gate = ReplicatedLinear(
            cfg.hidden_size,
            cfg.num_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.shared_expert_gate = ReplicatedLinear(
            cfg.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )
        self.shared_expert = _Qwen38SharedExpert(
            hidden_size=int(cfg.hidden_size),
            intermediate_size=int(cfg.shared_expert_intermediate_size),
            quant_config=None,
            expert_gate=self.shared_expert_gate,
            prefix=f"{prefix}.shared_expert",
        )

    def forward(self, hidden):
        group = get_tp_group()
        if group.world_size != 2:
            raise ValueError(
                "First remote Qwen38 integration admits TP2 attention only"
            )
        flat = hidden.reshape(-1, hidden.shape[-1])
        if group.rank_in_group == 0:
            transport = self.runtime_config.remote_expert_transport
            logits = torch.nn.functional.linear(flat.float(), self.gate.weight).to(
                flat.dtype
            )
            result = transport.forward(
                self.layer,
                flat,
                logits,
                self.shared_expert,
                priority=self.runtime_config.remote_expert_priority,
            )
        else:
            # Do not skip this: native shared down projection is collective.
            self.shared_expert(flat)
            result = torch.empty_like(flat)
        torch.distributed.broadcast(
            result, src=group.first_rank, group=group.device_group
        )
        return result.reshape_as(hidden)


class RemoteAscend(AscendArchitectureBinding):
    bindings = ArchBindings(
        {**AscendArchitectureBinding.bindings.classes, ArchQwen38MoE: RemoteMoE}
    )


class AttentionRoot(Qwen38ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        description = json.loads((MODEL / "quant_model_description.json").read_text())
        self.quantized_qsa = {}
        for name, module in list(self.named_modules()):
            if description.get(name + ".weight") != "W8A8_DYNAMIC":
                continue
            if not isinstance(module, nn.Linear) or ".self_attn." not in name:
                raise ValueError(f"Unexpected attention-side quantized module: {name}")
            replacement = QuantLinear(module, column_sharded=name.endswith(".o_proj"))
            parent, leaf = name.rsplit(".", 1)
            setattr(self.get_submodule(parent), leaf, replacement)
            self.quantized_qsa[name] = replacement
        if len(self.quantized_qsa) != 60:
            raise ValueError(
                f"Expected 12 QSA layers × 5 projections, got {len(self.quantized_qsa)}"
            )

    def accepts_checkpoint_tensor(self, name):
        return ".mlp.experts." not in name and super().accepts_checkpoint_tensor(name)

    def load_weights(self, weights):
        from ple_metadata import ExactPLEMetadata

        auxiliary = set()
        ple_metadata = ExactPLEMetadata()

        def base_weights():
            for name, value in weights:
                value = ple_metadata.restore(name, value)
                path, leaf = name.rsplit(".", 1)
                if path in self.quantized_qsa and leaf in (
                    "weight_scale",
                    "weight_offset",
                ):
                    parameter = getattr(self.quantized_qsa[path], leaf)
                    loader = getattr(parameter, "weight_loader", None)
                    if loader is not None:
                        loader(parameter, value)
                    else:
                        assert parameter.shape == value.shape, (
                            name,
                            parameter.shape,
                            value.shape,
                        )
                        parameter.data.copy_(value)
                    if name in auxiliary:
                        raise ValueError(f"Duplicate quant auxiliary: {name}")
                    auxiliary.add(name)
                else:
                    yield name, value

        loaded = super().load_weights(base_weights())
        expected = {
            name + "." + suffix
            for name in self.quantized_qsa
            for suffix in ("weight_scale", "weight_offset")
        }
        if auxiliary != expected:
            raise ValueError(
                f"QSA quantization coverage mismatch: {sorted(expected-auxiliary)[:4]}"
            )
        for module in self.quantized_qsa.values():
            module.finish()
        self.repaired_ple_metadata = tuple(ple_metadata.repaired)
        return loaded | auxiliary
