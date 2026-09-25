"""Pinned donor weight construction without native State/runner initialization."""

import json
from pathlib import Path


def loaded_parameter_names(model, reported):
    """Resolve the pinned MoERunner loader's flattened routed-owner receipts.

    MoERunner.load_weights forwards RoutedExperts.load_weights unchanged, so
    returned names omit the `routed_experts` child present in named_parameters.
    Expand only aliases evidenced by that actual module/parameter ownership.
    """
    resolved = set(reported)
    for prefix, module in model.named_modules():
        routed = getattr(module, "routed_experts", None)
        if routed is None or not callable(getattr(module, "load_weights", None)):
            continue
        for name, _ in routed.named_parameters():
            alias = f"{prefix}.{name}"
            if alias in reported:
                resolved.add(f"{prefix}.routed_experts.{name}")
    return resolved


def prepare_moe_routing(model):
    """Preserve Ascend's FP32 router contract without native Worker bootstrap.

    AscendUnquantizedLinearMethod normally creates this weight during native
    post-load. Core linear leaves do not; its absence silently switches the
    reused AscendMoERunner to BF16 routing. The immutable buffer is a weight,
    not resident/request State, and must exist before graph activation.
    """
    for layer in model.model.layers:
        gate = layer.mlp.gate
        gate.register_buffer(
            "weight_fp32", gate.weight.detach().float(), persistent=False
        )
        if not layer.mlp.experts.is_internal_router:
            raise RuntimeError("pinned Ascend MoE did not select FP32 internal routing")


def load_models(config, model_path, device):
    import torch
    from safetensors import safe_open
    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForCausalLM,
    )
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP, Qwen3_5MoeMTP

    moe = config.model_config.hf_text_config.model_type == "qwen3_5_moe_text"
    with torch.device(device):
        target = initialize_model(
            config, model_class=Qwen3_5MoeForCausalLM if moe else Qwen3_5ForCausalLM
        )
        draft = initialize_model(
            config, model_class=Qwen3_5MoeMTP if moe else Qwen3_5MTP
        )
    draft.model.embed_tokens = target.model.embed_tokens
    draft.lm_head = target.lm_head
    path = Path(model_path)
    files = sorted(
        set(
            json.loads((path / "model.safetensors.index.json").read_text())[
                "weight_map"
            ].values()
        )
    )

    def weights(for_draft=False):
        for filename in files:
            with safe_open(
                str(path / filename), framework="pt", device="cpu"
            ) as reader:
                for name in reader.keys():
                    if name.startswith("model.language_model."):
                        if for_draft:
                            continue  # embedding/head are shared, already loaded
                        yield (
                            name.replace("model.language_model.", "model.", 1),
                            reader.get_tensor(name),
                        )
                    elif for_draft and name.startswith("mtp."):
                        yield name, reader.get_tensor(name)
                    elif not for_draft and name == "lm_head.weight":
                        yield name, reader.get_tensor(name)

    loaded_target = loaded_parameter_names(target, target.load_weights(weights()))
    loaded_draft = loaded_parameter_names(draft, draft.load_weights(weights(True)))
    missing_target = set(dict(target.named_parameters())) - loaded_target
    shared = {"model.embed_tokens.weight", "lm_head.weight"}
    missing_draft = set(dict(draft.named_parameters())) - loaded_draft - shared
    if missing_target or missing_draft:
        raise ValueError(
            f"incomplete weights: target={sorted(missing_target)}, draft={sorted(missing_draft)}"
        )
    for model in (target, draft):
        process_weights_after_loading(model, config.model_config, torch.device(device))
        if moe:
            prepare_moe_routing(model)
        model.eval()
    return target, draft
