"""Qwen35 native attention/MTP with MOD-owned persistent routed experts.

Experimental admission only: source closure is not hardware qualification.
No dense-Qwen FULL prefill patch is implicitly inherited by this route.
"""
from ..patches.expert_service.config import ServiceConfig


def validate(config):
    service = ServiceConfig.from_vllm(config)
    p, m, s = config.parallel_config, config.model_config, config.scheduler_config
    hf = getattr(m.hf_config, 'text_config', m.hf_config)
    checks = {
        'Qwen35 BF16 geometry': (hf.model_type, hf.num_hidden_layers, hf.hidden_size,
            hf.moe_intermediate_size, hf.num_experts, hf.num_experts_per_tok) ==
            ('qwen3_5_moe_text', 40, 2048, 512, 256, 8),
        'BF16 unquantized auto loader': str(m.dtype) == 'torch.bfloat16' and
            m.quantization is None and config.load_config.load_format == 'auto',
        'independent TP1 attention rank': (p.tensor_parallel_size, p.data_parallel_size,
            p.pipeline_parallel_size, p.enable_expert_parallel,
            p.decode_context_parallel_size, p.prefill_context_parallel_size) == (1, 1, 1, False, 1, 1),
        'one physical draft expert layer': not service.draft_layers or getattr(hf, 'mtp_num_hidden_layers', 1) == 1,
        '4096 prefill / <=32 requests / native256K': s.max_num_batched_tokens == 4096
            and 1 <= s.max_num_seqs <= 32 and m.max_model_len <= 262144,
        'native eager or FULL decode without Dynamo': m.enforce_eager or (
            int(config.compilation_config.mode) == 0 and
            str(config.compilation_config.cudagraph_mode) == 'FULL_DECODE_ONLY'),
        'no LoRA or KV transfer': config.lora_config is None and config.kv_transfer_config is None,
        'text-only': m.multimodal_config is None or all(
            m.multimodal_config.get_limit_per_prompt(k) == 0 for k in ('image', 'video')),
    }
    if not all(checks.values()):
        raise ValueError('Outside experimental expert service: ' + '; '.join(k for k,v in checks.items() if not v))
    # Include topology/path identity: process-global hooks cannot serve two worlds.
    return ('qwen35-experts', service)


def check(config):
    from .qwen import check_runtime
    check_runtime('expert_pins.json')
    ServiceConfig.from_vllm(config).check_build()


def before_init(worker, config):
    service = ServiceConfig.from_vllm(config)
    sizes = config.compilation_config.cudagraph_capture_sizes or []
    rows = max([config.scheduler_config.max_num_seqs * (3 if service.draft_layers else 1), *sizes])
    service.bind(config.model_config.model, rows)
    from ..patches.expert_service.mamba_abi import install
    install()


def load_model(worker, native):
    from ..patches.expert_service.client import load_model
    result = load_model(worker, native)
    from ..patches.qwen_mtp_feedback import install
    install(worker.model_runner)
    return result


def expert_receipt(worker):
    from ..patches.expert_service.client import expert_receipt
    return expert_receipt(worker)


def close_expert_service(worker):
    from ..patches.expert_service.client import close_expert_service
    return close_expert_service(worker)


def shutdown(worker):
    close_expert_service(worker)
