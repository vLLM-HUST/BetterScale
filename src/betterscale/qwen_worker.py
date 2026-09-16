"""Independent opt-in Qwen hybrid entry; does not install the DSV4 patch stack."""

from functools import lru_cache
import hashlib
from importlib import metadata, resources
import json

from vllm_ascend.worker.worker import NPUWorker
from .patches.qwen_prefill import PREFILLS, install


@lru_cache(maxsize=1)
def check_runtime():
    from .compat import check_runtime as check_common_runtime

    check_common_runtime()
    pins = json.loads(
        resources.files("betterscale").joinpath("qwen_pins.json").read_text()
    )
    for item in pins:
        path = metadata.distribution(item["distribution"]).locate_file(item["path"])
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise RuntimeError(f"Unqualified Qwen donor source: {item['path']}")


def validate_config(config):
    p, s, m = config.parallel_config, config.scheduler_config, config.model_config
    hf = getattr(m.hf_config, "text_config", m.hf_config)
    checks = {
        "Qwen hybrid 27B BF16 text-only": (
            hf.model_type == "qwen3_5_text"
            and (
                hf.num_hidden_layers,
                hf.hidden_size,
                hf.head_dim,
                hf.num_attention_heads,
                hf.num_key_value_heads,
            )
            == (64, 5120, 256, 24, 4)
            and str(m.dtype) == "torch.bfloat16"
            and m.quantization is None
            and config.load_config.load_format == "auto"
        ),
        "TP2/DP1/PP1, no EP or context parallelism": (
            p.tensor_parallel_size,
            p.data_parallel_size,
            p.pipeline_parallel_size,
            p.enable_expert_parallel,
            p.decode_context_parallel_size,
            p.prefill_context_parallel_size,
        )
        == (2, 1, 1, False, 1, 1),
        "8 seats /2048-token budget /context<=8192": (
            s.max_num_seqs == 8
            and s.max_num_batched_tokens == 2048
            and m.max_model_len <= 8192
        ),
        "text-only input": m.multimodal_config is None
        or all(
            m.multimodal_config.get_limit_per_prompt(kind) == 0
            for kind in ("image", "video")
        ),
        "native asynchronous scheduler": s.scheduler_cls is None and s.async_scheduling,
        "non-speculative, APC off": config.speculative_config is None
        and not config.cache_config.enable_prefix_caching,
        "complete FULL capture set": str(config.compilation_config.cudagraph_mode)
        == "FULL"
        and set(config.compilation_config.cudagraph_capture_sizes)
        == {1, 2, 4, 8, *PREFILLS},
    }
    if not all(checks.values()):
        raise ValueError(
            "Outside Qwen FULL qualification: "
            + "; ".join(k for k, v in checks.items() if not v)
        )


class Worker(NPUWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        check_runtime()
        validate_config(vllm_config)
        install()
        super().__init__(vllm_config, *args, **kwargs)
