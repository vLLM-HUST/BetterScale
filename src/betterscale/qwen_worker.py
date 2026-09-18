"""Independent opt-in Qwen hybrid entry; does not install the DSV4 patch stack."""

from functools import lru_cache
import hashlib
from importlib import metadata, resources, util
from pathlib import Path
import json

from vllm_ascend.worker.worker import NPUWorker
from .patches.qwen_prefill import PREFILLS, install


@lru_cache(maxsize=2)
def check_runtime(pins_name="qwen_pins.json"):
    pins = json.loads(resources.files("betterscale").joinpath(pins_name).read_text())
    for name, expected in pins["versions"].items():
        if metadata.version(name).split("+", 1)[0] != expected:
            raise RuntimeError(f"Unqualified Qwen donor version: {name}")
    # Editable donors need not place source under distribution.locate_file().
    # Validate the package actually imported, never an unrelated installed copy.
    for item in pins["source_files"]:
        package, relative = item["path"].split("/", 1)
        spec = util.find_spec(package)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Missing Qwen donor package: {package}")
        path = Path(spec.origin).parent / relative
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise RuntimeError(f"Unqualified Qwen donor source: {item['path']}")


def validate_config(config):
    p, s, m = config.parallel_config, config.scheduler_config, config.model_config
    hf = getattr(m.hf_config, "text_config", m.hf_config)
    spec = config.speculative_config
    mode = str(config.compilation_config.cudagraph_mode)
    sizes = set(config.compilation_config.cudagraph_capture_sizes)
    native_mtp = spec is not None
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
        "APC off": not config.cache_config.enable_prefix_caching,
        "no speculation or native MTP2": spec is None
        or (
            getattr(spec, "method", None) == "mtp"
            and getattr(spec, "num_speculative_tokens", None) == 2
            and not getattr(spec, "enforce_eager", False)
        ),
        "route-specific graph configuration": (
            mode == "FULL_AND_PIECEWISE"
            and sizes
            and min(sizes) >= 1
            and max(sizes) == 24
            if native_mtp
            else mode == "FULL" and sizes == {1, 2, 4, 8, *PREFILLS}
        ),
    }
    if not all(checks.values()):
        raise ValueError(
            "Outside Qwen serving qualification: "
            + "; ".join(k for k, v in checks.items() if not v)
        )


class Worker(NPUWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        check_runtime()
        validate_config(vllm_config)
        self._native_mtp = vllm_config.speculative_config is not None
        if not self._native_mtp:
            install()
        super().__init__(vllm_config, *args, **kwargs)

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        if self._native_mtp:
            from .patches.qwen_layout import pack_conv_weights
            from vllm.logger import init_logger

            count = pack_conv_weights(self.model_runner.model)
            init_logger("vllm.betterscale.qwen").info(
                "Packed %d immutable GDN convolution weights; native MTP execution retained",
                count,
            )
        return result


class MixedWorker(NPUWorker):
    """Opt-in owned K-V state and elastic mixed FULL graphs; fresh process only."""

    def __init__(self, vllm_config, *args, **kwargs):
        check_runtime()
        check_runtime("qwen_mixed_pins.json")
        validate_config(vllm_config)
        if (
            vllm_config.speculative_config is not None
            or vllm_config.lora_config is not None
            or vllm_config.kv_transfer_config is not None
            or vllm_config.cache_config.mamba_cache_mode != "none"
        ):
            raise ValueError(
                "Owned GDN requires no MTP, LoRA, cache transfer or Mamba prefix cache"
            )
        hf = vllm_config.model_config.hf_text_config
        if tuple(
            getattr(hf, k, None)
            for k in (
                "linear_num_key_heads",
                "linear_num_value_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_conv_kernel_dim",
            )
        ) != (16, 48, 128, 128, 4):
            raise ValueError(
                "Owned GDN requires qk8/v24 TP-local heads, K/V128, convolution width4"
            )
        from .patches.qwen_gdn import install as install_owned
        from .patches.qwen_fia import install as install_fia

        install_owned()
        super().__init__(vllm_config, *args, **kwargs)
        # Preserve the qualified bootstrap order: donor initialization, then FIA.
        install_fia()

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        from .patches.qwen_layout import pack_conv_weights

        from .patches.qwen_gdn.execution import forward_core

        pack_conv_weights(self.model_runner.model, consumer=forward_core)
        return result
