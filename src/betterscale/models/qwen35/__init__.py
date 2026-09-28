"""Admission for the owned port of the qualified Qwen35 baseline.

Model forward, MTP and the async EngineCore remain the baseline implementation.
State ownership attaches below those owners, not through another serving loop.
"""

CAPTURE_SIZES = (
    3,
    6,
    12,
    16,
    24,
    32,
    40,
    48,
    64,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    4096,
)


def capture_sizes(requests=16):
    from .count_policy import spec_capacities

    return tuple(sorted(set(CAPTURE_SIZES) | set(spec_capacities(requests))))


SCHEDULER = "betterscale.models.qwen35.apc_boundary.BoundaryScheduler"
STATE_SCHEDULER = "betterscale.models.qwen35.seat_scheduler.LiveStateScheduler"


def using_live_state(config):
    return (
        getattr(config, "additional_config", {}).get("using_live_runtime", False)
        is True
    )


def validate(config):
    p, s, m = config.parallel_config, config.scheduler_config, config.model_config
    hf = m.hf_text_config
    spec = config.speculative_config
    if using_live_state(config):
        from .capacity import seat_counts

        seat_counts(config)
    checks = {
        "35B-A3B BF16 text geometry": (
            hf.model_type == "qwen3_5_moe_text"
            and (
                hf.num_hidden_layers,
                hf.hidden_size,
                hf.head_dim,
                hf.num_attention_heads,
                hf.num_key_value_heads,
            )
            == (40, 2048, 256, 16, 2)
            and (
                hf.linear_num_key_heads,
                hf.linear_num_value_heads,
                hf.linear_key_head_dim,
                hf.linear_value_head_dim,
                hf.linear_conv_kernel_dim,
            )
            == (16, 32, 128, 128, 4)
            and str(m.dtype) == "torch.bfloat16"
            and m.quantization is None
            and config.load_config.load_format == "auto"
        ),
        "TP2/DP1/PP1 without EP or context parallelism": (
            p.tensor_parallel_size,
            p.data_parallel_size,
            p.pipeline_parallel_size,
            p.enable_expert_parallel,
            p.decode_context_parallel_size,
            p.prefill_context_parallel_size,
        )
        == (2, 1, 1, False, 1, 1),
        "execution<=36/query4096/context<=262144": (
            (1 <= s.max_num_seqs <= 36 if using_live_state(config) else s.max_num_seqs == 16)
            and s.max_num_batched_tokens == 4096
            and 0 < m.max_model_len <= 262144
        ),
        "baseline asynchronous boundary scheduler": (
            s.async_scheduling
            and s.scheduler_cls
            == (STATE_SCHEDULER if using_live_state(config) else SCHEDULER)
        ),
        "native MTP2": spec is not None
        and spec.method == "mtp"
        and spec.num_speculative_tokens == 2
        and not spec.enforce_eager,
        "qualified mixed FULL keys": (
            str(config.compilation_config.cudagraph_mode) == "FULL"
            and set(config.compilation_config.cudagraph_capture_sizes)
            == set(capture_sizes(s.max_num_seqs))
        ),
        "APC align without connectors or LoRA": (
            config.cache_config.enable_prefix_caching
            and config.cache_config.mamba_cache_mode == "align"
            and config.kv_transfer_config is None
            and config.lora_config is None
        ),
        "text-only input": m.multimodal_config is None
        or all(
            m.multimodal_config.get_limit_per_prompt(kind) == 0
            for kind in ("image", "video")
        ),
    }
    if not all(checks.values()):
        raise ValueError(
            "Outside Qwen35 baseline qualification: "
            + "; ".join(name for name, passed in checks.items() if not passed)
        )
    return "qwen35-live-state" if using_live_state(config) else "qwen35-baseline"


def check(config):
    import os

    required = {
        "MTP_TOKENS": "2",
        "MTP_GDN_LAYOUT_FUSION": "1",
        "BETTERSCALE_MTP_GREEDY": "1",
        "BETTERSCALE_GDN_SMALL_COPIES": "1",
    }
    if any(os.environ.get(key) != value for key, value in required.items()):
        raise ValueError(
            "Qwen35 baseline requires its qualified MTP/layout/small-fish flags"
        )
    from ..qwen import check_runtime
    from ...patches import qwen_gdn, qwen_fia

    check_runtime("qwen35_pins.json")
    qwen_gdn.check_library()
    qwen_fia.check_library()
    from ...patches.qwen_fia.context_parallel import configure

    configure(os.environ)


def before_init(worker, config):
    from .integration import before_init

    before_init(worker, config)
    if using_live_state(config):
        from .state_backend import install

        install()


def after_init(worker):
    from .integration import after_init

    after_init(worker)
    if using_live_state(worker.vllm_config):
        from .state_address import install

        install()


def determine_available_memory(worker, native):
    if not using_live_state(worker.vllm_config):
        return native()
    from .state_backend import determine_available_memory

    return determine_available_memory(worker, native)


def model_loaded(worker):
    from .integration import model_loaded

    model_loaded(worker)
