"""Read-only admission for the qualified patch envelope; not a launch preset.

Keep runtime safety and measured scope explicit. Removing the custom CLI does
not qualify larger shapes, more seats or a different parallel layout. Model,
KV budget, networking and environment remain the user's native vLLM settings.
"""

PATCH_IDS = (
    "compat-lcm",
    "target-full",
    "ordered-replay",
    "cpu-qli",
    "private-draft-banks",
    "split-draft-context",
    "stable-receipt-cut",
)


DP_PATCH_IDS = (
    "compat-lcm",
    "target-full-dsa",
    "stable-receipt-cut",
    "dual-target-banks",
    "owned-ingress",
    "device-preparation",
    "device-metadata",
)


def validate_worker_config(config):
    """Reject unqualified combinations before allocating model weights."""
    p = config.parallel_config
    s = config.scheduler_config
    m = config.model_config
    spec = config.speculative_config
    extra = config.additional_config
    native_dp = (p.tensor_parallel_size, p.data_parallel_size) == (1, 8)
    checks = {
        "TP8/DP1 or TP1/DP8, PP1, EP": (
            p.tensor_parallel_size,
            p.data_parallel_size,
            p.pipeline_parallel_size,
            p.enable_expert_parallel,
        )
        in ((8, 1, 1, True), (1, 8, 1, True)),
        "DCP1, PCP1": p.decode_context_parallel_size == 1
        and p.prefill_context_parallel_size == 1,
        "DSV4 Flash43 layers /4096 hidden /256 experts": m.hf_config.model_type
        == "deepseek_v4"
        and (
            m.hf_config.num_hidden_layers,
            m.hf_config.hidden_size,
            m.hf_config.n_routed_experts,
        )
        == (43, 4096, 256),
        "qualified seats/budget/context": (s.max_num_seqs, s.max_num_batched_tokens)
        == ((2, 1026) if native_dp else (4, 4128))
        and m.max_model_len <= (16384 if native_dp else 15104),
        "DSpark K5 with native eager drafter": spec is not None
        and spec.method == "dspark"
        and spec.num_speculative_tokens == 5
        and spec.enforce_eager,
        "standard rejection": spec is not None
        and spec.rejection_sample_method == "standard",
        "layout-matched DSA backend": extra.get("enable_dsa_cp") is (not native_dp),
        "prefix caching disabled": not config.cache_config.enable_prefix_caching,
        "real W8A8 model": config.load_config.load_format == "auto"
        and m.quantization == "ascend",
        "native scheduler": s.scheduler_cls is None,
        "explicit graph mode": str(config.compilation_config.cudagraph_mode) == "FULL",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "Outside the qualified strengthen-dsv4 envelope: " + "; ".join(failed)
        )
