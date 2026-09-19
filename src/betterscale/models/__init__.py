"""Explicit model admission, not plugin discovery or a Worker factory."""


def select(config):
    model = config.model_config
    hf = getattr(model.hf_config, "text_config", model.hf_config)
    if hf.model_type == "deepseek_v4":
        from . import dsv4 as implementation
    elif hf.model_type == "qwen3_5_text":
        from . import qwen as implementation
    else:
        raise ValueError(f"Unsupported BetterScale model: {hf.model_type}")
    route = implementation.validate(config)
    return implementation, route
