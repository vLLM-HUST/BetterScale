"""Dummy-only draft shrink; target dict overrides do not propagate upstream."""
def small_dspark(config):
    from vllm.config import SpeculativeConfig
    config = SpeculativeConfig.hf_config_override(config)
    config.num_hidden_layers = 4
    config.compress_ratios = [0, 0, 4, 128, 0, 0, 0]
    config.n_routed_experts = 8
    config.num_hash_layers = 0
    config.num_nextn_predict_layers = 1
    config.dspark_target_layer_ids = [1, 2, 3]
    return config


def install_dummy_draft_config():
    from vllm.config import SpeculativeConfig
    SpeculativeConfig.compose_draft_hf_overrides = staticmethod(lambda _: small_dspark)
