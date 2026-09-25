"""TP sharding changes leaf geometry, not resident/page capacity ownership."""

import json
from pathlib import Path
import pytest
from betterscale.live.llm.qwen35 import Geometry


def config():
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "prototypes/qwen35-state-lanes/qwen35-35b-text-config.json"
        ).read_text()
    )


def test_qwen35_moe_tp2_is_local_not_global_geometry():
    g = Geometry.from_config(config(), tensor_parallel_size=2)
    assert len(g.layer_types) == 40
    assert (g.kv_heads, g.gdn_key_heads, g.gdn_value_heads) == (1, 8, 16)
    assert g.conv_channels == 4096 and g.hidden_size == 2048
    assert g.gdn_key_dim == g.gdn_value_dim == 128


def test_no_implicit_tp_or_fractional_head_shards():
    with pytest.raises(ValueError, match="dense TP1 or MoE TP2"):
        Geometry.from_config(config())
    c = config()
    c["num_key_value_heads"] = 3
    with pytest.raises(ValueError, match="divide evenly"):
        Geometry.from_config(c, tensor_parallel_size=2)
