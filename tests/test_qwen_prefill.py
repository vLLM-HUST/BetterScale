"""Separate Qwen entry admission and import isolation; no device initialization."""

import subprocess
import sys
import unittest


class QwenPrefill(unittest.TestCase):
    def test_bucket_rounds_up_and_leaves_decode_alone(self):
        from betterscale.patches.qwen_prefill import prefill_bucket

        for n, expected in [
            (0, None),
            (1, None),
            (2, 16),
            (16, 16),
            (17, 32),
            (512, 512),
            (513, 1024),
            (2051, None),
        ]:
            self.assertEqual(prefill_bucket(n), expected)

    def test_independent_worker_and_fail_closed_admission(self):
        code = """
import sys, types
from types import SimpleNamespace as S
native = types.ModuleType("vllm_ascend.worker.worker")
class Native:
    def __init__(self, *args, **kwargs): raise AssertionError("unexpected construction")
native.NPUWorker = Native
sys.modules[native.__name__] = native
from betterscale.qwen_worker import Worker, validate_config
from betterscale.patches.qwen_prefill import PREFILLS
assert Worker.__bases__ == (Native,)
assert "betterscale.models.dsv4" not in sys.modules
from betterscale.qwen_worker import MixedWorker
assert Worker is MixedWorker
assert "betterscale.patches.qwen_prefill.padding" not in sys.modules
hf = S(model_type="qwen3_5_text", num_hidden_layers=64, hidden_size=5120,
       head_dim=256, num_attention_heads=24, num_key_value_heads=4)
c = S(parallel_config=S(tensor_parallel_size=2, data_parallel_size=1,
        pipeline_parallel_size=1, enable_expert_parallel=False,
        decode_context_parallel_size=1, prefill_context_parallel_size=1),
      scheduler_config=S(max_num_seqs=8, max_num_batched_tokens=2048,
        scheduler_cls=None, async_scheduling=True),
      model_config=S(hf_config=S(text_config=hf), dtype="torch.bfloat16",
        quantization=None, max_model_len=8192, multimodal_config=None),
      load_config=S(load_format="auto"), speculative_config=None,
      cache_config=S(enable_prefix_caching=False),
      compilation_config=S(cudagraph_mode="FULL", cudagraph_capture_sizes=[1,2,4,8,*PREFILLS]))
validate_config(c)
for obj, key, bad in [(c, "speculative_config", S(method="mtp")),
                     (c.cache_config, "enable_prefix_caching", True),
                     (c.parallel_config, "tensor_parallel_size", 4),
                     (c.scheduler_config, "async_scheduling", False),
                     (c.compilation_config, "cudagraph_capture_sizes", [1,2,4,8,512]),
                     (c.model_config, "multimodal_config", S(get_limit_per_prompt=lambda kind: 1))]:
    old=getattr(obj,key); setattr(obj,key,bad)
    try:
        validate_config(c)
    except ValueError: pass
    else: raise AssertionError(key)
    setattr(obj,key,old)
c.cache_config.enable_prefix_caching=True
c.cache_config.mamba_cache_mode="align"
validate_config(c, mixed=True)
c.cache_config.mamba_cache_mode="all"
try:
    validate_config(c, mixed=True)
except ValueError: pass
else: raise AssertionError("owned APC all is not qualified")
c.cache_config.enable_prefix_caching=False
c.speculative_config=S(method="mtp",num_speculative_tokens=2,enforce_eager=False)
c.compilation_config.cudagraph_mode="FULL_AND_PIECEWISE"
c.compilation_config.cudagraph_capture_sizes=[1,2,4,8,16,24]
validate_config(c)
assert "betterscale.patches.qwen_layout" not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_weight_packing_preserves_parameter_identity_and_values(self):
        code = r"""
import sys, types, torch
from torch import nn
q = types.ModuleType("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
a = types.ModuleType("vllm_ascend.ops.gdn")
def forward_core(self): pass
class Qwen(nn.Module):
    _forward_core = forward_core
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(5120,5120,4,groups=5120,bias=False,dtype=torch.bfloat16)
class Ascend:
    _forward_core = forward_core
q.QwenGatedDeltaNetAttention = Qwen
a.AscendGatedDeltaNetAttention = Ascend
sys.modules[q.__name__]=q; sys.modules[a.__name__]=a
from betterscale.patches.qwen_layout import pack_conv_weights
model=nn.ModuleList([Qwen() for _ in range(48)])
refs=[(x.conv1d.weight, x.conv1d.weight.detach().clone()) for x in model]
for repeat in range(2):
    assert pack_conv_weights(model)==48
    for layer,(parameter,values) in zip(model,refs):
        assert layer.conv1d.weight is parameter
        assert torch.equal(parameter,values)
        assert parameter.view(5120,4).T.is_contiguous()
"""
        subprocess.run([sys.executable, "-c", code], check=True)
