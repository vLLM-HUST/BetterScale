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
assert "betterscale.worker" not in sys.modules
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
"""
        subprocess.run([sys.executable, "-c", code], check=True)
