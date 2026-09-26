"""CPU contracts for the baseline namespace port; no new execution loop."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import sys

from betterscale.models import qwen35

ROOT = Path(__file__).resolve().parents[1]


def config():
    hf = S(
        model_type="qwen3_5_moe_text",
        num_hidden_layers=40,
        hidden_size=2048,
        head_dim=256,
        num_attention_heads=16,
        num_key_value_heads=2,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    return S(
        model_config=S(
            hf_text_config=hf,
            hf_config=S(text_config=hf),
            dtype="torch.bfloat16",
            quantization=None,
            max_model_len=262144,
            multimodal_config=None,
        ),
        parallel_config=S(
            tensor_parallel_size=2,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=S(
            max_num_seqs=16,
            max_num_batched_tokens=4096,
            scheduler_cls=qwen35.SCHEDULER,
            async_scheduling=True,
        ),
        speculative_config=S(
            method="mtp", num_speculative_tokens=2, enforce_eager=False
        ),
        compilation_config=S(
            cudagraph_mode="FULL", cudagraph_capture_sizes=qwen35.CAPTURE_SIZES
        ),
        cache_config=S(enable_prefix_caching=True, mamba_cache_mode="align"),
        load_config=S(load_format="auto"),
        kv_transfer_config=None,
        lora_config=None,
    )


class Baseline(unittest.TestCase):
    def test_selection_and_rejected_protocol_changes(self):
        from betterscale.models import select

        c = config()
        self.assertEqual(select(c), (qwen35, "qwen35-baseline"))
        for owner, name, value in (
            ("scheduler_config", "async_scheduling", False),
            ("scheduler_config", "scheduler_cls", None),
            ("scheduler_config", "max_num_batched_tokens", 1024),
            ("parallel_config", "tensor_parallel_size", 1),
            ("speculative_config", "num_speculative_tokens", 1),
            ("compilation_config", "cudagraph_capture_sizes", [3, 6, 12, 24, 48]),
            ("model_config", "max_model_len", 262145),
        ):
            c = config()
            setattr(getattr(c, owner), name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                select(c)

    def test_capture_entry_preserves_mixed_keys_without_another_worker(self):
        import subprocess

        code = """
import sys
from types import SimpleNamespace as S
calls=[]
class Config:
    def adjust_cudagraph_sizes_for_spec_decode(self, *args, **kwargs):
        calls.append(self.cudagraph_mode)
class Worker: pass
sys.modules['vllm.config']=S(CUDAGraphMode=S(FULL='FULL'))
sys.modules['vllm.config.compilation']=S(CompilationConfig=Config)
sys.modules['betterscale.worker']=S(Worker=Worker)
from betterscale import qwen35_worker as entry
assert entry.Worker is Worker
wrapped=Config.adjust_cudagraph_sizes_for_spec_decode
entry._install_capture_policy()
assert Config.adjust_cudagraph_sizes_for_spec_decode is wrapped
c=Config(); c.cudagraph_mode='FULL'
c.adjust_cudagraph_sizes_for_spec_decode(2)
assert calls==[]
c.cudagraph_mode='FULL_AND_PIECEWISE'
c.adjust_cudagraph_sizes_for_spec_decode(2)
assert calls==['FULL_AND_PIECEWISE']
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_metadata_capacity_domains_keep_dense_default(self):
        import torch
        from betterscale.patches.qwen_gdn.metadata import Metadata, chunk_rows

        allocations = []

        def kernels(*args, **kwargs):
            allocations.append((args, kwargs))
            return S()

        with (
            patch.dict(
                sys.modules,
                {"betterscale.patches.qwen_gdn.runtime": S(Kernels=kernels)},
            ),
            patch.dict("os.environ", {"BETTERSCALE_GDN_LIBRARY": "cpu-fixture"}),
        ):
            dense = Metadata(4096, False, "cpu")
            moe = Metadata(4096, False, "cpu", requests=16, key_heads=8, value_heads=16)
        self.assertEqual((dense.requests, moe.requests), (9, 17))
        self.assertEqual((len(dense.cu), len(moe.cu)), (10, 18))
        self.assertEqual(moe.indices[64].shape, torch.Size([79, 2]))
        self.assertEqual([x[1]["value_heads"] for x in allocations], [24, 16])
        rows = chunk_rows([1] * 16, 64, 4096, requests=16)
        self.assertEqual(rows[:16], [(i, 0) for i in range(16)])
        self.assertEqual(rows[16:], [(16, 0)] * 63)
        with self.assertRaises(ValueError):
            chunk_rows([1] * 16, 64, 4096)

    def test_namespace_is_closed_and_pins_are_packaged(self):
        import tomllib

        package = ROOT / "src/betterscale/models/qwen35"
        local_names = {p.stem for p in package.glob("*.py")}
        for path in package.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module in local_names:
                    self.assertEqual(node.level, 1, (path, node.module))
                if isinstance(node, ast.Import):
                    self.assertFalse(
                        local_names.intersection(x.name for x in node.names), path
                    )
        data = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertIn(
            "qwen35_pins.json",
            data["tool"]["setuptools"]["package-data"]["betterscale"],
        )
        pins = json.loads((ROOT / "src/betterscale/qwen35_pins.json").read_text())
        self.assertTrue(
            any("llm_base_proposer" in x["path"] for x in pins["source_files"])
        )


if __name__ == "__main__":
    unittest.main()
