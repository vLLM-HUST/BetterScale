"""CPU proof that the opaque leaf does not specialize wave kind or token count."""

import sys
import types
import unittest
from unittest.mock import patch

import torch

from betterscale.patches.qwen_mc2 import install


class MC2Routing(unittest.TestCase):
    def test_compiled_leaf_reads_wave_kind_at_runtime(self):
        context = types.SimpleNamespace(attn_metadata=None)
        calls = []

        class Method:
            pass

        class Row:
            custom_op = bias = None
            quant_method = Method()
            tp_size = 2
            input_is_parallel = reduce_results = return_bias = True

            def __init__(self, k):
                self.weight = torch.empty((5120, k), dtype=torch.bfloat16, device="meta")

            def forward(self, x):
                calls.append("split")
                return torch.nn.functional.linear(x, self.weight), None

        layers = [(str(i), Row(3072 if i < 64 else 8704)) for i in range(128)]
        model = types.SimpleNamespace(named_modules=lambda: iter(layers))
        pg = types.SimpleNamespace(_get_backend=lambda device: types.SimpleNamespace(
            get_hccl_comm_name=lambda rank: "fixture"))
        tp = types.SimpleNamespace(world_size=2, rank_in_group=0, device_group=pg, cpu_group=None)

        def fused(x, weight, group, reduce_op):
            calls.append("fused")
            return x @ weight

        modules = {
            "torch_npu": types.SimpleNamespace(npu_mm_all_reduce_base=fused),
            "vllm.distributed": types.SimpleNamespace(get_tp_group=lambda: tp),
            "vllm.forward_context": types.SimpleNamespace(get_forward_context=lambda: context),
            "vllm.model_executor.layers.linear": types.SimpleNamespace(
                RowParallelLinear=Row, UnquantizedLinearMethod=Method),
            "vllm_ascend.distributed.parallel_state": types.SimpleNamespace(
                get_mc2_group=lambda: types.SimpleNamespace(device_group=pg)),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(torch, "npu", types.SimpleNamespace(synchronize=lambda: None), create=True),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "get_process_group_ranks", return_value=[0, 1]),
            patch.object(torch.distributed, "all_reduce"),
            patch.object(torch.distributed, "barrier"),
            # CPU Torch has no 'npu' device parser; production does.
            patch.object(torch, "device", side_effect=lambda name: name),
        ):
            self.assertEqual(install(model), 128)
        layer = layers[0][1]
        layer.weight = torch.ones((2, 4), dtype=torch.bfloat16)
        compiled = torch.compile(layer.forward, backend="eager", dynamic=True, fullgraph=True)
        for n, decode, expected in ((16, False, "fused"), (1, True, "split"),
                                    (32, False, "fused"), (8, True, "split"),
                                    (16, True, "split"), (16, False, "fused")):
            context.attn_metadata = {"gdn": types.SimpleNamespace(owned=types.SimpleNamespace(decode=decode))}
            x = torch.ones((n, 4), dtype=torch.bfloat16)
            out, bias = compiled(x)
            self.assertIsNone(bias)
            self.assertEqual(calls[-1], expected)
            torch.testing.assert_close(out, torch.full((n, 2), 4, dtype=torch.bfloat16))
        context.attn_metadata = None
        compiled(torch.ones((8, 4), dtype=torch.bfloat16))
        self.assertEqual(calls[-1], "split")


if __name__ == "__main__":
    unittest.main()
