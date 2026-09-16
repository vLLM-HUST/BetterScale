"""Protect the shared-expert dependency cut without importing a device runtime."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


class SubmitOrder(unittest.TestCase):
    def test_useful_work_precedes_collect_and_output_is_not_bank_alias(self):
        events = []
        fake = types.ModuleType("fake")
        deps = {
            name: fake
            for name in (
                "torch",
                "torch_npu",
                "common",
                "control",
                "settings",
                "device_joint",
            )
        }
        fake.acl_api = fake.connect = fake.Kernels = Mock()
        fake.ALIGN = 2**21
        fake.CONTRACT = {}
        fake.LAYERS = 4
        selector = types.ModuleType("vllm_ascend.ops.fused_moe.experts_selector")
        selector.select_experts = Mock(return_value=("probs", "ids"))
        deps[selector.__name__] = selector
        spec = importlib.util.spec_from_file_location(
            "next_remote_test", Path(__file__).with_name("next_remote.py")
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, deps):
            spec.loader.exec_module(module)
            session = module.Session.__new__(module.Session)
            output = Mock()
            output.__add__ = Mock(return_value="independent_result")
            bank = types.SimpleNamespace(
                config=[0] * 11,
                x=Mock(),
                ids=Mock(),
                probs=Mock(),
                submit_graph=types.SimpleNamespace(
                    replay=lambda: events.append("submit")
                ),
                collect_graph=types.SimpleNamespace(
                    replay=lambda: events.append("collect")
                ),
                output=output,
            )
            session.banks = {3: bank}

            def shared(x):
                events.append("shared")
                return "shared_output"

            result = session.forward(
                17, types.SimpleNamespace(shape=(3, 2048)), "logits", shared
            )
        self.assertEqual(events, ["submit", "shared", "collect"])
        self.assertEqual(bank.config[5], 17)
        self.assertEqual(result, "independent_result")
        output.__add__.assert_called_once_with("shared_output")


if __name__ == "__main__":
    unittest.main()
