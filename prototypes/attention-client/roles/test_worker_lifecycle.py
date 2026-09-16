"""CPU-only native shutdown ordering; no donor imports or accelerator runtime."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class LifecycleTest(unittest.TestCase):
    def module(self):
        torch = ModuleType("torch")
        torch.nn = SimpleNamespace(Module=object)
        donor = ModuleType("vllm_ascend.worker.worker")

        class Base:
            def shutdown(self):
                self.order.append("native")

        donor.NPUWorker = Base
        spec = importlib.util.spec_from_file_location(
            "test_role_worker", Path(__file__).with_name("role_worker.py")
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules, {"torch": torch, "vllm_ascend.worker.worker": donor}
        ):
            spec.loader.exec_module(module)
        return module

    def test_drain_precedes_native_release(self):
        module = self.module()
        worker = module.Worker()
        worker.order = []
        module.SESSION = object()
        worker.drain_experts = lambda: worker.order.append("drain")
        worker.shutdown()
        self.assertEqual(worker.order, ["drain", "native"])

    def test_explicitly_drained_shutdown(self):
        module = self.module()
        worker = module.Worker()
        worker.order = []
        worker.shutdown()
        self.assertEqual(worker.order, ["native"])

    def test_failed_drain_does_not_release_native_storage(self):
        module = self.module()
        worker = module.Worker()
        worker.order = []
        module.SESSION = object()

        def fail():
            raise TimeoutError("drain")

        worker.drain_experts = fail
        with self.assertRaises(TimeoutError):
            worker.shutdown()
        self.assertEqual(worker.order, [])
