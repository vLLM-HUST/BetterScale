"""The native fractional gate is bypassed only inside automatic initialization."""

from types import SimpleNamespace as NS
import unittest


class InitialAdmission(unittest.TestCase):
    def worker(self, manual=False, fail=False):
        from betterscale.patches.auto_kv import init_device

        seen = []

        class Native:
            def _init_device(self):
                seen.append(self.cache_config.gpu_memory_utilization)
                if fail:
                    raise ValueError("native initialization failed")
                self.init_snapshot = NS(free_memory=1234)
                return "device"

        class Memory(Native):
            def _init_device(self):
                return init_device(self, super()._init_device)

        obj = Memory()
        obj.cache_config = NS(
            kv_cache_memory_bytes=512 if manual else None, gpu_memory_utilization=0.9
        )
        return obj, seen

    def test_auto_uses_free_bytes_and_restores_user_configuration(self):
        obj, seen = self.worker()
        self.assertEqual(obj._init_device(), "device")
        self.assertEqual(seen, [0.0])
        self.assertEqual(obj.requested_memory, 1234)
        self.assertEqual(obj.cache_config.gpu_memory_utilization, 0.9)

    def test_native_failure_restores_configuration(self):
        obj, _ = self.worker(fail=True)
        with self.assertRaisesRegex(ValueError, "native initialization"):
            obj._init_device()
        self.assertEqual(obj.cache_config.gpu_memory_utilization, 0.9)

    def test_explicit_manual_budget_preserves_native_policy(self):
        obj, seen = self.worker(manual=True)
        obj._init_device()
        self.assertEqual(seen, [0.9])


class CapacityLogging(unittest.TestCase):
    def test_budget_and_ready_use_native_logger_namespace(self):
        from unittest.mock import patch
        from betterscale.patches.auto_kv import snapshot

        worker = NS()
        worker.rank = 2
        worker.model_config = NS(max_model_len=524288)
        worker.vllm_config = NS(scheduler_config=NS(max_num_seqs=4))
        npu = NS(
            synchronize=lambda: None,
            mem_get_info=lambda: (1 << 30, 64 << 30),
            memory_allocated=lambda: 59 << 30,
            memory_reserved=lambda: 60 << 30,
        )
        with patch.dict("sys.modules", {"torch": NS(npu=npu)}):
            with self.assertLogs("vllm", level="INFO") as captured:
                snapshot(
                    worker,
                    "physical_budget_after_trial_release",
                    kv_budget=15 << 30,
                    measured_target_graph=1 << 30,
                    safety=1 << 30,
                )
                snapshot(worker, "ready_after_capture")
        self.assertIn("budget=15.000 GiB", captured.output[0])
        self.assertIn("safety=1.000 GiB", captured.output[0])
        self.assertIn(
            "context ceiling=524288 tokens, active seats=4", captured.output[1]
        )
