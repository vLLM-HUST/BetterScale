"""The native fractional gate is bypassed only inside automatic initialization."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


class InitialAdmission(unittest.TestCase):
    def worker(self, manual=False, fail=False):
        path = Path(__file__).with_name("preflight_worker.py")
        cls = next(
            n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_init_device"
        )
        node = ast.ClassDef(
            name="PreflightWorker",
            bases=[ast.Name(id="MemoryWorker", ctx=ast.Load())],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        seen = []

        class Native:
            def _init_device(self):
                seen.append(self.cache_config.gpu_memory_utilization)
                if fail:
                    raise ValueError("native initialization failed")
                self.init_snapshot = NS(free_memory=1234)
                return "device"

        ns = {"MemoryWorker": Native}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                str(path),
                "exec",
            ),
            ns,
        )
        obj = ns["PreflightWorker"]()
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
