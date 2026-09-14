"""Public distribution identity and the native Worker entry; no NPU required."""

import json
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path


class PublicPackage(unittest.TestCase):
    def test_version_matches_distribution_metadata(self):
        import betterscale

        root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((root / "pyproject.toml").read_text())["project"]
        self.assertEqual(metadata["name"], "vllm-betterscale")
        self.assertEqual(betterscale.__version__, metadata["version"])
        self.assertEqual(metadata["dependencies"], [])

    def test_worker_is_defined_in_public_namespace_without_initialization(self):
        # A fresh interpreter avoids contaminating other tests' module fixtures.
        code = """
import json, sys, types
native = types.ModuleType('vllm_ascend.worker.worker')
class Native:
    def __init__(self, *args, **kwargs):
        raise AssertionError('Import must not construct a worker')
native.NPUWorker = Native
sys.modules[native.__name__] = native
from betterscale.worker import Worker
from betterscale.compat import pins
assert Worker.__module__ == "betterscale.worker"
assert issubclass(Worker, Native)
assert Worker.__bases__[-1] is Native
print(json.dumps({'native_namespace': True, 'pins': bool(pins()['source_files'])}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True, text=True
        )
        self.assertEqual(
            json.loads(result.stdout), {"native_namespace": True, "pins": True}
        )


if __name__ == "__main__":
    unittest.main()
