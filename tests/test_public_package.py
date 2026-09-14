"""Public distribution identity and the alias-only Worker entry; no NPU required."""

import json
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path


class PublicPackage(unittest.TestCase):
    def test_version_matches_distribution_metadata(self):
        import betterscale
        import strengthen_dsv4

        root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((root / "pyproject.toml").read_text())["project"]
        self.assertEqual(metadata["name"], "vllm-betterscale")
        self.assertEqual(betterscale.__version__, metadata["version"])
        self.assertEqual(strengthen_dsv4.__version__, metadata["version"])
        self.assertEqual(metadata["dependencies"], [])

    def test_worker_alias_is_the_same_class_without_initialization(self):
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
from strengthen_dsv4.worker import Worker as Original
from strengthen_dsv4.compat import pins
assert Worker is Original
assert Worker.__bases__ == (Native,)
print(json.dumps({'same_class': True, 'pins': bool(pins()['source_files'])}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True, text=True
        )
        self.assertEqual(json.loads(result.stdout), {"same_class": True, "pins": True})


if __name__ == "__main__":
    unittest.main()
