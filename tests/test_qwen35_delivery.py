"""Inert delivery contracts: isolated donor preparation and the native entry."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from betterscale.models.qwen35 import runtime
from betterscale.__main__ import prepare


class Delivery(unittest.TestCase):
    def test_runtime_staging_is_exact_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resources = root / "package/models/qwen35"
            resources.mkdir(parents=True)
            source = root / "donor"
            (source / "vllm_ascend").mkdir(parents=True)
            file = source / "vllm_ascend/leaf.py"
            for name in [
                "libvllm_ascend_kernels.so",
                "vllm_ascend_C.fixture.so",
                "_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so",
            ]:
                native = source / "vllm_ascend" / name
                native.parent.mkdir(parents=True, exist_ok=True)
                native.touch()
            before, after = b"value = 1\n", b"value = 2\n"
            digest = lambda data: hashlib.sha256(data).hexdigest()
            file.write_bytes(before)
            (resources / "runtime.patch").write_text(
                "--- a/vllm_ascend/leaf.py\n+++ b/vllm_ascend/leaf.py\n"
                "@@ -1 +1 @@\n-value = 1\n+value = 2\n"
            )
            (resources / "runtime.json").write_text(
                json.dumps(
                    dict(
                        files=[
                            dict(
                                path="vllm_ascend/leaf.py",
                                before=digest(before),
                                after=digest(after),
                            )
                        ]
                    )
                )
            )
            (resources.parents[1] / "qwen35_pins.json").write_text(
                json.dumps(
                    dict(
                        source_files=[
                            dict(path="vllm_ascend/leaf.py", sha256=digest(after))
                        ]
                    )
                )
            )
            with patch.object(runtime, "ROOT", resources):
                output = runtime.prepare(source, root / "runtime")
                self.assertEqual(file.read_bytes(), before)
                self.assertEqual((output / "vllm_ascend/leaf.py").read_bytes(), after)
                with self.assertRaises(FileExistsError):
                    runtime.prepare(source, output)
                runtime.prepare(output, root / "already-qualified")
                file.write_text("unqualified\n")
                with self.assertRaisesRegex(ValueError, "Unqualified donor input"):
                    runtime.prepare(source, root / "invalid")
                self.assertFalse((root / "invalid").exists())

    def test_live35_rejects_legacy_capacity_and_missing_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "config.json").write_text(
                json.dumps(
                    dict(
                        model_type="qwen3_5_moe_text",
                        num_hidden_layers=40,
                        hidden_size=2048,
                    )
                )
            )
            with self.assertRaisesRegex(ValueError, "qwen35-runtime-dir"):
                prepare(model, "0,1", 8000, model / "cache", runtime="live")
            for kwargs in [
                dict(resident_seats=40),
                dict(execution_seats=8),
                dict(token_pages=20),
            ]:
                with self.assertRaisesRegex(ValueError, "E16/R20"):
                    prepare(
                        model, "0,1", 8000, model / "cache", runtime="live", **kwargs
                    )
