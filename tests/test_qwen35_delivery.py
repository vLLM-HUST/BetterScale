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

    def test_live35_rejects_out_of_range_capacity_and_missing_runtime(self):
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
                dict(execution_seats=37, resident_seats=40),
                dict(token_pages=20),
            ]:
                with self.assertRaisesRegex(ValueError, "execution<=36"):
                    prepare(
                        model, "0,1", 8000, model / "cache", runtime="live", **kwargs
                    )


class BalancedDefault(unittest.TestCase):
    def test_default_model_entry_composes_resident_state_and_qualified_attention(self):
        import os
        from betterscale.patches.qwen_fia import context_parallel as cp

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({"text_config": dict(
                model_type="qwen3_5_moe_text", num_hidden_layers=40, hidden_size=2048)}))
            library = root / "libbs_fia_cp.so"
            library.write_bytes(b"qualified test artifact")
            (root / "native.json").write_text(json.dumps(dict(
                sha256=hashlib.sha256(library.read_bytes()).hexdigest())))
            overrides = {name: str(library) for name in (
                "BETTERSCALE_GDN_LIBRARY", "BETTERSCALE_GDN_HOST_LIBRARY",
                "BETTERSCALE_FIA_LIBRARY")}
            with (
                patch.dict(os.environ, overrides, clear=True),
                patch.object(cp, "__file__", str(root / "__init__.py")),
                patch("betterscale.models.qwen35.launch.validate", side_effect=lambda p: p),
            ):
                argv, env = prepare(root, "0,1", 8000, root / "cache",
                                    qwen35_runtime_dir=root / "donor")
                wide, _ = prepare(root, "0,1", 8000, root / "cache",
                                  qwen35_runtime_dir=root / "donor",
                                  execution_seats=36, resident_seats=36)
                self.assertEqual(wide[wide.index("--max-num-seqs")+1], "36")
                self.assertEqual(json.loads(wide[wide.index("--additional-config")+1])["state_resident_seats"], 36)
                self.assertIn(108, json.loads(wide[wide.index("--compilation-config")+1])["cudagraph_capture_sizes"])
                self.assertNotIn("BETTERSCALE_CONTEXT_PARALLEL", os.environ)
            self.assertEqual(env["BETTERSCALE_CONTEXT_PARALLEL"], "1")
            self.assertEqual(env["BETTERSCALE_CP_LIBRARY"], str(library))
            self.assertEqual(argv[argv.index("--scheduler-cls")+1],
                             "betterscale.models.qwen35.seat_scheduler.LiveStateScheduler")
            self.assertTrue(json.loads(argv[argv.index("--additional-config")+1])["using_live_runtime"])
            self.assertEqual(argv[argv.index("--kv-cache-memory-bytes")+1], "26038239232")
            self.assertEqual(argv[argv.index("--worker-cls")+1], "betterscale.qwen35_worker.Worker")
            with patch.object(cp, "__file__", str(root / "__init__.py")):
                library.write_bytes(b"wrong artifact")
                with self.assertRaisesRegex(ValueError, "qualified balanced"):
                    cp.configure({})
                library.unlink()
                with self.assertRaisesRegex(FileNotFoundError, "balanced attention"):
                    cp.configure({})
                disabled = {"BETTERSCALE_CONTEXT_PARALLEL": "0"}
                cp.configure(disabled)
                self.assertNotIn("BETTERSCALE_CP_LIBRARY", disabled)
                with self.assertRaisesRegex(ValueError, "0 or 1"):
                    cp.configure({"BETTERSCALE_CONTEXT_PARALLEL": "yes"})
