"""Installed-launch boundary: inert preparation, resource paths and process env."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from betterscale.__main__ import prepare


class QwenLauncher(unittest.TestCase):
    def test_prepares_existing_launcher_and_preserves_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "qualified.so"
            library.touch()
            overrides = {
                name: str(library)
                for name in (
                    "BETTERSCALE_GDN_LIBRARY",
                    "BETTERSCALE_GDN_HOST_LIBRARY",
                    "BETTERSCALE_FIA_LIBRARY",
                )
            }
            with patch.dict(os.environ, overrides):
                before = dict(os.environ)
                argv, env = prepare(Path("/models/qwen"), "5,6", 8123, Path(directory))
                self.assertEqual(dict(os.environ), before)
            self.assertEqual(argv[0], "bash")
            self.assertTrue(Path(argv[1]).is_file())
            self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "5,6")
            self.assertEqual(env["SERVING_PORT"], "8123")
            self.assertEqual(env["QWEN_MODEL_PATH"], "/models/qwen")
            self.assertEqual(env["BETTERSCALE_FIA_LIBRARY"], str(library))

    def test_bundled_libraries_are_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "qwen_gdn/libbs_gdn.so",
                "qwen_gdn/libbs_gdn_host.so",
                "qwen_fia/libbs_fia.so",
            ):
                file = root / "patches" / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.touch()
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("betterscale.__main__.__file__", str(root / "__main__.py")),
            ):
                _, env = prepare(Path("/model"), "0,1", 8000, root / "cache")
            self.assertEqual(
                env["BETTERSCALE_FIA_LIBRARY"],
                str(root / "patches/qwen_fia/libbs_fia.so"),
            )

    def test_rejects_invalid_device_or_port_before_launch(self):
        for devices, port in [("0,0", 8000), ("0", 8000), ("0,8", 8000), ("0,1", 0)]:
            with (
                self.subTest(devices=devices, port=port),
                self.assertRaises(ValueError),
            ):
                prepare(Path("/model"), devices, port, Path("/tmp/cache"))

    def test_missing_explicit_library_fails_before_exec(self):
        with patch.dict(os.environ, {"BETTERSCALE_GDN_LIBRARY": "/missing/native.so"}):
            with self.assertRaises(FileNotFoundError):
                prepare(Path("/model"), "0,1", 8000, Path("/tmp/cache"))

    def test_live_prepares_owned_entry_with_numerical_libraries(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5_moe_text",
                        "num_hidden_layers": 40,
                        "hidden_size": 2048,
                    }
                )
            )
            library = root / "numerical.so"
            library.touch()
            overrides = {
                name: str(library)
                for name in (
                    "BETTERSCALE_GDN_LIBRARY",
                    "BETTERSCALE_GDN_HOST_LIBRARY",
                    "BETTERSCALE_FIA_LIBRARY",
                )
            }
            with (
                patch.dict(os.environ, {**overrides, "BETTERSCALE_CONTEXT_PARALLEL": "0"}),
                patch(
                    "betterscale.models.qwen35.launch.validate", side_effect=lambda p: p
                ),
                patch(
                    "betterscale.models.qwen35.launch.matching_profile",
                    return_value={"pins_name": "qwen35_pins.json"},
                ),
            ):
                command, env = prepare(
                    root,
                    "0,1",
                    8000,
                    root / "cache",
                    runtime="live",
                    qwen35_runtime_dir=root / "donor",
                )
            self.assertTrue(command[1].endswith("models/qwen35/serve.sh"))
            self.assertEqual(env["BETTERSCALE_QWEN35_RUNTIME"], str(root / "donor"))
            self.assertEqual(command[command.index("--max-model-len") + 1], "262144")
            self.assertEqual(env["BETTERSCALE_FIA_LIBRARY"], str(library))
            with (
                patch.dict(
                    os.environ, {"BETTERSCALE_GDN_LIBRARY": "/missing/native.so"}
                ),
                patch(
                    "betterscale.models.qwen35.launch.validate", side_effect=lambda p: p
                ),
                patch(
                    "betterscale.models.qwen35.launch.matching_profile",
                    return_value={"pins_name": "qwen35_pins.json"},
                ),
            ):
                with self.assertRaises(FileNotFoundError):
                    prepare(
                        root,
                        "0,1",
                        8000,
                        root / "cache",
                        runtime="live",
                        qwen35_runtime_dir=root / "donor",
                    )
            with self.assertRaisesRegex(ValueError, "supports"):
                prepare(root, "0", 8000, root / "cache", runtime="live")
        with self.assertRaisesRegex(ValueError, "unknown runtime"):
            prepare(Path("/model"), "0,1", 8000, Path("/cache"), runtime="typo")
