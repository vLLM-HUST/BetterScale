"""Native selection is isolated, early, complete and fail-closed; no NPU needed."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from betterscale.patches import hc_workspace as hc


class HCWorkspace(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        self.source = self.root / "donor/_cann_ops_custom/vendors/custom_transformer"
        self.source.mkdir(parents=True)
        self.library = self.root / "libcust_opmaster_rt2.0.so"
        self.library.write_bytes(b"new host")
        (self.source / "host").write_bytes(b"old host")
        (self.source / "alias").symlink_to(self.source / "host")
        (self.source / "kernel").write_bytes(b"unchanged kernel")
        self.manifest = {
            "changed": ["host", "alias"],
            "original_library_sha256": hashlib.sha256(b"old host").hexdigest(),
            "candidate_library_sha256": hc._digest(self.library),
        }
        (self.root / "native.json").write_text(json.dumps(self.manifest))
        (self.root / "version.info").write_text("Version=9.0.1\n")

    def test_complete_copy_does_not_write_through_symlink(self):
        output = self.root / "copy"
        hc._prepare_vendor(self.source, output, self.library, self.manifest)
        self.assertEqual((self.source / "host").read_bytes(), b"old host")
        self.assertTrue((self.source / "alias").is_symlink())
        for name in ("host", "alias"):
            self.assertFalse((output / name).is_symlink())
            self.assertEqual((output / name).read_bytes(), b"new host")
        self.assertEqual((output / "kernel").read_bytes(), b"unchanged kernel")

    def test_unqualified_donor_fails_before_copy(self):
        (self.source / "host").write_bytes(b"other build")
        with self.assertRaisesRegex(RuntimeError, "donor differs"):
            hc._prepare_vendor(
                self.source, self.root / "copy", self.library, self.manifest
            )
        self.assertFalse((self.root / "copy").exists())

    def test_bad_payload_fails_before_copy(self):
        self.library.write_bytes(b"wrong")
        with self.assertRaisesRegex(RuntimeError, "corrupted"):
            hc._prepare_vendor(
                self.source, self.root / "copy", self.library, self.manifest
            )
        self.assertFalse((self.root / "copy").exists())

    def test_early_selection_is_idempotent_and_preserves_other_vendors(self):
        package = ModuleType("vllm_ascend")
        native = ModuleType("vllm_ascend.utils")
        package.utils = native
        native._CUSTOM_OP_ENABLED = None
        native._CUSTOM_OP_BASE_DIR = str(self.root / "donor")
        self.enterContext(
            patch.dict(
                sys.modules, {"vllm_ascend": package, "vllm_ascend.utils": native}
            )
        )
        self.enterContext(patch.object(hc, "__file__", str(self.root / "__init__.py")))
        self.enterContext(patch.object(hc, "_envelope", None))
        self.enterContext(patch.object(hc.platform, "machine", return_value="aarch64"))
        self.enterContext(
            patch.dict(
                os.environ,
                {
                    "ASCEND_OPP_PATH": str(self.root),
                    "ASCEND_CUSTOM_OPP_PATH": str(self.source) + ":/other-vendor",
                },
            )
        )
        hc.install()
        self.addCleanup(hc._envelope.cleanup)
        first = native._CUSTOM_OP_BASE_DIR
        hc.install()
        self.assertEqual(first, native._CUSTOM_OP_BASE_DIR)
        self.assertEqual(
            os.environ["ASCEND_CUSTOM_OPP_PATH"].split(":")[1:], ["/other-vendor"]
        )
        self.assertTrue(Path(first).is_dir())
        self.assertEqual((self.source / "host").read_bytes(), b"old host")


if __name__ == "__main__":
    unittest.main()
