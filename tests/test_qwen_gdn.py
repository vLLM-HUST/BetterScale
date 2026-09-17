"""CPU contracts for the owned mixed graph metadata and native-library boundary."""

import os
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from betterscale.patches.qwen_gdn import check_library
from betterscale.patches.qwen_gdn.metadata import chunk_rows


class OwnedGDN(unittest.TestCase):
    def test_chunk_capacity_and_empty_sentinel(self):
        for capacity in (16, 32, 64, 128, 256, 512, 1024, 1536, 2048):
            for count in range(1, min(8, capacity) + 1):
                for size in (64, 256, 1216):
                    lengths = [1] * (count - 1) + [capacity - count + 1]
                    rows = chunk_rows(lengths, size, capacity)
                    live = [
                        (i, j)
                        for i, n in enumerate(lengths)
                        for j in range((n + size - 1) // size)
                    ]
                    self.assertEqual(rows[: len(live)], live)
                    self.assertEqual(len(rows), (capacity + size - 1) // size + 7)
                    self.assertTrue(all(x == (8, 0) for x in rows[len(live) :]))

    def test_reject_bad_packed_metadata(self):
        for lengths in ([], [0], [1, -1], [1] * 9, [513]):
            with self.assertRaises(ValueError):
                chunk_rows(lengths, 64, 512)

    def test_native_library_is_fail_closed(self):
        with patch.dict(os.environ, {"TASK_QUEUE_ENABLE": "1"}):
            with self.assertRaisesRegex(ValueError, "TASK_QUEUE_ENABLE"):
                check_library()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "unqualified.so"
            path.write_bytes(b"wrong ABI or owned-init disabled")
            with patch.dict(
                os.environ,
                {"TASK_QUEUE_ENABLE": "0", "BETTERSCALE_GDN_LIBRARY": str(path)},
            ):
                with self.assertRaisesRegex(ValueError, "qualified"):
                    check_library()

    def test_host_adapter_is_required_and_qualified(self):
        with tempfile.TemporaryDirectory() as root:
            kernel = Path(root) / "kernel.so"
            host = Path(root) / "host.so"
            kernel.write_bytes(b"qualified kernel fixture")
            host.write_bytes(b"qualified host fixture")
            manifest = {
                "sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
                "host_sha256": hashlib.sha256(host.read_bytes()).hexdigest(),
            }
            with (
                patch.dict(
                    os.environ,
                    {
                        "TASK_QUEUE_ENABLE": "0",
                        "BETTERSCALE_GDN_LIBRARY": str(kernel),
                        "BETTERSCALE_GDN_HOST_LIBRARY": str(host),
                    },
                ),
                patch("betterscale.patches.qwen_gdn.json.loads", return_value=manifest),
            ):
                self.assertEqual(check_library(), str(kernel.resolve()))
                host.write_bytes(b"stale host ABI")
                with self.assertRaisesRegex(ValueError, "HOST_LIBRARY"):
                    check_library()
                host.unlink()
                with self.assertRaisesRegex(ValueError, "HOST_LIBRARY"):
                    check_library()
