import importlib.util
from concurrent.futures import Future
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import json

spec = importlib.util.spec_from_file_location(
    "pressure_monitor", Path(__file__).with_name("pressure_monitor.py")
)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class DurablePressure(unittest.TestCase):
    def test_partial_receipt_survives_next_sampling_failure(self):
        first, pending = Future(), Future()
        first.set_result({"request": 0})
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(
                    monitor.urllib.request, "urlopen", side_effect=OSError("offline")
                ),
                patch.object(monitor.time, "sleep", side_effect=KeyboardInterrupt),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    monitor.collect(
                        [first, pending], "http://127.0.0.1:30880", root, "wave", 0
                    )
            saved = json.loads((root / "wave-completed.jsonl").read_text())
            self.assertEqual(saved["result"], {"request": 0})
            row = json.loads((root / "wave-pressure.jsonl").read_text())
            self.assertIn("offline", row["observation_error"])

    def test_request_error_is_saved_and_propagated(self):
        failed = Future()
        failed.set_exception(ValueError("request failed"))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "request failed"):
                monitor.collect([failed], "http://127.0.0.1:30880", root, "wave", 0)
            self.assertIn("request failed", (root / "wave-completed.jsonl").read_text())
