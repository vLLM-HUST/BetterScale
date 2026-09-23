"""Captured capacity is not the live request count, especially with penalties."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'prototypes/qwen35-moe-serving/draft_output.py'
spec = importlib.util.spec_from_file_location('draft_output', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Rows:
    ndim = 2
    def __init__(self, rows, width=2):
        self.shape = (rows, width)
    def __getitem__(self, key):
        return Rows(key.stop, self.shape[1])


class DraftOutput(unittest.TestCase):
    def test_padding_never_becomes_request_history(self):
        captured = Rows(85)
        self.assertEqual(module.live_rows(captured, 1, 2).shape, (1, 2))
        self.assertEqual(module.live_rows(captured, 8, 2).shape, (8, 2))
        self.assertIs(module.live_rows(captured, 0, 2), captured)
        with self.assertRaises(RuntimeError):
            module.live_rows(Rows(1), 8, 2)
        with self.assertRaises(RuntimeError):
            module.live_rows(Rows(85, 3), 1, 2)
