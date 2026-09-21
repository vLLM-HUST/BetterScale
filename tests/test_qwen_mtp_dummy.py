"""Speculative capture rows must be legal even when native dummies overfill."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/'prototypes/qwen38-serving/mtp/count_policy.py'
spec = importlib.util.spec_from_file_location('mtp_count_policy',path)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)

class MTPDummyTest(unittest.TestCase):
    def test_general_dummy_capacity_and_inactive_padding(self):
        for width in range(2,6):
            for tokens in policy.BINS:
                n = min(tokens,8)
                raw = [tokens//n]*n
                raw[-1] += tokens%n
                lengths = policy.capture_lengths(raw,width)
                self.assertEqual(len(lengths),n)
                self.assertLessEqual(sum(lengths),tokens)
                self.assertTrue(all(1 <= m <= width for m in lengths))
                self.assertEqual(sum(raw),tokens)
    def test_valid_rows_are_preserved(self):
        self.assertEqual(policy.capture_lengths([1,2,3],3),(1,2,3))
    def test_invalid_rows_rejected(self):
        for lengths,width in (([],3),([0],3),([1]*9,3),([1],6)):
            with self.assertRaises(AssertionError):
                policy.capture_lengths(lengths,width)
