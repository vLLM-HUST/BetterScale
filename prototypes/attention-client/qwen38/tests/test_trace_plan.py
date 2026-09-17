"""Protect retained-prefix accounting independently from accelerator numerics."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace_plan import Seat


class PrefixAccounting(unittest.TestCase):
    def test_pending_anchor_is_encoded_once(self):
        s = Seat(
            dict(
                trace_id="a",
                turns=[dict(ids=[1, 2, 3], output=3), dict(ids=[9, 10], output=2)],
            )
        )
        self.assertEqual(s.take(2), [1, 2])
        self.assertFalse(s.finish_prefill(2, 99))
        self.assertEqual(s.take(2), [3])
        self.assertTrue(s.finish_prefill(1, 4))
        s.finish_decode(2, 6)
        self.assertEqual((s.cursor, s.generated), (5, 3))
        s.next_turn()
        self.assertEqual(s.todo, [6, 9, 10])
        self.assertEqual(s.reused, 5)
        self.assertEqual(s.take(8), [6, 9, 10])
        s.finish_prefill(3, 11)
        s.finish_decode(1, 12)
        s.next_turn()
        self.assertTrue(s.done)
        self.assertEqual((s.cursor, s.generated, s.prefilled), (9, 5, 6))
        # Logical history is five recorded input + five actual generated tokens.
        self.assertEqual(s.cursor + 1, 10)

    def test_output_budget_cannot_be_overrun(self):
        s = Seat(dict(trace_id="a", turns=[dict(ids=[1], output=1)]))
        s.take(8)
        s.finish_prefill(1, 2)
        with self.assertRaises(ValueError):
            s.finish_decode(1, 3)
        s.next_turn()
        self.assertTrue(s.done)


if __name__ == "__main__":
    unittest.main()
