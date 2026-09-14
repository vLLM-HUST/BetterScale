import random
import unittest

from timeline_windows import minus, union


class IntervalAccountingTest(unittest.TestCase):
    def test_union_and_subtraction_against_integer_coverage(self):
        rng = random.Random(17)
        for _ in range(100):
            xs, ys = [], []
            for target in (xs, ys):
                for _ in range(12):
                    a = rng.randrange(30)
                    target.append((a, a + rng.randrange(1, 10)))

            def covered(intervals):
                return {v for a, b in intervals for v in range(a, b)}

            self.assertEqual(covered(union(xs)), covered(xs))
            got = minus(xs, ys)
            self.assertEqual(covered(got), covered(xs) - covered(ys))
            self.assertEqual(sum(b - a for a, b in got), len(covered(got)))


if __name__ == "__main__":
    unittest.main()
