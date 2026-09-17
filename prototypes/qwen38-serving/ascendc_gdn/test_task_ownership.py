"""CPU check: initialization ownership matches the donor scheduler's wave rule."""

import unittest


class Ownership(unittest.TestCase):
    def test_paired_scheduler(self):
        cores = 24
        for requests in range(1, 65):
            tasks = requests * 24
            loops = (tasks + 2 * cores - 1) // (2 * cores)
            dummy = 0 < tasks % (2 * cores) <= cores
            observed = {}
            for core in range(cores):
                width = 2 if tasks > cores else 1
                first = core * width
                while first < tasks:
                    for task in range(first, min(first + width, tasks)):
                        self.assertNotIn(task, observed)
                        observed[task] = core
                    wave = first // (2 * cores)
                    width = 1 if wave + 2 == loops and dummy else 2
                    first = (wave + 1) * 2 * cores + width * core
            self.assertEqual(len(observed), tasks)
            for task, core in observed.items():
                wave = task // (2 * cores)
                remaining = tasks - wave * 2 * cores
                width = 1 if remaining <= cores else 2
                self.assertEqual((task % (2 * cores)) // width, core)


if __name__ == "__main__":
    unittest.main()
