import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expert_partition import ExpertPartition


class PartitionTests(unittest.TestCase):
    def test_every_expert_has_exactly_one_owner(self):
        for owners in (3, 4, 8):
            with self.subTest(owners=owners):
                partition = ExpertPartition(owners)
                seen = []
                for owner in range(owners):
                    begin, end = partition.bounds(owner)
                    self.assertLessEqual(end - begin, partition.slots)
                    for expert in range(begin, end):
                        self.assertEqual(
                            partition.locate(expert), (owner, expert - begin)
                        )
                        seen.append(expert)
                self.assertEqual(seen, list(range(512)))

    def test_e3_padding_has_no_route(self):
        partition = ExpertPartition(3)
        self.assertEqual(partition.slots, 171)
        self.assertEqual(partition.bounds(2), (342, 512))
        self.assertEqual(partition.locate(511), (2, 169))
        for expert in (-1, 512, 513):
            with self.assertRaises(ValueError):
                partition.locate(expert)
        for owner in (-1, 3):
            with self.assertRaises(ValueError):
                partition.bounds(owner)


if __name__ == "__main__":
    unittest.main()
