import dataclasses
from types import SimpleNamespace
import unittest

import torch

from storage import bank, backing_views, restore


class StorageContracts(unittest.TestCase):
    def test_repeated_references_are_owned_once(self):
        tensor = torch.arange(6)
        source = {'first': tensor, 'again': [tensor]}
        owned = bank(source)
        self.assertIs(owned['first'], owned['again'][0])
        self.assertNotEqual(owned['first'].data_ptr(), tensor.data_ptr())
        tensor.add_(10)
        self.assertEqual(owned['first'].tolist(), list(range(6)))

    def test_dataclass_cpu_metadata_remains_private(self):
        @dataclasses.dataclass
        class Metadata:
            offsets: torch.Tensor
            limit: int
        source = Metadata(torch.tensor([0, 6]), 6)
        owned = bank(source)
        source.offsets.add_(3)
        self.assertEqual(owned.offsets.tolist(), [0, 6])
        self.assertEqual(owned.limit, 6)

    def test_mixed_dtype_offset_views_snapshot_one_entire_backing(self):
        raw = torch.arange(64, dtype=torch.uint8)
        runner = SimpleNamespace(kv_caches=[raw[8:24].view(torch.float32), raw.view(torch.bfloat16)])
        pools = backing_views(runner)
        self.assertEqual(len(pools), 1)
        self.assertEqual(pools[0].numel(), 64)
        saved = [pools[0].clone()]
        raw.zero_()
        restore(pools, saved)
        self.assertTrue(torch.equal(raw, torch.arange(64, dtype=torch.uint8)))

    def test_snapshot_does_not_collapse_separate_allocations(self):
        runner = SimpleNamespace(kv_caches=[torch.zeros(4), torch.zeros(4)])
        self.assertEqual(len(backing_views(runner)), 2)


if __name__ == '__main__':
    unittest.main()
