import unittest
from types import SimpleNamespace
import torch
from betterscale.patches.auto_kv._state import zero_kv_backings


class KVBackingClear(unittest.TestCase):
    def test_offset_dtype_page_aliases_and_padding(self):
        a = torch.full((4096,), 73, dtype=torch.uint8)
        b = torch.full((2048,), 29, dtype=torch.uint8)
        untouched = torch.full((4096,), 19, dtype=torch.uint8)
        views = [
            a.view(torch.float32).as_strided((8, 4), (32, 1), 2),
            a.as_strided((8, 16), (128, 1), 48),
        ]
        addresses = [(v.data_ptr(), v.stride(), v.storage_offset()) for v in views]
        context = dict(
            first=SimpleNamespace(kv_cache=views),
            second=SimpleNamespace(kv_cache=(views[1], b)),
            empty=SimpleNamespace(kv_cache=torch.empty(0)),
            non_kv=SimpleNamespace(weight=untouched),
        )
        self.assertEqual(zero_kv_backings(context), 2)
        self.assertTrue(torch.equal(a, torch.zeros_like(a)))
        self.assertTrue(torch.equal(b, torch.zeros_like(b)))
        self.assertTrue(torch.all(untouched == 19))
        self.assertEqual(
            addresses, [(v.data_ptr(), v.stride(), v.storage_offset()) for v in views]
        )


if __name__ == "__main__":
    unittest.main()
