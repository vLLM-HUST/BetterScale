import unittest
import torch
from cache import PrefixLeases
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    FullAttentionSpec,
)


def allocator():
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=8, dtype=torch.float16
    )
    return PrefixLeases(KVCacheConfig(32, [], [KVCacheGroupSpec(["layer"], spec)]), 64)


class CacheTests(unittest.TestCase):
    def test_unretired_old_generation_pins_shared_prefix(self):
        a = allocator()
        x = a.admit("first", list(range(12)), 2, 0, 1)
        self.assertEqual(x.hit_tokens, 0)
        a.retain(x)
        a.retain(x)
        a.commit(x, cursor=12, token=50, terminal=True)
        self.assertFalse(x.freed)
        y = a.admit("second", list(range(12)) + [31], 2, 0, 2)
        self.assertEqual(y.hit_tokens, 12)
        self.assertEqual(y.blocks[:3], x.blocks[:3])
        a.commit(x, cursor=12, terminal=True)
        self.assertTrue(x.freed)
        self.assertFalse(y.freed)
        a.retain(y)
        a.commit(y, cursor=13, token=52, terminal=True)
        self.assertFalse(a.live)
        self.assertGreater(a.deferred_releases, 0)

    def test_allocation_is_not_cache_publication(self):
        a = allocator()
        x = a.admit("first", list(range(12)), 2, 0, 1)
        y = a.admit("second", list(range(12)), 2, 1, 1)
        self.assertEqual(y.hit_tokens, 0)
        self.assertFalse(set(x.blocks) & set(y.blocks))

    def test_whole_prompt_cache_hit_recomputes_last_block(self):
        a = allocator()
        prompt = list(range(12))
        x = a.admit("first", prompt, 1, 0, 1)
        a.retain(x)
        a.commit(x, cursor=12, token=50, terminal=True)
        y = a.admit("second", prompt, 1, 0, 2)
        self.assertEqual(y.hit_tokens, 8)
        with self.assertRaises(ValueError):
            a.retain(x)


if __name__ == "__main__":
    unittest.main()
