import unittest
import torch
from decode_shadow import geometry


class GeometryTests(unittest.TestCase):
    def test_k5_owner_rows_and_sampler_indices(self):
        for n in (1,2,4,16):
            g=geometry(n)
            self.assertEqual(g['qsl'].tolist(),list(range(0,6*n+1,6)))
            self.assertEqual(g['rows'].tolist(),[i for i in range(n) for _ in range(6)])
            expected=[6*i+j for i in range(n) for j in range(1,6)]
            self.assertEqual((g['target']+1).tolist(),expected)
            self.assertEqual(g['draft_index'].tolist(),expected)
            self.assertEqual(g['bonus'].tolist(),[6*i+5 for i in range(n)])
            self.assertEqual(g['samples'].dtype,torch.int32)
            self.assertEqual(g['target'].dtype,torch.int32)
            self.assertEqual(g['logits'].dtype,torch.int64)
