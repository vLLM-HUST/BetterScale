"""Persistent metadata views preserve routing, sentinels and bank isolation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1]/'prototypes/qwen38-serving/mtp/host_metadata.py'
spec = importlib.util.spec_from_file_location('mtp_host_metadata', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def core_fixture(capacity, width, decode):
    def tensor(shape, dtype=np.int64):
        array = np.zeros(shape, dtype=dtype)
        return NS(numpy=lambda: array)
    return NS(capacity=capacity, width=width, decode=decode,
        cu=tensor(10, np.int32), prefill_conv=tensor((9,1), np.int32),
        verify_conv=tensor((9,1), np.int32), initial=tensor(9, bool),
        accepted=tensor(9, np.int32), prefill_map=tensor(capacity),
        verify_map=tensor(8*width), restore=tensor(capacity), verify_ids=tensor(9),
        prefill=NS(cu=tensor(10), state=tensor((9,2)),
            indices={s:tensor(((capacity+s-1)//s+7,2)) for s in (64,256,1216)}),
        verify=NS(cu=tensor(10, np.int32), slots=tensor((9,width)),
                  accepted=tensor(9,np.int32)))


class HostMetadataTest(unittest.TestCase):
    def test_verify_widths_shrink_and_bank_isolation(self):
        for width in range(2,6):
            a = module.HostMetadata(core_fixture(40,width,True))
            b = module.HostMetadata(core_fixture(40,width,True))
            pointer = a.v.slots.__array_interface__['data'][0]
            for n in (8,1,4,2,8):
                lengths = [1+i%width for i in range(n)]
                slots = np.arange(n*width).reshape(n,width)*3
                a.prepare(lengths,[True]*n,slots,[True]*n)
                self.assertEqual(a.h.cu.tolist(), [0]+list(np.cumsum(lengths))+[sum(lengths)]*(9-n))
                np.testing.assert_array_equal(a.v.cu,a.h.cu)
                np.testing.assert_array_equal(a.v.slots[:n],slots)
                self.assertTrue((a.v.slots[n:]==-1).all())
                self.assertTrue((a.h.verify_conv[n:]==-1).all())
                self.assertEqual(a.h.verify_ids.tolist(),list(range(n))+[0]*(9-n))
                self.assertTrue((a.v.accepted==1).all())
                self.assertTrue((b.v.slots==0).all())
                self.assertEqual(pointer,a.v.slots.__array_interface__['data'][0])

    def test_mixed_role_changes_and_chunk_tails(self):
        h = module.HostMetadata(core_fixture(512,3,False))
        cases = [([3,33,97,257],[True,False,False,False]),
                 ([129,1,3],[False,True,True]),([1],[False]),
                 ([2,64,3,65],[True,False,True,False])]
        for lengths,roles in cases:
            n=len(lengths);slots=np.arange(n*3).reshape(n,3)*7
            initial=[i%2==0 for i in range(n)]
            h.prepare(lengths,roles,slots,initial)
            ends=[0]+list(np.cumsum(lengths)); expected_restore=[0]*512
            for role,cu,mapping,base in [(False,h.p.cu,h.h.prefill_map,0),(True,h.v.cu,h.h.verify_map,512)]:
                rows=[i for i in range(n) if roles[i]==role]
                tokens=[j for i in rows for j in range(ends[i],ends[i+1])]
                self.assertEqual(mapping.tolist(),tokens+[0]*(len(mapping)-len(tokens)))
                for packed,original in enumerate(tokens):expected_restore[original]=base+packed
                lens=[lengths[i] for i in rows]
                self.assertEqual(cu.tolist(),[0]+list(np.cumsum(lens))+[sum(lens)]*(9-len(lens)))
            self.assertEqual(h.h.restore.tolist(),expected_restore)
            for size,dest in h.indices.items():
                rows=[(j,k) for j,i in enumerate(i for i,r in enumerate(roles) if not r) for k in range((lengths[i]+size-1)//size)]
                self.assertEqual(dest.tolist(),[list(x) for x in rows]+[[8,0]]*(len(dest)-len(rows)))
            for i in range(n):
                self.assertEqual(h.h.verify_conv[i,0],slots[i,0] if roles[i] else -1)
                self.assertEqual(h.h.prefill_conv[i,0],slots[i,0] if not roles[i] else -1)
            self.assertTrue((h.h.initial[n:]==0).all())

    def test_invalid_roles_and_widths(self):
        h=module.HostMetadata(core_fixture(24,3,True))
        for lens,roles in [([4],[True]),([1],[False]),([] ,[]),([1]*9,[True]*9)]:
            with self.assertRaises(AssertionError):
                h.prepare(lens,roles,np.zeros((len(lens),3),dtype=int),[False]*len(lens))
