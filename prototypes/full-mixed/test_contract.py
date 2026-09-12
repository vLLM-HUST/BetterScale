"""CPU-only regression for the exact padding function exercised on device."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np

source=Path(__file__).with_name('extension.py')
function=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='_pad_dsa_capacity')
namespace={'CUDAGraphMode':NS(FULL='FULL'),'_original_pad':lambda *a: 'fallback'}
exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
pad=namespace['_pad_dsa_capacity']

class PaddingContract(unittest.TestCase):
    def test_fixed_capacity_after_shrink(self):
        q=NS(np=np.array([0,64,128,192,256],dtype=np.int32),copy_to_gpu=lambda:None)
        self.assertEqual(pad(None,q,256,2,2,'FULL',4),4)
        np.testing.assert_array_equal(q.np,[0,64,128,128,128])
    def test_token_padding_does_not_invent_request(self):
        q=NS(np=np.array([0,7,9,19,31],dtype=np.int32),copy_to_gpu=lambda:None)
        self.assertEqual(pad(None,q,256,4,4,'FULL',4),4)
        self.assertEqual(q.np[-1],31)
    def test_reject_capacity_overflow(self):
        with self.assertRaises(AssertionError):pad(None,None,256,5,5,'FULL',4)
    def test_preserve_non_full_route(self):
        self.assertEqual(pad(None,None,256,4,4,'NONE',4),'fallback')

class AlignmentContract(unittest.TestCase):
    def test_tp8_k5_lcm_without_changing_speculation(self):
        f=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='_adjust_joint_alignment')
        ns={'_original_adjust_sizes':lambda self,alignment,tp:(alignment,tp)}
        exec(compile(ast.Module(body=[f],type_ignores=[]),str(source),'exec'),ns)
        config=NS(pass_config=NS(enable_sp=True))
        self.assertEqual(ns['_adjust_joint_alignment'](config,6,8),(24,8))
        self.assertEqual(ns['_adjust_joint_alignment'](config,6,2),(6,2))
        config.pass_config.enable_sp=False
        self.assertEqual(ns['_adjust_joint_alignment'](config,6,8),(6,8))

class SharedPoolContract(unittest.TestCase):
    def test_aliases_snapshot_once_without_dtype_interpretation(self):
        import torch
        f=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='_unique_byte_pools')
        ns={'torch':torch}
        exec(compile(ast.Module(body=[f],type_ignores=[]),str(source),'exec'),ns)
        raw=torch.arange(128,dtype=torch.uint8)
        pools=ns['_unique_byte_pools']([raw.view(torch.bfloat16),raw.view(torch.float32)[4:]])
        self.assertEqual(len(pools),1)
        self.assertTrue(torch.equal(pools[0],raw))
        saved=pools[0].clone();raw.zero_();pools[0].copy_(saved)
        self.assertTrue(torch.equal(raw,torch.arange(128,dtype=torch.uint8)))

class ShadowContract(unittest.TestCase):
    def test_bounded_comparison_rejects_changed_element(self):
        import torch
        f=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='_compare_bounded')
        ns={'torch':torch}
        exec(compile(ast.Module(body=[f],type_ignores=[]),str(source),'exec'),ns)
        compare=ns['_compare_bounded']
        a=torch.ones((5,7),dtype=torch.bfloat16)
        self.assertEqual(compare(a,a)['max_diff'],0)
        b=a.clone();b[-1,-1]=2
        with self.assertRaises(AssertionError):compare(a,b)
        empty=a[:0]
        self.assertEqual(compare(empty,empty)['max_diff'],0)

if __name__=='__main__':unittest.main()
