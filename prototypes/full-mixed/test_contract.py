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

if __name__=='__main__':unittest.main()
