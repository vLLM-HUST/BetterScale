"""CPU contracts of the exact DSA normalization hooks (no NPU imports)."""
import ast
from contextvars import ContextVar
from copy import copy
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np

source = Path(__file__).with_name('dp_full.py')
names = {'split_queries', 'build', 'pad_queries'}
body = [x for x in ast.parse(source.read_text()).body if isinstance(x, ast.FunctionDef) and x.name in names]


class DPFullContract(unittest.TestCase):
    def setUp(self):
        self.ns = dict(copy=copy, native_reference=ContextVar('native', default=False),
                       unified_build=ContextVar('unified', default=False),
                       CUDAGraphMode=NS(FULL='FULL'), original_pad=lambda *a: 'native-pad',
                       original_split=lambda *a, **k: 'native-split')
        exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), self.ns)

    def test_normalization_preserves_real_offsets_and_request_clearing(self):
        common = NS(num_actual_tokens=129, num_input_tokens=264, num_reqs=2,
                    query_start_loc=[0, 129, 129])
        worker = NS(decode_threshold=6, vllm_config=NS(parallel_config=NS(tensor_parallel_size=1),
                                                     scheduler_config=NS(max_num_seqs=100)))
        seen = []
        def native(worker, prefix, cm, fast, **kw):
            seen.append((cm, kw, self.ns['split_queries'](cm)))
            return NS(decode=NS(num_reqs_actual=kw['num_reqs_actual']))
        self.ns['original_build'] = native
        result = self.ns['build'](worker, 0, common, num_reqs_actual=1)
        self.assertEqual(seen[0][0].num_actual_tokens, 264)
        self.assertEqual(seen[0][1]['num_reqs_actual'], 1)
        self.assertEqual(seen[0][2], (2, 0, 264, 0))
        self.assertEqual(common.num_actual_tokens, 129)
        self.assertIs(seen[0][0].query_start_loc, common.query_start_loc)
        self.assertEqual(result.decode.num_reqs_actual, 2)
        self.assertFalse(self.ns['unified_build'].get())
        self.assertEqual(self.ns['split_queries'](common), 'native-split')

    def test_native_reference_bypasses_normalization(self):
        self.ns['original_build'] = lambda *a, **k: a[2]
        cm = NS(num_actual_tokens=129)
        t = self.ns['native_reference'].set(True)
        try:
            self.assertIs(self.ns['build'](None, 0, cm), cm)
        finally:
            self.ns['native_reference'].reset(t)

    def test_zero_length_seats_not_fictitious_prefill(self):
        cm = NS(np=np.array([0, 129, 240, 999], dtype=np.int32), copy_to_gpu=lambda: None)
        self.assertEqual(self.ns['pad_queries'](None, cm, 264, 1, 1, 'FULL', 2), 2)
        np.testing.assert_array_equal(cm.np, [0, 129, 129, 999])
        self.assertEqual(self.ns['pad_queries'](None, cm, 264, 1, 1, 'NONE', 2), 'native-pad')

    def test_build_failure_restores_context(self):
        worker = NS(decode_threshold=6, vllm_config=NS(parallel_config=NS(tensor_parallel_size=1),
                                                     scheduler_config=NS(max_num_seqs=100)))
        def fail(*a, **k): raise RuntimeError('fixture')
        self.ns['original_build'] = fail
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            self.ns['build'](worker, 0, NS(num_input_tokens=12))
        self.assertFalse(self.ns['unified_build'].get())


class DPShadowMetadataContract(unittest.TestCase):
    def test_prepared_dummy_inputs_are_not_rebuilt(self):
        path=source.with_name('dp_full_shadow.py')
        node=next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n,ast.FunctionDef) and n.name=='reference_metadata')
        scope={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
        calls=[]; flag=ContextVar('reference',default=False)
        def rebuild(runner, **kwargs):
            calls.append((flag.get(),kwargs))
            if kwargs.get('fail'): raise ValueError('fixture')
            return ({'native':True},None)
        with patch.dict('sys.modules',dp_full=NS(native_reference=flag,original_metadata=rebuild)):
            prepared={'slot_mapping':'pre-clear-copy'}
            f=scope['reference_metadata']
            self.assertIs(f('unified',None,(),{},prepared),prepared)
            self.assertEqual(calls,[])
            self.assertEqual(f('native',None,(),{'tokens':12},prepared),{'native':True})
            self.assertEqual(calls,[('native',{'tokens':12})])
            self.assertFalse(flag.get())
            with self.assertRaises(ValueError): f('padded',None,(),{'fail':True},prepared)
            self.assertFalse(flag.get())


if __name__ == '__main__': unittest.main()
