"""CPU ownership check for native context ingestion vs replay-only query body."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
import copy
import dataclasses
from enum import Enum
from unittest.mock import patch
import torch


class SplitDraftTests(unittest.TestCase):
    def metadata_helpers(self):
        path=Path(__file__).with_name('draft_graph.py')
        nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)
               and n.name in ('signature','_signature','bank','_bank','refresh')]
        ns=dict(torch=torch,copy=copy,dataclasses=dataclasses,Enum=Enum,
                RopeDataProxy=type('RopeDataProxy',(),{}))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
        return ns

    def test_metadata_dag_is_banked_and_refreshed_once(self):
        ns=self.metadata_helpers()
        @dataclasses.dataclass
        class Metadata:
            rows: object
        t=torch.arange(4);meta=Metadata(t)
        tree=({'multi':[{'layer0':meta,'layer1':meta}]},{'layer0':meta})
        dst=ns['bank'](tree)
        self.assertIs(dst[0]['multi'][0]['layer0'],dst[1]['layer0'])
        self.assertNotEqual(dst[1]['layer0'].rows.data_ptr(),t.data_ptr())
        self.assertEqual(ns['signature'](tree),ns['signature'](dst))
        t.add_(1);calls=[];original=torch.Tensor.copy_
        def counted(d,s,**kw):calls.append(1);return original(d,s,**kw)
        with patch.object(torch.Tensor,'copy_',counted):ns['refresh'](dst,tree)
        self.assertEqual(len(calls),1)
        torch.testing.assert_close(dst[1]['layer0'].rows,t)
        calls.clear()
        with patch.object(torch.Tensor,'copy_',counted):ns['refresh'](dst,tree,False)
        self.assertEqual(len(calls),3)  # same-graph legacy traversal control
        self.assertEqual(ns['signature'](tree),ns['signature'](tree,False))

    def test_source_alias_split_cannot_overwrite_shared_destination(self):
        ns=self.metadata_helpers();t=torch.arange(4);dst=ns['bank']([t,t])
        with self.assertRaisesRegex(AssertionError,'alias split'):
            ns['refresh'](dst,[t,t.clone()])
        # Distinct view objects of the same exact region are compatible.
        ns['refresh'](dst,[t,t.view_as(t)])

    def test_context_hook_restored_after_success_and_failure(self):
        path = Path(__file__).with_name('split_draft.py')
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'query_body')
        ns = dict(contextmanager=contextmanager)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
        calls=[]
        original=lambda *args: calls.append(args)
        d=SimpleNamespace(build_model_inputs_first_pass=original)
        for fail in (False,True):
            d.build_model_inputs_first_pass(17)
            try:
                with ns['query_body'](d):
                    d.build_model_inputs_first_pass(17)
                    d.build_model_inputs_first_pass(17)
                    if fail: raise ValueError('capture failure')
            except ValueError: pass
            self.assertIs(d.build_model_inputs_first_pass,original)
        self.assertEqual(calls,[(17,),(17,)])
