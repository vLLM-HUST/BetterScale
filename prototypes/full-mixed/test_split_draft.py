"""CPU ownership check for native context ingestion vs replay-only query body."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest


class SplitDraftTests(unittest.TestCase):
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
