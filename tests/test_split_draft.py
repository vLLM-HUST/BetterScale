"""CPU ownership check for native context ingestion vs replay-only query body."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest


class SplitDraftTests(unittest.TestCase):
    def test_context_hook_restored_after_success_and_failure(self):
        path = (Path(__file__).resolve().parents[1]/'src/strengthen_dsv4/patches').joinpath('split_draft/__init__.py')
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

    def test_one_manager_owns_decode_and_query_routes(self):
        from contextlib import nullcontext
        path = Path(__file__).resolve().parents[1]/'src/strengthen_dsv4/patches/split_draft/__init__.py'
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name in ('query_body', 'SplitDraftGraphSet')]
        calls=[]
        original=lambda **kw: 'native'
        ingest=lambda *args: calls.append('context')
        d=SimpleNamespace(_runnable=original,parallel_drafting=True,num_speculative_tokens=5,
            _dflash_num_context=6,_context_slot_mapping_buffers=[],build_model_inputs_first_pass=ingest)
        metadata={'a':SimpleNamespace(num_prefills=0,attn_state=SimpleNamespace(name='mode'))}
        ctx=SimpleNamespace(capturing=False,attn_metadata=metadata)
        worker=SimpleNamespace(model_runner=SimpleNamespace(drafter=d,
            vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=4))))

        class Entry:
            def __init__(self,worker,original,count):
                self.original=original;self.count=count;self.fallbacks=0;self.fail=False
            def __call__(self,**kwargs):
                query=getattr(self,'query_only',False)
                calls.append('query' if query else 'decode')
                if query:
                    # This is the native merged-call hook: it must be suppressed
                    # only for the query body, since context was already ingested.
                    d.build_model_inputs_first_pass(1,[])
                    if self.fail:raise ValueError('query failure')
                return 'result'

        ns=dict(contextmanager=contextmanager,record_function=nullcontext,ExactDraftGraph=Entry,
            get_forward_context=lambda:ctx,graph_metadata=lambda value:dict(value))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
        manager=ns['SplitDraftGraphSet'](worker)
        d._runnable=manager
        self.assertEqual(manager(batch_size=1),'result')
        self.assertEqual(manager(batch_size=1),'result')
        self.assertEqual(calls,['decode','decode'])
        self.assertEqual(len(manager.decode_graphs),1)
        self.assertIs(manager.decode_graphs[1].original,original)

        d._dflash_num_context=17
        metadata['a'].num_prefills=1
        kwargs=dict(batch_size=1,num_input_tokens=17,inputs_embeds=None)
        self.assertEqual(manager(**kwargs),'result')
        self.assertEqual(calls[-2:],['context','query'])
        entry=next(iter(manager.query_graphs.values()))
        self.assertTrue(entry.strict_signature)
        self.assertIsNot(entry,manager.decode_graphs[1])
        self.assertIs(ctx.attn_metadata,metadata)
        self.assertIs(d.build_model_inputs_first_pass,ingest)

        entry.fail=True
        with self.assertRaisesRegex(ValueError,'query failure'):manager(**kwargs)
        self.assertEqual(calls[-2:],['context','query'])
        self.assertIs(ctx.attn_metadata,metadata)
        self.assertIs(d.build_model_inputs_first_pass,ingest)
        manager.enabled=False
        self.assertEqual(manager(**kwargs),'native')
        self.assertEqual(manager.context_calls,2)
