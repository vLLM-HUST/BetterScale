"""CPU lifecycle tests; device padding equivalence remains a hardware gate."""
import ast
import copy
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch


def load_class(filename, name, **namespace):
    path = Path(__file__).with_name(filename)
    node = next(x for x in ast.parse(path.read_text()).body
                if isinstance(x, ast.ClassDef) and x.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class NativeQueueFixture:
    def __init__(self):
        self.parallel_config = NS(pipeline_parallel_size=1)
        self.scheduler_config = NS(async_scheduling=True)
        self.requests = {}
        self.sched_step_seq = self.processed_step_seq = 0
        self.deferred_frees = deque()

    def schedule(self, ids):
        self.sched_step_seq += bool(ids)
        return NS(num_scheduled_tokens=dict.fromkeys(ids, 6))

    def update_from_output(self, output, result):
        self.processed_step_seq += bool(output.num_scheduled_tokens)
        return result

    def add_request(self, request):
        self.requests[request.request_id] = request


class N2Contracts(unittest.TestCase):
    def test_draft_mode_normalization_preserves_address_metadata(self):
        path = Path(__file__).with_name('full_draft.py')
        node = next(x for x in ast.parse(path.read_text()).body
                    if isinstance(x, ast.FunctionDef) and x.name == 'graph_metadata')
        namespace = dict(copy=copy)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        metadata = type('AscendDSAMetadata', (), {})()
        metadata.num_prefills = 2; metadata.num_decodes = 1; metadata.num_decode_tokens = 5
        metadata.req_metadata = NS(query_start_loc=torch.tensor([0, 5, 10, 15]))
        result = namespace['graph_metadata']([{'layer':metadata}])[0]['layer']
        self.assertEqual((result.num_prefills,result.num_decodes,result.num_decode_tokens),(1,0,0))
        self.assertEqual((metadata.num_prefills,metadata.num_decodes),(2,1))
        self.assertIs(result.req_metadata,metadata.req_metadata)

    def test_signed_zero_exception_is_confined_to_addressed_bf16(self):
        from draft_oracle import signed_zero_only
        a = torch.tensor([0., 0., 1., 0.], dtype=torch.bfloat16).view(torch.uint8)
        b = torch.tensor([-0., 0., 1., 0.], dtype=torch.bfloat16).view(torch.uint8)
        self.assertEqual(signed_zero_only(a, b, 100, {(100, 2)}), 1)
        self.assertIsNone(signed_zero_only(a, b, 100, {(102, 6)}))
        b[4] ^= 1
        self.assertIsNone(signed_zero_only(a, b, 100, {(100, 8)}))

    def scheduler(self):
        cls = load_class('n2_scheduler.py', 'N2Scheduler',
                         AsyncScheduler=NativeQueueFixture, deque=deque)
        result = cls()
        result.receipt = lambda: None
        return result

    def test_two_waves_and_fifo_including_empty_receipts(self):
        scheduler = self.scheduler()
        first = scheduler.schedule(['a'])
        second = scheduler.schedule([])
        with self.assertRaisesRegex(AssertionError, 'queue exceeded'):
            scheduler.schedule(['a'])
        with self.assertRaisesRegex(AssertionError, 'out of order'):
            scheduler.update_from_output(second, None)
        scheduler.update_from_output(first, None)
        third = scheduler.schedule(['b'])
        scheduler.update_from_output(second, None)
        scheduler.update_from_output(third, None)
        self.assertEqual((scheduler.issued, scheduler.retired, scheduler.max_inflight), (3, 3, 2))
        self.assertEqual(scheduler.processed_step_seq, 2)

    def test_cancelled_identity_cannot_alias_late_output(self):
        scheduler = self.scheduler()
        request = NS(request_id='a', resumable=False)
        scheduler.add_request(request)
        first = scheduler.schedule(['a'])
        second = scheduler.schedule(['a'])
        del scheduler.requests['a']  # Native abort removes request before drain.
        with self.assertRaisesRegex(AssertionError, 'still in flight'):
            scheduler.add_request(request)
        scheduler.update_from_output(first, None)
        with self.assertRaisesRegex(AssertionError, 'still in flight'):
            scheduler.add_request(request)
        scheduler.update_from_output(second, None)
        scheduler.add_request(request)

    def test_native_deferred_free_waits_for_last_device_wave(self):
        path = Path(__file__).resolve().parents[2] / 'upstream/vllm/vllm/v1/core/sched/scheduler.py'
        cls = next(x for x in ast.parse(path.read_text()).body if isinstance(x, ast.ClassDef) and x.name == 'Scheduler')
        methods = [x for x in cls.body if isinstance(x, ast.FunctionDef)
                   and x.name in ('_free_request_blocks', '_drain_deferred_frees')]
        namespace = dict(Request=object)
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
        returned = []
        state = NS(defer_block_free=True, processed_step_seq=0, sched_step_seq=2,
                   deferred_frees=deque(), kv_cache_manager=NS(
                       free=lambda request: returned.append('immediate'),
                       pop_blocks_for_free=lambda request: [1, 2],
                       block_pool=NS(free_blocks=lambda blocks: returned.extend(blocks))))
        namespace['_free_request_blocks'](state, NS(last_sched_seq=2))
        state.processed_step_seq = 1
        namespace['_drain_deferred_frees'](state)
        self.assertEqual(returned, [])
        state.processed_step_seq = 2
        namespace['_drain_deferred_frees'](state)
        self.assertEqual(returned, [2, 1])

    def test_full_draft_padding_and_unpadded_reference_restore(self):
        class Base:
            def __init__(self, worker):
                self.worker = worker; self.drafter = worker.model_runner.drafter
                self.original = self.drafter._runnable
                self.entries = {}; self.enabled = True
        seen = []
        class Entry:
            def __init__(self, worker, original, count, capacity, reference):
                self.original = original; self.reference = reference; self.fallbacks = 0
            def __call__(self, **kwargs):
                self.original(**kwargs)
                return self.reference(**kwargs)
        ctx=NS(capturing=False,attn_metadata={'x':NS(attn_state=NS(name='ChunkedPrefill'),num_prefills=1)})
        cls = load_class('full_draft.py', 'FullDraftGraphSet', DraftGraphSet=Base,
                         ExactDraftGraph=Entry, get_forward_context=lambda: ctx, graph_metadata=lambda x:x,
                         Path=Path, os=NS(environ={'FULL_MIXED_OUTPUT': '/tmp'}))
        d = NS(parallel_drafting=True, _dflash_num_context=7,
               _dflash_hidden_states=torch.ones(32, 2),
               _context_positions_buffer=torch.ones(32),
               _context_slot_mapping_buffers=[torch.ones(32, dtype=torch.int64)])
        d._runnable = lambda **kw: seen.append(d._dflash_num_context)
        worker = NS(rank=0, model_runner=NS(drafter=d,
                    vllm_config=NS(scheduler_config=NS(max_num_batched_tokens=32))))
        manager = cls(worker)
        manager(batch_size=2, inputs_embeds=None, is_prefill=True)
        self.assertEqual(seen, [32, 7])
        self.assertEqual(d._dflash_num_context, 7)
        self.assertIsNone(manager.actual_context)
        self.assertTrue((d._context_slot_mapping_buffers[0][7:] == -1).all())
        self.assertTrue((d._dflash_hidden_states[:7] == 1).all())
        self.assertTrue((d._dflash_hidden_states[7:] == 0).all())


if __name__ == '__main__':
    unittest.main()
