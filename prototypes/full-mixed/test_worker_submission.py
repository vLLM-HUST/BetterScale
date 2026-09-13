"""Generation-local receipt and submit-before-retire contracts, no NPU."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from worker_submission import WorkerSubmission


class Contract(unittest.TestCase):
    def make(self):
        log = []
        class Event:
            def record(self): log.append('record')
            def synchronize(self): log.append('receipt-ready')
        r = NS(use_async_scheduling=True, use_async_spec_decode=True,
               need_accepted_tokens=False, model_config=NS(is_hybrid=False),
               vllm_config=NS(parallel_config=NS(pipeline_parallel_size=1,tensor_parallel_size=2)),
               _decode_shadow=object(), num_prompt_logprobs=0, execute_model_state=object(),
               input_batch=NS(sampling_metadata=NS(all_greedy=True,output_token_ids=[]),
                              req_ids=['a','b'],prev_req_id_to_index={'a':0,'b':1},
                              prev_sampled_token_ids=torch.zeros(2,1)),
               valid_sampled_token_count_cpu=torch.tensor([2,6]),
               valid_sampled_token_count_event=Event())
        r._get_valid_sampled_token_count = lambda: r.valid_sampled_token_count_cpu.tolist()
        r.execute_model = lambda *a: log.append('target')
        def copy_counts(ids, counts):
            r.valid_sampled_token_count_cpu.copy_(counts)
            r.valid_sampled_token_count_event.record()
        r._copy_valid_sampled_token_count = copy_counts
        def sample(*args):
            log.append('sample-draft')
            r._copy_valid_sampled_token_count(None, torch.tensor([6,1]))
            return 'output-future'
        r.sample_tokens = sample
        worker = NS(model_runner=r,_decode_metadata=object(),_exact_draft_graph=object())
        allocate = torch.empty_like
        def cpu_empty(t, **kw): return allocate(t)
        patches = (patch.object(torch,'npu',NS(Event=Event),create=True),
                   patch.object(torch,'empty_like',cpu_empty))
        for p in patches: p.start(); self.addCleanup(p.stop)
        state = WorkerSubmission(worker)
        return r,state,log

    def test_generation_local_counts_retire_after_full_submit(self):
        r,s,log = self.make()
        r.execute_model(None)
        observed=[]
        original_getter=r._get_valid_sampled_token_count
        self.assertTrue(s.defer(lambda:observed.extend(r._get_valid_sampled_token_count())))
        self.assertEqual(observed,[])
        self.assertEqual(r.sample_tokens(), 'output-future')
        self.assertEqual(observed,[2,6])  # not the newly published [6,1]
        self.assertEqual(r.valid_sampled_token_count_cpu.tolist(),[6,1])
        self.assertIs(r._get_valid_sampled_token_count,original_getter)
        self.assertEqual(log,['target','sample-draft','record','record','receipt-ready'])
        self.assertEqual(s.receipt()['retired'],1)
        self.assertFalse(s.receipt()['numerical_state_banked'])
        # Next generation uses the other count bank without clobbering its old receipt.
        r.execute_model(None)
        self.assertTrue(s.defer(lambda:observed.extend(r._get_valid_sampled_token_count())))
        r.sample_tokens()
        self.assertEqual(observed,[2,6,6,1])

    def test_no_second_command_before_sample(self):
        r,s,_ = self.make();r.execute_model(None)
        with self.assertRaises(AssertionError):r.execute_model(None)

    def test_fail_closed_on_sample_error(self):
        r,s,log=self.make();r.execute_model(None)
        s.defer(lambda:log.append('retired'))
        def fail():raise RuntimeError('sample failure')
        s.sample=fail
        with self.assertRaisesRegex(RuntimeError,'sample failure'):r.sample_tokens()
        self.assertTrue(s.failed);self.assertNotIn('retired',log)
        with self.assertRaises(AssertionError):r.execute_model(None)

    def test_special_sampling_keeps_native_retirement(self):
        r,s,_=self.make();r.execute_model(None)
        r.input_batch.sampling_metadata.output_token_ids=[[1]]
        self.assertFalse(s.defer(lambda:None))
        self.assertIsNone(s.pending)

    def test_empty_execute_does_not_need_sample(self):
        r,s,_=self.make();r.execute_model_state=None
        r.execute_model(None);r.execute_model(None)
        self.assertIsNone(s.frame);self.assertEqual(s.receipt()['submitted'],0)


if __name__ == '__main__':unittest.main()
