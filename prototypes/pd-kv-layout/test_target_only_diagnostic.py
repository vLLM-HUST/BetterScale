"""Protect target-only sample feedback and fail-closed proposal suppression."""
import sys,unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import torch
from target_only_diagnostic import install,install_worker

class TargetOnlyTest(unittest.TestCase):
 def test_scheduler_keeps_current_accounting_but_clears_next_proposals(self):
  class Scheduler:
   def _update_after_schedule(self,output):
    self.observed=output.scheduled_spec_decode_tokens
    self.requests['r'].spec_token_ids=[-1,-1]
  install(Scheduler);s=Scheduler();s.requests={'r':NS(spec_token_ids=[])}
  output=NS(num_scheduled_tokens={'r':1},scheduled_spec_decode_tokens={})
  s._update_after_schedule(output)
  self.assertIs(s.observed,output.scheduled_spec_decode_tokens)
  self.assertEqual(s.requests['r'].spec_token_ids,[])
 def runner(self):
  class Runner:pass
  with patch.dict(sys.modules,{'vllm_ascend.worker.model_runner_v1':NS(NPUModelRunner=Runner)}):install_worker()
  r=Runner();r.use_async_spec_decode=True;r.requests={};r.input_batch=object()
  r.discard_request_indices=NS(gpu=object());r.num_discarded_requests=0
  ids=torch.tensor([17]);counts=torch.tensor([1])
  r.drafter=NS(prepare_next_token_ids_padded=Mock(return_value=(ids,counts)),_propose=Mock(side_effect=AssertionError('draft forward ran')))
  r._copy_valid_sampled_token_count=Mock();return r,ids,counts
 def test_feedback_without_draft_forward(self):
  r,ids,counts=self.runner();sample=torch.tensor([[17]])
  self.assertIsNone(r.propose_draft_token_ids(sample,None,NS(scheduled_spec_decode_tokens={})))
  r._copy_valid_sampled_token_count.assert_called_once_with(ids,counts)
  r.drafter._propose.assert_not_called()
 def test_reject_speculative_or_wrong_width(self):
  r,_,_=self.runner()
  for sample,proposal in [(torch.tensor([[17]]),{'r':[18]}),(torch.tensor([[17,18]]),{})]:
   with self.assertRaises(RuntimeError):r.propose_draft_token_ids(sample,None,NS(scheduled_spec_decode_tokens=proposal))
  r._copy_valid_sampled_token_count.assert_not_called()

if __name__=='__main__':unittest.main()
