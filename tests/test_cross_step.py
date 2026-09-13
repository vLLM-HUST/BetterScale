"""CPU contracts for conservative bounds and exact-state shadow reference."""
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from strengthen_dsv4.patches.cross_step import CrossStepBounds, stable_verification


def fixture():
    r=NS(input_batch=NS(req_ids=['a','b'],prev_req_id_to_index={'a':0,'b':1},
                       num_computed_tokens_cpu=[80,90],num_prompt_tokens=[64,64],num_reqs=2),
         valid_sampled_token_count_gpu=torch.tensor([1,6]),_draft_token_ids=torch.zeros(2,5),
         use_compress=True,use_async_spec_decode=True,use_dcp=False,num_spec_tokens=5,
         need_accepted_tokens=False,supports_mm_inputs=False,enable_prompt_embeds=False,
         vllm_config=NS(model_config=NS(hf_config=NS(model_type='deepseek_v4'))),
         max_model_len=2048,optimistic_seq_lens_cpu=torch.tensor([86,96]),seq_lens=torch.tensor([81,96]))
    Builder=type("AscendDSACPMetadataBuilder",(),{})
    r.attn_groups=[[NS(get_metadata_builder=lambda:Builder())]]
    r._update_states=lambda schedule:None
    r._model_forward=lambda **kwargs:None
    r.prepare_inputs_event=NS(synchronize=lambda:None)
    r._prepare_inputs=lambda *args: 'prepared'
    def correct(n):r.optimistic_seq_lens_cpu[:n].copy_(r.seq_lens[:n])
    r._correct_optimistic_seq_lens_cpu=correct
    r._build_attention_metadata=lambda *args,**kwargs:({'max':r.optimistic_seq_lens_cpu.max().item()},None)
    s=NS(scheduled_new_reqs=[],num_scheduled_tokens={'a':6,'b':6},scheduled_spec_decode_tokens={'a':[0]*5,'b':[0]*5})
    return r,s


class Contract(unittest.TestCase):
    def make(self,r):
        with patch('pathlib.Path.write_text',side_effect=AssertionError('No production receipts')):
            return CrossStepBounds(NS(model_runner=r,rank=0))

    def test_stable_only(self):
        r,s=fixture();self.assertTrue(stable_verification(r,s,[6,6]))
        self.assertFalse(stable_verification(r,s,[6,64]))
        r.input_batch.prev_req_id_to_index={'a':1,'b':0}
        self.assertFalse(stable_verification(r,s,[6,6]))
        r.input_batch.prev_req_id_to_index={'a':0,'b':1};s.scheduled_new_reqs=['new']
        self.assertFalse(stable_verification(r,s,[6,6]))

    def test_preserve_bound_and_restore_shadow(self):
        r,s=fixture();state=self.make(r)
        state.prepare_inputs(s,[6,6]);state.correct_bounds(2)
        self.assertEqual(r.optimistic_seq_lens_cpu.tolist(),[86,96])
        self.assertEqual(state.bypassed,1)
        state.build_metadata();ctx=NS(attn_metadata={})
        upper=state.reference_begin(ctx)
        self.assertEqual(r.optimistic_seq_lens_cpu.tolist(),[81,96])
        state.reference_end(ctx,upper)
        self.assertEqual(r.optimistic_seq_lens_cpu.tolist(),[86,96])
        self.assertEqual(r.seq_lens.tolist(),[81,96])

    def test_fallback_and_upper_bound_violation(self):
        r,s=fixture();state=self.make(r);state.prepare_inputs(s,[6,64]);state.correct_bounds(2)
        self.assertEqual(r.optimistic_seq_lens_cpu.tolist(),[81,96]);self.assertFalse(state.skipped)
        state.prepare_inputs(s,[6,6]);r.optimistic_seq_lens_cpu[0]=80;state.correct_bounds(2);state.build_metadata()
        with self.assertRaisesRegex(AssertionError,'underestimates'):
            state.reference_begin(NS(attn_metadata={}))

    def test_bookkeeping_after_forward_and_dma_retirement(self):
        r,s=fixture();events=[]
        r._update_states=lambda schedule:lambda:events.append('receipt')
        r._model_forward=lambda **kwargs:events.append('forward')
        r.prepare_inputs_event=NS(synchronize=lambda:events.append('dma_retired'))
        state=self.make(r);defer=r._update_states(s);defer()
        self.assertEqual(events,[])
        r._model_forward()
        self.assertEqual(events,['forward','dma_retired','receipt'])
        self.assertIsNone(state.pending)
        self.assertEqual(state.late_commits,1)

    def test_dcp_not_admitted(self):
        r,s=fixture();r.use_dcp=True
        with self.assertRaises(AssertionError):self.make(r)

    def test_unqualified_all_modes_rejected(self):
        r,s=fixture()
        with self.assertRaisesRegex(ValueError,'not part of the kept'):
            CrossStepBounds(NS(model_runner=r,rank=0),all_modes=True)
