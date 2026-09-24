"""D2H order versus mutable batch order, independent of accelerator timing."""
from types import SimpleNamespace as S
import unittest
import numpy as np
import torch
from betterscale.patches.qwen_mtp_feedback import install


class Event:
    def __init__(self):self.pending=None
    def synchronize(self):
        if self.pending is not None:self.pending();self.pending=None


def fixture():
    tensor=torch.ones(4,dtype=torch.int32)
    batch=S(num_accepted_tokens_cpu_tensor=tensor,num_accepted_tokens_cpu=tensor.numpy(), prev_req_id_to_index={"old":0})
    runner=S(input_batch=batch,speculative_config=True,model_config=S(is_hybrid=True),
             use_async_scheduling=True,num_accepted_tokens_event=Event())
    def post(values, *, fail=False):
        target=runner.input_batch.num_accepted_tokens_cpu_tensor
        runner.num_accepted_tokens_event.pending=lambda:target.copy_(torch.tensor(values,dtype=target.dtype))
        if fail:raise RuntimeError('fixture postprocess failure')
        return 'posted'
    def prepare(previous):
        # Execute the pinned Ascend remapping algebra, including new rows.
        runner.num_accepted_tokens_event.synchronize()
        old=runner.input_batch.num_accepted_tokens_cpu
        index=np.array(previous);fresh=index<0
        mapped=old[np.where(fresh,0,index)].copy();mapped[fresh]=1
        old[:len(mapped)]=mapped
        return mapped.tolist()
    runner._update_states_after_model_execute=post;runner._prepare_inputs=prepare
    return runner


class Feedback(unittest.TestCase):
    def test_private_receipt_survives_early_and_late_d2h_and_reorder(self):
        for early in (True,False):
            for previous in ([3,1,-1],[2,0,1],[1],[0,1,2,3]):
                with self.subTest(early=early,previous=previous):
                    r=fixture();destination=r.input_batch.num_accepted_tokens_cpu_tensor
                    install(r);prepare=r._prepare_inputs;install(r)
                    self.assertIs(prepare,r._prepare_inputs)
                    values=[3,2,1,3]
                    self.assertEqual(r._update_states_after_model_execute(values),'posted')
                    self.assertIs(r.input_batch.num_accepted_tokens_cpu_tensor,destination)
                    if early:r.num_accepted_tokens_event.synchronize()
                    # Batch mutation includes condense and new-request neutral1.
                    r.input_batch.num_accepted_tokens_cpu[:]=[3,2,1,1]
                    self.assertEqual(r._prepare_inputs(previous),[1 if i<0 else values[i] for i in previous])

    def test_original_alias_exposes_wrong_owner_after_condense(self):
        r=fixture();r._update_states_after_model_execute([3,2,3,1]);r.num_accepted_tokens_event.synchronize()
        # Previous row2 moves to row0; previous row3 moves to row2.
        r.input_batch.num_accepted_tokens_cpu[:]=[3,2,1,1]
        self.assertEqual(r._prepare_inputs([2,1,3]),[1,2,1]) # should be3,2,1

    def test_apc_reset_count_is_preserved_not_replaced_by_sampler_progress(self):
        r=fixture();install(r)
        r._update_states_after_model_execute([1,3,2,1])
        r.input_batch.num_accepted_tokens_cpu[:]=9
        self.assertEqual(r._prepare_inputs([1,0,2]),[3,1,2])

    def test_fresh_batch_does_not_inherit_finished_request_feedback(self):
        r=fixture();install(r)
        r._update_states_after_model_execute([3,2,3,1])
        r.input_batch.prev_req_id_to_index={}
        r.input_batch.num_accepted_tokens_cpu[:]=1
        self.assertEqual(r._prepare_inputs([0]),[1])

    def test_exception_restores_native_batch_tensor(self):
        r=fixture();destination=r.input_batch.num_accepted_tokens_cpu_tensor;install(r)
        with self.assertRaisesRegex(RuntimeError,'fixture'):r._update_states_after_model_execute([1]*4,fail=True)
        self.assertIs(r.input_batch.num_accepted_tokens_cpu_tensor,destination)

    def test_non_speculative_runner_unchanged(self):
        r=fixture();r.speculative_config=None;original=r._prepare_inputs;install(r)
        self.assertIs(r._prepare_inputs,original)
