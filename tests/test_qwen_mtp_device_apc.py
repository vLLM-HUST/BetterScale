"""Device APC identity remapping and raw-progress/selection separation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch

path = Path(__file__).resolve().parents[1]/'prototypes/qwen38-serving/mtp/device_apc.py'
spec = importlib.util.spec_from_file_location('mtp_device_apc', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Buffer:
    def __init__(self):
        self.gpu = torch.zeros(8, dtype=torch.int32)
        self.cpu = torch.zeros_like(self.gpu)
        self.np = self.cpu.numpy()
    def copy_to_gpu(self, n):
        self.gpu[:n].copy_(self.cpu[:n])


def fixture():
    ctx = NS(is_initialized=True, block_size=16,
             num_accepted_tokens_out=torch.ones(8,dtype=torch.int32))
    for name in ('mamba_state_idx','num_scheduled_tokens','num_computed_tokens','num_draft_tokens'):
        setattr(ctx,name+'_buf',Buffer())
    def copy(n,accepted,source,scheduled,computed,drafts):
        ctx.call = [x.clone() for x in (accepted,source,scheduled,computed,drafts)]
        ctx.num_accepted_tokens_out[:n].copy_(accepted[:n])
    ctx.run_fused_postprocess = copy
    runner = NS(_get_mamba_bufs=lambda:NS(postprocess_align=ctx),
        cache_config=NS(mamba_cache_mode='align'),scheduler_config=NS(max_num_seqs=8),
        input_batch=NS(num_reqs=0,req_ids=[]), device='cpu',
        num_computed_tokens=torch.zeros(8,dtype=torch.int32),
        num_scheduled_tokens=Buffer(),num_accepted_tokens=Buffer())
    return runner,ctx


def step(r, ids, computed, queries, **kw):
    s=NS(finished_req_ids=set(kw.get('finished',())),preempted_req_ids=set(kw.get('preempted',())),
        scheduled_cached_reqs=NS(resumed_req_ids=set(kw.get('resumed',()))),
        scheduled_spec_decode_tokens={req:[0]*(q-1) for req,q in zip(ids,queries)})
    r.input_batch.req_ids=ids;r.input_batch.num_reqs=len(ids)
    r.num_computed_tokens[:len(ids)]=torch.tensor(computed,dtype=torch.int32)
    r.num_scheduled_tokens.gpu[:len(ids)]=torch.tensor(queries,dtype=torch.int32)
    module.prepare(r,s)
    return s


class DeviceAPCTest(unittest.TestCase):
    def test_remap_pause_migration_and_progress(self):
        r,c=fixture()
        step(r,['a','b'],[13,20],[3,3])
        state=r._mtp_apc_state
        state['selection'][:2]=torch.tensor([3,2])
        # b changes order, a pauses for a wave but its candidate choice survives.
        step(r,['b'],[22],[3])
        self.assertEqual(c.call[0].tolist(),[2])
        self.assertEqual(c.call[1].tolist(),[1])
        step(r,['b','a'],[24,16],[3,3])
        self.assertEqual(c.call[0].tolist(),[2,3])
        self.assertEqual(c.call[1].tolist(),[1,0])
        self.assertEqual(r.num_accepted_tokens.gpu[:2].tolist(),[2,1])
        self.assertEqual(c.num_computed_tokens_buf.gpu[:2].tolist(),[24,16])
        self.assertEqual(c.mamba_state_idx_buf.gpu[:2].tolist(),[1,1])
        # Artificial boundary applies only to copy, not actual logical progress.
        self.assertEqual(c.call[3].tolist(),[0,29])

    def test_resume_and_empty_wave_cleanup(self):
        r,c=fixture()
        step(r,['a','b'],[35,20],[3,3])
        r._mtp_apc_state['selection'][:2]=torch.tensor([3,2])
        step(r,['a'],[16],[3],resumed=['a'])
        self.assertEqual(c.call[0].tolist(),[1])
        self.assertEqual(c.call[1].tolist(),[0])
        empty=NS(finished_req_ids={'a','b'},preempted_req_ids=None,
                 scheduled_cached_reqs=NS(resumed_req_ids=set()))
        module.wait_for_previous(r,empty)
        self.assertEqual(r._mtp_apc_state['requests'],{})
        step(r,['a'],[0],[3])
        self.assertEqual(c.call[1].tolist(),[-1])
        self.assertEqual(r.num_accepted_tokens.gpu[0].item(),1)

    def test_postprocess_preserves_raw_count_and_saves_reset_selection(self):
        r,c=fixture()
        schedule=step(r,['a','b'],[13,20],[3,3])
        def post(n,*args):
            c.num_accepted_tokens_out[:n].copy_(torch.tensor([1,2],dtype=torch.int32))
        c.run_fused_postprocess=post
        records=[]
        npu=NS(Event=lambda:NS(record=lambda:records.append('done')))
        with patch.object(torch,'npu',npu,create=True):
            module.postprocess(r,torch.tensor([[7,8,9],[10,11,-1]]),schedule)
        self.assertEqual(r.num_accepted_tokens.gpu[:2].tolist(),[3,2])
        self.assertEqual(r._mtp_apc_state['selection'][:2].tolist(),[1,2])
        self.assertEqual(records,['done'])

    def test_cross_stream_dependency_is_device_wait(self):
        r,_=fixture();event=object();r._mtp_apc_done=event
        calls=[]
        schedule=NS(finished_req_ids=set(),preempted_req_ids=None,
                    scheduled_cached_reqs=NS(resumed_req_ids=set()))
        npu=NS(current_stream=lambda:NS(wait_event=calls.append))
        with patch.object(torch,'npu',npu,create=True):
            module.wait_for_previous(r,schedule)
        self.assertEqual(calls,[event])
