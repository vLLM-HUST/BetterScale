"""Explicit stable-K5 host construction / banked ingress / device derivation.

Adapted from the pinned donor preparation arithmetic and LiveInfer's invocation
lifetime protocol. No generic FakeTensor replay is used in the serving path.
Prefill, turnover, hybrid feedback and other speculative widths stay native.
"""
import json
import os
from pathlib import Path
import numpy as np
import torch
from cross_step import stable_verification


def geometry(n):
    drafts=np.full(n,5,dtype=np.int32)
    samples=np.cumsum(drafts+1,dtype=np.int32)
    cumulative=np.cumsum(drafts,dtype=np.int32)
    target=np.repeat(samples-drafts-1,drafts)+(np.arange(n*5)-np.repeat(cumulative-drafts,drafts))
    return dict(qsl=torch.arange(n+1,dtype=torch.int32)*6,
                rows=torch.arange(n).repeat_interleave(6),query=torch.arange(6).repeat(n),
                scheduled=torch.full((n,),6,dtype=torch.int32),previous=torch.arange(n),
                logits=torch.arange(6*n),target=torch.from_numpy(target.astype(np.int32)),
                bonus=torch.from_numpy(samples-1),samples=torch.from_numpy(samples),
                drafts=torch.from_numpy(cumulative),sample_index=torch.arange(n)*6,
                draft_index=(torch.arange(n)[:,None]*6+torch.arange(1,6)).flatten())


class Slot:
    def __init__(self,r,n,ingress):
        self.n=n;self.r=r;self.ingress=ingress
        self.host={};self.device={}
        def field(name,example):
            self.host[name]=torch.empty_like(example,device='cpu',pin_memory=True)
            self.device[name]=torch.empty_like(example,device=r.device)
        field('budget',r.input_batch.num_computed_tokens_cpu_tensor[:n])
        field('previous_drafts',r.prev_num_draft_tokens.cpu)
        field('accepted',r.num_accepted_tokens.cpu)
        for i,table in enumerate(r.input_batch.block_table.block_tables):field(f'blocks{i}',table.block_table.cpu[:n])
        self.constants={k:v.to(r.device) for k,v in geometry(n).items()}
        self.valid=torch.empty_like(r.valid_sampled_token_count_gpu[:n])
        self.sampled=torch.empty_like(r.input_batch.prev_sampled_token_ids[:n,0])
        self.drafted=torch.empty_like(r._draft_token_ids[:n,:5])
        self.ready=None;self.consumed=None;self.graph=None;self.output=None

    def project(self):
        r=self.r;n=self.n
        # CPU sources and device destinations have different reuse gates.
        if self.ready is not None:self.ready.synchronize()
        self.host['budget'].copy_(r.input_batch.num_computed_tokens_cpu_tensor[:n])
        self.host['previous_drafts'].copy_(r.prev_num_draft_tokens.cpu)
        self.host['accepted'].copy_(r.num_accepted_tokens.cpu)
        for i,table in enumerate(r.input_batch.block_table.block_tables):
            self.host[f'blocks{i}'].copy_(table.block_table.cpu[:n])
        with torch.npu.stream(self.ingress):
            if self.consumed is not None:self.ingress.wait_event(self.consumed)
            for name,value in self.host.items():self.device[name].copy_(value,non_blocking=True)
            self.ready=torch.npu.Event();self.ready.record()

    def derive(self):
        from vllm_ascend.spec_decode.utils import update_num_computed_tokens_for_batch_change
        from vllm_ascend.worker.model_runner_v1 import SpecDecodeMetadata,lmhead_tp_enable
        r=self.r;n=self.n;t=n*6;c=self.constants;d=self.device
        for i,table in enumerate(r.input_batch.block_table.block_tables):
            table.block_table.gpu[:n].copy_(d[f'blocks{i}'])
        r.prev_positions.gpu[:n].copy_(c['previous'])
        r.prev_num_draft_tokens.gpu.copy_(d['previous_drafts'])
        r.num_accepted_tokens.gpu.copy_(d['accepted'])
        r.query_start_loc.gpu[:n+1].copy_(c['qsl'])
        r.query_start_loc.gpu[n+1:].fill_(-1)
        r.input_ids.gpu.scatter_(0,c['sample_index'],self.sampled)
        r.input_ids.gpu.scatter_(0,c['draft_index'],self.drafted.to(torch.int32).flatten())
        update_num_computed_tokens_for_batch_change(r.num_computed_tokens,r.num_accepted_tokens.gpu[:n],
            r.prev_positions.gpu[:n],self.valid,r.prev_num_draft_tokens.gpu,d['budget'])
        r.req_indices.gpu[:t].copy_(c['rows']);r.query_pos.gpu[:t].copy_(c['query'])
        r.num_scheduled_tokens.gpu[:n].copy_(c['scheduled'])
        r.positions[:t].copy_(r.num_computed_tokens[c['rows']].to(torch.int64)+c['query'])
        r.seq_lens[:n].copy_(r.num_computed_tokens[:n]+c['scheduled']);r.seq_lens[n:].zero_()
        # This is a real device action; it is NEVER called from host projection.
        r.input_batch.block_table.compute_slot_mapping(n,r.query_start_loc.gpu[:n+1],r.positions[:t])
        r.discard_request_mask.gpu[:n].zero_()
        r.num_decode_draft_tokens.gpu[:n].fill_(5);r.num_decode_draft_tokens.gpu[n:].fill_(-1)
        draft_ids=r.input_ids.gpu[c['logits']][c['target']+1]
        metadata=SpecDecodeMetadata(draft_token_ids=draft_ids,num_draft_tokens=[5]*n,
            cu_num_draft_tokens=c['drafts'],cu_num_sampled_tokens=c['samples'],target_logits_indices=c['target'],
            bonus_logits_indices=c['bonus'],logits_indices=c['logits'])
        logits=c['logits']
        if lmhead_tp_enable():logits=torch.nn.functional.pad(logits,(0,r.max_num_reqs*6-t))
        return logits,metadata,t

    def replay(self):
        r=self.r;n=self.n;compute=torch.npu.current_stream()
        compute.wait_event(self.ready)
        # Native feedback is numerical State, not host-predicted metadata.
        self.valid.copy_(r.valid_sampled_token_count_gpu[:n])
        self.sampled.copy_(r.input_batch.prev_sampled_token_ids[:n,0])
        self.drafted.copy_(r._draft_token_ids[:n,:5])
        if self.graph is None:
            torch.npu.synchronize() # one-time shape admission, never steady replay
            before=r.num_computed_tokens.clone()
            self.graph=torch.npu.NPUGraph()
            with torch.npu.graph(self.graph):self.output=self.derive()
            r.num_computed_tokens.copy_(before)
        self.graph.replay()
        self.consumed=torch.npu.Event();self.consumed.record()
        r.logits_indices=self.constants['logits']
        return self.output


def device_fields(r,n):
    t=n*6
    result=[r.num_computed_tokens,r.input_ids.gpu[:t],r.positions[:t],r.seq_lens,
        r.num_accepted_tokens.gpu,r.query_start_loc.gpu,r.req_indices.gpu[:t],r.query_pos.gpu[:t],
        r.num_scheduled_tokens.gpu[:n],r.num_decode_draft_tokens.gpu,r.discard_request_mask.gpu[:n]]
    for table in r.input_batch.block_table.block_tables:result += [table.block_table.gpu[:n],table.slot_mapping.gpu]
    return result


class DecodeShadow:
    def __init__(self,worker,verify=False):
        r=self.r=worker.model_runner
        assert not r.model_config.is_hybrid and not r.need_accepted_tokens
        assert r.use_async_spec_decode and not r.use_dcp and not r.lora_config
        self.native=r._cross_step_bounds.prepare
        self.ingress=torch.npu.Stream(device=r.device)
        r._decode_shadow=self
        self.enabled=True;self.slots={};self.active=None;self.sequence=0;self.checks=[];self.verify=verify
        p=r.vllm_config.parallel_config
        self.path=Path(os.environ['DONOR_DP_OUTPUT'])/f'decode-shadow-rank{p.data_parallel_rank*p.tensor_parallel_size+worker.rank}.json'
        r._cross_step_bounds.prepare=self.prepare

    def prepare(self,schedule,counts):
        r=self.r;n=r.input_batch.num_reqs
        self.active=None
        if not self.enabled or not stable_verification(r,schedule,counts,r.max_num_reqs):return self.native(schedule,counts)
        # Decline any special masked/partial query rather than approximating it.
        upper=r.input_batch.num_computed_tokens_cpu[:n]+counts
        if any(int(upper[i])<r.requests[rid].num_tokens or int(upper[i])>r.max_model_len
               for i,rid in enumerate(r.input_batch.req_ids)):
            return self.native(schedule,counts)
        assert r._draft_token_ids.ndim==2 and r._draft_token_ids.shape[1]==5
        r._build_attn_state(n,counts,np.ones(n,dtype=np.int32));r.with_prefill=False
        r._compute_prev_positions(n)
        cu=r._get_cumsum_and_arange(counts,r.query_pos.np)
        rows=np.repeat(r.arange_np[:n],counts)
        np.add(r.input_batch.num_computed_tokens_cpu[rows],r.query_pos.np[:n*6],out=r._positions_np_buf[:n*6])
        r.query_lens=torch.from_numpy(counts)
        r.query_start_loc.np[0]=0;r.query_start_loc.np[1:n+1]=cu
        r.optimistic_seq_lens_cpu[:n].copy_(torch.from_numpy(upper));r.optimistic_seq_lens_cpu[n:].zero_()
        r.num_discarded_requests=0;r.discard_request_mask.np[:n]=False
        r.num_accepted_tokens.np[:n]=r.input_batch.num_accepted_tokens_cpu[:n];r.num_accepted_tokens.np[n:]=1
        r.req_indices.np[:n*6]=rows;r.num_scheduled_tokens.np[:n]=counts
        r.num_decode_draft_tokens.np[:n]=5;r.num_decode_draft_tokens.np[n:]=-1
        key=(n,self.sequence%2)
        if key not in self.slots:self.slots[key]=Slot(r,n,self.ingress)
        slot=self.slots[key];slot.project();self.active=slot
        checking=self.verify and len(self.checks)<12
        fields=device_fields(r,n)
        before=[x.clone() for x in fields] if checking else None
        result=slot.replay();self.sequence+=1
        # Later builder needs a conservative CPU tiling bound, not a D2H read.
        r._cross_step_bounds.skipped=True;r._cross_step_bounds.bypassed+=1
        if checking:self.check(schedule,counts,fields,before,result,n)
        return result

    def check(self,schedule,counts,fields,before,result,n):
        def tensors(output):
            meta=output[1]
            return [output[0],meta.draft_token_ids,meta.cu_num_draft_tokens,meta.cu_num_sampled_tokens,
                    meta.target_logits_indices,meta.bonus_logits_indices,meta.logits_indices]
        torch.npu.synchronize();after=[x.clone() for x in fields]
        left=[x.clone() for x in tensors(result)]
        saved=self.r.logits_indices
        row=dict(sequence=self.sequence,requests=n,status='RUNNING')
        try:
            for dst,src in zip(fields,before):dst.copy_(src)
            reference=self.native(schedule,counts);torch.npu.synchronize()
            for i,(actual,expected) in enumerate(zip(after,fields)):
                assert torch.equal(actual,expected),f'device preparation field {i} differs'
            right=tensors(reference)
            assert result[2]==reference[2] and result[1].num_draft_tokens==reference[1].num_draft_tokens
            assert len(left)==len(right)
            for name,actual,expected in zip(('logits','draft_ids','cu_drafts','cu_samples','target','bonus','metadata_logits'),left,right):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0,msg=lambda message: f'{name}: {message}')
            row['status']='PASS'
        except Exception as error:row.update(status='FAIL',error=str(error));raise
        finally:
            for dst,src in zip(fields,after):dst.copy_(src)
            self.r.logits_indices=saved
            self.checks.append(row);self.path.write_text(json.dumps(self.checks,indent=2))

    def receipt(self):
        return dict(stage='explicit-host-ingress-device-preparation', replays=self.sequence,
                    banks=len(self.slots), checks=self.checks,
                    pinned_bytes=sum(x.numel()*x.element_size() for slot in self.slots.values() for x in slot.host.values()),
                    independent_h2d_stream=True, single_copy_progress=True)
