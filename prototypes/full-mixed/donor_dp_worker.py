"""Native donor observation and LCM startup repair; no replay optimizations."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch


class DonorDPWorker:
    def enable_worker_continuous(self):
        from worker_submission import WorkerSubmission
        self._worker_submission=WorkerSubmission(self)
        return {'scope':'whole-wave-submit-before-host-retirement'}

    def enable_producer_draft(self):
        from split_draft import SplitDraftGraphSet
        r=self.model_runner;p=r.vllm_config.parallel_config
        assert p.tensor_parallel_size>1, 'DSACP split-draft composition first; DSA requires its own gate'
        assert hasattr(self,'_decode_metadata') and not hasattr(self,'_exact_draft_graph')
        observer=SimpleNamespace(rank=p.data_parallel_rank*p.tensor_parallel_size+self.rank,model_runner=r)
        self._exact_draft_graph=SplitDraftGraphSet(observer,max_requests=r.max_num_reqs)
        r.drafter._runnable=self._exact_draft_graph
        return dict(policy='shadow-producer-plus-kept-split-draft',max_requests=r.max_num_reqs)

    def enable_shadow_decode(self,verify=False,metadata=False):
        from decode_shadow import DecodeShadow
        self._decode_shadow=DecodeShadow(self,verify)
        if metadata:
            if self.model_runner.vllm_config.parallel_config.tensor_parallel_size > 1:
                from qli_cpu import configure
                configure(self,True,False)
            from decode_metadata import DecodeMetadata
            self._decode_metadata=DecodeMetadata(self._decode_shadow)
        return {'scope':'explicit stable K5 input producer','verify':verify}

    def enable_producer_shadow_audit(self):
        from producer_shadow import install
        return install(self)

    def enable_pingpong(self):
        torch.npu.synchronize()
        pair=self.model_runner.model._decode_pair
        pair.stream=torch.npu.current_stream().npu_stream
        return pair.receipt()

    def set_pingpong_policy(self, policy):
        assert policy in ('native', 'cut', 'sources', 'pair', 'producer', 'metadata', 'draft', 'worker', 'worker-tree')
        torch.npu.synchronize()
        from draft_graph import ExactDraftGraph
        ExactDraftGraph.metadata_dag=policy!='worker-tree'
        execution_policy='worker' if policy=='worker-tree' else policy
        label=policy;policy=execution_policy
        r=self.model_runner;pair=r.model._decode_pair;slots=r._host_source_slots
        assert pair.reference_catalog
        pair.policy='native' if policy=='native' else 'pair' if policy in ('pair','producer','metadata','draft','worker') else 'ordered'
        r.synchronize_input_prep=slots.scope if policy in ('sources','pair','producer','metadata','draft','worker') else slots.original_scope
        r._cross_step_bounds.enabled=policy!='native'
        if hasattr(self,'_decode_shadow'):self._decode_shadow.enabled=policy in ('producer','metadata','draft','worker')
        if hasattr(self,'_decode_metadata'):self._decode_metadata.enabled=policy in ('metadata','draft','worker')
        if hasattr(self,'_exact_draft_graph'):self._exact_draft_graph.enabled=policy in ('draft','worker')
        if hasattr(self,'_worker_submission'):self._worker_submission.enabled=policy=='worker'
        if r.vllm_config.parallel_config.tensor_parallel_size > 1:
            from qli_cpu import configure
            configure(self,policy!='native',False)
        history=getattr(self,'_policy_memory',[])
        history.append(dict(policy=label,allocated=torch.npu.memory_allocated(),
                            reserved=torch.npu.memory_reserved(),
                            preparation_banks=len(self._decode_shadow.slots) if hasattr(self,'_decode_shadow') else 0,
                            metadata_graphs=len(self._decode_metadata.entries) if hasattr(self,'_decode_metadata') else 0))
        self._policy_memory=history
        rank=r.vllm_config.parallel_config.data_parallel_rank*r.vllm_config.parallel_config.tensor_parallel_size+self.rank
        (Path(os.environ['DONOR_DP_OUTPUT'])/f'policy-memory-rank{rank}.json').write_text(json.dumps(history,indent=2))
        return {'policy':label}

    def enable_pingpong_shadow(self):
        pair=self.model_runner.model._decode_pair
        assert pair.reference_catalog
        pair.shadow_runner=self.model_runner
        return {'shadow':'original-native-graph'}

    def enable_pingpong_continuous(self):
        from cross_step import CrossStepBounds
        r=self.model_runner;p=r.vllm_config.parallel_config
        os.environ['FULL_MIXED_OUTPUT']=os.environ['DONOR_DP_OUTPUT']
        observer=SimpleNamespace(rank=p.data_parallel_rank*p.tensor_parallel_size+self.rank,model_runner=r)
        r._cross_step_bounds=CrossStepBounds(observer,native_dsa=True,max_requests=r.max_num_reqs)
        return r._cross_step_bounds.receipt()

    def enable_pingpong_sources(self):
        from pingpong_sources import install
        return install(self)

    def start_window(self, label, profile=False):
        from diagnostics import start_decode_observation
        p = self.model_runner.vllm_config.parallel_config
        rank = p.data_parallel_rank * p.tensor_parallel_size + self.rank
        root = Path(os.environ['DONOR_DP_OUTPUT']) / label
        root.mkdir(parents=True,exist_ok=True)
        os.environ['FULL_MIXED_OUTPUT'] = str(root)
        self._donor_observer = SimpleNamespace(rank=rank,model_runner=self.model_runner)
        receipt=start_decode_observation(self._donor_observer,profile,label,
                                         profile_steps=int(os.environ.get('DONOR_DP_PROFILE_STEPS', '0')))
        runner=self.model_runner
        original=runner._determine_batch_execution_and_padding
        self._donor_modes=[]
        self._donor_dummy=False
        dummy=runner._dummy_run
        def observed_dummy(*args,**kwargs):
            previous=self._donor_dummy;self._donor_dummy=True
            try:return dummy(*args,**kwargs)
            finally:self._donor_dummy=previous
        self._donor_observer._decode_observation_originals.append((runner,'_dummy_run',dummy))
        runner._dummy_run=observed_dummy
        def observed(*args,**kwargs):
            result=original(*args,**kwargs)
            self._donor_modes.append(dict(mode=str(result[0]),padded=result[1].num_tokens,dummy=self._donor_dummy,
                across_dp=None if result[3] is None else result[3].tolist(),
                actual_tokens=int(kwargs.get('num_tokens',args[0] if args else -1)),
                actual_requests=int(kwargs.get('num_reqs',args[1] if len(args)>1 else -1))))
            return result
        self._donor_observer._decode_observation_originals.append((runner,'_determine_batch_execution_and_padding',original))
        runner._determine_batch_execution_and_padding=observed
        return receipt

    def stop_window(self):
        from diagnostics import stop_decode_observation
        result=stop_decode_observation(self._donor_observer)
        root=Path(os.environ['FULL_MIXED_OUTPUT']);label=self._donor_observer._decode_observation_label
        (root/f'{label}-modes-rank{self._donor_observer.rank}.json').write_text(json.dumps(self._donor_modes,indent=2))
        return result

    def donor_receipt(self):
        from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
        from vllm.distributed import get_ep_group
        r=self.model_runner;p=r.vllm_config.parallel_config
        draft=self._exact_draft_graph.receipt() if hasattr(self,'_exact_draft_graph') else None
        tokens,concurrency=get_kv_cache_capacity(r.vllm_config,r.kv_cache_config)
        return dict(worker_submission=(self._worker_submission.receipt() if hasattr(self,'_worker_submission') else None),
                    split_draft=draft,decode_shadow=(self._decode_shadow.receipt() if hasattr(self,'_decode_shadow') else None),
                    decode_metadata=(self._decode_metadata.receipt() if hasattr(self,'_decode_metadata') else None),
                    cross_step=(r._cross_step_bounds.receipt() if '_cross_step_bounds' in r.__dict__ else None),
                    host_source_slots=(r._host_source_slots.receipt() if '_host_source_slots' in r.__dict__ else None),
                    pingpong=(r.model._decode_pair.receipt() if '_decode_pair' in r.model.__dict__ else None),
                    dp_rank=p.data_parallel_rank,tp_rank=self.rank,ep_rank=get_ep_group().rank_in_group,
                    ep_size=get_ep_group().world_size,device=str(r.device),kv_capacity_tokens=tokens,
                    max_length_concurrency=concurrency,allocated=torch.npu.memory_allocated(),
                    reserved=torch.npu.memory_reserved(),peak=torch.npu.max_memory_allocated(),
                    attention_builders=sorted({type(g.get_metadata_builder()).__name__ for groups in r.attn_groups for g in groups}))


# Dummy fixture only: reproduce the checkpoint loader's wo_a layout conversion.
# No native real-weight code or graph admission is replaced.
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
_original_load=NPUModelRunner.load_model

def dummy_layout(self,*args,**kwargs):
    result=_original_load(self,*args,**kwargs)
    if self.vllm_config.load_config.load_format=='dummy':
        for model in [self.get_model(),getattr(getattr(self,'drafter',None),'model',None)]:
            if model is None:continue
            for name,m in model.named_modules():
                if name.endswith('wo_a') and m.weight.ndim==2:
                    m.weight.data=m.weight.data.view(m.n_local_groups,m.o_lora_rank,-1).transpose(2,1).contiguous()
    return result

NPUModelRunner.load_model=dummy_layout

# Existing K5/TP8 startup compatibility repair, not a replay optimization.
# The pinned upstream uses max(6,8) and rejects otherwise-valid LCM24 buckets.
from vllm.config import CompilationConfig
_original_adjust=CompilationConfig.adjust_cudagraph_sizes_for_spec_decode

def joint_alignment(self,query_len,tp):
    import math
    alignment=math.lcm(query_len,tp) if self.pass_config.enable_sp else query_len
    return _original_adjust(self,alignment,tp)

CompilationConfig.adjust_cudagraph_sizes_for_spec_decode=joint_alignment


if os.environ.get('DONOR_PINGPONG') == '1':
    import pingpong_graph  # opt-in before native startup graph capture
