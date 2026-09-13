"""Native donor observation and LCM startup repair; no replay optimizations."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch


class DonorDPWorker:
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
        def observed(*args,**kwargs):
            result=original(*args,**kwargs)
            self._donor_modes.append(dict(mode=str(result[0]),padded=result[1].num_tokens,
                across_dp=None if result[3] is None else result[3].tolist()))
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
        tokens,concurrency=get_kv_cache_capacity(r.vllm_config,r.kv_cache_config)
        return dict(dp_rank=p.data_parallel_rank,tp_rank=self.rank,ep_rank=get_ep_group().rank_in_group,
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
