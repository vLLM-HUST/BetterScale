"""Native EP control: benchmark sampler/ABI only, no separated expert hooks."""
import json
import os
from pathlib import Path
from vllm_ascend.worker.worker import NPUWorker


class Worker(NPUWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        from betterscale.models.qwen import check_runtime
        from betterscale.patches.expert_service.mamba_abi import install
        from betterscale.patches import benchmark_mtp
        check_runtime('expert_pins.json')
        install()
        benchmark_mtp.install(vllm_config)
        super().__init__(vllm_config, *args, **kwargs)

    def load_model(self, *args, **kwargs):
        result=super().load_model(*args, **kwargs)
        from betterscale.patches.qwen_mtp_feedback import install
        install(self.model_runner)
        # Same mode0 FULL-decode route as the separated client. The pinned
        # native flag otherwise keys off Dynamo mode and skips FIA workspaces.
        if not self.vllm_config.model_config.enforce_eager:
            assert int(self.vllm_config.compilation_config.mode)==0
            self.model_runner.use_aclgraph=True
        return result

    def ep_receipt(self):
        from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
        ep,dp,tp=get_ep_group(),get_dp_group(),get_tp_group()
        assert ep.world_size==8
        layers=[]
        for domain,model in [('target',self.model_runner.get_model()),
                             ('draft',self.model_runner.drafter.model)]:
            count=0
            for name,module in model.named_modules():
                if not hasattr(module,'w13_weight') or not hasattr(module,'expert_map'):continue
                assert module.expert_map is not None,name
                ids=(module.expert_map>=0).nonzero().flatten().cpu().tolist()
                shape=list(module.w13_weight.shape)
                assert len(ids)==shape[0]==32,(name,ids,shape)
                layers.append(dict(domain=domain,name=name,global_expert_ids=ids,w13_shape=shape))
                count+=1
            assert count==({'target':40,'draft':1}[domain]),(domain,count)
        receipt=dict(ep_size=ep.world_size,ep_rank=ep.rank_in_group,
                     dp_size=dp.world_size,tp_size=tp.world_size,layers=layers,
                     use_aclgraph=self.model_runner.use_aclgraph,
                     feedback_isolated=hasattr(self.model_runner,"_bs_mtp_feedback"))
        (Path(os.environ['EXPERT_BENCH_OUTPUT'])/f'ep-rank-{ep.rank_in_group}.json').write_text(json.dumps(receipt)+'\n')
        return receipt
