"""Explicit local TP2 pools for the two-host naive PD experiment.

P instances have DP1; the D instance has native DP4/TP2/EP8. No cross-host EP.
Target-only behavior and native dummy-rank protocol match the D6 experiment.
"""
from betterscale.models import qwen35
from checkpoint_entry import Scheduler as BaseScheduler, Worker as BaseWorker
from ep6_state_entry import idle_target_only

qwen35.STATE_SCHEDULER = "pool_state_entry.Scheduler"

class Scheduler(BaseScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Clearing next-step proposals alone misses native WAITING hot-hit
        # admission: alongside running decode it pads 1+num_spec_tokens before
        # _update_after_schedule. Declare zero proposals at the scheduling
        # source; keep native lookahead allocation and runner startup unchanged.
        self.num_spec_tokens=0
        self._spec_token_placeholders=[]

class Worker(BaseWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        p=vllm_config.parallel_config
        assert (p.tensor_parallel_size,p.data_parallel_size,p.enable_expert_parallel) in (
            (2,1,False),(2,4,True))
        assert not p.use_sequence_parallel_moe
        super().__init__(vllm_config,*args,**kwargs)
        self._pd_distributed = p.data_parallel_size > 1
        if self._pd_distributed:
            from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
            original=NPUModelRunner._dummy_run
            def dummy(runner,*args,**kwargs):
                return idle_target_only(runner,original,*args,**kwargs)
            NPUModelRunner._dummy_run=dummy

    def compile_or_warm_up_model(self):
        result=super().compile_or_warm_up_model()
        if self._pd_distributed:self.model_runner._pd_target_only_ready=True
        return result
