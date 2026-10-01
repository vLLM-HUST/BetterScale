"""Experimental DP3 State admission plus non-writing target-only idle ranks."""
from betterscale.models import qwen35
from checkpoint_entry import Scheduler as BaseScheduler,Worker as BaseWorker
qwen35.STATE_SCHEDULER='ep6_state_entry.Scheduler'
class Scheduler(BaseScheduler):pass


def idle_target_only(runner,native,*args,**kwargs):
    if not getattr(runner,'_pd_target_only_ready',False):return native(runner,*args,**kwargs)
    from vllm.config import CUDAGraphMode
    # Idle EP participants must execute target collectives but cannot write a
    # resident's State or execute extra MTP collectives absent on active peers.
    if len(args)>1 or kwargs.get('force_attention') or kwargs.get('is_graph_capturing') or kwargs.get('is_profile'):
        raise RuntimeError('Unqualified runtime idle-rank invocation')
    kwargs.update(cudagraph_runtime_mode=CUDAGraphMode.NONE,force_attention=False)
    drafter=runner.drafter
    try:
        runner.drafter=None
        return native(runner,*args,**kwargs)
    finally:runner.drafter=drafter

class Worker(BaseWorker):
 def __init__(self,vllm_config,*args,**kwargs):
  config=vllm_config
  p=config.parallel_config
  assert (p.tensor_parallel_size,p.data_parallel_size,p.enable_expert_parallel)==(2,3,True)
  assert not p.use_sequence_parallel_moe
  super().__init__(config,*args,**kwargs)
  from vllm_ascend.worker.model_runner_v1 import NPUModelRunner as Runner
  original=Runner._dummy_run
  def dummy(runner,*args,**kwargs):return idle_target_only(runner,original,*args,**kwargs)
  Runner._dummy_run=dummy
 def compile_or_warm_up_model(self):
  result=super().compile_or_warm_up_model()
  self.model_runner._pd_target_only_ready=True
  return result
