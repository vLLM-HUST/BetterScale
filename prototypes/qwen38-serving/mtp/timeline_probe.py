"""Diagnostic hooks on the existing Worker; no new Worker or timed-run syncs."""
import os
import time
from pathlib import Path


def install():
    if os.environ.get('MTP_PROFILE') != '1':
        return
    from vllm_ascend.worker.worker import NPUWorker
    from profiling import ProfileWindow
    original_load = NPUWorker.load_model
    original_sample = NPUWorker.sample_tokens

    def load(worker,*args,**kwargs):
        result = original_load(worker,*args,**kwargs)
        runner = worker.model_runner
        original = runner._determine_batch_execution_and_padding
        def determine(num_tokens,num_reqs,num_scheduled_tokens_np,*a,**kw):
            result = original(num_tokens,num_reqs,num_scheduled_tokens_np,*a,**kw)
            w = getattr(worker,'_mtp_window',None)
            if w is not None and w.started and not w.closed:
                w.schedule.append(dict(event='dispatch',sequence=w.count,wall_ns=time.time_ns(),
                    tokens=int(num_tokens),requests=int(num_reqs),
                    scheduled=num_scheduled_tokens_np.tolist(),mode=str(result[0]),
                    descriptor=repr(result[1]),
                    computed=runner.input_batch.num_computed_tokens_cpu[:num_reqs].tolist(),
                    prompt_tokens=runner.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs].tolist()))
            return result
        runner._determine_batch_execution_and_padding = determine
        return result

    def profile(worker,is_start=True,profile_prefix=None):
        if is_start:
            old = getattr(worker,'_mtp_window',None)
            assert old is None or old.closed
            index = getattr(worker,'_mtp_profile_index',0)
            assert index < 2
            label = ('decode','mixed')[index]
            worker._mtp_profile_index = index+1
            worker._mtp_window = ProfileWindow(str(Path(os.environ['CAPSULE'])/'profiles'/label),
                                               worker.rank,steps=6,warmup_steps=8 if index==0 else 0)
        else:
            worker._mtp_window.close()

    def sample(worker,*args,**kwargs):
        result = original_sample(worker,*args,**kwargs)
        w = getattr(worker,'_mtp_window',None)
        if w is not None and not w.closed:
            w.step()
        return result

    NPUWorker.load_model = load
    NPUWorker.profile = profile
    NPUWorker.sample_tokens = sample
