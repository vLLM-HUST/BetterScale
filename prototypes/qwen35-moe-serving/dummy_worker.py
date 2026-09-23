"""Bounded diagnostic observer over the native+ABI worker, never a MOD claim."""
import os
import time
from pathlib import Path
from native_worker import Worker as Native
from profiling import ProfileWindow


class ProfileObserver:
    def load_model(self,*args,**kwargs):
        result=super().load_model(*args,**kwargs)
        runner=self.model_runner;original=runner._determine_batch_execution_and_padding
        def determine(num_tokens,num_reqs,num_scheduled_tokens_np,*a,**kw):
            tokens,requests,scheduled=num_tokens,num_reqs,num_scheduled_tokens_np
            w=getattr(self,'window',None)
            if w is not None and not w.started and not w.closed and self.min_context:
                computed=runner.input_batch.num_computed_tokens_cpu[:requests]
                if len(computed) and max(computed)>=self.min_context:
                    import torch
                    torch.npu.synchronize()  # Diagnostic window boundary only.
                    w.prof.start();w.mark('profile_start');w.started=True
                    w.warmup_steps=w.count
            result=original(num_tokens,num_reqs,num_scheduled_tokens_np,*a,**kw)
            if w is not None and w.started and not w.closed:
                w.schedule.append({'event':'dispatch','sequence':w.count,'wall_ns':time.time_ns(),
                    'tokens':int(tokens),'requests':int(requests),'scheduled':scheduled.tolist(),
                    'mode':str(result[0]),'descriptor':repr(result[1]),
                    'computed':runner.input_batch.num_computed_tokens_cpu[:requests].tolist(),
                    'prompt_tokens':runner.input_batch.num_prompt_tokens_cpu_tensor[:requests].tolist()})
            return result
        runner._determine_batch_execution_and_padding=determine
        return result
    def begin_dummy_profile(self,label,warmup,min_context=0):
        assert getattr(self,'window',None) is None or self.window.closed
        self.min_context=min_context
        self.window=ProfileWindow(str(Path(os.environ['CAPSULE'])/'profiles'/label),self.rank,6,warmup_steps=(1<<30) if min_context else warmup)
        return {'rank':self.rank,'label':label}
    def end_dummy_profile(self):
        self.window.close()
        return {'rank':self.rank,'steps':self.window.count}
    def sample_tokens(self,*a,**kw):
        result=super().sample_tokens(*a,**kw)
        window=getattr(self,'window',None)
        if window is not None and not window.closed:window.step()
        return result


class Worker(ProfileObserver, Native):
    pass
