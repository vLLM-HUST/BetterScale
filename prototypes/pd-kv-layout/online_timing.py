"""Opt-in host boundary receipts; no synchronization, device reads or new steps."""
import atexit
import json
import os
from pathlib import Path
import time

class Recorder:
    def __init__(self,role):
        directory=Path(os.environ["BETTERSCALE_PD_TIMING_DIR"])
        directory.mkdir(parents=True,exist_ok=True)
        self.file=(directory/f"{role}-{os.getpid()}.jsonl").open("a",buffering=65536)
        atexit.register(self.file.close)
        self.count=0
        if role!="gc" and os.environ.get("BETTERSCALE_PD_GC_TIMING")=="1":observe_gc()
    def record(self,stage,**fields):
        self.file.write(json.dumps(dict(stage=stage,ns=time.perf_counter_ns(),**fields))+"\n")
        self.count+=1
        if self.count%64==0:self.file.flush()


_gc_started=False


def observe_gc():
    """Attribute Python collection pauses without changing GC policy or cadence."""
    global _gc_started
    if _gc_started:return
    # Set before opening the recorder: another recording thread may enter here.
    _gc_started=True
    import gc
    recorder=Recorder("gc");begins={}
    def callback(phase,info):
        generation=info["generation"]
        if phase=="start":begins[generation]=time.perf_counter_ns()
        elif phase=="stop":
            begin=begins.pop(generation,None)
            if begin is not None and not recorder.file.closed:
                recorder.record("python-gc",begin_ns=begin,generation=generation,
                    collected=info.get("collected",0),uncollectable=info.get("uncollectable",0))
    gc.callbacks.append(callback)
    # Registered after Recorder's close, so this runs before the file closes.
    atexit.register(lambda:gc.callbacks.remove(callback) if callback in gc.callbacks else None)


def observe_method(owner,name,role,stage):
    """Time a bound host call without awaiting or touching its result."""
    original=getattr(owner,name)
    if getattr(original,"_pd_timed_call",False):return
    recorder=Recorder(role)
    def observed(*args,**kwargs):
        begin=time.perf_counter_ns()
        fields={}
        if name=="collective_rpc":
            method=args[0] if args else kwargs.get("method")
            fields["method"]=method if isinstance(method,str) else getattr(method,"__name__","callable")
            fields["non_block"]=bool(kwargs.get("non_block",False))
        try:return original(*args,**kwargs)
        finally:recorder.record(stage,begin_ns=begin,**fields)
    observed._pd_timed_call=True
    setattr(owner,name,observed)

class OutputQueue:
    def __init__(self,queue,recorder):
        self.queue,self.recorder=queue,recorder
        self.sender=Recorder("sender")
    def __getattr__(self,name):return getattr(self.queue,name)
    def note(self,stage,value):
        if isinstance(value,tuple) and len(value)==2:
            outputs=getattr(value[1],"outputs",None)
            if outputs:
                recorder=self.sender if stage=="sender-dequeue" else self.recorder
                recorder.record(stage,requests=[(o.request_id,len(o.new_token_ids)) for o in outputs])
    def put_nowait(self,value):
        self.note("core-enqueue",value)
        return self.queue.put_nowait(value)
    def get(self,*args,**kwargs):
        value=self.queue.get(*args,**kwargs)
        self.note("sender-dequeue",value)
        return value

def install_core():
    from vllm.v1.engine.core import EngineCoreProc
    if getattr(EngineCoreProc,"_pd_timing_installed",False):return
    original=EngineCoreProc._process_engine_step
    def step(core):
        observe_method(core.model_executor,"collective_rpc","core-rpc","executor-rpc")
        if not isinstance(core.output_queue,OutputQueue):
            core.output_queue=OutputQueue(core.output_queue,Recorder("core"))
        recorder=core.output_queue.recorder
        begin=time.perf_counter_ns()
        result=original(core)
        recorder.record("core-step",begin_ns=begin)
        return result
    EngineCoreProc._process_engine_step=step
    EngineCoreProc._pd_timing_installed=True
