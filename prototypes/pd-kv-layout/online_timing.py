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
    def record(self,stage,**fields):
        self.file.write(json.dumps(dict(stage=stage,ns=time.perf_counter_ns(),**fields))+"\n")
        self.count+=1
        if self.count%64==0:self.file.flush()

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
        if not isinstance(core.output_queue,OutputQueue):
            core.output_queue=OutputQueue(core.output_queue,Recorder("core"))
        recorder=core.output_queue.recorder
        begin=time.perf_counter_ns()
        result=original(core)
        recorder.record("core-step",begin_ns=begin)
        return result
    EngineCoreProc._process_engine_step=step
    EngineCoreProc._pd_timing_installed=True
