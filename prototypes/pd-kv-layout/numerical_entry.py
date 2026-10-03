"""Explicit numerical probe: retired host metadata and optional fixed cold seat.

No model arithmetic, source pins, or shared runtime files are changed.
"""
import os
from concurrent.futures import Future
from ep6_state_entry import Worker as BaseWorker, Scheduler as BaseScheduler
from betterscale.models import qwen35
from betterscale.models.qwen35.resident_leases import Offer
from model_checkpoint import idle
qwen35.STATE_SCHEDULER='numerical_entry.Scheduler'


def snapshot(core, salt):
    scheduler=idle(core)
    seats=[s for s in scheduler.residents.seats if s.cache_salt==salt and s.tokens]
    if len(seats)!=1:raise ValueError('Expected one retired probe resident')
    s=seats[0]
    return dict(seat=s.index,epoch=s.epoch,cursor=s.cursor,
                pages=[[b.block_id for b in group] for group in s.blocks.blocks],
                retired_step=s.fence,processed_step=scheduler.processed_step_seq)


def observe(core, salt):
    try:idle(core)
    except RuntimeError:
        if getattr(core,'_numerical_pending',None) is not None:raise RuntimeError('Probe already pending')
        future=Future();core._numerical_pending=(future,salt);return future
    return snapshot(core,salt)


def service(core):
    pending=getattr(core,'_numerical_pending',None)
    if pending is None:return
    try:idle(core)
    except RuntimeError:return
    core._numerical_pending=None;future,salt=pending
    try:future.set_result(snapshot(core,salt))
    except BaseException as exc:future.set_exception(exc)


class Scheduler(BaseScheduler):
    def __init__(self,*args,**kwargs):
        from vllm.v1.engine.core import EngineCore,EngineCoreProc
        EngineCore.pd_probe_retired=observe
        original=EngineCoreProc._process_engine_step
        if not getattr(original,'_numerical_probe',False):
            def step(core,*a,**kw):
                result=original(core,*a,**kw);service(core);return result
            step._numerical_probe=True;EngineCoreProc._process_engine_step=step
        super().__init__(*args,**kwargs)
        mode=os.environ['BETTERSCALE_NUMERICAL_RESIDENCY']
        if mode not in ('natural','fixed-seat'):raise ValueError(mode)
        if mode=='fixed-seat':
            def offer(tokens,salt,completed_step,*,allow_hit=True):
                seat=self.residents.seats[0]
                if seat.owner is not None or seat.fence>completed_step:return None
                # Ordinary cold-victim retirement/allocation remains unchanged.
                return Offer(0,seat.epoch,False)
            self.residents.offer=offer


class Worker(BaseWorker):
    def compile_or_warm_up_model(self):
        activate=lambda:None
        if output:=os.environ.get('BETTERSCALE_NUMERICAL_TRACE'):
            from numerical_trace import install
            activate=install(self,output)
        result=super().compile_or_warm_up_model()
        activate()
        return result
