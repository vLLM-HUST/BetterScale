"""Opt-in pinned-donor DP finish cadence experiment; no manual peer wake."""
import os


def finish_sync(core,local_unfinished,interval,sync):
    # Preserve the donor step counter and pause-consensus state transitions.
    core.step_counter+=1
    if core.step_counter%interval:
        return True
    unfinished,paused=sync(core.dp_group,has_unfinished=local_unfinished,pending_pause=core.pending_pause)
    if paused:
        core.ignore_start_dp_wave=True
        core.pending_pause=False
    return unfinished


def install():
    interval=int(os.environ.get('BETTERSCALE_PD_FINISH_SYNC_STEPS','32'))
    if interval not in (1,4,8,32):raise ValueError('Unqualified DP finish cadence')
    if interval==32:return # Leave the pinned donor implementation untouched.
    from vllm.v1.engine.core import DPEngineCoreProc
    from vllm.config import ParallelConfig
    def check(core,local_unfinished):
        return finish_sync(core,local_unfinished,interval,ParallelConfig.sync_dp_state)
    check._pd_finish_sync_interval=interval
    DPEngineCoreProc._has_global_unfinished_reqs=check
    DPEngineCoreProc._pd_finish_sync_interval=interval
