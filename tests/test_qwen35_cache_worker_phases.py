from types import SimpleNamespace as S
import pytest
from betterscale.models.qwen35 import cache_worker as module


@pytest.mark.parametrize("fail_remote", [False,True])
def test_late_replica_ack_does_not_release_a_reused_rank_slot(monkeypatch,fail_remote):
    monkeypatch.setattr(module.torch,"npu",S(set_device=lambda device:None),raising=False)
    worker=module.CacheWorker.__new__(module.CacheWorker)
    worker.inflight={0};worker.verify={};worker.runner=S(device="test")
    messages=[]
    worker._send=lambda command,error=None,phase=None:messages.append((phase,error))
    command=dict(kind="store",seat=0,epoch=4,_device_staged=False)
    def result():
        worker._staged(command,"A",False,123)
        assert worker.inflight==set()
        # Core staged quorum permits a new State operation on the same rank.
        worker.inflight.add(0)
        if fail_remote:raise IOError("replica failed")
    worker._finish(command,S(result=result,byte_length=123),"A",False)
    assert worker.inflight=={0}
    assert messages[0]==("staged",None) and len(messages)==2
    assert bool(messages[1][1])==fail_remote


def test_legacy_single_phase_rank_release_is_unchanged(monkeypatch):
    monkeypatch.setattr(module.torch,"npu",S(set_device=lambda device:None),raising=False)
    worker=module.CacheWorker.__new__(module.CacheWorker)
    worker.inflight={0};worker.verify={};worker.runner=S(device="test")
    messages=[]
    worker._send=lambda command,error=None:messages.append(error)
    command=dict(kind="store",seat=0,epoch=4,_device_staged=False)
    worker._finish(command,S(result=lambda:None,byte_length=123),"A",False)
    assert worker.inflight==set() and messages==[None]
