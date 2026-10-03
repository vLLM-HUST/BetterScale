from types import SimpleNamespace
import pytest
from model_dp_control import finish_sync,install


@pytest.mark.parametrize('interval',[1,4,8,32])
def test_cadence_preserves_counter_and_sync_inputs(interval):
    core=SimpleNamespace(step_counter=0,dp_group='group',pending_pause=False,ignore_start_dp_wave=False)
    calls=[]
    def sync(group,**kw):calls.append((group,kw));return False,False
    for step in range(1,interval+1):
        assert finish_sync(core,False,interval,sync)==(step!=interval)
        assert core.step_counter==step
    assert calls==[('group',dict(has_unfinished=False,pending_pause=False))]


def test_pause_consensus_and_global_work_are_preserved():
    core=SimpleNamespace(step_counter=0,dp_group='group',pending_pause=True,ignore_start_dp_wave=False)
    def sync(group,**kw):
        assert kw==dict(has_unfinished=True,pending_pause=True)
        return True,True
    assert finish_sync(core,True,1,sync)
    assert core.ignore_start_dp_wave and not core.pending_pause


def test_default_install_does_not_import_or_patch_donor(monkeypatch):
    monkeypatch.delenv('BETTERSCALE_PD_FINISH_SYNC_STEPS',raising=False)
    install()


def test_unqualified_interval_fails_before_donor_import(monkeypatch):
    monkeypatch.setenv('BETTERSCALE_PD_FINISH_SYNC_STEPS','2')
    with pytest.raises(ValueError):install()
