import pytest
from pd_limits import context_limit,state_budget,checkpoint_limit
from naive_pool_node import validate_rpc,pack,unpack
from model_checkpoint import restore
from test_model_checkpoint import core,payload


def test_explicit_long_context_admission_and_checkpoint(monkeypatch):
    monkeypatch.setenv("BETTERSCALE_PD_CONTEXT","262144")
    assert context_limit()==262144 and state_budget()==26038239232
    assert checkpoint_limit()==6<<30
    tokens=[1]*262143
    validate_rpc("D",dict(instance=0,op="generate_batch",
                 args=dict(items=[dict(owner=3,salt="s",tokens=tokens,n=1)])))
    assert unpack(pack(tokens))==tokens
    p=payload();p["header"].update(tokens=[1]*262144,cursor=262143)
    c=core();result=restore(c,p,"s")
    assert len(result["blocks"])==128
    p["header"].update(tokens=[1]*262145,cursor=262144)
    with pytest.raises(ValueError):restore(core(),p,"too-long")


def test_short_context_and_unknown_configuration(monkeypatch):
    monkeypatch.delenv("BETTERSCALE_PD_CONTEXT",raising=False)
    assert context_limit()==8192 and state_budget()==8<<30
    monkeypatch.setenv("BETTERSCALE_PD_CONTEXT","65536")
    with pytest.raises(ValueError):context_limit()
