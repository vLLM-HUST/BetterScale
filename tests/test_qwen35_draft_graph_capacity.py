"""MTP step0 keeps the already-classified target graph envelope."""
from types import SimpleNamespace
import pytest
from betterscale.models.qwen35.draft_banks import target_capacity


def test_warm_prefill_does_not_redispatch_to_verification_bin():
    seen=[]
    def dispatch(n,**kw):
        seen.append((n,kw))
        return next(k for k in (3,6,12,16,24,32,40,48,64) if k>=n)
    d=SimpleNamespace(dispatch=dispatch)
    assert d.dispatch(17)==24
    with target_capacity(d,32):
        assert d.dispatch(num_tokens=17,uniform_decode=False)==32
        assert d.dispatch(40)==40  # never shrink a DP-wide requirement
    assert d.dispatch is dispatch
    assert seen[1]==(32,dict(uniform_decode=False))


def test_decode_capacity_and_exception_restore_dispatcher():
    def dispatch(n):return n
    d=SimpleNamespace(dispatch=dispatch)
    with pytest.raises(RuntimeError,match="probe"):
        with target_capacity(d,6):
            assert d.dispatch(4)==6
            raise RuntimeError("probe")
    assert d.dispatch is dispatch
    with pytest.raises(ValueError):
        with target_capacity(d,0):pass
