"""No NPU: idle ranks must skip State attention and draft collectives."""
import ast,sys
from pathlib import Path
from types import SimpleNamespace
import pytest

def helper(monkeypatch):
 node=next(n for n in ast.parse(Path(__file__).with_name('ep6_state_entry.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='idle_target_only')
 ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'idle','exec'),ns)
 monkeypatch.setitem(sys.modules,'vllm.config',SimpleNamespace(CUDAGraphMode=SimpleNamespace(NONE='NONE')))
 return ns['idle_target_only']

def test_idle_restores_drafter_on_success_and_failure(monkeypatch):
 call=helper(monkeypatch);draft=object();runner=SimpleNamespace(_pd_target_only_ready=True,drafter=draft)
 def native(r,*a,**kw):
  assert r.drafter is None and kw['cudagraph_runtime_mode']=='NONE' and kw['force_attention'] is False
  return a
 assert call(runner,native,3,uniform_decode=True)==(3,)
 assert runner.drafter is draft
 def broken(*a,**kw):raise ValueError('probe')
 with pytest.raises(ValueError,match='probe'):call(runner,broken,3)
 assert runner.drafter is draft
 with pytest.raises(RuntimeError):call(runner,native,3,force_attention=True)

def test_initial_capture_is_unchanged(monkeypatch):
 call=helper(monkeypatch);draft=object();r=SimpleNamespace(drafter=draft)
 def native(r,*a,**kw):
  assert r.drafter is draft and kw=={'is_graph_capturing':True};return 9
 assert call(r,native,3,is_graph_capturing=True)==9
