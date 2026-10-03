"""CPU checks for the disposable native-dummy metadata seam."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest

def helper():
    path=Path(__file__).with_name("dummy_decode_entry.py")
    fn=next(n for n in ast.parse(path.read_text()).body
            if isinstance(n,ast.FunctionDef) and n.name=="native_dummy")
    ns=dict(ast=ast,copy=copy,Path=Path)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),"exec"),ns)
    return ns["native_dummy"]

def test_keep_live_queries_separate_from_graph_request_padding(tmp_path):
    p=tmp_path/"runner.py"
    p.write_text("""class NPUModelRunner:
 def _dummy_run(self, num_reqs, batch_desc, num_scheduled_tokens):
  num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
  num_scheduled_tokens = num_scheduled_tokens.repeat(num_reqs_padded)
  return num_reqs_padded, num_scheduled_tokens
""")
    run=helper()(SimpleNamespace(__file__=str(p)))
    queries=[3]*28
    n,actual=run(None,28,SimpleNamespace(num_reqs=48),queries)
    assert n==28 and actual is queries and sum(actual)==84

def test_unknown_donor_does_not_silently_patch(tmp_path):
    p=tmp_path/"runner.py"
    p.write_text("class NPUModelRunner:\n def _dummy_run(self): return None\n")
    with pytest.raises(RuntimeError,match="Unknown pinned"):
        helper()(SimpleNamespace(__file__=str(p)))
