"""Opt-in execution envelope, distinct from head/tile geometry and residency."""
import os
from pathlib import Path
import subprocess
import sys
import pytest

@pytest.mark.parametrize("rows",(16,32,48,64,80))
def test_wider_decode_capacity_contract(rows):
    repo=Path(__file__).parents[1]
    env=dict(os.environ,TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        BETTERSCALE_PD_DECODE_ONLY="1",BETTERSCALE_QWEN35_DECODE_CAPACITY=str(rows),
        PYTHONPATH=f"{repo}/src:{repo}/prototypes/pd-kv-layout")
    program=r"""
import ast,copy,os
from pathlib import Path
from types import SimpleNamespace
from betterscale.models.qwen35.execution_capacity import EXECUTION
from betterscale.models.qwen35.count_policy import SPEC_CAPACITIES,capture_lengths
from betterscale.patches.qwen_fia.context_parallel.plan import schedule
from decode_graph_policy import capacity,verification_drafts
n=int(os.environ["BETTERSCALE_QWEN35_DECODE_CAPACITY"])
assert EXECUTION==n and max(SPEC_CAPACITIES)==3*n
assert capacity(3*n,n,[3]*n,[4096]*n,[4097]*n)==3*n
assert verification_drafts([3]*n,[4096]*n,[4097]*n,[-1]*n)==[0]*n
assert capture_lengths([7]*n)==(3,)*n
for lengths in ([4096]*n,[512+i*3072 for i in range(n)]):
    plan=schedule(lengths,[3]*n)
    covered={i:[] for i in range(n)}
    for group in plan["groups"]:
        for row,lo,hi in group:covered[row].append((lo,hi))
    for row,ranges in covered.items():
        ranges.sort()
        assert ranges[0][0]==0 and ranges[-1][1]==(lengths[row]+511)//512
        assert all(a[1]==b[0] for a,b in zip(ranges,ranges[1:]))
    assert len(plan["groups"])<=24 and plan["partials"]<=46
root=Path("src/betterscale/models/qwen35")
for filename,function in (("draft_fia.py","compact_padding"),("state_backend.py","resident_seats")):
    tree=ast.parse((root/filename).read_text())
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==function)
    scope={"EXECUTION":n,"copy":copy}
    exec(compile(ast.Module(body=[node],type_ignores=[]),filename,"exec"),scope)
    if function=="resident_seats":
        assert scope[function](SimpleNamespace(additional_config={}))==max(20,n+4)
    else:
        m=SimpleNamespace(seq_lens_list=[4096]*n+[0]*5,
            actual_seq_lengths_q=list(range(1,n+6)),seq_lens=list(range(n+5)),
            _mtp_device_seq_lens=list(range(n+5)))
        out=scope[function](m)
        assert out.seq_lens_list==[4096]*n+[0]
        assert out.actual_seq_lengths_q==list(range(1,n+1))+[n+5]
"""
    subprocess.run([sys.executable,"-c",program],cwd=repo,env=env,check=True,
                   capture_output=True,text=True,timeout=30)
