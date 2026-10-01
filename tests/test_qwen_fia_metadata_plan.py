"""CPU contract: owned attention borrows metadata, never native launch identity."""
import ast
import ctypes
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest


def fixture(owned=True):
    path=Path(__file__).resolve().parents[1]/'src/betterscale/patches/qwen_fia/wave.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Planner')
    tensor=SimpleNamespace(data_ptr=lambda:123,device='npu',shape=(64,))
    cp=SimpleNamespace(prepare=Mock())
    scope=dict(ctypes=ctypes,cp=cp,torch=SimpleNamespace(empty=lambda *a,**k:tensor,uint8='uint8',npu=SimpleNamespace(current_stream=lambda:SimpleNamespace(npu_stream=0))))
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),scope)
    lib=Mock()
    for name,value in dict(plan_native_queries=42,plan_is_fd=1,plan_blocks=9,plan_metadata=2528,plan_discard_latest=0,plan_release=0).items():
        getattr(lib,name).return_value=value
    planner=scope['Planner'](lib,heads=8,kvheads=1)
    planner.fixtures=(tensor,tensor,tensor,tensor,.0625)
    frame=SimpleNamespace(tokens=3,requests=17,columns=8,context_parallel=owned,h_tiling=tensor,table=tensor,plan=None)
    meta=SimpleNamespace(actual_seq_lengths_q=[3],seq_lens_list=[4096],attn_mask=tensor)
    return planner,frame,meta,lib,cp


def test_owned_plan_does_not_adopt_native_fd_function_or_grid():
    planner,frame,meta,lib,cp=fixture()
    planner.native(frame,meta)
    cp.prepare.assert_called_once_with(frame,meta)
    lib.plan_discard_latest.assert_called_once_with(42)
    lib.plan_replace.assert_not_called()
    lib.plan_is_fd.assert_not_called()
    assert frame.plan is None and planner.calls==1


def test_owned_metadata_validation_failure_still_retires_temporary():
    planner,frame,meta,lib,cp=fixture()
    cp.prepare.side_effect=ValueError('wrong geometry')
    with pytest.raises(ValueError,match='wrong geometry'):planner.native(frame,meta)
    lib.plan_discard_latest.assert_called_once_with(42)
    assert planner.calls==0


def test_native_launch_keeps_strict_variant_guard():
    planner,frame,meta,lib,cp=fixture(False)
    with pytest.raises(RuntimeError,match='fd=1 blocks=9'):planner.native(frame,meta)
    cp.prepare.assert_not_called()
    lib.plan_release.assert_called_once_with(42)
    lib.plan_discard_latest.assert_not_called()
