"""Scheduler and allocator share the resident count without widening execution."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest

def function():
    path=Path(__file__).parents[1]/"src/betterscale/models/qwen35/state_backend.py"
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="resident_seats")
    namespace={'EXECUTION':16}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),namespace)
    return namespace["resident_seats"]

def test_resident_configuration():
    seats=function()
    assert seats(SimpleNamespace(additional_config={}))==20
    for value in (20,40,80):
        assert seats(SimpleNamespace(additional_config={"state_resident_seats":value}))==value
    for value in (True,19,97,40.0,"40"):
        with pytest.raises(ValueError):
            seats(SimpleNamespace(additional_config={"state_resident_seats":value}))
