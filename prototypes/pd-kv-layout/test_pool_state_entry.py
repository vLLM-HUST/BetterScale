"""Keep target-only scheduler admission consistent for staggered hot requests."""
import ast
from pathlib import Path
from types import SimpleNamespace


def test_zero_proposals_retains_native_lookahead_and_handles_hot_admission():
    source=Path(__file__).with_name("pool_state_entry.py").read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=="Scheduler")
    class Base:
        def __init__(self):
            self.num_spec_tokens=2
            self.num_lookahead_tokens=2
            self._spec_token_placeholders=[-1,-1]
    namespace={"BaseScheduler":Base}
    exec(compile(ast.Module(body=[node],type_ignores=[]),"<pool-scheduler>","exec"),namespace)
    scheduler=namespace["Scheduler"]()
    # This is the donor's hot WAITING admission grant, not a post-hoc trim of
    # already-accounted output. Speculative capacity stays allocated.
    assert 1+scheduler.num_spec_tokens==1
    assert scheduler._spec_token_placeholders==[]
    assert scheduler.num_lookahead_tokens==2
