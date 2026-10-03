import asyncio
from types import SimpleNamespace
import pytest
import online_node


@pytest.mark.parametrize("constructor_failure",[False,True])
def test_partial_node_startup_closes_all_created_actors(tmp_path,monkeypatch,constructor_failure):
    created=[]
    class Actor:
        def __init__(self,kind,index):
            if constructor_failure and index==2:raise RuntimeError("spawn failure")
            self.index=index;self.quarantined=False;self.closed=False
            created.append(self)
        async def ready(self):
            if self.index==2:raise RuntimeError("allocation207001")
            await asyncio.Event().wait()
        async def close(self):
            assert self.quarantined
            self.closed=True
    monkeypatch.setattr(online_node,"Actor",Actor)
    node=online_node.Node.__new__(online_node.Node)
    node.args=SimpleNamespace(kind="P",output=tmp_path);node.actors=[];node.ready=False
    with pytest.raises(RuntimeError,match="spawn failure" if constructor_failure else "allocation207001"):
        asyncio.run(asyncio.wait_for(node.start(None),1))
    assert len(created)==(2 if constructor_failure else 4)
    assert all(a.closed for a in created)
    assert not node.ready and not (tmp_path/"ready.json").exists()
