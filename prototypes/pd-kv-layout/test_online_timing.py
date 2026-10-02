import json
import queue
from types import SimpleNamespace
from online_timing import Recorder,OutputQueue

def test_output_observer_preserves_identity_order_and_separate_writers(tmp_path,monkeypatch):
    monkeypatch.setenv("BETTERSCALE_PD_TIMING_DIR",str(tmp_path))
    recorder=Recorder("core")
    q=OutputQueue(queue.Queue(),recorder)
    values=[(0,SimpleNamespace(outputs=[SimpleNamespace(request_id=f"req{i}",new_token_ids=[17])])) for i in range(3)]
    for value in values:q.put_nowait(value)
    assert [q.get() for _ in values]==values
    assert q.empty()
    recorder.file.flush();q.sender.file.flush()
    rows=[json.loads(l) for l in next(tmp_path.glob("core-*.jsonl")).read_text().splitlines()]
    sent=[json.loads(l) for l in next(tmp_path.glob("sender-*.jsonl")).read_text().splitlines()]
    assert [r["requests"] for r in rows]==[r["requests"] for r in sent]
    assert all(a["ns"]<=b["ns"] for a,b in zip(rows,sent))



def test_host_call_observer_is_idempotent_and_preserves_lazy_result(tmp_path,monkeypatch):
    from online_timing import observe_method
    import pytest
    monkeypatch.setenv("BETTERSCALE_PD_TIMING_DIR",str(tmp_path))
    future=object();calls=[]
    class Owner:
        def collective_rpc(self,method,*,non_block=False):
            calls.append((method,non_block))
            if method=="fail":raise ValueError("original error")
            return future
    owner=Owner()
    observe_method(owner,"collective_rpc","test-rpc","rpc")
    wrapped=owner.collective_rpc
    observe_method(owner,"collective_rpc","test-rpc","rpc")
    assert owner.collective_rpc is wrapped
    for _ in range(63):assert owner.collective_rpc("execute",non_block=True) is future
    with pytest.raises(ValueError,match="original error"):owner.collective_rpc("fail")
    rows=[json.loads(x) for x in next(tmp_path.glob("test-rpc-*.jsonl")).read_text().splitlines()]
    assert len(rows)==64 and len(calls)==64
    assert all(r["begin_ns"]<=r["ns"] for r in rows)
    assert rows[0]["method"]=="execute" and rows[0]["non_block"] is True
    assert rows[-1]["method"]=="fail"
