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
