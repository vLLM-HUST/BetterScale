import json
from analyze_online_supply import analyze


def test_worker_window_excludes_gate_and_drain_but_control_cohort_is_explicit(tmp_path):
    rows=[dict(stage="worker-dispatch",begin_ns=t*10**9,
               scheduled={str(i):1 for i in range(n)},drafts={})
          for t,n in ((1,99),(2,7),(3,8),(4,99))]
    (tmp_path/"worker-1.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    control=tmp_path/"control.jsonl"
    values=[dict(op="ingress-admitted",start=1,end=2,session_commit_wait=.2,permit_wait=.8),
            dict(op="host-cache-admitted",session="s",start=1,end=4,used={"P0":20}),
            dict(op="host-cache-evicted",start=3,end=5)]
    control.write_text("\n".join(json.dumps(x) for x in values)+"\n{partial")
    result=analyze(tmp_path,control,start=2,end=4)
    worker=result["workers"]["worker-1"]
    assert worker["rows"]["mean"]==7.5
    assert worker["host_dispatch_interval_ms"]["p50"]==1000
    assert result["request_phase_seconds"]["host-cache-admitted"]["max"]==3
    assert result["host_reservation_peak_bytes"]=={"P0":20}
    assert result["control_counts"]["host-cache-evicted"]==1
    assert "NOT device" in result["scope"]


def test_live_tail_only_is_ignored_and_metadata_is_not_a_worker(tmp_path):
    from analyze_online_supply import records
    import pytest
    path=tmp_path/"worker-1.jsonl"
    path.write_text('{"stage":"other"}\n{partial')
    (tmp_path/"worker-metadata-1.jsonl").write_text('not a dispatch log\n')
    assert list(analyze(tmp_path,None)["workers"]) == ["worker-1"]
    assert list(records(path)) == [{"stage":"other"}]
    path.write_text('{broken}\n')
    with pytest.raises(json.JSONDecodeError):list(records(path))


def test_sender_window_counts_deltas_and_p_wait_excludes_host_admission(tmp_path):
    control=tmp_path/"control.jsonl"
    rows=[dict(op="host-cache-admitted",session="s",start=0,end=4,used={}),
          dict(op="owner-admitted",session="s",owner=2,p_instance=1,start=0,end=5),
          dict(op="turn-committed",events=[dict(stage="D",arrivals=[
              [2,10**9],[5,2*10**9],[8,3*10**9],[11,4*10**9]])])]
    control.write_text("\n".join(map(json.dumps,rows)))
    x=analyze(tmp_path,control,start=2,end=4)
    assert x["p_permit_wait_seconds_by_instance"][1]["p50"]==1
    assert x["sender_window"]["accepted_tokens"]==6
    assert x["sender_window"]["tokens_per_second"]==3
    assert x["sender_window"]["accepted_per_positive_chunk"]==3
