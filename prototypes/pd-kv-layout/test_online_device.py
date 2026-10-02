import sqlite3
from analyze_online_device import analyze


def test_dummy_owner_graph_envelopes_do_not_need_attention_anchors(tmp_path):
    p=tmp_path/"profile.db"
    with sqlite3.connect(p) as c:
        c.executescript("""
            create table STRING_IDS(id integer,value text);
            create table CANN_API(name integer,connectionId integer);
            create table TASK(startNs integer,endNs integer,deviceId integer,
                connectionId integer,globalTaskId integer,streamId integer);
            create table COMPUTE_TASK_INFO(globalTaskId integer,opType integer);
            insert into STRING_IDS values(1,'aclmdlRIExecuteAsync');
            insert into CANN_API values(1,10),(1,20);
            insert into TASK values
                (1000000,1100000,0,10,1,7),(1100000,2000000,0,10,2,7),
                (3000000,3100000,0,20,3,7),(3100000,4000000,0,20,4,7);
        """)
    result=analyze(p);graph=result["graph_device"]
    assert result["anchors"]==0 and result["replays"]==2
    assert graph["correlated_replays"]==2 and graph["device_records_per_replay"]==[2]
    assert graph["cycle_ms"]["p50"]==2 and graph["envelope_ms"]["p50"]==1
    assert graph["gap_ms"]["p50"]==1 and graph["cycles_over100ms"]==0
