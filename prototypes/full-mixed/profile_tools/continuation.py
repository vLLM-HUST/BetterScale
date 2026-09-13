"""Audit target graph arrivals in the retained single-compute-stream profile.

Link each FULL CPU call's replay API to its native MODEL_EXECUTE task by exact
connectionId. The next MODEL_EXECUTE on that stream bounds the graph envelope;
require one captured compute model within it. No size heuristic, timestamp-
nearest matching or assumption that draft runs eager is used.
Clock transforms are the existing display-only candidates, not calibration.
"""
import argparse
from bisect import bisect_left
import json
from pathlib import Path
import sqlite3
from statistics import median

p=argparse.ArgumentParser();p.add_argument('window',type=Path)
p.add_argument('--profile-prefix', action='store_true', help='Profile began at window start and ended early; match complete leading forward/draft pairs only')
a=p.parse_args();root=a.window.resolve();label=root.name
models={x['rank']:x for x in map(json.loads,(root/'analysis/clock/models.jsonl').read_text().splitlines())}
ranks=[];origin=None
for rank in range(8):
    db=next((root/'profile').glob(f'rank{rank}_*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_{rank}.db'))
    c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
    modes=json.loads((root/f'{label}-modes-rank{rank}.json').read_text())
    # (start,end,stream,task,model,name,connection), using real compute tasks.
    tasks=c.execute('select t.startNs,t.endNs,t.streamId,t.taskId,t.modelId,n.value,t.connectionId from TASK t join COMPUTE_TASK_INFO i using(globalTaskId) join STRING_IDS n on n.id=i.name order by t.startNs').fetchall()
    functions=c.execute("select cast(a.startNs as integer),cast(a.endNs as integer),n.value from PYTORCH_API a join STRING_IDS n on n.id=a.name where n.value like 'strengthen::%' order by cast(a.startNs as integer)").fetchall()
    forward=[x for x in functions if x[2]=='strengthen::target_forward']
    draft=[x for x in functions if x[2]=='strengthen::draft_forward']
    if a.profile_prefix:
        # diagnostics schedule(wait=0,warmup=0,active=N) stops before the full
        # cohort drains. Only use complete leading pairs; no timestamp matching
        # and no selection of faster interior waves.
        assert len(forward) == len(draft) <= len(modes)
        modes = modes[:len(forward)]
    assert len(forward)==len(modes)==len(draft)
    full=[i for i,m in enumerate(modes) if m['mode']=='FULL']
    apis=c.execute("select a.startNs,a.endNs,n.value,a.connectionId from CANN_API a join STRING_IDS n on n.id=a.name where n.value in ('aclmdlRIExecuteAsync','aclrtSynchronizeEvent') order by a.startNs").fetchall()
    launches=c.execute("select t.startNs,t.connectionId,t.streamId from TASK t join STRING_IDS n on n.id=t.taskType where n.value='MODEL_EXECUTE' order by t.startNs").fetchall()
    occurrences=[]
    for wave in full:
        calls=[x for x in apis if x[2]=='aclmdlRIExecuteAsync' and forward[wave][0]<=x[0]<forward[wave][1]]
        assert len(calls)==1,(rank,wave,calls)
        call=calls[0]
        linked=[x for x in launches if x[1]==call[3]]
        assert len(linked)==1,(rank,wave,call,linked)
        launch=linked[0]
        limit=next((x[0] for x in launches if x[2]==launch[2] and x[0]>launch[0]),max(x[1] for x in tasks)+1)
        body=[x for x in tasks if x[4]!=4294967295 and launch[0]<=x[0]<limit]
        assert body and len({x[4] for x in body})==1,(rank,wave,{x[4] for x in body})
        assert max(x[1] for x in body)<=limit,(rank,wave,'overlapping graph execution')
        occurrences.append((wave,body,call,launch))
    comm=c.execute("select o.startNs,o.endNs,n.value,ty.value,o.count,g.value,o.opId from COMMUNICATION_OP o join STRING_IDS n on n.id=o.opName join STRING_IDS ty on ty.id=o.opType join STRING_IDS g on g.id=o.groupName where ty.value like '%reduceScatter%' order by o.startNs").fetchall()
    comm_models={}
    for op,model in c.execute('select distinct i.opId,t.modelId from COMMUNICATION_TASK_INFO i join TASK t using(globalTaskId)'):
        comm_models.setdefault(op,set()).add(model)
    ended=sorted(tasks,key=lambda x:x[1]);ends=[x[1] for x in ended]
    if origin is None:origin=occurrences[0][1][0][0]
    def aligned(t):
        if rank==0:return (t-origin)/1e6
        m=models[rank]
        return ((t-m['reference_source_ns'])*m['scale']+(m['reference_target_ns']-origin))/1e6
    records=[]
    for wave,body,call,launch in occurrences:
        start=body[0]
        end=max(x[1] for x in body)
        rs=next(x for x in comm if start[0]<=x[0]<end and comm_models[x[6]]=={start[4]})
        previous=ended[max(0,bisect_left(ends,start[0])-8):bisect_left(ends,start[0])]
        sync=[x for x in apis if x[2]=='aclrtSynchronizeEvent' and forward[wave-1][0]<=x[0]<call[0]] if wave else []
        records.append(dict(wave=wave,requests=modes[wave]['actual_requests'],tokens=modes[wave]['actual_tokens'],model=start[4],
            launch_connection=launch[1],launch_stream=launch[2],
            graph_start_ms=aligned(start[0]),graph_end_ms=aligned(end),replay_host_start_ms=aligned(call[0]),replay_host_end_ms=aligned(call[1]),
            replay_to_graph_ms=(start[0]-call[1])/1e6,
            first_rs=dict(name=rs[2],type=rs[3],count=rs[4],group=rs[5],start_ms=aligned(rs[0]),end_ms=aligned(rs[1])),
            prefix_us=(rs[0]-start[0])/1e3,
            previous_draft_host_ms=(draft[wave-1][1]-draft[wave-1][0])/1e6 if wave else None,
            previous_draft_host_end_ms=aligned(draft[wave-1][1]) if wave else None,
            previous_compute=[dict(name=x[5],model=x[4],stream=x[2],start_ms=aligned(x[0]),end_ms=aligned(x[1])) for x in previous],
            host_sync=[dict(start_ms=aligned(x[0]),duration_ms=(x[1]-x[0])/1e6) for x in sync]))
    ranks.append(records);c.close()
assert all([x['wave'] for x in r]==[x['wave'] for x in ranks[0]] for r in ranks)
waves=[]
for records in zip(*ranks):
    # Semantic identity, not endpoint proximity, must agree across all ranks.
    matched=len({(x['first_rs']['name'],x['first_rs']['type'],x['first_rs']['count'],x['first_rs']['group']) for x in records})==1
    if not matched:
        # Some turnover captures have different native name ordinals. Preserve
        # them without inventing cross-rank correspondence from near timestamps.
        waves.append(dict(wave=records[0]['wave'],tokens=records[0]['tokens'],matched_provider_identity=False,ranks=list(records)))
        continue
    arrival=[x['first_rs']['start_ms'] for x in records]
    host=[x['replay_host_end_ms'] for x in records]
    late=max(range(8),key=lambda i:arrival[i]);early=min(range(8),key=lambda i:arrival[i])
    waves.append(dict(wave=records[0]['wave'],tokens=records[0]['tokens'],requests=records[0]['requests'],
        first_rs=records[0]['first_rs']['name'],last_arrival_rank=late,first_arrival_rank=early,
        rs_arrival_spread_ms=max(arrival)-min(arrival),host_replay_spread_ms=max(host)-min(host),
        first_arriver_rs_duration_ms=records[early]['first_rs']['end_ms']-arrival[early],
        last_arriver_rs_duration_ms=records[late]['first_rs']['end_ms']-arrival[late],
        median_graph_span_ms=median(x['graph_end_ms']-x['graph_start_ms'] for x in records),ranks=list(records)))
result=dict(contract=__doc__,window=str(root),profile_prefix=a.profile_prefix,waves=waves)
(root/'analysis/continuation.json').write_text(json.dumps(result,indent=2))
for w in waves:
    print({k:round(v,3) if isinstance(v,float) else v for k,v in w.items() if k!='ranks'})
