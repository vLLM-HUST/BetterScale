"""Audit target graph arrivals in the retained DSV4/eager-draft profile.

Match FULL calls to graph occurrences by execution order, never nearest time.
The >1000-compute-task model selector is specific to these full DSV4 graphs;
assert the observed FULL count and one target replay API per selected call.
Clock transforms are the existing display-only candidates, not calibration.
"""
import argparse
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path
import sqlite3
from statistics import median

p=argparse.ArgumentParser();p.add_argument('window',type=Path)
a=p.parse_args();root=a.window.resolve();label=root.name
models={x['rank']:x for x in map(json.loads,(root/'analysis/clock/models.jsonl').read_text().splitlines())}
ranks=[];origin=None
for rank in range(8):
    db=next((root/'profile').glob(f'rank{rank}_*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_{rank}.db'))
    c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
    modes=json.loads((root/f'{label}-modes-rank{rank}.json').read_text())
    full=[i for i,m in enumerate(modes) if m['mode']=='FULL']
    # (start,end,stream,task,model,name,connection), using real compute tasks.
    tasks=c.execute('select t.startNs,t.endNs,t.streamId,t.taskId,t.modelId,n.value,t.connectionId from TASK t join COMPUTE_TASK_INFO i using(globalTaskId) join STRING_IDS n on n.id=i.name order by t.startNs').fetchall()
    count=Counter(x[4] for x in tasks if x[4]!=4294967295)
    target_models={k for k,v in count.items() if v>1000};first={}
    for x in tasks:
        if x[4] in target_models:first.setdefault(x[4],x[2:4])
    starts=[x for x in tasks if x[4] in target_models and x[2:4]==first[x[4]]]
    assert len(starts)==len(full),(rank,len(starts),len(full),target_models)
    functions=c.execute("select cast(a.startNs as integer),cast(a.endNs as integer),n.value from PYTORCH_API a join STRING_IDS n on n.id=a.name where n.value like 'strengthen::%' order by cast(a.startNs as integer)").fetchall()
    forward=[x for x in functions if x[2]=='strengthen::target_forward']
    draft=[x for x in functions if x[2]=='strengthen::draft_forward']
    assert len(forward)==len(modes)==len(draft)
    apis=c.execute("select a.startNs,a.endNs,n.value,a.connectionId from CANN_API a join STRING_IDS n on n.id=a.name where n.value in ('aclmdlRIExecuteAsync','aclrtSynchronizeEvent') order by a.startNs").fetchall()
    comm=c.execute("select o.startNs,o.endNs,n.value,ty.value,o.count,g.value,o.opId from COMMUNICATION_OP o join STRING_IDS n on n.id=o.opName join STRING_IDS ty on ty.id=o.opType join STRING_IDS g on g.id=o.groupName where ty.value like '%reduceScatter%' order by o.startNs").fetchall()
    comm_models={}
    for op,model in c.execute('select distinct i.opId,t.modelId from COMMUNICATION_TASK_INFO i join TASK t using(globalTaskId)'):
        comm_models.setdefault(op,set()).add(model)
    ended=sorted(tasks,key=lambda x:x[1]);ends=[x[1] for x in ended]
    if origin is None:origin=starts[0][0]
    def aligned(t):
        if rank==0:return (t-origin)/1e6
        m=models[rank]
        return ((t-m['reference_source_ns'])*m['scale']+(m['reference_target_ns']-origin))/1e6
    records=[]
    for ordinal,(wave,start) in enumerate(zip(full,starts)):
        limit=starts[ordinal+1][0] if ordinal+1<len(starts) else max(x[1] for x in tasks)+1
        body=[x for x in tasks if x[4]==start[4] and start[0]<=x[0]<limit]
        end=max(x[1] for x in body)
        rs=next(x for x in comm if start[0]<=x[0]<end and comm_models[x[6]]=={start[4]})
        calls=[x for x in apis if x[2]=='aclmdlRIExecuteAsync' and forward[wave][0]<=x[0]<forward[wave][1]]
        assert len(calls)==1,(rank,wave,calls)
        call=calls[0]
        previous=ended[max(0,bisect_left(ends,start[0])-8):bisect_left(ends,start[0])]
        sync=[x for x in apis if x[2]=='aclrtSynchronizeEvent' and forward[wave-1][0]<=x[0]<call[0]]
        records.append(dict(wave=wave,requests=modes[wave]['actual_requests'],tokens=modes[wave]['actual_tokens'],model=start[4],
            graph_start_ms=aligned(start[0]),graph_end_ms=aligned(end),replay_host_start_ms=aligned(call[0]),replay_host_end_ms=aligned(call[1]),
            replay_to_graph_ms=(start[0]-call[1])/1e6,
            first_rs=dict(name=rs[2],type=rs[3],count=rs[4],group=rs[5],start_ms=aligned(rs[0]),end_ms=aligned(rs[1])),
            prefix_us=(rs[0]-start[0])/1e3,
            previous_draft_host_ms=(draft[wave-1][1]-draft[wave-1][0])/1e6,
            previous_draft_host_end_ms=aligned(draft[wave-1][1]),
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
result=dict(contract=__doc__,window=str(root),waves=waves)
(root/'analysis/continuation.json').write_text(json.dumps(result,indent=2))
for w in waves:
    print({k:round(v,3) if isinstance(v,float) else v for k,v in w.items() if k!='ranks'})
