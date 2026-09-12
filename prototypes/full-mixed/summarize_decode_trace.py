"""Bounded native trace summary; host scopes are not device idle-time claims."""
import argparse,collections,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=json.loads(a.trace.read_text());events=d.get('traceEvents',d) if isinstance(d,dict) else d
scopes=collections.defaultdict(list);cats=collections.Counter(); names=collections.Counter()
for e in events:
 if e.get('ph')!='X':continue
 name=e.get('name','');cat=e.get('cat','');cats[cat]+=1
 if name.startswith('strengthen::') or 'aclgraph' in name.lower() or 'aclmdl' in name.lower() or 'synchronize' in name.lower():
  scopes[name].append(e)
 if 'kernel' in cat.lower():names[name]+=1
out=dict(trace=str(a.trace),categories=dict(cats),scopes={})
for name,es in scopes.items():
 durations=[float(e.get('dur',0)) for e in es]
 out['scopes'][name]=dict(count=len(es),total_ms=sum(durations)/1000,median_ms=statistics.median(durations)/1000,max_ms=max(durations)/1000)
out['host_step_rows']=[dict(ts=e['ts'],dur=e['dur']) for e in scopes.get('strengthen::target_step',[])]
hardware={e['pid'] for e in events if e.get('ph')=='M' and e.get('args',{}).get('name')=='Ascend Hardware'}
tasks=[e for e in events if e.get('ph')=='X' and e.get('pid') in hardware and e.get('args',{}).get('Task Type') in ('AI_CORE','AI_VECTOR_CORE','AI_CPU','MIX_AIV','MIX_AIC','COMMUNICATION')]
all_hw=[e for e in events if e.get('ph')=='X' and e.get('pid') in hardware]
out['hardware_types']=dict(collections.Counter(e.get('args',{}).get('Task Type') for e in all_hw))
out['hardware_models']=dict(collections.Counter(str(e.get('args',{}).get('Model Id')) for e in all_hw))
intervals=sorted((float(e['ts']),float(e['ts'])+float(e.get('dur',0))) for e in tasks)
merged=[]
for lo,hi in intervals:
 if merged and lo<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],hi)
 else:merged.append([lo,hi])
gaps=[(b[0]-a[1])/1000 for a,b in zip(merged,merged[1:])]
out['compute_communication_coverage']=dict(task_count=len(tasks),span_ms=(merged[-1][1]-merged[0][0])/1000 if merged else 0,covered_ms=sum(hi-lo for lo,hi in merged)/1000,gaps_over_50us_ms=sum(x for x in gaps if x>=.05),largest_gaps_ms=sorted(gaps,reverse=True)[:10],nonclaim='Does not include every engine, is not removable-speedup estimate.')
markers=sorted(float(e['ts']) for e in all_hw if e.get('args',{}).get('Task Type')=='MODEL_EXECUTE')
steps=[]
for lo,hi in zip(markers,markers[1:]):
 active=[e for e in tasks if lo<=float(e['ts'])<hi]
 graph=[e for e in active if str(e.get('args',{}).get('Model Id'))!='4294967295']
 if graph:
  start=min(float(e['ts']) for e in graph);end=max(float(e['ts'])+float(e.get('dur',0)) for e in graph)
  steps.append(dict(interval_ms=(hi-lo)/1000,target_graph_span_ms=(end-start)/1000,after_target_to_next_replay_ms=(hi-end)/1000,non_graph_tasks=sum(str(e.get('args',{}).get('Model Id'))=='4294967295' for e in active)))
out['native_replay_intervals']=steps
out['replay_interval_summary']={k:statistics.median(x[k] for x in steps) for k in steps[0]} if steps else {}
a.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(categories=out['categories'],scopes=out['scopes'],hardware_types=out['hardware_types'],hardware_models=out['hardware_models'],coverage=out['compute_communication_coverage'],replay=out['replay_interval_summary']),indent=2))
