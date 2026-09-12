"""Read native rank DBs; bound device gaps with compute + communication coverage.
API timestamps use the provider's local host/device time mapping, not cross-rank fits.
The launch API interval bounds submission; it does not expose device-ready time.
"""
import argparse, glob, sqlite3, json
p=argparse.ArgumentParser();p.add_argument('profile');p.add_argument('output');a=p.parse_args()
result=[]
for rank in range(8):
 db=glob.glob(f'{a.profile}/rank-{rank}_*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_{rank}.db');assert len(db)==1
 c=sqlite3.connect(f'file:{db[0]}?mode=ro',uri=True)
 rows=c.execute('''select t.startNs,t.endNs,s.value,t.streamId,a.startNs,a.endNs,t.connectionId from TASK t join COMPUTE_TASK_INFO i using(globalTaskId) join STRING_IDS s on s.id=i.name left join CANN_API a on a.connectionId=t.connectionId order by t.startNs''').fetchall()
 comm=c.execute('select o.startNs,o.endNs,s.value from COMMUNICATION_OP o join STRING_IDS s on s.id=o.opName').fetchall()
 intervals=sorted([(x[0],x[1]) for x in rows]+[(x[0],x[1]) for x in comm]);merged=[]
 for lo,hi in intervals:
  if merged and lo<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],hi)
  else:merged.append([lo,hi])
 gaps=[(x[1],y[0]) for x,y in zip(merged,merged[1:]) if y[0]-x[1]>=50000]
 lookup={x[0]:x for x in rows};details=[]
 for lo,hi in gaps:
  nxt=lookup.get(hi)
  if nxt:
   _,end,name,stream,hs,he,conn=nxt
   kind='host_started_after_gap_start' if hs>=lo else ('launch_overlaps_gap_start' if he>lo else 'submitted_before_gap')
   details.append(dict(start_ms=(lo-rows[0][0])/1e6,gap_us=(hi-lo)/1e3,next=name,stream=stream,launch_start_after_gap_us=(hs-lo)/1e3,launch_end_before_device_us=(hi-he)/1e3,kind=kind,connection=conn,absolute_gap_start_ns=lo))
 lags=sorted((x[0]-x[5])/1e3 for x in rows if x[5] is not None)
 wait=c.execute("select count(*),sum(a.endNs-a.startNs)/1e6,max(a.endNs-a.startNs)/1e6,sum(a.endNs-a.startNs>1000000) from CANN_API a join STRING_IDS s on s.id=a.name where s.value='aclrtStreamWaitEvent'").fetchone()
 summary={k:dict(count=sum(d['kind']==k for d in details),gap_ms=sum(d['gap_us'] for d in details if d['kind']==k)/1000) for k in ['host_started_after_gap_start','launch_overlaps_gap_start','submitted_before_gap']}
 result.append(dict(rank=rank,launch_to_task_us_p50=lags[len(lags)//2],launch_to_task_us_p90=lags[int(len(lags)*.9)],wait_event=dict(count=wait[0],total_ms=wait[1],max_ms=wait[2],over_1ms=wait[3]),provider=db[0],task_count=len(rows),span_ms=(merged[-1][1]-merged[0][0])/1e6,coverage_ms=sum(hi-lo for lo,hi in merged)/1e6,gaps_over_50us=len(gaps),gaps_ms=sum(hi-lo for lo,hi in gaps)/1e6,next_compute_gaps=summary,largest_gaps=sorted(details,key=lambda d:-d['gap_us'])[:15]))
 print(json.dumps({k:v for k,v in result[-1].items() if k not in ['provider','largest_gaps']}))
open(a.output,'w').write(json.dumps(result,indent=2)+'\n')
