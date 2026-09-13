"""Compare equal-work four-seat K5 verification intervals, not cohort luck."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('engine',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--prefix',default='phase');a=p.parse_args()
result=json.loads((a.engine/'result.json').read_text());phases=[]
for index,cohort in enumerate(result['results']):
 ranks=[]
 for rank in range(8):
  path=a.engine/f'{a.prefix}{index}-timing-rank{rank}.json'
  if not path.exists():continue
  es=json.loads(path.read_text());ws=json.loads((a.engine/f'{a.prefix}{index}-waves-rank{rank}.json').read_text())
  positive=[x for x in ws if x['total'] and x['total']>0]
  target=[x for x in es if x['label']=='strengthen::target_forward']
  draft=[x for x in es if x['label']=='strengthen::draft_forward']
  assert len(positive)==len(target),(index,rank,len(positive),len(target))
  def full_verify(w):return len(w['scheduled'])==4 and all(x==6 for x in w['scheduled'].values())
  selected=[]
  for i,(t,nxt) in enumerate(zip(target,target[1:])):
   if not (full_verify(positive[i]) and full_verify(positive[i+1])):continue
   ds=[d for d in draft if t['device_start_ms']<=d['device_start_ms']<nxt['device_start_ms']]
   assert len(ds)==1,(index,rank,i,len(ds))
   d=ds[0]
   selected.append(dict(cycle_ms=nxt['device_start_ms']-t['device_start_ms'],target_ms=t['device_elapsed_ms'],draft_ms=d['device_elapsed_ms'],
    target_to_draft_ms=d['device_start_ms']-t['device_start_ms']-t['device_elapsed_ms'],draft_to_target_ms=nxt['device_start_ms']-d['device_start_ms']-d['device_elapsed_ms'],draft_host_ms=d['host_elapsed_ms']))
  stats={key:statistics.median(x[key] for x in selected) for key in selected[0]} if selected else {}
  ranks.append(dict(rank=rank,selected_intervals=len(selected),medians=stats))
 phases.append(dict(phase=index,policy=cohort.get('policy'),cohort_seconds=cohort['elapsed'],generated_tokens=sum(map(len,cohort['outputs'])),ranks=ranks))
summary=dict(engine=str(a.engine),selection='Consecutive positive waves with exactly four requests, each scheduling six target query tokens; median event intervals per rank.',nonclaims='Inter-forward intervals include metadata/copy/sampling work; not all idle. Cohort throughput is affected by speculative acceptance. Device event durations are spans, not kernel sums.',phases=phases)
a.output.write_text(json.dumps(summary,indent=2)+'\n')
for phase in phases:
 print(phase['phase'],phase['cohort_seconds'],phase['ranks'][0] if phase['ranks'] else None)
