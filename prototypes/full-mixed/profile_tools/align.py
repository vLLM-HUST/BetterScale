"""Use TraceLoom's fitter; matched collective endpoints are display-only candidates."""
from pathlib import Path
import sqlite3,collections,json,subprocess
import argparse
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--export',action='store_true');p.add_argument('--eager-markers',action='store_true');p.add_argument('--max-marker-count',type=int);p.add_argument('--max-holdout-us',type=float,default=50);a=p.parse_args()
root=a.root.resolve();clock=root/'analysis'/'clock';clock.mkdir(exist_ok=True);ranks=[]
tool=Path('/workspace/my-ascend-workspace/.tools/traceloom-37323af')
subprocess.run(['g++','-std=c++20','-O2','-I',str(tool/'native/include'),str(Path(__file__).with_name('fit.cpp')),
                str(tool/'native/src/analysis/clock_calibration.cpp'),'-o',str(clock/'fit')],check=True,timeout=60)
for rank in range(8):
 p=next((root/'profile').glob(f'rank{rank}_*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_{rank}.db'));c=sqlite3.connect(p)
 query='select n.value,t.value,g.value,o.count,o.startNs,o.endNs from COMMUNICATION_OP o join STRING_IDS n on n.id=o.opName join STRING_IDS t on t.id=o.opType join STRING_IDS g on g.id=o.groupName'
 if a.eager_markers:
  query += ' where exists (select 1 from COMMUNICATION_TASK_INFO i join TASK t on t.globalTaskId=i.globalTaskId where i.opId=o.opId) and not exists (select 1 from COMMUNICATION_TASK_INFO i join TASK t on t.globalTaskId=i.globalTaskId where i.opId=o.opId and t.modelId != 4294967295)'
 if a.max_marker_count is not None:
  assert a.max_marker_count>0
  query += (' and ' if a.eager_markers else ' where ')+f'o.count <= {a.max_marker_count}'
 rows=c.execute(query).fetchall();c.close()
 counts=collections.Counter(tuple(x[:4]) for x in rows)
 ranks.append({tuple(x[:4]):x[4:] for x in rows if counts[tuple(x[:4])]==1})
common=set.intersection(*(set(r) for r in ranks))
keys=sorted(common,key=lambda k:ranks[0][k][0]);stride=max(1,len(keys)//256);keys=keys[::stride][:256]
assert len(keys)>=20
models=[]
for rank in range(1,8):
 lines=[]
 for i,key in enumerate(keys):
  rs,re=ranks[0][key];ss,se=ranks[rank][key]
  lines.append(f'marker{i} {se} {re} {max(re-rs,se-ss,20000)}\n')
 text=''.join(lines);(clock/f'rank{rank}.tsv').write_text(text)
 r=subprocess.run([str(clock/'fit'),str(rank)],input=text,text=True,capture_output=True,check=True)
 models.append(json.loads(r.stdout))
(clock/'models.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in models))
(clock/'markers.json').write_text(json.dumps(dict(contract='unique identical provider opName/opType/group/count; no timestamp-nearest pairing; end-affine candidate display only',eager_markers=a.eager_markers,max_marker_count=a.max_marker_count,common_count=len(common),selected=[list(k) for k in keys]),indent=2))
print([(m['rank'],m['holdout_p95_ns'],m['drift_ppm']) for m in models])

assert all(m['holdout_p95_ns'] <= a.max_holdout_us*1000 for m in models), 'Clock marker residual exceeds display gate; inspect identities, do not filter by timestamp residual'

if a.export:
    output=root/'analysis/target-draft-tp8-end-aligned.json.gz'
    partial=root/'analysis/target-draft-tp8-end-aligned.partial.json.gz'
    args=[str(tool/'build/traceloom'),'export-perfetto',str(root/'analysis/rank0.db'),
          '--output',str(partial),
          '--distributed-clock-model',str(clock/'models.jsonl')]
    for rank in range(8):args+=['--distributed-rank',f'{rank}={root}/analysis/rank{rank}.db']
    with (root/'analysis/export.log').open('w') as f:
        subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
    partial.replace(output)
