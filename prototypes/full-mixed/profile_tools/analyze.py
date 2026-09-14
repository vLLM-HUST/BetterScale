"""Analyze eight rank-bound official profiler DBs with the frozen TraceLoom tool."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,json,sqlite3,hashlib
import argparse
p=argparse.ArgumentParser();p.add_argument('root',type=Path)
p.add_argument('--jobs',type=int,choices=range(1,9),default=1)
p.add_argument('--resume',action='store_true')
p.add_argument('--ranks',type=int,choices=range(2,9),default=8)
a=p.parse_args();root=a.root.resolve();(root/'analysis').mkdir(exist_ok=True)
exe=Path('/workspace/my-ascend-workspace/.tools/traceloom-37323af/build/traceloom')
manifest=[]
for rank in range(a.ranks):
 paths=list((root/'profile').glob(f'rank{rank}_*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_{rank}.db'))
 assert len(paths)==1
 source=paths[0];c=sqlite3.connect(source)
 assert c.execute('pragma integrity_check').fetchone()[0]=='ok'
 manifest.append(dict(rank=rank,source=str(source.resolve()),bytes=source.stat().st_size,
                      rank_device_map=c.execute('select * from RANK_DEVICE_MAP').fetchall()))
 c.close()

def index_export_edges(output):
 # Frozen37323af's recursive tree view has a correlated root anti-join.
 # Without this index the exporter scans all edges for every node (quadratic).
 # Change only the derived queryable DB; raw profiles and rows stay untouched.
 c=sqlite3.connect(output)
 c.execute('create index if not exists strengthen_viz_edge_child on traceloom_viz_edge(child_node_id)')
 c.commit();c.close()

def analyze(item):
 rank=item['rank'];source=Path(item['source']);output=root/'analysis'/f'rank{rank}.db'
 if a.resume and output.exists():
  # TraceLoom publishes the final filename atomically only after materialization.
  # Reject an unrelated artifact; .tmp outputs are never resumable rank DBs.
  c=sqlite3.connect(f'file:{output}?mode=ro',uri=True)
  metadata=dict(c.execute('select key,value from traceloom_metadata'))
  assert metadata['source_path']==str(source)
  assert metadata['source_sha256']==hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
  assert c.execute('pragma quick_check').fetchone()[0]=='ok'
  c.close();index_export_edges(output);print('resumed',rank,flush=True);return
 with (root/'analysis'/f'rank{rank}.log').open('w') as f:
  subprocess.run([str(exe),str(source),'--threads','4','--output',str(output)],
                 stdout=f,stderr=subprocess.STDOUT,timeout=300,check=True)
 index_export_edges(output)
 print('analyzed',rank,flush=True)

with ThreadPoolExecutor(max_workers=a.jobs) as pool:
 list(pool.map(analyze,manifest))
(root/'analysis'/'sources.json').write_text(json.dumps(dict(
 analyzer=(exe.parent.parent/'SOURCE_COMMIT').read_text().strip(),
 archive_sha256=hashlib.file_digest((root/'profile-sqlite.tar.gz').open('rb'),'sha256').hexdigest(),
 sources=manifest),indent=2))
