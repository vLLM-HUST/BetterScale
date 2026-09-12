"""Analyze eight rank-bound official profiler DBs with the frozen TraceLoom tool."""
from pathlib import Path
import subprocess,json,sqlite3,hashlib
import argparse
p=argparse.ArgumentParser();p.add_argument("root",type=Path);a=p.parse_args()
root=a.root.resolve();(root/"analysis").mkdir(exist_ok=True)
exe=Path('/workspace/my-ascend-workspace/.tools/traceloom-37323af/build/traceloom')
manifest=[]
for rank in range(8):
 paths=list((root/'profile').glob(f'rank{rank}_*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_{rank}.db'))
 assert len(paths)==1
 p=paths[0];c=sqlite3.connect(p)
 assert c.execute('pragma integrity_check').fetchone()[0]=='ok'
 manifest.append(dict(rank=rank,source=str(p.resolve()),bytes=p.stat().st_size,rank_device_map=c.execute('select * from RANK_DEVICE_MAP').fetchall()))
 c.close()
 with (root/'analysis'/f'rank{rank}.log').open('w') as f:
  subprocess.run([str(exe),str(p),'--threads','4','--output',str(root/'analysis'/f'rank{rank}.db')],stdout=f,stderr=subprocess.STDOUT,timeout=180,check=True)
 print('analyzed',rank,flush=True)
(root/'analysis'/'sources.json').write_text(json.dumps(dict(analyzer=(exe.parent.parent/'SOURCE_COMMIT').read_text().strip(),archive_sha256=hashlib.file_digest((root/'profile-sqlite.tar.gz').open('rb'),'sha256').hexdigest(),sources=manifest),indent=2))
