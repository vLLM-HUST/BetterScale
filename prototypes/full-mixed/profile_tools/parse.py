"""Parse copied native profiles in fresh processes; never reuse rank singletons."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('root',type=Path,help='Window containing profile/rankN_*')
p.add_argument('--jobs',type=int,choices=(1,2,4),default=2)
p.add_argument('--devices', default='0,1,2,3,4,5,6,7', help='Physical device IDs in rank order')
a=p.parse_args();root=a.root.resolve()
devices=[int(x) for x in a.devices.split(',')]
assert len(devices)==len(set(devices)) and devices
code='from torch_npu.profiler.profiler import analyse; import sys; analyse(sys.argv[1],max_process_number=1,export_type="db")'


def parse(rank):
    paths=list((root/'profile').glob(f'rank{rank}_*'))
    assert len(paths)==1
    path=paths[0];output=path/'ASCEND_PROFILER_OUTPUT'
    if output.exists():
        saved=path/'REUSED_PARSER_OUTPUT'
        assert not saved.exists(), f'Preserve/resolve prior parse attempt: {saved}'
        output.rename(saved)
    with (root/f'parse-rank{rank}.log').open('w') as log:
        subprocess.run([sys.executable,'-c',code,str(path)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    files=list(output.glob('ascend_pytorch_profiler_*.db'))
    assert len(files)==1 and files[0].name==f'ascend_pytorch_profiler_{rank}.db',files
    with sqlite3.connect(files[0]) as c:
        assert c.execute('select * from RANK_DEVICE_MAP').fetchall()==[(rank,devices[rank])]
    print('parsed rank',rank,flush=True)


with ThreadPoolExecutor(max_workers=a.jobs) as pool:
    list(pool.map(parse,range(len(devices))))
