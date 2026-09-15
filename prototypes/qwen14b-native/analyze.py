"""Offline native parsing and single-rank TraceLoom export; no clock fitting."""
import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
a = p.parse_args()
tool = Path('/workspace/my-ascend-workspace/.tools/traceloom-37323af/build/traceloom')
for label in ('c1-4k', 'c8-512'):
    window = a.root.resolve() / label
    assert (window / 'complete.json').exists()
    paths = list((window / 'profile').glob('rank0_*'))
    assert len(paths) == 1, paths
    native = paths[0]
    with (window / 'parse.log').open('w') as log:
        subprocess.run([sys.executable, '-c',
            'from torch_npu.profiler.profiler import analyse; import sys; '
            'analyse(sys.argv[1],max_process_number=1,export_type="db")', str(native)],
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
    dbs = list((native / 'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch_profiler_*.db'))
    assert len(dbs) == 1, dbs
    with sqlite3.connect(dbs[0]) as c:
        assert c.execute('pragma quick_check').fetchone()[0] == 'ok'
        mapping = c.execute('select * from RANK_DEVICE_MAP').fetchall()
        assert len(mapping) == 1 and mapping[0][0] == 0, mapping
    out = window / 'analysis'
    out.mkdir()
    with (out / 'analyze.log').open('w') as log:
        subprocess.run([str(tool), str(dbs[0]), '--threads', '4', '--output', str(out/'rank0.db')],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    with sqlite3.connect(out/'rank0.db') as c:
        c.execute('create index if not exists strengthen_viz_edge_child on traceloom_viz_edge(child_node_id)')
    partial = out / (label + '.partial.json.gz')
    with (out / 'export.log').open('w') as log:
        subprocess.run([str(tool), 'export-perfetto', str(out/'rank0.db'), '--output', str(partial)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    final = out / (label + '.json.gz')
    partial.replace(final)
    (out/'source.json').write_text(json.dumps(dict(native=str(dbs[0]),
        rank_device_map=mapping, scope='single rank; no distributed clock model'), indent=2))
    print(final, flush=True)
