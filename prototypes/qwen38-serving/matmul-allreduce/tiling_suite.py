"""Collect real kernel tiling bytes and PMU; instruction export may be partial."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ['CAPSULE'])
rows = []
for n in (1024, 1408, 1536, 1792):
    out = root / 'measurements' / str(n)
    out.parent.mkdir(exist_ok=True)
    cmd = ['/usr/local/Ascend/cann-9.0.1/bin/msprof', 'op', f'--output={out}',
           '--aic-metrics=TimelineDetail,Default', '--replay-mode=kernel',
           '--warm-up=0', '--launch-count=1', '--kernel-name=MatMul*',
           f'--application={sys.executable} {Path(__file__).with_name("tiling_app.py")} {n}']
    with (out.parent / f'{n}.log').open('w') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=420)
    csvs = list(out.rglob('*.csv'))
    tilings = list(out.rglob('input_tiling.bin'))
    rows.append(dict(tokens=n, exit_code=result.returncode, csv_count=len(csvs),
                     tiling_paths=[str(p.relative_to(root)) for p in tilings],
                     scope='first selected eager ND GEMM; kernel replay, not warm graph timing. Detail success must be checked separately.'))
    (root/'receipt.json').write_text(json.dumps(rows,indent=2))
    print(rows[-1],flush=True)
    assert result.returncode == 0 and csvs and tilings, rows[-1]
