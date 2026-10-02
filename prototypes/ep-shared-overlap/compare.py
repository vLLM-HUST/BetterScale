"""Compare saved native outputs and measured communication/shared-matmul overlap."""
import argparse
import csv
import json
from pathlib import Path
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('reference', type=Path)
p.add_argument('candidates', type=Path, nargs='+')
a = p.parse_args()
for directory in [a.reference, *a.candidates]:
    for rank in range(2):
        ref = torch.load(a.reference / f'rank{rank}-outputs.pt', weights_only=True)
        got = torch.load(directory / f'rank{rank}-outputs.pt', weights_only=True)
        assert len(ref) == len(got)
        for x, y in zip(ref, got):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        paths = list(directory.glob(f'profile-rank{rank}/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv'))
        measured = {}
        if paths:
            rows = list(csv.DictReader(paths[0].open()))
            shared = [r for r in rows if r['Name'].startswith('aclnnMatmul_')]
            for kind in ('allGather', 'reduceScatter'):
                comm = [r for r in rows if r['Name'].startswith('hcom_' + kind)]
                # Shared MoE is the only ordinary matmul in this leaf fixture;
                # routed experts use GroupedMatmul. Device intervals, not host calls.
                overlaps = []
                for x in shared:
                    xs, xd = float(x['Start Time(us)']), float(x['Duration(us)'])
                    for y in comm:
                        ys, yd = float(y['Start Time(us)']), float(y['Duration(us)'])
                        overlaps.append(max(0, min(xs + xd, ys + yd) - max(xs, ys)))
                measured[kind + '_shared_matmul_overlap_us'] = sum(overlaps)
            measured['shared_matmul_count'] = len(shared)
        print(json.dumps(dict(directory=str(directory), rank=rank, bitwise_equal=True,
                              **json.loads((directory / f'rank{rank}.json').read_text()), **measured)))
