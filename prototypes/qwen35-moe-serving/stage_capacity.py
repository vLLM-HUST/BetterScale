"""Stage a separate, unqualified sixteen-request MoE capsule.

Exact substitutions preserve head geometry, ABI pointer counts and alignment.
The host adapter must be rebuilt; CPU checks do not qualify the new envelope.
"""
import argparse
import difflib
import json
from pathlib import Path
import shutil


def stage(seed, output):
    assert not output.exists()
    shutil.copytree(seed, output, ignore=shutil.ignore_patterns('__pycache__', '*.log'))
    changes = {
        'count_policy.py': [('BINS = (3, 6, 12, 24, 40)', 'BINS = (3, 6, 12, 24, 40, 48, 80)'),
                            ('8*WIDTH', '16*WIDTH'), ('len(lengths) <= 8', 'len(lengths) <= 16')],
        'service_adapter.py': [('max_num_seqs == 8', 'max_num_seqs == 16')],
        'service_metadata.py': [('torch.zeros(9,', 'torch.zeros(17,'), ('n <= 8', 'n <= 16')],
        'device_metadata.py': [('torch.zeros(9,', 'torch.zeros(17,')],
        'device_slots.py': [('meta.live <= 8', 'meta.live <= 16')],
        'host_metadata.py': [('n <= 8', 'n <= 16'), ('= 8  # permanent empty sentinel', '= 16  # permanent empty sentinel')],
        'mixed_core.py': [('torch.zeros(10,', 'torch.zeros(18,'), ('torch.full((9,', 'torch.full((17,'),
                          ('torch.zeros(9,', 'torch.zeros(17,'), ('torch.ones(9,', 'torch.ones(17,'),
                          ('torch.zeros(8*WIDTH,', 'torch.zeros(16*WIDTH,'),
                          ('len(lengths) <= 8', 'len(lengths) <= 16'), ('(10-len(', '(18-len(')],
        'draft_fia.py': [('only8 real requests', 'only16 real requests'),
                         ('<= 9:', '<= 17:'), ('[8:]', '[16:]'), ('[:8]', '[:16]'),
                         ('live ninth request', 'live seventeenth request')],
        'package/betterscale/patches/qwen_gdn/metadata.py': [
            ('requests=8', 'requests=16'), ('tokens if decode else 9', 'tokens if decode else 17'),
            ('// size + 7, 2', '// size + 15, 2'), ('tokens <= 8', 'tokens <= 16')],
        'package/betterscale/patches/qwen_gdn/graphs.py': [('num_reqs=8', 'num_reqs=16'), ('tokens > 8', 'tokens > 16')],
        'package/betterscale/patches/qwen_gdn/host.cpp': [('requests <= 9', 'requests <= 17'), ('requests == 9', 'requests == 17')],
        'package/betterscale/patches/qwen_fia/wave.py': [
            ('len(lengths) <= 9', 'len(lengths) <= 17'),
            ('2528 + 9 * 16 + 9 * columns * 4', '2528 + 17 * 16 + 17 * columns * 4'),
            ('t[2528:2600]', 't[2528:2664]'), ('t[2600:2672]', 't[2664:2800]'),
            ('t[2672:].view(torch.int32).view(9, columns)', 't[2800:].view(torch.int32).view(17, columns)')],
    }
    diff = []
    for name, replacements in changes.items():
        path = output / name
        old = new = path.read_text()
        for before, after in replacements:
            assert before in new, (name, before)
            new = new.replace(before, after)
        compile(new, name, 'exec') if name.endswith('.py') else None
        path.write_text(new)
        diff.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                        fromfile='capacity8/'+name, tofile='capacity16/'+name))
    # A stale host binary must never accidentally execute against enlarged PODs.
    (output/'native/libbs_gdn_host.so').unlink()
    (output/'capacity.diff').write_text(''.join(diff))
    (output/'capacity-contract.json').write_text(json.dumps({
        'status': 'UNQUALIFIED', 'max_requests': 16, 'sentinel_row': 16,
        'query_capacity': 4096, 'required_gates': [
            'rebuild host adapter and update its pin', 'CPU metadata and padding checks',
            'GDN state oracle with sixteen rows', 'FIA target/draft sixteen-row oracle',
            'real serving and FULL capture at new capacities'],
        'source': str(seed)}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    stage(args.seed, args.output)
