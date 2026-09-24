"""Stage TP2 attention / EP2 experts without mutating a running TP capsule.

Attention geometry is unchanged. Native BF16 MoE owns expert partition and
communication. This source port is UNQUALIFIED until real graph/retrieval gates.
"""
import argparse
import difflib
import json
from pathlib import Path
import shutil


def stage(seed, output):
    source = (seed / 'service_adapter.py').read_text()
    before = 'parallel.pipeline_parallel_size, parallel.enable_expert_parallel) == (2,1,1,False)'
    after = 'parallel.pipeline_parallel_size, parallel.enable_expert_parallel) == (2,1,1,True)'
    if source.count(before) != 1:
        raise ValueError('Expected exactly one qualified TP2/DP1/EP-off contract')
    if output.exists():
        raise FileExistsError(output)
    shutil.copytree(seed, output, ignore=shutil.ignore_patterns('__pycache__', '*.log'))
    changed = source.replace(before, after)
    (output / 'service_adapter.py').write_text(changed)
    (output / 'parallel.diff').write_text(''.join(difflib.unified_diff(
        source.splitlines(True), changed.splitlines(True),
        fromfile='expert-tp2/service_adapter.py', tofile='expert-ep2/service_adapter.py')))
    (output / 'parallel-contract.json').write_text(json.dumps({
        'status': 'UNQUALIFIED', 'source': str(seed),
        'attention_tp': 2, 'attention_dp': 1, 'expert_ep': 2,
        'required_gates': ['loaded target and draft expert partition receipts',
                           'FULL prefill/mixed/decode with native EP communication',
                           'cold/warm retrieval and real MTP',
                           'near-native context and concurrent requests'],
        'unchanged': 'Attention geometry, graph envelope, request slots and native libraries',
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    stage(args.seed, args.output)
