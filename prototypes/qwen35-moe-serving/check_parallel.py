"""Validate pinned BF16 Ascend weight geometry across two physical workers.

This proves loaded partition geometry and EP coverage, not numerical correctness.
Ascend's loaded w13 is transposed relative to the checkpoint representation.
"""
import argparse
import json
from pathlib import Path


def check(receipts):
    if len(receipts) != 2:
        raise ValueError('Exactly two physical worker receipts required')
    tp = receipts[0]['attention_tp']
    dp = receipts[0]['attention_dp']
    ep = receipts[0]['enable_expert_parallel']
    assert (tp, dp) in ((2, 1), (1, 2))
    assert {(r['data_parallel_rank'], r['rank']) for r in receipts} == (
        {(0, 0), (0, 1)} if tp == 2 else {(0, 0), (1, 0)})
    by_worker = []
    for receipt in receipts:
        assert (receipt['attention_tp'], receipt['attention_dp'],
                receipt['enable_expert_parallel']) == (tp, dp, ep)
        experts = {}
        for scope, expected in [('target', 40), ('draft', 1)]:
            rows = receipt[scope]
            moe = [row for row in rows if 'expert_parallel' in row]
            assert len(moe) == expected, (scope, len(moe))
            for row in moe:
                key = scope + ':' + row['name']
                assert key not in experts
                experts[key] = row
                p = row['expert_parallel']
                assert (p['tp_size'], p['ep_size'], p['dp_size'], p['use_ep']) == (
                    1 if ep else 2, 2 if ep else 1, dp, ep)
                n, width = (128, 512) if ep else (256, 256)
                assert row['weights']['w13_weight'] == [n, 2048, 2 * width]
                assert row['weights']['w2_weight'] == [n, width, 2048]
                assert row['intermediate_size'] == 512
            qkv = [row for row in rows if row['name'].endswith('.qkv_proj')]
            assert len(qkv) == (10 if scope == 'target' else 1)
            assert all(row['weights']['weight'] == [9216 // tp, 2048] for row in qkv)
        by_worker.append(experts)
    assert by_worker[0].keys() == by_worker[1].keys()
    for key in by_worker[0]:
        pair = [worker[key] for worker in by_worker]
        if ep:
            maps = [row['expert_map'] for row in pair]
            assert all(len(mapping) == 256 for mapping in maps)
            owned = [{i for i, local in enumerate(mapping) if local >= 0} for mapping in maps]
            assert not owned[0] & owned[1]
            assert owned[0] | owned[1] == set(range(256))
            assert all(sorted(v for v in mapping if v >= 0) == list(range(128)) for mapping in maps)
            assert {row['expert_parallel']['ep_rank'] for row in pair} == {0, 1}
        else:
            assert all(row['expert_map'] is None for row in pair)
            assert {row['expert_parallel']['tp_rank'] for row in pair} == {0, 1}
    return {'status': 'PASS', 'attention_tp': tp, 'attention_dp': dp,
            'expert_partition': 'EP2' if ep else 'TP2', 'physical_moe_layers': 41,
            'scope': 'loaded weight geometry and expert-map coverage, not semantic qualification'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(check([json.loads(p.read_text())
                            for p in sorted(args.directory.glob('partition-*.json'))]), indent=2))
