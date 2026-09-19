"""Fail-closed all-rank compatibility receipt, no quality/performance inference."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('capsule', type=Path)
p.add_argument('--tp', type=int, required=True)
p.add_argument('--dp', type=int, required=True)
a = p.parse_args()
root = a.capsule / 'engine'
complete = json.loads((root / 'complete.json').read_text())
assert complete['status'] == 'PASS' and complete['exitcodes'] == [0]*a.dp
ranks = [json.loads((root/f'rank{rank}.json').read_text()) for rank in range(a.tp*a.dp)]
assert {r['manifest']['ep_rank'] for r in ranks} == set(range(a.tp*a.dp))
assert {(r['manifest']['dp_rank'], r['manifest']['tp_rank']) for r in ranks} == {
    (dp,tp) for dp in range(a.dp) for tp in range(a.tp)}
for rank in ranks:
    m, rows = rank['manifest'], rank['rows']
    assert m['ep_size'] == a.tp*a.dp and m['tp_size'] == a.tp
    assert len(rows) == 6 and all(r['status'] == 'PASS' and r['all_kv_bytes_exact'] for r in rows)
    assert all(r['replays'] == 2 and r['capture_scope'] == 'fixed native snapshot' for r in rows)
weight_scopes = set()
for dp in range(a.dp):
    config = json.loads((root/f'config-dp{dp}.json').read_text())
    weight_scopes.add(config.get('load_format', 'auto-real-checkpoint'))
    tokens = [[x['sampled'] for x in r['rows']] for r in ranks if r['manifest']['dp_rank'] == dp]
    assert all(t == tokens[0] for t in tokens), 'TP peers disagree'
    result = json.loads((root/f'result-dp{dp}.json').read_text())
    assert result['status'] == 'PASS' and len(result['receipts']) == a.tp
    assert len(result['outputs']) == 1 and len(result['outputs'][0]) == 12
assert len(weight_scopes) == 1
summary = dict(weights=weight_scopes.pop(), status='PASS', tp=a.tp, dp=a.dp, ep=a.tp*a.dp,
               manifests=[r['manifest'] for r in ranks], checks_per_rank=6,
               snapshot_captures_per_rank=6, replays_per_snapshot=2, all_sampled_and_kv_exact=True,
               scope='fixed-snapshot joint capture only; native PA; no device continuation or draft')
(a.capsule/'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
