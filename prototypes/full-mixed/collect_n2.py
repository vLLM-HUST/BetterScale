"""Reduce copied shadow receipts; keep full per-wave material in the capsule."""
import argparse
import json
from pathlib import Path


def collect(engine, capsule):
    read = lambda name: json.loads((engine / name).read_text())
    protocol, config = read('protocol.json'), read('config.json')
    rows = []
    for rank in range(config['tensor_parallel_size']):
        target = read(f'shadow-rank{rank}.json')
        graphs = [json.loads(p.read_text()) for p in engine.glob(f'full-draft-rank{rank}-*.json')
                  if not any(s in p.name for s in ('failure', 'signature'))]
        bounds = read(f'cross-step-rank{rank}.json')
        assert target and graphs and all(x['status'] == 'PASS' for x in target)
        rows.append(dict(rank=rank, target_checks=len(target),
            mixed_checks=sum(x['num_prefills'] > 0 and x['num_decodes'] > 0 for x in target),
            prefill_checks=sum(x['num_prefills'] > 0 and x['num_decodes'] == 0 for x in target),
            decode_checks=sum(x['num_prefills'] == 0 for x in target),
            target_max_diff=max(m['max_diff'] for x in target for m in x['checks']),
            draft_banks=len(graphs), draft_first_checks=sum(x['capture_checked'] for x in graphs),
            draft_replay_checks=sum(x['checks'] for x in graphs),
            draft_signed_zero_words=sum(x.get('signed_zero_words', 0) for x in graphs),
            draft_fallbacks=sum(x['fallbacks'] for x in graphs),
            draft_references=sorted({x.get('reference', 'unpadded') for x in graphs}),
            exact_metadata_checks=bounds['exact_metadata_shadow_checks'],
            full_forwards=bounds.get('full_forwards'), late_commits=bounds['late_commits']))
    turnover = read('turnover.json'); turnover.pop('events')
    return dict(capsule=capsule, real=protocol['real_weights'],
        budget=config['max_num_batched_tokens'], tp=config['tensor_parallel_size'],
        draft_reference=protocol.get('draft_reference', 'unpadded'),
        scheduler=read('n2-scheduler.json'), turnover=turnover, ranks=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('engine', type=Path); p.add_argument('--capsule', required=True)
    p.add_argument('--name', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = json.loads(a.output.read_text()) if a.output.exists() else {}
    result[a.name] = collect(a.engine, a.capsule)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
