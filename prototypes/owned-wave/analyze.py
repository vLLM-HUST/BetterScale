"""Validate the bounded owned-wave capsule, including every distributed rank."""
import json
from pathlib import Path
import sys


def analyze(path):
    engine = path / 'engine'
    complete = json.loads((engine / 'complete.json').read_text())
    assert complete['status'] == 'PASS' and all(x == 0 for x in complete['exitcodes'])
    config = json.loads((engine / 'config-dp0.json').read_text())
    tp, dp = config['tensor_parallel_size'], len(complete['exitcodes'])
    rows = [json.loads((engine / f'rank{i}.json').read_text()) for i in range(tp*dp)]
    full_n2 = 'full_prefill_width' in rows[0]
    for row in rows:
        assert row['status'] == 'PASS'
        assert row['ep_size'] == tp*dp and row['tp_size'] == tp
        assert row['runner_calls'] == []
        assert row['activation_restores_adopted_state']
        assert row['terminal_drains'] == 2
        if full_n2:
            assert row['startup_graphs'] == 4 and row['numerical_forward_calls'] == 8
            limit = row.get('nplus2_output_limit', 7)
            total = 2 * (limit + 1)
            assert row['shadow_actions'] == 8 + 9 + total
            assert row['full_prefill_width'] == 32
            assert row['quorum_size'] == tp*dp and row['turnover_behind_old_drain']
            trace = row['nplus2_trace']
            live, retired, submitted = [], [], []
            for event in trace:
                seq = event['sequence']
                if event['action'] == 'submit':
                    assert seq == len(submitted)
                    if seq >= 2:
                        assert seq-2 in retired, 'N+2 issued without N quorum'
                    live.append(seq)
                    submitted.append(seq)
                else:
                    assert live.pop(0) == seq
                    retired.append(seq)
                assert len(live) <= 2 and event['outstanding'] == live
            assert not live and submitted == retired == list(range(total))
            turnover = next(e for e in trace if e['action'] == 'submit' and e['generation'] == 2)
            assert turnover['sequence'] == limit+1 and turnover['outstanding'] == [limit, limit+1]
            assert row['tokens_by_generation']['1'] == row['tokens_by_generation']['2']
        else:
            assert row['startup_graphs'] == 2 and row['numerical_forward_calls'] == 4
            assert row['shadow_actions'] == 18
            assert row['admission_requires_retirement'] and row['generation'] == 2
        assert row['max_outstanding'] == 2
        assert len(row['checks']) == 9
        assert all(x['all_kv_bytes_exact'] for x in row['checks'])
        assert row['layers'] == (2 if config.get('load_format') == 'dummy' else 48)
    for rank in range(dp):
        peers = [r for r in rows if r['dp_rank'] == rank]
        assert len(peers) == tp
        tokens = lambda r: r['tokens_by_generation']['1'] if full_n2 else r['autonomous_tokens']
        assert all(tokens(r) == tokens(peers[0]) for r in peers)
        result = json.loads((engine / f'result-dp{rank}.json').read_text())
        assert result['status'] == 'PASS'
        assert (result['outputs'][0][:peers[0].get('nplus2_output_limit', 7)] if full_n2 else result['outputs'][0][1:]) == tokens(peers[0])
    return dict(status='PASS', tp=tp, dp=dp, ep=tp*dp, layers=rows[0]['layers'], full_prefill_nplus2=full_n2,
                tokens_by_dp={r: next(tokens(x) for x in rows if x['dp_rank']==r)
                              for r in range(dp)})


if __name__ == '__main__':
    print(json.dumps(analyze(Path(sys.argv[1])), indent=2))
