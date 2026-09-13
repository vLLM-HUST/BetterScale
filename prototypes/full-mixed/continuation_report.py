"""Same-ordinal, occupied decode windows; event spans are NOT idle estimates.

All ranks must have the requested real geometry at both ends of each cycle.
The output preserves per-wave values before median-of-rank-medians aggregation.
Profile windows are deliberately excluded from this performance report.
"""
import argparse
import json
from pathlib import Path
from statistics import median


def analyze(window, ranks, requests, count=10):
    data = []
    for rank in range(ranks):
        events = json.loads((window / f'{window.name}-timing-rank{rank}.json').read_text())
        modes = json.loads((window / f'{window.name}-modes-rank{rank}.json').read_text())
        target = [e for e in events if e['label'] == 'strengthen::target_forward']
        draft = [e for e in events if e['label'] == 'strengthen::draft_forward']
        assert len(target) == len(draft) == len(modes), (window, rank)
        data.append((target, draft, modes))

    def eligible(mode):
        return (mode['mode'] == 'FULL' and not mode['dummy']
                and mode['actual_requests'] == requests
                and mode['actual_tokens'] == requests * 6)

    candidates = [i for i in range(min(len(t) for t, d, m in data) - 1)
                  if all(eligible(m[i]) and eligible(m[i + 1]) for t, d, m in data)]
    # Preserve a contiguous stretch, not a selection of the fastest steps.
    chosen = next((candidates[j:j + count] for j in range(len(candidates) - count + 1)
                   if candidates[j + count - 1] - candidates[j] == count - 1), None)
    assert chosen is not None, f'No {count}-cycle common occupied window: {window}'
    observations = []
    for rank, (target, draft, modes) in enumerate(data):
        rows = []
        for i in chosen:
            t, d, nxt = target[i], draft[i], target[i + 1]
            start, end = t['device_start_ms'], nxt['device_start_ms']
            ds = d['device_start_ms']
            assert start <= ds <= end and ds + d['device_elapsed_ms'] <= end + .001
            rows.append(dict(wave=i, ms=dict(
                cycle=end - start,
                target=t['device_elapsed_ms'], draft=d['device_elapsed_ms'],
                target_to_draft=ds - start - t['device_elapsed_ms'],
                draft_to_target=end - ds - d['device_elapsed_ms'],
                target_host=t['host_elapsed_ms'], draft_host=d['host_elapsed_ms'])))
        observations.append(dict(rank=rank, waves=rows,
                                 median_ms={k: median(r['ms'][k] for r in rows)
                                            for k in rows[0]['ms']}))
    return dict(window=str(window.resolve()), common_waves=chosen,
                requests_per_rank_view=requests, query_rows_per_rank_view=requests * 6,
                ranks=observations,
                median_ms={k: median(r['median_ms'][k] for r in observations)
                           for k in observations[0]['median_ms']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine', type=Path)
    parser.add_argument('--ranks', type=int, default=8)
    parser.add_argument('--requests-per-rank', type=int, required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    windows = sorted(p for p in args.engine.iterdir()
                     if p.is_dir() and p.name.startswith('decode'))
    assert windows, 'No unprofiled decode windows'
    rows = [analyze(p, args.ranks, args.requests_per_rank, args.count) for p in windows]
    args.output.write_text(json.dumps(dict(scope=__doc__, windows=rows), indent=2) + '\n')
    for row in rows:
        print(Path(row['window']).name, {k: round(v, 3) for k, v in row['median_ms'].items()})


if __name__ == '__main__':
    main()
