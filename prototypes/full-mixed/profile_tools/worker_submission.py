"""Separate worker CPU issue phases in a retained eight-rank profile.

Uses enclosing CPU ranges and same-thread CANN calls, not timestamp-nearest
device attribution. The post-sample gap includes RPC/worker wrapper work; it
does not measure when a command first arrived in the queue. Durations are
within-rank and need no cross-rank clock fit. Profiling overhead remains.
"""
import argparse
import json
from pathlib import Path
import sqlite3
from statistics import median


def audit(root, rank, first, last):
    sources = list((root / 'profile').glob(
        f'rank{rank}_*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_{rank}.db'))
    assert len(sources) == 1, (rank, sources)
    with sqlite3.connect(f'file:{sources[0]}?mode=ro', uri=True) as db:
        functions = db.execute("""select cast(a.startNs as integer),
            cast(a.endNs as integer),n.value,a.globalTid from PYTORCH_API a
            join STRING_IDS n on n.id=a.name where n.value like 'strengthen::%'
            order by cast(a.startNs as integer)""").fetchall()
        apis = db.execute("""select a.startNs,a.endNs,n.value,a.globalTid
            from CANN_API a join STRING_IDS n on n.id=a.name where n.value in
            ('aclmdlRIExecuteAsync','aclrtSynchronizeEvent') order by a.startNs""").fetchall()
    def named(name):
        return [f for f in functions if f[2] == 'strengthen::' + name]
    forwards, samples, drafts, steps = map(named, (
        'target_forward', 'sample_and_draft', 'draft_forward', 'target_step'))
    assert len(forwards) == len(samples) == len(drafts)
    assert 1 <= first <= last < len(forwards)
    modes = json.loads((root / f'{root.name}-modes-rank{rank}.json').read_text())
    assert len(modes) == len(forwards)
    rows = []
    for wave in range(first, last + 1):
        f, s, d = forwards[wave], samples[wave - 1], drafts[wave - 1]
        enclosing = [x for x in steps if x[0] <= f[0] <= f[1] <= x[1]]
        assert len(enclosing) == 1
        step = enclosing[0]
        assert s[0] <= d[0] <= d[1] <= s[1] <= step[0] <= f[0]
        assert len({x[3] for x in (f, s, d, step)}) == 1
        replay = [x for x in apis if x[2] == 'aclmdlRIExecuteAsync'
                  and x[3] == f[3] and f[0] <= x[0] <= x[1] <= f[1]]
        assert len(replay) == 1, (rank, wave, replay)
        cuts = (d[0], d[1], s[1], step[0], replay[0][0])
        labels = ('previous_draft_host', 'sample_tail', 'dispatch_gap', 'target_issue')
        durations = {name: (b - a) / 1e6
                     for name, a, b in zip(labels, cuts, cuts[1:])}
        sync = [x for x in apis if x[2] == 'aclrtSynchronizeEvent'
                and x[3] == f[3] and cuts[0] <= x[0] < cuts[-1]]
        rows.append(dict(wave=wave, requests=modes[wave]['actual_requests'],
                         tokens=modes[wave]['actual_tokens'], ms=durations,
                         worker_event_wait_ms=sum(x[1]-x[0] for x in sync)/1e6))
    return dict(rank=rank, source=str(sources[0]), waves=rows,
                median_ms={k: median(x['ms'][k] for x in rows) for k in labels},
                median_worker_event_wait_ms=median(x['worker_event_wait_ms'] for x in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('window', type=Path)
    parser.add_argument('--first-wave', type=int, required=True)
    parser.add_argument('--last-wave', type=int, required=True)
    args = parser.parse_args()
    root = args.window.resolve()
    ranks = [audit(root, r, args.first_wave, args.last_wave) for r in range(8)]
    geometry = [[(w['wave'], w['requests'], w['tokens']) for w in r['waves']] for r in ranks]
    assert all(x == geometry[0] for x in geometry)
    output = root / 'analysis/worker-submission.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(dict(contract=__doc__, ranks=ranks), indent=2))
    for r in ranks:
        print(r['rank'], {k: round(v, 3) for k, v in r['median_ms'].items()},
              'worker_event_wait_ms', round(r['median_worker_event_wait_ms'], 3))
