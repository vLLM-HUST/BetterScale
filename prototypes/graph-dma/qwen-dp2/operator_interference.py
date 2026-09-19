"""Attribute the separate sustained-DMA profile; never replace timing trials."""
import argparse
import collections
import json
import sqlite3
from pathlib import Path


def union_ms(rows):
    end = 0
    total = 0
    for start, stop in sorted(rows):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total / 1e6


def analyze(path, phase="d2d_sustained"):
    conn = sqlite3.connect(path)
    strings = dict(conn.execute('SELECT id,value FROM STRING_IDS'))
    phases = conn.execute("""SELECT s.value,p.startNs,p.endNs FROM PYTORCH_API p
        JOIN STRING_IDS s ON p.name=s.id WHERE s.value LIKE 'dma_probe::%'
        ORDER BY CAST(p.startNs AS INTEGER)""").fetchall()
    phases = [p for p in phases if p[0] in ['dma_probe::compute', 'dma_probe::'+phase]]
    assert [p[0] for p in phases] == ['dma_probe::compute', 'dma_probe::'+phase]
    groups, coverage = [], []
    for name, a, b in phases:
        rows = conn.execute('''SELECT t.startNs,t.endNs,i.opType,i.inputShapes,
            i.inputDataTypes,t.streamId,t.taskType,t.globalTaskId
            FROM TASK t JOIN COMPUTE_TASK_INFO i USING(globalTaskId)
            WHERE t.startNs>=? AND t.endNs<=? ORDER BY t.startNs''', (int(a), int(b))).fetchall()
        assert len({r[-1] for r in rows}) == len(rows), 'duplicate native task'
        group = collections.defaultdict(list)
        for st, en, op, shape, dtype, stream, typ, _ in rows:
            group[(strings[op], strings.get(shape), strings.get(dtype), stream,
                   strings.get(typ))].append((st, en))
        groups.append(group)
        latest_end, prior_op, gaps = 0, None, []
        for st, en, op, *_ in rows:
            if prior_op is not None and st > latest_end:
                gaps.append(dict(duration_ms=(st-latest_end)/1e6,
                                 start_from_scope_ms=(latest_end-int(a))/1e6,
                                 end_from_scope_ms=(st-int(a))/1e6,
                                 left_op=prior_op, right_op=strings[op]))
            if en > latest_end:
                latest_end, prior_op = en, strings[op]
        spans = [(r[0], r[1]) for r in rows]
        cover = union_ms(spans)
        envelope = (max(e for _, e in spans) - min(s for s, _ in spans))/1e6
        coverage.append(dict(phase=name, host_scope_ms=(int(b)-int(a))/1e6,
                             model_task_envelope_ms=envelope, model_task_union_ms=cover,
                             uncovered_inside_envelope_ms=envelope-cover,
                             largest_gaps=sorted(gaps,key=lambda g:-g["duration_ms"])[:3]))
    assert groups[0].keys() == groups[1].keys()
    details, totals = [], collections.defaultdict(lambda: [0, 0, 0])
    for key, before in groups[0].items():
        after = groups[1][key]
        assert len(before) == len(after), key
        x, y = [sum(e-s for s,e in rs)/1e6 for rs in [before,after]]
        details.append(dict(op=key[0], shape=key[1], dtype=key[2], stream=key[3],
                            task_type=key[4], count=len(before), baseline_ms=x,
                            dma_ms=y, delta_ms=y-x, ratio=y/x))
        t = totals[key[0]]
        t[0] += len(before)
        t[1] += x
        t[2] += y
    summary = [dict(op=k,count=n,baseline_ms=x,dma_ms=y,delta_ms=y-x,ratio=y/x)
               for k,(n,x,y) in totals.items()]
    return dict(source=str(path), coverage=coverage,
                operators=sorted(summary,key=lambda x:-x['delta_ms']),
                shapes=sorted(details,key=lambda x:-x['delta_ms']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis_dir',type=Path)
    parser.add_argument('--phase', default='d2d_sustained')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = dict(protocol='One separate profile pair per rank; see the source run for transfer size. '
                  'Matched op, shape, dtype, stream, task type and count. '
                  'Operator times are task duration sums, not exclusive critical-path attribution.',
                  phase=args.phase, ranks=[analyze(args.analysis_dir/f'rank{r}.db',args.phase) for r in range(2)])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r, data in enumerate(result['ranks']):
        print('rank',r,json.dumps(data['coverage']))
        for op in data['operators'][:10]:
            print(op['op'],op['count'],*[round(op[k],3) for k in ['baseline_ms','dma_ms','delta_ms','ratio']])
