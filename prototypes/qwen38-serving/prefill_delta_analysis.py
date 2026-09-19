"""Attribute paired cold-prefill growth without summing overlapping observations."""
import argparse
import collections
import json
from pathlib import Path
import re
import sqlite3
import statistics

from compare_full_prefill import duration


def family(name):
    if name.startswith('hcom_allReduce'):
        return 'allreduce'
    if name in ('MatMulV2', 'MatMulV3'):
        return 'matmul'
    if name.startswith('FusedInferAttentionScore'):
        return 'fia'
    if name.startswith(('bs_gdn_', 'chunk_', 'solve_tril_', 'merge_16x16_', 'recompute_w_u_', 'preprocess_kernel', '_layer_norm_fwd_1pass')) or name == 'CausalConv1d':
        return 'gdn_kernels'
    if name in ('Transpose', 'TensorMove', 'Slice'):
        return 'layout'
    return 'other_compute'


def inspect(root, rank, evidence):
    path = root / 'traceloom' / f'{root.name}-rank{rank}.db'
    c = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    rows = c.execute('''select t.startNs,t.endNs,s.value,n.value,sh.value,i.blockNum,i.mixBlockNum
        from TASK t join COMPUTE_TASK_INFO i using(globalTaskId)
        join STRING_IDS s on s.id=i.opType join STRING_IDS n on n.id=i.name
        left join STRING_IDS sh on sh.id=i.inputShapes order by t.startNs''').fetchall()
    result = []
    expected = [1024, 1536, 512, 1536, 512, 1024]
    assert [d['scheduled'] for d in evidence['dispatch']] == [[n] for n in expected]
    for step, dispatch, n in zip(evidence['steps'], evidence['dispatch'], expected):
        assert dispatch['computed'] == ([1536] if n == 512 else [0])
        assert 'FULL' in dispatch['mode']
        lo, hi = step['start_ns'], step['end_ns']
        body = [r for r in rows if lo <= r[0] < hi]
        groups, ops, matrix = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
        for row in body:
            groups[family(row[2])].append((row[0], row[1]))
            key = re.sub(r'_\d+$', '', row[2])
            ops[key].append(row)
            if family(row[2]) == 'matmul':
                matrix[row[2] + ' ' + str(row[4])].append(row)
        union_ms = {k: duration(v) for k, v in groups.items()}
        all_union = duration([(r[0], r[1]) for r in body])
        conv = [r for r in body if r[2] == 'CausalConv1d']
        norm = [r for r in body if r[2].startswith('_layer_norm_fwd_1pass')]
        assert len(conv) == len(norm) == 48
        spans = []
        for a, b in zip(conv, norm):
            assert a[0] < b[1]
            projection = next(r for r in body if r[0] >= b[1] and family(r[2]) == 'matmul')
            spans.append((projection[0] - a[0]) / 1e6)
        def aggregate(group):
            return {k: dict(count=len(v), sum_ms=sum(r[1]-r[0] for r in v)/1e6,
                            median_us=statistics.median((r[1]-r[0])/1000 for r in v),
                            blocks=sorted(set((r[5], r[6]) for r in v))) for k,v in group.items()}
        comm = c.execute('select count, endNs-startNs from COMMUNICATION_OP where startNs>=? and endNs<=?', (lo,hi)).fetchall()
        assert len(comm) == 128
        result.append(dict(rank=rank, step=step['step'], tokens=n, computed=dispatch['computed'],
                           descriptor=dispatch['descriptor'], body_ms=step['body_ms'],
                           category_union_ms=union_ms, overlap_across_categories_ms=sum(union_ms.values())-all_union,
                           uncovered_ms=step['body_ms']-all_union,
                           gdn_conv_to_outproj_ms=sum(spans), gdn_layer_spans_ms=spans,
                           operators=aggregate(ops), matmul_shapes=aggregate(matrix),
                           communication_counts=sorted(set(r[0] for r in comm))))
    c.close()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('round',type=Path);a=p.parse_args()
    evidence=json.loads((a.round/'traceloom/profile-summary.json').read_text())
    steps=[s for rank in evidence['ranks'] for s in inspect(a.round,rank['rank'],rank)]
    summary=[]
    for rank in (0,1):
        groups={n:[s for s in steps if s['rank']==rank and s['tokens']==n] for n in (1024,1536)}
        assert all(len(v)==2 for v in groups.values())
        def pair(getter):
            values=[statistics.mean(getter(s) for s in groups[n]) for n in (1024,1536)]
            return dict(before_ms=values[0], after_ms=values[1], delta_ms=values[1]-values[0])
        categories={key:pair(lambda s:s['category_union_ms'].get(key,0)) for key in set().union(*(s['category_union_ms'] for s in steps))}
        operators={key:pair(lambda s:s['operators'].get(key,{}).get('sum_ms',0)) for key in set().union(*(s['operators'] for s in steps))}
        summary.append(dict(rank=rank, body=pair(lambda s:s['body_ms']), categories=categories,
                            overlap=pair(lambda s:s['overlap_across_categories_ms']), uncovered=pair(lambda s:s['uncovered_ms']),
                            gdn_span=pair(lambda s:s['gdn_conv_to_outproj_ms']),
                            operators=dict(sorted(operators.items(),key=lambda kv:-kv[1]['delta_ms']))))
    output=dict(scope='Same candidate process; two separately profiled bodies per length/rank. ABBA1024/1536 actual cold work;1536 is first chunk of2048. Categories are device interval unions, not FLOPs or guaranteed recoverable time; allreduce includes peer waits. GDN regional spans overlap operator categories and must not be added to them. No cross-rank clock alignment.',steps=steps,summary=summary)
    (a.round.parent/'delta-analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    for s in summary:
        print('rank',s['rank'],'body',s['body'])
        print('categories',s['categories']);print('gdn',s['gdn_span'])
        print('top operator deltas',list(s['operators'].items())[:10])


if __name__=='__main__':
    main()
