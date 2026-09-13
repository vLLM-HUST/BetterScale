"""Summarize native DP cohort receipts; not a workload-normalized speedup claim."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median


def summarize(root):
    completion=json.loads((root/'dp-completion.json').read_text())
    assert all(code==0 for code in completion['exitcodes'])
    dp=completion['dp']
    ranks=[json.loads((root/f'dp{r}-results.json').read_text()) for r in range(dp)]
    assert all([x['label'] for x in rows]==[x['label'] for x in ranks[0]] for rows in ranks)
    cohorts=[]
    for rows in zip(*ranks):
        label=rows[0]['label']
        start=min(x['started_monotonic'] for x in rows)
        end=max(x['started_monotonic']+x['elapsed'] for x in rows)
        output=sum(sum(map(len,x['outputs'])) for x in rows)
        item=dict(label=label,profile=rows[0]['profile'],seconds=end-start,
                  launch_skew_ms=1000*(max(x['started_monotonic'] for x in rows)-start),
                  input_tokens=sum(sum(x['lengths']) for x in rows),output_tokens=output,
                  output_tokens_per_second=output/(end-start),rank_completion_seconds=[x['elapsed'] for x in rows])
        windows=[]
        for rank in range(8):
            path=root/label/f'{label}-timing-rank{rank}.json'
            if not path.exists():continue
            events=json.loads(path.read_text())
            target=[x for x in events if x['label']=='strengthen::target_forward']
            draft=[x['device_elapsed_ms'] for x in events if x['label']=='strengthen::draft_forward']
            intervals=[b['device_start_ms']-a['device_start_ms'] for a,b in zip(target,target[1:])]
            modes=json.loads((root/label/f'{label}-modes-rank{rank}.json').read_text())
            waves=json.loads((root/label/f'{label}-waves-rank{rank}.json').read_text())
            windows.append(dict(rank=rank,real_scheduler_waves=len(waves),target_calls=len(target),
                target_forward_median_ms=median(x['device_elapsed_ms'] for x in target) if target else None,
                draft_forward_median_ms=median(draft) if draft else None,
                inter_target_median_ms=median(intervals) if intervals else None,
                modes=dict(Counter(x['mode'] for x in modes)),
                padded_shapes=dict(Counter(x['padded'] for x in modes))))
        if label in ('decode0','decode1'):
            matched=[]
            for rank in range(8):
                w=json.loads((root/label/f'{label}-waves-rank{rank}.json').read_text())
                m=json.loads((root/label/f'{label}-modes-rank{rank}.json').read_text())
                t=[x for x in json.loads((root/label/f'{label}-timing-rank{rank}.json').read_text())
                   if x['label']=='strengthen::target_forward']
                assert len(m)==len(t)
                start=next(i for i in range(min(len(w),len(m))-10)
                           if all(w[j]['total']==96//dp and len(w[j]['scheduled'])==16//dp
                                  and m[j]['mode']=='FULL' for j in range(i,i+11)))
                assert all(x['total']>0 for x in w[:start+11])
                matched.append(dict(rank=rank,start=start,
                    target_ms=median(x['device_elapsed_ms'] for x in t[start:start+10]),
                    cycle_ms=median(t[j+1]['device_start_ms']-t[j]['device_start_ms']
                                    for j in range(start,start+10))))
            assert len({x['start'] for x in matched})==1
            item['first_10_full_occupancy_forwards']=dict(global_requests=16,global_query_rows=96,
                ranks=matched,all_rank_target_median_ms=median(x['target_ms'] for x in matched),
                all_rank_cycle_median_ms=median(x['cycle_ms'] for x in matched))
        item['ranks']=windows
        cohorts.append(item)
    receipts=[x for rank in range(dp) for x in json.loads((root/f'dp{rank}-receipt.json').read_text())]
    # TP ranks within a DP shard describe the same logical cache, not additive copies.
    capacities=[max(x['kv_capacity_tokens'] for x in receipts if x['dp_rank']==r) for r in range(dp)]
    return dict(configuration=completion,logical_kv_tokens_per_dp_shard=capacities,
                aggregate_logical_kv_tokens=sum(capacities),receipts=receipts,cohorts=cohorts,
                interpretation='Fixed synthetic token cohorts, cold prefix cache, greedy K5 with EOS ignored. Cohort times include native scheduling, acceptance and drain; per-step distributions are not matched-work causal speedups. Timings with profile=true are not throughput evidence.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    args=parser.parse_args();result=summarize(args.root)
    (args.root/'dp-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('configuration','aggregate_logical_kv_tokens')}))
    for row in result['cohorts']:
        print(row['label'],round(row['seconds'],3),round(row['output_tokens_per_second'],2))
