"""Summarize immutable paired count capsules, preserving cache-policy confounds."""
import argparse
import csv
import json
import re
import statistics as stats
from pathlib import Path


def metric(lines,name):
    values=[float(s.rsplit(' ',1)[1]) for s in lines if s.startswith('vllm:'+name+'{')]
    return sum(values) if values else None


def main():
    p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    rows=[];profile_rows=[];unqualified=[]
    for k in range(5):
        hits=[]
        for arm in ('native','candidate'):
            family='qwen-mtp-count-service-20260920' if k==0 or (k==1 and arm=='native') else 'qwen-mtp-count-service-20260920-v2'
            root=a.runs/family/f'{arm}-k{k}'
            receipt=root/'receipt.json'
            x=json.loads(receipt.read_text()) if receipt.exists() else {}
            if x.get('status')!='PASS' or x.get('server_exit')!=0:
                unqualified.append(dict(mtp_tokens=k,arm=arm,status=x.get('status','NOT_RUN'),error=x.get('error'),source=str(root.resolve())))
                continue
            assert len(x['benchmarks'])==9
            log=(root/'server.log').read_text()
            capture=re.findall(r'Graph capturing finished in ([0-9.]+) secs, took ([0-9.]+) GiB',log)
            capacities=re.findall(r'Maximum concurrency for .*? tokens per request: ([0-9.]+)x',log)
            cached=metric(x['metrics'],'prefix_cache_hits_total');hits.append(cached)
            proposed=metric(x['metrics'],'spec_decode_num_draft_tokens_total')
            accepted=metric(x['metrics'],'spec_decode_num_accepted_tokens_total')
            drafts=metric(x['metrics'],'spec_decode_num_drafts_total')
            for c in (1,4,8):
                samples=[b for b in x['benchmarks'] if b['concurrency']==c]
                values=[b['output_tokens_per_second'] for b in samples]
                row=dict(mtp_tokens=k,arm=arm,concurrency=c,
                    output_tokens_per_second=stats.median(values),min=min(values),max=max(values),
                    ttft_ms=1000*stats.median(t for b in samples for t in b['ttft_s']),
                    capture_gib=float(capture[-1][1]) if capture else None,configured_context_concurrency=float(capacities[-1]) if capacities else None,
                    prefix_hit_tokens=cached,prefix_query_tokens=metric(x['metrics'],'prefix_cache_queries_total'),
                    draft_acceptance_rate=accepted/proposed if proposed else None,
                    mean_acceptance_length=1+accepted/drafts if drafts else None,
                    source=str(root.resolve()))
                rows.append(row)
            for rank in (0,1):
                profile=json.loads((root/'traceloom'/f'{arm}-k{k}-decode-rank{rank}-summary.json').read_text())
                targets=[g for g in profile['graphs'] if g['kind']=='target' and g['evidence']=='exact_direct']
                draft=[g for g in profile['graphs'] if g['kind']=='draft' and g['evidence']=='exact_direct']
                assert len(targets)==6 and len(profile['dispatch'])==6,(root,rank,len(targets))
                assert len(draft)==(6 if k else 0),(root,rank,len(draft))
                profile_rows.append(dict(mtp_tokens=k,arm=arm,rank=rank,
                    target_ms=stats.median(g['duration_us'] for g in targets)/1000,
                    draft_ms=stats.median(g['duration_us'] for g in draft)/1000 if draft else 0,
                    target_period_ms=stats.median(g['target_period_us'] for g in targets if 'target_period_us' in g)/1000,
                    after_target_ms=stats.median(g['after_target_us'] for g in targets if 'after_target_us' in g)/1000,
                    dispatch=profile['dispatch']))
        if len(hits)==2: assert hits[0]==hits[1],(k,hits)
    result=dict(scope='Same hw3 TP2 pair, APC on, synthetic shared prompts3073+8*i and64 outputs; one warm-up and three measured cohorts. Output throughput includes prefill. APC work differs between K0 and speculative counts. Profiles are separate diagnostic windows, not throughput samples.',throughput=rows,decode_profiles=profile_rows,unqualified=unqualified)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2))
    with (a.output/'throughput.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# MTP-count comparison','',result['scope'],'','## Median end-to-end output tokens/s','','| MTP | Native C1/C4/C8 | Candidate C1/C4/C8 | Candidate accepted length |','|---|---|---|---|']
    for k in range(5):
        cells=[]
        for arm in ('native','candidate'):
            cells.append(' / '.join(f"{r['output_tokens_per_second']:.2f}" for r in rows if r['mtp_tokens']==k and r['arm']==arm) or 'Not qualified')
        r=next((r for r in rows if r['mtp_tokens']==k and r['arm']=='candidate'),None)
        length=f"{r['mean_acceptance_length']:.3f}" if k and r is not None else '—'
        lines.append(f'|{k}|{cells[0]}|{cells[1]}|{length}|')
    if unqualified:
        lines += ['','## Unqualified (excluded, including partial measurements)']
        lines += [f"- {r['arm']} K{r['mtp_tokens']}: {r['status']}; {r['error'] or 'no completed receipt'}" for r in unqualified]
    lines += ['','## C4 decode timeline (rank0 medians, ms)','','| MTP | Arm | Target | Draft | Target-to-target | After target* |','|---|---|---|---|---|---|']
    for r in profile_rows:
        if r['rank']==0:
            lines.append(f"|{r['mtp_tokens']}|{r['arm']}|{r['target_ms']:.3f}|{r['draft_ms']:.3f}|{r['target_period_ms']:.3f}|{r['after_target_ms']:.3f}|")
    lines += ['','*After target includes draft, sampling, metadata and waits; it is not pure idle. Both-rank full observations and actual dispatch descriptors remain in summary.json. No SWE, quality, statistical-significance or universal-best-count claim.']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(a.output/'README.md')

if __name__=='__main__':main()
