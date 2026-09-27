"""Describe matched C16 replay without turning arrival-limited scores into capacity."""
import argparse,json,re
from pathlib import Path

FIELDS={'output_tokens_s':('output_token_throughput','avg'),
        'ttft_mean_ms':('time_to_first_token','avg'),
        'ttft_p50_ms':('time_to_first_token','p50'),
        'ttft_p95_ms':('time_to_first_token','p95'),
        'itl_mean_ms':('inter_token_latency','avg'),
        'itl_p95_ms':('inter_token_latency','p95'),
        'request_latency_mean_ms':('request_latency','avg'),
        'request_latency_p95_ms':('request_latency','p95')}


def read(root,server):
    run=json.loads((root/'run.json').read_text())
    report=json.loads((root/'aiperf/profile_export_aiperf.json').read_text())
    audit=json.loads((root/'replay_audit.json').read_text())
    assert run['returncode']==0 and report['metadata']['submission_valid']
    assert not report['error_summary'] and not report['was_cancelled']
    assert audit['records']['malformed']==0 and audit['configured_session_trees']==16
    assert not (set(audit['findings'])-{'first_request_ramp_exceeds_measurement_window'})
    receipt=json.loads((server/'receipt.json').read_text())
    assert receipt['status']=='STOPPED_BY_CLIENT' and receipt['server_exit_code']==0
    def counter(file,name):
        return sum(float(x) for x in re.findall(r'^vllm:'+name+r'(?:\{[^\n]*\})?\s+([\d.eE+\-]+)$',file.read_text(),re.M))
    delta={key:counter(server/'metrics-after.txt',key)-counter(server/'metrics-before.txt',key)
           for key in ('spec_decode_num_drafts_total','spec_decode_num_accepted_tokens_total')}
    assert delta['spec_decode_num_drafts_total']>0
    result={'artifact':str(root.resolve()),'server':str(server.resolve()),
        'upstream_submission_valid':True,'wrapper_status':run['status'],
        'review_findings':audit['findings'],
        'profiling_requests':audit['records']['profiling'],
        'mean_http_inflight':audit['http_occupancy']['mean_inflight_requests'],
        'source_trace_count':len(audit['source_traces']),
        'first_request_spread_seconds':audit['startup']['first_request_spread_seconds'],
        'all_phase_acceptance_length':1+delta['spec_decode_num_accepted_tokens_total']/delta['spec_decode_num_drafts_total'],
        'metrics':{name:report[key][stat] for name,(key,stat) in FIELDS.items()}}
    result['metrics']['output_tokens_s_chip']=result['metrics']['output_tokens_s']/2
    return run,result


def main():
    p=argparse.ArgumentParser()
    for name in ('baseline','candidate','baseline-server','candidate-server','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    old,b=read(a.baseline,a.baseline_server);new,c=read(a.candidate,a.candidate_server)
    assert old['protocol']==new['protocol'] and old['profile']==new['profile']=='smoke'
    assert old['concurrent_agent_clients']==new['concurrent_agent_clients']==16
    assert old['target']['speculative_decoding']==new['target']['speculative_decoding']
    # Configuration differs only in the claimed implementation revision/arm.
    assert old['target']['engine']['launch_command_without_secrets']==new['target']['engine']['launch_command_without_secrets']
    result={'scope':'one same-host same-protocol A/B; descriptive replay, not saturated throughput or statistical significance',
            'source_commit':'a1d543e','baseline':b,'candidate':c,
            'relative_change_pct':{key:100*(c['metrics'][key]/value-1) for key,value in b['metrics'].items()}}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'baseline':b['metrics'],'candidate':c['metrics'],'change_pct':result['relative_change_pct']},indent=2))

if __name__=='__main__':main()
