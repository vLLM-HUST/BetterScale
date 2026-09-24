"""Real-MTP mixed-length/C64 execution gate through the installed MOD group.

This checks liveness, output lengths, physical routing and native acceptance;
it deliberately does not label generated text as a semantic-quality oracle.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import signal
import time
from betterscale.patches.expert_service.deployment import Deployment, add_arguments, http


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--concurrency',type=int,default=64)
    parser.add_argument('--requests',type=int,default=128)
    args=parser.parse_args()
    assert args.synthetic_acceptance_length is None
    assert args.mtp_tokens==2 and args.requests>=args.concurrency>=args.sources
    group=Deployment(args)
    receipt=dict(status='STARTED',scope='real MTP execution, not quality or frontier throughput',requests=[])
    def save():
        (group.output/'workload.json').write_text(json.dumps(receipt,indent=2)+'\n')
    def cancel(*_):
        raise KeyboardInterrupt('execution gate cancelled')
    signal.signal(signal.SIGTERM,cancel)
    with group:
        ports=group.ports
        receipt['startup']=group.receipt['startup']
        assert all(len(r['native_shadows'])==41 for r in receipt['startup'])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        filler = tokenizer.encode(' The repository contains Python modules and their unit tests.', add_special_tokens=False)
        suffix = tokenizer.encode('\nWrite a Python function to validate a configuration dictionary, and explain its checks.\n', add_special_tokens=False)
        # Boundary-changing waves exercise partial frames, mixed target/draft
        # work, APC reuse and independent attention timelines on all sources.
        lengths = (257, 2049, 4095, 4096, 4097, 8193, 16385, 32769)
        def request(index):
            source = index % args.sources
            length = lengths[(index // args.sources) % len(lengths)]
            prefix = tokenizer.encode(f'Repository audit lane {index % args.concurrency}.\n', add_special_tokens=False)
            n = length-len(prefix)-len(suffix)
            tokens = prefix + (filler*((n+len(filler)-1)//len(filler)))[:n] + suffix
            started = time.monotonic()
            result = http(ports[source], '/v1/completions', dict(model='qwen35', prompt=tokens,
                max_tokens=128, ignore_eos=True, temperature=0), timeout=600)
            assert result['usage']['prompt_tokens'] == length, result['usage']
            assert result['usage']['completion_tokens'] == 128, result['usage']
            assert result['choices'][0]['finish_reason'] == 'length'
            return dict(index=index, source=source, prompt_tokens=length,
                        output_tokens=128, seconds=time.monotonic()-started)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(request, i) for i in range(args.requests)]
            for future in as_completed(futures):
                receipt['requests'].append(future.result()); group.check(); save()
        receipt['final'] = [group.rpc(port, 'expert_receipt') for port in ports]
        receipt['speculation'] = []
        for source, port in enumerate(ports):
            metrics = http(port, '/metrics')
            (group.output/f'attention{source}-metrics.txt').write_text(metrics)
            counters = {}
            for name in ('num_drafts', 'num_draft_tokens', 'num_accepted_tokens'):
                counters[name] = sum(float(line.rsplit(' ',1)[1]) for line in metrics.splitlines()
                                     if line.startswith('vllm:spec_decode_'+name+'_total{'))
            assert counters['num_drafts'] > 0 and counters['num_accepted_tokens'] > 0, counters
            receipt['speculation'].append(counters)
        assert all(r['host_forward_requests'] == 0 and r['native_weight_release']['expert_parameter_storage_bytes'] == 164
                   for r in receipt['final'])
        receipt['status']='PASS';save()
    print('PASS',group.output,flush=True)


if __name__=='__main__':main()
