"""Eight-device workload endpoint: installed expert group or native true EP control.

Caller owns admission/watchdog. This fixture owns only HTTP routing, benchmark
configuration and native control observation; expert execution lives in the MOD.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from betterscale.patches.expert_service.deployment import Deployment,add_arguments,http,capture_sizes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--arm',choices=['a4e4','a6e2','dp8ep8','tp8ep8'],required=True)
    parser.add_argument('--router-port',type=int,default=32900)
    args=parser.parse_args()
    assert args.mtp_tokens==2
    assert args.synthetic_acceptance_length is None, 'SWE continuation uses real MTP acceptance'
    assert args.devices=='0,1,2,3,4,5,6,7'
    separated=args.arm.startswith('a')
    group=Deployment(args) if separated else None
    out=args.output.resolve();children=[];stopping=False
    ports=group.ports if group else [args.port_base]
    deadline=time.monotonic()+args.lifetime
    receipt=dict(status='STARTED',arm=args.arm,commands=[],scope='real MTP fixed-shape serving, not task-solving quality')
    def stop(*_):
        nonlocal stopping
        stopping=True
        if group:group.stopping=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    def save(): (out/'benchmark-deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
    def launch(name,command):
        receipt['commands'].append(dict(role=name,argv=command))
        with (out/f'{name}.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES=args.devices,EXPERT_BENCH_OUTPUT=str(out)))
        children.append(child)
    def check():
        if stopping:raise InterruptedError('cancelled')
        if time.monotonic()>deadline:raise TimeoutError('bounded benchmark lifetime')
        if group:group.check()
        assert all(c.poll() is None for c in children),'owned child exited'
    def ready(port):
        try:http(port,'/health',timeout=2);return True
        except OSError:return False
    try:
        if group:group.start()
        else:
            out.mkdir(parents=True,exist_ok=False)
            tp=8 if args.arm=='tp8ep8' else 1
            sizes=capture_sizes(args.max_seqs,2)
            cmd=[sys.executable,'-m','vllm.entrypoints.cli.main','serve',str(args.model),
                '--host','127.0.0.1','--port',str(ports[0]),'--served-model-name','qwen35',
                '--worker-cls','native_benchmark_worker.Worker','--tensor-parallel-size',str(tp),
                '--enable-expert-parallel','--dtype','bfloat16','--max-model-len','262144',
                '--max-num-seqs',str(args.max_seqs),'--max-num-batched-tokens','4096',
                '--kv-cache-memory-bytes',str(int(args.kv_gib*1024**3)),
                '--enable-prefix-caching','--mamba-cache-mode','align','--async-scheduling',
                '--limit-mm-per-prompt','{"image":0,"video":0}','--additional-config','{"enable_cpu_binding":false}',
                '--seed','17','--generation-config','vllm','--override-generation-config','{"temperature":0}',
                '--default-chat-template-kwargs','{"enable_thinking":false}',
                '--shutdown-timeout','60','--safetensors-load-strategy','lazy',
                '--compilation-config',json.dumps(dict(mode=0,cudagraph_mode='FULL_DECODE_ONLY',
                    cudagraph_capture_sizes=sizes,max_cudagraph_capture_size=max(sizes))),
                '--speculative-config',json.dumps(dict(method='mtp',num_speculative_tokens=2,
                    rejection_sample_method='standard'))]
            if tp==1:cmd+=['--data-parallel-size','8','--data-parallel-size-local','8','--api-server-count','1']
            launch('native',cmd);save()
            while not ready(ports[0]):check();time.sleep(1)
            http(ports[0],'/collective_rpc',dict(method='ep_receipt',timeout=180),timeout=200)
            ep=[json.loads((out/f'ep-rank-{i}.json').read_text()) for i in range(8)]
            assert all(r['ep_size']==8 and r['tp_size']==tp and r['dp_size']==8//tp for r in ep)
            for layer in range(41):
                assert sorted(e for r in ep for e in r['layers'][layer]['global_expert_ids'])==list(range(256))
            receipt['ep_verified_layers']=41
        # Bounded HTTP integration before exposing the endpoint to long replay.
        def gate(i):
            result=http(ports[i%len(ports)],'/v1/completions',dict(model='qwen35',
                prompt='Implement a dictionary validator in Python.\n'*128,max_tokens=128,
                ignore_eos=True,temperature=0),timeout=600)
            assert result['usage']['completion_tokens']==128
        with ThreadPoolExecutor(max_workers=16) as pool:list(pool.map(gate,range(32)))
        receipt['http_gate_requests']=32
        launch('router',[sys.executable,str(Path(__file__).with_name('agentx_router.py')),
            '--backends',*[f'http://127.0.0.1:{p}' for p in ports],
            '--dp-size','8' if args.arm=='dp8ep8' else '1',
            '--port',str(args.router_port),'--receipt',str(out/'routing.json')])
        while not ready(args.router_port):check();time.sleep(1)
        receipt['status']='READY';save();(out/'benchmark-ready').write_text('ready\n')
        while not stopping and not (out/'stop').exists():check();time.sleep(1)
        receipt['metrics']={}
        for port in ports:
            metrics=http(port,'/metrics');(out/f'metrics-{port}.txt').write_text(metrics)
            counts={name:sum(float(line.rsplit(' ',1)[1]) for line in metrics.splitlines()
                      if line.startswith('vllm:spec_decode_'+name+'_total{'))
                    for name in ('num_drafts','num_draft_tokens','num_accepted_tokens')}
            assert counts['num_drafts']>0
            counts['observed_acceptance_length']=1+counts['num_accepted_tokens']/counts['num_drafts']
            assert counts['num_accepted_tokens']>0,counts
            counts['prefix_cache_hits']=sum(float(line.rsplit(' ',1)[1]) for line in metrics.splitlines()
                if line.startswith('vllm:prefix_cache_hits_total{'))
            counts['prefix_cache_queries']=sum(float(line.rsplit(' ',1)[1]) for line in metrics.splitlines()
                if line.startswith('vllm:prefix_cache_queries_total{'))
            receipt['metrics'][str(port)]=counts
        receipt['status']='PASS'
    except BaseException as error:
        receipt['status']='FAIL';receipt['error']=repr(error);raise
    finally:
        for c in reversed(children):
            if c.poll() is None:c.send_signal(signal.SIGINT)
            try:c.wait(timeout=90)
            except subprocess.TimeoutExpired:c.kill();c.wait(timeout=10)
        if group:group.close()
        receipt['exit_codes']=[c.returncode for c in children]
        if any(receipt['exit_codes']) or (group and group.receipt['status']!='PASS'):receipt['status']='FAIL'
        if out.exists():save()
    assert receipt['status']=='PASS',receipt


if __name__=='__main__':main()
