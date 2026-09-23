"""Profile native BF16/MTP2 execution using dummy weights, never quality/perf claims."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request
from probe import server_command


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--port',type=int,default=32281)
    p.add_argument('--worker',default='dummy_worker.Worker')
    p.add_argument('--long-context-only',action='store_true')
    p.add_argument('--candidate-full',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=args.output;url=f'http://127.0.0.1:{args.port}'
    command=server_command(args)+['--load-format','dummy']
    if args.candidate_full:
        assert args.worker == 'candidate_worker.Worker'
        # APC block2048 needs room alongside live verification rows. At a2048
        # budget, residual prefill capacity can round to zero in a mixed wave.
        command[command.index('--max-num-batched-tokens')+1]='4096'
        command[command.index('--compilation-config')+1]=json.dumps({
            'cudagraph_mode':'FULL',
            'cudagraph_capture_sizes':[3,6,12,16,24,32,64,128,256,512,1024,1536,2048,4096],
            'max_cudagraph_capture_size':4096})
        command+=['--scheduler-cls','apc_boundary.BoundaryScheduler']
    receipt={'status':'STARTED','command':command,'weights':'dummy',
             'scope':'execution/graph/communication structure; NOT model quality or real-workload throughput',
             'phases':[]}
    def save():(root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    def cancel(signum,frame):raise KeyboardInterrupt(f'cancelled {signum}')
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,cancel)
    def post(path,body):
        req=urllib.request.Request(url+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=1800) as response:
            raw=response.read();return json.loads(raw) if raw else None
    def request(length,outputs,slot=0,event=None):
        # Legal fixed IDs for controlled dummy geometry, not semantic prompts.
        tokens=[500+slot]+[1000]*(length-1)
        body={'model':'qwen35-moe','prompt':tokens,'max_tokens':outputs,'temperature':0,
              'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True}}
        req=urllib.request.Request(url+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        usage=None
        with urllib.request.urlopen(req,timeout=1800) as response:
            for line in response:
                if not line.startswith(b'data: '):continue
                payload=line[6:].strip()
                if payload==b'[DONE]':break
                x=json.loads(payload)
                if x.get('usage'):usage=x['usage']
                if event and x.get('choices') and x['choices'][0].get('text'):event.set()
        assert usage and usage['prompt_tokens']==length and usage['completion_tokens']==outputs,usage
        return usage
    def rpc(name,*values):return post('/collective_rpc',{'method':name,'args':list(values),'timeout':180})
    server=None;save()
    try:
        env=dict(os.environ,CAPSULE=str(root))
        with (root/'server.log').open('w') as log:
            server=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+1800
        while time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError(f'server exit{server.returncode}')
            try:
                with urllib.request.urlopen(url+'/health',timeout=2):break
            except OSError:time.sleep(2)
        else:raise TimeoutError('startup')
        # First-use compilation/capture costs stay outside the observed windows.
        request(8193,48)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i:request(1024,48,i),range(8)))
        # Three native windows first. Actual scheduler dispatch is the evidence,
        # not an assertion that the submitted workload forces every desired shape.
        labels=('long-cold','long-warm') if args.long_context_only else ('decode-c8','prefill-8k','mixed-c4')
        for label in labels:
            if label != 'long-warm':post('/reset_prefix_cache',{})
            if label != 'mixed-c4':
                # Trigger long prefill by actual computed context, not an
                # assumed chunk count (the final APC split changes that count).
                warmup={'decode-c8':8,'long-warm':4}.get(label,0)
                rpc('begin_dummy_profile',label,warmup,240000 if label=='long-cold' else 0)
            started=time.time_ns()
            if label.startswith('long-'):
                responses=[request(262016,128,20)]
                cached=responses[0]['prompt_tokens_details']['cached_tokens']
                assert cached == 0 if label=='long-cold' else cached > 0, responses
            elif label=='decode-c8':
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    responses=list(pool.map(lambda i:request(1024,128,i),range(8)))
            elif label=='prefill-8k':
                responses=[request(8193,1,i) for i in range(6)]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                    events=[threading.Event() for _ in range(4)]
                    leads=[pool.submit(request,1024,128,i,events[i]) for i in range(4)]
                    assert all(e.wait(180) for e in events),'lead requests did not begin decode'
                    rpc('begin_dummy_profile',label,0)
                    joined=pool.submit(request,2049,32,9)
                    responses=[f.result() for f in leads]+[joined.result()]
            rpc('end_dummy_profile')
            receipt['phases'].append({'label':label,'start_ns':started,'end_ns':time.time_ns(),'usage':responses});save()
        with urllib.request.urlopen(url+'/metrics',timeout=5) as response:(root/'metrics.txt').write_bytes(response.read())
        receipt['status']='CAPTURED'
    except BaseException as e:
        receipt.update(status='FAIL',error=f'{type(e).__name__}: {e}');raise
    finally:
        if server is not None:
            if server.poll() is None:
                os.killpg(server.pid,signal.SIGINT)
                try:server.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid,signal.SIGTERM)
                    try:server.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
                    receipt.update(status='FAIL',shutdown_timeout=True)
            receipt['server_exit_code']=server.returncode
            if server.returncode:receipt['status']='FAIL'
        save()
    if receipt['status']!='CAPTURED':raise RuntimeError('dummy capture failed')


if __name__=='__main__':main()
