"""Exercise the shipped HTTP entry with retained OpenCompass tokenized inputs.

Run under the repository's shared-host NPU lease/admission supervisor. This
script owns only its child server; it never installs packages or edits donors.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    engine=args.output/'engine';engine.mkdir(exist_ok=True)
    artifacts=args.output/'service'
    env=dict(os.environ,STRENGTHEN_PYTHON=sys.executable)
    entry=args.package/'bin/strengthen-dsv4'
    command=[str(entry),'serve','--model',args.model,'--profile',args.profile,
             '--artifacts',str(artifacts),'--port',str(args.port),'--host','127.0.0.1']
    base=f'http://127.0.0.1:{args.port}'
    def post(payload):
        request=urllib.request.Request(base+'/v1/completions',data=json.dumps(payload).encode(),
                                       headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=180) as response:
            return json.load(response)
    log=(args.output/'server.log').open('w')
    server=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+660
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Server exited before readiness: {server.returncode}; see server.log')
            try:
                with urllib.request.urlopen(base+'/health',timeout=2) as response:
                    if response.status==200:break
            except (urllib.error.URLError,TimeoutError):pass
            if time.monotonic()>deadline:raise TimeoutError('HTTP readiness exceeded660 seconds')
            time.sleep(3)
        ready=[json.loads(p.read_text()) for p in sorted(artifacts.glob('ready-rank*.json'))]
        assert len(ready)==8 and {r['rank'] for r in ready}==set(range(8))
        assert all(r['profile']==args.profile and r['max_length_concurrency']>=4 for r in ready)
        requests=json.loads(args.requests.read_text())
        assert len(requests)==32
        (engine/'quality-inputs.json').write_text(json.dumps(requests))
        started=time.monotonic();results=[]
        for start in range(0,len(requests),4):
            group=requests[start:start+4]
            assert len({r['max_new_tokens'] for r in group})==1
            assert len({r['eos_token_id'] for r in group})==1
            reply=post(dict(model='dsv4',prompt=[r['prompt_token_ids'] for r in group],temperature=0,
                max_tokens=group[0]['max_new_tokens'],stop_token_ids=[group[0]['eos_token_id']],
                ignore_eos=False,return_token_ids=True))
            choices=sorted(reply['choices'],key=lambda c:c['index'])
            assert len(choices)==4 and [x['index'] for x in choices]==list(range(4))
            for request,choice in zip(group,choices):
                assert isinstance(choice['token_ids'],list) and len(choice['token_ids'])<=request['max_new_tokens']
                results.append(dict(request_id=request['request_id'],prompt_tokens=len(request['prompt_token_ids']),
                    token_ids=choice['token_ids'],finish_reason=choice['finish_reason'],stop_reason=choice.get('stop_reason')))
            (engine/'quality-partial.json').write_text(json.dumps(results,indent=2))
            print(f'HTTP quality completed {len(results)}/32',flush=True)
        # Exercise streaming through the same public endpoint, not a private RPC.
        req=urllib.request.Request(base+'/v1/completions',data=json.dumps(dict(model='dsv4',
            prompt=[17]*64,temperature=0,max_tokens=16,ignore_eos=True,stream=True)).encode(),
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=120) as response:
            stream=response.read().decode()
        assert 'data: [DONE]' in stream and '"choices"' in stream
        (engine/'quality-result.json').write_text(json.dumps(dict(status='COMPLETED_UNSCORED',
            scope='Shipped HTTP entry: retained32 OpenCompass LongBench English retrieval items',
            elapsed_seconds=time.monotonic()-started,requests=results,streaming_completed=True,
            receipts=ready),indent=2))
    finally:
        if server.poll() is None:
            server.terminate()
            try:server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                server.kill();server.wait(timeout=10)
        log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--requests',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--port',type=int,default=30880)
    p.add_argument('--profile',choices=['baseline','optimized'],default='optimized')
    main(p.parse_args())
