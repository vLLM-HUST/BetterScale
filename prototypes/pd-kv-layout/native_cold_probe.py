"""D6-only, no Store or checkpoint calls: repeated fresh native cold requests."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import os
import uuid
from native_async_pool import worker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,choices=range(2,17),default=12)
    p.add_argument('--owner',type=int,choices=range(3),default=1)
    p.add_argument('--residency',choices=('natural','fixed-seat'))
    p.add_argument('--warm-prefix-control',action='store_true',help='Same-owner prefix then continuation, without PD transfer')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert not (a.warm_prefix_control and a.residency=='fixed-seat'),'Warm control requires natural hits'
    tokens=json.loads(a.prompt_receipt.read_text())['prompt_token_ids']
    if os.environ.get('BETTERSCALE_NUMERICAL_TRACE'):
        assert a.owner==1 and len(tokens)==281 and a.residency,'Trace requires owner1,281 tokens and diagnostic worker'
    (a.output/'input.json').write_text(json.dumps(dict(tokens=tokens,source=str(a.prompt_receipt),owner=a.owner)))
    ctx=mp.get_context('spawn');parent,child=ctx.Pipe()
    proc=ctx.Process(target=worker,args=('D',child,a.output),name='cold-D6')
    if a.residency:os.environ['BETTERSCALE_NUMERICAL_RESIDENCY']=a.residency
    proc.start();child.close();results=[]
    def receive(timeout):
        if not parent.poll(timeout):raise TimeoutError('Native cold probe deadline')
        status,data=parent.recv()
        if status=='error':raise RuntimeError(data)
        return data
    try:
        assert receive(900)=='D'
        for i in range(a.repeats):
            if a.warm_prefix_control:
                warm_salt='warm-'+uuid.uuid4().hex
                parent.send(('generate',dict(owner=a.owner,tokens=tokens[:-1],salt=warm_salt,n=1,diagnostic=True)))
                prefix=receive(240);assert prefix['cached']==0
                (a.output/f'prefix-{i}.json').write_text(json.dumps(prefix,indent=2))
                parent.send(('generate',dict(owner=a.owner,tokens=tokens,salt=warm_salt,n=16,diagnostic=True)))
                warm=receive(240);assert warm['cached']==len(tokens)-1
                (a.output/f'warm-{i}.json').write_text(json.dumps(warm,indent=2))
            salt='cold-'+uuid.uuid4().hex
            parent.send(('generate',dict(owner=a.owner,tokens=tokens,salt=salt,n=16,diagnostic=True)))
            result=receive(240);assert result['cached']==0
            if a.residency:
                parent.send(('observe',dict(owner=a.owner,salt=salt)))
                result['resident']=receive(30)
                if a.residency=='fixed-seat':assert result['resident']['seat']==0
            (a.output/f'cold-{i}.json').write_text(json.dumps(result,indent=2))
            results.append(result['token_ids']);print('COLD_REPEAT',i,result['token_ids'],flush=True)
        (a.output/'complete.json').write_text(json.dumps(dict(status='completed',scope='Native D6 repeatability; no P, checkpoint or Store',warm_prefix_control=a.warm_prefix_control,repeats=a.repeats,residency=a.residency,unique_token_sequences=len({tuple(x) for x in results})),indent=2))
    finally:
        try:parent.send(('stop',{}))
        except (BrokenPipeError,EOFError,OSError):pass
        proc.join(timeout=60)
        if proc.is_alive():proc.terminate();proc.join(timeout=10)
        parent.close()

if __name__=='__main__':main()
