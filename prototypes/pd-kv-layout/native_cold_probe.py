"""D6-only, no Store or checkpoint calls: repeated fresh native cold requests."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import uuid
from native_async_pool import worker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,choices=range(2,17),default=12)
    p.add_argument('--owner',type=int,choices=range(3),default=1)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    tokens=json.loads(a.prompt_receipt.read_text())['prompt_token_ids']
    (a.output/'input.json').write_text(json.dumps(dict(tokens=tokens,source=str(a.prompt_receipt),owner=a.owner)))
    ctx=mp.get_context('spawn');parent,child=ctx.Pipe()
    proc=ctx.Process(target=worker,args=('D',child,a.output),name='cold-D6')
    proc.start();child.close();results=[]
    def receive(timeout):
        if not parent.poll(timeout):raise TimeoutError('Native cold probe deadline')
        status,data=parent.recv()
        if status=='error':raise RuntimeError(data)
        return data
    try:
        assert receive(900)=='D'
        for i in range(a.repeats):
            parent.send(('generate',dict(owner=a.owner,tokens=tokens,salt='cold-'+uuid.uuid4().hex,n=16,diagnostic=True)))
            result=receive(240);assert result['cached']==0
            (a.output/f'cold-{i}.json').write_text(json.dumps(result,indent=2))
            results.append(result['token_ids']);print('COLD_REPEAT',i,result['token_ids'],flush=True)
        (a.output/'complete.json').write_text(json.dumps(dict(status='completed',scope='Native D6 cold requests only; no P, checkpoint or Store',repeats=a.repeats,unique_token_sequences=len({tuple(x) for x in results})),indent=2))
    finally:
        try:parent.send(('stop',{}))
        except (BrokenPipeError,EOFError,OSError):pass
        proc.join(timeout=60)
        if proc.is_alive():proc.terminate();proc.join(timeout=10)
        parent.close()

if __name__=='__main__':main()
