"""Matched complete MOD FFN-call timings, real two-layer weights, two cards.

A1E1 leaf only: do not infer full-model E1 capacity or serving throughput.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(a):
    os.environ.update(BETTERSCALE_EXPERT_DRAFT_LAYERS='1',BETTERSCALE_EXPERT_MODEL=a.model,
        BETTERSCALE_EXPERT_PERSISTENT_BUILD=a.build,BETTERSCALE_EXPERT_FINE_PACK='0',
        BETTERSCALE_EXPERT_RETURN_MODE=a.return_mode,BETTERSCALE_EXPERT_GRAPH_BATCH='4')
    import torch
    import torch_npu
    torch.npu.set_device(0)
    from betterscale.patches.expert_service.placement import Placement
    placement=Placement('layer',1)
    if a.role=='server':
        from betterscale.patches.expert_service.persistent_server import serve
        receipt=serve(a.output/'control',placement,0,1,layers=(0,40))
        (a.output/'server.json').write_text(json.dumps(receipt,indent=2));return
    if a.client_plan:
        from client_plan import install
        install()
    from betterscale.patches.expert_service.persistent_remote import PersistentRemote
    from betterscale.patches.expert_service.checkpoint import Checkpoint,plain_experts
    remote=PersistentRemote(a.output/'control',placement,0);cp=Checkpoint()
    torch.manual_seed(20260924);results=[]
    for layer in [0,40]:
        weights=cp.experts(layer)
        for rows in a.rows:
            for pattern in a.patterns:
                if pattern.startswith('remote128'):torch.manual_seed(20260924)
                x=(torch.randn(rows,2048,device='npu')*.1).bfloat16()
                ids=(torch.rand(rows,256,device='npu').topk(8,dim=-1).indices if pattern=='random' else
                     (torch.arange(rows*8,device='npu').reshape(rows,8)%(8 if pattern=='hot8' else 128 if pattern.startswith('remote128') else 256)).long())
                if pattern.startswith('remote128'):ids=ids.add_(128).int()
                probs=torch.full((rows,8),.125,device='npu')
                if pattern=='remote128-sentinel':x[-1].zero_()
                remote.prepare_graph_rows(rows)
                expected=plain_experts(weights,x,ids,probs)
                remote(layer,x,ids,probs);torch.npu.synchronize()
                graph=torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    for _ in range(a.repeats):actual=remote(layer,x,ids,probs)
                graph.replay();torch.npu.synchronize()
                error=float(torch.linalg.vector_norm(actual.float()-expected.float())/torch.linalg.vector_norm(expected.float()).clamp_min(1e-12))
                assert torch.isfinite(actual).all() and error<.02,(layer,rows,pattern,error)
                times=[]
                for _ in range(5):
                    start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                    start.record();graph.replay();end.record();end.synchronize()
                    times.append(start.elapsed_time(end)*1000/a.repeats)
                # Mutate every input after timing; independent full-output oracle.
                x.mul_(.5)
                if pattern.startswith('remote128'):ids.add_(17).remainder_(128).add_(128)
                else:ids.add_(17).remainder_(256)
                probs.copy_(torch.softmax(torch.randn(rows,8,device='npu'),dim=-1))
                expected=plain_experts(weights,x,ids,probs)
                graph.replay();torch.npu.synchronize()
                changed_error=float(torch.linalg.vector_norm(actual.float()-expected.float())/torch.linalg.vector_norm(expected.float()).clamp_min(1e-12))
                assert torch.isfinite(actual).all() and changed_error<.02,changed_error
                row=dict(layer=layer,rows=rows,pattern=pattern,us=times,relative_l2=error,changed_l2=changed_error)
                results.append(row);print(json.dumps(row),flush=True);del graph
        del weights
    receipt=remote.receipt();remote.close()
    (a.output/'client.json').write_text(json.dumps(dict(status='PASS',results=results,receipt=receipt),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='/workspace/models/Qwen3.5-35B-A3B')
    p.add_argument('--devices',required=True);p.add_argument('--build',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--return-mode',choices=['pull','push'],default='pull')
    p.add_argument('--rows',type=int,nargs='+',default=[3,96,4096])
    p.add_argument('--patterns',nargs='+',choices=['hot8','broad','random','remote128','remote128-sentinel'],default=['hot8','broad'])
    p.add_argument('--client-plan',action='store_true')
    p.add_argument('--role',choices=['client','server']);p.add_argument('--repeats',type=int,default=16)
    a=p.parse_args();devices=list(map(int,a.devices.split(',')));assert len(devices)==2 and len(set(devices))==2
    if a.role:return worker(a)
    a.output.mkdir(exist_ok=False);(a.output/'control').mkdir(mode=0o700)
    jobs=[]
    def launch(role,device):
        with (a.output/f'{role}.log').open('x') as log:
            child=subprocess.Popen([sys.executable,__file__,*sys.argv[1:],'--role',role],stdout=log,stderr=subprocess.STDOUT,
                                   env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES=str(device)))
        jobs.append(child);return child
    try:
        server=launch('server',devices[1]);deadline=time.monotonic()+300
        while not (a.output/'control/e0.sock').exists():
            assert server.poll() is None,'server startup failure'
            assert time.monotonic()<deadline,'server startup timeout'
            time.sleep(.5)
        launch('client',devices[0]);deadline=time.monotonic()+600
        while any(job.poll() is None for job in jobs):
            assert all(job.poll() in (None,0) for job in jobs),'FFN worker failed'
            assert time.monotonic()<deadline,'FFN gate timeout'
            time.sleep(1)
        client=json.loads((a.output/'client.json').read_text());server=json.loads((a.output/'server.json').read_text())
        assert client['receipt']['peer_generations']['0']==server['completed']['0']
        (a.output/'result.json').write_text(json.dumps(dict(status='PASS',devices=devices,return_mode=a.return_mode,scope='real-weight two-layer A1E1 leaf; not full serving'),indent=2))
    finally:
        for job in jobs:
            if job.poll() is None:job.terminate()
        for job in jobs:
            try:job.wait(timeout=15)
            except subprocess.TimeoutExpired:job.kill();job.wait()

if __name__=='__main__':main()
