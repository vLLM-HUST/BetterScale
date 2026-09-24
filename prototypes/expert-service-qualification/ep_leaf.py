"""Bounded real-weight EP leaf: changing routes, empty owners and FULL replay.

Run under external admission; every child is scoped to exactly one physical NPU.
No model server or benchmark throughput claim is made by this leaf.
"""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

def worker(a):
    from betterscale.patches.expert_service.config import ServiceConfig
    c=ServiceConfig(str(a.output/'control'),a.build,a.owners,a.sources,a.source,1,placement='expert')
    c.validate();c.check_build();c.bind(a.model,96)
    import torch,torch_npu
    assert torch.npu.device_count()==1
    torch.npu.set_device(0)
    from betterscale.patches.expert_service.placement import Placement
    placement=Placement('expert',a.owners)
    if a.owner>=0:
        from betterscale.patches.expert_service.persistent_server import serve
        result=serve(a.output/'control',placement,a.owner,a.sources,layers=(0,40))
        (a.output/f'owner{a.owner}.json').write_text(json.dumps(result,indent=2));return
    from betterscale.patches.expert_service.persistent_remote import PersistentRemote
    from betterscale.patches.expert_service.checkpoint import Checkpoint,plain_experts
    torch.manual_seed(20260924+a.source);cp=Checkpoint();remote=PersistentRemote(a.output/'control',placement,a.source)
    results=[]
    for layer in (0,40):
        weights=cp.experts(layer)
        for rows,pattern in [(1,'owner0'),(3,'last'),(33,'boundary'),(96,'spread'),(4096,'owner0'),(4096,'spread')]:
            x=(torch.randn(rows,2048)*.1).bfloat16().npu()
            ids=torch.arange(rows*8).reshape(rows,8)
            if pattern=='owner0':ids%=8
            elif pattern=='last':ids=248+ids%8
            elif pattern=='boundary':ids=(ids+256//a.owners-4)%256
            else:ids%=256
            ids=ids.npu();probs=torch.full((rows,8),.125,device='npu')
            if a.sources>1:
                p=torch.rand(rows,8);p/=p.sum(-1,keepdim=True);probs=p.bfloat16().npu()
            before=[v.clone() for v in (x,ids,probs)]
            expected=plain_experts(weights,x,ids,probs);actual=remote(layer,x,ids,probs)
            torch.npu.synchronize()
            err=float(torch.linalg.vector_norm(actual.float()-expected.float())/torch.linalg.vector_norm(expected.float()).clamp_min(1e-12))
            assert torch.isfinite(actual).all() and err<=.02,(layer,rows,pattern,err)
            assert all(torch.equal(v,b) for v,b in zip((x,ids,probs),before))
            results.append(dict(layer=layer,rows=rows,pattern=pattern,relative_l2=err,mode='eager'))
        # Changed inputs/routes under one exact captured 3-row shape.
        remote.prepare_graph_rows(3)
        x=torch.zeros(3,2048,dtype=torch.bfloat16,device='npu');ids=torch.zeros(3,8,dtype=torch.int64,device='npu');probs=torch.full((3,8),.125,device='npu')
        remote(layer,x,ids,probs);torch.npu.synchronize()
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph):actual=remote(layer,x,ids,probs)
        for i in range(8):
            x.copy_((torch.randn(3,2048)*.1).bfloat16().npu())
            ids.copy_(((torch.arange(24).reshape(3,8)+i*37)%256).npu())
            expected=plain_experts(weights,x,ids,probs);graph.replay();torch.npu.synchronize()
            err=float(torch.linalg.vector_norm(actual.float()-expected.float())/torch.linalg.vector_norm(expected.float()).clamp_min(1e-12))
            assert torch.isfinite(actual).all() and err<=.02,(layer,i,err)
            results.append(dict(layer=layer,rows=3,generation=i,relative_l2=err,mode='FULL'))
        del graph,weights
    if a.sources>1:
        # Protocol-only colocation: two client processes share one physical card.
        # Freeze a broad4096 frame, rendezvous once, then issue independently.
        x=(torch.randn(4096,2048)*.1).bfloat16().npu()
        ids=((torch.arange(4096*8).reshape(4096,8)+a.source*31)%256).npu()
        p=torch.rand(4096,8);p/=p.sum(-1,keepdim=True);probs=p.bfloat16().npu()
        weights=cp.experts(40);expected=plain_experts(weights,x,ids,probs)
        remote.prepare_graph_rows(4096);remote(40,x,ids,probs);torch.npu.synchronize()
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph):actual=remote(40,x,ids,probs)
        (a.output/f'burst-ready{a.source}').touch()
        deadline=time.monotonic()+300
        while not all((a.output/f'burst-ready{i}').exists() for i in range(a.sources)):
            assert time.monotonic()<deadline,'burst rendezvous timeout';time.sleep(.01)
        for _ in range(32):graph.replay()
        torch.npu.synchronize()
        err=float(torch.linalg.vector_norm(actual.float()-expected.float())/torch.linalg.vector_norm(expected.float()).clamp_min(1e-12))
        assert torch.isfinite(actual).all() and err<=.02,err
        results.append(dict(layer=40,rows=4096,relative_l2=err,mode='FULL-burst-final',replays=32))
        del graph,weights
    receipt=remote.receipt();remote.close()
    (a.output/f'client{a.source}.json').write_text(json.dumps(dict(status='PASS',cases=results,receipt=receipt),indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--output',type=Path,required=True);p.add_argument('--build',required=True);p.add_argument('--owners',type=int,choices=(2,4),required=True);p.add_argument('--devices',required=True,help='Explicit physical client,owner0,... device IDs');p.add_argument('--sources',type=int,choices=(1,2),default=1);p.add_argument('--shared-client-device',action='store_true');p.add_argument('--source',type=int,default=0);p.add_argument('--worker',action='store_true');p.add_argument('--owner',type=int,default=-1);a=p.parse_args()
    devices=[int(x) for x in a.devices.split(',')]
    if len(devices)!=a.owners+1 or len(set(devices))!=len(devices) or not set(devices)<=set(range(8)):
        p.error('devices must name distinct physical cards for client and every owner')
    if a.sources>1 and not a.shared_client_device:p.error('two clients require explicit --shared-client-device in this bounded leaf')
    if not 0<=a.source<a.sources:p.error('invalid source')
    if a.worker:return worker(a)
    a.output.mkdir(exist_ok=False);(a.output/'control').mkdir(mode=0o700)
    children=[]
    try:
        for owner in range(a.owners):
            cmd=[sys.executable,__file__,a.model,'--output',str(a.output),'--build',a.build,
                 '--owners',str(a.owners),'--devices',a.devices,'--sources',str(a.sources),
                 *(['--shared-client-device'] if a.shared_client_device else []),
                 '--worker','--owner',str(owner)]
            with (a.output/f'owner{owner}.log').open('x') as f:
                children.append(subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,
                    env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES=str(devices[owner+1]))))
        deadline=time.monotonic()+300
        while not all((a.output/f'control/e{i}.sock').exists() for i in range(a.owners)):
            assert all(c.poll() is None for c in children),'expert exited before listen'
            assert time.monotonic()<deadline,'startup timeout';time.sleep(.5)
        for source in range(a.sources):
            cmd=[sys.executable,__file__,a.model,'--output',str(a.output),'--build',a.build,
                 '--owners',str(a.owners),'--devices',a.devices,'--sources',str(a.sources),
                 *(['--shared-client-device'] if a.shared_client_device else []),'--source',str(source),'--worker']
            with (a.output/f'client{source}.log').open('x') as f:
                child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,
                                       env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES=str(devices[0])))
            children.append(child)
        deadline=time.monotonic()+900
        while any(c.poll() is None for c in children):
            assert all(c.poll() in (None,0) for c in children),'worker failed'
            assert time.monotonic()<deadline,'leaf timeout';time.sleep(1)
        assert all(c.returncode==0 for c in children),'worker failed'
        for owner in range(a.owners):
            receipt=json.loads((a.output/f'owner{owner}.json').read_text())
            for source in range(a.sources):
                client=json.loads((a.output/f'client{source}.json').read_text())
                assert receipt['completed'][str(source)]==client['receipt']['peer_generations'][str(owner)]
            if a.sources>1:
                assert sum(receipt['completed_counts'])>receipt['waves'],'no observed source coalescing'
        (a.output/'PASS').write_text('exact generations and all roles exit0\n')
    finally:
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=15)
            except subprocess.TimeoutExpired:c.kill();c.wait()
if __name__=='__main__':main()
