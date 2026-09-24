"""Two physical client cards and one owner: bounded real-weight co-batching.

Only layers0/40 are loaded. This is not a full-model E1 deployment and does not
relax the production topology admission. No serving throughput claim is made.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def wait_for(path, timeout=300):
    deadline=time.monotonic()+timeout
    while not path.exists():
        if time.monotonic()>deadline:raise TimeoutError(str(path))
        time.sleep(.01)


def worker(a):
    # Leaf-only two-layer geometry: full41-layer E1 would not fit. The packaged
    # source/ABI still contains41 physical layer IDs and the unchanged runtime.
    abi=json.loads((Path(a.build)/'abi.json').read_text())
    assert abi['placement']=='layer' and abi['layer_count']==41
    assert abi['sources_per_wave'] in (1,7) and abi['combined_return']
    os.environ.update(BETTERSCALE_EXPERT_DRAFT_LAYERS='1',BETTERSCALE_EXPERT_MODEL=a.model,
        BETTERSCALE_EXPERT_PERSISTENT_BUILD=a.build,BETTERSCALE_EXPERT_FINE_PACK='0',
        BETTERSCALE_EXPERT_RETURN_MODE='pull',BETTERSCALE_EXPERT_GRAPH_BATCH='4')
    import torch,torch_npu
    assert torch.npu.device_count()==1
    torch.npu.set_device(0)
    from betterscale.patches.expert_service.placement import Placement
    placement=Placement('layer',1)
    if a.role=='owner':
        from betterscale.patches.expert_service.persistent_server import serve
        receipt=serve(a.output/'control',placement,0,2,layers=(0,40))
        (a.output/'owner.json').write_text(json.dumps(receipt,indent=2));return
    from betterscale.patches.expert_service.persistent_remote import PersistentRemote
    from betterscale.patches.expert_service.checkpoint import Checkpoint,plain_experts
    torch.manual_seed(20260924+a.source)
    cp=Checkpoint();remote=PersistentRemote(a.output/'control',placement,a.source)
    results=[]
    for layer in (0,40):
        weights=cp.experts(layer)
        for rows in (3,127,4096):
            x=torch.zeros(rows,2048,dtype=torch.bfloat16,device='npu')
            ids=torch.zeros(rows,8,dtype=torch.int64,device='npu')
            probs=torch.full((rows,8),.125,device='npu')
            remote.prepare_graph_rows(rows)
            remote(layer,x,ids,probs);torch.npu.synchronize()
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph):actual=remote(layer,x,ids,probs)
            # Unequal sources and repeated routes; mutate the same captured pointers.
            for case in range(3):
                x.copy_((torch.randn(rows,2048)*.1).bfloat16().npu())
                pattern=(torch.arange(rows*8).reshape(rows,8)+a.source*37+case*71)%256
                if case==2:pattern%=8
                ids.copy_(pattern.npu())
                p=torch.rand(rows,8);p/=p.sum(-1,keepdim=True)
                probs.copy_(p.npu())
                before=[v.clone() for v in (x,ids,probs)]
                expected=plain_experts(weights,x,ids,probs)
                graph.replay();torch.npu.synchronize()
                error=float(torch.linalg.vector_norm(actual.float()-expected.float())/torch.linalg.vector_norm(expected.float()).clamp_min(1e-12))
                assert torch.isfinite(actual).all() and error<=.02,(layer,rows,case,error)
                assert all(torch.equal(v,b) for v,b in zip((x,ids,probs),before))
                results.append(dict(layer=layer,rows=rows,case=case,relative_l2=error))
            # Start together once, then independent back-to-back graphs, no per-wave
            # client rendezvous or deliberate server batching delay.
            stem=f'{layer}-{rows}'
            (a.output/f'{stem}-ready{a.source}').touch()
            wait_for(a.output/f'{stem}-ready{1-a.source}')
            for _ in range(32):graph.replay()
            torch.npu.synchronize()
            error=float(torch.linalg.vector_norm(actual.float()-expected.float())/torch.linalg.vector_norm(expected.float()).clamp_min(1e-12))
            assert torch.isfinite(actual).all() and error<=.02,error
            del graph
        del weights
    receipt=remote.receipt()
    # The owner drains all registered sources; clients must close concurrently.
    (a.output/f'close-ready{a.source}').touch();wait_for(a.output/f'close-ready{1-a.source}')
    remote.close()
    (a.output/f'client{a.source}.json').write_text(json.dumps(dict(status='PASS',cases=results,receipt=receipt),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model');p.add_argument('--build',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--devices',required=True,help='physical client0,client1,owner')
    p.add_argument('--role',choices=('client','owner'));p.add_argument('--source',type=int,default=0)
    a=p.parse_args();devices=[int(x) for x in a.devices.split(',')]
    if len(devices)!=3 or len(set(devices))!=3 or not set(devices)<=set(range(8)):p.error('three distinct physical devices required')
    if a.role:return worker(a)
    a.output.mkdir(exist_ok=False);(a.output/'control').mkdir(mode=0o700)
    children=[]
    def launch(role,source,device):
        cmd=[sys.executable,__file__,a.model,'--build',a.build,'--output',str(a.output),
             '--devices',a.devices,'--role',role,'--source',str(source)]
        with (a.output/f'{role}{source}.log').open('x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
                                   env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES=str(device)))
        children.append(child);return child
    try:
        owner=launch('owner',0,devices[2])
        deadline=time.monotonic()+300
        while not (a.output/'control/e0.sock').exists():
            assert owner.poll() is None,'owner exited before listen'
            assert time.monotonic()<deadline,'owner startup timeout'
            time.sleep(.5)
        clients=[launch('client',i,devices[i]) for i in range(2)]
        deadline=time.monotonic()+900
        while any(c.poll() is None for c in children):
            assert all(c.poll() in (None,0) for c in children),'worker failed'
            assert time.monotonic()<deadline,'leaf timeout'
            time.sleep(1)
        assert all(c.returncode==0 for c in children),'worker failed'
        receipt=json.loads((a.output/'owner.json').read_text())
        for i in range(2):
            client=json.loads((a.output/f'client{i}.json').read_text())
            assert receipt['completed'][str(i)]==client['receipt']['peer_generations']['0']
        frames=sum(receipt['completed_counts']);waves=receipt['waves']
        assert waves<=frames<=2*waves,(frames,waves)
        cap=receipt['sources_per_wave']
        assert (frames==waves if cap==1 else frames>waves),(cap,frames,waves)
        result=dict(status='PASS',cap=cap,source_frames=frames,server_waves=waves,
                    two_source_waves=frames-waves,devices=devices,scope='two-layer real-weight FULL graph leaf; no throughput claim')
        (a.output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    finally:
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=15)
            except subprocess.TimeoutExpired:c.kill();c.wait()

if __name__=='__main__':main()
