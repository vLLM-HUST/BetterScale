"""Two-process IPC transport lower bound. Run only under selected-device admission.

No weights/GEMM; same production pack/publish/collect/retire functions and wire.
Graph device-event timing, changed input exact echo, bounded parent lifecycle.
"""
import argparse
import ctypes as C
import json
import multiprocessing as mp
import os
from pathlib import Path
import time


def worker(role, device, channel, args):
    os.environ['ASCEND_RT_VISIBLE_DEVICES']=str(device)
    import torch
    import torch_npu
    from betterscale.patches.expert_service.ipc_acl import PrefixCopyACL
    from betterscale.patches.expert_service.runtime import LIB
    torch.npu.set_device(0)
    api=PrefixCopyACL(LIB)
    lib=C.CDLL(str(Path(args.build)/'launch.so'));p=C.c_void_p
    lib.load_server.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(p),C.POINTER(p)]
    lib.launch_blocks.argtypes=[p]*5+[C.c_uint32]
    lib.unload_server.argtypes=[p]
    binaries=[]; functions={}
    def call(name,cfg,x=None,ids=None,blocks=1):
        if name not in functions:
            binary,fn=p(),p()
            assert lib.load_server(str(Path(args.build)/'queue_service.o').encode(),name.encode(),C.byref(binary),C.byref(fn))==0
            binaries.append(binary);functions[name]=fn
        assert lib.launch_blocks(functions[name],torch.npu.current_stream().npu_stream,cfg.data_ptr(),x.data_ptr() if x is not None else 0,ids.data_ptr() if ids is not None else 0,blocks)==0
    channel.send(api.pid());peerpid=channel.recv()
    size=20<<20;local=api.allocate_staging(size)
    zero=torch.zeros(256,dtype=torch.int32,device='npu')
    api.copy(torch.npu.current_stream().npu_stream,local,zero.data_ptr(),1024);torch.npu.synchronize()
    key=api.export(local,size,(peerpid,));channel.send(key);peerkey=channel.recv();remote=api.import_memory(peerkey)
    if role=='server':
        ack=torch.zeros(16,16,dtype=torch.int32,device='npu')
        cfg=torch.tensor([remote,local,ack.data_ptr()],dtype=torch.int64,device='npu')
        torch.npu.synchronize();call('transport_echo',cfg,blocks=args.server_blocks)
        channel.send('ready');torch.npu.synchronize()
        channel.send({'completed':ack.cpu().tolist()[0][0]})
    else:
        assert channel.recv()=='ready'
        counter=torch.zeros(8,dtype=torch.int32,device='npu');results=[]
        for rows in args.rows:
            x=torch.zeros(rows,2048,dtype=torch.bfloat16,device='npu')
            ids=torch.zeros(max(8,rows*8),dtype=torch.int32,device='npu')
            probs=torch.zeros(max(16,rows*8+16),dtype=torch.bfloat16,device='npu');out=torch.empty_like(x)
            cfg=torch.tensor([local,remote,0,0,0,0,rows,counter.data_ptr(),1,20000000,0,probs.data_ptr(),out.data_ptr(),0,250,0],dtype=torch.int64,device='npu')
            for mode in args.modes:
                def step():
                    call('neural_pack',cfg,x,ids,16)
                    call('neural_publish',cfg)
                    if mode=='split':
                        call('transport_wait',cfg)
                        call('transport_copy',cfg,blocks=16)
                    elif mode=='pipeline':call('transport_collect_pipeline',cfg,blocks=16)
                    else:call('neural_collect_reduced',cfg,blocks=int(mode))
                    call('neural_retire',cfg)
                step();torch.npu.synchronize()
                graph=torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    for _ in range(args.repeats):step()
                graph.replay();torch.npu.synchronize()
                times=[]
                for sample in range(args.samples):
                    # BF16-exact per-element changes catch stale/partial returned data.
                    x.copy_(((torch.arange(rows*2048,device='npu',dtype=torch.int32)+sample*17)%127).reshape(rows,2048).to(torch.bfloat16))
                    torch.npu.synchronize()
                    start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                    start.record();graph.replay();end.record();end.synchronize()
                    times.append(start.elapsed_time(end)*1000/args.repeats)
                    assert torch.equal(out,x),(rows,mode,sample)
                assert counter.cpu()[0].item()>0
                row=dict(rows=rows,collect=mode,server_blocks=args.server_blocks,roundtrip_us=times,principal_bytes=rows*8192)
                print(json.dumps(row),flush=True);results.append(row)
                del graph
        copy_results=[]
        # Remote server is idle at last READY; no new generation published here.
        # Copy into local+256 does not touch READY/descriptor at local+[0,32].
        for rows in [3,96,512,4096]:
            payload=torch.arange(rows*2048,device='npu',dtype=torch.int32).remainder(127).to(torch.bfloat16).reshape(rows,2048)
            api.copy(torch.npu.current_stream().npu_stream,local+256,payload.data_ptr(),rows*4096)
            torch.npu.synchronize()
            output=torch.empty_like(payload)
            for location,address in [('local',local),('remote',remote)]:
                cfg=torch.tensor([local,address,0,0,0,0,rows,counter.data_ptr(),1,20000000,0,probs.data_ptr(),output.data_ptr(),0,250,0],dtype=torch.int64,device='npu')
                for kernel in ['transport_copy','transport_copy_pipeline']:
                    for blocks in [1,4,16]:
                        call(kernel,cfg,blocks=blocks);torch.npu.synchronize()
                        graph=torch.npu.NPUGraph()
                        with torch.npu.graph(graph):
                            for _ in range(args.repeats):call(kernel,cfg,blocks=blocks)
                        graph.replay();torch.npu.synchronize();times=[]
                        for _ in range(args.samples):
                            start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                            start.record();graph.replay();end.record();end.synchronize()
                            times.append(start.elapsed_time(end)*1000/args.repeats)
                        expected=payload if location=='local' else x[:rows]
                        assert torch.equal(output,expected),(rows,location,kernel,blocks)
                        row=dict(rows=rows,location=location,kernel=kernel,blocks=blocks,copy_us=times,bytes=rows*4096)
                        print(json.dumps(row),flush=True);copy_results.append(row)
                        del graph
        direction_results=[]
        for rows in [3,96,512,4096]:
            payload=torch.arange(rows*2048,device='npu',dtype=torch.int32).remainder(113).to(torch.bfloat16).reshape(rows,2048)
            api.copy(torch.npu.current_stream().npu_stream,local+256,payload.data_ptr(),rows*4096)
            torch.npu.synchronize()
            output=torch.empty_like(payload)
            cfg=torch.tensor([local,local,0,0,0,0,rows,counter.data_ptr(),1,20000000,0,probs.data_ptr(),remote+256,0,250,0],dtype=torch.int64,device='npu')
            for kernel in ['transport_copy','transport_copy_pipeline','sdma']:
                for blocks in ([1] if kernel=='sdma' else [1,4,16]):
                    def push():
                        if kernel=='sdma':api.copy(torch.npu.current_stream().npu_stream,remote+256,local+256,rows*4096)
                        else:call(kernel,cfg,blocks=blocks)
                    push();torch.npu.synchronize();graph=torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        for _ in range(args.repeats):push()
                    graph.replay();torch.npu.synchronize();times=[]
                    for _ in range(args.samples):
                        start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                        start.record();graph.replay();end.record();end.synchronize()
                        times.append(start.elapsed_time(end)*1000/args.repeats)
                    api.copy(torch.npu.current_stream().npu_stream,output.data_ptr(),remote+256,rows*4096)
                    torch.npu.synchronize();assert torch.equal(output,payload)
                    row=dict(rows=rows,direction='push',kernel=kernel,blocks=blocks,copy_us=times,bytes=rows*4096)
                    print(json.dumps(row),flush=True);direction_results.append(row);del graph
            # Same API and memory endpoints in the reverse direction.
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(args.repeats):api.copy(torch.npu.current_stream().npu_stream,output.data_ptr(),remote+256,rows*4096)
            graph.replay();torch.npu.synchronize();times=[]
            for _ in range(args.samples):
                start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                start.record();graph.replay();end.record();end.synchronize()
                times.append(start.elapsed_time(end)*1000/args.repeats)
            assert torch.equal(output,payload)
            row=dict(rows=rows,direction='pull',kernel='sdma',blocks=1,copy_us=times,bytes=rows*4096)
            print(json.dumps(row),flush=True);direction_results.append(row);del graph
        count=int(counter.cpu()[0]);stop=torch.full((8,),-(count+1),dtype=torch.int32,device='npu')
        api.copy(torch.npu.current_stream().npu_stream,local,stop.data_ptr(),32);torch.npu.synchronize()
        server=channel.recv();assert server['completed']==count,(server,count)
        Path(args.output).write_text(json.dumps(dict(status='PASS',devices=args.devices,kind='echo-not-FFN',repeats=args.repeats,samples=args.samples,generations=count,results=results,copy_results=copy_results,direction_results=direction_results),indent=2)+'\n')
    # Both streams drained before either mapping/allocation is released.
    channel.send('drained');assert channel.recv()=='drained'
    api.close_mapping(peerkey)
    channel.send('unmapped');assert channel.recv()=='unmapped'
    api.close_mapping(key);api.free_staging(local)
    for binary in binaries:assert lib.unload_server(binary)==0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices',default='4,5');parser.add_argument('--build',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--server-blocks',type=int,choices=[1,16],default=16)
    parser.add_argument('--rows',type=int,nargs='+',default=[0,3,16,96,512,4096])
    parser.add_argument('--modes',nargs='+',choices=['1','16','split','pipeline'],default=['16','pipeline','split','1'])
    parser.add_argument('--repeats',type=int,default=64);parser.add_argument('--samples',type=int,default=5)
    args=parser.parse_args();assert args.rows[-1]==4096;devices=list(map(int,args.devices.split(',')));assert len(devices)==2 and len(set(devices))==2
    ctx=mp.get_context('spawn');left,right=ctx.Pipe()
    jobs=[ctx.Process(target=worker,args=(role,dev,ch,args)) for role,dev,ch in zip(('client','server'),devices,(left,right))]
    for job in jobs:job.start()
    try:
        deadline=time.monotonic()+600
        while any(job.is_alive() for job in jobs):
            if any(job.exitcode not in (None,0) for job in jobs):raise RuntimeError('transport child failed')
            if time.monotonic()>deadline:raise TimeoutError('transport gate exceeded 600s')
            time.sleep(.5)
        assert all(job.exitcode==0 for job in jobs)
    finally:
        for job in jobs:
            if job.is_alive():job.terminate()
        for job in jobs:job.join(10)
        for job in jobs:
            if job.is_alive():job.kill();job.join()

if __name__=='__main__':main()
