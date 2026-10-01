"""Retired model pages -> two pinned slots -> acknowledged immutable Store chunks.

Core's synchronous idle RPC owns page lifetime throughout. This overlaps only
host transfer stages, NOT compute. Event completion gates Store reads; Store
completion gates slot reuse. No background job outlives the RPC, including failure.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys,time


def export_dense(runner,root,rank,header,*,lifetime=None,expected_dense=None):
    if lifetime is None:lifetime={}
    lifetime['drained']=True
    import torch
    from betterscale.live.llm.qwen35.state import GDNState
    from model_checkpoint import dense_span
    from dram_store_fixture import CPU_ENV
    options=header['stream_store'];prefix=options['prefix'];ports=options['ports']
    if (not isinstance(prefix,str) or not prefix.startswith('pd/') or len(ports)!=2
            or any(type(p) is not int or not 55421<=p<=55428 for p in ports)):
        raise ValueError('Unqualified model Store endpoint/namespace')
    site=CPU_ENV/'lib/python3.12/site-packages';sys.path.insert(0,str(site))
    import mooncake.store
    from mooncake.store import MooncakeDistributedStore,ReplicateConfig
    if not Path(mooncake.store.__file__).resolve().is_relative_to(site):
        raise RuntimeError('Wrong Mooncake runtime in model worker')
    planes=[(f'{name}/{kind}/head{rank}',getattr(leaf,kind).tensor)
            for name,leaf in sorted(root.target.items()) if not isinstance(leaf,GDNState)
            for kind in ('key','value')]
    if len(planes)!=20:raise ValueError('Expected twenty local dense planes')
    first=header['dense_start'];cursor=header['cursor'];chunk_tokens=2048
    if not 0<=first<=cursor:raise ValueError('Invalid stream frontier')
    size=20*chunk_tokens*512
    slots=[torch.empty(size,dtype=torch.uint8,pin_memory=True) for _ in range(2)]
    # Retain gathered device tensors until their slot's DMA and Store are done.
    retained=[[],[]];pending=[None,None];registered=[];chunks=[]
    stream=torch.npu.Stream(device=runner.device)
    indices=torch.tensor(header['blocks'],dtype=torch.int64,device=runner.device)
    stream.wait_stream(torch.npu.current_stream(runner.device))
    client=MooncakeDistributedStore();pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='pd-model-store')
    config=ReplicateConfig();config.replica_num=1
    started=time.perf_counter();concurrent_enqueues=0
    def put(event,slot,keys,nbytes):
        event.synchronize()
        pointers=[[slots[slot].data_ptr()+i*chunk_tokens*512] for i in range(20)]
        rc=client.batch_put_from_multi_buffers(list(keys.values()),pointers,[[nbytes]]*20,config)
        if rc!=[0]*20:raise IOError(f'Model dense Store put failed: {rc}')
        if expected_dense is not None:
            for name,key in keys.items():
                begin=int(key.rsplit('/dense/',1)[1].split('/',1)[0])-first
                expected=expected_dense[name][begin:begin+nbytes//512].view(torch.uint8).numpy().tobytes()
                if bytes(client.get(key))!=expected:raise RuntimeError(f'Concurrent dense Store bytes differ: {name}, token {begin+first}')
    try:
        rc=client.setup(f'127.0.0.1:{ports[rank]}','http://127.0.0.1:55402/metadata',
                        0,64*1024**2,'tcp','', '127.0.0.1:55401')
        if rc!=0:raise RuntimeError(f'Model dense Store setup failed: {rc}')
        for slot in slots:
            if client.register_buffer(slot.data_ptr(),size)!=0:raise RuntimeError('Model pinned registration failed')
            registered.append(slot)
        for index,start in enumerate(range(first,cursor,chunk_tokens)):
            slot=index%2
            if pending[slot] is not None:pending[slot].result()
            retained[slot]=[];stop=min(start+chunk_tokens,cursor);nbytes=(stop-start)*512
            keys={name:f'{prefix}/dense/{start}/{name}' for name,_ in planes}
            if any(f is not None and not f.done() for f in pending):concurrent_enqueues+=1
            lifetime['drained']=False
            lifetime['retained']=(slots,retained,indices,stream,client)
            with torch.npu.stream(stream):
                for i,(_,tensor) in enumerate(planes):
                    source=dense_span(tensor,indices,header['block_size'],stop,start)
                    retained[slot].append(source)
                    slots[slot][i*chunk_tokens*512:i*chunk_tokens*512+nbytes].view(torch.bfloat16).view(stop-start,1,256).copy_(source,non_blocking=True)
                event=torch.npu.Event();event.record(stream)
            pending[slot]=pool.submit(put,event,slot,keys,nbytes)
            chunks.append(dict(start=start,stop=stop,keys=keys))
        for future in pending:
            if future is not None:future.result()
        return dict(acknowledged=True,chunks=chunks,streams=[n for n,_ in planes],
                    dense_bytes=(cursor-first)*20*512,ring_slots=2,
                    copy_enqueued_while_store_pending=concurrent_enqueues,
                    seconds=time.perf_counter()-started,exact_transfer_oracle=expected_dense is not None)
    finally:
        # Even failed Store work must stop reading before unregister/free.
        pool.shutdown(wait=True)
        stream.synchronize()
        lifetime['drained']=True
        for slot in reversed(registered):client.unregister_buffer(slot.data_ptr())
        client.close()
        lifetime.pop('retained',None)
