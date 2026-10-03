"""Store descriptor ingress: dense payload bypasses controller/Core RPC."""
import json
import msgspec
from model_checkpoint import IDENTITY
from model_store import tensor
from session import CacheMiss


def validate_chunks(chunks,cursor,streams):
    frontier=0;seen=set()
    if not chunks or len(set(streams))!=len(streams):raise ValueError('Invalid dense stream plan')
    for chunk in chunks:
        start,stop,keys=chunk['start'],chunk['stop'],chunk['keys']
        if (type(start) is not int or type(stop) is not int or start!=frontier
                or not start<stop<=min(start+2048,cursor) or set(keys)!=set(streams)
                or any(not isinstance(k,str) or not k for k in keys.values())
                or len(set(keys.values()))!=len(keys) or seen.intersection(keys.values())):
            raise ValueError('Invalid dense chunk dependency')
        seen.update(keys.values());frontier=stop
    if frontier!=cursor:raise ValueError('Incomplete dense frontier')


def load_plan(objects,key,identity,streams,ports,*,verify=False):
    """Stage target checkpoint blobs only; workers validate every dense read."""
    try:
        manifest=json.loads(objects.get(key));cursor=manifest['cursor']
        if (identity!=IDENTITY or manifest['schema'] not in (1,2) or manifest['identity']!=identity
                or manifest['streams']!=list(streams) or manifest['token_bytes']!=512
                or type(cursor) is not int or not 0<cursor<=8192
                or len(manifest['tokens'])!=cursor or len(streams)!=40
                or any(type(t) is not int or t<0 for t in manifest['tokens'])
                or type(manifest['pending_token']) is not int or manifest['pending_token']<0):
            raise ValueError('Invalid target manifest')
        validate_chunks(manifest['chunks'],cursor,streams)
        if manifest['schema']==2:
            from model_store_checkpoint import FORMAT,validate_descriptors
            if manifest.get('checkpoint_format')!=FORMAT:raise ValueError('Unknown checkpoint format')
            validate_descriptors(manifest['checkpoint'])
            return dict(header=dict(identity=identity,cursor=cursor,
                        tokens=manifest['tokens']+[manifest['pending_token']],block_size=2048,
                        checkpoint_format=FORMAT,
                        dense_store=dict(streams=list(streams),chunks=manifest['chunks'],ports=ports,verify=verify)),
                        shards=[dict(rank=rank,draft_valid=False,checkpoint_store=manifest['checkpoint'][f'rank{rank}'])
                                for rank in (0,1)])
        if manifest.get('checkpoint_format') is not None:raise ValueError('Unknown checkpoint format')
        if set(manifest['checkpoint'])!={'gdn','conv'}:raise ValueError('Missing target checkpoint')
        checkpoints={}
        for name,descriptor in manifest['checkpoint'].items():
            data=objects.get(descriptor['key'])
            if len(data)!=descriptor['bytes']:raise ValueError('Truncated target checkpoint')
            checkpoints[name]=msgspec.msgpack.decode(data)
        gdn,conv=checkpoints['gdn'],checkpoints['conv']
        if set(gdn)!={0,1} or set(conv)!={0,1}:raise ValueError('Incomplete TP checkpoint')
        shards=[]
        for rank in (0,1):
            if set(gdn[rank])!=set(conv[rank]) or len(gdn[rank])!=30:
                raise ValueError('Incomplete GDN census')
            layers={}
            for name in gdn[rank]:
                tensor(gdn[rank][name],(16,128,128),'float32')
                tensor(conv[rank][name],(3,4096),'bfloat16')
                layers[name]=dict(recurrent=gdn[rank][name],conv=conv[rank][name])
            shards.append(dict(rank=rank,layers=layers,draft_valid=False))
        if set(gdn[0])!=set(gdn[1]):raise ValueError('Mismatched target layers')
        return dict(header=dict(identity=identity,cursor=cursor,
                    tokens=manifest['tokens']+[manifest['pending_token']],block_size=2048,
                    dense_store=dict(streams=list(streams),chunks=manifest['chunks'],ports=ports,verify=verify)),
                    shards=shards)
    except (KeyError,ValueError,TypeError,msgspec.DecodeError) as exc:
        raise CacheMiss('Invalid streamed target snapshot') from exc


def import_dense(worker,runner,root,rank,header):
    import sys,time
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    import torch
    from betterscale.live.llm.qwen35.state import GDNState
    from model_checkpoint import dense_span
    from dram_store_fixture import CPU_ENV
    plan=header['dense_store'];cursor=header['cursor'];block=header['block_size']
    planes=[(f'{name}/{kind}/head{rank}',getattr(leaf,kind).tensor)
            for name,leaf in sorted(root.target.items()) if not isinstance(leaf,GDNState)
            for kind in ('key','value')]
    local=[name for name,_ in planes]
    validate_chunks(plan['chunks'],cursor,plan['streams'])
    if len(local)!=20 or set(local)!={s for s in plan['streams'] if s.endswith(f'/head{rank}')}:
        raise ValueError('Dense import plane mismatch')
    ports=plan['ports']
    if len(ports)!=2 or any(type(p) is not int or not 55431<=p<=55438 for p in ports):
        raise ValueError('Unqualified import Store endpoint')
    site=CPU_ENV/'lib/python3.12/site-packages';sys.path.insert(0,str(site))
    import mooncake.store
    from mooncake.store import MooncakeDistributedStore
    if not Path(mooncake.store.__file__).resolve().is_relative_to(site):raise RuntimeError('Wrong Store runtime')
    stride=2048*512;size=20*stride
    slots=[torch.empty(size,dtype=torch.uint8,pin_memory=True) for _ in range(2)]
    stream=torch.npu.Stream(device=runner.device)
    indices=torch.tensor(header['blocks'],dtype=torch.int64,device=runner.device)
    stream.wait_stream(torch.npu.current_stream(runner.device))
    from model_store_clients import acquire,release
    client=acquire(runner,ports[rank],(slots,stream,indices,planes),MooncakeDistributedStore)
    registered=[];closed=False
    # Keep every DMA operand and Store registration reachable across failed fences.
    retained=(slots,stream,indices,planes,client)
    def cleanup():
        nonlocal closed
        _=retained
        while registered:
            slot=registered[-1]
            if client.unregister_buffer(slot.data_ptr())!=0:raise RuntimeError('Import unregister failed')
            registered.pop()
        if not closed:release(runner,ports[rank]);closed=True
    worker._pd_import_cleanup=cleanup
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='pd-store-ingress')
    events=[None,None];start_time=time.perf_counter();chunks=plan['chunks']
    def fetch(index):
        slot=index%2
        if events[slot] is not None:events[slot].synchronize()
        chunk=chunks[index];count=(chunk['stop']-chunk['start'])*512
        result=client.get_into_ranges(
            [slots[slot].data_ptr()+i*stride for i in range(20)],
            [[chunk['keys'][name]] for name in local],
            [[[0]] for _ in local],[[[0]] for _ in local],[[[count]] for _ in local])
        if result!=[[[count]] for _ in local]:raise CacheMiss('Dense Store read incomplete')
    try:
        for slot in slots:
            if client.register_buffer(slot.data_ptr(),size)!=0:raise RuntimeError('Import registration failed')
            registered.append(slot)
        with torch.npu.stream(stream):
            for _,plane in planes:plane.view(-1,block,1,256).index_fill_(0,indices,0)
        pending=pool.submit(fetch,0)
        for index,chunk in enumerate(chunks):
            pending.result();slot=index%2;begin,stop=chunk['start'],chunk['stop']
            with torch.npu.stream(stream):
                for i,(_,plane) in enumerate(planes):
                    host=slots[slot][i*stride:i*stride+(stop-begin)*512].view(torch.bfloat16).view(-1,1,256)
                    logical=plane.view(-1,block,1,256);position=begin
                    while position<stop:
                        end=min(stop,(position//block+1)*block)
                        page=header['blocks'][position//block]
                        logical[page,position%block:position%block+end-position].copy_(
                            host[position-begin:end-begin],non_blocking=True)
                        position=end
                event=torch.npu.Event();event.record(stream);events[slot]=event
            if plan.get('verify'):
                event.synchronize()
                for i,(_,plane) in enumerate(planes):
                    actual=dense_span(plane,indices,block,stop,begin).cpu().view(torch.uint8).flatten()
                    expected=slots[slot][i*stride:i*stride+(stop-begin)*512]
                    if not torch.equal(actual,expected):raise RuntimeError('Dense H2D byte oracle failed')
            if index+1<len(chunks):pending=pool.submit(fetch,index+1)
        stream.synchronize()
        return dict(dense_bytes=cursor*20*512,chunks=len(chunks),ring_slots=2,
                    exact_transfer_oracle=bool(plan.get('verify')),seconds=time.perf_counter()-start_time)
    finally:
        pool.shutdown(wait=True)
        # The caller's drain fence owns cleanup, including on partial-copy failure.

