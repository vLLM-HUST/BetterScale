"""Quiescent TP2 target-only model checkpoint ingress/egress prototype.

Core owns seats/page references and publishes only after both workers acknowledge.
CPU tensors are the first transport; no MTP state, background writers, distributed
consensus, live-session snapshotting or production connector claim.
"""
import math
import uuid

IDENTITY='qwen35-35b-a3b-bf16-tp2-target-only-v1'



def wire_encode(value):
    """Explicit bytes envelope: untyped native utility RPC does not decode tensors."""
    import torch
    if isinstance(value,torch.Tensor):
        if value.device.type!='cpu' or value.dtype not in (torch.bfloat16,torch.float32):
            raise ValueError('Unsupported checkpoint wire tensor')
        return dict(checkpoint_tensor=True,dtype=str(value.dtype).removeprefix('torch.'),
                    shape=list(value.shape),data=value.contiguous().view(torch.uint8).numpy().tobytes())
    if isinstance(value,dict):return {k:wire_encode(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [wire_encode(v) for v in value]
    return value


def wire_decode(value):
    import torch
    if isinstance(value,dict) and value.get('checkpoint_tensor') is True:
        if set(value)!={'checkpoint_tensor','dtype','shape','data'}:
            raise ValueError('Malformed checkpoint wire fields')
        dtype={'bfloat16':torch.bfloat16,'float32':torch.float32}.get(value['dtype'])
        shape=value['shape'];data=value['data']
        if (dtype is None or not isinstance(shape,list) or not shape
                or any(type(n) is not int or n<=0 for n in shape)
                or not isinstance(data,bytes) or len(data)!=math.prod(shape)*dtype.itemsize):
            raise ValueError('Malformed checkpoint wire tensor')
        return torch.frombuffer(bytearray(data),dtype=dtype).view(shape)
    if isinstance(value,dict):return {k:wire_decode(v) for k,v in value.items()}
    if isinstance(value,list):return [wire_decode(v) for v in value]
    return value


def idle(core):
    scheduler=core.scheduler
    if scheduler.has_requests() or getattr(core,'batch_queue',None) or scheduler._pending_hot:
        raise RuntimeError('Checkpoint RPC requires a fully retired idle engine')
    return scheduler


def export(core,tokens,salt,dense_start=0):
    scheduler=idle(core)
    candidates=[s for s in scheduler.residents.seats if s.owner is None and s.tokens==tuple(tokens)
                and s.cache_salt==salt and s.fence<=scheduler.processed_step_seq]
    if len(candidates)!=1:raise ValueError('Expected exactly one fully retired matching resident')
    seat=candidates[0]
    if type(dense_start) is not int or not 0<=dense_start<=seat.cursor:
        raise ValueError('Invalid dense export interval')
    header=dict(identity=IDENTITY,tokens=list(tokens),cursor=seat.cursor,seat=seat.index,epoch=seat.epoch,
                dense_start=dense_start,block_size=scheduler.block_size,blocks=[b.block_id for b in seat.blocks.blocks[0]])
    shards=core.model_executor.collective_rpc('pd_export_target',args=(header,))
    if len(shards)!=2:raise RuntimeError('TP2 export requires two worker acknowledgements')
    return dict(header=header,shards=wire_encode(shards))


def drop(core,salt):
    scheduler=idle(core);removed=[]
    for seat in scheduler.residents.seats:
        if seat.owner is None and seat.cache_salt==salt and seat.blocks is not None:
            scheduler._release_resident_blocks(seat.blocks)
            removed.append(seat.index);seat.tokens=();seat.blocks=None;seat.cache_salt=None;seat.epoch+=1
    return removed


def restore(core,payload,salt):
    scheduler=idle(core);header=payload['header'];tokens=header['tokens'];cursor=header['cursor']
    if (header.get('dense_start',0)!=0 or header['identity']!=IDENTITY or cursor!=len(tokens)-1 or not 0<cursor<=8192
            or header['block_size']!=scheduler.block_size or len(payload['shards'])!=2
            or any(type(t) is not int or t<0 for t in tokens)):
        raise ValueError('Incompatible target checkpoint')
    offer=scheduler.residents.offer(tokens,salt,scheduler.processed_step_seq,allow_hit=False)
    if offer is None:raise RuntimeError('No quiescent destination resident')
    offer=scheduler.residents.discard_victim(offer,scheduler.processed_step_seq)
    owner='pd-import-'+uuid.uuid4().hex
    seat=scheduler.residents.claim(owner,offer,scheduler.processed_step_seq)
    pool=scheduler.kv_cache_manager.block_pool;blocks=[]
    try:
        blocks=pool.get_new_blocks(math.ceil(cursor/scheduler.block_size))
        destination=dict(header,seat=seat.index,epoch=seat.epoch,blocks=[b.block_id for b in blocks])
        responses=core.model_executor.collective_rpc('pd_import_target',args=(destination,wire_decode(payload['shards'])))
        if len(responses)!=2 or any(r['epoch']!=seat.epoch or r['seat']!=seat.index for r in responses):
            raise RuntimeError('Incomplete target import acknowledgements')
        wrapped=scheduler.kv_cache_manager.create_kv_cache_blocks((blocks,))
        scheduler.residents.retire(owner,fence=scheduler.processed_step_seq,tokens=tokens,cache_salt=salt,blocks=wrapped)
        return dict(seat=seat.index,epoch=seat.epoch,cursor=cursor,blocks=destination['blocks'],workers=responses)
    except BaseException:
        if blocks:pool.free_blocks(reversed(blocks))
        if owner in scheduler.residents.requests:scheduler.residents.retire(owner,fence=scheduler.processed_step_seq)
        seat.epoch+=1
        raise



def export_retired(core,tokens,salt,dense_start=0):
    """Return a native utility Future when async output precedes State retirement."""
    from concurrent.futures import Future
    try:idle(core)
    except RuntimeError:
        if getattr(core,'_pd_pending_export',None) is not None:
            raise RuntimeError('One pending checkpoint export per engine')
        future=Future();core._pd_pending_export=(future,tokens,salt,dense_start)
        return future
    return export(core,tokens,salt,dense_start)


def service_export(core):
    pending=getattr(core,'_pd_pending_export',None)
    if pending is None:return
    try:idle(core)
    except RuntimeError:return
    core._pd_pending_export=None
    future,tokens,salt,dense_start=pending
    try:future.set_result(export(core,tokens,salt,dense_start))
    except BaseException as exc:future.set_exception(exc)


def install_core():
    from vllm.v1.engine.core import EngineCore
    EngineCore.pd_export_target=export
    EngineCore.pd_export_retired=export_retired
    from vllm.v1.engine.core import EngineCoreProc
    original=EngineCoreProc._process_engine_step
    if not getattr(original,'_pd_retirement_service',False):
        def step(core,*args,**kwargs):
            result=original(core,*args,**kwargs)
            service_export(core)
            return result
        step._pd_retirement_service=True
        EngineCoreProc._process_engine_step=step
    EngineCore.pd_drop_target=drop
    EngineCore.pd_import_target=restore


def worker_root(worker,header,*,source):
    import torch
    from vllm.distributed import get_tensor_model_parallel_world_size,get_tensor_model_parallel_rank
    runner=worker.model_runner;root=runner._live_state_root;seat=header['seat']
    if (get_tensor_model_parallel_world_size()!=2 or header['identity']!=IDENTITY
            or not 0<=seat<20 or not 0<header['cursor']<=8192 or len(root.target)!=40
            or header['block_size']!=runner.block_size):
        raise ValueError('Unqualified model checkpoint geometry')
    torch.npu.synchronize()
    if source and (runner._live_resident_epochs[seat]!=header['epoch'] or root.remaining_outputs.tensor[seat].item()!=0):
        raise RuntimeError('Source is not the retired length frontier')
    return runner,root,get_tensor_model_parallel_rank()



def dense_span(tensor,indices,block,cursor,start):
    """Gather only intersecting logical pages and expose only new token bytes."""
    if type(start) is not int or not 0<=start<=cursor<=len(indices)*block:
        raise ValueError('Invalid dense page span')
    if start==cursor:return tensor.new_empty((0,*tensor.shape[2:]))
    logical=tensor.view(-1,block,*tensor.shape[2:])
    gathered=logical.index_select(0,indices[start//block:(cursor+block-1)//block]).flatten(0,1)
    return gathered[start%block:start%block+cursor-start]


def export_worker(worker,header):
    import torch
    from betterscale.live.llm.qwen35.state import GDNState
    runner,root,rank=worker_root(worker,header,source=True)
    seat=header['seat'];cursor=header['cursor'];block=header['block_size']
    selected=int(root.continuation.selection.tensor[seat]);conv_selected=int(root.conv_selection.tensor[seat])
    if not 1<=selected<=3 or not 1<=conv_selected<=3:raise RuntimeError('Invalid committed target selectors')
    indices=torch.tensor(header['blocks'],dtype=torch.int64,device=runner.device)
    layers={}
    for name,leaf in root.target.items():
        if isinstance(leaf,GDNState):
            layers[name]=dict(conv=leaf.conv.tensor[seat,conv_selected-1:conv_selected+2].cpu().clone(),
                              recurrent=leaf.recurrent.tensor[seat*3+selected-1].cpu().clone())
        else:
            layers[name]={}
            for kind in ('key','value'):
                tensor=getattr(leaf,kind).tensor
                layers[name][kind]=dense_span(tensor,indices,block,cursor,header.get('dense_start',0)).cpu().clone()
    return dict(rank=rank,layers=layers,draft_valid=False)


def import_worker(worker,header,shards):
    import torch
    from betterscale.live.llm.qwen35.state import GDNState
    runner,root,rank=worker_root(worker,header,source=False)
    shard=shards[rank];seat=header['seat'];cursor=header['cursor'];block=header['block_size']
    if shard['rank']!=rank or shard['draft_valid'] is not False or set(shard['layers'])!=set(root.target):
        raise ValueError('Incomplete rank target checkpoint')
    # Validate every shape/type before touching the destination. A failed rank
    # still cannot expose a partially installed resident through Core publication.
    for name,leaf in root.target.items():
        data=shard['layers'][name]
        expected=({'conv':((3,4096),torch.bfloat16),'recurrent':((16,128,128),torch.float32)}
                  if isinstance(leaf,GDNState) else {k:((cursor,1,256),torch.bfloat16) for k in ('key','value')})
        if set(data)!=set(expected):raise ValueError('Wrong checkpoint fields')
        for k,(shape,dtype) in expected.items():
            t=data[k]
            if not isinstance(t,torch.Tensor) or t.device.type!='cpu' or tuple(t.shape)!=shape or t.dtype!=dtype:
                raise ValueError(f'Wrong checkpoint tensor geometry: {name}/{k}: {type(t).__name__} {getattr(t, "shape", None)} {getattr(t, "dtype", None)}, expected {shape}/{dtype}')
    root.clear_state_blocks((seat,),domain=root.residents)
    indices=torch.tensor(header['blocks'],dtype=torch.int64,device=runner.device)
    for name,leaf in root.target.items():
        data=shard['layers'][name]
        if isinstance(leaf,GDNState):
            leaf.conv.tensor[seat,:3].copy_(data['conv']);leaf.recurrent.tensor[seat*3].copy_(data['recurrent'])
        else:
            for kind in ('key','value'):
                tensor=getattr(leaf,kind).tensor
                padded=torch.zeros(len(header['blocks'])*block,1,256,dtype=torch.bfloat16)
                padded[:cursor].copy_(data[kind])
                tensor.view(-1,block,1,256).index_copy_(0,indices,padded.view(-1,block,1,256).to(runner.device))
    root.continuation.selection.tensor[seat]=1;root.conv_selection.tensor[seat]=1
    root.continuation.resident_epoch.tensor[seat]=header['epoch'];root.remaining_outputs.tensor[seat]=0
    runner._live_resident_epochs[seat]=header['epoch'];runner._live_previous_verify.discard(seat)
    torch.npu.synchronize()
    return dict(rank=rank,seat=seat,epoch=header['epoch'],cursor=cursor,draft_valid=False)


def install_worker():
    from vllm_ascend.worker.worker import NPUWorker
    NPUWorker.pd_export_target=export_worker
    NPUWorker.pd_import_target=import_worker
