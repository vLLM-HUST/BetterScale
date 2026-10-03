"""Quiescent TP2 target-only model checkpoint ingress/egress prototype.

Core owns seats/page references and publishes only after both workers acknowledge.
CPU tensors are the first transport; no MTP state, background writers, distributed
consensus, live-session snapshotting or production connector claim.
"""
import math
from pd_limits import context_limit
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


def export(core,tokens,salt,dense_start=0,stream_store=None):
    scheduler=idle(core)
    candidates=[s for s in scheduler.residents.seats if s.owner is None and s.tokens==tuple(tokens)
                and s.cache_salt==salt and s.fence<=scheduler.processed_step_seq]
    if len(candidates)!=1:raise ValueError('Expected exactly one fully retired matching resident')
    seat=candidates[0]
    if type(dense_start) is not int or not 0<=dense_start<=seat.cursor:
        raise ValueError('Invalid dense export interval')
    header=dict(identity=IDENTITY,tokens=list(tokens),cursor=seat.cursor,seat=seat.index,epoch=seat.epoch,
                dense_start=dense_start,block_size=scheduler.block_size,blocks=[b.block_id for b in seat.blocks.blocks[0]])
    if hasattr(core,'step_counter'):
        header['dp_control']=dict(finish_sync_steps=getattr(core,'_pd_finish_sync_interval',32),
                                 step_counter=core.step_counter,engines_running=core.engines_running)
    if stream_store is not None:header['stream_store']=stream_store
    if stream_store is not None and stream_store.get('asynchronous'):
        from model_async_export import begin
        return begin(core,header,seat)
    shards=core.model_executor.collective_rpc('pd_export_target',args=(header,))
    if len(shards)!=2:raise RuntimeError('TP2 export requires two worker acknowledgements')
    return dict(header=header,shards=wire_encode(shards))


def capacity(core):
    """Idle admission snapshot, not a reservation or a concurrent allocator."""
    scheduler=idle(core);pool=scheduler.kv_cache_manager.block_pool
    return dict(block_size=scheduler.block_size,free_blocks=pool.get_num_free_blocks(),
                total_blocks=pool.num_gpu_blocks,max_requests=scheduler.max_num_running_reqs,
                context_limit=scheduler.max_model_len)


def drop(core,salt):
    scheduler=idle(core);removed=[]
    for seat in scheduler.residents.seats:
        if seat.owner is None and seat.cache_salt==salt and seat.blocks is not None:
            scheduler._release_resident_blocks(seat.blocks)
            removed.append(seat.index);seat.tokens=();seat.blocks=None;seat.cache_salt=None;seat.epoch+=1
    return removed


def restore(core,payload,salt):
    from model_import_lifetime import validate_acks,release
    scheduler=idle(core);header=payload['header'];tokens=header['tokens'];cursor=header['cursor']
    if getattr(core,'_pd_import',None) is not None:
        raise RuntimeError('A quarantined import must drain before another import')
    if (header.get('dense_start',0)!=0 or header['identity']!=IDENTITY or cursor!=len(tokens)-1 or not 0<cursor<context_limit()
            or header['block_size']!=scheduler.block_size or len(payload['shards'])!=2
            or any(type(t) is not int or t<0 for t in tokens)):
        raise ValueError('Incompatible target checkpoint')
    shards=wire_decode(payload['shards'])
    offer=scheduler.residents.offer(tokens,salt,scheduler.processed_step_seq,allow_hit=False)
    if offer is None:raise RuntimeError('No quiescent destination resident')
    offer=scheduler.residents.discard_victim(offer,scheduler.processed_step_seq)
    owner='pd-import-'+uuid.uuid4().hex
    seat=scheduler.residents.claim(owner,offer,scheduler.processed_step_seq)
    pool=scheduler.kv_cache_manager.block_pool
    job=dict(owner=owner,seat=seat,blocks=[],header=None)
    dispatched=False;drained=False
    try:
        job['blocks']=pool.get_new_blocks(math.ceil(cursor/scheduler.block_size))
        destination=dict(header,seat=seat.index,epoch=seat.epoch,
                         blocks=[b.block_id for b in job['blocks']],transfer_id=uuid.uuid4().hex)
        job['header']=destination;core._pd_import=job
        dispatched=True
        responses=core.model_executor.collective_rpc('pd_import_target',args=(destination,shards))
        validate_acks(responses,destination);drained=True
        errors=[r['error'] for r in responses if r.get('error')]
        if errors:raise RuntimeError('Target import failed after drain: '+str(errors))
        wrapped=scheduler.kv_cache_manager.create_kv_cache_blocks((job['blocks'],))
        scheduler.residents.retire(owner,fence=scheduler.processed_step_seq,tokens=tokens,cache_salt=salt,blocks=wrapped)
        core._pd_import=None
        return dict(seat=seat.index,epoch=seat.epoch,cursor=cursor,blocks=destination['blocks'],workers=responses)
    except BaseException:
        if not dispatched or drained:release(core,job)
        # Otherwise keep both seat and native pages quarantined. Explicit abort
        # can retry a lost drain acknowledgement; never infer completion from RPC.
        raise



def export_retired(core,tokens,salt,dense_start=0,stream_store=None):
    """Return a native utility Future when async output precedes State retirement."""
    from concurrent.futures import Future
    try:idle(core)
    except RuntimeError:
        if getattr(core,'_pd_pending_export',None) is not None:
            raise RuntimeError('One pending checkpoint export per engine')
        future=Future();core._pd_pending_export=(future,tokens,salt,dense_start,stream_store)
        return future
    return export(core,tokens,salt,dense_start,stream_store)


def service_export(core):
    pending=getattr(core,'_pd_pending_export',None)
    if pending is None:return
    try:idle(core)
    except RuntimeError:return
    core._pd_pending_export=None
    future,tokens,salt,dense_start,stream_store=pending
    try:future.set_result(export(core,tokens,salt,dense_start,stream_store))
    except BaseException as exc:future.set_exception(exc)


def install_core():
    from model_dp_control import install
    install()
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
    EngineCore.pd_capacity=capacity
    EngineCore.pd_drop_target=drop
    EngineCore.pd_import_target=restore
    from model_import_lifetime import abort,failure_probe
    EngineCore.pd_abort_import=abort
    EngineCore.pd_import_failure_probe=failure_probe
    from model_async_export import finish,wait
    EngineCore.pd_finish_export=finish
    EngineCore.pd_wait_export=wait
    from model_store_clients import close_core
    EngineCore.pd_close_store_clients=close_core


def worker_root(worker,header,*,source):
    import torch
    from vllm.distributed import get_tensor_model_parallel_world_size,get_tensor_model_parallel_rank
    runner=worker.model_runner;root=runner._live_state_root;seat=header['seat']
    if (get_tensor_model_parallel_world_size()!=2 or header['identity']!=IDENTITY
            or not 0<=seat<20 or not 0<header['cursor']<context_limit() or len(root.target)!=40
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
            if header.get('stream_store') is not None:continue
            layers[name]={}
            for kind in ('key','value'):
                tensor=getattr(leaf,kind).tensor
                layers[name][kind]=dense_span(tensor,indices,block,cursor,header.get('dense_start',0)).cpu().clone()
    result=dict(rank=rank,layers=layers,draft_valid=False)
    if header.get('stream_store') is not None:
        from model_stream_export import export_dense
        result['dense_store']=export_dense(runner,root,rank,header)
    return result


def import_worker(worker,header,shards):
    from model_import_lifetime import install
    return install(worker,header,shards,_import_worker)


def _import_worker(worker,header,shards):
    import torch
    from betterscale.live.llm.qwen35.state import GDNState
    runner,root,rank=worker_root(worker,header,source=False)
    shard=shards[rank];seat=header['seat'];cursor=header['cursor'];block=header['block_size']
    checkpoint_receipt=None
    if header.get('checkpoint_format') is not None:
        from model_store_checkpoint import FORMAT,load_worker
        if header['checkpoint_format']!=FORMAT:raise ValueError('Unknown target checkpoint format')
        shard,checkpoint_receipt=load_worker(runner,header,rank,shard['checkpoint_store'])
    streamed=header.get('dense_store') is not None
    expected_layers={n for n,l in root.target.items() if not streamed or isinstance(l,GDNState)}
    if shard['rank']!=rank or shard['draft_valid'] is not False or set(shard['layers'])!=expected_layers:
        raise ValueError('Incomplete rank target checkpoint')
    # Validate every shape/type before touching the destination. A failed rank
    # still cannot expose a partially installed resident through Core publication.
    for name,leaf in root.target.items():
        if streamed and not isinstance(leaf,GDNState):continue
        data=shard['layers'][name]
        expected=({'conv':((3,4096),torch.bfloat16),'recurrent':((16,128,128),torch.float32)}
                  if isinstance(leaf,GDNState) else {k:((cursor,1,256),torch.bfloat16) for k in ('key','value')})
        if set(data)!=set(expected):raise ValueError('Wrong checkpoint fields')
        for k,(shape,dtype) in expected.items():
            t=data[k]
            if not isinstance(t,torch.Tensor) or t.device.type!='cpu' or tuple(t.shape)!=shape or t.dtype!=dtype:
                raise ValueError(f'Wrong checkpoint tensor geometry: {name}/{k}: {type(t).__name__} {getattr(t, "shape", None)} {getattr(t, "dtype", None)}, expected {shape}/{dtype}')
    if streamed:
        from model_stream_import import validate_chunks
        streams={f'{name}/{kind}/head{r}' for name,leaf in root.target.items()
                 if not isinstance(leaf,GDNState) for kind in ('key','value') for r in (0,1)}
        plan=header['dense_store']
        if set(plan['streams'])!=streams or len(plan['streams'])!=len(streams):
            raise ValueError('Invalid imported head streams')
        validate_chunks(plan['chunks'],cursor,streams)
    root.clear_state_blocks((seat,),domain=root.residents)
    indices=torch.tensor(header['blocks'],dtype=torch.int64,device=runner.device)
    for name,leaf in root.target.items():
        if streamed and not isinstance(leaf,GDNState):continue
        data=shard['layers'][name]
        if isinstance(leaf,GDNState):
            leaf.conv.tensor[seat,:3].copy_(data['conv']);leaf.recurrent.tensor[seat*3].copy_(data['recurrent'])
        else:
            for kind in ('key','value'):
                tensor=getattr(leaf,kind).tensor
                padded=torch.zeros(len(header['blocks'])*block,1,256,dtype=torch.bfloat16)
                padded[:cursor].copy_(data[kind])
                tensor.view(-1,block,1,256).index_copy_(0,indices,padded.view(-1,block,1,256).to(runner.device))
    dense_receipt=None
    if streamed:
        from model_stream_import import import_dense
        dense_receipt=import_dense(worker,runner,root,rank,header)
    if checkpoint_receipt is not None and header['dense_store'].get('verify'):
        from model_store_checkpoint import verify_installed
        verify_installed(root,seat,shard);checkpoint_receipt['exact_transfer_oracle']=True
    root.continuation.selection.tensor[seat]=1;root.conv_selection.tensor[seat]=1
    root.continuation.resident_epoch.tensor[seat]=header['epoch'];root.remaining_outputs.tensor[seat]=0
    runner._live_resident_epochs[seat]=header['epoch'];runner._live_previous_verify.discard(seat)
    return dict(rank=rank,seat=seat,epoch=header['epoch'],cursor=cursor,draft_valid=False,dense_import=dense_receipt,checkpoint_import=checkpoint_receipt)


def install_worker():
    from vllm_ascend.worker.worker import NPUWorker
    NPUWorker.pd_export_target=export_worker
    NPUWorker.pd_import_target=import_worker
    from model_import_lifetime import drain
    NPUWorker.pd_abort_import=drain
    from model_async_export import begin_worker,finish_worker,wait_worker
    NPUWorker.pd_begin_export=begin_worker
    NPUWorker.pd_finish_export=finish_worker
    NPUWorker.pd_wait_export=wait_worker
    from model_store_clients import close_worker
    NPUWorker.pd_close_store_clients=close_worker
