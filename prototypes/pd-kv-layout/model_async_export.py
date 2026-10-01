"""One retired export per engine, with native page pins until both workers drain.

Only retired immutable target intervals may enter. Selected GDN/conv is staged
before begin returns; dense jobs may then outlive hot-seat eviction and overlap
other requests. This does not export the currently writing request's pages.
"""
from concurrent.futures import ThreadPoolExecutor
import time,traceback,uuid


def begin(core,header,seat):
    if getattr(core,'_pd_async_export',None) is not None:
        raise RuntimeError('One asynchronous export per engine')
    groups=tuple(tuple(group) for group in seat.blocks.blocks)
    pool=core.scheduler.kv_cache_manager.block_pool
    for group in groups:pool.touch(group)
    header=dict(header,transfer_id=uuid.uuid4().hex)
    core._pd_async_export=dict(header=header,groups=groups)
    # Any transport exception quarantines the pins. Never infer remote DMA
    # completion from a failed RPC. Explicit finish can retry a still-live engine.
    responses=core.model_executor.collective_rpc('pd_begin_export',args=(header,))
    if len(responses)!=2 or {r['rank'] for r in responses}!={0,1} or any(not r['started'] or r.get('transfer_id')!=header['transfer_id'] for r in responses):
        # Worker methods return errors as data, after local work has drained.
        finish(core,header['transfer_id'])
        raise RuntimeError('Incomplete async export start')
    return dict(transfer_id=header['transfer_id'],pinned_blocks=[b.block_id for group in groups for b in group])


def finish(core,transfer_id):
    job=getattr(core,'_pd_async_export',None)
    if job is None or job['header']['transfer_id']!=transfer_id:
        raise ValueError('Unknown asynchronous export')
    responses=core.model_executor.collective_rpc('pd_finish_export',args=(transfer_id,))
    if (len(responses)!=2 or {r['rank'] for r in responses}!={0,1}
            or any(r.get('transfer_id')!=transfer_id or r.get('drained') is not True for r in responses)):
        raise RuntimeError('Cannot release page pins without both worker drain acknowledgements')
    pool=core.scheduler.kv_cache_manager.block_pool
    counts=[b.ref_cnt for group in job['groups'] for b in group]
    if any(n<1 for n in counts):raise RuntimeError('Async export lost native page ownership')
    for group in job['groups']:pool.free_blocks(reversed(group))
    core._pd_async_export=None
    errors=[r['error'] for r in responses if r.get('error')]
    if errors:raise RuntimeError('Asynchronous export failed: '+str(errors))
    return dict(header=job['header'],shards=[r['shard'] for r in sorted(responses,key=lambda r:r['rank'])],
                page_pin_receipt=dict(blocks=[b.block_id for group in job['groups'] for b in group],refs_at_release=counts))


def begin_worker(worker,header):
    from vllm.distributed import get_tensor_model_parallel_rank
    from model_checkpoint import export_worker
    rank=get_tensor_model_parallel_rank();transfer_id=header['transfer_id']
    previous=getattr(worker,'_pd_async_export',None)
    if previous is not None and 'receipt' not in previous:
        return dict(rank=rank,started=False,error='Worker export already active')
    job=dict(transfer_id=transfer_id,pool=None,future=None,shard=None,error=None,lifetime=dict(drained=False))
    worker._pd_async_export=job
    try:
        # Empty dense interval avoids a second implementation of selected
        # recurrent/conv snapshotting. Only those CPU planes survive this call.
        check=header['stream_store'].get('verify_transfer',False)
        checkpoint_header=dict(header,dense_start=header['dense_start'] if check else header['cursor']);checkpoint_header.pop('stream_store')
        shard=export_worker(worker,checkpoint_header)
        expected={f'{name}/{kind}/head{rank}':value for name,data in shard['layers'].items()
                  if set(data)=={'key','value'} for kind,value in data.items()} if check else None
        shard['layers']={k:v for k,v in shard['layers'].items() if set(v)=={'conv','recurrent'}}
        job['shard']=shard;job['lifetime']['drained']=True
        runner=worker.model_runner;root=runner._live_state_root
        def transfer():
            import torch
            from model_stream_export import export_dense
            torch.npu.set_device(runner.device)
            start=time.monotonic()
            result=export_dense(runner,root,rank,header,lifetime=job['lifetime'],expected_dense=expected)
            result['host_job_interval']=[start,time.monotonic()]
            return result
        job['pool']=ThreadPoolExecutor(max_workers=1,thread_name_prefix='pd-retired-export')
        job['future']=job['pool'].submit(transfer)
        return dict(rank=rank,started=True,transfer_id=transfer_id)
    except BaseException:
        job['error']=traceback.format_exc()
        return dict(rank=rank,started=False,transfer_id=transfer_id,error=job['error'])


def finish_worker(worker,transfer_id):
    from vllm.distributed import get_tensor_model_parallel_rank
    from model_checkpoint import wire_encode
    rank=get_tensor_model_parallel_rank();job=getattr(worker,'_pd_async_export',None)
    if job is None or job['transfer_id']!=transfer_id:
        return dict(rank=rank,transfer_id=transfer_id,drained=False,error='Unknown worker transfer')
    # Keep the drained receipt for retry after a Core-side transport exception.
    if 'receipt' in job:return job['receipt']
    result=dict(rank=rank,transfer_id=transfer_id,drained=True,error=job['error'])
    try:
        if job['future'] is not None:
            job['shard']['dense_store']=job['future'].result()
        if not result['error']:result['shard']=wire_encode(job['shard'])
    except BaseException:result['error']=traceback.format_exc()
    finally:
        if job['pool'] is not None:job['pool'].shutdown(wait=True)
    result['drained']=job['lifetime']['drained']
    if result['drained']:
        job.clear();job.update(transfer_id=transfer_id,receipt=result)
    return result


def wait(core,transfer_id):
    job=getattr(core,'_pd_async_export',None)
    if job is None or job['header']['transfer_id']!=transfer_id:raise ValueError('Unknown asynchronous export')
    responses=core.model_executor.collective_rpc('pd_wait_export',args=(transfer_id,))
    if len(responses)!=2 or not all(responses):raise RuntimeError('Incomplete wait acknowledgement')
    return True


def wait_worker(worker,transfer_id):
    job=getattr(worker,'_pd_async_export',None)
    if job is None or job['transfer_id']!=transfer_id:raise ValueError('Unknown worker transfer')
    if job.get('future') is not None:job['future'].result()
    return job.get('receipt',{}).get('drained',job['lifetime']['drained'] if 'lifetime' in job else False)
