"""Destination pages stay reserved until both TP workers prove DMA drained.

A lost RPC is not a completion fence. Quarantined imports are never published;
explicit abort may release them only after matching worker drain receipts.
"""
import traceback


def validate_acks(responses, header):
    if (len(responses)!=2 or {r.get('rank') for r in responses}!={0,1}
            or any(r.get('transfer_id')!=header['transfer_id']
                   or r.get('seat')!=header['seat'] or r.get('epoch')!=header['epoch']
                   or r.get('drained') is not True for r in responses)):
        raise RuntimeError('Import quarantined: matching TP2 drain acknowledgements required')


def release(core, job):
    scheduler=core.scheduler
    if job['blocks']:
        scheduler.kv_cache_manager.block_pool.free_blocks(reversed(job['blocks']))
    if job['owner'] in scheduler.residents.requests:
        scheduler.residents.retire(job['owner'],fence=scheduler.processed_step_seq)
    job['seat'].epoch+=1
    core._pd_import=None


def abort(core, transfer_id):
    from model_checkpoint import idle
    idle(core)
    job=getattr(core,'_pd_import',None)
    if job is None or job['header']['transfer_id']!=transfer_id:
        raise ValueError('Unknown quarantined import')
    responses=core.model_executor.collective_rpc('pd_abort_import',args=(job['header'],))
    validate_acks(responses,job['header'])
    release(core,job)
    return dict(transfer_id=transfer_id,released=True)


def drain(worker, header):
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank
    result=dict(rank=get_tensor_model_parallel_rank(),seat=header['seat'],
                epoch=header['epoch'],transfer_id=header['transfer_id'],drained=False)
    if getattr(worker,'_pd_import_ticket',None)!=header['transfer_id']:
        return dict(result,error='Unknown worker import ticket')
    try:
        torch.npu.synchronize()
        cleanup=getattr(worker,'_pd_import_cleanup',None)
        if cleanup is not None:
            cleanup();worker._pd_import_cleanup=None
        result['drained']=True
    except BaseException:
        result['error']=traceback.format_exc()
    return result


def install(worker, header, shards, copy):
    worker._pd_import_ticket=header['transfer_id']
    result={};error=None
    try:
        result=copy(worker,header,shards)
        if header.get('_probe_fail_after_copy'):
            from vllm.distributed import get_tensor_model_parallel_rank
            if get_tensor_model_parallel_rank()==1:
                raise RuntimeError('Injected import failure after enqueue')
    except BaseException:
        error=traceback.format_exc()
    receipt=drain(worker,header)
    # Drain errors must never be hidden by an earlier copy exception.
    result.update(receipt)
    if error:result['error']=error+('\n'+receipt['error'] if receipt.get('error') else '')
    return result



def failure_probe(core,payload,salt):
    """One explicit diagnostic: rank1 fails after H2D enqueue, then clean retry."""
    from model_checkpoint import restore
    injected=dict(payload,header=dict(payload['header'],_probe_fail_after_copy=True))
    try:
        restore(core,injected,salt)
    except RuntimeError as exc:
        if 'Injected import failure after enqueue' not in str(exc):raise
        if getattr(core,'_pd_import',None) is not None:
            raise RuntimeError('Injected failure did not receive both drain acknowledgements') from exc
        if any(s.cache_salt==salt and s.tokens for s in core.scheduler.residents.seats):
            raise RuntimeError('Failed import published a resident') from exc
    else:
        raise RuntimeError('Expected the diagnostic import to fail')
    result=restore(core,payload,salt)
    result['failure_probe']=dict(rank=1,after_enqueue=True,drained_before_release=True,
                                 no_failed_publication=True,retry_succeeded=True)
    return result
