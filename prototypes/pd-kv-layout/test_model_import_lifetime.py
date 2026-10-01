from types import SimpleNamespace as NS
from unittest.mock import Mock
import sys
import pytest
import torch
from model_import_lifetime import validate_acks,install,abort
from test_model_checkpoint import core


def header():
    return dict(transfer_id='ticket',seat=0,epoch=1)


def receipts(h):
    return [dict(h,rank=i,drained=True) for i in (0,1)]


@pytest.mark.parametrize('fault',('missing','duplicate','not_drained','wrong_ticket','wrong_epoch'))
def test_reject_unknown_completion(fault):
    h=header();rs=receipts(h)
    if fault=='missing':rs.pop()
    elif fault=='duplicate':rs[1]['rank']=0
    elif fault=='not_drained':rs[1]['drained']=False
    elif fault=='wrong_ticket':rs[1]['transfer_id']='other'
    else:rs[1]['epoch']=2
    with pytest.raises(RuntimeError,match='quarantined'):validate_acks(rs,h)


@pytest.mark.parametrize('copy_error,drain_error',((False,False),(True,False),(True,True)))
def test_worker_returns_error_only_with_explicit_drain_state(monkeypatch,copy_error,drain_error):
    sync=Mock(side_effect=RuntimeError('device unavailable') if drain_error else None)
    monkeypatch.setattr(torch,'npu',NS(synchronize=sync),raising=False)
    monkeypatch.setitem(sys.modules,'vllm.distributed',NS(get_tensor_model_parallel_rank=lambda:0))
    worker=NS()
    def copy(*args):
        if copy_error:raise IOError('injected after enqueue')
        return dict(cursor=10)
    result=install(worker,header(),[],copy)
    sync.assert_called_once()
    assert result['drained']==(not drain_error)
    assert bool(result.get('error'))==copy_error
    if drain_error:assert 'device unavailable' in result['error']
    if not copy_error:assert result['cursor']==10


def test_abort_retains_allocations_until_matching_drains():
    c=core();s=c.scheduler
    offer=s.residents.offer([1,2],'salt',s.processed_step_seq,allow_hit=False)
    seat=s.residents.claim('import',offer,s.processed_step_seq)
    blocks=s.kv_cache_manager.block_pool.get_new_blocks(1)
    h=dict(header(),seat=seat.index,epoch=seat.epoch)
    c._pd_import=dict(header=h,seat=seat,owner='import',blocks=blocks)
    c.model_executor.collective_rpc.side_effect=IOError('RPC lost')
    with pytest.raises(IOError):abort(c,'ticket')
    assert s.kv_cache_manager.block_pool.live and 'import' in s.residents.requests
    c.model_executor.collective_rpc.side_effect=lambda *a,**kw:receipts(h)
    with pytest.raises(ValueError):abort(c,'wrong')
    assert abort(c,'ticket')['released']
    assert not s.kv_cache_manager.block_pool.live and not s.residents.requests
    assert c._pd_import is None
    with pytest.raises(ValueError):abort(c,'ticket')


def test_reported_copy_error_releases_only_after_both_drains():
    from model_checkpoint import restore
    from test_model_checkpoint import payload
    c=core();native=c.model_executor.collective_rpc.side_effect
    def rpc(name,args):
        rs=native(name,args);rs[1]['error']='copy failed after enqueue'
        return rs
    c.model_executor.collective_rpc.side_effect=rpc
    with pytest.raises(RuntimeError,match='failed after drain'):
        restore(c,payload(),'salt')
    assert c._pd_import is None
    assert not c.scheduler.kv_cache_manager.block_pool.live
    assert not c.scheduler.residents.requests
    assert all(not s.tokens for s in c.scheduler.residents.seats)


def test_restore_transport_failure_blocks_reimport_until_abort():
    from model_checkpoint import restore
    from test_model_checkpoint import payload
    c=core();native=c.model_executor.collective_rpc.side_effect
    c.model_executor.collective_rpc.side_effect=IOError('unknown completion')
    with pytest.raises(IOError):restore(c,payload(),'salt')
    job=c._pd_import;ids=set(c.scheduler.kv_cache_manager.block_pool.live)
    with pytest.raises(RuntimeError,match='quarantined'):restore(c,payload(),'salt2')
    assert c.scheduler.kv_cache_manager.block_pool.live==ids
    c.model_executor.collective_rpc.side_effect=native
    abort(c,job['header']['transfer_id'])
    assert not c.scheduler.kv_cache_manager.block_pool.live
    assert restore(c,payload(),'salt')['workers'][0]['drained']


def test_failure_probe_retries_without_publishing_failed_copy():
    from model_import_lifetime import failure_probe
    from test_model_checkpoint import payload
    c=core();native=c.model_executor.collective_rpc.side_effect
    def rpc(name,args):
        rs=native(name,args)
        if args[0].get('_probe_fail_after_copy'):
            rs[1]['error']='Injected import failure after enqueue'
        return rs
    c.model_executor.collective_rpc.side_effect=rpc
    result=failure_probe(c,payload(),'salt')
    assert result['failure_probe']['retry_succeeded']
    assert len(c.scheduler.kv_cache_manager.block_pool.live)==3
    assert sum(bool(s.tokens) for s in c.scheduler.residents.seats)==1

