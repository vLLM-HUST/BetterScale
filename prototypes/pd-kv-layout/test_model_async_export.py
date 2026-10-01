"""Native ownership survives hot eviction; failures cannot pretend DMA drained."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from model_async_export import begin,finish


def fixture():
 blocks=[NS(block_id=i,ref_cnt=1) for i in (5,2)]
 def touch(group):
  for b in group:b.ref_cnt+=1
 def free(group):
  for b in group:b.ref_cnt-=1
 pool=NS(touch=touch,free_blocks=free)
 def rpc(name,args):
  if name=='pd_begin_export':return [dict(rank=i,started=True,transfer_id=args[0]['transfer_id']) for i in (0,1)]
  return [dict(rank=i,transfer_id=args[0],drained=True,error=None,shard={'rank':i}) for i in (0,1)]
 core=NS(scheduler=NS(kv_cache_manager=NS(block_pool=pool)),model_executor=NS(collective_rpc=Mock(side_effect=rpc)))
 seat=NS(blocks=NS(blocks=(blocks,)))
 return core,seat,blocks,pool


def test_pins_outlive_source_eviction_then_release_once():
 core,seat,blocks,pool=fixture();ticket=begin(core,{},seat)
 assert [b.ref_cnt for b in blocks]==[2,2]
 pool.free_blocks(blocks) # Independent hot LRU eviction may now retire its refs.
 assert [b.ref_cnt for b in blocks]==[1,1]
 with pytest.raises(RuntimeError,match='One asynchronous'):begin(core,{},seat)
 with pytest.raises(ValueError):finish(core,'foreign')
 result=finish(core,ticket['transfer_id'])
 assert result['page_pin_receipt']['refs_at_release']==[1,1]
 assert [b.ref_cnt for b in blocks]==[0,0]
 with pytest.raises(ValueError):finish(core,ticket['transfer_id'])


@pytest.mark.parametrize('failure',['transport','missing','not_drained','wrong_ticket'])
def test_unknown_remote_completion_quarantines_pins(failure):
 core,seat,blocks,pool=fixture();ticket=begin(core,{},seat);original=core.model_executor.collective_rpc.side_effect
 def rpc(name,args):
  if failure=='transport':raise IOError('RPC lost')
  result=original(name,args)
  if failure=='missing':return result[:1]
  if failure=='not_drained':result[0]['drained']=False
  if failure=='wrong_ticket':result[0]['transfer_id']='other'
  return result
 core.model_executor.collective_rpc.side_effect=rpc
 with pytest.raises((IOError,RuntimeError)):finish(core,ticket['transfer_id'])
 assert [b.ref_cnt for b in blocks]==[2,2]
 core.model_executor.collective_rpc.side_effect=original
 finish(core,ticket['transfer_id']);assert [b.ref_cnt for b in blocks]==[1,1]


def test_store_error_releases_only_after_both_workers_drained():
 core,seat,blocks,pool=fixture();ticket=begin(core,{},seat);original=core.model_executor.collective_rpc.side_effect
 def rpc(name,args):
  result=original(name,args);result[1]['error']='Store write failed';return result
 core.model_executor.collective_rpc.side_effect=rpc
 with pytest.raises(RuntimeError,match='Store write failed'):finish(core,ticket['transfer_id'])
 assert [b.ref_cnt for b in blocks]==[1,1]
 assert core._pd_async_export is None


def test_worker_waits_for_job_and_retains_idempotent_drain_receipt(monkeypatch):
 import sys,threading,torch
 from concurrent.futures import ThreadPoolExecutor
 import model_checkpoint,model_stream_export
 from model_async_export import begin_worker,finish_worker
 monkeypatch.setitem(sys.modules,'vllm.distributed',NS(get_tensor_model_parallel_rank=lambda:0))
 monkeypatch.setattr(torch,'npu',NS(set_device=lambda d:None),raising=False)
 monkeypatch.setattr(model_checkpoint,'export_worker',lambda *args:dict(rank=0,draft_valid=False,
  layers={'gdn':dict(conv='c',recurrent='r'),'fa':dict(key=b'',value=b'')}))
 entered=threading.Event();gate=threading.Event()
 def transfer(*args,lifetime,expected_dense=None):
  lifetime['drained']=False;entered.set()
  assert gate.wait(5)
  lifetime['drained']=True
  return dict(acknowledged=True)
 monkeypatch.setattr(model_stream_export,'export_dense',transfer)
 worker=NS(model_runner=NS(device='test',_live_state_root=object()))
 h=dict(transfer_id='a',cursor=5,stream_store={})
 assert begin_worker(worker,h)['started']
 assert entered.wait(5)
 assert not begin_worker(worker,dict(h,transfer_id='b'))['started']
 with ThreadPoolExecutor(1) as pool:
  f=pool.submit(finish_worker,worker,'a')
  assert not f.done();gate.set();r=f.result(timeout=5)
 assert r['drained'] and not r['error'] and set(r['shard']['layers'])=={'gdn'}
 assert finish_worker(worker,'a') is r
 def failed(*args,lifetime,expected_dense=None):
  lifetime['drained']=True
  raise IOError('injected Store failure after drain')
 monkeypatch.setattr(model_stream_export,'export_dense',failed)
 assert begin_worker(worker,dict(h,transfer_id='b'))['started']
 r=finish_worker(worker,'b')
 assert r['drained'] and 'injected Store failure' in r['error'] and 'shard' not in r


def test_serialization_control_wait_does_not_release_native_pins():
 from model_async_export import wait
 core,seat,blocks,pool=fixture();ticket=begin(core,{},seat);original=core.model_executor.collective_rpc.side_effect
 core.model_executor.collective_rpc.side_effect=lambda name,args:[True,True] if name=='pd_wait_export' else original(name,args)
 assert wait(core,ticket['transfer_id'])
 assert [b.ref_cnt for b in blocks]==[2,2]
 finish(core,ticket['transfer_id']);assert [b.ref_cnt for b in blocks]==[1,1]
