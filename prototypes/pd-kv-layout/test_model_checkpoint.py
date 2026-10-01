"""Core publication/refcount contract, using actual resident ownership logic."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import sys,unittest
from unittest.mock import Mock
from model_checkpoint import IDENTITY,restore,export,drop
path=Path(__file__).resolve().parents[2]/'src/betterscale/models/qwen35/resident_leases.py'
spec=importlib.util.spec_from_file_location('checkpoint_resident_leases',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

class Pool:
 def __init__(self):self.live=set();self.next=0
 def get_new_blocks(self,n):
  result=[NS(block_id=i) for i in range(self.next,self.next+n)];self.next+=n;self.live.update(b.block_id for b in result);return result
 def free_blocks(self,blocks):
  for b in blocks:self.live.remove(b.block_id)

def core():
 pool=Pool();manager=NS(block_pool=pool,create_kv_cache_blocks=lambda blocks:NS(blocks=blocks))
 scheduler=NS(kv_cache_manager=manager,block_size=2048,processed_step_seq=5,_pending_hot={},has_requests=lambda:False)
 def release(blocks):
  for group in blocks.blocks:pool.free_blocks(group)
 scheduler._release_resident_blocks=release;scheduler.residents=m.ResidentLeases(3,release_blocks=release)
 def rpc(name,args):
  h=args[0]
  if name=='pd_export_target':return [{'rank':0},{'rank':1}]
  return [dict(rank=r,seat=h['seat'],epoch=h['epoch'],transfer_id=h['transfer_id'],drained=True) for r in range(2)]
 return NS(scheduler=scheduler,batch_queue=[],model_executor=NS(collective_rpc=Mock(side_effect=rpc)))

def payload():return dict(header=dict(identity=IDENTITY,tokens=list(range(4098)),cursor=4097,block_size=2048),shards=[{},{}])

class CheckpointTest(unittest.TestCase):
 def test_publish_then_export_drop(self):
  c=core();result=restore(c,payload(),'s');s=c.scheduler.residents.seats[result['seat']]
  self.assertIsNone(s.owner);self.assertEqual(s.cursor,4097);self.assertEqual(len(c.scheduler.kv_cache_manager.block_pool.live),3)
  out=export(c,payload()['header']['tokens'],'s');self.assertEqual(out['header']['blocks'],result['blocks'])
  self.assertEqual(drop(c,'unrelated'),[]);self.assertEqual(drop(c,'s'),[s.index]);self.assertFalse(c.scheduler.kv_cache_manager.block_pool.live)
 def test_no_publication_without_all_worker_acks(self):
  c=core();c.model_executor.collective_rpc.side_effect=lambda *a,**kw:[]
  with self.assertRaises(RuntimeError):restore(c,payload(),'s')
  self.assertTrue(c.scheduler.residents.requests);self.assertTrue(c.scheduler.kv_cache_manager.block_pool.live)
  self.assertTrue(all(not s.tokens and not s.blocks for s in c.scheduler.residents.seats))
 def test_unknown_worker_completion_quarantines_allocations(self):
  c=core();c.model_executor.collective_rpc.side_effect=IOError('copy failed')
  with self.assertRaises(IOError):restore(c,payload(),'s')
  self.assertTrue(c.scheduler.kv_cache_manager.block_pool.live);self.assertTrue(c.scheduler.residents.requests)
 def test_reject_active_or_wrong_frontier_before_reserving(self):
  c=core();c.batch_queue=[object()]
  with self.assertRaises(RuntimeError):restore(c,payload(),'s')
  c.batch_queue=[];p=payload();p['header']['cursor']-=1
  with self.assertRaises(ValueError):restore(c,p,'s')
  self.assertFalse(c.scheduler.kv_cache_manager.block_pool.live)

if __name__=='__main__':unittest.main()


def test_explicit_wire_bytes_roundtrip():
    import torch
    from model_checkpoint import wire_encode,wire_decode
    original={'layers':[torch.arange(12).reshape(3,4).bfloat16(),torch.randn(2,3)]}
    encoded=wire_encode(original)
    assert isinstance(encoded['layers'][0]['data'],bytes)
    # The actual utility codec's untyped decode must preserve this envelope.
    import msgspec
    decoded=wire_decode(msgspec.msgpack.decode(msgspec.msgpack.encode(encoded)))
    for a,b in zip(original['layers'],decoded['layers']):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    encoded['layers'][0]['shape']=[100]
    import pytest
    with pytest.raises(ValueError,match='Malformed'):wire_decode(encoded)


def test_deferred_export_waits_for_real_retirement(monkeypatch):
    from types import SimpleNamespace
    import model_checkpoint as checkpoint
    core=SimpleNamespace(batch_queue=[object()],scheduler=SimpleNamespace(has_requests=lambda:False,_pending_hot=[]))
    monkeypatch.setattr(checkpoint,'export',lambda c,t,s,start=0,stream_store=None:dict(tokens=t,salt=s))
    future=checkpoint.export_retired(core,[1,2],'s')
    assert not future.done()
    checkpoint.service_export(core);assert not future.done()
    import pytest
    with pytest.raises(RuntimeError,match='One pending'):checkpoint.export_retired(core,[1,2],'s')
    core.batch_queue=[];core.scheduler._pending_hot=[object()]
    checkpoint.service_export(core);assert not future.done()
    core.scheduler._pending_hot=[];checkpoint.service_export(core)
    assert future.result()=={'tokens':[1,2],'salt':'s'}
    assert core._pd_pending_export is None


def test_dense_span_permuted_partial_pages():
    import torch
    from model_checkpoint import dense_span
    pages=torch.arange(12*2*1*3).view(12,2,1,3)
    ids=torch.tensor([4,1,5]);logical=pages.view(6,4,1,3)
    full=logical[ids].flatten(0,1)[:9]
    for start in (0,1,4,5,8,9):
        assert torch.equal(dense_span(pages,ids,4,9,start),full[start:])
