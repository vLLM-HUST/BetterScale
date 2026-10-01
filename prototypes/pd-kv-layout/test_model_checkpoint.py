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
  return [dict(rank=r,seat=h['seat'],epoch=h['epoch']) for r in range(2)]
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
  self.assertFalse(c.scheduler.residents.requests);self.assertFalse(c.scheduler.kv_cache_manager.block_pool.live)
  self.assertTrue(all(not s.tokens and not s.blocks for s in c.scheduler.residents.seats))
 def test_worker_failure_releases_allocations(self):
  c=core();c.model_executor.collective_rpc.side_effect=IOError('copy failed')
  with self.assertRaises(IOError):restore(c,payload(),'s')
  self.assertFalse(c.scheduler.kv_cache_manager.block_pool.live);self.assertFalse(c.scheduler.residents.requests)
 def test_reject_active_or_wrong_frontier_before_reserving(self):
  c=core();c.batch_queue=[object()]
  with self.assertRaises(RuntimeError):restore(c,payload(),'s')
  c.batch_queue=[];p=payload();p['header']['cursor']-=1
  with self.assertRaises(ValueError):restore(c,p,'s')
  self.assertFalse(c.scheduler.kv_cache_manager.block_pool.live)

if __name__=='__main__':unittest.main()
