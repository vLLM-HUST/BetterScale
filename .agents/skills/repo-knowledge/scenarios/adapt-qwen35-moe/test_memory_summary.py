"""CPU regressions for native allocator ownership accounting."""
import unittest
from summarize_memory import pool_totals


def segment(pool,stream,active,inactive):
    return dict(device=0,segment_pool_id=pool,stream=stream,total_size=active+inactive,
                blocks=[dict(size=active,state='active_allocated'),dict(size=inactive,state='inactive')])


class PoolAccounting(unittest.TestCase):
    def test_multiple_streams_do_not_duplicate_one_pool(self):
        pools=pool_totals([segment([1,1],10,60,40),segment([1,1],11,0,20)])
        self.assertEqual(len(pools),1)
        self.assertEqual(next(iter(pools.values())),dict(reserved=120,active=60,inactive=60,segments=2,streams=[10,11]))

    def test_ordinary_and_private_pools_stay_separate(self):
        pools=pool_totals([segment([0,0],10,20,80),segment([1,1],10,0,100)])
        self.assertEqual(len(pools),2)
        self.assertEqual(pools['device0:pool(0, 0)']['inactive'],80)

    def test_unknown_or_inconsistent_snapshots_fail_closed(self):
        bad=segment([1,1],10,60,40);bad['total_size']=120
        with self.assertRaises(ValueError):pool_totals([bad])
        bad=segment([1,1],10,60,40);bad['blocks'][0]['state']='unknown'
        with self.assertRaises(ValueError):pool_totals([bad])


if __name__=='__main__':unittest.main()
