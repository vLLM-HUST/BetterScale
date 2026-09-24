"""CPU ownership/dependency model for the bounded compact-ready prototype."""
import random
import unittest

class Prefix(unittest.TestCase):
    def test_every_released_tile_has_all_payload_contributions(self):
        rng=random.Random(20260924)
        for n in (1024,1025,2049,4095,4096):
            ids=[rng.randrange(256) for _ in range(n*8)]
            order=sorted(range(n*8),key=ids.__getitem__)
            inverse=[0]*(n*8)
            for dest,source in enumerate(order):inverse[source]=dest
            counts=[ids.count(e) for e in range(256)]
            starts=[];bases=[];row=tiles=0
            for count in counts:
                starts.append(row);bases.append(tiles)
                row+=count;tiles+=(count+63)//64
            deps=[[0]*tiles for _ in range(16)];tile_sources=[[] for _ in range(tiles)]
            for source,e in enumerate(ids):
                token=source//8;worker=(token//8)%16
                tile=bases[e]+(inverse[source]-starts[e])//64
                deps[worker][tile]=max(deps[worker][tile],token+1)
                tile_sources[tile].append(token)
            threshold=[max(deps[w][t] for w in range(16)) for t in range(tiles)]
            rounds=(n+127)//128
            for _ in range(30):
                progress=[rng.randrange(rounds+1) for _ in range(16)]
                prefix=min(n,min(progress)*128)
                for tile,need in enumerate(threshold):
                    if need<=prefix:
                        self.assertTrue(all(token//128<progress[(token//8)%16]
                                            for token in tile_sources[tile]))
            self.assertEqual(max(threshold),n)
            self.assertLessEqual(tiles,1024)

if __name__=='__main__':unittest.main()
