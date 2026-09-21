import hashlib
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/'prototypes/qwen38-serving/mtp/apc_protocol.py'
spec = importlib.util.spec_from_file_location('apc_protocol',path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def hashes(tokens,old=(),salt=None):
    return m.extend_hashes(tokens,old,4,
        lambda p,t,e: hashlib.sha256(repr((p,t,e)).encode()).digest(),
        lambda a,b: (salt,))


class LookaheadIdentity(unittest.TestCase):
    def test_wait_for_boundary_token(self):
        self.assertEqual(hashes([0,1,2,3]),[])
        self.assertEqual(len(hashes([0,1,2,3,4])),1)

    def test_divergent_next_token_never_aliases(self):
        self.assertNotEqual(hashes([0,1,2,3,4]),hashes([0,1,2,3,5]))
        self.assertEqual(hashes([0,1,2,3,4,6]),hashes([0,1,2,3,4,7]))

    def test_incremental_and_salt(self):
        tokens=list(range(13));old=hashes(tokens[:5])
        self.assertEqual(old+hashes(tokens,old),hashes(tokens))
        self.assertNotEqual(hashes(tokens,salt='a'),hashes(tokens,salt='b'))

if __name__=='__main__':unittest.main()
