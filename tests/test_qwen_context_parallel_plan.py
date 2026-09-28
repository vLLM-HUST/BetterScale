"""Partition coverage and byte-level producer/reducer ownership; CPU only."""
import importlib.util
from pathlib import Path
import random
import ctypes
import sys
from types import SimpleNamespace
import struct
import unittest

path = Path(__file__).parents[1]/'src/betterscale/patches/qwen_fia/context_parallel/plan.py'
spec = importlib.util.spec_from_file_location('cp_plan', path)
cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)
sys.path.insert(0,str(path.parents[4]))
from betterscale.patches.qwen_fia.context_parallel import adapter


def native(batch):
    raw = bytearray(2528)
    for offset, value in ((0,8),(4,256),(8,256),(16,128),(28,1),(32,batch),(40,1),(44,batch)):
        struct.pack_into('<I', raw, offset, value)
    struct.pack_into('<Q', raw, 56, 66060288)
    return raw


class PlanTest(unittest.TestCase):
    def verify(self, lengths, queries):
        plan = cp.schedule(lengths, queries)
        encoded = cp.encode(native(len(lengths)), plan)
        self.assertEqual(len(encoded), 4096)
        tasks = [[] for _ in lengths]
        for core, group in enumerate(plan['groups']):
            self.assertGreater(len(group), 0)
            for task, lo, hi in group:
                self.assertLess(lo, hi)
                tasks[task].append((lo, hi, core))
        self.assertLessEqual(len(plan['groups']), 24)
        written = set()
        node = 0
        for task, pieces in enumerate(tasks):
            last = 0
            for lo, hi, _ in pieces:
                self.assertEqual(lo, last); last = hi
            self.assertEqual(last, (lengths[task]+511)//512)
            if len(pieces) == 1:
                continue
            self.assertEqual(struct.unpack_from('<i', encoded, 1448+4*node)[0], task)
            self.assertEqual(struct.unpack_from('<i', encoded, 1448+104*4+4*node)[0], queries[task])
            base = struct.unpack_from('<Q', encoded, 2072+8*node)[0]
            self.assertEqual(struct.unpack_from('<Q', encoded, 2280+8*node)[0], base*256)
            for part, (_, _, core) in enumerate(pieces):
                before = sum(queries[t]*8 for t, _, _ in plan['groups'][core] if t < task and len(tasks[t])>1)
                start = struct.unpack_from('<I', encoded, 2560+32*core+20)[0] + before
                self.assertEqual(start, base + part*queries[task]*8)
                rows = set(range(start, start+queries[task]*8))
                self.assertFalse(rows & written); written |= rows
            node += 1
        self.assertEqual(node, plan['split_nodes'])
        self.assertLessEqual(len(written)*4, cp.LSE_BYTES)
        self.assertLessEqual(len(written)*256*4, cp.PARTIAL_O_BYTES)
        return plan

    def test_no_merge_for_short_balanced(self):
        for b in (2,4,8,16):
            self.assertEqual(self.verify([1024]*b, [3]*b)['split_nodes'], 0)

    def test_skew_and_query_rows(self):
        p = self.verify([32768]*14+[262144]*2, [1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,3])
        self.assertGreater(p['split_nodes'], 0)
        self.assertLess(p['cap'], p['uncut_cap'])

    def test_tile_and_capacity_boundaries(self):
        rng = random.Random(173)
        for _ in range(100):
            b = rng.randint(1,96)
            lengths = [rng.choice((3,127,128,129,511,512,513,1023,1024,1025,32768,262144)) for _ in range(b)]
            self.verify(lengths, [rng.randint(1,3) for _ in range(b)])

    def test_device_length_scaling_and_padding_abi(self):
        rng = random.Random(731)
        for _ in range(100):
            b = rng.randint(2,96)
            upper = [rng.choice((513,32768,262144)) for _ in range(b)]
            qs = [rng.randint(1,3) for _ in range(b)]
            plan = self.verify(upper, qs)
            encoded = cp.encode(native(b+1), plan)
            self.assertEqual(struct.unpack_from('<I',encoded,3328)[0], b)
            for actual in ([3]*b, upper, [rng.randint(3,262144) for _ in range(b)]):
                ends = [0]*b
                for group in cp.scaled_groups(plan,actual):
                    for task,lo,hi in group:
                        self.assertEqual(lo,ends[task])
                        self.assertLessEqual(lo,hi)
                        ends[task] = hi
                self.assertEqual(ends,[(n+511)//512 for n in actual])
            for core,group in enumerate(plan['groups']):
                denoms = struct.unpack_from('<2I',encoded,2560+32*core+24)
                self.assertEqual(denoms,tuple((upper[t]+511)//512 for t in (group[0][0],group[-1][0])))

    def test_adapter_publication_keeps_native_workspace(self):
        storage = ctypes.create_string_buffer(bytes(native(17))+bytes(4096-2528))
        frame = SimpleNamespace(context_parallel=True,tokens=48,workspace=123,
            h_tiling=SimpleNamespace(data_ptr=lambda:ctypes.addressof(storage)))
        ends = list(range(3,49,3))+[61]
        metadata = SimpleNamespace(actual_seq_lengths_q=ends,
            seq_lens_list=[32768]*14+[262144]*2+[0])
        adapter.prepare(frame,metadata)
        self.assertEqual(frame.workspace,123)
        self.assertEqual(struct.unpack_from('<I',storage.raw,3328)[0],16)
        self.assertGreater(frame.cp_statistics[1],0)
        self.assertLessEqual(frame.cp_workspace,128<<20)
        # Published metadata lives within the slab; adjacent q/kv/table are not touched.
        self.assertEqual(len(storage.raw),4097)
        frame.context_parallel=False
        before=storage.raw
        adapter.prepare(frame,SimpleNamespace())
        self.assertEqual(storage.raw,before)

    def test_disposable_capture_does_not_relax_live_admission(self):
        for tokens in (6,12,24,40,48):
            n = min(tokens,16)
            q = [tokens//n]*n
            q[-1] += tokens%n
            ends = list(__import__('itertools').accumulate(q))
            source = object()
            m = SimpleNamespace(actual_seq_lengths_q=ends,seq_lens_list=[tokens]*n,
                                _mtp_device_seq_lens=source)
            adjusted = adapter.capture_metadata(m,tokens)
            self.assertEqual(m.actual_seq_lengths_q,ends)
            self.assertIs(adjusted._mtp_device_seq_lens,source)
            lengths = [hi-lo for lo,hi in zip([0]+adjusted.actual_seq_lengths_q,adjusted.actual_seq_lengths_q)]
            self.assertEqual(lengths[:n],[min(x,3) for x in q])
            self.assertEqual(adjusted.actual_seq_lengths_q[-1],tokens)
            self.assertEqual(len(adjusted.seq_lens_list),n+int(sum(lengths[:n])<tokens))
            self.assertEqual(cp.schedule([tokens]*n,lengths[:n])['split_nodes'],0)
            if max(q)>3:
                with self.assertRaises(ValueError):cp.schedule([tokens]*n,q)

    def test_reject_wrong_geometry_and_non_verification(self):
        for lengths, queries in (([0],[1]),([2],[3]),([262145],[3]),([1024],[4]),([1024]*2,[3])):
            with self.assertRaises(ValueError):cp.schedule(lengths, queries)
        plan = cp.schedule([1024,2048], [3,3])
        raw = native(2);struct.pack_into('<I',raw,4,128)
        with self.assertRaises(ValueError):cp.encode(raw,plan)


if __name__ == '__main__':unittest.main()
