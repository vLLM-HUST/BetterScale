"""Only inactive draft padding may collapse to one zero-KV FIA row."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

path=Path(__file__).resolve().parents[1]/'prototypes/qwen38-serving/mtp/draft_fia.py'
spec=importlib.util.spec_from_file_location('mtp_draft_fia',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class DraftFIATest(unittest.TestCase):
    def test_preserve_live_prefix_and_exact_total_query_extent(self):
        for capacity in (12,24,512,2048):
            lengths=[3073+i for i in range(8)]+[0]*(capacity-8)
            m=NS(actual_seq_lengths_q=list(range(1,capacity+1)),seq_lens_list=lengths,
                 seq_lens=torch.tensor(lengths,dtype=torch.int32),
                 _mtp_device_seq_lens=torch.tensor(lengths,dtype=torch.int32),attn_mask=object())
            out=module.compact_padding(m)
            self.assertEqual(out.actual_seq_lengths_q,list(range(1,9))+[capacity])
            self.assertEqual(out.seq_lens_list,lengths[:8]+[0])
            self.assertEqual(out.seq_lens.data_ptr(),m.seq_lens.data_ptr())
            self.assertEqual(len(m.actual_seq_lengths_q),capacity)
            self.assertIs(out.attn_mask,m.attn_mask)

    def test_reject_live_tail_and_preserve_small_envelope(self):
        m=NS(actual_seq_lengths_q=list(range(1,13)),seq_lens_list=[9]*9+[0]*3)
        with self.assertRaisesRegex(ValueError,'live ninth'):
            module.compact_padding(m)
        m=NS(actual_seq_lengths_q=[3,6,12])
        self.assertIs(module.compact_padding(m),m)

    def test_cpu_mirror_is_not_mistaken_for_device_plane(self):
        cpu=torch.tensor([123],dtype=torch.int32)
        device=NS(device=NS(type='npu'))
        result=NS(seq_lens=cpu)
        self.assertIs(module.bind_device_lengths(result,NS(seq_lens=device)),result)
        self.assertIs(result.seq_lens,cpu)
        self.assertIs(result._mtp_device_seq_lens,device)
        with self.assertRaisesRegex(ValueError,'device-authoritative'):
            module.bind_device_lengths(result,NS(seq_lens=cpu))
