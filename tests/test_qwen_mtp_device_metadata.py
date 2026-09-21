"""Device slot construction preserves mixed routing and inactive sentinels."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

path=Path(__file__).resolve().parents[1]/'prototypes/qwen38-serving/mtp/device_metadata.py'
spec=importlib.util.spec_from_file_location('mtp_device_metadata',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class DeviceMetadataTest(unittest.TestCase):
    def test_mixed_device_lengths_and_zero_slot(self):
        table=torch.arange(32,dtype=torch.int32).reshape(4,8)
        table[1,1]=0
        seq=torch.tensor([17,19,3],dtype=torch.int32)
        meta=NS(live=3,width=3,decode=False,device_slot_source=(table,seq,16,[0,2],[1]),
            cu=torch.tensor([0,17,20,23,23,23,23,23,23,23],dtype=torch.int32),
            initial=torch.zeros(9,dtype=torch.bool),
            prefill_ids=torch.tensor([0,2]+[0]*7),verify_ids=torch.tensor([1]+[0]*8),
            prefill_conv=torch.full((9,1),-1,dtype=torch.int32),
            verify_conv=torch.full((9,1),-1,dtype=torch.int32),
            prefill=NS(state=torch.zeros((9,2),dtype=torch.int64)),
            verify=NS(slots=torch.full((9,3),-1,dtype=torch.int64)))
        module.publish_slots(meta)
        self.assertEqual(meta.prefill_conv[:3,0].tolist(),[1,-1,16])
        self.assertEqual(meta.verify_conv[:3,0].tolist(),[-1,0,-1])
        self.assertEqual(meta.prefill.state[:2].tolist(),[[1,0],[16,0]])
        self.assertEqual(meta.verify.slots[0].tolist(),[0,10,11])
        self.assertEqual(meta.initial[:3].tolist(),[False,True,False])
        self.assertTrue((meta.verify.slots[1:]==-1).all())

    def test_verify_crosses_column_boundary_on_device(self):
        table=torch.arange(16,dtype=torch.int32).reshape(2,8)
        meta=NS(live=2,width=3,decode=True,
            device_slot_source=(table,torch.tensor([16,17],dtype=torch.int32),16,[],[0,1]),
            verify_conv=torch.full((9,1),-1,dtype=torch.int32),
            verify=NS(slots=torch.full((9,3),-1,dtype=torch.int64)))
        module.publish_slots(meta)
        self.assertEqual(meta.verify.slots[:2].tolist(),[[0,1,2],[9,10,11]])
        self.assertEqual(meta.verify_conv[:2,0].tolist(),[0,9])
        self.assertTrue((meta.verify_conv[2:]==-1).all())
