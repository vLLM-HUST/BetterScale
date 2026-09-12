"""CPU contracts for opt-in stream admission and stable draft metadata banks."""
import ast
import copy
import dataclasses
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

class Proxy:
    def __init__(self,data,idx=0):self._data=data;self.idx=idx

class Protocol(unittest.TestCase):
    def test_private_bank_updates_without_rebinding(self):
        p=Path(__file__).with_name('draft_graph.py')
        fs=[x for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef) and x.name in ('bank','refresh','signature')]
        ns=dict(torch=torch,copy=copy,dataclasses=dataclasses,Enum=Enum,RopeDataProxy=Proxy)
        exec(compile(ast.Module(body=fs,type_ignores=[]),str(p),'exec'),ns)
        @dataclasses.dataclass
        class Metadata:
            rope:object
            query:object
            count:int
        original=Metadata(Proxy({'c4':torch.arange(4)}),torch.tensor([0,2,4]),2)
        private=ns['bank'](original);ptr=private.rope._data['c4'].data_ptr()
        original.rope._data['c4'].add_(9)
        self.assertNotEqual(private.rope._data['c4'].tolist(),original.rope._data['c4'].tolist())
        ns['refresh'](private,original)
        self.assertEqual(ptr,private.rope._data['c4'].data_ptr())
        self.assertEqual(private.rope._data['c4'].tolist(),original.rope._data['c4'].tolist())
        sig=ns['signature'](original);original.count=3
        self.assertNotEqual(sig,ns['signature'](original))
        original.count=2;original.rope.idx=1
        self.assertNotEqual(sig,ns['signature'](original))

    def test_cpu_qli_uses_existing_mirrors_and_checks_them(self):
        p=Path(__file__).with_name('qli_cpu.py')
        f=next(x for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='_cpu_qli_metadata')
        recorded=[]
        fake=NS(ops=NS(_C_ascend=NS(npu_vllm_quant_lightning_indexer_metadata=lambda **kw:(recorded.append(kw) or torch.zeros(1024)))))
        ns=dict(torch=fake,_enabled=True,_verify=True,_original=lambda *a:'fallback')
        exec(compile(ast.Module(body=[f],type_ignores=[]),str(p),'exec'),ns)
        qsl=torch.tensor([0,2,6]);sl=torch.tensor([12,18]);ql=torch.tensor([2,4])
        builder=NS(compressor_ratio=4,common_ratio_to_sas_metadata={'_cpu_local':{'qsl_cpu':qsl,'sl_cpu':sl}},
                   model_config=NS(hf_config=NS(index_n_heads=64,index_head_dim=128,index_topk=512)),
                   seqused_q=torch.zeros(2),req_qli_metadata=torch.zeros(1024))
        ns['_cpu_qli_metadata'](builder,qsl,sl,ql,2)
        self.assertEqual((recorded[0]['max_seqlen_q'],recorded[0]['max_seqlen_k']),(4,18))
        builder.common_ratio_to_sas_metadata.pop('cp_qli')
        with self.assertRaises(AssertionError):ns['_cpu_qli_metadata'](builder,qsl,sl+1,ql,2)
        ns['_enabled']=False;self.assertEqual(ns['_cpu_qli_metadata'](builder,qsl,sl,ql,2),'fallback')

    def test_only_admitted_stream_replays(self):
        p=Path(__file__).with_name('ordered_replay.py')
        f=next(x for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='call')
        calls=[];ctx=NS(batch_descriptor='x',cudagraph_runtime_mode='FULL',capturing=False)
        fake=NS(npu=NS(current_stream=lambda:NS(npu_stream=7)))
        ns=dict(torch=fake,CUDAGraphMode=NS(FULL='FULL'),get_forward_context=lambda:ctx)
        exec(compile(ast.Module(body=[f],type_ignores=[]),str(p),'exec'),ns)
        w=NS(_ordered_replay_stream=7,runtime_mode='FULL',is_debugging_mode=False,
             concrete_aclgraph_entries={'x':NS(aclgraph=NS(replay=lambda:calls.append('replay')),output='result')})
        original=lambda *a,**kw:'fallback'
        self.assertEqual(ns['call'](original,w),'result');self.assertEqual(calls,['replay'])
        ctx.capturing=True;self.assertEqual(ns['call'](original,w),'fallback')
        ctx.capturing=False;w._ordered_replay_stream=8
        with self.assertRaises(AssertionError):ns['call'](original,w)
        w._ordered_replay_stream=None;self.assertEqual(ns['call'](original,w),'fallback')

if __name__=='__main__':unittest.main()
