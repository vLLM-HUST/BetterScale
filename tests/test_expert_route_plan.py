"""CPU-only admission/emission guards for the native route-plan wire extension."""
import ast
from importlib.machinery import ModuleSpec
import json
from types import ModuleType, SimpleNamespace as S
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest
from betterscale.patches.expert_service.build import emit
from betterscale.patches.expert_service.route_plan import threshold


class RoutePlan(unittest.TestCase):
    def test_legacy_abi_remains_disabled(self):
        self.assertEqual(threshold({}),0)
        for bad in [{'client_route_plan_min_rows':0},{'client_route_plan':False}]:
            with self.assertRaises(ValueError):threshold(bad)

    def test_explicit_wire_and_topology(self):
        valid=dict(client_route_plan='native-v2-cap1',client_route_plan_min_rows=1024,
                   placement='layer',sources_per_wave=1)
        self.assertEqual(threshold(valid),1024)
        for key,values in [('client_route_plan_min_rows',[0,4097,True,'1024',None]),
                           ('placement',['expert',None]),('sources_per_wave',[2,7,None]),
                           ('client_route_plan',['unknown'])]:
            for value in values:
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    threshold(dict(valid,**{key:value}))

    def test_frame_planning_precedes_publication_and_skips_small_tail(self):
        import torch
        source=Path(__file__).resolve().parents[1]/'src/betterscale/patches/expert_service/persistent_remote.py'
        cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='PersistentRemote')
        call=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
        scope=dict(torch=torch,H=4,K=2)
        exec(compile(ast.Module(body=[call],type_ignores=[]),str(source),'exec'),scope)
        trace=[];bank=S(config=torch.zeros(18,dtype=torch.int64),x=torch.empty(3,4),
                       id_storage=torch.empty(6,dtype=torch.int32),probs=torch.empty(6),
                       output=torch.zeros(3,4),plan_input=torch.zeros(3,32))
        def plan(x,ids,**kwargs):
            self.assertEqual(ids.dtype,torch.int32)
            trace.append(('plan',ids.clone()))
            return (torch.empty(0),torch.arange(ids.numel(),dtype=torch.int32),
                    torch.zeros(256,dtype=torch.int64),torch.empty(0))
        def launch(fn,*args):trace.append((fn,None))
        obj=S(placement=S(targets=lambda _:(0,)),rows=3,python_submissions=0,
              route_plan_min_rows=2,bank=lambda owner,n:bank,kernels=S(call=launch),
              pack='pack',publish='publish',collect='collect',retire='retire',promote='promote',shared_callback=None)
        ids=torch.arange(14).reshape(7,2)
        # Keep this AST-level CPU test from importing the real vLLM/Ascend stack.
        fake_npu=ModuleType('torch_npu');fake_npu.__spec__=ModuleSpec('torch_npu',loader=None)
        fake_npu.npu_moe_init_routing_v2=plan
        fake_vllm=ModuleType('vllm');fake_vllm.__path__=[]
        fake_vllm.__spec__=ModuleSpec('vllm',loader=None,is_package=True)
        fake_forward=ModuleType('vllm.forward_context')
        fake_forward.get_forward_context=lambda:None
        fake_forward.is_forward_context_available=lambda:False
        with patch.dict('sys.modules',{'torch_npu':fake_npu,'vllm':fake_vllm,
                                       'vllm.forward_context':fake_forward}):
            scope['__call__'](obj,40,torch.ones(7,4),ids,torch.ones(7,2))
        self.assertEqual([name for name,_ in trace],
                         ['plan','pack','publish','collect','retire']*2+['pack','publish','collect','retire'])
        self.assertTrue(torch.equal(trace[0][1],ids[:3].int()))
        self.assertTrue(torch.equal(trace[5][1],ids[3:6].int()))

    def test_emission_binds_threshold_and_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            for minimum in [0,1,1024,4096]:
                root=emit(Path(tmp)/str(minimum),draft_layers=1,route_plan_min_rows=minimum)
                abi=json.loads((root/'abi.json').read_text())
                self.assertEqual(threshold(abi),minimum)
                self.assertIn(f'constexpr int MIN_ROWS = {minimum};',(root/'source/route_plan.hpp').read_text())
                source_bytes=((abi['payload_words']*4+abi['rows']*2048*2+(2<<20)-1)//(2<<20))*(2<<20)
                self.assertLessEqual((50176+4096*2048//2+4096*8+512)*4,source_bytes)
            for kwargs in [dict(route_plan_min_rows=True),dict(route_plan_min_rows=-1),
                           dict(route_plan_min_rows=1024,placement='expert'),
                           dict(route_plan_min_rows=1024,sources_per_wave=7)]:
                with self.assertRaises(ValueError):emit(Path(tmp)/'invalid',**kwargs)
                self.assertFalse((Path(tmp)/'invalid').exists())

if __name__=='__main__':unittest.main()
