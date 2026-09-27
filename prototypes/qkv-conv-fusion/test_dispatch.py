"""CPU regression for polymorphic dispatch and fake/real gate strides."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch


class DispatchTests(unittest.TestCase):
    def test_runtime_shapes_and_fake_contract(self):
        callbacks={};used=[]
        ctx=NS(no_compile_layers={},attn_metadata={})
        qwen=ModuleType('betterscale.models.qwen');qwen.before_init=lambda *args:None
        models=ModuleType('betterscale.models');models.qwen=qwen
        utils=ModuleType('vllm.utils.torch_utils')
        utils.direct_register_custom_op=lambda **kw:callbacks.update(kw)
        context=ModuleType('vllm.forward_context');context.get_forward_context=lambda:ctx
        adapter=ModuleType('pool_adapter')
        def project(x,w,*args):
            used.append(x.shape[0]);z=x@w.T;return z,z
        adapter.project_conv=project
        mods={'betterscale':ModuleType('betterscale'),'betterscale.models':models,
              'vllm':ModuleType('vllm'),'vllm.utils':ModuleType('vllm.utils'),
              'vllm.utils.torch_utils':utils,'vllm.forward_context':context,'pool_adapter':adapter}
        with patch.dict(sys.modules,mods),tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'CAPSULE':tmp}):
            spec=importlib.util.spec_from_file_location('tested_fusion',Path(__file__).with_name('service_fusion.py'))
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.install()
            # Small channel geometry keeps this CPU test cheap; token dispatch,
            # GQA gate head flattening and real/fake layout are the real contracts.
            w=torch.arange(20,dtype=torch.float32).view(10,2)*.01
            calls=[]
            class Projection:
                weight=w
                def __call__(self,x):return x@w.T,None
            layer=NS(in_proj_qkvz=Projection(),key_dim=1,value_dim=4,tp_size=1,
                     num_v_heads=2,head_v_dim=2,conv1d=NS(weight=torch.zeros(6,1,4)),
                     A_log=None,dt_bias=None,kv_cache=(None,None))
            layer.in_proj_ba=lambda x:(torch.zeros(x.shape[0],4),None)
            layer._split_ba_for_tp=lambda ba:ba.chunk(2,dim=-1)
            def old(mixed_qkv,b,a,core_attn_out):
                calls.append(mixed_qkv.shape[0]);core_attn_out.copy_(mixed_qkv[:,:4].reshape(-1,2,2))
            layer._forward_core=old;ctx.no_compile_layers['layer']=layer
            core,fake=callbacks['op_func'],callbacks['fake_impl']
            for m in (1,3,48,128,512,1536,2048,4096):
                x=torch.arange(m*2,dtype=torch.float32).view(m,2)*.001
                out=torch.zeros(m,2,2)
                ctx.attn_metadata['layer']=NS(owned=NS(tokens=m,decode=False,cu=None,
                    prefill_conv=None,initial=None,verify_conv=None,accepted=None,
                    after_conv=lambda y,*args:y[:,:4].reshape(1,-1,2,2)))
                gate=core(x,out,'layer');abstract=fake(x,out,'layer')
                torch.testing.assert_close(gate,(x@w.T)[:,6:].reshape(-1,2))
                torch.testing.assert_close(out,(x@w.T)[:,:4].reshape(m,2,2))
                self.assertEqual(gate.shape,abstract.shape)
                self.assertEqual(gate.stride(),abstract.stride())
            self.assertEqual(used,[2048,4096])
            self.assertEqual(calls,[1,3,48,128,512,1536])
            self.assertEqual(callbacks['mutates_args'],['out'])

if __name__=='__main__':unittest.main()
