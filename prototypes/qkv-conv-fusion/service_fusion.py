"""Default fusion for qualified35B TP2 envelopes, with runtime shape dispatch.

Dynamo's one polymorphic backbone covers every capture capacity: shape policy
must live INSIDE the opaque operator, never in the traced Python forward.
"""
import json
import os
from pathlib import Path
import torch
from pool_adapter import project_conv


def install():
    from betterscale.models import qwen
    from vllm.utils.torch_utils import direct_register_custom_op
    from vllm.forward_context import get_forward_context

    seen = set()

    def core(x: torch.Tensor, out: torch.Tensor, prefix: str) -> torch.Tensor:
        ctx = get_forward_context()
        layer = ctx.no_compile_layers[prefix]
        n = (layer.key_dim*2+layer.value_dim)//layer.tp_size
        m = x.shape[0]
        shape = (m,layer.num_v_heads//layer.tp_size,layer.head_v_dim)
        # This callback executes with real tensors for each graph capture/eager
        # call. Replays retain their captured branch and device-dynamic metadata.
        if m not in (2048,4096) or ctx.attn_metadata is None:
            projection,_ = layer.in_proj_qkvz(x)
            qkv,gate = projection.split([n,layer.value_dim//layer.tp_size],dim=-1)
            ba,_ = layer.in_proj_ba(x)
            b,a = layer._split_ba_for_tp(ba)
            layer._forward_core(mixed_qkv=qkv,b=b.contiguous(),a=a.contiguous(),core_attn_out=out)
            return gate.reshape(-1,layer.head_v_dim)

        meta = ctx.attn_metadata[prefix].owned
        assert not meta.decode and meta.tokens == m
        key = (prefix,int(m))
        if key not in seen:
            seen.add(key)
            root = Path(os.environ['CAPSULE'])
            (root/f'fusion-dispatch-{os.getpid()}.json').write_text(json.dumps(sorted(seen),indent=2)+'\n')
        w = layer.in_proj_qkvz.weight
        cw = layer.conv1d.weight.view(n,4).T
        transformed,_ = project_conv(x,w[:n],cw,layer.kv_cache[0],meta.cu,
            meta.prefill_conv,meta.initial,meta.verify_conv,meta.accepted)
        ba,_ = layer.in_proj_ba(x)
        b,a = layer._split_ba_for_tp(ba)
        gate = torch.mm(x,w[n:].T).reshape(shape)
        result = meta.after_conv(transformed,a.contiguous(),b.contiguous(),
                                 layer.A_log,layer.dt_bias,layer.kv_cache[1])
        out.copy_(result.squeeze(0))
        return gate.reshape(-1,layer.head_v_dim)

    def fake(x: torch.Tensor, out: torch.Tensor, prefix: str) -> torch.Tensor:
        return torch.empty((out.shape[0]*out.shape[1],out.shape[2]),dtype=out.dtype,device=out.device)

    direct_register_custom_op(op_name='betterscale_qkv_conv_core',op_func=core,
                              mutates_args=['out'],fake_impl=fake)
    before = qwen.before_init

    def before_init(worker,config):
        before(worker,config)
        from vllm_ascend.patch.worker.patch_qwen3_5 import _GDN_PATCH_TARGET as cls

        def forward(self,hidden_states,output=None):
            # Model/geometry admission is static; token dispatch is inside core.
            if self.gqa_interleaved_layout or hasattr(self,'in_proj_qkv'):
                raise ValueError('fusion requires qualified contiguous Qwen35 QKVZ')
            m = hidden_states.shape[0]
            shape = (m,self.num_v_heads//self.tp_size,self.head_v_dim)
            out = torch.zeros(shape,dtype=hidden_states.dtype,device=hidden_states.device)
            gate = torch.ops.vllm.betterscale_qkv_conv_core(hidden_states,out,self.prefix)
            norm = self.norm(out.reshape(-1,self.head_v_dim),gate)
            result,_ = self.out_proj(norm.reshape(m,-1))
            if output is not None:output[:m]=result
            return result
        cls.forward = forward
    qwen.before_init = before_init
