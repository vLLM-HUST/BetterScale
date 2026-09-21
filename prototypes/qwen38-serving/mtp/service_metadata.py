"""Experimental service adapter for the already-probed mixed core.

Kept outside product admission until the real-model continuation and APC checks
pass. No new Worker: these replace the same builder/publication leaf interfaces.
"""
import copy
from math import prod
from types import SimpleNamespace

import numpy as np
import torch

from mixed_core import MixedCore
from host_metadata import HostMetadata
from betterscale.patches.qwen_gdn.publication import Frame

from count_policy import WIDTH, SPEC_CAPACITIES


class Core(MixedCore):
    def __init__(self, tokens, device):
        # The first service experiment deliberately uses the mixed core for all
        # capacities. Pure verification takes the small branch below; no chunk
        # work is recorded into its graph.
        super().__init__(tokens, device)
        self.tokens = tokens
        self.decode = tokens in SPEC_CAPACITIES
        self.mtp = True
        self.verify_ids = torch.zeros(9, dtype=torch.int64, device=device)
        self.accepted_source = None
        self.live = 0
        self.verify_count = 0

    def __call__(self, x, a, b, weight, log, bias, conv, state):
        if not self.decode:
            return super().__call__(x,a,b,weight,log,bias,conv,state)
        from betterscale.patches.qwen_gdn.preprocess import preprocess
        from betterscale.patches.qwen_gdn.decode_kv import fused_recurrent_gated_delta_rule_fwd

        y = torch.empty_like(x)
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            y,x,weight,conv_state=conv,bias_opt=None,query_start_loc_opt=self.cu,
            cache_indices_opt=self.verify_conv,initial_state_mode_opt=None,
            num_accepted_tokens_opt=self.accepted,activation_mode=1,pad_slot_id=-1,run_mode=1)
        q,k,v,g,beta = preprocess(y,a,b,log,bias)
        return fused_recurrent_gated_delta_rule_fwd(q,k,v,g,beta,128**-.5,state,
            cu_seqlens=self.verify.cu,ssm_state_indices=self.verify.slots,
            num_accepted_tokens=self.verify.accepted)[0]


class MTPFrame(Frame):
    def __init__(self, tokens, groups, device, stream):
        self.stream = stream
        self.uploaded, self.consumed = torch.npu.Event(), torch.npu.Event()
        self.has_upload = self.has_consumer = False
        self.metas = {key:Core(tokens,device) for key in groups}
        fields, offset = [], 0
        for core in self.metas.values():
            host = copy.copy(core)
            host.prefill = copy.copy(core.prefill)
            host.prefill.indices = dict(core.prefill.indices)
            host.verify = copy.copy(core.verify)
            core.host_view = host
            for obj, host_obj in ((core,host),(core.prefill,host.prefill),(core.verify,host.verify)):
                for name, value in vars(obj).copy().items():
                    if not isinstance(value,torch.Tensor):
                        continue
                    offset = (offset+7)//8*8
                    fields.append((obj,host_obj,name,value.shape,value.dtype,offset))
                    offset += value.numel()*value.element_size()
            for size,value in core.prefill.indices.items():
                offset = (offset+7)//8*8
                fields.append((core.prefill.indices,host.prefill.indices,size,value.shape,value.dtype,offset))
                offset += value.numel()*value.element_size()
        self.host = torch.empty(offset,dtype=torch.uint8,pin_memory=True)
        self.device = torch.empty(offset,dtype=torch.uint8,device=device)
        for obj,host_obj,name,shape,dtype,start in fields:
            size = prod(shape)*torch.empty((),dtype=dtype).element_size()
            for target,storage in ((obj,self.device),(host_obj,self.host)):
                value = storage[start:start+size].view(dtype).view(shape)
                if isinstance(target,dict):
                    target[name] = value
                else:
                    setattr(target,name,value)
        # All inactive fields are deterministic, including padded request rows.
        self.host.zero_()
        for core in self.metas.values():
            core.host_metadata = HostMetadata(core.host_view)
        self.stream.wait_stream(torch.npu.current_stream())

    def fill_mtp(self,key,m,lengths,table,builder,accepted,drafts):
        meta = self.metas[key]
        n = len(lengths)
        assert 0 < n <= 8 and sum(lengths) <= meta.tokens
        seq = m.seq_lens_cpu if m.seq_lens_cpu is not None else m._seq_lens_cpu
        if seq is None:
            raise ValueError('MTP requires the pinned runner corrected CPU lengths')
        seq = seq[:n].numpy()
        aligned = builder.vllm_config.cache_config.mamba_cache_mode == 'align'
        columns = np.maximum((seq-1)//builder.kv_cache_spec.block_size,0) if aligned else np.zeros(n,dtype=np.int64)
        slots = table[np.arange(n)[:,None],columns[:,None]+np.arange(WIDTH)]
        roles = ([meta.decode]*n if drafts is None else (drafts[:n].numpy()>=0).tolist())
        meta.host_metadata.prepare(lengths,roles,slots,seq>lengths)
        meta.live,meta.verify_count = n,sum(roles)
        meta.accepted_source = accepted
        return meta,roles

    def publish(self):
        super().publish()
        for meta in self.metas.values():
            if meta.accepted_source is not None:
                # Device feedback is not staged through CPU. The native runner
                # has already remapped requests and applied APC boundary reset.
                meta.accepted[:meta.live].copy_(meta.accepted_source[:meta.live])
                if meta.verify_count:
                    meta.verify.accepted[:meta.verify_count].copy_(
                        meta.accepted_source.index_select(0,meta.verify_ids[:meta.verify_count]))
