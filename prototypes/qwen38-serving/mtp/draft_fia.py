"""Experimental wave-FIA publication for native merged MTP draft graphs.

The runnable wrapper is outside ACLGraphWrapper, so one publication precedes
replay. Each bank/shape/draft-position owns a distinct metadata frame. Actual KV
lengths come from device tensors; CPU lengths only describe a tiling envelope.
"""
import copy
import ctypes
from functools import wraps
import torch


def compact_padding(metadata):
    # Native merged draft pads its one-token phase to the *token* capacity,
    # potentially2048 rows although this service admits only8 real requests.
    # Zero-KV padding rows can share one ignored tail query; never fold live rows.
    if len(metadata.actual_seq_lengths_q) <= 9:
        return metadata
    if any(metadata.seq_lens_list[8:]):
        raise ValueError('Draft padding compaction encountered a live ninth request')
    result = copy.copy(metadata)
    result.actual_seq_lengths_q = metadata.actual_seq_lengths_q[:8] + [metadata.actual_seq_lengths_q[-1]]
    result.seq_lens_list = metadata.seq_lens_list[:8] + [0]
    result.seq_lens = metadata.seq_lens[:8]
    result._mtp_device_seq_lens = metadata._mtp_device_seq_lens[:8]
    return result


def bind_device_lengths(metadata, common):
    # AscendMetadata.seq_lens is a CPU mirror in this pinned builder, despite
    # its name. Preserve the CommonAttentionMetadata device plane explicitly.
    source = common.seq_lens
    if source.device.type == 'cpu':
        raise ValueError('MTP FIA actual lengths must remain device-authoritative')
    metadata._mtp_device_seq_lens = source
    return metadata


def install():
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackendImpl as Impl, AscendAttentionMetadataBuilder as Builder)
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer as Proposer
    from betterscale.patches.qwen_fia import check_library
    from betterscale.patches.qwen_fia.wave import Planner, Frame, install_replay_ordering
    if getattr(Proposer, '_mtp_wave_fia', False):
        return
    library = check_library()
    active = {}
    original_fia = Impl.forward_fused_infer_attention
    original_load = Proposer.load_model
    original_update = Proposer._update_full_graph_params
    original_prepare = Frame.prepare
    original_build = Builder.build

    def build(builder, common_prefix_len, common_attn_metadata, fast_build=False):
        result = original_build(builder,common_prefix_len,common_attn_metadata,fast_build)
        return bind_device_lengths(result,common_attn_metadata)

    Builder.build = build

    def prepare(frame, planner, metadata, cpu, num_reqs):
        original_prepare(frame, planner, metadata, cpu, num_reqs)
        # The pinned donor owns the actual device lengths; never reconstruct
        # them from a host receipt. Copy AFTER the bank's slab publication.
        source = metadata._mtp_device_seq_lens
        if source.device.type == 'cpu':
            raise ValueError('FIA publication received a CPU actual-length mirror')
        n = min(source.numel(), len(metadata.actual_seq_lengths_q))
        frame.kv[:n].copy_(source[:n])

    Frame.prepare = prepare

    class Runnable:
        def __init__(self, proposer, inner):
            self.proposer, self.inner = proposer, inner
            self.frames = {}
            self.planner = Planner(library)

        def __call__(self, **kw):
            context = get_forward_context()
            if context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
                if (getattr(self.proposer.runner, '_mtp_device_lengths', False)
                        and self.proposer.runner.input_batch.num_reqs):
                    raise RuntimeError('Device-length MTP requires FULL draft execution')
                return self.inner(**kw)
            proposer = self.proposer
            runner = proposer.runner
            entries = {}
            for step, metadata in enumerate(kw['multi_steps_attn_metadata']):
                layer, m = next(iter(metadata.items()))
                gid = next(i for i,g in enumerate(runner.kv_cache_config.kv_cache_groups)
                           if layer in g.layer_names)
                cpu = runner.input_batch.block_table[gid].get_cpu_tensor()
                key = kw['num_input_tokens'], runner._owned_bank, step
                plan_metadata = compact_padding(m)
                entry = dict(key=key, metadata=plan_metadata, cpu=cpu,
                             num_reqs=min(len(plan_metadata.actual_seq_lengths_q), cpu.shape[0]))
                entries[id(m)] = entry
                if key in self.frames:
                    self.frames[key].prepare(self.planner,plan_metadata,cpu,entry['num_reqs'])
            if active:
                raise RuntimeError('Draft FIA requires a serial non-reentrant proposer')
            active.update(owner=self, entries=entries)
            try:
                if any(entry['key'] not in self.frames for entry in entries.values()):
                    # Native dummy warmups may be NONE and carry no draft
                    # metadata. Warm this exact FULL envelope explicitly,
                    # before ACLGraphWrapper flips context.capturing on.
                    if (runner.input_batch.num_reqs or context.capturing
                            or getattr(runner,'_owned_capture_bank',None) is None):
                        raise RuntimeError('Missing draft FIA envelope outside disposable startup')
                    saved_metadata = context.attn_metadata
                    try:
                        self.inner.runnable(**kw)
                        torch.npu.current_stream().synchronize()
                    finally:
                        context.attn_metadata = saved_metadata
                return self.inner(**kw)
            finally:
                for entry in entries.values():
                    frame = self.frames.get(entry['key'])
                    if frame is not None:
                        frame.release()
                active.clear()

    @wraps(original_load)
    def load(proposer, *a, **kw):
        result = original_load(proposer, *a, **kw)
        assert proposer.method == 'mtp' and proposer.num_speculative_tokens == 2
        proposer._runnable = Runnable(proposer, proposer._runnable)
        return result

    def fia(impl, query, key, value, m, output, kv_cache=None):
        if not active:
            return original_fia(impl,query,key,value,m,output,kv_cache)
        entry = active['entries'][id(m)]
        owner = active['owner']
        planner = owner.planner
        if ((impl.num_heads,impl.num_kv_heads,impl.head_size)!=(12,2,256)
                or impl.sinks is not None or impl.sliding_window is not None or not m.causal):
            raise ValueError('Unqualified MTP draft FIA geometry')
        key,value,block_size,_,_ = impl._get_fia_params(key,value,m,kv_cache)
        if block_size!=128 or any(t.dtype!=torch.bfloat16 or not t.is_contiguous()
                                 for t in (query,key,value,output)):
            raise ValueError('Draft wave FIA requires contiguous BF16 paged128')
        if planner.fixtures is None:
            fixture = torch.empty((2048,12,256),device=query.device,dtype=query.dtype)
            planner.fixtures = fixture,key,value,torch.empty_like(fixture),impl.scale
        if entry['key'] not in owner.frames:
            if get_forward_context().capturing:
                raise RuntimeError('Draft wave FIA must warm before capture')
            frame = Frame(query.shape[0],entry['cpu'].shape[1],query.device,
                          owner.proposer.runner._owned_ingress)
            frame.prepare(planner,entry['metadata'],entry['cpu'],entry['num_reqs'])
            owner.frames[entry['key']] = frame
        frame = owner.frames[entry['key']]
        if query.shape[0]!=frame.tokens or key.shape!=planner.fixtures[1].shape:
            raise ValueError('Draft graph changed its captured FIA capacity')
        plan = planner.check(library.plan_clone(frame.plan))
        try:
            scratch = torch.empty(frame.workspace,dtype=torch.uint8,device=query.device)
            ptrs = (ctypes.c_uint64*9)(*[t.data_ptr() for t in
                (query,m.attn_mask,frame.q,frame.kv,frame.table,output,scratch,key,value)])
            planner.status(library.plan_bind(plan,ptrs))
            planner.status(library.plan_bind_metadata(plan,frame.tiling.data_ptr()))
            planner.status(library.plan_launch(plan,torch.npu.current_stream().npu_stream))
        finally:
            planner.status(library.plan_release(plan))
        return output

    def update(proposer, context, *a, **kw):
        if isinstance(proposer._runnable,Runnable) and context.cudagraph_runtime_mode==CUDAGraphMode.FULL:
            return
        return original_update(proposer,context,*a,**kw)

    def prime_startup():
        runner = active['owner'].proposer.runner
        if getattr(runner,'_owned_capture_bank',None) is None:
            return False
        if runner.input_batch.num_reqs:
            raise RuntimeError('Draft graph priming requires empty startup state')
        return True

    # Without native task updates, the donor's pre-replay host drain is no
    # longer the owner of ordering. Reuse the qualified target replay boundary.
    install_replay_ordering(lambda:bool(active),prime_startup)
    Proposer.load_model = load
    Proposer._update_full_graph_params = update
    Impl.forward_fused_infer_attention = fia
    Proposer._mtp_wave_fia = True
