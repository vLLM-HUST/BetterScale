"""Publish actual GDN slot addresses from device lengths, not host estimates.

Host still supplies query partition, roles and resource-table entries. This
NPU path fuses addressing/publication in one kernel. The tensor reference below
is retained for independent CPU checks of routing and inactive sentinels.
"""
import torch


def publish_slots_reference(meta):
    table, seq, block, pre, verify = meta.device_slot_source
    n = meta.live
    columns = (seq[:n] - 1).clamp_min(0) // block
    rows = torch.arange(n,device=seq.device)
    offsets = torch.arange(meta.width,device=seq.device)
    slots = table[rows[:,None],columns[:,None]+offsets]
    if meta.decode:
        meta.verify_conv[:n,0].copy_(slots[:,0])
        meta.verify.slots[:n].copy_(slots)
        return
    initial = seq[:n] > (meta.cu[1:n+1]-meta.cu[:n])
    meta.initial[:n].copy_(initial)
    # These are host-known membership indices, never accepted lengths.
    pre = meta.prefill_ids[:len(pre)]
    verify = meta.verify_ids[:len(verify)]
    meta.prefill_conv[:,0].index_copy_(0,pre,slots[pre,0].to(meta.prefill_conv.dtype))
    meta.prefill.state[:pre.numel(),0].copy_(slots[pre,0])
    meta.prefill.state[:pre.numel(),1].copy_(initial[pre])
    meta.verify_conv[:,0].index_copy_(0,verify,slots[verify,0].to(meta.verify_conv.dtype))
    meta.verify.slots[:verify.numel()].copy_(slots[verify])


def publish_slots(meta):
    if meta.device_slot_source[1].device.type == 'cpu':
        return publish_slots_reference(meta)
    from device_slots import publish_slots as publish_device_slots
    return publish_device_slots(meta)


def install():
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner as Runner
    original_inputs = Runner._prepare_inputs

    def inputs(runner, *a, **kw):
        assert not runner.use_compress and runner.use_async_spec_decode
        runner._needs_seq_lens_cpu_sync = False
        runner._mtp_device_lengths = True
        return original_inputs(runner, *a, **kw)

    Runner._prepare_inputs = inputs
    from service_metadata import Core, MTPFrame
    original_core = Core.__init__

    def core_init(core,tokens,device):
        original_core(core,tokens,device)
        # MTPFrame includes this field automatically in its banked pinned slab.
        core.prefill_ids = torch.zeros(9,dtype=torch.int64,device=device)

    Core.__init__ = core_init
    original_fill,original_publish = MTPFrame.fill_mtp,MTPFrame.publish

    def fill(frame,key,m,lengths,table,builder,accepted,drafts):
        meta,roles = original_fill(frame,key,m,lengths,table,builder,accepted,drafts)
        host_ids = meta.host_view.prefill_ids.numpy()
        host_ids.fill(0)
        pre = [i for i,role in enumerate(roles) if not role]
        host_ids[:len(pre)] = pre
        meta.device_slot_source = (m.block_table_tensor,m.seq_lens,builder.kv_cache_spec.block_size,
            [i for i,role in enumerate(roles) if not role],
            [i for i,role in enumerate(roles) if role])
        return meta,roles

    def publish(frame):
        original_publish(frame)
        for meta in frame.metas.values():
            if hasattr(meta,'device_slot_source'):
                publish_slots(meta)

    MTPFrame.fill_mtp=fill
    MTPFrame.publish=publish
