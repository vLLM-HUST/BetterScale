"""Graph-compatible target-only EP participation with no resident State writer.

The native scheduler owns the wave; this does not create requests or advance I/O.
Dummy attention reads the allocator's permanently reserved, zeroed null FA page.
GDN uses the qualified negative-slot padding ABI. Startup capture is unchanged.
"""
import copy


def blank_gdn(meta):
    meta.verify_conv.fill_(-1)
    meta.verify.slots.fill_(-1)
    meta.verify.accepted.fill_(1)
    if not meta.decode:
        # A peer can select a prefill bucket during its first target wave.
        # Empty prefill rows never commit recurrent State; cold metadata avoids
        # even reading a resident. Route every output to one defined-zero
        # negative-slot verify token, not uninitialized empty-prefill storage.
        meta.prefill_conv.fill_(-1)
        meta.initial.zero_()
        meta.prefill.cu.zero_()
        meta.prefill.state.zero_()
        for rows in meta.prefill.indices.values():
            rows[:, 0].fill_(meta.prefill.max_requests)
            rows[:, 1].zero_()
        meta.verify.cu.fill_(1);meta.verify.cu[0]=0
        meta.restore.fill_(meta.capacity)


def attention_metadata(metadata,tokens):
    result=copy.copy(metadata)
    result.actual_seq_lengths_q=[1,tokens] if tokens>1 else [1]
    result.seq_lens_list=[1,0] if tokens>1 else [1]
    result.num_actual_tokens=1
    return result


def install():
    from betterscale.models.qwen35 import device_metadata
    from betterscale.patches.qwen_fia.context_parallel import adapter
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner as Runner
    if getattr(Runner,"_pd_idle_graph_installed",False):return
    original_slots=device_metadata.publish_slots
    def slots(meta):
        if getattr(meta.resident_runner,"_pd_idle_graph",False):
            blank_gdn(meta)
            return  # No normalization, selectors, epochs or previous-role edits.
        return original_slots(meta)
    device_metadata.publish_slots=slots
    original_attention=Runner._build_attention_metadata
    def build(runner,*args,**kwargs):
        result=original_attention(runner,*args,**kwargs)
        if getattr(runner,"_pd_idle_graph",False):
            for meta in result[0].values():
                if hasattr(meta,"slot_mapping"):
                    # Native may cast/copy this mapping before it clears its
                    # source. Clear the actual consumer too, not only the source.
                    meta.slot_mapping.fill_(-1)
                    meta._pd_idle_graph=True
                    if getattr(meta,"seq_lens",None) is not None:
                        meta.seq_lens.zero_();meta.seq_lens[0]=1
        return result
    Runner._build_attention_metadata=build
    original_target=adapter.target_metadata
    def target(metadata,live_requests,tokens):
        if getattr(metadata,"_pd_idle_graph",False):
            if live_requests!=0:raise RuntimeError("Idle graph has live native requests")
            return attention_metadata(metadata,tokens)
        return original_target(metadata,live_requests,tokens)
    adapter.target_metadata=target
    Runner._pd_idle_graph_installed=True


def initialize_null_page(runner):
    from betterscale.live.llm.qwen35.state import AttentionState
    # Pinned BlockPool removes block0 as its null block. Scheduler asserts this
    # invariant independently. Only its first128-token kernel page is read here.
    for leaf in runner._live_state_root.target.values():
        if isinstance(leaf,AttentionState):
            for tensor in leaf.numerical_tensors():tensor[0].zero_()


def target_only(runner,native,*args,**kwargs):
    if not getattr(runner,"_pd_target_only_ready",False):
        return native(runner,*args,**kwargs)
    if (len(args)>1 or kwargs.get("force_attention") or kwargs.get("is_graph_capturing")
            or kwargs.get("is_profile") or runner.input_batch.num_reqs):
        raise RuntimeError("Unqualified idle graph entry")
    # Let the native dispatcher choose the synchronized graph mode and bucket.
    kwargs.pop("cudagraph_runtime_mode",None)
    kwargs["force_attention"]=False
    draft=runner.drafter
    runner.drafter=None;runner._pd_idle_graph=True
    try:return native(runner,*args,**kwargs)
    finally:runner.drafter=draft;runner._pd_idle_graph=False
