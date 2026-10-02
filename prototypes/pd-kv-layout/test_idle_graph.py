from types import SimpleNamespace as NS
import pytest
import torch
from idle_graph import blank_gdn,attention_metadata,target_only


def test_idle_masks_metadata_without_any_resident_access():
    meta=NS(decode=True,verify_conv=torch.zeros(16,1,dtype=torch.int32),
            verify=NS(slots=torch.zeros(16,3,dtype=torch.int32),accepted=torch.zeros(16,dtype=torch.int32)))
    blank_gdn(meta)
    assert torch.all(meta.verify_conv==-1) and torch.all(meta.verify.slots==-1)
    assert torch.all(meta.verify.accepted==1)
    meta.decode=False;meta.capacity=1024
    meta.prefill_conv=torch.zeros(17,1,dtype=torch.int32)
    meta.initial=torch.ones(17,dtype=torch.bool)
    meta.prefill=NS(cu=torch.ones(18,dtype=torch.int64),
        state=torch.ones(17,2,dtype=torch.int64),max_requests=16,
        indices={64:torch.ones(31,2,dtype=torch.int64)})
    meta.verify.cu=torch.zeros(18,dtype=torch.int32)
    meta.restore=torch.zeros(1024,dtype=torch.int64)
    blank_gdn(meta)
    assert not meta.prefill.cu.any() and not meta.prefill.state.any()
    assert torch.all(meta.prefill_conv==-1)
    assert meta.verify.cu.tolist()==[0]+[1]*17
    assert torch.all(meta.restore==1024)


def test_idle_attention_uses_one_null_page_reader_without_mutating_original():
    original=NS(actual_seq_lengths_q=[3,6],seq_lens_list=[2048,4096],num_actual_tokens=6)
    result=attention_metadata(original,24)
    assert result.actual_seq_lengths_q==[1,24] and result.seq_lens_list==[1,0]
    assert original.seq_lens_list==[2048,4096]


def test_idle_preserves_native_graph_dispatch_and_restores_on_error():
    draft=object();r=NS(_pd_target_only_ready=True,drafter=draft,input_batch=NS(num_reqs=0))
    def native(r,*a,**kw):
        assert r._pd_idle_graph and r.drafter is None
        assert 'cudagraph_runtime_mode' not in kw and not kw['force_attention']
        raise ValueError("sentinel")
    with pytest.raises(ValueError,match="sentinel"):target_only(r,native,1)
    assert r.drafter is draft and not r._pd_idle_graph
    r.input_batch.num_reqs=1
    with pytest.raises(RuntimeError):target_only(r,native,1)


def test_startup_capture_is_unchanged():
    r=NS(drafter=object())
    assert target_only(r,lambda r,*a,**kw:kw,1,is_graph_capturing=True)==dict(is_graph_capturing=True)


def test_idle_mtp_keeps_drafter_for_peer_collectives():
    draft=object()
    r=NS(_pd_target_only_ready=True,_pd_mtp_enabled=True,drafter=draft,input_batch=NS(num_reqs=0))
    def native(r,*a,**kw):
        assert r._pd_idle_graph and r.drafter is draft
        return "participated"
    assert target_only(r,native,3)=="participated"
    assert r.drafter is draft and not r._pd_idle_graph


def test_idle_draft_metadata_masks_writes_without_changing_device_feedback():
    from betterscale.models.qwen35.draft_fia import idle_metadata
    original=NS(slot_mapping=torch.arange(16),actual_seq_lengths_q=[3,6],
        seq_lens_list=[2048,4096],_mtp_device_seq_lens=torch.tensor([2048,4096]))
    result=idle_metadata(original,16)
    assert result.actual_seq_lengths_q==[1,16] and result.seq_lens_list==[1,0]
    assert result._mtp_device_seq_lens.tolist()==[1,0]
    assert original._mtp_device_seq_lens.tolist()==[2048,4096]
    assert (original.slot_mapping==-1).all()
