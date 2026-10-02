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
    with pytest.raises(RuntimeError):blank_gdn(NS(decode=False))


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
