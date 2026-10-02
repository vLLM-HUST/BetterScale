"""Donor adapter ordering; hardware overlap needs the separate NPU gate."""
from types import SimpleNamespace
import pytest


def test_only_dispatch_fence_moves_before_native_prepare(monkeypatch):
    import torch
    import torch_npu
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    import vllm_ascend.ascend_forward_context as context
    import vllm_ascend.utils as utils
    monkeypatch.setattr(utils, "shared_expert_dp_enabled", lambda: False)
    from vllm_ascend.ascend_forward_context import MoECommType
    from vllm_ascend.quantization.quant_type import QuantType
    from betterscale.models.qwen35.moe_overlap import install

    order = []
    early, late, combine, output = object(), object(), object(), object()
    def record():
        order.append("local_input_ready")
        return early
    def original(self, hidden, logits, return_with_event=False):
        order.extend(("prepare_gather", "routing", "expert", "combine", "finalize"))
        if not return_with_event:
            return output
        return SimpleNamespace(routed_out=output, before_dispatch_evt=late,
                               before_combine_evt=combine)
    monkeypatch.setattr(AscendMoERunner, "no_shared_forward_impl", original)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: SimpleNamespace(record_event=record))
    extra = SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)
    monkeypatch.setattr(context, "_EXTRA_CTX", extra)
    obj = SimpleNamespace(_shared_experts=object(), multistream_overlap_shared_expert=True,
                          quant_type=QuantType.NONE, moe_config=SimpleNamespace(sp_size=1, pcp_size=1))
    x = torch.empty(3, 4, dtype=torch.bfloat16)
    install()
    method = AscendMoERunner.no_shared_forward_impl
    install()
    assert AscendMoERunner.no_shared_forward_impl is method
    result = method(obj, x, x, True)
    assert order[0:2] == ["local_input_ready", "prepare_gather"]
    assert result.before_dispatch_evt is early
    assert result.before_combine_evt is combine and result.routed_out is output
    order.clear()
    assert method(obj, x, x, False) is output
    assert "local_input_ready" not in order
    obj.multistream_overlap_shared_expert = False
    with pytest.raises(ValueError, match="multistream"):
        method(obj, x, x, True)
    obj.multistream_overlap_shared_expert = True
    with pytest.raises(ValueError, match="BF16"):
        method(obj, x.float(), x, True)
    for field in ("sp_size", "pcp_size"):
        setattr(obj.moe_config, field, 2)
        with pytest.raises(ValueError, match="excludes"):
            method(obj, x, x, True)
        setattr(obj.moe_config, field, 1)
    extra.moe_comm_type = MoECommType.MC2
    with pytest.raises(ValueError, match="AllGather"):
        method(obj, x, x, True)


def test_config_opt_in_requires_donor_stream(monkeypatch):
    from betterscale.models.qwen35 import moe_overlap
    calls = []
    monkeypatch.setattr(moe_overlap, "install", lambda: calls.append(True))
    moe_overlap.configure(SimpleNamespace(additional_config={}))
    assert not calls
    extra = {"betterscale_shared_expert_overlap": True}
    with pytest.raises(ValueError, match="multistream"):
        moe_overlap.configure(SimpleNamespace(additional_config=extra))
    extra["multistream_overlap_shared_expert"] = True
    moe_overlap.configure(SimpleNamespace(additional_config=extra))
    assert calls == [True]
