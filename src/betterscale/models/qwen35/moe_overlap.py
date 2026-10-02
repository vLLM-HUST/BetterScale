"""Opt-in BF16 AllGather/shared-expert overlap for the pinned Ascend runner.

The donor's before_dispatch event is after DP input gathering. Shared gate/up
needs only the local input, so it may start before that gather. Keep the donor's
before_combine fence for the second shared stage and its final stream join and
TP reduction; no arithmetic, routing, or collective order is replaced here.
"""
from functools import wraps


def configure(config):
    """Explicit opt-in; enabling the donor stream alone does not install a patch."""
    extra = config.additional_config
    if not extra.get("betterscale_shared_expert_overlap", False):
        return
    if not extra.get("multistream_overlap_shared_expert", False):
        raise ValueError("shared overlap requires donor multistream flag")
    install()


def install():
    import torch
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
    from vllm_ascend.quantization.quant_type import QuantType
    from vllm_ascend.utils import shared_expert_dp_enabled

    original = AscendMoERunner.no_shared_forward_impl
    if getattr(original, "_betterscale_shared_overlap", False):
        return

    @wraps(original)
    def forward(self, hidden_states, router_logits, return_with_event=False):
        early = None
        if return_with_event and self._shared_experts is not None:
            if not self.multistream_overlap_shared_expert:
                raise ValueError("shared overlap requires donor multistream flag")
            if (self.moe_config.sp_size != 1 or self.moe_config.pcp_size != 1
                    or shared_expert_dp_enabled()):
                raise ValueError("shared overlap excludes SP, PCP and shared-expert DP")
            if self.quant_type != QuantType.NONE:
                raise ValueError("shared overlap supports only unquantized BF16")
            if hidden_states.dtype != torch.bfloat16:
                raise ValueError("shared overlap requires BF16 input")
            if _EXTRA_CTX.moe_comm_type != MoECommType.ALLGATHER:
                raise ValueError("shared overlap requires AllGather communication")
            early = torch.npu.current_stream().record_event()
        result = original(self, hidden_states, router_logits, return_with_event)
        if early is not None:
            result.before_dispatch_evt = early
        return result

    forward._betterscale_shared_overlap = True
    AscendMoERunner.no_shared_forward_impl = forward
