"""Qwen35 baseline hooks; model execution remains in the original runner."""

from types import SimpleNamespace
from betterscale.patches import qwen_gdn, qwen_fia
from betterscale.patches.qwen_layout import pack_conv_weights
from .count_policy import WIDTH, capture_lengths

PREFILLS = (16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 4096)


def forward_core(self, mixed_qkv, b, a, core_attn_out):
    from vllm.forward_context import get_forward_context

    ctx = get_forward_context()
    if ctx.attn_metadata is None:
        return
    meta = ctx.attn_metadata[self.prefix].owned
    t = meta.tokens
    weight = self.conv1d.weight.view(self.conv1d.weight.size(0), 4).T
    output = meta(
        mixed_qkv[:t], a[:t], b[:t], weight, self.A_log, self.dt_bias, *self.kv_cache
    )
    core_attn_out[:t] = output.squeeze(0)


def before_init(worker, config):
    from .moe_overlap import configure as configure_moe_overlap

    configure_moe_overlap(config)
    from .service_metadata import MTPFrame, SPEC_CAPACITIES
    from betterscale.patches.qwen_gdn import publication, graphs
    from vllm_ascend.ops.gdn_attn_builder import (
        AscendGDNAttentionMetadataBuilder as Builder,
    )
    from vllm_ascend.patch.worker.patch_qwen3_5 import _GDN_PATCH_TARGET
    from .apc_boundary import install_draft_boundary

    install_draft_boundary()
    qwen_gdn.install()
    graphs.PREFILLS = PREFILLS
    import dataclasses

    graphs.descriptor = lambda result: (
        dataclasses.replace(result, num_reqs=16)
        if result.num_tokens in PREFILLS
        else result
    )
    from .device_apc import install as install_device_apc

    install_device_apc()
    from .draft_banks import install as install_draft_banks

    install_draft_banks()
    from .mamba_abi import install as install_mamba_abi

    install_mamba_abi()
    publication.Frame = MTPFrame
    _GDN_PATCH_TARGET._forward_core = forward_core
    from .small_fish_runtime import install as install_small_fish

    install_small_fish()

    def capacity(runner, num_tokens, num_reqs, scheduled):
        if getattr(runner, "_elastic_dummy", False):
            return num_tokens
        computed = runner.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]
        prompts = runner.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs]
        verify = bool((computed >= prompts).all()) and max(scheduled) <= WIDTH
        choices = SPEC_CAPACITIES if verify else graphs.PREFILLS
        return next((n for n in choices if n >= num_tokens), num_tokens)

    def attention(runner, tokens):
        from vllm_ascend.attention.attention_v1 import AscendAttentionState

        runner.attn_state = AscendAttentionState.ChunkedPrefill

    graphs.capacity = capacity
    graphs.attention = attention

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        **kwargs,
    ):
        m = common_attn_metadata
        raw = m.query_start_loc_cpu.diff().tolist()
        lengths = tuple((n for n in raw if n > 0))
        assert raw[: len(lengths)] == list(lengths), "non-packed requests"
        if num_accepted_tokens is None:
            num_accepted_tokens = getattr(self, "_mtp_feedback", None)
        if (
            getattr(self, "_mtp_dummy", False)
            and self._owned_publication[0].metas[self._owned_publication[1]].decode
        ):
            lengths = capture_lengths(lengths)
        frame, key, table = self._owned_publication
        meta, roles = frame.fill_mtp(
            key,
            m,
            lengths,
            table,
            self,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
        )
        return SimpleNamespace(
            owned=meta,
            num_actual_tokens=m.num_input_tokens,
            num_prefills=len(roles) - sum(roles),
            num_decodes=0,
            num_spec_decodes=sum(roles),
            spec_sequence_masks=None,
            spec_state_indices_tensor=meta.verify.slots,
        )

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner as Runner

    original_attention_build = Runner._build_attention_metadata

    def attention_build(runner, *args, **kwargs):
        builders = [
            group.get_metadata_builder(0)
            for groups in runner.attn_groups
            for group in groups
        ]
        builders = [builder for builder in builders if isinstance(builder, Builder)]
        for builder in builders:
            builder._mtp_dummy = getattr(runner, "_elastic_dummy", False)
            builder._mtp_feedback = (
                None
                if getattr(runner, "_elastic_dummy", False)
                else runner.num_accepted_tokens.gpu
            )
        try:
            return original_attention_build(runner, *args, **kwargs)
        finally:
            for builder in builders:
                del builder._mtp_feedback
                del builder._mtp_dummy

    Runner._build_attention_metadata = attention_build
    Builder.build = build
    Builder.build_for_cudagraph_capture = lambda self, m: self.build(0, m)


def model_loaded(worker):
    pack_conv_weights(
        worker.model_runner.model, consumer=forward_core, expected_layers=30
    )


def after_init(worker):
    qwen_fia.install(heads=8, kvheads=1, requests=17, tokens=4096)
    from .draft_fia import install as install_draft_fia
    from .device_metadata import install as install_device_metadata

    install_draft_fia()
    install_device_metadata()
