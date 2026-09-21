"""Bounded integration of MTP leaves; no additional Worker or donor edits."""
from types import SimpleNamespace
from count_policy import MTP_TOKENS, WIDTH, SPEC_CAPACITIES, capture_lengths


def install():
    import torch
    from betterscale.models import qwen
    from betterscale.patches import qwen_gdn, qwen_fia, qwen_mc2
    from betterscale.patches.qwen_layout import pack_conv_weights
    from vllm.config import CUDAGraphMode
    from vllm.config.compilation import CompilationConfig

    import os
    from timeline_probe import install as install_timeline
    install_timeline()
    if os.environ.get('MTP_APC_BOUNDARY') == '1':
        assert MTP_TOKENS == 2 and os.environ.get('NATIVE_BASELINE') != '1'
        from apc_boundary import install_draft_boundary
        install_draft_boundary()
    if MTP_TOKENS == 0 and os.environ.get("NATIVE_BASELINE") != "1":
        return
    if os.environ.get('NATIVE_BASELINE') == '1':
        def native_validate(config):
            if MTP_TOKENS:
                assert config.speculative_config.method == 'mtp'
                assert config.speculative_config.num_speculative_tokens == MTP_TOKENS
            else:
                assert config.speculative_config is None
            assert config.parallel_config.tensor_parallel_size == 2
            assert config.cache_config.mamba_cache_mode == 'align'
            assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
            return 'qwen-mtp'
        def native_before(worker,config):
            from betterscale.patches.qwen_gdn.mamba_abi import install as install_abi
            install_abi()
        qwen.validate = native_validate
        qwen.check = lambda config:qwen.check_runtime()
        qwen.before_init = native_before
        qwen.after_init = lambda worker:None
        qwen.model_loaded = lambda worker:None
        return

    original_adjust = CompilationConfig.adjust_cudagraph_sizes_for_spec_decode

    def adjust(self, *args, **kwargs):
        if self.cudagraph_mode == CUDAGraphMode.FULL:
            # FULL here has one general mixed routine, no uniform-only keys.
            # Rounding prefill capacity2048 to a multiple3 would drop its graph.
            return
        return original_adjust(self, *args, **kwargs)

    CompilationConfig.adjust_cudagraph_sizes_for_spec_decode = adjust

    def validate(config):
        assert config.speculative_config.method == 'mtp'
        assert config.speculative_config.num_speculative_tokens == MTP_TOKENS
        parallel = config.parallel_config
        assert (parallel.tensor_parallel_size, parallel.data_parallel_size,
                parallel.pipeline_parallel_size, parallel.enable_expert_parallel) == (2,1,1,False)
        hf = config.model_config.hf_text_config
        assert (hf.num_hidden_layers,hf.hidden_size,hf.head_dim) == (64,5120,256)
        assert (hf.linear_num_key_heads,hf.linear_num_value_heads,
                hf.linear_key_head_dim,hf.linear_value_head_dim) == (16,48,128,128)
        assert str(config.model_config.dtype) == 'torch.bfloat16'
        assert config.scheduler_config.max_num_seqs == 8
        assert config.scheduler_config.max_num_batched_tokens <= 2048
        assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
        assert config.cache_config.mamba_cache_mode == 'align'
        return 'qwen-owned'

    def check(config):
        qwen.check_runtime(); qwen.check_runtime('qwen_mixed_pins.json')
        qwen_gdn.check_library(); qwen_fia.check_library()

    def forward_core(self, mixed_qkv, b, a, core_attn_out):
        from vllm.forward_context import get_forward_context
        ctx = get_forward_context()
        if ctx.attn_metadata is None:
            return
        meta = ctx.attn_metadata[self.prefix].owned
        t = meta.tokens
        weight = self.conv1d.weight.view(self.conv1d.weight.size(0),4).T
        output = meta(mixed_qkv[:t],a[:t],b[:t],weight,self.A_log,self.dt_bias,*self.kv_cache)
        core_attn_out[:t] = output.squeeze(0)

    def before_init(worker,config):
        from service_metadata import MTPFrame, SPEC_CAPACITIES
        from betterscale.patches.qwen_gdn import publication, graphs
        from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder as Builder
        from vllm_ascend.patch.worker.patch_qwen3_5 import _GDN_PATCH_TARGET

        qwen_gdn.install()
        from draft_banks import install as install_draft_banks
        install_draft_banks()
        from betterscale.patches.qwen_gdn.mamba_abi import install as install_mamba_abi
        install_mamba_abi()
        publication.Frame = MTPFrame
        _GDN_PATCH_TARGET._forward_core = forward_core

        def capacity(runner,num_tokens,num_reqs,scheduled):
            if getattr(runner,'_elastic_dummy',False):
                return num_tokens
            computed = runner.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]
            prompts = runner.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs]
            verify = bool((computed>=prompts).all()) and max(scheduled)<=WIDTH
            choices = SPEC_CAPACITIES if verify else graphs.PREFILLS
            return next((n for n in choices if n>=num_tokens),num_tokens)

        def attention(runner,tokens):
            from vllm_ascend.attention.attention_v1 import AscendAttentionState
            runner.attn_state = AscendAttentionState.ChunkedPrefill

        graphs.capacity = capacity
        graphs.attention = attention

        def build(self,common_prefix_len,common_attn_metadata,num_accepted_tokens=None,
                  num_decode_draft_tokens_cpu=None,**kwargs):
            m = common_attn_metadata
            raw = m.query_start_loc_cpu.diff().tolist()
            lengths = tuple(n for n in raw if n>0)
            assert raw[:len(lengths)] == list(lengths), 'non-packed requests'
            if num_accepted_tokens is None:
                num_accepted_tokens = getattr(self,'_mtp_feedback',None)
            if getattr(self,'_mtp_dummy',False) and self._owned_publication[0].metas[self._owned_publication[1]].decode:
                # Native full dummy may put excess padding in the last row.
                # Give GDN only legal verification rows; FIA keeps the padded
                # query envelope. These are empty-request capture inputs only.
                lengths = capture_lengths(lengths)
            frame,key,table = self._owned_publication
            meta,roles = frame.fill_mtp(key,m,lengths,table,self,num_accepted_tokens,num_decode_draft_tokens_cpu)
            return SimpleNamespace(owned=meta,num_actual_tokens=m.num_input_tokens,
                num_prefills=len(roles)-sum(roles),num_decodes=0,num_spec_decodes=sum(roles),
                spec_sequence_masks=None,spec_state_indices_tensor=meta.verify.slots)

        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner as Runner
        original_attention_build = Runner._build_attention_metadata

        def attention_build(runner,*args,**kwargs):
            builders = [group.get_metadata_builder(0) for groups in runner.attn_groups for group in groups]
            builders = [builder for builder in builders if isinstance(builder,Builder)]
            for builder in builders:
                builder._mtp_dummy = getattr(runner,'_elastic_dummy',False)
                builder._mtp_feedback = (None if getattr(runner,'_elastic_dummy',False)
                                         else runner.num_accepted_tokens.gpu)
            try:
                return original_attention_build(runner,*args,**kwargs)
            finally:
                for builder in builders:
                    del builder._mtp_feedback
                    del builder._mtp_dummy

        Runner._build_attention_metadata = attention_build
        Builder.build = build
        Builder.build_for_cudagraph_capture = lambda self,m:self.build(0,m)

    def model_loaded(worker):
        pack_conv_weights(worker.model_runner.model,consumer=forward_core)
        qwen_mc2.install(worker.model_runner.model)

    qwen.validate = validate
    qwen.check = check
    qwen.before_init = before_init
    qwen.after_init = lambda worker:qwen_fia.install()
    qwen.model_loaded = model_loaded
