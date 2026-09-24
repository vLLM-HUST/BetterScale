"""Opt-in lifecycle flags for a sealed, source-patched Qwen MTP2 capsule.

Reuse donor reduced greedy selection only for the draft. No global AscendConfig
mutation, target logits change, or replacement collective implementation.
"""
import os
from functools import wraps


def admit_draft(proposer, target, ascend_config, lmhead_tp):
    p = proposer.vllm_config.parallel_config
    hf = proposer.vllm_config.model_config.hf_text_config
    if ((p.tensor_parallel_size, p.data_parallel_size, p.pipeline_parallel_size,
         p.enable_expert_parallel) != (2, 1, 1, False)
            or proposer.method != 'mtp' or proposer.num_speculative_tokens != 2
            or proposer.parallel_drafting or proposer.extra_slots_per_request != 1
            or lmhead_tp or ascend_config.enable_reduce_sample
            or hf.model_type != 'qwen3_5_moe_text' or hf.vocab_size != 248320
            or str(proposer.vllm_config.model_config.dtype) != 'torch.bfloat16'
            or proposer.vllm_config.model_config.quantization is not None):
        raise ValueError('Draft-only greedy requires the admitted BF16 Qwen35MoE TP2/MTP2 route')
    draft = proposer.model
    target = target.get_language_model() if hasattr(target, 'get_language_model') else target
    processor = draft.logits_processor
    if processor is target.logits_processor:
        raise ValueError('Draft-only logits processor must not alias the target processor')
    head = draft.lm_head
    if (head.num_org_embeddings_per_partition != 124160
            or head.num_embeddings_per_partition != 124160):
        raise ValueError('Unqualified padded/added vocabulary partition')
    if getattr(target.logits_processor, '_betterscale_draft_greedy', False):
        raise ValueError('Target logits processor already has a draft-only flag')
    return processor


def install():
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer as Proposer
    from vllm_ascend.patch.worker.patch_qwen3_5 import _GDN_PATCH_TARGET
    if getattr(Proposer, '_betterscale_small_fish_installed', False):
        return
    greedy = os.environ.get('BETTERSCALE_MTP_GREEDY', '0') == '1'
    copies = os.environ.get('BETTERSCALE_GDN_SMALL_COPIES', '0') == '1'
    if _GDN_PATCH_TARGET._forward_core.__module__ != 'service_adapter':
        raise ValueError('Strided gates require the explicitly owned GDN consumer')
    _GDN_PATCH_TARGET._betterscale_strided_gates = copies
    original = Proposer.load_model

    @wraps(original)
    def load(self, target, *args, **kwargs):
        result = original(self, target, *args, **kwargs)
        if greedy:
            from vllm_ascend.ascend_config import get_ascend_config
            from vllm_ascend.utils import lmhead_tp_enable
            processor = admit_draft(self, target, get_ascend_config(), lmhead_tp_enable())
            # Separate objects: shared weight storage does not share processor policy.
            processor._betterscale_draft_greedy = True
            self._betterscale_draft_greedy = True
        return result

    Proposer.load_model = load
    Proposer._betterscale_small_fish_installed = True
