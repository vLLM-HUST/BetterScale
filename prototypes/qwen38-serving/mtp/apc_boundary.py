"""Opt-in scheduler entry for the bounded Qwen MTP APC experiment, not product."""
import json
import os
from pathlib import Path
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.kv_cache_utils import hash_block_tokens, generate_block_hash_extra_keys
from vllm.utils.hashing import sha256
from apc_protocol import extend_hashes


class BoundaryScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.use_eagle and self.cache_config.mamba_cache_mode == 'align'
        assert self.vllm_config.speculative_config.method == 'mtp'
        assert self.vllm_config.model_config.hf_text_config.model_type == 'qwen3_5_text'
        assert self.connector is None
        coord = self.kv_cache_manager.coordinator
        coord.eagle_group_ids.clear()
        assert type(coord).__name__ == 'AscendHybridKVCacheCoordinator'
        assert all(len(g) == 3 for g in coord.attention_groups)
        coord.eagle_attn_group_indices.clear()
        coord.use_eagle = False
        for manager in coord.single_type_managers:
            manager.use_eagle = False
        original_cache = coord.cache_blocks
        def cache(request, computed):
            # Delayed hash publication: no key until its lookahead is known.
            maximum = len(request.block_hashes) * coord.block_pool.hash_block_size
            return original_cache(request, min(computed, maximum))
        coord.cache_blocks = cache
        self._boundary_block_size = coord.block_pool.hash_block_size
        original_get = self.kv_cache_manager.get_computed_blocks
        def get(request):
            result = original_get(request)
            with open(Path(os.environ['CAPSULE'])/'apc-hits.jsonl','a') as out:
                out.write(json.dumps(dict(request=request.request_id,prompt=request.num_prompt_tokens,hits=result[1]))+'\n')
            return result
        self.kv_cache_manager.get_computed_blocks = get

    def add_request(self, request):
        assert not request.resumable
        assert not request.mm_features and request.lora_request is None
        assert request.prompt_token_ids is not None
        def hasher(r):
            return extend_hashes(r.all_token_ids,r.block_hashes,self._boundary_block_size,
                lambda parent,tokens,extra: hash_block_tokens(sha256,parent,tokens,extra),
                lambda start,end: generate_block_hash_extra_keys(r,start,end,0)[0])
        request.block_hashes.clear()
        request._block_hasher = hasher
        request.update_block_hashes()
        return super().add_request(request)

    def _mamba_block_aligned_split(self,*args,**kwargs):
        # Only this synchronous helper loses the old whole-block retreat.
        # Other speculative scheduling/lookahead behavior remains unchanged.
        previous = self.use_eagle
        self.use_eagle = False
        try:
            return super()._mamba_block_aligned_split(*args,**kwargs)
        finally:
            self.use_eagle = previous


def install_draft_boundary():
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
    original = AscendSpecDecodeBaseProposer.set_inputs_first_pass
    def prepare(self,*args,**kwargs):
        assert not args
        cad = kwargs['cad']
        seq = cad._seq_lens_cpu if cad._seq_lens_cpu is not None else cad.seq_lens_cpu
        replacements = []
        for i,req_id in enumerate(self.runner.input_batch.req_ids):
            request = self.runner.requests[req_id]
            end = int(seq[i])
            if end < len(request.prompt_token_ids):
                replacements.append((i,request.get_token_id(end)))
        if replacements:
            ids = kwargs['next_token_ids'].clone()
            for i,token in replacements:
                ids[i] = token
            kwargs['next_token_ids'] = ids
        return original(self,**kwargs)
    AscendSpecDecodeBaseProposer.set_inputs_first_pass = prepare
