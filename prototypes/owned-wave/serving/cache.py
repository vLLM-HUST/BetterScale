"""Native APC allocator with an explicit in-flight read-lease boundary."""

from dataclasses import dataclass


@dataclass
class Lease:
    request: object
    slot: int
    generation: int
    blocks: list
    hit_tokens: int
    references: int = 0
    terminal: bool = False
    freed: bool = False


class PrefixLeases:
    def __init__(self, config, max_length, *, enable_caching=True):
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
        from vllm.utils.hashing import sha256

        assert (
            len(config.kv_cache_groups) == 1
        ), "ordinary Qwen full-attention group only"
        self.block_size = config.kv_cache_groups[0].kv_cache_spec.block_size
        init_none_hash(sha256)
        self.hasher = get_request_block_hasher(self.block_size, sha256)
        self.manager = KVCacheManager(
            config,
            max_length,
            self.block_size,
            self.block_size,
            enable_caching=enable_caching,
            log_stats=True,
        )
        self.max_length = max_length
        self.live = {}
        self.hit_tokens = 0
        self.peak_used_blocks = 0
        self.deferred_releases = 0

    def admit(self, request_id, prompt, output_budget, slot, generation):
        from vllm import SamplingParams
        from vllm.v1.request import Request

        if (
            not prompt
            or output_budget < 1
            or len(prompt) + output_budget > self.max_length
        ):
            raise ValueError("request outside explicit context grant")
        key = (slot, generation)
        if key in self.live:
            raise ValueError("duplicate resident generation")
        request = Request(
            request_id,
            prompt,
            SamplingParams(temperature=0, max_tokens=output_budget, ignore_eos=True),
            None,
            block_hasher=self.hasher,
        )
        computed, hit = self.manager.get_computed_blocks(request)
        result = self.manager.allocate_slots(
            request,
            num_new_tokens=len(prompt) + output_budget - hit,
            num_new_computed_tokens=hit,
            new_computed_blocks=computed,
            delay_cache_blocks=True,
        )
        if result is None:
            return None
        request.num_computed_tokens = hit
        blocks = self.manager.get_block_ids(request_id)[0]
        lease = Lease(request, slot, generation, blocks, hit)
        self.live[key] = lease
        self.hit_tokens += hit
        used = (
            self.manager.block_pool.num_gpu_blocks
            - self.manager.block_pool.get_num_free_blocks()
        )
        self.peak_used_blocks = max(self.peak_used_blocks, used)
        return lease

    def retain(self, lease):
        if lease.freed:
            raise ValueError("retaining released KV")
        lease.references += 1

    def commit(self, lease, *, cursor, token=None, terminal=False):
        if lease.freed or lease.references < 1:
            raise ValueError("receipt has no live read lease")
        if cursor < lease.request.num_computed_tokens:
            raise ValueError("device cursor regressed")
        if token is not None:
            lease.request.append_output_token_ids(token)
        if cursor > lease.request.num_tokens:
            raise ValueError("cursor exceeds known numerical token history")
        lease.request.num_computed_tokens = cursor
        # Only the completed receipt's prefix is published, never an allocated
        # or merely submitted future range. Native hashing/refcounts own reuse.
        self.manager.cache_blocks(lease.request, cursor)
        lease.terminal |= terminal
        if terminal and lease.references > 1:
            self.deferred_releases += 1
        lease.references -= 1
        self.release_if_retired(lease)

    def release_if_retired(self, lease):
        if lease.terminal and lease.references == 0 and not lease.freed:
            self.manager.free(lease.request)
            lease.freed = True
            del self.live[(lease.slot, lease.generation)]
