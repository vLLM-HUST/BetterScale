"""Substitute declared State at the native cache-allocation seam only."""

from .execution_capacity import EXECUTION

from copy import deepcopy
from dataclasses import replace

from betterscale.live import LiveRuntime, TorchStateBackend, live_runtime
from betterscale.live.llm.qwen35.state import (
    Capacity,
    Geometry,
    AttentionState,
    GDNState,
)
from betterscale.live.llm.qwen35.root import state_budget_bytes
from .state_binding import BaselineStateRoot


def geometry(config):
    return Geometry.from_config(
        config.model_config.hf_text_config.to_dict(),
        tensor_parallel_size=config.parallel_config.tensor_parallel_size,
    )


def resident_seats(config):
    """One shared scheduler/allocation contract."""
    value = config.additional_config.get("state_resident_seats", max(20, EXECUTION + 4))
    if type(value) is not int or not max(20, EXECUTION) <= value <= 96:
        raise ValueError("resident State seats must cover execution and fit20..96")
    return value


def fixed_state_bytes(config):
    g = geometry(config)
    c = Capacity(EXECUTION, resident_seats(config), token_pages=1, prefill_tokens=4096)
    attention_layers = g.layer_types.count("full_attention") + g.draft_layers
    page_bytes = (
        attention_layers * c.page_tokens * g.kv_heads * g.attention_head_dim * 4
    )
    # Baseline MTP2 candidate set plus convolution selector and write budget.
    # Neither resident State nor shared FA capacity scales with async queue depth.
    return state_budget_bytes(g, c, 1) - page_bytes + c.resident_seats * 8


def determine_available_memory(worker, native):
    available = native()
    fixed = fixed_state_bytes(worker.vllm_config)
    if available <= fixed:
        raise ValueError("KV budget cannot hold the fixed resident State domain")
    worker._live_state_budget = {
        "total_bytes": available,
        "resident_bytes": fixed,
        "attention_bytes": available - fixed,
    }
    return available - fixed


def install():
    from vllm.v1.kv_cache_interface import MambaSpec, KVCacheGroupSpec
    from vllm.v1.worker.utils import bind_kv_cache
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner as Runner

    original_specs = Runner.get_kv_cache_spec
    original_init = Runner.initialize_kv_cache

    def specs(runner):
        native = original_specs(runner)
        runner._live_state_specs = native
        runner._live_gdn_specs = {
            name: spec for name, spec in native.items() if isinstance(spec, MambaSpec)
        }
        if len(runner._live_gdn_specs) != 30 or len(native) != 41:
            raise ValueError("live State census requires 30 target GDN and11 FA leaves")
        # Retain the baseline scheduler/kernel page granularity for this cut.
        # Remove only heterogeneous padding and GDN's claim on the token pool.
        return {
            name: replace(spec, page_size_padded=None)
            for name, spec in native.items()
            if not isinstance(spec, MambaSpec)
        }

    def initialize(runner, config):
        if len(config.kv_cache_groups) != 1 or config.has_mamba_layers:
            raise ValueError("native scheduler must expose one shared FA page pool")
        config = deepcopy(config)
        gdn = runner._live_gdn_specs
        prototype = next(iter(gdn.values()))
        if any(spec != prototype for spec in gdn.values()):
            raise ValueError(
                "resident GDN metadata requires uniform per-layer geometry"
            )
        # Worker-only metadata group: this is not an allocator group and receives
        # no KVCacheTensor. It has resident row descriptors, not pooled GDN pages.
        config.kv_cache_groups.append(KVCacheGroupSpec(list(gdn), prototype))
        runner._live_gdn_group = len(config.kv_cache_groups) - 1
        return original_init(runner, config)

    def allocate(runner, config):
        if runner.kv_caches:
            raise RuntimeError(
                "native KV storage already bound before State activation"
            )
        logical_block = config.kv_cache_groups[0].kv_cache_spec.block_size
        if logical_block % 128:
            raise ValueError(
                "FA scheduler page must consist of whole128-token kernel pages"
            )
        capacity = Capacity(
            EXECUTION,
            resident_seats(runner.vllm_config),
            token_pages=config.num_blocks * (logical_block // 128),
            prefill_tokens=4096,
        )
        context = runner.compilation_config.static_forward_context
        consumers = {name: context[name] for name in runner._live_state_specs}
        for name, consumer in consumers.items():
            # The installed State publisher supplies distinct conv/resident and
            # recurrent/candidate indexes to the unchanged baseline consumer.
            consumer.state_binding_abi = (
                GDNState.binding_abi
                if name in runner._live_gdn_specs
                else AttentionState.binding_abi
            )
        with live_runtime(
            LiveRuntime(
                device=runner.device, state_backend=TorchStateBackend(runner.device)
            )
        ):
            root = BaselineStateRoot(
                geometry(runner.vllm_config),
                capacity,
                consumers=consumers,
                draft_names=runner.drafter.attn_layer_names,
            )
        root.activate()
        caches = root.kv_caches()
        bind_kv_cache(caches, context, runner.kv_caches)
        runner._live_state_root = root
        runner._live_resident_epochs = [0] * capacity.resident_seats
        for group in runner.attn_groups[runner._live_gdn_group]:
            group.get_metadata_builder(0)._live_state_runner = runner
        from .state_address import initialize

        initialize(runner)
        return caches

    Runner.get_kv_cache_spec = specs
    Runner.initialize_kv_cache = initialize
    Runner.initialize_kv_cache_tensors = allocate
