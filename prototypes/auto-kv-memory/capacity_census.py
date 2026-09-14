"""Re-run pinned native pool math from a REAL worker's CACHE_SPEC_CENSUS.

CPU only. Report physical allocation and common-pool page demand separately
from vLLM's length-weighted equivalent-token metric. This is sizing evidence,
not a proof that a long-context graph or workload executes successfully.
"""

import argparse
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
from vllm.v1.kv_cache_interface import KVCacheConfig, KVQuantMode
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    group_and_unify_kv_cache_specs,
    _get_kv_cache_groups_uniform_groups,
    _get_kv_cache_config_deepseek_v4,
)


def evaluate(census, budget, wave_budget, lengths):
    constructors = {
        c.__name__: c for c in (AscendMLAAttentionSpec, AscendSlidingWindowMLASpec)
    }
    specs = {}
    for name, serialized in census["specs"].items():
        values = dict(serialized)
        constructor = constructors[values.pop("type")]
        for key in ("dtype", "scale_dtype"):
            if key in values:
                values[key] = getattr(torch, values[key].removeprefix("torch."))
        if "kv_quant_mode" in values:
            values["kv_quant_mode"] = KVQuantMode(values["kv_quant_mode"])
        specs[name] = constructor(**values)
    groups = _get_kv_cache_groups_uniform_groups(group_and_unify_kv_cache_specs(specs))
    cfg = NS(
        cache_config=NS(num_gpu_blocks_override=None),
        model_config=NS(max_model_len=lengths[0]),
        scheduler_config=NS(max_num_batched_tokens=wave_budget),
        parallel_config=NS(
            decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
    )
    blocks, tensors = _get_kv_cache_config_deepseek_v4(cfg, groups, budget)
    physical = sum(t.size for t in tensors)
    output = dict(
        scope=__doc__,
        device=census["device"],
        requested_bytes=budget,
        allocated_pool_bytes=physical,
        unallocated_budget_bytes=budget - physical,
        pool_blocks=blocks,
        physical_bytes_per_pool_block=physical // blocks,
        wave_budget=wave_budget,
        groups=[],
        horizons=[],
    )
    for group in groups:
        kinds = {}
        for spec in group.kv_cache_spec.kv_cache_specs.values():
            value = dict(
                type=type(spec).__name__,
                block=spec.block_size,
                ratio=getattr(spec, "compress_ratio", 1),
                window=getattr(spec, "sliding_window", None),
                head=spec.head_size,
                dtype=str(spec.dtype),
                page_bytes=spec.page_size_bytes,
            )
            key = json.dumps(value, sort_keys=True)
            kinds.setdefault(key, dict(**value, count=0))["count"] += 1
        output["groups"].append(
            dict(
                layers=len(group.layer_names),
                examples=group.layer_names[:2],
                spec_kinds=list(kinds.values()),
            )
        )
    for length in lengths:
        cfg.model_config.max_model_len = length
        config = KVCacheConfig(
            num_blocks=blocks, kv_cache_tensors=tensors, kv_cache_groups=groups
        )
        tokens, concurrency = get_kv_cache_capacity(cfg, config)
        pages = [g.kv_cache_spec.max_memory_usage_pages(cfg) for g in groups]
        output["horizons"].append(
            dict(
                length=length,
                group_peak_pages=pages,
                common_pool_peak_bytes=sum(pages) * (physical // blocks),
                common_pool_peak_page_fit=(blocks - 1) / sum(pages),
                native_equivalent_tokens=tokens,
                native_equivalent_concurrency=concurrency,
            )
        )
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("census", type=Path)
    p.add_argument("--kv-gib", type=float, default=8)
    p.add_argument("--wave-budget", type=int, default=1026)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = evaluate(
        json.loads(a.census.read_text()),
        int(a.kv_gib * (1 << 30)),
        a.wave_budget,
        [16384, 65536, 131072, 262144, 524288, 1048576],
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("pool_blocks", "allocated_pool_bytes", "horizons")},
            indent=2,
        )
    )
