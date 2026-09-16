"""Invocation tensor banks and alias-safe KV snapshots for the joint oracle."""
import copy
import dataclasses
import torch
from torch.utils._pytree import tree_flatten


def bank(value, memo=None):
    """Own invocation tensors, retaining repeated-object aliases."""
    memo = {} if memo is None else memo
    key = id(value)
    if key in memo:
        return memo[key]
    if isinstance(value, torch.Tensor):
        result = value.clone()
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = copy.copy(value)
        memo[key] = result
        for field in dataclasses.fields(value):
            setattr(result, field.name, bank(getattr(value, field.name), memo))
    elif isinstance(value, dict):
        result = {k: bank(v, memo) for k, v in value.items()}
    elif isinstance(value, list):
        result = [bank(v, memo) for v in value]
    elif isinstance(value, tuple):
        result = tuple(bank(v, memo) for v in value)
    else:
        return value
    memo[key] = result
    return result


def backing_views(runner):
    leaves, _ = tree_flatten(runner.kv_caches)
    pools = {}
    for tensor in leaves:
        if not isinstance(tensor, torch.Tensor):
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key not in pools:
            pools[key] = torch.as_strided(
                tensor, (storage.nbytes() // tensor.element_size(),), (1,),
                storage_offset=0,
            ).view(torch.uint8)
    return list(pools.values())


def restore(destinations, values):
    for dst, src in zip(destinations, values, strict=True):
        dst.copy_(src)
