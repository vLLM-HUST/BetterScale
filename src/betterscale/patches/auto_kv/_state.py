"""Clear the qualified native KV allocations, not their typed page views.

The pinned Ascend compressed-cache allocator owns one raw INT8 allocation per
KV tensor specification; its shared layers expose strided, possibly overlapping
views of that same allocation. Before first admission all bytes, including page
padding, may be cleared. This helper is deliberately not a generic tensor clearer:
its caller must supply only current, KV-exclusive native backing views, after
all warmup writers have completed. It never changes storage or graph addresses.
"""

import torch


def zero_kv_backings(context):
    backings = {}

    def collect(value):
        if isinstance(value, torch.Tensor) and value.numel():
            storage = value.untyped_storage()
            key = (value.device, storage.data_ptr(), storage.nbytes())
            backings.setdefault(key, storage)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    for layer in context.values():
        if hasattr(layer, "kv_cache"):
            collect(layer.kv_cache)
    for (device, _, size), storage in backings.items():
        # set_ only constructs a byte view; it allocates no payload or second KV
        # pool. Use storage identity rather than data_ptr of an offset/dtype view.
        raw = torch.empty(0, dtype=torch.uint8, device=device)
        raw.set_(storage, 0, (size,), (1,))
        raw.zero_()
    return len(backings)
