"""Contiguous State lanes submitted through the pinned Ascend DMA operator.

This is only a copy submitter: host_state/page_state own stream events, pinned
payload lifetime, address leases and quarantine. No publication occurs here.
"""
from itertools import pairwise

import torch


def descriptors(lanes, payloads, *, to_host):
    """Validate the whole batch before native submission; coalesce adjacent rows."""
    sources, destinations, sizes = [], [], []
    for name, state, blocks in lanes:
        device, host = state.tensor, payloads[name].tensor
        if not device.is_contiguous() or not host.is_contiguous():
            raise ValueError("Batched State DMA requires contiguous physical lanes")
        if device.dtype != host.dtype or device.shape[1:] != host.shape[1:]:
            raise ValueError("Batched State DMA payload geometry mismatch")
        span = state.physical_blocks_per_logical_block
        leading = state.leading_physical_blocks
        if (not blocks or span < 1 or leading < 0
                or any(type(b) is not int or b < 0 for b in blocks)
                or len(set(blocks)) != len(blocks)):
            raise ValueError("Invalid batched State DMA selection")
        if host.shape[0] != len(blocks) * span:
            raise ValueError("Batched State DMA payload length mismatch")
        row_bytes = device.stride(0) * device.element_size()
        first, count, runs = blocks[0], 1, []
        for previous, current in pairwise(blocks):
            if current == previous + 1:
                count += 1
            else:
                runs.append((first, count))
                first, count = current, 1
        runs.append((first, count))
        offset = 0
        for first, count in runs:
            start, rows = leading + first * span, count * span
            if start + rows > device.shape[0]:
                raise ValueError("Batched State DMA exceeds owned lane")
            d, h = device.data_ptr() + start * row_bytes, host.data_ptr() + offset * row_bytes
            sources.append(d if to_host else h)
            destinations.append(h if to_host else d)
            sizes.append(rows * row_bytes)
            offset += rows
    return tuple(torch.tensor(v, dtype=torch.int64) for v in (sources, destinations, sizes))


def enqueue(lanes, payloads, *, to_host):
    if any(s.tensor.device.type != "npu" for _, s, _ in lanes):
        raise ValueError("Ascend State DMA requires NPU lanes")
    if any(not p.tensor.is_pinned() for p in payloads.values()):
        raise ValueError("Ascend State DMA requires pinned host payloads")
    args = descriptors(lanes, payloads, to_host=to_host)
    # The pinned upstream operator consumes CPU descriptors during submission,
    # and queues copies on the current NPU stream. Data buffers outlive its event.
    torch.ops._C_ascend.swap_blocks_batch(*args, 1 if to_host else 0)
