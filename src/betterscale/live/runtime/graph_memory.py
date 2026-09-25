# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Native graph-pool attribution, separate from device-wide waterlines."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from betterscale.live.core.error import LiveModuleError


@dataclass(frozen=True, slots=True)
class GraphPoolMemorySnapshot:
    """Observed residency of one native private pool on one device.

    Inactive storage remains graph-owned. Only inactive blocks on the selected
    capture stream are reuse candidates; their sum is not an allocation-fit
    guarantee (size classes, alignment and fragmentation still apply).
    """

    device_index: int
    pool_id: tuple[int, int]
    stream_id: int
    reserved_bytes: int
    active_bytes: int
    inactive_bytes: int
    reusable_bytes: int
    largest_reusable_block_bytes: int

    def __post_init__(self) -> None:
        values = (self.device_index, self.stream_id, self.reserved_bytes,
                  self.active_bytes, self.inactive_bytes, self.reusable_bytes,
                  self.largest_reusable_block_bytes, *self.pool_id)
        if (len(self.pool_id) != 2 or self.pool_id == (0, 0)
                or any(type(value) is not int or value < 0 for value in values)
                or self.active_bytes + self.inactive_bytes != self.reserved_bytes
                or not 0 <= self.largest_reusable_block_bytes <= self.reusable_bytes <= self.inactive_bytes):
            raise LiveModuleError("invalid-graph-pool-memory-snapshot",
                                  "graph pool byte geometry or identity is invalid")


def graph_pool_snapshot(
    segments: Sequence[Mapping[str, object]],
    *,
    device_index: int,
    pool_id: tuple[int, int],
    stream_id: int,
) -> GraphPoolMemorySnapshot | None:
    """Read torch allocator segments without crediting foreign or live blocks.

    Unknown snapshot layouts return no evidence, never guessed capacity. The
    default pool is not a serial graph lifetime contract and cannot be credited.
    """
    if pool_id == (0, 0):
        return None
    reserved = active = inactive = reusable = largest = 0
    try:
        for segment in segments:
            if segment['device'] != device_index:
                continue
            if tuple(segment['segment_pool_id']) != pool_id:
                continue
            size = segment['total_size']
            if type(size) is not int or size < 0:
                return None
            same_stream = segment['stream'] == stream_id
            block_total = 0
            for block in segment['blocks']:
                block_size = block['size']
                if type(block_size) is not int or block_size < 0:
                    return None
                block_total += block_size
                state = block['state']
                if state == 'inactive':
                    inactive += block_size
                    if same_stream:
                        reusable += block_size
                        largest = max(largest, block_size)
                elif state in ('active_allocated', 'active_pending_free', 'active_awaiting_free'):
                    active += block_size
                else:
                    return None
            if block_total != size:
                return None
            reserved += size
    except (KeyError, TypeError, ValueError):
        return None
    return GraphPoolMemorySnapshot(device_index, pool_id, stream_id, reserved,
                                   active, inactive, reusable, largest)


def state_allocation_cache_bytes(
    segments: Sequence[Mapping[str, object]], *, device_index: int, stream_id: int,
) -> int | None:
    """Upper-bound inactive default-pool storage available to eager State supply.

    Private capture pools and other streams cannot supply these allocations.
    This is a search ceiling only: fragmentation/alignment still require the
    fitter to allocate and verify the actual driver-free floor.
    """
    available = 0
    try:
        for segment in segments:
            if segment['device'] != device_index:
                continue
            if (tuple(segment['segment_pool_id']) != (0, 0)
                    or segment['stream'] != stream_id):
                continue
            total = 0
            for block in segment['blocks']:
                size = block['size']
                if type(size) is not int or size < 0:
                    return None
                total += size
                if block['state'] == 'inactive':
                    available += size
                elif block['state'] not in ('active_allocated', 'active_pending_free',
                                             'active_awaiting_free'):
                    return None
            if total != segment['total_size']:
                return None
    except (KeyError, TypeError, ValueError):
        return None
    return available
