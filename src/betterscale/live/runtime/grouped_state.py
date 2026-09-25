# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Cross-layer homogeneous State backings; numerical lanes remain disjoint views.

Grouping is per capacity domain and generation. Neither owner paths nor seats
are new allocation boundaries. Physical ports supply a backing, while the
common planner accounts for its whole padded extent before admission.
"""
from dataclasses import dataclass
from math import prod

import torch

from betterscale.live.runtime.state_backend import StateBackend


def round_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class StateBackingPlan:
    lanes: tuple
    offsets: tuple[int, ...]
    lane_bytes: int
    nbytes: int
    shape: tuple[int, ...]


def plan_state_backings(plan, *, allocation_alignment, lane_alignment=512):
    """Stable first-occurrence groups; never hash model requirement objects."""
    groups = []
    for lane in plan.schema.lanes:
        state = lane.state
        signature = (type(state), state.role, state.requirement, state.storage_dtype,
                     state.physical_shape(plan.num_blocks), state.block_shape,
                     state.physical_blocks_per_logical_block, state.leading_physical_blocks)
        for key, members in groups:
            if signature == key:
                members.append(lane)
                break
        else:
            groups.append((signature, [lane]))
    result = []
    for _, lanes in groups:
        state = lanes[0].state
        shape = state.physical_shape(plan.num_blocks)
        size = prod(shape) * state.storage_dtype.itemsize
        stride = round_up(size, lane_alignment)
        offsets = tuple(i * stride for i in range(len(lanes)))
        result.append(StateBackingPlan(tuple(lanes), offsets, size,
            round_up(offsets[-1] + size, allocation_alignment), shape))
    return tuple(result)


class StateBacking:
    """Exact owned physical extent. Export users pin it, not allocator parents."""
    def __init__(self, tensor, plan, *, exportable=False):
        if (tensor.dtype != torch.uint8 or tensor.ndim != 1
            or not tensor.is_contiguous() or tensor.storage_offset() != 0
            or tensor.numel() != plan.nbytes):
            raise ValueError('State backing provider returned a non-exact byte extent')
        self.tensor, self.plan, self.exportable = tensor, plan, exportable
        self._leases = 0
        self.closed = False

    @property
    def address(self):
        if self.closed:
            raise RuntimeError('State backing is closed')
        return self.tensor.data_ptr()

    def acquire(self):
        if self.closed:
            raise RuntimeError('State backing is closed')
        self._leases += 1
        released = False
        def release():
            nonlocal released
            if not released:
                self._leases -= 1
                released = True
        return release

    def require_releasable(self):
        if self._leases:
            raise RuntimeError('State backing still has IPC/transfer leases')

    def close(self):
        self.require_releasable()
        self.closed = True
        self.tensor = None  # The physical Tensor owner/deleter outlives retained views.


@dataclass(frozen=True)
class StateBackingSpan:
    backing: StateBacking
    offset: int
    nbytes: int


class GroupedStateAllocation(dict):
    def __init__(self):
        super().__init__()
        self.backings = []
        self.spans = {}


class GroupedStateBackend(StateBackend):
    """Reference grouped provider; physical ports override only backing supply.

    CPU tensors deliberately are NOT IPC-exportable. Ascend must supply a
    standalone driver allocation, not mark a caching-allocator tensor as owned.
    """
    def __init__(self, device, *, allocation_alignment=512, lane_alignment=512,
                 memory_budget_bytes=None, capacity_coordinator=None):
        super().__init__(memory_budget_bytes=memory_budget_bytes,
                         capacity_coordinator=capacity_coordinator)
        if any(type(x) is not int or x < 8 or x & (x-1)
               for x in (allocation_alignment, lane_alignment)):
            raise ValueError('State allocation alignments must be powers of two >= 8')
        self.device = torch.device(device)
        self.allocation_alignment = allocation_alignment
        self.lane_alignment = lane_alignment
        self._spans = {}

    def _plans(self, plan):
        return plan_state_backings(plan, allocation_alignment=self.allocation_alignment,
                                   lane_alignment=self.lane_alignment)

    def _allocation_bytes(self, plan):
        return sum(group.nbytes for group in self._plans(plan))

    def _supply_backing(self, plan):
        return StateBacking(torch.zeros(plan.nbytes, dtype=torch.uint8, device=self.device), plan)

    def _allocate_state_domain(self, plan):
        allocation = GroupedStateAllocation()
        try:
            for group in self._plans(plan):
                backing = self._supply_backing(group)
                allocation.backings.append(backing)
                for lane, offset in zip(group.lanes, group.offsets, strict=True):
                    if lane.state in self._spans:
                        raise RuntimeError('State already belongs to a live grouped generation')
                    value = backing.tensor.narrow(0, offset, group.lane_bytes)
                    value = value.view(lane.state.storage_dtype).reshape(group.shape)
                    allocation[lane.state] = value
                    span = StateBackingSpan(backing, offset, group.lane_bytes)
                    allocation.spans[lane.state] = self._spans[lane.state] = span
        except BaseException:
            self._dispose(allocation)
            raise
        return allocation

    def backing_span(self, state):
        try:
            return self._spans[state]
        except KeyError:
            raise ValueError('State has no backing in this backend generation') from None

    def release_state(self, realization):
        # Preflight all domains before partially freeing a pinned generation.
        for domain in realization.domains:
            for backing in domain.allocation.backings:
                backing.require_releasable()
        super().release_state(realization)

    def _dispose(self, allocation):
        for state in allocation.spans:
            self._spans.pop(state, None)
        allocation.clear()
        allocation.spans.clear()
        for backing in reversed(allocation.backings):
            backing.close()
        allocation.backings.clear()

    def _release_state_domain(self, realization):
        self._dispose(realization.allocation)
