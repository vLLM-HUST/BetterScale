# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Narrow logical-capacity agreement for one rank-local LiveModule Runtime."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist

from betterscale.live.core.error import LiveModuleError


class StateCapacityCoordinator(Protocol):
    """Agree on one logical block count without observing rank-local layout."""

    def admit_num_blocks(self, local_max_blocks: int, /) -> int:
        """Return the jointly admitted count for one published local maximum.

        Plain elastic domains count logical blocks. Coupled State domains
        count complete capacity units (e.g. residents), never separate pages
        and control rows. All ranks must select the same unit geometry.
        """


@runtime_checkable
class StateFitCoordinator(Protocol):
    """Two fixed agreement phases, independent of local allocation attempts."""

    def admit_fitted_units(self, local_max_units: int, /) -> int:
        """Take the group minimum; zero publishes local failure and aborts all."""

    def confirm_fitted_allocation(self, succeeded: bool, /) -> bool:
        """Confirm final storage on every rank after any capacity reduction."""


class GlooStateCapacityCoordinator:
    """Use an explicitly supplied CPU control group, with no NPU/CUDA buffers.

    Bootstrap owns group membership, timeout and destruction. This coordinator
    never creates an implicit WORLD group or initializes accelerator memory.
    """

    def __init__(self, group: dist.ProcessGroup) -> None:
        if (not isinstance(group, dist.ProcessGroup)
                or dist.get_backend(group) != 'gloo'):
            raise LiveModuleError("invalid-state-capacity-control-group",
                                  "State capacity agreement requires an explicit Gloo group")
        self._group = group

    def _minimum(self, value: int) -> int:
        if type(value) is not int or not 0 <= value < 2**63:
            raise LiveModuleError("invalid-state-capacity-control-value",
                                  "State control values must fit a nonnegative int64")
        control = torch.tensor(value, dtype=torch.int64, device='cpu')
        dist.all_reduce(control, op=dist.ReduceOp.MIN, group=self._group)
        return int(control.item())

    def admit_num_blocks(self, local_max_blocks: int, /) -> int:
        return self._minimum(local_max_blocks)

    def admit_fitted_units(self, local_max_units: int, /) -> int:
        return self._minimum(local_max_units)

    def confirm_fitted_allocation(self, succeeded: bool, /) -> bool:
        if type(succeeded) is not bool:
            raise LiveModuleError("invalid-state-allocation-confirmation",
                                  "State allocation confirmation requires a bool")
        return bool(self._minimum(int(succeeded)))


def admit_state_capacity(
    coordinator: StateCapacityCoordinator | None,
    local_max_blocks: int,
) -> int:
    """Publish one local maximum and validate the returned logical count."""

    if type(local_max_blocks) is not int or local_max_blocks <= 0:
        raise LiveModuleError(
            "invalid-local-state-capacity",
            "local State capacity must be one positive logical block count",
            local_max_blocks=local_max_blocks,
        )
    if coordinator is None:
        return local_max_blocks
    admit = getattr(coordinator, "admit_num_blocks", None)
    if not callable(admit):
        raise LiveModuleError(
            "invalid-state-capacity-coordinator",
            "State capacity coordinator must admit one logical block count",
            coordinator_type=type(coordinator).__qualname__,
        )
    admitted = admit(local_max_blocks)
    if type(admitted) is not int or admitted <= 0:
        raise LiveModuleError(
            "invalid-admitted-state-capacity",
            "admitted State capacity must be one positive logical block count",
            local_max_blocks=local_max_blocks,
            admitted_num_blocks=admitted,
        )
    if admitted > local_max_blocks:
        raise LiveModuleError(
            "admitted-state-capacity-exceeds-local-maximum",
            "a rank cannot realize more logical State blocks than its local maximum",
            local_max_blocks=local_max_blocks,
            admitted_num_blocks=admitted,
        )
    return admitted


class DistributedStateCapacityCoordinator:
    """MIN agreement on an explicitly supplied logical-capacity process group.

    Group creation, membership, timeout and destruction belong to bootstrap.
    All members must call in the same capacity-domain order, before allocation;
    this is a synchronous generation-build operation, never graph/replay work.
    The explicit device must be supported by the group's collective backend.
    Physical State layouts and budgets are not exchanged.
    """

    def __init__(self, *, group: object, device: object) -> None:
        import torch
        import torch.distributed as dist

        if (not dist.is_initialized() or not isinstance(group, dist.ProcessGroup)
                or dist.get_rank(group) < 0):
            raise ValueError('capacity agreement requires an explicit member process group')
        self._group = group
        self._device = torch.device(device)

    def admit_num_blocks(self, local_max_blocks: int, /) -> int:
        import torch
        import torch.distributed as dist

        if (type(local_max_blocks) is not int
                or not 0 < local_max_blocks <= torch.iinfo(torch.int64).max):
            raise ValueError('collective capacity must be a positive int64 block count')
        count = torch.tensor(local_max_blocks, dtype=torch.int64, device=self._device)
        dist.all_reduce(count, op=dist.ReduceOp.MIN, group=self._group)
        return int(count.item())

    def publish_domain_capacities(
        self, domain_keys: tuple[tuple[str, ...], ...],
        local_counts: tuple[int | None, ...],
    ) -> tuple[int, ...]:
        """Publish already-admitted counts to owners AND shadow-only readers.

        None means no physical contribution, not zero capacity. This group may
        differ from the physical admission group. Domain keys are ordered State
        declaration paths, never process-local object identities. Disagreement
        fails rather than silently re-admitting an already allocated domain.
        """
        import hashlib
        import json
        import torch
        import torch.distributed as dist

        maximum = torch.iinfo(torch.int64).max
        valid = (len(domain_keys) == len(local_counts)
                 and all(type(n) is int and 0 < n <= maximum or n is None
                         for n in local_counts))
        digest = hashlib.sha256(json.dumps(domain_keys).encode()).digest()
        # Fixed-size preflight before count vectors: different domain counts or
        # ordering must fail coherently instead of mismatching collective sizes.
        header = torch.tensor([int(valid), *digest], dtype=torch.int64, device=self._device)
        headers = [torch.empty_like(header) for _ in range(dist.get_world_size(self._group))]
        dist.all_gather(headers, header, group=self._group)
        if any(not torch.equal(item, header) for item in headers) or not valid:
            raise ValueError('capacity publication domain layout or input disagreement')
        if not domain_keys:
            return ()
        low = torch.tensor([maximum if n is None else n for n in local_counts],
                           dtype=torch.int64, device=self._device)
        high = torch.tensor([0 if n is None else n for n in local_counts],
                            dtype=torch.int64, device=self._device)
        dist.all_reduce(low, op=dist.ReduceOp.MIN, group=self._group)
        dist.all_reduce(high, op=dist.ReduceOp.MAX, group=self._group)
        if not torch.equal(low, high) or bool((high == 0).any().item()):
            raise ValueError('capacity publication has missing or disagreeing owners')
        return tuple(int(n) for n in high.tolist())


__all__ = (
    "StateCapacityCoordinator", "StateFitCoordinator", "GlooStateCapacityCoordinator",
    "DistributedStateCapacityCoordinator",
    "admit_state_capacity",
)
