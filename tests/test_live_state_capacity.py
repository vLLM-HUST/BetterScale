"""Fixed-first capacity arithmetic, without allocator-cache credit or retries."""

import pytest
import torch

from betterscale.live import (
    ElasticStateCapacity,
    ExactStateCapacity,
    SIMDStateLane,
    StateCapacityUnit,
    StateDomain,
    StateTensor,
    StateTensorError,
    TorchStateBackend,
    compile_simd_state_schema,
)
from betterscale.live.runtime.memory import (
    DeviceMemorySnapshot,
    TorchDeviceMemoryObserver,
)


def schema(requirement, lanes):
    domain = StateDomain(requirement)
    return compile_simd_state_schema(
        tuple(
            SIMDStateLane(
                str(i),
                "state",
                StateTensor(
                    role="test",
                    requirement="test",
                    block_shape=(1,),
                    storage_dtype=torch.float32,
                    domain=domain,
                ),
            )
            for i in range(lanes)
        ),
        domain=domain,
    )


class Backend(TorchStateBackend):
    def __init__(self, coordinator=None):
        super().__init__(
            "cpu", allocation_granularity_bytes=16, capacity_coordinator=coordinator
        )
        self.allocations = []
        self.released = []

    def _allocate_state_domain(self, plan):
        self.allocations.append(plan.num_blocks)
        return super()._allocate_state_domain(plan)

    def _release_state_domain(self, result):
        self.released.append(result.plan.num_blocks)
        super()._release_state_domain(result)


class Observer(TorchDeviceMemoryObserver):
    def __init__(self, backend, final_free=300):
        self._device = torch.device("cpu")
        self.backend, self.final_free = backend, final_free
        self.reads = 0

    def reclaim(self):
        pass

    def snapshot_state_cache_bytes(self):
        raise AssertionError("cache fragments are not capacity")

    def snapshot(self):
        self.reads += 1
        assert self.backend.allocations[0] == 3  # Fixed State exists before sizing.
        free = 1000 if self.reads == 1 else self.final_free
        return DeviceMemorySnapshot(free, 2000, 0, 0, 0, 0, 0, 0)


class Coordinator:
    def __init__(self):
        self.votes = []
        self.confirmations = []

    def admit_fitted_units(self, value):
        self.votes.append(value)
        return min(64, value)

    def confirm_fitted_allocation(self, value):
        self.confirmations.append(value)
        return value


def specs():
    return (
        schema(ExactStateCapacity(3), 1),
        schema(ElasticStateCapacity(StateCapacityUnit(2)), 2),
    )


def test_fixed_first_remaining_bytes_become_shared_units_once():
    backend = Backend()
    result = backend.realize_state_fitting(
        specs(), memory_observer=Observer(backend), minimum_free_bytes=100
    )
    # (1000 free - 100 floor - 2*(16-1) rounding) / 8 bytes per shared unit.
    assert backend.allocations == [3, 108]
    assert backend.capacity_report["shared_budget_bytes"] == 870
    assert [d.plan.num_blocks for d in result.domains] == [3, 108]
    backend.release_state(result)
    assert backend.released == [108, 3]


def test_ranks_agree_before_the_only_shared_allocation():
    coordinator = Coordinator()
    backend = Backend(coordinator)
    result = backend.realize_state_fitting(
        specs(), memory_observer=Observer(backend), minimum_free_bytes=100
    )
    assert coordinator.votes == [108] and coordinator.confirmations == [True]
    assert backend.allocations == [3, 64]
    backend.release_state(result)


def test_physical_mismatch_fails_and_releases_instead_of_searching():
    coordinator = Coordinator()
    backend = Backend(coordinator)
    with pytest.raises(StateTensorError, match="floor"):
        backend.realize_state_fitting(
            specs(), memory_observer=Observer(backend, 90), minimum_free_bytes=100
        )
    assert backend.allocations == [3, 64] and backend.released == [64, 3]
    assert coordinator.confirmations == [False]


def test_insufficient_budget_votes_zero_and_releases_fixed_state():
    coordinator = Coordinator()
    backend = Backend(coordinator)
    with pytest.raises(StateTensorError, match="no shared"):
        backend.realize_state_fitting(
            specs(), memory_observer=Observer(backend), minimum_free_bytes=1100
        )
    assert coordinator.votes == [0] and coordinator.confirmations == []
    assert backend.allocations == [3] and backend.released == [3]
