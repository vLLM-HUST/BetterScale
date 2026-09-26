# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""State resource planning, admission and physical realization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from betterscale.live.core.error import LiveModuleError
from betterscale.live.core.state_tensor import (
    ElasticStateCapacity,
    ExactStateCapacity,
    ScaledStateCapacity,
    SIMDStatePlan,
    SIMDStateSchema,
    StateTensor,
    StateTensorError,
    admit_simd_state_plan,
    compile_exact_simd_state_plan,
    compile_simd_state_plan,
    validate_state_domain_allocations,
)
from betterscale.live.runtime.memory import TorchDeviceMemoryObserver
from betterscale.live.runtime.capacity import (
    StateCapacityCoordinator,
    StateFitCoordinator,
    admit_state_capacity,
)


@dataclass(frozen=True, slots=True)
class _PlannedStateDomain:
    plan: SIMDStatePlan
    local_max_capacity: int


@dataclass(frozen=True, slots=True)
class StateDomainRealization:
    """One backend-supplied domain plan and its physical allocation."""

    plan: SIMDStatePlan
    local_max_capacity: int
    allocation: object
    allocated_state_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class StateGenerationRealization:
    """One complete backend-supplied State generation before module binding."""

    domains: tuple[StateDomainRealization, ...]


class StateBackend:
    """Own State budget, capacity admission, planning and physical supply."""

    __slots__ = ("_capacity_coordinator", "_memory_budget_bytes")

    def __init__(
        self,
        *,
        memory_budget_bytes: int | None = None,
        capacity_coordinator: StateCapacityCoordinator | None = None,
    ) -> None:
        if memory_budget_bytes is not None and (
            type(memory_budget_bytes) is not int or memory_budget_bytes <= 0
        ):
            raise LiveModuleError(
                "invalid-state-memory-budget",
                "StateBackend memory budget must be one positive integer byte count",
                memory_budget_bytes=memory_budget_bytes,
            )
        self._memory_budget_bytes = memory_budget_bytes
        self._capacity_coordinator = capacity_coordinator

    def realize_state(
        self,
        schemas: Sequence[SIMDStateSchema],
        *,
        memory_budget_bytes: int | None = None,
    ) -> StateGenerationRealization:
        """Plan and allocate one generation within its logical State budget.

        A generation may supply a smaller budget without mutating the backend's
        configured ceiling. This permits disposable calibration and final State
        to use the same physical provider. These are logical tensor bytes, not a
        prediction of allocator padding, segment residency or device free space.
        """

        plans = self._plan_state(
            schemas, memory_budget_bytes=memory_budget_bytes,
            capacity_coordinator=self._capacity_coordinator,
        )
        return self._realize_state_plans(plans)

    def _plan_state(
        self,
        schemas: Sequence[SIMDStateSchema],
        *,
        memory_budget_bytes: int | None,
        capacity_coordinator: StateCapacityCoordinator | None,
    ) -> tuple[_PlannedStateDomain, ...]:
        """Compile domains before physical supply; None means local admission.

        Physical fit searches must remain rank-local until their local maximum
        is established. They must not mutate the provider's coordinator or make
        a different number of collective admission calls on different ranks.
        """

        if memory_budget_bytes is not None and (
            type(memory_budget_bytes) is not int or memory_budget_bytes <= 0
        ):
            raise StateTensorError(
                "invalid-state-memory-budget",
                "a generation State budget must be a positive integer byte count",
                memory_budget_bytes=memory_budget_bytes,
            )

        ordered = tuple(schemas)
        if not ordered or any(
            not isinstance(schema, SIMDStateSchema) for schema in ordered
        ):
            raise StateTensorError(
                "invalid-state-domain-request",
                "StateBackend requires one nonempty sequence of typed schemas",
            )

        elastic = tuple(
            schema
            for schema in ordered
            if isinstance(schema.capacity_requirement, ElasticStateCapacity)
        )
        if len(elastic) > 1:
            raise LiveModuleError(
                "ambiguous-elastic-state-domains",
                "one State budget cannot be divided between multiple elastic "
                "domains without an explicit backend partition policy",
                elastic_domain_count=len(elastic),
            )
        unsupported = tuple(
            type(schema.capacity_requirement).__qualname__
            for schema in ordered
            if not isinstance(
                schema.capacity_requirement,
                (ElasticStateCapacity, ExactStateCapacity, ScaledStateCapacity),
            )
        )
        if unsupported:
            raise LiveModuleError(
                "unsupported-state-capacity-requirement",
                "StateBackend received an unknown capacity constraint",
                requirement_types=unsupported,
            )

        scaled = tuple(schema for schema in ordered
                       if isinstance(schema.capacity_requirement, ScaledStateCapacity))
        if scaled and (not elastic or any(
            schema.capacity_requirement.unit is not elastic[0].capacity_requirement.unit
            for schema in scaled
        )):
            raise StateTensorError("unbound-scaled-state-capacity",
                                   "scaled State domains must share the primary elastic unit")

        exact_plans = tuple(
            compile_exact_simd_state_plan(
                schema,
                num_blocks=schema.capacity_requirement.capacity,
            )
            for schema in ordered
            if isinstance(schema.capacity_requirement, ExactStateCapacity)
        )
        exact_state_bytes = sum(self._allocation_bytes(plan) for plan in exact_plans)
        budget = self._memory_budget_bytes
        if memory_budget_bytes is not None:
            budget = (memory_budget_bytes if budget is None
                      else min(budget, memory_budget_bytes))
        if budget is not None and exact_state_bytes > budget:
            raise StateTensorError(
                "insufficient-state-memory",
                "exact State domains exceed the backend State budget",
                memory_budget_bytes=budget,
                exact_state_bytes=exact_state_bytes,
            )

        elastic_plan: SIMDStatePlan | None = None
        elastic_local_max: int | None = None
        if elastic:
            if budget is None:
                raise StateTensorError(
                    "missing-state-memory-budget",
                    "an elastic State domain requires a backend-owned memory budget",
                )
            requirement = elastic[0].capacity_requirement
            if requirement.unit is None:
                local_plan = compile_simd_state_plan(
                    elastic[0],
                    usable_state_bytes=budget - exact_state_bytes,
                )
                # Physical providers may round/group allocations. Fit their actual
                # extent before cross-rank capacity admission, not after allocation.
                available = budget - exact_state_bytes
                if self._allocation_bytes(local_plan) > available:
                    low, high = 0, local_plan.num_blocks
                    while low < high:
                        mid = (low + high + 1) // 2
                        candidate = admit_simd_state_plan(local_plan, admitted_num_blocks=mid)
                        if self._allocation_bytes(candidate) <= available:
                            low = mid
                        else:
                            high = mid - 1
                    if low == 0:
                        raise StateTensorError("insufficient-state-memory",
                            "State allocation alignment exceeds the available budget")
                    local_plan = admit_simd_state_plan(local_plan, admitted_num_blocks=low)
                elastic_local_max = local_plan.num_blocks
                elastic_plan = admit_simd_state_plan(
                    local_plan,
                    admitted_num_blocks=admit_state_capacity(
                        capacity_coordinator,
                        elastic_local_max,
                    ),
                )
            else:
                family = (*elastic, *scaled)
                fixed = sum(schema.fixed_state_bytes for schema in family)
                per_unit = sum(schema.bytes_per_simd_block
                               * schema.capacity_requirement.blocks_per_unit
                               for schema in family)
                if per_unit <= 0:
                    raise StateTensorError("invalid-state-capacity-unit-cost",
                                           "a complete State capacity unit must have positive storage cost")
                local_units = (budget - exact_state_bytes - fixed) // per_unit
                if local_units < requirement.unit.minimum_units:
                    raise StateTensorError("insufficient-state-memory",
                                           "State budget cannot cover the minimum complete capacity units",
                                           minimum_units=requirement.unit.minimum_units,
                                           local_max_units=max(0, local_units))
                admitted_units = admit_state_capacity(capacity_coordinator, local_units)
                if admitted_units < requirement.unit.minimum_units:
                    raise StateTensorError("insufficient-admitted-state-capacity",
                                           "coordinated capacity is below the declared minimum units")
                elastic_local_max = local_units * requirement.blocks_per_unit
                committed_related = sum(
                    schema.fixed_state_bytes + admitted_units
                    * schema.capacity_requirement.blocks_per_unit * schema.bytes_per_simd_block
                    for schema in scaled
                )
                elastic_plan = compile_exact_simd_state_plan(
                    elastic[0], num_blocks=admitted_units * requirement.blocks_per_unit,
                )
                # The primary receipt owns the budget residual; related domains
                # retain exact receipts. No family member is charged twice.
                remaining = budget - exact_state_bytes - committed_related
                elastic_plan = SIMDStatePlan(
                    schema=elastic[0], usable_state_bytes=remaining,
                    num_blocks=elastic_plan.num_blocks,
                    committed_state_bytes=elastic_plan.committed_state_bytes,
                    residual_state_bytes=remaining - elastic_plan.committed_state_bytes,
                )

        plans_by_schema_id = {id(plan.schema): plan for plan in exact_plans}
        local_max_by_schema_id = {
            id(plan.schema): plan.num_blocks for plan in exact_plans
        }
        if elastic_plan is not None:
            assert elastic_local_max is not None
            plans_by_schema_id[id(elastic_plan.schema)] = elastic_plan
            local_max_by_schema_id[id(elastic_plan.schema)] = elastic_local_max

        for schema in scaled:
            multiplier = schema.capacity_requirement.blocks_per_unit
            plans_by_schema_id[id(schema)] = compile_exact_simd_state_plan(
                schema, num_blocks=admitted_units * multiplier,
            )
            local_max_by_schema_id[id(schema)] = local_units * multiplier

        return tuple(
            _PlannedStateDomain(plans_by_schema_id[id(schema)],
                                local_max_by_schema_id[id(schema)])
            for schema in ordered
        )

    def _realize_state_plans(
        self, plans: Sequence[_PlannedStateDomain],
    ) -> StateGenerationRealization:
        """Allocate an already planned set with partial-supply rollback."""

        realized: list[StateDomainRealization] = []
        try:
            for domain in plans:
                plan = domain.plan
                realized.append(
                    StateDomainRealization(
                        plan=plan,
                        local_max_capacity=domain.local_max_capacity,
                        allocation=self._allocate_state_domain(plan),
                        allocated_state_bytes=self._allocation_bytes(plan),
                    )
                )
        except BaseException:
            for domain in reversed(realized):
                self._release_state_domain(domain)
            raise
        return StateGenerationRealization(tuple(realized))

    def release_state(self, realization: StateGenerationRealization) -> None:
        """Release one complete realization after module bindings are gone."""

        if not isinstance(realization, StateGenerationRealization):
            raise LiveModuleError(
                "invalid-state-generation-realization",
                "StateBackend can release only its typed generation resource",
                realization_type=type(realization).__qualname__,
            )
        for domain in reversed(realization.domains):
            self._release_state_domain(domain)

    def _allocation_bytes(self, plan: SIMDStatePlan) -> int:
        """Monotone physical charge, including provider alignment/padding."""
        return plan.committed_state_bytes

    def _allocate_state_domain(self, plan: SIMDStatePlan) -> object:
        raise NotImplementedError

    def _release_state_domain(self, realization: StateDomainRealization) -> None:
        del realization


def validate_state_generation_realization(
    schemas: Sequence[SIMDStateSchema],
    realization: object,
) -> tuple[tuple[StateDomainRealization, tuple[torch.Tensor, ...]], ...]:
    """Validate a provider result before any domain or StateTensor is bound."""

    expected = tuple(schemas)
    if not isinstance(realization, StateGenerationRealization):
        raise StateTensorError(
            "invalid-state-generation-realization",
            "StateBackend must return one typed complete State realization",
            realization_type=type(realization).__qualname__,
        )
    if len(realization.domains) != len(expected):
        raise StateTensorError(
            "incomplete-state-generation-realization",
            "StateBackend realization must cover every requested domain",
            expected_domain_count=len(expected),
            actual_domain_count=len(realization.domains),
        )
    unit_counts = {}
    primary_units = set()
    for schema, domain in zip(expected, realization.domains, strict=True):
        if (
            not isinstance(domain, StateDomainRealization)
            or domain.plan.schema is not schema
        ):
            raise StateTensorError(
                "foreign-state-domain-realization",
                "StateBackend must preserve exact requested domain schemas and order",
            )
        if (
            type(domain.local_max_capacity) is not int
            or domain.local_max_capacity < domain.plan.num_blocks
        ):
            raise StateTensorError(
                "invalid-local-state-capacity",
                "backend local maximum must cover admitted domain capacity",
                local_max_capacity=domain.local_max_capacity,
                admitted_capacity=domain.plan.num_blocks,
            )
        if domain.allocated_state_bytes is not None and (
            type(domain.allocated_state_bytes) is not int
            or domain.allocated_state_bytes < domain.plan.committed_state_bytes
        ):
            raise StateTensorError("invalid-state-allocation-charge",
                "physical State allocation charge must cover the logical State extent")
        requirement = schema.capacity_requirement
        if (
            isinstance(requirement, ExactStateCapacity)
            and domain.plan.num_blocks != requirement.capacity
        ):
            raise StateTensorError(
                "unsatisfied-exact-state-capacity",
                "backend realization does not satisfy the declared exact capacity",
                required_capacity=requirement.capacity,
                realized_capacity=domain.plan.num_blocks,
            )
        if isinstance(requirement, (ElasticStateCapacity, ScaledStateCapacity)) and requirement.unit is not None:
            unit = requirement.unit
            count, remainder = divmod(domain.plan.num_blocks, requirement.blocks_per_unit)
            if (remainder or count < unit.minimum_units
                    or unit_counts.setdefault(unit, count) != count):
                raise StateTensorError("inconsistent-state-capacity-unit",
                                       "related State domains must realize the same complete admission units")
            if isinstance(requirement, ElasticStateCapacity):
                primary_units.add(unit)
    if set(unit_counts) != primary_units:
        raise StateTensorError("unbound-scaled-state-capacity",
                               "scaled State realization has no primary elastic unit")
    values = validate_state_domain_allocations(
        tuple((domain.plan, domain.allocation) for domain in realization.domains)
    )
    return tuple(zip(realization.domains, values, strict=True))


class TorchStateBackend(StateBackend):
    """Supply every admitted domain lane as one independent Torch tensor."""

    __slots__ = ("_device", "_allocation_granularity_bytes", "capacity_report")

    def __init__(
        self,
        device: torch.device | str,
        *,
        memory_budget_bytes: int | None = None,
        capacity_coordinator: StateCapacityCoordinator | None = None,
        allocation_granularity_bytes: int = 1,
    ) -> None:
        super().__init__(
            memory_budget_bytes=memory_budget_bytes,
            capacity_coordinator=capacity_coordinator,
        )
        self._device = torch.device(device)
        if (
            type(allocation_granularity_bytes) is not int
            or allocation_granularity_bytes < 1
        ):
            raise ValueError("allocation granularity must be a positive byte count")
        self._allocation_granularity_bytes = allocation_granularity_bytes
        self.capacity_report = None

    @property
    def device(self) -> torch.device:
        return self._device

    def realize_calibration_state(
        self, schemas: Sequence[SIMDStateSchema],
    ) -> StateGenerationRealization:
        """Allocate the minimum complete unit family, agreeing on failures.

        Calibration must not enter positive-only capacity negotiation: a local
        allocation failure would otherwise leave a peer waiting for its vote.
        """
        coordinator = self._capacity_coordinator
        if coordinator is not None and not isinstance(coordinator, StateFitCoordinator):
            raise StateTensorError("unsupported-distributed-state-fit",
                                   "calibration requires failure-aware State agreement")
        held = None
        error = None
        try:
            try:
                ordered = tuple(schemas)
                primary = tuple(schema for schema in ordered if isinstance(
                    schema.capacity_requirement, ElasticStateCapacity))
                if len(primary) != 1 or primary[0].capacity_requirement.unit is None:
                    raise StateTensorError("unsupported-state-fit-geometry",
                                           "calibration requires one complete State capacity unit")
                units = primary[0].capacity_requirement.unit.minimum_units
                budget = 0
                for schema in ordered:
                    req = schema.capacity_requirement
                    blocks = (req.capacity if isinstance(req, ExactStateCapacity)
                              else units * req.blocks_per_unit)
                    budget += schema.fixed_state_bytes + blocks * schema.bytes_per_simd_block
                plans = self._plan_state(ordered, memory_budget_bytes=budget,
                                         capacity_coordinator=None)
                held = self._realize_state_plans(plans)
            except Exception as exc:
                error = exc
            confirmed = (error is None if coordinator is None
                         else coordinator.confirm_fitted_allocation(error is None))
            if error is not None:
                raise error
            if confirmed is not True:
                raise StateTensorError("peer-calibration-state-failed",
                                       "a rank could not allocate calibration State")
            result, held = held, None
            assert result is not None
            return result
        finally:
            if held is not None:
                self.release_state(held)

    def realize_state_fitting(
        self,
        schemas: Sequence[SIMDStateSchema],
        *,
        memory_observer: TorchDeviceMemoryObserver,
        minimum_free_bytes: int,
    ) -> StateGenerationRealization:
        """Allocate fixed State, size the shared domain, agree, allocate once.

        Calibration already holds execution resources. Cache fragments are not
        budget: only driver-free bytes after fixed State count. Reserve at most
        one configured allocator quantum per elastic lane for segment rounding.
        No OOM-driven search, speculative allocation or largest-fit claim.
        """
        if (
            not isinstance(memory_observer, TorchDeviceMemoryObserver)
            or memory_observer.device != self.device
        ):
            raise StateTensorError(
                "invalid-state-fit-observer",
                "State sizing requires its exact device observer",
            )
        if type(minimum_free_bytes) is not int or minimum_free_bytes < 0:
            raise StateTensorError(
                "invalid-state-fit-budget", "invalid free-memory floor"
            )
        coordinator = self._capacity_coordinator
        if coordinator is not None and not isinstance(coordinator, StateFitCoordinator):
            raise StateTensorError(
                "unsupported-distributed-state-fit",
                "State sizing requires failure-aware rank agreement",
            )
        ordered = tuple(schemas)
        fixed = elastic = None
        error = None
        local_units = 0
        self.capacity_report = None
        try:
            try:
                fixed_schemas = tuple(
                    s
                    for s in ordered
                    if isinstance(s.capacity_requirement, ExactStateCapacity)
                )
                variable = tuple(
                    s
                    for s in ordered
                    if not isinstance(s.capacity_requirement, ExactStateCapacity)
                )
                primary = tuple(
                    s
                    for s in variable
                    if isinstance(s.capacity_requirement, ElasticStateCapacity)
                )
                if len(primary) != 1 or primary[0].capacity_requirement.unit is None:
                    raise StateTensorError(
                        "unsupported-state-fit-geometry",
                        "State sizing requires one elastic capacity unit",
                    )
                requirement = primary[0].capacity_requirement
                if fixed_schemas:
                    fixed = self._realize_state_plans(
                        self._plan_state(
                            fixed_schemas,
                            memory_budget_bytes=None,
                            capacity_coordinator=None,
                        )
                    )
                memory_observer.reclaim()
                before = memory_observer.snapshot()
                padding = sum(len(s.lanes) for s in variable) * (
                    self._allocation_granularity_bytes - 1
                )
                budget = before.free_bytes - minimum_free_bytes - padding
                if self._memory_budget_bytes is not None:
                    fixed_bytes = (
                        sum(d.plan.committed_state_bytes for d in fixed.domains)
                        if fixed
                        else 0
                    )
                    budget = min(budget, self._memory_budget_bytes - fixed_bytes)
                if budget <= 0:
                    raise StateTensorError(
                        "insufficient-state-memory",
                        "no shared State budget remains after fixed State",
                    )
                plans = self._plan_state(
                    variable, memory_budget_bytes=budget, capacity_coordinator=None
                )
                primary_plan = next(d for d in plans if d.plan.schema is primary[0])
                local_units = (
                    primary_plan.plan.num_blocks // requirement.blocks_per_unit
                )
                self.capacity_report = dict(
                    free_after_fixed_bytes=before.free_bytes,
                    minimum_free_bytes=minimum_free_bytes,
                    allocation_padding_bytes=padding,
                    shared_budget_bytes=budget,
                    local_units=local_units,
                )
            except Exception as caught:
                error = caught
            admitted = (
                coordinator.admit_fitted_units(local_units)
                if coordinator
                else local_units
            )
            if type(admitted) is not int or not 0 <= admitted <= local_units:
                raise StateTensorError(
                    "invalid-fitted-state-capacity", "invalid shared capacity agreement"
                )
            if admitted == 0:
                if error is not None:
                    raise error
                raise StateTensorError(
                    "peer-state-fit-failed", "a rank could not size shared State"
                )
            try:
                if admitted < requirement.unit.minimum_units:
                    raise StateTensorError(
                        "insufficient-admitted-state-capacity",
                        "shared capacity is below the declared minimum",
                    )
                final_plans = tuple(
                    _PlannedStateDomain(
                        compile_exact_simd_state_plan(
                            d.plan.schema,
                            num_blocks=admitted
                            * d.plan.schema.capacity_requirement.blocks_per_unit,
                        ),
                        d.local_max_capacity,
                    )
                    for d in plans
                )
                elastic = self._realize_state_plans(final_plans)
                memory_observer.reclaim()
                free = memory_observer.snapshot().free_bytes
                self.capacity_report.update(
                    admitted_units=admitted, free_after_allocation_bytes=free
                )
                if free < minimum_free_bytes:
                    raise StateTensorError(
                        "state-allocation-exceeds-budget",
                        "shared State allocation breached the explicit free-memory floor",
                        **self.capacity_report,
                    )
            except Exception as caught:
                error = caught
            confirmed = (
                coordinator.confirm_fitted_allocation(error is None)
                if coordinator
                else error is None
            )
            if type(confirmed) is not bool:
                raise StateTensorError(
                    "invalid-state-allocation-confirmation",
                    "invalid allocation agreement",
                )
            if error is not None:
                raise error
            if not confirmed:
                raise StateTensorError(
                    "peer-state-allocation-failed",
                    "a rank could not allocate shared State",
                )
            by_schema = {
                id(d.plan.schema): d for r in (fixed, elastic) if r for d in r.domains
            }
            result = StateGenerationRealization(
                tuple(by_schema[id(s)] for s in ordered)
            )
            fixed = elastic = None
            return result
        finally:
            for held in (elastic, fixed):
                if held is not None:
                    self.release_state(held)
            memory_observer.reclaim()

    def _allocate_state_domain(
        self,
        plan: SIMDStatePlan,
    ) -> dict[StateTensor, torch.Tensor]:
        return {
            lane.state: torch.zeros(
                lane.state.physical_shape(plan.num_blocks),
                dtype=lane.state.storage_dtype,
                device=self._device,
            )
            for lane in plan.schema.lanes
        }

    def _release_state_domain(self, realization: StateDomainRealization) -> None:
        # A retired receipt may remain available for diagnostics. It must not
        # keep owning device tensors after this provider releases the domain.
        allocation = realization.allocation
        if not isinstance(allocation, dict):
            raise StateTensorError("invalid-torch-state-allocation",
                                   "Torch State release requires its tensor mapping")
        allocation.clear()


__all__ = (
    "StateBackend",
    "StateDomainRealization",
    "StateGenerationRealization",
    "TorchStateBackend",
    "validate_state_generation_realization",
)
