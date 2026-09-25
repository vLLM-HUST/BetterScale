# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Flat block-unit State declarations and homogeneous SIMD planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from betterscale.live.core.error import LiveModuleError
from betterscale.live.core.state_shadow import state_shadow_resolver

if TYPE_CHECKING:
    from betterscale.live.core.live_module import LiveModule


TensorT = TypeVar("TensorT", bound=torch.Tensor)
RoleT = TypeVar("RoleT")
RequirementT = TypeVar("RequirementT")


class StateTensorError(LiveModuleError):
    """Machine-classified failure of a LiveModule State declaration."""


class StateCapacityRequirement:
    """Marker for one module-declared logical capacity constraint."""


@dataclass(frozen=True, slots=True, eq=False)
class StateCapacityUnit:
    """One shared admission unit across distinct State addressing domains.

    Identity, not equal geometry, joins domains. For example one resident can
    require one owner-local ring, several canonical pages and all-rank control
    rows. Their costs must be charged together without merging address spaces.
    """

    minimum_units: int = 1

    def __post_init__(self) -> None:
        if type(self.minimum_units) is not int or self.minimum_units <= 0:
            raise StateTensorError("invalid-state-capacity-unit",
                                   "a State capacity unit requires a positive minimum")


@dataclass(frozen=True, slots=True)
class ElasticStateCapacity(StateCapacityRequirement):
    """Admit one primary domain, optionally in a shared indivisible unit."""

    unit: StateCapacityUnit | None = None
    blocks_per_unit: int = 1

    def __post_init__(self) -> None:
        if (self.unit is not None and not isinstance(self.unit, StateCapacityUnit)
                or type(self.blocks_per_unit) is not int or self.blocks_per_unit <= 0
                or self.unit is None and self.blocks_per_unit != 1):
            raise StateTensorError("invalid-elastic-state-capacity",
                                   "elastic capacity requires a valid unit and block multiplier")


@dataclass(frozen=True, slots=True)
class ScaledStateCapacity(StateCapacityRequirement):
    """Couple this domain's count to the primary domain's admission unit."""

    unit: StateCapacityUnit
    blocks_per_unit: int

    def __post_init__(self) -> None:
        if (not isinstance(self.unit, StateCapacityUnit)
                or type(self.blocks_per_unit) is not int or self.blocks_per_unit <= 0):
            raise StateTensorError("invalid-scaled-state-capacity",
                                   "scaled capacity requires a unit and positive block multiplier")


@dataclass(frozen=True, slots=True)
class ExactStateCapacity(StateCapacityRequirement):
    """Require one exact construction-known logical capacity."""

    capacity: int

    def __post_init__(self) -> None:
        if type(self.capacity) is not int or self.capacity <= 0:
            raise StateTensorError(
                "invalid-exact-state-capacity",
                "an exact State capacity must be one positive integer",
                capacity=self.capacity,
            )


@dataclass(frozen=True, slots=True)
class _StateDomainBinding:
    generation: object
    plan: SIMDStatePlan
    local_max_capacity: int


class StateDomain:
    """One exact addressing domain and capacity demand shared by State lanes."""

    __slots__ = ("_binding", "_capacity_requirement")

    def __init__(self, capacity: StateCapacityRequirement) -> None:
        if not isinstance(capacity, StateCapacityRequirement):
            raise StateTensorError(
                "invalid-state-capacity-requirement",
                "a State domain requires one typed capacity constraint",
                capacity_type=type(capacity).__qualname__,
            )
        self._capacity_requirement = capacity
        self._binding: _StateDomainBinding | None = None

    @property
    def capacity_requirement(self) -> StateCapacityRequirement:
        return self._capacity_requirement

    @property
    def is_bound(self) -> bool:
        return self._binding is not None

    @property
    def plan(self) -> SIMDStatePlan:
        binding = self._binding
        if binding is None:
            raise StateTensorError(
                "state-domain-not-bound",
                "a State domain has no active generation plan",
                domain_type=type(self).__qualname__,
            )
        return binding.plan

    @property
    def capacity(self) -> int:
        """Return the logical capacity admitted for the active generation."""

        resolver = state_shadow_resolver.get()
        if resolver is not None:
            return resolver(self)[0]
        return self.plan.num_blocks

    @property
    def local_max_capacity(self) -> int:
        """Return the rank-local maximum observed before domain admission."""

        binding = self._binding
        if binding is None:
            raise StateTensorError(
                "state-domain-not-bound",
                "a State domain has no active local capacity",
                domain_type=type(self).__qualname__,
            )
        return binding.local_max_capacity

    def _bind(
        self,
        *,
        generation: object,
        plan: SIMDStatePlan,
        local_max_capacity: int,
    ) -> None:
        if self._binding is not None:
            raise StateTensorError(
                "state-domain-already-bound",
                "a State domain already belongs to an active generation",
                domain_type=type(self).__qualname__,
            )
        if plan.schema.domain is not self:
            raise StateTensorError(
                "foreign-state-domain-plan",
                "a State domain can bind only its exact compiled plan",
                domain_type=type(self).__qualname__,
            )
        if type(local_max_capacity) is not int or local_max_capacity < plan.num_blocks:
            raise StateTensorError(
                "invalid-local-state-capacity",
                "a State domain local maximum must cover its admitted capacity",
                local_max_capacity=local_max_capacity,
                admitted_capacity=plan.num_blocks,
            )
        self._binding = _StateDomainBinding(
            generation,
            plan,
            local_max_capacity,
        )

    def _unbind(self, *, generation: object) -> None:
        binding = self._binding
        if binding is None:
            raise StateTensorError(
                "state-domain-not-bound",
                "a State domain cannot unbind without an active generation",
                domain_type=type(self).__qualname__,
            )
        if binding.generation is not generation:
            raise StateTensorError(
                "foreign-state-domain-generation",
                "a State domain received an unbind from a foreign generation",
                domain_type=type(self).__qualname__,
            )
        self._binding = None


@dataclass(frozen=True, slots=True)
class _StateBinding(Generic[TensorT]):
    generation: object
    num_blocks: int
    value: TensorT


class StateTensor(Generic[TensorT, RoleT, RequirementT]):
    """One immutable block-unit State declaration and generation binding.

    ``block_shape`` and ``storage_dtype`` describe one native physical block.
    ``physical_blocks_per_logical_block`` lowers one shared logical block id
    into this State's native page domain, while ``leading_physical_blocks``
    accounts an optional generation-fixed prefix.  Capacity, device placement,
    allocation containers and block-table ingress are deliberately absent.
    One StateTensor always owns one exclusive storage span in an activated
    generation.
    """

    __slots__ = (
        "_binding",
        "_block_shape",
        "_domain",
        "_leading_physical_blocks",
        "_name",
        "_owner",
        "_physical_blocks_per_logical_block",
        "_requirement",
        "_role",
        "_storage_dtype",
    )

    def __init__(
        self,
        *,
        role: RoleT,
        requirement: RequirementT,
        block_shape: Sequence[int],
        storage_dtype: torch.dtype,
        domain: StateDomain | None = None,
        physical_blocks_per_logical_block: int = 1,
        leading_physical_blocks: int = 0,
    ) -> None:
        if role is None:
            raise StateTensorError(
                "invalid-state-role",
                "StateTensor requires one non-None semantic role",
            )
        if requirement is None:
            raise StateTensorError(
                "invalid-state-requirement",
                "StateTensor requires one non-None semantic requirement",
                role=repr(role),
            )
        if domain is not None and not isinstance(domain, StateDomain):
            raise StateTensorError(
                "invalid-state-domain",
                "StateTensor domain must be one typed StateDomain declaration",
                role=repr(role),
                domain_type=type(domain).__qualname__,
            )
        try:
            declared_shape = tuple(block_shape)
        except TypeError as error:
            raise StateTensorError(
                "invalid-state-block-shape",
                "StateTensor block_shape must be one finite integer sequence",
                role=repr(role),
            ) from error
        if any(
            type(dimension) is not int or dimension <= 0 for dimension in declared_shape
        ):
            raise StateTensorError(
                "invalid-state-block-shape",
                "StateTensor block dimensions must be positive integers",
                role=repr(role),
                block_shape=declared_shape,
            )
        if not isinstance(storage_dtype, torch.dtype):
            raise StateTensorError(
                "invalid-state-storage-dtype",
                "StateTensor storage_dtype must be one concrete torch dtype",
                role=repr(role),
                storage_dtype=repr(storage_dtype),
            )
        if (
            type(physical_blocks_per_logical_block) is not int
            or physical_blocks_per_logical_block <= 0
        ):
            raise StateTensorError(
                "invalid-state-logical-block-span",
                "StateTensor logical blocks must span a positive number of "
                "physical blocks",
                physical_blocks_per_logical_block=(physical_blocks_per_logical_block),
            )
        if type(leading_physical_blocks) is not int or leading_physical_blocks < 0:
            raise StateTensorError(
                "invalid-state-leading-blocks",
                "StateTensor leading physical block count must be nonnegative",
                leading_physical_blocks=leading_physical_blocks,
            )
        self._owner: LiveModule | None = None
        self._name: str | None = None
        self._role = role
        self._requirement = requirement
        self._domain = domain
        self._block_shape = declared_shape
        self._storage_dtype = storage_dtype
        self._physical_blocks_per_logical_block = physical_blocks_per_logical_block
        self._leading_physical_blocks = leading_physical_blocks
        self._binding: _StateBinding[TensorT] | None = None

    @property
    def owner(self) -> LiveModule:
        owner = self._owner
        if owner is None:
            raise StateTensorError(
                "unregistered-state-tensor",
                "StateTensor must be registered on one LiveModule",
                role=repr(self._role),
            )
        return owner

    @property
    def name(self) -> str:
        name = self._name
        if name is None:
            raise StateTensorError(
                "unregistered-state-tensor",
                "StateTensor must be registered on one LiveModule",
                role=repr(self._role),
            )
        return name

    def _register(self, *, owner: LiveModule, name: str) -> None:
        """Bind this construction-time declaration to one local registry key."""

        if self._owner is not None:
            raise StateTensorError(
                "shared-state-tensor-declaration",
                "one StateTensor cannot be registered by more than one attribute",
                registered_owner_type=type(self._owner).__qualname__,
                registered_name=self._name,
                requested_owner_type=type(owner).__qualname__,
                requested_name=name,
            )
        self._owner = owner
        self._name = name

    @property
    def role(self) -> RoleT:
        return self._role

    @property
    def requirement(self) -> RequirementT:
        return self._requirement

    @property
    def domain(self) -> StateDomain | None:
        """Return the exact explicit addressing domain, or the implicit default."""

        return self._domain

    @property
    def block_shape(self) -> tuple[int, ...]:
        return self._block_shape

    @property
    def storage_dtype(self) -> torch.dtype:
        return self._storage_dtype

    @property
    def block_bytes(self) -> int:
        """Return bytes in one native physical State block."""

        return prod(self._block_shape) * self._storage_dtype.itemsize

    @property
    def physical_blocks_per_logical_block(self) -> int:
        """Return this lane's native page span for one shared logical id."""

        return self._physical_blocks_per_logical_block

    @property
    def leading_physical_blocks(self) -> int:
        """Return fixed native pages reserved before logical block zero."""

        return self._leading_physical_blocks

    @property
    def logical_block_bytes(self) -> int:
        """Return this lane's variable payload for one shared logical id."""

        return self.block_bytes * self._physical_blocks_per_logical_block

    @property
    def fixed_bytes(self) -> int:
        """Return generation-fixed storage preceding the logical domain."""

        return self.block_bytes * self._leading_physical_blocks

    def physical_shape(self, num_blocks: int) -> tuple[int, ...]:
        """Lower a shared logical capacity into this lane's native shape."""

        if type(num_blocks) is not int or num_blocks <= 0:
            raise StateTensorError(
                "invalid-state-logical-capacity",
                "StateTensor physical shape requires positive logical capacity",
                num_blocks=num_blocks,
            )
        physical_blocks = self._leading_physical_blocks + (
            num_blocks * self._physical_blocks_per_logical_block
        )
        return (physical_blocks, *self._block_shape)

    @property
    def is_bound(self) -> bool:
        return self._binding is not None

    @property
    def num_blocks(self) -> int:
        resolver = state_shadow_resolver.get()
        if resolver is not None:
            return resolver(self)[0]
        binding = self._binding
        if binding is None:
            raise StateTensorError(
                "state-not-bound",
                "StateTensor has no active block capacity",
                owner_type=type(self._owner).__qualname__,
                role=repr(self._role),
            )
        return binding.num_blocks

    @property
    def tensor(self) -> TensorT:
        """Return physical storage, or a build-scoped remote shape view."""

        resolver = state_shadow_resolver.get()
        if resolver is not None:
            return resolver(self)[1]
        binding = self._binding
        if binding is None:
            raise StateTensorError(
                "state-not-bound",
                "StateTensor has no active root generation binding",
                owner_type=type(self._owner).__qualname__,
                role=repr(self._role),
            )
        return binding.value

    def validate_physical_value(self, value: TensorT, *, num_blocks: int) -> None:
        """Validate one proposed exclusive State span before binding."""

        if not isinstance(value, torch.Tensor):
            raise StateTensorError(
                "invalid-state-value",
                "State allocation must return one torch Tensor",
                owner_type=type(self._owner).__qualname__,
                role=repr(self._role),
                value_type=type(value).__qualname__,
            )
        expected_shape = self.physical_shape(num_blocks)
        if tuple(value.shape) != expected_shape:
            raise StateTensorError(
                "invalid-state-value-shape",
                "State allocation does not match its admitted block geometry",
                owner_type=type(self._owner).__qualname__,
                role=repr(self._role),
                expected_shape=expected_shape,
                actual_shape=tuple(value.shape),
            )
        if value.dtype is not self._storage_dtype:
            raise StateTensorError(
                "invalid-state-value-dtype",
                "State allocation does not match its declared storage dtype",
                owner_type=type(self._owner).__qualname__,
                role=repr(self._role),
                expected_dtype=str(self._storage_dtype),
                actual_dtype=str(value.dtype),
            )
        if not value.is_contiguous():
            raise StateTensorError(
                "noncontiguous-state-value",
                "State allocation must expose one contiguous exclusive span",
                owner_type=type(self._owner).__qualname__,
                role=repr(self._role),
            )

    def clear_logical_block(self, block_id: int) -> None:
        """Clear the exact native page run belonging to one logical id."""

        if type(block_id) is not int or block_id < 0 or block_id >= self.num_blocks:
            raise StateTensorError(
                "state-block-reset-out-of-range",
                "StateTensor logical block reset exceeds active capacity",
                block_id=block_id,
                state_capacity=self.num_blocks,
            )
        first = self._leading_physical_blocks + (
            block_id * self._physical_blocks_per_logical_block
        )
        last = first + self._physical_blocks_per_logical_block
        self.tensor[first:last].zero_()

    def _bind(
        self,
        *,
        generation: object,
        num_blocks: int,
        value: TensorT,
    ) -> None:
        if self._binding is not None:
            raise StateTensorError(
                "state-already-bound",
                "StateTensor already belongs to an active root generation",
                owner_type=type(self._owner).__qualname__,
                role=repr(self._role),
            )
        self.validate_physical_value(value, num_blocks=num_blocks)
        self._binding = _StateBinding(generation, num_blocks, value)

    def _unbind(self, *, generation: object) -> None:
        binding = self._binding
        if binding is None:
            raise StateTensorError(
                "state-not-bound",
                "StateTensor cannot unbind without an active generation",
                owner_type=type(self._owner).__qualname__,
                role=repr(self._role),
            )
        if binding.generation is not generation:
            raise StateTensorError(
                "foreign-state-generation",
                "StateTensor received an unbind from a foreign generation",
                owner_type=type(self._owner).__qualname__,
                role=repr(self._role),
            )
        self._binding = None


class StateGenerationSnapshot:
    """Host images of selected State spans for synchronous generation build.

    These images are construction-only backups, not replay workspace. Keeping
    them on the host avoids a second device charge beside final State storage.
    Snapshot transfers are blocking; the graph backend synchronizes before and
    after restore. Restores copy into the original captured destinations, never
    replace their storage.
    """

    __slots__ = ("_entries", "_released")

    def __init__(
        self,
        states: Sequence[StateTensor],
        *,
        domain_blocks: Mapping[StateDomain | None, Sequence[int]],
    ) -> None:
        ordered = tuple(states)
        if not isinstance(domain_blocks, Mapping):
            raise StateTensorError(
                "missing-state-generation-snapshot-blocks",
                "State snapshots require an explicit domain-to-block selection",
                selection_type=type(domain_blocks).__qualname__,
            )
        selected_by_domain: dict[StateDomain | None, tuple[int, ...]] = {}
        for domain, block_ids in domain_blocks.items():
            if domain is not None and not isinstance(domain, StateDomain):
                raise StateTensorError(
                    "invalid-state-generation-snapshot-domain",
                    "State snapshot domains must be typed StateDomain values",
                    domain_type=type(domain).__qualname__,
                )
            try:
                requested = tuple(block_ids)
            except TypeError as error:
                raise StateTensorError(
                    "invalid-state-generation-snapshot-blocks",
                    "State snapshot block ids must be one finite sequence",
                ) from error
            if any(
                type(block_id) is not int or block_id < 0 for block_id in requested
            ):
                raise StateTensorError(
                    "invalid-state-generation-snapshot-blocks",
                    "State snapshot block ids must be nonnegative integers",
                    block_ids=requested,
                )
            selected_by_domain[domain] = tuple(dict.fromkeys(requested))
        seen: set[int] = set()
        entries: list[
            tuple[
                StateTensor,
                torch.Tensor,
                tuple[tuple[int, int, torch.Tensor], ...],
            ]
        ] = []
        for state in ordered:
            if not isinstance(state, StateTensor):
                raise StateTensorError(
                    "invalid-state-generation-snapshot",
                    "a State generation snapshot accepts only StateTensor values",
                    state_type=type(state).__qualname__,
                )
            if id(state) in seen:
                raise StateTensorError(
                    "duplicate-state-generation-snapshot",
                    "one StateTensor cannot occur twice in a generation snapshot",
                )
            seen.add(id(state))
            destination = state.tensor
            if state.domain not in selected_by_domain:
                continue
            selected_blocks = selected_by_domain[state.domain]
            if any(block_id >= state.num_blocks for block_id in selected_blocks):
                raise StateTensorError(
                    "state-generation-snapshot-block-out-of-range",
                    "State snapshot exceeds the active logical capacity",
                    owner_type=type(state.owner).__qualname__,
                    role=repr(state.role),
                    block_ids=selected_blocks,
                    state_capacity=state.num_blocks,
                )
            selected_spans: list[tuple[int, int]] = []
            if state.leading_physical_blocks:
                selected_spans.append((0, state.leading_physical_blocks))
            for block_id in selected_blocks:
                first = state.leading_physical_blocks + (
                    block_id * state.physical_blocks_per_logical_block
                )
                last = first + state.physical_blocks_per_logical_block
                selected_spans.append((first, last))
            spans = tuple(
                # copy=True also gives CPU State an independent initial image.
                # Blocking transfers are intentional: graph construction owns
                # these images, with no asynchronous host-buffer lifetime.
                (first, last, destination[first:last].to(device="cpu", copy=True))
                for first, last in selected_spans
            )
            if spans:
                entries.append((state, destination, spans))
        self._entries = tuple(entries)
        self._released = False

    def restore(self) -> None:
        """Restore every exact destination without assuming a zero initializer."""

        if self._released:
            raise StateTensorError(
                "released-state-generation-snapshot",
                "a released State generation snapshot cannot restore State",
            )
        with torch.no_grad():
            for state, destination, spans in self._entries:
                if state.tensor is not destination:
                    raise StateTensorError(
                        "replaced-state-generation-destination",
                        "State storage identity changed after snapshot capture",
                        owner_type=type(state.owner).__qualname__,
                        role=repr(state.role),
                    )
                for first, last, initial in spans:
                    destination[first:last].copy_(initial)

    def release(self) -> None:
        """Drop retained initial images before the State generation is unbound."""

        self._entries = ()
        self._released = True


@dataclass(frozen=True, slots=True)
class SIMDStateLane:
    """One exact StateTensor at its deterministic root-relative lane key."""

    module_path: str
    state_name: str
    state: StateTensor[torch.Tensor, object, object]

    @property
    def key(self) -> tuple[str, str]:
        return (self.module_path, self.state_name)


@dataclass(frozen=True, slots=True)
class SIMDStateSchema:
    """One immutable homogeneous block schema before capacity admission."""

    lanes: tuple[SIMDStateLane, ...]
    bytes_per_simd_block: int
    fixed_state_bytes: int
    domain: StateDomain | None = None
    capacity_requirement: StateCapacityRequirement = ElasticStateCapacity()


@dataclass(frozen=True, slots=True)
class SIMDStatePlan:
    """One admitted common block capacity for an exact State schema."""

    schema: SIMDStateSchema
    usable_state_bytes: int
    num_blocks: int
    committed_state_bytes: int
    residual_state_bytes: int


def compile_simd_state_schema(
    lanes: Sequence[SIMDStateLane],
    *,
    domain: StateDomain | None = None,
) -> SIMDStateSchema:
    """Compile exact State lanes into one homogeneous logical block schema."""

    ordered = tuple(lanes)
    if not ordered:
        raise StateTensorError(
            "empty-simd-state-schema",
            "SIMD State schema requires at least one exact State lane",
        )
    seen_states: set[int] = set()
    seen_keys: set[tuple[str, str]] = set()
    for lane in ordered:
        if not isinstance(lane, SIMDStateLane):
            raise StateTensorError(
                "invalid-simd-state-lane",
                "SIMD State compiler accepts only typed State lanes",
                lane_type=type(lane).__qualname__,
            )
        if id(lane.state) in seen_states:
            raise StateTensorError(
                "duplicate-simd-state",
                "one exact StateTensor cannot occupy multiple SIMD lanes",
                module_path=lane.module_path,
                state_name=lane.state_name,
            )
        if lane.key in seen_keys:
            raise StateTensorError(
                "duplicate-simd-state-lane-key",
                "SIMD State lane keys must be unique inside one root",
                module_path=lane.module_path,
                state_name=lane.state_name,
            )
        seen_states.add(id(lane.state))
        seen_keys.add(lane.key)
    return SIMDStateSchema(
        lanes=ordered,
        bytes_per_simd_block=sum(lane.state.logical_block_bytes for lane in ordered),
        fixed_state_bytes=sum(lane.state.fixed_bytes for lane in ordered),
        domain=domain,
        capacity_requirement=(
            ElasticStateCapacity() if domain is None else domain.capacity_requirement
        ),
    )


def compile_simd_state_plan(
    schema: SIMDStateSchema,
    *,
    usable_state_bytes: int,
) -> SIMDStatePlan:
    """Select one common block count from an explicit generation budget."""

    if not isinstance(schema, SIMDStateSchema):
        raise StateTensorError(
            "invalid-simd-state-schema",
            "SIMD capacity compilation requires one typed schema",
            schema_type=type(schema).__qualname__,
        )
    if type(usable_state_bytes) is not int or usable_state_bytes <= 0:
        raise StateTensorError(
            "invalid-state-memory-budget",
            "usable State memory must be one positive integer byte count",
            usable_state_bytes=usable_state_bytes,
        )
    variable_budget = usable_state_bytes - schema.fixed_state_bytes
    num_blocks = variable_budget // schema.bytes_per_simd_block
    if num_blocks <= 0:
        raise StateTensorError(
            "insufficient-state-memory",
            "usable State memory cannot hold one complete SIMD block",
            usable_state_bytes=usable_state_bytes,
            bytes_per_simd_block=schema.bytes_per_simd_block,
            fixed_state_bytes=schema.fixed_state_bytes,
        )
    committed = schema.fixed_state_bytes + (num_blocks * schema.bytes_per_simd_block)
    return SIMDStatePlan(
        schema=schema,
        usable_state_bytes=usable_state_bytes,
        num_blocks=num_blocks,
        committed_state_bytes=committed,
        residual_state_bytes=usable_state_bytes - committed,
    )


def admit_simd_state_plan(
    local_plan: SIMDStatePlan,
    *,
    admitted_num_blocks: int,
) -> SIMDStatePlan:
    """Lower one admitted logical count into an exact rank-local State plan."""

    if not isinstance(local_plan, SIMDStatePlan):
        raise StateTensorError(
            "invalid-local-simd-state-plan",
            "SIMD State admission requires one typed local capacity plan",
            plan_type=type(local_plan).__qualname__,
        )
    if type(admitted_num_blocks) is not int or admitted_num_blocks <= 0:
        raise StateTensorError(
            "invalid-admitted-state-capacity",
            "admitted SIMD State capacity must be one positive logical block count",
            admitted_num_blocks=admitted_num_blocks,
        )
    if admitted_num_blocks > local_plan.num_blocks:
        raise StateTensorError(
            "admitted-state-capacity-exceeds-local-maximum",
            "admitted SIMD State capacity exceeds the compiled local maximum",
            local_max_blocks=local_plan.num_blocks,
            admitted_num_blocks=admitted_num_blocks,
        )
    committed = local_plan.schema.fixed_state_bytes + (
        admitted_num_blocks * local_plan.schema.bytes_per_simd_block
    )
    return SIMDStatePlan(
        schema=local_plan.schema,
        usable_state_bytes=local_plan.usable_state_bytes,
        num_blocks=admitted_num_blocks,
        committed_state_bytes=committed,
        residual_state_bytes=local_plan.usable_state_bytes - committed,
    )


def compile_exact_simd_state_plan(
    schema: SIMDStateSchema,
    *,
    num_blocks: int,
) -> SIMDStatePlan:
    """Compile one exact-capacity domain without provider admission."""

    if not isinstance(schema, SIMDStateSchema):
        raise StateTensorError(
            "invalid-simd-state-schema",
            "exact SIMD capacity compilation requires one typed schema",
            schema_type=type(schema).__qualname__,
        )
    if type(num_blocks) is not int or num_blocks <= 0:
        raise StateTensorError(
            "invalid-exact-state-capacity",
            "an exact State plan requires one positive logical capacity",
            num_blocks=num_blocks,
        )
    committed = schema.fixed_state_bytes + (num_blocks * schema.bytes_per_simd_block)
    return SIMDStatePlan(
        schema=schema,
        usable_state_bytes=committed,
        num_blocks=num_blocks,
        committed_state_bytes=committed,
        residual_state_bytes=0,
    )


def validate_simd_state_allocation(
    plan: SIMDStatePlan,
    values: object,
) -> tuple[torch.Tensor, ...]:
    """Validate a complete backend allocation and prove disjoint State spans."""

    if not isinstance(values, Mapping):
        raise StateTensorError(
            "invalid-simd-state-allocation",
            "SIMD State backend must return a mapping keyed by exact StateTensor",
            allocation_type=type(values).__qualname__,
        )
    expected_by_id = {id(lane.state): lane.state for lane in plan.schema.lanes}
    actual_by_id: dict[int, tuple[object, object]] = {}
    for state, value in values.items():
        if not isinstance(state, StateTensor):
            raise StateTensorError(
                "foreign-simd-state-allocation",
                "SIMD State allocation contains a non-StateTensor key",
                key_type=type(state).__qualname__,
            )
        actual_by_id[id(state)] = (state, value)
    if actual_by_id.keys() != expected_by_id.keys() or any(
        actual_by_id[state_id][0] is not expected_state
        for state_id, expected_state in expected_by_id.items()
        if state_id in actual_by_id
    ):
        raise StateTensorError(
            "incomplete-simd-state-allocation",
            "SIMD State allocation must cover every exact lane and no others",
            expected_state_count=len(expected_by_id),
            actual_state_count=len(actual_by_id),
        )

    ordered_values: list[torch.Tensor] = []
    occupied: list[tuple[tuple[str, int | None], int, int, SIMDStateLane]] = []
    for lane in plan.schema.lanes:
        value = actual_by_id[id(lane.state)][1]
        lane.state.validate_physical_value(value, num_blocks=plan.num_blocks)
        assert isinstance(value, torch.Tensor)
        start = value.data_ptr()
        end = start + value.numel() * value.element_size()
        device_key = (value.device.type, value.device.index)
        for other_device, other_start, other_end, other_lane in occupied:
            if device_key == other_device and start < other_end and other_start < end:
                raise StateTensorError(
                    "overlapping-state-storage",
                    "distinct StateTensors must receive disjoint exclusive spans",
                    first_lane=other_lane.key,
                    second_lane=lane.key,
                )
        occupied.append((device_key, start, end, lane))
        ordered_values.append(value)
    return tuple(ordered_values)


def validate_state_domain_allocations(
    allocations: Sequence[tuple[SIMDStatePlan, object]],
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Validate complete per-domain allocations and cross-domain exclusivity."""

    ordered = tuple(allocations)
    validated = tuple(
        validate_simd_state_allocation(plan, values) for plan, values in ordered
    )
    occupied: list[tuple[tuple[str, int | None], int, int, SIMDStateLane]] = []
    seen_states: set[int] = set()
    for (plan, _), values in zip(ordered, validated, strict=True):
        for lane, value in zip(plan.schema.lanes, values, strict=True):
            if id(lane.state) in seen_states:
                raise StateTensorError(
                    "state-in-multiple-domains",
                    "one exact StateTensor cannot belong to multiple State domains",
                    lane=lane.key,
                )
            seen_states.add(id(lane.state))
            start = value.data_ptr()
            end = start + value.numel() * value.element_size()
            device_key = (value.device.type, value.device.index)
            for other_device, other_start, other_end, other_lane in occupied:
                if (
                    device_key == other_device
                    and start < other_end
                    and other_start < end
                ):
                    raise StateTensorError(
                        "overlapping-state-storage",
                        "StateTensors from distinct domains need disjoint spans",
                        first_lane=other_lane.key,
                        second_lane=lane.key,
                    )
            occupied.append((device_key, start, end, lane))
    return validated


__all__ = (
    "ElasticStateCapacity",
    "ExactStateCapacity",
    "ScaledStateCapacity",
    "StateCapacityUnit",
    "SIMDStateLane",
    "SIMDStatePlan",
    "SIMDStateSchema",
    "StateCapacityRequirement",
    "StateDomain",
    "StateGenerationSnapshot",
    "StateTensor",
    "StateTensorError",
    "admit_simd_state_plan",
    "compile_exact_simd_state_plan",
    "compile_simd_state_plan",
    "compile_simd_state_schema",
    "validate_simd_state_allocation",
    "validate_state_domain_allocations",
)
