# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Structured accelerator-memory evidence for one LiveModule activation."""

from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from betterscale.live.core.error import LiveModuleError
from betterscale.live.runtime.graph_memory import (
    GraphPoolMemorySnapshot, graph_pool_snapshot, state_allocation_cache_bytes,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from betterscale.live.core.live_module import LiveModule
    from betterscale.live.runtime.state_backend import StateDomainRealization


@dataclass(frozen=True, slots=True)
class DeviceMemorySnapshot:
    """One synchronized allocator and driver waterline."""

    free_bytes: int
    total_bytes: int
    allocated_bytes: int
    active_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_active_bytes: int
    peak_reserved_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.free_bytes,
            self.total_bytes,
            self.allocated_bytes,
            self.active_bytes,
            self.reserved_bytes,
            self.peak_allocated_bytes,
            self.peak_active_bytes,
            self.peak_reserved_bytes,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise LiveModuleError(
                "invalid-device-memory-snapshot",
                "device memory waterlines must be nonnegative integer bytes",
            )
        if self.total_bytes <= 0 or self.free_bytes > self.total_bytes:
            raise LiveModuleError(
                "invalid-device-memory-snapshot",
                "device free memory must fit inside one positive total extent",
                free_bytes=self.free_bytes,
                total_bytes=self.total_bytes,
            )
        if (
            self.peak_allocated_bytes < self.allocated_bytes
            or self.peak_active_bytes < self.active_bytes
            or self.peak_reserved_bytes < self.reserved_bytes
        ):
            raise LiveModuleError(
                "invalid-device-memory-peak",
                "device peak waterlines cannot be below their current values",
            )

    @property
    def driver_used_bytes(self) -> int:
        return self.total_bytes - self.free_bytes

    @property
    def non_allocator_bytes(self) -> int:
        """Device-wide residual, not attribution to this process or root."""

        return max(0, self.driver_used_bytes - self.reserved_bytes)

    @property
    def live_allocator_bytes(self) -> int:
        """Return live framework storage without unused allocator reserve."""

        return max(self.allocated_bytes, self.active_bytes)

    @property
    def attributed_live_bytes(self) -> int:
        """Legacy mixed-scope waterline; not root-owned live memory or a KV budget."""

        return self.live_allocator_bytes + self.non_allocator_bytes


@runtime_checkable
class DeviceMemoryObserver(Protocol):
    """Observe one already selected rank-local accelerator device."""

    def reset_peak_stats(self) -> None:
        """Begin one bounded allocator peak window."""

    def snapshot(self) -> DeviceMemorySnapshot:
        """Return one synchronized driver and allocator waterline."""


class TorchDeviceMemoryObserver:
    """Use the native torch accelerator allocator for one exact device."""

    def __init__(self, device: torch.device | str) -> None:
        self._device = torch.device(device)
        self._api = getattr(torch, self._device.type, None)
        required = (
            "synchronize",
            "reset_peak_memory_stats",
            "mem_get_info",
            "memory_allocated",
            "memory_reserved",
            "memory_stats",
        )
        missing = tuple(
            name for name in required if not callable(getattr(self._api, name, None))
        )
        if missing:
            raise LiveModuleError(
                "missing-torch-device-memory-api",
                "memory profiling requires the native torch device memory API",
                device_type=self._device.type,
                missing=missing,
            )

    @property
    def device(self) -> torch.device:
        return self._device

    def reclaim(self) -> None:
        """Synchronously release dead objects and reclaimable allocator cache."""

        self._api.synchronize(self._device)
        gc.collect()
        empty_cache = getattr(self._api, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
        self._api.synchronize(self._device)

    def reset_peak_stats(self) -> None:
        self._api.synchronize(self._device)
        self._api.reset_peak_memory_stats(self._device)

    def snapshot(self) -> DeviceMemorySnapshot:
        self._api.synchronize(self._device)
        free_bytes, total_bytes = self._api.mem_get_info(self._device)
        stats = self._api.memory_stats(self._device)
        allocated = int(self._api.memory_allocated(self._device))
        reserved = int(self._api.memory_reserved(self._device))
        active = int(stats.get("active_bytes.all.current", allocated))
        return DeviceMemorySnapshot(
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
            allocated_bytes=allocated,
            active_bytes=active,
            reserved_bytes=reserved,
            peak_allocated_bytes=int(stats.get("allocated_bytes.all.peak", allocated)),
            peak_active_bytes=int(stats.get("active_bytes.all.peak", active)),
            peak_reserved_bytes=int(stats.get("reserved_bytes.all.peak", reserved)),
        )

    def snapshot_state_cache_bytes(self) -> int | None:
        """Observe default-pool cache on the stream supplying State allocations."""
        api = getattr(self, "_api", None)
        snapshot = getattr(api, "memory_snapshot", None)
        current_stream = getattr(api, "current_stream", None)
        if not callable(snapshot) or not callable(current_stream):
            return None
        api.synchronize(self._device)
        index = self._device.index
        if index is None:
            index = api.current_device()
        stream = current_stream(self._device)
        stream_id = getattr(stream, "npu_stream", getattr(stream, "cuda_stream", None))
        if type(stream_id) is not int:
            return None
        return state_allocation_cache_bytes(snapshot(), device_index=index, stream_id=stream_id)

    def snapshot_pool(
        self, pool_id: tuple[int, int], stream_id: int, *, device: torch.device,
    ) -> GraphPoolMemorySnapshot | None:
        """Observe an explicitly selected private pool, when native APIs support it."""
        snapshot = getattr(self._api, "memory_snapshot", None)
        if not callable(snapshot):
            return None
        self._api.synchronize(self._device)
        device_index = self._device.index
        if device_index is None:
            device_index = self._api.current_device()
        requested_index = device.index
        if requested_index is None and device.type == self._device.type:
            requested_index = self._api.current_device()
        if device.type != self._device.type or requested_index != device_index:
            raise LiveModuleError("graph-memory-observer-device-mismatch",
                                  "graph pool and memory observer must select the same device")
        return graph_pool_snapshot(snapshot(), device_index=device_index,
                                   pool_id=pool_id, stream_id=stream_id)


@dataclass(frozen=True, slots=True)
class StateLaneMemoryProfile:
    """Exact byte geometry of one realized StateTensor lane."""

    module_path: str
    state_name: str
    fixed_bytes: int
    logical_block_bytes: int
    realized_bytes: int


@dataclass(frozen=True, slots=True)
class StateDomainMemoryProfile:
    """State budget and admitted extent for one addressing domain."""

    ordinal: int
    capacity_kind: str
    admitted_blocks: int
    local_max_blocks: int
    fixed_bytes: int
    logical_block_bytes: int
    committed_bytes: int
    realized_bytes: int
    lanes: tuple[StateLaneMemoryProfile, ...]
    allocated_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class MemoryProfileFailure:
    """The exact activation phase and exception that stopped profiling."""

    phase: str
    graph_key: tuple[str, str] | None
    error_type: str
    error_message: str
    snapshot: DeviceMemorySnapshot | None


@dataclass(frozen=True, slots=True)
class MemoryPhaseProfile:
    """One bounded activation phase with its allocator peak window."""

    phase: str
    graph_key: tuple[str, str] | None
    before: DeviceMemorySnapshot
    after: DeviceMemorySnapshot | None

    @property
    def retained_allocated_bytes(self) -> int | None:
        if self.after is None:
            return None
        return self.after.allocated_bytes - self.before.allocated_bytes

    @property
    def retained_reserved_bytes(self) -> int | None:
        """Return allocator residency retained across this phase."""

        if self.after is None:
            return None
        return self.after.reserved_bytes - self.before.reserved_bytes

    @property
    def retained_driver_bytes(self) -> int | None:
        """Return driver-visible residency retained across this phase."""

        if self.after is None:
            return None
        return self.after.driver_used_bytes - self.before.driver_used_bytes

    @property
    def peak_allocated_delta_bytes(self) -> int | None:
        if self.after is None:
            return None
        return max(0, self.after.peak_allocated_bytes - self.before.allocated_bytes)


@dataclass(frozen=True, slots=True)
class GraphMemoryAdmission:
    """Warmup-derived preflight estimate, not proof a capture will fit.

    No pool evidence means zero reuse credit, not proof the pool is empty.
    Native overhead is a historical device residual, not process attribution.
    """

    graph_key: tuple[str, str]
    warmup_graph_key: tuple[str, str]
    memory_graph_key: tuple[str, str]
    warmup_peak_bytes: int
    shared_pool_capacity_bytes: int
    incremental_allocator_bytes: int
    observed_graph_exec_overhead_bytes: int
    required_free_bytes: int
    available_free_bytes: int
    pool: GraphPoolMemorySnapshot | None = None

    @property
    def admitted(self) -> bool:
        return self.available_free_bytes >= self.required_free_bytes

    @property
    def deficit_bytes(self) -> int:
        return max(0, self.required_free_bytes - self.available_free_bytes)


@dataclass(frozen=True, slots=True)
class LiveMemoryProfile:
    """Immutable memory receipt retained by one LiveModule generation attempt."""

    model_storage_bytes: int
    initial: DeviceMemorySnapshot
    final: DeviceMemorySnapshot | None
    state_domains: tuple[StateDomainMemoryProfile, ...]
    phases: tuple[MemoryPhaseProfile, ...]
    graph_admissions: tuple[GraphMemoryAdmission, ...]
    failure: MemoryProfileFailure | None
    # Latest successful observation per native pool, not a complete final seal.
    # Empty means no attribution evidence, NOT zero graph residency.
    graph_pools: tuple[GraphPoolMemorySnapshot, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.failure is None and self.final is not None

    @property
    def state_committed_bytes(self) -> int:
        return sum(domain.committed_bytes for domain in self.state_domains)

    @property
    def peak_allocated_bytes(self) -> int:
        snapshots = [self.initial]
        snapshots.extend(
            snapshot
            for phase in self.phases
            for snapshot in (phase.before, phase.after)
            if snapshot is not None
        )
        if self.final is not None:
            snapshots.append(self.final)
        return max(snapshot.peak_allocated_bytes for snapshot in snapshots)

    @property
    def max_observed_driver_used_bytes(self) -> int:
        """Return the largest synchronized driver waterline in the receipt."""

        return max(snapshot.driver_used_bytes for snapshot in self._snapshots())

    @property
    def graph_portfolio_retained_bytes(self) -> int:
        """Legacy positive net allocated change, NOT absolute graph residency."""

        graph_phases = tuple(
            phase
            for phase in self.phases
            if phase.graph_key is not None and phase.after is not None
        )
        if not graph_phases:
            return 0
        after = graph_phases[-1].after
        assert after is not None
        return max(0, after.allocated_bytes - graph_phases[0].before.allocated_bytes)

    @property
    def graph_portfolio_retained_reserved_bytes(self) -> int:
        """Legacy positive net reserved change, NOT absolute graph residency."""

        bounds = self._graph_portfolio_bounds()
        if bounds is None:
            return 0
        before, after = bounds
        return max(0, after.reserved_bytes - before.reserved_bytes)

    @property
    def graph_portfolio_retained_driver_bytes(self) -> int:
        """Legacy positive device net change, NOT graph-owned native residency."""

        bounds = self._graph_portfolio_bounds()
        if bounds is None:
            return 0
        before, after = bounds
        return max(0, after.driver_used_bytes - before.driver_used_bytes)

    @property
    def graph_portfolio_net_allocated_bytes(self) -> int:
        bounds = self._graph_portfolio_bounds()
        return 0 if bounds is None else bounds[1].allocated_bytes - bounds[0].allocated_bytes

    @property
    def graph_portfolio_net_reserved_bytes(self) -> int:
        """Signed device-allocator change across preparation, including reclaim."""
        bounds = self._graph_portfolio_bounds()
        return 0 if bounds is None else bounds[1].reserved_bytes - bounds[0].reserved_bytes

    @property
    def graph_portfolio_net_driver_bytes(self) -> int:
        bounds = self._graph_portfolio_bounds()
        return 0 if bounds is None else bounds[1].driver_used_bytes - bounds[0].driver_used_bytes

    def _snapshots(self) -> tuple[DeviceMemorySnapshot, ...]:
        snapshots = [self.initial]
        snapshots.extend(
            snapshot
            for phase in self.phases
            for snapshot in (phase.before, phase.after)
            if snapshot is not None
        )
        if self.final is not None:
            snapshots.append(self.final)
        return tuple(snapshots)

    def _graph_portfolio_bounds(
        self,
    ) -> tuple[DeviceMemorySnapshot, DeviceMemorySnapshot] | None:
        graph_phases = tuple(
            phase
            for phase in self.phases
            if phase.graph_key is not None and phase.after is not None
        )
        if not graph_phases:
            return None
        after = graph_phases[-1].after
        assert after is not None
        return graph_phases[0].before, after


@dataclass(slots=True)
class _ActiveMemoryPhase:
    phase: str
    graph_key: tuple[str, str] | None
    before: DeviceMemorySnapshot


class LiveMemoryProfileSession:
    """Mutable activation-local builder used only by LiveModule and backends."""

    __slots__ = (
        "_active",
        "_failure",
        "_final",
        "_graph_admissions",
        "_graph_pools",
        "_initial",
        "_model_storage_bytes",
        "_observer",
        "_phases",
        "_state_domains",
    )

    def __init__(
        self,
        root: LiveModule,
        observer: DeviceMemoryObserver,
    ) -> None:
        if not isinstance(observer, DeviceMemoryObserver):
            raise LiveModuleError(
                "invalid-device-memory-observer",
                "LiveModule memory profiling requires one typed observer",
                observer_type=type(observer).__qualname__,
            )
        self._observer = observer
        self._model_storage_bytes = _unique_module_storage_bytes(root)
        self._initial = observer.snapshot()
        observer.reset_peak_stats()
        self._active: _ActiveMemoryPhase | None = None
        self._phases: list[MemoryPhaseProfile] = []
        self._graph_admissions: list[GraphMemoryAdmission] = []
        self._graph_pools: dict[tuple[int, tuple[int, int]], GraphPoolMemorySnapshot] = {}
        self._state_domains: tuple[StateDomainMemoryProfile, ...] = ()
        self._failure: MemoryProfileFailure | None = None
        self._final: DeviceMemorySnapshot | None = None

    @contextmanager
    def phase(
        self,
        phase: str,
        *,
        graph_key: tuple[str, str] | None = None,
    ) -> Iterator[None]:
        """Measure one non-overlapping activation phase and preserve failures."""

        if not phase:
            raise LiveModuleError(
                "invalid-memory-profile-phase",
                "memory profile phase names must be nonempty",
            )
        if self._active is not None:
            raise LiveModuleError(
                "nested-memory-profile-phase",
                "LiveModule activation memory phases cannot overlap",
                active_phase=self._active.phase,
                requested_phase=phase,
            )
        before = self._observer.snapshot()
        self._observer.reset_peak_stats()
        self._active = _ActiveMemoryPhase(phase, graph_key, before)
        try:
            yield
        except BaseException as error:
            self._fail_active(error)
            raise
        else:
            after = self._observer.snapshot()
            self._phases.append(MemoryPhaseProfile(phase, graph_key, before, after))
            self._active = None

    def record_state_domains(
        self,
        domains: Sequence[tuple[StateDomainRealization, tuple[torch.Tensor, ...]]],
    ) -> None:
        """Record exact declared and physical bytes after State realization."""

        profiles: list[StateDomainMemoryProfile] = []
        for ordinal, (domain, values) in enumerate(domains):
            plan = domain.plan
            lanes = tuple(
                StateLaneMemoryProfile(
                    module_path=lane.module_path,
                    state_name=lane.state_name,
                    fixed_bytes=lane.state.fixed_bytes,
                    logical_block_bytes=lane.state.logical_block_bytes,
                    realized_bytes=value.numel() * value.element_size(),
                )
                for lane, value in zip(plan.schema.lanes, values, strict=True)
            )
            profiles.append(
                StateDomainMemoryProfile(
                    ordinal=ordinal,
                    capacity_kind=type(plan.schema.capacity_requirement).__qualname__,
                    admitted_blocks=plan.num_blocks,
                    local_max_blocks=domain.local_max_capacity,
                    fixed_bytes=plan.schema.fixed_state_bytes,
                    logical_block_bytes=plan.schema.bytes_per_simd_block,
                    committed_bytes=plan.committed_state_bytes,
                    realized_bytes=sum(lane.realized_bytes for lane in lanes),
                    lanes=lanes,
                    allocated_bytes=domain.allocated_state_bytes,
                )
            )
        self._state_domains = tuple(profiles)

    def reclaim_warmup_cache(self) -> None:
        """Release calibration-only allocator cache before graph admission."""

        reclaim = getattr(self._observer, "reclaim", None)
        if not callable(reclaim):
            return
        with self.phase("graph.warmup.reclaim"):
            reclaim()

    def observe_graph_pool(
        self, native_pool: object, stream_id: int | None, *, device: torch.device,
    ) -> GraphPoolMemorySnapshot | None:
        """Observe the backend-selected native pool, never a schema estimate class."""
        observe = getattr(self._observer, "snapshot_pool", None)
        if (not callable(observe) or stream_id is None
                or not isinstance(native_pool, tuple) or len(native_pool) != 2
                or any(type(part) is not int for part in native_pool)):
            return None
        evidence = observe(native_pool, stream_id, device=device)
        if evidence is not None:
            if (not isinstance(evidence, GraphPoolMemorySnapshot)
                    or evidence.pool_id != native_pool or evidence.stream_id != stream_id):
                raise LiveModuleError("invalid-graph-pool-memory-evidence",
                                      "pool observer returned evidence for another resource")
            self._graph_pools[(evidence.device_index, evidence.pool_id)] = evidence
        return evidence

    def admit_graph_capture(
        self,
        graph_key: tuple[str, str],
        *,
        warmup_graph_key: tuple[str, str],
        memory_graph_key: tuple[str, str],
        pool: GraphPoolMemorySnapshot | None = None,
    ) -> None:
        """Reject when the warmup-derived estimate exceeds current free memory.

        This is a preflight heuristic, not a lower/upper bound or a memory seal:
        eager and capture can differ, reuse can fragment, and first-use native
        allocations can be absent from all preceding observations.
        """

        warmup_peak = max(
            (
                phase.peak_allocated_delta_bytes or 0
                for phase in self._phases
                if phase.phase == "graph.warmup" and phase.graph_key == warmup_graph_key
            ),
            default=0,
        )
        capture_classes = {
            admission.graph_key: admission.memory_graph_key
            for admission in self._graph_admissions
            if admission.admitted
        }
        graph_exec_overhead = max(
            (
                max(
                    0,
                    phase.after.non_allocator_bytes - phase.before.non_allocator_bytes,
                )
                for phase in self._phases
                if phase.phase == "graph.capture"
                and phase.after is not None
                and capture_classes.get(phase.graph_key) == memory_graph_key
            ),
            default=0,
        )
        snapshot = self._observer.snapshot()
        shared_pool_capacity = 0 if pool is None else pool.reusable_bytes
        incremental_allocator = max(0, warmup_peak - shared_pool_capacity)
        admission = GraphMemoryAdmission(
            graph_key=graph_key,
            warmup_graph_key=warmup_graph_key,
            memory_graph_key=memory_graph_key,
            warmup_peak_bytes=warmup_peak,
            shared_pool_capacity_bytes=shared_pool_capacity,
            incremental_allocator_bytes=incremental_allocator,
            observed_graph_exec_overhead_bytes=graph_exec_overhead,
            required_free_bytes=incremental_allocator + graph_exec_overhead,
            available_free_bytes=snapshot.free_bytes,
            pool=pool,
        )
        self._graph_admissions.append(admission)
        if admission.admitted:
            return
        error = LiveModuleError(
            "insufficient-graph-capture-memory",
            "estimated graph warmup and observed device overhead exceed free device memory",
            graph_key=graph_key,
            warmup_peak_bytes=warmup_peak,
            shared_pool_capacity_bytes=shared_pool_capacity,
            incremental_allocator_bytes=incremental_allocator,
            observed_graph_exec_overhead_bytes=graph_exec_overhead,
            required_free_bytes=admission.required_free_bytes,
            available_free_bytes=admission.available_free_bytes,
            deficit_bytes=admission.deficit_bytes,
        )
        if self._failure is None:
            self._failure = MemoryProfileFailure(
                phase="graph.admission",
                graph_key=graph_key,
                error_type=type(error).__qualname__,
                error_message=str(error),
                snapshot=snapshot,
            )
        raise error

    def finish(self, error: BaseException | None = None) -> LiveMemoryProfile:
        """Close the receipt after READY publication or failure cleanup."""

        if self._active is not None:
            if error is None:
                raise AssertionError("memory profile finished inside an active phase")
            self._fail_active(error)
        if error is not None and self._failure is None:
            snapshot = self._safe_snapshot()
            self._failure = MemoryProfileFailure(
                phase="activation",
                graph_key=None,
                error_type=type(error).__qualname__,
                error_message=str(error),
                snapshot=snapshot,
            )
        self._final = self._safe_snapshot()
        return self.receipt()

    def receipt(self) -> LiveMemoryProfile:
        return LiveMemoryProfile(
            model_storage_bytes=self._model_storage_bytes,
            initial=self._initial,
            final=self._final,
            state_domains=self._state_domains,
            phases=tuple(self._phases),
            graph_admissions=tuple(self._graph_admissions),
            failure=self._failure,
            graph_pools=tuple(self._graph_pools.values()),
        )

    def _fail_active(self, error: BaseException) -> None:
        active = self._active
        if active is None:
            return
        snapshot = self._safe_snapshot()
        self._phases.append(
            MemoryPhaseProfile(
                active.phase,
                active.graph_key,
                active.before,
                snapshot,
            )
        )
        if self._failure is None:
            self._failure = MemoryProfileFailure(
                phase=active.phase,
                graph_key=active.graph_key,
                error_type=type(error).__qualname__,
                error_message=str(error),
                snapshot=snapshot,
            )
        self._active = None

    def _safe_snapshot(self) -> DeviceMemorySnapshot | None:
        try:
            return self._observer.snapshot()
        except Exception:  # noqa: BLE001 - never mask the activation failure
            return None


def _unique_module_storage_bytes(root: LiveModule) -> int:
    """Count Parameters and buffers once by exact physical storage identity."""

    storages: dict[tuple[str, int | None, int, int], int] = {}
    for tensor in (*root.parameters(), *root.buffers()):
        if tensor.device.type == "meta":
            continue
        storage = tensor.untyped_storage()
        size = storage.nbytes()
        key = (tensor.device.type, tensor.device.index, storage.data_ptr(), size)
        storages.setdefault(key, size)
    return sum(storages.values())


__all__ = (
    "DeviceMemoryObserver",
    "DeviceMemorySnapshot",
    "GraphMemoryAdmission",
    "LiveMemoryProfile",
    "MemoryPhaseProfile",
    "MemoryProfileFailure",
    "StateDomainMemoryProfile",
    "StateLaneMemoryProfile",
    "TorchDeviceMemoryObserver",
)
