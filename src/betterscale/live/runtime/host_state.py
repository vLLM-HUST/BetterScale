# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-scoped host residency for address-stable LiveModule State."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from threading import Lock

import torch

from betterscale.live.core.error import LiveModuleError
from betterscale.live.core.state_tensor import StateDomain, StateTensor


class HostStateError(LiveModuleError):
    """A host State residency or transfer invariant failed."""


@dataclass(frozen=True, slots=True)
class HostStateKey:
    """One request/session incarnation stored outside the hot arena."""

    request_id: str
    generation: int

    def __post_init__(self) -> None:
        if (
            not self.request_id
            or type(self.generation) is not int
            or self.generation <= 0
        ):
            raise ValueError(
                "host State key needs a request id and positive generation"
            )


@dataclass(frozen=True, slots=True)
class HostStateDomainSelection:
    """Logical rows from one exact State addressing domain."""

    domain: StateDomain | None
    block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.block_ids:
            raise ValueError("host State selection needs at least one logical block")
        if any(type(value) is not int or value < 0 for value in self.block_ids):
            raise ValueError("host State logical blocks must be nonnegative integers")
        if len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError("host State selection cannot repeat a logical block")


@dataclass(frozen=True, slots=True)
class HostStateSelection:
    """Complete per-domain physical placement of one logical session State."""

    domains: tuple[HostStateDomainSelection, ...]

    def __post_init__(self) -> None:
        if not self.domains:
            raise ValueError("host State selection needs at least one domain")
        identities = [id(item.domain) for item in self.domains]
        if len(set(identities)) != len(identities):
            raise ValueError("host State selection repeats one domain")

    def blocks_for(self, state: StateTensor) -> tuple[int, ...] | None:
        for selection in self.domains:
            if selection.domain is state.domain:
                return selection.block_ids
        return None


@dataclass(frozen=True, slots=True)
class _LanePayload:
    lane_id: str
    dtype: torch.dtype
    block_shape: tuple[int, ...]
    physical_blocks_per_logical_block: int
    logical_block_count: int
    tensor: torch.Tensor

    @property
    def byte_length(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()


@dataclass(slots=True)
class _HostSnapshot:
    key: HostStateKey
    payloads: dict[str, _LanePayload]
    byte_length: int
    pins: int = 0


class HostStateTransferHandle:
    """Nonblocking transfer whose result publishes or releases exactly once."""

    __slots__ = (
        "_backend",
        "_byte_length",
        "_event",
        "_finalized",
        "_key",
        "_payloads",
        "_restore",
    )

    def __init__(
        self,
        backend: TorchHostStateBackend,
        *,
        key: HostStateKey,
        payloads: dict[str, _LanePayload],
        byte_length: int,
        event: torch.cuda.Event | None,
        restore: bool,
    ) -> None:
        self._backend = backend
        self._key = key
        self._payloads = payloads
        self._byte_length = byte_length
        self._event = event
        self._restore = restore
        self._finalized = False

    @property
    def key(self) -> HostStateKey:
        return self._key

    @property
    def byte_length(self) -> int:
        return self._byte_length

    def wait_on(self, stream: torch.cuda.Stream) -> None:
        """Make one caller-owned CUDA stream wait for this transfer.

        Waiting publishes only a device ordering edge.  The caller must still
        observe ``done()`` or ``result()`` before releasing either side of the
        transaction.
        """

        if not isinstance(stream, torch.cuda.Stream):
            raise TypeError("host State transfer wait requires a CUDA stream")
        if self._event is not None:
            stream.wait_event(self._event)

    def done(self) -> bool:
        if self._finalized:
            return True
        if self._event is not None and not self._event.query():
            return False
        self._finalize()
        return True

    def result(self) -> HostStateKey:
        if not self._finalized:
            if self._event is not None:
                self._event.synchronize()
            self._finalize()
        return self._key

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._backend._finish_transfer(
            self._key,
            self._payloads,
            self._byte_length,
            restore=self._restore,
        )
        self._finalized = True


class TorchHostStateBackend:
    """Pinned CPU payload pool with explicit asynchronous Torch copies.

    The backend never chooses migration policy.  Callers retain the source
    device lease until an OFFLOAD handle is terminal and keep a restored
    destination unpublished until its LOAD handle is terminal or its compute
    stream has waited on the caller-selected copy stream.
    """

    def __init__(self, *, memory_budget_bytes: int) -> None:
        if type(memory_budget_bytes) is not int or memory_budget_bytes <= 0:
            raise ValueError("host State memory budget must be positive")
        self._memory_budget_bytes = memory_budget_bytes
        self._committed_bytes = 0
        self._snapshots: dict[HostStateKey, _HostSnapshot] = {}
        self._inflight: set[HostStateKey] = set()
        self._lock = Lock()

    @property
    def memory_budget_bytes(self) -> int:
        return self._memory_budget_bytes

    @property
    def committed_bytes(self) -> int:
        return self._committed_bytes

    @property
    def resident_keys(self) -> frozenset[HostStateKey]:
        return frozenset(self._snapshots)

    def offload(
        self,
        states: Iterable[tuple[str, StateTensor]],
        key: HostStateKey,
        selection: HostStateSelection,
        *,
        stream: object | None,
    ) -> HostStateTransferHandle:
        lanes = self._select_lanes(states, selection)
        payloads: dict[str, _LanePayload] = {}
        byte_length = 0
        for lane_id, state, block_ids in lanes:
            shape = (
                len(block_ids) * state.physical_blocks_per_logical_block,
                *state.block_shape,
            )
            host = torch.empty(
                shape,
                dtype=state.storage_dtype,
                device="cpu",
                pin_memory=state.tensor.device.type == "cuda",
            )
            payload = _LanePayload(
                lane_id=lane_id,
                dtype=state.storage_dtype,
                block_shape=state.block_shape,
                physical_blocks_per_logical_block=(
                    state.physical_blocks_per_logical_block
                ),
                logical_block_count=len(block_ids),
                tensor=host,
            )
            payloads[lane_id] = payload
            byte_length += payload.byte_length
        with self._lock:
            if key in self._snapshots or key in self._inflight:
                raise HostStateError(
                    "duplicate-host-state-key",
                    "one session generation already has host State",
                    request_id=key.request_id,
                    generation=key.generation,
                )
            if self._committed_bytes + byte_length > self._memory_budget_bytes:
                raise HostStateError(
                    "insufficient-host-state-memory",
                    "host State payload exceeds the admitted CPU budget",
                    requested_bytes=byte_length,
                    committed_bytes=self._committed_bytes,
                    memory_budget_bytes=self._memory_budget_bytes,
                )
            self._inflight.add(key)
        try:
            event = self._copy_lanes(
                lanes,
                payloads,
                to_host=True,
                stream=stream,
            )
        except BaseException:
            with self._lock:
                self._inflight.remove(key)
            raise
        return HostStateTransferHandle(
            self,
            key=key,
            payloads=payloads,
            byte_length=byte_length,
            event=event,
            restore=False,
        )

    def restore(
        self,
        states: Iterable[tuple[str, StateTensor]],
        key: HostStateKey,
        selection: HostStateSelection,
        *,
        stream: object | None,
    ) -> HostStateTransferHandle:
        lanes = self._select_lanes(states, selection)
        with self._lock:
            snapshot = self._snapshots.get(key)
            if snapshot is None:
                raise HostStateError(
                    "missing-host-state",
                    "no host State belongs to this session generation",
                    request_id=key.request_id,
                    generation=key.generation,
                )
            if key in self._inflight:
                raise HostStateError(
                    "host-state-transfer-inflight",
                    "one session generation already has a State transfer",
                    request_id=key.request_id,
                    generation=key.generation,
                )
            self._validate_payloads(lanes, snapshot.payloads)
            snapshot.pins += 1
            self._inflight.add(key)
        try:
            event = self._copy_lanes(
                lanes,
                snapshot.payloads,
                to_host=False,
                stream=stream,
            )
        except BaseException:
            with self._lock:
                snapshot.pins -= 1
                self._inflight.remove(key)
            raise
        return HostStateTransferHandle(
            self,
            key=key,
            payloads=snapshot.payloads,
            byte_length=snapshot.byte_length,
            event=event,
            restore=True,
        )

    def release(self, key: HostStateKey) -> int:
        with self._lock:
            if key in self._inflight:
                raise HostStateError(
                    "host-state-transfer-inflight",
                    "host State cannot be released during transfer",
                    request_id=key.request_id,
                    generation=key.generation,
                )
            try:
                snapshot = self._snapshots.pop(key)
            except KeyError as error:
                raise HostStateError(
                    "missing-host-state",
                    "no host State belongs to this session generation",
                    request_id=key.request_id,
                    generation=key.generation,
                ) from error
            if snapshot.pins:
                raise AssertionError("unpinned host State snapshot was expected")
            self._committed_bytes -= snapshot.byte_length
            return snapshot.byte_length

    def _finish_transfer(
        self,
        key: HostStateKey,
        payloads: dict[str, _LanePayload],
        byte_length: int,
        *,
        restore: bool,
    ) -> None:
        with self._lock:
            if key not in self._inflight:
                raise HostStateError(
                    "stale-host-state-completion",
                    "host State transfer completed twice or under a stale key",
                    request_id=key.request_id,
                    generation=key.generation,
                )
            self._inflight.remove(key)
            if restore:
                snapshot = self._snapshots[key]
                if snapshot.payloads is not payloads or snapshot.pins <= 0:
                    raise AssertionError("restore lost its pinned host snapshot")
                snapshot.pins -= 1
            else:
                self._snapshots[key] = _HostSnapshot(key, payloads, byte_length)
                self._committed_bytes += byte_length

    @staticmethod
    def _select_lanes(
        states: Iterable[tuple[str, StateTensor]],
        selection: HostStateSelection,
    ) -> tuple[tuple[str, StateTensor, tuple[int, ...]], ...]:
        lanes: list[tuple[str, StateTensor, tuple[int, ...]]] = []
        seen_domains: set[int] = set()
        for lane_id, state in states:
            blocks = selection.blocks_for(state)
            if blocks is None:
                continue
            if any(block >= state.num_blocks for block in blocks):
                raise HostStateError(
                    "host-state-block-out-of-range",
                    "host State selection exceeds one lane's active capacity",
                    lane_id=lane_id,
                    state_capacity=state.num_blocks,
                )
            lanes.append((lane_id, state, blocks))
            seen_domains.add(id(state.domain))
        expected_domains = {id(item.domain) for item in selection.domains}
        if seen_domains != expected_domains:
            raise HostStateError(
                "empty-host-state-domain",
                "every selected State domain must contribute at least one lane",
            )
        if not lanes:
            raise HostStateError(
                "empty-host-state-selection",
                "host State transfer selected no StateTensor lanes",
            )
        return tuple(lanes)

    @staticmethod
    def _validate_payloads(
        lanes: Sequence[tuple[str, StateTensor, tuple[int, ...]]],
        payloads: Mapping[str, _LanePayload],
    ) -> None:
        if set(payloads) != {lane_id for lane_id, _, _ in lanes}:
            raise HostStateError(
                "incompatible-host-state-closure",
                "restore lane closure differs from the stored State closure",
            )
        for lane_id, state, block_ids in lanes:
            payload = payloads[lane_id]
            if (
                payload.dtype is not state.storage_dtype
                or payload.block_shape != state.block_shape
                or payload.physical_blocks_per_logical_block
                != state.physical_blocks_per_logical_block
                or payload.logical_block_count != len(block_ids)
            ):
                raise HostStateError(
                    "incompatible-host-state-format",
                    "restore destination differs from the stored lane format",
                    lane_id=lane_id,
                )

    @classmethod
    def _copy_lanes(
        cls,
        lanes: Sequence[tuple[str, StateTensor, tuple[int, ...]]],
        payloads: Mapping[str, _LanePayload],
        *,
        to_host: bool,
        stream: object | None,
    ) -> torch.cuda.Event | None:
        devices = {state.tensor.device.type for _, state, _ in lanes}
        if len(devices) != 1:
            raise HostStateError(
                "mixed-host-state-devices",
                "one host State transaction requires one device kind",
            )
        device_type = next(iter(devices))
        if device_type == "cuda":
            if not isinstance(stream, torch.cuda.Stream):
                raise HostStateError(
                    "missing-host-state-stream",
                    "CUDA host State transfer requires a caller-owned CUDA stream",
                )
            with torch.cuda.stream(stream):
                cls._enqueue_copies(lanes, payloads, to_host=to_host)
                event = torch.cuda.Event()
                event.record(stream)
            return event
        if stream is not None:
            raise HostStateError(
                "unexpected-host-state-stream",
                "CPU State transfer does not consume a device stream",
            )
        cls._enqueue_copies(lanes, payloads, to_host=to_host)
        return None

    @staticmethod
    def _enqueue_copies(
        lanes: Sequence[tuple[str, StateTensor, tuple[int, ...]]],
        payloads: Mapping[str, _LanePayload],
        *,
        to_host: bool,
    ) -> None:
        for lane_id, state, block_ids in lanes:
            payload = payloads[lane_id]
            span = state.physical_blocks_per_logical_block
            offset = 0
            run_start = block_ids[0]
            run_length = 1
            runs: list[tuple[int, int]] = []
            for previous, block_id in pairwise(block_ids):
                if block_id == previous + 1:
                    run_length += 1
                else:
                    runs.append((run_start, run_length))
                    run_start = block_id
                    run_length = 1
            runs.append((run_start, run_length))
            for first_block, logical_count in runs:
                physical_count = logical_count * span
                first = state.leading_physical_blocks + first_block * span
                device = state.tensor[first : first + physical_count]
                host = payload.tensor[offset : offset + physical_count]
                if to_host:
                    host.copy_(device, non_blocking=device.device.type == "cuda")
                else:
                    device.copy_(host, non_blocking=device.device.type == "cuda")
                offset += physical_count


__all__ = (
    "HostStateDomainSelection",
    "HostStateError",
    "HostStateKey",
    "HostStateSelection",
    "HostStateTransferHandle",
    "TorchHostStateBackend",
)
