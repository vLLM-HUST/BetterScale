# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""One module-owned graph declaration and its active physical execution."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from betterscale.live.core.error import LiveModuleError
from betterscale.live.runtime.meta_tensor import MetaTensorRealization


@runtime_checkable
class GraphIngressProjection(Protocol):
    """Project one live call onto a graph's stable ingress plane."""

    def project_call(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[tuple[object, ...], dict[str, object]]: ...


@runtime_checkable
class CapturedGraphExecution(Protocol):
    """One completely captured physical graph generation."""

    @property
    def metadata(self) -> MetaTensorRealization: ...

    @property
    def ingress_projection(self) -> GraphIngressProjection | None: ...

    @property
    def shadow_device_types(self) -> tuple[str, ...]: ...

    def stream_scope(self, stream: object) -> AbstractContextManager[object]: ...

    def publish_ingress(self) -> None: ...

    def replay(self, *, stream: object) -> None: ...

    def close(self) -> None: ...


class LiveGraph:
    """Retain one semantic graph while physical generations come and go."""

    def __init__(
        self,
        *,
        entry: Callable[..., object],
        schema: object | None = None,
        allow_forward_shadow: bool = False,
    ) -> None:
        if not callable(entry):
            raise LiveModuleError(
                "invalid-live-graph-entry",
                "a LiveGraph entry must be callable",
                entry_type=type(entry).__qualname__,
            )
        if not isinstance(allow_forward_shadow, bool):
            raise TypeError("allow_forward_shadow must be a bool")
        self._allow_forward_shadow = allow_forward_shadow
        self._entry = entry
        self._schema = schema
        self._graph_key: Hashable | None = None
        self._execution: CapturedGraphExecution | None = None

    @property
    def allow_forward_shadow(self) -> bool:
        """Explicit compatibility permission; never inferred from metadata."""

        return self._allow_forward_shadow

    @property
    def entry(self) -> Callable[..., object]:
        return self._entry

    @property
    def schema(self) -> object | None:
        return self._schema

    @property
    def prepared(self) -> bool:
        """Return whether this graph owns one READY physical generation."""

        return self._execution is not None

    @property
    def captured(self) -> bool:
        """Return whether this graph owns one READY captured execution."""

        return self._execution is not None

    @property
    def graph_key(self) -> Hashable | None:
        return self._graph_key

    @property
    def metadata(self) -> MetaTensorRealization | None:
        execution = self._execution
        return None if execution is None else execution.metadata

    @property
    def ingress_projection(self) -> GraphIngressProjection | None:
        execution = self._execution
        return None if execution is None else execution.ingress_projection

    @property
    def shadow_device_types(self) -> tuple[str, ...]:
        execution = self._execution
        return () if execution is None else execution.shadow_device_types

    def _bind_execution(
        self,
        *,
        graph_key: Hashable,
        execution: CapturedGraphExecution,
    ) -> None:
        """Atomically publish one already captured physical generation."""

        if self._execution is not None:
            raise LiveModuleError(
                "live-graph-already-prepared",
                "a LiveGraph already belongs to a physical generation",
            )
        try:
            hash(graph_key)
        except Exception as exc:
            raise LiveModuleError(
                "invalid-graph-key",
                "a captured LiveGraph key must be hashable",
                graph_key_type=type(graph_key).__qualname__,
            ) from exc
        if not isinstance(execution, CapturedGraphExecution):
            raise LiveModuleError(
                "invalid-captured-graph-execution",
                "a LiveGraph requires one typed captured execution",
                execution_type=type(execution).__qualname__,
            )
        metadata = execution.metadata
        if not isinstance(metadata, MetaTensorRealization):
            raise LiveModuleError(
                "invalid-captured-graph-metadata",
                "a captured execution requires one MetaTensorRealization",
                metadata_type=type(metadata).__qualname__,
            )
        if metadata.graph_key != graph_key:
            raise LiveModuleError(
                "metadata-realization-key-mismatch",
                "captured graph and metadata keys must agree",
                graph_key=graph_key,
                metadata_graph_key=metadata.graph_key,
            )
        if metadata.owner is not self:
            raise LiveModuleError(
                "metadata-realization-owner-mismatch",
                "captured metadata must be owned by its exact LiveGraph",
                graph_key=graph_key,
                metadata_owner_type=type(metadata.owner).__qualname__,
            )
        if not metadata.sealed or metadata.closed:
            raise LiveModuleError(
                "unready-captured-graph-execution",
                "a LiveGraph publishes only sealed live metadata executions",
                graph_key=graph_key,
                metadata_sealed=metadata.sealed,
                metadata_closed=metadata.closed,
            )
        projection = execution.ingress_projection
        if projection is not None and not isinstance(
            projection,
            GraphIngressProjection,
        ):
            raise LiveModuleError(
                "invalid-graph-ingress-projection",
                "a captured execution ingress must implement project_call",
                projection_type=type(projection).__qualname__,
            )
        devices = execution.shadow_device_types
        if (
            not isinstance(devices, tuple)
            or not devices
            or any(
                not isinstance(device_type, str) or not device_type.strip()
                for device_type in devices
            )
        ):
            raise LiveModuleError(
                "invalid-shadow-device",
                "captured execution shadow devices must be nonempty names",
                shadow_device_types=devices,
            )
        self._graph_key = graph_key
        self._execution = execution

    def _stream_scope(self, stream: object) -> AbstractContextManager[object]:
        execution = self._execution
        if execution is None:
            raise LiveModuleError(
                "unprepared-live-graph",
                "a LiveGraph must be prepared before stream selection",
            )
        return execution.stream_scope(stream)

    def _publish_ingress(self) -> None:
        execution = self._execution
        if execution is None:
            raise LiveModuleError(
                "unprepared-live-graph",
                "a LiveGraph must be prepared before ingress publication",
            )
        execution.publish_ingress()

    def _execute(self, *, stream: object) -> None:
        execution = self._execution
        if execution is None:
            raise LiveModuleError(
                "unprepared-live-graph",
                "a LiveGraph must be prepared before execution",
            )
        execution.replay(stream=stream)

    def _release_before_state(self) -> None:
        """Release physical graph resources before State unbinds."""

        execution = self._execution
        if execution is None:
            return
        try:
            execution.close()
        finally:
            self._execution = None
            self._graph_key = None


__all__ = (
    "CapturedGraphExecution",
    "GraphIngressProjection",
    "LiveGraph",
)
