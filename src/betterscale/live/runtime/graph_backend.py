# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Common synchronous graph build and stream-qualified replay lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from betterscale.live.core.error import LiveModuleError
from betterscale.live.core.meta_tensor import meta_tensor_scope
from betterscale.live.core.state_tensor import StateGenerationSnapshot
from betterscale.live.runtime.ingress import GraphCallSchema, TensorTreeIngress
from betterscale.live.runtime.invocation import LiveInvocationContext
from betterscale.live.runtime.live_graph import (
    CapturedGraphExecution,
    GraphIngressProjection,
    LiveGraph,
)
from betterscale.live.runtime.meta_tensor import MetaTensorRealization
from betterscale.live.runtime.phase import LivePhase, live_phase_scope

if TYPE_CHECKING:
    from betterscale.live.runtime.memory import LiveMemoryProfileSession


@runtime_checkable
class CapturedGraphSchema(Protocol):
    """Supply one CPU-safe call shape to a physical graph builder."""

    @property
    def call_schema(self) -> GraphCallSchema: ...


@runtime_checkable
class CapturedGraphStateSchema(Protocol):
    """Declare logical State changed by a graph's fixed build exemplar."""

    @property
    def capture_state_blocks(self) -> Mapping[object | None, Sequence[int]]: ...


@runtime_checkable
class CapturedGraphWarmupSchema(Protocol):
    """Identify graph entries with one equivalent physical warmup program."""

    @property
    def graph_warmup_key(self) -> object: ...


@runtime_checkable
class CapturedGraphMetadataSchema(Protocol):
    """Identify graph entries with one equivalent metadata construction program."""

    @property
    def graph_metadata_key(self) -> object: ...


@runtime_checkable
class CapturedGraphMemorySchema(Protocol):
    """Identify captures with one conservative executable-memory class."""

    @property
    def graph_memory_key(self) -> object: ...


@runtime_checkable
class CapturedGraphPoolSchema(Protocol):
    """Opt into one serial capture pool within a physical generation.

    Members must replay on one stream. Cross-call values must live in retained State/Meta/ingress tensors, never freed scratch.
    This is a lifetime contract, not the graph_memory_key estimation class.
    """

    @property
    def graph_pool_key(self) -> object | None: ...


@dataclass(slots=True)
class _SerialCapturePool:
    generation: object | None
    owner: object
    stream: object
    native_pool: object
    users: int = 0
    submission_stream: object | None = None

    def require_stream(self, stream: object) -> None:
        if self.submission_stream is None:
            self.submission_stream = stream
        elif stream != self.submission_stream:
            raise LiveModuleError(
                "serial-graph-pool-stream-mismatch",
                "graphs sharing scratch must replay on one stream",
            )


class _PhysicalGraph(Protocol):
    def replay(self) -> None: ...

    def reset(self) -> None: ...


@dataclass(slots=True)
class CapturePoolHandoff:
    """Keep one retired calibration graph solely as a native pool owner.

    It cannot replay. Close after a successor capture owns the same pool, or
    when abandoning calibration. No execution metadata or ingress is retained.
    """

    backend: TorchDeviceGraphBackend
    owner: object
    source_generation: object | None
    stream: object
    native_pool: object
    _graph: _PhysicalGraph | None

    @property
    def closed(self) -> bool:
        return self._graph is None

    def close(self) -> None:
        graph = self._graph
        if graph is None:
            return
        self.backend._synchronize()
        graph.reset()
        self._graph = None


@dataclass(slots=True)
class CaptureInputHandoff:
    """Own fixed input buffers between two unpublished captures of one graph.

    The source must retire before the handoff is consumed. Closing an unused
    handoff returns ownership to a live source, or frees a retired source's
    buffers. Final capture consumes it, so there is never a second input copy.
    """

    _source: _TorchDeviceGraphState | None
    _ingress: TensorTreeIngress | None

    def take(self, backend: TorchDeviceGraphBackend, graph: LiveGraph,
             schema: GraphCallSchema) -> TensorTreeIngress:
        source, ingress = self._source, self._ingress
        if (source is None or ingress is None or not source.closed
                or source.backend is not backend or source.metadata.owner is not graph
                or ingress.schema is not schema):
            raise LiveModuleError("invalid-capture-input-handoff",
                                  "input reuse requires the same graph/schema and a retired source")
        self._source = self._ingress = None
        return ingress

    def close(self) -> None:
        source, ingress = self._source, self._ingress
        if source is not None and ingress is not None:
            if source.closed:
                ingress.release()
            else:
                source.inputs_retained = False
        self._source = self._ingress = None


@dataclass(slots=True)
class _TorchDeviceGraphState:
    """One completely captured torch device-graph execution."""

    backend: TorchDeviceGraphBackend
    metadata: MetaTensorRealization
    ingress_projection: GraphIngressProjection | None
    shadow_device_types: tuple[str, ...]
    graph: _PhysicalGraph | None
    capture_stream: object
    capture_pool: _SerialCapturePool | None = None
    closed: bool = False
    inputs_retained: bool = False

    def stream_scope(self, stream: object) -> AbstractContextManager[object]:
        if self.closed:
            raise LiveModuleError(
                f"closed-{self.backend.graph_kind}-backend",
                f"a closed {self.backend.graph_display_name} graph cannot select "
                "a submission stream",
            )
        return self.backend._stream_scope(stream)

    def publish_ingress(self) -> None:
        self.backend._publish_ingress(self)

    def replay(self, *, stream: object) -> None:
        self.backend._replay(self, stream=stream)

    def close(self) -> None:
        self.backend._close(self)


class TorchDeviceGraphBackend(ABC):
    """Build READY physical graph executions and own their device lifecycle."""

    graph_kind = "graph"
    graph_display_name = "device graph"
    externalize_graph_ingress_effects = False

    def __init__(
        self,
        *,
        device: torch.device | str,
        warmup_iterations: int = 1,
        capture_error_mode: str = "thread_local",
    ) -> None:
        self._device = torch.device(device)
        if (
            isinstance(warmup_iterations, bool)
            or not isinstance(warmup_iterations, int)
            or warmup_iterations < 0
        ):
            raise LiveModuleError(
                "invalid-warmup-iterations",
                f"{self.graph_display_name} backend warmup iterations must be "
                "nonnegative",
                warmup_iterations=warmup_iterations,
            )
        if capture_error_mode not in {"global", "thread_local", "relaxed"}:
            raise LiveModuleError(
                f"invalid-{self.graph_kind}-capture-error-mode",
                f"{self.graph_display_name} capture error mode is unsupported",
                capture_error_mode=capture_error_mode,
            )
        self._warmup_iterations = warmup_iterations
        self._capture_error_mode = capture_error_mode
        self._warmed_graph_keys: set[tuple[int, int]] = set()
        self._warmup_representatives: dict[
            tuple[int, int],
            tuple[str, str],
        ] = {}
        self._memory_representatives: dict[
            tuple[int, int],
            tuple[str, str],
        ] = {}
        self._metadata_programs: dict[
            tuple[int, int],
            tuple[object, object, MetaTensorRealization],
        ] = {}
        self._capture_pools: dict[tuple[int, int], _SerialCapturePool] = {}

    def warmup_graph(
        self,
        *,
        graph: LiveGraph,
        graph_key: tuple[str, str],
        generation_token: object | None = None,
        warmup_context: LiveInvocationContext | None = None,
        restore_point: StateGenerationSnapshot | None = None,
        memory_profile: LiveMemoryProfileSession | None = None,
    ) -> None:
        """Warm one physical program before any graph pool becomes resident.

        Portfolio activation calls this for every unique program before the
        first capture.  Besides keeping JIT and eager workspace out of later
        graph-pool headroom, the measured warmup becomes the Runtime's first
        capture-memory estimate.
        """

        if not self._warmup_iterations:
            return
        schema, call_schema = self._graph_schema(graph, graph_key=graph_key)
        self._validate_context(warmup_context, role="warmup")
        self._validate_restore_point(restore_point)
        warmup_key = self._warmup_key(
            schema,
            graph,
            generation_token=generation_token,
        )
        memory_key = self._memory_key(
            schema,
            graph,
            generation_token=generation_token,
        )
        self._memory_representatives.setdefault(memory_key, graph_key)
        if warmup_key in self._warmed_graph_keys:
            return
        ingress = TensorTreeIngress(call_schema, device=self._device)
        warmup_scope = (
            nullcontext()
            if memory_profile is None
            else memory_profile.phase("graph.warmup", graph_key=graph_key)
        )
        with torch.inference_mode(), warmup_scope:
            for iteration in range(self._warmup_iterations):
                ingress.stage_exemplar()
                try:
                    with (
                        live_phase_scope(LivePhase.WARMUP),
                        meta_tensor_scope(("warmup", graph_key, iteration)),
                    ):
                        self._invoke_entry(graph, ingress, warmup_context)
                finally:
                    self._finish_build_step(restore_point)
        self._warmed_graph_keys.add(warmup_key)
        self._warmup_representatives[warmup_key] = graph_key

    def capture_graph(
        self,
        *,
        graph: LiveGraph,
        graph_key: tuple[str, str],
        generation_token: object | None = None,
        warmup_context: LiveInvocationContext | None = None,
        capture_context: LiveInvocationContext | None = None,
        restore_point: StateGenerationSnapshot | None = None,
        memory_profile: LiveMemoryProfileSession | None = None,
        pool_handoff: CapturePoolHandoff | None = None,
        input_handoff: CaptureInputHandoff | None = None,
    ) -> CapturedGraphExecution:
        """Synchronously build one complete physical execution."""

        schema, call_schema = self._graph_schema(graph, graph_key=graph_key)
        self._validate_context(warmup_context, role="warmup")
        self._validate_context(capture_context, role="capture")
        self._validate_restore_point(restore_point)

        prepare_scope = (
            nullcontext()
            if memory_profile is None
            else memory_profile.phase("graph.prepare", graph_key=graph_key)
        )
        devices = (self._device.type,)
        warmup_key = self._warmup_key(
            schema,
            graph,
            generation_token=generation_token,
        )
        memory_key = self._memory_key(
            schema,
            graph,
            generation_token=generation_token,
        )
        self._memory_representatives.setdefault(memory_key, graph_key)
        metadata_owner = (
            schema.graph_metadata_key
            if isinstance(schema, CapturedGraphMetadataSchema)
            else None
        )
        metadata_key = (
            None
            if metadata_owner is None
            else (id(generation_token), id(metadata_owner))
        )
        metadata_template = None
        if metadata_key is not None:
            cached = self._metadata_programs.get(metadata_key)
            if (
                cached is not None
                and cached[0] is generation_token
                and cached[1] is metadata_owner
                and not cached[2].closed
                and not cached[2].requires_forward_replay
            ):
                metadata_template = cached[2]
        native_graph: _PhysicalGraph | None = None
        capture_pool: _SerialCapturePool | None = None
        ingress: TensorTreeIngress | None = None
        metadata: MetaTensorRealization | None = None
        try:
            with prepare_scope:
                if input_handoff is not None and not isinstance(input_handoff, CaptureInputHandoff):
                    raise LiveModuleError("invalid-capture-input-handoff", "capture input handoff is not typed")
                ingress = (TensorTreeIngress(call_schema, device=self._device)
                           if input_handoff is None else input_handoff.take(self, graph, call_schema))
                if input_handoff is not None:
                    # A State rebind may update host exemplar contents (for
                    # example owner offsets). Refresh, never replace, the IO.
                    with torch.inference_mode():
                        ingress.project_call(call_schema.args, call_schema.kwargs)
                metadata = MetaTensorRealization(
                    graph,
                    graph_key=graph_key,
                    generation=generation_token,
                )
            with torch.inference_mode():
                if (
                    self._warmup_iterations
                    and warmup_key not in self._warmed_graph_keys
                ):
                    self.warmup_graph(
                        graph=graph,
                        graph_key=graph_key,
                        generation_token=generation_token,
                        warmup_context=warmup_context,
                        restore_point=restore_point,
                        memory_profile=memory_profile,
                    )

                capture_scope = (
                    nullcontext()
                    if memory_profile is None
                    else memory_profile.phase(
                        "graph.capture",
                        graph_key=graph_key,
                    )
                )
                capture_pool = self._acquire_capture_pool(schema, generation_token, pool_handoff)
                pool_memory = None
                if memory_profile is not None and capture_pool is not None:
                    pool_memory = memory_profile.observe_graph_pool(
                        capture_pool.native_pool,
                        self._capture_stream_id(capture_pool.stream),
                        device=self._device,
                    )
                if memory_profile is not None:
                    memory_profile.admit_graph_capture(
                        graph_key,
                        warmup_graph_key=self._warmup_representatives.get(
                            warmup_key,
                            graph_key,
                        ),
                        memory_graph_key=self._memory_representatives[memory_key],
                        pool=pool_memory,
                    )
                with capture_scope:
                    ingress.stage_exemplar()
                    self._synchronize()
                    native_graph, capture_stream = self._capture_entry(
                        graph=graph,
                        ingress=ingress,
                        context=capture_context,
                        metadata=metadata,
                        metadata_template=metadata_template,
                        shadow_device_types=devices,
                        restore_point=restore_point,
                        capture_pool=capture_pool,
                    )
                if memory_profile is not None and capture_pool is not None:
                    memory_profile.observe_graph_pool(
                        capture_pool.native_pool,
                        self._capture_stream_id(capture_pool.stream),
                        device=self._device,
                    )
            if not metadata.sealed:
                raise AssertionError("successful graph capture left metadata unsealed")
            if (
                metadata_key is not None
                and metadata_template is None
                and not metadata.requires_forward_replay
            ):
                self._metadata_programs[metadata_key] = (
                    generation_token,
                    metadata_owner,
                    metadata,
                )
            return _TorchDeviceGraphState(
                backend=self,
                metadata=metadata,
                ingress_projection=ingress,
                shadow_device_types=devices,
                graph=native_graph,
                capture_stream=capture_stream,
                capture_pool=capture_pool,
            )
        except BaseException:
            try:
                if native_graph is not None:
                    native_graph.reset()
            finally:
                try:
                    if metadata is not None:
                        metadata.close()
                finally:
                    if ingress is not None:
                        ingress.release()
                    self._release_capture_pool(capture_pool)
            raise

    def _capture_stream_id(self, stream: object) -> int | None:
        """Return the port's native allocator stream identity, if supported."""
        return None

    def _acquire_capture_pool(
        self, schema: object, generation: object | None,
        handoff: CapturePoolHandoff | None = None,
    ) -> _SerialCapturePool | None:
        owner = (
            schema.graph_pool_key
            if isinstance(schema, CapturedGraphPoolSchema)
            else None
        )
        if handoff is not None and (
            handoff.backend is not self or handoff.closed or owner is not handoff.owner
            or generation is handoff.source_generation
        ):
            raise LiveModuleError("invalid-capture-pool-handoff",
                                  "recapture requires a live same-family handoff from another generation")
        if owner is None:
            return None
        key = (id(generation), id(owner))
        pool = self._capture_pools.get(key)
        if pool is None:
            stream = self._create_stream() if handoff is None else handoff.stream
            native_pool = self._create_graph_pool() if handoff is None else handoff.native_pool
            pool = _SerialCapturePool(generation, owner, stream, native_pool)
            self._capture_pools[key] = pool
        elif handoff is not None and (
            pool.native_pool != handoff.native_pool or pool.stream != handoff.stream
        ):
            raise LiveModuleError("conflicting-capture-pool-handoff",
                                  "successor generation already owns another pool")
        pool.users += 1
        return pool

    def _release_capture_pool(self, pool: _SerialCapturePool | None) -> None:
        if pool is None:
            return
        pool.users -= 1
        if not pool.users:
            del self._capture_pools[(id(pool.generation), id(pool.owner))]

    def _graph_schema(
        self,
        graph: LiveGraph,
        *,
        graph_key: tuple[str, str],
    ) -> tuple[object, GraphCallSchema]:
        schema = graph.schema
        if isinstance(schema, GraphCallSchema):
            return schema, schema
        if isinstance(schema, CapturedGraphSchema):
            return schema, schema.call_schema
        raise LiveModuleError(
            f"unsupported-{self.graph_kind}-schema",
            f"{self.graph_display_name} capture requires one typed graph schema",
            graph_name=graph_key[1],
            schema_type=type(schema).__qualname__,
        )

    @staticmethod
    def _warmup_key(
        schema: object,
        graph: LiveGraph,
        *,
        generation_token: object | None,
    ) -> tuple[int, int]:
        warmup_owner = (
            schema.graph_warmup_key
            if isinstance(schema, CapturedGraphWarmupSchema)
            else graph
        )
        return id(generation_token), id(warmup_owner)

    @staticmethod
    def _memory_key(
        schema: object,
        graph: LiveGraph,
        *,
        generation_token: object | None,
    ) -> tuple[int, int]:
        memory_owner = (
            schema.graph_memory_key
            if isinstance(schema, CapturedGraphMemorySchema)
            else graph
        )
        return id(generation_token), id(memory_owner)

    @staticmethod
    def _validate_restore_point(
        restore_point: StateGenerationSnapshot | None,
    ) -> None:
        if restore_point is not None and not isinstance(
            restore_point,
            StateGenerationSnapshot,
        ):
            raise LiveModuleError(
                "invalid-state-restore-point",
                "graph build requires one root-generation State restore point",
                restore_point_type=type(restore_point).__qualname__,
            )

    def _capture_entry(
        self,
        *,
        graph: LiveGraph,
        ingress: TensorTreeIngress,
        context: LiveInvocationContext | None,
        metadata: MetaTensorRealization,
        metadata_template: MetaTensorRealization | None,
        shadow_device_types: tuple[str, ...],
        restore_point: StateGenerationSnapshot | None,
        capture_pool: _SerialCapturePool | None = None,
    ) -> tuple[_PhysicalGraph, object]:
        native_graph = self._create_graph()
        capture_stream = (
            self._create_stream() if capture_pool is None else capture_pool.stream
        )
        native_pool = None if capture_pool is None else capture_pool.native_pool
        try:
            if self.externalize_graph_ingress_effects:
                try:
                    with metadata.capture_graph_ingress_effects(
                        shadow_device_types,
                        externalize=True,
                    ):
                        if metadata_template is None:
                            with metadata.capture():
                                self._invoke_entry(graph, ingress, context)
                        else:
                            metadata.capture_construction_program(
                                metadata_template,
                                context,
                            )
                finally:
                    self._finish_build_step(restore_point)
                try:
                    with (
                        self._capture_scope_with_pool(native_graph, capture_stream, native_pool),
                        metadata.capture_reuse(),
                    ):
                        self._invoke_entry(graph, ingress, context)
                finally:
                    self._finish_build_step(restore_point)
            else:
                try:
                    with (
                        self._capture_scope_with_pool(native_graph, capture_stream, native_pool),
                        metadata.capture_graph_ingress_effects(
                            shadow_device_types,
                        ),
                        metadata.capture(),
                    ):
                        self._invoke_entry(graph, ingress, context)
                finally:
                    self._finish_build_step(restore_point)
            return native_graph, capture_stream
        except BaseException:
            native_graph.reset()
            raise

    def _invoke_entry(
        self,
        graph: LiveGraph,
        ingress: TensorTreeIngress,
        context: LiveInvocationContext | None,
    ) -> object:
        scope = nullcontext() if context is None else context.activate()
        if not isinstance(scope, AbstractContextManager):
            raise LiveModuleError(
                "invalid-live-invocation-context-scope",
                "graph build context activation must return a context manager",
                context_type=type(context).__qualname__,
                scope_type=type(scope).__qualname__,
            )
        args, kwargs = ingress.stable_call
        with scope:
            return graph.entry(*args, **kwargs)

    def _finish_build_step(
        self,
        restore_point: StateGenerationSnapshot | None,
    ) -> None:
        self._synchronize()
        if restore_point is not None:
            restore_point.restore()
            self._synchronize()

    def _publish_ingress(
        self,
        state: _TorchDeviceGraphState,
    ) -> None:
        if state.closed:
            raise LiveModuleError(
                f"closed-{self.graph_kind}-backend",
                f"a closed {self.graph_display_name} graph cannot publish ingress",
            )
        effects = state.metadata.graph_ingress_effects
        if effects is not None:
            effects.publish_externalized()

    def _replay(
        self,
        state: _TorchDeviceGraphState,
        *,
        stream: object,
    ) -> None:
        if state.closed:
            raise LiveModuleError(
                f"closed-{self.graph_kind}-backend",
                f"a closed {self.graph_display_name} graph cannot replay",
            )
        if state.capture_pool is not None:
            state.capture_pool.require_stream(stream)
        with state.stream_scope(stream):
            assert state.graph is not None
            state.graph.replay()

    def retire_generation(self, generation_token: object) -> None:
        """Discard build census only after the root has retired its executions.

        Calibration retirement deliberately keeps this census for final capture.
        Root retirement must not: recycled Python identities could otherwise
        make a later generation skip its ordinary warmup.
        """
        generation = id(generation_token)
        if (any(key[0] == generation for key in self._capture_pools)
                or any(key[0] == generation for key in self._metadata_programs)):
            raise LiveModuleError("graph-generation-still-live",
                                  "close generation captures before retiring their build census")
        self._warmed_graph_keys.difference_update(
            tuple(key for key in self._warmed_graph_keys if key[0] == generation))
        for cache in (self._warmup_representatives, self._memory_representatives):
            for key in tuple(cache):
                if key[0] == generation:
                    del cache[key]

    def prime_unpublished_capture(
        self, state: CapturedGraphExecution, *, restore_point=None,
    ) -> None:
        """Synchronously exercise the capture exemplar without binding replay's stream.

        This is startup work, not a substitute for a recipe's full execution
        envelope. The production submission stream remains unclaimed.
        """
        if (not isinstance(state, _TorchDeviceGraphState) or state.backend is not self
                or state.closed or not isinstance(state.metadata.owner, LiveGraph)
                or state.metadata.owner.prepared):
            raise LiveModuleError("invalid-unpublished-capture-prime",
                                  "only this backend's unpublished capture can be primed")
        self._synchronize()
        try:
            with state.stream_scope(state.capture_stream):
                state.publish_ingress()
                state.graph.replay()
        finally:
            self._finish_build_step(restore_point)

    def retain_calibration_pool(self, state: CapturedGraphExecution) -> CapturePoolHandoff:
        """Retire an unpublished execution without freeing its native pool."""
        if (not isinstance(state, _TorchDeviceGraphState) or state.backend is not self
                or state.closed or state.capture_pool is None
                or not isinstance(state.metadata.owner, LiveGraph)
                or state.metadata.owner.prepared):
            raise LiveModuleError("invalid-calibration-pool-retirement",
                                  "only an unpublished live pooled capture can hand off its pool")
        self._synchronize()
        pool, graph = state.capture_pool, state.graph
        assert graph is not None
        try:
            self._retire_graph_metadata(state)
        except BaseException:
            graph.reset()
            raise
        return CapturePoolHandoff(self, pool.owner, pool.generation, pool.stream,
                                  pool.native_pool, graph)

    def validate_calibration_inputs(self, graph: LiveGraph, *, graph_key) -> None:
        """Reject unsafe exemplars before allocating calibration State or graphs."""
        _, call_schema = self._graph_schema(graph, graph_key=graph_key)
        if not call_schema.has_host_exemplar:
            raise LiveModuleError("invalid-calibration-input-exemplar",
                                  "calibration requires host-only non-opaque call exemplars",
                                  graph_name=graph_key)

    def retain_calibration_inputs(self, state: CapturedGraphExecution) -> CaptureInputHandoff:
        """Keep host-exemplared input destinations while calibration retires."""
        if (not isinstance(state, _TorchDeviceGraphState) or state.backend is not self
                or state.closed or state.inputs_retained
                or not isinstance(state.metadata.owner, LiveGraph)
                or state.metadata.owner.prepared
                or not isinstance(state.ingress_projection, TensorTreeIngress)
                or not state.ingress_projection.has_host_exemplar):
            raise LiveModuleError("invalid-calibration-input-retirement",
                                  "only unpublished host-exemplared capture inputs can be retained once")
        state.inputs_retained = True
        return CaptureInputHandoff(state, state.ingress_projection)

    def _close(self, state: _TorchDeviceGraphState) -> None:
        if state.closed:
            return
        try:
            self._synchronize()
            assert state.graph is not None
            state.graph.reset()
        finally:
            self._retire_graph_metadata(state)

    def _retire_graph_metadata(self, state: _TorchDeviceGraphState) -> None:
        stale_keys = tuple(
            key for key, (_, _, metadata) in self._metadata_programs.items()
            if metadata is state.metadata
        )
        for key in stale_keys:
            del self._metadata_programs[key]
        try:
            state.metadata.close()
        finally:
            state.closed = True
            self._release_capture_pool(state.capture_pool)
            state.capture_pool = None
            if not state.inputs_retained and isinstance(state.ingress_projection, TensorTreeIngress):
                state.ingress_projection.release()
            state.ingress_projection = None
            state.graph = None

    def _create_graph_pool(self) -> object:
        raise LiveModuleError(
            "unsupported-serial-graph-pool",
            "this graph backend does not implement serial capture pools",
        )

    def _capture_scope_with_pool(
        self, graph: _PhysicalGraph, stream: object, pool: object | None,
    ) -> AbstractContextManager[object]:
        if pool is not None:
            raise LiveModuleError(
                "unsupported-serial-graph-pool",
                "this graph backend cannot capture into a shared pool",
            )
        return self._capture_scope(graph, stream)

    def _validate_context(
        self,
        context: LiveInvocationContext | None,
        *,
        role: str,
    ) -> None:
        if context is not None and not isinstance(context, LiveInvocationContext):
            raise LiveModuleError(
                f"invalid-{self.graph_kind}-{role}-context",
                f"{self.graph_display_name} {role} requires one typed context",
                context_type=type(context).__qualname__,
            )

    @abstractmethod
    def _create_graph(self) -> _PhysicalGraph:
        """Create one architecture-native graph object."""

    @abstractmethod
    def _stream_scope(self, stream: object) -> AbstractContextManager[object]:
        """Enter the caller-selected native stream scope."""

    @abstractmethod
    def _create_stream(self) -> object:
        """Create one architecture-native capture stream."""

    @abstractmethod
    def _capture_scope(
        self,
        graph: _PhysicalGraph,
        stream: object,
    ) -> AbstractContextManager[object]:
        """Enter one architecture-native graph capture scope."""

    @abstractmethod
    def _synchronize(self) -> None:
        """Wait for prior work on the selected architecture device."""


__all__ = (
    "CaptureInputHandoff",
    "CapturePoolHandoff",
    "CapturedGraphMetadataSchema",
    "CapturedGraphPoolSchema",
    "CapturedGraphSchema",
    "CapturedGraphStateSchema",
    "CapturedGraphWarmupSchema",
    "TorchDeviceGraphBackend",
)
