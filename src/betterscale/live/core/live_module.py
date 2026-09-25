# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""The minimal composable Module root for the LiveModule prototype."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import nullcontext
from enum import Enum
from threading import RLock
from typing import TYPE_CHECKING

from torch import nn

from betterscale.live.core.error import LiveModuleError
from betterscale.live.runtime.live_graph import LiveGraph
from betterscale.live.runtime.phase import LivePhase, current_live_phase

if TYPE_CHECKING:
    import torch

    from betterscale.live.core.meta_tensor import MetaTensor
    from betterscale.live.core.state_tensor import SIMDStatePlan, StateDomain, StateTensor
    from betterscale.live.runtime.invocation import (
        LiveInvocation,
        LiveInvocationContext,
    )
    from betterscale.live.runtime.memory import LiveMemoryProfile
    from betterscale.live.runtime.pipeline import PipelineExecution


class LiveModulePhase(Enum):
    """Physical lifecycle phases exposed by ``LiveModule``."""

    CREATED = "created"
    READY = "ready"
    POISONED = "poisoned"
    CLOSING = "closing"
    CLOSED = "closed"


class LiveModule(nn.Module):
    """An ``nn.Module`` whose original forward may remain live around replay.

    Ordinary ``nn.Module`` child ownership and recursive ``forward()`` calls
    remain the composition authority. Persistent StateTensor and MetaTensor
    declarations register explicitly by local attribute name and expose
    recursive views over that same tree. Anonymous MetaTensor recipes remain
    local to one prepared graph generation.
    """

    def __init__(self) -> None:
        super().__init__()
        from betterscale.live.runtime.live_runtime import current_live_runtime

        object.__setattr__(self, "_live_declarations_sealed", False)
        # Persistent serving declarations use local named registries, just as
        # nn.Module locally owns parameters, buffers and child modules.
        # Recursive views are derived from the Module tree rather than cached.
        object.__setattr__(self, "_live_states", {})
        object.__setattr__(self, "_live_meta_tensors", {})
        object.__setattr__(self, "_live_graphs", {})
        object.__setattr__(self, "_live_runtime", current_live_runtime())
        object.__setattr__(self, "_live_phase", LiveModulePhase.CREATED)
        object.__setattr__(self, "_live_lock", RLock())
        object.__setattr__(self, "_live_generation_token", None)
        object.__setattr__(self, "_live_generation_modules", ())
        object.__setattr__(self, "_live_state_tensors", ())
        object.__setattr__(self, "_live_generation_state_domains", ())
        object.__setattr__(self, "_live_state_realization", None)
        object.__setattr__(self, "_live_generation_meta_tensors", ())
        object.__setattr__(self, "_live_state_plan", None)
        object.__setattr__(self, "_live_local_max_state_blocks", None)
        object.__setattr__(self, "_live_active_invocations", set())
        object.__setattr__(self, "_live_memory_profile", None)

    def _process_live_weights_after_loading(
        self,
        model_dtype: torch.dtype,
    ) -> None:
        """Apply one module-local layout transition after donor loading.

        The one-shot LiveModule loader invokes this fixed phase recursively
        after vLLM's ordinary quantization post-processing and before runtime
        activation.  Most modules retain the loaded representation unchanged;
        a module whose execution ABI needs another stable physical layout owns
        that transition here.
        """

        del model_dtype

    def _initialize_live_generation(self) -> None:
        """Initialize generation-local metadata after State binding.

        This fixed recursive phase runs after every StateTensor and module-held
        MetaTensor has its physical generation value and before any graph is
        warmed or captured. Modules may publish immutable descriptors derived
        from those stable addresses; ordinary request State remains the
        responsibility of admission and device programs.
        """

    def _rebind_live_state(self) -> None:
        """Reinitialize replaced calibration State without reallocating fixed resources.

        Automatic fitting requires modules with custom generation initialization
        to override this hook explicitly. Captured addresses are still unpublished;
        the final generation cannot rebind State after READY.
        """

    def _finalize_live_generation_warmup(self) -> None:
        """Seal generation resources after portfolio warmup and before capture.

        Architecture or model modules may override this hook when a complete
        finite eager census must freeze an address-stable resource before any
        graph is captured. The default has no generation-local work.
        """

    def _release_live_generation(self) -> None:
        """Release module-local resources owned by the active generation.

        This hook runs in reverse module order after captured graphs stop using
        their addresses and before StateTensor storage is unbound.  Overrides
        must be idempotent: a failed close leaves the root poisoned so the
        release may be retried.
        """

    @property
    def phase(self) -> LiveModulePhase:
        """Return this module's physical lifecycle phase."""

        return self._live_phase

    @property
    def active_invocation_count(self) -> int:
        with self._live_lock:
            return len(self._live_active_invocations)

    @property
    def state_plan(self) -> SIMDStatePlan | None:
        """Return the root's unique elastic State-domain plan, if present."""

        return self._live_state_plan

    @property
    def local_max_state_blocks(self) -> int | None:
        """Return the unique elastic domain's pre-admission block capacity."""

        return self._live_local_max_state_blocks

    @property
    def live_device(self) -> torch.device | str | None:
        """Return the process device captured during module construction."""

        return self._live_runtime.device

    @property
    def memory_profile(self) -> LiveMemoryProfile | None:
        """Return the latest activation memory receipt, when observation is enabled."""

        return self._live_memory_profile

    def named_live_modules(self) -> Iterator[tuple[str, LiveModule]]:
        """Yield every LiveModule in this subtree with its relative path."""

        for path, module in self.named_modules():
            if isinstance(module, LiveModule):
                yield path, module

    def named_states(
        self,
        *,
        recurse: bool = True,
    ) -> Iterator[tuple[str, StateTensor]]:
        """Yield local or recursive StateTensor declarations by attribute path."""

        modules = self.named_live_modules() if recurse else iter((("", self),))
        for module_path, module in modules:
            for state_name, state in module._live_states.items():
                qualified_name = (
                    f"{module_path}.{state_name}" if module_path else state_name
                )
                yield qualified_name, state

    def named_graphs(
        self,
        *,
        recurse: bool = True,
    ) -> Iterator[tuple[str, LiveGraph]]:
        """Yield local or recursive graph declarations by declaration path."""

        modules = self.named_live_modules() if recurse else iter((("", self),))
        for module_path, module in modules:
            for graph_name, graph in module._live_graphs.items():
                qualified_name = (
                    f"{module_path}.{graph_name}" if module_path else graph_name
                )
                yield qualified_name, graph

    def named_meta_tensors(
        self,
        *,
        recurse: bool = True,
    ) -> Iterator[tuple[str, MetaTensor[torch.Tensor]]]:
        """Yield local or recursive persistent metadata declarations."""

        modules = self.named_live_modules() if recurse else iter((("", self),))
        for module_path, module in modules:
            for member_name, meta_tensor in module._live_meta_tensors.items():
                qualified_name = (
                    f"{module_path}.{member_name}" if module_path else member_name
                )
                yield qualified_name, meta_tensor

    def activate(self, *, state_memory_floor_bytes: int | None = None,
                 pipeline: PipelineExecution | None = None) -> LiveModule:
        """Seal owned State/graphs and atomically publish this subtree READY."""

        from betterscale.live.core.state_tensor import (
            ElasticStateCapacity,
            SIMDStateLane,
            SIMDStateSchema,
            StateGenerationSnapshot,
            compile_simd_state_schema,
        )
        from betterscale.live.runtime.graph_backend import CapturedGraphStateSchema, TorchDeviceGraphBackend
        from betterscale.live.runtime.ingress import GraphCallSchema
        from betterscale.live.runtime.invocation import LiveGraphContextBackend
        from betterscale.live.runtime.live_graph import CapturedGraphExecution
        from betterscale.live.runtime.memory import LiveMemoryProfileSession, TorchDeviceMemoryObserver
        from betterscale.live.runtime.state_backend import (
            StateBackend,
            TorchStateBackend,
            StateGenerationRealization,
            validate_state_generation_realization,
        )

        if pipeline is not None and state_memory_floor_bytes is not None:
            raise ValueError("pipeline State fitting is not yet qualified; use explicit State capacity")
        runtime = self._live_runtime
        with self._live_lock:
            if self._live_phase not in (
                LiveModulePhase.CREATED,
                LiveModulePhase.CLOSED,
            ):
                raise LiveModuleError(
                    "live-module-already-active",
                    "a LiveModule is already active",
                    phase=self._live_phase.value,
                )

            named_live_modules = tuple(self.named_live_modules())
            for _, module in named_live_modules:
                module._live_declarations_sealed = True
            named_states = tuple(self.named_states())
            if pipeline is not None:
                from betterscale.live.runtime.pipeline import PipelineExecution
                if not isinstance(pipeline, PipelineExecution):
                    raise TypeError('pipeline activation requires PipelineExecution')
                named_live_modules, named_states = pipeline._activation_views(self)
            live_modules = tuple(module for _, module in named_live_modules)
            state_lanes = tuple(
                SIMDStateLane(
                    module_path=state_name.rpartition(".")[0],
                    state_name=state.name,
                    state=state,
                )
                for state_name, state in named_states
            )
            state_domain_list: list[StateDomain] = []
            lanes_by_domain_id: dict[int, list[SIMDStateLane]] = {}
            implicit_lanes: list[SIMDStateLane] = []
            for lane in state_lanes:
                domain = lane.state.domain
                if domain is None:
                    implicit_lanes.append(lane)
                    continue
                domain_id = id(domain)
                if domain_id not in lanes_by_domain_id:
                    state_domain_list.append(domain)
                    lanes_by_domain_id[domain_id] = []
                lanes_by_domain_id[domain_id].append(lane)
            state_domains = tuple(state_domain_list)
            meta_tensors = tuple(
                meta_tensor for _, meta_tensor in self.named_meta_tensors()
            )
            live_graphs = tuple(
                (
                    module_path,
                    graph_name,
                    graph,
                )
                for module_path, module in named_live_modules
                for graph_name, graph in module._live_graphs.items()
            )
            for module_path, graph_name, graph in live_graphs:
                if graph.prepared:
                    qualified_name = (
                        f"{module_path}.{graph_name}" if module_path else graph_name
                    )
                    raise LiveModuleError(
                        "live-graph-already-prepared",
                        "a graph in this subtree already belongs to a physical generation",
                        graph_name=qualified_name,
                    )
            capture_graph = None
            if live_graphs:
                if runtime._graph_backend is None:
                    raise LiveModuleError(
                        "missing-graph-backend",
                        "LiveGraph declarations require a graph backend during activation",
                        graph_count=len(live_graphs),
                    )
                capture_graph = getattr(runtime._graph_backend, "capture_graph", None)
                if not callable(capture_graph):
                    raise LiveModuleError(
                        "invalid-graph-backend",
                        "a graph backend must provide one capture_graph capability",
                        backend_type=type(runtime._graph_backend).__qualname__,
                    )
            state_tensors = tuple(lane.state for lane in state_lanes)
            state_plan: SIMDStatePlan | None = None
            local_max_state_blocks: int | None = None
            state_backend = runtime._state_backend
            state_schemas: tuple[SIMDStateSchema, ...] = ()
            if state_tensors and not isinstance(state_backend, StateBackend):
                raise LiveModuleError(
                    "missing-state-backend",
                    "StateTensor declarations require one explicit StateBackend provider",
                    state_count=len(state_tensors),
                    backend_type=type(state_backend).__qualname__,
                )
            if state_tensors:
                missing_state_schemas = tuple(
                    (f"{module_path}.{graph_name}" if module_path else graph_name)
                    for module_path, graph_name, graph in live_graphs
                    if not isinstance(graph.schema, CapturedGraphStateSchema)
                    or not isinstance(graph.schema.capture_state_blocks, Mapping)
                )
                if missing_state_schemas:
                    raise LiveModuleError(
                        "missing-captured-graph-state-blocks",
                        "every graph in a stateful LiveModule must explicitly "
                        "declare the State blocks changed during graph build",
                        graph_names=missing_state_schemas,
                    )
            if state_tensors:
                schemas: list[SIMDStateSchema] = []
                if implicit_lanes:
                    schemas.append(compile_simd_state_schema(implicit_lanes))
                for domain in state_domains:
                    schemas.append(
                        compile_simd_state_schema(
                            lanes_by_domain_id[id(domain)],
                            domain=domain,
                        )
                    )
                state_schemas = tuple(schemas)
            if meta_tensors and runtime.device is None:
                raise LiveModuleError(
                    "missing-metatensor-device",
                    "module-held MetaTensor declarations require one process device",
                    meta_tensor_count=len(meta_tensors),
                )
            fitting = state_memory_floor_bytes is not None
            if fitting:
                if (type(state_memory_floor_bytes) is not int or state_memory_floor_bytes < 0
                        or not state_schemas or not live_graphs
                        or not isinstance(state_backend, TorchStateBackend)
                        or not isinstance(runtime._graph_backend, TorchDeviceGraphBackend)
                        or not isinstance(runtime._memory_observer, TorchDeviceMemoryObserver)):
                    raise LiveModuleError("unsupported-memory-fit-activation",
                                          "State fitting requires graphs, complete State units, "
                                          "Torch providers and an explicit nonnegative free-memory floor")
                for module_path, graph_name, graph in live_graphs:
                    runtime._graph_backend.validate_calibration_inputs(
                        graph, graph_key=(module_path, graph_name))
                for module in live_modules:
                    if (type(module)._initialize_live_generation is not LiveModule._initialize_live_generation
                            and type(module)._rebind_live_state is LiveModule._rebind_live_state):
                        raise LiveModuleError("unqualified-state-rebinding",
                                              "a generation initializer must explicitly support calibration State replacement",
                                              module_type=type(module).__qualname__)
            generation_token = object()
            bound_states: list[StateTensor] = []
            bound_state_domains: list[StateDomain] = []
            bound_meta_tensors: list[MetaTensor[torch.Tensor]] = []
            initialized_modules: list[LiveModule] = []
            captured_graphs: list[
                tuple[LiveGraph, tuple[str, str], CapturedGraphExecution]
            ] = []
            published_graphs: list[LiveGraph] = []
            input_handoffs = []
            pool_handoffs = []
            restore_point: StateGenerationSnapshot | None = None
            state_realization: StateGenerationRealization | None = None
            memory_profile = None
            if runtime._memory_observer is not None:
                memory_profile = LiveMemoryProfileSession(
                    self,
                    runtime._memory_observer,
                )
                self._live_memory_profile = memory_profile.receipt()
            def bind_state(realization):
                nonlocal state_plan, local_max_state_blocks
                validated_state_domains = validate_state_generation_realization(state_schemas, realization)
                if memory_profile is not None:
                    memory_profile.record_state_domains(validated_state_domains)
                for schema, (realized_domain, values) in zip(
                    state_schemas,
                    validated_state_domains,
                    strict=True,
                ):
                    plan = realized_domain.plan
                    if isinstance(
                        schema.capacity_requirement,
                        ElasticStateCapacity,
                    ):
                        state_plan = plan
                        local_max_state_blocks = realized_domain.local_max_capacity
                    domain = plan.schema.domain
                    if domain is not None:
                        domain._bind(
                            generation=generation_token,
                            plan=plan,
                            local_max_capacity=realized_domain.local_max_capacity,
                        )
                        bound_state_domains.append(domain)
                    for lane, value in zip(plan.schema.lanes, values, strict=True):
                        lane.state._bind(
                            generation=generation_token,
                            num_blocks=plan.num_blocks,
                            value=value,
                        )
                        bound_states.append(lane.state)

            def release_state():
                nonlocal state_realization
                for state in reversed(bound_states):
                    state._unbind(generation=generation_token)
                bound_states.clear()
                for domain in reversed(bound_state_domains):
                    domain._unbind(generation=generation_token)
                bound_state_domains.clear()
                if state_realization is not None:
                    state_backend.release_state(state_realization)
                    state_realization = None

            try:
                state_scope = (nullcontext() if memory_profile is None
                               else memory_profile.phase("state.realization"))
                with state_scope:
                    if state_schemas:
                        state_realization = (state_backend.realize_calibration_state(state_schemas)
                                             if fitting else state_backend.realize_state(state_schemas))
                        bind_state(state_realization)
                    if pipeline is not None:
                        pipeline._publish_state_capacities()
                    if runtime.device is not None:
                        for meta_tensor in meta_tensors:
                            meta_tensor._bind_generation(
                                generation=generation_token,
                                device=runtime.device,
                            )
                            bound_meta_tensors.append(meta_tensor)
                initialize_scope = (
                    nullcontext()
                    if memory_profile is None
                    else memory_profile.phase("generation.initialize")
                )
                with initialize_scope:
                    for module in live_modules:
                        module._initialize_live_generation()
                        initialized_modules.append(module)

                def build_graph_contexts():
                    graph_builds = []
                    for module_path, graph_name, graph in live_graphs:
                        warmup_context = None
                        capture_context = None
                        if graph.schema is not None and not isinstance(
                            graph.schema,
                            GraphCallSchema,
                        ):
                            context_backend = runtime._context_backend
                            if not isinstance(context_backend, LiveGraphContextBackend):
                                raise LiveModuleError(
                                    "missing-live-graph-context-backend",
                                    "a contextual graph schema requires one process "
                                    "graph context backend",
                                    graph_name=(module_path, graph_name),
                                    backend_type=type(context_backend).__qualname__,
                                )
                            contexts = context_backend.bind_graph_build_contexts(
                                graph.schema
                            )
                            if not isinstance(contexts, tuple) or len(contexts) != 2:
                                raise LiveModuleError(
                                    "invalid-live-graph-build-contexts",
                                    "a graph context backend must return warmup and "
                                    "capture contexts",
                                    graph_name=(module_path, graph_name),
                                    contexts_type=type(contexts).__qualname__,
                                )
                            warmup_context, capture_context = contexts
                        graph_key = (module_path, graph_name)
                        graph_builds.append(
                            (
                                graph,
                                graph_key,
                                warmup_context,
                                capture_context,
                            )
                        )

                    return graph_builds

                graph_builds = build_graph_contexts()

                def snapshot_graph_state(
                    graph: LiveGraph,
                    graph_key: tuple[str, str],
                ) -> StateGenerationSnapshot | None:
                    if state_tensors:
                        schema = graph.schema
                        if not isinstance(
                            schema, CapturedGraphStateSchema
                        ) or not isinstance(schema.capture_state_blocks, Mapping):
                            raise AssertionError(
                                "stateful graph escaped State-block prevalidation"
                            )
                        snapshot_scope = (
                            nullcontext()
                            if memory_profile is None
                            else memory_profile.phase(
                                "state.snapshot",
                                graph_key=graph_key,
                            )
                        )
                        with snapshot_scope:
                            return StateGenerationSnapshot(
                                state_tensors,
                                domain_blocks=schema.capture_state_blocks,
                            )
                    return None

                warmup_graph = getattr(runtime._graph_backend, "warmup_graph", None)
                if callable(warmup_graph):
                    for graph, graph_key, warmup_context, _ in graph_builds:
                        restore_point = snapshot_graph_state(graph, graph_key)
                        try:
                            warmup_graph(
                                graph=graph,
                                graph_key=graph_key,
                                generation_token=generation_token,
                                warmup_context=warmup_context,
                                restore_point=restore_point,
                                memory_profile=memory_profile,
                            )
                        finally:
                            if restore_point is not None:
                                restore_point.release()
                                restore_point = None
                    if memory_profile is not None:
                        memory_profile.reclaim_warmup_cache()

                for module in live_modules:
                    module._finalize_live_generation_warmup()

                def capture_portfolio(handoffs=()):
                    for index, (
                        graph,
                        graph_key,
                        warmup_context,
                        capture_context,
                    ) in enumerate(graph_builds):
                        assert capture_graph is not None
                        restore_point = snapshot_graph_state(graph, graph_key)
                        try:
                            execution = capture_graph(
                                graph=graph,
                                graph_key=graph_key,
                                generation_token=generation_token,
                                warmup_context=warmup_context,
                                capture_context=capture_context,
                                restore_point=restore_point,
                                memory_profile=memory_profile,
                                **({"input_handoff": handoffs[index]} if handoffs else {}),
                            )
                        finally:
                            if restore_point is not None:
                                restore_point.release()
                                restore_point = None
                        if not isinstance(execution, CapturedGraphExecution):
                            raise LiveModuleError(
                                "graph-backend-did-not-return-execution",
                                "graph backend returned no READY captured execution",
                                graph_name=graph_key,
                                execution_type=type(execution).__qualname__,
                            )
                        captured_graphs.append((graph, graph_key, execution))
                        if fitting:
                            restore_point = snapshot_graph_state(graph, graph_key)
                            try:
                                runtime._graph_backend.prime_unpublished_capture(
                                    execution, restore_point=restore_point)
                            finally:
                                if restore_point is not None:
                                    restore_point.release()

                capture_portfolio()
                if fitting:
                    backend = runtime._graph_backend
                    observer = runtime._memory_observer
                    for _, _, calibration in captured_graphs:
                        input_handoffs.append(backend.retain_calibration_inputs(calibration))
                        pool_handoffs.append(backend.retain_calibration_pool(calibration))
                    captured_graphs.clear()
                    graph_builds.clear()
                    # Build contexts may contain old State projections. Drop all
                    # loop aliases before asking the provider for the final fit.
                    warmup_context = _ = None
                    release_state()
                    state_realization = state_backend.realize_state_fitting(
                        state_schemas, memory_observer=observer,
                        minimum_free_bytes=state_memory_floor_bytes,
                    )
                    bind_state(state_realization)
                    for module in live_modules:
                        module._rebind_live_state()
                    for handoff in pool_handoffs:
                        handoff.close()
                    pool_handoffs.clear()
                    # Reset retires native owners, but their allocator segments
                    # can remain cached. Return only that now-reclaimable cache
                    # before the final capture's driver-free admission check.
                    # The fitted State and retained input addresses stay live.
                    with memory_profile.phase("graph.calibration.reclaim"):
                        observer.reclaim()
                    graph_builds = build_graph_contexts()
                    capture_portfolio(input_handoffs)
                    input_handoffs.clear()
                    observer.reclaim()
                    if observer.snapshot().free_bytes < state_memory_floor_bytes:
                        raise LiveModuleError("final-capture-exceeds-memory-budget",
                                              "final captures exceed the calibrated State budget; "
                                              "this generation cannot become READY")
                for graph, graph_key, execution in captured_graphs:
                    graph._bind_execution(
                        graph_key=graph_key,
                        execution=execution,
                    )
                    published_graphs.append(graph)
            except BaseException as error:
                release_errors: list[BaseException] = []
                try:
                    for graph in reversed(published_graphs):
                        graph._release_before_state()
                    published_ids = {id(graph) for graph in published_graphs}
                    for graph, _, execution in reversed(captured_graphs):
                        if id(graph) not in published_ids:
                            execution.close()
                finally:
                    if restore_point is not None:
                        restore_point.release()
                    for handoff in reversed(input_handoffs):
                        handoff.close()
                    for handoff in reversed(pool_handoffs):
                        handoff.close()
                    retire_generation = getattr(runtime._graph_backend, "retire_generation", None)
                    if callable(retire_generation):
                        retire_generation(generation_token)
                    for module in reversed(initialized_modules):
                        try:
                            module._release_live_generation()
                        except BaseException as release_error:
                            release_errors.append(release_error)
                    for meta_tensor in reversed(bound_meta_tensors):
                        meta_tensor._unbind(generation=generation_token)
                    release_state()
                if memory_profile is not None:
                    self._live_memory_profile = memory_profile.finish(error)
                for release_error in release_errors:
                    error.add_note(
                        "LiveModule generation rollback also failed while "
                        f"releasing a module resource: {release_error!r}"
                    )
                raise

            self._live_generation_modules = live_modules
            self._live_state_tensors = state_tensors
            self._live_generation_state_domains = state_domains
            self._live_state_realization = state_realization
            self._live_generation_meta_tensors = meta_tensors
            self._live_state_plan = state_plan
            self._live_local_max_state_blocks = local_max_state_blocks
            for member in live_modules:
                member._live_generation_token = generation_token
                member._live_phase = LiveModulePhase.READY
            if memory_profile is not None:
                self._live_memory_profile = memory_profile.finish()
            return self

    def register_graph(
        self,
        name: str,
        *,
        entry: Callable[..., object],
        schema: object | None = None,
        allow_forward_shadow: bool = False,
    ) -> None:
        """Declare a graph; full forward shadow requires explicit compatibility opt-in."""

        if self._live_declarations_sealed:
            raise LiveModuleError(
                "sealed-live-declarations",
                "LiveModule graph declarations cannot change after root seal",
                module_type=type(self).__qualname__,
                graph_name=name,
            )
        if not isinstance(name, str) or not name or "." in name:
            raise LiveModuleError(
                "invalid-live-graph-name",
                "a local graph name must be a nonempty path component",
                module_type=type(self).__qualname__,
                graph_name=name,
            )
        if not callable(entry) or getattr(entry, "__self__", None) is not self:
            raise LiveModuleError(
                "foreign-live-graph-entry",
                "a LiveGraph entry must be a bound method of its owner",
                module_type=type(self).__qualname__,
                graph_name=name,
                entry_type=type(entry).__qualname__,
            )
        if name in self._live_graphs:
            raise LiveModuleError(
                "duplicate-live-graph",
                "a LiveModule cannot register one local graph name twice",
                module_type=type(self).__qualname__,
                graph_name=name,
            )
        self._live_graphs[name] = LiveGraph(
            entry=entry,
            schema=schema,
            allow_forward_shadow=allow_forward_shadow,
        )

    def replay(
        self,
        graph_name: str,
        /,
        *args: object,
        stream: object,
        context: LiveInvocationContext | None = None,
        **kwargs: object,
    ) -> LiveInvocation:
        """Compose shadow and graph enqueue on one caller-selected stream."""

        invocation = self.invoke(
            graph_name,
            *args,
            context=context,
            **kwargs,
        )
        invocation.shadow_replay(stream=stream)
        invocation.replay(stream=stream)
        return invocation

    def invoke(
        self,
        graph_name: str,
        /,
        *args: object,
        context: LiveInvocationContext | None = None,
        **kwargs: object,
    ) -> LiveInvocation:
        """Admit one stateful invocation for explicit asynchronous control."""

        graph = self._require_live_graph(graph_name)
        from betterscale.live.runtime.invocation import current_live_invocation

        phase = current_live_phase()
        if phase in (LivePhase.WARMUP, LivePhase.CAPTURE):
            raise LiveModuleError(
                "nested-replay-during-graph-build",
                "graph warmup and capture entries must call child forward explicitly",
                graph_name=graph_name,
                phase=phase.value,
            )
        if not graph.prepared:
            raise LiveModuleError(
                "unprepared-live-graph",
                "this LiveModule has not prepared the graph",
                graph_name=graph_name,
                module_type=type(self).__qualname__,
            )
        current = current_live_invocation()
        if current is not None:
            if current.generation_token is not self._live_generation_token:
                raise LiveModuleError(
                    "foreign-nested-live-invocation",
                    "a LiveGraph cannot borrow another generation's invocation",
                    graph_name=graph_name,
                )
            raise LiveModuleError(
                "nested-replay-during-captured-call",
                "a captured parent entry must call child forward explicitly",
                graph_name=graph_name,
            )
        return self._admit(graph, context, args, kwargs)

    def bind_replay(self, graph_name: str, /, **facts: object) -> object:
        """Bind CPU context facts for one prepared LiveGraph."""

        graph = self._require_live_graph(graph_name)
        return self._bind_graph_context(graph_name, graph, facts)

    def bind_eager_context(self, **facts: object) -> LiveInvocationContext:
        """Bind CPU facts for an ordinary invocation of this module."""

        from betterscale.live.runtime.invocation import LiveInvocationContextBackend

        self._require_active("bind eager context on")
        backend = self._live_runtime._context_backend
        if not isinstance(backend, LiveInvocationContextBackend):
            raise LiveModuleError(
                "missing-live-context-backend",
                "ordinary contextual invocation requires a process context backend",
                backend_type=type(backend).__qualname__,
            )
        return backend.bind_eager_context(**facts)

    def clear_state_blocks(
        self,
        block_ids: object,
        *,
        domain: StateDomain | None = None,
    ) -> None:
        """Clear logical units across every lane in one exact State domain."""

        try:
            requested = tuple(block_ids)  # type: ignore[arg-type]
        except TypeError as exc:
            raise LiveModuleError(
                "invalid-state-block-reset",
                "State block reset requires one finite iterable of logical ids",
                block_ids_type=type(block_ids).__qualname__,
            ) from exc
        with self._live_lock:
            if self._live_phase is not LiveModulePhase.READY:
                raise LiveModuleError(
                    "live-root-not-ready",
                    "State blocks may be reset only while the LiveModule is READY",
                    phase=self._live_phase.value,
                )
            if self._live_active_invocations:
                raise LiveModuleError(
                    "live-root-busy",
                    "State blocks cannot be reset during an active invocation",
                    active_invocation_count=len(self._live_active_invocations),
                )
            if domain is None:
                plan = self._live_state_plan
            else:
                if not any(
                    candidate is domain
                    for candidate in self._live_generation_state_domains
                ):
                    raise LiveModuleError(
                        "foreign-state-domain",
                        "State reset domain does not belong to this active root",
                        domain_type=type(domain).__qualname__,
                    )
                plan = domain.plan
            capacity = 0 if plan is None else plan.num_blocks
            normalized: list[int] = []
            seen: set[int] = set()
            for value in requested:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise LiveModuleError(
                        "invalid-state-block-reset",
                        "State block ids must be non-bool integers",
                        block_id=value,
                    )
                if value < 0 or value >= capacity:
                    raise LiveModuleError(
                        "state-block-reset-out-of-range",
                        "State block reset exceeds the active logical capacity",
                        block_id=value,
                        state_capacity=capacity,
                    )
                if value not in seen:
                    normalized.append(value)
                    seen.add(value)
            states = (
                () if plan is None else tuple(lane.state for lane in plan.schema.lanes)
            )
            for state in states:
                for block_id in normalized:
                    state.clear_logical_block(block_id)

    def offload_state(
        self,
        key: object,
        selection: object,
        *,
        stream: object | None,
    ) -> object:
        """Enqueue one caller-selected device-to-host State transaction."""

        from betterscale.live.runtime.host_state import (
            HostStateKey,
            HostStateSelection,
            TorchHostStateBackend,
        )

        self._require_active("offloading State")
        backend = self._live_runtime._host_state_backend
        if not isinstance(backend, TorchHostStateBackend):
            raise LiveModuleError(
                "missing-host-state-backend",
                "State offload requires one process host-State capability",
                backend_type=type(backend).__qualname__,
            )
        if not isinstance(key, HostStateKey) or not isinstance(
            selection, HostStateSelection
        ):
            raise TypeError("State offload requires typed key and selection")
        return backend.offload(
            ((name, state) for name, state in self.named_states()
             if state.is_bound and state._binding.generation is self._live_generation_token),
            key, selection, stream=stream)

    def restore_state(
        self,
        key: object,
        selection: object,
        *,
        stream: object | None,
    ) -> object:
        """Enqueue one caller-selected host-to-device State transaction."""

        from betterscale.live.runtime.host_state import (
            HostStateKey,
            HostStateSelection,
            TorchHostStateBackend,
        )

        self._require_active("restoring State")
        backend = self._live_runtime._host_state_backend
        if not isinstance(backend, TorchHostStateBackend):
            raise LiveModuleError(
                "missing-host-state-backend",
                "State restore requires one process host-State capability",
                backend_type=type(backend).__qualname__,
            )
        if not isinstance(key, HostStateKey) or not isinstance(
            selection, HostStateSelection
        ):
            raise TypeError("State restore requires typed key and selection")
        return backend.restore(
            ((name, state) for name, state in self.named_states()
             if state.is_bound and state._binding.generation is self._live_generation_token),
            key, selection, stream=stream)

    def release_host_state(self, key: object) -> int:
        """Release one terminal host-resident session generation."""

        from betterscale.live.runtime.host_state import HostStateKey, TorchHostStateBackend

        if not isinstance(key, HostStateKey):
            raise TypeError("host State release requires one typed key")
        backend = self._live_runtime._host_state_backend
        if not isinstance(backend, TorchHostStateBackend):
            raise LiveModuleError(
                "missing-host-state-backend",
                "host State release requires one process host-State capability",
                backend_type=type(backend).__qualname__,
            )
        return backend.release(key)

    def close(self) -> None:
        """Stop admission and detach this module's physical resources."""

        with self._live_lock:
            if self._live_phase is LiveModulePhase.CLOSED:
                return
            if self._live_phase is LiveModulePhase.CREATED:
                self._live_phase = LiveModulePhase.CLOSED
                return
            if self._live_phase not in (
                LiveModulePhase.READY,
                LiveModulePhase.POISONED,
            ):
                raise LiveModuleError(
                    "invalid-live-module-phase",
                    "LiveModule cannot close from its current phase",
                    phase=self._live_phase.value,
                )
            if self._live_active_invocations:
                raise LiveModuleError(
                    "live-root-busy",
                    "a LiveModule cannot close with active invocations",
                    active_invocation_count=len(self._live_active_invocations),
                )

            self._live_phase = LiveModulePhase.CLOSING
            generation_token = self._require_generation_token()
            for module in reversed(self._live_generation_modules):
                for graph in reversed(tuple(module._live_graphs.values())):
                    graph._release_before_state()
            retire_generation = getattr(self._live_runtime._graph_backend, "retire_generation", None)
            if callable(retire_generation):
                retire_generation(generation_token)
            try:
                for module in reversed(self._live_generation_modules):
                    module._release_live_generation()
            except BaseException:
                for module in self._live_generation_modules:
                    module._live_phase = LiveModulePhase.POISONED
                raise
            for meta_tensor in reversed(self._live_generation_meta_tensors):
                meta_tensor._unbind(generation=generation_token)
            for state in reversed(self._live_state_tensors):
                state._unbind(generation=generation_token)
            for domain in reversed(self._live_generation_state_domains):
                domain._unbind(generation=generation_token)
            state_realization = self._live_state_realization
            if state_realization is not None:
                from betterscale.live.runtime.state_backend import StateBackend

                state_backend = self._live_runtime._state_backend
                if not isinstance(state_backend, StateBackend):
                    raise AssertionError("active State generation lost its backend")
                state_backend.release_state(state_realization)
            for module in reversed(self._live_generation_modules):
                module._live_generation_token = None
                module._live_phase = LiveModulePhase.CLOSED
            self._live_generation_modules = ()
            self._live_state_tensors = ()
            self._live_generation_state_domains = ()
            self._live_state_realization = None
            self._live_generation_meta_tensors = ()
            self._live_state_plan = None
            self._live_local_max_state_blocks = None

    def _require_active(self, operation: str) -> None:
        if self._live_generation_token is None:
            raise LiveModuleError(
                "live-module-not-active",
                f"a LiveModule must be activated before {operation}",
                module_type=type(self).__qualname__,
            )

    def _bind_graph_context(
        self,
        graph_name: str,
        graph: LiveGraph,
        facts: dict[str, object],
    ) -> object:
        from betterscale.live.runtime.invocation import LiveGraphContextBackend

        with self._live_lock:
            if self._live_phase is not LiveModulePhase.READY:
                raise LiveModuleError(
                    "live-root-not-ready",
                    "graph context binding requires a READY LiveModule",
                    phase=self._live_phase.value,
                )
            if not graph.prepared:
                raise LiveModuleError(
                    "unprepared-live-graph",
                    "this LiveModule has not prepared the graph",
                    graph_name=graph_name,
                )
            backend = self._live_runtime._context_backend
            if not isinstance(backend, LiveGraphContextBackend):
                raise LiveModuleError(
                    "missing-live-graph-context-backend",
                    "graph context binding requires one process context backend",
                    graph_name=graph_name,
                    backend_type=type(backend).__qualname__,
                )
            return backend.bind_graph_replay_context(graph.schema, **facts)

    def _admit(
        self,
        graph: LiveGraph,
        context: LiveInvocationContext | None,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> LiveInvocation:
        from betterscale.live.runtime.invocation import LiveInvocation

        with self._live_lock:
            if self._live_phase is not LiveModulePhase.READY:
                raise LiveModuleError(
                    "live-root-not-ready",
                    "a LiveModule admits invocations only while READY",
                    phase=self._live_phase.value,
                )
            generation_token = self._require_generation_token()
            invocation = LiveInvocation(
                graph=graph,
                generation_token=generation_token,
                context=context,
                args=args,
                kwargs=kwargs,
                owner=self,
            )
            self._live_active_invocations.add(invocation)
            return invocation

    def _retire(self, invocation: LiveInvocation) -> None:
        with self._live_lock:
            if invocation not in self._live_active_invocations:
                raise AssertionError("retiring an invocation not admitted here")
            self._live_active_invocations.remove(invocation)

    def _poison_after_failed_invocation(
        self,
        invocation: LiveInvocation,
    ) -> None:
        """Fail-stop after an invocation may have produced effects."""

        with self._live_lock:
            if invocation not in self._live_active_invocations:
                raise AssertionError("poisoning an invocation not admitted here")
            if self._live_phase is not LiveModulePhase.READY:
                raise AssertionError("an active invocation escaped its READY module")
            self._live_phase = LiveModulePhase.POISONED

    def _require_generation_token(self) -> object:
        token = self._live_generation_token
        if token is None:  # pragma: no cover - internal invariant.
            raise AssertionError("active LiveModule has no generation token")
        return token

    def register_state(self, name: str, state: StateTensor) -> None:
        """Register one local StateTensor under its module attribute name."""

        from betterscale.live.core.state_tensor import StateTensor

        if self._live_declarations_sealed:
            raise LiveModuleError(
                "sealed-live-declarations",
                "LiveModule State declarations cannot change after root seal",
                module_type=type(self).__qualname__,
                state_name=name,
            )
        if not isinstance(name, str) or not name or "." in name:
            raise LiveModuleError(
                "invalid-state-name",
                "a local StateTensor name must be one nonempty path component",
                module_type=type(self).__qualname__,
                state_name=name,
            )
        if not isinstance(state, StateTensor):
            raise LiveModuleError(
                "invalid-state-declaration",
                "LiveModule accepts only StateTensor State declarations",
                module_type=type(self).__qualname__,
                state_name=name,
                state_type=type(state).__qualname__,
            )
        if name in self._live_states:
            raise LiveModuleError(
                "duplicate-state-name",
                "a LiveModule cannot register one local StateTensor name twice",
                module_type=type(self).__qualname__,
                state_name=name,
            )
        existing = getattr(self, name, None)
        if existing is not None:
            raise LiveModuleError(
                "state-attribute-conflict",
                "a StateTensor name conflicts with an existing module attribute",
                module_type=type(self).__qualname__,
                state_name=name,
                existing_type=type(existing).__qualname__,
            )
        state._register(owner=self, name=name)
        self._live_states[name] = state
        nn.Module.__setattr__(self, name, state)

    def register_meta_tensor(
        self,
        name: str,
        meta_tensor: MetaTensor[torch.Tensor],
    ) -> None:
        """Register one fixed-contract MetaTensor as a local member."""

        from betterscale.live.core.meta_tensor import MetaTensor

        if self._live_declarations_sealed:
            raise LiveModuleError(
                "sealed-live-declarations",
                "LiveModule MetaTensor declarations cannot change after root seal",
                module_type=type(self).__qualname__,
                member_name=name,
            )
        if not isinstance(name, str) or not name or "." in name:
            raise LiveModuleError(
                "invalid-metatensor-name",
                "a local MetaTensor name must be one nonempty path component",
                module_type=type(self).__qualname__,
                member_name=name,
            )
        if not isinstance(meta_tensor, MetaTensor):
            raise LiveModuleError(
                "invalid-metatensor-declaration",
                "LiveModule accepts only MetaTensor metadata declarations",
                module_type=type(self).__qualname__,
                member_name=name,
                member_type=type(meta_tensor).__qualname__,
            )
        if name in self._live_meta_tensors:
            raise LiveModuleError(
                "duplicate-metatensor-name",
                "a LiveModule cannot register one local MetaTensor name twice",
                module_type=type(self).__qualname__,
                member_name=name,
            )
        existing = getattr(self, name, None)
        if existing is not None:
            raise LiveModuleError(
                "metatensor-attribute-conflict",
                "a MetaTensor name conflicts with an existing module attribute",
                module_type=type(self).__qualname__,
                member_name=name,
                existing_type=type(existing).__qualname__,
            )
        meta_tensor._register(owner=self, name=name)
        self._live_meta_tensors[name] = meta_tensor
        nn.Module.__setattr__(self, name, meta_tensor)

    def _require_live_graph(self, graph_name: str) -> LiveGraph:
        graph = self._live_graphs.get(graph_name)
        if graph is None:
            raise LiveModuleError(
                "unknown-live-graph",
                "LiveModule replay requires one exact local LiveGraph",
                module_type=type(self).__qualname__,
                graph_name=graph_name,
                live_graphs=tuple(self._live_graphs),
            )
        return graph
