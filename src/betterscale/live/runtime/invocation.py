# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""One admitted ForwardContext transaction for a LiveModule call."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from enum import Enum
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import torch

from betterscale.live.core.error import LiveModuleError
from betterscale.live.runtime.live_graph import LiveGraph
from betterscale.live.runtime.phase import LivePhase, live_phase_scope
from betterscale.live.runtime.shadow import replay_shadow_forward

if TYPE_CHECKING:
    from betterscale.live.core.live_module import LiveModule


_current_live_invocation: ContextVar[object | None] = ContextVar(
    "stateharbor_current_live_invocation",
    default=None,
)


@runtime_checkable
class LiveInvocationContext(Protocol):
    """One typed lexical environment around an admitted model call."""

    def activate(self) -> AbstractContextManager[object]: ...


@runtime_checkable
class LiveInvocationContextBackend(Protocol):
    """Compile pure CPU invocation facts into one lexical eager context."""

    def bind_eager_context(self, **facts: object) -> LiveInvocationContext: ...


@runtime_checkable
class LiveGraphContextBackend(Protocol):
    """Bind graph build and live-shadow contexts from one retained schema."""

    def bind_graph_build_contexts(
        self,
        schema: object,
    ) -> tuple[LiveInvocationContext, LiveInvocationContext]: ...

    def bind_graph_replay_context(
        self,
        schema: object,
        **facts: object,
    ) -> LiveInvocationContext: ...


class LiveInvocationPhase(Enum):
    """Observable phases implemented by the first invocation vertical."""

    BOUND = "bound"
    HOST_PROJECTED = "host-projected"
    PREPARED = "prepared"
    IN_FLIGHT = "in-flight"
    ABORTING = "aborting"
    RETIRED = "retired"


class LiveInvocation:
    """Execute one captured LiveGraph fixed at admission time.

    Construction is LiveModule-owned. The invocation snapshots its selected
    program and generation token and never consults mutable selection state.
    """

    def __init__(
        self,
        *,
        graph: LiveGraph,
        generation_token: object,
        context: LiveInvocationContext | None,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        owner: LiveModule | None = None,
    ) -> None:
        if not isinstance(graph, LiveGraph):
            raise LiveModuleError(
                "invalid-live-invocation-graph",
                "a graph LiveInvocation requires one LiveGraph",
                graph_type=type(graph).__qualname__,
            )
        if not graph.captured:
            raise LiveModuleError(
                "live-invocation-graph-not-captured",
                "LiveInvocation accepts only a captured LiveGraph",
            )
        self._graph = graph
        self._generation_token = generation_token
        if context is not None and not isinstance(context, LiveInvocationContext):
            raise LiveModuleError(
                "invalid-live-invocation-context",
                "LiveInvocation context must provide one typed activation scope",
                context_type=type(context).__qualname__,
            )
        self._context = context
        self._owner = owner
        self._args = args
        self._kwargs = dict(kwargs)
        self._phase = LiveInvocationPhase.BOUND

    @property
    def phase(self) -> LiveInvocationPhase:
        return self._phase

    @property
    def generation_token(self) -> object:
        """Return the exact opaque generation borrowed at admission."""

        return self._generation_token

    @property
    def graph(self) -> LiveGraph:
        """Return the exact captured LiveGraph fixed at admission."""

        return self._graph

    @property
    def context(self) -> LiveInvocationContext | None:
        """Return the joint lexical ForwardContext borrowed by this call."""

        return self._context

    def shadow_replay(self, *, stream: object) -> None:
        """Project host state and enqueue ingress on the supplied stream."""

        if self._phase is not LiveInvocationPhase.BOUND:
            raise LiveModuleError(
                "live-invocation-shadow-replay-out-of-order",
                "shadow replay requires a newly bound LiveInvocation",
                phase=self._phase.value,
            )

        try:
            graph = self._graph
            metadata = graph.metadata
            assert metadata is not None
            if metadata.requires_forward_replay and not graph.allow_forward_shadow:
                raise LiveModuleError(
                    "forward-shadow-disabled",
                    "graph metadata requires full forward shadow; declare complete "
                    "construct_meta_tensors actions, or explicitly opt in with "
                    "register_graph(..., allow_forward_shadow=True) for compatibility",
                    graph_key=graph.graph_key,
                    construction_action_count=metadata.construction_action_count,
                )
            with (
                graph._stream_scope(stream),
                self._activation_scope() as construction_context,
            ):
                self._transition(LiveInvocationPhase.HOST_PROJECTED)

                metadata = graph.metadata
                assert metadata is not None
                projection = graph.ingress_projection
                projected_args = self._args
                projected_kwargs = self._kwargs
                if projection is not None:
                    projected = projection.project_call(
                        self._args,
                        dict(self._kwargs),
                    )
                    if (
                        not isinstance(projected, tuple)
                        or len(projected) != 2
                        or not isinstance(projected[0], tuple)
                        or not isinstance(projected[1], dict)
                    ):
                        raise LiveModuleError(
                            "invalid-graph-ingress-projection-result",
                            "graph ingress projection must return "
                            "(tuple args, dict kwargs)",
                            projection_type=type(projection).__qualname__,
                        )
                    projected_args, projected_kwargs = projected

                if metadata.requires_forward_replay:

                    def forward_only() -> object:
                        return graph.entry(*projected_args, **projected_kwargs)

                    with metadata.replay():
                        replay_shadow_forward(
                            forward_only,
                            shadow_device_types=graph.shadow_device_types,
                            graph_ingress_effects=metadata.graph_ingress_effects,
                        )
                elif metadata.construction_action_count:
                    replay_shadow_forward(
                        lambda: metadata.replay_construction_program(
                            construction_context
                        ),
                        shadow_device_types=graph.shadow_device_types,
                        graph_ingress_effects=metadata.graph_ingress_effects,
                    )
                graph._publish_ingress()
                self._transition(LiveInvocationPhase.PREPARED)
        except BaseException:
            self._abort()
            raise

    def replay(self, *, stream: object) -> None:
        """Enqueue the captured graph on the supplied stream and return."""

        if self._phase is not LiveInvocationPhase.PREPARED:
            raise LiveModuleError(
                "live-invocation-replay-out-of-order",
                "graph replay requires one prepared LiveInvocation",
                phase=self._phase.value,
            )

        try:
            with torch.inference_mode():
                self._graph._execute(stream=stream)
            self._transition(LiveInvocationPhase.IN_FLIGHT)
        except BaseException:
            self._abort()
            raise

    def retire(self) -> None:
        """Assert caller-owned completion and release this invocation."""

        if self._phase is not LiveInvocationPhase.IN_FLIGHT:
            raise LiveModuleError(
                "live-invocation-retire-out-of-order",
                "retirement requires one in-flight invocation",
                phase=self._phase.value,
            )
        self._transition(LiveInvocationPhase.RETIRED)
        if self._owner is not None:
            self._owner._retire(self)

    @contextmanager
    def _activation_scope(self) -> Iterator[object | None]:
        invocation_token = _current_live_invocation.set(self)
        try:
            scope = nullcontext() if self._context is None else self._context.activate()
            if not isinstance(scope, AbstractContextManager):
                raise LiveModuleError(
                    "invalid-live-invocation-context-scope",
                    "LiveInvocation context activation must return a context manager",
                    context_type=type(self._context).__qualname__,
                    scope_type=type(scope).__qualname__,
                )
            with (
                scope as active_context,
                torch.inference_mode(),
                live_phase_scope(LivePhase.LIVE_REPLAY),
            ):
                yield active_context
        finally:
            _current_live_invocation.reset(invocation_token)

    def _abort(self) -> None:
        self._transition(LiveInvocationPhase.ABORTING)
        if self._owner is not None:
            self._owner._poison_after_failed_invocation(self)
        self._transition(LiveInvocationPhase.RETIRED)
        if self._owner is not None:
            self._owner._retire(self)

    def _transition(self, phase: LiveInvocationPhase) -> None:
        self._phase = phase


def current_live_invocation() -> LiveInvocation | None:
    """Return the lexical invocation, if execution currently owns one."""

    invocation = _current_live_invocation.get()
    if invocation is None:
        return None
    return cast("LiveInvocation", invocation)


__all__ = (
    "LiveGraphContextBackend",
    "LiveInvocation",
    "LiveInvocationContext",
    "LiveInvocationContextBackend",
    "LiveInvocationPhase",
    "current_live_invocation",
)
