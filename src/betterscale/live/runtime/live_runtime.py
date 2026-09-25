# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Process-installed physical capabilities captured by ``LiveModule``."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

import torch

from betterscale.live.arch.binding import ArchitectureBinding
from betterscale.live.core.error import LiveModuleError
from betterscale.live.runtime.distributed import LiveRankBinding

if TYPE_CHECKING:
    from collections.abc import Iterator


class LiveRuntime:
    """Process-installed physical capabilities available to LiveModules.

    The process Runtime deliberately owns no model root, generation phase,
    invocation lane or activation protocol.  A LiveModule captures this object
    during construction and owns every externally visible lifecycle operation.
    """

    def __init__(
        self,
        *,
        architecture: ArchitectureBinding | None = None,
        state_backend: object | None = None,
        host_state_backend: object | None = None,
        memory_observer: object | None = None,
        graph_backend: object | None = None,
        context_backend: object | None = None,
        rank_binding: LiveRankBinding | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if architecture is not None and not isinstance(
            architecture,
            ArchitectureBinding,
        ):
            raise LiveModuleError(
                "invalid-live-architecture-binding",
                "LiveRuntime architecture must be one typed ArchitectureBinding",
                architecture_type=type(architecture).__qualname__,
            )
        self._architecture = architecture
        self._state_backend = state_backend
        self._host_state_backend = host_state_backend
        self._memory_observer = memory_observer
        self._graph_backend = graph_backend
        self._context_backend = context_backend
        if rank_binding is not None and not isinstance(rank_binding, LiveRankBinding):
            raise LiveModuleError(
                "invalid-live-rank-binding",
                "Runtime requires one typed immutable rank binding",
                binding_type=type(rank_binding).__qualname__,
            )
        self._rank_binding = rank_binding
        if self._architecture is not None and device is not None:
            try:
                self._architecture.require_device(device)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise LiveModuleError(
                    "live-architecture-device-mismatch",
                    "LiveRuntime architecture does not admit its selected device",
                    architecture=self._architecture.name,
                    device=str(device),
                ) from exc
        # Keep a PrivateUse1 device name representable before its extension
        # registers that name with torch. Physical consumers normalize it only
        # after process bootstrap has activated the selected architecture.
        self._device = device

    def _prepare(self) -> None:
        if self._architecture is not None and self._rank_binding is not None:
            if self._device is None:
                raise LiveModuleError(
                    "missing-distributed-runtime-device",
                    "distributed Runtime startup requires its selected device",
                )
            self._architecture.prepare_runtime(
                device=self._device, rank_binding=self._rank_binding,
            )

    @property
    def architecture(self) -> ArchitectureBinding | None:
        """Return the architecture selected before module construction."""

        return self._architecture

    @property
    def device(self) -> torch.device | str | None:
        """Return the process/rank device selected for physical capabilities."""

        return self._device

    @property
    def rank_binding(self) -> LiveRankBinding | None:
        """Return the immutable distributed identity admitted at construction."""

        return self._rank_binding


_CURRENT_LIVE_RUNTIME: ContextVar[LiveRuntime | None] = ContextVar(
    "stateharbor_live_runtime",
    default=None,
)


def current_live_runtime() -> LiveRuntime:
    """Return the exact process Runtime or fail before module construction."""

    runtime = _CURRENT_LIVE_RUNTIME.get()
    if runtime is None:
        raise LiveModuleError(
            "missing-live-runtime",
            "LiveModule construction requires one installed LiveRuntime",
        )
    return runtime


def current_architecture_binding() -> ArchitectureBinding:
    """Return the architecture selected by the installed process Runtime."""

    runtime = current_live_runtime()
    architecture = runtime.architecture
    if architecture is None:
        raise LiveModuleError(
            "missing-live-architecture-binding",
            "architecture-dependent construction requires an explicit process binding",
        )
    return architecture


def live_runtime_installed() -> bool:
    """Return whether this construction context has one installed Runtime."""

    return _CURRENT_LIVE_RUNTIME.get() is not None


def install_live_runtime(runtime: LiveRuntime) -> LiveRuntime:
    """Install one process-long Runtime before constructing LiveModules."""

    if not isinstance(runtime, LiveRuntime):
        raise LiveModuleError(
            "invalid-live-runtime",
            "only a typed LiveRuntime may be installed",
            runtime_type=type(runtime).__qualname__,
        )
    current = _CURRENT_LIVE_RUNTIME.get()
    if current is not None and current is not runtime:
        raise LiveModuleError(
            "live-runtime-already-installed",
            "one execution context cannot replace its installed LiveRuntime",
        )
    runtime._prepare()
    _CURRENT_LIVE_RUNTIME.set(runtime)
    return runtime


@contextmanager
def live_runtime(runtime: LiveRuntime) -> Iterator[LiveRuntime]:
    """Install one bounded Runtime while constructing and testing a root."""

    if not isinstance(runtime, LiveRuntime):
        raise LiveModuleError(
            "invalid-live-runtime",
            "only a typed LiveRuntime may enter a construction scope",
            runtime_type=type(runtime).__qualname__,
        )
    if _CURRENT_LIVE_RUNTIME.get() is not None:
        raise LiveModuleError(
            "live-runtime-already-installed",
            "nested LiveRuntime construction scopes are not admitted",
        )
    runtime._prepare()
    token = _CURRENT_LIVE_RUNTIME.set(runtime)
    try:
        yield runtime
    finally:
        _CURRENT_LIVE_RUNTIME.reset(token)


__all__ = (
    "LiveRuntime",
    "current_architecture_binding",
    "current_live_runtime",
    "install_live_runtime",
    "live_runtime",
    "live_runtime_installed",
)
