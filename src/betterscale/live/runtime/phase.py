# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Dynamically scoped execution phase for live host behavior."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum


class LivePhase(Enum):
    """Observable phase of one LiveModule call."""

    EAGER = "eager"
    WARMUP = "warmup"
    CAPTURE = "capture"
    LIVE_REPLAY = "live-replay"


_CURRENT_LIVE_PHASE: ContextVar[LivePhase] = ContextVar(
    "stateharbor_betterscale.live_phase",
    default=LivePhase.EAGER,
)


def current_live_phase() -> LivePhase:
    """Return the phase visible to current Python host behavior."""

    return _CURRENT_LIVE_PHASE.get()


@contextmanager
def live_phase_scope(phase: LivePhase) -> Iterator[None]:
    """Publish one phase for a bounded recursive Module call."""

    if not isinstance(phase, LivePhase):
        raise TypeError(f"live phase must be LivePhase, got {type(phase).__name__}")
    token = _CURRENT_LIVE_PHASE.set(phase)
    try:
        yield
    finally:
        _CURRENT_LIVE_PHASE.reset(token)
