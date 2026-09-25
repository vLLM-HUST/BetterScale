# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Build-scoped State shape access, deliberately separate from physical binding."""
from contextlib import contextmanager
from contextvars import ContextVar


state_shadow_resolver = ContextVar('betterscale.live_state_shadow_resolver', default=None)


@contextmanager
def shadow_state_access(resolver):
    token = state_shadow_resolver.set(resolver)
    try:
        yield
    finally:
        state_shadow_resolver.reset(token)
