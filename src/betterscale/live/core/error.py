# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Machine-classified LiveModule contract failures."""

from __future__ import annotations

from types import MappingProxyType


class LiveModuleError(RuntimeError):
    """Machine-classified failure of a LiveModule lifecycle contract."""

    def __init__(self, code: str, message: str, **context: object) -> None:
        super().__init__(message)
        self.code = code
        self.context = MappingProxyType(dict(context))


__all__ = ("LiveModuleError",)
