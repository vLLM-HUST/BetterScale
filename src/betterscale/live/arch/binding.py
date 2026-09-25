# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Explicit process architecture capabilities consumed by common code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType

import torch


class ArchBindingError(RuntimeError):
    """Reject an invalid or incomplete architecture leaf binding."""


class ArchABC(ABC):
    """Mark one common leaf class as architecture-dispatched.

    Only classes that inherit ``ArchABC`` directly are dispatch roots. Their
    architecture implementations inherit the common leaf and therefore keep
    its public type and construction contract without dispatching again.
    """

    __slots__ = ()

    def __new__(cls, *args: object, **kwargs: object):
        del args, kwargs
        if ArchABC not in cls.__bases__:
            return super().__new__(cls)

        from betterscale.live.runtime.live_runtime import (
            current_architecture_binding,
        )

        implementation = current_architecture_binding().bindings.resolve(cls)
        return super().__new__(implementation)


@dataclass(frozen=True, slots=True)
class ArchClassRef:
    """Import one concrete architecture leaf only when its role is used."""

    module: str
    name: str

    def __post_init__(self) -> None:
        if not self.module or not self.name:
            raise ArchBindingError(
                "lazy architecture class references require module and name"
            )

    def resolve(self, role: type[ArchABC]) -> type[ArchABC]:
        try:
            implementation = getattr(import_module(self.module), self.name)
        except (AttributeError, ImportError) as exc:
            raise ArchBindingError(
                f"cannot load architecture leaf {self.module}:{self.name}"
            ) from exc
        return _validate_arch_implementation(role, implementation)


def _validate_arch_implementation(
    role: type[ArchABC],
    implementation: object,
) -> type[ArchABC]:
    if not isinstance(implementation, type) or not issubclass(
        implementation,
        role,
    ):
        raise ArchBindingError(
            f"architecture implementation must inherit {role.__qualname__}"
        )
    if implementation is role:
        raise ArchBindingError(
            f"architecture role {role.__qualname__} requires a concrete subclass"
        )
    return implementation


@dataclass(frozen=True, slots=True)
class ArchBindings:
    """One port's immutable common-leaf to concrete-class registry."""

    classes: Mapping[type[ArchABC], type[ArchABC] | ArchClassRef]

    def __init__(
        self,
        classes: Mapping[
            type[ArchABC],
            type[ArchABC] | ArchClassRef,
        ]
        | None = None,
    ) -> None:
        admitted: dict[type[ArchABC], type[ArchABC] | ArchClassRef] = {}
        for role, implementation in (classes or {}).items():
            if not isinstance(role, type) or ArchABC not in role.__bases__:
                raise ArchBindingError(
                    "architecture binding keys must directly inherit ArchABC"
                )
            if isinstance(implementation, ArchClassRef):
                admitted[role] = implementation
            else:
                admitted[role] = _validate_arch_implementation(
                    role,
                    implementation,
                )
        object.__setattr__(self, "classes", MappingProxyType(admitted))

    def resolve(self, role: type[ArchABC]) -> type[ArchABC]:
        """Resolve one exact common leaf or fail before construction."""
        try:
            implementation = self.classes[role]
        except KeyError as exc:
            raise ArchBindingError(
                f"architecture does not bind common leaf {role.__qualname__}"
            ) from exc
        if isinstance(implementation, ArchClassRef):
            return implementation.resolve(role)
        return implementation


@dataclass(frozen=True, slots=True)
class PlatformTraits:
    """Small torch-facing facts that differ between architecture ports."""

    device_type: str
    dispatch_key: str
    opaque_attention_op: bool
    cuda_alike: bool

    def __post_init__(self) -> None:
        if not self.device_type or not self.dispatch_key:
            raise ValueError("architecture platform traits require nonempty names")


class ArchitectureBinding(ABC):
    """One immutable architecture port selected for a process Runtime.

    The binding contains only seams currently exercised by common code. New
    subsystems should add their own narrow capability rather than introduce
    architecture conditionals into model code.
    """

    __slots__ = ()

    name: str
    supported_device_types: frozenset[str]
    platform: PlatformTraits
    bindings: ArchBindings

    def __init__(self) -> None:
        if not self.name:
            raise ValueError("architecture binding requires a nonempty name")
        if not self.supported_device_types:
            raise ValueError("architecture binding requires a device domain")
        if self.platform.device_type not in self.supported_device_types:
            raise ValueError(
                "architecture platform device must belong to its device domain"
            )
        if not isinstance(self.bindings, ArchBindings):
            raise TypeError("architecture binding requires typed leaf bindings")

    def require_device(self, device: torch.device | str) -> None:
        # PrivateUse1 names such as ``npu`` become valid torch.device strings
        # only after their extension registers the backend. Architecture
        # selection must be checkable before that native activation.
        device_type = (
            device.type
            if isinstance(device, torch.device)
            else str(device).partition(":")[0]
        )
        if not device_type or device_type not in self.supported_device_types:
            raise ValueError(
                f"architecture {self.name!r} does not support device {device}"
            )

    def prepare_runtime(self, *, device: torch.device | str, rank_binding: object) -> None:
        """Finish port-specific startup before publishing a distributed Runtime.

        Called collectively at bootstrap, never from graph capture or replay.
        Ports without lazy physical startup need no additional work.
        """

    @abstractmethod
    def get_attention_backend(self, *args: object, **kwargs: object) -> type:
        """Return this port's ordinary-attention backend class."""

    @abstractmethod
    def create_graph_backend(
        self,
        *,
        device: torch.device | str,
        warmup_iterations: int = 1,
    ) -> object:
        """Create this port's physical LiveGraph execution capability."""


__all__ = (
    "ArchABC",
    "ArchBindingError",
    "ArchBindings",
    "ArchClassRef",
    "ArchitectureBinding",
    "PlatformTraits",
)
