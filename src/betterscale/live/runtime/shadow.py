# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Torch-dispatch projection of a live forward onto its host behavior."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TypeVar

import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

from betterscale.live.core.error import LiveModuleError
from betterscale.live.runtime.phase import LivePhase, live_phase_scope

ResultT = TypeVar("ResultT")

__all__ = (
    "GraphIngressEffectTrace",
    "LiveModuleError",
    "ShadowForwardReceipt",
    "real_ingress_tensor_write",
    "replay_shadow_forward",
    "write_ingress_tensor_value",
)


_MISSING = object()


@dataclass(frozen=True, slots=True)
class ShadowForwardReceipt:
    """Receipt for one host projection of an original recursive forward."""

    shadow_output: object
    shadowed_operation_count: int
    real_ingress_write_count: int
    shadow_device_types: frozenset[str]


@dataclass(frozen=True, slots=True)
class _GraphIngressEffectKey:
    """One physical host-to-device occurrence inside a fixed graph."""

    operation: Hashable
    source_shape: tuple[int, ...]
    source_stride: tuple[int, ...]
    source_dtype: torch.dtype
    target_device: torch.device
    target_dtype: torch.dtype
    target_layout: torch.layout
    target_memory_format: object


@dataclass(frozen=True, slots=True)
class _GraphIngressEffect:
    key: _GraphIngressEffectKey
    stable_host_source: torch.Tensor
    stable_device_destination: torch.Tensor | None = None


class GraphIngressEffectTrace:
    """Graph-owned stable host roots for captured H2D occurrences.

    This is a physical capture trace, not a metadata semantic identity.  Exact
    MetaTensor demand matching remains the fixed host-program guard.
    """

    __slots__ = ("_effects", "_sealed")

    def __init__(self) -> None:
        self._effects: tuple[_GraphIngressEffect, ...] = ()
        self._sealed = False

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def retained_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(effect.stable_host_source for effect in self._effects)

    @contextmanager
    def capture(
        self,
        shadow_device_types: Iterable[str],
        *,
        externalize: bool = False,
    ) -> Iterator[None]:
        """Bind H2D sources encountered by one graph ingress program."""

        if self._sealed:
            raise LiveModuleError(
                "sealed-graph-ingress-effects",
                "graph ingress effects may be captured only once",
            )
        mode = _GraphIngressCaptureMode(
            _normalize_device_types(shadow_device_types),
            externalize=externalize,
        )
        try:
            with mode:
                yield
        except BaseException:
            self._effects = ()
            raise
        self._effects = tuple(mode.effects)
        self._sealed = True

    def stage(
        self,
        func: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        occurrence: int,
        shadow_device_types: frozenset[str],
    ) -> int | None:
        """Stage one matching live H2D source and advance its occurrence."""

        if not self._sealed:
            return None
        key = _host_to_device_effect_key(
            func,
            args,
            kwargs,
            shadow_device_types,
        )
        if key is None:
            return None
        if occurrence >= len(self._effects):
            raise LiveModuleError(
                "unexpected-graph-ingress-effect",
                "shadow forward produced an H2D ingress absent from capture",
                operation=_operation_name(func),
                occurrence=occurrence,
            )
        expected = self._effects[occurrence]
        if key != expected.key:
            raise LiveModuleError(
                "changed-graph-ingress-effect",
                "shadow H2D ingress no longer matches its captured occurrence",
                occurrence=occurrence,
                expected=repr(expected.key),
                actual=repr(key),
            )
        source = args[0]
        assert isinstance(source, torch.Tensor)
        expected.stable_host_source.copy_(source)
        return occurrence + 1

    def require_complete(self, occurrence: int) -> None:
        """Reject a shadow path that skipped a captured H2D occurrence."""

        if not self._sealed or occurrence == len(self._effects):
            return
        raise LiveModuleError(
            "missing-graph-ingress-effect",
            "shadow forward skipped an H2D ingress required by capture",
            expected_count=len(self._effects),
            actual_count=occurrence,
        )

    def publish_externalized(self) -> None:
        """Publish staged host values to graph-external stable device roots."""

        with torch.no_grad():
            for effect in self._effects:
                destination = effect.stable_device_destination
                if destination is not None:
                    destination.copy_(
                        effect.stable_host_source,
                        non_blocking=True,
                    )

    def close(self) -> None:
        self._effects = ()


class _GraphIngressCaptureMode(TorchDispatchMode):
    """Substitute stable host roots while graph ingress is materialized."""

    def __init__(
        self,
        shadow_device_types: frozenset[str],
        *,
        externalize: bool,
    ) -> None:
        super().__init__()
        self._shadow_device_types = shadow_device_types
        self._externalize = externalize
        self.effects: list[_GraphIngressEffect] = []

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        call_kwargs = {} if kwargs is None else kwargs
        key = _host_to_device_effect_key(
            func,
            args,
            call_kwargs,
            self._shadow_device_types,
        )
        if key is None:
            return func(*args, **call_kwargs)
        source = args[0]
        assert isinstance(source, torch.Tensor)
        stable_source = _stable_host_copy(
            source,
            pin_memory=(source.is_pinned() or key.target_device.type == "cuda"),
        )
        result = func(stable_source, *args[1:], **call_kwargs)
        destination = (
            result if self._externalize and isinstance(result, torch.Tensor) else None
        )
        self.effects.append(_GraphIngressEffect(key, stable_source, destination))
        return result


_AUTHORIZED_INGRESS_DESTINATIONS: ContextVar[tuple[torch.Tensor, ...]] = ContextVar(
    "stateharbor_real_ingress_tensor_destinations", default=()
)


@contextmanager
def real_ingress_tensor_write(destination: torch.Tensor) -> Iterator[torch.Tensor]:
    """Authorize real mutations of one exact stable graph ingress Tensor.

    Pure operations remain shadowed.  A mutable Torch operation is allowed to
    reach real storage only when every mutated Tensor is the exact authorized
    object.  The capability is intentionally destination-based rather than
    caller-name-based.
    """

    if not isinstance(destination, torch.Tensor):
        raise LiveModuleError(
            "invalid-ingress-destination",
            "real graph ingress writes require one exact Tensor destination",
            destination_type=type(destination).__qualname__,
        )
    active = _AUTHORIZED_INGRESS_DESTINATIONS.get()
    if any(candidate is destination for candidate in active):
        yield destination
        return
    token = _AUTHORIZED_INGRESS_DESTINATIONS.set((*active, destination))
    try:
        yield destination
    finally:
        _AUTHORIZED_INGRESS_DESTINATIONS.reset(token)


def write_ingress_tensor_value(
    destination: torch.Tensor,
    value: torch.Tensor | float | bool,
) -> None:
    """Write one invocation value into an exact stable graph ingress Tensor."""

    with real_ingress_tensor_write(destination):
        if isinstance(value, torch.Tensor):
            destination.copy_(value, non_blocking=True)
        else:
            destination.fill_(value)


class _ShadowForwardMode(TorchDispatchMode):
    """Virtualize selected-device Torch ops while admitting ingress writes."""

    def __init__(
        self,
        shadow_device_types: frozenset[str],
        graph_ingress_effects: GraphIngressEffectTrace | None,
    ) -> None:
        super().__init__()
        self._shadow_device_types = shadow_device_types
        self._fake_mode = FakeTensorMode()
        self._graph_ingress_effects = graph_ingress_effects
        self._graph_ingress_occurrence = 0
        self.shadowed_operation_count = 0
        self.real_ingress_write_count = 0

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        call_kwargs = {} if kwargs is None else kwargs
        if self._graph_ingress_effects is not None:
            occurrence = self._graph_ingress_effects.stage(
                func,
                args,
                call_kwargs,
                self._graph_ingress_occurrence,
                self._shadow_device_types,
            )
            if occurrence is not None:
                self._graph_ingress_occurrence = occurrence
                self.real_ingress_write_count += 1
        write_targets = _write_targets(func, args, call_kwargs)
        authorized = _AUTHORIZED_INGRESS_DESTINATIONS.get()
        authorized_targets = tuple(
            target
            for target in write_targets
            if any(target is candidate for candidate in authorized)
        )

        if authorized and write_targets:
            unauthorized = tuple(
                target
                for target in write_targets
                if not any(target is candidate for candidate in authorized)
            )
            if unauthorized:
                raise LiveModuleError(
                    "unauthorized-real-tensor-write",
                    "an ingress write scope attempted to mutate another Tensor",
                    operation=_operation_name(func),
                    authorized_count=len(authorized),
                    unauthorized_count=len(unauthorized),
                )
            if authorized_targets:
                self.real_ingress_write_count += 1
                return func(*args, **call_kwargs)

        if not _touches_shadow_device(
            func,
            args,
            call_kwargs,
            self._shadow_device_types,
        ):
            return func(*args, **call_kwargs)

        fake_args = tree_map(self._to_fake_tensor, args)
        fake_kwargs = tree_map(self._to_fake_tensor, call_kwargs)
        self.shadowed_operation_count += 1
        with self._fake_mode:
            return func(*fake_args, **fake_kwargs)

    def _to_fake_tensor(self, value: object) -> object:
        if not isinstance(value, torch.Tensor) or is_fake(value):
            return value
        return self._fake_mode.from_tensor(value)


class _ShadowCollectiveWork:
    """Completed Python-level work returned by a suppressed collective."""

    __slots__ = ()

    def wait(self, timeout: object = None) -> bool:
        del timeout
        return True

    def is_completed(self) -> bool:
        return True


@contextmanager
def _shadow_python_collectives(mode: _ShadowForwardMode) -> Iterator[None]:
    """Suppress selected-device collectives lacking FakeTensor kernels.

    ACLGraph capture already retained the real communication. Host projection
    replays only Python control and metadata effects, so issuing the collective
    again would be wrong even if c10d supplied a FakeTensor implementation.
    Keep this list narrow and preserve ordinary collectives on unselected
    devices.
    """

    original_all_to_all_single = dist.all_to_all_single

    def all_to_all_single(
        output: torch.Tensor,
        input: torch.Tensor,
        output_split_sizes: object = None,
        input_split_sizes: object = None,
        group: object = None,
        async_op: bool = False,
    ) -> object:
        if not _tensor_on_shadow_device(
            output,
            mode._shadow_device_types,
        ) and not _tensor_on_shadow_device(input, mode._shadow_device_types):
            return original_all_to_all_single(
                output,
                input,
                output_split_sizes,
                input_split_sizes,
                group,
                async_op,
            )
        mode.shadowed_operation_count += 1
        return _ShadowCollectiveWork() if async_op else None

    dist.all_to_all_single = all_to_all_single
    try:
        yield
    finally:
        dist.all_to_all_single = original_all_to_all_single


def replay_shadow_forward(
    forward: Callable[..., ResultT],
    /,
    *args: object,
    shadow_device_types: Iterable[str] = ("cuda", "npu"),
    graph_ingress_effects: GraphIngressEffectTrace | None = None,
    **kwargs: object,
) -> ShadowForwardReceipt:
    """Execute the original forward with selected-device effects virtualized."""

    if not callable(forward):
        raise LiveModuleError(
            "invalid-live-forward",
            "shadow replay requires one callable forward program",
            forward_type=type(forward).__qualname__,
        )
    normalized_devices = _normalize_device_types(shadow_device_types)
    mode = _ShadowForwardMode(normalized_devices, graph_ingress_effects)
    with (
        live_phase_scope(LivePhase.LIVE_REPLAY),
        mode,
        _shadow_python_collectives(mode),
    ):
        output = forward(*args, **kwargs)
    if graph_ingress_effects is not None:
        graph_ingress_effects.require_complete(mode._graph_ingress_occurrence)
    return ShadowForwardReceipt(
        shadow_output=output,
        shadowed_operation_count=mode.shadowed_operation_count,
        real_ingress_write_count=mode.real_ingress_write_count,
        shadow_device_types=normalized_devices,
    )


def _normalize_device_types(device_types: Iterable[str]) -> frozenset[str]:
    normalized = []
    for device_type in device_types:
        if not isinstance(device_type, str) or not device_type.strip():
            raise LiveModuleError(
                "invalid-shadow-device",
                "shadow device types must be nonempty strings",
                device_type=device_type,
            )
        normalized.append(device_type.strip().lower())
    if not normalized:
        raise LiveModuleError(
            "missing-shadow-device",
            "shadow replay requires at least one selected device type",
        )
    return frozenset(normalized)


def _host_to_device_effect_key(
    func: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    shadow_device_types: frozenset[str],
) -> _GraphIngressEffectKey | None:
    schema = getattr(func, "_schema", None)
    if schema is None or not args:
        return None
    if not (
        schema.name == "aten::_to_copy"
        or (schema.name == "aten::to" and schema.overload_name == "dtype_layout")
    ):
        return None
    source = args[0]
    if not isinstance(source, torch.Tensor) or source.device.type != "cpu":
        return None
    target_value = _argument_value(func, args, kwargs, "device")
    if target_value is _MISSING or target_value is None:
        return None
    target = torch.device(target_value)
    if target.type not in shadow_device_types or target.type == "cpu":
        return None
    dtype = _argument_value(func, args, kwargs, "dtype")
    layout = _argument_value(func, args, kwargs, "layout")
    memory_format = _argument_value(func, args, kwargs, "memory_format")
    return _GraphIngressEffectKey(
        operation=_operation_name(func),
        source_shape=tuple(source.shape),
        source_stride=tuple(source.stride()),
        source_dtype=source.dtype,
        target_device=target,
        target_dtype=source.dtype if dtype in (_MISSING, None) else dtype,
        target_layout=source.layout if layout in (_MISSING, None) else layout,
        target_memory_format=(
            None if memory_format in (_MISSING, None) else memory_format
        ),
    )


def _stable_host_copy(
    source: torch.Tensor,
    *,
    pin_memory: bool | None = None,
) -> torch.Tensor:
    stable = torch.empty_strided(
        tuple(source.shape),
        tuple(source.stride()),
        dtype=source.dtype,
        device="cpu",
        pin_memory=(source.is_pinned() if pin_memory is None else pin_memory),
    )
    stable.copy_(source)
    return stable


def _touches_shadow_device(
    func: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    shadow_device_types: frozenset[str],
) -> bool:
    flat_values, _ = tree_flatten((args, kwargs))
    for value in flat_values:
        if not isinstance(value, torch.Tensor):
            continue
        if is_fake(value) or value.device.type in shadow_device_types:
            return True

    device = _argument_value(func, args, kwargs, "device")
    if device is _MISSING or device is None:
        return False
    try:
        return torch.device(device).type in shadow_device_types
    except (RuntimeError, TypeError, ValueError):
        return False


def _tensor_on_shadow_device(
    value: object,
    shadow_device_types: frozenset[str],
) -> bool:
    return isinstance(value, torch.Tensor) and value.device.type in shadow_device_types


def _write_targets(
    func: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[torch.Tensor, ...]:
    schema = getattr(func, "_schema", None)
    if schema is None or not bool(getattr(schema, "is_mutable", False)):
        return ()
    targets = []
    for argument in schema.arguments:
        alias_info = argument.alias_info
        if alias_info is None or not alias_info.is_write:
            continue
        value = _argument_value(func, args, kwargs, argument.name)
        if value is _MISSING:
            continue
        flat_values, _ = tree_flatten(value)
        targets.extend(
            candidate
            for candidate in flat_values
            if isinstance(candidate, torch.Tensor)
        )
    return tuple(targets)


def _argument_value(
    func: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    name: str,
) -> object:
    schema = getattr(func, "_schema", None)
    if schema is None:
        return kwargs.get(name, _MISSING)
    positional_index = 0
    for argument in schema.arguments:
        if argument.kwarg_only:
            value = kwargs.get(argument.name, _MISSING)
        else:
            if positional_index < len(args):
                value = args[positional_index]
            else:
                value = kwargs.get(argument.name, _MISSING)
            positional_index += 1
        if argument.name == name:
            return value
    return _MISSING


def _operation_name(func: Callable[..., object]) -> Hashable:
    schema = getattr(func, "_schema", None)
    if schema is not None:
        return schema.name
    return getattr(func, "__qualname__", repr(func))
