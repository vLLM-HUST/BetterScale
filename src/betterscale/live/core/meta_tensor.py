# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Stable metadata declarations and lexical recipe resolution."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generic, Protocol, TypeVar

import torch

from betterscale.live.core.error import LiveModuleError


class MetaTensorError(LiveModuleError):
    """Machine-classified failure of metadata declaration or resolution."""


TensorT = TypeVar("TensorT", bound=torch.Tensor)


class _MetaTensorScope(Protocol):
    def _resolve(
        self,
        recipe: MetaTensor[TensorT],
    ) -> TensorT: ...


def _construct_recipe(
    recipe: MetaTensor[TensorT],
    graph_key: object,
) -> TensorT:
    try:
        value = recipe.construct()
    except Exception as exc:
        if isinstance(exc, MetaTensorError):
            raise
        raise MetaTensorError(
            "metatensor-construction-failed",
            "a MetaTensor recipe failed during realization",
            recipe_type=type(recipe).__qualname__,
            graph_key=graph_key,
        ) from exc
    if not isinstance(value, torch.Tensor):
        raise MetaTensorError(
            "invalid-metatensor-construction-result",
            "a MetaTensor recipe must produce one torch.Tensor",
            recipe_type=type(recipe).__qualname__,
            result_type=type(value).__qualname__,
        )
    return value


@contextmanager
def _activate_meta_tensor_scope(
    scope: _MetaTensorScope,
) -> Iterator[None]:
    token = _CURRENT_META_TENSOR_SCOPE.set(scope)
    try:
        yield
    finally:
        _CURRENT_META_TENSOR_SCOPE.reset(token)


class _EagerMetaTensorScope:
    """Exact-object recipe cache for one eager or warmup entry."""

    def __init__(self, graph_key: object) -> None:
        self.graph_key = graph_key
        self._values: dict[
            int,
            tuple[MetaTensor[torch.Tensor], torch.Tensor | None],
        ] = {}

    @property
    def materialization_count(self) -> int:
        return len(self._values)

    def _resolve(self, recipe: MetaTensor[TensorT]) -> TensorT:
        if recipe._owner is not None:
            return recipe._require_bound_value()
        key = id(recipe)
        binding = self._values.get(key)
        if binding is not None:
            value = binding[1]
            if value is None:
                raise MetaTensorError(
                    "recursive-metatensor-construction",
                    "one MetaTensor recipe recursively requested itself",
                    recipe_type=type(recipe).__qualname__,
                    graph_key=self.graph_key,
                )
            return value  # type: ignore[return-value]
        self._values[key] = (recipe, None)
        try:
            value = _construct_recipe(recipe, self.graph_key)
        except BaseException:
            del self._values[key]
            raise
        self._values[key] = (recipe, value)
        return value


_CURRENT_META_TENSOR_SCOPE: ContextVar[_MetaTensorScope | None] = ContextVar(
    "stateharbor_betterscale.live_metadata_realization",
    default=None,
)


@contextmanager
def meta_tensor_scope(graph_key: object) -> Iterator[_EagerMetaTensorScope]:
    """Open one ephemeral exact-object metadata resolution scope."""

    scope = _EagerMetaTensorScope(graph_key)
    with _activate_meta_tensor_scope(scope):
        yield scope


class MetaTensor(Generic[TensorT]):
    """Declare stable metadata or describe one lexical metadata recipe.

    A contract-bearing instance may be registered on one ``LiveModule`` and
    receives one physical Tensor binding per activated generation. Existing
    subclasses without a contract remain anonymous entry-local recipes whose
    ``construct()`` method is interpreted by the current resolution scope.
    """

    __slots__ = ("_contract", "_generation", "_name", "_owner", "_value")

    def __new__(cls, *_args: object, **_kwargs: object):
        instance = super().__new__(cls)
        instance._owner = None
        instance._name = None
        instance._contract = None
        instance._generation = None
        instance._value = None
        return instance

    def __init__(
        self,
        shape: Sequence[int] | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        if shape is None:
            if dtype is not None:
                raise MetaTensorError(
                    "incomplete-metatensor-contract",
                    "a MetaTensor dtype requires one fixed physical shape",
                )
            return
        try:
            declared_shape = tuple(shape)
        except TypeError as error:
            raise MetaTensorError(
                "invalid-metatensor-contract",
                "a MetaTensor contract requires a finite integer shape and dtype",
            ) from error
        if any(
            type(dimension) is not int or dimension < 0 for dimension in declared_shape
        ) or not isinstance(dtype, torch.dtype):
            raise MetaTensorError(
                "invalid-metatensor-contract",
                "a MetaTensor contract requires a nonnegative integer shape and dtype",
            )
        self._contract = (declared_shape, dtype)

    @property
    def is_bound(self) -> bool:
        return self._generation is not None

    def construct(self) -> TensorT:
        """Build one anonymous recipe value from current lexical inputs."""

        raise MetaTensorError(
            "missing-metatensor-construction",
            "an anonymous MetaTensor recipe must implement construct()",
            recipe_type=type(self).__qualname__,
        )

    @property
    def tensor(self) -> TensorT:
        """Resolve through the current entry scope or materialize eagerly."""

        scope = _CURRENT_META_TENSOR_SCOPE.get()
        if scope is not None:
            return scope._resolve(self)
        if self._owner is not None:
            return self._require_bound_value()
        with meta_tensor_scope(("eager", id(self))) as eager_scope:
            return eager_scope._resolve(self)

    def _register(self, *, owner: object, name: str) -> None:
        """Bind this declaration to one exact LiveModule member name."""

        prior_owner = self._owner
        if prior_owner is not None:
            raise MetaTensorError(
                "shared-metatensor-declaration",
                "one MetaTensor cannot be registered by more than one attribute",
                registered_owner_type=type(prior_owner).__qualname__,
                registered_name=self._name,
                requested_owner_type=type(owner).__qualname__,
                requested_name=name,
            )
        if self._contract is None:
            raise MetaTensorError(
                "missing-metatensor-contract",
                "a module-held MetaTensor requires a fixed shape and dtype",
            )
        self._owner = owner
        self._name = name

    def _bind_generation(
        self,
        *,
        generation: object,
        device: torch.device | str,
    ) -> None:
        if self._generation is not None:
            raise MetaTensorError(
                "metatensor-already-bound",
                "a MetaTensor already belongs to an active generation",
                member_name=self._name,
            )
        contract = self._contract
        if contract is None:  # pragma: no cover - registration invariant.
            raise AssertionError("registered MetaTensor has no physical contract")
        shape, dtype = contract
        self._value = torch.empty(
            shape,
            dtype=dtype,
            device=torch.device(device),
        )
        self._generation = generation

    def _unbind(self, *, generation: object) -> None:
        if self._generation is None:
            raise MetaTensorError(
                "metatensor-not-bound",
                "a MetaTensor cannot unbind without an active generation",
                member_name=self._name,
            )
        if self._generation is not generation:
            raise MetaTensorError(
                "foreign-metatensor-generation",
                "a MetaTensor received an unbind from a foreign generation",
                member_name=self._name,
            )
        self._generation = None
        self._value = None

    def _require_bound_value(self, *, generation: object | None = None) -> TensorT:
        value = self._value
        if self._generation is None or value is None:
            raise MetaTensorError(
                "metatensor-not-bound",
                "a module-held MetaTensor has no active generation binding",
                member_name=self._name,
            )
        if generation is not None and self._generation is not generation:
            raise MetaTensorError(
                "foreign-metatensor-generation",
                "a graph cannot borrow a MetaTensor from another generation",
                member_name=self._name,
            )
        return value  # type: ignore[return-value]


__all__ = (
    "MetaTensor",
    "MetaTensorError",
    "meta_tensor_scope",
)
