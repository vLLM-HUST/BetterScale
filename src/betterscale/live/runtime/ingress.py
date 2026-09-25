# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Automatic stable Tensor ingress for one LiveGraph call."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from betterscale.live.core.error import LiveModuleError
from betterscale.live.runtime.shadow import write_ingress_tensor_value


@dataclass(frozen=True, slots=True)
class GraphCallSchema:
    """One CPU-safe exemplar of a registered entry's call boundary."""

    args: tuple[object, ...]
    kwargs: dict[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.args, tuple) or not isinstance(self.kwargs, dict):
            raise LiveModuleError(
                "invalid-graph-call-schema",
                "a graph call schema requires tuple args and dict kwargs",
                args_type=type(self.args).__qualname__,
                kwargs_type=type(self.kwargs).__qualname__,
            )

    @property
    def has_host_exemplar(self) -> bool:
        leaves, _ = tree_flatten((self.args, self.kwargs))
        return _has_host_exemplar(leaves)


def _has_host_exemplar(leaves) -> bool:
    constants = (bool, int, float, str, bytes, torch.dtype, torch.device)
    return all(
        leaf.device.type == 'cpu' if isinstance(leaf, torch.Tensor)
        else leaf is None or type(leaf) in constants
        for leaf in leaves
    )


class TensorTreeIngress:
    """Allocate and rewrite stable destinations for every Tensor PyTree leaf."""

    def __init__(
        self,
        schema: GraphCallSchema,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(schema, GraphCallSchema):
            raise LiveModuleError(
                "invalid-graph-call-schema",
                "automatic graph ingress requires one GraphCallSchema",
                schema_type=type(schema).__qualname__,
            )
        exemplar_leaves, tree_spec = tree_flatten((schema.args, schema.kwargs))
        target = None if device is None else torch.device(device)
        stable_leaves: list[object] = []
        for leaf in exemplar_leaves:
            if not isinstance(leaf, torch.Tensor):
                stable_leaves.append(leaf)
                continue
            destination = torch.empty_like(leaf, device=target)
            stable_leaves.append(destination)
        stable_tree = tree_unflatten(stable_leaves, tree_spec)
        self._schema = schema
        self._tree_spec = tree_spec
        self._exemplar_leaves = tuple(exemplar_leaves)
        self._stable_leaves = tuple(stable_leaves)
        self._stable_args, self._stable_kwargs = stable_tree
        self._released = False

    @property
    def schema(self) -> GraphCallSchema:
        return self._schema

    @property
    def has_host_exemplar(self) -> bool:
        """Reject opaque leaves that could conceal an old device State alias."""
        return _has_host_exemplar(self._exemplar_leaves)

    def release(self) -> None:
        """Revoke the call boundary and drop its owned device destinations."""
        self._released = True
        self._stable_leaves = ()
        self._stable_args, self._stable_kwargs = (), {}

    @property
    def stable_call(self) -> tuple[tuple[object, ...], dict[str, object]]:
        """Return the exact captured call tree."""

        if self._released:
            raise LiveModuleError("released-graph-ingress", "graph ingress has been released")
        return self._stable_args, self._stable_kwargs

    @property
    def stable_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return stable Tensor leaves in PyTree traversal order."""

        return tuple(
            leaf for leaf in self._stable_leaves if isinstance(leaf, torch.Tensor)
        )

    def project_call(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        """Validate one call and rewrite all dynamic Tensor destinations."""

        if self._released:
            raise LiveModuleError("released-graph-ingress", "graph ingress has been released")
        if not isinstance(args, tuple) or not isinstance(kwargs, dict):
            raise LiveModuleError(
                "invalid-graph-call-tree",
                "graph replay requires tuple args and dict kwargs",
                args_type=type(args).__qualname__,
                kwargs_type=type(kwargs).__qualname__,
            )
        actual_leaves, actual_spec = tree_flatten((args, kwargs))
        if actual_spec != self._tree_spec:
            raise LiveModuleError(
                "graph-call-pytree-mismatch",
                "graph replay call structure differs from its registered schema",
            )
        for ordinal, (actual, exemplar, destination) in enumerate(
            zip(
                actual_leaves,
                self._exemplar_leaves,
                self._stable_leaves,
                strict=True,
            )
        ):
            if isinstance(exemplar, torch.Tensor):
                if not isinstance(actual, torch.Tensor):
                    raise LiveModuleError(
                        "graph-call-tensor-mismatch",
                        "a dynamic graph call leaf must remain a Tensor",
                        leaf_ordinal=ordinal,
                        actual_type=type(actual).__qualname__,
                    )
                assert isinstance(destination, torch.Tensor)
                if (
                    tuple(actual.shape) != tuple(destination.shape)
                    or actual.dtype is not destination.dtype
                ):
                    raise LiveModuleError(
                        "graph-call-tensor-schema-mismatch",
                        "a graph call Tensor shape or dtype differs from its "
                        "admitted realization",
                        leaf_ordinal=ordinal,
                        expected_shape=tuple(destination.shape),
                        actual_shape=tuple(actual.shape),
                        expected_dtype=str(destination.dtype),
                        actual_dtype=str(actual.dtype),
                        expected_device=str(destination.device),
                        actual_device=str(actual.device),
                    )
                write_ingress_tensor_value(destination, actual)
            elif type(actual) is not type(exemplar) or not _static_equal(
                actual,
                exemplar,
            ):
                raise LiveModuleError(
                    "graph-call-static-leaf-mismatch",
                    "a static graph call leaf changed after capture",
                    leaf_ordinal=ordinal,
                    expected_type=type(exemplar).__qualname__,
                    actual_type=type(actual).__qualname__,
                )
        return self.stable_call

    def stage_exemplar(self) -> tuple[tuple[object, ...], dict[str, object]]:
        """Seed capture destinations from CPU-safe registered exemplar values."""

        for exemplar, destination in zip(
            self._exemplar_leaves,
            self._stable_leaves,
            strict=True,
        ):
            if isinstance(exemplar, torch.Tensor):
                assert isinstance(destination, torch.Tensor)
                write_ingress_tensor_value(destination, exemplar)
        return self.stable_call


def _static_equal(actual: object, expected: object) -> bool:
    try:
        result = actual == expected
    except (TypeError, ValueError, RuntimeError):
        return actual is expected
    return result if isinstance(result, bool) else actual is expected


__all__ = ("GraphCallSchema", "TensorTreeIngress")
