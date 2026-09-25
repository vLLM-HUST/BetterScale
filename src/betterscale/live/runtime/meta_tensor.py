# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Ordered graph-local MetaTensor realization programs."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import TypeVar

import torch

from betterscale.live.core.meta_tensor import (
    _CURRENT_META_TENSOR_SCOPE,
    MetaTensor,
    MetaTensorError,
    _activate_meta_tensor_scope,
    _construct_recipe,
)
from betterscale.live.runtime.phase import LivePhase, live_phase_scope
from betterscale.live.runtime.shadow import GraphIngressEffectTrace

TensorT = TypeVar("TensorT", bound=torch.Tensor)
ResultT = TypeVar("ResultT")


@dataclass(slots=True)
class _CapturedSlot:
    exact_type: type[MetaTensor[torch.Tensor]]
    member: MetaTensor[torch.Tensor] | None = None
    construction_trace: tuple[int, ...] = ()
    root: torch.Tensor | None = None


@dataclass(slots=True)
class _ValidationFrame:
    label: str
    trace: tuple[int, ...]
    cursor: int = 0


class MetaTensorRealization:
    """One LiveGraph-owned positional metadata program and physical binding."""

    def __init__(
        self,
        owner: object,
        *,
        graph_key: object | None = None,
        generation: object | None = None,
    ) -> None:
        if owner is None:
            raise MetaTensorError(
                "invalid-metadata-realization-owner",
                "a metadata realization requires one exact logical owner",
            )
        self.owner = owner
        self.graph_key = owner if graph_key is None else graph_key
        self._generation = generation
        self._graph_ingress_effects = GraphIngressEffectTrace()
        self._slots: list[_CapturedSlot] = []
        self._entry_trace: tuple[int, ...] = ()
        self._construction_actions: list[Callable[[object], object]] = []
        self._construction_action_traces: list[tuple[int, ...]] = []
        self._construction_action_cursor = 0
        self._construction_action_depth = 0
        self._requires_forward_replay = False
        self._sealed = False
        self._closed = False
        self._mode: str | None = None
        self._recipe_slots: dict[
            int,
            tuple[MetaTensor[torch.Tensor], int],
        ] = {}
        self._slot_recipes: dict[int, MetaTensor[torch.Tensor]] = {}
        self._values: dict[int, torch.Tensor] = {}
        self._record_frames: list[list[int]] = []
        self._validation_frames: list[_ValidationFrame] = []

    @property
    def materialization_count(self) -> int:
        return len(self._slots)

    @property
    def construction_action_count(self) -> int:
        """Return the number of explicit host construction actions."""

        return len(self._construction_actions)

    @property
    def requires_forward_replay(self) -> bool:
        """Return whether replay must still re-enter the original forward."""

        return self._requires_forward_replay or not self._construction_actions

    @property
    def retained_tensors(self) -> tuple[torch.Tensor, ...]:
        roots = tuple(
            slot.root
            for slot in self._slots
            if slot.member is None and slot.root is not None
        )
        return (*roots, *self._graph_ingress_effects.retained_tensors)

    @property
    def graph_ingress_effects(self) -> GraphIngressEffectTrace | None:
        if self._graph_ingress_effects.sealed:
            return self._graph_ingress_effects
        return None

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def closed(self) -> bool:
        return self._closed

    @contextmanager
    def capture(self) -> Iterator[MetaTensorRealization]:
        """Record one ordered entry and materialize its graph-local slots."""

        if self._closed:
            raise MetaTensorError(
                "closed-metadata-realization",
                "a closed graph metadata realization cannot be captured",
                graph_key=self.graph_key,
            )
        if self._sealed:
            raise MetaTensorError(
                "sealed-metadata-realization",
                "a graph metadata realization may be captured only once",
                graph_key=self.graph_key,
            )
        self._begin("record")
        self._record_frames.append([])
        try:
            with (
                live_phase_scope(LivePhase.CAPTURE),
                _activate_meta_tensor_scope(self),
            ):
                yield self
            self._entry_trace = tuple(self._record_frames[0])
            if any(slot.root is None for slot in self._slots):
                raise AssertionError("MetaTensor capture left an incomplete slot")
            self._sealed = True
        except BaseException:
            self._slots.clear()
            self._entry_trace = ()
            self._construction_actions.clear()
            self._construction_action_traces.clear()
            self._requires_forward_replay = False
            raise
        finally:
            self._finish()

    def capture_reuse(self) -> AbstractContextManager[MetaTensorRealization]:
        """Validate a native-capture entry while reusing physical roots."""

        return self._validation_entry("reuse", capture_phase=True)

    def capture_construction_program(
        self,
        source: MetaTensorRealization,
        context: object,
    ) -> None:
        """Materialize an equivalent graph's explicit metadata program.

        The positional program is copied, but every anonymous graph-local root
        is constructed again. Module-held MetaTensors continue to resolve to
        their generation binding. This keeps ping-pong graphs physically
        independent without re-entering the model forward merely to rediscover
        the same host construction program.
        """

        if self._closed:
            raise MetaTensorError(
                "closed-metadata-realization",
                "a closed graph metadata realization cannot be captured",
                graph_key=self.graph_key,
            )
        if self._sealed:
            raise MetaTensorError(
                "sealed-metadata-realization",
                "a graph metadata realization may be captured only once",
                graph_key=self.graph_key,
            )
        if not isinstance(source, MetaTensorRealization):
            raise MetaTensorError(
                "invalid-metadata-program-source",
                "metadata construction reuse requires one captured realization",
                source_type=type(source).__qualname__,
            )
        if source.closed or not source.sealed:
            raise MetaTensorError(
                "unready-metadata-program-source",
                "metadata construction reuse requires one live sealed realization",
                source_graph_key=source.graph_key,
                source_sealed=source.sealed,
                source_closed=source.closed,
            )
        if source.requires_forward_replay:
            raise MetaTensorError(
                "implicit-metadata-program-source",
                "metadata construction reuse requires a complete explicit program",
                source_graph_key=source.graph_key,
            )
        if self._generation is not source._generation:
            raise MetaTensorError(
                "foreign-metadata-program-generation",
                "metadata construction programs may be reused only within one generation",
                graph_key=self.graph_key,
                source_graph_key=source.graph_key,
            )

        self._slots = [
            _CapturedSlot(
                slot.exact_type,
                slot.member,
                slot.construction_trace,
                None,
            )
            for slot in source._slots
        ]
        self._entry_trace = source._entry_trace
        self._construction_actions = list(source._construction_actions)
        self._construction_action_traces = list(source._construction_action_traces)
        self._requires_forward_replay = False
        self._begin("materialize")
        try:
            for slot in self._slots:
                member = slot.member
                if member is not None:
                    slot.root = member._require_bound_value(
                        generation=self._generation,
                    )
            with (
                live_phase_scope(LivePhase.CAPTURE),
                _activate_meta_tensor_scope(self),
            ):
                for action, trace in zip(
                    self._construction_actions,
                    self._construction_action_traces,
                    strict=True,
                ):
                    self._validation_frames.append(
                        _ValidationFrame("construction action", trace)
                    )
                    try:
                        self._construct_action(action, context)
                        self._complete_frame()
                    finally:
                        self._validation_frames.pop()
            self._complete_construction_actions()
            if any(slot.root is None for slot in self._slots):
                raise MetaTensorError(
                    "incomplete-metadata-construction-program",
                    "an explicit metadata program did not materialize every graph-local root",
                    graph_key=self.graph_key,
                    source_graph_key=source.graph_key,
                )
            self._sealed = True
        except BaseException:
            self._slots.clear()
            self._entry_trace = ()
            self._construction_actions.clear()
            self._construction_action_traces.clear()
            self._requires_forward_replay = False
            raise
        finally:
            self._finish()

    def replay(self) -> AbstractContextManager[MetaTensorRealization]:
        """Validate one shadow entry and construct fresh shadow values."""

        return self._validation_entry("shadow", capture_phase=False)

    def replay_construction_program(self, context: object) -> None:
        """Run only the explicit host construction actions captured by this graph."""

        with self.replay():
            for action in self._construction_actions:
                self._construct_action(action, context)

    def capture_graph_ingress_effects(
        self,
        shadow_device_types: Sequence[str],
        *,
        externalize: bool = False,
    ) -> AbstractContextManager[None]:
        return self._graph_ingress_effects.capture(
            shadow_device_types,
            externalize=externalize,
        )

    @contextmanager
    def _validation_entry(
        self,
        mode: str,
        *,
        capture_phase: bool,
    ) -> Iterator[MetaTensorRealization]:
        if self._closed:
            raise MetaTensorError(
                "closed-metadata-realization",
                "a closed graph metadata realization cannot replay",
                graph_key=self.graph_key,
            )
        if not self._sealed:
            raise MetaTensorError(
                "uncaptured-metadata-realization",
                "metadata execution requires one captured ordered program",
                graph_key=self.graph_key,
            )
        self._begin(mode)
        self._validation_frames.append(_ValidationFrame("entry", self._entry_trace))
        phase = live_phase_scope(LivePhase.CAPTURE) if capture_phase else nullcontext()
        try:
            with phase, _activate_meta_tensor_scope(self):
                yield self
            self._complete_frame()
            self._complete_construction_actions()
        finally:
            self._finish()

    def _resolve(self, recipe: MetaTensor[TensorT]) -> TensorT:
        if self._mode == "record":
            return self._record(recipe)
        if self._mode in {"reuse", "shadow", "materialize"}:
            return self._validate(recipe)
        raise MetaTensorError(
            "inactive-metadata-realization",
            "MetaTensor values resolve only inside their active realization scope",
            recipe_type=type(recipe).__qualname__,
        )

    def _record(self, recipe: MetaTensor[TensorT]) -> TensorT:
        binding = self._recipe_slots.get(id(recipe))
        if binding is not None:
            slot = binding[1]
            if (
                self._slots[slot].member is None
                and self._construction_action_depth == 0
            ):
                self._requires_forward_replay = True
            self._record_frames[-1].append(slot)
            root = self._slots[slot].root
            if root is None:
                raise MetaTensorError(
                    "recursive-metatensor-construction",
                    "one MetaTensor recipe recursively requested itself",
                    recipe_type=type(recipe).__qualname__,
                    graph_key=self.graph_key,
                    slot=slot,
                )
            return root  # type: ignore[return-value]

        slot = len(self._slots)
        self._recipe_slots[id(recipe)] = (recipe, slot)
        self._slot_recipes[slot] = recipe
        self._record_frames[-1].append(slot)
        member = recipe if recipe._owner is not None else None
        if member is None and self._construction_action_depth == 0:
            self._requires_forward_replay = True
        if member is not None and self._generation is None:
            raise MetaTensorError(
                "missing-metatensor-generation",
                "a graph backend must bind module members to one root generation",
                graph_key=self.graph_key,
                owner_type=type(member._owner).__qualname__,
                member_name=member._name,
            )
        captured = _CapturedSlot(type(recipe), member)
        self._slots.append(captured)
        children: list[int] = []
        self._record_frames.append(children)
        try:
            root = (
                recipe._require_bound_value(generation=self._generation)
                if member is not None
                else _construct_recipe(recipe, self.graph_key)
            )
        finally:
            self._record_frames.pop()
        captured.construction_trace = tuple(children)
        captured.root = root
        return root

    def _construct_action(
        self,
        action: Callable[[object], ResultT],
        context: object,
    ) -> ResultT:
        if self._construction_action_depth:
            raise MetaTensorError(
                "nested-metatensor-construction-action",
                "metadata construction actions form one flat captured program",
                graph_key=self.graph_key,
                action=_construction_action_name(action),
            )
        record_frame: list[int] | None = None
        record_start = 0
        if self._mode == "record":
            self._construction_actions.append(action)
            record_frame = self._record_frames[-1]
            record_start = len(record_frame)
        elif self._mode in {"reuse", "shadow", "materialize"}:
            cursor = self._construction_action_cursor
            if cursor >= len(self._construction_actions):
                raise MetaTensorError(
                    "unexpected-metatensor-construction-action",
                    "entry executed a metadata construction action absent from capture",
                    graph_key=self.graph_key,
                    position=cursor,
                    action=_construction_action_name(action),
                )
            expected = self._construction_actions[cursor]
            if _construction_action_key(action) != _construction_action_key(expected):
                raise MetaTensorError(
                    "changed-metatensor-construction-action",
                    "entry metadata construction action differs from capture",
                    graph_key=self.graph_key,
                    position=cursor,
                    expected=_construction_action_name(expected),
                    actual=_construction_action_name(action),
                )
            self._construction_action_cursor += 1
        else:
            raise MetaTensorError(
                "inactive-metadata-realization",
                "metadata construction actions require an active realization",
                graph_key=self.graph_key,
            )

        self._construction_action_depth += 1
        try:
            result = action(context)
            if record_frame is not None:
                self._construction_action_traces.append(
                    tuple(record_frame[record_start:])
                )
            return result
        finally:
            self._construction_action_depth -= 1

    def _complete_construction_actions(self) -> None:
        if self._construction_action_cursor == len(self._construction_actions):
            return
        raise MetaTensorError(
            "missing-metatensor-construction-action",
            "entry skipped metadata construction actions required by capture",
            graph_key=self.graph_key,
            position=self._construction_action_cursor,
            missing_count=(
                len(self._construction_actions) - self._construction_action_cursor
            ),
        )

    def _validate(
        self,
        recipe: MetaTensor[TensorT],
    ) -> TensorT:
        frame = self._validation_frames[-1]
        if frame.cursor >= len(frame.trace):
            raise MetaTensorError(
                "unexpected-metatensor-demand",
                "entry visited metadata beyond its captured ordered trace",
                graph_key=self.graph_key,
                frame=frame.label,
                position=frame.cursor,
                recipe_type=type(recipe).__qualname__,
            )
        slot = frame.trace[frame.cursor]
        frame.cursor += 1
        captured = self._slots[slot]
        member = captured.member
        matches = (
            recipe is member
            if member is not None
            else recipe._owner is None and type(recipe) is captured.exact_type
        )
        if not matches:
            raise MetaTensorError(
                "unexpected-metatensor-demand",
                "entry metadata differs from its captured trace position",
                graph_key=self.graph_key,
                frame=frame.label,
                position=frame.cursor - 1,
                slot=slot,
                expected_type=(
                    captured.exact_type.__qualname__
                    if member is None
                    else f"{type(member._owner).__qualname__}.{member._name}"
                ),
                actual_type=type(recipe).__qualname__,
            )

        binding = self._recipe_slots.get(id(recipe))
        prior_recipe = self._slot_recipes.get(slot)
        if (
            binding is not None
            and binding[1] != slot
            or prior_recipe is not None
            and prior_recipe is not recipe
        ):
            raise MetaTensorError(
                "metatensor-alias-pattern-mismatch",
                "entry changed whether repeated metadata uses share one recipe object",
                graph_key=self.graph_key,
                recipe_type=type(recipe).__qualname__,
                expected_slot=slot,
                actual_slot=None if binding is None else binding[1],
            )
        self._recipe_slots[id(recipe)] = (recipe, slot)
        self._slot_recipes[slot] = recipe

        if self._mode == "reuse" or captured.member is not None:
            root = captured.root
            if root is None:  # pragma: no cover - capture invariant.
                raise AssertionError("captured MetaTensor slot has no root")
            if captured.member is not None:
                captured.member._require_bound_value(
                    generation=self._generation,
                )
            return root  # type: ignore[return-value]

        if self._mode == "materialize" and captured.root is not None:
            return captured.root  # type: ignore[return-value]

        value = self._values.get(slot)
        if value is not None:
            return value  # type: ignore[return-value]
        self._validation_frames.append(
            _ValidationFrame(
                f"slot {slot} construction",
                captured.construction_trace,
            )
        )
        try:
            value = _construct_recipe(recipe, self.graph_key)
            self._complete_frame()
        finally:
            self._validation_frames.pop()
        if self._mode == "materialize":
            captured.root = value
        else:
            self._values[slot] = value
        return value

    def _complete_frame(self) -> None:
        frame = self._validation_frames[-1]
        if frame.cursor == len(frame.trace):
            return
        missing = frame.trace[frame.cursor :]
        raise MetaTensorError(
            "missing-metatensor-demand",
            "entry skipped metadata required by its captured ordered trace",
            graph_key=self.graph_key,
            frame=frame.label,
            position=frame.cursor,
            missing_count=len(missing),
            missing_slots=missing,
        )

    def _begin(self, mode: str) -> None:
        if self._mode is not None:
            raise MetaTensorError(
                "metadata-realization-already-active",
                "one metadata realization cannot run two entries concurrently",
                graph_key=self.graph_key,
            )
        self._mode = mode
        self._recipe_slots = {}
        self._slot_recipes = {}
        self._values = {}
        self._record_frames = []
        self._validation_frames = []
        self._construction_action_cursor = 0
        self._construction_action_depth = 0

    def _finish(self) -> None:
        self._mode = None
        self._recipe_slots.clear()
        self._slot_recipes.clear()
        self._values.clear()
        self._record_frames.clear()
        self._validation_frames.clear()
        self._construction_action_cursor = 0
        self._construction_action_depth = 0

    def close(self) -> None:
        """Release retained graph-owned Tensor roots exactly once."""

        if self._mode is not None:
            raise MetaTensorError(
                "active-metadata-realization-close",
                "an active metadata realization cannot close",
                graph_key=self.graph_key,
            )
        if self._closed:
            return
        self._graph_ingress_effects.close()
        self._slots.clear()
        self._entry_trace = ()
        self._construction_actions.clear()
        self._construction_action_traces.clear()
        self._closed = True


def construct_meta_tensors(
    action: Callable[[object], ResultT],
    *,
    context: object,
) -> ResultT:
    """Execute and, during graph build, retain one explicit construction action."""

    if not callable(action):
        raise MetaTensorError(
            "invalid-metatensor-construction-action",
            "a metadata construction action must be callable",
            action_type=type(action).__qualname__,
        )
    scope = _CURRENT_META_TENSOR_SCOPE.get()
    if isinstance(scope, MetaTensorRealization):
        return scope._construct_action(action, context)
    return action(context)


def _construction_action_key(action: Callable[[object], object]) -> object:
    owner = getattr(action, "__self__", None)
    function = getattr(action, "__func__", None)
    if owner is not None and function is not None:
        return (id(owner), function)
    return action


def _construction_action_name(action: Callable[[object], object]) -> str:
    return getattr(action, "__qualname__", type(action).__qualname__)


__all__ = ("MetaTensorRealization", "construct_meta_tensors")
