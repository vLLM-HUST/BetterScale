# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Immutable identity of one rank-bound LiveModule process."""

from __future__ import annotations

from dataclasses import dataclass

from betterscale.live.core.error import LiveModuleError


@dataclass(frozen=True, slots=True)
class LiveRankBinding:
    """Exact static group membership observed after rank/device binding.

    This is an admission receipt, not a distributed layout language. Dynamic
    request, token, expert and State placement remains invocation data.
    """

    global_rank: int
    local_rank: int
    world_ranks: tuple[int, ...]
    tensor_parallel_ranks: tuple[int, ...]
    pipeline_parallel_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.global_rank) is not int or self.global_rank < 0:
            self._reject("global_rank must be a nonnegative non-bool integer")
        if type(self.local_rank) is not int or self.local_rank < 0:
            self._reject("local_rank must be a nonnegative non-bool integer")
        for name in (
            "world_ranks",
            "tensor_parallel_ranks",
            "pipeline_parallel_ranks",
        ):
            ranks = getattr(self, name)
            if (
                type(ranks) is not tuple
                or not ranks
                or any(type(rank) is not int or rank < 0 for rank in ranks)
                or len(set(ranks)) != len(ranks)
            ):
                self._reject(f"{name} must be one nonempty unique rank tuple")
            if self.global_rank not in ranks:
                self._reject(f"global_rank must belong to {name}")
        world = set(self.world_ranks)
        if not set(self.tensor_parallel_ranks).issubset(world):
            self._reject("tensor_parallel_ranks must belong to world_ranks")
        if not set(self.pipeline_parallel_ranks).issubset(world):
            self._reject("pipeline_parallel_ranks must belong to world_ranks")

    @property
    def world_size(self) -> int:
        return len(self.world_ranks)

    @property
    def tensor_parallel_size(self) -> int:
        return len(self.tensor_parallel_ranks)

    @property
    def tensor_parallel_rank(self) -> int:
        return self.tensor_parallel_ranks.index(self.global_rank)

    @property
    def pipeline_parallel_size(self) -> int:
        return len(self.pipeline_parallel_ranks)

    @property
    def pipeline_parallel_rank(self) -> int:
        return self.pipeline_parallel_ranks.index(self.global_rank)

    def _reject(self, detail: str) -> None:
        raise LiveModuleError(
            "invalid-live-rank-binding",
            "LiveModule rank binding is malformed",
            detail=detail,
        )


__all__ = ("LiveRankBinding",)
