# From LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: owned namespace; explicit architecture entry.
# SPDX-License-Identifier: Apache-2.0
"""torch-npu ACLGraph specialization of the common graph backend."""

from __future__ import annotations

from contextlib import AbstractContextManager

import torch

from betterscale.live.runtime.graph_backend import TorchDeviceGraphBackend


class ACLGraphBackend(TorchDeviceGraphBackend):
    """Compile and own ACLGraph realizations through the common lifecycle."""

    graph_kind = "aclgraph"
    graph_display_name = "ACLGraph"

    def _create_graph(self) -> object:
        return torch.npu.NPUGraph()  # type: ignore[attr-defined]

    def _create_stream(self) -> object:
        return torch.npu.Stream()  # type: ignore[attr-defined]

    def _stream_scope(self, stream: object) -> AbstractContextManager[object]:
        return torch.npu.stream(stream)  # type: ignore[attr-defined,no-any-return]

    def _capture_scope(
        self,
        graph: object,
        stream: object,
    ) -> AbstractContextManager[object]:
        return self._capture_scope_with_pool(graph, stream, None)

    def _capture_stream_id(self, stream: object) -> int | None:
        return getattr(stream, "npu_stream", None)

    def _create_graph_pool(self) -> object:
        return torch.npu.graph_pool_handle()  # type: ignore[attr-defined]

    def _capture_scope_with_pool(
        self, graph: object, stream: object, pool: object | None,
    ) -> AbstractContextManager[object]:
        return torch.npu.graph(  # type: ignore[attr-defined,no-any-return]
            graph,
            stream=stream,
            pool=pool,
            capture_error_mode=self._capture_error_mode,
        )

    def _synchronize(self) -> None:
        torch.npu.synchronize()  # type: ignore[attr-defined]


__all__ = ("ACLGraphBackend",)
