# SPDX-License-Identifier: Apache-2.0
# Transferred from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# Imports relocated and construction-scope helper inlined.
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import patch

import torch

from betterscale.live import (
    GraphCallSchema,
    LiveGraph,
    LiveInvocationPhase,
    LiveModule,
    LiveModuleError,
    LivePhase,
    LiveRuntime,
    MetaTensor,
    SIMDStatePlan,
    StateBackend,
    StateTensor,
    current_live_phase,
    write_ingress_tensor_value,
)
from betterscale.live.arch.ascend.graph import ACLGraphBackend
from betterscale.live import live_runtime


def construct_live(runtime, factory):
    with live_runtime(runtime):
        return factory()


@dataclass(frozen=True, slots=True, kw_only=True)
class _StateGraphCallSchema(GraphCallSchema):
    capture_state_blocks: dict[object | None, tuple[int, ...]]


class _FakeGraph:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def replay(self) -> None:
        self.events.append("graph-replay")

    def reset(self) -> None:
        self.events.append("graph-reset")


class _FakeStream:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def wait_event(self, _event: object) -> None:
        self.events.append("stream-wait-event")


class _FakeEvent:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def record(self) -> None:
        self.events.append("event-record")

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        self.events.append("event-synchronize")


class _FakeNPU:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def NPUGraph(self) -> _FakeGraph:
        self.events.append("graph-create")
        return _FakeGraph(self.events)

    def Stream(self) -> object:
        self.events.append("stream-create")
        return _FakeStream(self.events)

    def Event(self) -> object:
        self.events.append("event-create")
        return _FakeEvent(self.events)

    def current_stream(self) -> _FakeStream:
        return _FakeStream(self.events)

    @contextmanager
    def stream(self, stream: object):
        self.events.append("stream-enter")
        try:
            yield stream
        finally:
            self.events.append("stream-exit")

    @contextmanager
    def graph(self, _graph: object, **_kwargs: object):
        self.events.append("graph-enter")
        try:
            yield
        finally:
            self.events.append("graph-exit")

    def synchronize(self) -> None:
        self.events.append("synchronize")


class _Context:
    def __init__(self, value: float, ingress: torch.Tensor) -> None:
        self.value = value
        self.ingress = ingress
        self.projections = 0

    def project(self) -> None:
        write_ingress_tensor_value(self.ingress, self.value)
        self.projections += 1


class _Meta(MetaTensor[torch.Tensor]):
    def __init__(self, ingress: torch.Tensor) -> None:
        self.ingress = ingress

    def construct(self) -> torch.Tensor:
        return self.ingress * 2.0


class _Root(LiveModule):
    def forward(self, context: _Context) -> torch.Tensor:
        if current_live_phase() is not LivePhase.CAPTURE:
            context.project()
        return _Meta(context.ingress).tensor + 1.0


class _RegisteredContext:
    def __init__(self, events: list[str], name: str) -> None:
        self._events = events
        self._name = name

    @contextmanager
    def activate(self):
        self._events.append(f"{self._name}-enter")
        try:
            yield self
        finally:
            self._events.append(f"{self._name}-exit")


class _RegisteredSchema:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.call_schema = GraphCallSchema((torch.tensor([2.0]),), {})


class _RegisteredContextBackend:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def bind_graph_build_contexts(
        self,
        _schema: object,
    ) -> tuple[_RegisteredContext, _RegisteredContext]:
        return (
            _RegisteredContext(self.events, "warmup-context"),
            _RegisteredContext(self.events, "capture-context"),
        )

    def bind_graph_replay_context(
        self,
        _schema: object,
        **facts: object,
    ) -> _RegisteredContext:
        return _RegisteredContext(self.events, str(facts["name"]))


class _RegisteredACLGraphBackend(ACLGraphBackend):
    def __init__(
        self,
        events: list[str],
        *,
        device: torch.device | str,
        warmup_iterations: int = 1,
    ) -> None:
        self._events = events
        super().__init__(
            device=device,
            warmup_iterations=warmup_iterations,
        )


class _RegisteredRoot(LiveModule):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events
        self.register_graph(
            "decode",
            entry=self.forward,
            schema=_RegisteredSchema(events),
            # This lifecycle fixture intentionally re-enters forward.
            allow_forward_shadow=True,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.events.append("forward")
        return value * 2.0


class _RegisteredPortfolioRoot(_RegisteredRoot):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.register_graph(
            "verify",
            entry=self.forward,
            schema=_RegisteredSchema(events),
            # This lifecycle fixture intentionally re-enters forward.
            allow_forward_shadow=True,
        )


class _WarmupSealingPortfolioRoot(_RegisteredPortfolioRoot):
    def _finalize_live_generation_warmup(self) -> None:
        self.events.append("warmup-seal")


class _BuildState(StateTensor[torch.Tensor, str, tuple[int, ...]]):
    pass


class _BuildStateBackend(StateBackend):
    def _allocate_state_domain(
        self,
        plan: SIMDStatePlan,
    ) -> dict[StateTensor, torch.Tensor]:
        return {
            lane.state: torch.full(
                (plan.num_blocks, *lane.state.block_shape),
                3.0,
                dtype=lane.state.storage_dtype,
            )
            for lane in plan.schema.lanes
        }


class _StateMutatingPortfolioRoot(LiveModule):
    def __init__(self) -> None:
        super().__init__()
        self.register_state(
            "state",
            _BuildState(
                role="build-state",
                requirement=(1,),
                block_shape=(),
                storage_dtype=torch.float32,
            ),
        )
        self.observed: list[float] = []
        schema = _StateGraphCallSchema(
            args=(torch.tensor([1.0]),),
            kwargs={},
            capture_state_blocks={None: (0,)},
        )
        self.register_graph("first", entry=self.forward, schema=schema)
        self.register_graph("second", entry=self.forward, schema=schema)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.observed.append(float(self.state.tensor.item()))
        self.state.tensor.add_(1.0)
        return value + self.state.tensor


class LiveModuleACLGraphBackendTests(unittest.TestCase):
    def test_generation_resource_seal_runs_after_portfolio_warmup_before_capture(
        self,
    ) -> None:
        events: list[str] = []
        runtime = LiveRuntime(
            graph_backend=_RegisteredACLGraphBackend(
                events,
                device="cpu",
                warmup_iterations=1,
            ),
            context_backend=_RegisteredContextBackend(events),
        )
        root = construct_live(runtime, lambda: _WarmupSealingPortfolioRoot(events))

        with patch.object(torch, "npu", _FakeNPU(events), create=True):
            root.activate()
            root.close()

        self.assertEqual(events.count("warmup-seal"), 1)
        self.assertEqual(events.count("forward"), 4)
        self.assertEqual(events[: events.index("warmup-seal")].count("forward"), 2)
        self.assertLess(events.index("warmup-seal"), events.index("graph-create"))

    def test_process_backend_prepares_live_graph_and_context_binder(self):
        events: list[str] = []
        fake_npu = _FakeNPU(events)
        runtime = LiveRuntime(
            graph_backend=_RegisteredACLGraphBackend(
                events,
                device="cpu",
                warmup_iterations=1,
            ),
            context_backend=_RegisteredContextBackend(events),
        )
        root = construct_live(runtime, lambda: _RegisteredRoot(events))

        with patch.object(torch, "npu", fake_npu, create=True):
            root.activate()
            context = root.bind_replay("decode", name="live-context")
            invocation = root.replay(
                "decode",
                torch.tensor([7.0]),
                stream=fake_npu.Stream(),
                context=context,
            )
            self.assertIs(invocation.phase, LiveInvocationPhase.IN_FLIGHT)
            invocation.retire()
            root.close()

        self.assertIn("warmup-context-enter", events)
        self.assertIn("capture-context-enter", events)
        self.assertIn("live-context-enter", events)
        self.assertIn("graph-replay", events)
        self.assertLess(events.index("live-context-exit"), events.index("graph-replay"))

    def test_one_backend_owns_independent_portfolio_graph_states(self) -> None:
        events: list[str] = []
        runtime = LiveRuntime(
            graph_backend=_RegisteredACLGraphBackend(
                events,
                device="cpu",
                warmup_iterations=0,
            ),
            context_backend=_RegisteredContextBackend(events),
        )
        root = construct_live(runtime, lambda: _RegisteredPortfolioRoot(events))

        with patch.object(torch, "npu", _FakeNPU(events), create=True):
            root.activate()
            stream = torch.npu.Stream()  # type: ignore[attr-defined]
            first = root.replay(
                "decode",
                torch.tensor([3.0]),
                stream=stream,
            )
            second = root.replay(
                "verify",
                torch.tensor([5.0]),
                stream=stream,
            )
            first.retire()
            second.retire()
            root.close()

        self.assertEqual(events.count("graph-create"), 2)
        self.assertEqual(events.count("graph-replay"), 2)
        self.assertEqual(events.count("graph-reset"), 2)

    def test_one_root_snapshot_restores_state_between_all_build_entries(self) -> None:
        events: list[str] = []
        runtime = LiveRuntime(
            graph_backend=_RegisteredACLGraphBackend(
                events,
                device="cpu",
                warmup_iterations=1,
            ),
            state_backend=_BuildStateBackend(memory_budget_bytes=4),
        )
        root = construct_live(runtime, _StateMutatingPortfolioRoot)

        with patch.object(torch, "npu", _FakeNPU(events), create=True):
            root.activate()
            torch.testing.assert_close(root.state.tensor, torch.tensor([3.0]))
            root.close()

        # Each graph runs one warmup entry and one captured entry. Every entry
        # must see the generation's exact initial image, including graph two.
        self.assertEqual(root.observed, [3.0, 3.0, 3.0, 3.0])

    def test_live_graph_capture_and_close_own_backend_resources(self):
        events: list[str] = []
        ingress = torch.tensor([0.0])
        root = construct_live(LiveRuntime(), _Root)
        capture_context = _Context(3.0, ingress)

        backend = _RegisteredACLGraphBackend(
            events,
            device="cpu",
        )
        graph = LiveGraph(
            entry=root.forward,
            schema=GraphCallSchema((capture_context,), {}),
        )
        fake_npu = _FakeNPU(events)
        with patch.object(torch, "npu", fake_npu, create=True):
            execution = backend.capture_graph(
                graph=graph,
                graph_key=("", "decode"),
            )
            graph._bind_execution(
                graph_key=("", "decode"),
                execution=execution,
            )
            metadata = graph.metadata
            assert metadata is not None
            self.assertTrue(metadata.sealed)
            self.assertEqual(capture_context.projections, 1)
            self.assertEqual(metadata.materialization_count, 1)
            stream = torch.npu.Stream()  # type: ignore[attr-defined]
            with graph._stream_scope(stream):
                graph._publish_ingress()
            graph._execute(stream=stream)
            graph._release_before_state()

        self.assertEqual(
            events,
            [
                "synchronize",
                "synchronize",
                "graph-create",
                "stream-create",
                "graph-enter",
                "graph-exit",
                "synchronize",
                "stream-create",
                "stream-enter",
                "stream-exit",
                "stream-enter",
                "graph-replay",
                "stream-exit",
                "synchronize",
                "graph-reset",
            ],
        )
        self.assertTrue(metadata.closed)
        self.assertEqual(metadata.retained_tensors, ())

    def test_capture_rejects_schema_free_graph(self):
        events: list[str] = []
        backend = _RegisteredACLGraphBackend(
            events,
            device="cpu",
        )
        graph = LiveGraph(entry=lambda: None)

        with self.assertRaises(LiveModuleError) as rejected:
            backend.capture_graph(
                graph=graph,
                graph_key=("", "decode"),
            )

        # A missing call schema fails before architecture resources are made.
        self.assertEqual(rejected.exception.code, "unsupported-aclgraph-schema")
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()


def test_qwen_graph_declarations_build_without_forward_shadow():
    """CPU contract probe of the actual entry with stand-in numerical kernels."""
    import json
    from pathlib import Path
    import sys
    import types
    from betterscale.live import TorchStateBackend
    from betterscale.live.llm.qwen35 import Geometry, Capacity
    from betterscale.live.llm.qwen35 import gdn_graph as module

    prototype = Path(__file__).resolve().parents[1] / 'prototypes/qwen35-state-lanes'
    geometry = Geometry.from_config(json.loads((prototype / 'qwen35-0.8b-text-config.json').read_text()))
    events = []

    def conv(output, x, weight, *, conv_state, **kwargs):
        output.copy_(x)
        conv_state[:4].add_(1)

    def recurrence(q, k, v, g, beta, scale, state, **kwargs):
        state[:12].add_(1)
        return torch.zeros_like(v), state

    kernel = types.ModuleType('betterscale.live.llm.qwen35.gdn_candidates')
    kernel.fused_recurrent_gated_delta_rule_fwd = recurrence
    with (patch.object(torch, 'npu', _FakeNPU(events), create=True),
          patch.object(torch.ops._C_ascend, 'npu_causal_conv1d_custom', conv, create=True),
          patch.dict(sys.modules, {'betterscale.live.llm.qwen35.gdn_candidates': kernel})):
        root = construct_live(LiveRuntime(
            device='cpu', state_backend=TorchStateBackend('cpu', memory_budget_bytes=512 << 20),
            graph_backend=ACLGraphBackend(device='cpu')),
            lambda: module.GDNGraphRoot(geometry, Capacity(4, 5, token_pages=32),
                                       torch.ones(4, 6144, dtype=torch.bfloat16)))
        root.activate()
        try:
            assert all(not graph.metadata.requires_forward_replay for _, graph in root.named_graphs())
            assert torch.all(root.target['0'].recurrent.tensor == .125)
            assert torch.all(root.target['0'].conv.tensor == .25)
        finally:
            root.close()
        assert events.count('graph-reset') == 2


def test_full_qwen_root_declares_and_retires_all_four_graphs():
    """Actual model-root metadata protocol; mocked arithmetic is not a model oracle."""
    from betterscale.live import TorchStateBackend
    from betterscale.live.llm.qwen35 import Capacity, Geometry
    from betterscale.live.llm.qwen35.graphs import QwenLiveLLMRoot

    geometry = Geometry(('linear_attention', 'full_attention'), 1, 4, 1, 2, 4, 4, 4, 8)
    target = torch.nn.Module()
    target.model = torch.nn.Module()
    target.model.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    target.model.vocab_size = 32

    def target_math(self, ids, positions, write, read, **kwargs):
        self.target['0'].recurrent.tensor.add_(1)
        return torch.ones(len(ids), 8, dtype=torch.bfloat16), torch.ones(len(ids), 32, dtype=torch.bfloat16)

    def draft_math(self, ids, seed, positions, write, read):
        self.draft.key.tensor.add_(1)
        return target_math(self, ids, positions, write, read)

    events = []
    with (patch.object(torch, 'npu', _FakeNPU(events), create=True),
          patch.object(QwenLiveLLMRoot, '_target_forward', target_math),
          patch.object(QwenLiveLLMRoot, '_draft_forward', draft_math)):
        root = construct_live(LiveRuntime(device='cpu',
            state_backend=TorchStateBackend('cpu', memory_budget_bytes=1 << 20),
            graph_backend=ACLGraphBackend(device='cpu')),
            lambda: QwenLiveLLMRoot(geometry, Capacity(1, 1, token_pages=1), target, torch.nn.Identity()))
        root.activate()
        assert [name for name, _ in root.named_graphs()] == ['target1', 'target3', 'draft1', 'draft2']
        assert all(not graph.metadata.requires_forward_replay for _, graph in root.named_graphs())
        assert torch.count_nonzero(root.target['0'].recurrent.tensor) == 0
        assert torch.count_nonzero(root.draft.key.tensor) == 0
        root.close()
        assert events.count('graph-reset') == 4
