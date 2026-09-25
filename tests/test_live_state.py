# SPDX-License-Identifier: Apache-2.0
# From LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0, tests/test_livemodule_simd_state.py
# Imports relocated; construction helper inlined to avoid upstream tests dependency.
import unittest
from dataclasses import dataclass

import torch
from torch import nn

from betterscale.live import (
    ElasticStateCapacity,
    ExactStateCapacity,
    GraphCallSchema,
    LiveModule,
    LiveModuleError,
    LiveModulePhase,
    LiveRuntime,
    SIMDStateLane,
    SIMDStatePlan,
    StateBackend,
    StateDomain,
    StateDomainRealization,
    StateGenerationSnapshot,
    StateTensor,
    StateTensorError,
    TorchStateBackend,
    compile_simd_state_schema,
)
from betterscale.live import live_runtime


def construct_live(runtime, factory):
    with live_runtime(runtime):
        return factory()


class _StateLeaf(LiveModule):
    def __init__(
        self,
        *,
        role: str,
        block_shape: tuple[int, ...],
        storage_dtype: torch.dtype,
        domain: StateDomain | None = None,
        physical_blocks_per_logical_block: int = 1,
        leading_physical_blocks: int = 0,
    ) -> None:
        super().__init__()
        self.register_state(
            "state",
            StateTensor(
                role=role,
                requirement=(role, block_shape, storage_dtype),
                block_shape=block_shape,
                storage_dtype=storage_dtype,
                domain=domain,
                physical_blocks_per_logical_block=(physical_blocks_per_logical_block),
                leading_physical_blocks=leading_physical_blocks,
            ),
        )

    def forward(self) -> torch.Tensor:
        return self.state.tensor


class _PairRoot(LiveModule):
    def __init__(self, left: _StateLeaf, right: _StateLeaf) -> None:
        super().__init__()
        self.left = left
        self.right = right

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.left(), self.right())


class _NestedRoot(LiveModule):
    def __init__(self, leaf: _StateLeaf) -> None:
        super().__init__()
        self.container = nn.Sequential(nn.Identity(), leaf)

    def forward(self) -> torch.Tensor:
        return self.container[1]()


class _ExactDomainLeaf(LiveModule):
    def __init__(self, *, capacity: int, role: str = "slot") -> None:
        super().__init__()
        self.slots = StateDomain(ExactStateCapacity(capacity))
        self.register_state(
            "token",
            StateTensor(
                role=f"{role}-token",
                requirement=(role, "token"),
                block_shape=(),
                storage_dtype=torch.int64,
                domain=self.slots,
            ),
        )
        self.register_state(
            "length",
            StateTensor(
                role=f"{role}-length",
                requirement=(role, "length"),
                block_shape=(),
                storage_dtype=torch.int32,
                domain=self.slots,
            ),
        )

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.token.tensor, self.length.tensor


class _MixedDomainRoot(LiveModule):
    def __init__(self) -> None:
        super().__init__()
        self.kv = _StateLeaf(
            role="kv",
            block_shape=(2,),
            storage_dtype=torch.float32,
        )
        self.continuation = _ExactDomainLeaf(capacity=3)

    def forward(self) -> tuple[torch.Tensor, ...]:
        token, length = self.continuation()
        return self.kv(), token, length


class _OneExactDomainLeaf(LiveModule):
    def __init__(self, *, role: str, capacity: int = 2) -> None:
        super().__init__()
        self.domain = StateDomain(ExactStateCapacity(capacity))
        self.register_state(
            "state",
            StateTensor(
                role=role,
                requirement=(role,),
                block_shape=(1,),
                storage_dtype=torch.float32,
                domain=self.domain,
            ),
        )

    def forward(self) -> torch.Tensor:
        return self.state.tensor


class _TwoExactDomainRoot(LiveModule):
    def __init__(self) -> None:
        super().__init__()
        self.left = _OneExactDomainLeaf(role="left")
        self.right = _OneExactDomainLeaf(role="right")

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.left(), self.right()


@dataclass(frozen=True, slots=True, kw_only=True)
class _StateBlockSelection(GraphCallSchema):
    capture_state_blocks: dict[object, tuple[int, ...]]


class _ExactDomainGraphRoot(_OneExactDomainLeaf):
    def __init__(self) -> None:
        super().__init__(role="exact-graph")
        self.register_graph(
            "step",
            entry=self.forward,
            schema=_StateBlockSelection(
                args=(),
                kwargs={},
                capture_state_blocks={self.domain: (0, 1)},
            ),
        )


class _MissingStateGraphSelectionRoot(_OneExactDomainLeaf):
    def __init__(self) -> None:
        super().__init__(role="missing-selection")
        self.register_graph("step", entry=self.forward)


class _SharedElasticDomainRoot(LiveModule):
    def __init__(self) -> None:
        super().__init__()
        self.kv_domain = StateDomain(ElasticStateCapacity())
        self.target = _StateLeaf(
            role="target-kv",
            block_shape=(2,),
            storage_dtype=torch.float32,
            domain=self.kv_domain,
        )
        self.draft = _StateLeaf(
            role="draft-kv",
            block_shape=(1,),
            storage_dtype=torch.float32,
            domain=self.kv_domain,
        )

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.target(), self.draft()


class _ElasticDomainLeaf(LiveModule):
    def __init__(self, role: str) -> None:
        super().__init__()
        self.domain = StateDomain(ElasticStateCapacity())
        self.register_state(
            "state",
            StateTensor(
                role=role,
                requirement=(role,),
                block_shape=(1,),
                storage_dtype=torch.float32,
                domain=self.domain,
            ),
        )

    def forward(self) -> torch.Tensor:
        return self.state.tensor


class _IndependentBackend(StateBackend):
    def __init__(
        self,
        *,
        memory_budget_bytes: int | None = None,
        capacity_coordinator: object | None = None,
    ) -> None:
        super().__init__(
            memory_budget_bytes=memory_budget_bytes,
            capacity_coordinator=capacity_coordinator,
        )
        self.plans: list[SIMDStatePlan] = []
        self.released: list[StateDomainRealization] = []
        self.all_unbound_during_allocation: bool | None = None
        self.all_unbound_during_release: list[bool] = []

    def _allocate_state_domain(
        self,
        plan: SIMDStatePlan,
    ) -> dict[StateTensor, torch.Tensor]:
        self.plans.append(plan)
        self.all_unbound_during_allocation = all(
            not lane.state.is_bound for lane in plan.schema.lanes
        )
        return {
            lane.state: torch.full(
                lane.state.physical_shape(plan.num_blocks),
                lane_index + 1,
                dtype=lane.state.storage_dtype,
            )
            for lane_index, lane in enumerate(plan.schema.lanes)
        }

    def _release_state_domain(self, realization: StateDomainRealization) -> None:
        self.released.append(realization)
        self.all_unbound_during_release.append(
            all(not lane.state.is_bound for lane in realization.plan.schema.lanes)
        )


class _CapacityCoordinator:
    def __init__(self, admitted_num_blocks: object) -> None:
        self.admitted_num_blocks = admitted_num_blocks
        self.local_maxima: list[int] = []

    def admit_num_blocks(self, local_max_blocks: int, /) -> object:
        self.local_maxima.append(local_max_blocks)
        return self.admitted_num_blocks


class _CoalescingBackend(StateBackend):
    def __init__(self, *, memory_budget_bytes: int) -> None:
        super().__init__(memory_budget_bytes=memory_budget_bytes)
        self.arena: torch.Tensor | None = None

    def _allocate_state_domain(
        self,
        plan: SIMDStatePlan,
    ) -> dict[StateTensor, torch.Tensor]:
        first = plan.schema.lanes[0].state
        assert all(
            lane.state.block_shape == first.block_shape
            and lane.state.storage_dtype is first.storage_dtype
            for lane in plan.schema.lanes
        )
        self.arena = torch.empty(
            (len(plan.schema.lanes), plan.num_blocks, *first.block_shape),
            dtype=first.storage_dtype,
        )
        return {
            lane.state: self.arena[index]
            for index, lane in enumerate(plan.schema.lanes)
        }


class _OverlappingBackend(StateBackend):
    def _allocate_state_domain(
        self,
        plan: SIMDStatePlan,
    ) -> dict[StateTensor, torch.Tensor]:
        first = plan.schema.lanes[0].state
        shared = torch.empty(
            (plan.num_blocks, *first.block_shape),
            dtype=first.storage_dtype,
        )
        return {lane.state: shared for lane in plan.schema.lanes}


class _IncompleteBackend(StateBackend):
    def _allocate_state_domain(
        self,
        plan: SIMDStatePlan,
    ) -> dict[StateTensor, torch.Tensor]:
        lane = plan.schema.lanes[0]
        return {
            lane.state: torch.empty(
                (plan.num_blocks, *lane.state.block_shape),
                dtype=lane.state.storage_dtype,
            )
        }


class _WrongShapeBackend(StateBackend):
    def _allocate_state_domain(
        self,
        plan: SIMDStatePlan,
    ) -> dict[StateTensor, torch.Tensor]:
        lane = plan.schema.lanes[0]
        return {
            lane.state: torch.empty(
                (plan.num_blocks, *lane.state.block_shape, 2),
                dtype=lane.state.storage_dtype,
            )
        }


class _CrossDomainOverlappingBackend(StateBackend):
    def __init__(self) -> None:
        super().__init__()
        self.shared: torch.Tensor | None = None

    def _allocate_state_domain(
        self,
        plan: SIMDStatePlan,
    ) -> dict[StateTensor, torch.Tensor]:
        lane = plan.schema.lanes[0]
        if self.shared is None:
            self.shared = torch.empty(
                lane.state.physical_shape(plan.num_blocks),
                dtype=lane.state.storage_dtype,
            )
        return {lane.state: self.shared}


class _FailingGraphBackend:
    def capture_graph(self, **_: object) -> object:
        raise RuntimeError("intentional graph capture failure")


class LiveModuleSIMDStateTests(unittest.TestCase):
    def test_common_ancestor_domain_spans_target_and_draft_children(self) -> None:
        coordinator = _CapacityCoordinator(2)
        root = construct_live(
            LiveRuntime(
                state_backend=_IndependentBackend(
                    memory_budget_bytes=36,
                    capacity_coordinator=coordinator,
                ),
            ),
            _SharedElasticDomainRoot,
        )

        root.activate()

        self.assertIs(root.state_plan, root.kv_domain.plan)
        self.assertEqual(root.local_max_state_blocks, 3)
        self.assertEqual(root.kv_domain.local_max_capacity, 3)
        self.assertEqual(root.kv_domain.capacity, 2)
        self.assertEqual(root.target.state.num_blocks, 2)
        self.assertEqual(root.draft.state.num_blocks, 2)
        self.assertEqual(coordinator.local_maxima, [3])

        root.target.state.tensor.fill_(5)
        root.draft.state.tensor.fill_(7)
        root.clear_state_blocks((1,), domain=root.kv_domain)
        torch.testing.assert_close(root.target.state.tensor[1], torch.zeros(2))
        torch.testing.assert_close(root.draft.state.tensor[1], torch.zeros(1))

    def test_elastic_kv_and_exact_slot_domains_share_one_generation(self) -> None:
        coordinator = _CapacityCoordinator(4)
        backend = _IndependentBackend(
            memory_budget_bytes=68,
            capacity_coordinator=coordinator,
        )
        # Exact slots commit 3 * (8 + 4) = 36 bytes. The remaining
        # 32 bytes admit four 8-byte KV logical blocks.
        runtime = LiveRuntime(state_backend=backend)
        root = construct_live(runtime, _MixedDomainRoot)

        root.activate()

        assert root.state_plan is not None
        self.assertEqual(root.state_plan.num_blocks, 4)
        self.assertEqual(root.local_max_state_blocks, 4)
        self.assertEqual(coordinator.local_maxima, [4])
        self.assertEqual(root.kv.state.num_blocks, 4)
        self.assertEqual(tuple(root.kv.state.tensor.shape), (4, 2))
        self.assertEqual(root.continuation.slots.plan.num_blocks, 3)
        self.assertEqual(root.continuation.slots.local_max_capacity, 3)
        self.assertEqual(root.continuation.slots.capacity, 3)
        self.assertEqual(root.continuation.token.num_blocks, 3)
        self.assertEqual(root.continuation.length.num_blocks, 3)
        self.assertEqual(tuple(root.continuation.token.tensor.shape), (3,))
        self.assertEqual(tuple(root.continuation.length.tensor.shape), (3,))
        self.assertEqual(tuple(plan.num_blocks for plan in backend.plans), (4, 3))

        root.kv.state.tensor.fill_(9)
        root.continuation.token.tensor.fill_(11)
        root.continuation.length.tensor.fill_(13)
        root.clear_state_blocks((1,))
        torch.testing.assert_close(root.kv.state.tensor[1], torch.zeros(2))
        torch.testing.assert_close(
            root.continuation.token.tensor,
            torch.full((3,), 11, dtype=torch.int64),
        )
        root.clear_state_blocks((2,), domain=root.continuation.slots)
        torch.testing.assert_close(
            root.continuation.token.tensor[2],
            torch.zeros((), dtype=torch.int64),
        )
        torch.testing.assert_close(
            root.continuation.length.tensor[2],
            torch.zeros((), dtype=torch.int32),
        )

        root.close()
        self.assertFalse(root.continuation.slots.is_bound)
        self.assertFalse(root.kv.state.is_bound)
        self.assertFalse(root.continuation.token.is_bound)

    def test_exact_domains_need_no_kv_budget_or_capacity_coordination(self) -> None:
        runtime = LiveRuntime(state_backend=_IndependentBackend())
        root = construct_live(runtime, _TwoExactDomainRoot)

        root.activate()

        self.assertIsNone(root.state_plan)
        self.assertIsNone(root.local_max_state_blocks)
        self.assertEqual(root.left.domain.plan.num_blocks, 2)
        self.assertEqual(root.right.domain.plan.num_blocks, 2)

    def test_cross_domain_storage_overlap_fails_before_any_binding(self) -> None:
        root = construct_live(
            LiveRuntime(state_backend=_CrossDomainOverlappingBackend()),
            _TwoExactDomainRoot,
        )

        with self.assertRaises(StateTensorError) as rejected:
            root.activate()

        self.assertEqual(rejected.exception.code, "overlapping-state-storage")
        self.assertFalse(root.left.domain.is_bound)
        self.assertFalse(root.right.domain.is_bound)
        self.assertFalse(root.left.state.is_bound)
        self.assertFalse(root.right.state.is_bound)

    def test_graph_capture_failure_rolls_back_exact_domain_and_state(self) -> None:
        backend = _IndependentBackend()
        root = construct_live(
            LiveRuntime(
                state_backend=backend,
                graph_backend=_FailingGraphBackend(),
            ),
            _ExactDomainGraphRoot,
        )

        with self.assertRaisesRegex(RuntimeError, "graph capture failure"):
            root.activate()

        self.assertFalse(root.domain.is_bound)
        self.assertFalse(root.state.is_bound)
        self.assertEqual(root.phase, LiveModulePhase.CREATED)
        self.assertFalse(root._live_graphs["step"].prepared)
        self.assertEqual(len(backend.released), 1)
        self.assertEqual(backend.all_unbound_during_release, [True])

    def test_stateful_graph_requires_explicit_capture_block_selection(self) -> None:
        backend = _IndependentBackend()
        root = construct_live(
            LiveRuntime(
                state_backend=backend,
                graph_backend=_FailingGraphBackend(),
            ),
            _MissingStateGraphSelectionRoot,
        )

        with self.assertRaises(LiveModuleError) as rejected:
            root.activate()

        self.assertEqual(
            rejected.exception.code,
            "missing-captured-graph-state-blocks",
        )
        self.assertEqual(rejected.exception.context["graph_names"], ("step",))
        self.assertEqual(backend.plans, [])
        self.assertFalse(root.domain.is_bound)
        self.assertFalse(root.state.is_bound)
        self.assertEqual(root.phase, LiveModulePhase.CREATED)

    def test_multiple_elastic_domains_require_a_partition_policy(self) -> None:
        runtime = LiveRuntime(
            state_backend=_IndependentBackend(memory_budget_bytes=32),
        )
        root = construct_live(
            runtime,
            lambda: _PairRoot(
                _ElasticDomainLeaf("left"),
                _ElasticDomainLeaf("right"),
            ),
        )

        with self.assertRaises(LiveModuleError) as rejected:
            root.activate()

        self.assertEqual(
            rejected.exception.code,
            "ambiguous-elastic-state-domains",
        )
        self.assertFalse(root.left.domain.is_bound)
        self.assertFalse(root.right.domain.is_bound)

    def test_named_states_follow_deep_ordinary_module_composition(self) -> None:
        runtime = LiveRuntime(
            state_backend=_IndependentBackend(memory_budget_bytes=8),
        )
        root = construct_live(
            runtime,
            lambda: _NestedRoot(
                _StateLeaf(
                    role="deep",
                    block_shape=(1,),
                    storage_dtype=torch.float32,
                )
            ),
        )
        leaf = root.container[1]
        assert isinstance(leaf, _StateLeaf)

        self.assertEqual(
            tuple(path for path, _ in root.named_live_modules()),
            ("", "container.1"),
        )
        self.assertEqual(tuple(root.named_states(recurse=False)), ())
        self.assertEqual(
            tuple(leaf.named_states(recurse=False)),
            (("state", leaf.state),),
        )
        self.assertEqual(
            tuple(root.named_states()),
            (("container.1.state", leaf.state),),
        )
        self.assertIs(leaf.state.owner, leaf)
        self.assertEqual(leaf.state.name, "state")

        root.activate()

        plan = root.state_plan
        assert plan is not None
        self.assertEqual(
            tuple(lane.key for lane in plan.schema.lanes),
            (("container.1", "state"),),
        )

    def test_native_page_span_and_fixed_prefix_preserve_logical_capacity(
        self,
    ) -> None:
        runtime = LiveRuntime(
            state_backend=_IndependentBackend(memory_budget_bytes=56),
            # 8 fixed bytes plus three 16-byte logical blocks.
        )
        root = construct_live(
            runtime,
            lambda: _StateLeaf(
                role="paged",
                block_shape=(2,),
                storage_dtype=torch.float32,
                physical_blocks_per_logical_block=2,
                leading_physical_blocks=1,
            ),
        )

        root.activate()

        plan = root.state_plan
        assert plan is not None
        self.assertEqual(plan.schema.fixed_state_bytes, 8)
        self.assertEqual(plan.schema.bytes_per_simd_block, 16)
        self.assertEqual(plan.num_blocks, 3)
        self.assertEqual(plan.committed_state_bytes, 56)
        self.assertEqual(root.state.num_blocks, 3)
        self.assertEqual(tuple(root.state.tensor.shape), (7, 2))

        root.state.tensor.fill_(5)
        root.clear_state_blocks((1,))
        torch.testing.assert_close(root.state.tensor[:3], torch.full((3, 2), 5.0))
        torch.testing.assert_close(root.state.tensor[3:5], torch.zeros((2, 2)))
        torch.testing.assert_close(root.state.tensor[5:], torch.full((2, 2), 5.0))

    def test_process_device_does_not_supply_a_state_backend(self) -> None:
        runtime = LiveRuntime(device="cpu")
        root = construct_live(
            runtime,
            lambda: _PairRoot(
                _StateLeaf(
                    role="left",
                    block_shape=(2,),
                    storage_dtype=torch.float32,
                ),
                _StateLeaf(
                    role="right",
                    block_shape=(3,),
                    storage_dtype=torch.float16,
                ),
            ),
        )
        with self.assertRaises(LiveModuleError) as rejected:
            root.activate()
        self.assertEqual(rejected.exception.code, "missing-state-backend")

    def test_common_torch_backend_can_be_selected_explicitly(self) -> None:
        runtime = LiveRuntime(
            state_backend=TorchStateBackend("cpu", memory_budget_bytes=16),
        )
        root = construct_live(
            runtime,
            lambda: _StateLeaf(
                role="state",
                block_shape=(2,),
                storage_dtype=torch.float32,
            ),
        )
        root.activate()

        self.assertEqual(root.state.num_blocks, 2)
        self.assertEqual(root.state.tensor.device.type, "cpu")
        torch.testing.assert_close(
            root.state.tensor,
            torch.zeros((2, 2), dtype=torch.float32),
        )

    def test_runtime_clears_new_logical_blocks_across_state_lanes(self) -> None:
        runtime = LiveRuntime(
            state_backend=_IndependentBackend(memory_budget_bytes=48),
        )
        root = construct_live(
            runtime,
            lambda: _PairRoot(
                _StateLeaf(
                    role="left",
                    block_shape=(2,),
                    storage_dtype=torch.float32,
                ),
                _StateLeaf(
                    role="right",
                    block_shape=(2,),
                    storage_dtype=torch.float32,
                ),
            ),
        )
        left, right = root.left, root.right
        root.activate()

        root.clear_state_blocks([1, 1])

        torch.testing.assert_close(left.state.tensor[0], torch.ones(2))
        torch.testing.assert_close(left.state.tensor[1], torch.zeros(2))
        torch.testing.assert_close(left.state.tensor[2], torch.ones(2))
        torch.testing.assert_close(right.state.tensor[0], torch.full((2,), 2.0))
        torch.testing.assert_close(right.state.tensor[1], torch.zeros(2))
        torch.testing.assert_close(right.state.tensor[2], torch.full((2,), 2.0))

    def test_generation_snapshot_restores_exact_nonzero_initial_image(self) -> None:
        runtime = LiveRuntime(
            state_backend=_IndependentBackend(memory_budget_bytes=16),
        )
        root = construct_live(
            runtime,
            lambda: _StateLeaf(
                role="state",
                block_shape=(2,),
                storage_dtype=torch.float32,
            ),
        )
        root.activate()
        destination = root.state.tensor
        initial = destination.clone()
        snapshot = StateGenerationSnapshot(
            (root.state,),
            domain_blocks={None: tuple(range(root.state.num_blocks))},
        )

        destination.fill_(19.0)
        snapshot.restore()

        self.assertIs(root.state.tensor, destination)
        torch.testing.assert_close(destination, initial)
        snapshot.release()
        with self.assertRaises(StateTensorError) as released:
            snapshot.restore()
        self.assertEqual(
            released.exception.code,
            "released-state-generation-snapshot",
        )

    def test_generation_snapshot_rejects_a_missing_block_selection(self) -> None:
        runtime = LiveRuntime(
            state_backend=_IndependentBackend(memory_budget_bytes=16),
        )
        root = construct_live(
            runtime,
            lambda: _StateLeaf(
                role="state",
                block_shape=(2,),
                storage_dtype=torch.float32,
            ),
        )
        root.activate()

        with self.assertRaises(StateTensorError) as rejected:
            StateGenerationSnapshot(
                (root.state,),
                domain_blocks=None,  # type: ignore[arg-type]
            )

        self.assertEqual(
            rejected.exception.code,
            "missing-state-generation-snapshot-blocks",
        )

    def test_generation_snapshot_can_select_domain_blocks_and_fixed_prefix(
        self,
    ) -> None:
        domain = StateDomain(ExactStateCapacity(3))
        runtime = LiveRuntime(state_backend=_IndependentBackend())
        root = construct_live(
            runtime,
            lambda: _StateLeaf(
                role="state",
                block_shape=(2,),
                storage_dtype=torch.float32,
                domain=domain,
                physical_blocks_per_logical_block=2,
                leading_physical_blocks=1,
            ),
        )
        root.activate()
        destination = root.state.tensor
        initial = destination.clone()
        snapshot = StateGenerationSnapshot(
            (root.state,),
            domain_blocks={domain: (1,)},
        )

        destination.fill_(19.0)
        snapshot.restore()

        torch.testing.assert_close(destination[0], initial[0])
        torch.testing.assert_close(destination[3:5], initial[3:5])
        torch.testing.assert_close(destination[1:3], torch.full((2, 2), 19.0))
        torch.testing.assert_close(destination[5:7], torch.full((2, 2), 19.0))

    def test_root_compiles_one_schema_and_common_capacity(self) -> None:
        backend = _IndependentBackend(memory_budget_bytes=43)
        runtime = LiveRuntime(state_backend=backend)
        root = construct_live(
            runtime,
            lambda: _PairRoot(
                _StateLeaf(
                    role="left",
                    block_shape=(2,),
                    storage_dtype=torch.float32,
                ),
                _StateLeaf(
                    role="right",
                    block_shape=(3,),
                    storage_dtype=torch.float16,
                ),
            ),
        )

        root.activate()

        plan = root.state_plan
        assert plan is not None
        self.assertEqual(
            tuple(lane.key for lane in plan.schema.lanes),
            (("left", "state"), ("right", "state")),
        )
        self.assertEqual(plan.schema.bytes_per_simd_block, 14)
        self.assertEqual(plan.num_blocks, 3)
        self.assertEqual(plan.committed_state_bytes, 42)
        self.assertEqual(plan.residual_state_bytes, 1)
        self.assertEqual(len(backend.plans), 1)
        self.assertTrue(backend.all_unbound_during_allocation)
        self.assertEqual(root.left.state.num_blocks, 3)
        self.assertEqual(tuple(root.left.state.tensor.shape), (3, 2))
        self.assertEqual(tuple(root.right.state.tensor.shape), (3, 3))
        self.assertNotEqual(
            root.left.state.tensor.data_ptr(),
            root.right.state.tensor.data_ptr(),
        )

    def test_backend_coordinates_only_the_logical_block_count(self) -> None:
        coordinator = _CapacityCoordinator(3)
        backend = _IndependentBackend(
            memory_budget_bytes=70,
            capacity_coordinator=coordinator,
        )
        runtime = LiveRuntime(state_backend=backend)
        root = construct_live(
            runtime,
            lambda: _PairRoot(
                _StateLeaf(
                    role="left",
                    block_shape=(2,),
                    storage_dtype=torch.float32,
                ),
                _StateLeaf(
                    role="right",
                    block_shape=(3,),
                    storage_dtype=torch.float16,
                ),
            ),
        )

        root.activate()

        plan = root.state_plan
        assert plan is not None
        self.assertEqual(coordinator.local_maxima, [5])
        self.assertEqual(root.local_max_state_blocks, 5)
        self.assertEqual(plan.num_blocks, 3)
        self.assertEqual(plan.committed_state_bytes, 42)
        self.assertEqual(plan.residual_state_bytes, 28)
        self.assertEqual(tuple(root.left.state.tensor.shape), (3, 2))
        self.assertEqual(tuple(root.right.state.tensor.shape), (3, 3))

    def test_impossible_capacity_decision_fails_before_state_allocation(self) -> None:
        coordinator = _CapacityCoordinator(3)
        backend = _IndependentBackend(
            memory_budget_bytes=16,
            capacity_coordinator=coordinator,
        )
        runtime = LiveRuntime(state_backend=backend)
        root = construct_live(
            runtime,
            lambda: _StateLeaf(
                role="state",
                block_shape=(2,),
                storage_dtype=torch.float32,
            ),
        )

        with self.assertRaises(LiveModuleError) as rejected:
            root.activate()

        self.assertEqual(
            rejected.exception.code,
            "admitted-state-capacity-exceeds-local-maximum",
        )
        self.assertEqual(coordinator.local_maxima, [2])
        self.assertEqual(backend.plans, [])
        self.assertFalse(root.state.is_bound)
        self.assertIsNone(root.local_max_state_blocks)
        self.assertEqual(root.phase, LiveModulePhase.CREATED)

    def test_same_schema_arena_slices_are_disjoint_and_legal(self) -> None:
        backend = _CoalescingBackend(memory_budget_bytes=64)
        runtime = LiveRuntime(state_backend=backend)
        root = construct_live(
            runtime,
            lambda: _PairRoot(
                _StateLeaf(
                    role="first",
                    block_shape=(2,),
                    storage_dtype=torch.float32,
                ),
                _StateLeaf(
                    role="second",
                    block_shape=(2,),
                    storage_dtype=torch.float32,
                ),
            ),
        )
        root.activate()

        assert backend.arena is not None
        first = root.left.state.tensor
        second = root.right.state.tensor
        self.assertEqual(root.left.state.num_blocks, 4)
        self.assertEqual(first.untyped_storage().data_ptr(), backend.arena.data_ptr())
        self.assertEqual(second.untyped_storage().data_ptr(), backend.arena.data_ptr())
        self.assertEqual(
            second.data_ptr() - first.data_ptr(),
            first.numel() * first.element_size(),
        )

    def test_overlapping_state_spans_fail_before_any_binding(self) -> None:
        runtime = LiveRuntime(
            state_backend=_OverlappingBackend(memory_budget_bytes=64),
        )
        root = construct_live(
            runtime,
            lambda: _PairRoot(
                _StateLeaf(
                    role="first",
                    block_shape=(2,),
                    storage_dtype=torch.float32,
                ),
                _StateLeaf(
                    role="second",
                    block_shape=(2,),
                    storage_dtype=torch.float32,
                ),
            ),
        )

        with self.assertRaises(StateTensorError) as rejected:
            root.activate()

        self.assertEqual(rejected.exception.code, "overlapping-state-storage")
        self.assertFalse(root.left.state.is_bound)
        self.assertFalse(root.right.state.is_bound)
        self.assertEqual(root.phase, LiveModulePhase.CREATED)

    def test_incomplete_generation_fails_before_any_binding(self) -> None:
        runtime = LiveRuntime(
            state_backend=_IncompleteBackend(memory_budget_bytes=16),
        )
        root = construct_live(
            runtime,
            lambda: _PairRoot(
                _StateLeaf(
                    role="first",
                    block_shape=(1,),
                    storage_dtype=torch.float32,
                ),
                _StateLeaf(
                    role="second",
                    block_shape=(1,),
                    storage_dtype=torch.float32,
                ),
            ),
        )

        with self.assertRaises(StateTensorError) as rejected:
            root.activate()

        self.assertEqual(
            rejected.exception.code,
            "incomplete-simd-state-allocation",
        )
        self.assertFalse(root.left.state.is_bound)
        self.assertFalse(root.right.state.is_bound)

    def test_insufficient_budget_fails_before_backend_allocation(self) -> None:
        backend = _IndependentBackend(memory_budget_bytes=13)
        runtime = LiveRuntime(state_backend=backend)
        root = construct_live(
            runtime,
            lambda: _PairRoot(
                _StateLeaf(
                    role="left",
                    block_shape=(2,),
                    storage_dtype=torch.float32,
                ),
                _StateLeaf(
                    role="right",
                    block_shape=(3,),
                    storage_dtype=torch.float16,
                ),
            ),
        )

        with self.assertRaises(StateTensorError) as rejected:
            root.activate()

        self.assertEqual(rejected.exception.code, "insufficient-state-memory")
        self.assertEqual(backend.plans, [])
        self.assertFalse(root.left.state.is_bound)
        self.assertFalse(root.right.state.is_bound)

    def test_elastic_state_backend_requires_budget_and_typed_provider(self) -> None:
        missing_budget_runtime = LiveRuntime(state_backend=_IndependentBackend())
        missing_budget_root = construct_live(
            missing_budget_runtime,
            lambda: _StateLeaf(
                role="state",
                block_shape=(1,),
                storage_dtype=torch.float32,
            ),
        )
        with self.assertRaises(StateTensorError) as missing_budget:
            missing_budget_root.activate()
        self.assertEqual(
            missing_budget.exception.code,
            "missing-state-memory-budget",
        )

        missing_allocator_runtime = LiveRuntime(state_backend=object())
        missing_allocator_root = construct_live(
            missing_allocator_runtime,
            lambda: _StateLeaf(
                role="state",
                block_shape=(1,),
                storage_dtype=torch.float32,
            ),
        )
        with self.assertRaises(LiveModuleError) as missing_allocator:
            missing_allocator_root.activate()
        self.assertEqual(
            missing_allocator.exception.code,
            "missing-state-backend",
        )

    def test_malformed_lane_geometry_fails_before_binding(self) -> None:
        runtime = LiveRuntime(
            state_backend=_WrongShapeBackend(memory_budget_bytes=8),
        )
        root = construct_live(
            runtime,
            lambda: _StateLeaf(
                role="state",
                block_shape=(1,),
                storage_dtype=torch.float32,
            ),
        )

        with self.assertRaises(StateTensorError) as rejected:
            root.activate()

        self.assertEqual(rejected.exception.code, "invalid-state-value-shape")
        self.assertFalse(root.state.is_bound)

    def test_state_declaration_rejects_nonphysical_block_facts(self) -> None:
        with self.assertRaises(StateTensorError) as bad_shape:
            StateTensor(
                role="bad-shape",
                requirement=True,
                block_shape=(2, 0),
                storage_dtype=torch.float32,
            )
        self.assertEqual(bad_shape.exception.code, "invalid-state-block-shape")

        with self.assertRaises(StateTensorError) as bad_dtype:
            StateTensor(
                role="bad-dtype",
                requirement=True,
                block_shape=(2,),
                storage_dtype="float32",  # type: ignore[arg-type]
            )
        self.assertEqual(
            bad_dtype.exception.code,
            "invalid-state-storage-dtype",
        )

    def test_schema_rejects_duplicate_state_or_lane_identity(self) -> None:
        runtime = LiveRuntime()
        owner = construct_live(
            runtime,
            lambda: _StateLeaf(
                role="only",
                block_shape=(1,),
                storage_dtype=torch.float32,
            ),
        )
        first = SIMDStateLane("", "state", owner.state)
        with self.assertRaises(StateTensorError) as duplicate_state:
            compile_simd_state_schema(
                (first, SIMDStateLane("other", "state", owner.state))
            )
        self.assertEqual(duplicate_state.exception.code, "duplicate-simd-state")

        other = construct_live(
            runtime,
            lambda: _StateLeaf(
                role="other",
                block_shape=(1,),
                storage_dtype=torch.float32,
            ),
        )
        with self.assertRaises(StateTensorError) as duplicate_key:
            compile_simd_state_schema((first, SIMDStateLane("", "state", other.state)))
        self.assertEqual(
            duplicate_key.exception.code,
            "duplicate-simd-state-lane-key",
        )

    def test_close_drops_the_plan_and_unbinds_every_lane(self) -> None:
        backend = _IndependentBackend(memory_budget_bytes=16)
        runtime = LiveRuntime(
            state_backend=backend,
        )
        root = construct_live(
            runtime,
            lambda: _StateLeaf(
                role="state",
                block_shape=(2,),
                storage_dtype=torch.float32,
            ),
        )
        root.activate()

        root.close()

        self.assertIsNone(root.state_plan)
        self.assertFalse(root.state.is_bound)
        self.assertEqual(root.phase, LiveModulePhase.CLOSED)
        self.assertEqual(len(backend.released), 1)
        self.assertEqual(backend.all_unbound_during_release, [True])


if __name__ == "__main__":
    unittest.main()
