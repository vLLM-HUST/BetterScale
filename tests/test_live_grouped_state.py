# SPDX-License-Identifier: Apache-2.0
# From LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0, tests/test_grouped_state_backend.py
# Imports relocated; construction helper inlined to avoid upstream tests dependency.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from betterscale.live import StateTensor, StateDomain, ExactStateCapacity, SIMDStateLane, compile_simd_state_schema
from betterscale.live.runtime.state_backend import validate_state_generation_realization
from betterscale.live.runtime.grouped_state import GroupedStateBackend


def schema(*, layers=3, domain=None, dtype=torch.float32, leading=1):
    states = [StateTensor(role='swa', requirement=('swa', 8), block_shape=(8,),
        storage_dtype=dtype, domain=domain, leading_physical_blocks=leading,
        physical_blocks_per_logical_block=2) for _ in range(layers)]
    return compile_simd_state_schema(tuple(SIMDStateLane(f'layer.{i}', 'swa', s)
        for i, s in enumerate(states)), domain=domain)


def test_cross_layer_views_keep_all_seats_and_sentinels_without_aliasing():
    spec = schema(domain=StateDomain(ExactStateCapacity(3)))
    backend = GroupedStateBackend('cpu', memory_budget_bytes=4096)
    result = backend.realize_state((spec,))
    validated = validate_state_generation_realization((spec,), result)
    allocation = result.domains[0].allocation
    assert len(allocation.backings) == 1
    a, b, c = validated[0][1]
    assert a.shape == b.shape == c.shape == (7, 8)
    assert all(t.is_contiguous() for t in (a, b, c))
    assert a.untyped_storage()._cdata == b.untyped_storage()._cdata
    assert b.storage_offset() > 0
    a.fill_(7); b[3:5].fill_(9)
    assert torch.all(a == 7) and torch.all(c == 0)
    assert torch.all(b[:3] == 0) and torch.all(b[5:] == 0)
    span = backend.backing_span(spec.lanes[1].state)
    assert b.data_ptr() == span.backing.address + span.offset
    release = span.backing.acquire()
    with pytest.raises(RuntimeError, match='leases'):
        backend.release_state(result)
    assert not span.backing.closed
    release(); release()
    backend.release_state(result)
    assert span.backing.closed
    assert torch.all(a == 7)  # Retained Tensor views still own physical storage.


def test_grouping_does_not_merge_dtypes_requirements_or_domains():
    domain = StateDomain(ExactStateCapacity(2))
    first, second = schema(domain=domain), schema(domain=domain, dtype=torch.int64)
    combined = compile_simd_state_schema(first.lanes + tuple(SIMDStateLane("scale."+lane.module_path, lane.state_name, lane.state) for lane in second.lanes), domain=domain)
    other = schema(domain=StateDomain(ExactStateCapacity(2)))
    backend = GroupedStateBackend('cpu', memory_budget_bytes=8192)
    result = backend.realize_state((combined, other))
    validate_state_generation_realization((combined, other), result)
    assert [len(d.allocation.backings) for d in result.domains] == [2, 1]
    backend.release_state(result)


def test_padding_is_charged_before_elastic_admission():
    exact = schema(layers=1, domain=StateDomain(ExactStateCapacity(1)))
    elastic = schema(layers=3)
    backend = GroupedStateBackend('cpu', allocation_alignment=2048, memory_budget_bytes=6144)
    result = backend.realize_state((exact, elastic))
    assert sum(d.allocated_state_bytes for d in result.domains) <= 6144
    admitted = result.domains[1].plan.num_blocks
    from betterscale.live.core.state_tensor import compile_exact_simd_state_plan
    too_large = compile_exact_simd_state_plan(elastic, num_blocks=admitted+1)
    assert backend._allocation_bytes(too_large) > 4096
    backend.release_state(result)


def test_exact_padding_over_budget_rejects_before_allocation():
    backend = GroupedStateBackend('cpu', allocation_alignment=2048, memory_budget_bytes=1024)
    with pytest.raises(Exception, match='exceed'):
        backend.realize_state((schema(layers=1, domain=StateDomain(ExactStateCapacity(1))),))
    assert not backend._spans


def test_partial_domain_failure_releases_all_backings():
    class Failing(GroupedStateBackend):
        calls = 0
        held = []
        def _supply_backing(self, plan):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('injected allocation failure')
            b = super()._supply_backing(plan); self.held.append(b); return b
    backend = Failing('cpu', memory_budget_bytes=8192)
    specs = tuple(schema(domain=StateDomain(ExactStateCapacity(2))) for _ in range(2))
    with pytest.raises(RuntimeError, match='injected'):
        backend.realize_state(specs)
    assert not backend._spans and all(b.closed for b in backend.held)
