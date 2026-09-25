"""CPU tests exercise real LiveInference allocation/binding, not a fake backend."""

import unittest
import json
from pathlib import Path

import torch
from torch import nn
from livemodule import LiveRuntime, StateTensorError, TorchStateBackend, live_runtime
from livemodule.runtime.grouped_state import GroupedStateBackend
from state import AttentionState, Capacity, GDNState, Geometry, QwenStateRoot


SMALL = Geometry(("linear_attention", "full_attention"), 1, 4, 1, 2, 4, 4, 4)


class Consumer(nn.Module):
    def __init__(self, abi):
        super().__init__()
        self.state_binding_abi = abi
        self.kv_cache = []


class CountBackend(TorchStateBackend):
    def __init__(self, budget=1 << 20):
        super().__init__("cpu", memory_budget_bytes=budget)
        self.allocations = self.releases = 0

    def _allocate_state_domain(self, plan):
        self.allocations += 1
        return super()._allocate_state_domain(plan)

    def _release_state_domain(self, realization):
        self.releases += 1
        return super()._release_state_domain(realization)


def root(capacity=None, backend=None, kind=QwenStateRoot):
    backend = backend or CountBackend()
    with live_runtime(LiveRuntime(device="cpu", state_backend=backend)):
        result = kind(SMALL, capacity or Capacity(2, 4, page_tokens=8, token_pages=7))
    return result, backend


def consumers(r):
    target = {name: Consumer(leaf.binding_abi) for name, leaf in r.target.items()}
    return target, Consumer(r.draft.binding_abi)


class StateTests(unittest.TestCase):
    def test_full_small_model_geometry_and_budget(self):
        config = json.loads(
            Path(__file__).with_name("qwen35-0.8b-text-config.json").read_text()
        )
        geometry = Geometry.from_config(config)
        backend = CountBackend(512 << 20)
        with live_runtime(LiveRuntime(device="cpu", state_backend=backend)):
            r = QwenStateRoot(geometry, Capacity(4, 5, token_pages=32))
        self.assertEqual(backend.allocations, 0)
        r.activate()
        try:
            numerical = [
                s
                for name, s in r.named_states()
                if not name.startswith("continuation.")
            ]
            self.assertEqual(len(numerical), 50)
            resident_bytes = sum(
                s.tensor.numel() * s.tensor.element_size()
                for s in numerical
                if s.domain is r.residents
            )
            page_bytes = sum(
                s.tensor.numel() * s.tensor.element_size()
                for s in numerical
                if s.domain is r.pages
            )
            self.assertEqual(
                resident_bytes, 5 * 18 * (6144 * 5 * 2 + 3 * 16 * 128 * 128 * 4)
            )
            self.assertEqual(page_bytes, 32 * 128 * 7 * 2 * 2 * 256 * 2)
        finally:
            r.close()

    def test_declaration_does_not_allocate(self):
        r, backend = root()
        self.assertEqual(backend.allocations, 0)
        self.assertFalse(any(s.is_bound for _, s in r.named_states()))
        with self.assertRaises(StateTensorError):
            _ = r.target["0"].conv.tensor
        r.close()
        self.assertEqual(backend.releases, 0)

    def test_real_activation_has_two_domains_and_exclusive_lanes(self):
        r, backend = root()
        r.activate()
        self.assertEqual(backend.allocations, 2)
        self.assertEqual(r.residents.capacity, 4)
        self.assertEqual(r.pages.capacity, 7)
        self.assertEqual(r.target["0"].conv.tensor.shape, (4, 5, 16))
        self.assertEqual(r.target["0"].recurrent.tensor.shape, (12, 2, 4, 4))
        self.assertEqual(r.target["1"].key.tensor.shape, (7, 8, 1, 4))
        values = [s.tensor for _, s in r.named_states()]
        spans = sorted(
            (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in values
        )
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])))
        self.assertTrue(torch.all(r.continuation.selection.tensor == 1))
        self.assertTrue(torch.all(r.continuation.proposal.tensor == -1))
        r.close()
        self.assertEqual(backend.releases, 2)
        self.assertFalse(any(s.is_bound for _, s in r.named_states()))

    def test_execution_capacity_does_not_resize_resident_state(self):
        receipts = []
        for execution in (1, 4):
            r, _ = root(Capacity(execution, 4, page_tokens=8, token_pages=7))
            r.activate()
            receipts.append([(n, tuple(s.tensor.shape)) for n, s in r.named_states()])
            r.close()
        self.assertEqual(*receipts)

    def test_more_residents_do_not_reserve_more_token_pages(self):
        sizes = []
        for residents in (4, 6):
            r, _ = root(Capacity(2, residents, page_tokens=8, token_pages=7))
            r.activate()
            sizes.append(
                (r.target["1"].key.tensor.numel(), r.target["0"].conv.tensor.numel())
            )
            r.close()
        self.assertEqual(sizes[0][0], sizes[1][0])
        self.assertEqual(sizes[1][1] * 2, sizes[0][1] * 3)

    def test_elastic_pages_pay_fixed_resident_cost_first(self):
        admitted = []
        for residents in (4, 6):
            r, _ = root(Capacity(2, residents, page_tokens=8), CountBackend(16384))
            r.activate()
            self.assertEqual(r.residents.capacity, residents)
            admitted.append(r.pages.capacity)
            r.close()
        self.assertGreater(admitted[0], admitted[1])

    def test_candidate_addressing_and_seat_reset_are_distinct(self):
        r, _ = root()
        r.activate()
        g = r.target["0"]
        g.recurrent.tensor.fill_(9)
        g.conv.tensor.fill_(7)
        # StateTensor lowers resident1 into candidate rows3,4,5, not row1.
        g.recurrent.clear_logical_block(1)
        g.conv.clear_logical_block(1)
        self.assertTrue(torch.all(g.recurrent.tensor[3:6] == 0))
        self.assertTrue(torch.all(g.recurrent.tensor[:3] == 9))
        self.assertTrue(torch.all(g.recurrent.tensor[6:] == 9))
        self.assertTrue(torch.all(g.conv.tensor[0] == 7))
        r.close()

    def test_consumer_borrows_only_during_live_generation(self):
        r, _ = root()
        target, draft = consumers(r)
        r.attach_consumers(target, draft)
        self.assertEqual(target["0"].kv_cache, [])
        r.activate()
        self.assertIs(target["0"].kv_cache[0], r.target["0"].conv.tensor)
        self.assertIs(draft.kv_cache[0], r.draft.key.tensor)
        target["0"].kv_cache[1][5].fill_(11)
        self.assertTrue(torch.all(r.target["0"].recurrent.tensor[5] == 11))
        r.close()
        self.assertEqual(target["0"].kv_cache, [])
        self.assertEqual(draft.kv_cache, [])

    def test_native_cache_or_unknown_abi_rejected_atomically(self):
        r, _ = root()
        target, draft = consumers(r)
        draft.kv_cache = [torch.ones(1)]
        with self.assertRaises(ValueError):
            r.attach_consumers(target, draft)
        self.assertIsNone(r.target["0"]._consumer)
        draft.kv_cache = []
        target["0"].state_binding_abi = "native-common-block-id"
        with self.assertRaises(ValueError):
            r.attach_consumers(target, draft)
        self.assertIsNone(r.target["1"]._consumer)
        r.close()

    def test_late_native_allocation_fails_and_revokes_earlier_borrows(self):
        r, backend = root()
        target, draft = consumers(r)
        r.attach_consumers(target, draft)
        draft.kv_cache = [torch.ones(1)]
        with self.assertRaises(RuntimeError):
            r.activate()
        self.assertEqual(target["0"].kv_cache, [])
        self.assertFalse(any(s.is_bound for _, s in r.named_states()))
        self.assertEqual(backend.allocations, backend.releases)

    def test_initialization_failure_rolls_back_all_bound_state(self):
        class Broken(QwenStateRoot):
            def _initialize_live_generation(self):
                raise RuntimeError("injected before publication")

        r, b = root(kind=Broken)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            r.activate()
        self.assertEqual(b.allocations, b.releases)
        self.assertFalse(r.residents.is_bound)

    def test_budget_failure_leaves_no_bound_generation(self):
        r, b = root(backend=CountBackend(1))
        with self.assertRaises(Exception):
            r.activate()
        self.assertFalse(any(s.is_bound for _, s in r.named_states()))
        self.assertEqual(b.allocations, b.releases)

    def test_grouped_backend_still_produces_disjoint_semantic_lanes(self):
        r, _ = root(backend=GroupedStateBackend("cpu", memory_budget_bytes=65536))
        r.activate()
        r.target["1"].key.tensor.fill_(17)
        self.assertTrue(torch.all(r.draft.key.tensor == 0))
        r.close()

    def test_invalid_capacity_and_geometry_fail_before_allocation(self):
        for args in [(4, 2), (True, 4), (2, 0)]:
            with self.assertRaises(ValueError):
                Capacity(*args)
        with self.assertRaises(ValueError):
            Capacity(2, 4, speculative_tokens=3)
        with self.assertRaises(ValueError):
            Geometry.from_config({"model_type": "qwen3_5_moe_text"})


if __name__ == "__main__":
    unittest.main()
