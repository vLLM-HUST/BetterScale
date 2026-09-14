"""Protect the MP wrapper seam without constructing a model or NPU context."""

import unittest
from types import SimpleNamespace
from early_rpc import execute_with_budget


class WrapperTests(unittest.TestCase):
    def test_cache_precedes_worker_and_budget_is_unchanged(self):
        calls = []
        schedule, budget, output = object(), object(), object()

        def cache(value):
            self.assertIs(value, schedule)
            calls.append("cache")

        def execute(value, early_budget):
            self.assertIs(value, schedule)
            self.assertIs(early_budget, budget)
            calls.append("model")
            return output

        wrapper = SimpleNamespace(
            _apply_mm_cache=cache,
            worker=SimpleNamespace(execute_model=execute),
        )
        self.assertIs(execute_with_budget(wrapper, schedule, budget), output)
        self.assertEqual(calls, ["cache", "model"])
