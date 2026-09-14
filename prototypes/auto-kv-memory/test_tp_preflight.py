"""CPU ownership contract for retiring trial draft graphs and scratch bindings."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


class DraftRetirement(unittest.TestCase):
    def test_restore_native_runnable_without_retaining_packets(self):
        path = Path(__file__).with_name("tp_physical_worker.py")
        cls = next(
            n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "retire_trial_program"
        )
        original, first, second, binding = object(), object(), object(), object()
        manager = NS(
            original=original,
            decode_graphs={1: NS(graph=first)},
            query_graphs={1: NS(graph=second)},
        )
        drafter = NS(_runnable=manager, saved="scratch", transient="scratch")
        worker = NS(
            _exact_draft_graph=manager,
            model_runner=NS(drafter=drafter),
            _preflight_retired_graphs=[],
            _trial_draft_bindings={
                "saved": (True, binding),
                "transient": (False, None),
            },
        )
        observed = []
        ns = dict(
            torch=NS(
                npu=NS(synchronize=lambda: observed.append(bool(manager.decode_graphs)))
            )
        )
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
        ns["retire_trial_program"](worker)
        self.assertEqual(observed, [True])
        self.assertEqual(worker._preflight_retired_graphs, [first, second])
        self.assertIs(drafter._runnable, original)
        self.assertIs(drafter.saved, binding)
        self.assertFalse(hasattr(drafter, "transient"))
        self.assertFalse(hasattr(worker, "_exact_draft_graph"))
        self.assertFalse(hasattr(worker, "_trial_draft_bindings"))
        self.assertEqual(manager.decode_graphs, {})
        self.assertEqual(manager.query_graphs, {})
        ns["retire_trial_program"](worker)
        self.assertEqual(observed, [True])


class DummyEntry(unittest.TestCase):
    def entry(self, events):
        path = Path(__file__).with_name("tp_preflight_worker.py")
        cls = next(
            n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
        )

        class Physical:
            def __init__(self, config, *args, **kwargs):
                events.append(("native_init", config, args, kwargs))

        ns = dict(
            TPPhysicalWorker=Physical,
            install_dummy_layout=lambda: events.append("dummy_hook"),
        )
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), ns)
        return ns["TPPreflightWorker"]

    def test_real_load_rejected_before_hook_or_native_init(self):
        events = []
        entry = self.entry(events)
        with self.assertRaises(AssertionError):
            entry(NS(load_config=NS(load_format="auto")))
        self.assertEqual(events, [])

    def test_dummy_hook_precedes_native_initialization(self):
        events = []
        config = NS(load_config=NS(load_format="dummy"))
        self.entry(events)(config, "worker", rank=3)
        self.assertEqual(
            events, ["dummy_hook", ("native_init", config, ("worker",), {"rank": 3})]
        )


if __name__ == "__main__":
    unittest.main()
