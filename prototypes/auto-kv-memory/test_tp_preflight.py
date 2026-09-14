"""CPU ownership contract for retiring trial draft graphs and scratch bindings."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


class DraftRetirement(unittest.TestCase):
    def test_restore_native_runnable_without_retaining_packets(self):
        path = Path(__file__).with_name("tp_preflight_worker.py")
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


if __name__ == "__main__":
    unittest.main()
