"""A vanished container PID is not permission to admit an unknown device owner."""

import ast
from pathlib import Path
import unittest

source = Path(__file__).parents[1] / "device-service/admit_subset.py"
node = next(
    n
    for n in ast.parse(source.read_text()).body
    if isinstance(n, ast.FunctionDef) and n.name == "previously_owned"
)
namespace = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
owned = namespace["previously_owned"]


class OwnerIdentity(unittest.TestCase):
    def test_missing_container_pid_keeps_only_known_live_identity(self):
        known = {100: (10, "start", 50.0)}
        self.assertTrue(owned(100, 0, known, "start", 51.0))
        self.assertFalse(owned(101, 0, known, None, 51.0))
        self.assertFalse(owned(100, 0, known, "reused", 51.0))
        self.assertFalse(owned(100, 11, known, "start", 51.0))

    def test_departed_process_grace_is_bounded(self):
        known = {100: (10, "start", 50.0)}
        self.assertTrue(owned(100, 0, known, None, 51.0))
        self.assertFalse(owned(100, 0, known, None, 66.0))
