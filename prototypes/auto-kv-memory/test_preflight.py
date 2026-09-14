"""CPU lifetime contract for the disposable Ascend catalog; no NPU imports."""

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


class TrialCleanup(unittest.TestCase):
    def test_retire_both_catalogs_and_metadata_only_after_sync(self):
        path = Path(__file__).with_name("preflight_worker.py")
        node = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "clear_trial_graphs"
        )

        @dataclasses.dataclass
        class Params:
            events: dict
            workspaces: dict
            handles: dict
            attn_params: dict

        params = Params(
            {6: [object()]}, {6: object()}, {6: [object()]}, {6: [object()]}
        )
        pair = NS(
            catalogs=[{6: object()}, {6: object()}],
            packets=[{6: object()}, {6: object()}],
        )
        wrapper = NS(concrete_aclgraph_entries={6: object()}, _decode_pair=pair)
        observations = []

        def sync():
            observations.append(
                bool(wrapper.concrete_aclgraph_entries and pair.catalogs[0])
            )

        namespace = dict(
            dataclasses=dataclasses,
            torch=NS(npu=NS(synchronize=sync)),
            acl_graph=NS(_graph_params=params),
        )
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        namespace["clear_trial_graphs"]([wrapper])
        self.assertEqual(observations, [True])
        self.assertFalse(hasattr(wrapper, "_decode_pair"))
        self.assertEqual(pair.catalogs, [{}, {}])
        self.assertEqual(pair.packets, [{}, {}])
        self.assertEqual(wrapper.concrete_aclgraph_entries, {})
        self.assertEqual(params, Params({6: []}, {6: None}, {6: []}, {6: []}))
        self.assertIsNone(namespace["acl_graph"]._graph_params)
        namespace["clear_trial_graphs"]([wrapper])
        self.assertEqual(observations, [True, False])


if __name__ == "__main__":
    unittest.main()
