"""CPU ownership checks; native shape/state correctness is checked on TP2."""

import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch


@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int
    num_reqs: int | None = None


class PartitionOwnership(unittest.TestCase):
    def setUp(self):
        self.forward = types.ModuleType("vllm.forward_context")
        self.forward.BatchDescriptor = BatchDescriptor
        self.acl = types.ModuleType("vllm_ascend.compilation.acl_graph")
        self.acl.GraphParams = types.SimpleNamespace
        self.original = object()
        self.acl._graph_params = self.original
        ascend = types.ModuleType("vllm_ascend")
        compilation = types.ModuleType("vllm_ascend.compilation")
        ascend.compilation = compilation
        compilation.acl_graph = self.acl
        self.modules = patch.dict(
            sys.modules,
            {
                "vllm.forward_context": self.forward,
                "vllm_ascend": ascend,
                "vllm_ascend.compilation": compilation,
                "vllm_ascend.compilation.acl_graph": self.acl,
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location(
            "partition_graphs_test_subject",
            Path(__file__).with_name("partition_graphs.py"),
        )
        self.subject = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.subject
        self.addCleanup(sys.modules.pop, spec.name)
        spec.loader.exec_module(self.subject)
        self.runner = types.SimpleNamespace()

    def test_same_total_and_request_count_do_not_alias(self):
        a = self.subject.descriptor_for((1, 1, 1024, 1022))
        b = self.subject.descriptor_for((1, 1, 1022, 1024))
        self.assertEqual((a.num_tokens, a.num_reqs), (b.num_tokens, b.num_reqs))
        self.assertEqual(len({a, b}), 2)
        with self.subject.graph_resources(self.runner, a):
            first = self.acl._graph_params
            first.handles[2048].append("first")
        with self.subject.graph_resources(self.runner, b):
            self.assertIsNot(self.acl._graph_params, first)
            self.assertEqual(self.acl._graph_params.handles[2048], [])
        with self.subject.graph_resources(self.runner, a):
            self.assertIs(self.acl._graph_params, first)
            self.assertEqual(first.handles[2048], ["first"])
        self.assertIs(self.acl._graph_params, self.original)

    def test_exception_restores_native_bank(self):
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with self.subject.graph_resources(
                self.runner, self.subject.descriptor_for((2048,))
            ):
                raise RuntimeError("probe")
        self.assertIs(self.acl._graph_params, self.original)
        self.assertFalse(self.runner._partition_resources_active)

    def test_native_decode_does_not_allocate_partition_bank(self):
        with self.subject.graph_resources(self.runner, BatchDescriptor(4, 4)):
            self.assertIs(self.acl._graph_params, self.original)
        self.assertFalse(hasattr(self.runner, "_partition_resources"))


if __name__ == "__main__":
    unittest.main()
