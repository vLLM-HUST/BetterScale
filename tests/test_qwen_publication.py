"""Ownership tests independent of accelerator availability."""

from contextlib import nullcontext
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch
import numpy as np

from betterscale.patches.qwen_gdn.publication import Frame


class Publication(unittest.TestCase):
    def test_fill_replaces_warm_slots_and_empty_suffix(self):
        h = {
            "cu": np.empty(10, dtype=np.int64),
            "conv_cu": np.empty(10, dtype=np.int32),
            "slots": np.empty(9, dtype=np.int64),
            "conv_slots": np.empty((9, 1), dtype=np.int32),
            "conv_initial": np.empty(9, dtype=bool),
            "state": np.empty((9, 2), dtype=np.int64),
            **{
                s: np.empty(((1536 + s - 1) // s + 7, 2), dtype=np.int64)
                for s in (64, 256, 1216)
            },
        }
        frame = Frame.__new__(Frame)
        frame.metas = {
            0: NS(host=h, tokens=1536, requests=9, indices={64: 0, 256: 0, 1216: 0})
        }

        class CPU:
            def __init__(self, values):
                self.values = np.array(values)

            def __getitem__(self, index):
                return CPU(self.values[index])

            def numpy(self):
                return self.values

        frame.fill(
            0, NS(seq_lens_cpu=CPU([1025, 1472])), (1, 1472), np.array([[7], [3]])
        )
        self.assertEqual(h["cu"].tolist(), [0, 1] + [1473] * 8)
        self.assertEqual(h["slots"].tolist(), [7, 3] + [-1] * 7)
        self.assertEqual(h["conv_initial"].tolist(), [True] + [False] * 8)
        self.assertEqual(h[1216].tolist()[:3], [[0, 0], [1, 0], [1, 1]])
        self.assertEqual(h[1216].tolist()[3:], [[8, 0]] * 6)
        frame.fill(0, NS(seq_lens_cpu=CPU([17])), (17,), np.array([[11]]))
        self.assertEqual(h["state"].tolist(), [[11, 0]] + [[-1, 0]] * 8)
        self.assertEqual(h["cu"].tolist(), [0] + [17] * 9)
        self.assertEqual(h[1216].tolist(), [[0, 0]] + [[8, 0]] * 8)
        # align snapshots live at the last logical block, not column zero.
        # Exact boundaries, crossing, a warm prefix and dummy zero length use
        # the same clamped floor division as the donor's GPU gather.
        frame.fill(
            0,
            NS(seq_lens_cpu=CPU([0, 512, 513, 1024, 1025])),
            (1, 512, 1, 17, 1),
            np.array(
                [[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42], [50, 51, 52]]
            ),
            aligned_block_size=512,
        )
        self.assertEqual(h["slots"].tolist(), [10, 20, 31, 41, 52] + [-1] * 4)
        self.assertEqual(
            h["conv_initial"].tolist(), [False, False, True, True, True] + [False] * 4
        )
        self.assertEqual(h["conv_slots"][:, 0].tolist(), h["slots"].tolist())
        self.assertEqual(h["state"][:, 0].tolist(), h["slots"].tolist())

    def test_dma_source_and_device_consumer_are_different_fences(self):
        log = []

        class Event:
            def __init__(self, name):
                self.name = name

            def query(self):
                return False

            def synchronize(self):
                log.append(("host_wait", self.name))

            def record(self, stream):
                log.append(("record", self.name, stream.name))

        class Stream:
            def __init__(self, name):
                self.name = name

            def wait_event(self, event):
                log.append(("device_wait", self.name, event.name))

        frame = Frame.__new__(Frame)
        frame.uploaded, frame.consumed = Event("upload"), Event("consume")
        frame.has_upload = frame.has_consumer = True
        frame.stream, compute = Stream("ingress"), Stream("compute")
        frame.host = object()
        frame.device = NS(
            copy_=lambda src, non_blocking: log.append(
                ("copy", src is frame.host, non_blocking)
            )
        )
        fake = NS(
            npu=NS(stream=lambda s: nullcontext(), current_stream=lambda: compute)
        )
        with patch.dict(sys.modules, {"torch": fake}):
            frame.acquire()
            frame.publish()
            frame.release()
        self.assertEqual(
            log,
            [
                ("host_wait", "upload"),
                ("device_wait", "ingress", "consume"),
                ("copy", True, True),
                ("record", "upload", "ingress"),
                ("device_wait", "compute", "upload"),
                ("record", "consume", "compute"),
            ],
        )


class GraphBanks(unittest.TestCase):
    def test_bank_keys_and_fia_resources_are_distinct(self):
        import types
        from dataclasses import dataclass
        from betterscale.patches.qwen_gdn.publication import install, graph_resources

        @dataclass(frozen=True)
        class Descriptor:
            num_tokens: int
            num_reqs: int

        class Dispatcher:
            def _create_padded_batch_descriptor(self, n):
                return Descriptor(n, min(n, 8))

            def initialize_cudagraph_keys(self):
                self.cudagraph_keys = {
                    2: {self._create_padded_batch_descriptor(n) for n in (1, 8, 1536)}
                }

        class Runner:
            def __init__(self):
                self.cudagraph_dispatcher = Dispatcher()

            def _warmup_and_capture(self, desc):
                return self._determine_batch_execution_and_padding()

            def _determine_batch_execution_and_padding(self):
                return self.cudagraph_dispatcher._create_padded_batch_descriptor(1536)

            def _build_attention_metadata(self):
                pass

            def _model_forward(self):
                pass

        acl = types.ModuleType("vllm_ascend.compilation.acl_graph")
        acl.GraphParams = NS
        original = acl._graph_params = object()
        compilation = types.ModuleType("vllm_ascend.compilation")
        compilation.acl_graph = acl
        ascend = types.ModuleType("vllm_ascend")
        ascend.compilation = compilation
        modules = {
            "vllm.config": NS(CUDAGraphMode=NS(FULL=2)),
            "vllm.forward_context": NS(
                BatchDescriptor=Descriptor, get_forward_context=lambda: None
            ),
            "vllm.v1.cudagraph_dispatcher": NS(CudagraphDispatcher=Dispatcher),
            "vllm_ascend": ascend,
            "vllm_ascend.compilation": compilation,
            "vllm_ascend.compilation.acl_graph": acl,
            "vllm_ascend.worker.model_runner_v1": NS(NPUModelRunner=Runner),
            "vllm_ascend.ops.gdn_attn_builder": NS(
                AscendGDNAttentionMetadataBuilder=type("Builder", (), {})
            ),
        }
        with patch.dict(sys.modules, modules):
            install()
            runner = Runner()
            runner.cudagraph_dispatcher.initialize_cudagraph_keys()
            keys = runner.cudagraph_dispatcher.cudagraph_keys[2]
            self.assertEqual(len(keys), 6)
            for desc in keys:
                result = runner._warmup_and_capture(desc)
                self.assertEqual(result.bank, desc.bank)
                self.assertIsNone(runner._owned_capture_bank)
            runner._owned_next_bank = 1
            self.assertEqual(runner._determine_batch_execution_and_padding().bank, 1)
            with graph_resources(runner, 0, 1536):
                first = acl._graph_params
                first.handles[1536].append("bank0")
            with self.assertRaisesRegex(RuntimeError, "probe"):
                with graph_resources(runner, 1, 1536):
                    self.assertIsNot(acl._graph_params, first)
                    self.assertEqual(acl._graph_params.handles[1536], [])
                    raise RuntimeError("probe")
            self.assertIs(acl._graph_params, original)
            with graph_resources(runner, 0, 1536):
                self.assertIs(acl._graph_params, first)
