"""CPU checks for the fail-closed native FIA preload boundary."""

import ctypes
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from betterscale.patches.qwen_fia import check_library


class WaveFIA(unittest.TestCase):
    def test_reject_queue_and_unqualified_library(self):
        with patch.dict(os.environ, {"TASK_QUEUE_ENABLE": "1"}):
            with self.assertRaisesRegex(ValueError, "TASK_QUEUE_ENABLE"):
                check_library()
        with patch.dict(
            os.environ, {"TASK_QUEUE_ENABLE": "0", "BETTERSCALE_FIA_LIBRARY": ""}
        ):
            with self.assertRaisesRegex(ValueError, "qualified native planner"):
                check_library()

    def test_preload_must_match_loaded_library(self):
        callback = ctypes.CFUNCTYPE(ctypes.c_int)
        entry = callback(lambda: 0)
        different = callback(lambda: 0)
        library = SimpleNamespace(plan_native_queries=entry)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "lib.so"
            path.write_bytes(b"qualified fixture")
            manifest = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            with (
                patch.dict(
                    os.environ,
                    {"TASK_QUEUE_ENABLE": "0", "BETTERSCALE_FIA_LIBRARY": str(path)},
                ),
                patch("betterscale.patches.qwen_fia.json.loads", return_value=manifest),
            ):
                for process, error in (
                    (SimpleNamespace(), "Preload"),
                    (SimpleNamespace(plan_native_queries=different), "binding"),
                ):
                    with patch(
                        "betterscale.patches.qwen_fia.ctypes.CDLL",
                        side_effect=[library, process],
                    ):
                        with self.assertRaisesRegex(RuntimeError, error):
                            check_library()
                with patch(
                    "betterscale.patches.qwen_fia.ctypes.CDLL",
                    side_effect=[library, library],
                ):
                    self.assertIs(check_library(), library)
                path.write_bytes(b"different ABI")
                with self.assertRaisesRegex(ValueError, "qualified native planner"):
                    check_library()


class OrderedReplay(unittest.TestCase):
    def test_only_owned_full_skips_old_update_barrier_and_restores_flag(self):
        import sys
        from betterscale.patches.qwen_fia.wave import install_replay_ordering

        context = SimpleNamespace(cudagraph_runtime_mode="FULL")
        state = {"owned": True}

        class Wrapper:
            runtime_mode = "FULL"
            enable_enpu = False

            def __call__(self, fail=False):
                if fail:
                    raise RuntimeError("launch failed")
                return self.enable_enpu

        with patch.dict(
            sys.modules,
            {
                "vllm_ascend.compilation.acl_graph": SimpleNamespace(
                    ACLGraphWrapper=Wrapper
                ),
                "vllm.config": SimpleNamespace(
                    CUDAGraphMode=SimpleNamespace(FULL="FULL")
                ),
                "vllm.forward_context": SimpleNamespace(
                    get_forward_context=lambda: context
                ),
            },
        ):
            install_replay_ordering(lambda: state["owned"])
            wrapper = Wrapper()
            self.assertTrue(wrapper())
            self.assertFalse(wrapper.enable_enpu)
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                wrapper(fail=True)
            self.assertFalse(wrapper.enable_enpu)
            state["owned"] = False
            self.assertFalse(wrapper())
            state["owned"] = True
            context.cudagraph_runtime_mode = "NONE"
            self.assertFalse(wrapper())
            context.cudagraph_runtime_mode = "FULL"
            wrapper.runtime_mode = "PIECEWISE"
            self.assertFalse(wrapper())
            wrapper.enable_enpu = True
            self.assertTrue(wrapper())
            self.assertTrue(wrapper.enable_enpu)

    def test_startup_priming_only_exercises_a_new_capture(self):
        import sys
        from betterscale.patches.qwen_fia.wave import install_replay_ordering

        context = SimpleNamespace(
            cudagraph_runtime_mode="FULL", batch_descriptor=(16, 1)
        )
        calls = []
        graph = SimpleNamespace(replay=lambda: calls.append("replay"))

        class Wrapper:
            runtime_mode = "FULL"
            enable_enpu = False

            def __init__(self):
                self.concrete_aclgraph_entries = {}

            def __call__(self):
                self.concrete_aclgraph_entries.setdefault(
                    context.batch_descriptor, SimpleNamespace(aclgraph=graph)
                )
                return "output"

        with (
            patch.dict(
                sys.modules,
                {
                    "vllm_ascend.compilation.acl_graph": SimpleNamespace(
                        ACLGraphWrapper=Wrapper
                    ),
                    "vllm.config": SimpleNamespace(
                        CUDAGraphMode=SimpleNamespace(FULL="FULL")
                    ),
                    "vllm.forward_context": SimpleNamespace(
                        get_forward_context=lambda: context
                    ),
                },
            ),
            patch(
                "betterscale.patches.qwen_fia.wave.torch.npu",
                SimpleNamespace(
                    current_stream=lambda: SimpleNamespace(
                        synchronize=lambda: calls.append("drain")
                    )
                ),
                create=True,
            ),
        ):
            install_replay_ordering(lambda: True, lambda: True)
            wrapper = Wrapper()
            self.assertEqual(wrapper(), "output")
            self.assertEqual(calls, ["replay", "drain"])
            wrapper()
            self.assertEqual(calls, ["replay", "drain"])
            self.assertFalse(wrapper.enable_enpu)


class WorkspaceContract(unittest.TestCase):
    def test_refresh_cannot_outgrow_captured_scratch(self):
        from betterscale.patches.qwen_fia.wave import Planner

        pointer = SimpleNamespace(data_ptr=lambda: 4096, device="cpu", shape=(16,))
        metadata = SimpleNamespace(
            actual_seq_lengths_q=[1], seq_lens_list=[1], attn_mask=pointer
        )
        released = []
        planner = Planner.__new__(Planner)
        planner.fixtures = (pointer, pointer, pointer, pointer, 0.0625)
        planner.calls = 0
        planner.lib = SimpleNamespace(
            plan_native_queries=lambda *a: 5,
            plan_is_fd=lambda *a: 0,
            plan_blocks=lambda *a: 24,
            plan_metadata=lambda *a: 2528,
            plan_workspace=lambda *a: 2,
            plan_release=lambda p: released.append(p),
        )
        frame = SimpleNamespace(
            tokens=16, columns=64, table=pointer, h_tiling=pointer, plan=3, workspace=1
        )
        with (
            patch(
                "betterscale.patches.qwen_fia.wave.torch.empty", return_value=pointer
            ),
            patch(
                "betterscale.patches.qwen_fia.wave.torch.npu",
                SimpleNamespace(current_stream=lambda: SimpleNamespace(npu_stream=1)),
                create=True,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "workspace changed"):
                planner.native(frame, metadata)
        self.assertEqual(released, [5])
        self.assertEqual(frame.workspace, 1)
        self.assertEqual(frame.plan, 3)
