"""Each patch owns its hooks: real module imports against CPU donor doubles."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1] / "src/betterscale/patches"


class Installation(unittest.TestCase):
    def setUp(self):
        class CompilationConfig:
            def adjust_cudagraph_sizes_for_spec_decode(self, alignment, tp):
                return alignment, tp

        class Builder:
            def build(self):
                return "native-build"

            def _build_qli_metadata(self, *args):
                return "native-qli"

            @classmethod
            def get_cudagraph_support(cls, *args):
                return "native-support"

        class Runner:
            def _pad_query_start_loc_for_fia(self, *args):
                return "native-pad"

        class Wrapper:
            def __call__(self, *args, **kwargs):
                return "native-call"

        self.CompilationConfig, self.Builder, self.Runner, self.Wrapper = (
            CompilationConfig,
            Builder,
            Runner,
            Wrapper,
        )
        self.context = NS(
            batch_descriptor="bucket", cudagraph_runtime_mode="FULL", capturing=False
        )
        exports = {
            "vllm.platforms": dict(
                current_platform=NS(get_global_graph_pool=lambda: "shared")
            ),
            "vllm.config": dict(
                CompilationConfig=CompilationConfig,
                CUDAGraphMode=NS(FULL="FULL", NONE="NONE"),
            ),
            "vllm.forward_context": dict(get_forward_context=lambda: self.context),
            "vllm.v1.attention.backend": dict(AttentionCGSupport=NS(ALWAYS="always")),
            "vllm_ascend.attention.context_parallel.dsa_cp": dict(
                AscendDSACPMetadataBuilder=Builder,
                RopeDataProxy=type("RopeDataProxy", (), {}),
                get_cos_and_sin_dsa=lambda: "native-rope",
            ),
            "vllm_ascend.worker.model_runner_v1": dict(NPUModelRunner=Runner),
            "vllm_ascend.compilation.acl_graph": dict(
                ACLGraphWrapper=Wrapper,
                logger=NS(info_once=lambda *a: None),
                _EXTRA_CTX=NS(is_draft_model=False),
            ),
        }
        modules = {}
        for name, values in exports.items():
            module = modules.setdefault(name, ModuleType(name))
            module.__dict__.update(values)
            parts = name.split(".")
            for i in range(1, len(parts)):
                parent = ".".join(parts[:i])
                child = ".".join(parts[: i + 1])
                modules.setdefault(parent, ModuleType(parent))
                modules.setdefault(child, ModuleType(child))
                setattr(modules[parent], parts[i], modules[child])
        self.dsa = modules["vllm_ascend.attention.context_parallel.dsa_cp"]
        self.enterContext(patch.dict(sys.modules, modules))
        self.enterContext(
            patch.object(
                torch,
                "npu",
                NS(
                    synchronize=lambda: None,
                    current_stream=lambda: NS(npu_stream=7, synchronize=lambda: None),
                ),
                create=True,
            )
        )

    def load(self, name):
        key = "_isolated_patch_" + name
        spec = importlib.util.spec_from_file_location(
            key,
            ROOT / name / "__init__.py",
            submodule_search_locations=[str(ROOT / name)],
        )
        module = importlib.util.module_from_spec(spec)
        self.enterContext(patch.dict(sys.modules, {key: module}))
        spec.loader.exec_module(module)
        return module

    def hooks(self):
        return (
            self.CompilationConfig.adjust_cudagraph_sizes_for_spec_decode,
            self.Builder.build,
            self.Builder._build_qli_metadata,
            self.Builder.__dict__["get_cudagraph_support"],
            self.Runner._pad_query_start_loc_for_fia,
            self.Wrapper.__call__,
            self.dsa.get_cos_and_sin_dsa,
        )

    def test_importing_each_closed_package_does_not_install_hooks(self):
        before = self.hooks()
        for name in (
            "compat_lcm",
            "target_full",
            "ordered_replay",
            "qli_cpu",
            "cross_step",
            "split_draft",
        ):
            with self.subTest(name=name):
                self.assertTrue(callable(self.load(name).install))
                self.assertEqual(before, self.hooks())

    def test_patch_imports_stay_within_their_own_directory(self):
        import ast

        for source in ROOT.glob("**/*.py"):
            if source.parent == ROOT:
                continue
            for node in ast.walk(ast.parse(source.read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertLessEqual(node.level, 1, str(source))
                    self.assertFalse(
                        (node.module or "").startswith("betterscale.patches"),
                        str(source),
                    )
                elif isinstance(node, ast.Import):
                    self.assertFalse(
                        any(
                            n.name.startswith("betterscale.patches") for n in node.names
                        ),
                        str(source),
                    )

    def test_alignment_does_not_install_target_or_replay(self):
        before = self.hooks()
        module = self.load("compat_lcm")
        module.install()
        module.install()
        self.assertEqual(self.hooks()[1:], before[1:])
        config = NS(pass_config=NS(enable_sp=True))
        self.assertEqual(
            self.CompilationConfig.adjust_cudagraph_sizes_for_spec_decode(config, 6, 8),
            (24, 8),
        )

    def test_target_does_not_install_alignment_or_replay_or_qli(self):
        before = self.hooks()
        module = self.load("target_full")
        module.install()
        after = self.hooks()
        module.install()
        self.assertEqual(after, self.hooks())
        for index in (0, 2, 5):
            self.assertIs(after[index], before[index])
        self.assertEqual(self.Builder.get_cudagraph_support(None, None), "always")
        self.assertIs(self.Builder.build, module._build_fixed_capacity)

    def test_ordered_replay_installs_without_target_patch(self):
        module = self.load("ordered_replay")
        before = self.hooks()
        calls = []
        model = self.Wrapper()
        model.runtime_mode = "FULL"
        model.is_debugging_mode = False
        model.use_eagle = False
        model.enable_enpu = False
        model.concrete_aclgraph_entries = {
            "bucket": NS(
                aclgraph=NS(replay=lambda: calls.append("replay")), output="output"
            )
        }
        runner = NS(
            model=model,
            use_compress=True,
            vllm_config=NS(model_config=NS(hf_config=NS(model_type="deepseek_v4"))),
        )
        worker = NS(rank=0, model_runner=runner)
        module.install(worker)
        module.install(worker)
        self.assertEqual(self.hooks()[:5], before[:5])
        self.assertEqual(model(), "output")
        self.assertEqual(calls, ["replay"])
        self.context.capturing = True
        self.assertEqual(model(), "output")
        unadmitted = self.Wrapper()
        unadmitted.runtime_mode = "FULL"
        unadmitted.runnable = lambda: "native-eager"
        self.context.cudagraph_runtime_mode = "NONE"
        self.assertEqual(unadmitted(), "native-eager")

    def test_qli_explicit_install_is_local_and_idempotent(self):
        module = self.load("qli_cpu")
        before = self.hooks()
        module.install(NS(rank=0))
        module.install(NS(rank=0))
        for i in (0, 1, 3, 4, 5, 6):
            self.assertIs(self.hooks()[i], before[i])
        builder = self.Builder()
        builder.compressor_ratio = 1
        self.assertEqual(builder._build_qli_metadata(None, None, None, 0), "native-qli")
