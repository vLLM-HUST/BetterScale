"""CPU contracts for opt-in stream admission and stable draft metadata banks."""

import ast
import copy
import dataclasses
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch


class Proxy:
    def __init__(self, data, idx=0):
        self._data = data
        self.idx = idx


class Protocol(unittest.TestCase):
    def test_private_bank_updates_without_rebinding(self):
        p = (
            Path(__file__).resolve().parents[1] / "src/betterscale/patches"
        ).joinpath("split_draft/_graph.py")
        fs = [
            x
            for x in ast.parse(p.read_text()).body
            if isinstance(x, ast.FunctionDef)
            and x.name in ("bank", "refresh", "signature")
        ]
        ns = dict(
            torch=torch,
            copy=copy,
            dataclasses=dataclasses,
            Enum=Enum,
            RopeDataProxy=Proxy,
        )
        exec(compile(ast.Module(body=fs, type_ignores=[]), str(p), "exec"), ns)

        @dataclasses.dataclass
        class Metadata:
            rope: object
            query: object
            count: int

        original = Metadata(Proxy({"c4": torch.arange(4)}), torch.tensor([0, 2, 4]), 2)
        private = ns["bank"](original)
        ptr = private.rope._data["c4"].data_ptr()
        original.rope._data["c4"].add_(9)
        self.assertNotEqual(
            private.rope._data["c4"].tolist(), original.rope._data["c4"].tolist()
        )
        ns["refresh"](private, original)
        self.assertEqual(ptr, private.rope._data["c4"].data_ptr())
        self.assertEqual(
            private.rope._data["c4"].tolist(), original.rope._data["c4"].tolist()
        )
        sig = ns["signature"](original)
        original.count = 3
        self.assertNotEqual(sig, ns["signature"](original))
        original.count = 2
        original.rope.idx = 1
        self.assertNotEqual(sig, ns["signature"](original))

    def test_capture_first_call_executes_before_returning(self):
        # Emulate capture that records but does not commit writes. A successful
        # later-replay test alone would miss an uninitialized first invocation.
        import contextlib, json, os
        from unittest.mock import patch

        p = (
            Path(__file__).resolve().parents[1] / "src/betterscale/patches"
        ).joinpath("split_draft/_graph.py")
        nodes = [
            x
            for x in ast.parse(p.read_text()).body
            if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name != "install"
        ]
        box = {}

        class Graph:
            def __init__(self):
                self.executions = 0

            def replay(self):
                self.executions += 1
                box["output"].fill_(7)

        @contextlib.contextmanager
        def capture(graph):
            yield

        fake = NS(
            Tensor=torch.Tensor,
            npu=NS(synchronize=lambda: None, NPUGraph=Graph, graph=capture),
        )

        @dataclasses.dataclass
        class Metadata:
            num_prefills: int = 0

        ctx = NS(attn_metadata={"q": Metadata()}, capturing=False)
        ns = dict(
            torch=fake,
            copy=copy,
            dataclasses=dataclasses,
            Enum=Enum,
            RopeDataProxy=Proxy,
            os=os,
            json=json,
            Path=Path,
            get_forward_context=lambda: ctx,
        )
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), "exec"), ns)
        Drafter = type("AscendDSparkProposer", (), {})
        d = Drafter()
        d.use_cuda_graph = False
        d._dflash_num_context = 24
        d.num_speculative_tokens = 5
        d.parallel_drafting = True
        for name in (
            "input_ids",
            "positions",
            "_dflash_hidden_states",
            "_context_positions_buffer",
            "_dspark_seed_buffer",
            "_dspark_draft_buffer",
        ):
            setattr(d, name, torch.zeros(1))
        d._context_slot_mapping_buffers = [torch.zeros(1)]
        box["output"] = torch.zeros(1)
        d._runnable = lambda **kw: box["output"]
        worker = NS(
            rank=0,
            model_runner=NS(
                drafter=d, vllm_config=NS(scheduler_config=NS(max_num_seqs=4))
            ),
        )
        with patch(
            "pathlib.Path.write_text",
            side_effect=AssertionError("No production receipts"),
        ):
            graph = ns["ExactDraftGraph"](worker)
            out = graph(batch_size=4)
        self.assertEqual(out.item(), 7)
        self.assertEqual(graph.graph.executions, 1)
        self.assertFalse(ctx.capturing)
        manager_source = p.with_name("__init__.py")
        manager = next(
            n
            for n in ast.parse(manager_source.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "DraftGraphRunner"
        )
        exec(
            compile(
                ast.Module(body=[manager], type_ignores=[]), str(manager_source), "exec"
            ),
            ns,
        )
        bank_set = ns["DraftGraphRunner"](worker)
        for count in range(1, 5):
            d._dflash_num_context = 6 * count
            bank_set(batch_size=count)
        self.assertEqual(set(bank_set.decode_graphs), {1, 2, 3, 4})
        self.assertTrue(
            all(g.graph.executions == 1 for g in bank_set.decode_graphs.values())
        )
        d._dflash_num_context = 24
        bank_set(batch_size=4)
        self.assertEqual(bank_set.decode_graphs[4].graph.executions, 2)
        self.assertEqual(bank_set.query_graphs, {})
        d._dflash_num_context = 30
        with self.assertRaises(AssertionError):
            bank_set(batch_size=5)
        self.assertEqual(len(bank_set.decode_graphs), 4)

    def test_cpu_qli_uses_existing_mirrors_and_checks_them(self):
        p = (
            Path(__file__).resolve().parents[1] / "src/betterscale/patches"
        ).joinpath("qli_cpu/__init__.py")
        f = next(
            x
            for x in ast.parse(p.read_text()).body
            if isinstance(x, ast.FunctionDef) and x.name == "_cpu_qli_metadata"
        )
        recorded = []
        fake = NS(
            ops=NS(
                _C_ascend=NS(
                    npu_vllm_quant_lightning_indexer_metadata=lambda **kw: (
                        recorded.append(kw) or torch.zeros(1024)
                    )
                )
            )
        )
        ns = dict(
            torch=fake, _enabled=True, _verify=True, _original=lambda *a: "fallback"
        )
        exec(compile(ast.Module(body=[f], type_ignores=[]), str(p), "exec"), ns)
        qsl = torch.tensor([0, 2, 6])
        sl = torch.tensor([12, 18])
        ql = torch.tensor([2, 4])
        builder = NS(
            compressor_ratio=4,
            common_ratio_to_sas_metadata={"_cpu_local": {"qsl_cpu": qsl, "sl_cpu": sl}},
            model_config=NS(
                hf_config=NS(index_n_heads=64, index_head_dim=128, index_topk=512)
            ),
            seqused_q=torch.zeros(2),
            req_qli_metadata=torch.zeros(1024),
        )
        ns["_cpu_qli_metadata"](builder, qsl, sl, ql, 2)
        self.assertEqual(
            (recorded[0]["max_seqlen_q"], recorded[0]["max_seqlen_k"]), (4, 18)
        )
        builder.common_ratio_to_sas_metadata.pop("cp_qli")
        with self.assertRaises(AssertionError):
            ns["_cpu_qli_metadata"](builder, qsl, sl + 1, ql, 2)
        ns["_enabled"] = False
        self.assertEqual(ns["_cpu_qli_metadata"](builder, qsl, sl, ql, 2), "fallback")

    def test_only_admitted_stream_replays(self):
        p = (
            Path(__file__).resolve().parents[1] / "src/betterscale/patches"
        ).joinpath("ordered_replay/__init__.py")
        f = next(
            x
            for x in ast.parse(p.read_text()).body
            if isinstance(x, ast.FunctionDef) and x.name == "_ordered_call"
        )
        calls = []
        ctx = NS(batch_descriptor="x", cudagraph_runtime_mode="FULL", capturing=False)
        fake = NS(
            npu=NS(
                current_stream=lambda: NS(
                    npu_stream=7, synchronize=lambda: calls.append("sync")
                )
            )
        )
        native = NS(
            logger=NS(info_once=lambda *a: None), _EXTRA_CTX=NS(is_draft_model=False)
        )
        ns = dict(
            torch=fake,
            native=native,
            CUDAGraphMode=NS(FULL="FULL", NONE="NONE"),
            get_forward_context=lambda: ctx,
        )
        exec(compile(ast.Module(body=[f], type_ignores=[]), str(p), "exec"), ns)
        w = NS(
            _ordered_replay_stream=7,
            runtime_mode="FULL",
            is_debugging_mode=False,
            use_eagle=False,
            enable_enpu=False,
            runnable=lambda *a, **kw: "eager",
            concrete_aclgraph_entries={
                "x": NS(
                    aclgraph=NS(replay=lambda: calls.append("replay")), output="result"
                )
            },
        )
        run = ns["_ordered_call"]
        self.assertEqual(run(w), "result")
        self.assertEqual(calls, ["replay"])
        calls.clear()
        ctx.capturing = True
        self.assertEqual(run(w), "result")
        self.assertEqual(calls, ["sync", "replay"])
        ctx.capturing = False
        w._ordered_replay_stream = 8
        calls.clear()
        with self.assertRaises(AssertionError):
            run(w)
        self.assertEqual(calls, [])
        w._ordered_replay_stream = None
        self.assertEqual(run(w), "result")
        self.assertEqual(calls, ["sync", "replay"])
        for enpu, draft, eagle in ((True, False, False), (False, True, True)):
            calls.clear()
            w.enable_enpu = enpu
            native._EXTRA_CTX.is_draft_model = draft
            w.use_eagle = eagle
            self.assertEqual(run(w), "result")
            self.assertEqual(calls, ["replay"])
        calls.clear()
        ctx.cudagraph_runtime_mode = "NONE"
        self.assertEqual(run(w), "eager")
        self.assertEqual(calls, [])

    def test_inlined_dispatch_and_capture_match_pinned_donor(self):
        # The copied capture body must keep upstream offloader, workspace and
        # error handling. Ignore only qualification of donor-owned globals.
        root = Path(__file__).resolve().parents[1]
        donor = ast.parse(
            (
                root / "upstream/vllm-ascend/vllm_ascend/compilation/acl_graph.py"
            ).read_text()
        )
        cls = next(
            n
            for n in donor.body
            if isinstance(n, ast.ClassDef) and n.name == "ACLGraphWrapper"
        )
        original = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__call__"
        )
        patched = ast.parse(
            (
                root / "src/betterscale/patches/ordered_replay/__init__.py"
            ).read_text()
        )
        current = next(
            n
            for n in patched.body
            if isinstance(n, ast.FunctionDef) and n.name == "_ordered_call"
        )

        class Normalize(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == "native":
                    return ast.Name(id=node.attr, ctx=node.ctx)
                return self.generic_visit(node)

            def visit_Global(self, node):
                return None

        # Everything before the donor replay log/fence is preserved verbatim
        # modulo formatting and live native-module global references.
        prefix = []
        for node in original.body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "info_once"
            ):
                break
            prefix.append(node)
        self.assertEqual(
            ast.dump(Normalize().visit(ast.Module(body=prefix, type_ignores=[]))),
            ast.dump(
                Normalize().visit(
                    ast.Module(body=current.body[: len(prefix)], type_ignores=[])
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
