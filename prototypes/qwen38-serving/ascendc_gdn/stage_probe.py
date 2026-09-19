"""Matched H/O graph timing: identical head-major inputs, no state glue/transposes."""

import json, os
from pathlib import Path
import torch, torch_npu
from vllm_ascend.utils import enable_custom_op
from runtime import Kernels

assert enable_custom_op()
torch.npu.set_device(0)
torch.manual_seed(91)
root = Path(os.environ["CAPSULE"])
receipt = []


def timing(graph):
    a, b = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    a.record()
    for _ in range(30):
        graph.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 30


with torch.inference_mode():
    for lengths in [(512,), (1, 511), (1, 1, 256, 254)]:
        ends = [0]
        for n in lengths:
            ends.append(ends[-1] + n)
        rows = [(i, j) for i, n in enumerate(lengths) for j in range((n + 63) // 64)]
        engine = Kernels(os.environ["ASCENDC_GDN_LIB"], 512, len(lengths), len(rows))
        bf = dict(device="npu", dtype=torch.bfloat16)
        q = torch.randn(1, 8, 512, 128, **bf) * 0.05
        k = torch.randn_like(q) * 0.05
        w = torch.randn(1, 24, 512, 128, **bf) * 0.01
        u = torch.randn_like(w) * 0.1
        g = -torch.rand(1, 24, 512, device="npu") * 0.1
        initial = torch.randn(len(lengths), 24, 128, 128, device="npu") * 0.01
        cu = torch.tensor(ends, device="npu", dtype=torch.int64)
        idx = torch.tensor(rows, device="npu", dtype=torch.int64)
        flat = tuple(v for row in rows for v in row)

        def native_h():
            return torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
                k,
                w,
                u,
                g=g,
                gk=None,
                initial_state=initial,
                output_final_state=True,
                chunk_size=64,
                save_new_value=True,
                cu_seqlens=tuple(ends),
                chunk_indices=flat,
                use_exp2=False,
                transpose_state_layout=False,
            )

        ho, vn, fs = native_h()

        def native_o():
            return torch.ops._C_ascend.chunk_fwd_o(
                q,
                k,
                vn,
                ho,
                g=g,
                scale=128**-0.5,
                g_gamma=None,
                cu_seqlens=tuple(ends),
                chunk_indices=flat,
                chunk_size=64,
                transpose_state_layout=False,
            )

        def owned_h():
            engine.launch(
                "h",
                [
                    k,
                    w,
                    u,
                    g,
                    initial,
                    cu,
                    idx,
                    engine.h,
                    engine.v,
                    engine.final,
                    engine.ws,
                    engine.th,
                ],
            )
            return engine.h, engine.v, engine.final

        def owned_o():
            # Same oracle H/V inputs, isolates O instead of changing two stages.
            engine.launch(
                "o", [q, k, vn, ho, g, cu, idx, engine.o, engine.ws, engine.to]
            )
            return engine.o

        row = {"lengths": lengths, "stages": {}}
        for stage, native, owned in [
            ("h", native_h, owned_h),
            ("o", native_o, owned_o),
        ]:
            graphs = {}
            outputs = {}
            for name, fn in [("native", native), ("owned", owned)]:
                fn()
                torch.npu.synchronize()
                gr = torch.npu.NPUGraph()
                with torch.npu.graph(gr):
                    out = fn()
                gr.replay()
                torch.npu.synchronize()
                graphs[name] = gr
                outputs[name] = out
            aa = outputs["native"]
            bb = outputs["owned"]
            if stage == "o":
                aa = (aa,)
                bb = (bb,)
            errors = [
                float((a.float() - b.float()).abs().max()) for a, b in zip(aa, bb)
            ]
            assert all(e == 0 for e in errors), errors
            times = {"native": [], "owned": []}
            for name in ["native", "owned", "owned", "native"] * 2:
                times[name].append(timing(graphs[name]))
            row["stages"][stage] = {"max_abs": errors, "ms": times}
        receipt.append(row)
(root / "receipt.json").write_text(
    json.dumps({"status": "PASS", "cases": receipt}, indent=2)
)
print("PASS", flush=True)
