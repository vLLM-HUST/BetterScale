"""Bounded nine-row owned core graph/NONE witness, including poisoned padding."""

import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op

assert enable_custom_op()
torch.npu.set_device(0)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

init_device_properties_triton()
from betterscale.patches.qwen_gdn.metadata import Metadata, chunk_rows
from betterscale.patches.qwen_gdn.execution import forward_core
import vllm.forward_context

T, N, C = 1024, 9, 5120
meta = Metadata(T, False, torch.device("npu"))
meta.engine.retain_intermediates = True
if os.environ.get("ELASTIC_SEPARATE_INITIAL") == "1":
    launch = meta.engine.launch

    def separate_initial(kind, tensors):
        if kind == "h":
            tensors = list(tensors)
            tensors[4] = tensors[4].clone()
        return launch(kind, tensors)

    meta.engine.launch = separate_initial
vllm.forward_context.get_forward_context = lambda: NS(
    attn_metadata={"core": NS(owned=meta)}
)
torch.manual_seed(17)
x = torch.randn(T, C, dtype=torch.bfloat16, device="npu") * 0.1
clean = x.clone()
a = torch.randn(T, 24, dtype=torch.bfloat16, device="npu")
b = torch.randn_like(a)
conv = torch.randn(21, 3, C, dtype=torch.bfloat16, device="npu") * 0.1
bank = torch.randn(21, 24, 128, 128, dtype=torch.float32, device="npu") * 0.1
conv_seed = conv.clone()
seed = bank.clone()
w = torch.randn(4, C, dtype=torch.bfloat16, device="npu") * 0.1
layer = NS(
    prefix="core",
    kv_cache=(conv, bank),
    conv1d=NS(weight=w.T.unsqueeze(1), bias=None),
    activation=True,
    A_log=torch.randn(24, device="npu"),
    dt_bias=torch.randn(24, device="npu"),
)


def arrange(z):
    q, k, v = z.split([1024, 1024, 3072], dim=-1)
    return (
        q.reshape(1, T, 8, 128).contiguous(),
        k.reshape(1, T, 8, 128).contiguous(),
        v.reshape(1, T, 24, 128).contiguous(),
    )


layer.rearrange_mixed_qkv = arrange


def prep(lengths):
    ends = [0]
    for n in lengths:
        ends.append(ends[-1] + n)
    ends += [ends[-1]] * (N - len(lengths))
    meta.cu.copy_(torch.tensor(ends, dtype=torch.int64))
    meta.conv_cu.copy_(meta.cu)
    slots = [15, 11, 7, 3, 18, 19, 20, 16][: len(lengths)] + [-1] * (N - len(lengths))
    meta.slots.copy_(torch.tensor(slots, dtype=torch.int64))
    meta.conv_slots[:, 0].copy_(meta.slots)
    flags = [i < 2 for i in range(N)]
    meta.conv_initial.copy_(torch.tensor(flags, dtype=torch.bool))
    meta.state[:, 0].copy_(meta.slots)
    meta.state[:, 1].copy_(meta.conv_initial)
    for size, dest in meta.indices.items():
        dest.copy_(torch.tensor(chunk_rows(lengths, size, T), dtype=torch.int64))


def run():
    out = torch.empty(T, 24, 128, dtype=torch.bfloat16, device="npu")
    forward_core(layer, x, b, a, out)
    return out


def compare(x, y):
    return dict(
        close=bool(torch.allclose(x, y, atol=0.01, rtol=0.01, equal_nan=True)),
        max_abs=float((x.float() - y.float()).abs().nan_to_num().max()),
        finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
    )


rows = []
with torch.inference_mode():
    prep([128] * 8)
    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = run()
    captured_h = meta.engine.h
    for lengths in (
        [513],
        [1, 1, 7, 129, 513],
        [1, 1, 1, 1, 513],
        [1] * 8,
        [128] * 8,
        [1, 1, 17, 129],
    ):
        for poison in (False, True):
            prep(lengths)
            x.copy_(clean)
            if poison:
                x[sum(lengths) :] = float("nan")
            conv.copy_(conv_seed)
            bank.copy_(seed)
            graph.replay()
            result = out[: sum(lengths)].clone()
            cb = conv.clone()
            sb = bank.clone()
            initial_checks = []
            offset = 0
            for i, n in enumerate(lengths):
                slot = [15, 11, 7, 3, 18, 19, 20, 16][i]
                expected_h = (
                    seed[slot].to(torch.bfloat16)
                    if i < 2
                    else torch.zeros_like(seed[slot], dtype=torch.bfloat16)
                )
                actual_h = captured_h[0, :, offset]
                initial_checks.append(
                    dict(
                        request=i,
                        heads=(actual_h.float() - expected_h.float())
                        .abs()
                        .flatten(1)
                        .amax(1)
                        .cpu()
                        .tolist(),
                    )
                )
                offset += (n + 63) // 64
            conv.copy_(conv_seed)
            bank.copy_(seed)
            expected = run()
            torch.npu.synchronize()
            rows.append(
                dict(
                    lengths=lengths,
                    poison=poison,
                    initial_h=initial_checks,
                    output=compare(result, expected[: sum(lengths)]),
                    conv=compare(cb, conv),
                    state=compare(sb, bank),
                )
            )
    receipt = dict(
        rows=rows,
        passed=all(
            r[k]["max_abs"] == 0 and r[k]["finite"]
            for r in rows
            for k in ("output", "conv", "state")
        )
        and all(max(check["heads"]) == 0 for r in rows for check in r["initial_h"]),
    )
    Path(os.environ["CAPSULE"], "receipt.json").write_text(
        json.dumps(receipt, indent=2)
    )
    print(json.dumps(dict(passed=receipt["passed"], cases=len(rows))), flush=True)
    assert receipt[
        "passed"
    ], "Core output/state/initial-H oracle failed; inspect receipt.json"
