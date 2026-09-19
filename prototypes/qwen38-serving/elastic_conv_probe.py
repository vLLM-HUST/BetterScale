"""Isolate dynamic padded convolution metadata from the model/graph adapter."""

import json
import os
from pathlib import Path
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op

assert enable_custom_op()
torch.npu.set_device(0)
torch.manual_seed(17)
root = Path(os.environ["CAPSULE"])
T, N, C = 1024, 9, 5120
x = torch.randn(T, C, device="npu", dtype=torch.bfloat16)
w = torch.randn(4, C, device="npu", dtype=torch.bfloat16)
bank = torch.randn(21, 3, C, device="npu", dtype=torch.bfloat16)
seed = bank.clone()
cu = torch.empty(N + 1, dtype=torch.int32, device="npu")
slots = torch.empty(N, 1, dtype=torch.int32, device="npu")
flags = torch.empty(N, dtype=torch.bool, device="npu")


def prep(lengths):
    ends = [0]
    for n in lengths:
        ends.append(ends[-1] + n)
    ends += [ends[-1]] * (N - len(lengths))
    cu.copy_(torch.tensor(ends, dtype=torch.int32))
    slots.copy_(
        torch.tensor(
            [*range(11, 11 + len(lengths)), *([-1] * (N - len(lengths)))],
            dtype=torch.int32,
        )[:, None]
    )
    flags.copy_(torch.tensor([i < 2 for i in range(N)], dtype=torch.bool))


def forward():
    out = torch.empty_like(x)
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        out,
        x,
        w,
        conv_state=bank,
        bias_opt=None,
        query_start_loc_opt=cu,
        cache_indices_opt=slots,
        initial_state_mode_opt=flags,
        num_accepted_tokens_opt=None,
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=0,
    )
    return out


rows = []
with torch.inference_mode():
    prep([128] * 8)
    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = forward()
    for lengths in (
        [513],
        [1, 1, 7, 129, 513],
        [1, 1, 1, 1, 514],
        [1] * 8,
        [128] * 8,
        [1, 1, 17, 129],
    ):
        prep(lengths)
        bank.copy_(seed)
        graph.replay()
        actual = result[: sum(lengths)].clone()
        after = bank.clone()
        bank.copy_(seed)
        expected = forward()[: sum(lengths)]
        torch.npu.synchronize()
        rows.append(
            dict(
                lengths=lengths,
                output=float((actual.float() - expected.float()).abs().max()),
                state=float((after.float() - bank.float()).abs().max()),
            )
        )
    (root / "receipt.json").write_text(
        json.dumps(
            dict(rows=rows, passed=all(r["output"] == r["state"] == 0 for r in rows)),
            indent=2,
        )
    )
    print(rows)
