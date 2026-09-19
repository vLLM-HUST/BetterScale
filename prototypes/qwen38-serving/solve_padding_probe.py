"""Compare owned empty-task skip against the pinned donor on valid rows."""

import json
import os
from pathlib import Path
import torch
import torch_npu
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.fla.solve_tril import solve_tril as baseline
from betterscale.patches.qwen_gdn.solve_tril import solve_tril as candidate
from betterscale.patches.qwen_gdn.metadata import chunk_rows

torch.npu.set_device(0)
init_device_properties_triton()
torch.manual_seed(41)
rows = []
for lengths in (
    [1, 1472],
    [1536],
    [1] * 8,
    [1, 1, 512, 513, 17],
    [1215, 1],
    [1216, 1],
    [1217, 1],
):
    T = 1536
    A = torch.randn((1, T, 24, 64), device="npu", dtype=torch.float32) * 0.01
    # Mimic the true strictly lower triangular chunk matrix.
    r = torch.arange(T, device="npu") % 64
    c = torch.arange(64, device="npu")
    A *= (r[:, None] > c[None, :])[None, :, None, :]
    A[:, sum(lengths) :] = float("nan")
    ends = [0]
    for n in lengths:
        ends.append(ends[-1] + n)
    ends += [ends[-1]] * (9 - len(lengths))
    cu = torch.tensor(ends, dtype=torch.int64, device="npu")
    large = torch.tensor(chunk_rows(lengths, 1216, T), dtype=torch.int64, device="npu")
    small = torch.tensor(chunk_rows(lengths, 64, T), dtype=torch.int64, device="npu")
    args = (A, cu, large, small, torch.bfloat16)
    a, b = baseline(*args), candidate(*args)
    torch.npu.synchronize()
    valid = sum(lengths)
    equal = torch.equal(a[:, :valid], b[:, :valid])
    assert equal, lengths
    graphs = {}
    for name, fn in (("baseline", baseline), ("candidate", candidate)):
        graph = torch.npu.NPUGraph()
        torch.npu.synchronize()
        with torch.npu.graph(graph):
            for _ in range(20):
                output = fn(*args)
        graphs[name] = graph
    times = {}
    for name, fn in (
        ("baseline", baseline),
        ("candidate", candidate),
        ("candidate2", candidate),
        ("baseline2", baseline),
    ):
        for _ in range(3):
            graphs[name.rstrip("2")].replay()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
            enable_timing=True
        )
        start.record()
        for _ in range(30):
            graphs[name.rstrip("2")].replay()
        end.record()
        end.synchronize()
        times[name] = start.elapsed_time(end) / (30 * 20)
    rows.append(dict(lengths=lengths, equal=equal, milliseconds=times))
    print(json.dumps(rows[-1]), flush=True)
Path(os.environ["CAPSULE"], "solve-padding.json").write_text(json.dumps(rows, indent=2))
