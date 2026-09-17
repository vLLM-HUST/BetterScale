"""Real PLE weight: native eager vs graph-capable convolution dispatch."""

import argparse, json, os
from pathlib import Path
import torch
import torch_npu
from weights import Checkpoint
from ple_conv import GraphPLEConv

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--physical-device", required=True)
a = p.parse_args()
assert (
    os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == a.physical_device
), "Admission and runtime devices must agree"
torch.npu.set_device(0)
torch.set_num_threads(2)
torch.manual_seed(18)
torch.npu.config.allow_internal_format = True
w = Checkpoint().tensor("model.language_model.layers.1.ple.conv1d.weight")
base = torch.nn.Conv1d(
    10240,
    10240,
    4,
    dilation=3,
    groups=10240,
    bias=False,
    dtype=torch.bfloat16,
    device="npu",
)
base.weight.data.copy_(w)
new = GraphPLEConv(base)
records = []
with torch.inference_mode():
    for rows in (1, 3, 32):
        x = torch.randn(1, 10240, rows + 9, dtype=torch.bfloat16, device="npu")
        expected = base(x)
        actual = new(x)
        torch.npu.synchronize()
        error = float(
            (actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-9)
        )
        assert error < 0.003, (rows, error)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        graph = torch.npu.NPUGraph()
        with torch.npu.stream(stream):
            with torch.npu.graph(graph):
                output = new(x)
        stream.synchronize()
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.npu.synchronize()
        replay_error = float(
            (output.float() - new(x).float()).norm()
            / output.float().norm().clamp_min(1e-9)
        )
        assert replay_error == 0
        assert torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable"
        graph.reset()
        records.append(
            dict(rows=rows, relative_l2=error, replay_relative_l2=replay_error)
        )
a.output.write_text(json.dumps(dict(status="PASS", cases=records), indent=2))
print(records, flush=True)
