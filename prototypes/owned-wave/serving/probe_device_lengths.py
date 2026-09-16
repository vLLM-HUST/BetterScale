"""Bounded device-length FIA feasibility: GE tiling sink, then ACL nesting."""

import json
import sys
from pathlib import Path
import torch
import torch_npu

sys.path.insert(0, str(Path(torch_npu.__file__).parent / "dynamo"))
import torchair
from torchair.configs.compiler_config import CompilerConfig

torch.npu.set_device(0)
torch.npu.set_stream(torch.npu.Stream())
torch.manual_seed(123)
config = CompilerConfig()
config.mode = "max-autotune"
config.experimental_config.tiling_schedule_optimize = True
# A bounded paged decode, Qwen TP2 head geometry.
q = torch.randn(4, 16, 128, device="npu", dtype=torch.bfloat16)
k = torch.randn(32, 128, 256, device="npu", dtype=torch.bfloat16)
v = torch.randn_like(k)
table = torch.arange(32, device="npu", dtype=torch.int32).reshape(4, 8)
ql = torch.arange(1, 5, device="npu", dtype=torch.int64)
kl = torch.tensor([17, 129, 257, 513], device="npu", dtype=torch.int64)
mask = torch.ones(2048, 2048, dtype=torch.bool, device="npu").triu_(1)
kwargs = dict(
    num_heads=16,
    num_key_value_heads=2,
    input_layout="TND",
    scale=128**-0.5,
    block_size=128,
    sparse_mode=3,
    next_tokens=0,
)


def attention(q, k, v, table, ql, kl, mask):
    return torchair.ops.npu_fused_infer_attention_score(
        q,
        k,
        v,
        block_table=table,
        actual_seq_lengths=ql,
        actual_seq_lengths_kv=kl,
        atten_mask=mask,
        **kwargs,
    )[0]


compiled = torch.compile(
    attention,
    backend=torchair.get_npu_backend(compiler_config=config),
    fullgraph=True,
    dynamic=False,
)
checks = []
for lengths in ([17, 129, 257, 513], [31, 128, 300, 700], [8, 33, 64, 256]):
    kl.copy_(torch.tensor(lengths, device="npu"))
    got = compiled(q, k, v, table, ql, kl, mask)
    ref = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        block_table=table,
        actual_seq_lengths=[1, 2, 3, 4],
        actual_seq_lengths_kv=lengths,
        atten_mask=mask,
        **kwargs,
    )[0]
    torch.npu.synchronize()
    torch.testing.assert_close(got, ref, rtol=0.01, atol=0.002)
    checks.append(dict(lengths=lengths, max_error=(got - ref).abs().max().item()))
    print("device lengths passed", checks[-1], flush=True)
Path("device-lengths.json").write_text(json.dumps(checks, indent=2))
stream = torch.npu.current_stream()
stream.wait_stream(torch.npu.current_stream())
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph, stream=stream):
    result = compiled(q, k, v, table, ql, kl, mask)
for lengths in ([15, 111, 222, 444], [63, 257, 511, 900]):
    kl.copy_(torch.tensor(lengths, device="npu"))
    graph.replay()
    torch.npu.synchronize()
    ref = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        block_table=table,
        actual_seq_lengths=[1, 2, 3, 4],
        actual_seq_lengths_kv=lengths,
        atten_mask=mask,
        **kwargs,
    )[0]
    torch.testing.assert_close(result, ref, rtol=0.01, atol=0.002)
    print("nested replay passed", lengths, flush=True)
Path("complete.json").write_text(json.dumps(dict(status="PASS", checks=checks)))
