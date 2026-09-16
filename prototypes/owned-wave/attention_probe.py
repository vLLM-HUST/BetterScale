"""Bounded native device-length alternative; no model or runner."""
import json
import os
from pathlib import Path
import torch
import torch_npu
from betterscale.patches.async_decode._metadata import DeviceOnly

out = Path(os.environ['OWNED_OUTPUT'])
torch.npu.set_device(0)
torch.manual_seed(123)
rows = []
try:
    q = torch.randn(1, 16, 128, device='npu', dtype=torch.bfloat16)
    k = torch.randn(2, 128, 2, 128, device='npu', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.tensor([[0, 1]], device='npu', dtype=torch.int32)
    length = torch.tensor([33], device='npu', dtype=torch.int32)
    qlength = torch.ones(1, device='npu', dtype=torch.int32)
    axis = torch.arange(256, device='npu', dtype=torch.int32)
    def body():
        indices = torch.where(axis < length, axis, -1).view(1, 1, 256).expand(1, 2, 256).contiguous()
        return torch_npu.npu_sparse_flash_attention(
            q[..., :64].contiguous(), k[..., :64].contiguous(), v, indices, 128**-0.5,
            query_rope=q[..., 64:].contiguous(), key_rope=k[..., 64:].contiguous(),
            sparse_block_size=1,
            block_table=table, actual_seq_lengths_query=qlength,
            actual_seq_lengths_kv=length, layout_query='TND',
            layout_kv='PA_BSND', sparse_mode=3)[0]
    with DeviceOnly():
        eager = body()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph), DeviceOnly():
        actual = body()
    for n in (33, 34, 65, 129, 255):
        length.fill_(n)
        graph.replay()
        reference = torch.empty_like(q)
        torch_npu._npu_paged_attention(query=q, key_cache=k, value_cache=v,
            num_kv_heads=2, num_heads=16, scale_value=128**-0.5,
            block_table=table, context_lens=torch.tensor([n], dtype=torch.int32), out=reference)
        torch.npu.synchronize()
        err = (actual.float()-reference.float()).abs().max().item()
        rows.append(dict(length=n, exact=torch.equal(actual, reference), max_abs=err))
        torch.testing.assert_close(actual, reference, rtol=.01, atol=.01)
    result = dict(status='PASS', rows=rows, captures=1)
except BaseException as exc:
    result = dict(status='FAIL', rows=rows, error=f'{type(exc).__name__}: {exc}')
    raise
finally:
    (out/'attention.json').write_text(json.dumps(result, indent=2))
