"""Single-card, dummy two-KV-head TP1 page publication/gather/FIA graph gate."""

import argparse
import os
import json
import torch
import torch_npu

p = argparse.ArgumentParser()
p.add_argument("--device", required=True)
a = p.parse_args()
if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != a.device:
    raise RuntimeError("physical device binding must match admission")
torch.npu.set_device(0)
torch.set_num_threads(2)
from livemodule.arch.ascend.llm.qwen38.qsa_attention import AscendQwen38QSAAttention
from livemodule.arch.ascend.llm.qwen38.qsa_gather import gather_qsa_kv
from livemodule.arch.ascend.llm.qwen38.qsa_publish import publish_kv

backend = AscendQwen38QSAAttention()
torch.manual_seed(17)
records = []
for heads in (1, 2):
    for n in (1, 4):
        cache = torch.zeros(5, 64, heads, 256, dtype=torch.bfloat16, device="npu")
        value_cache = torch.zeros_like(cache)
        table = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device="npu")
        # Two request rows, non-identity page mapping, visible head-specific data.
        requests = torch.arange(n, dtype=torch.int64, device="npu") % 2
        positions = torch.arange(n, dtype=torch.int64, device="npu") + 64
        valid = torch.ones(n, dtype=torch.bool, device="npu")
        keys = torch.randn(n, heads, 256, dtype=torch.bfloat16, device="npu")
        values = torch.randn_like(keys)
        query = torch.randn(n, 1, heads * 12, 256, dtype=torch.bfloat16, device="npu")
        indices = torch.stack((positions.int(), torch.ones_like(positions).int()), -1)

        def run():
            publish_kv(
                cache,
                value_cache,
                table,
                positions,
                keys,
                values,
                valid,
                key_leading_pages=2,
                value_leading_pages=2,
                request_rows=requests,
            )
            return backend.decode_paged(
                query,
                cache,
                value_cache,
                indices,
                table,
                leading_pages=2,
                request_rows=requests,
                scale=256**-0.5,
            )

        run()
        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream), torch.npu.graph(graph):
            output = run()
        stream.synchronize()
        for generation in range(2):
            keys.copy_(torch.randn_like(keys))
            values.copy_(torch.randn_like(values))
            graph.replay()
            torch.npu.synchronize()
            gathered = gather_qsa_kv(
                query, cache, value_cache, indices, table, requests, leading_pages=2
            )
            assert torch.equal(gathered[0][:, 0], keys)
            assert torch.equal(gathered[1][:, 0], values)
            # With exactly one selected row, attention must return that row,
            # repeated within each GQA head group, not across the two KV heads.
            expected = values.repeat_interleave(12, dim=1)[:, None]
            assert torch.equal(output, expected), (heads, n, generation)
        graph.reset()
        records.append(dict(kv_heads=heads, rows=n, status="PASS"))
# Non-degenerate head-major layout, inactive tiles, and both sides of the
# 128-query workspace boundary. Independent page indexing feeds unchunked FIA.
for heads in (1, 2):
    for n in (129, 257):
        cache = torch.randn(6, 64, heads, 256, dtype=torch.bfloat16, device="npu")
        value_cache = torch.randn_like(cache)
        query = torch.randn(n, 1, heads * 12, 256, dtype=torch.bfloat16, device="npu")
        table = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32, device="npu")
        requests = torch.arange(n, device="npu") % 2
        selected = 33
        indices = (torch.arange(selected, device="npu") * 3).int().repeat(n, 1)
        indices[:, 5] = -1
        counts = (torch.arange(n, device="npu") % (selected + 1)).int()
        packed = torch.cat((indices, counts[:, None]), dim=1)

        def run_many():
            return backend.decode_paged(
                query,
                cache,
                value_cache,
                packed,
                table,
                leading_pages=2,
                request_rows=requests,
                scale=256**-0.5,
            )

        run_many()
        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream), torch.npu.graph(graph):
            output = run_many()
        stream.synchronize()
        for generation in range(2):
            cache.copy_(torch.randn_like(cache))
            value_cache.copy_(torch.randn_like(value_cache))
            graph.replay()
            torch.npu.synchronize()
            safe = indices.clamp_min(0).long()
            page = table[requests[:, None], safe // 64].long() + 2
            keys = cache[page, safe % 64]
            values = value_cache[page, safe % 64]
            valid = (indices >= 0) & (
                torch.arange(selected, device="npu")[None] < counts[:, None]
            )
            keys = torch.where(valid[:, :, None, None], keys, 0)
            values = torch.where(valid[:, :, None, None], values, 0)
            got = gather_qsa_kv(
                query, cache, value_cache, packed, table, requests, leading_pages=2
            )
            assert torch.equal(got[0], keys) and torch.equal(got[1], values)
            assert got[0].transpose(1, 2).is_contiguous()
            assert got[1].transpose(1, 2).is_contiguous()
            reference = backend._attend(
                query, keys, values, valid, counts, scale=256**-0.5
            )
            assert torch.equal(output, reference), (heads, n, generation, "chunked FIA")
        graph.reset()
        records.append(dict(kv_heads=heads, rows=n, selected=selected, status="PASS"))
print(json.dumps(dict(status="PASS", cases=records)))
