"""Isolate GEMM graph output overwrite from communication (no HCCL)."""
import json
import os
from pathlib import Path
import torch
import torch_npu

root = Path(os.environ['CAPSULE'])
torch.npu.set_device(0)
rows = []
with torch.inference_mode():
    for k in (3072, 8704):
        torch.manual_seed(419 + k)
        w = (torch.randn((5120, k), device='npu') / k**.5).to(torch.bfloat16)
        for n in (1, 1024, 1536):
            banks = []
            for b in (0, 1):
                x = torch.randn((n, k), device='npu', dtype=torch.bfloat16)
                def body():
                    y = torch.nn.functional.linear(x, w)
                    before = y.clone()
                    y.mul_(2)
                    return before, y, y + .125
                for _ in range(3): body()
                torch.npu.synchronize()
                g = torch.npu.NPUGraph()
                with torch.npu.graph(g):
                    before, y, out = body()
                banks.append((g, x, before, y, out))
            for generation in range(6):
                torch.manual_seed(1909 + 13 * generation)
                value = torch.randn((n, k), device='npu').to(torch.bfloat16)
                ref = value.float() @ w.float().T
                g, x, before, y, out = banks[generation % 2]
                x.copy_(value); g.replay(); torch.npu.synchronize()
                checks = []
                for actual, expected in ((before, ref), (out, 2*ref+.125)):
                    error = actual.float()-expected
                    checks.append(dict(passed=bool(torch.allclose(actual.float(),expected,atol=.04,rtol=.03)),rmse=float(error.square().mean().sqrt())))
                rows.append(dict(k=k,n=n,generation=generation,checks=checks))
            print(k,n,rows[-1],flush=True)
receipt = dict(status='PASS' if all(c['passed'] for r in rows for c in r['checks']) else 'FAIL', rows=rows)
(root/'receipt.json').write_text(json.dumps(receipt,indent=2))
print(receipt['status'],flush=True)
