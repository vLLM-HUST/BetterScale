"""Short shape-boundary graph profile; no profiler timings used as benchmark."""
import os
from pathlib import Path
import torch
import torch_npu
from profiling import ProfileWindow

root = Path(os.environ['CAPSULE'])
torch.npu.set_device(0)
with torch.inference_mode():
    torch.manual_seed(1909)
    w = (torch.randn((17408, 5120), device='npu') / 5120**.5).to(torch.bfloat16)
    banks = []
    for n in (1024, 1280, 1408, 1536, 1792, 2048):
        x = torch.randn((n, 5120), device='npu', dtype=torch.bfloat16)
        for _ in range(3):
            torch.nn.functional.linear(x, w)
        torch.npu.synchronize()
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            y = torch.nn.functional.linear(x, w)
        banks.append((n, x, y, g))
    for _, _, _, g in banks:
        for _ in range(3):
            g.replay()
    torch.npu.synchronize()
    prof = ProfileWindow(str(root / 'profiles'), 0, 12)
    for _ in range(2):
        for n, _, _, g in banks:
            prof.mark(f'gate_up_{n}')
            g.replay()
            torch.npu.synchronize()
            prof.step()
    prof.close()
print('PROFILE PASS', flush=True)
