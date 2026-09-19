"""One shape's native ND gate/up kernel for bounded msprof op selection."""
import sys
import torch
import torch_npu

torch.set_num_threads(4)
torch.npu.set_device(0)
torch.manual_seed(1909)
n = int(sys.argv[1])
with torch.inference_mode():
    x = torch.randn((n, 5120), device='npu', dtype=torch.bfloat16)
    w = (torch.randn((17408, 5120), device='npu') / 5120**.5).to(torch.bfloat16)
    for _ in range(4):
        out = torch.nn.functional.linear(x, w)
        torch.npu.synchronize()
    assert torch.isfinite(out).all()
print('PROFILE_APP_COMPLETE', n, flush=True)
