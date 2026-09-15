import argparse
import ctypes
import torch
import torch_npu

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["native", "pair", "full"], required=True)
a = p.parse_args()
torch.set_num_threads(4)
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True
torch.manual_seed(1904)
m, h, i = 4096, 5120, 13824
x = torch.randn(m, h, device="npu", dtype=torch.bfloat16)
w = (torch.randn(2 * i, h, device="npu") / h**0.5).bfloat16()
b = torch_npu.npu_format_cast(w.t().contiguous(), 29)
y = torch.empty(m, i, device="npu", dtype=torch.bfloat16)
scratch = torch.empty(m * 2 * i, device="npu", dtype=torch.bfloat16)
if a.mode != "native":
    build = "build-v7-pair" if a.mode == "pair" else "build-v8-full"
    lib = ctypes.CDLL(
        "/workspace/strengthen-dsv4/runs/qwen-native-ffn-20260915/"
        + build
        + "/libqwen_native_ffn.so"
    )
    launch = lib.launch_native_ffn
    launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32] * 3 + [ctypes.c_void_p]
    launch.restype = ctypes.c_int


def run():
    if a.mode == "native":
        return torch_npu.npu_swiglu(torch.mm(x, b))
    assert (
        launch(
            x.data_ptr(),
            b.data_ptr(),
            y.data_ptr(),
            scratch.data_ptr(),
            m,
            256,
            256,
            torch.npu.current_stream().npu_stream,
        )
        == 0
    )
    return y


for _ in range(3):
    run()
torch.npu.synchronize()

o = run()
torch.npu.synchronize()

assert torch.isfinite(o).all()
print("PROFILE_APP_COMPLETE", a.mode, flush=True)
