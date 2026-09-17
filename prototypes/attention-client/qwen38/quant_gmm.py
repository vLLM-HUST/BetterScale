"""Raw-ACL actual-count INT8 GEMM for the persistent server math boundary."""

import ctypes as C
from pathlib import Path

import torch
import torch_npu


class QuantGmm:
    def __init__(self, build, weight, groups, capacity):
        assert weight.dtype == torch.int8 and weight.ndim == 3
        assert int(torch_npu.get_npu_format(weight)) == 29
        count, k, n = weight.shape
        assert k % 32 == n % 32 == 0
        assert groups.dtype == torch.int64 and groups.shape == (count,)
        self.weight, self.groups = weight, groups
        self.k, self.n, self.capacity = k, n, capacity
        root = Path(build)
        self.lib = C.CDLL(str(root / "launch.so"))
        self.lib.load_server.argtypes = [
            C.c_char_p,
            C.c_char_p,
            C.POINTER(C.c_void_p),
            C.POINTER(C.c_void_p),
        ]
        self.lib.launch_cube.argtypes = [C.c_void_p] * 5 + [C.c_uint32]
        self.lib.unload_server.argtypes = [C.c_void_p]
        self.binary, self.fn = C.c_void_p(), C.c_void_p()
        rc = self.lib.load_server(
            str(root / "quant_gmm.o").encode(),
            b"quant_gmm",
            C.byref(self.binary),
            C.byref(self.fn),
        )
        assert rc == 0, rc
        self.config = torch.tensor(
            [k, n, count, weight.data_ptr(), groups.data_ptr(), capacity],
            device=weight.device,
            dtype=torch.int64,
        )

    def __call__(self, x, output):
        assert x.shape == (self.capacity, self.k)
        assert output.shape == (self.capacity, self.n)
        assert x.dtype == torch.int8 and output.dtype == torch.int32
        assert x.is_contiguous() and output.is_contiguous()
        assert x.device == output.device == self.weight.device
        rc = self.lib.launch_cube(
            self.fn,
            torch.npu.current_stream().npu_stream,
            self.config.data_ptr(),
            x.data_ptr(),
            output.data_ptr(),
            24,
        )
        assert rc == 0, rc
        return output

    def close(self):
        # All graphs using this binary must first be reset and work synchronized.
        assert self.lib.unload_server(self.binary) == 0
