"""Bounded raw-ACL adapter; matrix arithmetic belongs to installed CANN CATLASS."""

import ctypes as C
import os
from pathlib import Path

import torch
import torch_npu


class ActualGmm:
    def __init__(self, weight, groups, capacity=512):
        assert weight.dtype == torch.bfloat16 and weight.ndim == 3
        assert int(torch_npu.get_npu_format(weight)) == 29
        count, k, n = weight.shape
        assert count in (64, 128) and (k, n) in ((2048, 1536), (768, 2048))
        assert groups.dtype == torch.int64 and groups.shape == (count,)
        assert 1 <= capacity <= 512
        self.weight, self.groups = weight, groups
        self.k, self.n, self.capacity = k, n, capacity
        root = Path(os.environ["ACTUAL_GMM_BUILD"])
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
            str(root / "actual_gmm.o").encode(),
            b"actual_gmm",
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
        assert x.shape == (self.capacity, self.k) and output.shape == (
            self.capacity,
            self.n,
        )
        assert x.dtype == output.dtype == torch.bfloat16
        assert x.is_contiguous() and output.is_contiguous()
        assert (
            int(torch_npu.get_npu_format(x))
            == int(torch_npu.get_npu_format(output))
            == 2
        )
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
        # Caller must synchronize and reset all graphs that reference this binary.
        assert self.lib.unload_server(self.binary) == 0
