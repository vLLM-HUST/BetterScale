"""Owned raw AscendC launcher. Requires TASK_QUEUE_ENABLE=0, BF16/FP32.

Mixed mode uses a native host operator: invocation-local temporary tensors are
allocated by the framework, hence capture uses the shared graph pool. Only
immutable tiling survives on this object. The non-pool raw path is a standalone
operator probe, not a service path. Metadata has an active prefix and empty tail.
"""

import ctypes as C
from functools import lru_cache
import os
from pathlib import Path
import re
import torch


def tiling_type(kind):
    types = {"int64_t": C.c_int64, "bool": C.c_bool, "float": C.c_float}
    fields = re.findall(
        r"    (int64_t|bool|float) (\w+);",
        Path(__file__).with_name(f"{kind}_tiling.h").read_text(),
    )
    return type(
        f"Tiling_{kind}",
        (C.Structure,),
        {"_fields_": [(n, types[t]) for t, n in fields]},
    )


def device_struct(value):
    return torch.tensor(list(bytes(value)), dtype=torch.uint8, device="npu")


@lru_cache(maxsize=1)
def host_adapter(host_path, kernel_path):
    torch.ops.load_library(host_path)
    torch.ops.betterscale_gdn.initialize(kernel_path)
    return torch.ops.betterscale_gdn


class Kernels:
    def __init__(self, library, tokens, requests, chunks, cores=24, state_pool=False):
        assert os.environ.get("TASK_QUEUE_ENABLE") == "0"
        self.lib = C.CDLL(str(library))
        self.state_pool = state_pool
        self.cores = cores
        self.T, self.N, self.C = tokens, requests, chunks
        host_path = os.environ["BETTERSCALE_GDN_HOST_LIBRARY"] if state_pool else None
        self.host = host_adapter(host_path, library) if state_pool else None
        self.retain_intermediates = False
        self.ws = (
            None
            if self.host
            else torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="npu")
        )
        self.h = (
            None
            if self.host
            else torch.empty(
                (1, 24, chunks, 128, 128), dtype=torch.bfloat16, device="npu"
            )
        )
        self.v = (
            None
            if self.host
            else torch.empty((1, 24, tokens, 128), dtype=torch.bfloat16, device="npu")
        )
        self.final = (
            None
            if state_pool
            else torch.empty(
                (requests, 24, 128, 128), dtype=torch.float32, device="npu"
            )
        )
        self.o = None if self.host else torch.empty_like(self.v)
        th, to = tiling_type("h")(), tiling_type("o")()
        for t in (th, to):
            t.seqlen = tokens
            t.kNumHead = 8
            t.vNumHead = 24
            t.kHeadDim = t.vHeadDim = 128
            t.chunkSize = 64
            t.isVariedLen = t.shapeBatch = 1
            t.tokenBatch = requests
            t.dataType = 1
            t.gDataType = 2
            t.chunkCapacity = chunks
        th.statePoolMode = state_pool
        th.batch = requests
        th.initalStateStride0 = 128
        th.useInitialState = th.storeFinalState = True
        th.stateDataType = 2
        to.scale = 128**-0.5
        # Match the native host workspace query and immutable kernel offsets.
        for t, sizes in (
            (
                th,
                [
                    ("vWorkspaceOffset", cores * 64 * 128 * 4 * 2),
                    ("vUpdateWorkspaceOffset", cores * 64 * 128 * 4 * 2),
                    ("hWorkspaceOffset", cores * 128 * 128 * 4 * 2),
                    ("numSeqWorkspaceOffset", (requests + 1) * 8),
                    ("numChunksWorkspaceOffset", (requests + 1) * 8),
                ],
            ),
            (
                to,
                [
                    ("vWorkspaceOffset", cores * 64 * 128 * 4 * 2),
                    ("hWorkspaceOffset", cores * 64 * 128 * 4 * 2),
                    ("attnWorkspaceOffset", cores * 64 * 64 * 4 * 2),
                    ("aftermaskWorkspaceOffset", cores * 64 * 64 * 4 * 2),
                    ("maskWorkspaceOffset", 64 * 64),
                ],
            ),
        ):
            offset = 16 * 1024 * 1024
            for name, size in sizes:
                setattr(t, name, offset)
                offset += (size + 511) // 512 * 512
            available = (
                self.host.workspace_size(cores, requests)
                if self.host
                else self.ws.numel()
            )
            assert offset <= available
        self.th, self.to = device_struct(th), device_struct(to)
        self.fn = {}
        for kind, count in [("h", 12), ("o", 10)]:
            f = getattr(self.lib, f"aclrtlaunch_bs_gdn_{kind}")
            f.argtypes = [C.c_uint32, C.c_void_p] + [C.c_void_p] * count
            f.restype = C.c_uint32
            self.fn[kind] = f

    def launch(self, kind, tensors):
        stream = torch.npu.current_stream().npu_stream
        code = self.fn[kind](self.cores, stream, *[t.data_ptr() for t in tensors])
        if code:
            raise RuntimeError(f"AscendC {kind} launch returned {code}")

    def pool_forward(self, q, k, w, u, g, bank, cu, state_meta, indices):
        """Exclusive valid slots; packed active prefix, empty suffix; K-V FP32 bank.

        The scheduler validates unique in-range slots and token/chunk capacities
        before publishing metadata. No data-dependent host read occurs here.
        """
        assert self.state_pool and bank.shape[1:] == (24, 128, 128)
        assert bank.dtype == torch.float32 and bank.is_contiguous()
        assert state_meta.shape == (self.N, 2) and state_meta.dtype == torch.int64
        assert cu.shape == (self.N + 1,) and cu.dtype == torch.int64
        assert indices.shape == (self.C, 2) and indices.dtype == torch.int64
        assert q.shape == k.shape == (1, 8, self.T, 128)
        assert w.shape == u.shape == (1, 24, self.T, 128)
        assert g.shape == (1, 24, self.T) and g.dtype == torch.float32
        assert all(t.dtype == torch.bfloat16 for t in (q, k, w, u))
        assert all(
            t.device == bank.device and t.is_contiguous()
            for t in (q, k, w, u, g, cu, state_meta, indices)
        )
        if self.host:
            output, h, v = self.host.pool_forward(
                q,
                k,
                w,
                u,
                g,
                bank,
                cu,
                state_meta,
                indices,
                self.th,
                self.to,
                self.cores,
            )
            if self.retain_intermediates:
                self.h, self.v = h, v
            return output
        self.launch(
            "h",
            [k, w, u, g, bank, cu, state_meta, self.h, self.v, bank, self.ws, self.th],
        )
        self.launch(
            "o", [q, k, self.v, self.h, g, cu, indices, self.o, self.ws, self.to]
        )
        return self.o

    def __call__(self, q, k, w, u, g, initial, cu, indices):
        assert not self.state_pool
        for x in (q, k, w, u, g, initial, cu, indices):
            assert x.is_contiguous() and x.device.type == "npu"
        assert q.shape == k.shape == (1, 8, self.T, 128)
        assert w.shape == u.shape == (1, 24, self.T, 128)
        assert g.shape == (1, 24, self.T) and g.dtype == torch.float32
        assert (
            initial.shape == (self.N, 24, 128, 128) and initial.dtype == torch.float32
        )
        assert all(x.dtype == torch.bfloat16 for x in (q, k, w, u))
        assert cu.dtype == indices.dtype == torch.int64
        assert cu.numel() == self.N + 1 and indices.numel() >= 2 * self.C
        self.launch(
            "h",
            [
                k,
                w,
                u,
                g,
                initial,
                cu,
                indices,
                self.h,
                self.v,
                self.final,
                self.ws,
                self.th,
            ],
        )
        self.launch(
            "o", [q, k, self.v, self.h, g, cu, indices, self.o, self.ws, self.to]
        )
        return self.o, self.final
