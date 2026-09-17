"""Qwen38 channel identity and raw kernels; no private forward-path RPC."""

import ctypes as C
import json
from pathlib import Path

import torch
from ipc_acl import PrefixCopyACL

from channel_layout import ALIGN, ChannelLayout

CONTRACT = ChannelLayout().contract()


def acl_api():
    return PrefixCopyACL("/usr/local/Ascend/cann-9.0.1/lib64/libascendcl.so")


class Kernels:
    def __init__(self, build):
        root = Path(build)
        abi = json.loads((root / "abi.json").read_text())
        assert (
            abi.get("kernel_timeout_us") == 1200000000
        ), "Reject microbench launch lifetime"
        assert (
            abi["model"] == "qwen38"
            and abi["client_config_words"] == 17
            and not abi["prefix_pipeline"]
        )
        self.layout = ChannelLayout.from_abi(abi)
        self.root = root
        self.lib = C.CDLL(str(root / "launch.so"))
        self.binaries = []
        self.lib.load_server.argtypes = [
            C.c_char_p,
            C.c_char_p,
            C.POINTER(C.c_void_p),
            C.POINTER(C.c_void_p),
        ]
        self.lib.launch_blocks.argtypes = [C.c_void_p] * 5 + [C.c_uint32]
        self.lib.unload_server.argtypes = [C.c_void_p]

    def load(self, name):
        binary, fn = C.c_void_p(), C.c_void_p()
        assert (
            self.lib.load_server(
                str(self.root / "queue_service.o").encode(),
                name.encode(),
                C.byref(binary),
                C.byref(fn),
            )
            == 0
        )
        self.binaries.append(binary)
        return fn

    def call(self, fn, config, x, ids, blocks=1):
        assert (
            self.lib.launch_blocks(
                fn,
                torch.npu.current_stream().npu_stream,
                config.data_ptr(),
                x.data_ptr(),
                ids.data_ptr(),
                blocks,
            )
            == 0
        )

    def close(self):
        for binary in self.binaries:
            assert self.lib.unload_server(binary) == 0
