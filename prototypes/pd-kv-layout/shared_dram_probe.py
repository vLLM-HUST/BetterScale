"""Prove direct DMA and cross-process ownership of one memfd DRAM object."""
import argparse
import ctypes
import json
import mmap
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch_npu


def run(args):
    torch.npu.set_device(args.device)
    size = 16 << 20
    owned = args.fd is None
    fd = os.memfd_create("betterscale-state-capability", os.MFD_CLOEXEC) if owned else args.fd
    if owned:
        os.ftruncate(fd, size)
    if args.readonly and owned:
        raise ValueError("Read-only mode requires an inherited initialized FD")
    mapping = mmap.mmap(fd, size, flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ if args.readonly else mmap.PROT_READ | mmap.PROT_WRITE)
    view = np.ndarray((size,), dtype=np.uint8, buffer=mapping)
    expected = np.full(size, 37, dtype=np.uint8)
    expected[::4096] = 91
    if owned:
        view[:] = expected  # First-touch under the launcher's explicit NUMA policy.
    pointer = view.ctypes.data
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    acl.aclrtHostRegister.restype = ctypes.c_int
    acl.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    acl.aclrtHostUnregister.restype = ctypes.c_int
    alias = ctypes.c_void_p()
    begin = time.perf_counter()
    register_type = 8 if args.readonly else 0
    rc = acl.aclrtHostRegister(pointer, size, register_type, ctypes.byref(alias))
    result = dict(scope="16MiB MAP_SHARED memfd; no Store/model/performance qualification",
                  device=args.device, readonly=args.readonly, register_type=register_type, register_rc=rc, register_seconds=time.perf_counter()-begin)
    registered, drained = rc == 0, True
    try:
        if not registered:
            return result
        import vllm_ascend.vllm_ascend_C
        device = torch.empty(size, dtype=torch.uint8, device=f"npu:{args.device}")
        stream = torch.npu.Stream(device=args.device)

        def copy(to_host):
            nonlocal drained
            src, dst = (device.data_ptr(), pointer) if to_host else (pointer, device.data_ptr())
            with torch.npu.stream(stream):
                drained = False
                torch.ops._C_ascend.swap_blocks_batch(
                    torch.tensor([src],dtype=torch.int64),
                    torch.tensor([dst],dtype=torch.int64),
                    torch.tensor([size],dtype=torch.int64),1 if to_host else 0)
                event = torch.npu.Event()
                event.record(stream)
            event.synchronize()
            drained = True

        copy(False)
        assert np.array_equal(device.cpu().numpy(), expected)
        result["exact_h2d"] = True
        if args.readonly:
            return result
        view[:] = 0
        if not owned:
            with torch.npu.stream(stream):
                device.fill_(113)
            expected[:] = 113
        copy(True)
        assert np.array_equal(view, expected)
        result["exact_roundtrip"] = True
        if args.peer is not None:
            peer_output = args.output.with_suffix(".peer.json")
            with args.output.with_suffix(".peer.log").open("x") as log:
                child = subprocess.run(
                    [sys.executable, __file__, "--device", str(args.peer),
                     "--fd", str(fd), "--output", str(peer_output)]
                    + (["--readonly"] if args.peer_readonly else []),
                    pass_fds=(fd,),stdout=log,stderr=subprocess.STDOUT,timeout=90)
            assert child.returncode == 0, child.returncode
            peer = json.loads(peer_output.read_text())
            if args.peer_readonly:
                assert peer.get("exact_h2d"), peer
                assert np.array_equal(view, expected), "Read-only peer changed owner bytes"
            else:
                assert peer.get("exact_roundtrip"), peer
                assert (view == 113).all(), "Child device write did not reach owner's mapping"
            result["exact_cross_process_shared_pages"] = True
            result["peer"] = peer
        return result
    finally:
        if not drained:
            # A failed drain intentionally leaves registration/mapping retained.
            stream.synchronize()
        if registered:
            result["unregister_rc"] = acl.aclrtHostUnregister(pointer)
            assert result["unregister_rc"] == 0
        del view
        mapping.close()
        os.close(fd)
        args.output.write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",type=int,default=0)
    p.add_argument("--peer",type=int)
    p.add_argument("--fd",type=int)
    p.add_argument("--readonly",action="store_true")
    p.add_argument("--peer-readonly",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    if args.output.exists(): raise ValueError("Preserve existing evidence")
    run(args)
