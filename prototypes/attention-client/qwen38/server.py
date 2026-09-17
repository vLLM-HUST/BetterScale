"""Qwen38 expert process. Host only discovers channels and drains at shutdown.

One attention TP group has one publishing leader. The second device mailbox is
closed from the outset, preserving the existing two-source coordinator ABI.
"""

import argparse
import ctypes as C
import json
import os
from pathlib import Path

import torch
import torch_npu
from catalog import load
from control import Channel, listen
from server_engine import Engine
from wire import ALIGN, CONTRACT, acl_api


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--owner", type=int, choices=range(4), required=True)
    p.add_argument("--layers", type=int, default=48)
    p.add_argument("--mtp", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch_npu.npu.config.allow_internal_format = True
    acl = C.CDLL("/usr/local/Ascend/cann-9.0.1/lib64/libascendcl.so")
    acl.aclrtSetOpExecuteTimeOut.argtypes = [C.c_uint32]
    assert acl.aclrtSetOpExecuteTimeOut(1200) == 0
    catalog = load(a.owner, layers=a.layers, mtp=a.mtp)
    api = acl_api()
    path = a.directory / f"expert{a.owner}.sock"
    listener = listen(path)
    listener.settimeout(1200)
    channel = Channel(listener.accept()[0])
    channel.sock.settimeout(1200)
    hello = channel.expect("hello")
    assert hello["source"] == 0 and hello["contract"] == CONTRACT
    output = api.allocate_staging(ALIGN)
    zero = torch.zeros(ALIGN // 4, dtype=torch.int32, device="npu")
    api.copy(torch.npu.current_stream().npu_stream, output, zero.data_ptr(), ALIGN)
    torch.npu.synchronize()
    key = api.export(output, ALIGN, (hello["pid"],))
    channel.send(
        dict(
            op="window",
            owner=a.owner,
            pid=api.pid(),
            key=key.decode(),
            contract=CONTRACT,
        )
    )
    source_key = channel.expect("source")["key"].encode()
    source = api.import_memory(source_key)
    channel.send(dict(op="registered"))
    listener.close()
    closed_source = torch.zeros_like(zero)
    closed_source[0] = -1
    unused_output = torch.zeros_like(zero)
    engine = Engine(
        a.build,
        [source, closed_source.data_ptr()],
        [output, unused_output.data_ptr()],
        catalog,
        a.owner,
        tasks=32,
        open_service=True,
    )
    engine.replay()
    channel.send(dict(op="ready"))
    while True:
        message = channel.read()
        if (
            message.get("op") == "inspect"
            and os.environ.get("QWEN38_DIAGNOSTICS") == "1"
        ):
            sample = torch.empty(32, dtype=torch.int32, device="npu")
            api.copy(
                torch.npu.current_stream().npu_stream, sample.data_ptr(), source, 128
            )
            source_head = sample.cpu().tolist()
            api.copy(
                torch.npu.current_stream().npu_stream, sample.data_ptr(), output, 128
            )
            output_head = sample.cpu().tolist()
            state = engine.control.cpu().tolist()
            channel.send(
                dict(
                    op="inspection",
                    source=source_head,
                    output=output_head,
                    control={i: state[i][:8] for i in (0, 1, 2, 43)},
                    slot_headers=[
                        [s[5][c, :3].cpu().tolist() for c in range(2)]
                        for s in engine.slots
                    ],
                    enabled=int(engine.config[10].cpu()),
                )
            )
            continue
        if message.get("op") != "drain":
            raise ValueError("Expected drain or diagnostic inspection")
        count = message["generation"]
        break
    receipt = engine.finish()
    assert receipt["completed_counts"] == [count, 0], receipt
    channel.send(dict(op="drained"))
    channel.expect("unmapped")
    api.close_mapping(source_key)
    api.close_mapping(key)
    api.free_staging(output)
    channel.send(dict(op="released"))
    channel.close()
    engine.close()
    path.unlink()
    (a.directory / f"expert{a.owner}.json").write_text(
        json.dumps(
            dict(
                status="pass",
                completed=count,
                layers=len(catalog),
                weight_bytes=sum(
                    t.numel() * t.element_size()
                    for layer in catalog
                    for t in layer
                    if t is not None
                ),
                **receipt,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
