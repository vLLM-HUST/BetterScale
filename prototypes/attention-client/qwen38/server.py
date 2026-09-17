"""Qwen38 expert process. Host only discovers channels and drains at shutdown.

Each attention TP group has one publishing leader and its own mailbox.
Unused sources are closed; active sources drain independently before reclamation.
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
from wire import ALIGN, acl_api
from channel_layout import ChannelLayout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--owner", type=int, choices=range(4), required=True)
    p.add_argument("--layers", type=int, default=48)
    p.add_argument("--mtp", action="store_true")
    p.add_argument("--sources", type=int, choices=(1, 2, 4, 5), default=1)
    a = p.parse_args()
    layout = ChannelLayout.from_abi(json.loads((a.build / "abi.json").read_text()))
    contract = layout.contract()
    assert a.sources <= layout.sources
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch_npu.npu.config.allow_internal_format = True
    acl = C.CDLL("/usr/local/Ascend/cann-9.0.1/lib64/libascendcl.so")
    acl.aclrtSetOpExecuteTimeOut.argtypes = [C.c_uint32]
    assert acl.aclrtSetOpExecuteTimeOut(1200) == 0
    catalog = load(a.owner, layers=a.layers, mtp=a.mtp, owners=layout.owners)
    api = acl_api()
    path = a.directory / f"expert{a.owner}.sock"
    listener = listen(path)
    listener.settimeout(1200)
    channels = {}
    outputs, output_keys = {}, {}
    sources, source_keys = {}, {}
    zero = torch.zeros(layout.output_bytes // 4, dtype=torch.int32, device="npu")
    # Respond to each hello before waiting for source registration: clients
    # discover all four owners before exporting their source to those PIDs.
    for _ in range(a.sources):
        channel = Channel(listener.accept()[0])
        channel.sock.settimeout(1200)
        hello = channel.expect("hello")
        source_id = hello["source"]
        assert 0 <= source_id < a.sources and source_id not in channels
        assert hello["contract"] == contract
        output = api.allocate_staging(layout.output_bytes)
        api.copy(
            torch.npu.current_stream().npu_stream,
            output,
            zero.data_ptr(),
            layout.output_bytes,
        )
        torch.npu.synchronize()
        key = api.export(output, layout.output_bytes, (hello["pid"],))
        channel.send(
            dict(
                op="window",
                owner=a.owner,
                pid=api.pid(),
                key=key.decode(),
                contract=contract,
            )
        )
        channels[source_id] = channel
        outputs[source_id], output_keys[source_id] = output, key
    listener.close()
    for source_id, channel in channels.items():
        key = channel.expect("source")["key"].encode()
        source_keys[source_id] = key
        sources[source_id] = api.import_memory(key)
        channel.send(dict(op="registered"))
    closed_source = torch.zeros_like(zero)
    closed_source[0] = -1
    unused_output = torch.zeros_like(zero)
    engine = Engine(
        a.build,
        [sources.get(i, closed_source.data_ptr()) for i in range(layout.sources)],
        [outputs.get(i, unused_output.data_ptr()) for i in range(layout.sources)],
        catalog,
        a.owner,
        tasks=32,
        open_service=True,
    )
    engine.replay()
    # Host-side accounting only; never synchronize the persistent service here.
    free_bytes, total_bytes = torch.npu.mem_get_info()
    memory_after_start = dict(
        allocated=torch.npu.memory_allocated(),
        reserved=torch.npu.memory_reserved(),
        driver_free=free_bytes,
        driver_total=total_bytes,
    )
    for channel in channels.values():
        channel.send(dict(op="ready"))
    counts = [0] * layout.sources
    for source_id, channel in channels.items():
        while True:
            message = channel.read()
            if (
                message.get("op") == "inspect"
                and os.environ.get("QWEN38_DIAGNOSTICS") == "1"
            ):
                sample = torch.empty(32, dtype=torch.int32, device="npu")
                api.copy(
                    torch.npu.current_stream().npu_stream,
                    sample.data_ptr(),
                    sources[source_id],
                    128,
                )
                source_head = sample.cpu().tolist()
                api.copy(
                    torch.npu.current_stream().npu_stream,
                    sample.data_ptr(),
                    outputs[source_id],
                    128,
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
                            [s[5][c, :3].cpu().tolist() for c in range(layout.sources)]
                            for s in engine.slots
                        ],
                        enabled=int(engine.config[10].cpu()),
                    )
                )
                continue
            if message.get("op") != "drain":
                raise ValueError("Expected drain or diagnostic inspection")
            counts[source_id] = message["generation"]
            break
    receipt = engine.finish()
    assert receipt["completed_counts"] == counts, receipt
    for channel in channels.values():
        channel.send(dict(op="drained"))
    for source_id, channel in channels.items():
        channel.expect("unmapped")
        api.close_mapping(source_keys[source_id])
        api.close_mapping(output_keys[source_id])
        api.free_staging(outputs[source_id])
        channel.send(dict(op="released"))
        channel.close()
    engine.close()
    path.unlink()
    (a.directory / f"expert{a.owner}.json").write_text(
        json.dumps(
            dict(
                status="pass",
                completed=sum(counts),
                sources=a.sources,
                layers=len(catalog),
                memory_after_start=memory_after_start,
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
