"""Independent-process CPU Store reader; deterministic opaque KV fixture."""
import argparse
import ctypes
import json
from pathlib import Path

from mooncake.store import MooncakeDistributedStore

p = argparse.ArgumentParser()
p.add_argument("--key", required=True)
p.add_argument("--base-port", type=int, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
store = MooncakeDistributedStore()
destination = ctypes.create_string_buffer(2 * 65536 + 64)
registered = False
try:
    rc = store.setup(f"127.0.0.1:{a.base_port+5}",
                     f"http://127.0.0.1:{a.base_port+1}/metadata",
                     0, 16*1024**2, "tcp", "", f"127.0.0.1:{a.base_port}")
    assert rc == 0
    expected = [bytes((i * 7 + h * 31) % 251 for i in range(65536)) for h in range(2)]
    assert store.get(a.key) == b"".join(expected)
    assert store.register_buffer(ctypes.addressof(destination), ctypes.sizeof(destination)) == 0
    registered = True
    ctypes.memset(ctypes.addressof(destination), 0xFE, ctypes.sizeof(destination))
    rc = store.get_into_ranges([ctypes.addressof(destination)], [[a.key]],
                               [[[32, 65568]]], [[[65536, 0]]], [[[65536, 65536]]])
    assert rc == [[[65536, 65536]]]
    assert destination.raw == b"\xfe"*32 + expected[1] + expected[0] + b"\xfe"*32
    a.output.write_text(json.dumps({"full_object": "exact", "head_scatter": "exact",
                                   "guards": "intact", "scope": "separate-process CPU"})+"\n")
finally:
    if registered:
        store.unregister_buffer(ctypes.addressof(destination))
    store.close()
