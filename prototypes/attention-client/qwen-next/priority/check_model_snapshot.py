"""Check pinned snapshot structure and tensor placement without rehashing payloads."""

import argparse
import collections
import json
import math
import struct
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", type=Path)
parser.add_argument(
    "manifest", type=Path, help="ModelScope files API response at exact revision"
)
parser.add_argument("output", type=Path)
args = parser.parse_args()
ROOT = args.model
WIDTHS = {
    "F64": 8,
    "I64": 8,
    "F32": 4,
    "I32": 4,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}
manifest = json.loads(args.manifest.read_text())["Data"]["Files"]
for item in manifest:
    if item["Type"] == "blob":
        file = ROOT / item["Path"]
        assert file.is_file() and file.stat().st_size == item["Size"], str(file)
index = json.loads((ROOT / "quant_model_weights.safetensors.index.json").read_text())
seen = {}
bytes_by_role = collections.Counter()
bytes_by_dtype = collections.Counter()
counts = collections.Counter()
for file in sorted(ROOT.glob("*.safetensors")):
    with file.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        assert 0 < size < 64 * 1024**2, file.name
        header = json.loads(stream.read(size))
    cursor = 0
    entries = sorted(
        ((k, v) for k, v in header.items() if k != "__metadata__"),
        key=lambda pair: pair[1]["data_offsets"],
    )
    for name, tensor in entries:
        start, end = tensor["data_offsets"]
        assert start == cursor and end >= start, (file.name, name, start, cursor)
        assert end - start == math.prod(tensor["shape"]) * WIDTHS[tensor["dtype"]], name
        assert name not in seen, name
        assert index["weight_map"].get(name) == file.name, name
        seen[name] = file.name
        role = (
            "ple"
            if ".ple." in name
            else (
                "mtp_routed"
                if name.startswith("mtp.") and ".experts." in name
                else "target_routed" if ".experts." in name else "other"
            )
        )
        bytes_by_role[role] += end - start
        bytes_by_dtype[tensor["dtype"]] += end - start
        counts[role] += 1
        cursor = end
    assert 8 + size + cursor == file.stat().st_size, file.name
assert set(seen) == set(index["weight_map"]), "index/header coverage differs"
assert len(list(ROOT.glob("*.safetensors"))) == 61
receipt = dict(
    model=str(ROOT),
    files=61,
    tensor_count=len(seen),
    bytes_by_role=bytes_by_role,
    bytes_by_dtype=bytes_by_dtype,
    tensor_counts_by_role=counts,
    index_declared_bytes=index.get("metadata", {}).get("total_size"),
    header_payload_bytes=sum(bytes_by_role.values()),
    validation="pinned manifest sizes; complete index/header coverage; dtype/shape/offset bounds; SDK handles hashes",
)
args.output.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
