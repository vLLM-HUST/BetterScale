"""Read safetensors headers only; checkpoint bytes are not runtime peak memory."""

import argparse
import collections
import json
import struct
from pathlib import Path


def census(directory):
    totals = collections.Counter()
    files = sorted(Path(directory).glob("*.safetensors"))
    if not files:
        raise ValueError("no safetensors files")
    for path in files:
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            role = "draft" if name.startswith("mtp.") else "target"
            kind = (
                "routed"
                if ".experts." in name
                else (
                    "shared"
                    if ".shared_experts." in name or ".shared_expert." in name
                    else "other"
                )
            )
            start, end = tensor["data_offsets"]
            totals[f"{role}/{kind}"] += end - start
    return dict(
        path=str(directory),
        files=len(files),
        bytes=dict(totals),
        gib={key: value / 2**30 for key, value in totals.items()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="+")
    args = parser.parse_args()
    print(json.dumps([census(path) for path in args.directory], indent=2))
