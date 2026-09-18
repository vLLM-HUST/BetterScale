"""Extend a frozen pack-capable client with token-owned fused collection."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from expert_partition import ExpertPartition


def append_reduce(client, owners):
    assert "void\nneural_collect_fused(" not in client
    constants = f"\nconstexpr int COLLECT_OWNERS = {owners}, COLLECT_EXPERTS = {ExpertPartition(owners).slots};\n"
    return (
        client + constants + Path(__file__).with_name("client_reduce.cpp").read_text()
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    shutil.copytree(a.base, a.output)
    abi = json.loads((a.output / "abi.json").read_text())
    source = a.output / "source/client_kernel.cpp"
    source.write_text(append_reduce(source.read_text(), abi["owners"]))
    here = Path(__file__).resolve().parent
    env = dict(
        os.environ,
        OUTPUT_DIR=str(a.output.resolve()),
        SOURCE=str(source.resolve()),
        OBJECT_NAME="queue_service",
        LAUNCH_SOURCE=str(here / "launch.cpp"),
    )
    with (a.output / "client_reduce.build.log").open("w") as log:
        subprocess.run(
            ["bash", str(here.parent / "device-service/build.sh")],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    abi["fused_client_collect"] = True
    (a.output / "abi.json").write_text(json.dumps(abi, indent=2) + "\n")
