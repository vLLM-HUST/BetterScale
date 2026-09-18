"""Freeze an existing server closure; rebuild only the compatible client binary."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def append_pipeline(client):
    assert "META(neural_collect_fused)" in client
    assert "META(neural_collect_pipelined)" not in client
    return (
        client
        + "\n"
        + Path(__file__).with_name("client_reduce_pipeline.cpp").read_text()
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    shutil.copytree(a.base, a.output)
    here = Path(__file__).resolve().parent
    source = a.output.resolve() / "source/client_kernel.cpp"
    source.write_text(append_pipeline(source.read_text()))
    env = dict(
        os.environ,
        OUTPUT_DIR=str(a.output.resolve()),
        SOURCE=str(source),
        OBJECT_NAME="queue_service",
        LAUNCH_SOURCE=str(here / "launch.cpp"),
    )
    with (a.output / "client_pipeline.build.log").open("w") as log:
        subprocess.run(
            ["bash", str(here.parent / "device-service/build.sh")],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    abi = json.loads((a.output / "abi.json").read_text())
    abi["pipelined_client_collect"] = True
    (a.output / "abi.json").write_text(json.dumps(abi, indent=2) + "\n")
