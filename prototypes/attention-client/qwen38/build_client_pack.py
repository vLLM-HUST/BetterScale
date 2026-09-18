"""Add the pack/publish exports without rebuilding or changing the server ABI."""

import json
import os
from pathlib import Path
import shutil
import subprocess
from channel_layout import ChannelLayout


def append_pack(client, layout):
    assert "void\nneural_pack(" not in client
    constants = f"\nconstexpr int PACK_SCALES = {layout.scales}, PACK_PAYLOAD = {layout.payload};\n"
    return client + constants + Path(__file__).with_name("client_pack.cpp").read_text()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    shutil.copytree(a.base, a.output)
    abi = json.loads((a.output / "abi.json").read_text())
    layout = ChannelLayout.from_abi(abi)
    source = a.output / "source/client_kernel.cpp"
    source.write_text(append_pack(source.read_text(), layout))
    here = Path(__file__).resolve().parent
    env = dict(
        os.environ,
        OUTPUT_DIR=str(a.output.resolve()),
        SOURCE=str(source.resolve()),
        OBJECT_NAME="queue_service",
        LAUNCH_SOURCE=str(here / "launch.cpp"),
    )
    with (a.output / "client_pack.build.log").open("w") as log:
        subprocess.run(
            ["bash", str(here.parent / "device-service/build.sh")],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    abi["parallel_client_pack"] = True
    (a.output / "abi.json").write_text(json.dumps(abi, indent=2) + "\n")
