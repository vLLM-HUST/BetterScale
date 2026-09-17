"""Independent A(TP2)+E4 roles; bounded fail-stop, no inherited IPC handles."""

import argparse
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--devices", required=True)
p.add_argument("--directory", type=Path, required=True)
p.add_argument("--build", type=Path, required=True)
p.add_argument("--construct-only", action="store_true")
p.add_argument("--decode-graph", action="store_true")
p.add_argument("--artifacts", type=Path)
a = p.parse_args()
devices = a.devices.split(",")
assert len(devices) == len(set(devices)) == (2 if a.construct_only else 6)
a.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
children = []


def launch(name, script, args, device, **env):
    with (a.directory / f"{name}.log").open("w") as log:
        children.append(
            subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name(script)), *args],
                env=dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=device, **env),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )


common = ["--directory", str(a.directory), "--build", str(a.build)]
try:
    if not a.construct_only:
        for owner in range(4):
            launch(
                f"expert{owner}",
                "server.py",
                [*common, "--owner", str(owner)],
                devices[owner + 2],
            )
        deadline = time.monotonic() + 900
        while not all((a.directory / f"expert{i}.sock").exists() for i in range(4)):
            if any(c.poll() is not None for c in children):
                raise RuntimeError("server weight startup failed")
            if time.monotonic() > deadline:
                raise TimeoutError("server weight load")
            time.sleep(0.5)
    for rank in range(2):
        launch(
            f"attention{rank}",
            "model_client.py",
            common
            + (["--construct-only"] if a.construct_only else [])
            + (["--decode-graph"] if a.decode_graph else []),
            devices[rank],
            RANK=str(rank),
            LOCAL_RANK="0",
            WORLD_SIZE="2",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="37652",
        )
    deadline = time.monotonic() + 1200
    while any(c.poll() is None for c in children):
        if any(c.poll() not in (None, 0) for c in children):
            raise RuntimeError(
                f"role failure; launch-order exit codes: {[c.poll() for c in children]}"
            )
        if time.monotonic() > deadline:
            raise TimeoutError("model gate")
        time.sleep(0.5)
finally:
    for c in children:
        if c.poll() is None:
            c.terminate()
    for c in children:
        try:
            c.wait(10)
        except subprocess.TimeoutExpired:
            c.kill()
            c.wait()

    if a.artifacts is not None:
        a.artifacts.mkdir(parents=True, exist_ok=True)
        for pattern in ("*.log", "*.json", "last-client-input.pt"):
            for path in a.directory.glob(pattern):
                shutil.copyfile(path, a.artifacts / path.name)
