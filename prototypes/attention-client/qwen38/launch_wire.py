"""Bounded five-device wire gate supervisor; stops only its own children."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--devices", required=True)
p.add_argument("--directory", type=Path, required=True)
p.add_argument("--build", type=Path, required=True)
a = p.parse_args()
devices = a.devices.split(",")
assert len(devices) == len(set(devices)) == 5
a.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
children = []


def launch(name, args, device):
    with (a.directory / f"{name}.log").open("w") as log:
        children.append(
            subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name(args[0])), *args[1:]],
                env=dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=device),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )


try:
    for owner in range(4):
        launch(
            f"expert{owner}",
            [
                "server.py",
                "--owner",
                str(owner),
                "--layers",
                "1",
                "--build",
                str(a.build),
                "--directory",
                str(a.directory),
            ],
            devices[owner + 1],
        )
    deadline = time.monotonic() + 600
    while not all((a.directory / f"expert{i}.sock").exists() for i in range(4)):
        if any(c.poll() is not None for c in children):
            raise RuntimeError("server startup failed")
        if time.monotonic() > deadline:
            raise TimeoutError("weight load")
        time.sleep(0.5)
    launch(
        "client",
        ["probe_wire.py", "--build", str(a.build), "--directory", str(a.directory)],
        devices[0],
    )
    deadline = time.monotonic() + 600
    while any(c.poll() is None for c in children):
        if any(c.poll() not in (None, 0) for c in children):
            raise RuntimeError("role failure")
        if time.monotonic() > deadline:
            raise TimeoutError("wire gate")
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
