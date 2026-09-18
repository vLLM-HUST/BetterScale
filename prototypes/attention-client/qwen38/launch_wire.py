"""Bounded five-device wire gate supervisor; stops only its own children."""

import argparse
import json
from channel_layout import ChannelLayout
import os
from pathlib import Path
import subprocess
import shutil
import signal
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--devices", required=True)
p.add_argument("--directory", type=Path, required=True)
p.add_argument("--build", type=Path, required=True)
p.add_argument(
    "--client-probe",
    choices=("probe_wire.py", "probe_shared_overlap.py"),
    default="probe_wire.py",
)
p.add_argument("--sources", type=int, default=1)
p.add_argument("--artifacts", type=Path)
a = p.parse_args()
devices = a.devices.split(",")
layout = ChannelLayout.from_abi(json.loads((a.build / "abi.json").read_text()))
owners = layout.owners
assert 1 <= a.sources <= layout.sources
assert len(devices) == len(set(devices)) == a.sources + owners
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


def cancelled(signum, frame):
    raise KeyboardInterrupt("wire gate cancelled")


signal.signal(signal.SIGTERM, cancelled)
try:
    for owner in range(owners):
        launch(
            f"expert{owner}",
            [
                "server.py",
                "--owner",
                str(owner),
                "--layers",
                "1",
                "--sources",
                str(a.sources),
                "--build",
                str(a.build),
                "--directory",
                str(a.directory),
            ],
            devices[owner + a.sources],
        )
    deadline = time.monotonic() + 600
    while not all((a.directory / f"expert{i}.sock").exists() for i in range(owners)):
        if any(c.poll() is not None for c in children):
            raise RuntimeError("server startup failed")
        if time.monotonic() > deadline:
            raise TimeoutError("weight load")
        time.sleep(0.5)
    for source in range(a.sources):
        launch(
            f"client{source}",
            [
                a.client_probe,
                "--build",
                str(a.build),
                "--directory",
                str(a.directory),
                "--source",
                str(source),
            ],
            devices[source],
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

    if a.artifacts is not None:
        a.artifacts.mkdir(parents=True, exist_ok=True)
        for pattern in ("*.log", "*.json"):
            for path in a.directory.glob(pattern):
                shutil.copyfile(path, a.artifacts / path.name)
