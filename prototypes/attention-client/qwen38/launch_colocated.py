"""Bounded EP8 leaf supervisor. Admission is owned by the outer wrapper."""

import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--devices", required=True)
p.add_argument("--directory", type=Path, required=True)
a = p.parse_args()
devices = a.devices.split(",")
assert len(devices) == len(set(devices)) == 8
a.directory.mkdir(parents=True, exist_ok=False)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
children = []


def interrupted(signum, frame):
    raise InterruptedError(f"launcher signal {signum}")


signal.signal(signal.SIGTERM, interrupted)
try:
    for rank, device in enumerate(devices):
        with (a.directory / f"rank{rank}.log").open("w") as log:
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("probe_colocated.py")),
                        "--output",
                        str(a.directory),
                    ],
                    env=dict(
                        os.environ,
                        ASCEND_RT_VISIBLE_DEVICES=device,
                        RANK=str(rank),
                        LOCAL_RANK="0",
                        WORLD_SIZE="8",
                        MASTER_ADDR="127.0.0.1",
                        MASTER_PORT=str(port),
                    ),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
    deadline = time.monotonic() + 600
    while any(c.poll() is None for c in children):
        if any(c.poll() not in (None, 0) for c in children):
            raise RuntimeError("EP rank failed; inspect rank logs")
        if time.monotonic() > deadline:
            raise TimeoutError("EP leaf exceeded600s")
        time.sleep(0.5)
    if any(c.returncode != 0 for c in children):
        raise RuntimeError("EP ranks exited unsuccessfully")
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
