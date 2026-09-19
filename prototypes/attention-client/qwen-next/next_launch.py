"""Test supervisor only: roles discover UNIX endpoints, not inherited IPC pipes."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--devices", required=True)
p.add_argument("--directory", type=Path, required=True)
p.add_argument("--artifacts", type=Path)
a = p.parse_args()
devices = a.devices.split(",")
assert len(set(devices)) == 6
# Short explicit path: UNIX socket paths have a small fixed-size limit.
a.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
children = []
try:
    for role in (2, 3, 4, 5, 0, 1):
        env = dict(
            os.environ,
            ASCEND_RT_VISIBLE_DEVICES=devices[role],
            EXPERT_ROLE_DIRECTORY=str(a.directory),
            EXPERT_SOURCE_ID=str(role),
            VLLM_PORT=str(36300 + role * 100),
            MASTER_PORT=str(36301 + role * 100),
        )
        command = (
            [
                sys.executable,
                str(Path(__file__).with_name("next_server.py")),
                "--owner",
                str(role - 2),
                "--directory",
                str(a.directory),
            ]
            if role >= 2
            else [
                sys.executable,
                str(
                    Path(__file__).with_name(
                        "next_wire_client.py"
                        if os.environ.get("NEXT_WIRE_ONLY") == "1"
                        else "next_client.py"
                    )
                ),
            ]
        )
        log = (a.directory / f"role{role}.log").open("w")
        children.append(
            subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        )
        log.close()
    deadline = time.monotonic() + 1200
    while any(child.poll() is None for child in children):
        if any(child.poll() not in (None, 0) for child in children):
            raise RuntimeError(
                f"role exits (server0..3,client0,client1): {[c.poll() for c in children]}"
            )
        if time.monotonic() > deadline:
            raise TimeoutError("independent role gate")
        time.sleep(0.25)
    assert all(child.returncode == 0 for child in children)
finally:
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()

    if a.artifacts is not None:
        a.artifacts.mkdir(parents=True, exist_ok=True)
        for pattern in ("*.json", "*.log"):
            for source in a.directory.glob(pattern):
                shutil.copyfile(source, a.artifacts / source.name)
