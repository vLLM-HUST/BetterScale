"""Independent A(TP2)+E4 roles; bounded fail-stop, no inherited IPC handles."""

import argparse
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
p.add_argument("--construct-only", action="store_true")
p.add_argument("--decode-graph", action="store_true")
p.add_argument("--artifacts", type=Path)
p.add_argument("--sources", type=int, choices=(1, 2), default=1)
p.add_argument("--decode-steps", type=int, default=3)
p.add_argument("--align-steady-start", action="store_true")
p.add_argument("--observe-pauses", action="store_true")
p.add_argument("--defer-steady-gc", action="store_true")
p.add_argument("--batch-size", type=int, choices=range(1, 33), default=1)
p.add_argument("--state-gib", type=float, default=4)
p.add_argument("--prompt-width", type=int, choices=(1, 3), default=3)
a = p.parse_args()
assert a.batch_size * a.prompt_width <= 32
devices = a.devices.split(",")
assert (
    len(devices)
    == len(set(devices))
    == (2 * a.sources if a.construct_only else 2 * a.sources + 4)
)
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


def cancelled(signum, frame):
    raise KeyboardInterrupt("model gate cancelled")


signal.signal(signal.SIGTERM, cancelled)
common = ["--directory", str(a.directory), "--build", str(a.build)]
try:
    if not a.construct_only:
        for owner in range(4):
            launch(
                f"expert{owner}",
                "server.py",
                [*common, "--owner", str(owner), "--sources", str(a.sources)],
                devices[owner + 2 * a.sources],
            )
        deadline = time.monotonic() + 900
        while not all((a.directory / f"expert{i}.sock").exists() for i in range(4)):
            if any(c.poll() is not None for c in children):
                raise RuntimeError("server weight startup failed")
            if time.monotonic() > deadline:
                raise TimeoutError("server weight load")
            time.sleep(0.5)
    for physical_rank in range(2 * a.sources):
        source, rank = divmod(physical_rank, 2)
        launch(
            f"attention{physical_rank}",
            "model_client.py",
            common
            + [
                "--source",
                str(source),
                "--sources",
                str(a.sources),
                "--decode-steps",
                str(a.decode_steps),
            ]
            + [
                "--batch-size",
                str(a.batch_size),
                "--state-gib",
                str(a.state_gib),
                "--prompt-width",
                str(a.prompt_width),
            ]
            + (["--align-steady-start"] if a.align_steady_start else [])
            + (["--observe-pauses"] if a.observe_pauses else [])
            + (["--defer-steady-gc"] if a.defer_steady_gc else [])
            + (["--construct-only"] if a.construct_only else [])
            + (["--decode-graph"] if a.decode_graph else []),
            devices[physical_rank],
            RANK=str(rank),
            LOCAL_RANK="0",
            WORLD_SIZE="2",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(37652 + source),
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
