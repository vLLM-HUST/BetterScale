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
p.add_argument("--sources", type=int, choices=(1, 2, 4, 5, 8), default=1)
p.add_argument("--tp-size", type=int, choices=(1, 2), default=2)
p.add_argument("--decode-steps", type=int, default=3)
p.add_argument("--align-steady-start", action="store_true")
p.add_argument("--observe-pauses", action="store_true")
p.add_argument("--defer-steady-gc", action="store_true")
p.add_argument("--batch-size", type=int, choices=range(1, 33), default=1)
p.add_argument("--state-gib", type=float, default=4)
p.add_argument("--prompt-width", type=int, default=3)
p.add_argument("--mtp-tokens", type=int, choices=range(0, 6), default=0)
p.add_argument("--reference-tokens", type=int, default=0)
p.add_argument("--colocated", action="store_true")
p.add_argument("--distinct-prompts", action="store_true")
p.add_argument("--trace-plan", type=Path)
p.add_argument("--trace-count", type=int, default=4)
p.add_argument("--trace-turns", type=int, default=0)
p.add_argument("--trace-output-cap", type=int, default=0)
p.add_argument("--trace-max-context", type=int, default=32768)
p.add_argument("--capacity-probe", action="store_true")
a = p.parse_args()

assert a.reference_tokens == 0 or 2 <= a.reference_tokens <= min(32, a.decode_steps + 3)
assert not a.reference_tokens or a.mtp_tokens
from channel_layout import ChannelLayout
import json

layout = ChannelLayout.from_abi(json.loads((a.build / "abi.json").read_text()))
token_capacity = layout.rows
expert_owners = layout.owners
assert (a.sources * a.tp_size == 8) if a.colocated else (a.sources <= layout.sources)
assert 1 <= a.prompt_width <= token_capacity
assert a.batch_size * max(a.prompt_width, a.mtp_tokens + 1) <= token_capacity
devices = a.devices.split(",")
assert (
    len(devices)
    == len(set(devices))
    == (
        a.tp_size * a.sources
        if a.construct_only or a.colocated
        else a.tp_size * a.sources + expert_owners
    )
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
    if not a.construct_only and not a.colocated:
        for owner in range(expert_owners):
            launch(
                f"expert{owner}",
                "server.py",
                [*common, "--owner", str(owner), "--sources", str(a.sources)]
                + (["--mtp"] if a.mtp_tokens else []),
                devices[owner + a.tp_size * a.sources],
            )
        deadline = time.monotonic() + 900
        while not all(
            (a.directory / f"expert{i}.sock").exists() for i in range(expert_owners)
        ):
            if any(c.poll() is not None for c in children):
                raise RuntimeError("server weight startup failed")
            if time.monotonic() > deadline:
                raise TimeoutError("server weight load")
            time.sleep(0.5)
    for physical_rank in range(a.tp_size * a.sources):
        source, rank = divmod(physical_rank, a.tp_size)
        launch(
            f"attention{physical_rank}",
            "model_client.py",
            common
            + [
                "--source",
                str(source),
                "--tp-size",
                str(a.tp_size),
                "--sources",
                str(a.sources),
                "--decode-steps",
                str(a.decode_steps),
            ]
            + [
                "--reference-tokens",
                str(a.reference_tokens),
                "--mtp-tokens",
                str(a.mtp_tokens),
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
            + (["--decode-graph"] if a.decode_graph else [])
            + (["--colocated"] if a.colocated else [])
            + (["--distinct-prompts"] if a.distinct_prompts else [])
            + (
                ["--capacity-probe", "--trace-max-context", str(a.trace_max_context)]
                if a.capacity_probe
                else []
            )
            + (
                [
                    "--trace-plan",
                    str(a.trace_plan),
                    "--trace-count",
                    str(a.trace_count),
                    "--trace-turns",
                    str(a.trace_turns),
                    "--trace-output-cap",
                    str(a.trace_output_cap),
                    "--trace-max-context",
                    str(a.trace_max_context),
                ]
                if a.trace_plan
                else []
            ),
            devices[physical_rank],
            RANK=str(physical_rank if a.colocated else rank),
            LOCAL_RANK="0",
            WORLD_SIZE="8" if a.colocated else str(a.tp_size),
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(37652 if a.colocated else 37652 + source),
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
    if any(c.returncode != 0 for c in children):
        raise RuntimeError("Model roles exited unsuccessfully")
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
