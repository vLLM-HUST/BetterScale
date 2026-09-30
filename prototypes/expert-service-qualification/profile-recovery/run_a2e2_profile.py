#!/usr/bin/env python3
"""Start validated A2E2, run one bounded fixed msprof window, and drain."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

CAPSULE = Path(__file__).resolve().parent
WT = Path("/root/my-ascend-workspace/betterscale/.worktrees/a2e2-revival")
PY = "/workspace/my-ascend-workspace/runs/rp-legacy/20260903T155041Z-layout/rp-upstream-0.25.1/.venv/bin/python"
MODEL = "/workspace/models/Qwen3.5-35B-A3B"
BUILD = "/root/my-ascend-workspace/runs/expert-transport/20260924/mod-fused-build2"


def wait(predicate, timeout, label):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(1)
    raise TimeoutError(label)


def listening(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def stop(proc, timeout=90):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=30)


def device_processes():
    text = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
    found = {}
    for line in text.splitlines():
        cells = [x.strip() for x in line.split("|")[1:-1]]
        if (
            len(cells) >= 5
            and re.fullmatch(r"\d+\s+\d+", cells[0])
            and cells[1].isdigit()
            and cells[4].isdigit()
        ):
            found[int(cells[0].split()[0])] = {
                "host_pid": int(cells[1]),
                "container_pid": int(cells[4]),
            }
    return found


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one bounded A2E2 msprof diagnostic (not a benchmark)."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="New absolute output directory below the Stage10 msprof root.",
    )
    args = parser.parse_args()
    if not args.run_dir.is_absolute():
        parser.error("--run-dir must be an absolute path")
    run = args.run_dir.resolve()
    allowed = Path(
        "/root/my-ascend-workspace/runs/a2e2-revival/20260930-stage11-profile-recovery"
    )
    if run == allowed or allowed not in run.parents:
        parser.error(f"--run-dir must be a child of {allowed}")
    if len(str(run / "deployment/control/e0.sock").encode()) >= 108:
        parser.error("AF_UNIX control path exceeds Linux socket bound")
    if run.exists():
        parser.error(f"--run-dir already exists: {run}")
    return run


RUN = parse_args()
RUN.mkdir(parents=True, exist_ok=False)
env = dict(os.environ)
env["BETTERSCALE_EXPERT_EXTERNAL_WATCHDOG"] = "1"
env["PROFILING_MODE"] = "dynamic"
env["PYTHONPATH"] = str(WT / "src") + os.pathsep + env.get("PYTHONPATH", "")
deploy = [PY, "-m", "betterscale", "serve-experts", MODEL,
          "--output", str(RUN / "deployment"), "--build", BUILD,
          "--devices", "0,1,2,3", "--sources", "2", "--owners", "2",
          "--placement", "layer", "--return-mode", "push", "--port-base", "33640",
          "--max-seqs", "16", "--max-model-len", "262144", "--kv-gib", "32",
          "--mtp-tokens", "2", "--lifetime", "1800"]
router_cmd = [PY, str(WT / "prototypes/expert-service-qualification/agentx_router.py"),
              "--backends", "http://127.0.0.1:33640", "http://127.0.0.1:33641",
              "--dp-size", "1", "--port", "33650", "--receipt", str(RUN / "routing.json")]
(RUN / "commands.json").write_text(json.dumps({"deployment": deploy, "router": router_cmd}, indent=2) + "\n")
deployment = router = None
status = {"status": "STARTED", "scope": "fixed diagnostic; not SWE headline", "started": time.time()}
try:
    with (RUN / "deployment-launch.log").open("w") as log:
        deployment = subprocess.Popen(deploy, cwd=WT, env=env, stdout=log, stderr=subprocess.STDOUT)
    wait(lambda: (RUN / "deployment/ready").exists() or deployment.poll() is not None, 1200, "deployment ready")
    if deployment.poll() is not None:
        raise RuntimeError(f"deployment exited {deployment.returncode}")
    with (RUN / "router.log").open("w") as log:
        router = subprocess.Popen(router_cmd, cwd=WT, env=env, stdout=log, stderr=subprocess.STDOUT)
    wait(lambda: listening(33650) or router.poll() is not None, 30, "router ready")
    if router.poll() is not None:
        raise RuntimeError(f"router exited {router.returncode}")
    wait(lambda: set(device_processes()) >= {0, 1, 2, 3}, 30, "device PID map")
    processes = device_processes()
    role_devices = {"attention0": 0, "attention1": 1, "expert0": 2, "expert1": 3}
    roles = {role: processes[device] for role, device in role_devices.items()}
    for role, row in roles.items():
        command = Path(f"/proc/{row['container_pid']}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        # EngineCore is a multiprocessing child, so its argv need not contain the
        # friendly log name; reject the HTTP wrapper instead. The npu-smi device
        # row itself is the positive evidence that this child owns the NPU.
        if (role.startswith("attention") and "api_server" in command) or (
            role.startswith("expert") and "expert_service.server" not in command
        ):
            raise RuntimeError(f"unexpected {role} container process: {command}")
        # Read only the nonsecret profiler assignment, never persist process environments.
        entries = Path(f"/proc/{row['container_pid']}/environ").read_bytes().split(b"\0")
        if b"PROFILING_MODE=dynamic" not in entries:
            raise RuntimeError(f"{role} missing early dynamic profiling initialization")
        row["profiling_mode"] = "dynamic"
        row["command"] = command
    (RUN / "device-pids.json").write_text(json.dumps(roles, indent=2) + "\n")
    profile_pids = {role: row["container_pid"] for role, row in roles.items()}
    profile_cmd = [PY, str(CAPSULE / "fixed_profile.py"),
                   "--endpoint", "http://127.0.0.1:33650/v1/completions",
                   "--model", "qwen35",
                   "--pids", json.dumps(profile_pids), "--output", str(RUN / "profile")]
    with (RUN / "profile.log").open("w") as log:
        completed = subprocess.run(profile_cmd, cwd=WT, env=env, stdout=log, stderr=subprocess.STDOUT)
    status["profile_exit"] = completed.returncode
    if completed.returncode:
        raise RuntimeError(f"profile exited {completed.returncode}")
    status["status"] = "PROFILED"
finally:
    stop(router)
    if deployment is not None and deployment.poll() is None:
        (RUN / "deployment/stop").touch()
        try:
            deployment.wait(timeout=240)
        except subprocess.TimeoutExpired:
            stop(deployment)
    status["deployment_exit"] = None if deployment is None else deployment.poll()
    status["router_exit"] = None if router is None else router.poll()
    if status["status"] == "PROFILED" and status["deployment_exit"] == 0:
        status["status"] = "PASS"
    else:
        status["status"] = "FAIL"
    status["finished"] = time.time()
    (RUN / "status.json").write_text(json.dumps(status, indent=2) + "\n")
