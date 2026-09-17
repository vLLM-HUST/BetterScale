"""Matched TP2 services: pair-swap in parallel, or ABBA on one admitted pair."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from probe_host_npus import parse_devices

root = Path(os.environ["CAPSULE"])
pairs = os.environ.get("SWE_DEVICE_PAIRS", "0,1;6,7").split(";")
devices = {int(x) for pair in pairs for x in pair.split(",")}
assert len(pairs) in (1, 2) and len(devices) == 2 * len(pairs)
assert devices <= set(range(8))


def wait_reclaimed(wave):
    # Leases stay held by admission; an API process exit is not proof that
    # native workers/driver resources have left. Recheck before every reload.
    deadline = time.monotonic() + 600
    while True:
        sample = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
        (wave / "reload-admission.txt").write_text(sample)
        readings = parse_devices(sample)
        if all(
            d in readings
            and readings[d].hbm_used_mb <= 4096
            and readings[d].aicore_percent == 0
            and re.search(rf"\|\s*{d}\s+910B2\s*\|\s*OK", sample)
            and f"No running processes found in NPU {d}" in sample
            for d in devices
        ):
            return
        if time.monotonic() > deadline:
            raise TimeoutError("selected pair resources not reclaimed before reload")
        time.sleep(2)


receipt = dict(
    status="RUNNING",
    rounds=[],
    scope=(
        "Sequential same-pair ABBA; closed-loop whole-trajectory cohorts."
        if len(pairs) == 1
        else "Simultaneous same-host pairs; swap arms across pairs; closed-loop cohorts synchronized only at start. Shared-host CPU effects are not isolated."
    ),
)
plan = (
    [
        (0, [(0, "baseline")]),
        (0, [(0, "candidate")]),
        (1, [(0, "candidate")]),
        (1, [(0, "baseline")]),
    ]
    if len(pairs) == 1
    else [
        (
            repeat,
            [
                (pair, ("baseline", "candidate")[(pair + repeat) % 2])
                for pair in range(2)
            ],
        )
        for repeat in range(2)
    ]
)
# Candidate-only regression reuses a retained native control; never labels it a
# fresh paired comparison. Profiles remain opt-in for this abbreviated route.
if os.environ.get("SWE_CANDIDATE_ONLY") == "1":
    assert len(pairs) == 1
    plan = [(0, [(0, "candidate")]), (1, [(0, "candidate")])]
    receipt["scope"] = (
        "Two same-pair candidate-only cohorts; retained controls, no fresh baseline."
    )
profile_enabled = os.environ.get("SWE_PROFILE", "1") == "1"
try:
    for repeat, entries in plan:
        wave = root / f"round{repeat}"
        wave.mkdir(exist_ok=True)
        for phase in ("c4", "c8", "profile"):
            (wave / f"{phase}.go").unlink(missing_ok=True)
        wait_reclaimed(wave)
        children, logs = [], []
        try:
            for pair, arm in entries:
                visible = pairs[pair]
                out = wave / arm
                out.mkdir()
                env = os.environ.copy()
                env.update(
                    ASCEND_RT_VISIBLE_DEVICES=visible,
                    COMPARE_NO_MTP=arm,
                    ELASTIC_CANDIDATE="1",
                    TASK_QUEUE_ENABLE="0" if arm == "candidate" else "1",
                    VLLM_CACHE_ROOT=str(root / f"cache-{arm}"),
                    SERVING_PROFILE=str(out / "profiles"),
                    MASTER_PORT=str(32282 + pair * 10),
                    HCCL_NPU_SOCKET_PORT_RANGE=("29800-29863", "29864-29927")[pair],
                )
                command = [
                    sys.executable,
                    str(root / "source/swe_service.py"),
                    str(out),
                    "--arm",
                    arm,
                    "--port",
                    str(32281 + pair * 10),
                ]
                if repeat == 1 and profile_enabled:
                    command.append("--profile")
                log = (out / "run.log").open("w")
                logs.append(log)
                children.append(
                    subprocess.Popen(
                        command, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                )
            for phase in (
                ["c4", "c8", "profile"]
                if repeat == 1 and profile_enabled
                else ["c4", "c8"]
            ):
                deadline = time.monotonic() + 1800
                while not all(
                    (wave / arm / f"{phase}.ready").exists() for _, arm in entries
                ):
                    if any(c.poll() is not None for c in children):
                        raise RuntimeError(f"paired service exited before {phase}")
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"paired readiness {phase}")
                    time.sleep(1)
                (wave / f"{phase}.go").touch()
            for child in children:
                if child.wait(timeout=1800):
                    raise RuntimeError("paired service failed")
            for _, arm in entries:
                result = json.loads((wave / arm / "receipt.json").read_text())
                assert result["status"] == "PASS"
                receipt["rounds"].append(
                    dict(
                        repeat=repeat, arm=arm, receipt=str(wave / arm / "receipt.json")
                    )
                )
            (root / "comparison.json").write_text(json.dumps(receipt, indent=2))
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for log in logs:
                log.close()
    receipt["status"] = "PASS"
except BaseException as exc:
    receipt.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    raise
finally:
    (root / "comparison.json").write_text(json.dumps(receipt, indent=2))
