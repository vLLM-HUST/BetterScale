"""Two simultaneous TP2 services, swapped physical pairs in the second round."""

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
assert len(pairs) == 2 and len(devices) == 4 and devices <= set(range(8))


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
    scope="Simultaneous same-host pairs; swap arms across pairs; closed-loop cohorts synchronized only at start. Shared-host CPU effects are not isolated.",
)
try:
    for repeat in range(2):
        wave = root / f"round{repeat}"
        wave.mkdir()
        wait_reclaimed(wave)
        children, logs = [], []
        try:
            for pair, visible in enumerate(pairs):
                arm = ("baseline", "candidate")[(pair + repeat) % 2]
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
                if repeat == 1:
                    command.append("--profile")
                log = (out / "run.log").open("w")
                logs.append(log)
                children.append(
                    subprocess.Popen(
                        command, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                )
            for phase in (["c4", "c8", "profile"] if repeat == 1 else ["c4", "c8"]):
                deadline = time.monotonic() + 1800
                while not all(
                    (wave / arm / f"{phase}.ready").exists()
                    for arm in ("baseline", "candidate")
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
            for arm in ("baseline", "candidate"):
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
