"""Two simultaneous TP2 services, swapped physical pairs in the second round."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(os.environ["CAPSULE"])
receipt = dict(
    status="RUNNING",
    rounds=[],
    scope="Simultaneous same-host pairs; swap arms across pairs; closed-loop cohorts synchronized only at start. Shared-host CPU effects are not isolated.",
)
try:
    for repeat in range(2):
        wave = root / f"round{repeat}"
        wave.mkdir()
        children, logs = [], []
        try:
            for pair, devices in enumerate(("0,1", "6,7")):
                arm = ("baseline", "candidate")[(pair + repeat) % 2]
                out = wave / arm
                out.mkdir()
                env = os.environ.copy()
                env.update(
                    ASCEND_RT_VISIBLE_DEVICES=devices,
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
