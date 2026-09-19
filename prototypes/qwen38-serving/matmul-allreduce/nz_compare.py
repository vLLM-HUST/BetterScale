"""Prototype gate/up-only NZ vs current ND candidate; same-pair ABBA."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from probe_host_npus import parse_devices


def reclaimed(root):
    deadline = time.monotonic() + 600
    while True:
        sample = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
        (root / "reclaimed.txt").write_text(sample)
        readings = parse_devices(sample)
        if all(d in readings and readings[d].hbm_used_mb <= 4096
               and readings[d].aicore_percent == 0
               and re.search(rf"\|\s*{d}\s+910B2\s*\|\s*OK", sample)
               and f"No running processes found in NPU {d}" in sample for d in (6, 7)):
            return
        if time.monotonic() > deadline:
            raise TimeoutError("selected pair not reclaimed")
        time.sleep(2)


def main():
    root = Path(os.environ["CAPSULE"])
    receipt = dict(status="RUNNING", rounds=[], source_commit=os.environ["SWE_SOURCE_COMMIT"])
    try:
        for index, arm in enumerate(("nd", "nz", "nz", "nd")):
            out = root / f"{index}-{arm}"
            out.mkdir()
            reclaimed(root)
            env = os.environ.copy()
            env.update(ASCEND_RT_VISIBLE_DEVICES="6,7", COMPARE_NO_MTP="candidate",
                       ELASTIC_CANDIDATE="1", TASK_QUEUE_ENABLE="0", STEP_PREFILL_DELTA="1",
                       PROBE_NZ_GATE_UP=str(int(arm == "nz")),
                       VLLM_CACHE_ROOT=str(root / f"cache-{arm}"), SERVING_PROFILE=str(out / "profiles"),
                       MASTER_PORT="32382", HCCL_NPU_SOCKET_PORT_RANGE="30000-30063",
                       VLLM_SERVER_DEV_MODE="1", STEP_REVERSE=str(int(index >= 2)))
            env.pop("LD_PRELOAD", None)
            env["LD_PRELOAD"] = env["BETTERSCALE_FIA_LIBRARY"]
            with (out / "run.log").open("w") as log:
                subprocess.run([sys.executable, str(root / "source/step_probe.py"), str(out)],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
            result = json.loads((out / "receipt.json").read_text())
            assert result["status"] == "PASS", result.get("error")
            receipt["rounds"].append(dict(index=index, arm=arm, path=str(out)))
            (root / "comparison.json").write_text(json.dumps(receipt, indent=2))
        reclaimed(root)
        receipt["status"] = "PASS"
    except BaseException as exc:
        receipt.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (root / "comparison.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
