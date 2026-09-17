"""One admitted TP2 window, separate native/candidate servers in ABBA order."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path(os.environ["CAPSULE"])
receipt = dict(
    status="RUNNING",
    order=["baseline", "candidate", "candidate", "baseline"],
    rounds=[],
    scope="No MTP in either arm. Async, TP2, APC off, 6GiB KV, 8 seats, 2048 batch budget, 64 output tokens. Separate-process ABBA after warmup; timings exclude startup/profile.",
)
path = root / "comparison.json"
try:
    for index, arm in enumerate(receipt["order"]):
        out = root / f"{index}-{arm}"
        out.mkdir()
        shutil.copyfile(root / "prompt.json", out / "prompt.json")
        env = os.environ.copy()
        for key in [
            "FULL_MTP",
            "PACKAGED_QWEN",
            "SERVING_PACK_CONV",
            "PROFILE_ONLY",
            "PADDED_PREFILL",
        ]:
            env.pop(key, None)
        env["COMPARE_NO_MTP"] = arm
        env["VLLM_CACHE_ROOT"] = str(root / f"cache-{arm}")
        path.write_text(json.dumps(receipt, indent=2))
        print(f"round {index}: {arm} starting", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(root / "source/service_probe.py"),
                "--capsule",
                str(out),
                "--arm",
                "async",
            ],
            env=env,
            check=True,
            timeout=650,
        )
        result = json.loads((out / "receipt.json").read_text())
        assert (
            result["status"] == "PASS"
            and "--speculative-config" not in result["command"]
        )
        receipt["rounds"].append(
            dict(index=index, arm=arm, path=str(out / "receipt.json"))
        )
        path.write_text(json.dumps(receipt, indent=2))
        print(f"round {index}: {arm} PASS", flush=True)
        time.sleep(3)
    receipt["status"] = "PASS"
except BaseException as e:
    receipt.update(status="FAIL", error=f"{type(e).__name__}: {e}")
    raise
finally:
    path.write_text(json.dumps(receipt, indent=2))
