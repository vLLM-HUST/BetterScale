"""One admitted TP2 window, separate native/candidate servers in the recorded order."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path(os.environ["CAPSULE"])
if os.environ.get("SERVING_QUALIFICATION_RECEIPT"):
    qualification = json.loads(
        Path(os.environ["SERVING_QUALIFICATION_RECEIPT"]).read_text()
    )
    if qualification["status"] != "PASS":
        raise RuntimeError(
            "Service correctness qualification failed; refusing timing run"
        )
receipt = dict(
    status="RUNNING",
    order=(
        os.environ.get(
            "SERVING_COMPARE_ORDER", "baseline,candidate,candidate,baseline"
        ).split(",")
    ),
    rounds=[],
    scope="No MTP in either arm. Async, TP2, APC off, 6GiB KV, 8 seats, 2048 batch budget, 64 output tokens. Separate-process order as recorded after warmup; timings exclude startup/profile.",
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
        if os.environ.get("ELASTIC_CANDIDATE") == "1":
            env["TASK_QUEUE_ENABLE"] = "0" if arm == "candidate" else "1"
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
