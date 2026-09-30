#!/usr/bin/env python3
"""Bounded fixed-token diagnostic load with dynamic msprof; not a benchmark score."""
import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import time
import msprof_helpers as h

import aiohttp

sys.path.insert(0, "/root/swe-prefix-reuse/src")
from swe_prefix_reuse.client import request as swe_request  # noqa: E402


ROLES = {"attention0", "attention1", "expert0", "expert1"}


async def batch(endpoint, model, prompts, phase, max_tokens):
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        outputs = await asyncio.gather(*(
            swe_request(
                session, endpoint, model, prompt, max_tokens,
                f"stage10-{phase}-lane{lane}", lane,
            )
            for lane, prompt in enumerate(prompts)
        ))
    records = []
    for lane, (prompt, output) in enumerate(zip(prompts, outputs)):
        row = output.record(prompt)
        row.update({"lane": lane, "phase": phase})
        records.append(row)
    failed = [row for row in records if not row["success"]]
    if failed:
        raise RuntimeError(f"{phase} protocol failures: {[x['error'] for x in failed]}")
    if any(len(row["token_ids"]) != max_tokens for row in records):
        raise RuntimeError(f"{phase} did not return the exact token-ID budget")
    return records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pids", required=True, help="JSON object role to PID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hccl", action="store_true")
    args = parser.parse_args()
    pids = json.loads(args.pids)
    if set(pids) != ROLES:
        parser.error(f"--pids must contain exactly {sorted(ROLES)}")
    if any(type(pid) is not int or pid <= 1 for pid in pids.values()):
        parser.error("all profile PIDs must be positive integers greater than 1")
    if len(set(pids.values())) != len(pids):
        parser.error("profile PIDs must be unique")
    args.pids = pids
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    workload = json.load(open(
        "/root/my-ascend-workspace/runs/qwen35-state-lanes/20260928-width-matched-cache/workload.json"
    ))
    base = [session["turns"][0]["input_ids"] for session in workload["sessions"]]
    prompts = base + base
    (args.output / "shape.json").write_text(json.dumps({
        "scope": "fixed diagnostic replay; not official SWE throughput",
        "model": args.model,
        "lanes": 16,
        "prompt_lengths": [len(x) for x in prompts],
        "warm_output_tokens": 8,
        "profile_output_tokens": 128,
        "profile_control": "start acknowledgement -> fixed requests -> stop acknowledgement",
    }, indent=2) + "\n")
    warm = asyncio.run(batch(args.endpoint, args.model, prompts, "warm", 8))
    (args.output / "warm.json").write_text(json.dumps(warm, indent=2) + "\n")
    # Resident expert kernels predate attachment. Capture attention-side cost only;
    # missing expert task rows cannot establish compute idle time.
    profilers = []
    receipt = {"status": "STARTED", "scope": "two attention ranks; no expert-math attribution",
               "owners": args.pids, "events": []}
    def mark(event):
        receipt["events"].append(dict(event=event, wall_ns=time.time_ns(), monotonic_ns=time.monotonic_ns()))
    try:
        for rank, role in enumerate(("attention0", "attention1")):
            out = args.output / "raw" / role
            out.mkdir(parents=True)
            cmd = h.msprof_command(Path("/usr/local/Ascend/cann-9.0.1/bin/msprof"),
                                   target_pid=args.pids[role], output=out)
            if not args.hccl: cmd.remove("--hccl=on")
            stdout = args.output / f"msprof-{role}.stdout.log"
            stderr = args.output / f"msprof-{role}.stderr.log"
            with stdout.open("w") as a, stderr.open("w") as b:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=a, stderr=b)
            profilers.append(dict(rank=rank,role=role,output=str(out),stdout=str(stdout),
                                  stderr=str(stderr),command=cmd,process=proc))
        mark("start_commands")
        for child in profilers: h._send_command(child,"start")
        h._wait_for_ack(profilers,command="start",timeout=60)
        mark("all_started")
        profile = asyncio.run(batch(args.endpoint, args.model, prompts, "profile", 128))
        mark("requests_finished")
        (args.output / "profile-requests.json").write_text(json.dumps(profile, indent=2)+"\n")
        for child in profilers: h._send_command(child,"stop")
        h._wait_for_ack(profilers,command="stop",timeout=120)
        mark("all_stopped")
        for child in profilers:
            h._send_command(child,"quit"); child["process"].stdin.close()
        for child in profilers:
            if child["process"].wait(timeout=120) != 0:
                raise RuntimeError(f"profiler {child['role']} failed")
        receipt["status"]="CAPTURED"
    except BaseException as error:
        receipt.update(status="FAIL",error=repr(error))
        raise
    finally:
        h._terminate(profilers)
        receipt["records"]=[{k:v for k,v in child.items() if k!="process"} for child in profilers]
        (args.output / "receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__ == "__main__":
    main()
