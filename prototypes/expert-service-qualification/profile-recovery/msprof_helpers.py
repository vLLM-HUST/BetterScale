#!/usr/bin/env python3
"""Attach bounded dynamic msprof windows to exact rank-bound workers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def wait_for_file(
    path: Path,
    *,
    timeout: float,
    watched_pid: int | None = None,
    poll_interval: float = 0.2,
) -> None:
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if watched_pid is not None:
            try:
                os.kill(watched_pid, 0)
            except ProcessLookupError as error:
                raise RuntimeError(
                    f"watched process {watched_pid} exited before {path}"
                ) from error
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(poll_interval)


def load_rank_owners(path: Path, expected_ranks: int) -> list[dict[str, int]]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "pass":
        raise ValueError(f"owner receipt is not passing: {path}")
    owners = sorted(payload.get("owners", []), key=lambda row: row.get("rank", -1))
    expected = list(range(expected_ranks))
    for field in ("rank", "local_rank", "current_device"):
        values = [int(row[field]) for row in owners]
        if values != expected:
            raise ValueError(f"owner {field} values {values} != {expected}")
    pids = [int(row["pid"]) for row in owners]
    if len(set(pids)) != expected_ranks or any(pid <= 1 for pid in pids):
        raise ValueError(f"owner PIDs are not unique positive workers: {pids}")
    return [
        {
            "rank": int(row["rank"]),
            "local_rank": int(row["local_rank"]),
            "current_device": int(row["current_device"]),
            "pid": int(row["pid"]),
        }
        for row in owners
    ]


def msprof_command(
    executable: Path,
    *,
    target_pid: int,
    output: Path,
) -> list[str]:
    return [
        str(executable),
        "--dynamic=on",
        f"--pid={target_pid}",
        f"--output={output}",
        "--type=db",
        "--task-time=l1",
        "--ai-core=on",
        "--aic-metrics=PipeUtilization",
        "--hccl=on",
    ]


def _send_command(child: dict[str, Any], command: str) -> None:
    process: subprocess.Popen[bytes] = child["process"]
    if process.stdin is None:
        raise RuntimeError(f"rank {child['rank']} msprof stdin is unavailable")
    process.stdin.write(f"{command}\n".encode())
    process.stdin.flush()


def _wait_for_ack(
    children: list[dict[str, Any]], *, command: str, timeout: float
) -> None:
    pending = {int(child["rank"]) for child in children}
    deadline = time.monotonic() + timeout
    while pending:
        for child in children:
            rank = int(child["rank"])
            if rank not in pending:
                continue
            returncode = child["process"].poll()
            if returncode is not None:
                raise RuntimeError(
                    f"rank {rank} msprof exited {returncode} before {command} ack"
                )
            stdout = Path(child["stdout"]).read_text(errors="replace")
            if f" {command} success" in stdout:
                pending.remove(rank)
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for msprof {command} ack from ranks "
                f"{sorted(pending)}"
            )
        time.sleep(0.1)


def _terminate(children: list[dict[str, Any]]) -> None:
    for child in children:
        process: subprocess.Popen[bytes] = child["process"]
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10
    for child in children:
        process = child["process"]
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _export_databases(
    children: list[dict[str, Any]],
    *,
    executable: Path,
    timeout: float,
) -> None:
    """Export dynamic-profile raw roots after their controllers have quit.

    CANN dynamic attachment seals a ``PROF_*`` tree on ``quit`` but does not
    materialize its SQLite database.  The database is an explicit offline
    export of that raw root, not an output of the interactive controller.
    """

    deadline = time.monotonic() + timeout
    for child in children:
        rank = int(child["rank"])
        rank_root = Path(child["output"])
        raw_roots = sorted(
            path
            for path in rank_root.glob("PROF_*")
            if path.is_dir()
        )
        if len(raw_roots) != 1:
            raise RuntimeError(
                f"rank {rank} produced {len(raw_roots)} raw PROF roots; "
                "expected exactly one"
            )
        raw_root = raw_roots[0]
        command = [
            str(executable),
            "--export=on",
            f"--output={raw_root}",
            "--type=db",
        ]
        stdout_path = rank_root.parent / f"rank-{rank}.export.stdout.log"
        stderr_path = rank_root.parent / f"rank-{rank}.export.stderr.log"
        remaining = max(0.0, deadline - time.monotonic())
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            result = subprocess.run(
                command,
                stdout=stdout,
                stderr=stderr,
                timeout=remaining,
                check=False,
            )
        child.update(
            {
                "raw_profile_root": str(raw_root),
                "export_command": command,
                "export_stdout": str(stdout_path),
                "export_stderr": str(stderr_path),
                "export_returncode": result.returncode,
            }
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"rank {rank} offline msprof export failed: {result.returncode}"
            )


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if args.profile_root.exists():
        raise ValueError(f"profile root already exists: {args.profile_root}")
    args.profile_root.mkdir(parents=True)
    owners = load_rank_owners(args.owner_receipt, args.expected_ranks)
    wait_for_file(
        args.ready,
        timeout=args.ready_timeout,
        watched_pid=args.driver_pid,
    )

    children: list[dict[str, Any]] = []
    launched_utc = utc_now()
    started_monotonic = time.monotonic()
    try:
        for owner in owners:
            rank = owner["rank"]
            rank_root = args.profile_root / f"rank-{rank}"
            rank_root.mkdir()
            stdout_path = args.profile_root / f"rank-{rank}.stdout.log"
            stderr_path = args.profile_root / f"rank-{rank}.stderr.log"
            command = msprof_command(
                args.msprof,
                target_pid=owner["pid"],
                output=rank_root,
            )
            stdout = stdout_path.open("xb")
            stderr = stderr_path.open("xb")
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=stdout,
                    stderr=stderr,
                )
            finally:
                stdout.close()
                stderr.close()
            children.append(
                {
                    "rank": rank,
                    "target_pid": owner["pid"],
                    "profile_process_pid": process.pid,
                    "output": str(rank_root),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "command": command,
                    "process": process,
                }
            )

        for child in children:
            _send_command(child, "start")
        _wait_for_ack(children, command="start", timeout=args.activation_grace)

        start_payload = {
            "schema": "stateharbor-external-msprof-window-start/v1",
            "status": "pass",
            "created_utc": utc_now(),
            "controller_pid": os.getpid(),
            "duration_seconds": args.duration,
            "owners": [
                {
                    key: child[key]
                    for key in (
                        "rank",
                        "target_pid",
                        "profile_process_pid",
                        "output",
                    )
                }
                for child in children
            ],
        }
        write_json_once(args.start, start_payload)

        completion_deadline = time.monotonic() + args.workload_timeout
        while not args.complete.is_file():
            exited = [
                (child["rank"], child["process"].poll())
                for child in children
                if child["process"].poll() is not None
            ]
            if exited:
                raise RuntimeError(
                    f"msprof exited before measured workload completed: {exited}"
                )
            try:
                os.kill(args.driver_pid, 0)
            except ProcessLookupError as error:
                raise RuntimeError(
                    "engine driver exited before measured workload completion"
                ) from error
            if time.monotonic() >= completion_deadline:
                raise TimeoutError(
                    f"timed out waiting for measured completion: {args.complete}"
                )
            time.sleep(0.2)

        for child in children:
            _send_command(child, "stop")
        _wait_for_ack(children, command="stop", timeout=args.export_timeout)
        for child in children:
            _send_command(child, "quit")
            assert child["process"].stdin is not None
            child["process"].stdin.close()

        export_deadline = time.monotonic() + args.export_timeout
        for child in children:
            remaining = max(0.0, export_deadline - time.monotonic())
            child["returncode"] = child["process"].wait(timeout=remaining)
        failed = [
            (child["rank"], child["returncode"])
            for child in children
            if child["returncode"] != 0
        ]
        if failed:
            raise RuntimeError(f"msprof dynamic capture failed: {failed}")

        _export_databases(
            children,
            executable=args.msprof,
            timeout=args.export_timeout,
        )

        records = []
        for child in children:
            databases = sorted(
                str(path)
                for path in Path(child["output"]).rglob("*.db")
                if path.is_file()
            )
            if not databases:
                raise RuntimeError(
                    f"rank {child['rank']} produced no SQLite database"
                )
            records.append(
                {
                    key: child[key]
                    for key in (
                        "rank",
                        "target_pid",
                        "profile_process_pid",
                        "output",
                        "stdout",
                        "stderr",
                        "command",
                        "returncode",
                        "raw_profile_root",
                        "export_command",
                        "export_stdout",
                        "export_stderr",
                        "export_returncode",
                    )
                }
                | {"databases": databases}
            )
        receipt = {
            "schema": "stateharbor-external-msprof-window/v1",
            "status": "pass",
            "launched_utc": launched_utc,
            "finished_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "ready": str(args.ready),
            "start": str(args.start),
            "complete": str(args.complete),
            "duration_seconds": args.duration,
            "records": records,
        }
        write_json_once(args.output, receipt)
        write_json_once(
            args.ack,
            {
                "schema": "stateharbor-external-msprof-window-ack/v1",
                "status": "pass",
                "created_utc": utc_now(),
                "receipt": str(args.output),
            },
        )
        return receipt
    except BaseException as error:
        _terminate(children)
        failure = {
            "schema": "stateharbor-external-msprof-window/v1",
            "status": "fail",
            "created_utc": utc_now(),
            "error": f"{type(error).__name__}: {error}",
            "records": [
                {
                    key: child[key]
                    for key in (
                        "rank",
                        "target_pid",
                        "profile_process_pid",
                        "output",
                        "stdout",
                        "stderr",
                        "command",
                    )
                }
                | {
                    key: child[key]
                    for key in (
                        "raw_profile_root",
                        "export_command",
                        "export_stdout",
                        "export_stderr",
                        "export_returncode",
                    )
                    if key in child
                }
                | {"returncode": child["process"].poll()}
                for child in children
            ],
        }
        write_json_once(args.output, failure)
        if args.start.is_file():
            write_json_once(
                args.ack,
                {
                    "schema": "stateharbor-external-msprof-window-ack/v1",
                    "status": "fail",
                    "created_utc": utc_now(),
                    "receipt": str(args.output),
                    "error": failure["error"],
                },
            )
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--owner-receipt", type=Path, required=True)
    result.add_argument("--ready", type=Path, required=True)
    result.add_argument("--start", type=Path, required=True)
    result.add_argument("--complete", type=Path, required=True)
    result.add_argument("--ack", type=Path, required=True)
    result.add_argument("--profile-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--msprof", type=Path, required=True)
    result.add_argument("--driver-pid", type=int, required=True)
    result.add_argument("--duration", type=int, default=30)
    result.add_argument("--expected-ranks", type=int, default=8)
    result.add_argument("--activation-grace", type=float, default=30.0)
    result.add_argument("--ready-timeout", type=float, default=1800.0)
    result.add_argument("--workload-timeout", type=float, default=300.0)
    result.add_argument("--export-timeout", type=float, default=180.0)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.duration < 1:
        raise ValueError("duration must be positive")
    payload = capture(args)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
