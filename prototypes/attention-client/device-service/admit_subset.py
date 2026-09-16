"""Fletcher-authorized subset admission: per-device locks, never a global lease.

Foreign occupancy still rejects admission and aborts only our owned process group.
"""

import argparse, fcntl, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

sys.path.insert(
    0,
    os.environ.get(
        "PROBE_HELPERS",
        "/workspace/my-ascend-workspace/runs/query-gang/20260911-lhtb-long-real-gang-v1/harness",
    ),
)
from probe_host_npus import parse_devices
from supervise import descendants, stop_group, group_members

p = argparse.ArgumentParser()
p.add_argument("--devices", default="0")
p.add_argument("--output", type=Path, required=True)
p.add_argument("--wait-seconds", type=float, default=1800)
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
devices = {int(x) for x in a.devices.split(",")}
assert devices and devices <= set(range(8))
a.output.mkdir(parents=True, exist_ok=False)
# Explicit user override on 2026-09-16: idle subsets need not wait for tp8.lock.
# Order per-device locks so two cooperating subset jobs cannot overlap/deadlock.
locks = []
lease_deadline = time.monotonic() + a.wait_seconds
for device in sorted(devices):
    lock = open(Path.home() / f"npu-device-{device}.lock", "a")
    locks.append(lock)
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() > lease_deadline:
                raise TimeoutError("subset lease wait expired")
            time.sleep(2)
known_host_owners = {}


def process_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def previously_owned(host_pid, local_pid, known, current_start, now):
    previous = known.get(host_pid)
    if previous is None:
        return False
    old_pid, old_start, last_seen = previous
    if local_pid not in (0, old_pid):
        return False
    # A driver row may outlive /proc, or its process may be reparented during
    # teardown. An actually reused visible PID must have a different start time.
    return (
        (now - last_seen <= 15) if current_start is None else current_start == old_start
    )


def inspect(owned=None):
    text = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
    readings = parse_devices(text)
    owners = set()
    for line in text.split("Process id", 1)[1].splitlines():
        cells = [x.strip() for x in line.split("|")[1:-1]]
        if (
            cells
            and re.fullmatch(r"\d+\s+\d+", cells[0])
            and int(cells[0].split()[0]) in devices
        ):
            assert len(cells) >= 5 and cells[4].isdigit()
            host_pid, local_pid = int(cells[1]), int(cells[4])
            start = process_start(
                local_pid or known_host_owners.get(host_pid, (0, None, 0))[0]
            )
            if owned is not None and local_pid in owned and start is not None:
                known_host_owners[host_pid] = (local_pid, start, time.monotonic())
            if owned is not None and previously_owned(
                host_pid, local_pid, known_host_owners, start, time.monotonic()
            ):
                continue
            owners.add(local_pid or host_pid)
    return text, readings, owners


# The queued job, not the model, polls its selected subset. A rejected sample
# never reaches Popen; the selected-device leases remain held through admission and cleanup.
deadline = time.monotonic() + a.wait_seconds
while True:
    text, readings, owners = inspect()
    (a.output / "admission.txt").write_text(text)
    ready = not owners and all(
        d in readings
        and readings[d].hbm_used_mb <= 4096
        and readings[d].aicore_percent == 0
        and re.search(rf"\|\s*{d}\s+910B2\s*\|\s*OK", text)
        for d in devices
    )
    if ready:
        break
    if time.monotonic() > deadline:
        raise TimeoutError("Selected-card admission wait expired")
    time.sleep(2)
command = a.command[1:] if a.command[:1] == ["--"] else a.command
source = a.output / "source"
source.mkdir()
for f in Path(__file__).parent.glob("*.py"):
    shutil.copyfile(f, source / f.name)
command = [
    str(source / "probe.py") if x.endswith("/full-mixed/probe.py") else x
    for x in command
]
os.environ["PYTHONPATH"] = str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")
log = open(a.output / "run.log", "w")
child = subprocess.Popen(
    command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=a.output
)
try:
    deadline = time.monotonic() + 1500
    while child.poll() is None:
        if time.monotonic() > deadline:
            raise TimeoutError("bounded probe exceeded 1500s")
        text, _, owners = inspect(descendants(child.pid) | group_members(child.pid))
        (a.output / "latest.txt").write_text(text)
        foreign = owners - (descendants(child.pid) | group_members(child.pid))
        if foreign:
            (a.output / "foreign.txt").write_text(text)
            raise RuntimeError(f"foreign owner on selected cards: {foreign}")
        time.sleep(3)
    rc = child.returncode
finally:
    stop_group(child)
    log.close()
    (a.output / "release.txt").write_text(
        subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
    )
(a.output / "exit.txt").write_text(str(rc) + "\n")
print("probe exit", rc, flush=True)
sys.exit(rc)
