"""Real Mooncake CPU/TCP smoke; separate clients, no accelerator access."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from mooncake.store import MooncakeDistributedStore, ReplicateConfig


def check_ports(ports):
    reservations = []
    try:
        for port in ports:
            s = socket.socket()
            s.bind(("127.0.0.1", port))
            reservations.append(s)
    finally:
        for s in reservations:
            s.close()


def session_roundtrip(path, p, d):
    from session import Directory, Objects, Turn, Conflict, CacheMiss, restore
    directory = Directory(path)
    sid = f"session-{os.getpid()}"
    identity = "target-only/synthetic-state/salt-A"
    streams = ("layer0/k/head0", "layer0/v/head0", "layer0/k/head1", "layer0/v/head1")
    directory.create(sid, identity, "P0")
    checks = []
    def payload(start, stop):
        return {name: bytes((i + j) % 251 for j in range(start * 512, stop * 512))
                for i, name in enumerate(streams)}
    latest = None
    tokens = []
    for owner, next_owner, stop, client in [
        ("P0", "D0", 129, p), ("D0", "P0", 145, d), ("P0", "D0", 257, p)]:
        lease = directory.claim(sid, owner, identity)
        objects = Objects(client)
        worker = Turn(directory, objects, lease, streams, 512)
        try:
            begin = worker.cursor
            middle = begin + (stop - begin) // 2
            worker.append(middle, payload(begin, middle))
            worker.append(stop, payload(middle, stop))
            assert directory.current(sid) == latest
            tokens = list(range(stop))
            state = {"gdn": f"opaque-gdn-at-{stop}".encode(),
                     "conv": f"opaque-conv-at-{stop}".encode()}
            latest = worker.finish(tokens, state, next_owner, writer_retired=True,
                                   pending_token=stop)
            manifest, dense, checkpoint = restore(Objects(d if owner == "P0" else p),
                                                 latest, identity, streams, 512)
            assert dense == payload(0, stop) and checkpoint == state
            assert manifest["pending_token"] == stop
            checks.append(f"{owner}->{next_owner}: exact {stop}-token snapshot")
        finally:
            worker.close()
    # A missing dense object must not yield a usable partial checkpoint.
    victim = manifest["chunks"][0]["keys"][streams[0]]
    assert p.remove(victim, force=True) == 0  # deliberate test-owned fault injection
    try:
        restore(Objects(d), latest, identity, streams, 512)
    except CacheMiss:
        checks.append("evicted dependency fails closed")
    else:
        raise AssertionError("missing dense chunk accepted")
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-port", type=int, default=55181)
    parser.add_argument("--ssd", action="store_true")
    parser.add_argument("--npu-staging", action="store_true")
    parser.add_argument("--gdn-resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    port = args.base_port
    check_ports(range(port, port + 6))
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env.pop("LD_LIBRARY_PATH", None)
    disk_path = args.output_dir / "disk-owner"
    if args.ssd:
        disk_path.mkdir(exist_ok=False)
        disk_env = {
            "MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR": "file_per_key_storage_backend",
            "MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES": str(32 * 1024**2),
            "MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES": str(64 * 1024**2),
            "MOONCAKE_OFFLOAD_TOTAL_KEYS_LIMIT": "1024",
            "MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS": "1",
        }
        env.update(disk_env)
        os.environ.update(disk_env)
    master = Path(sys.executable).parent / "mooncake_master"
    clients, registered = [], []
    receipt = {"scope": "real CPU Store TCP; same-host clients; no NPU",
               "disk_offload": args.ssd,
               "checks": [], "master_exit": None}
    with (args.output_dir / "master.log").open("w") as log:
        proc = subprocess.Popen(
            [str(master), "--rpc_address=127.0.0.1", f"--rpc_port={port}",
             "--enable_http_metadata_server=true",
             "--http_metadata_server_host=127.0.0.1",
             f"--http_metadata_server_port={port+1}",
             "--metrics_host=127.0.0.1", f"--metrics_port={port+2}",
             "--enable_metric_reporting=false", "--default_kv_lease_ttl=2000",
             f"--enable_offload={str(args.ssd).lower()}"], env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"master exited {proc.returncode}; see master.log")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("master startup")
                    time.sleep(0.1)
            for index in range(2):
                client = MooncakeDistributedStore()
                clients.append(client)
                rc = client.setup(
                    f"127.0.0.1:{port+3+index}",
                    f"http://127.0.0.1:{port+1}/metadata",
                    32 * 1024**2 if index == 0 else 0,
                    16 * 1024**2, "tcp", "", f"127.0.0.1:{port}",
                    enable_ssd_offload=args.ssd and index == 0,
                    ssd_offload_path=str(disk_path) if args.ssd and index == 0 else "")
                assert rc == 0, ("setup", index, rc)
            p, d = clients
            size = 128 * 256 * 2
            heads = [bytes((i * 7 + h * 31) % 251 for i in range(size)) for h in range(2)]
            sources = [ctypes.create_string_buffer(b, len(b)) for b in heads]
            destination = ctypes.create_string_buffer(2 * size + 64)
            for client, buf in [(p, b) for b in sources] + [(d, destination)]:
                rc = client.register_buffer(ctypes.addressof(buf), ctypes.sizeof(buf))
                assert rc == 0, ("register", rc)
                registered.append((client, buf))
            config = ReplicateConfig()
            config.replica_num = 1
            key = f"betterscale-pd-smoke-{os.getpid()}"
            rc = p.batch_put_from_multi_buffers(
                [key], [[ctypes.addressof(b) for b in sources]], [[size, size]], config)
            assert rc == [0], ("put", rc)
            receipt["checks"].append("two-head multi-buffer put")
            # Full bytes establish ordering before testing partial scatter.
            assert d.get(key) == heads[0] + heads[1]
            receipt["checks"].append("cross-client full-object exact bytes")
            with (args.output_dir / "reader.log").open("w") as reader_log:
                subprocess.run([sys.executable, str(Path(__file__).with_name("store_reader.py")),
                                "--key", key, "--base-port", str(port),
                                "--output", str(args.output_dir / "reader.json")],
                               env=env, stdout=reader_log, stderr=reader_log,
                               timeout=25, check=True)
            receipt["checks"].append("independent-process full/ranged reads and clean exit")
            ctypes.memset(ctypes.addressof(destination), 0xFE, ctypes.sizeof(destination))
            # Swap head destinations with interior guards; no tensor arithmetic.
            rc = d.get_into_ranges([ctypes.addressof(destination)], [[key]],
                                   [[[32 + size, 32]]], [[[0, size]]], [[[size, size]]])
            assert rc == [[[size, size]]], ("ranged get", rc)
            expected = b"\xfe" * 32 + heads[1] + heads[0] + b"\xfe" * 32
            assert destination.raw == expected
            receipt["checks"].append("two-range head scatter and guard bytes")
            # Partial token interval: token 17..33 of head zero.
            ctypes.memset(ctypes.addressof(destination), 0xFE, ctypes.sizeof(destination))
            offset, length = 17 * 512, 16 * 512
            rc = d.get_into_ranges([ctypes.addressof(destination)], [[key]],
                                   [[[32]]], [[[offset]]], [[[length]]])
            assert rc == [[[length]]]
            assert destination.raw == (b"\xfe" * 32 + heads[0][offset:offset+length]
                                       + b"\xfe" * (ctypes.sizeof(destination)-32-length))
            receipt["checks"].append("partial-token interval and untouched suffix")
            if args.ssd:
                time.sleep(3)  # Background eager offload; one-second heartbeat.
                replicas = p.get_replica_desc(key)
                assert any(r.is_local_disk_replica() for r in replicas), "no disk replica"
                time.sleep(2.2)  # Descriptor query renews the read lease.
                cleared = p.batch_replica_clear([key], segment_name=p.get_hostname())
                assert key in cleared, ("clear memory replica", cleared)
                replicas = p.get_replica_desc(key)
                assert replicas and not any(r.is_memory_replica() for r in replicas)
                assert d.get(key) == heads[0] + heads[1]
                receipt["checks"].append("disk-only full read exact after memory replica removal")
                ctypes.memset(ctypes.addressof(destination), 0xFE, ctypes.sizeof(destination))
                # Released wheel disk-native range read failed RPC decoding in
                # disk-store capsule. Safe first cut: whole bounded chunk -> slice.
                full_chunk = d.get(key)
                part = full_chunk[17*512:33*512]
                ctypes.memmove(ctypes.addressof(destination)+32, part, len(part))
                assert destination.raw == (b"\xfe"*32 + heads[0][17*512:33*512]
                                           + b"\xfe"*(ctypes.sizeof(destination)-32-len(part)))
                receipt["checks"].append("disk full-chunk fallback yields exact partial interval")
                receipt["native_disk_range"] = "known RPC decode failure; not retried"
                receipt["disk_offload_rpc_reads"] = d.get_offload_rpc_read_count()
            receipt["normal_remove_result"] = p.remove(key)
            assert receipt["normal_remove_result"] in (0, -706)
            if receipt["normal_remove_result"] == -706:
                time.sleep(2.2)  # Explicit two-second test lease; no refresh.
                assert p.remove(key) == 0
                receipt["checks"].append("lease expiry permits normal deletion")
            receipt["checks"].append("remove test-owned object")
            receipt["session_checks"] = session_roundtrip(args.output_dir / "directory.sqlite", p, d)
            if args.gdn_resume:
                from gdn_resume_probe import check as check_gdn
                import torch
                def checkpoint_transport(checkpoint):
                    # Quiescent, selected recurrent target State, not a random blob.
                    key = f"gdn-resume-{os.getpid()}"
                    payload = checkpoint.numpy().tobytes()
                    assert p.put(key, payload) == 0
                    returned = d.get(key)
                    assert returned == payload
                    return torch.frombuffer(bytearray(returned), dtype=torch.float32).reshape(checkpoint.shape).clone()
                receipt["gdn_resume"] = check_gdn(args.output_dir / "gdn-resume", checkpoint_transport)
                receipt["scope"] = "real DRAM Store TCP, owned GDN checkpoint/new-slot continuation; same-host"
            if args.npu_staging:
                from npu_store_staging import check
                receipt["npu_staging"] = check(p, d)
                receipt["scope"] = "real DRAM Store TCP plus NPU0 pinned staging; same-host"
        finally:
            for client, buf in reversed(registered):
                client.unregister_buffer(ctypes.addressof(buf))
            for client in reversed(clients):
                client.close()
            registered.clear()
            clients.clear()
            p = d = client = None
            import gc
            gc.collect()  # Retire embedded disk services before their master.
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            receipt["master_exit"] = proc.returncode
            (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
