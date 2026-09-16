"""Local startup/drain protocol. No pickled objects, no forward-path RPC."""

import json
import os
import socket
import struct
import time

CONTRACT = dict(
    version=1,
    layers=2,
    hidden=2048,
    intermediate=768,
    experts=128,
    topk=8,
    dtype="bf16",
    rows=32,
)


class Channel:
    def __init__(self, sock):
        self.sock = sock
        sock.settimeout(180)

    def send(self, value):
        data = json.dumps(value).encode()
        if len(data) > 16384:
            raise ValueError("control message too large")
        self.sock.sendall(struct.pack("!I", len(data)) + data)

    def read(self):
        def exact(n):
            out = b""
            while len(out) < n:
                part = self.sock.recv(n - len(out))
                if not part:
                    raise EOFError("peer disconnected; session must fail-stop")
                out += part
            return out

        size = struct.unpack("!I", exact(4))[0]
        if size > 16384:
            raise ValueError("control message too large")
        return json.loads(exact(size))

    def expect(self, op):
        value = self.read()
        if value.get("op") != op:
            raise ValueError(f"expected {op}, got {value.get('op')}")
        return value

    def close(self):
        self.sock.close()


def connect(path):
    deadline = time.monotonic() + 180
    while True:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(str(path))
            return Channel(sock)
        except (FileNotFoundError, ConnectionRefusedError):
            sock.close()
            if time.monotonic() > deadline:
                raise TimeoutError("expert startup rendezvous")
            time.sleep(0.1)


def listen(path):
    # Never remove somebody else's endpoint. The launcher supplies a private dir.
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(path))
    except Exception:
        sock.close()
        raise
    os.chmod(path, 0o600)
    sock.listen(2)
    sock.settimeout(180)
    return sock
