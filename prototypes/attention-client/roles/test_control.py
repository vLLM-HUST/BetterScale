"""CPU-only framing/permission gates for the startup-only channel."""

import os
from pathlib import Path
import socket
import struct
import tempfile
import unittest
from control import Channel, connect, listen


class ControlTest(unittest.TestCase):
    def pair(self):
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        return Channel(a), Channel(b)

    def test_roundtrip(self):
        a, b = self.pair()
        a.send(dict(op="ready", number=3))
        self.assertEqual(b.expect("ready")["number"], 3)

    def test_wrong_transition(self):
        a, b = self.pair()
        a.send(dict(op="ready"))
        with self.assertRaises(ValueError):
            b.expect("drained")

    def test_disconnect(self):
        a, b = self.pair()
        a.close()
        with self.assertRaises(EOFError):
            b.read()

    def test_outbound_limit(self):
        a, b = self.pair()
        with self.assertRaises(ValueError):
            a.send(dict(blob="x" * 16385))

    def test_inbound_limit(self):
        a, b = self.pair()
        a.sock.sendall(struct.pack("!I", 16385))
        with self.assertRaises(ValueError):
            b.read()

    def test_private_endpoint_no_clobber(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "expert.sock"
            listener = listen(path)
            self.addCleanup(listener.close)
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            with self.assertRaises(OSError):
                listen(path)
            client = connect(path)
            self.addCleanup(client.close)
            accepted = Channel(listener.accept()[0])
            self.addCleanup(accepted.close)
            client.send(dict(op="hello"))
            accepted.expect("hello")


if __name__ == "__main__":
    unittest.main()
