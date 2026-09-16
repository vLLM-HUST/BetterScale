"""Two outstanding LiveInvocations; distinct graph-reader and D2H reuse fences."""

from collections import deque
from dataclasses import dataclass
import torch


@dataclass
class Ticket:
    plan: object
    invocation: object
    pinned: tuple
    host: object
    metadata: object
    done: object
    copied: object


class Reactor:
    def __init__(self, root):
        self.root = root
        self.pending = deque()
        self.previous = [None, None]
        self.ingress = torch.npu.Stream()
        self.compute = root.execution_stream or torch.npu.Stream()
        self.copy = torch.npu.Stream()
        self.sequence = 0
        self.failed = False
        ready = torch.npu.Event()
        ready.record()
        for stream in (self.ingress, self.compute, self.copy):
            stream.wait_event(ready)

    def submit(self, plan):
        if self.failed or len(self.pending) >= 2 or plan["sequence"] != self.sequence:
            raise RuntimeError("reactor window/sequence/liveness violation")
        root = self.root
        bank = self.sequence % 2
        key = plan["key"]
        attention_metadata = None
        if hasattr(root.static_attention, "prepare"):
            key, attention_metadata = root.static_attention.prepare(
                key, plan["lengths"]
            )
        f = root.frames[key]
        if f["bank"] != bank:
            raise ValueError("wrong graph bank")
        old = self.previous[bank]
        pinned = []
        inv = root.invoke(key, key)
        try:
            with torch.npu.stream(self.ingress):
                if old is not None:
                    self.ingress.wait_event(old.done)
                if attention_metadata is not None:
                    f["tiling"].copy_(attention_metadata, non_blocking=True)
                    pinned.append(attention_metadata)
                for name, values in plan["inputs"].items():
                    dst = f[name]
                    host = torch.tensor(values, dtype=dst.dtype, pin_memory=True)
                    assert host.shape == dst.shape, (name, host.shape, dst.shape)
                    dst.copy_(host, non_blocking=True)
                    pinned.append(host)
                f["lengths"] = plan["lengths"]
                before = root.forward_calls
                inv.shadow_replay(stream=self.ingress)
                assert root.forward_calls == before
                ready = torch.npu.Event()
                ready.record(self.ingress)
            self.compute.wait_event(ready)
            if old is not None:
                self.compute.wait_event(old.copied)
            root.publish(key, self.compute)
            inv.replay(stream=self.compute)
            done = torch.npu.Event()
            done.record(self.compute)
            host = torch.empty_like(root.egress[bank], device="cpu", pin_memory=True)
            with torch.npu.stream(self.copy):
                self.copy.wait_event(done)
                host.copy_(root.egress[bank], non_blocking=True)
                copied = torch.npu.Event()
                copied.record(self.copy)
            ticket = Ticket(
                plan, inv, tuple(pinned), host, f["metadata"].seq_lens, done, copied
            )
            self.pending.append(ticket)
            self.previous[bank] = ticket
            self.sequence += 1
        except BaseException:
            self.failed = True
            raise

    def receive(self):
        if self.failed or not self.pending:
            raise RuntimeError("no live receipt")
        ticket = self.pending[0]
        try:
            ticket.copied.synchronize()
            rows = ticket.host.tolist()
            for row in rows:
                if row[1] >= 0 and (row[0] != ticket.plan["sequence"] or row[7]):
                    raise RuntimeError(f"failed/stale receipt {row}")
            ticket.invocation.retire()
            self.pending.popleft()
            return ticket.plan, rows
        except BaseException:
            self.failed = True
            raise
