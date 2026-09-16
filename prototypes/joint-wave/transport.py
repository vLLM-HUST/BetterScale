"""Bounded two-bank transport for a single ordered device wave program."""
from dataclasses import dataclass

import torch


@dataclass
class Ticket:
    sequence: int
    bank: int
    done: object
    host: object
    copied: object = None


class TwoBankExecutor:
    def __init__(self, graph, egress, *, authorization=None, backend=None):
        self.backend = backend or torch.npu
        self.graph = graph
        self.egress = egress
        self.compute = self.backend.Stream()
        self.copy = self.backend.Stream()
        self.ingress = self.backend.Stream()
        self.authorization = authorization
        self.retired = []
        self.banks = [None, None]
        self.tickets = []
        self.closed = False
        self.failed = False
        ready = self.backend.Event()
        ready.record()
        self.compute.wait_event(ready)

    def submit(self, grant=None):
        if self.closed or self.failed:
            raise RuntimeError('executor is closed or failed')
        sequence = len(self.tickets)
        if sequence - len(self.retired) >= 2:
            raise RuntimeError('two-wave submission window is full')
        bank = sequence % 2
        previous = self.banks[bank]
        if previous is not None and previous.copied is None:
            raise RuntimeError('output bank still has an unsubmitted reader')
        host = torch.empty_like(self.egress[bank], device='cpu', pin_memory=True)
        try:
            if self.authorization is not None:
                if grant is None or grant.sequence != sequence:
                    raise ValueError('missing or out-of-order wave grant')
                command = torch.tensor([grant.sequence, grant.generation, grant.end],
                                       dtype=torch.int64, device='cpu', pin_memory=True)
                with self.backend.stream(self.ingress):
                    if previous is not None:
                        self.ingress.wait_event(previous.done)
                    self.authorization[bank].copy_(command, non_blocking=True)
                    ready = self.backend.Event()
                    ready.record(self.ingress)
                self.compute.wait_event(ready)
            with self.backend.stream(self.compute):
                if previous is not None:
                    self.compute.wait_event(previous.copied)
                self.graph.replay()
                done = self.backend.Event()
                done.record(self.compute)
        except BaseException:
            self.failed = True
            raise
        ticket = Ticket(sequence, bank, done, host)
        # Keep the pinned grant alive through H2D and the invocation.
        ticket.command = command if self.authorization is not None else None
        self.banks[bank] = ticket
        self.tickets.append(ticket)
        return ticket

    def copy_out(self, ticket):
        if self.closed or self.failed or self.banks[ticket.bank] is not ticket:
            raise RuntimeError('foreign or overwritten output ticket')
        if ticket.copied is not None:
            raise RuntimeError('output already submitted')
        try:
            with self.backend.stream(self.copy):
                self.copy.wait_event(ticket.done)
                ticket.host.copy_(self.egress[ticket.bank], non_blocking=True)
                ticket.copied = self.backend.Event()
                ticket.copied.record(self.copy)
        except BaseException:
            self.failed = True
            raise

    def finish(self):
        if self.closed or self.failed or any(t.copied is None for t in self.tickets):
            raise RuntimeError('unsubmitted egress or already closed')
        while len(self.retired) < len(self.tickets):
            self.receive_oldest()
        self.closed = True
        return self.retired

    def receive_oldest(self):
        if self.closed or self.failed or len(self.retired) == len(self.tickets):
            raise RuntimeError('no live outstanding receipt')
        ticket = self.tickets[len(self.retired)]
        if ticket.copied is None:
            raise RuntimeError('egress has not been submitted')
        try:
            ticket.copied.synchronize()
            row = ticket.host.tolist()
            if row[0] != ticket.sequence:
                raise RuntimeError('stale or torn wave receipt')
            self.retired.append(row)
            return row
        except BaseException:
            self.failed = True
            raise
