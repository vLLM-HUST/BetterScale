"""Adapter for the *integer oracle* ABI in workspace commit3532418.

Not a BF16/GEMM transport. Each packet owns a copy of its rows; retiring a packet
cannot free the originating layer input, which Continuations retains separately.
"""

from dataclasses import dataclass

DEPTH, SLOT_BYTES, WIDTH, MAX_ROWS = 8, 16384, 64, 8
READY_BYTES, DESCRIPTOR_BYTES, PAYLOAD_BYTES = 0, 32, 256


@dataclass(frozen=True)
class Packet:
    task_id: int
    ticket: object
    expert: int
    phase: int
    routes: tuple
    payload: tuple

    @property
    def slot(self):
        return self.task_id % DEPTH

    @property
    def generation(self):
        return self.task_id // DEPTH + 1

    @property
    def descriptor(self):
        return (
            self.generation,
            self.task_id,
            self.ticket.layer,
            self.expert,
            self.phase,
            len(self.routes),
            0,
            0,
        )


class OraclePackets:
    def __init__(self, continuations):
        self.continuations = continuations
        self.next_task = 0
        self.active = {}

    def publish(self, ticket, expert, phase, routes, rows):
        # Strictly reject incompatible real-model payloads; no implicit conversion.
        if ticket.layer not in (0, 1) or expert not in range(3) or phase not in (0, 1):
            raise ValueError("outside server oracle descriptor envelope")
        routes, rows = tuple(routes), tuple(tuple(row) for row in rows)
        if not 1 <= len(routes) == len(rows) <= MAX_ROWS:
            raise ValueError("invalid packet row count")
        if any(
            len(row) != WIDTH
            or any(type(x) is not int or not -(2**31) <= x < 2**31 for x in row)
            for row in rows
        ):
            raise ValueError("server accepts INT32 width64 only, not real model hidden")
        p = self.continuations.pending.get(ticket.lane)
        if (
            p is None
            or p.ticket != ticket
            or any(
                p.routes.get((r.row, r.slot)) != r or r.expert != expert for r in routes
            )
        ):
            raise ValueError("packet does not belong to active layer routes")
        keys = {(ticket, r.row, r.slot) for r in routes}
        issued = {
            (packet.ticket, r.row, r.slot)
            for packet in self.active.values()
            for r in packet.routes
        }
        if (
            len(keys) != len(routes)
            or keys & issued
            or any((r.row, r.slot) in p.contributions for r in routes)
            or p.width != WIDTH
        ):
            raise ValueError("duplicate route publication or incompatible hidden width")
        slot = self.next_task % DEPTH
        if slot in self.active:
            raise BufferError("packet slot still owned")
        packet = Packet(self.next_task, ticket, expert, phase, routes, rows)
        self.active[slot] = packet
        self.next_task += 1
        return packet

    def complete(self, slot, generation, output):
        packet = self.active.get(slot)
        if packet is None or packet.generation != generation:
            raise ValueError("stale or future DONE")
        if len(output) != len(packet.routes) or any(
            len(row) != WIDTH for row in output
        ):
            raise ValueError("invalid result shape")
        for route, row in zip(packet.routes, output):
            self.continuations.receive(
                packet.ticket, route.row, route.slot, route.expert, row
            )
        # A device implementation must finish pulling/consuming before publishing
        # the next generation; deletion here denotes that completed consumption.
        del self.active[slot]
