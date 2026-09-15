"""CPU/control-plane ownership oracle for asynchronous routed contributions.

No torch/device synchronization here. A transport must make result data visible
before delivering Reply, and retain input/output backing until retirement.
"""
from dataclasses import dataclass, field
from collections import deque


@dataclass(frozen=True)
class Ticket:
    lane: int
    generation: int
    layer: int


@dataclass(frozen=True)
class Route:
    row: int
    slot: int
    expert: int
    weight: float


@dataclass
class Pending:
    ticket: Ticket
    rows: int
    width: int
    routes: dict
    contributions: dict = field(default_factory=dict)
    shared: list | None = None
    cancelled: bool = False


class Continuations:
    """One outstanding MoE per lane; immutable identity until complete/drained.

    Retire contributions in routing-slot order, not arrival order: network
    completion order must not change floating-point reduction order.
    """
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError('capacity must be positive')
        self.capacity = capacity
        self.pending = {}
        self.generations = {}
        self.ready = deque()

    def submit(self, lane, layer, rows, width, routes, shared=None):
        if lane in self.pending or len(self.pending) >= self.capacity:
            raise BufferError('lane active or capacity exhausted')
        if rows <= 0 or width <= 0 or layer < 0:
            raise ValueError('invalid dimensions/layer')
        mapped = {}
        for route in routes:
            key = route.row, route.slot
            if key in mapped or not 0 <= route.row < rows or route.slot < 0 or route.expert < 0:
                raise ValueError('invalid or duplicate routing slot')
            mapped[key] = route
        if any(not any(row == r for row, _ in mapped) for r in range(rows)):
            raise ValueError('every live row needs routed work')
        if shared is not None and (len(shared) != rows or any(len(x) != width for x in shared)):
            raise ValueError('invalid shared result shape')
        gen = self.generations.get(lane, 0) + 1
        self.generations[lane] = gen
        ticket = Ticket(lane, gen, layer)
        self.pending[lane] = Pending(ticket, rows, width, mapped, shared=shared)
        return ticket

    def receive(self, ticket, row, slot, expert, values):
        p = self.pending.get(ticket.lane)
        if p is None or p.ticket != ticket:
            raise ValueError('stale/unknown completion')
        key = row, slot
        route = p.routes.get(key)
        if route is None or route.expert != expert:
            raise ValueError('wrong routed contribution')
        if key in p.contributions:
            raise ValueError('duplicate completion')
        if len(values) != p.width:
            raise ValueError('wrong hidden width')
        p.contributions[key] = tuple(values)
        if len(p.contributions) == len(p.routes):
            self.ready.append(ticket)

    def cancel(self, ticket):
        p = self.pending.get(ticket.lane)
        if p is None or p.ticket != ticket:
            raise ValueError('stale/unknown cancellation')
        p.cancelled = True  # Retain backing until all issued contributions return.

    def retire(self):
        if not self.ready:
            return None
        ticket = self.ready.popleft()
        p = self.pending.pop(ticket.lane)
        assert p.ticket == ticket
        if p.cancelled:
            return ticket, None
        out = [[0.0] * p.width for _ in range(p.rows)]
        for key in sorted(p.routes):
            route = p.routes[key]
            for col, value in enumerate(p.contributions[key]):
                out[route.row][col] += route.weight * value
        if p.shared is not None:
            for row in range(p.rows):
                for col in range(p.width):
                    out[row][col] += p.shared[row][col]
        return ticket, out
