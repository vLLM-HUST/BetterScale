"""Experimental host-scheduled EP waves, independent of the persistent ABI.

Only immutable control descriptors live here. Token payloads and route maps stay
on device. Each owner runs its own Scheduler; no all-client/all-owner barrier.
"""
from dataclasses import dataclass
from enum import Enum


class Kind(str, Enum):
    DECODE = 'decode'
    PREFILL = 'prefill'
    MIXED = 'mixed'


class ReturnMode(str, Enum):
    PUSH = 'push'
    PULL = 'pull'


@dataclass(frozen=True)
class Request:
    client: int
    generation: int
    layer: int
    tokens: int
    kind: Kind


@dataclass(frozen=True)
class Wave:
    sequence: int
    primary: int
    requests: tuple[Request, ...]
    slot: int

    @property
    def tokens(self):
        return sum(r.tokens for r in self.requests)

    @property
    def layer(self):
        return self.requests[0].layer


@dataclass(frozen=True)
class Output:
    """Owner-local output lease; payload addresses are resolved out of band.

    For pull, slot remains pinned until every member acknowledges its completed
    copy. Neither notification delivery nor DOWN completion retires this lease.
    """
    wave: int
    slot: int
    request: Request


class Scheduler:
    def __init__(self, clients, max_tokens, *, layers=41, return_mode=ReturnMode.PUSH):
        if any(type(x) is not int or x <= 0 for x in (clients, max_tokens, layers)):
            raise ValueError('positive integer geometry required')
        self.clients, self.max_tokens, self.layers = clients, max_tokens, layers
        self.return_mode = ReturnMode(return_mode)
        self.cursor = 0
        self.sequence = 0
        self.retired = [0] * clients
        self.last_retired = [None] * clients
        self.ready = {}
        self.active = {}  # client -> exact immutable request until retirement
        self.waves = {}  # sequence -> wave
        self.slots = [None, None]
        self.completed = set()
        self.published = {}  # client -> Output
        self.closed = set()

    def _client(self, client):
        if type(client) is not int or not 0 <= client < self.clients:
            raise ValueError('client outside admitted geometry')

    def publish(self, request):
        """Observe READY after the producer's payload/metadata visibility fence.

        Re-observing the same immutable descriptor is allowed. A changed live
        descriptor or skipped generation is a protocol error, not a new request.
        """
        r = request
        self._client(r.client)
        if (type(r.generation) is not int or type(r.layer) is not int
                or type(r.tokens) is not int or not isinstance(r.kind, Kind)
                or not 0 <= r.layer < self.layers or not 0 < r.tokens <= self.max_tokens):
            raise ValueError('invalid request descriptor')
        # READY may retain the last descriptor after its ACK is observed.
        # Do not require the client to clear a mailbox before republishing.
        if r == self.last_retired[r.client]:
            return
        if r.client in self.closed:
            raise ValueError('publication after EOF')
        if r.generation != self.retired[r.client] + 1:
            raise ValueError('request generation is not the next unretired generation')
        old = self.ready.get(r.client, self.active.get(r.client))
        if old is not None:
            if old != r:
                raise ValueError('live request mutated before retirement')
            return
        self.ready[r.client] = r

    def admit(self):
        """Freeze one finite wave from a single ready snapshot; never wait to fill.

        Prefill/mixed are singleton waves. Decode may piggyback same-layer decode
        in circular order if the entire request fits. Cursor advances after the
        primary, never after the last passenger. Two output slots bound lookahead.
        """
        if not self.ready or None not in self.slots:
            return None
        order = [(self.cursor + i) % self.clients for i in range(self.clients)]
        primary = next(c for c in order if c in self.ready)
        first = self.ready[primary]
        members, tokens = [first], first.tokens
        if first.kind is Kind.DECODE:
            for i in range(1, self.clients):
                r = self.ready.get((primary + i) % self.clients)
                if (r is not None and r.kind is Kind.DECODE and r.layer == first.layer
                        and tokens + r.tokens <= self.max_tokens):
                    members.append(r)
                    tokens += r.tokens
        self.sequence += 1
        slot = self.slots.index(None)
        wave = Wave(self.sequence, primary, tuple(members), slot)
        self.slots[slot] = wave.sequence
        self.waves[wave.sequence] = wave
        for r in members:
            self.active[r.client] = self.ready.pop(r.client)
        self.cursor = (primary + 1) % self.clients
        return wave

    def down_complete(self, wave):
        self._wave(wave)
        if wave.sequence in self.completed:
            raise ValueError('duplicate DOWN completion')
        self.completed.add(wave.sequence)

    def _wave(self, wave):
        if self.waves.get(wave.sequence) != wave:
            raise ValueError('unknown or retired wave')

    def publish_output(self, wave):
        """Called AFTER push writes, or after local output writes for pull.

        Completion must be visible after payload, not merely after enqueue. An
        empty-owner request follows the same notification/retirement protocol.
        """
        self._wave(wave)
        if wave.sequence not in self.completed:
            raise ValueError('output before DOWN completion')
        if any(r.client in self.published for r in wave.requests):
            raise ValueError('duplicate output publication')
        outputs = tuple(Output(wave.sequence, wave.slot, r) for r in wave.requests)
        self.published.update((o.request.client, o) for o in outputs)
        if self.return_mode is ReturnMode.PUSH:
            # Owner scratch is free after all pushes finish. Client destinations
            # are NOT free: active keeps them generation-owned until retirement.
            self.slots[wave.slot] = None
        return outputs

    def retire(self, output):
        """Client ack follows payload consumption (and completed pull, if used)."""
        r = output.request
        self._client(r.client)
        if self.published.get(r.client) != output:
            raise ValueError('stale, duplicate or premature retirement')
        del self.published[r.client]
        del self.active[r.client]
        self.retired[r.client] = r.generation
        self.last_retired[r.client] = r
        wave = self.waves[output.wave]
        if not any(self.active.get(x.client) == x for x in wave.requests):
            if self.return_mode is ReturnMode.PULL:
                self.slots[wave.slot] = None
            del self.waves[wave.sequence]
            self.completed.remove(wave.sequence)

    def close(self, client, generation):
        self._client(client)
        if (type(generation) is not int or generation != self.retired[client]
                or client in self.ready or client in self.active):
            raise ValueError('EOF requires exact fully retired generation')
        self.closed.add(client)

    @property
    def drained(self):
        return len(self.closed) == self.clients and not self.waves


def route_destination(token, topk_slot, *, tokens, topk, hidden):
    """BF16-element offset in client [token, original top-k slot, hidden]."""
    if (any(type(x) is not int for x in (token, topk_slot, tokens, topk, hidden))
            or min(tokens, topk, hidden) <= 0 or not 0 <= token < tokens
            or not 0 <= topk_slot < topk):
        raise ValueError('invalid route destination')
    return (token * topk + topk_slot) * hidden


def expert_owner(expert, *, experts=256, owners=2):
    if (any(type(x) is not int for x in (expert, experts, owners))
            or experts <= 0 or owners <= 0 or experts % owners
            or not 0 <= expert < experts):
        raise ValueError('invalid EP geometry or expert ID')
    return expert // (experts // owners)
