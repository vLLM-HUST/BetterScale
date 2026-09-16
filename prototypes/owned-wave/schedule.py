"""Bounded N+2 authority ledger; no numerical token feedback enters planning."""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Wave:
    sequence: int
    generation: int
    kind: str
    length: int
    max_tokens: int

    @property
    def bank(self):
        return self.sequence % 2


class NPlusTwo:
    """One balanced resident per DP owner, serial request generations.

    Receive the COMPLETE rank quorum for N, then authorize N+2 while N+1
    remains outstanding. A terminal receipt allows the next generation behind
    that old-generation drain; a slot is never overwritten on ingress.
    """
    def __init__(self, *, width, max_tokens, grant_end, requests, ranks):
        if min(width, max_tokens, requests, ranks) < 1 or width + max_tokens - 1 > grant_end:
            raise ValueError('request exceeds finite resource grant')
        self.width, self.max_tokens = width, max_tokens
        self.requests, self.ranks = requests, ranks
        self.pending = deque()
        self.sequence = 0
        self.generation = 0
        self.decode_index = 0
        self.terminal = set()
        self.trace = []

    def next_wave(self):
        if len(self.pending) >= 2:
            return None
        if self.generation == 0 or self.generation in self.terminal:
            if self.generation == self.requests:
                return None
            self.generation += 1
            self.decode_index = 0
            kind, length = 'prefill', self.width
        else:
            self.decode_index += 1
            kind = 'decode'
            length = self.width + min(self.decode_index, self.max_tokens - 1)
        wave = Wave(self.sequence, self.generation, kind, length, self.max_tokens)
        self.sequence += 1
        self.pending.append(wave)
        self.trace.append(dict(action='submit', sequence=wave.sequence,
                               generation=wave.generation, kind=kind,
                               outstanding=[x.sequence for x in self.pending]))
        return wave

    def receive(self, rows):
        if not self.pending or len(rows) != self.ranks:
            raise ValueError('complete rank quorum required')
        wave = self.pending[0]
        # rows are ordered by the communicator, not an untrusted rank-0 summary.
        for row in rows:
            if len(row) != 7 or row[0] != wave.sequence or row[1] != wave.generation or row[6]:
                raise ValueError('stale or failed rank receipt')
        if len({(r[4], r[5]) for r in rows}) != 1:
            raise ValueError('this balanced portfolio cannot admit asymmetric termination')
        if rows[0][5]:
            self.terminal.add(wave.generation)
        self.pending.popleft()
        self.trace.append(dict(action='quorum', sequence=wave.sequence,
                               generation=wave.generation, terminal=bool(rows[0][5]),
                               outstanding=[x.sequence for x in self.pending]))
        return wave
