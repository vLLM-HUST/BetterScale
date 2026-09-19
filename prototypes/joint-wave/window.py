"""Single-resident, two-wave resource authorization; no token reconstruction."""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Grant:
    sequence: int
    generation: int
    end: int


class WaveWindow:
    def __init__(self, *, position, granted_end, generation=1, width=5):
        if position < 0 or granted_end < position or generation < 1 or width < 1:
            raise ValueError('invalid resident resource grant')
        self.position = position
        self.granted_end = granted_end
        self.generation = generation
        self.width = width
        self.next_sequence = 0
        self.pending = deque()
        self.done = False
        self.failed = False

    @property
    def required_end(self):
        return self.position + (len(self.pending) + 1) * (self.width + 1) + self.width

    def can_authorize(self):
        return (not self.failed and not self.done and len(self.pending) < 2
                and self.required_end <= self.granted_end)

    def authorize(self):
        if self.failed or self.done:
            raise RuntimeError('resident is terminal or failed')
        if len(self.pending) == 2:
            raise RuntimeError('two-wave window is full')
        # Receipt-derived floor plus worst-case outstanding target progress and
        # draft queries. No predicted acceptance and no full-lifetime prelease.
        end = self.required_end
        if end > self.granted_end:
            raise RuntimeError('KV grant does not cover the submission horizon')
        grant = Grant(self.next_sequence, self.generation, self.granted_end)
        self.pending.append(grant)
        self.next_sequence += 1
        return grant

    def retire(self, *, sequence, generation, position, count, done, ok=True):
        if self.failed or not self.pending:
            raise RuntimeError('no live wave to retire')
        grant = self.pending[0]
        valid = (
            ok and sequence == grant.sequence and generation == grant.generation
            and 0 <= count <= self.width + 1
            and position == self.position + count
            and position <= grant.end
            and (not self.done or (done and count == 0))
        )
        if not valid:
            self.failed = True
            raise RuntimeError('invalid or stale completion; resident fail-stopped')
        self.pending.popleft()
        self.position = position
        self.done = self.done or done
