"""Closed-loop multi-session scheduler over immutable resource leases and N+2."""

import time
from dataclasses import dataclass
from cache import PrefixLeases


@dataclass
class Resident:
    lease: object
    session: int
    turn: int
    call: dict
    projected: int
    started: bool = False
    ready: bool = False
    completed: bool = False
    start_time: float = 0
    first_time: float | None = None
    tokens: list | None = None


class SessionScheduler:
    def __init__(
        self, cache, sessions, *, slots, chunks, table_width, generations=None
    ):
        self.cache = cache
        self.sessions = sessions
        self.slots = [None] * slots
        self.chunks = sorted(chunks, reverse=True)
        self.table_width = table_width
        self.generations = generations or [0] * slots
        self.waiting = [(i, 0) for i in range(len(sessions))]
        self.pending = []
        self.sequence = 0
        self.prefill_cursor = 0
        self.last_kind = "d"
        self.records = []
        self.events = []
        self.started = time.perf_counter()
        self.arrival = {(i, 0): self.started for i in range(len(sessions))}

    def admit(self):
        for slot in range(len(self.slots)):
            if not self.waiting:
                return
            old = self.slots[slot]
            if old is not None and not old.completed:
                continue
            session, turn = self.waiting[0]
            call = self.sessions[session]["calls"][turn]
            gen = self.generations[slot] + 1
            lease = self.cache.admit(
                f"s{session}t{turn}g{gen}",
                call["prompt_ids"],
                call["output_tokens"],
                slot,
                gen,
            )
            if lease is None:
                continue
            self.waiting.pop(0)
            self.generations[slot] = gen
            self.slots[slot] = Resident(
                lease,
                session,
                turn,
                call,
                lease.hit_tokens,
                start_time=self.arrival[(session, turn)],
                tokens=[],
            )

    def next_plan(self):
        if len(self.pending) >= 2:
            return None
        self.admit()
        prefills = [
            r for r in self.slots if r is not None and not r.completed and not r.ready
        ]
        decodes = [
            r for r in self.slots if r is not None and not r.completed and r.ready
        ]
        if not prefills and not decodes:
            return None
        seq = self.sequence
        bank = seq % 2
        if prefills and (self.last_kind == "d" or not decodes):
            r = prefills[self.prefill_cursor % len(prefills)]
            self.prefill_cursor += 1
            remaining = len(r.call["prompt_ids"]) - r.projected
            q = next(x for x in self.chunks if x <= remaining)
            end = r.projected + q
            command = [
                seq,
                r.lease.slot,
                r.lease.generation,
                int(not r.started),
                r.lease.hit_tokens,
                len(r.call["prompt_ids"]),
                r.call["output_tokens"],
            ]
            plan = dict(
                sequence=seq,
                key=f"p{q}b{bank}",
                kind="p",
                lengths=[end],
                residents=[r],
                inputs=dict(
                    command=command,
                    ids=r.call["prompt_ids"][r.projected : end],
                    block_row=r.lease.blocks
                    + [0] * (self.table_width - len(r.lease.blocks)),
                ),
            )
            r.started = True
            r.projected = end
            r.ready = end == len(r.call["prompt_ids"])
        else:
            lengths = [1] * len(self.slots)
            gens = [0] * len(self.slots)
            for r in decodes:
                last = len(r.call["prompt_ids"]) + r.call["output_tokens"] - 1
                active = r.projected < last
                r.projected = min(r.projected + 1, last)
                # N+1 can be an already-authorized terminal drain. Its actual
                # attention row is length1, not the completed resident's length.
                lengths[r.lease.slot] = r.projected if active else 1
                gens[r.lease.slot] = r.lease.generation
            plan = dict(
                sequence=seq,
                key=f"d{len(self.slots)}b{bank}",
                kind="d",
                lengths=lengths,
                residents=decodes,
                inputs=dict(command=[seq, 0, 0, 0, 0, 0, 0], generations=gens),
            )
        for r in plan["residents"]:
            self.cache.retain(r.lease)
        self.pending.append(plan)
        self.sequence += 1
        self.last_kind = plan["kind"]
        self.events.append(("submit", seq, len(self.pending)))
        return plan

    def receive(self, plan, rank_rows):
        if not self.pending or self.pending[0] is not plan:
            raise ValueError("out-of-order quorum")
        # TP ranks replicate token/cursor truth. Each must return the SAME rows.
        if not rank_rows or any(rows != rank_rows[0] for rows in rank_rows):
            raise ValueError("TP quorum disagrees")
        rows = {row[1]: row for row in rank_rows[0] if row[1] >= 0}
        expected = {r.lease.slot for r in plan["residents"]}
        if rows.keys() != expected:
            raise ValueError("receipt resident set differs from authorization")
        now = time.perf_counter()
        for r in plan["residents"]:
            row = rows[r.lease.slot]
            if row[0] != plan["sequence"] or row[2] != r.lease.generation or row[7]:
                raise ValueError("stale resident generation")
            _, slot, gen, cursor, token, count, terminal, error = row
            if count not in (0, 1):
                raise ValueError("invalid non-speculative output count")
            self.cache.commit(
                r.lease,
                cursor=cursor,
                token=token if count else None,
                terminal=bool(terminal),
            )
            if count:
                if r.first_time is None:
                    r.first_time = now
                r.tokens.append(token)
            if terminal and not r.completed:
                r.completed = True
                if len(r.tokens) != r.call["output_tokens"]:
                    raise ValueError("wrong terminal output budget")
                self.records.append(
                    dict(
                        session=r.session,
                        turn=r.turn,
                        prompt_tokens=len(r.call["prompt_ids"]),
                        output_tokens=len(r.tokens),
                        token_ids=r.tokens,
                        hit_tokens=r.lease.hit_tokens,
                        ttft_s=r.first_time - r.start_time,
                        latency_s=now - r.start_time,
                        tpot_s=(now - r.first_time) / max(1, len(r.tokens) - 1),
                    )
                )
                if r.turn + 1 < len(self.sessions[r.session]["calls"]):
                    self.waiting.append((r.session, r.turn + 1))
                    self.arrival[(r.session, r.turn + 1)] = now
        self.pending.pop(0)
        self.events.append(("quorum", plan["sequence"], len(self.pending)))

    @property
    def done(self):
        return (
            not self.waiting
            and not self.pending
            and all(r is None or r.completed for r in self.slots)
        )
