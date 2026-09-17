"""CPU-only retained-session workload state; tools remain inert token data."""

import json
from pathlib import Path


def load_sessions(path, count=4, turns=0, output_cap=0):
    data = json.loads(Path(path).read_text())
    sessions = data["conversations"][:count]
    if len(sessions) != count:
        raise ValueError("not enough distinct sessions")
    if len({s["trace_id"] for s in sessions}) != count:
        raise ValueError("duplicate sessions")
    result = []
    for session in sessions:
        selected = session["turns"][:turns] if turns else session["turns"]
        out = []
        for turn in selected:
            ids = list(turn["prompt_delta_token_ids"])
            n = int(turn["output_tokens"])
            if not ids or n < 1:
                raise ValueError("empty turn requires an explicit continuation policy")
            out.append(dict(ids=ids, output=min(n, output_cap) if output_cap else n))
        result.append(dict(trace_id=session["trace_id"], turns=out))
    return result


class Seat:
    def __init__(self, session):
        self.session = session
        self.turn = 0
        self.cursor = 0  # target-encoded rows; the pending output is not encoded
        self.pending = None
        self.generated = 0
        self.remaining = 0
        self.todo = list(session["turns"][0]["ids"])
        self.prefilled = 0
        self.reused = 0
        self.done = False
        self.events = []

    def take(self, width):
        part, self.todo = self.todo[:width], self.todo[width:]
        return part

    def finish_prefill(self, count, token):
        self.cursor += count
        self.prefilled += count
        if not self.todo:
            self.pending = int(token)
            self.generated += 1
            self.remaining = self.session["turns"][self.turn]["output"] - 1
            return True
        return False

    def finish_decode(self, count, token):
        if count < 0 or count > self.remaining:
            raise ValueError("decoder exceeded fixed output budget")
        self.cursor += count
        self.generated += count
        self.remaining -= count
        if count:
            self.pending = int(token)

    def next_turn(self):
        if self.todo or self.remaining:
            return False
        self.turn += 1
        if self.turn == len(self.session["turns"]):
            self.done = True
        else:
            # Exactly one already-emitted token remains outside target KV.
            self.reused += self.cursor
            self.todo = [self.pending, *self.session["turns"][self.turn]["ids"]]
        return True
