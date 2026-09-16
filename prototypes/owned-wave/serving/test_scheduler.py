"""CPU numerical-state double checks multi-session authority and drain lifetime."""

import unittest
from types import SimpleNamespace
from scheduler import SessionScheduler


class Cache:
    def __init__(self):
        self.live = {}
        self.deferred = 0

    def admit(self, key, prompt, budget, slot, generation):
        lease = SimpleNamespace(
            slot=slot,
            generation=generation,
            hit_tokens=0,
            blocks=[1, 2],
            references=0,
            terminal=False,
            freed=False,
        )
        self.live[(slot, generation)] = lease
        return lease

    def retain(self, lease):
        assert not lease.freed
        lease.references += 1

    def commit(self, lease, *, cursor, token=None, terminal=False):
        lease.terminal |= terminal
        if terminal and lease.references > 1:
            self.deferred += 1
        lease.references -= 1
        if lease.references == 0 and lease.terminal:
            lease.freed = True
            del self.live[(lease.slot, lease.generation)]


class SchedulerTests(unittest.TestCase):
    def test_skewed_sessions_chunk_tails_and_turnover(self):
        sessions = [
            dict(
                calls=[
                    dict(prompt_ids=list(range(n)), output_tokens=m) for n, m in calls
                ]
            )
            for calls in [[(7, 2), (11, 1)], [(5, 4), (13, 3)]]
        ]
        cache = Cache()
        s = SessionScheduler(cache, sessions, slots=2, chunks=[1, 2, 4], table_width=4)
        state = {}
        quorums = []
        seen = []
        while not s.done:
            while (p := s.next_plan()) is not None:
                rows = [[-1, -1, 0, 0, -1, 0, 0, 0] for _ in range(2)]
                for r in p["residents"]:
                    key = (r.lease.slot, r.lease.generation)
                    cursor, generated = state.get(key, (0, 0))
                    if p["kind"] == "p":
                        cursor = p["lengths"][0]
                    else:
                        cursor += int(generated < r.call["output_tokens"])
                    count = int(
                        (p["kind"] == "d" or cursor == len(r.call["prompt_ids"]))
                        and generated < r.call["output_tokens"]
                    )
                    generated += count
                    state[key] = (cursor, generated)
                    row = [
                        p["sequence"],
                        key[0],
                        key[1],
                        cursor,
                        42 if count else -1,
                        count,
                        int(generated == r.call["output_tokens"]),
                        0,
                    ]
                    rows[key[0]] = row
                quorums.append((p, rows))
                seen.append(p["kind"])
            p, rows = quorums.pop(0)
            s.receive(p, [rows, rows])
        self.assertEqual(len(s.records), 4)
        self.assertFalse(cache.live)
        self.assertEqual(sum(r["output_tokens"] for r in s.records), 10)
        self.assertIn("d", seen)
        self.assertIn("p", seen)
        self.assertTrue(
            all(len(r["token_ids"]) == r["output_tokens"] for r in s.records)
        )
        self.assertGreater(cache.deferred, 0)

    def test_tp_disagreement_cannot_release(self):
        cache = Cache()
        s = SessionScheduler(
            cache,
            [dict(calls=[dict(prompt_ids=[1], output_tokens=1)])],
            slots=1,
            chunks=[1],
            table_width=2,
        )
        p = s.next_plan()
        with self.assertRaises(ValueError):
            s.receive(p, [[[0, 0, 1, 1, 5, 1, 1, 0]], [[0, 0, 1, 1, 6, 1, 1, 0]]])
        self.assertEqual(p["residents"][0].lease.references, 1)


if __name__ == "__main__":
    unittest.main()
