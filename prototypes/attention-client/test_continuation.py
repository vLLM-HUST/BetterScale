import unittest
from continuation import Continuations, Route, Ticket


class ContinuationTests(unittest.TestCase):
    def test_out_of_order_and_shared(self):
        c = Continuations(2)
        t = c.submit(0, 3, 1, 2, [Route(0, 0, 2, 0.25), Route(0, 1, 7, 0.75)], [[1, 2]])
        c.receive(t, 0, 1, 7, [8, 12])
        self.assertIsNone(c.retire())
        c.receive(t, 0, 0, 2, [4, 8])
        self.assertEqual(c.retire(), (t, [[8, 13]]))

    def test_slow_lane_does_not_block_ready_lane(self):
        c = Continuations(2)
        slow = c.submit(0, 0, 1, 1, [Route(0, 0, 0, 1)])
        fast = c.submit(1, 0, 1, 1, [Route(0, 0, 0, 1)])
        c.receive(fast, 0, 0, 0, [3])
        self.assertEqual(c.retire(), (fast, [[3]]))
        self.assertEqual(c.pending[0].ticket, slow)

    def test_cancel_drains_before_reuse(self):
        c = Continuations(1)
        args = (0, 0, 1, 1, [Route(0, 0, 0, 1)])
        old = c.submit(*args)
        c.cancel(old)
        with self.assertRaises(BufferError):
            c.submit(*args)
        c.receive(old, 0, 0, 0, [3])
        self.assertEqual(c.retire(), (old, None))
        new = c.submit(*args)
        self.assertGreater(new.generation, old.generation)
        with self.assertRaises(ValueError):
            c.receive(old, 0, 0, 0, [3])

    def test_reject_wrong_and_duplicate_completion(self):
        c = Continuations(1)
        t = c.submit(0, 0, 1, 1, [Route(0, 0, 4, 1)])
        for ticket, row, slot, expert, value in [
            (Ticket(0, 1, 1), 0, 0, 4, [1]),
            (t, 0, 0, 5, [1]),
            (t, 1, 0, 4, [1]),
            (t, 0, 0, 4, [1, 2]),
        ]:
            with self.assertRaises(ValueError):
                c.receive(ticket, row, slot, expert, value)
        c.receive(t, 0, 0, 4, [1])
        with self.assertRaises(ValueError):
            c.receive(t, 0, 0, 4, [1])

    def test_reduction_not_arrival_order(self):
        routes = [Route(0, i, i, 1) for i in range(3)]
        results = []
        for order in [[0, 1, 2], [2, 0, 1]]:
            c = Continuations(1)
            t = c.submit(0, 0, 1, 1, routes)
            for i in order:
                c.receive(t, 0, i, i, [[1e20], [-1e20], [3]][i])
            results.append(c.retire()[1])
        self.assertEqual(results, [[[3]], [[3]]])

    def test_reject_missing_row(self):
        with self.assertRaises(ValueError):
            Continuations(1).submit(0, 0, 2, 1, [Route(0, 0, 0, 1)])


class LayerSchedulerTests(unittest.TestCase):
    def test_progress_independent_of_slow_expert(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from scheduler import Lane, LayerScheduler, ReferenceTransport

        class Norm:
            def __call__(self, x, residual=None):
                return x if residual is None else (x + residual, x + residual)

        layer = SimpleNamespace(
            input_layernorm=Norm(),
            post_attention_layernorm=Norm(),
            self_attn=lambda positions, hidden_states: hidden_states + 1,
            mlp=lambda x: x * 2,
        )
        model = SimpleNamespace(layers=[layer, layer], norm=Norm())

        class Delayed(ReferenceTransport):
            def __init__(self):
                super().__init__()
                self.slow = set()
                self.release = False

            def submit(self, identity, *args):
                h = super().submit(identity, *args)
                if identity == "slow":
                    self.slow.add(h)
                return h

            def poll(self, h):
                if h in self.slow and not self.release:
                    return None
                return super().poll(h)

        transport = Delayed()
        s = LayerScheduler(transport)
        for name in ["slow", "fast"]:
            s.admit(Lane(name, model, 0, 1, nullcontext))
        for _ in range(4):
            s.tick()
        self.assertEqual(s.completed.popleft()[0], "fast")
        self.assertEqual(s.lanes["slow"].layer, 0)
        transport.release = True
        for _ in range(3):
            s.tick()
        self.assertEqual(s.completed.popleft()[0], "slow")
        self.assertFalse(s.lanes)


if __name__ == "__main__":
    unittest.main()
