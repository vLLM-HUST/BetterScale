import unittest
from continuation import Continuations, Route
from server_contract import OraclePackets


class ServerContractTests(unittest.TestCase):
    def test_topk_out_of_order_packet_results(self):
        c = Continuations(1)
        routes = [Route(0, 0, 0, 0.25), Route(0, 1, 2, 0.75)]
        t = c.submit(0, 1, 1, 64, routes, [[1000] * 64])
        transport = OraclePackets(c)
        packets = [transport.publish(t, r.expert, 1, [r], [[7] * 64]) for r in routes]
        for p in reversed(packets):
            # Exact arithmetic of the delivered server, not a neural expert.
            out = [
                [x + t.layer * 100 + p.expert * 10 for x in row] for row in p.payload
            ]
            transport.complete(p.slot, p.generation, out)
        self.assertEqual(c.retire(), (t, [[1122.0] * 64]))
        self.assertEqual(packets[0].descriptor, (1, 0, 1, 0, 1, 1, 0, 0))

    def test_reject_bf16_and_future_done(self):
        c = Continuations(1)
        r = Route(0, 0, 0, 1)
        t = c.submit(0, 0, 1, 64, [r])
        p = OraclePackets(c)
        with self.assertRaises(ValueError):
            p.publish(t, 0, 0, [r], [[1.0] * 64])
        packet = p.publish(t, 0, 0, [r], [[1] * 64])
        with self.assertRaises(ValueError):
            p.complete(packet.slot, 2, [[1] * 64])
        self.assertIn(packet.slot, p.active)

    def test_slot_wrap_requires_consumption(self):
        c = Continuations(1)
        routes = [Route(0, i, i % 3, 1) for i in range(9)]
        t = c.submit(0, 0, 1, 64, routes)
        p = OraclePackets(c)
        packets = [p.publish(t, r.expert, 0, [r], [[1] * 64]) for r in routes[:8]]
        with self.assertRaises(BufferError):
            p.publish(t, 2, 0, [routes[8]], [[1] * 64])
        first = packets[0]
        p.complete(first.slot, first.generation, [[1] * 64])
        new = p.publish(t, 2, 0, [routes[8]], [[1] * 64])
        self.assertEqual((new.slot, new.generation), (0, 2))

    def test_duplicate_publication_is_rejected_before_state_changes(self):
        c = Continuations(1)
        r = Route(0, 0, 0, 1)
        t = c.submit(0, 0, 1, 64, [r])
        p = OraclePackets(c)
        with self.assertRaises(ValueError):
            p.publish(t, 0, 0, [r, r], [[1] * 64, [1] * 64])
        packet = p.publish(t, 0, 0, [r], [[1] * 64])
        with self.assertRaises(ValueError):
            p.publish(t, 0, 0, [r], [[1] * 64])
        p.complete(packet.slot, packet.generation, [[1] * 64])
        with self.assertRaises(ValueError):
            p.publish(t, 0, 0, [r], [[1] * 64])
        self.assertEqual(p.next_task, 1)
