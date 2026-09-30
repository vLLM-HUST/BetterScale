"""Executable CPU protocol oracle, not NPU transport/performance qualification."""
import math
import unittest
from dataclasses import dataclass

from betterscale.patches.expert_service.wave_protocol import (
    Kind, ReturnMode, Request, Scheduler, expert_owner, route_destination,
)
from betterscale.patches.expert_service.wave_server import ServerLoop


def req(c, n=1, layer=0, kind=Kind.DECODE, gen=1):
    return Request(c, gen, layer, n, kind)


class Admission(unittest.TestCase):
    def test_same_layer_decode_snapshot_and_primary_cursor(self):
        s = Scheduler(5, 8)
        for r in (req(0, 3), req(1, 4, 1), req(2, 3), req(3, 4), req(4, 1)):
            s.publish(r)
        w = s.admit()
        self.assertEqual([r.client for r in w.requests], [0, 2, 4])
        self.assertEqual(w.tokens, 7)
        self.assertEqual(s.cursor, 1)  # not after passenger 4
        next_wave = s.admit()
        self.assertEqual(next_wave.primary, 1)
        self.assertEqual(next_wave.layer, 1)
        self.assertIsNone(s.admit())  # both slots reserved

    def test_prefill_and_mixed_cannot_hitchhike_or_take_passengers(self):
        for kind in (Kind.PREFILL, Kind.MIXED):
            s = Scheduler(3, 8)
            for r in (req(0, 2), req(1, 4, kind=kind), req(2, 2)):
                s.publish(r)
            self.assertEqual([r.client for r in s.admit().requests], [0, 2])
            self.assertEqual([r.client for r in s.admit().requests], [1])
            s = Scheduler(2, 8)
            s.publish(req(0, 4, kind=kind)); s.publish(req(1, 1))
            self.assertEqual(len(s.admit().requests), 1)

    def test_no_wait_to_fill_no_partial_request_no_live_wave_extension(self):
        s = Scheduler(3, 4)
        s.publish(req(0, 3)); s.publish(req(1, 2))
        w = s.admit()
        self.assertEqual(w.requests, (req(0, 3),))
        s.publish(req(2, 1))
        self.assertEqual(w.requests, (req(0, 3),))
        self.assertEqual(s.admit().requests, (req(1, 2), req(2, 1)))

    def test_primary_round_robin_including_prefill(self):
        s = Scheduler(4, 8)
        primaries = []
        for _ in range(12):
            for c in range(4):
                if c not in s.ready:
                    s.publish(req(c, 8, kind=Kind.PREFILL, gen=s.retired[c]+1))
            w = s.admit(); primaries.append(w.primary)
            s.down_complete(w)
            for o in s.publish_output(w): s.retire(o)
        self.assertEqual(primaries, [0, 1, 2, 3]*3)

    def test_generations_reobserve_mutation_skip_and_eof(self):
        s = Scheduler(2, 8)
        r = req(0)
        s.publish(r); s.publish(r)
        with self.assertRaises(ValueError): s.publish(req(0, 2))
        with self.assertRaises(ValueError): s.publish(req(1, gen=2))
        w = s.admit(); s.publish(r)
        with self.assertRaises(ValueError): s.close(0, 0)
        with self.assertRaises(ValueError): s.publish_output(w)
        s.down_complete(w)
        o, = s.publish_output(w)
        with self.assertRaises(ValueError): s.publish(req(0, gen=2))
        s.retire(o)
        with self.assertRaises(ValueError): s.retire(o)
        s.publish(r)  # old READY may remain visible after ACK
        self.assertFalse(s.ready)
        with self.assertRaises(ValueError): s.publish(req(0, 2))
        s.close(0, 1); s.close(1, 0)
        self.assertTrue(s.drained)
        with self.assertRaises(ValueError): s.publish(req(0, gen=2))

    def test_descriptor_validation_is_fail_before_claim(self):
        s = Scheduler(2, 8)
        for r in (req(-1), req(2), req(0, 0), req(0, 9), req(0, layer=41),
                  req(0, kind='decode'), req(True), req(0, gen=True)):
            with self.assertRaises(ValueError): s.publish(r)
        self.assertFalse(s.ready)
        self.assertEqual(s.cursor, 0)

    def test_push_releases_server_slot_not_client_destination(self):
        s = Scheduler(3, 8)
        s.publish(req(0)); w = s.admit(); s.down_complete(w)
        old, = s.publish_output(w)
        self.assertEqual(s.slots, [None, None])
        with self.assertRaises(ValueError): s.publish(req(0, gen=2))
        s.publish(req(1)); new = s.admit()
        self.assertEqual(new.slot, w.slot)
        s.retire(old)  # must not release new wave's reused slot
        self.assertEqual(s.slots[new.slot], new.sequence)

    def test_pull_pins_slot_until_every_client_copy_is_acknowledged(self):
        s = Scheduler(4, 8, return_mode=ReturnMode.PULL)
        s.publish(req(0)); s.publish(req(1))
        w = s.admit(); s.down_complete(w); a, b = s.publish_output(w)
        s.retire(a)
        self.assertEqual(s.slots[w.slot], w.sequence)
        s.publish(req(2, layer=1)); w2 = s.admit()
        s.publish(req(3, layer=2)); self.assertIsNone(s.admit())
        s.retire(b)
        w3 = s.admit()
        self.assertEqual(w3.slot, w.slot)
        self.assertEqual(s.slots[w2.slot], w2.sequence)

    def test_route_offsets_and_ep(self):
        offsets = {route_destination(t, k, tokens=3, topk=8, hidden=2048)
                   for t in range(3) for k in range(8)}
        self.assertEqual(len(offsets), 24)
        self.assertEqual(max(offsets), 23*2048)
        self.assertEqual([expert_owner(x) for x in (0, 127, 128, 255)], [0, 0, 1, 1])
        with self.assertRaises(ValueError): expert_owner(256)
        with self.assertRaises(ValueError): expert_owner(2, experts=5, owners=2)
        with self.assertRaises(ValueError): route_destination(0, 8, tokens=1, topk=8, hidden=2)


@dataclass
class Event:
    op: str
    wave: int
    start: int
    end: int
    action: object
    complete: bool = False


class Oracle:
    """Two FIFO engines with delayed completion and actual toy route arithmetic.

    Simulated durations only test dependencies. They predict no hardware timing.
    """
    def __init__(self, mode, requests, *, owner=0, pull_time=3, ack_delay=4):
        self.mode, self.owner = mode, owner
        self.requests = {r.client: r for r in requests}
        self.now = 0
        self.tails = {'cube': 0, 'vector': 0}
        self.events, self.acks, self.notices = [], [], []
        self.scratch, self.destinations, self.received = {}, {}, {}
        self.pull_time, self.ack_delay = pull_time, ack_delay

    def snapshot(self): return tuple(self.requests.values())
    def acknowledgements(self):
        result, self.acks = self.acks, []
        return result
    def done(self, e): return e.complete
    def idle(self): self.advance()

    @staticmethod
    def routes(r):
        # Change inputs and routes with generation, including one empty EP owner.
        for t in range(r.tokens):
            for k in range(2):
                expert = (r.client+t+k+r.generation) % 4
                yield (r.client, r.generation, t, k), expert, (t+1)*.1+r.generation*.2

    @staticmethod
    def reference(r, expert, x):
        # Independent scalar FFN oracle; not BF16 or Qwen numerical qualification.
        gate = x*(expert+1)
        up = x*(r.layer+2)
        return (gate/(1+math.exp(-gate)))*up*(expert+r.layer+1)

    def launch(self, op, wave, stream, waits):
        start = max([self.now, self.tails[stream]]+[e.end for e in waits])
        duration = {'pull': self.pull_time, 'up': 5, 'gate': 2, 'down': 5, 'push': 2}[op]
        def action():
            if op == 'pull':
                self.scratch[wave.slot] = dict(wave=wave.sequence, inputs={}, gate={}, values={})
                for r in wave.requests:
                    for key, expert, x in self.routes(r):
                        if expert_owner(expert, experts=4, owners=2) == self.owner:
                            self.scratch[wave.slot]['inputs'][key] = (r, expert, x)
            else:
                slot = self.scratch[wave.slot]
                assert slot['wave'] == wave.sequence, 'slot reused before read completion'
                if op == 'up':
                    slot['gate'] = {key:(x*(e+1), x*(r.layer+2), e+r.layer+1)
                                    for key,(r,e,x) in slot['inputs'].items()}
                elif op == 'gate':
                    slot['values'] = {key:(g/(1+math.exp(-g)))*u
                                      for key,(g,u,_) in slot['gate'].items()}
                elif op == 'down':
                    slot['values'] = {key:y*slot['gate'][key][2] for key,y in slot['values'].items()}
                elif op == 'push':
                    self.destinations.update(slot['values'])
        e = Event(op, wave.sequence, start, start+duration, action)
        self.tails[stream] = e.end
        self.events.append(e)
        return e

    def notify(self, outputs, mode):
        self.notices.extend(outputs)
        for o in outputs:
            def consume(o=o):
                if mode is ReturnMode.PULL:
                    slot = self.scratch[o.slot]
                    assert slot['wave'] == o.wave, 'pull output overwritten before ack'
                    data = slot['values']
                else:
                    data = self.destinations
                r = o.request
                for key, expert, x in self.routes(r):
                    if expert_owner(expert, experts=4, owners=2) == self.owner:
                        expected = self.reference(r, expert, x)
                        assert abs(data[key]-expected) < 1e-12
                        self.received[key] = data[key]
                del self.requests[r.client]
                self.acks.append(o)
            self.events.append(Event('consume', o.wave, self.now, self.now+self.ack_delay, consume))

    def advance(self):
        self.now += 1
        for e in sorted(self.events, key=lambda e:e.end):
            if not e.complete and e.end <= self.now:
                e.action(); e.complete = True


def execute(mode, requests, **kwargs):
    s = Scheduler(max(r.client for r in requests)+1, 8, return_mode=mode)
    b = Oracle(mode, requests, **kwargs)
    loop = ServerLoop(s, b)
    for _ in range(250):
        loop.tick(); b.advance()
        if not b.requests and not s.active and not loop.pending:
            break
    else: raise AssertionError('CPU protocol failed to drain')
    for c in range(s.clients): s.close(c, s.retired[c])
    assert s.drained
    return s,b,loop


class Execution(unittest.TestCase):
    def test_final_single_wave_returns_without_next_wave(self):
        for mode in ReturnMode:
            s,b,_ = execute(mode, [req(0, 2)])
            self.assertTrue(s.drained)
            self.assertEqual(len(b.notices), 1)
            self.assertEqual([e.op for e in b.events].count('pull'), 1)

    def test_gate_before_next_pull_and_dual_engine_overlap(self):
        _,b,_ = execute(ReturnMode.PUSH, [req(0, 2), req(1, 2, layer=1)])
        event = {(e.wave,e.op):e for e in b.events}
        self.assertGreaterEqual(event[2,'pull'].start, event[1,'gate'].end)
        self.assertLess(event[2,'pull'].start, event[1,'down'].end)
        self.assertGreaterEqual(event[2,'up'].start, event[2,'pull'].end)
        self.assertGreaterEqual(event[2,'up'].start, event[1,'down'].end)
        self.assertLess(event[1,'push'].start, event[2,'up'].end)
        self.assertLess(event[2,'up'].start, event[1,'push'].end)

    def test_long_pull_is_allowed_bubble_not_deadlock(self):
        for mode in ReturnMode:
            _,b,_ = execute(mode, [req(0, 2),req(1, 8, layer=1, kind=Kind.PREFILL)], pull_time=20)
            self.assertEqual(len(b.notices), 2)
            event = {(e.wave,e.op):e for e in b.events}
            self.assertGreater(event[2,'pull'].end, event[1,'down'].end)

    def test_push_pull_identical_route_outputs_and_empty_owner(self):
        requests = [req(0, 1), req(1, 3), req(2, 4, layer=1, kind=Kind.MIXED)]
        for owner in (0,1):
            _,push,_ = execute(ReturnMode.PUSH, requests, owner=owner)
            _,pull,_ = execute(ReturnMode.PULL, requests, owner=owner, ack_delay=25)
            self.assertEqual(push.received, pull.received)
            self.assertEqual(len(push.notices), len(requests))
        # client0/gen3 routes are experts3/0; client0/gen1 routes1/2. Choose
        # client1/gen1 => experts2/3: owner0 has no route, still must complete.
        _,b,_ = execute(ReturnMode.PULL, [req(1)], owner=0)
        self.assertFalse(b.received)
        self.assertEqual(len(b.notices), 1)

    def test_stop_is_not_successful_drain(self):
        s = Scheduler(1, 8)
        loop = ServerLoop(s, Oracle(ReturnMode.PUSH, [req(0)]))
        self.assertFalse(loop.run(lambda: True))
        with self.assertRaises(TimeoutError): loop.run(lambda: False, max_ticks=0)


if __name__ == '__main__': unittest.main()
