"""NumPy execution of the finite-wave contract; no NPU or BF16 claim."""
import unittest
import numpy as np

from test_expert_wave_service import Event, Oracle, req
from betterscale.patches.expert_service.wave_protocol import Kind, ReturnMode, Scheduler
from betterscale.patches.expert_service.wave_server import ServerLoop


class MatrixOracle(Oracle):
    def __init__(self, mode, requests, *, owner, owners, payloads, weights):
        super().__init__(mode, requests, owner=owner, ack_delay=13)
        self.owners, self.payloads, self.weights = owners, payloads, weights
        self.experts = weights[0][0].shape[0]

    def launch(self, op, wave, stream, waits):
        start = max([self.now, self.tails[stream]]+[e.end for e in waits])
        def action():
            if op == 'pull':
                groups = {}
                for r in wave.requests:
                    x, ids, _ = self.payloads[r.client, r.generation]
                    assert len(x) == r.tokens
                    for token, routes in enumerate(ids):
                        for k, expert in enumerate(routes):
                            if expert // (self.experts//self.owners) == self.owner:
                                key = (r.client, r.generation, token, k)
                                groups.setdefault(int(expert), []).append((key, x[token].copy()))
                self.scratch[wave.slot] = dict(wave=wave.sequence, groups=groups, values={})
                return
            state = self.scratch[wave.slot]
            assert state['wave'] == wave.sequence
            gate, up, down = self.weights[wave.layer]
            if op == 'up':
                state['intermediate'] = {}
                for expert, routes in state['groups'].items():
                    x = np.stack([row for _, row in routes])
                    state['intermediate'][expert] = (x @ gate[expert].T, x @ up[expert].T)
            elif op == 'gate':
                state['activated'] = {e:(g/(1+np.exp(-g)))*u
                                      for e,(g,u) in state['intermediate'].items()}
            elif op == 'down':
                for expert, a in state['activated'].items():
                    y = a @ down[expert].T
                    for (key,_), row in zip(state['groups'][expert], y):
                        state['values'][key] = row.copy()
            elif op == 'push':
                for key, row in state['values'].items():
                    self.destinations[key] = row.copy()
        event = Event(op, wave.sequence, start, start+3, action)
        self.tails[stream] = event.end
        self.events.append(event)
        return event

    def notify(self, outputs, mode):
        self.notices.extend(outputs)
        for output in outputs:
            def consume(o=output):
                if mode is ReturnMode.PULL:
                    state = self.scratch[o.slot]
                    assert state['wave'] == o.wave
                    data = state['values']
                else:
                    data = self.destinations
                r = o.request
                for key, value in data.items():
                    if key[:2] == (r.client, r.generation):
                        assert key not in self.received
                        self.received[key] = value.copy()
                del self.requests[r.client]
                self.acks.append(o)
            self.events.append(Event('consume', output.wave, self.now,
                                     self.now+self.ack_delay, consume))


class Numerics(unittest.TestCase):
    def test_ep2_ep4_push_pull_changing_routes_layers_and_slot_reuse(self):
        rng = np.random.default_rng(20260930)
        experts, hidden, inner, topk = 8, 5, 7, 3
        weights = {layer: tuple(rng.normal(0,.15,shape).astype(np.float32) for shape in
                               ((experts,inner,hidden),(experts,inner,hidden),(experts,hidden,inner)))
                   for layer in (0,40)}
        payloads = {}
        for generation in range(1,6):
            for c,rows in enumerate((1,2,3,8)):
                x = rng.normal(size=(rows,hidden)).astype(np.float32)
                ids = np.stack([rng.choice(experts,topk,replace=False) for _ in range(rows)])
                # Skew and an empty-owner case: all routes belong to EP2 owner0.
                if generation == 2: ids[:] = [0,1,2]
                probs = rng.random((rows,topk),dtype=np.float32)
                probs /= probs.sum(axis=1,keepdims=True)
                payloads[c,generation] = x,ids,probs
        returned = {}
        for owners in (2,4):
            for mode in ReturnMode:
                schedulers = [Scheduler(4,8,return_mode=mode) for _ in range(owners)]
                backends = [MatrixOracle(mode,[],owner=o,owners=owners,payloads=payloads,weights=weights)
                            for o in range(owners)]
                loops = [ServerLoop(s,b) for s,b in zip(schedulers,backends)]
                for generation in range(1,6):
                    requests = [req(c,rows,layer=40 if generation%2 else 0,
                                    kind=Kind.MIXED if c==3 else Kind.DECODE,gen=generation)
                                for c,rows in enumerate((1,2,3,8))]
                    for b in backends: b.requests = {r.client:r for r in requests}
                    for tick in range(300):
                        # Deliberately different owner progress: no EP-wave barrier.
                        for o,(loop,b) in enumerate(zip(loops,backends)):
                            if tick%(o+1)==0: loop.tick()
                            b.advance()
                        if all(not b.requests and not s.active and not loop.pending
                               for b,s,loop in zip(backends,schedulers,loops)):
                            break
                    else: self.fail('independent owners failed to drain')
                    for r in requests:
                        x,ids,probs = payloads[r.client,generation]
                        route_values = np.empty((r.tokens,topk,hidden),np.float32)
                        gate,up,down = weights[r.layer]
                        expected = np.zeros_like(x)
                        for token in range(r.tokens):
                            for k,expert in enumerate(ids[token]):
                                key = (r.client,generation,token,k)
                                owner = expert//(experts//owners)
                                self.assertEqual(sum(key in b.received for b in backends),1)
                                route_values[token,k] = backends[owner].received[key]
                                # Token-by-token independent oracle, no grouped layout.
                                g = gate[expert] @ x[token]
                                u = up[expert] @ x[token]
                                y = down[expert] @ ((g/(1+np.exp(-g)))*u)
                                expected[token] += probs[token,k]*y
                        actual = np.zeros_like(x)
                        for k in range(topk): actual += probs[:,k,None]*route_values[:,k]
                        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-7)
                        returned[owners,mode,r.client,generation] = actual
                for s in schedulers:
                    for c in range(4): s.close(c,5)
                    self.assertTrue(s.drained)
            for c in range(4):
                for generation in range(1,6):
                    np.testing.assert_array_equal(returned[owners,ReturnMode.PUSH,c,generation],
                                                  returned[owners,ReturnMode.PULL,c,generation])


if __name__ == '__main__': unittest.main()
