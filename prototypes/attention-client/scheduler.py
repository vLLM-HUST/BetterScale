"""Cooperative layer scheduler, independent of token admission and sampling.

A lane owns its full forward context/IO until retirement. Transport.poll must
be nonblocking; None means not ready, never a zero-valued result. Device-backed
transports must enqueue stream waits before exposing readable result tensors.
"""
from collections import deque
from dataclasses import dataclass
from qwen_layer import attention_half


@dataclass
class Lane:
    identity: object
    model: object
    positions: object
    hidden: object
    context: object
    residual: object = None
    layer: int = 0
    handle: object = None
    pending: object = None


class LayerScheduler:
    def __init__(self, transport, capacity=2):
        if capacity < 1:
            raise ValueError('positive capacity required')
        self.transport = transport
        self.capacity = capacity
        self.lanes = {}
        self.round_robin = deque()
        self.completed = deque()

    def admit(self, lane):
        if lane.identity in self.lanes or len(self.lanes) >= self.capacity:
            raise BufferError('active identity or no free lane')
        self.lanes[lane.identity] = lane
        self.round_robin.append(lane.identity)

    def tick(self):
        """Visit each lane once; no device/host-wide barrier or busy wait."""
        progress = False
        for _ in range(len(self.round_robin)):
            identity = self.round_robin.popleft()
            lane = self.lanes[identity]
            with lane.context():
                if lane.handle is not None:
                    result = self.transport.poll(lane.handle)
                    if result is None:
                        self.round_robin.append(identity)
                        continue
                    lane.hidden = result
                    lane.residual = lane.pending.residual
                    lane.pending = lane.handle = None
                    lane.layer += 1
                    progress = True
                if lane.layer == len(lane.model.layers):
                    output, _ = lane.model.norm(lane.hidden, lane.residual)
                    self.completed.append((identity, output))
                    del self.lanes[identity]
                    continue
                layer = lane.model.layers[lane.layer]
                lane.pending = attention_half(layer, lane.positions, lane.hidden, lane.residual)
                lane.handle = self.transport.submit(identity, lane.layer, layer, lane.pending)
                if lane.handle is None:
                    raise RuntimeError('transport returned no accepted handle')
                progress = True
            self.round_robin.append(identity)
        return progress


class ReferenceTransport:
    """Native full MLP reference, NOT a remote or asynchronous expert service.

    Defer invocation to poll so the scheduler's pause/resume cut is exercised.
    Full MLP preserves native gating/shared semantics; do not add shared twice.
    """
    def __init__(self):
        self.pending = {}
        self.sequence = 0

    def submit(self, identity, layer_id, layer, pending):
        self.sequence += 1
        self.pending[self.sequence] = (layer, pending)
        return self.sequence

    def poll(self, handle):
        layer, pending = self.pending.pop(handle)
        return layer.mlp(pending.normalized)
