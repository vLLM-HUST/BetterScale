"""Bounded object transport for incremental State, outside the model RPC path.

The sink owns shared host storage. A worker stages one object at a time, retaining
it through both DMA and sink acknowledgement. Keys must include model/layout,
TP head and session incarnation; the scheduler owns page versions and TP quorum.
This module does not grant execution ownership or infer a reusable frontier.
"""
from dataclasses import dataclass, field
from collections import deque
from threading import Condition, Lock
import hashlib
import json
import struct
import time
from collections import defaultdict

import torch
import numpy as np

from .host_state import HostStateKey, TorchHostStateBackend
from .page_state import PageStateStore


MAX_OBJECT_BYTES = 128 << 20
MAX_HEADER_BYTES = 1 << 20
DTYPES = {str(t): t for t in (torch.bfloat16, torch.float32, torch.int64,
                             torch.int32, torch.bool, torch.float16, torch.uint8)}


def encode_snapshot(snapshot):
    """Portable lane schema, never pickle or physical device addresses."""
    lanes, chunks, offset = [], [], 0
    for name, value in sorted(snapshot.payloads.items()):
        data = memoryview(value.tensor.contiguous().view(torch.uint8).numpy().reshape(-1))
        lanes.append(dict(name=name, dtype=str(value.dtype), shape=list(value.block_shape),
                          span=value.physical_blocks_per_logical_block,
                          count=value.logical_block_count, offset=offset, size=len(data)))
        chunks.append(data)
        offset += len(data)
    header = json.dumps(dict(schema=1, lanes=lanes), separators=(",", ":")).encode()
    if len(header) > MAX_HEADER_BYTES or offset > MAX_OBJECT_BYTES:
        raise ValueError("State object exceeds bounded wire envelope")
    return b"".join((struct.pack("<I", len(header)), header, *chunks))


def decode_snapshot(data, backend, key, states, selection):
    """Validate destination-derived geometry before allocating/publishing."""
    from .host_state import _LanePayload, _HostSnapshot

    if not 4 < len(data) <= MAX_OBJECT_BYTES + MAX_HEADER_BYTES + 4:
        raise ValueError("Invalid State object length")
    header_size = struct.unpack_from("<I", data)[0]
    if not 0 < header_size <= min(MAX_HEADER_BYTES, len(data)-4):
        raise ValueError("Invalid State object header")
    header = json.loads(data[4:4+header_size])
    if set(header) != {"schema", "lanes"} or header["schema"] != 1:
        raise ValueError("Unknown State object schema")
    lanes = backend._select_lanes(states, selection)
    expected = {name: (state, blocks) for name, state, blocks in lanes}
    if len(header["lanes"]) != len(expected):
        raise ValueError("State lane census mismatch")
    position, checked = 0, {}
    for lane in header["lanes"]:
        if set(lane) != {"name", "dtype", "shape", "span", "count", "offset", "size"}:
            raise ValueError("Malformed State lane")
        name = lane["name"]
        if name not in expected or name in checked:
            raise ValueError("Foreign or duplicate State lane")
        state, blocks = expected[name]
        size = state.logical_block_bytes * len(blocks)
        if (lane["dtype"] != str(state.storage_dtype)
                or lane["shape"] != list(state.block_shape)
                or lane["span"] != state.physical_blocks_per_logical_block
                or lane["count"] != len(blocks) or lane["offset"] != position
                or lane["size"] != size):
            raise ValueError("State lane geometry mismatch")
        checked[name] = lane
        position += size
    if position != len(data)-4-header_size:
        raise ValueError("State payload length mismatch")
    with backend._lock:
        if key in backend._snapshots or key in backend._inflight:
            raise ValueError("State staging key is already owned")
        if backend.committed_bytes + backend._reserved_bytes + position > backend.memory_budget_bytes:
            raise ValueError("State staging budget exceeded")
        payloads = {}
        for name, lane in checked.items():
            state, blocks = expected[name]
            begin = 4 + header_size + lane["offset"]
            shape = (len(blocks)*state.physical_blocks_per_logical_block, *state.block_shape)
            host = torch.empty(shape, dtype=state.storage_dtype,
                               pin_memory=state.tensor.device.type != "cpu")
            # One synchronous CPU copy into the owned pinned destination.
            # The immutable wire buffer stays alive until this returns; avoid
            # bytes slice + bytearray + Tensor copy (three full payload passes).
            np.copyto(host.view(torch.uint8).numpy().reshape(-1),
                      np.frombuffer(data, dtype=np.uint8, count=lane["size"], offset=begin))

            payloads[name] = _LanePayload(name, state.storage_dtype, state.block_shape,
                state.physical_blocks_per_logical_block, len(blocks), host)
        backend._snapshots[key] = _HostSnapshot(key, payloads, position)
        backend._committed_bytes += position


@dataclass
class ObjectTransfer:
    run: object
    byte_length: int = 0
    finished: bool = False
    lock: object = field(default_factory=Lock, repr=False)
    phase_seconds: dict = field(default_factory=dict)

    def result(self):
        with self.lock:
            if not self.finished:
                self.byte_length = self.run()
                self.finished = True


class ObjectStateTransport:
    """PageStateStore DMA + immutable shared sink; no per-worker durable mirror.

    sink.ensure(identity) confirms/replicates an existing complete object, put
    acknowledges all required replicas, get validates stored content. They are
    blocking host calls, performed only by the cache completion thread.
    """
    def __init__(self, sink, namespace, *, staging_bytes=MAX_OBJECT_BYTES, verify=False,
                 max_transfers=1, enqueue_copies=None):
        if not namespace:
            raise ValueError("State transport namespace is required")
        if type(max_transfers) is not int or max_transfers < 1:
            raise ValueError("positive State transfer concurrency required")
        self.sink, self.namespace = sink, namespace
        self.verify = verify
        self.staging = PageStateStore(memory_budget_bytes=staging_bytes, enqueue_copies=enqueue_copies)
        # PageStateStore._CopyBackend deliberately has no per-object event: its
        # batch wrapper supplies that event. Standalone restore/audit must use
        # the ordinary backend, whose handle owns a real completion event.
        self.restore_backend = TorchHostStateBackend(memory_budget_bytes=staging_bytes, enqueue_copies=enqueue_copies)
        self.quarantined = []
        self.max_transfers = max_transfers
        self.staging_budget_bytes = staging_bytes * max_transfers
        self.condition = Condition()
        self.failure = None
        self.active_transfers = self.peak_transfers = 0
        self.available = deque([(self.staging, self.restore_backend)])
        self.available.extend((
            PageStateStore(memory_budget_bytes=staging_bytes, enqueue_copies=enqueue_copies),
            TorchHostStateBackend(memory_budget_bytes=staging_bytes, enqueue_copies=enqueue_copies),
        ) for _ in range(max_transfers - 1))

    def object_id(self, identity):
        return hashlib.sha256((self.namespace + "\0" + identity).encode()).hexdigest()

    def release(self, key):
        # Shared sink LRU owns cache eviction, not an individual worker manifest.
        # Removing scheduler metadata cannot delete a page used by another owner.
        return None

    def transfer(self, key, states, objects, *, store, stream):
        states = tuple(states)
        objects = dict(objects)
        phases = defaultdict(float)
        created = time.perf_counter()
        def timed(name, function, *args, **kwargs):
            begin = time.perf_counter()
            try:return function(*args, **kwargs)
            finally:phases[name] += time.perf_counter() - begin
        @torch.inference_mode()
        def copy(staging, restore_backend):
            device_type = states[0][1].tensor.device.type
            device = getattr(torch, device_type) if device_type != "cpu" else None
            if device is not None:
                device.set_device(states[0][1].tensor.device)
            moved = 0
            for identity, selection in objects.items():
                remote = self.object_id(identity)
                if store and timed("ensure", self.sink.ensure, remote):
                    continue
                local = HostStateKey(remote, 1)
                try:
                    if store:
                        transfer = timed("d2h_submit", staging.transfer, local, states,
                            {identity: selection}, store=True, stream=stream)
                        timed("d2h_wait", transfer.result)
                        objkey = staging.object_key(identity)
                        snapshot = staging.backend._snapshots[objkey]
                        data = timed("encode", encode_snapshot, snapshot)
                        timed("put", self.sink.put, remote, data)
                        moved += transfer.byte_length
                        staging.release(local)
                    else:
                        data = timed("get", self.sink.get, remote)
                        timed("decode", decode_snapshot, data, restore_backend, local, states, selection)
                        transfer = timed("h2d_submit", restore_backend.restore, states, local, selection, stream=stream)
                        timed("h2d_wait", transfer.result)
                        moved += transfer.byte_length
                        restore_backend.release(local)
                        if self.verify:
                            audit = restore_backend.offload(states, local, selection, stream=stream)
                            audit.result()
                            actual = encode_snapshot(restore_backend._snapshots[local])
                            if actual != data:
                                raise RuntimeError("Post-H2D State object byte mismatch")
                            restore_backend.release(local)
                except BaseException:
                    # Leave uncertain storage reachable through this transport.
                    # The caller fails its TP receipt and retains device pins.
                    self.quarantined.append((key, identity, states, selection, stream, staging, restore_backend))
                    raise
            return moved
        def run():
            phases["completion_queue"] = time.perf_counter() - created
            waiting = time.perf_counter()
            with self.condition:
                self.condition.wait_for(lambda: self.available or self.failure)
                if self.failure:
                    raise RuntimeError("State transport quarantined: " + self.failure)
                phases["staging_queue"] = time.perf_counter() - waiting
                lane = self.available.popleft()
                self.active_transfers += 1
                self.peak_transfers = max(self.peak_transfers, self.active_transfers)
            completed = False
            try:
                result = copy(*lane)
                completed = True
                return result
            except BaseException as error:
                with self.condition:
                    self.failure = repr(error)
                raise
            finally:
                with self.condition:
                    self.active_transfers -= 1
                    if completed:
                        self.available.append(lane)
                    self.condition.notify_all()
        return ObjectTransfer(run, phase_seconds=phases)
