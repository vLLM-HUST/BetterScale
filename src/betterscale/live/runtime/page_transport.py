"""Bounded object transport for incremental State, outside the model RPC path.

The sink owns shared host storage. A worker stages one object at a time, retaining
it through both DMA and sink acknowledgement. Keys must include model/layout,
TP head and session incarnation; the scheduler owns page versions and TP quorum.
This module does not grant execution ownership or infer a reusable frontier.
"""
from dataclasses import dataclass
import hashlib
import json
import struct

import torch

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
        data = value.tensor.contiguous().view(torch.uint8).numpy().tobytes()
        lanes.append(dict(name=name, dtype=str(value.dtype), shape=list(value.block_shape),
                          span=value.physical_blocks_per_logical_block,
                          count=value.logical_block_count, offset=offset, size=len(data)))
        chunks.append(data)
        offset += len(data)
    header = json.dumps(dict(schema=1, lanes=lanes), separators=(",", ":")).encode()
    if len(header) > MAX_HEADER_BYTES or offset > MAX_OBJECT_BYTES:
        raise ValueError("State object exceeds bounded wire envelope")
    return struct.pack("<I", len(header)) + header + b"".join(chunks)


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
            view = torch.frombuffer(bytearray(data[begin:begin+lane["size"]]),
                                    dtype=DTYPES[lane["dtype"]])
            shape = (len(blocks)*state.physical_blocks_per_logical_block, *state.block_shape)
            host = torch.empty(shape, dtype=state.storage_dtype,
                               pin_memory=state.tensor.device.type != "cpu")
            host.copy_(view.reshape(shape))
            payloads[name] = _LanePayload(name, state.storage_dtype, state.block_shape,
                state.physical_blocks_per_logical_block, len(blocks), host)
        backend._snapshots[key] = _HostSnapshot(key, payloads, position)
        backend._committed_bytes += position


@dataclass
class ObjectTransfer:
    run: object
    byte_length: int = 0
    finished: bool = False

    def result(self):
        if not self.finished:
            self.byte_length = self.run()
            self.finished = True


class ObjectStateTransport:
    """PageStateStore DMA + immutable shared sink; no per-worker durable mirror.

    sink.ensure(identity) confirms/replicates an existing complete object, put
    acknowledges all required replicas, get validates stored content. They are
    blocking host calls, performed only by the cache completion thread.
    """
    def __init__(self, sink, namespace, *, staging_bytes=MAX_OBJECT_BYTES, verify=False):
        if not namespace:
            raise ValueError("State transport namespace is required")
        self.sink, self.namespace = sink, namespace
        self.verify = verify
        self.staging = PageStateStore(memory_budget_bytes=staging_bytes)
        # PageStateStore._CopyBackend deliberately has no per-object event: its
        # batch wrapper supplies that event. Standalone restore/audit must use
        # the ordinary backend, whose handle owns a real completion event.
        self.restore_backend = TorchHostStateBackend(memory_budget_bytes=staging_bytes)
        self.quarantined = []

    def object_id(self, identity):
        return hashlib.sha256((self.namespace + "\0" + identity).encode()).hexdigest()

    def release(self, key):
        # Shared sink LRU owns cache eviction, not an individual worker manifest.
        # Removing scheduler metadata cannot delete a page used by another owner.
        return None

    def transfer(self, key, states, objects, *, store, stream):
        states = tuple(states)
        objects = dict(objects)
        @torch.inference_mode()
        def run():
            device_type = states[0][1].tensor.device.type
            device = getattr(torch, device_type) if device_type != "cpu" else None
            if device is not None:
                device.set_device(states[0][1].tensor.device)
            moved = 0
            for identity, selection in objects.items():
                remote = self.object_id(identity)
                if store and self.sink.ensure(remote):
                    continue
                local = HostStateKey(remote, 1)
                try:
                    if store:
                        transfer = self.staging.transfer(local, states,
                            {identity: selection}, store=True, stream=stream)
                        transfer.result()
                        objkey = self.staging.object_key(identity)
                        snapshot = self.staging.backend._snapshots[objkey]
                        self.sink.put(remote, encode_snapshot(snapshot))
                        moved += transfer.byte_length
                        self.staging.release(local)
                    else:
                        data = self.sink.get(remote)
                        decode_snapshot(data, self.restore_backend, local, states, selection)
                        transfer = self.restore_backend.restore(states, local, selection, stream=stream)
                        transfer.result()
                        moved += transfer.byte_length
                        self.restore_backend.release(local)
                        if self.verify:
                            audit = self.restore_backend.offload(states, local, selection, stream=stream)
                            audit.result()
                            actual = encode_snapshot(self.restore_backend._snapshots[local])
                            if actual != data:
                                raise RuntimeError("Post-H2D State object byte mismatch")
                            self.restore_backend.release(local)
                except BaseException:
                    # Leave uncertain storage reachable through this transport.
                    # The caller fails its TP receipt and retains device pins.
                    self.quarantined.append((key, identity, states, selection, stream))
                    raise
            return moved
        return ObjectTransfer(run)
