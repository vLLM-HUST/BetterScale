"""Reference-counted host State objects, batched under one DMA completion.

A manifest owns a private resident snapshot and shared immutable FA objects.
Objects allocate independently so releasing one really releases its byte charge;
there are no retained slab views hiding memory outside the budget.
"""

from threading import RLock

import torch

from .host_state import HostStateKey, TorchHostStateBackend


class _CopyBackend(TorchHostStateBackend):
    def _copy_lanes(self, lanes, payloads, *, to_host, stream):
        # PageStateStore supplies the stream scope, completion event and error
        # drain for the entire batch. Individual handles are not exposed.
        self._enqueue_copies(lanes, payloads, to_host=to_host)


class PageTransfer:
    def __init__(self, store, key, handles, event, byte_length):
        self.store, self.key = store, key
        self.handles, self.event = handles, event
        self.byte_length = byte_length
        self.finished = False

    def result(self):
        if self.finished:
            return self.key
        if self.event is not None:
            self.event.synchronize()
        with self.store.lock:
            if self.finished:
                return self.key
            for handle in self.handles:
                handle.result()
            self.store.pending.remove(self.key)
            self.finished = True
        return self.key


class PageStateStore:
    def __init__(self, *, memory_budget_bytes):
        self.backend = _CopyBackend(memory_budget_bytes=memory_budget_bytes)
        self.manifests = {}
        self.references = {}
        self.pending = set()
        self.lock = RLock()
        self.quarantined = []

    @property
    def committed_bytes(self):
        return self.backend.committed_bytes

    @staticmethod
    def object_key(identity):
        return HostStateKey(identity, 1)

    def release(self, key):
        with self.lock:
            if key in self.pending:
                raise ValueError("host manifest has pending DMA")
            for identity in self.manifests.pop(key):
                self.references[identity] -= 1
                if self.references[identity] == 0:
                    del self.references[identity]
                    self.backend.release(self.object_key(identity))

    def transfer(self, key, states, objects, *, store, stream):
        """objects is an ordered identity -> State selection mapping.

        Store references every object but copies only new ones. Restore lists
        only the resident plus missing FA objects, mapped to new destinations.
        """
        states = tuple(states)
        devices = {state.tensor.device.type for _, state in states}
        if len(devices) != 1:
            raise ValueError("one State transfer requires one device kind")
        kind = next(iter(devices))
        if kind not in ("cpu", "npu", "cuda"):
            raise ValueError("unsupported State device")
        device = getattr(torch, kind) if kind != "cpu" else None
        if device is not None and not isinstance(stream, device.Stream):
            raise ValueError("copy stream required")
        with self.lock:
            if key in self.pending or (store and key in self.manifests):
                raise ValueError("host manifest already exists or is in flight")
            if not store and not set(objects) <= set(self.manifests[key]):
                raise ValueError("restore asks for objects outside its manifest")
            existing = self.backend.resident_keys
            new = [
                identity
                for identity in objects
                if self.object_key(identity) not in existing
            ]
            if store:
                # Do not share objects whose producer has not completed.
                if any(identity in self.references for identity in new):
                    raise ValueError("shared host object is still in flight")
                requested = sum(
                    sum(
                        s.logical_block_bytes * len(ids)
                        for _, s, ids in self.backend._select_lanes(
                            states, objects[identity]
                        )
                    )
                    for identity in new
                )
                if (
                    self.backend._committed_bytes
                    + self.backend._reserved_bytes
                    + requested
                    > self.backend.memory_budget_bytes
                ):
                    raise ValueError("host State capacity exhausted before batch")
                self.manifests[key] = tuple(objects)
                for identity in objects:
                    self.references[identity] = self.references.get(identity, 0) + 1
            self.pending.add(key)
            handles = []
            event = None
            try:
                from contextlib import nullcontext

                with device.stream(stream) if device is not None else nullcontext():
                    for identity, selection in objects.items():
                        if store and identity not in new:
                            continue
                        method = self.backend.offload if store else self.backend.restore
                        handles.append(
                            method(
                                states,
                                self.object_key(identity),
                                selection,
                                stream=stream,
                            )
                        )
                    if device is not None:
                        event = device.Event()
                        event.record(stream)
            except BaseException:
                # Any uncertain partial DMA keeps all ownership until teardown.
                # Never let rollback race with an already submitted copy.
                self.quarantined.append((key, states, objects, handles))
                if device is not None:
                    stream.synchronize()
                raise
            return PageTransfer(
                self, key, handles, event, sum(h.byte_length for h in handles)
            )
