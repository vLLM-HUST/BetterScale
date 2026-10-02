"""Rank-owned immutable State objects; no shared mapping or object service.

The allocator returns an owned buffer with data_ptr() and numel(). Production
uses NUMA-local torch pinned tensors; CPU tests inject ordinary CPU tensors.
Checkpoint refs and active readers independently prevent reclamation. A source
DMA completion seals its final DRAM home, rather than copying into another cache.
"""
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock


@dataclass
class Object:
    key: str
    buffer: object
    size: int
    generation: int
    state: str = "writing"
    readers: int = 0
    digest: str | None = None


class Writer:
    def __init__(self, pool, item):
        self.pool, self.item = pool, item
        self._size = item.size

    @property
    def pointer(self):
        with self.pool.lock:
            if self.item is None:raise RuntimeError("State writer sealed")
            self.pool._owns(self.item)
            if self.item.state not in ("writing", "draining"):
                raise RuntimeError("State writer no longer owns mutable memory")
            return self.item.buffer.data_ptr()

    @property
    def size(self):
        return self._size

    @property
    def state(self):
        return self.item.state if self.item is not None else "sealed"

    def seal(self, wait_dma):
        with self.pool.lock:
            if self.item is None:raise RuntimeError("State writer sealed")
            self.pool._owns(self.item)
            if self.item.state != "writing":
                raise RuntimeError("State writer is not writable")
            self.item.state = "draining"
        try:
            wait_dma()
        except BaseException:
            with self.pool.lock:
                self.item.state = "quarantined"
            raise
        with self.pool.lock:
            self.pool._owns(self.item)
            self.item.state = "sealed"
            self.pool.objects.move_to_end(self.item.key)
        self.item = None
        return self._size


class Reader:
    def __init__(self, pool, item):
        self.pool, self.item, self.closed = pool, item, False
        self._size = item.size

    @property
    def pointer(self):
        with self.pool.lock:
            if self.closed:
                raise RuntimeError("State reader is closed")
            self.pool._owns(self.item)
            return self.item.buffer.data_ptr()

    @property
    def size(self):
        return self._size

    @property
    def state(self):
        return "closed" if self.closed else "sealed"

    def close(self):
        with self.pool.lock:
            if not self.closed:
                self.pool._owns(self.item)
                self.item.readers -= 1
                self.closed = True
                self.item = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class RankStatePool:
    def __init__(self, owner, byte_budget, allocate):
        if not isinstance(owner, tuple) or len(owner) != 3:
            raise ValueError("owner must be (P-or-D, attention-group, TP-rank)")
        side, group, rank = owner
        if side not in ("P", "D") or type(group) is not int or group < 0 or type(rank) is not int or rank not in (0, 1):
            raise ValueError("invalid TP2 State owner")
        if type(byte_budget) is not int or byte_budget < 1:
            raise ValueError("positive rank-local DRAM budget required")
        self.owner, self.budget, self.allocate = owner, byte_budget, allocate
        self.lock = RLock()
        self.objects, self.groups, self.references = OrderedDict(), {}, {}
        self.bytes = self.generation = 0
        self.closed = False

    def _healthy(self):
        if self.closed:
            raise RuntimeError("State pool closed")

    def _owns(self, item):
        self._healthy()
        if self.objects.get(item.key) is not item:
            raise RuntimeError("stale State object generation")

    def retain(self, checkpoint, keys):
        keys = tuple(keys)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate manifest object")
        with self.lock:
            self._healthy()
            if checkpoint in self.groups:
                if self.groups[checkpoint] != frozenset(keys):
                    raise ValueError("checkpoint manifest changed")
            else:
                self.groups[checkpoint] = frozenset(keys)
                for key in keys:
                    self.references[key] = self.references.get(key, 0) + 1
            return tuple(key in self.objects and self.objects[key].state == "sealed" for key in keys)

    def drop(self, checkpoint):
        with self.lock:
            self._healthy()
            for key in self.groups.pop(checkpoint,()):
                self.references[key] -= 1
                if not self.references[key]:
                    del self.references[key]

    def reserve(self, key, size):
        if type(size) is not int or not 0 < size <= self.budget:
            raise ValueError("State object exceeds rank budget")
        with self.lock:
            self._healthy()
            if key in self.objects:
                raise RuntimeError("State object already present or in flight")
            if key not in self.references:
                raise ValueError("reserve requires a checkpoint reference")
            candidates = [(item.key, item.size) for item in self.objects.values()
                          if item.state == "sealed" and not item.readers
                          and not self.references.get(item.key)]
            reclaim = max(0, self.bytes + size - self.budget)
            if sum(size for _, size in candidates) < reclaim:
                raise MemoryError("rank-local State capacity pinned or in flight")
            for old_key, old_size in candidates:
                if self.bytes + size <= self.budget:
                    break
                del self.objects[old_key]
                self.bytes -= old_size
            buffer = self.allocate(size)
            if buffer.numel() != size or buffer.element_size() != 1:
                raise ValueError("State allocator returned wrong byte capacity")
            self.generation += 1
            item = Object(key, buffer, size, self.generation)
            self.objects[key] = item
            self.bytes += size
            return Writer(self, item)

    def read(self, key):
        with self.lock:
            self._healthy()
            item = self.objects[key]
            if item.state != "sealed":
                raise RuntimeError("State object not locally committed")
            item.readers += 1
            self.objects.move_to_end(key)
            return Reader(self, item)

    def complete(self, checkpoint):
        with self.lock:
            self._healthy()
            return all(key in self.objects and self.objects[key].state == "sealed"
                       for key in self.groups[checkpoint])

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.groups or any(item.readers or item.state != "sealed" for item in self.objects.values()):
                raise RuntimeError("State references, DMA or quarantined memory still live")
            self.objects.clear()
            self.bytes = 0
            self.closed = True
