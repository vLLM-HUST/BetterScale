"""Bounded reusable native shared staging, not a second persistent KV cache.

The caller establishes NUMA policy before construction and may register the
single arena with its device runtime. Raw pointers are borrowed capabilities:
only the acquired writer may modify them, and seal waits its DMA completion.
Store calls happen in background threads; no disk/replica ACK is a DMA fence.
A failed completion quarantines storage rather than recycling uncertain bytes.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections import deque
from threading import Condition
import ctypes
import time


@dataclass
class Stage:
    arena: object
    index: int
    generation: int
    size: int
    state: str = "writing"

    @property
    def pointer(self):
        if self.state not in ("writing", "draining", "sealed", "sending"):
            raise RuntimeError("Staging lease no longer owns storage")
        return self.arena.base + self.index * self.arena.slot_bytes

    def seal(self, wait_dma):
        """wait_dma must retire every producer write, not just enqueue it."""
        with self.arena.condition:
            self.arena._check(self, "writing")
            self.state = "draining"
        try:
            wait_dma()
        except BaseException:
            with self.arena.condition:
                self.state = "quarantined"
                self.arena.condition.notify_all()
            raise
        with self.arena.condition:
            self.arena._check(self, "draining")
            self.state = "sealed"

    def replicate(self, key):
        """Return remote-ready Future; retain this region through Store ACK."""
        with self.arena.condition:
            self.arena._check(self, "sealed")
            self.state = "sending"
            return self.arena.executor.submit(self.arena._put, self, key)

    def discard(self):
        # Only a drained immutable buffer may be discarded. An unsealed writer
        # might already have submitted DMA, even if its host call raised.
        with self.arena.condition:
            self.arena._check(self, "sealed")
            self.arena._release(self)


class NativeDramStaging:
    def __init__(self, store, *, slot_bytes, slots, allocator=None,
                 register_device=None, unregister_device=None):
        if type(slots) is not int or slots < 1 or type(slot_bytes) is not int or slot_bytes < 1:
            raise ValueError("positive bounded staging geometry required")
        if bool(register_device) != bool(unregister_device):
            raise ValueError("device registration needs paired teardown")
        if allocator is None:
            from mooncake.store import MooncakeHostMemAllocator
            allocator = MooncakeHostMemAllocator()
        self.store, self.allocator = store, allocator
        self.slot_bytes, self.slots = slot_bytes, slots
        self.total_bytes = slot_bytes * slots
        self.base = allocator.alloc(self.total_bytes)
        if not self.base:
            raise MemoryError("native shared arena allocation failed")
        self.condition = Condition()
        self.available = deque(range(slots))
        self.active = {}
        self.generation = 0
        self.closed = False
        self.closing = False
        self.unregister_device = unregister_device
        store_registered = False
        try:
            # First touch is intentional: the caller's NUMA memory policy owns
            # physical placement. Do this before timed admission / device work.
            ctypes.memset(self.base, 0, self.total_bytes)
            if store.register_buffer(self.base, self.total_bytes) != 0:
                raise RuntimeError("Store arena registration failed")
            store_registered = True
            if register_device:
                register_device(self.base, self.total_bytes)
        except BaseException as error:
            error.quarantined_arena = self
            # A failed device registration may be uncertain. Keep the arena
            # reachable in this object; do not unmap underneath a device.
            if not register_device:
                if store_registered:
                    if store.unregister_buffer(self.base) != 0:
                        raise RuntimeError("Failed registration teardown quarantined")
                allocator.free(self.base)
            raise
        self.executor = ThreadPoolExecutor(max_workers=slots, thread_name_prefix="state-dram-egress")

    def acquire(self, size, *, timeout=0):
        if type(size) is not int or not 0 < size <= self.slot_bytes:
            raise ValueError("object does not fit staging slot")
        if timeout is not None and timeout < 0:
            raise ValueError("nonnegative timeout required")
        with self.condition:
            ready = self.condition.wait_for(lambda: self.available or self.closed or self.closing, timeout)
            if self.closed or self.closing:
                raise RuntimeError("staging arena closed or quarantined during teardown")
            if not ready:
                raise TimeoutError("bounded DRAM staging is full")
            index = self.available.popleft()
            self.generation += 1
            lease = Stage(self, index, self.generation, size)
            self.active[index] = lease
            return lease

    def _check(self, lease, state):
        if self.active.get(lease.index) is not lease or lease.state != state:
            raise RuntimeError("stale or incorrectly phased staging lease")

    def _release(self, lease):
        del self.active[lease.index]
        lease.state = "released"
        self.available.append(lease.index)
        self.condition.notify_all()

    def _put(self, lease, key):
        try:
            start = time.perf_counter()
            rc = self.store.put_from(key, lease.pointer, lease.size)
            if rc != 0:
                raise RuntimeError(f"background State replication failed: {rc}")
            result = dict(bytes=lease.size, seconds=time.perf_counter()-start)
        except BaseException:
            with self.condition:
                lease.state = "quarantined"
                self.condition.notify_all()
            raise
        with self.condition:
            self._check(lease, "sending")
            self._release(lease)
        return result

    def close(self):
        with self.condition:
            if self.closed:
                return
            if self.closing:
                raise RuntimeError("failed/in-progress arena teardown retains storage")
            if self.active:
                raise RuntimeError("drain or quarantine active staging leases before close")
            self.closing = True
            self.condition.notify_all()
        self.executor.shutdown(wait=True)
        # Every producer and Store call has retired. Failed unregister keeps the
        # mapping live; close must not claim that teardown succeeded.
        if self.unregister_device:
            self.unregister_device(self.base)
        if self.store.unregister_buffer(self.base) != 0:
            raise RuntimeError("Store unregister failed; arena retained")
        if self.allocator.free(self.base) != 0:
            raise RuntimeError("arena free failed")
        self.closed = True
