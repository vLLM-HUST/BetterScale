"""Startup-reserved rank-private pinned memory, never a pageable cache tier.

CANN VMM gives explicit NUMA placement and known-fast DMA addresses. The current
driver still requires physical high-order pages: reservation may fail at startup.
Suballocations reuse that fixed reservation; there is no driver allocation in a
request. RankStatePool's existing writers/readers/checkpoint refs own each buffer.
"""
import bisect
import ctypes as C
from threading import RLock
import weakref


class Location(C.Structure):
    _fields_ = [("id", C.c_uint32), ("type", C.c_int)]


class PhysicalProperties(C.Structure):
    _fields_ = [("handle_type", C.c_int), ("allocation_type", C.c_int),
                ("memory_attribute", C.c_int), ("location", Location),
                ("reserved", C.c_uint64)]


class PointerAttributes(C.Structure):
    _fields_ = [("location", Location), ("page_size", C.c_uint32),
                ("reserved", C.c_uint32 * 4)]


class NativeRegion:
    """One private CANN physical allocation and its CPU-accessible mapping."""
    def __init__(self, size, node):
        self.size, self.node, self.state = size, node, "creating"
        self.handle, self.address = C.c_void_p(), C.c_void_p()
        self.api = C.CDLL("libascendcl.so")
        signatures = {
            "aclrtMemGetAllocationGranularity": [C.POINTER(PhysicalProperties), C.c_int, C.POINTER(C.c_size_t)],
            "aclrtMallocPhysical": [C.POINTER(C.c_void_p), C.c_size_t, C.POINTER(PhysicalProperties), C.c_uint64],
            "aclrtReserveMemAddress": [C.POINTER(C.c_void_p), C.c_size_t, C.c_size_t, C.c_void_p, C.c_uint64],
            "aclrtMapMem": [C.c_void_p, C.c_size_t, C.c_size_t, C.c_void_p, C.c_uint64],
            "aclrtPointerGetAttributes": [C.c_void_p, C.POINTER(PointerAttributes)],
        }
        for name in ("aclrtUnmapMem", "aclrtFreePhysical", "aclrtReleaseMemAddress"):
            signatures[name] = [C.c_void_p]
        for name, args in signatures.items():
            getattr(self.api, name).argtypes = args
            getattr(self.api, name).restype = C.c_int
        # PINNED, DDR_NORMAL, HOST_NUMA. NORMAL is not a promise of 4KiB
        # physical backing on this driver; the capacity probes disprove that.
        prop = PhysicalProperties(0, 0, 3, Location(node, 4), 0)
        granularity = C.c_size_t()
        self._call("aclrtMemGetAllocationGranularity", C.byref(prop), 0, C.byref(granularity))
        if not granularity.value or size % granularity.value:
            raise ValueError("arena region must match native allocation granularity")
        try:
            self._call("aclrtMallocPhysical", C.byref(self.handle), size, C.byref(prop), 0)
            self._call("aclrtReserveMemAddress", C.byref(self.address), size, 0, None, 0)
            self._call("aclrtMapMem", self.address, size, 0, self.handle, 0)
            attrs = PointerAttributes()
            self._call("aclrtPointerGetAttributes", self.address, C.byref(attrs))
            if attrs.location.type != 0:
                raise RuntimeError("private arena mapping is not host memory")
            self.pointer = self.address.value
            self.state = "ready"
        except BaseException as error:
            # No guessed teardown after partial native setup. The failed worker
            # incarnation owns these handles until process exit.
            self.state = "quarantined"
            error.quarantined_region = self
            raise

    def _call(self, name, *args):
        result = getattr(self.api, name)(*args)
        if result:
            raise RuntimeError(f"{name} failed: {result}")

    def close(self):
        if self.state == "closed":
            return
        if self.state != "ready":
            raise RuntimeError("private native mapping is quarantined")
        self.state = "closing"
        try:
            self._call("aclrtUnmapMem", self.address)
            self._call("aclrtFreePhysical", self.handle)
            self._call("aclrtReleaseMemAddress", self.address)
        except BaseException:
            self.state = "quarantined"
            raise
        self.state = "closed"


class Buffer:
    __slots__ = ("arena", "pointer", "size", "finalizer", "__weakref__")

    def __init__(self, arena, token, pointer, size):
        self.arena, self.pointer, self.size = arena, pointer, size
        self.finalizer = weakref.finalize(self, arena._recycle, token)
        self.finalizer.atexit = False

    def data_ptr(self):
        if self.arena.closed:
            raise RuntimeError("private arena closed")
        return self.pointer

    def numel(self):
        return self.size

    def element_size(self):
        return 1


class RankPinnedArena:
    """Variable-size allocations over an actually reserved, bounded rank pool."""
    def __init__(self, byte_budget, node, *, region_factory=NativeRegion,
                 region_bytes=64 << 30):
        if (type(byte_budget) is not int or byte_budget <= 0
                or type(node) is not int or node < 0
                or type(region_bytes) is not int or region_bytes <= 0):
            raise ValueError("positive private arena geometry and NUMA node required")
        self.budget, self.node = byte_budget, node
        self.lock = RLock()
        self.regions, self.free, self.live = [], [], {}
        self.sequence = self.used = self.peak = 0
        self.closed = self.closing = False
        try:
            remaining = byte_budget
            while remaining:
                size = min(region_bytes, remaining)
                region = region_factory(size, node)
                if region.size != size or not region.pointer:
                    raise ValueError("native region returned wrong capacity")
                self.free.append((len(self.regions), 0, size))
                self.regions.append(region)
                remaining -= size
        except BaseException as error:
            # Setup is not admitted to service. Keep even successful earlier
            # mappings owned by this failed incarnation, never auto-free by GC.
            self.closing = True
            error.quarantined_arena = self
            raise

    def __call__(self, size):
        if type(size) is not int or size <= 0:
            raise ValueError("positive byte allocation required")
        aligned = (size + 63) // 64 * 64
        with self.lock:
            if self.closed or self.closing:
                raise RuntimeError("private arena closed or quarantined")
            candidates = [(length, i) for i, (_, _, length) in enumerate(self.free)
                          if length >= aligned]
            if not candidates:
                raise MemoryError("reserved rank arena full or fragmented")
            _, index = min(candidates)
            region, offset, length = self.free.pop(index)
            if length > aligned:
                self.free.insert(index, (region, offset + aligned, length - aligned))
            self.sequence += 1
            token = self.sequence
            self.live[token] = (region, offset, aligned)
            self.used += aligned
            self.peak = max(self.peak, self.used)
            return Buffer(self, token, self.regions[region].pointer + offset, size)

    def _recycle(self, token):
        with self.lock:
            region, offset, length = self.live.pop(token)
            self.used -= length
            index = bisect.bisect_left(self.free, (region, offset, length))
            if index and self.free[index - 1][0] == region:
                _, before, before_size = self.free[index - 1]
                if before + before_size == offset:
                    index -= 1
                    self.free.pop(index)
                    offset, length = before, length + before_size
            if index < len(self.free):
                after_region, after, after_size = self.free[index]
                if after_region == region and offset + length == after:
                    self.free.pop(index)
                    length += after_size
            self.free.insert(index, (region, offset, length))

    def stats(self):
        with self.lock:
            return dict(backend="rank-private-vmm-arena", numa_node=self.node,
                        reserved_bytes=self.budget, used_bytes=self.used,
                        peak_used_bytes=self.peak, live_buffers=len(self.live),
                        largest_free_bytes=max((n for _, _, n in self.free), default=0))

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.closing:
                raise RuntimeError("private arena teardown is quarantined")
            if self.live:
                raise RuntimeError("private arena still has live DMA/cache buffers")
            self.closing = True
            for region in reversed(self.regions):
                region.close()
            self.closed = True
