import ctypes
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import pytest
from native_dram_staging import NativeDramStaging


class Allocator:
    def alloc(self, size):
        self.buffer = ctypes.create_string_buffer(size)
        return ctypes.addressof(self.buffer)
    def free(self, pointer):
        assert pointer == ctypes.addressof(self.buffer)
        self.buffer = None
        return 0


class Store:
    def __init__(self):
        self.entered, self.release = Event(), Event()
        self.values = {}
        self.fail = False
    def register_buffer(self, pointer, size): return 0
    def unregister_buffer(self, pointer): return 0
    def put_from(self, key, pointer, size):
        self.entered.set()
        assert self.release.wait(3)
        if self.fail: return -1
        self.values[key] = ctypes.string_at(pointer, size)
        return 0


def fixture():
    store, allocator = Store(), Allocator()
    return store, allocator, NativeDramStaging(store,slot_bytes=16,slots=1,allocator=allocator)


def test_dma_local_ready_precedes_egress_and_slot_cannot_be_reused():
    store, allocator, arena = fixture()
    lease = arena.acquire(16)
    pointer = lease.pointer
    ctypes.memset(pointer, 91, 16)
    dma = Event()
    with ThreadPoolExecutor(1) as worker:
        sealing = worker.submit(lease.seal, lambda: dma.wait(3) or (_ for _ in ()).throw(TimeoutError()))
        with pytest.raises(RuntimeError): lease.replicate("early")
        dma.set();sealing.result(3)
    assert lease.state == "sealed" and not store.entered.is_set()
    result = lease.replicate("A")
    assert store.entered.wait(3) and not result.done()
    with pytest.raises(TimeoutError): arena.acquire(1)
    with pytest.raises(RuntimeError): arena.close()
    store.release.set()
    assert result.result(3)["bytes"] == 16
    assert store.values["A"] == bytes([91])*16
    newer = arena.acquire(8)
    assert newer.pointer == pointer and newer.generation > lease.generation
    with pytest.raises(RuntimeError): lease.discard()
    newer.seal(lambda: None);newer.discard()
    arena.close()
    assert allocator.buffer is None


@pytest.mark.parametrize("failure", ["dma", "replica"])
def test_failure_quarantines_not_reuses_or_unmaps(failure):
    store, allocator, arena = fixture()
    lease = arena.acquire(8)
    if failure == "dma":
        with pytest.raises(IOError):
            lease.seal(lambda: (_ for _ in ()).throw(IOError("uncertain DMA")))
    else:
        lease.seal(lambda: None)
        store.fail=True;store.release.set()
        with pytest.raises(RuntimeError): lease.replicate("A").result(3)
    assert lease.state == "quarantined"
    with pytest.raises(RuntimeError,match="quarantined"): arena.acquire(8)
    with pytest.raises(RuntimeError): arena.close()
    assert allocator.buffer is not None
    # Test fake owns no actual asynchronous DMA. Shut down threads without
    # pretending a real quarantined mapping is safe to reclaim.
    arena.executor.shutdown(wait=True)


def test_failed_unregister_cannot_be_reported_as_closed_on_retry():
    store, allocator, arena = fixture()
    store.unregister_buffer = lambda pointer: -1
    with pytest.raises(RuntimeError, match="retained"): arena.close()
    assert not arena.closed and allocator.buffer is not None
    with pytest.raises(RuntimeError, match="teardown"): arena.close()
    with pytest.raises(RuntimeError): arena.acquire(1)
