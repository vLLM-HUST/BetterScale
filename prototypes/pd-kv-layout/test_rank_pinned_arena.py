import ctypes
import gc
from concurrent.futures import ThreadPoolExecutor
import pytest
from rank_pinned_arena import RankPinnedArena
from rank_state_pool import RankStatePool


class Region:
    def __init__(self, size, node):
        self.size, self.node = size, node
        self.buffer = ctypes.create_string_buffer(size)
        self.pointer = ctypes.addressof(self.buffer)
        self.closed = False

    def close(self):
        self.closed = True
        self.buffer = None


def arena(size=1024, region=512):
    return RankPinnedArena(size, 5, region_factory=Region, region_bytes=region)


def test_exact_byte_interface_alignment_reuse_and_coalescing():
    a = arena(512)
    x, y, z = [a(n) for n in (65, 100, 129)]
    assert [b.numel() for b in (x, y, z)] == [65, 100, 129]
    assert all(b.element_size() == 1 for b in (x, y, z))
    assert y.data_ptr() - x.data_ptr() == 128
    with pytest.raises(MemoryError):
        a(65)
    middle = y.data_ptr()
    del y
    reused = a(100)
    assert reused.data_ptr() == middle
    del x, reused, z
    gc.collect()
    assert a.stats()["largest_free_bytes"] == 512
    whole = a(512)
    assert a.stats()["used_bytes"] == 512
    with pytest.raises(RuntimeError, match="live"):
        a.close()
    del whole
    a.close()
    with pytest.raises(RuntimeError, match="closed"):
        a(1)


def test_regions_do_not_coalesce_across_different_native_addresses():
    a = arena()
    x, y = a(512), a(512)
    del x, y
    assert a.stats()["reserved_bytes"] == 1024
    assert a.stats()["largest_free_bytes"] == 512
    with pytest.raises(MemoryError):
        a(513)
    a.close()
    assert all(r.closed for r in a.regions)


def test_reader_and_checkpoint_refs_prevent_arena_reuse():
    a = arena(512)
    pool = RankStatePool(("D", 0, 0), 512, a)
    pool.retain("old", ["page"])
    writer = pool.reserve("page", 512)
    pointer = writer.pointer
    ctypes.memset(pointer, 73, 512)
    writer.seal(lambda: None)
    reader = pool.read("page")
    pool.drop("old")
    assert a.stats()["live_buffers"] == 1
    with pytest.raises(MemoryError):
        a(512)
    assert ctypes.string_at(reader.pointer, 512) == bytes([73]) * 512
    reader.close()
    assert a.stats()["live_buffers"] == 0
    pool.close()
    a.close()


def test_uncertain_dma_keeps_physical_region_owned():
    a = arena(512)
    pool = RankStatePool(("D", 0, 0), 512, a)
    pool.retain("cp", ["page"])
    writer = pool.reserve("page", 512)
    def failed():
        raise IOError("uncertain DMA")
    with pytest.raises(IOError):
        writer.seal(failed)
    pool.drop("cp")
    gc.collect()
    assert a.stats()["live_buffers"] == 1
    with pytest.raises(RuntimeError, match="live"):
        a.close()
    assert not a.regions[0].closed


def test_concurrent_buffers_keep_disjoint_owned_bytes():
    a = arena(4096, 4096)
    def work(value):
        for _ in range(100):
            b = a(256)
            ctypes.memset(b.data_ptr(), value, b.numel())
            assert ctypes.string_at(b.data_ptr(), b.numel()) == bytes([value]) * b.numel()
            del b
    with ThreadPoolExecutor(8) as executor:
        list(executor.map(work, range(1, 9)))
    assert a.stats()["used_bytes"] == 0
    assert a.stats()["largest_free_bytes"] == 4096
    a.close()


def test_failed_native_teardown_is_not_retried_or_reported_closed():
    a = arena(512)
    def failed():
        raise IOError("unmap uncertain")
    a.regions[0].close = failed
    with pytest.raises(IOError):
        a.close()
    with pytest.raises(RuntimeError, match="quarantined"):
        a.close()
    with pytest.raises(RuntimeError):
        a(1)
    assert not a.closed
