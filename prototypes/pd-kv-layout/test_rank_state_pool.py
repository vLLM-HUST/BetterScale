import ctypes
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
from rank_state_pool import RankStatePool


def pool(owner=("D", 0, 0), size=64):
    return RankStatePool(owner, size, lambda n: torch.empty(n, dtype=torch.uint8))


def write(p, key, value, n=32):
    w = p.reserve(key, n)
    ctypes.memset(w.pointer, value, n)
    w.seal(lambda: None)
    return w


def test_tp_and_attention_owners_are_isolated():
    pools = [pool((side, group, rank)) for side in ("P", "D") for group in range(4) for rank in range(2)]
    pointers = []
    for i, p in enumerate(pools):
        p.retain("checkpoint", ("same-logical-key",))
        write(p, "same-logical-key", i)
        with p.read("same-logical-key") as r:
            pointers.append(r.pointer)
            assert ctypes.string_at(r.pointer, r.size) == bytes([i]) * r.size
    assert len(set(pointers)) == 16
    for p in pools:
        p.drop("checkpoint"); p.close()


def test_read_and_manifest_refs_independently_pin():
    p = pool()
    p.retain("a", ("shared",)); p.retain("b", ("shared",))
    w = write(p, "shared", 7, 64)
    with pytest.raises(RuntimeError): _ = w.pointer
    p.drop("a")
    p.retain("c", ("new",))
    with pytest.raises(MemoryError): p.reserve("new", 64)
    r = p.read("shared"); p.drop("b")
    with pytest.raises(MemoryError): p.reserve("new", 64)
    assert ctypes.string_at(r.pointer, 64) == bytes([7]) * 64
    r.close(); r.close()
    with pytest.raises(RuntimeError): _ = r.pointer
    write(p, "new", 9, 64)
    assert p.bytes == 64 and "shared" not in p.objects
    p.drop("c"); p.close()


def test_dma_failure_quarantines_without_reclaim():
    p = pool()
    p.retain("a", ("x",))
    w = p.reserve("x", 64)
    def fail(): raise RuntimeError("DMA")
    with pytest.raises(RuntimeError, match="DMA"): w.seal(fail)
    with pytest.raises(RuntimeError): p.read("x")
    p.drop("a"); p.retain("b", ("y",))
    with pytest.raises(MemoryError): p.reserve("y", 64)
    with pytest.raises(RuntimeError): p.close()
    assert p.bytes == 64 and w.state == "quarantined"


def test_inflight_not_visible_and_manifest_identity():
    p = pool(); p.retain("a", ("x",))
    with pytest.raises(ValueError): p.retain("a", ("y",))
    w = p.reserve("x", 32)
    with pytest.raises(RuntimeError): p.read("x")
    assert not p.complete("a")
    w.seal(lambda: None)
    assert p.complete("a")
    p.drop("a"); p.close()


def test_independent_concurrent_writers():
    p = pool(size=20 * 32)
    def run(i):
        key=str(i); p.retain(key, (key,)); write(p, key, i)
        with p.read(key) as r:
            assert ctypes.string_at(r.pointer, 32) == bytes([i]) * 32
        p.drop(key)
    with ThreadPoolExecutor(20) as executor:
        list(executor.map(run, range(20)))
    assert p.bytes <= p.budget
    p.close()
