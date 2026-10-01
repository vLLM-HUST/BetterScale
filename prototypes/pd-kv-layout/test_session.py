import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from session import CacheMiss, Conflict, Directory, Objects, Turn, restore

STREAMS = ("l0/k/h0", "l0/v/h0", "l0/k/h1", "l0/v/h1")
IDENTITY = "qwen35-target-bf16-fixture/salt-A"
CHECKPOINT = {"gdn": b"accepted-gdn", "conv": b"normalized-conv"}


class MemoryStore:
    def __init__(self):
        self.data = {}
        self.fail = False
        self.gate = None

    def put(self, key, value):
        if self.gate is not None:
            assert self.gate.wait(5)
        if self.fail:
            return -1
        self.data.setdefault(key, bytes(value))
        return 0

    def get(self, key):
        return self.data.get(key)

    def get_size(self,key):
        return len(self.data[key]) if key in self.data else -1


@pytest.fixture
def setup(tmp_path):
    directory = Directory(tmp_path / "directory.sqlite")
    directory.create("session", IDENTITY, "P0")
    return directory, MemoryStore()


def payload(start, stop):
    return {s: bytes((t + i) % 255 for t in range(start*2, stop*2))
            for i, s in enumerate(STREAMS)}


def turn(directory, store, owner):
    return Turn(directory, Objects(store), directory.claim("session", owner, IDENTITY),
                STREAMS, 2)


def complete(writer, tokens, next_owner):
    try:
        return writer.finish(tokens, CHECKPOINT, next_owner, writer_retired=True)
    finally:
        writer.close()


def test_bidirectional_incremental_handoff(setup):
    directory, store = setup
    p = turn(directory, store, "P0")
    p.append(3, payload(0, 3))
    complete(p, [1, 2, 3], "D0")
    d = turn(directory, store, "D0")
    d.append(5, payload(3, 5))
    complete(d, [1, 2, 3, 4, 5], "P0")
    p2 = turn(directory, store, "P0")
    p2.append(7, payload(5, 7))
    key = complete(p2, list(range(1, 8)), "D0")
    manifest, data, state = restore(Objects(store), key, IDENTITY, STREAMS, 2)
    assert data == payload(0, 7)
    assert state == CHECKPOINT
    assert len(manifest["chunks"]) == 3
    assert len(store.data) == 3 * (4 + 2 + 1)  # old chunks referenced, not rewritten


def test_one_writer_with_two_independent_directory_connections(setup):
    directory, _ = setup
    barrier = threading.Barrier(2)
    def claim():
        other = Directory(directory.path)
        barrier.wait()
        try:
            return other.claim("session", "P0", IDENTITY)
        except Conflict:
            return None
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(claim) for _ in range(2)]
        results = [f.result() for f in futures]
    assert sum(r is not None for r in results) == 1


def test_wrong_owner_and_identity(setup):
    directory, _ = setup
    with pytest.raises(Conflict):
        directory.claim("session", "D0", IDENTITY)
    with pytest.raises(Conflict):
        directory.claim("session", "P0", "salt-B")


def test_async_snapshot_is_not_published_or_mutated(setup):
    directory, store = setup
    store.gate = threading.Event()
    p = turn(directory, store, "P0")
    raw = {s: bytearray(v) for s, v in payload(0, 2).items()}
    try:
        p.append(2, raw)
        assert directory.current("session") is None
        for value in raw.values():
            value[:] = b"\xff" * len(value)
        store.gate.set()
        key = complete(p, [1, 2], "D0")
        assert restore(Objects(store), key, IDENTITY, STREAMS, 2)[1] == payload(0, 2)
    finally:
        store.gate.set()
        p.close()


def test_final_writer_fence_is_required(setup):
    directory, store = setup
    p = turn(directory, store, "P0")
    try:
        p.append(1, payload(0, 1))
        with pytest.raises(Conflict):
            p.finish([1], CHECKPOINT, "D0", writer_retired=False)
        assert directory.current("session") is None
    finally:
        p.close()


def test_failed_transfer_keeps_old_version(setup):
    directory, store = setup
    p = turn(directory, store, "P0")
    p.append(1, payload(0, 1))
    old = complete(p, [1], "D0")
    d = turn(directory, store, "D0")
    try:
        store.fail = True
        d.append(2, payload(1, 2))
        with pytest.raises(IOError):
            d.finish([1, 2], CHECKPOINT, "P0", writer_retired=True)
        assert directory.current("session") == old
    finally:
        d.close()


def test_late_writer_cannot_publish_after_revoke(setup):
    directory, store = setup
    old = turn(directory, store, "P0")
    try:
        old.append(1, payload(0, 1))
        directory.revoke(old.lease)
        fresh = turn(directory, store, "P0")
        fresh.append(2, payload(0, 2))
        key = complete(fresh, [1, 2], "D0")
        with pytest.raises(Conflict):
            old.finish([9], CHECKPOINT, "D0", writer_retired=True)
        assert directory.current("session") == key
    finally:
        old.close()


@pytest.mark.parametrize("kind", ["dense", "gdn", "conv", "manifest"])
def test_eviction_is_a_miss_not_partial_state(setup, kind):
    directory, store = setup
    p = turn(directory, store, "P0")
    p.append(2, payload(0, 2))
    key = complete(p, [1, 2], "D0")
    m = json.loads(store.data[key])
    victim = (m["chunks"][0]["keys"][STREAMS[0]] if kind == "dense"
              else key if kind == "manifest" else m["checkpoint"][kind]["key"])
    del store.data[victim]
    with pytest.raises(CacheMiss):
        restore(Objects(store), key, IDENTITY, STREAMS, 2)


def test_directory_reopens_without_losing_writer_fence(setup):
    directory, _ = setup
    lease = directory.claim("session", "P0", IDENTITY)
    restarted = Directory(directory.path)
    with pytest.raises(Conflict):
        restarted.claim("session", "P0", IDENTITY)
    restarted.revoke(lease)
    assert restarted.claim("session", "P0", IDENTITY).epoch > lease.epoch


def test_incomplete_increment_and_history_rejected(setup):
    directory, store = setup
    p = turn(directory, store, "P0")
    try:
        with pytest.raises(ValueError):
            p.append(1, {STREAMS[0]: b"ab"})
        p.append(1, payload(0, 1))
        with pytest.raises(ValueError):
            p.finish([], CHECKPOINT, "D0", writer_retired=True)
        assert directory.current("session") is None
    finally:
        p.close()

def test_staging_byte_budget_rejects_oversized_delta(setup):
    directory, store = setup
    lease = directory.claim("session", "P0", IDENTITY)
    writer = Turn(directory, Objects(store), lease, STREAMS, 2, max_pending_bytes=8)
    try:
        with pytest.raises(ValueError):
            writer.append(2, payload(0, 2))
        assert writer.cursor == 0 and not writer.pending
        writer.append(1, payload(0, 1))
        assert writer.pending_bytes == 8
        complete(writer, [1], "D0")
        assert writer.pending_bytes == 0
    finally:
        writer.close()


def test_bad_checkpoint_cannot_be_retried_after_partial_write(setup):
    directory, store = setup
    writer = turn(directory, store, "P0")
    writer.append(1, payload(0, 1))
    original = store.put
    def fail_checkpoint(key, data):
        return -1 if "/checkpoint/conv" in key else original(key, data)
    store.put = fail_checkpoint
    try:
        with pytest.raises(IOError):
            writer.finish([1], CHECKPOINT, "D0", writer_retired=True)
        assert directory.current("session") is None and writer.failed
        with pytest.raises(Conflict):
            writer.finish([1], CHECKPOINT, "D0", writer_retired=True)
    finally:
        writer.close()


def test_acknowledged_external_chunks_keep_namespace_and_epoch_fences(setup):
    directory,store=setup
    writer=turn(directory,store,'P0');prefix=writer.prefix
    keys={name:f'{prefix}/dense/0/{i}' for i,name in enumerate(STREAMS)}
    try:
        for name,data in payload(0,2).items():assert store.put(keys[name],data)==0
        with pytest.raises(ValueError):writer.append_acknowledged(2,keys,prefix='pd/other/1/x')
        aliases=dict.fromkeys(STREAMS,next(iter(keys.values())))
        with pytest.raises(ValueError):writer.append_acknowledged(2,aliases,prefix=prefix)
        writer.append_acknowledged(2,keys,prefix=prefix)
        with pytest.raises(ValueError):writer.append_acknowledged(4,keys,prefix=prefix)
        assert writer.cursor==2
        directory.revoke(writer.lease)
        with pytest.raises(Conflict,match='stale writer'):complete(writer,[1,2],'D0')
        assert directory.current('session') is None
    finally:writer.close()


@pytest.mark.parametrize('failure',[None,'dense-missing','dense-short','checkpoint-missing','checkpoint-short'])
def test_incremental_publish_checks_base_without_payload_reads(setup,failure):
    directory,store=setup
    first=turn(directory,store,'P0');first.append(3,payload(0,3))
    key=complete(first,[1,2,3],'D0');m=json.loads(store.data[key])
    if failure:
        victim=(next(iter(m['chunks'][0]['keys'].values())) if failure.startswith('dense')
                else m['checkpoint']['gdn']['key'])
        if failure.endswith('missing'):del store.data[victim]
        else:store.data[victim]=store.data[victim][:-1]
    reads=[];get=store.get
    def record(k):
        reads.append(k)
        assert k==key,'Publisher must not reread historical payloads'
        return get(k)
    store.get=record
    if failure:
        with pytest.raises(CacheMiss):turn(directory,store,'D0')
    else:
        resumed=turn(directory,store,'D0');assert resumed.cursor==3;resumed.close()
    assert reads==[key]


@pytest.mark.parametrize('field,value',[('checkpoint',['gdn','conv']),('chunks',None),('cursor',True)])
def test_metadata_only_base_rejects_malformed_manifest(setup,field,value):
    directory,store=setup
    first=turn(directory,store,'P0');first.append(3,payload(0,3))
    key=complete(first,[1,2,3],'D0');m=json.loads(store.data[key]);m[field]=value
    store.data[key]=json.dumps(m).encode()
    with pytest.raises(CacheMiss):turn(directory,store,'D0')
