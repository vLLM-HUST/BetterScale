"""Bounded CPU PD session protocol prototype; NOT a production coordinator.

SQLite is the local CAS/fencing oracle. Mooncake owns immutable byte objects.
There is no distributed consensus, scheduler hook, device fence or model math.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import json
import sqlite3
import uuid


class Conflict(RuntimeError):
    pass


class CacheMiss(RuntimeError):
    pass


@dataclass(frozen=True)
class Lease:
    session: str
    epoch: int
    owner: str
    base: str | None
    identity: str


class Directory:
    def __init__(self, path):
        self.path = str(path)
        with self.transaction() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS sessions(
                id TEXT PRIMARY KEY, identity TEXT NOT NULL, epoch INTEGER NOT NULL,
                owner TEXT NOT NULL, active INTEGER NOT NULL, manifest TEXT)""")

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def create(self, session, identity, owner):
        with self.transaction() as db:
            db.execute("INSERT INTO sessions VALUES(?,?,0,?,0,NULL)",
                       (session, identity, owner))

    def claim(self, session, owner, identity):
        with self.transaction() as db:
            row = db.execute("SELECT identity,epoch,owner,active,manifest FROM sessions WHERE id=?",
                             (session,)).fetchone()
            if row is None or row[0] != identity or row[2] != owner or row[3]:
                raise Conflict("wrong identity/owner or another active writer")
            epoch = row[1] + 1
            db.execute("UPDATE sessions SET epoch=?,active=1 WHERE id=?", (epoch, session))
            return Lease(session, epoch, owner, row[4], identity)

    def publish(self, lease, manifest, next_owner):
        with self.transaction() as db:
            changed = db.execute(
                "UPDATE sessions SET manifest=?,owner=?,active=0 "
                "WHERE id=? AND epoch=? AND owner=? AND active=1",
                (manifest, next_owner, lease.session, lease.epoch, lease.owner)).rowcount
            if changed != 1:
                raise Conflict("stale writer cannot publish")

    def revoke(self, lease):
        with self.transaction() as db:
            changed = db.execute(
                "UPDATE sessions SET epoch=epoch+1,active=0 "
                "WHERE id=? AND epoch=? AND owner=? AND active=1",
                (lease.session, lease.epoch, lease.owner)).rowcount
            if changed != 1:
                raise Conflict("stale revoke")

    def current(self, session):
        with self.transaction() as db:
            row = db.execute("SELECT manifest FROM sessions WHERE id=?", (session,)).fetchone()
            if row is None:
                raise Conflict("unknown session")
            return row[0]


class Objects:
    """Only immutable unique keys; no in-place upsert or silent Store errors."""
    def __init__(self, store):
        self.store = store

    def put(self, key, payload):
        if self.store.put(key, payload) != 0:
            raise IOError("Store put failed: " + key)

    def get(self, key):
        payload = self.store.get(key)
        if payload is None:
            raise CacheMiss(key)
        return bytes(payload)



def restore_manifest(objects,key,identity,streams,token_bytes):
    """Check immutable dependency sizes without rereading historical payloads.

    This is a point-in-time check, not a pin/lease. A later eviction still makes
    receiver restore fail closed, just as after the old full-payload check.
    """
    if key is None:return dict(tokens=[],chunks=[],cursor=0)
    try:
        m=json.loads(objects.get(key));cursor=m['cursor']
        if (m['schema']!=1 or m['identity']!=identity or m['streams']!=list(streams)
                or m['token_bytes']!=token_bytes or type(cursor) is not int or cursor<0
                or not isinstance(m['tokens'],list) or len(m['tokens'])!=cursor
                or any(type(t) is not int or t<0 for t in m['tokens'])
                or not isinstance(m['chunks'],list)):
            raise ValueError('Invalid base manifest')
        sizes={};frontier=0
        def dependency(k,size):
            if not isinstance(k,str) or not k or k in sizes or type(size) is not int or size<=0:
                raise ValueError('Invalid dependency')
            sizes[k]=size
        for chunk in m['chunks']:
            stop=chunk['stop'];keys=chunk['keys']
            if (type(chunk['start']) is not int or chunk['start']!=frontier
                    or type(stop) is not int or not frontier<stop<=cursor
                    or not isinstance(keys,dict) or set(keys)!=set(streams)):
                raise ValueError('Invalid base frontier')
            for k in keys.values():dependency(k,(stop-frontier)*token_bytes)
            frontier=stop
        if (frontier!=cursor or not isinstance(m['checkpoint'],dict)
                or set(m['checkpoint'])!={'gdn','conv'}):
            raise ValueError('Incomplete base snapshot')
        for d in m['checkpoint'].values():dependency(d['key'],d['bytes'])
        for k,size in sizes.items():
            if objects.store.get_size(k)!=size:raise CacheMiss('Missing/truncated base dependency: '+k)
        return m
    except (KeyError,TypeError,ValueError) as exc:
        raise CacheMiss('Invalid base snapshot') from exc


def restore(objects, key, identity, streams, token_bytes):
    """Stage ALL required bytes before returning anything consumable.

    Eviction at any point gives a miss. This prototype materializes a CPU copy,
    so it needs no remote lease after the complete staging result is returned.
    It does not install partial bytes into live State.
    """
    if key is None:
        return {"tokens": [], "chunks": [], "cursor": 0}, {}, {}
    try:
        manifest = json.loads(objects.get(key))
        if (manifest["schema"] != 1 or manifest["identity"] != identity
                or manifest["streams"] != list(streams)
                or manifest["token_bytes"] != token_bytes
                or len(manifest["tokens"]) != manifest["cursor"]):
            raise ValueError("incompatible manifest")
        dense = {name: bytearray() for name in streams}
        frontier = 0
        for chunk in manifest["chunks"]:
            if chunk["start"] != frontier or chunk["stop"] <= frontier:
                raise ValueError("non-contiguous token history")
            if set(chunk["keys"]) != set(streams):
                raise ValueError("incomplete stream set")
            size = (chunk["stop"] - frontier) * token_bytes
            for name in streams:
                data = objects.get(chunk["keys"][name])
                if len(data) != size:
                    raise ValueError("truncated dense chunk")
                dense[name].extend(data)
            frontier = chunk["stop"]
        if frontier != manifest["cursor"]:
            raise ValueError("checkpoint/dense frontier mismatch")
        checkpoint = {}
        if set(manifest["checkpoint"]) != {"gdn", "conv"}:
            raise ValueError("missing target checkpoint")
        for name, descriptor in manifest["checkpoint"].items():
            data = objects.get(descriptor["key"])
            if len(data) != descriptor["bytes"]:
                raise ValueError("truncated checkpoint")
            checkpoint[name] = data
        return manifest, {k: bytes(v) for k, v in dense.items()}, checkpoint
    except (KeyError, ValueError, TypeError) as exc:
        raise CacheMiss("invalid/incomplete session snapshot") from exc


class Turn:
    def __init__(self, directory, objects, lease, streams, token_bytes,
                 max_pending_batches=2, max_pending_bytes=64 * 1024**2):
        if token_bytes <= 0 or max_pending_batches <= 0 or max_pending_bytes <= 0 or not streams or len(set(streams)) != len(streams):
            raise ValueError("invalid transfer geometry")
        self.directory, self.objects, self.lease = directory, objects, lease
        self.streams, self.token_bytes = tuple(streams), token_bytes
        self.base = restore_manifest(objects, lease.base, lease.identity, streams, token_bytes)
        self.cursor = self.base["cursor"]
        self.chunks = list(self.base["chunks"])
        self.pending = []
        self.max_pending_batches = max_pending_batches
        self.max_pending_bytes = max_pending_bytes
        self.pending_bytes = 0
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pd-store")
        self.failed = False
        self.closed = False
        self.prefix = f"pd/{lease.session}/{lease.epoch}/{uuid.uuid4().hex}"

    def _drain_one(self):
        batch, size = self.pending.pop(0)
        self.pending_bytes -= size
        try:
            for task in batch:
                task.result()
        except BaseException:
            self.failed = True
            raise

    def append(self, stop, payloads):
        if self.closed or self.failed:
            raise Conflict("turn no longer writable")
        if stop <= self.cursor or set(payloads) != set(self.streams):
            raise ValueError("invalid increment")
        size = (stop - self.cursor) * self.token_bytes
        if any(len(payloads[name]) != size for name in self.streams):
            raise ValueError("wrong increment byte count")
        # Caller must already have retired final device writes before supplying
        # these buffers. Copy now so later host mutation cannot race Store.
        batch_bytes = size * len(self.streams)
        if batch_bytes > self.max_pending_bytes:
            raise ValueError("increment exceeds staging byte budget; split it")
        while (self.pending and (len(self.pending) >= self.max_pending_batches
                                or self.pending_bytes + batch_bytes > self.max_pending_bytes)):
            self._drain_one()
        frozen = {name: bytes(payloads[name]) for name in self.streams}
        index = len(self.chunks)
        keys = {name: f"{self.prefix}/chunk/{index}/{i}" for i, name in enumerate(self.streams)}
        tasks = [self.pool.submit(self.objects.put, keys[name], frozen[name])
                 for name in self.streams]
        self.pending.append((tasks, batch_bytes))
        self.pending_bytes += batch_bytes
        self.chunks.append({"start": self.cursor, "stop": stop, "keys": keys})
        self.cursor = stop

    def append_acknowledged(self, stop, keys, *, prefix):
        """Adopt trusted worker Store acknowledgements, never uncompleted puts.

        This is an internal prototype seam, not proof that a remote object cannot
        subsequently be evicted. Restore still validates every dependency.
        """
        if self.closed or self.failed:
            raise Conflict("turn no longer writable")
        expected = f"pd/{self.lease.session}/{self.lease.epoch}/"
        if (not isinstance(prefix, str) or not prefix.startswith(expected)
                or not prefix[len(expected):] or '/' in prefix[len(expected):]
                or type(stop) is not int or stop <= self.cursor
                or set(keys) != set(self.streams)
                or (stop-self.cursor)*self.token_bytes*len(keys) > self.max_pending_bytes):
            raise ValueError("invalid acknowledged increment")
        old = {key for chunk in self.chunks for key in chunk['keys'].values()}
        values = list(keys.values())
        if (any(not isinstance(key, str) or not key.startswith(prefix+'/dense/') for key in values)
                or len(set(values)) != len(values) or old.intersection(values)):
            raise ValueError("invalid immutable chunk keys")
        self.chunks.append(dict(start=self.cursor, stop=stop, keys=dict(keys)))
        self.cursor = stop

    def finish(self, tokens, checkpoint, next_owner, *, writer_retired, pending_token=None):
        if self.closed or self.failed or not writer_retired:
            raise Conflict("cannot publish before final writer retirement")
        if (len(tokens) != self.cursor or list(tokens[:len(self.base["tokens"])]) != self.base["tokens"]
                or set(checkpoint) != {"gdn", "conv"} or any(not v for v in checkpoint.values())):
            raise ValueError("incompatible frontier/checkpoint")
        if any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("invalid token history")
        if pending_token is not None and (type(pending_token) is not int or pending_token < 0):
            raise ValueError("invalid uncomputed tail")
        try:
            while self.pending:
                self._drain_one()
            descriptors = {}
            for name, payload in checkpoint.items():
                key = f"{self.prefix}/checkpoint/{name}"
                frozen = bytes(payload)
                self.objects.put(key, frozen)
                descriptors[name] = {"key": key, "bytes": len(frozen)}
            manifest = dict(schema=1, identity=self.lease.identity, streams=list(self.streams),
                            token_bytes=self.token_bytes, cursor=self.cursor, tokens=list(tokens),
                            chunks=self.chunks, checkpoint=descriptors, pending_token=pending_token,
                            epoch=self.lease.epoch)
            key = f"{self.prefix}/manifest"
            self.objects.put(key, json.dumps(manifest).encode())
            # Only this CAS exposes the new version/next writer. Stale writes are
            # unreachable orphan objects, never overwrites of published data.
            self.directory.publish(self.lease, key, next_owner)
            self.closed = True
            self.pool.shutdown(wait=True)
            return key
        except BaseException:
            self.failed = True
            raise

    def close(self):
        self.closed = True
        self.pool.shutdown(wait=True)
