"""Prototype direct-DRAM State transport; not enabled in the online launcher.

sink: batch_is_exist/get_into plus arena's two-replica put_from facade.
submit(descriptors, to_host, stream): enqueue on an owned stream and return a
real completion wait callable. The enclosing worker supplies device context.
"""
from concurrent.futures import Future
import ctypes
import hashlib
import hmac
from threading import Lock

from betterscale.live.runtime.host_state import TorchHostStateBackend
from native_state_frame import FramePlan


class CheckedReplicas:
    """Native pointer facade: checksum trailer is prepared after local DMA.

    No full payload Python bytes assembly. Configuration must request the
    required DRAM replicas; the launcher's placement gate verifies both nodes.
    """
    frame_integrity = True

    def __init__(self, client, config, leases):
        self.client, self.config, self.leases = client, config, leases

    def register_buffer(self, *args): return self.client.register_buffer(*args)
    def unregister_buffer(self, *args): return self.client.unregister_buffer(*args)

    def put_from(self, key, pointer, size):
        if size <= 32:raise ValueError("missing State frame body")
        body = (ctypes.c_uint8 * (size-32)).from_address(pointer)
        digest = hashlib.sha256(body).digest()
        ctypes.memmove(pointer+size-32,digest,32)
        rc = self.client.put_from(key,pointer,size,self.config)
        if rc == 0:
            # Store ACK alone is insufficient: acquire eviction protection
            # before the arena can recycle this object's last staging copy.
            self.leases.publish(key)
        return rc


class NativeTransfer:
    def __init__(self, run):
        self.run = run
        self.lock = Lock()
        self.done = Future()
        self.byte_length = 0

    def result(self):
        with self.lock:
            if not self.done.done():
                try:
                    self.byte_length = self.run()
                    self.done.set_result(self.byte_length)
                except BaseException as error:
                    self.done.set_exception(error)
        return self.done.result()


class NativeStateTransport:
    # Ledger integration is CPU-tested, but keep the live gate CLOSED until
    # the combined native DMA/replica/lease path has passed qualification.
    supports_staged_receipts = False

    def __init__(self, sink, arena, namespace, submit, *, leases, verify=False):
        if not namespace:raise ValueError("State namespace required")
        if not getattr(arena.store,"frame_integrity",False):
            raise ValueError("native State writes require checked replica facade")
        if getattr(arena.store,"leases",None) is not leases:
            raise ValueError("staging and checkpoint must share eviction lease ownership")
        self.leases = leases
        self.sink, self.arena, self.namespace = sink, arena, namespace
        self.submit, self.verify = submit, verify
        self.quarantined = []

    def object_id(self, identity):
        return hashlib.sha256((self.namespace+"/native-frame-v1\0"+identity).encode()).hexdigest()

    def release(self, key):
        # Core checkpoint/LRU metadata already owns the reference policy.
        # Drop only this manifest's refs; shared sealed pages retain others.
        with self.leases.condition:
            if key in self.leases.groups:self.leases.drop(key)

    @staticmethod
    def digest(lease):
        view = (ctypes.c_uint8 * lease.size).from_address(lease.pointer)
        return hashlib.sha256(view).digest()

    def transfer(self, key, states, objects, *, store, stream, on_staged=None):
        states, objects = tuple(states), dict(objects)
        if on_staged is not None and not store:
            raise ValueError("only exports have a local staged completion")
        def run():
            replicas, moved = [], 0
            keys = [self.object_id(k) for k in objects]
            exists = self.leases.begin(key,keys)
            if len(exists) != len(keys) or any(x not in (0,1) for x in exists):
                raise RuntimeError("native State presence query failed")
            if not store and not all(exists):
                raise KeyError("native State checkpoint missing a complete replica")
            for (identity,selection),remote,present in zip(objects.items(),keys,exists):
                if store and present:continue
                lanes = TorchHostStateBackend._select_lanes(states,selection)
                plan = FramePlan.build(lanes)
                lease = self.arena.acquire(plan.byte_length+32,timeout=None)
                try:
                    if store:
                        wait = self.submit(plan.copy_descriptors(lease,to_host=True),True,stream)
                        lease.seal(wait)
                        replicas.append(lease.replicate(remote))
                    else:
                        if self.sink.get_into(remote,lease.pointer,lease.size) != lease.size:
                            raise RuntimeError("native State object missing or truncated")
                        body = (ctypes.c_uint8 * plan.byte_length).from_address(lease.pointer)
                        checksum = ctypes.string_at(lease.pointer+plan.byte_length,32)
                        if not hmac.compare_digest(hashlib.sha256(body).digest(),checksum):
                            raise RuntimeError("native State frame checksum mismatch")
                        expected = self.digest(lease) if self.verify else None
                        wait = self.submit(plan.copy_descriptors(lease,to_host=False),False,stream)
                        wait()
                        if self.verify:
                            self.submit(plan.copy_descriptors(lease,to_host=True),True,stream)()
                            if self.digest(lease) != expected:
                                raise RuntimeError("post-H2D State frame byte mismatch")
                        lease.seal(lambda:None)
                        lease.discard()
                    moved += plan.payload_bytes
                except BaseException:
                    # A partial native enqueue can fail before returning its
                    # event. Keep device references and arena storage reachable;
                    # never infer a safe lifetime from a host exception.
                    with self.arena.condition:
                        if lease.state != "released":
                            lease.state = "quarantined"
                        self.arena.failure = "State frame transfer failed"
                        self.arena.condition.notify_all()
                    self.quarantined.append((key,states,stream,lease))
                    raise
            self.leases.check(key)
            if store and on_staged is not None:
                # Earlier objects are either still held by the staging ring or
                # already acknowledged in Store. All source DMA has retired.
                on_staged(moved)
            for replica in replicas:
                replica.result()
            self.leases.check(key,complete=True)
            return moved
        return NativeTransfer(run)
