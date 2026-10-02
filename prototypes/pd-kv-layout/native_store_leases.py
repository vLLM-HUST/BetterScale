"""Scope native expiring read leases to existing checkpoint references.

query(keys) MUST perform fresh lease-granting native metadata queries and return
the set having all required COMPLETE DRAM replicas, never mere is_exist results.
ttl_seconds is a verified lower bound from the owned master launch, not a guess.
No new LRU lives here: existing Core checkpoint drop releases these references.
"""
from threading import Condition, Lock, Thread
import time


class LeaseLost(RuntimeError):
    pass


class NativeStoreLeases:
    def __init__(self, query, *, ttl_seconds, clock=time.monotonic, background=True):
        if ttl_seconds <= 0:
            raise ValueError("verified positive native lease TTL required")
        self.query, self.ttl, self.clock = query, ttl_seconds, clock
        self.condition, self.query_lock = Condition(), Lock()
        self.groups, self.references, self.deadlines = {}, {}, {}
        self.failure = None
        self.closed = False
        self.thread = None
        if background:
            self.thread = Thread(target=self._run,name="state-store-leases",daemon=True)
            self.thread.start()

    def _fail(self, reason):
        self.failure = self.failure or str(reason)
        self.condition.notify_all()
        raise LeaseLost(self.failure)

    def _healthy(self):
        if self.closed:raise LeaseLost("native lease ledger closed")
        if self.failure:raise LeaseLost(self.failure)
        now = self.clock()
        if any(now >= deadline for deadline in self.deadlines.values()):
            self._fail("native State eviction lease expired")

    def _query(self, keys, *, required):
        # Serialize native calls, not the model or device copy stream.
        with self.query_lock:
            with self.condition:
                self._healthy()
                start = self.clock()
            try:
                found = set(self.query(tuple(keys)))
                if not found.issubset(keys):
                    raise ValueError("foreign State key in lease query")
            except BaseException as error:
                with self.condition:self._fail("native lease query failed: "+str(error))
            with self.condition:
                self._healthy()  # no silent recovery across an expired interval
                if self.clock() >= start+self.ttl:
                    self._fail("native lease reply arrived after conservative expiry")
                wanted = set(keys).intersection(self.references)
                protected = wanted.intersection(self.deadlines)
                if not protected.issubset(found) or (required and not wanted.issubset(found)):
                    self._fail("previously held State replica disappeared")
                for key in wanted.intersection(found):
                    self.deadlines[key] = start+self.ttl
                self.condition.notify_all()
                return found

    def begin(self, group, keys):
        keys = tuple(keys)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate State object in checkpoint")
        with self.condition:
            self._healthy()
            if group in self.groups:
                if self.groups[group] != set(keys):
                    raise ValueError("checkpoint State manifest changed")
                return [key in self.deadlines for key in keys]
            self.groups[group] = set(keys)
            for key in keys:self.references[key] = self.references.get(key,0)+1
        # Absent objects are provisional: the staging ring owns their bytes
        # until publish() has acquired the native lease after successful PUT.
        found = self._query(keys,required=False)
        return [key in found for key in keys]

    def publish(self, key):
        with self.condition:
            self._healthy()
            if key not in self.references:
                raise ValueError("unowned State object publication")
        self._query((key,),required=True)

    def check(self, group, *, complete=False):
        with self.condition:
            self._healthy()
            if group not in self.groups:raise ValueError("unknown checkpoint")
            if complete and not self.groups[group].issubset(self.deadlines):
                raise LeaseLost("checkpoint still has uncommitted State objects")

    def drop(self, group):
        with self.condition:
            for key in self.groups.pop(group):
                self.references[key] -= 1
                if not self.references[key]:
                    del self.references[key]
                    self.deadlines.pop(key,None)
            self.condition.notify_all()

    def renew(self):
        with self.condition:
            self._healthy()
            keys = tuple(self.deadlines)
        if keys:self._query(keys,required=True)

    def _run(self):
        try:
            while True:
                with self.condition:
                    if self.closed or self.failure:return
                    if not self.deadlines:
                        self.condition.wait()
                        continue
                    delay = min(self.deadlines.values())-self.clock()-self.ttl/2
                    if delay > 0:
                        self.condition.wait(timeout=delay)
                        continue
                self.renew()
        except LeaseLost:
            return  # sticky failure is checked by every State operation

    def close(self):
        # Caller first drains State operations. Ordinary native leases expire;
        # no force removal or permanent hard pin is introduced at teardown.
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise RuntimeError("native lease RPC still active; retain client")
