"""Per-rank receiving half of private State replication.

Control RPCs may call these methods concurrently. Payloads travel through the
existing TransferEngine directly into owned pinned writers. No device forward,
node-shared cache, whole-owner transfer lock, or timeout-based buffer reuse.
"""
import ctypes
import hashlib
import re
from threading import RLock


class RankReplicaReceiver:
    def __init__(self,pool,engine,*,max_pending=20):
        if type(max_pending) is not int or not 1<=max_pending<=20:
            raise ValueError("bounded transfer concurrency required")
        self.pool,self.engine,self.limit=pool,engine,max_pending
        self.lock=RLock()
        self.pending,self.intents={},{}
        self.completed={}
        self.accepting=True
        self.failure=None
        self.quarantined=[]

    def _healthy(self):
        if self.failure:raise RuntimeError("rank replica quarantined: "+self.failure)

    def begin(self,checkpoint,objects):
        objects=tuple(objects)
        if not 1<=len(objects)<=256 or any(not isinstance(k,str) or not 1<=len(k)<=128 for k in objects):
            raise ValueError("bounded State manifest required")
        with self.lock:
            self._healthy()
            if not self.accepting:raise RuntimeError("rank receiver draining")
            return self.pool.retain(checkpoint,objects)

    @staticmethod
    def digest(pointer,size):
        return hashlib.sha256((ctypes.c_uint8*size).from_address(pointer)).hexdigest()

    def prepare(self,source,checkpoint,oid,size,digest,operation):
        source=tuple(source)
        side,group,rank=self.pool.owner
        if (len(source)!=3 or source[0] not in ("P","D") or source[0]==side
                or type(source[1]) is not int or not 0<=source[1]<4
                or type(source[2]) is not int or source[2]!=rank):
            raise ValueError("State peer does not match opposite-side TP shard")
        if (type(size) is not int or not 0<size<=(129<<20)+4
                or not isinstance(digest,str) or not re.fullmatch("[0-9a-f]{64}",digest)
                or not isinstance(operation,str) or not re.fullmatch("[0-9a-f]{32}",operation)):
            raise ValueError("invalid bounded State transfer descriptor")
        identity=(source,checkpoint,oid,size,digest)
        with self.lock:
            self._healthy()
            if not self.accepting:raise RuntimeError("rank receiver draining")
            if operation in self.completed:
                old=self.completed[operation]
                if old!=identity:raise ValueError("transfer operation identity changed")
                return dict(status="complete")
            if operation in self.pending:
                row=self.pending[operation]
                if row["identity"]!=identity:raise ValueError("transfer operation identity changed")
                if row["state"]=="receiving":
                    return dict(status="write",pointer=row["writer"].pointer,size=size)
                return dict(status="busy")
            with self.pool.lock:
                if oid not in self.pool.groups.get(checkpoint,()):
                    raise ValueError("transfer absent from retained checkpoint")
                item=self.pool.objects.get(oid)
                if oid in self.intents or (item is not None and item.state!="sealed"):
                    return dict(status="busy")
                existing=self.pool.read(oid) if item is not None else None
            if existing is None:
                if len(self.pending)>=self.limit:return dict(status="busy")
                self.intents[oid]=operation
                row=dict(identity=identity,writer=None,state="allocating")
                self.pending[operation]=row
        if existing is not None:
            try:
                if existing.size!=size:raise ValueError("immutable State size collision")
                if item.digest is None:
                    actual=self.digest(existing.pointer,size)
                    with self.pool.lock:item.digest=actual
                if item.digest!=digest:
                    raise ValueError("immutable State identity collision")
                return dict(status="complete")
            finally:existing.close()
        try:
            writer=self.pool.reserve(oid,size)
            row["writer"]=writer
            if self.engine.register_memory(writer.pointer,size)!=0:
                raise RuntimeError("private State receive registration failed")
            with self.lock:row["state"]="receiving"
            return dict(status="write",pointer=writer.pointer,size=size)
        except MemoryError as error:
            # reserve rejects capacity before allocating or registering anything.
            # This is backpressure, not uncertain DMA and not a poisoned owner.
            if row["writer"] is not None:
                with self.lock:
                    self.failure=str(error);self.quarantined.append(row)
                raise
            with self.lock:
                del self.pending[operation];del self.intents[oid]
            return dict(status="busy")
        except BaseException as error:
            with self.lock:
                self.failure=str(error);self.quarantined.append(row)
            raise

    def commit(self,operation):
        with self.lock:
            self._healthy()
            if operation in self.completed:return True
            row=self.pending[operation]
            if row["state"]=="checking":return False
            if row["state"]!="receiving":raise RuntimeError("State transfer not ready to commit")
            row["state"]="checking"
        source,checkpoint,oid,size,expected=row["identity"]
        writer=row["writer"]
        try:
            # Caller issues commit only after a successful synchronous native
            # transfer completion. Verify bytes before exposing them to H2D.
            if self.digest(writer.pointer,size)!=expected:
                raise RuntimeError("received State checksum mismatch")
            if self.engine.unregister_memory(writer.pointer)!=0:
                raise RuntimeError("private State receive deregistration failed")
            writer.seal(lambda:None)
            with self.pool.lock:self.pool.objects[oid].digest=expected
            with self.lock:
                del self.pending[operation];del self.intents[oid]
                self.completed[operation]=row["identity"]
            return True
        except BaseException as error:
            with self.lock:
                self.failure=str(error);self.quarantined.append(row)
            raise

    def drop(self,checkpoint):
        with self.lock:
            self._healthy()
            if any(row["identity"][1]==checkpoint for row in self.pending.values()):
                raise RuntimeError("incoming State is still in flight")
            self.pool.drop(checkpoint)
            # A delayed commit retry remains idempotent for the checkpoint's
            # whole lifetime, not merely the last N unrelated transactions.
            for operation,identity in tuple(self.completed.items()):
                if identity[1]==checkpoint:del self.completed[operation]
