"""State DMA into final rank-private pinned objects, independent of egress.

replicate(checkpoint, object_ids) owns peer routing/transfer and returns only
after the required replica is committed. It must hold pool read leases during
network access. This module does not invent a node-shared cache or LRU.
"""
from collections import defaultdict
import ctypes
import hashlib
import time

from betterscale.live.runtime.host_state import TorchHostStateBackend
from native_state_frame import FramePlan
from native_state_transport import NativeTransfer


class RankStateTransport:
    supports_staged_receipts = True
    requires_complete_manifest = True

    def __init__(self,pool,namespace,submit,replicate,*,verify=False,release_checkpoint=None):
        if not namespace:raise ValueError("State namespace required")
        self.pool,self.namespace,self.submit,self.replicate=pool,namespace,submit,replicate
        self.verify=verify
        self.release_checkpoint=release_checkpoint or pool.drop
        self.quarantined=[]

    def object_id(self,identity):
        return hashlib.sha256((self.namespace+"/rank-frame-v1\\0"+identity).encode()).hexdigest()

    def release(self,key):
        self.release_checkpoint(key)

    def transfer(self,key,states,objects,*,store,stream,on_staged=None,manifest=None):
        states,objects=tuple(states),dict(objects)
        manifest=tuple(objects) if manifest is None else tuple(manifest)
        if len(set(manifest))!=len(manifest) or not set(objects).issubset(manifest):
            raise ValueError("invalid complete State manifest")
        if on_staged is not None and not store:raise ValueError("only exports can stage")
        phases=defaultdict(float)
        def timed(name,fn,*args,**kwargs):
            begin=time.perf_counter()
            try:return fn(*args,**kwargs)
            finally:phases[name]+=time.perf_counter()-begin
        def wait_copy(name,wait):
            timed(name+"_wait",wait)
            seconds=getattr(wait,"device_seconds",None)
            if seconds is not None:phases[name+"_device"]+=seconds
        def run():
            keys=[self.object_id(identity) for identity in manifest]
            present=dict(zip(keys,self.pool.retain(key,keys),strict=True))
            exporting={self.object_id(identity) for identity in objects}
            if store and any(not value for identity,value in present.items() if identity not in exporting):
                raise ValueError("missing source object in complete checkpoint")
            if not store and not all(present.values()):
                raise KeyError("rank-local checkpoint needs peer fetch before restore")
            moved=0
            for identity,selection in objects.items():
                oid=self.object_id(identity)
                if store and present[oid]:continue
                lanes=TorchHostStateBackend._select_lanes(states,selection)
                plan=timed("frame_plan",FramePlan.build,lanes)
                lease=None;scratch=None
                try:
                    if store:
                        lease=timed("allocate",self.pool.reserve,oid,plan.byte_length)
                        descriptors=timed("descriptors",plan.copy_descriptors,lease,to_host=True)
                        wait=timed("d2h_submit",self.submit,descriptors,True,stream)
                        lease.seal(lambda:wait_copy("d2h",wait))
                    else:
                        lease=self.pool.read(oid)
                        descriptors=timed("descriptors",plan.copy_descriptors,lease,to_host=False)
                        wait=timed("h2d_submit",self.submit,descriptors,False,stream)
                        wait_copy("h2d",wait)
                        if self.verify:
                            # Audit cannot overwrite the immutable cached source
                            # while another network/DMA reader may be using it.
                            from types import SimpleNamespace
                            buffer=self.pool.allocate(plan.byte_length)
                            scratch=SimpleNamespace(pointer=buffer.data_ptr(),size=plan.byte_length,
                                                    state="writing",buffer=buffer)
                            expected=hashlib.sha256((ctypes.c_uint8*plan.byte_length).from_address(lease.pointer)).digest()
                            desc=plan.copy_descriptors(scratch,to_host=True)
                            wait=self.submit(desc,True,stream);wait_copy("audit",wait)
                            actual=hashlib.sha256((ctypes.c_uint8*plan.byte_length).from_address(scratch.pointer)).digest()
                            if actual!=expected:raise RuntimeError("post-H2D rank State mismatch")
                        lease.close()
                    moved+=plan.payload_bytes
                except BaseException:
                    # An enqueue can fail before returning a completion event.
                    # Never let an uncertain asynchronous read/write unpin bytes.
                    self.quarantined.append((states,stream,lease,scratch))
                    raise
            if not self.pool.complete(key):raise RuntimeError("local State commit incomplete")
            if store:
                if on_staged is not None:on_staged(moved)
                timed("replica_wait",self.replicate,key,tuple(keys))
            return moved
        return NativeTransfer(run,phases)
