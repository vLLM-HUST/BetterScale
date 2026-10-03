"""Exact native dual-node DRAM replica placement/read gate; no model."""
import argparse
import ctypes
import json
from pathlib import Path
import time
from mooncake.store import MooncakeDistributedStore, ReplicateConfig, ReplicaStatus
from native_dram_staging import NativeDramStaging

HOSTS={"D":"10.244.2.32","P":"10.244.1.16"}
KEY="pd-native-two-replica-gate2"
SIZE=16<<20

class BothNodes:
    def __init__(self,client):
        self.client=client
        self.config=ReplicateConfig()
        self.config.replica_num=2
        self.config.preferred_segments=[ip+":55307" for ip in HOSTS.values()]
    def register_buffer(self,*args):return self.client.register_buffer(*args)
    def unregister_buffer(self,*args):return self.client.unregister_buffer(*args)
    def put_from(self,*args):return self.client.put_from(*args,self.config)

def run(role,write,output):
    if output.exists():raise ValueError("Preserve existing evidence")
    client=MooncakeDistributedStore();arena=None;query=None
    try:
        assert client.setup_dummy(32<<20,32<<20,HOSTS[role]+":55306")==0
        arena=NativeDramStaging(BothNodes(client),slot_bytes=SIZE,slots=2)
        result=dict(role=role,bytes=SIZE,write=write)
        if write:
            lease=arena.acquire(SIZE)
            ctypes.memset(lease.pointer,83,SIZE)
            lease.seal(lambda:None)
            begin=time.perf_counter()
            result["commit"]=lease.replicate(KEY).result(timeout=60)
            result["commit_wall_seconds"]=time.perf_counter()-begin
        # Installed DummyClient descriptor RPC is not registered by its
        # native service. Inspect master metadata through a bounded ordinary
        # client instead; never dereference the returned remote addresses.
        query=MooncakeDistributedStore()
        assert query.setup(HOSTS[role]+":55308",
            "http://10.244.2.32:55302/metadata",0,1<<20,"tcp","",
            "10.244.2.32:55301")==0
        descriptors=query.get_replica_desc(KEY)
        replicas=[]
        for replica in descriptors:
            assert replica.status==ReplicaStatus.COMPLETE and replica.is_memory_replica()
            desc=replica.get_memory_descriptor().buffer_descriptor
            replicas.append(dict(endpoint=desc.transport_endpoint,size=desc.size))
        assert len(replicas)==2,replicas
        assert {r["endpoint"] for r in replicas}=={ip+":55307" for ip in HOSTS.values()},replicas
        assert all(r["size"]==SIZE for r in replicas),replicas
        lease=arena.acquire(SIZE);ctypes.memset(lease.pointer,0,SIZE)
        begin=time.perf_counter()
        assert client.get_into(KEY,lease.pointer,SIZE)==SIZE
        result["read_seconds"]=time.perf_counter()-begin
        assert ctypes.string_at(lease.pointer,SIZE)==bytes([83])*SIZE
        lease.seal(lambda:None);lease.discard()
        result.update(exact=True,replicas=replicas)
        output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    finally:
        if arena is not None:
            arena.executor.shutdown(wait=True)
            if not arena.active:arena.close()
        if query is not None:query.close()
        client.close()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role",choices=HOSTS,required=True)
    p.add_argument("--write",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();run(a.role,a.write,a.output)
