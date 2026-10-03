"""Background replication between explicitly selected sticky rank owners."""
import ctypes
import hashlib
import http.client
import json
from threading import Lock
import time
import uuid

HOSTS={"P":"10.244.1.16","D":"10.244.2.32"}
CONTROL_BASE=27400  # outside this deployment's32768..60999 ephemeral range


class RankReplicator:
    def __init__(self,pool,engine,*,control_base=CONTROL_BASE):
        self.pool,self.engine,self.control_base=pool,engine,control_base
        self.routes={}
        self.lock=Lock()
        self.registered={}
        self.quarantined=[]

    def set_peer(self,key,group):
        if group is not None and (type(group) is not int or not 0<=group<4):
            raise ValueError("invalid sticky peer group")
        with self.lock:
            if key in self.routes and self.routes[key]!=group:
                raise ValueError("checkpoint peer owner changed")
            self.routes[key]=group

    @staticmethod
    def rpc(host,port,op,args):
        payload=json.dumps(dict(op=op,args=args),separators=(",",":")).encode()
        if len(payload)>64<<10:raise ValueError("rank control request too large")
        for attempt in range(3):
            connection=http.client.HTTPConnection(host,port,timeout=30)
            try:
                connection.request("POST","/rpc",payload,{"Content-Type":"application/json"})
                response=connection.getresponse()
                raw=response.read((64<<10)+1)
                if response.status==503:
                    time.sleep(.05*(attempt+1));continue
                if len(raw)>64<<10:raise ValueError("rank control response too large")
                data=json.loads(raw)
                if response.status!=200 or not data.get("ok"):
                    raise RuntimeError("rank control failed: "+str(data.get("error")))
                return data["result"]
            except (OSError,http.client.HTTPException):
                if attempt==2:raise
                time.sleep(.05*(attempt+1))
            finally:connection.close()
        raise TimeoutError("rank control remained busy")

    def _register(self,reader):
        with self.lock:
            pointer=reader.pointer
            if pointer not in self.registered:
                if self.engine.register_memory(pointer,reader.size)!=0:
                    raise RuntimeError("source registration failed")
                self.registered[pointer]=0
            self.registered[pointer]+=1

    def _unregister(self,reader):
        with self.lock:
            pointer=reader.pointer
            if self.registered[pointer]==1:
                if self.engine.unregister_memory(pointer)!=0:
                    raise RuntimeError("source deregistration failed")
                del self.registered[pointer]
            else:self.registered[pointer]-=1

    def replicate(self,key,objects):
        with self.lock:
            if key not in self.routes:raise ValueError("checkpoint has no explicit peer placement")
            group=self.routes[key]
        if group is None:return  # explicitly P-only turn, not an accidental miss
        side,_,rank=self.pool.owner
        peer_side="P" if side=="D" else "D"
        host=HOSTS[peer_side];port=self.control_base+2*group+rank
        info=self.rpc(host,port,"info",{})
        endpoint=info["endpoint"]
        if (tuple(info["owner"])!=(peer_side,group,rank) or not isinstance(endpoint,str)
                or not endpoint.startswith(host+":") or not 0<int(endpoint.rsplit(":",1)[1])<65536):
            raise ValueError("peer endpoint/shard differs from sticky placement")
        wire=[key.request_id,key.generation]
        self.rpc(host,port,"begin",dict(checkpoint=wire,objects=list(objects)))
        for oid in objects:
            reader=self.pool.read(oid)
            registered=False
            try:
                item=reader.item
                if item.digest is None:
                    digest=hashlib.sha256((ctypes.c_uint8*reader.size).from_address(reader.pointer)).hexdigest()
                    with self.pool.lock:item.digest=digest
                operation=uuid.uuid4().hex
                args=dict(source=self.pool.owner,checkpoint=wire,oid=oid,size=reader.size,
                          digest=item.digest,operation=operation)
                deadline=time.monotonic()+180
                while True:
                    target=self.rpc(host,port,"prepare",args)
                    if target["status"]!="busy":break
                    if time.monotonic()>deadline:raise TimeoutError("peer State capacity/frontier deadline")
                    time.sleep(.05)
                if target["status"]=="complete":
                    reader.close();continue
                if (target["status"]!="write" or target["size"]!=reader.size
                        or type(target["pointer"]) is not int or target["pointer"]<=0):
                    raise ValueError("invalid private State target")
                self._register(reader);registered=True
                rc=self.engine.transfer_sync_write(endpoint,reader.pointer,target["pointer"],reader.size)
                if rc:raise RuntimeError(f"private State transfer failed: {rc}")
                self._unregister(reader);registered=False
                while not self.rpc(host,port,"commit",dict(operation=operation)):
                    if time.monotonic()>deadline:raise TimeoutError("peer State verification deadline")
                    time.sleep(.05)
                reader.close()
            except BaseException:
                # A failed native submission may still own its source address.
                # Likewise an unknown prepare/commit outcome must not be hidden.
                self.quarantined.append((key,reader,registered))
                raise
