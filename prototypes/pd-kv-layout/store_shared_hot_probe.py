"""Read-only probe of Mooncake's existing shared hot-cache lease API."""
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from dram_store_fixture import CPU_ENV, dram_store


def mapping_for(pointer):
    for line in Path("/proc/self/maps").read_text().splitlines():
        lo,hi=[int(x,16) for x in line.split()[0].split("-")]
        if lo <= pointer < hi:
            return " ".join(line.split()[1:])
    raise AssertionError("unmapped buffer")



class NpuStageDriver:
    """Bounded probe adapter; one arena registration, explicit completion."""
    def __init__(self, device, size):
        import torch
        import torch_npu
        import vllm_ascend.vllm_ascend_C
        self.torch, self.size = torch, size
        torch.npu.set_device(device)
        self.tensor = torch.empty(size,dtype=torch.uint8,device=f"npu:{device}")
        self.stream = torch.npu.Stream(device=device)
        self.acl = ctypes.CDLL("libascendcl.so")
        self.acl.aclrtHostRegister.argtypes = [ctypes.c_void_p,ctypes.c_uint64,ctypes.c_int,ctypes.POINTER(ctypes.c_void_p)]
        self.acl.aclrtHostRegister.restype = ctypes.c_int
        self.acl.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
        self.acl.aclrtHostUnregister.restype = ctypes.c_int

    def register(self, pointer, size):
        alias = ctypes.c_void_p()
        rc = self.acl.aclrtHostRegister(pointer,size,0,ctypes.byref(alias))
        if rc: raise RuntimeError(f"HostRegister: {rc}")

    def unregister(self, pointer):
        rc = self.acl.aclrtHostUnregister(pointer)
        if rc: raise RuntimeError(f"HostUnregister: {rc}")

    def copy(self, pointer, to_host, fill):
        torch = self.torch
        src,dst = (self.tensor.data_ptr(),pointer) if to_host else (pointer,self.tensor.data_ptr())
        with torch.npu.stream(self.stream):
            self.tensor.fill_(fill)
            torch.ops._C_ascend.swap_blocks_batch(
                torch.tensor([src],dtype=torch.int64),torch.tensor([dst],dtype=torch.int64),
                torch.tensor([self.size],dtype=torch.int64),1 if to_host else 0)
            event = torch.npu.Event()
            event.record(self.stream)
        return event

    def produce(self, lease, value): return self.copy(lease.pointer,True,value)
    def consume(self, lease): return self.copy(lease.pointer,False,0)
    def assert_value(self, value): assert bool((self.tensor.cpu() == value).all())
    def synchronize(self): self.stream.synchronize()



class NpuFrameDriver(NpuStageDriver):
    """Mixed dtype / noncontiguous row State frames, not bandwidth measurement."""
    def __init__(self, device):
        super().__init__(device, 0)
        from types import SimpleNamespace as S
        from native_state_frame import FramePlan
        torch = self.torch
        self.states = []
        for name,dtype,width in [("a",torch.bool,3),("b",torch.bfloat16,5),
                                 ("c",torch.float32,7),("d",torch.int64,2)]:
            tensor=torch.empty((4,width),dtype=dtype,device=f"npu:{device}")
            state=S(tensor=tensor,storage_dtype=dtype,block_shape=(width,),
                physical_blocks_per_logical_block=1,leading_physical_blocks=0,
                logical_block_bytes=width*tensor.element_size())
            self.states.append((name,state))
        self.export = FramePlan.build([(n,s,(2,0)) for n,s in self.states])
        self.restore = FramePlan.build([(n,s,(1,3)) for n,s in self.states])
        self.size = self.export.byte_length

    def produce(self, lease, value):
        torch = self.torch
        self.expected = {}
        with torch.npu.stream(self.stream):
            for name,state in self.states:
                cpu=torch.arange(state.tensor.numel()).reshape(state.tensor.shape)+value
                if state.storage_dtype == torch.bool: cpu=cpu%2
                cpu=cpu.to(state.storage_dtype)
                self.expected[name]=cpu[[2,0]].clone()
                state.tensor.copy_(cpu)
            args=self.export.copy_descriptors(lease,to_host=True)
            torch.ops._C_ascend.swap_blocks_batch(*args,1)
            event=torch.npu.Event();event.record(self.stream)
        return event

    def consume(self, lease):
        torch = self.torch
        with torch.npu.stream(self.stream):
            for _,state in self.states:state.tensor.zero_()
            args=self.restore.copy_descriptors(lease,to_host=False)
            torch.ops._C_ascend.swap_blocks_batch(*args,0)
            event=torch.npu.Event();event.record(self.stream)
        return event

    def assert_value(self, value):
        for name,state in self.states:
            assert self.torch.equal(state.tensor.cpu()[[1,3]],self.expected[name])


def main(out, staging=False, npu=None, state_frame=False):
    out=Path(out)
    out.mkdir(exist_ok=False)
    with socket.socket() as reservation:reservation.bind(("127.0.0.1",55206))
    with dram_store(out/"store",55201,segment_bytes=512<<20) as stores:
        from mooncake.store import MooncakeDistributedStore
        env=os.environ.copy()
        env.pop("LD_PRELOAD",None);env.pop("LD_LIBRARY_PATH",None)
        env.update(MC_STORE_LOCAL_HOT_CACHE_SIZE=str(256<<20),
                   MC_STORE_LOCAL_HOT_BLOCK_SIZE=str(32<<20),
                   MC_STORE_LOCAL_HOT_CACHE_USE_SHM="1",
                   MC_STORE_LOCAL_HOT_ADMISSION_THRESHOLD="1")
        command=[str(CPU_ENV/"bin/mooncake_client"),"--host=127.0.0.1:55207",
                 "--port=55206","--metadata_server=http://127.0.0.1:55202/metadata",
                 "--master_server_address=127.0.0.1:55201","--protocol=tcp",
                 "--global_segment_size=0","--local_buffer_size=64MB","--threads=4",
                 "--enable_http_server=false","--enable_offload=false"]
        dummy=None;handle=None;arena=None;driver=None
        with (out/"client.log").open("w") as log:
            server=subprocess.Popen(command,env=env,stdout=log,stderr=log)
            try:
                deadline=time.monotonic()+20
                while True:
                    if server.poll() is not None:raise RuntimeError("native service failed")
                    try:
                        with socket.create_connection(("127.0.0.1",55206),timeout=.2):break
                    except OSError:
                        if time.monotonic()>deadline:raise TimeoutError("native service")
                        time.sleep(.1)
                dummy=MooncakeDistributedStore()
                assert dummy.setup_dummy(64<<20,64<<20,"127.0.0.1:55206")==0
                data=bytes(range(256))*(65536)
                assert stores[0].put("shared-hot-test",data)==0
                observations=[]
                # First read seeds the asynchronous local cache; following reads
                # determine whether the public lease actually points into shm.
                for i in range(4):
                    begin=time.perf_counter();handle=dummy.get_buffer("shared-hot-test")
                    elapsed=time.perf_counter()-begin
                    assert handle is not None and handle.size()==len(data)
                    assert ctypes.string_at(handle.ptr(),handle.size())==data
                    observations.append(dict(seconds=elapsed,mapping=mapping_for(handle.ptr())))
                    handle=None;gc.collect();time.sleep(.05)
                staging_result = None
                if staging:
                    from mooncake.store import MooncakeHostMemAllocator
                    from native_dram_staging import NativeDramStaging
                    size = 16 << 20
                    if npu is not None:
                        driver = NpuFrameDriver(npu) if state_frame else NpuStageDriver(npu, size)
                        size = driver.size
                    arena = NativeDramStaging(dummy,slot_bytes=size,slots=2,
                                              allocator=MooncakeHostMemAllocator(),
                                              register_device=driver.register if driver else None,
                                              unregister_device=driver.unregister if driver else None)
                    cycles = []
                    for i in range(3):
                        lease = arena.acquire(size)
                        if driver:
                            event = driver.produce(lease, 41+i)
                            lease.seal(event.synchronize)
                        else:
                            ctypes.memset(lease.pointer, 41+i, size)
                            lease.seal(lambda: None)
                        start = time.perf_counter()
                        committed = lease.replicate("direct-stage-"+str(i))
                        receipt = committed.result(timeout=20)
                        put_seconds = time.perf_counter()-start
                        reader = arena.acquire(size)
                        ctypes.memset(reader.pointer, 0, size)
                        start = time.perf_counter()
                        rc = dummy.get_into("direct-stage-"+str(i), reader.pointer, size)
                        get_seconds = time.perf_counter()-start
                        assert rc == size, rc
                        if not state_frame:
                            assert ctypes.string_at(reader.pointer, size) == bytes([41+i])*size
                        if driver:
                            event = driver.consume(reader)
                            reader.seal(event.synchronize)
                            driver.assert_value(41+i)
                        else:
                            reader.seal(lambda: None)
                        reader.discard()
                        cycles.append(dict(put_seconds=put_seconds,get_seconds=get_seconds))
                    staging_result = dict(exact=True,cycles=cycles,mapping=mapping_for(arena.base),
                                          fixed_reused_buffer=True,bytes=size,arena_bytes=arena.total_bytes,
                                          device=npu,npu_roundtrip_exact=driver is not None,state_frame=state_frame)
                    arena.close()
                result=dict(scope="native shared-cache/read and optional shared staging; no model or throughput qualification",
                            exact=True,reads=observations,staging=staging_result)
                (out/"summary.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
            finally:
                if driver is not None:
                    driver.synchronize()
                if arena is not None:
                    arena.executor.shutdown(wait=True)
                    if not arena.active:
                        arena.close()
                handle=None;gc.collect()
                if dummy is not None:dummy.close()
                server.terminate()
                try:server.wait(timeout=10)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--staging",action="store_true")
    parser.add_argument("--npu",type=int)
    parser.add_argument("--state-frame",action="store_true")
    args=parser.parse_args()
    if args.npu is not None and not args.staging:parser.error("--npu requires --staging")
    if args.state_frame and args.npu is None:parser.error("--state-frame requires --npu")
    main(args.output,args.staging,args.npu,args.state_frame)
