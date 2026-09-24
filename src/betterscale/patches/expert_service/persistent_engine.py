"""Geometry adapter for the existing BF16 two-slot persistent AIV/AIC engine."""
import ctypes as C
import hashlib,json,os
from pathlib import Path
import torch
import torch_npu
from .checkpoint import H,M,K,E
from .model_geometry import GEOMETRY as G


def geometry():
    root=Path(os.environ['BETTERSCALE_EXPERT_PERSISTENT_BUILD']);abi=json.loads((root/'abi.json').read_text())
    assert (abi['version'],abi['hidden'],abi['inner'],abi['topk'],abi['experts'],abi['sources'])==(1,H,M,K,E,7)
    assert abi['layer_count']==G.total_layers and abi['model']==f'{G.name}-bf16-persistent'
    assert abi['rows'] in (1024,4096) and (abi['rows']==1024 or abi.get('combined_return'))
    assert type(abi['persistent_launch_timeout_us']) is int
    lifetime=abi.get('service_lifetime_seconds',abi['persistent_launch_timeout_us']//1000000)
    assert 1200 <= lifetime <= 14400
    assert abi['persistent_launch_timeout_us']==lifetime*1000000 or (abi['persistent_launch_timeout_us']==0 and abi.get('external_watchdog') is True)
    assert abi['persistent_launch_timeout_us'] % 1000000 == 0
    return root,abi

def return_mode():
    mode=os.environ.get('BETTERSCALE_EXPERT_RETURN_MODE','pull');assert mode in ('pull','push')
    return mode

def aligned(n):return (n+(2<<20)-1)//(2<<20)*(2<<20)

def extents(abi):
    return aligned(abi['payload_words']*4+abi['rows']*H*2),aligned(256+abi['rows']*(1 if abi.get('combined_return') else K)*H*2)

class Kernels:
    def __init__(self):
        self.root,self.abi=geometry();self.identity={name:hashlib.sha256((self.root/name).read_bytes()).hexdigest() for name in ('abi.json','launch.so','persistent_vector.o','persistent_cube.o','queue_service.o')}
        self.lib=C.CDLL(str(self.root/'launch.so'));p=C.c_void_p
        self.lib.load_server.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(p),C.POINTER(p)]
        self.lib.launch_blocks.argtypes=[p]*5+[C.c_uint32]
        self.lib.launch_cube.argtypes=[p]*5+[C.c_uint32]
        self.lib.unload_server.argtypes=[p];self.binaries=[]
    def load(self,name,unit='queue_service'):
        binary,fn=C.c_void_p(),C.c_void_p()
        assert self.lib.load_server(str(self.root/f'{unit}.o').encode(),name.encode(),C.byref(binary),C.byref(fn))==0,name
        self.binaries.append(binary);return fn
    def call(self,fn,config,x,ids,blocks=1,cube=False):
        launch=self.lib.launch_cube if cube else self.lib.launch_blocks
        rc=launch(fn,torch.npu.current_stream().npu_stream,config.data_ptr(),
                  x.data_ptr() if x is not None else 0,ids.data_ptr() if ids is not None else 0,blocks)
        assert rc==0,rc
    def close(self):
        for binary in self.binaries:assert self.lib.unload_server(binary)==0

class Engine:
    def __init__(self,sources,outputs,weights):
        self.kernels=Kernels();a=self.kernels.abi
        s,rows=a['sources'],a['rows'];capacity=s*rows*K
        assert len(sources)==len(outputs)==s
        self.weights=weights
        self.fine_pack=os.environ.get('BETTERSCALE_EXPERT_FINE_PACK','1')=='1'
        assert not a.get('pipelined_copy') or not self.fine_pack,'pipelined movement requires full-PACK completion, not per-row readiness'
        self.combined_return=a.get('combined_return',False)
        for up,down in weights.values():
            assert up.shape==(E,H,2*M) and down.shape==(E,M,H)
            assert int(torch_npu.get_npu_format(up))==int(torch_npu.get_npu_format(down))==29
        self.weight_table=torch.tensor([[weights[l][0].data_ptr(),weights[l][1].data_ptr()] if l in weights else [0,0] for l in range(G.total_layers)],dtype=torch.int64,device='npu')
        self.control=torch.zeros(128,16,dtype=torch.int32,device='npu')
        self.trace=torch.zeros(64,32,dtype=torch.int32,device='npu')
        self.events=torch.zeros(a.get('event_capacity',512),8,dtype=torch.int64,device='npu')
        self.pack_ready=torch.zeros(2,capacity,16,dtype=torch.int32,device='npu')
        self.slots=[];table=[]
        for _ in range(2):
            slot=[torch.empty(s,rows,H,dtype=torch.bfloat16,device='npu'),
                  torch.empty(capacity,H,dtype=torch.bfloat16,device='npu'),
                  torch.empty(capacity,2*M,dtype=torch.bfloat16,device='npu'),
                  torch.empty(capacity,M,dtype=torch.bfloat16,device='npu'),
                  torch.empty(capacity,H,dtype=torch.bfloat16,device='npu'),
                  torch.zeros(s,rows*K+8,dtype=torch.int32,device='npu'),
                  torch.zeros(E,dtype=torch.int64,device='npu')]
            for k,n in [(H,2*M),(M,H)]:
                slot.append(torch.tensor([k,n,E,0,slot[6].data_ptr(),capacity],dtype=torch.int64,device='npu'))
            slot.append(torch.zeros(128,dtype=torch.int64,device='npu')) # donor prefix boundary at [127]
            ids=torch.empty(s,rows*K,dtype=torch.int32,device='npu');slot.append(ids)
            table.append([x.data_ptr() for x in slot[:10]]+[0]*5+[ids.data_ptr()]);self.slots.append(slot)
        self.table=torch.tensor(table,dtype=torch.int64,device='npu')
        self.source_pointers=torch.tensor(sources,dtype=torch.int64,device='npu')
        self.output_pointers=torch.tensor(outputs,dtype=torch.int64,device='npu')
        # Existing cfg: whole-layer ownership has all64 local experts (owner=0).
        values=[self.control.data_ptr(),self.table.data_ptr(),0,0,0,0,32,0,
                self.trace.data_ptr(),20000000,0,self.events.data_ptr(),0,0,
                2,0,0,0,1,0,self.pack_ready.data_ptr() if self.fine_pack else 0,0,0 if self.combined_return else 1,0,1,
                self.weight_table.data_ptr(),G.total_layers,self.source_pointers.data_ptr(),self.output_pointers.data_ptr()]
        assert len(values)==29
        self.config=torch.tensor(values,dtype=torch.int64,device='npu')
        self.streams=[];self.graphs=[];self.functions=[]
        self.server_graph=os.environ.get('BETTERSCALE_EXPERT_PERSISTENT_SERVER_GRAPH','1')=='1'
        for name,blocks,cube in [('persistent_vector',17,False),('persistent_cube',24,True)]:
            fn=self.kernels.load(name,name);stream=torch.npu.Stream()
            if self.server_graph:
                graph=torch.npu.NPUGraph()
                with torch.npu.stream(stream):
                    with torch.npu.graph(graph):self.kernels.call(fn,self.config,None,None,blocks,cube)
                self.graphs.append(graph)
            self.streams.append(stream);self.functions.append((fn,blocks,cube))
        self.config[10]=1;torch.npu.synchronize()
    def start(self):
        for index,(stream,(fn,blocks,cube)) in enumerate(zip(self.streams,self.functions)):
            with torch.npu.stream(stream):
                if self.server_graph:self.graphs[index].replay()
                else:self.kernels.call(fn,self.config,None,None,blocks,cube)
    def finish(self):
        for stream in self.streams:stream.synchronize()
        c=self.control.cpu().tolist();assert c[0][0]==c[43][0]==1,c[:3]+c[43:44]
        return dict(completed_counts=c[43][4:11],waves=c[43][1],pulls_during_cube=c[43][2],
                    admitted_promotions=c[43][11],events=self.events.cpu().tolist(),
                    trace=self.trace.cpu().tolist(),host_forward_requests=0,persistent_graph_launches=2 if self.server_graph else 0,
                    persistent_kernel_launches=2,server_launch_mode='graph' if self.server_graph else 'direct',
                    internal_pipeline=True,fine_pack=self.fine_pack,early_down=True,early_return=not self.combined_return,combined_return=self.combined_return)
    def close(self):
        for graph in self.graphs:graph.reset()
        self.kernels.close()
