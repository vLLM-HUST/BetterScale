"""Standalone native route range checker; validates rejection without stopping a server."""
import argparse
import ctypes as C
import json
from pathlib import Path
import statistics

def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import torch
    import torch_npu
    torch.npu.set_device(0)
    lib=C.CDLL(str(a.build/'launch.so'));ptr=C.c_void_p
    lib.load_server.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(ptr),C.POINTER(ptr)]
    lib.launch_blocks.argtypes=[ptr]*5+[C.c_uint32];lib.unload_server.argtypes=[ptr]
    binary,fn=ptr(),ptr()
    assert lib.load_server(str(a.build/'queue_service.o').encode(),b'route_validate_probe',C.byref(binary),C.byref(fn))==0
    results=[]
    for rows in (1,3,96,1023,1024,4096):
        ids=torch.arange(rows*8,device='npu',dtype=torch.int32)%256
        out=torch.zeros(8,device='npu',dtype=torch.int32)
        cfg=torch.tensor([ids.data_ptr(),ids.numel(),out.data_ptr()],device='npu',dtype=torch.int64)
        def call():
            assert lib.launch_blocks(fn,torch.npu.current_stream().npu_stream,cfg.data_ptr(),0,0,1)==0
        call();torch.npu.synchronize();assert out.cpu()[0].item()==1
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph):call()
        checked=0
        for pos in (0,ids.numel()//2,ids.numel()-1):
            for value in (-2147483648,-1,0,255,256,2147483647):
                ids.fill_(0);ids[pos]=value
                graph.replay();torch.npu.synchronize()
                assert out.cpu()[0].item()==int(0<=value<256),(rows,pos,value,out.cpu().tolist())
                checked+=1
        ids.fill_(255)
        batch=torch.npu.NPUGraph()
        with torch.npu.graph(batch):
            for _ in range(64):call()
        batch.replay();torch.npu.synchronize()
        times=[]
        for _ in range(5):
            start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            start.record();batch.replay();end.record();end.synchronize()
            times.append(start.elapsed_time(end)*1000/64)
        results.append(dict(rows=rows,checks=checked,us=times,median_us=statistics.median(times)))
        print(json.dumps(results[-1]),flush=True)
        batch.reset();graph.reset()
    lib.unload_server(binary)
    a.output.write_text(json.dumps(dict(status='PASS',results=results),indent=2))

if __name__=='__main__':main()
