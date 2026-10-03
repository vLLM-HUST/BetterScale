"""Bounded native Mooncake pointer-buffer probe, no HTTP/Python payload bytes."""
import argparse
import json
import statistics
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
import torch_npu
from dram_store_fixture import dram_store


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--parallel",action="store_true")
    p.add_argument("--warm-segment",action="store_true")
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    torch.set_num_threads(8);torch.npu.set_device(0)
    size=64<<20;count=4
    src=[torch.empty(size,dtype=torch.uint8,pin_memory=True) for _ in range(count)]
    dst=[torch.empty_like(x,pin_memory=True) for x in src]
    for i,x in enumerate(src):x.numpy()[:]=37+i;x.numpy()[::4096]=91+i
    rows=[]
    with dram_store(args.output/"store",55051,segment_bytes=2<<30) as stores:
        store=stores[0];registered=[]
        try:
            for x in src+dst:
                assert store.register_buffer(x.data_ptr(),size)==0
                registered.append(x)
            if args.warm_segment:
                warm_keys=[f"prefault-{i}" for i in range(24)]
                for i,key in enumerate(warm_keys):
                    assert store.put_from(key,src[i%count].data_ptr(),size)==0
                for key in warm_keys:
                    assert store.remove(key,force=True)==0
            for repeat in range(4):
                keys=[f"direct-{repeat}-{i}" for i in range(count)]
                for op in ("put","get"):
                    faults=resource.getrusage(resource.RUSAGE_SELF).ru_minflt
                    begin=time.perf_counter()
                    if args.parallel:
                        def transfer(i):
                            return (store.put_from(keys[i],src[i].data_ptr(),size) if op=="put"
                                else store.get_into(keys[i],dst[i].data_ptr(),size))
                        with ThreadPoolExecutor(max_workers=count) as pool:values=list(pool.map(transfer,range(count)))
                    else:
                        values=(store.batch_put_from(keys,[x.data_ptr() for x in src],[size]*count)
                            if op=="put" else store.batch_get_into(keys,[x.data_ptr() for x in dst],[size]*count))
                    seconds=time.perf_counter()-begin
                    if op=="put":assert all(v==0 for v in values),values
                    else:assert all(v>=0 for v in values),values
                    if repeat:rows.append(dict(op=op,seconds=seconds,effective_GB_s=size*count/seconds/1e9,returns=values,minor_faults=resource.getrusage(resource.RUSAGE_SELF).ru_minflt-faults))
                assert all(torch.equal(a,b) for a,b in zip(src,dst)),"pointer byte mismatch"
            summary=dict(scope="native registered pinned CPU buffers and local DRAM Store; no device DMA, HTTP, hash or wire encoding",
                parallel=args.parallel,warm_segment=args.warm_segment,exact=True,samples=rows,median_GB_s={op:statistics.median(v["effective_GB_s"] for v in rows if v["op"]==op) for op in ("put","get")})
            (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
            print(json.dumps(summary),flush=True)
        finally:
            for x in reversed(registered):assert store.unregister_buffer(x.data_ptr())==0


if __name__=="__main__":main()
