"""Pinned, preallocated, independently owned H2D/D2H rings; no model or Store."""
import argparse
import json
import statistics
import time
from pathlib import Path
import torch
import torch_npu


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mib",type=int,default=128)
    p.add_argument("--depth",type=int,default=4)
    p.add_argument("--copies",type=int,default=64)
    p.add_argument("--barrier-dir",type=Path)
    p.add_argument("--participants",type=int,default=1)
    args=p.parse_args()
    if not (1<=args.mib<=512 and 1<=args.depth<=8 and args.depth<=args.copies<=256):
        raise ValueError("Unbounded DMA probe")
    torch.set_num_threads(8)
    torch.npu.set_device(args.device)
    size=args.mib*1024**2
    hs=[torch.empty(size,dtype=torch.uint8,pin_memory=True) for _ in range(args.depth)]
    hd=[torch.empty_like(x,pin_memory=True) for x in hs]
    ds=[torch.empty(size,dtype=torch.uint8,device=f"npu:{args.device}") for _ in hs]
    dd=[torch.empty_like(x) for x in ds]
    for i,(source,dest,device_source) in enumerate(zip(hs,hd,ds)):
        source.numpy()[:]=17+i;source.numpy()[::4096]=91+i
        dest.numpy()[:]=173+i;dest.numpy()[::4096]=29+i
        device_source.copy_(dest)
        dest.zero_()
    torch.npu.synchronize()
    streams=[torch.npu.Stream(device=args.device) for _ in range(2)]
    results=[]
    for mode in ("h2d","d2h","duplex"):
        samples=[]
        active=(0,1) if mode=="duplex" else ((0,) if mode=="h2d" else (1,))
        for repeat in range(4):
            if args.barrier_dir:
                prefix=f"{mode}-{repeat}-"
                (args.barrier_dir/f"{prefix}{args.device}.ready").write_text("ready")
                deadline=time.monotonic()+180
                while len(list(args.barrier_dir.glob(prefix+"*.ready")))<args.participants:
                    if time.monotonic()>deadline:raise TimeoutError("DMA cohort barrier")
                    time.sleep(.005)
            starts=[torch.npu.Event(enable_timing=True) for _ in range(2)]
            ends=[torch.npu.Event(enable_timing=True) for _ in range(2)]
            begin=time.perf_counter()
            for direction in active:starts[direction].record(streams[direction])
            for j in range(args.copies):
                slot=j%args.depth
                for direction in active:
                    with torch.npu.stream(streams[direction]):
                        (dd[slot] if direction==0 else hd[slot]).copy_(
                            hs[slot] if direction==0 else ds[slot],non_blocking=True)
            for direction in active:ends[direction].record(streams[direction])
            for direction in active:ends[direction].synchronize()
            wall=time.perf_counter()-begin
            if repeat:
                samples.append(dict(wall_seconds=wall,bytes_per_direction=size*args.copies,
                    effective_GB_s_per_direction=size*args.copies/wall/1e9,
                    stream_ms={("h2d" if d==0 else "d2h"):starts[d].elapsed_time(ends[d]) for d in active}))
        for i in range(args.depth):
            if 0 in active:assert torch.equal(dd[i].cpu(),hs[i]),"H2D byte mismatch"
            if 1 in active:
                a=hd[i].numpy()
                assert (a[::4096]==29+i).all()
                # Check every byte, not merely the sparse sentinels.
                expected=hs[i].numpy().copy();expected[:]=173+i;expected[::4096]=29+i
                assert (a==expected).all(),"D2H byte mismatch"
        results.append(dict(mode=mode,exact=True,samples=samples,
            median_effective_GB_s_per_direction=statistics.median(x["effective_GB_s_per_direction"] for x in samples)))
    receipt=dict(scope="pinned preallocated DMA, independent directions and synchronized cohort; not State gather/scatter, Store or model coexistence",
        participants=args.participants,device=args.device,mib=args.mib,depth=args.depth,copies=args.copies,
        torch=torch.__version__,torch_npu=torch_npu.__version__,results=results)
    args.output.write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":main()
