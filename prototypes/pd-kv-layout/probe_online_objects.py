"""Bounded CPU-only object service probe; leaves opaque payloads to shared LRU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from online_objects import PeerObjectSink


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local",required=True);p.add_argument("--remote",required=True)
    p.add_argument("--mib",type=int,default=80);p.add_argument("--parallel",type=int,default=4)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.mib<=128 or not 1<=a.parallel<=8:raise ValueError("Probe bound exceeded")
    if a.output.exists():raise ValueError("Output already exists")
    sink=PeerObjectSink([a.local,a.remote]);rows=[]
    # Unique content avoids measuring Store chunk deduplication instead of PUT.
    values=[(hashlib.sha256(("cpu-object-probe:"+uuid.uuid4().hex).encode()).hexdigest(),
             os.urandom(a.mib<<20)) for _ in range(a.parallel)]
    def call(op,url,key,data):
        start=time.perf_counter()
        if op=="PUT":sink._put(url,key,data)
        else:
            result=sink.request(url,key,op)
            if op=="GET" and result!=data:raise RuntimeError("Payload mismatch")
            if result is None:raise RuntimeError("Missing probe object")
        end=time.perf_counter()
        return dict(op=op,url=url,bytes=len(data) if op!="HEAD" else 0,seconds=end-start)
    for url in sink.urls:
        for op in ("PUT","HEAD","GET"):
            rows.append(dict(mode="serial",**call(op,url,*values[0])))
    # The first key is already cached; new identities/content for every writer.
    values=[(hashlib.sha256(("cpu-object-probe:"+uuid.uuid4().hex).encode()).hexdigest(),
             os.urandom(a.mib<<20)) for _ in range(a.parallel)]
    parallel=[]
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        for url in sink.urls:
            for op in ("PUT","GET"):
                start=time.perf_counter()
                result=list(pool.map(lambda pair:call(op,url,*pair),values))
                elapsed=time.perf_counter()-start
                parallel.append(dict(op=op,url=url,seconds=elapsed,
                    MiB_s=a.parallel*a.mib/elapsed,requests=result))
    answer=dict(mib=a.mib,parallelism=a.parallel,serial=rows,parallel=parallel,
        retained_payload_MiB=(a.parallel+1)*a.mib,
        scope="Opaque CPU payloads, no NPU DMA; Store/client/network combined. Isolated from model requests.")
    a.output.write_text(json.dumps(answer,indent=2))
    print(json.dumps(answer))

if __name__=="__main__":main()
