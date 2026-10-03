"""Bounded native two-host DRAM fixture; explicit owned addresses, no model."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

BIN=Path("/workspace/pd-kv-layout-results/store-cpu-venv/bin")
HOSTS={"D":"10.244.2.32","P":"10.244.1.16"}

def ready(host,port,child,seconds=20):
    until=time.monotonic()+seconds
    while True:
        if child.poll() is not None:raise RuntimeError("owned native service exited")
        try:
            with socket.create_connection((host,port),timeout=.2):return
        except OSError:
            if time.monotonic()>until:raise TimeoutError(f"{host}:{port}")
            time.sleep(.1)

def run(role,out,cache_gib=None,ttl_ms=2000,lifetime=300):
    out.mkdir(exist_ok=False)
    ip=HOSTS[role];children=[];logs=[]
    env=os.environ.copy()
    env.pop("LD_PRELOAD",None);env.pop("LD_LIBRARY_PATH",None)
    env["MC_STORE_LOCAL_HOT_CACHE_SIZE"]="0"
    try:
        ports=(55301,55302,55303,55306,55307) if role=="D" else (55306,55307)
        for port in ports:
            with socket.socket() as s:s.bind((ip,port))
        def spawn(name,args):
            log=(out/(name+".log")).open("w");logs.append(log)
            child=subprocess.Popen([str(BIN/name),*args],stdout=log,stderr=log,env=env)
            children.append(child);return child
        if role=="D":
            master=spawn("mooncake_master",[
                "--rpc_address="+ip,"--rpc_port=55301",
                "--enable_http_metadata_server=true","--http_metadata_server_host="+ip,
                "--http_metadata_server_port=55302","--metrics_host="+ip,
                "--metrics_port=55303","--enable_metric_reporting=false",
                "--default_kv_lease_ttl="+str(ttl_ms),"--enable_offload=false"])
            ready(ip,55301,master)
        client=spawn("mooncake_client",[
            "--host="+ip+":55307","--port=55306",
            "--metadata_server=http://10.244.2.32:55302/metadata",
            "--master_server_address=10.244.2.32:55301","--protocol=tcp",
            "--global_segment_size="+(str(cache_gib)+"GB" if cache_gib else "512MB"),"--local_buffer_size=64MB","--threads=4",
            "--enable_http_server=false","--enable_offload=false"])
        ready(ip,55306,client)
        (out/"ready.json").write_text(json.dumps(dict(role=role,pids=[p.pid for p in children],ttl_ms=ttl_ms,cache_gib=cache_gib)))
        until=time.monotonic()+lifetime
        while not (out/"stop").exists() and time.monotonic()<until:
            if any(p.poll() is not None for p in children):raise RuntimeError("owned native service failed")
            time.sleep(.5)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        for log in logs:log.close()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role",choices=HOSTS,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cache-gib",type=int)
    p.add_argument("--ttl-ms",type=int,default=2000)
    p.add_argument("--lifetime",type=int,default=300)
    a=p.parse_args()
    if a.cache_gib is not None and not 1<=a.cache_gib<=512:p.error("bounded cache requires1..512GiB")
    if not 2000<=a.ttl_ms<=120000 or not 1<=a.lifetime<=14400:p.error("bounded TTL/lifetime required")
    run(a.role,a.output,a.cache_gib,a.ttl_ms,a.lifetime)
