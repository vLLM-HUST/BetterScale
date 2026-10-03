"""Bounded two-host TCP transport probe; explicit peer and port only."""
import argparse
import json
import socket
import statistics
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--role",choices=["server","client"],required=True)
p.add_argument("--bind",required=True);p.add_argument("--peer",required=True)
p.add_argument("--port",type=int,default=55481)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
chunk=b"BetterScale-PD-network-probe\0"*32768
chunk=chunk[:1<<20]
nbytes=128<<20
def receive(sock,n,validate=False):
    got=0
    while got<n:
        data=sock.recv(min(len(chunk),n-got))
        if not data:raise EOFError()
        if validate:
            # Stream position, not recv packet boundaries, defines payload.
            pos=got%len(chunk)
            expected=(chunk+chunk)[pos:pos+len(data)]
            assert data==expected
        got+=len(data)
def send(sock):
    sent=0
    while sent<nbytes:
        data=chunk[:min(len(chunk),nbytes-sent)]
        sock.sendall(data);sent+=len(data)
if a.role=="server":
    with socket.socket() as listener:
        listener.settimeout(30);listener.bind((a.bind,a.port));listener.listen(1)
        print("LISTENING",flush=True)
        sock,peer=listener.accept()
        assert peer[0]==a.peer,peer
else:
    sock=socket.socket();sock.settimeout(15);sock.bind((a.bind,0))
    sock.connect((a.peer,a.port))
with sock:
    sock.settimeout(20);sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    timings=[]
    for _ in range(64):
        if a.role=="client":
            start=time.perf_counter();sock.sendall(b"P");receive(sock,1)
            timings.append((time.perf_counter()-start)*1000)
        else:
            receive(sock,1);sock.sendall(b"P")
    transfers=[]
    for direction in ("client_to_server","server_to_client"):
        sender=(direction=="client_to_server")== (a.role=="client")
        start=time.perf_counter()
        if sender:
            send(sock);receive(sock,1)
        else:
            receive(sock,nbytes,True);sock.sendall(b"K")
        elapsed=time.perf_counter()-start
        transfers.append(dict(direction=direction,bytes=nbytes,seconds=elapsed,
                              GiB_per_second=nbytes/elapsed/2**30))
    result=dict(role=a.role,bind=a.bind,peer=a.peer,transfers=transfers,
                median_rtt_ms=statistics.median(timings) if timings else None,
                validation="exact patterned payload in both directions")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
