"""Bounded private control RPCs for rank-owned State; never carries payloads."""
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from threading import BoundedSemaphore,Thread
from betterscale.live.runtime.host_state import HostStateKey

LIMIT=64<<10


def checkpoint(value):
    if (not isinstance(value,list) or len(value)!=2 or not isinstance(value[0],str)
            or not 1<=len(value[0])<=128 or type(value[1]) is not int or value[1]<1):
        raise ValueError("invalid checkpoint incarnation")
    return HostStateKey(*value)


class RankControl:
    def __init__(self,receiver,host,port,allowed_peers):
        self.receiver=receiver
        parent=self
        class Server(ThreadingHTTPServer):
            daemon_threads=False
            request_queue_size=32
            def process_request(self,request,address):
                if not parent.slots.acquire(blocking=False):
                    request.settimeout(5)
                    request.sendall(b"HTTP/1.1 503 Busy\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    request.close();return
                try:super().process_request(request,address)
                except BaseException:
                    parent.slots.release();raise
            def process_request_thread(self,*args):
                try:super().process_request_thread(*args)
                finally:parent.slots.release()
        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup();self.connection.settimeout(30)
            def log_message(self,*args):pass
            def do_POST(self):
                self.connection.settimeout(30)
                if self.client_address[0] not in allowed_peers:
                    self.send_error(403);return
                try:
                    length=int(self.headers.get("Content-Length","0"))
                    if self.path!="/rpc" or not 0<length<=LIMIT:raise ValueError("invalid control envelope")
                    data=json.loads(self.rfile.read(length))
                    result=parent.dispatch(data)
                    payload=json.dumps(dict(ok=True,result=result)).encode()
                    self.send_response(200)
                except (KeyError,TypeError,ValueError) as error:
                    payload=json.dumps(dict(ok=False,error=str(error))).encode();self.send_response(400)
                except Exception as error:
                    payload=json.dumps(dict(ok=False,error=str(error))).encode();self.send_response(409)
                self.send_header("Content-Type","application/json")
                self.send_header("Content-Length",str(len(payload)));self.end_headers()
                self.wfile.write(payload)
        self.slots=BoundedSemaphore(32)
        self.server=Server((host,port),Handler)
        self.host=host
        self.thread=Thread(target=self.server.serve_forever,name="rank-state-control",daemon=True)

    def dispatch(self,data):
        if not isinstance(data,dict) or set(data)!={"op","args"} or not isinstance(data["args"],dict):
            raise ValueError("invalid rank control operation")
        op,a=data["op"],data["args"]
        fields={"info":set(),"begin":{"checkpoint","objects"},
                "prepare":{"source","checkpoint","oid","size","digest","operation"},
                "commit":{"operation"}}
        if op not in fields or set(a)!=fields[op]:raise ValueError("unqualified rank control operation")
        r=self.receiver
        if op=="info":
            return dict(owner=r.pool.owner,endpoint=f"{self.host}:{r.engine.get_rpc_port()}")
        if op=="begin":return r.begin(checkpoint(a["checkpoint"]),a["objects"])
        if op=="prepare":
            return r.prepare(a["source"],checkpoint(a["checkpoint"]),a["oid"],a["size"],a["digest"],a["operation"])
        return r.commit(a["operation"])

    def start(self):
        self.thread.start();return self

    def close(self):
        # The owner first drains outgoing work and incoming commits; no timed
        # shutdown is allowed to make a remotely writable address reusable.
        with self.receiver.lock:
            if self.receiver.pending:raise RuntimeError("incoming rank State not drained")
            self.receiver.accepting=False
        self.server.shutdown();self.thread.join();self.server.server_close()
