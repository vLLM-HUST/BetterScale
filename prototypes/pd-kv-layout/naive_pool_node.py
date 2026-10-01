"""Bounded naive PD node: one local DRAM Store and native model actors.

Control/data HTTP accepts only explicitly configured peers. No network pickle,
arbitrary utility calls, retry after unknown model completion, or HA claim.
"""
import argparse
import asyncio
from contextlib import ExitStack
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time

import msgpack
from aiohttp import web

from pd_limits import context_limit,checkpoint_limit

MAX_STATE=checkpoint_limit()

def pack(value):
    return msgpack.packb(value,use_bin_type=True)

def unpack(data):
    return msgpack.unpackb(data,raw=False,strict_map_key=False,max_bin_len=MAX_STATE,
                           max_array_len=context_limit()+8192,max_map_len=8192,max_str_len=1<<20)

def validate_rpc(kind,body):
    if not isinstance(body,dict) or set(body)!={"instance","op","args"}:
        raise ValueError("Malformed RPC")
    instance,op,args=body["instance"],body["op"],body["args"]
    dp=1 if kind=="P" else 4
    if type(instance) is not int or not 0<=instance<(4 if kind=="P" else 1):
        raise ValueError("Invalid instance")
    fields={"generate_batch":{"items"},"export":{"owner","tokens","salt"},
            "import":{"owner","key","salt"},"drop":{"owner","salt"}}
    if not isinstance(op,str) or op not in fields or not isinstance(args,dict) or set(args)!=fields[op]:
        raise ValueError("Unqualified RPC/arguments")
    def owner_salt(item):
        if type(item["owner"]) is not int or not 0<=item["owner"]<dp:
            raise ValueError("Invalid attention owner")
        if not isinstance(item["salt"],str) or not 1<=len(item["salt"])<=128:
            raise ValueError("Invalid cache salt")
    def tokens(value):
        if (not isinstance(value,list) or not 1<=len(value)<=context_limit()
                or any(type(t) is not int or not 0<=t<248320 for t in value)):
            raise ValueError("Invalid token sequence")
    if op=="generate_batch":
        items=args["items"]
        if not isinstance(items,list) or not 1<=len(items)<=16*dp:
            raise ValueError("Invalid batch")
        counts=[0]*dp;salts=set()
        for item in items:
            if not isinstance(item,dict) or set(item)!={"owner","tokens","salt","n"}:
                raise ValueError("Invalid batch item")
            owner_salt(item);tokens(item["tokens"])
            if (type(item["n"]) is not int or not 1<=item["n"]<=512
                    or len(item["tokens"])+item["n"]>context_limit() or item["salt"] in salts):
                raise ValueError("Invalid request envelope/duplicate writer")
            salts.add(item["salt"]);counts[item["owner"]]+=1
        if max(counts)>16:raise ValueError("Owner capacity exceeded")
    else:
        owner_salt(args)
        if op=="export":tokens(args["tokens"])
        if op=="import" and (not isinstance(args["key"],str) or len(args["key"])!=64
                or any(c not in "0123456789abcdef" for c in args["key"])):
            raise ValueError("Invalid checkpoint key")
    return instance,op,args

class Actor:
    def __init__(self,kind,instance):
        from native_pool_service_actor import worker
        context=mp.get_context("spawn")
        self.pipe,child=context.Pipe()
        self.process=context.Process(target=worker,args=(kind,instance,child),
                                     name=f"pd-{kind}{instance}")
        self.process.start();child.close()
        self.lock=asyncio.Lock()
        self.quarantined=False
        self.info=None

    def _receive(self,seconds):
        if not self.pipe.poll(seconds):raise TimeoutError("Actor response deadline")
        status,value=self.pipe.recv()
        if status not in ("ok","ready"):raise RuntimeError(value)
        return value

    async def ready(self):
        self.info=await asyncio.to_thread(self._receive,1200)
        return self.info

    async def call(self,op,args,timeout=1800):
        async with self.lock:
            if self.quarantined or not self.process.is_alive():
                raise RuntimeError("Actor quarantined/dead; no automatic replay")
            def exchange():
                self.pipe.send((op,args))
                return self._receive(timeout)
            try:
                return await asyncio.wait_for(asyncio.to_thread(exchange),timeout+5)
            except BaseException:
                # Timeout/cancellation cannot prove whether a mutation executed.
                self.quarantined=True
                raise

    async def close(self):
        if self.process.is_alive() and not self.quarantined:
            try:await self.call("stop",{},timeout=30)
            except Exception:pass
        await asyncio.to_thread(self.process.join,60)
        if self.process.is_alive():
            # Worker creates a dedicated session before spawning Core/workers.
            if os.getpgid(self.process.pid)!=self.process.pid:
                raise RuntimeError("Cannot safely retire actor process group")
            os.killpg(self.process.pid,signal.SIGTERM)
            await asyncio.to_thread(self.process.join,20)
            if self.process.is_alive():
                os.killpg(self.process.pid,signal.SIGKILL)
                await asyncio.to_thread(self.process.join,10)
        self.pipe.close()

class Cache:
    def __init__(self,store):
        self.store=store
        self.lock=asyncio.Lock()

    async def put(self,key,data):
        if not 0<len(data)<=MAX_STATE or hashlib.sha256(data).hexdigest()!=key:
            raise ValueError("Checkpoint content identity/size mismatch")
        value=unpack(data)
        from model_checkpoint import IDENTITY
        if value.get("header",{}).get("identity")!=IDENTITY or len(value.get("shards",[]))!=2:
            raise ValueError("Checkpoint is not the admitted TP2 target representation")
        async with self.lock:
            def publish():
                chunks=[]
                for offset in range(0,len(data),8<<20):
                    chunk=data[offset:offset+(8<<20)]
                    chunk_key="chunk:"+hashlib.sha256(chunk).hexdigest()
                    rc=self.store.put(chunk_key,chunk)
                    if rc!=0:raise RuntimeError(f"DRAM chunk put failed: {rc}")
                    chunks.append((chunk_key,len(chunk)))
                # A generation becomes discoverable only after every chunk ack.
                manifest=pack(dict(size=len(data),chunks=chunks))
                rc=self.store.put("manifest:"+key,manifest)
                if rc!=0:raise RuntimeError(f"DRAM manifest put failed: {rc}")
            await asyncio.to_thread(publish)

    async def get(self,key):
        async with self.lock:
            def read():
                raw=self.store.get("manifest:"+key)
                if not isinstance(raw,bytes) or not raw:
                    raise KeyError("Checkpoint manifest evicted/missing")
                manifest=unpack(raw)
                if not 0<manifest["size"]<=MAX_STATE:
                    raise ValueError("Invalid checkpoint manifest")
                chunks=[]
                for chunk_key,size in manifest["chunks"]:
                    chunk=self.store.get(chunk_key)
                    if not isinstance(chunk,bytes) or len(chunk)!=size:
                        raise KeyError("Checkpoint chunk evicted/missing")
                    if "chunk:"+hashlib.sha256(chunk).hexdigest()!=chunk_key:
                        raise ValueError("Checkpoint chunk corrupted")
                    chunks.append(chunk)
                data=b"".join(chunks)
                if len(data)!=manifest["size"] or hashlib.sha256(data).hexdigest()!=key:
                    raise ValueError("Checkpoint content identity mismatch")
                return data
            return await asyncio.to_thread(read)

class Node:
    def __init__(self,args,store):
        self.args=args;self.cache=Cache(store);self.actors=[]
        self.ready=False

    async def start(self,app):
        count=4 if self.args.kind=="P" else 1
        self.actors=[Actor(self.args.kind,i) for i in range(count)]
        receipts=await asyncio.gather(*(a.ready() for a in self.actors))
        self.ready=True
        (self.args.output/"ready.json").write_text(json.dumps(receipts,indent=2))

    async def close(self,app):
        self.ready=False
        await asyncio.gather(*(a.close() for a in self.actors))

    async def health(self,request):
        return web.json_response(dict(ready=self.ready,kind=self.args.kind,context_limit=context_limit(),
            checkpoint_limit=MAX_STATE,
            actors=[dict(info=a.info,alive=a.process.is_alive(),quarantined=a.quarantined)
                    for a in self.actors]))

    async def checkpoint(self,request):
        key=request.match_info["key"]
        if len(key)!=64 or any(c not in "0123456789abcdef" for c in key):
            raise web.HTTPBadRequest()
        if request.method=="PUT":
            data=await request.read()
            await self.cache.put(key,data)
            return web.json_response(dict(key=key,bytes=len(data)))
        data=await self.cache.get(key)
        return web.Response(body=data,content_type="application/msgpack")

    async def rpc(self,request):
        body=unpack(await request.read())
        instance,op,args=validate_rpc(self.args.kind,body)
        if op=="import":
            args=dict(args);key=args.pop("key")
            args["payload"]=unpack(await self.cache.get(key))
        result=await self.actors[instance].call(op,args)
        if op=="export":
            data=pack(result);key=hashlib.sha256(data).hexdigest()
            await self.cache.put(key,data)
            result=dict(key=key,bytes=len(data),cursor=result["header"]["cursor"])
        return web.Response(body=pack(result),content_type="application/msgpack")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",choices=("P","D"),required=True)
    parser.add_argument("--bind",required=True)
    parser.add_argument("--peer",action="append",required=True)
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cache-gib",type=int,default=16)
    args=parser.parse_args()
    if not 1<=args.cache_gib<=512:raise ValueError("Cache budget must be1..512GiB")
    args.output.mkdir(parents=True,exist_ok=False)
    peers=set(args.peer)|{"127.0.0.1"}
    @web.middleware
    async def boundary(request,handler):
        if request.remote not in peers:raise web.HTTPForbidden()
        try:return await handler(request)
        except web.HTTPException:raise
        except KeyError as error:raise web.HTTPNotFound(text=str(error))
        except ValueError as error:raise web.HTTPBadRequest(text=str(error))
        except Exception as error:
            return web.Response(status=500,body=pack(dict(error=str(error))),
                                content_type="application/msgpack")
    from dram_store_fixture import dram_store
    with dram_store(args.output/"store",55401,segment_bytes=args.cache_gib<<30) as stores:
        node=Node(args,stores[0])
        app=web.Application(client_max_size=MAX_STATE,middlewares=[boundary])
        app.router.add_get("/health",node.health)
        app.router.add_get("/checkpoint/{key}",node.checkpoint)
        app.router.add_put("/checkpoint/{key}",node.checkpoint)
        app.router.add_post("/rpc",node.rpc)
        app.on_startup.append(node.start);app.on_cleanup.append(node.close)
        web.run_app(app,host=args.bind,port=args.port,access_log=None,
                    shutdown_timeout=30,handler_cancellation=False)

if __name__=="__main__":main()
