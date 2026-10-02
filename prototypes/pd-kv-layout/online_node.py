"""Private P4/D4 online node, multiplexed control and shared immutable objects."""
import argparse
import asyncio
import json
import time
from pathlib import Path
from aiohttp import web
from naive_pool_node import pack,unpack,validate_rpc,Node as BaseNode
from online_actor import Actor
from online_objects import Objects,LIMIT
from pd_limits import context_limit


def validate(kind,body):
    if not isinstance(body,dict) or set(body)!={"instance","op","args"}:raise ValueError("Bad RPC")
    instance,op,args=body["instance"],body["op"],body["args"]
    if type(instance) is not int or not 0<=instance<(4 if kind=="P" else 1):raise ValueError("Bad instance")
    if op=="generate":
        if not isinstance(args,dict) or set(args)!={"owner","tokens","salt","n"}:raise ValueError("Bad generation")
        validate_rpc(kind,dict(instance=instance,op="generate_batch",args={"items":[args]}))
    elif op=="audit":
        if (not isinstance(args,dict) or set(args)!={"owner","enabled"}
                or type(args["owner"]) is not int or args["owner"]!=0
                or type(args["enabled"]) is not bool):raise ValueError("Bad audit RPC")
    elif op=="profile":
        if (not isinstance(args,dict) or set(args)!={"owner","start"}
                or type(args["owner"]) is not int or args["owner"]!=0
                or type(args["start"]) is not bool):raise ValueError("Bad profile RPC")
    elif op=="cache":
        if not isinstance(args,dict) or set(args)!={"owner","command"}:raise ValueError("Bad cache RPC")
        if type(args["owner"]) is not int or not 0<=args["owner"]<(1 if kind=="P" else 4):raise ValueError("Bad owner")
        c=args["command"]
        fields={"capacity":set(),"snapshot":set(),"wait":{"operation"},"describe":{"key"},"drop":{"key"},
                "store_match":{"tokens","salt","key"},"adopt":{"checkpoint"},"load_match":{"key"}}
        if not isinstance(c,dict) or c.get("kind") not in fields or set(c)!={"kind"}|fields[c["kind"]]:raise ValueError("Bad cache command")
        if "key" in c and (not isinstance(c["key"],str) or not 1<=len(c["key"])<=128):raise ValueError("Bad key")
        if "operation" in c and (type(c["operation"]) is not int or c["operation"]<1):raise ValueError("Bad operation")
        if c["kind"]=="store_match":
            validate_rpc(kind,dict(instance=instance,op="export",args=dict(owner=args["owner"],tokens=c["tokens"],salt=c["salt"])))
        if c["kind"]=="adopt":
            cp=c["checkpoint"]
            if not isinstance(cp,dict) or set(cp)!={"key","tokens","salt","block_count","byte_length","pages"}:raise ValueError("Bad checkpoint")
            validate_rpc(kind,dict(instance=instance,op="export",args=dict(owner=args["owner"],tokens=cp["tokens"],salt=cp["salt"])))
    else:raise ValueError("Bad operation")
    return instance,op,args


class Node(BaseNode):
    async def start(self,app):
        self.actors=[Actor(self.args.kind,i) for i in range(4 if self.args.kind=="P" else 1)]
        receipts=await asyncio.gather(*(a.ready() for a in self.actors))
        self.ready=True
        (self.args.output/"ready.json").write_text(json.dumps(receipts,indent=2))

    async def generate(self,request):
        from online_stream import Progress
        import struct
        instance,op,args=validate(self.args.kind,unpack(await request.read()))
        if op!="generate":raise ValueError("Only generation can stream")
        progress=Progress()
        async def run():
            try:
                value=await self.actors[instance].call(op,args,
                    lambda value:progress.push(dict(kind="tokens",value=value)))
                progress.push(dict(kind="done",value=value));progress.finish()
            except BaseException as error:progress.fail(error)
        task=asyncio.create_task(run())
        response=web.StreamResponse(headers={"Content-Type":"application/octet-stream"})
        try:
            await response.prepare(request)
            while (value:=await progress.read()) is not None:
                if value["kind"]=="tokens":
                    value["value"]["node_send_ns"]=time.perf_counter_ns()
                data=pack(value)
                await response.write(struct.pack("!I",len(data))+data)
        except (ConnectionError,RuntimeError,asyncio.CancelledError):
            # Do not turn network backpressure into native-model cancellation.
            # This handler remains responsible for draining its generation.
            progress.detach()
            raise
        finally:
            await asyncio.shield(task)
        await response.write_eof()
        return response

    async def rpc(self,request):
        instance,op,args=validate(self.args.kind,unpack(await request.read()))
        result=await self.actors[instance].call(op,args)
        if op=="audit":self.actors[instance].info["object_audit"]=args["enabled"]
        return web.Response(body=pack(result),content_type="application/msgpack")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind",choices=("P","D"),required=True)
    p.add_argument("--bind",required=True);p.add_argument("--peer",action="append",required=True)
    p.add_argument("--port",type=int,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cache-gib",type=int,default=512)
    args=p.parse_args()
    if not 1<=args.cache_gib<=512:raise ValueError("Cache budget out of range")
    args.output.mkdir(parents=True,exist_ok=False)
    peers=set(args.peer)|{args.bind,"127.0.0.1"}
    @web.middleware
    async def boundary(request,handler):
        if request.remote not in peers:raise web.HTTPForbidden()
        try:return await handler(request)
        except web.HTTPException:raise
        except (ValueError,KeyError,TypeError) as exc:raise web.HTTPBadRequest(text=str(exc))
    from dram_store_fixture import dram_store
    with dram_store(args.output/"store",55401,segment_bytes=args.cache_gib<<30) as stores:
        node=Node(args,stores[0]);objects=Objects(stores[0])
        app=web.Application(client_max_size=LIMIT,middlewares=[boundary])
        app.router.add_get("/health",node.health)
        app.router.add_get("/objects/{key}",objects.route)
        app.router.add_put("/objects/{key}",objects.route)
        app.router.add_post("/rpc",node.rpc)
        app.router.add_post("/generate",node.generate)
        app.on_startup.append(node.start);app.on_cleanup.append(node.close)
        web.run_app(app,host=args.bind,port=args.port,access_log=None,
                    shutdown_timeout=60,handler_cancellation=False)

if __name__=="__main__":main()
