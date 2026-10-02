"""Private exact-token SSE ingress; output completion is not checkpoint commit."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from aiohttp import web
from online_coordinator import Coordinator
from online_stream import Progress
from pd_limits import context_limit


def validate(body,model):
    fields={"model","prompt","max_tokens","ignore_eos","temperature","seed","stream",
            "stream_options","return_token_ids","cache_salt"}
    if not isinstance(body,dict) or set(body)-fields:raise ValueError("Unsupported completion fields")
    prompt=body.get("prompt");n=body.get("max_tokens");salt=body.get("cache_salt")
    if (body.get("model")!=model or body.get("stream") is not True
            or body.get("ignore_eos") is not True or body.get("return_token_ids") is not True
            or body.get("temperature")!=0 or body.get("stream_options")!={"include_usage":True}):
        raise ValueError("Only greedy fixed-budget exact-token streaming is qualified")
    if (not isinstance(prompt,list) or not prompt
            or any(type(t) is not int or not 0<=t<248320 for t in prompt)
            or type(n) is not int or n<1 or len(prompt)+n>context_limit()
            or not isinstance(salt,str) or not 1<=len(salt)<=512):
        raise ValueError("Invalid prompt, budget or cache salt")
    return hashlib.sha256(salt.encode()).hexdigest(),prompt,n


def application(coordinator,model,peers):
    tasks=set()
    async def completion(request):
        if request.remote not in peers:raise web.HTTPForbidden()
        try:session,prompt,n=validate(await request.json(),model)
        except (ValueError,TypeError) as error:raise web.HTTPBadRequest(text=str(error))
        progress=Progress()
        async def run():
            try:
                value=await coordinator.submit(session,prompt,n,
                    on_tokens=lambda item:progress.push(("tokens",item["token_ids"])),
                    output_ready=True)
                progress.push(("done",value));progress.finish()
            except BaseException as error:progress.fail(error)
        task=asyncio.create_task(run());tasks.add(task);task.add_done_callback(tasks.discard)
        response=web.StreamResponse(headers={"Content-Type":"text/event-stream",
            "Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        async def send(value):
            await response.write(("data: "+json.dumps(value,separators=(",",":"))+"\n\n").encode())
        try:
            await response.prepare(request)
            await send(dict(choices=[dict(index=0,prompt_token_ids=prompt,token_ids=[])]))
            while (item:=await progress.read()) is not None:
                kind,value=item
                if kind=="tokens":
                    await send(dict(choices=[dict(index=0,token_ids=value)]))
                else:
                    await send(dict(choices=[dict(index=0,token_ids=[],finish_reason="length")],
                        usage=dict(prompt_tokens=len(prompt),completion_tokens=n,total_tokens=len(prompt)+n,
                            prompt_tokens_details=dict(cached_tokens=value["cached"]))))
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
        except (ConnectionError,asyncio.CancelledError):
            progress.detach()
            # submit is shielded; backend generation/commit remains owned.
            raise
        except Exception as error:
            progress.detach()
            try:await send(dict(error=dict(message=str(error)[:256])))
            except ConnectionError:pass
        return response

    async def health(request):
        if request.remote not in peers:raise web.HTTPForbidden()
        return web.json_response(dict(ready=coordinator.failure is None,
            failure=coordinator.failure,inflight=len(coordinator.inflight)))

    async def startup(app):await coordinator.start()
    async def cleanup(app):
        await asyncio.gather(*tasks,return_exceptions=True)
        await coordinator.close()
    app=web.Application(client_max_size=8<<20)
    app.router.add_post("/v1/completions",completion);app.router.add_get("/health",health)
    app.on_startup.append(startup);app.on_cleanup.append(cleanup)
    return app


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bind",required=True);p.add_argument("--port",type=int,required=True)
    p.add_argument("--peer",action="append",default=[])
    p.add_argument("--p-url",required=True);p.add_argument("--d-url",required=True)
    p.add_argument("--model",default="Qwen3.5-35B-A3B")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/"control.jsonl").open("w",buffering=1) as log:
        def trace(value):log.write(json.dumps(value,separators=(",",":"))+"\n")
        coordinator=Coordinator(a.output/"sessions.sqlite",a.p_url,a.d_url,trace)
        web.run_app(application(coordinator,a.model,set(a.peer)|{a.bind,"127.0.0.1"}),
            host=a.bind,port=a.port,access_log=None,handler_cancellation=False)


if __name__=="__main__":main()
