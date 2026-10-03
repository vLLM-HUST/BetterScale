"""Loopback token-ID frontend for the experimental two-host PD coordinator.

Use SSH forwarding for remote clients. No public auth/TLS, OpenAI compatibility,
streaming, automatic failover or exactly-once response replay is claimed.
"""
import argparse
from pathlib import Path

from aiohttp import web
from naive_pd_coordinator import Coordinator
from pd_limits import context_limit
from session import Conflict


def application(coordinator):
    async def startup(app):
        await coordinator.start()

    async def cleanup(app):
        await coordinator.close()

    async def health(request):
        return web.json_response(dict(ready=coordinator.failure is None,
            context_limit=context_limit(),inflight=len(coordinator.inflight),
            failed_closed=coordinator.failure is not None),
            status=200 if coordinator.failure is None else 503)

    async def generate(request):
        try:
            body=await request.json()
            if not isinstance(body,dict) or set(body)!={"session","prompt_token_ids","max_tokens"}:
                raise ValueError("Expected session, prompt_token_ids and max_tokens")
            result=await coordinator.submit(body["session"],body["prompt_token_ids"],body["max_tokens"])
        except (ValueError,TypeError) as error:
            raise web.HTTPBadRequest(text=str(error))
        except Conflict as error:
            raise web.HTTPConflict(text=str(error))
        except RuntimeError as error:
            raise web.HTTPServiceUnavailable(text=str(error))
        return web.json_response(result)

    app=web.Application(client_max_size=4<<20)
    app.router.add_get("/health",health)
    app.router.add_post("/generate",generate)
    app.on_startup.append(startup);app.on_cleanup.append(cleanup)
    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p",required=True);parser.add_argument("--d",required=True)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--port",type=int,default=55686)
    parser.add_argument("--max-batch",type=int,default=64)
    args=parser.parse_args()
    coordinator=Coordinator(args.directory,args.p,args.d,max_batch=args.max_batch)
    web.run_app(application(coordinator),host="127.0.0.1",port=args.port,
                handler_cancellation=True,access_log=None)


if __name__=="__main__":main()
