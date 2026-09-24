"""Session-affine streaming relay; workload bodies and SSE bytes are unchanged.

The official harness supplies X-Correlation-ID per session. New sessions are
assigned round-robin; subsequent turns stay on that attention/DP rank. Child
sessions get their own placement. This is not root-tree affinity or caching.
Only inference/readiness paths are exposed; native development RPC stays private.
"""
import argparse
import collections
import json
from pathlib import Path
from aiohttp import ClientSession, ClientTimeout, web

HOP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
       'te', 'trailer', 'transfer-encoding', 'upgrade', 'host', 'content-length'}

class Affinity:
    def __init__(self, ranks):
        assert ranks > 0
        self.ranks = ranks
        self.sessions = {}
        self.counts = collections.Counter()

    def select(self, session):
        if not session:
            raise ValueError('X-Correlation-ID is required for inference')
        if session not in self.sessions:
            if len(self.sessions) >= 100000:
                raise ValueError('bounded session table exhausted')
            self.sessions[session] = len(self.sessions) % self.ranks
        rank = self.sessions[session]
        self.counts[rank] += 1
        return rank


def application(backends, dp_size, receipt):
    assert dp_size == 1 or len(backends) == 1
    affinity = Affinity(dp_size if dp_size > 1 else len(backends))
    app = web.Application(client_max_size=16*1024*1024)

    async def lifetime(app):
        async with ClientSession(timeout=ClientTimeout(total=None, sock_connect=10,
                                                       sock_read=10800),
                                 auto_decompress=False, trust_env=False) as client:
            app['client'] = client
            yield
        receipt.write_text(json.dumps({'policy':'round-robin new X-Correlation-ID, sticky turns; independent children',
                                       'sessions':len(affinity.sessions),
                                       'requests_per_rank':dict(affinity.counts)}, indent=2)+'\n')
    app.cleanup_ctx.append(lifetime)

    async def relay(request):
        if request.path not in ('/health', '/v1/models', '/v1/chat/completions', '/v1/completions'):
            raise web.HTTPNotFound()
        rank = 0
        if request.method == 'POST':
            try:
                rank = affinity.select(request.headers.get('X-Correlation-ID'))
            except ValueError as error:
                raise web.HTTPBadRequest(text=str(error))
        headers = {k:v for k,v in request.headers.items()
                   if k.lower() not in HOP | {'x-data-parallel-rank'}}
        if dp_size > 1:
            headers['X-data-parallel-rank'] = str(rank)
        backend = backends[0 if dp_size > 1 else rank]
        async with app['client'].request(request.method, backend+request.path,
                                        data=await request.read(), headers=headers) as upstream:
            response = web.StreamResponse(status=upstream.status,
                headers={k:v for k,v in upstream.headers.items() if k.lower() not in HOP})
            await response.prepare(request)
            async for block in upstream.content.iter_any():
                await response.write(block)
            await response.write_eof()
            return response
    app.router.add_get('/health', relay)
    app.router.add_get('/v1/models', relay)
    app.router.add_post('/v1/chat/completions', relay)
    app.router.add_post('/v1/completions', relay)
    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backends', nargs='+', required=True)
    parser.add_argument('--dp-size', type=int, default=1)
    parser.add_argument('--port', type=int, default=18900)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    assert all(url.startswith('http://127.0.0.1:') for url in args.backends)
    web.run_app(application(args.backends, args.dp_size, args.receipt), host='127.0.0.1',
                port=args.port, access_log=None, handler_cancellation=True)
