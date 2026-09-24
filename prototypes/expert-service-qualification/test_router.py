"""CPU checks: stable rank ownership and unchanged HTTP/SSE transport."""
import tempfile
import unittest
from pathlib import Path
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from agentx_router import Affinity, application

class Placement(unittest.TestCase):
    def test_sticky_round_robin(self):
        affinity=Affinity(4)
        self.assertEqual([affinity.select(s) for s in ('a','b','c','d','e','a','d')],[0,1,2,3,0,0,3])
        with self.assertRaises(ValueError):affinity.select(None)

class Transport(unittest.IsolatedAsyncioTestCase):
    async def test_dp_stream_and_private_rpc(self):
        seen=[]
        async def upstream(request):
            seen.append((await request.read(),request.headers.get('X-data-parallel-rank')))
            response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
            await response.prepare(request)
            for chunk in (b'data: {"x":1}\n\n',b'data: [DONE]\n\n'):
                await response.write(chunk)
            await response.write_eof()
            return response
        backend=web.Application();backend.router.add_post('/v1/chat/completions',upstream)
        async with TestServer(backend) as server:
            with tempfile.TemporaryDirectory() as temp:
                async with TestClient(TestServer(application([str(server.make_url('')).rstrip('/')],8,Path(temp)/'receipt.json'))) as client:
                    for session in ('one','two','one'):
                        response=await client.post('/v1/chat/completions',data=('{"cache_salt":"'+session+'"}').encode(),headers={'X-data-parallel-rank':'7'})
                        self.assertEqual(await response.read(),b'data: {"x":1}\n\ndata: [DONE]\n\n')
                    self.assertEqual((await client.post('/collective_rpc',json={})).status,404)
                    self.assertEqual((await client.post('/v1/chat/completions',json={})).status,400)
                self.assertTrue((Path(temp)/'receipt.json').is_file())
        self.assertEqual(seen,[(b'{"cache_salt":"one"}','0'),(b'{"cache_salt":"two"}','1'),(b'{"cache_salt":"one"}','0')])

if __name__=='__main__':unittest.main()
