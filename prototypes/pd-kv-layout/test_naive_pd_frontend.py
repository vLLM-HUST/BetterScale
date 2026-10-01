import asyncio
from aiohttp.test_utils import TestClient,TestServer
from naive_pd_frontend import application
from session import Conflict


class Coordinator:
    failure=None
    inflight={}
    async def start(self):return self
    async def close(self):self.closed=True
    async def submit(self,session,prompt,n):
        if session=="busy":raise Conflict("active writer")
        if session=="bad":raise ValueError("invalid tokens")
        if session=="failed":raise RuntimeError("uncertain execution")
        return dict(session=session,token_ids=[7]*n)


def test_frontend_envelope_status_and_cleanup():
    async def scenario():
        coordinator=Coordinator()
        async with TestClient(TestServer(application(coordinator))) as client:
            assert (await client.get("/health")).status==200
            assert (await client.post("/generate",json={"text":"unsupported"})).status==400
            for session,status in (("busy",409),("bad",400),("failed",503),("ok",200)):
                response=await client.post("/generate",json=dict(session=session,prompt_token_ids=[1],max_tokens=2))
                assert response.status==status
                if status==200:assert (await response.json())["token_ids"]==[7,7]
        assert coordinator.closed
    asyncio.run(scenario())
