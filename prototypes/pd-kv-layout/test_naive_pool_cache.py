import asyncio
import hashlib
import pytest
from naive_pool_node import Cache,pack,unpack
from model_checkpoint import IDENTITY


class Store:
    def __init__(self):self.objects={};self.fail=None;self.writes=[]
    def put(self,key,data):
        assert len(data)<64<<20
        self.writes.append(key)
        if self.fail and key.startswith(self.fail):return -1
        self.objects[key]=data
        return 0
    def get(self,key):return self.objects.get(key)
    def get_size(self,key):return len(self.objects[key]) if key in self.objects else -1


def payload():
    return pack(dict(header=dict(identity=IDENTITY),shards=[{"data":b"x"*(17<<20)},{}]))


def test_chunked_publish_and_missing_chunk():
    async def scenario():
        store=Store();cache=Cache(store);data=payload();key=hashlib.sha256(data).hexdigest()
        await cache.put(key,data)
        assert store.writes[-1]=="manifest:"+key
        assert await cache.get(key)==data
        written=len(store.writes)
        await cache.put(key,data)
        assert len(store.writes)==written
        manifest=unpack(store.objects["manifest:"+key])
        del store.objects[manifest["chunks"][0][0]]
        with pytest.raises(KeyError):await cache.get(key)
        await cache.put(key,data)
        assert await cache.get(key)==data
    asyncio.run(scenario())


def test_no_manifest_after_failed_chunk():
    async def scenario():
        store=Store();store.fail="chunk:";cache=Cache(store);data=payload()
        key=hashlib.sha256(data).hexdigest()
        with pytest.raises(RuntimeError):await cache.put(key,data)
        assert "manifest:"+key not in store.objects
    asyncio.run(scenario())


@pytest.mark.parametrize("body",[
    {}, {"instance":True,"op":"drop","args":{"owner":0,"salt":"s"}},
    {"instance":0,"op":"drop","args":{"owner":4,"salt":"s"}},
    {"instance":0,"op":"import","args":{"owner":0,"salt":"s","key":"bad"}},
    {"instance":0,"op":"generate_batch","args":{"items":[{"owner":0,"salt":"s","tokens":[True],"n":1}]}},
    {"instance":0,"op":"generate_batch","args":{"items":[{"owner":0,"salt":"s","tokens":[1],"n":1}]*17}},
])
def test_bad_rpc_rejected_before_actor(body):
    from naive_pool_node import validate_rpc
    with pytest.raises(ValueError):validate_rpc("D",body)


def test_capacity_rpc_is_read_only_and_has_no_extra_arguments():
    from naive_pool_node import validate_rpc
    assert validate_rpc("D",dict(instance=0,op="capacity",args={}))==(0,"capacity",{})
    with pytest.raises(ValueError):
        validate_rpc("D",dict(instance=0,op="capacity",args={"owner":0}))
