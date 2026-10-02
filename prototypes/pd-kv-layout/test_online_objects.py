import hashlib
import pytest
from online_objects import Objects,PeerObjectSink

class Store:
    def __init__(self):self.values={};self.fail=False
    def get(self,k):return self.values.get(k,b"")
    def get_size(self,k):return len(self.values[k]) if k in self.values else -1
    def put(self,k,v):
        if self.fail:return -600
        self.values[k]=v;return 0

def test_shared_page_publication_collision_eviction_and_repair():
    s=Store();o=Objects(s);key="a"*64;data=b"a"*(9<<20)
    o.write(key,data);assert o.read(key)==data
    with pytest.raises(ValueError,match="collision"):o.write(key,b"other")
    chunk=o.inspect(key)["chunks"][0][0];del s.values[chunk]
    with pytest.raises(KeyError):o.inspect(key)
    o.write(key,data);assert o.read(key)==data

def test_failed_put_never_publishes_manifest():
    s=Store();o=Objects(s);s.fail=True
    with pytest.raises(RuntimeError):o.write("a"*64,b"payload")
    assert not s.values

def test_sink_requires_two_acks_and_repairs_only_missing_replica():
    sink=PeerObjectSink(["http://10.244.1.16:55581","http://10.244.2.32:55586"])
    data={sink.urls[0]:b"payload"};puts=[]
    def request(url,key,method="GET",data_arg=None):
        if method=="HEAD":return b"" if url in data else None
        return data.get(url)
    sink.request=request
    sink._put=lambda url,key,payload:(puts.append(url),data.update({url:payload}))
    assert sink.ensure("a"*64) and puts==[sink.urls[1]]
    assert sink.ensure("a"*64) and len(puts)==1

def test_peer_endpoints_are_explicit():
    with pytest.raises(ValueError):PeerObjectSink(["http://example.com","http://10.244.1.16:55581"])
