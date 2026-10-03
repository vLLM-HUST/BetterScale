import ctypes
import pytest
import torch
from rank_state_pool import RankStatePool
from rank_replica_receiver import RankReplicaReceiver
from rank_peer_control import RankControl
from rank_replicator import RankReplicator
from betterscale.live.runtime.host_state import HostStateKey


class Engine:
    def __init__(self):self.registered={};self.writes=0
    def get_rpc_port(self):return 12345
    def register_memory(self,pointer,size):
        assert pointer not in self.registered
        self.registered[pointer]=size;return 0
    def unregister_memory(self,pointer):
        del self.registered[pointer];return 0
    def transfer_sync_write(self,endpoint,source,target,size):
        assert self.registered[source]==self.registered[target]==size
        ctypes.memmove(target,source,size);self.writes+=1;return 0


def test_real_control_rpc_replicates_once_and_preserves_sticky_shards(monkeypatch):
    import rank_replicator
    monkeypatch.setattr(rank_replicator,"HOSTS",{"P":"127.0.0.1","D":"127.0.0.1"})
    allocate=lambda n:torch.empty(n,dtype=torch.uint8)
    source=RankStatePool(("D",0,0),1024,allocate)
    target=RankStatePool(("P",0,0),1024,allocate)
    engine=Engine();receiver=RankReplicaReceiver(target,engine)
    server=RankControl(receiver,"127.0.0.1",0,{"127.0.0.1"}).start()
    replica=RankReplicator(source,engine,control_base=server.server.server_address[1])
    a=HostStateKey("A",1);b=HostStateKey("B",1)
    try:
        source.retain(a,("a",))
        writer=source.reserve("a",32)
        ctypes.memset(writer.pointer,73,32);writer.seal(lambda:None)
        replica.set_peer(a,0);replica.replicate(a,("a",))
        assert target.complete(a) and not engine.registered and engine.writes==1
        with target.read("a") as reader:
            assert ctypes.string_at(reader.pointer,32)==bytes([73])*32
        source.retain(b,("a",));replica.set_peer(b,0);replica.replicate(b,("a",))
        assert target.complete(b) and engine.writes==1
        with pytest.raises(ValueError,match="owner changed"):replica.set_peer(a,1)
        with pytest.raises(RuntimeError,match="unqualified"):
            replica.rpc("127.0.0.1",server.server.server_address[1],"arbitrary",{})
        from rank_state_transport import RankStateTransport
        transport=RankStateTransport(target,"test",None,None,release_checkpoint=receiver.drop)
        source.drop(a);source.drop(b);transport.release(a);transport.release(b)
        transport.release(b)  # repeated release is harmless
        assert not receiver.completed
    finally:server.close()
    assert not receiver.accepting
    source.close();target.close()


def test_control_cannot_close_with_remote_writer_in_flight():
    from hashlib import sha256
    engine=Engine();pool=RankStatePool(("P",0,0),64,lambda n:torch.empty(n,dtype=torch.uint8))
    receiver=RankReplicaReceiver(pool,engine)
    server=RankControl(receiver,"127.0.0.1",0,{"127.0.0.1"}).start()
    key=HostStateKey("A",1);receiver.begin(key,("a",))
    result=receiver.prepare(("D",0,0),key,"a",32,sha256(bytes(32)).hexdigest(),"a"*32)
    with pytest.raises(RuntimeError,match="not drained"):server.close()
    ctypes.memset(result["pointer"],0,32);receiver.commit("a"*32)
    receiver.drop(key);server.close();pool.close()


def test_listener_preflight_rejects_ephemeral_overlap_before_model_start(monkeypatch):
    from pathlib import Path
    from rank_peer_control import preflight
    from rank_replicator import CONTROL_BASE
    monkeypatch.setattr(Path,"read_text",lambda *a,**k:f"{CONTROL_BASE} {CONTROL_BASE+100}")
    import pytest
    with pytest.raises(ValueError,match="ephemeral"):
        preflight("127.0.0.1")


def test_preflight_import_does_not_need_model_package():
    import subprocess
    import sys
    from pathlib import Path
    folder=str(Path(__file__).parent)
    code=("import sys; sys.path.insert(0,"+repr(folder)+"); "
          "from rank_peer_control import preflight; "
          "assert 'betterscale' not in sys.modules; "
          "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable,"-I","-c",code],check=True)
