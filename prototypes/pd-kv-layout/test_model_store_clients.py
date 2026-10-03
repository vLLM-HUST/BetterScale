from types import SimpleNamespace
import pytest
from model_store_clients import acquire,release,close_worker


class Client:
    def __init__(self):self.setups=0;self.closes=0;self.fail=False
    def setup(self,*args):self.setups+=1;return 0
    def close(self):
        if self.fail:raise RuntimeError('close failed')
        self.closes+=1


def test_reuses_connection_but_retains_each_transfer_until_release():
    owner=SimpleNamespace();client=Client();held=object()
    assert acquire(owner,55421,held,lambda:client) is client
    assert owner._pd_store_clients[55421]['retained'] is held
    with pytest.raises(RuntimeError,match='owns'):acquire(owner,55421,None,Client)
    with pytest.raises(RuntimeError,match='owns'):close_worker(SimpleNamespace(model_runner=owner))
    release(owner,55421);assert owner._pd_store_clients[55421]['retained'] is None
    assert acquire(owner,55421,None,Client) is client
    release(owner,55421);assert client.setups==1 and client.closes==0
    assert close_worker(SimpleNamespace(model_runner=owner))
    assert client.closes==1 and not owner._pd_store_clients


def test_close_failure_preserves_client_for_retry():
    owner=SimpleNamespace();client=Client()
    acquire(owner,55431,None,lambda:client);release(owner,55431);client.fail=True
    with pytest.raises(RuntimeError,match='failed'):close_worker(SimpleNamespace(model_runner=owner))
    assert owner._pd_store_clients[55431]['client'] is client
    client.fail=False;close_worker(SimpleNamespace(model_runner=owner));assert client.closes==1


def test_setup_failure_never_caches_a_client():
    owner=SimpleNamespace();client=Client();client.setup=lambda *args:-1
    with pytest.raises(RuntimeError,match='setup'):acquire(owner,55421,None,lambda:client)
    assert not owner._pd_store_clients and client.closes==1


@pytest.mark.parametrize('port',[True,55401,55429,55439])
def test_rejects_unqualified_endpoint(port):
    with pytest.raises(ValueError):acquire(SimpleNamespace(),port,None,Client)
