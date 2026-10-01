import asyncio
from types import SimpleNamespace
import pytest
from native_async_pool import owner_utility

@pytest.mark.parametrize('owner', [0, 1, 2])
def test_utility_targets_exact_owner_without_broadcast(owner):
    calls = []
    async def targeted(method, *args, engine):
        calls.append((method, args, engine))
        return 'ack'
    async def broadcast(*args):
        raise AssertionError('Owner-specific mutation must never broadcast')
    client = SimpleNamespace(core_engines=[b'0', b'1', b'2'],
        _call_utility_async=targeted, call_utility_async=broadcast)
    assert asyncio.run(owner_utility(client, owner, 'pd_import_target', 'bytes', 'salt')) == 'ack'
    assert calls == [('pd_import_target', ('bytes', 'salt'), client.core_engines[owner])]

@pytest.mark.parametrize('owner', [-1, 3, True, '1'])
def test_invalid_owner_rejected_before_rpc(owner):
    with pytest.raises(ValueError, match='owner'):
        asyncio.run(owner_utility(SimpleNamespace(core_engines=[b'0', b'1', b'2']), owner,
                                 'pd_drop_target', 'salt'))

def test_manual_wake_not_supported():
    with pytest.raises(ValueError, match='utility'):
        asyncio.run(owner_utility(SimpleNamespace(core_engines=[b'0']), 0, 'pd_start_wave'))
