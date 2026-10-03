import pytest
from native_store_leases import NativeStoreLeases,LeaseLost


def fixture():
    now=[0.0];available={"a"}
    ledger=NativeStoreLeases(lambda keys:set(keys)&available,ttl_seconds=2,
                             clock=lambda:now[0],background=False)
    return now,available,ledger


def test_shared_checkpoint_refs_renew_and_only_last_drop_releases():
    now,available,ledger=fixture()
    assert ledger.begin("A",("a","b"))==[True,False]
    with pytest.raises(LeaseLost,match="uncommitted"):ledger.check("A",complete=True)
    available.add("b");ledger.publish("b");ledger.check("A",complete=True)
    assert ledger.begin("B",("a",))==[True]
    ledger.drop("A")
    assert ledger.references=={"a":1} and set(ledger.deadlines)=={"a"}
    now[0]=1.2;ledger.renew();now[0]=2.1;ledger.check("B",complete=True)
    ledger.drop("B")
    assert not ledger.references and not ledger.deadlines
    ledger.close()


def test_expired_lease_is_not_silently_reacquired():
    now,available,ledger=fixture()
    ledger.begin("A",("a",));now[0]=2.01
    for action in (ledger.renew,lambda:ledger.check("A",complete=True),lambda:ledger.begin("B",("a",))):
        with pytest.raises(LeaseLost,match="expired"):action()
    ledger.close()


def test_lost_replica_is_sticky_and_does_not_publish_checkpoint():
    now,available,ledger=fixture()
    ledger.begin("A",("a",));available.clear()
    with pytest.raises(LeaseLost,match="disappeared"):ledger.renew()
    available.add("a")
    with pytest.raises(LeaseLost):ledger.check("A",complete=True)
    ledger.close()


def test_slow_native_reply_cannot_claim_a_fresh_full_ttl():
    now=[0.0]
    def query(keys):now[0]+=2.1;return set(keys)
    ledger=NativeStoreLeases(query,ttl_seconds=2,clock=lambda:now[0],background=False)
    with pytest.raises(LeaseLost,match="conservative expiry"):ledger.begin("A",("a",))
    assert not ledger.deadlines
    ledger.close()
