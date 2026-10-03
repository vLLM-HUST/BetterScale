"""Check installed native metadata queries really renew eviction leases."""
import argparse
import json
from pathlib import Path
import time
from dram_store_fixture import dram_store

def run(output):
    output.mkdir(exist_ok=False)
    results=[]
    with dram_store(output/"store",55401,segment_bytes=64<<20) as (owner,query):
        from native_store_leases import NativeStoreLeases
        from mooncake.store import ReplicaStatus
        for batch in (False,True):
            key="lease-batch" if batch else "lease-single"
            assert owner.put(key,b"lease-test")==0
            time.sleep(2.2)  # fixture master explicitly uses a2000ms TTL
            def renew():
                begin=time.monotonic()
                rows=query.batch_get_replica_desc([key])[key] if batch else query.get_replica_desc(key)
                assert rows
                return time.monotonic()-begin
            first=renew()
            immediate=owner.remove(key,False)
            assert immediate!=0,"Query did not prevent eviction"
            time.sleep(1.2)
            second=renew()
            time.sleep(1.2)  # old lease expired, renewed lease still valid
            renewed=owner.remove(key,False)
            assert renewed!=0,"Repeated query did not extend eviction protection"
            time.sleep(1.0)
            expired=owner.remove(key,False)
            assert expired==0,"Expired unpinned object did not become removable"
            results.append(dict(batch=batch,query_seconds=[first,second],
                immediate_remove=immediate,renewed_remove=renewed,expired_remove=expired))
        key="lease-ledger"
        assert owner.put(key,b"ledger-test")==0
        def query_complete(keys):
            rows=query.batch_get_replica_desc(list(keys))
            return {k for k,v in rows.items() if v and all(r.status==ReplicaStatus.COMPLETE for r in v)}
        ledger=NativeStoreLeases(query_complete,ttl_seconds=2)
        try:
            assert ledger.begin("checkpoint",(key,))==[True]
            time.sleep(3.2)
            ledger.check("checkpoint",complete=True)
            held=owner.remove(key,False)
            assert held!=0,"renewing checkpoint lost protection"
            ledger.drop("checkpoint")
            time.sleep(2.2)
            released=owner.remove(key,False)
            assert released==0,"last checkpoint drop retained a permanent pin"
            results.append(dict(ledger=True,held_remove=held,released_remove=released))
        finally:
            ledger.close()
    receipt=dict(scope="installed wheel, explicit2000ms master TTL; no device or failure recovery",results=results)
    (output/"summary.json").write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    run(p.parse_args().output)
