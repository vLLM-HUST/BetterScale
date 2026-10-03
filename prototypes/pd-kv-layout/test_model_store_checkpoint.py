import copy
import json
from types import SimpleNamespace
import msgspec
import pytest
from model_checkpoint import IDENTITY
from model_store import publish_streamed,split
from model_stream_import import load_plan
from model_store_checkpoint import FORMAT,encode_snapshot,decode_snapshot,validate_descriptors,load_worker
from session import Directory,Objects,CacheMiss,restore_manifest
from test_model_store import MemoryStore,payload,streamed_payload


def direct_payload(data,store,lease):
    data=streamed_payload(data,store,lease);h=data['header']
    h['stream_store']['direct_checkpoint']=True
    for shard in data['shards']:
        snapshot={k:shard[k] for k in ('rank','layers','draft_valid')}
        blob=encode_snapshot(h,snapshot)
        key=f"{h['stream_store']['prefix']}/checkpoint/rank{shard['rank']}"
        store.put(key,blob)
        shard['checkpoint_store']=dict(rank=shard['rank'],key=key,bytes=len(blob),
                                      layers=sorted(shard.pop('layers')),acknowledged=True)
    return data


def test_direct_checkpoint_roundtrip_and_no_controller_payload_reads(tmp_path):
    store=MemoryStore();objects=Objects(store);directory=Directory(tmp_path/'directory.sqlite')
    directory.create('s',IDENTITY,'P')
    for owner,dest,start,end in [('P','D0',0,5),('D0','P',5,8)]:
        lease=directory.claim('s',owner,IDENTITY)
        full=payload(end);streams=tuple(sorted(split(full)[0]));data=copy.deepcopy(full)
        data['header']['dense_start']=start
        for shard in data['shards']:
            for layer in shard['layers'].values():
                if set(layer)=={'key','value'}:
                    for wire in layer.values():wire['shape'][0]-=start;wire['data']=wire['data'][start*512:]
        data=direct_payload(data,store,lease)
        assert len(msgspec.msgpack.encode(data))<25000
        key=publish_streamed(directory,objects,lease,data,dest)
        reads=[];original=store.get
        def get(k):reads.append(k);return original(k)
        store.get=get
        plan=load_plan(objects,key,IDENTITY,streams,[55433,55434],verify=True)
        assert reads==[key] and plan['header']['checkpoint_format']==FORMAT
        assert len(msgspec.msgpack.encode(plan))<25000
        store.get=original
        for rank in (0,1):
            descriptor=plan['shards'][rank]['checkpoint_store']
            restored=decode_snapshot(store.get(descriptor['key']),plan['header'],rank,descriptor)
            expected={n:v for n,v in full['shards'][rank]['layers'].items() if set(v)=={'conv','recurrent'}}
            assert restored['layers']==expected
        assert restore_manifest(objects,key,IDENTITY,streams,512)['cursor']==end
    victim=plan['shards'][1]['checkpoint_store']['key'];del store.values[victim]
    with pytest.raises(CacheMiss):restore_manifest(objects,key,IDENTITY,streams,512)
    # Planning is not an availability lease; the worker still rejects a later miss.
    assert load_plan(objects,key,IDENTITY,streams,[55433,55434])


@pytest.mark.parametrize('fault',['rank','ack','bytes','writer','census'])
def test_descriptor_rejects_wrong_identity(fault):
    from model_store_checkpoint import MIN_BYTES
    prefix='pd/s/1/id'
    ds={f'rank{r}':dict(rank=r,key=f'{prefix}/checkpoint/rank{r}',bytes=MIN_BYTES+100,
                       layers=[f'l{i}' for i in range(30)],acknowledged=True) for r in (0,1)}
    if fault=='rank':ds['rank1']['rank']=0
    elif fault=='ack':ds['rank1']['acknowledged']=False
    elif fault=='bytes':ds['rank1']['bytes']=1
    elif fault=='writer':ds['rank1']['key']='pd/another/writer'
    else:ds['rank1']['layers'][0]='different'
    with pytest.raises(ValueError):validate_descriptors(ds,prefix=prefix)


def test_checkpoint_blob_rejects_frontier_geometry_and_truncation():
    full=payload(2);shard=full['shards'][0]
    shard['layers']={n:d for n,d in shard['layers'].items() if set(d)=={'conv','recurrent'}}
    header=full['header'];blob=encode_snapshot(header,shard)
    desc=dict(bytes=len(blob),layers=sorted(shard['layers']))
    with pytest.raises(CacheMiss):decode_snapshot(blob[:-1],header,0,desc)
    with pytest.raises(CacheMiss):decode_snapshot(blob,dict(header,cursor=3),0,desc)
    with pytest.raises(CacheMiss):decode_snapshot(blob,header,1,desc)
    bad=msgspec.msgpack.decode(blob)
    next(iter(bad['shard']['layers'].values()))['conv']['shape']=[1]
    broken=msgspec.msgpack.encode(bad)
    with pytest.raises(CacheMiss):decode_snapshot(broken,header,0,dict(desc,bytes=len(broken)))


def test_missing_worker_blob_releases_cpu_store_operation(monkeypatch):
    import model_store_checkpoint
    class Client:
        def setup(self,*args):return 0
        def get(self,key):return None
    monkeypatch.setattr(model_store_checkpoint,'client_type',lambda:Client)
    runner=SimpleNamespace()
    with pytest.raises(CacheMiss):
        load_worker(runner,dict(dense_store=dict(ports=[55433,55434])),0,dict(key='missing'))
    assert runner._pd_store_clients[55433]['busy'] is False
