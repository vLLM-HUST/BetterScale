import json
import pytest
from model_checkpoint import IDENTITY
from model_store import publish,split
from model_stream_import import load_plan,validate_chunks
from session import Directory,Objects,CacheMiss
from test_model_store import MemoryStore,payload


def fixture(tmp_path):
    store=MemoryStore();objects=Objects(store)
    directory=Directory(tmp_path/'sessions.sqlite');directory.create('s',IDENTITY,'P')
    data=payload(5);streams=tuple(sorted(split(data)[0]))
    key=publish(directory,objects,directory.claim('s','P',IDENTITY),data,'D0')
    return store,objects,key,streams,data


def test_plan_reads_only_manifest_and_target_checkpoint(tmp_path):
    store,objects,key,streams,data=fixture(tmp_path)
    reads=[];original=store.get
    def get(k):reads.append(k);return original(k)
    store.get=get
    plan=load_plan(objects,key,IDENTITY,streams,[55433,55434],verify=True)
    assert len(reads)==3 and reads[0]==key
    assert all('/checkpoint/' in name for name in reads[1:])
    assert plan['header']['tokens']==data['header']['tokens']
    assert plan['header']['dense_store']['verify']
    assert all(len(shard['layers'])==30 for shard in plan['shards'])
    for rank,shard in enumerate(plan['shards']):
        for name,value in shard['layers'].items():assert value==data['shards'][rank]['layers'][name]


def test_plan_is_not_a_promise_that_store_dependencies_cannot_evict(tmp_path):
    store,objects,key,streams,_=fixture(tmp_path)
    manifest=json.loads(objects.get(key));missing=manifest['chunks'][0]['keys'][streams[0]]
    del store.values[missing]
    plan=load_plan(objects,key,IDENTITY,streams,[55433,55434])
    assert plan['header']['dense_store']['chunks'][0]['keys'][streams[0]]==missing
    assert store.get(missing) is None # Device-side range reads must still fail closed.


@pytest.mark.parametrize('fault',('gap','duplicate_key','missing_head','wrong_frontier','oversized'))
def test_reject_invalid_chunk_geometry(fault):
    chunks=[dict(start=0,stop=4,keys={'k':'one','v':'two'})]
    cursor=4
    if fault=='gap':chunks[0]['start']=1
    elif fault=='duplicate_key':chunks[0]['keys']['v']='one'
    elif fault=='missing_head':del chunks[0]['keys']['v']
    elif fault=='wrong_frontier':cursor=5
    else:chunks[0]['stop']=cursor=2049
    with pytest.raises(ValueError):validate_chunks(chunks,cursor,['k','v'])


def test_reject_truncated_target_checkpoint(tmp_path):
    store,objects,key,streams,_=fixture(tmp_path)
    manifest=json.loads(objects.get(key));checkpoint=manifest['checkpoint']['gdn']['key']
    store.values[checkpoint]=store.values[checkpoint][:-1]
    with pytest.raises(CacheMiss):load_plan(objects,key,IDENTITY,streams,[55433,55434])

