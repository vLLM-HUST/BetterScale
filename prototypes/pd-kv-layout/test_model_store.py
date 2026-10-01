import pytest
from model_checkpoint import IDENTITY
from model_store import envelope,split,publish,load
from session import Directory,Objects,CacheMiss

class MemoryStore:
 def __init__(self):self.values={}
 def put(self,k,v):self.values[k]=v;return 0
 def get(self,k):return self.values.get(k)

def payload(cursor):
 shards=[]
 for rank in (0,1):
  layers={}
  for i in range(40):
   if i%4==3:
    layers[f'layer{i}']={kind:envelope(bytes([i+rank])*cursor*512,(cursor,1,256),'bfloat16') for kind in ('key','value')}
   else:layers[f'layer{i}']=dict(conv=envelope(bytes([cursor])*3*4096*2,(3,4096),'bfloat16'),
       recurrent=envelope(bytes([cursor])*16*128*128*4,(16,128,128),'float32'))
  shards.append(dict(rank=rank,draft_valid=False,layers=layers))
 return dict(header=dict(identity=IDENTITY,block_size=2048,cursor=cursor,tokens=list(range(cursor+1)),seat=0,epoch=1,blocks=[1]),shards=shards)

def test_bidirectional_model_streams_only_append_delta(tmp_path):
 store=MemoryStore();objects=Objects(store);directory=Directory(tmp_path/'directory.sqlite');directory.create('s',IDENTITY,'P')
 previous_keys=None
 for owner,next_owner,cursor in [('P','D0',5),('D0','P',8)]:
  data=payload(cursor);dense,_=split(data);streams=tuple(sorted(dense))
  key=publish(directory,objects,directory.claim('s',owner,IDENTITY),data,next_owner)
  restored=load(objects,key,IDENTITY,streams)
  assert restored['shards']==data['shards'] and restored['header']['tokens']==data['header']['tokens']
  import json
  manifest=json.loads(objects.get(key))
  if previous_keys is not None:
   assert manifest['chunks'][0]['keys']==previous_keys
   assert [(c['start'],c['stop']) for c in manifest['chunks']]==[(0,5),(5,8)]
  previous_keys=manifest['chunks'][0]['keys']
 victim=manifest['chunks'][0]['keys'][streams[0]];del store.values[victim]
 with pytest.raises(CacheMiss):load(objects,key,IDENTITY,streams)

def test_reject_missing_target_plane():
 data=payload(1);del data['shards'][1]['layers']['layer3']['value']
 with pytest.raises(ValueError):split(data)


def test_incremental_device_payload_reconstructs_full_model(tmp_path):
 store=MemoryStore();objects=Objects(store);directory=Directory(tmp_path/'dir.sqlite');directory.create('s',IDENTITY,'P')
 initial=payload(5);streams=tuple(sorted(split(initial)[0]))
 publish(directory,objects,directory.claim('s','P',IDENTITY),initial,'D0')
 delta=payload(8);expected=payload(8);delta['header']['dense_start']=5
 for shard in delta['shards']:
  for data in shard['layers'].values():
   if set(data)=={'key','value'}:
    for value in data.values():value['shape'][0]=3;value['data']=value['data'][5*512:]
 key=publish(directory,objects,directory.claim('s','D0',IDENTITY),delta,'P')
 assert load(objects,key,IDENTITY,streams)['shards']==expected['shards']


def streamed_payload(data,store,lease):
 prefix=f'pd/{lease.session}/{lease.epoch}/cpu-fixture'
 data['header']['stream_store']=dict(prefix=prefix,ports=[55421,55422])
 start=data['header'].get('dense_start',0);cursor=data['header']['cursor']
 for shard in data['shards']:
  dense={}
  for name in list(shard['layers']):
   plane=shard['layers'][name]
   if set(plane)=={'key','value'}:
    del shard['layers'][name]
    for kind in ('key','value'):dense[f'{name}/{kind}/head{shard["rank"]}']=plane[kind]['data']
  chunks=[]
  for begin in range(start,cursor,2048):
   end=min(begin+2048,cursor);keys={name:f'{prefix}/dense/{begin}/{name}' for name in dense}
   for name,key in keys.items():store.put(key,dense[name][(begin-start)*512:(end-start)*512])
   chunks.append(dict(start=begin,stop=end,keys=keys))
  shard['dense_store']=dict(acknowledged=True,streams=list(dense),chunks=chunks,dense_bytes=(cursor-start)*20*512)
 return data


def test_streamed_acknowledgements_reconstruct_and_fail_closed(tmp_path):
 from model_store import publish_streamed
 import copy
 store=MemoryStore();objects=Objects(store);directory=Directory(tmp_path/'dir.sqlite');directory.create('s',IDENTITY,'P')
 for owner,dest,start,end in [('P','D0',0,5),('D0','P',5,8)]:
  lease=directory.claim('s',owner,IDENTITY);full=payload(end);streams=tuple(sorted(split(full)[0]));delta=copy.deepcopy(full)
  delta['header']['dense_start']=start
  for shard in delta['shards']:
   for data in shard['layers'].values():
    if set(data)=={'key','value'}:
     for value in data.values():value['shape'][0]-=start;value['data']=value['data'][start*512:]
  data=streamed_payload(delta,store,lease)
  prior=directory.current('s')
  bad=copy.deepcopy(data);bad['shards'][1]['dense_store']['acknowledged']=False
  with pytest.raises(ValueError,match='Unacknowledged'):publish_streamed(directory,objects,lease,bad,dest)
  bad=copy.deepcopy(data);bad['shards'][1]['dense_store']['chunks'][0]['start']+=1
  with pytest.raises(ValueError,match='span'):publish_streamed(directory,objects,lease,bad,dest)
  bad=copy.deepcopy(data);k=next(iter(bad['shards'][0]['dense_store']['chunks'][0]['keys']))
  bad['shards'][0]['dense_store']['chunks'][0]['keys'][k]='other-writer/key'
  with pytest.raises(ValueError,match='span'):publish_streamed(directory,objects,lease,bad,dest)
  assert directory.current('s')==prior
  key=publish_streamed(directory,objects,lease,data,dest)
  assert load(objects,key,IDENTITY,streams)['shards']==full['shards']
 victim=data['shards'][0]['dense_store']['chunks'][0]['keys'][streams[0]]
 del store.values[victim]
 with pytest.raises(CacheMiss):load(objects,key,IDENTITY,streams)
