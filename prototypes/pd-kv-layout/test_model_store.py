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
