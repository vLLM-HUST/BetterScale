"""Map complete TP2 target checkpoints onto the existing fenced Store prototype.

Head-major dense streams append only new tokens. Target GDN/conv are turn-end
objects. This adapter starts after writer retirement; it does not overlap DMA
with model compute, and SQLite remains a same-host coordinator, not consensus.
"""
import math
import msgspec
from model_checkpoint import IDENTITY
from session import Turn,restore as restore_session,CacheMiss
TOKEN_BYTES=512


def tensor(value,shape,dtype):
    if (not isinstance(value,dict) or set(value)!={'checkpoint_tensor','shape','dtype','data'}
            or value['checkpoint_tensor'] is not True or value['shape']!=list(shape)
            or value['dtype']!=dtype or not isinstance(value['data'],bytes)
            or len(value['data'])!=math.prod(shape)*({'bfloat16':2,'float32':4}[dtype])):
        raise ValueError('Incompatible target wire tensor')
    return value['data']


def envelope(data,shape,dtype):
    result=dict(checkpoint_tensor=True,shape=list(shape),dtype=dtype,data=data)
    tensor(result,shape,dtype)
    return result


def split(payload):
    h=payload['header'];cursor=h['cursor'];shards=payload['shards'];start=h.get('dense_start',0)
    if (type(start) is not int or not 0<=start<=cursor or h['identity']!=IDENTITY or h['block_size']!=2048 or not 0<cursor<=8192
            or len(h['tokens'])!=cursor+1 or len(shards)!=2
            or [s['rank'] for s in shards]!=[0,1]):
        raise ValueError('Incompatible target Store geometry')
    dense={};gdn={};conv={};layouts=[]
    for shard in shards:
        rank=shard['rank'];gdn[rank]={};conv[rank]={};fa=0
        if shard['draft_valid'] is not False:raise ValueError('Draft is not a shared checkpoint')
        for name,data in sorted(shard['layers'].items()):
            if set(data)=={'key','value'}:
                fa+=1
                for kind in ('key','value'):
                    dense[f'{name}/{kind}/head{rank}']=tensor(data[kind],(cursor-start,1,256),'bfloat16')
            elif set(data)=={'conv','recurrent'}:
                tensor(data['recurrent'],(16,128,128),'float32');tensor(data['conv'],(3,4096),'bfloat16')
                gdn[rank][name]=data['recurrent'];conv[rank][name]=data['conv']
            else:raise ValueError('Unexpected target layer fields')
        if fa!=10 or len(gdn[rank])!=30:raise ValueError('Incomplete target layer census')
        layouts.append(set(shard['layers']))
    if layouts[0]!=layouts[1] or set(gdn[0])!=set(gdn[1]):raise ValueError('Mismatched TP layouts')
    return dense,dict(gdn=msgspec.msgpack.encode(gdn),conv=msgspec.msgpack.encode(conv))


def publish(directory,objects,lease,payload,next_owner):
    dense,checkpoint=split(payload);header=payload['header'];streams=tuple(sorted(dense))
    turn=Turn(directory,objects,lease,streams,TOKEN_BYTES)
    try:
        start=header.get('dense_start',0)
        if start not in (0,turn.cursor):raise ValueError('Dense increment does not start at committed Store frontier')
        # At most40 MiB per immutable batch, two pending batches/64 MiB total.
        while turn.cursor<header['cursor']:
            begin=turn.cursor;end=min(begin+2048,header['cursor'])
            turn.append(end,{s:dense[s][(begin-start)*TOKEN_BYTES:(end-start)*TOKEN_BYTES] for s in streams})
        return turn.finish(header['tokens'][:-1],checkpoint,next_owner,writer_retired=True,
                           pending_token=header['tokens'][-1])
    finally:turn.close()


def load(objects,key,identity,streams):
    manifest,dense,checkpoint=restore_session(objects,key,identity,streams,TOKEN_BYTES)
    try:
        cursor=manifest['cursor'];pending=manifest['pending_token']
        if type(pending) is not int or pending<0:raise ValueError('Missing pending token')
        gdn=msgspec.msgpack.decode(checkpoint['gdn']);conv=msgspec.msgpack.decode(checkpoint['conv'])
        if set(gdn)!={0,1} or set(conv)!={0,1}:raise ValueError('Incomplete TP checkpoint')
        shards=[]
        for rank in (0,1):
            if set(gdn[rank])!=set(conv[rank]):raise ValueError('Inconsistent target checkpoint')
            layers={name:dict(recurrent=gdn[rank][name],conv=conv[rank][name]) for name in gdn[rank]}
            shards.append(dict(rank=rank,layers=layers,draft_valid=False))
        for stream,data in dense.items():
            name,kind,head=stream.rsplit('/',2)
            if kind not in ('key','value') or head not in ('head0','head1'):raise ValueError('Invalid head stream')
            shards[int(head[-1])]['layers'].setdefault(name,{})[kind]=envelope(data,(cursor,1,256),'bfloat16')
        header=dict(identity=IDENTITY,cursor=cursor,tokens=manifest['tokens']+[pending],block_size=2048,
                    seat=-1,epoch=0,blocks=[])
        payload=dict(header=header,shards=shards)
        # Geometry/census validation, without repacking large checkpoint blobs.
        for shard in shards:
            if len(shard['layers'])!=40:raise ValueError('Incomplete restored model')
        return payload
    except (KeyError,ValueError,TypeError,msgspec.DecodeError) as exc:
        raise CacheMiss('Invalid target model snapshot') from exc
