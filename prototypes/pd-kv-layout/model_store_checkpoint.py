"""Target-only checkpoint shards bypass Core/controller payload RPC."""
import time
import msgspec

FORMAT='tp2-target-shards-v1'
MIN_BYTES=30*(16*128*128*4+3*4096*2)


def validate_descriptors(descriptors,*,prefix=None):
    if not isinstance(descriptors,dict) or set(descriptors)!={'rank0','rank1'}:
        raise ValueError('Incomplete checkpoint descriptors')
    names=None;keys=set()
    for rank in (0,1):
        d=descriptors[f'rank{rank}'];layers=d['layers']
        if (type(d['rank']) is not int or d['rank']!=rank or d['acknowledged'] is not True
                or type(d['bytes']) is not int or not MIN_BYTES<d['bytes']<40*1024**2
                or not isinstance(d['key'],str) or not d['key'] or d['key'] in keys
                or not isinstance(layers,list) or len(layers)!=30
                or any(not isinstance(n,str) or not n for n in layers) or len(set(layers))!=30):
            raise ValueError('Invalid checkpoint descriptor')
        if prefix is not None and d['key']!=f'{prefix}/checkpoint/rank{rank}':
            raise ValueError('Checkpoint belongs to another writer')
        if names is not None and names!=set(layers):raise ValueError('Mismatched checkpoint census')
        names=set(layers);keys.add(d['key'])


def encode_snapshot(header,shard):
    from model_checkpoint import wire_encode
    return msgspec.msgpack.encode(dict(schema=1,identity=header['identity'],
                                      cursor=header['cursor'],shard=wire_encode(shard)))


def decode_snapshot(blob,header,rank,descriptor):
    from model_store import tensor
    from session import CacheMiss
    try:
        if len(blob)!=descriptor['bytes']:raise ValueError('Truncated target shard')
        data=msgspec.msgpack.decode(blob);shard=data['shard']
        if (data['schema']!=1 or data['identity']!=header['identity'] or data['cursor']!=header['cursor']
                or shard['rank']!=rank or shard['draft_valid'] is not False
                or len(shard['layers'])!=30 or set(shard['layers'])!=set(descriptor['layers'])):
            raise ValueError('Wrong checkpoint frontier/census')
        for layer in shard['layers'].values():
            if set(layer)!={'conv','recurrent'}:raise ValueError('Unexpected target plane')
            tensor(layer['conv'],(3,4096),'bfloat16');tensor(layer['recurrent'],(16,128,128),'float32')
        return shard
    except (KeyError,ValueError,TypeError,msgspec.DecodeError) as exc:
        raise CacheMiss('Invalid direct target checkpoint') from exc


def client_type():
    import sys
    from pathlib import Path
    from dram_store_fixture import CPU_ENV
    site=CPU_ENV/'lib/python3.12/site-packages';sys.path.insert(0,str(site))
    import mooncake.store
    if not Path(mooncake.store.__file__).resolve().is_relative_to(site):raise RuntimeError('Wrong Store runtime')
    return mooncake.store.MooncakeDistributedStore


def publish_worker(runner,header,shard):
    from model_store_clients import acquire,release
    start=time.perf_counter();rank=shard['rank'];options=header['stream_store']
    key=f"{options['prefix']}/checkpoint/rank{rank}";blob=encode_snapshot(header,shard)
    port=options['ports'][rank];client=acquire(runner,port,blob,client_type())
    try:
        if client.put(key,blob)!=0:raise IOError('Target checkpoint Store put failed')
        verify=bool(options.get('verify_transfer'))
        if verify and bytes(client.get(key))!=blob:raise RuntimeError('Checkpoint Store byte oracle failed')
    finally:release(runner,port)
    return dict(rank=rank,key=key,bytes=len(blob),layers=sorted(shard['layers']),acknowledged=True,
                exact_transfer_oracle=verify,seconds=time.perf_counter()-start)


def load_worker(runner,header,rank,descriptor):
    from model_store_clients import acquire,release
    from model_checkpoint import wire_decode
    from session import CacheMiss
    start=time.perf_counter();port=header['dense_store']['ports'][rank]
    client=acquire(runner,port,None,client_type())
    try:
        blob=client.get(descriptor['key'])
        if blob is None:raise CacheMiss('Missing direct target checkpoint')
        shard=wire_decode(decode_snapshot(blob,header,rank,descriptor))
    finally:release(runner,port)
    return shard,dict(bytes=descriptor['bytes'],seconds=time.perf_counter()-start,exact_transfer_oracle=False)


def verify_installed(root,seat,shard):
    import torch
    for name,data in shard['layers'].items():
        leaf=root.target[name]
        for actual,expected in ((leaf.conv.tensor[seat,:3],data['conv']),
                                (leaf.recurrent.tensor[seat*3],data['recurrent'])):
            if not torch.equal(actual.cpu().contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8)):
                raise RuntimeError('Checkpoint H2D byte oracle failed')
