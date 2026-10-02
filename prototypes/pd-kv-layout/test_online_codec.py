import os
import pytest
import online_codec as wire
from online_codec import ResidentWireSink
from types import SimpleNamespace


def fixture():
    return ResidentWireSink(SimpleNamespace())


def test_lossless_compressed_and_raw_frames(monkeypatch):
    monkeypatch.setattr(wire,"MIN_COMPRESS",1)
    codec=fixture()
    for data in (b"state"*10000,os.urandom(10000)):
        encoded=codec.encode(data)
        assert codec.decode(encoded)==data
    assert codec.encode(b"state"*10000).startswith(wire.MAGIC)
    random=os.urandom(10000)
    assert codec.encode(random)==random


def test_corruption_trailing_frames_and_embedded_limits(monkeypatch):
    monkeypatch.setattr(wire,"MIN_COMPRESS",1)
    codec=fixture();data=b"state"*10000;encoded=codec.encode(data)
    with pytest.raises(ValueError):codec.decode(encoded[:-1])
    with pytest.raises(ValueError):codec.decode(encoded+b"trailing")
    with pytest.raises(ValueError):codec.decode(encoded+encoded)
    changed=encoded[:-1]+bytes([encoded[-1]^1])
    with pytest.raises(ValueError):codec.decode(changed)
    monkeypatch.setattr(wire,"MAX_WIRE",1000)
    with pytest.raises(ValueError):codec.decode(encoded)


def test_unknown_size_and_missing_checksum_are_not_admitted():
    codec=fixture();data=b"state"*10000
    for options in (dict(write_content_size=False,write_checksum=True),dict(write_checksum=False)):
        frame=codec.codec.ZstdCompressor(**options).compress(data)
        with pytest.raises(ValueError):codec.decode(frame)


def test_replica_repair_remains_wire_opaque(monkeypatch):
    monkeypatch.setattr(wire,"MIN_COMPRESS",1)
    values={};ensured=[]
    raw=SimpleNamespace(put=lambda k,v:values.update({k:v}),get=values.__getitem__,
        ensure=lambda k:ensured.append(k) or k in values)
    codec=ResidentWireSink(raw);data=b"state"*10000
    codec.put("k",data)
    assert values["k"].startswith(wire.MAGIC)
    assert codec.ensure("k") and ensured==["k"] and codec.get("k")==data
