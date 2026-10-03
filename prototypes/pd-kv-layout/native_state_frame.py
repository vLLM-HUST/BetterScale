"""Direct State-lane views over an owned pinned host lease.

Only metadata is serialized. Payload bytes are DMA destinations/sources in the
arena; no torch.empty pinned allocation or full-object Python bytes assembly.
The portable schema remains compatible with page_transport.decode_snapshot.
"""
import ctypes
from dataclasses import dataclass
import json
import struct
from types import SimpleNamespace

import torch
from betterscale.live.runtime.page_transport import MAX_HEADER_BYTES, MAX_OBJECT_BYTES
from betterscale.models.qwen35.state_dma import descriptors


@dataclass(frozen=True)
class FramePlan:
    lanes: tuple
    header: bytes
    payload_bytes: int

    @property
    def byte_length(self):
        return len(self.header) + self.payload_bytes

    @classmethod
    def build(cls, lanes):
        lanes = tuple(sorted(lanes, key=lambda item: item[0]))
        schema, offset, names = [], 0, set()
        for name, state, blocks in lanes:
            if name in names or not blocks:
                raise ValueError("duplicate or empty State lane")
            names.add(name)
            size = state.logical_block_bytes * len(blocks)
            schema.append(dict(name=name,dtype=str(state.storage_dtype),
                shape=list(state.block_shape),
                span=state.physical_blocks_per_logical_block,
                count=len(blocks),offset=offset,size=size))
            offset += size
        raw = json.dumps(dict(schema=1,lanes=schema),separators=(",",":")).encode()
        # Whitespace is legal JSON. Align the first payload without changing
        # the existing wire schema or adding a full-payload encoding pass.
        raw += b" " * (-(4+len(raw)) % 64)
        if not lanes or len(raw) > MAX_HEADER_BYTES or offset > MAX_OBJECT_BYTES:
            raise ValueError("State frame exceeds bounded envelope")
        return cls(lanes, struct.pack("<I",len(raw))+raw, offset)

    def views(self, lease, *, initialize):
        if lease.size < self.byte_length or lease.state not in (("writing",) if initialize else ("writing", "sealed")):
            raise ValueError("frame needs a live correctly sized writer or immutable reader")
        base = lease.pointer
        if initialize:
            ctypes.memmove(base,self.header,len(self.header))
        elif ctypes.string_at(base,len(self.header)) != self.header:
            raise ValueError("State frame metadata differs from destination geometry")
        payloads, offset = {}, len(self.header)
        for name, state, blocks in self.lanes:
            size = state.logical_block_bytes * len(blocks)
            buffer = (ctypes.c_uint8 * size).from_address(base+offset)
            tensor = torch.frombuffer(buffer,dtype=state.storage_dtype).reshape(
                len(blocks)*state.physical_blocks_per_logical_block,*state.block_shape)
            payloads[name] = SimpleNamespace(tensor=tensor)
            offset += size
        # All consumers must keep the lease through their actual DMA event.
        # These borrowed views do not own the arena and may not escape it.
        return payloads

    def copy_descriptors(self, lease, *, to_host):
        payloads = self.views(lease,initialize=to_host)
        return descriptors(self.lanes,payloads,to_host=to_host)
