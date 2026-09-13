"""Private call packets; preserve storage aliases, never clone model State.

Only explicit model arguments and attention metadata enter this tree. KV,
weights, runner progress and draft feedback do not. No donor import is needed.
"""
import copy
import dataclasses
from enum import Enum
import torch


class CallPacket:
    def __init__(self, tree, *, shared_fields=()):
        self.shared_fields = frozenset(shared_fields)
        self.shared = {}
        self.storages = {}
        self.memo = {}
        self.tree = self._clone(tree)

    @staticmethod
    def raw(t):
        storage = t.untyped_storage()
        return torch.empty(0, dtype=torch.uint8, device=t.device).set_(
            storage, 0, (storage.nbytes(),), (1,))

    @staticmethod
    def storage_key(t):
        storage = t.untyped_storage()
        return (str(t.device), storage.data_ptr(), storage.nbytes())

    def _clone(self, value):
        if id(value) in self.memo:
            return self.memo[id(value)]
        if isinstance(value, torch.Tensor):
            key = self.storage_key(value)
            if key not in self.storages:
                raw = self.raw(value)
                dst = torch.empty_like(raw, pin_memory=value.is_pinned()) if value.device.type == 'cpu' else torch.empty_like(raw)
                dst.copy_(raw)
                self.storages[key] = dst
            result = torch.empty(0, dtype=value.dtype, device=value.device).set_(
                self.storages[key].untyped_storage(), value.storage_offset(), value.shape, value.stride())
        elif dataclasses.is_dataclass(value):
            result = copy.copy(value)
            for f in dataclasses.fields(value):
                source = getattr(value, f.name)
                if f.name in self.shared_fields and isinstance(source, torch.Tensor):
                    self.shared[id(source)] = source
                    setattr(result, f.name, source)
                else:
                    setattr(result, f.name, self._clone(source))
        elif type(value).__name__ == 'RopeDataProxy':
            result = copy.copy(value)
            result._data = self._clone(value._data)
        elif isinstance(value, dict):
            result = {k: self._clone(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            result = type(value)(self._clone(v) for v in value)
        elif value is None or isinstance(value, (str, int, float, bool, Enum)):
            return value
        else:
            raise TypeError(f'Unreviewed call-packet field: {type(value)}')
        self.memo[id(value)] = result
        return result

    def refresh(self, source):
        copies = {}
        self._bind(self.tree, source, copies)
        # Validate the complete packet before modifying any bank.
        for destination, origin in copies.values():
            destination.copy_(origin, non_blocking=True)

    def _bind(self, dst, src, copies):
        if isinstance(src, torch.Tensor):
            assert isinstance(dst, torch.Tensor)
            if id(dst) in self.shared:
                assert (self.storage_key(dst), dst.shape, dst.stride(), dst.storage_offset()) == (self.storage_key(src), src.shape, src.stride(), src.storage_offset()), "Immutable metadata backing changed"
                return
            assert (dst.shape, dst.stride(), dst.dtype, dst.device, dst.storage_offset()) == (src.shape, src.stride(), src.dtype, src.device, src.storage_offset())
            origin, destination = self.raw(src), self.raw(dst)
            assert origin.numel() == destination.numel()
            key = self.storage_key(dst)
            if key in copies:
                assert self.storage_key(copies[key][1]) == self.storage_key(origin), 'Source alias topology changed'
            else:
                copies[key] = (destination, origin)
        elif dataclasses.is_dataclass(src):
            assert type(dst) is type(src)
            for f in dataclasses.fields(src): self._bind(getattr(dst, f.name), getattr(src, f.name), copies)
        elif type(src).__name__ == 'RopeDataProxy':
            assert dst.idx == src.idx
            self._bind(dst._data, src._data, copies)
        elif isinstance(src, dict):
            assert isinstance(dst, dict) and dst.keys() == src.keys()
            for key in src: self._bind(dst[key], src[key], copies)
        elif isinstance(src, (list, tuple)):
            assert type(dst) is type(src) and len(dst) == len(src)
            for d, s in zip(dst, src): self._bind(d, s, copies)
        else:
            # Native FULL already fixes scalar bounds at capture. Do not
            # rebuild a graph on changing logging counts / CPU sequence lists.
            assert type(dst) is type(src)

    @property
    def bytes(self):
        return sum(t.numel() for t in self.storages.values())
