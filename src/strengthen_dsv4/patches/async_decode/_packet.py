"""Private call packets; preserve storage aliases, never clone model State.

Only explicit model arguments and attention metadata enter this tree. KV,
weights, runner progress and draft feedback do not. No donor import is needed.
"""

import copy
import dataclasses
from enum import Enum
import torch


class CallPacket:
    def __init__(self, tree, *, shared_fields=(), native_views=False):
        self.native_views = native_views
        self.shared_fields = frozenset(shared_fields)
        self.shared = {}
        self.storages = {}
        self.memo = {}
        self.tree = self._clone(tree)

    @staticmethod
    def raw(t):
        storage = t.untyped_storage()
        return torch.empty(0, dtype=torch.uint8, device=t.device).set_(
            storage, 0, (storage.nbytes(),), (1,)
        )

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
                dst = (
                    torch.empty_like(raw, pin_memory=value.is_pinned())
                    if value.device.type == "cpu"
                    else torch.empty_like(raw)
                )
                dst.copy_(raw)
                self.storages[key] = dst
            result = torch.empty(0, dtype=value.dtype, device=value.device).set_(
                self.storages[key].untyped_storage(),
                value.storage_offset(),
                value.shape,
                value.stride(),
            )
        elif dataclasses.is_dataclass(value):
            result = copy.copy(value)
            for f in dataclasses.fields(value):
                source = getattr(value, f.name)
                if f.name in self.shared_fields and isinstance(source, torch.Tensor):
                    self.shared[id(source)] = source
                    setattr(result, f.name, source)
                else:
                    setattr(result, f.name, self._clone(source))
        elif type(value).__name__ == "RopeDataProxy":
            result = copy.copy(value)
            result._data = self._clone(value._data)
        elif isinstance(value, dict):
            result = {k: self._clone(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            result = type(value)(self._clone(v) for v in value)
        elif value is None or isinstance(value, (str, int, float, bool, Enum)):
            return value
        else:
            raise TypeError(f"Unreviewed call-packet field: {type(value)}")
        self.memo[id(value)] = result
        return result

    def captured_copies(self, source):
        """Freeze native FULL input backings, not live Python metadata views.

        The native graph already reads these persistent capture addresses.
        Put the bank copies in the same graph so no per-layer Python traversal
        or per-copy host launches survive in the serving path.
        """
        copies = {}
        self._bind(self.tree, source, copies, seen=set())
        return [(d, s) for d, s in copies.values() if d.device.type != "cpu"]

    def refresh(self, source):
        """Publish a live packet before replay, preserving all storage aliases."""
        copies = {}
        self._bind(self.tree, source, copies, seen=set())
        for destination, origin in copies.values():
            destination.copy_(origin, non_blocking=True)

    def _bind(self, dst, src, copies, path=(), seen=None):
        # Metadata are a DAG shared across model layers, not a tree. Validate
        # each actual source/destination pair once without hiding alias splits.
        key = (id(dst), id(src))
        if key in seen:
            return
        seen.add(key)
        if isinstance(src, torch.Tensor):
            assert isinstance(dst, torch.Tensor)
            if id(dst) in self.shared:
                assert (
                    self.storage_key(dst),
                    dst.shape,
                    dst.stride(),
                    dst.storage_offset(),
                ) == (
                    self.storage_key(src),
                    src.shape,
                    src.stride(),
                    src.storage_offset(),
                ), "Immutable metadata backing changed"
                return
            layout = (dst.stride(), dst.dtype, dst.device, dst.storage_offset())
            source_layout = (src.stride(), src.dtype, src.device, src.storage_offset())
            assert (
                layout == source_layout
            ), f"{path}: capture {dst.shape}/{layout}, runtime {src.shape}/{source_layout}"
            if dst.shape != src.shape:
                # Native DSACP changes the visible leading row count while
                # FULL retains the capture-time view over the SAME allocation.
                # Copy the full backing, as native graph observes it, not just
                # the shorter live view. Arguments and non-leading dims stay fixed.
                assert (
                    self.native_views
                    and path[:1] == (2,)
                    and dst.ndim == src.ndim
                    and dst.shape[1:] == src.shape[1:]
                ), f"{path}: capture {dst.shape}, runtime {src.shape}"
            origin, destination = self.raw(src), self.raw(dst)
            assert (
                origin.numel() == destination.numel()
            ), f"{path}: backing extent {destination.numel()} -> {origin.numel()}"
            key = self.storage_key(dst)
            if key in copies:
                assert self.storage_key(copies[key][1]) == self.storage_key(
                    origin
                ), "Source alias topology changed"
            else:
                copies[key] = (destination, origin)
        elif dataclasses.is_dataclass(src):
            assert type(dst) is type(src)
            for f in dataclasses.fields(src):
                self._bind(
                    getattr(dst, f.name),
                    getattr(src, f.name),
                    copies,
                    path + (f.name,),
                    seen,
                )
        elif type(src).__name__ == "RopeDataProxy":
            assert dst.idx == src.idx
            self._bind(dst._data, src._data, copies, path + ("rope",), seen)
        elif isinstance(src, dict):
            assert isinstance(dst, dict) and dst.keys() == src.keys()
            for key in src:
                self._bind(dst[key], src[key], copies, path + (key,), seen)
        elif isinstance(src, (list, tuple)):
            assert type(dst) is type(src) and len(dst) == len(src)
            for i, (d, s) in enumerate(zip(dst, src)):
                self._bind(d, s, copies, path + (i,), seen)
        else:
            # Native FULL already fixes scalar bounds at capture. Do not
            # rebuild a graph on changing logging counts / CPU sequence lists.
            assert type(dst) is type(src)

    @property
    def bytes(self):
        return sum(t.numel() for t in self.storages.values())
