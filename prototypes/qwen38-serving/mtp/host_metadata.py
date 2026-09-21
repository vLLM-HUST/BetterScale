"""One bank's persistent CPU views; filling never constructs a Torch tensor.

The caller owns the pinned-source reuse fence. Publication and device-authored
accepted counts remain in MTPFrame, not in this host construction program.
"""
from types import SimpleNamespace

import numpy as np


class HostMetadata:
    def __init__(self, core):
        self.capacity, self.width, self.decode = core.capacity, core.width, core.decode
        for owner, names in (
            (core, ('cu', 'prefill_conv', 'verify_conv', 'initial', 'accepted',
                    'prefill_map', 'verify_map', 'restore', 'verify_ids')),
            (core.prefill, ('cu', 'state')),
            (core.verify, ('cu', 'slots', 'accepted')),
        ):
            views = SimpleNamespace(**{name: getattr(owner, name).numpy() for name in names})
            if owner is core:
                self.h = views
            elif owner is core.prefill:
                self.p = views
            else:
                self.v = views
        self.indices = {size: value.numpy() for size, value in core.prefill.indices.items()}

    @staticmethod
    def offsets(destination, lengths):
        destination[0] = 0
        np.cumsum(lengths, out=destination[1:len(lengths)+1])
        destination[len(lengths)+1:] = destination[len(lengths)]

    def prepare(self, lengths, roles, slots, initial):
        n = len(lengths)
        assert 0 < n <= 8 and sum(lengths) <= self.capacity
        assert len(roles) == len(slots) == len(initial) == n
        assert all(x > 0 for x in lengths)
        assert all(x <= self.width for x, role in zip(lengths, roles) if role)
        h, p, v = self.h, self.p, self.v
        self.offsets(h.cu, lengths)
        h.verify_conv.fill(-1)
        v.slots.fill(-1)
        h.accepted.fill(1)
        v.accepted.fill(1)
        ids = np.flatnonzero(roles)
        h.verify_ids.fill(0)
        h.verify_ids[:len(ids)] = ids
        if self.decode:
            assert all(roles), 'prefill reached a verification-only graph'
            v.cu[:] = h.cu
            h.verify_conv[:n, 0] = slots[:, 0]
            v.slots[:n] = slots
            return

        pre = np.flatnonzero(np.logical_not(roles))
        assert len(pre), 'pure verification belongs to its small graph, not mixed'
        h.prefill_conv.fill(-1)
        h.prefill_conv[pre, 0] = slots[pre, 0]
        h.verify_conv[ids, 0] = slots[ids, 0]
        h.initial.fill(False)
        h.initial[:n] = initial
        p.state.fill(0)
        p.state[:len(pre), 0] = slots[pre, 0]
        p.state[:len(pre), 1] = np.asarray(initial)[pre]
        v.slots[:len(ids)] = slots[ids]
        h.restore.fill(0)
        for rows, cu, mapping, base in ((pre, p.cu, h.prefill_map, 0),
                                       (ids, v.cu, h.verify_map, self.capacity)):
            sub_lengths = [lengths[i] for i in rows]
            self.offsets(cu, sub_lengths)
            mapping.fill(0)
            cursor = 0
            for i in rows:
                length = lengths[i]
                mapping[cursor:cursor+length] = np.arange(h.cu[i], h.cu[i+1])
                h.restore[h.cu[i]:h.cu[i+1]] = np.arange(base+cursor, base+cursor+length)
                cursor += length
        for size, dest in self.indices.items():
            dest[:, 0] = 8  # permanent empty sentinel row
            dest[:, 1] = 0
            cursor = 0
            for packed, i in enumerate(pre):
                count = (lengths[i]+size-1)//size
                dest[cursor:cursor+count, 0] = packed
                dest[cursor:cursor+count, 1] = np.arange(count)
                cursor += count
