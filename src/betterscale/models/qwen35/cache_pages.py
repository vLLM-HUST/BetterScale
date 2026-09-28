"""Weak FA residency: free pages remain hits until the native pool reuses them.

Only the scheduler thread mutates this index. It owns no physical references;
execution and transfer lifetimes remain native pool references.
"""

import hashlib
import json
import struct


def page_keys(tokens, salt, block_size, count, generation):
    # The last accepted block stays private, including at an exact boundary.
    # A sealed key includes the next token: draft KV may use that lookahead.
    sealed = max(0, (len(tokens) - 2) // block_size)
    chain = hashlib.sha256(json.dumps(salt).encode())
    keys = []
    for i in range(count):
        if i < sealed:
            chunk = tokens[i * block_size : (i + 1) * block_size]
            chain.update(struct.pack(f"<{len(chunk)}q", *chunk))
            identity = chain.copy()
            identity.update(struct.pack("<q", tokens[(i + 1) * block_size]))
            keys.append("fa:" + identity.hexdigest())
        else:
            keys.append(f"tail:{generation}:{i}")
    return tuple(keys)


class PageResidency:
    def __init__(self, pool):
        self.pool = pool
        self.by_key = {}
        self.by_slot = {}
        allocate = pool.get_new_blocks

        def reuse(count):
            blocks = allocate(count)
            self.invalidate(blocks)
            return blocks

        pool.get_new_blocks = reuse
        release = pool.free_blocks

        def free(blocks):
            blocks = tuple(blocks)
            release(blocks)
            # Native unhashed frees prepend. Our FA index intentionally does
            # not publish native hash hits, but valid cold pages still deserve
            # LRU retention behind genuinely unused/invalid free capacity.
            retained = [
                b for b in blocks if b.ref_cnt == 0 and b.block_id in self.by_slot
            ]
            for block in retained:
                pool.free_block_queue.remove(block)
            pool.free_block_queue.append_n(retained)

        pool.free_blocks = free

    def invalidate(self, blocks):
        for block in blocks:
            key = self.by_slot.pop(block.block_id, None)
            if key is not None:
                self.by_key.pop(key, None)

    def remember(self, keys, blocks):
        for key, block in zip(keys, blocks, strict=True):
            self.invalidate((block,))
            old = self.by_key.pop(key, None)
            if old is not None:
                self.by_slot.pop(old.block_id, None)
            self.by_key[key] = block
            self.by_slot[block.block_id] = key

    def acquire(self, keys):
        # Do not alias a writable tail held by another request/seat. The first
        # cut reuses only free placements, even for sealed pages.
        hits = {
            i: self.by_key[key]
            for i, key in enumerate(keys)
            if key in self.by_key and self.by_key[key].ref_cnt == 0
        }
        self.pool.touch(tuple(hits.values()))
        return hits

    def clear(self):
        self.by_key.clear()
        self.by_slot.clear()
