"""NPU -> two pinned staging slots -> DRAM Store -> NPU byte oracle.

No model, no distributed ownership, no claim of compute/transfer overlap.
Store threads wait device completion before reading; slots are reused only
when Store has finished reading them. All operations are drained on failure.
"""
from concurrent.futures import ThreadPoolExecutor
import os
import time


def check(p, d):
    import torch
    import torch_npu
    from mooncake.store import ReplicateConfig
    torch.npu.set_device(0)
    size = 128 * 10 * 2 * 2 * 256 * 2  # 128 tokens of global target dense KV.
    host = [torch.empty(size, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
    output = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    reference = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    source = torch.empty(size, dtype=torch.uint8, device='npu:0')
    restored = torch.empty_like(source)
    stream = torch.npu.Stream(device=0)
    config = ReplicateConfig()
    config.replica_num = 1
    registered, keys, pending = [], [], [None, None]
    count = 8
    pool = ThreadPoolExecutor(max_workers=1)
    start = time.perf_counter()

    def put_after_event(event, slot, key):
        # Background host thread owns no device tensor operations.
        event.synchronize()
        rc = p.batch_put_from_multi_buffers([key], [[slot.data_ptr()]], [[size]], config)
        assert rc == [0], ('put', rc)

    try:
        for client, tensor in [(p, h) for h in host] + [(d, output)]:
            assert client.register_buffer(tensor.data_ptr(), size) == 0
            registered.append((client, tensor))
        for i in range(count):
            slot = i % 2
            if pending[slot] is not None:
                pending[slot].result()  # Includes D2H fence and Store read completion.
            key = f'pd-npu-staging-{os.getpid()}-{i}'
            keys.append(key)
            with torch.npu.stream(stream):
                source.fill_(i * 17 + 3)
                host[slot].copy_(source, non_blocking=True)
                ready = torch.npu.Event()
                ready.record(stream)
            pending[slot] = pool.submit(put_after_event, ready, host[slot], key)
        for future in pending:
            future.result()
        published_ms = (time.perf_counter() - start) * 1000
        for i, key in enumerate(keys):
            rc = d.get_into_ranges(
                [output.data_ptr()], [[key]], [[[0]]], [[[0]]], [[[size]]])
            assert rc == [[[size]]], ('restore', rc)
            with torch.npu.stream(stream):
                restored.copy_(output, non_blocking=True)
                reference.copy_(restored, non_blocking=True)
                done = torch.npu.Event()
                done.record(stream)
            done.synchronize()  # Host validation and next output reuse are now safe.
            assert (reference.numpy() == i * 17 + 3).all(), ('bytes', i)
        return {'chunks': count, 'chunk_bytes': size, 'ring_slots': 2,
                'exact': True, 'publish_ms': published_ms,
                'scope': 'NPU0 pinned staging + same-host DRAM Store; no model/compute overlap'}
    finally:
        pool.shutdown(wait=True)
        stream.synchronize()
        for client, tensor in reversed(registered):
            client.unregister_buffer(tensor.data_ptr())
        for key in keys:
            p.remove(key, force=True)  # Only this probe's uniquely named objects.
