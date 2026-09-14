"""Prepare the finite decode envelope before the worker advertises READY.

Only input/metadata programs run here, never model forward or KV writes. Use
native persistent destinations and both pinned CPU carriers, just as serving
does. Numerical feedback is copied into owned producer inputs during replay;
no fake request or fabricated acceptance is installed in the live input batch.
"""

from contextlib import contextmanager
import logging

import numpy as np
import torch

from ._producer import Slot

log = logging.getLogger(__name__)


def shapes(r):
    """Actual captured target descriptors, not every integer token budget."""
    limit = min(r.max_num_reqs, r._cross_step_bounds.max_requests)
    descriptors = r.model._decode_pair.packets[0]
    return sorted(
        {
            (n, key.num_reqs, key.num_tokens)
            for key in descriptors
            for n in range(1, limit + 1)
            if key.num_reqs is not None
            and n <= key.num_reqs
            and n * 6 <= key.num_tokens <= r.max_num_reqs * 6
        },
        reverse=True,
    )


@contextmanager
def preserve_inputs(r):
    """Restore native input exemplars, without cloning weights or the KV pool."""
    fields = r._host_source_slots.fields
    tensors = {}

    def save(t):
        if isinstance(t, torch.Tensor) and id(t) not in tensors:
            tensors[id(t)] = (t, t.clone())

    for field in fields:
        for slot in field.slots:
            save(slot)
        save(getattr(field.owner, "gpu", None))
    for name in ("num_computed_tokens", "positions", "seq_lens"):
        save(getattr(r, name))
    attrs = {
        name: getattr(r, name)
        for name in ("attn_state", "with_prefill", "query_lens", "logits_indices")
        if hasattr(r, name)
    }
    selected = [(field, getattr(field.owner, field.name)) for field in fields]
    build_args = r._cross_step_bounds.build_args
    try:
        yield
    finally:
        torch.npu.synchronize()
        for tensor, before in tensors.values():
            tensor.copy_(before)
        for field, before in selected:
            setattr(field.owner, field.name, before)
            if field.numpy_name is not None:
                setattr(field.owner, field.numpy_name, before.numpy())
        for name, value in attrs.items():
            setattr(r, name, value)
        r._cross_step_bounds.build_args = build_args
        torch.npu.synchronize()


@torch.inference_mode()
def prepare(producer, metadata):
    from vllm.config import CUDAGraphMode

    r = producer.r
    envelope = shapes(r)
    assert envelope, "No captured target descriptor for stable K5 preparation"
    assert r.input_batch.num_reqs == 0, "Warmup must precede request admission"
    assert not producer.slots and not metadata.entries
    with preserve_inputs(r):
        # Warm every request count in both independently reused ingress banks.
        # Largest first encourages scratch reuse by the shared graph pool.
        for n in sorted({shape[0] for shape in envelope}, reverse=True):
            r.num_computed_tokens.fill_(128)
            for bank in (0, 1):
                slot = Slot(r, n, producer.ingress, producer.pool)
                slot.capture()
                producer.slots[n, bank] = slot

        for bank in (0, 1):
            for field in r._host_source_slots.fields:
                field.select(bank)
            for n, nr, nt in envelope:
                counts = np.full(n, 6, dtype=np.int32)
                r.input_batch.num_computed_tokens_cpu_tensor.fill_(128)
                r.input_batch.num_prompt_tokens_cpu_tensor.zero_()
                r.num_computed_tokens.fill_(128)
                r._build_attn_state(n, counts, np.ones(n, dtype=np.int32))
                r.with_prefill = False
                r.query_lens = torch.from_numpy(counts)
                r.optimistic_seq_lens_cpu.zero_()
                r.optimistic_seq_lens_cpu[:n].fill_(134)
                r._dsa_positions_cpu_buf.zero_()
                r._dsa_positions_cpu_buf[: n * 6].copy_(
                    (torch.arange(6).repeat(n) + 128)
                )
                # Execute the captured producer, then apply the native FULL
                # padding protocol exactly where execute_model would apply it.
                producer.slots[n, 0].graph.replay()
                r.query_start_loc.cpu.zero_()
                r.query_start_loc.cpu[: n + 1].copy_(
                    torch.arange(n + 1, dtype=torch.int32) * 6
                )
                actual_nr = r._pad_query_start_loc_for_fia(
                    r.query_start_loc, nt, nr, n, CUDAGraphMode.FULL, nr
                )
                assert actual_nr == nr
                metadata.capture(
                    num_tokens=n * 6,
                    num_reqs=n,
                    num_tokens_padded=nt,
                    num_reqs_padded=nr,
                    max_query_len=6,
                    use_spec_decode=True,
                    num_scheduled_tokens_np=counts,
                )
    log.info(
        "Prepared decode auxiliary graphs before READY: producer=%d metadata=%d",
        len(producer.slots),
        len(metadata.entries),
    )
