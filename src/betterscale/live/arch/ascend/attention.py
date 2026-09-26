# SPDX-License-Identifier: Apache-2.0
"""Paged head256 FIA numerical leaf, without the native runner or graph owner.

The model root creates/rebinds this resource before capture, publishes each
wave before replay, and closes it after its captured readers retire. Only the
qualified native planner/launch adapter is reused; no patch is installed.
"""

import ctypes
from types import SimpleNamespace

import torch

from betterscale.patches.qwen_fia import check_library
from betterscale.patches.qwen_fia.wave import Frame, Planner


class PagedAttentionWave:
    def __init__(self, *, tokens, requests, context_tokens, heads, key, value):
        if (
            tokens < requests
            or requests < 1
            or context_tokens < 1
            or key.shape != value.shape
            or key.ndim != 4
            or key.shape[1] != 128
            or key.shape[-1] != 256
            or key.dtype != torch.bfloat16
            or value.dtype != key.dtype
            or key.device != value.device
        ):
            raise ValueError("paged FIA requires BF16 [pages,128,KVheads,256]")
        self.planner = Planner(check_library(), heads=heads, kvheads=key.shape[2])
        self.device = key.device
        self.tokens, self.requests = tokens, requests
        self.context_tokens = context_tokens
        self.heads, self.kvheads = heads, key.shape[2]
        # These are planning descriptors, never model activations. They survive
        # capture and are accounted before State fitting with other fixed costs.
        self.query = torch.empty(tokens, heads, 256, dtype=key.dtype, device=key.device)
        self.output = torch.empty_like(self.query)
        self.mask = torch.triu(
            torch.ones(2048, 2048, dtype=torch.bool, device=key.device), diagonal=1
        )
        self.frame = Frame(
            tokens,
            (context_tokens + 127) // 128,
            key.device,
            torch.npu.current_stream(key.device),
            requests=requests,
        )
        self.bind_cache(key, value)
        self.closed = False

    def bind_cache(self, key, value):
        """Refresh descriptors without retaining calibration State storage.

        Borrowed tensor views would keep the old allocation alive during fit.
        Only shape/address descriptors survive; the root owns their validity
        and must rebind before planning or capturing against replacement State.
        """

        def descriptor(tensor):
            address = tensor.data_ptr()
            return SimpleNamespace(shape=tensor.shape, data_ptr=lambda: address)

        self.planner.fixtures = (
            self.query,
            descriptor(key),
            descriptor(value),
            self.output,
            256**-0.5,
        )

    def prepare(self, query_lengths, sequence_lengths, block_rows):
        """Host metadata only; no full-context KV gather or host task updates."""
        if self.closed:
            raise RuntimeError("paged attention wave is closed")
        if (
            not len(query_lengths)
            == len(sequence_lengths)
            == len(block_rows)
            == self.requests
        ):
            raise ValueError("FIA metadata must describe every real request")
        if sum(query_lengths) != self.tokens:
            raise ValueError("FIA query lengths must cover the captured token shape")
        table = torch.zeros(self.requests, self.frame.columns, dtype=torch.int32)
        ends = []
        for i, (q, length, blocks) in enumerate(
            zip(query_lengths, sequence_lengths, block_rows, strict=True)
        ):
            if not 0 < q <= length <= self.context_tokens:
                raise ValueError("invalid FIA sequence/query envelope")
            if len(blocks) != (length + 127) // 128 or any(
                type(b) is not int or not 0 <= b < self.planner.fixtures[1].shape[0]
                for b in blocks
            ):
                raise ValueError(
                    "FIA block row must cover its sequence in the owned pool"
                )
            if len(set(blocks)) != len(blocks):
                raise ValueError("FIA sequence cannot alias its own pages")
            table[i, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
            ends.append((ends[-1] if ends else 0) + q)
        metadata = SimpleNamespace(
            actual_seq_lengths_q=ends,
            seq_lens_list=sequence_lengths,
            attn_mask=self.mask,
        )
        self.frame.prepare(self.planner, metadata, table, self.requests)

    def __call__(self, query, key, value):
        if self.closed or self.frame.plan is None:
            raise RuntimeError("prepare paged attention before numerical launch")
        if (
            query.shape != (self.tokens, self.heads, 256)
            or key.shape != self.planner.fixtures[1].shape
            or value.shape != key.shape
            or not all(t.is_contiguous() for t in (query, key, value))
        ):
            raise ValueError("FIA launch differs from its planned geometry")
        output = torch.empty_like(query)
        planner, frame = self.planner, self.frame
        plan = planner.check(planner.lib.plan_clone(frame.plan))
        try:
            scratch = torch.empty(
                frame.workspace, dtype=torch.uint8, device=query.device
            )
            tensors = (
                query,
                self.mask,
                frame.q,
                frame.kv,
                frame.table,
                output,
                scratch,
                key,
                value,
            )
            pointers = (ctypes.c_uint64 * 9)(*[t.data_ptr() for t in tensors])
            planner.status(planner.lib.plan_bind(plan, pointers))
            planner.status(
                planner.lib.plan_bind_metadata(plan, frame.tiling.data_ptr())
            )
            planner.status(
                planner.lib.plan_launch(
                    plan, torch.npu.current_stream(query.device).npu_stream
                )
            )
        finally:
            planner.status(planner.lib.plan_release(plan))
        return output

    def consumed(self):
        self.frame.release()

    def close(self):
        if self.closed:
            return
        torch.npu.current_stream(self.device).synchronize()
        if self.frame.plan is not None:
            self.planner.status(self.planner.lib.plan_release(self.frame.plan))
            self.frame.plan = None
        self.planner.fixtures = None
        self.query = self.output = self.mask = None
        self.frame = None
        self.closed = True
