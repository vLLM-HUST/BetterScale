# SPDX-License-Identifier: Apache-2.0
# Adapted from pinned vLLM-Ascend worker/model_runner_v1.py.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
"""Explicit stable-K5 host construction / banked ingress / device derivation.

Adapted from the pinned donor preparation arithmetic and LiveInfer's invocation
lifetime protocol. No generic FakeTensor replay is used in the serving path.
Prefill, turnover, hybrid feedback and other speculative widths stay native.
"""

import numpy as np
import torch


def geometry(n):
    drafts = np.full(n, 5, dtype=np.int32)
    samples = np.cumsum(drafts + 1, dtype=np.int32)
    cumulative = np.cumsum(drafts, dtype=np.int32)
    target = np.repeat(samples - drafts - 1, drafts) + (
        np.arange(n * 5) - np.repeat(cumulative - drafts, drafts)
    )
    return dict(
        qsl=torch.arange(n + 1, dtype=torch.int32) * 6,
        rows=torch.arange(n).repeat_interleave(6),
        query=torch.arange(6).repeat(n),
        scheduled=torch.full((n,), 6, dtype=torch.int32),
        previous=torch.arange(n),
        logits=torch.arange(6 * n),
        target=torch.from_numpy(target.astype(np.int32)),
        bonus=torch.from_numpy(samples - 1),
        samples=torch.from_numpy(samples),
        drafts=torch.from_numpy(cumulative),
        sample_index=torch.arange(n) * 6,
        draft_index=(torch.arange(n)[:, None] * 6 + torch.arange(1, 6)).flatten(),
    )


class Slot:
    def __init__(self, r, n, ingress, pool):
        self.n = n
        self.r = r
        self.ingress = ingress
        self.pool = pool
        self.host = {}
        self.device = {}

        def field(name, example):
            self.host[name] = torch.empty_like(example, device="cpu", pin_memory=True)
            self.device[name] = torch.empty_like(example, device=r.device)

        field("budget", r.input_batch.num_computed_tokens_cpu_tensor[:n])
        field("previous_drafts", r.prev_num_draft_tokens.cpu)
        field("accepted", r.num_accepted_tokens.cpu)
        for i, table in enumerate(r.input_batch.block_table.block_tables):
            field(f"blocks{i}", table.block_table.cpu[:n])
        self.constants = {k: v.to(r.device) for k, v in geometry(n).items()}
        # Owned destinations have known K5 geometry before any request exists.
        # Native feedback may still be None at startup; replay copies its actual
        # values here, rather than capturing the address of a first request.
        self.valid = torch.ones(n, dtype=torch.int32, device=r.device)
        self.sampled = torch.zeros(n, dtype=r.input_ids.gpu.dtype, device=r.device)
        self.drafted = torch.zeros((n, 5), dtype=torch.int32, device=r.device)
        self.ready = None
        self.consumed = None
        self.graph = None
        self.output = None

    def project(self):
        r = self.r
        n = self.n
        # CPU sources and device destinations have different reuse gates.
        if self.ready is not None:
            self.ready.synchronize()
        self.host["budget"].copy_(r.input_batch.num_computed_tokens_cpu_tensor[:n])
        self.host["previous_drafts"].copy_(r.prev_num_draft_tokens.cpu)
        self.host["accepted"].copy_(r.num_accepted_tokens.cpu)
        for i, table in enumerate(r.input_batch.block_table.block_tables):
            self.host[f"blocks{i}"].copy_(table.block_table.cpu[:n])
        with torch.npu.stream(self.ingress):
            if self.consumed is not None:
                self.ingress.wait_event(self.consumed)
            for name, value in self.host.items():
                self.device[name].copy_(value, non_blocking=True)
            self.ready = torch.npu.Event()
            self.ready.record()

    def derive(self):
        from vllm_ascend.spec_decode.utils import (
            update_num_computed_tokens_for_batch_change,
        )
        from vllm_ascend.worker.model_runner_v1 import (
            SpecDecodeMetadata,
            lmhead_tp_enable,
        )

        r = self.r
        n = self.n
        t = n * 6
        c = self.constants
        d = self.device
        for i, table in enumerate(r.input_batch.block_table.block_tables):
            table.block_table.gpu[:n].copy_(d[f"blocks{i}"])
        r.prev_positions.gpu[:n].copy_(c["previous"])
        r.prev_num_draft_tokens.gpu.copy_(d["previous_drafts"])
        r.num_accepted_tokens.gpu.copy_(d["accepted"])
        r.query_start_loc.gpu[: n + 1].copy_(c["qsl"])
        r.query_start_loc.gpu[n + 1 :].fill_(-1)
        r.input_ids.gpu.scatter_(0, c["sample_index"], self.sampled)
        r.input_ids.gpu.scatter_(
            0, c["draft_index"], self.drafted.to(torch.int32).flatten()
        )
        update_num_computed_tokens_for_batch_change(
            r.num_computed_tokens,
            r.num_accepted_tokens.gpu[:n],
            r.prev_positions.gpu[:n],
            self.valid,
            r.prev_num_draft_tokens.gpu,
            d["budget"],
        )
        r.req_indices.gpu[:t].copy_(c["rows"])
        r.query_pos.gpu[:t].copy_(c["query"])
        r.num_scheduled_tokens.gpu[:n].copy_(c["scheduled"])
        r.positions[:t].copy_(
            r.num_computed_tokens[c["rows"]].to(torch.int64) + c["query"]
        )
        r.seq_lens[:n].copy_(r.num_computed_tokens[:n] + c["scheduled"])
        r.seq_lens[n:].zero_()
        # This is a real device action; it is NEVER called from host projection.
        r.input_batch.block_table.compute_slot_mapping(
            n, r.query_start_loc.gpu[: n + 1], r.positions[:t]
        )
        r.discard_request_mask.gpu[:n].zero_()
        r.num_decode_draft_tokens.gpu[:n].fill_(5)
        r.num_decode_draft_tokens.gpu[n:].fill_(-1)
        draft_ids = r.input_ids.gpu[c["logits"]][c["target"] + 1]
        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_ids,
            num_draft_tokens=[5] * n,
            cu_num_draft_tokens=c["drafts"],
            cu_num_sampled_tokens=c["samples"],
            target_logits_indices=c["target"],
            bonus_logits_indices=c["bonus"],
            logits_indices=c["logits"],
        )
        logits = c["logits"]
        if lmhead_tp_enable():
            logits = torch.nn.functional.pad(logits, (0, r.max_num_reqs * 6 - t))
        return logits, metadata, t

    def capture(self):
        """Startup only; the caller owns initialized exemplars and restoration."""
        assert self.graph is None
        for name, value in self.device.items():
            value.zero_()
        self.device["budget"].fill_(128)
        self.device["accepted"].fill_(1)
        self.device["previous_drafts"].fill_(5)
        before = self.r.num_computed_tokens.clone()
        try:
            self.derive()  # compile/warm kernels outside graph capture
            self.r.num_computed_tokens.copy_(before)
            torch.npu.synchronize()
            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(self.graph, pool=self.pool):
                self.output = self.derive()
        finally:
            self.r.num_computed_tokens.copy_(before)

    def replay(self):
        r = self.r
        n = self.n
        compute = torch.npu.current_stream()
        compute.wait_event(self.ready)
        # Native feedback is numerical State, not host-predicted metadata.
        self.valid.copy_(r.valid_sampled_token_count_gpu[:n])
        self.sampled.copy_(r.input_batch.prev_sampled_token_ids[:n, 0])
        self.drafted.copy_(r._draft_token_ids[:n, :5])
        assert self.graph is not None, "Producer shape was not prepared before READY"
        self.graph.replay()
        self.consumed = torch.npu.Event()
        self.consumed.record()
        r.logits_indices = self.constants["logits"]
        return self.output


class DecodeProducer:
    def __init__(self, worker):
        r = self.r = worker.model_runner
        assert not r.model_config.is_hybrid and not r.need_accepted_tokens
        assert r.use_async_spec_decode and not r.use_dcp and not r.lora_config
        self.native = r._cross_step_bounds.prepare
        self.ingress = torch.npu.Stream(device=r.device)
        # All preparation graphs are serialized; their returned tensors stay
        # alive in Slot.output. Share scratch instead of creating a private
        # expandable-segment reservation for every request count/bank.
        self.pool = torch.npu.graph_pool_handle()
        r._decode_shadow = self
        self.slots = {}
        self.active = None
        self.sequence = 0
        r._cross_step_bounds.prepare = self.prepare

    def prepare(self, schedule, counts):
        r = self.r
        n = r.input_batch.num_reqs
        self.active = None
        if not r._cross_step_bounds.authorized(schedule, counts):
            return self.native(schedule, counts)
        key = (n, self.sequence % 2)
        if key not in self.slots:
            return self.native(schedule, counts)
        # Decline any special masked/partial query rather than approximating it.
        upper = r.input_batch.num_computed_tokens_cpu[:n] + counts
        if any(
            int(upper[i]) < r.requests[rid].num_tokens
            or int(upper[i]) > r.max_model_len
            for i, rid in enumerate(r.input_batch.req_ids)
        ):
            return self.native(schedule, counts)
        assert r._draft_token_ids.ndim == 2 and r._draft_token_ids.shape[1] == 5
        r._build_attn_state(n, counts, np.ones(n, dtype=np.int32))
        r.with_prefill = False
        r._compute_prev_positions(n)
        cu = r._get_cumsum_and_arange(counts, r.query_pos.np)
        rows = np.repeat(r.arange_np[:n], counts)
        np.add(
            r.input_batch.num_computed_tokens_cpu[rows],
            r.query_pos.np[: n * 6],
            out=r._positions_np_buf[: n * 6],
        )
        r.query_lens = torch.from_numpy(counts)
        r.query_start_loc.np[0] = 0
        r.query_start_loc.np[1 : n + 1] = cu
        r.optimistic_seq_lens_cpu[:n].copy_(torch.from_numpy(upper))
        r.optimistic_seq_lens_cpu[n:].zero_()
        r.num_discarded_requests = 0
        r.discard_request_mask.np[:n] = False
        r.num_accepted_tokens.np[:n] = r.input_batch.num_accepted_tokens_cpu[:n]
        r.num_accepted_tokens.np[n:] = 1
        r.req_indices.np[: n * 6] = rows
        r.num_scheduled_tokens.np[:n] = counts
        r.num_decode_draft_tokens.np[:n] = 5
        r.num_decode_draft_tokens.np[n:] = -1
        slot = self.slots[key]
        slot.project()
        self.active = slot
        result = slot.replay()
        self.sequence += 1
        # Later builder needs a conservative CPU tiling bound, not a D2H read.
        r._cross_step_bounds.skipped = True
        r._cross_step_bounds.bypassed += 1
        return result
