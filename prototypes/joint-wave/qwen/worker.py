"""Qwen snapshot compatibility and the retained negative device-length probe."""
import copy
import json
import os
from pathlib import Path

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, override_forward_context
from vllm_ascend.worker.worker import NPUWorker
from betterscale.patches.async_decode._metadata import DeviceOnly
from storage import bank, backing_views, restore
from transport import TwoBankExecutor


class JointWorker(NPUWorker):
    def enable_joint_oracle(self):
        self.oracle = Oracle(self.model_runner)

    def joint_receipt(self):
        assert len(self.oracle.rows) == 6
        assert all(x['status'] == 'PASS' for x in self.oracle.rows)
        return dict(manifest=self.oracle.manifest, rows=self.oracle.rows)


class Oracle:
    def __init__(self, runner):
        from vllm.distributed import get_ep_group, get_tp_group
        from vllm_ascend.attention.utils import using_paged_attention
        self.r = r = runner
        assert using_paged_attention(1, r.vllm_config, 128), 'native PA envelope required'
        self.manifest = dict(tp_size=get_tp_group().world_size,
                             tp_rank=get_tp_group().rank_in_group,
                             dp_rank=r.parallel_config.data_parallel_rank,
                             ep_size=get_ep_group().world_size,
                             ep_rank=get_ep_group().rank_in_group,
                             device=str(r.device), attention='native paged attention')
        self.path = Path(os.environ['QWEN_JOINT_OUTPUT']) / f"rank{get_ep_group().rank_in_group}.json"
        self.forward = r._model_forward
        self.sample = r._sample
        self.retire = r.sample_tokens
        self.pending = None
        self.rows = []
        self.mode = os.environ.get('QWEN_JOINT_MODE', 'snapshot')
        assert self.mode in ('snapshot', 'continuation')
        self.manifest['oracle_mode'] = self.mode
        self.graph = None
        self.expected = []
        r._model_forward = self.observe_forward
        r._sample = self.observe_sample
        r.sample_tokens = self.observe_retirement
        self.write()

    def write(self):
        self.path.write_text(json.dumps(dict(manifest=self.manifest, rows=self.rows), indent=2))

    def observe_forward(self, *args, **kwargs):
        r = self.r
        if len(self.rows) < 6 and r.input_batch.num_reqs == 1 and args[0] == 1:
            torch.npu.synchronize()
            ctx = copy.copy(get_forward_context())
            ctx.attn_metadata = bank(ctx.attn_metadata)
            pools = backing_views(r)
            self.pending = dict(args=bank(args), kwargs=bank(kwargs), context=ctx,
                                pools=pools, before=[x.clone() for x in pools])
        return self.forward(*args, **kwargs)

    def observe_sample(self, *args, **kwargs):
        result = self.sample(*args, **kwargs)
        if self.pending is not None:
            self.sampled = result.sampled_token_ids.clone()
        return result

    def observe_retirement(self, grammar_output):
        if self.pending is None:
            return self.retire(grammar_output)
        assert grammar_output is None
        packet = self.pending
        packet['indices'] = self.r.logits_indices.clone()
        output = self.retire(grammar_output)
        torch.npu.synchronize()
        golden = [x.clone() for x in packet['pools']]
        expected = self.sampled.clone()
        row = dict(step=len(self.rows), status='STARTED')
        self.rows.append(row)
        self.write()
        try:
            if self.mode == 'snapshot':
                self.check_snapshot(packet, expected, golden, row)
            else:
                self.check_step(packet, expected, golden, row)
        except BaseException as exc:
            row.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            torch.npu.synchronize()
            restore(packet['pools'], golden)
            self.pending = None
            self.write()
        return output

    def check_snapshot(self, packet, expected, golden, row):
        self.manifest.update(native_mode=str(packet['context'].cudagraph_runtime_mode),
                             moe_comm=str(packet['context'].moe_comm_type),
                             layers=self.r.model_config.hf_text_config.num_hidden_layers)
        sm = self.r.input_batch.sampling_metadata
        assert sm.all_greedy and sm.no_penalties and sm.max_num_logprobs is None
        def numerical():
            ctx = copy.copy(packet['context'])
            ctx.cudagraph_runtime_mode = CUDAGraphMode.NONE
            ctx.moe_layer_index = 0
            with override_forward_context(ctx):
                hidden = self.forward(*packet['args'], **packet['kwargs'])
            logits = self.r.model.compute_logits(hidden[packet['indices']])
            return self.r.sampler(logits=logits, sampling_metadata=sm).sampled_token_ids
        restore(packet['pools'], packet['before'])
        with DeviceOnly():
            eager = numerical()
        torch.npu.synchronize()
        self.exact(eager, expected, packet['pools'], golden)
        row['eager_exact'] = True
        restore(packet['pools'], packet['before'])
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph), DeviceOnly():
            actual = numerical()
        torch.npu.synchronize()
        for _ in range(2):
            restore(packet['pools'], packet['before'])
            graph.replay()
            torch.npu.synchronize()
            self.exact(actual, expected, packet['pools'], golden)
        row.update(status='PASS', capture_scope='fixed native snapshot', captures=1,
                   replays=2, input_position=int(packet['args'][2][0].item()),
                   sampled=int(expected.item()), all_kv_bytes_exact=True)

    def initialize(self, packet):
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        self.packet = packet
        self.manifest.update(native_mode=str(packet['context'].cudagraph_runtime_mode),
                             moe_comm=str(packet['context'].moe_comm_type),
                             layers=self.r.model_config.hf_text_config.num_hidden_layers)
        self.cursor = packet['args'][2][:1].clone()
        self.anchor = packet['args'][1][:1].clone()
        self.sequence = torch.zeros(1, dtype=torch.int64, device=self.r.device)
        self.egress = torch.empty((2, 3), dtype=torch.int64, device=self.r.device)
        self.state = [self.cursor, self.anchor, self.sequence]
        self.initial = [x.clone() for x in self.state]
        self.metadata = list({id(m): m for m in packet['context'].attn_metadata.values()}.values())
        assert all(m.attn_state == AscendAttentionState.DecodeOnly for m in self.metadata)
        # Deliberate negative probe: the Tensor-typed native PA carrier is CPU.
        # Direct device substitution failed in smoke3; this is NOT a working
        # continuation adapter. Snapshot mode leaves the native carrier intact.
        for metadata in self.metadata:
            metadata.seq_lens = metadata.seq_lens.to(self.r.device).clone()
        sm = self.r.input_batch.sampling_metadata
        assert sm.all_greedy and sm.no_penalties and sm.max_num_logprobs is None
        assert not sm.bad_words_token_ids and sm.allowed_token_ids_mask is None
        self.sampling_metadata = sm
        self.block_size = self.r.input_batch.block_table.block_tables[0].block_size
        self.grant_end = min(int(t.num_blocks_per_row[0]) * t.block_size
                             for t in self.r.input_batch.block_table.block_tables)
        assert int(self.cursor.item()) + 6 <= self.grant_end

    def program(self):
        packet = self.packet
        packet['args'][1].copy_(self.anchor)
        packet['args'][2].copy_(self.cursor)
        for metadata in self.metadata:
            metadata.seq_lens.copy_(self.cursor + 1)
            block = metadata.block_tables[0].gather(0, self.cursor // self.block_size)
            metadata.slot_mapping.copy_((block * self.block_size + self.cursor % self.block_size)
                                        .to(metadata.slot_mapping.dtype))
        ctx = copy.copy(packet['context'])
        ctx.cudagraph_runtime_mode = CUDAGraphMode.NONE  # no nested target-only replay
        ctx.moe_layer_index = 0
        with override_forward_context(ctx):
            hidden = self.forward(*packet['args'], **packet['kwargs'])
        logits = self.r.model.compute_logits(hidden[packet['indices']])
        sampled = self.r.sampler(logits=logits, sampling_metadata=self.sampling_metadata).sampled_token_ids
        self.anchor.copy_(sampled.flatten())
        self.cursor.add_(1)
        receipt = torch.cat((self.sequence, self.cursor, self.anchor.to(torch.int64)))
        self.egress.index_copy_(0, self.sequence.remainder(2), receipt[None])
        self.sequence.add_(1)
        return sampled

    @staticmethod
    def exact(actual, expected, pools, golden):
        assert torch.equal(actual, expected), 'sampled IDs differ from original native PA'
        for i, (a, b) in enumerate(zip(pools, golden, strict=True)):
            assert torch.equal(a, b), f'KV backing {i} differs from original native PA'

    def check_step(self, packet, expected, golden, row):
        if self.graph is None:
            self.initialize(packet)
            restore(packet['pools'], packet['before'])
            with DeviceOnly():
                eager = self.program()
            torch.npu.synchronize()
            self.exact(eager, expected, packet['pools'], golden)
            row['eager_exact'] = True
            restore(self.state, self.initial)
            restore(packet['pools'], packet['before'])
            torch.npu.synchronize()
            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(self.graph), DeviceOnly():
                self.output = self.program()
            torch.npu.synchronize()
            restore(self.state, self.initial)
        assert torch.equal(self.cursor, packet['args'][2][:1])
        restore(packet['pools'], packet['before'])
        self.graph.replay()
        torch.npu.synchronize()
        self.exact(self.output, expected, packet['pools'], golden)
        self.expected.append([len(self.expected), int(self.cursor.item()), int(expected.item())])
        row.update(status='PASS', captures=1, next_position=int(self.cursor.item()),
                   sampled=int(expected.item()), all_kv_bytes_exact=True)
        if len(self.rows) == 6:
            restore(self.state, self.initial)
            restore(self.packet['pools'], self.packet['before'])
            torch.npu.synchronize()
            executor = TwoBankExecutor(self.graph, self.egress)
            a, b = executor.submit(), executor.submit()
            executor.copy_out(a)
            executor.copy_out(b)
            for _ in range(4):
                executor.receive_oldest()
                executor.copy_out(executor.submit())
            receipts = executor.finish()
            assert receipts == self.expected, (receipts, self.expected)
            for a, b in zip(packet['pools'], golden, strict=True):
                assert torch.equal(a, b), 'autonomous final KV differs'
            row['autonomous'] = dict(waves=6, max_outstanding=2, receipts_exact=True,
                                     final_kv_exact=True, native_steps_between=0,
                                     assigned_grant_end=self.grant_end)
