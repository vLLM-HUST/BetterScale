"""Actual LiveInference lifecycle around donated native Qwen model bodies.

No runner reference exists in this module. PA's host tiling carrier is explicit;
positions, anchor, committed progress and receipt publication remain on device.
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from livemodule import LiveModule, MetaTensor, StateTensor, StateDomain, ExactStateCapacity
from livemodule.runtime.ingress import GraphCallSchema
from livemodule.runtime.meta_tensor import construct_meta_tensors
from livemodule.runtime.phase import LivePhase, current_live_phase
from livemodule.runtime.state_backend import TorchStateBackend
from vllm.config import CUDAGraphMode
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention import attention_v1
from vllm_ascend.attention.attention_v1 import AscendMetadata, AscendAttentionState
from vllm_ascend.compilation import acl_graph


class AdoptedStateBackend(TorchStateBackend):
    """Transfer existing fixed buffers into State ownership without reallocating.

    Donor model views retain these allocations. No native scheduler may use them
    during the adopted generation. Release drops this provider's lease only.
    """
    def __init__(self, device, *, memory_budget_bytes=4 * 1024**3):
        super().__init__(device, memory_budget_bytes=memory_budget_bytes)
        self.buffers = {}

    def _allocate_state_domain(self, plan):
        assert plan.num_blocks == 1
        return {lane.state: self.buffers[lane.state].view(1, -1) for lane in plan.schema.lanes}


@dataclass(frozen=True)
class WaveSchema(GraphCallSchema):
    capture_state_blocks: dict


class PALength(MetaTensor):
    def __init__(self, value):
        self.value = value

    def construct(self):
        return torch.tensor([self.value], dtype=torch.int32, device='cpu')


@dataclass
class ModelBundle:
    model: object
    sampler: object
    sampling: object
    config: object
    layer_names: tuple
    table: torch.Tensor
    pools: list
    device: object
    block_size: int
    attention_backend: object
    grant_end: int


class OwnedRoot(LiveModule):
    def __init__(self, bundle, backend, *, prefill_width=32):
        super().__init__()
        self.bundle = bundle
        self.prefill_width = prefill_width
        self.prompt_ids = torch.zeros((2, prefill_width), dtype=torch.int64, device=bundle.device)
        self.authorization = torch.tensor([[0, 1, 7], [1, 1, 7]], dtype=torch.int64, device=bundle.device)
        self.domain = StateDomain(ExactStateCapacity(1))
        self.progress = torch.zeros(6, dtype=torch.int64, device=bundle.device)
        # cursor, anchor, sequence, remaining, done, generation
        self.progress[3] = 16
        self.progress[5] = 1
        self.egress = torch.zeros((2, 7), dtype=torch.int64, device=bundle.device)
        for i, buffer in enumerate([*bundle.pools, self.progress, self.egress]):
            byte_view = buffer.view(torch.uint8).flatten()
            state = StateTensor(role=f'owned-backing-{i}', requirement='fixed adopted generation',
                                block_shape=(byte_view.numel(),), storage_dtype=torch.uint8,
                                domain=self.domain)
            self.register_state(f'backing_{i}', state)
            backend.buffers[state] = byte_view
        self.host_lengths = [1, 1]
        self.metadata = [self.make_metadata(1, 1, decode=True) for _ in range(2)]
        self.prefill_metadata = [self.make_metadata(prefill_width, prefill_width, decode=False) for _ in range(2)]
        self.params = [acl_graph.GraphParams(
            {1: [], prefill_width: []}, {1: None, prefill_width: None},
            {1: [], prefill_width: []}, {1: [], prefill_width: []}) for _ in range(2)]
        self.update_stream = torch.npu.Stream()
        self.forward_calls = 0
        self.shadow_actions = 0
        schema = WaveSchema((), {}, {self.domain: (0,)})
        self.register_graph('decode0', entry=self.decode0, schema=schema)
        self.register_graph('decode1', entry=self.decode1, schema=schema)
        self.register_graph('prefill0', entry=self.prefill0, schema=schema)
        self.register_graph('prefill1', entry=self.prefill1, schema=schema)

    def make_metadata(self, tokens, length, *, decode):
        b = self.bundle
        # Only host-admitted geometry is built here, never sampled-token feedback.
        positions = torch.arange(length-tokens, length, device=b.device, dtype=torch.int64)
        blocks = b.table[0].gather(0, positions // b.block_size)
        slots = (blocks * b.block_size + positions % b.block_size).to(torch.int32)
        cpu_lengths = torch.tensor([length], dtype=torch.int32)
        m = AscendMetadata(attn_state=AscendAttentionState.DecodeOnly if decode else AscendAttentionState.ChunkedPrefill,
            num_actual_tokens=tokens, num_decode_tokens=tokens if decode else 0,
            num_prefills=0 if decode else 1, num_decodes=1 if decode else 0,
            seq_lens=cpu_lengths, seq_lens_cpu=cpu_lengths, seq_lens_list=[length],
            actual_seq_lengths_q=[tokens], query_start_loc=torch.tensor([0,tokens],device=b.device,dtype=torch.int32),
            max_query_len=tokens, block_tables=b.table, slot_mapping=slots, causal=True,
            model_runner_type=b.config.model_config.runner_type)
        if not decode:
            from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
            m.attn_mask = AttentionMaskBuilder(b.device).get_attention_mask(True, b.config.model_config)
        return m

    @contextmanager
    def registry(self, bank):
        old_params, old_keys = acl_graph._graph_params, attention_v1._ATTN_KEYS_BUFFER
        acl_graph._graph_params = self.params[bank]
        attention_v1._ATTN_KEYS_BUFFER = list(self.bundle.layer_names)
        try:
            yield
        finally:
            acl_graph._graph_params, attention_v1._ATTN_KEYS_BUFFER = old_params, old_keys

    @contextmanager
    def context(self, metadata, tokens, *, capturing=False):
        from vllm.forward_context import get_forward_context
        b = self.bundle
        across = torch.full((b.config.parallel_config.data_parallel_size,), tokens, dtype=torch.int32)
        with set_ascend_forward_context(
            {key: metadata for key in b.layer_names}, b.config, num_tokens=tokens,
            num_tokens_across_dp=across, num_actual_tokens=tokens,
            aclgraph_runtime_mode=CUDAGraphMode.NONE, model_instance=b.model):
            ctx = get_forward_context()
            ctx.capturing = capturing
            yield ctx

    def construct0(self, context):
        return self.construct_metadata(0)

    def construct1(self, context):
        return self.construct_metadata(1)

    def construct_metadata(self, bank):
        self.shadow_actions += 1
        m = self.metadata[bank]
        m.seq_lens = PALength(self.host_lengths[bank]).tensor
        m.seq_lens_cpu = m.seq_lens
        m.seq_lens_list = [self.host_lengths[bank]]
        return m

    def numerical(self, ids, positions, metadata, *, bank=None, capturing=False):
        b = self.bundle
        with self.context(metadata, ids.shape[0], capturing=capturing):
            hidden = b.model(input_ids=ids, positions=positions, intermediate_tensors=None, inputs_embeds=None)
        logits = b.model.compute_logits(hidden[-1:])
        return b.sampler(logits=logits, sampling_metadata=b.sampling).sampled_token_ids.flatten()

    def construct_prefill0(self, context):
        return self.construct_prefill_metadata(0)

    def construct_prefill1(self, context):
        return self.construct_prefill_metadata(1)

    def construct_prefill_metadata(self, bank):
        self.shadow_actions += 1
        m = self.prefill_metadata[bank]
        m.seq_lens = PALength(self.prefill_width).tensor
        m.seq_lens_cpu = m.seq_lens
        m.seq_lens_list = [self.prefill_width]
        return m

    def prefill0(self):
        return self.prefill(0)

    def prefill1(self):
        return self.prefill(1)

    def prefill(self, bank):
        self.forward_calls += 1
        m = construct_meta_tensors(
            self.construct_prefill0 if bank == 0 else self.construct_prefill1, context=None)
        n = self.prefill_width
        positions = torch.arange(n, device=self.bundle.device, dtype=torch.int64)
        with self.registry(bank):
            sampled = self.numerical(self.prompt_ids[bank], positions, m,
                                    capturing=current_live_phase() == LivePhase.CAPTURE)
        sequence, generation, max_tokens = self.authorization[bank].unbind()
        self.progress[0] = n
        self.progress[1:2].copy_(sampled)
        self.progress[2].copy_(sequence + 1)
        self.progress[3].copy_(max_tokens - 1)
        self.progress[4].copy_((max_tokens == 1).to(torch.int64))
        self.progress[5].copy_(generation)
        self.egress[bank].copy_(torch.stack((
            sequence, generation, self.progress[0], sampled[0].to(torch.int64),
            torch.ones_like(sequence), self.progress[4], torch.zeros_like(sequence))))
        return self.egress[bank]

    def decode0(self):
        return self.decode(0)

    def decode1(self):
        return self.decode(1)

    def decode(self, bank):
        self.forward_calls += 1
        m = construct_meta_tensors(self.construct0 if bank == 0 else self.construct1, context=None)
        cursor, anchor, sequence, remaining, done, generation = self.progress.unbind()
        # Clone aliased State inputs before in-place commit.
        positions = cursor.view(1).clone()
        ids = anchor.view(1).clone()
        command = self.authorization[bank]
        qualified = (command[0] == sequence) & (command[1] == generation)
        active = qualified & (remaining > 0) & (done == 0) & (cursor < self.bundle.grant_end)
        block = self.bundle.table[0].gather(0, positions.clamp(max=self.bundle.grant_end-1) // self.bundle.block_size)
        slots = block * self.bundle.block_size + positions % self.bundle.block_size
        m.slot_mapping.copy_(torch.where(active, slots, -1).to(torch.int32))
        with self.registry(bank):
            sampled = self.numerical(ids, positions, m, bank=bank,
                                    capturing=current_live_phase() == LivePhase.CAPTURE)
        count = active.to(torch.int64)
        self.progress[0].add_(count)
        self.progress[1:2].copy_(torch.where(active, sampled, self.progress[1:2]))
        self.progress[3].sub_(count)
        self.progress[4].copy_((self.progress[3] == 0).to(torch.int64))
        receipt = torch.stack((sequence.clone(), generation, self.progress[0],
                               torch.where(active, sampled[0].to(torch.int64), -1), count,
                               self.progress[4], (~qualified).to(torch.int64)))
        self.egress[bank].copy_(receipt)
        self.progress[2].add_(qualified.to(torch.int64))
        return self.egress[bank]

    def publish_attention(self, bank, compute_stream, *, kind):
        # Native PA's host tiling ABI remains explicit, outside the numerical graph.
        ready = torch.npu.Event()
        ready.record(compute_stream)
        self.update_stream.wait_event(ready)
        metadata = self.metadata[bank] if kind == "decode" else self.prefill_metadata[bank]
        tokens = 1 if kind == "decode" else self.prefill_width
        with self.registry(bank), self.context(metadata, tokens) as ctx:
            acl_graph.update_full_graph_params(self.bundle.attention_backend, self.update_stream,
                                               ctx, tokens, self.bundle.config, None)
