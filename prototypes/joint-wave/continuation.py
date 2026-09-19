"""One stable K5 seat: graph-owned numerical progress, native model bodies.

cursor is the next target input position (not LiveInference's emitted-token
cursor). The admitted cohort and KV grants do not change in this first gate.
"""
import copy
import torch

from betterscale.patches.async_decode._metadata import DeviceOnly
from storage import restore


class ContinuingWave:
    def __init__(self, oracle, packet):
        self.oracle = oracle
        self.r = r = oracle.r
        self.packet = packet
        self.cursor = packet['args'][2][:1].clone()
        self.anchor = packet['args'][1][:1].clone()
        self.draft = packet['spec'].draft_token_ids.view(1, 5).clone()
        self.axis = torch.arange(6, dtype=torch.int64, device=r.device)
        self.sequence = torch.zeros(1, dtype=torch.int64, device=r.device)
        self.done = torch.zeros(1, dtype=torch.bool, device=r.device)
        self.remaining = torch.full((1,), 1000, dtype=torch.int64, device=r.device)
        self.eos = torch.full((1,), -1, dtype=torch.int64, device=r.device)
        self.generation = torch.ones(1, dtype=torch.int64, device=r.device)
        self.poisoned = torch.zeros(1, dtype=torch.bool, device=r.device)
        self.refresh_grant()
        self.authorization = torch.tensor([[0, 1, self.granted_end],
                                           [1, 1, self.granted_end]],
                                          dtype=torch.int64, device=r.device)
        self.egress = torch.empty((2, 17), dtype=torch.int64, device=r.device)
        self.state = [self.cursor, self.anchor, self.draft, self.sequence,
                      self.done, self.remaining, self.eos, self.generation, self.poisoned]
        self.initial = [x.clone() for x in self.state]
        self.graph = None
        self.output = None
        self.replays = 0
        self.expected_receipts = []

    def refresh_grant(self):
        r = self.r
        # Compression groups count compressed slots, not original token positions.
        # Keep the per-group evidence; never infer a grant from buffer capacity.
        self.grant_layout = []
        for table, group in zip(r.input_batch.block_table.block_tables,
                                r.kv_cache_config.kv_cache_groups, strict=True):
            spec = group.kv_cache_spec
            specs = list(spec.kv_cache_specs.values()) if hasattr(spec, 'kv_cache_specs') else [spec]
            ratios = {getattr(s, 'compress_ratio', 1) for s in specs}
            assert len(ratios) == 1, 'heterogeneous compression grant unsupported'
            ratio = ratios.pop()
            assert ratio >= 1
            blocks = int(table.num_blocks_per_row[0])
            self.grant_layout.append(dict(kind=type(specs[0]).__name__, blocks=blocks,
                                          block_size=table.block_size, ratio=ratio,
                                          end=blocks * table.block_size * ratio))
        self.granted_end = min(x['end'] for x in self.grant_layout)

    def program(self):
        r = self.r
        packet = dict(self.packet)
        grant = self.authorization.index_select(0, self.sequence.remainder(2))[0]
        authorized = ((grant[0] == self.sequence) & (grant[1] == self.generation)
                      & (self.done | (self.cursor + 11 <= grant[2])))
        self.poisoned.logical_or_(~authorized)
        active = ~self.done & ~self.poisoned
        packet['active'] = active
        ids = torch.cat((self.anchor.view(1, 1), self.draft), dim=1).flatten()
        r.input_ids.gpu[:6].copy_(torch.where(active, ids, 0))
        r.positions[:6].copy_(torch.where(active, self.cursor + self.axis, 0))
        r.num_computed_tokens[:1].copy_(torch.where(active, self.cursor, 0))
        r.seq_lens[:1].copy_(torch.where(active, self.cursor + 6, 0))
        r.query_start_loc.gpu[1:2].copy_(active.to(torch.int32) * 6)
        r.input_batch.block_table.compute_slot_mapping(
            1, r.query_start_loc.gpu[:2], r.positions[:6],
        )
        for table in r.input_batch.block_table.block_tables:
            table.slot_mapping.gpu[:6].masked_fill_(~active, -1)
        metadata, common = self.oracle.build_metadata(**self.packet['build_kwargs'])
        packet['context'] = copy.copy(self.packet['context'])
        packet['context'].attn_metadata = metadata
        packet['common'] = common
        packet['spec'] = copy.copy(self.packet['spec'])
        packet['spec'].draft_token_ids = self.draft.flatten()
        packet['args'] = (6, r.input_ids.gpu[:6], r.positions[:6], None, None)
        sampled, draft = self.oracle.numeric(packet)
        _, raw_count = self.oracle.next_tokens(sampled, None, r.input_batch, None, 0)
        count = torch.minimum(raw_count, self.remaining)
        eos_hit = (sampled == self.eos[:, None]) & (self.eos[:, None] >= 0)
        eos_hit = eos_hit & (self.axis[None] < count[:, None])
        eos_end = torch.where(eos_hit, self.axis[None] + 1, 7).amin(dim=1)
        count = torch.minimum(count, eos_end)
        count = torch.where(active, count, 0)
        sampled = sampled.masked_fill(self.axis[None] >= count[:, None], -1)
        anchor, _ = self.oracle.next_tokens(sampled, None, r.input_batch, None, 0)
        terminal = active & ((count >= self.remaining) | (eos_end <= count))
        self.cursor.add_(count)
        self.anchor.copy_(torch.where(count > 0, anchor, self.anchor))
        draft = draft.masked_fill(terminal[:, None], -1)
        self.draft.copy_(torch.where(active[:, None], draft, self.draft))
        self.done.logical_or_(terminal)
        self.remaining.sub_(count)
        receipt = torch.cat((self.sequence, self.cursor, count,
                             sampled.flatten().to(torch.int64),
                             self.draft.flatten().to(torch.int64),
                             self.done.to(torch.int64), self.generation,
                             (~self.poisoned).to(torch.int64)))
        self.egress.index_copy_(0, self.sequence.remainder(2), receipt[None])
        self.sequence.add_(1)
        return sampled, draft

    def run(self, packet, expected, golden, row):
        r = self.r
        # The native reference has just run; do not derive graph progress from
        # its CPU receipt. This assertion observes independently advanced State.
        assert torch.equal(self.cursor, packet['args'][2][:1])
        row['input_position'] = int(self.cursor.item())
        # The independent native scheduler may allocate another block between
        # reference steps. Read only actually assigned rows, not pool capacity.
        self.refresh_grant()
        row['grant_layout'] = self.grant_layout
        upper = r.optimistic_seq_lens_cpu.clone()
        old_async_spec = r.use_async_spec_decode
        try:
            r.optimistic_seq_lens_cpu[:1].fill_(r.max_model_len)
            r.use_async_spec_decode = True
            if self.graph is None:
                restore(packet['pools'], packet['before'])
                with DeviceOnly():
                    eager = self.program()
                torch.npu.synchronize()
                self.oracle.check(eager, expected, packet['pools'], golden)
                row['device_preparation_eager_exact'] = True
                restore(self.state, self.initial)
                restore(packet['pools'], packet['before'])
                torch.npu.synchronize()
                self.graph = torch.npu.NPUGraph()
                with torch.npu.graph(self.graph), DeviceOnly():
                    self.output = self.program()
                torch.npu.synchronize()
                restore(self.state, self.initial)
            restore(packet['pools'], packet['before'])
            self.write_authorization(self.replays)
            self.graph.replay()
            torch.npu.synchronize()
            self.oracle.check(self.output, expected, packet['pools'], golden)
            self.replays += 1
            count = ((expected[0] >= 0) & (expected[0] < r.input_batch.vocab_size)).sum().item()
            self.expected_receipts.append(
                [self.replays - 1, row['input_position'] + count, count]
                + expected[0].flatten().cpu().tolist()
                + expected[1].flatten().cpu().tolist()
                + [0, 1, 1]
            )
            row.update(status='PASS', graph_captures=1, replay=self.replays,
                       next_position=int(self.cursor.item()),
                       sampled=self.output[0].cpu().tolist(),
                       draft=self.output[1].cpu().tolist())
            if self.replays == self.oracle.limit:
                row['queued_episode'] = self.check_queued_episode(golden)
                row['terminal_drains'] = self.check_terminal_drains()
                row['rejected_authorizations'] = self.check_rejected_authorizations()
        finally:
            r.optimistic_seq_lens_cpu.copy_(upper)
            r.use_async_spec_decode = old_async_spec

    def check_queued_episode(self, golden):
        """Six graph replays with no native step/preparation or token readback.

        Delay egress submission until both banks have been produced. Subsequent
        bank reuse waits on D2H completion on device, not CPU synchronization.
        """
        from transport import TwoBankExecutor
        from window import WaveWindow
        restore(self.state, self.initial)
        restore(self.packet['pools'], self.packet['before'])
        torch.npu.synchronize()
        window = WaveWindow(position=int(self.initial[0].item()), granted_end=self.granted_end)
        executor = TwoBankExecutor(self.graph, self.egress, authorization=self.authorization)
        first = executor.submit(window.authorize())
        second = executor.submit(window.authorize())
        executor.copy_out(first)
        executor.copy_out(second)
        def retire(row):
            window.retire(sequence=row[0], generation=row[15], position=row[1],
                          count=row[2], done=bool(row[14]), ok=bool(row[16]))
        grant_backpressure = 0
        while len(executor.tickets) < len(self.expected_receipts):
            retire(executor.receive_oldest())
            # A two-wave window is a maximum, not an obligation to speculate
            # beyond assigned KV. Retire the second receipt if it tightens the
            # worst-case horizon enough. If empty and still short, renew outside
            # this stable-table episode instead of publishing an unsafe wave.
            while not window.can_authorize():
                if not window.pending:
                    raise RuntimeError('resident needs a KV grant renewal')
                grant_backpressure += 1
                retire(executor.receive_oldest())
            executor.copy_out(executor.submit(window.authorize()))
        while len(executor.retired) < len(executor.tickets):
            retire(executor.receive_oldest())
        actual = executor.finish()
        assert actual == self.expected_receipts, (actual, self.expected_receipts)
        for i, (lhs, rhs) in enumerate(zip(self.packet['pools'], golden, strict=True)):
            assert torch.equal(lhs, rhs), f'queued episode KV backing {i} differs'
        return dict(replays=len(actual), native_calls_between_replays=0,
                    receipts_exact=True, final_kv_exact=True,
                    max_outstanding=2, granted_end=self.granted_end,
                    grant_backpressure=grant_backpressure,
                    delayed_copy_until_two_banks_produced=True)

    def check_terminal_drains(self):
        receipts = []
        for policy in ('length', 'eos'):
            restore(self.state, self.initial)
            restore(self.packet['pools'], self.packet['before'])
            if policy == 'length':
                self.remaining.fill_(1)
            else:
                self.eos.fill_(self.expected_receipts[0][3])
            self.write_authorization(0)
            self.graph.replay()
            torch.npu.synchronize()
            assert self.done.item() and self.egress[0, 2].item() == 1
            final_kv = [x.clone() for x in self.packet['pools']]
            final_progress = [x.clone() for x in (self.cursor, self.anchor, self.draft)]
            for drain in range(3):
                self.write_authorization(drain + 1)
                self.graph.replay()
                torch.npu.synchronize()
                bank = (drain + 1) % 2
                assert self.egress[bank, 2].item() == 0
                assert torch.all(self.egress[bank, 3:9] == -1).item()
                for lhs, rhs in zip((self.cursor, self.anchor, self.draft), final_progress):
                    assert torch.equal(lhs, rhs), 'terminal drain changed continuation'
                for i, (lhs, rhs) in enumerate(zip(self.packet['pools'], final_kv)):
                    assert torch.equal(lhs, rhs), f'terminal drain changed KV backing {i}'
            receipts.append(dict(policy=policy, emitted=1, drained_waves=3,
                                 continuation_unchanged=True, kv_unchanged=True))
        return receipts

    def write_authorization(self, sequence, *, generation=1, end=None):
        self.authorization[sequence % 2].copy_(torch.tensor(
            [sequence, generation, self.granted_end if end is None else end],
            dtype=torch.int64, device='cpu'))

    def check_rejected_authorizations(self):
        receipts = []
        for reason in ('generation', 'grant'):
            restore(self.state, self.initial)
            restore(self.packet['pools'], self.packet['before'])
            self.write_authorization(0, generation=2 if reason == 'generation' else 1,
                                     end=int(self.initial[0].item()) + 10
                                     if reason == 'grant' else self.granted_end)
            self.graph.replay()
            torch.npu.synchronize()
            assert self.poisoned.item() and not self.egress[0, 16].item()
            assert self.egress[0, 2].item() == 0
            for lhs, rhs in zip(self.packet['pools'], self.packet['before']):
                assert torch.equal(lhs, rhs), 'rejected grant mutated KV'
            assert torch.equal(self.cursor, self.initial[0])
            self.write_authorization(1)
            self.graph.replay()
            torch.npu.synchronize()
            assert self.poisoned.item() and not self.egress[1, 16].item()
            for lhs, rhs in zip(self.packet['pools'], self.packet['before']):
                assert torch.equal(lhs, rhs), 'poisoned generation resumed KV writes'
            receipts.append(dict(reason=reason, fail_closed=True, kv_unchanged=True,
                                 rejected_retry=True))
        return receipts
