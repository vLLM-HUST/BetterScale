"""Opt-in same-state oracle for one native DSV4 target/sample/DSpark graph.

No product Worker hooks. Native serving remains the independent reference.
The first envelope is TP1, one occupied K5 seat, fixed greedy, no discards.
"""

import copy
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch
from vllm.forward_context import get_forward_context, override_forward_context

# Existing dummy-only loader repair, not a numerical/graph replacement.
import donor_dp_worker  # noqa: F401
from storage import bank, backing_views, restore


@contextmanager
def replace_attributes(obj, **values):
    old = {key: getattr(obj, key) for key in values}
    try:
        for key, value in values.items():
            setattr(obj, key, value)
        yield
    finally:
        for key, value in old.items():
            setattr(obj, key, value)


class JointWaveOracle:
    def __init__(self, runner):
        self.r = runner
        self.forward = runner._model_forward
        self.sample = runner.sample_tokens
        self.native_sample = runner._sample
        self.build_metadata = runner._build_attention_metadata
        self.build_kwargs = None
        self.mode = os.environ.get('JOINT_WAVE_MODE', 'snapshot')
        self.continuing = None
        self.pending = None
        self.rows = []
        self.sampled = None
        self.limit = 6 if self.mode == 'continuation' else 3
        self.path = Path(os.environ['JOINT_WAVE_OUTPUT']) / 'oracle.json'
        runner._model_forward = self.observe_forward
        runner._sample = self.observe_sample
        runner.sample_tokens = self.observe_retirement
        runner._build_attention_metadata = self.observe_metadata

    def observe_metadata(self, *args, **kwargs):
        if not args:
            self.build_kwargs = dict(kwargs)
        return self.build_metadata(*args, **kwargs)

    def observe_sample(self, *args, **kwargs):
        output = self.native_sample(*args, **kwargs)
        if self.pending is not None:
            self.sampled = output.sampled_token_ids.clone()
        return output

    def observe_forward(self, *args, **kwargs):
        r = self.r
        if len(self.rows) < self.limit and r.input_batch.num_reqs == 1 and args[0] == 6:
            metadata = r.input_batch.sampling_metadata
            assert metadata.all_greedy and metadata.no_penalties
            assert metadata.max_num_logprobs is None
            assert not metadata.bad_words_token_ids
            assert metadata.allowed_token_ids_mask is None
            assert r.num_discarded_requests == 0
            torch.npu.synchronize()
            context = copy.copy(get_forward_context())
            context.attn_metadata = bank(context.attn_metadata)
            pools = backing_views(r)
            self.pending = dict(
                args=bank(args), kwargs=bank(kwargs), context=context,
                pools=pools, before=[x.clone() for x in pools],
                build_kwargs=self.build_kwargs,
            )
        return self.forward(*args, **kwargs)

    def observe_retirement(self, grammar_output):
        assert grammar_output is None
        if self.pending is None:
            return self.sample(grammar_output)
        packet = self.pending
        state = self.r.execute_model_state
        packet['state'] = state
        packet['common'] = bank(state.spec_decode_common_attn_metadata)
        packet['spec'] = bank(state.spec_decode_metadata)
        packet['logits_indices'] = self.r.logits_indices.clone()
        output = self.sample(grammar_output)
        torch.npu.synchronize()
        golden = [x.clone() for x in packet['pools']]
        expected = (self.sampled.clone(), self.r._draft_token_ids.clone())
        row = dict(step=len(self.rows), status='STARTED', phase='eager-composition')
        self.rows.append(row)
        self.write()
        try:
            with replace_attributes(self.r, _copy_valid_sampled_token_count=lambda *a: None), \
                 replace_attributes(self.r.drafter, prepare_next_token_ids_padded=self.next_tokens):
                if self.mode == 'continuation':
                    from continuation import ContinuingWave
                    if self.continuing is None:
                        self.continuing = ContinuingWave(self, packet)
                    row['phase'] = 'continuous-joint-graph'
                    self.continuing.run(packet, expected, golden, row)
                    return output
                restore(packet['pools'], packet['before'])
                eager = self.numeric(packet)
                torch.npu.synchronize()
                self.check(eager, expected, packet['pools'], golden)
                row['eager_composition_exact'] = True
                row['phase'] = 'joint-capture'
                self.write()
                restore(packet['pools'], packet['before'])
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    captured = self.numeric(packet)
                torch.npu.synchronize()
                # Capture is initialization, never count it as a serving invocation.
                for repeat in range(2):
                    restore(packet['pools'], packet['before'])
                    graph.replay()
                    torch.npu.synchronize()
                    self.check(captured, expected, packet['pools'], golden)
                row.update(status='PASS', replays=2,
                           sampled=expected[0].cpu().tolist(),
                           draft=expected[1].cpu().tolist())
        except BaseException as exc:
            row.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            restore(packet['pools'], golden)
            torch.npu.synchronize()
            self.pending = None
            self.write()
        return output

    @staticmethod
    def next_tokens(sampled, requests, batch, discard_indices, num_discarded):
        # Native padded selection minus unreachable backup-token H2D for this
        # explicitly admitted no-discard, nonempty verification envelope.
        assert num_discarded == 0
        valid = (sampled != -1) & (sampled < batch.vocab_size)
        count = valid.sum(dim=1)
        last = (count - 1).clamp(min=0)
        selected = sampled.gather(1, last[:, None]).squeeze(1)
        return torch.where(count > 0, selected, 0), count

    def numeric(self, packet):
        r = self.r
        state = packet['state']
        ctx = copy.copy(packet['context'])
        ctx.moe_layer_index = 0
        with override_forward_context(ctx):
            hidden = self.forward(*packet['args'], **packet['kwargs'])
        auxiliary = None
        if r.use_aux_hidden_state_outputs:
            hidden, auxiliary = hidden
        selected = hidden[packet['logits_indices']]
        logits = r.model.compute_logits(selected)
        sampled = r.rejection_sampler(
            packet['spec'], None, logits, r.input_batch.sampling_metadata,
        ).sampled_token_ids
        active = packet.get('active')
        propose_inputs = r.drafter.set_inputs_first_pass
        def masked_inputs(*args, **kwargs):
            result = propose_inputs(*args, **kwargs)
            if active is not None:
                common = result[2]
                common.seq_lens.masked_fill_(~active, 0)
                common.query_start_loc.mul_(active.to(common.query_start_loc.dtype))
                for mapping in (*r.drafter._per_group_context_slot_mapping_buffers.values(),
                                *r.drafter._per_group_query_slot_mapping_buffers.values()):
                    mapping.masked_fill_(~active, -1)
                r.drafter.positions.masked_fill_(~active, 0)
                r.drafter.input_ids.masked_fill_(~active, 0)
            return result
        draft_sampled = sampled if active is None else sampled.masked_fill(~active, -1)
        with replace_attributes(r.drafter, set_inputs_first_pass=masked_inputs):
            draft = r.propose_draft_token_ids(
                draft_sampled, r.input_batch.sampling_metadata, state.scheduler_output,
                packet['spec'], copy.copy(packet['common']), packet['args'][2],
                6, hidden, auxiliary, selected, state.batch_desc,
            )
        return sampled, draft

    @staticmethod
    def check(actual, expected, pools, golden):
        for name, lhs, rhs in zip(('sampled', 'draft'), actual, expected, strict=True):
            assert torch.equal(lhs, rhs), f'{name} differs from native serving'
        for index, (lhs, rhs) in enumerate(zip(pools, golden, strict=True)):
            assert torch.equal(lhs, rhs), f'KV backing {index} differs from native serving'

    def write(self):
        self.path.write_text(json.dumps(self.rows, indent=2))


class JointWaveProbeWorker:
    def enable_joint_oracle(self):
        r = self.model_runner
        assert r.parallel_config.tensor_parallel_size == 1
        assert r.parallel_config.data_parallel_size == 1
        assert r.num_spec_tokens == 5
        self.joint_oracle = JointWaveOracle(r)

    def joint_receipt(self):
        rows = self.joint_oracle.rows
        assert len(rows) == self.joint_oracle.limit and all(row['status'] == 'PASS' for row in rows), rows
        return rows
