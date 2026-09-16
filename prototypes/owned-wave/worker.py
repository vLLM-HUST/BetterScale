"""Reference first; then transfer model/State and revoke runner execution."""
import json
import os
from pathlib import Path

import torch
from vllm_ascend.worker.worker import NPUWorker
from storage import backing_views, bank, restore
from boundary import forbid_runner_execution


class OwnedWorker(NPUWorker):
    def collect_reference(self):
        from vllm.forward_context import get_forward_context
        r = self.model_runner
        self.reference = []
        self.pools = backing_views(r)
        original_forward, original_sample = r._model_forward, r._sample
        self.original = original_forward, original_sample
        self.pending_reference = False
        def forward(*args, **kwargs):
            if r.input_batch.num_reqs == 1 and len(self.reference) < 7:
                if not self.reference:
                    torch.npu.synchronize()
                    self.before = [x.clone() for x in self.pools]
                    metadata = get_forward_context().attn_metadata
                    self.layer_names = tuple(metadata)
                    self.table = next(iter(metadata.values())).block_tables.clone()
                    bt = r.input_batch.block_table.block_tables[0]
                    self.grant_end = int(bt.num_blocks_per_row[0]) * bt.block_size
                self.pending_reference = True
            return original_forward(*args, **kwargs)
        def sample(*args, **kwargs):
            result = original_sample(*args, **kwargs)
            if self.pending_reference:
                torch.npu.synchronize()
                self.reference.append(dict(token=int(result.sampled_token_ids.item()),
                                           kv=[x.clone() for x in self.pools]))
                self.sampling = bank(r.input_batch.sampling_metadata)
                self.pending_reference = False
            return result
        r._model_forward, r._sample = forward, sample

    def run_owned(self, prompt):
        from vllm.distributed import get_ep_group, get_tp_group
        from livemodule import LiveRuntime, live_runtime
        from livemodule.arch.ascend.runtime.aclgraph import ACLGraphBackend
        from live_root import AdoptedStateBackend, ModelBundle, OwnedRoot
        from executor import WaveExecutor
        from schedule import NPlusTwo, Wave
        import torch.distributed as dist
        r = self.model_runner
        r._model_forward, r._sample = self.original
        assert len(self.reference) == 7
        rank = get_ep_group().rank_in_group
        path = Path(os.environ['OWNED_OUTPUT']) / f'rank{rank}.json'
        receipt = dict(status='STARTED', ep_rank=rank, ep_size=get_ep_group().world_size,
                       tp_rank=get_tp_group().rank_in_group, tp_size=get_tp_group().world_size,
                       dp_rank=r.parallel_config.data_parallel_rank, checks=[])
        path.write_text(json.dumps(receipt, indent=2))
        model = r.model
        while hasattr(model, 'unwrap'):
            model = model.unwrap()
        b = ModelBundle(model, r.sampler, self.sampling, r.vllm_config,
                        self.layer_names, self.table, self.pools, r.device,
                        r.input_batch.block_table.block_tables[0].block_size, r.attn_backend, self.grant_end)
        assert b.sampling.all_greedy and b.sampling.no_penalties
        assert len(b.layer_names) == b.config.model_config.hf_text_config.num_hidden_layers
        backend = AdoptedStateBackend(r.device)
        runtime = LiveRuntime(device=r.device, state_backend=backend,
                              graph_backend=ACLGraphBackend(device=r.device))
        root = None
        try:
            restore(self.pools, self.before)
            with forbid_runner_execution(r) as forbidden_calls:
                with live_runtime(runtime):
                    root = OwnedRoot(b, backend, prefill_width=len(prompt))
                root.activate()
                receipt['startup_graphs'] = 4
                for a, expected in zip(self.pools, self.before, strict=True):
                    assert torch.equal(a, expected), 'activation mutated adopted KV'
                receipt['activation_restores_adopted_state'] = True
                executor = WaveExecutor(root)
                # Every prefill/decode, including the FIRST invocation, is replayed.
                for seq in range(7):
                    wave = Wave(seq, 1, 'prefill' if seq == 0 else 'decode',
                                len(prompt)+seq, 7)
                    executor.submit(wave, prompt)
                    row = executor.receive_oldest()
                    assert row[4] == 1
                    self.exact(row[3], self.reference[seq], receipt, f'full-wave-{seq}')
                state = root.progress.clone()
                for seq in (7, 8):
                    executor.submit(Wave(seq, 1, 'decode', len(prompt)+6, 7), prompt)
                    row = executor.receive_oldest()
                    assert row[3:5] == [-1, 0] and row[5] == 1
                assert torch.equal(root.progress[:2], state[:2])
                self.exact(self.reference[-1]['token'], self.reference[-1], receipt, 'terminal-drains')
                receipt['terminal_drains'] = 2

                # Continuous episode: no KV reset between requests, no host anchor,
                # no all-device synchronization between N and submission of N+2.
                restore(self.pools, self.before)
                executor = WaveExecutor(root)
                scheduler = NPlusTwo(width=len(prompt), max_tokens=6,
                    grant_end=b.grant_end, requests=2, ranks=get_ep_group().world_size)
                tokens = {1: [], 2: []}
                while True:
                    while (wave := scheduler.next_wave()) is not None:
                        executor.submit(wave, prompt)
                    if not scheduler.pending:
                        break
                    row = executor.receive_oldest()
                    rows = [None] * get_ep_group().world_size
                    dist.all_gather_object(rows, row, group=get_ep_group().cpu_group)
                    scheduler.receive(rows)
                    if row[4]:
                        tokens[row[1]].append(row[3])
                expected = [x['token'] for x in self.reference[:6]]
                assert tokens == {1: expected, 2: expected}
                self.exact(tokens[2][-1], self.reference[5], receipt, 'nplus2-final')
                assert root.active_invocation_count == 0
                assert not forbidden_calls
                submissions = [x for x in scheduler.trace if x['action'] == 'submit']
                turnover = next(x for x in submissions if x['generation'] == 2)
                assert turnover['outstanding'] == [6, 7], turnover
                receipt.update(status='PASS', layers=len(b.layer_names),
                    runner_calls=list(forbidden_calls), tokens_by_generation=tokens,
                    nplus2_trace=scheduler.trace, nplus2_output_limit=6, quorum_size=get_ep_group().world_size,
                    turnover_behind_old_drain=True, max_outstanding=2,
                    full_prefill_width=len(prompt),
                    numerical_forward_calls=root.forward_calls, shadow_actions=root.shadow_actions,
                    model_type=type(model).__name__,
                    scope='FULL prefill+decode; actual LiveModule; three-stream N+2; all-rank quorum; PA/FIA host task updates')
                root.close()
                root = None
        except BaseException as exc:
            receipt.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            path.write_text(json.dumps(receipt, indent=2))
        return receipt

    def exact(self, token, reference, receipt, label):
        assert token == reference['token'], f'{label}: token mismatch {token} != {reference["token"]}'
        for i, (actual, expected) in enumerate(zip(self.pools, reference['kv'], strict=True)):
            assert torch.equal(actual, expected), f'{label}: KV backing {i} mismatch'
        receipt['checks'].append(dict(label=label, token=token, all_kv_bytes_exact=True))
