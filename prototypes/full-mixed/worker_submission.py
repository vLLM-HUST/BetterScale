"""Stable-K5 worker submission/retirement cut, retaining native RPC ordering.

Ingress/target metadata use the existing shadow producer's banks and events.
Do not retire a host receipt between target and sampler/draft submission.
Keep count receipts generation-local when the following wave publishes its
own counts. Numerical progress and KV remain native single-copy device State.
"""
from dataclasses import dataclass
import time
import torch
from torch.profiler import record_function


@dataclass
class CountReceipt:
    cpu: object
    ready: object
    rows: int

    def read(self):
        self.ready.synchronize()
        return self.cpu[:self.rows].tolist()


class WorkerSubmission:
    def __init__(self, worker):
        r = self.r = worker.model_runner
        assert r.use_async_scheduling and r.use_async_spec_decode
        assert not r.need_accepted_tokens and not r.model_config.is_hybrid
        assert r.vllm_config.parallel_config.pipeline_parallel_size == 1
        assert r.vllm_config.parallel_config.tensor_parallel_size > 1
        assert hasattr(r, '_decode_shadow') and hasattr(worker, '_decode_metadata')
        assert hasattr(worker, '_exact_draft_graph')
        self.execute = r.execute_model
        self.sample = r.sample_tokens
        self.copy_counts = r._copy_valid_sampled_token_count
        self.buffers = [torch.empty_like(r.valid_sampled_token_count_cpu,
                        device='cpu', pin_memory=True) for _ in range(2)]
        self.copy_sequence = self.sequence = self.retired = 0
        self.pending = None
        self.frame = None
        self.failed = False
        self.enabled = True
        self.rows = []
        r.execute_model = self.execute_model
        r.sample_tokens = self.sample_tokens
        r._copy_valid_sampled_token_count = self.publish_counts
        r._worker_submission = self

    def execute_model(self, *args, **kwargs):
        assert not self.failed and self.pending is None and self.frame is None
        self.frame = dict(sequence=self.sequence, host_begin_ns=time.monotonic_ns())
        try:
            result = self.execute(*args, **kwargs)
            if self.r.execute_model_state is None:
                # Empty/connector/dummy-only RPCs have no sampling successor.
                assert self.pending is None
                self.frame = None
            return result
        except BaseException:
            self.failed = True
            raise

    def defer(self, callback):
        r = self.r
        # Native prefill, turnover, logits processors and special sampling keep
        # the already-qualified earlier retirement point.
        sampling = r.input_batch.sampling_metadata
        if (not self.enabled or self.frame is None or not sampling.all_greedy
                or sampling.output_token_ids or r.num_prompt_logprobs):
            return False
        previous = r.input_batch.prev_sampled_token_ids
        if previous is None or r.valid_sampled_token_count_event is None:
            return False
        assert self.pending is None
        receipt = CountReceipt(r.valid_sampled_token_count_cpu,
                               r.valid_sampled_token_count_event, previous.shape[0])
        mapping = dict(r.input_batch.prev_req_id_to_index)
        assert mapping == {rid: i for i, rid in enumerate(r.input_batch.req_ids)}
        self.pending = callback, receipt, mapping
        return True

    def publish_counts(self, next_ids, counts):
        # A count published by wave N must not overwrite the receipt from N-1
        # that the worker will retire after all of N has been submitted.
        assert not self.failed
        buffer = self.buffers[self.copy_sequence % 2]
        if self.pending is not None:
            assert buffer.data_ptr() != self.pending[1].cpu.data_ptr()
        self.r.valid_sampled_token_count_cpu = buffer
        self.r.valid_sampled_token_count_event = torch.npu.Event()
        self.copy_counts(next_ids, counts)
        self.copy_sequence += 1

    def sample_tokens(self, *args, **kwargs):
        assert not self.failed
        try:
            with record_function('strengthen::submit_wave_tail'):
                result = self.sample(*args, **kwargs)
            if self.frame is None:
                assert self.pending is None
                return result
            # Record, do not synchronize: target, sampling and draft are now
            # queued on the same compute stream. Native async output owns its
            # D2H-ready event; this event is the full compute-tail boundary.
            done = torch.npu.Event()
            done.record()
            self.frame['host_submitted_ns'] = time.monotonic_ns()
            if self.pending is not None:
                callback, receipt, mapping = self.pending
                assert self.r.input_batch.prev_req_id_to_index == mapping
                getter = self.r._get_valid_sampled_token_count
                self.r._get_valid_sampled_token_count = receipt.read
                try:
                    with record_function('strengthen::retire_wave_receipt'):
                        callback()
                finally:
                    self.r._get_valid_sampled_token_count = getter
                self.pending = None
                self.retired += 1
            self.frame['host_retired_ns'] = time.monotonic_ns()
            if len(self.rows) < 256:
                self.rows.append(self.frame)
            self.last_done = done
            self.frame = None
            self.sequence += 1
            return result
        except BaseException:
            self.failed = True
            raise

    def receipt(self):
        assert self.pending is None and self.frame is None and not self.failed
        return dict(scope='stable-K5-complete-wave-before-host-retirement',
                    submitted=self.sequence, retired=self.retired,
                    count_publications=self.copy_sequence,
                    count_pinned_bytes=sum(x.numel()*x.element_size() for x in self.buffers),
                    numerical_state_banked=False, rows=self.rows)
