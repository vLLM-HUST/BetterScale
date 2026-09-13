"""Stable-K5 DSV4 dependency cut: CPU authorization bounds, device exact progress.

Not a second scheduler. Admission is stable pure K5 verification only;
all_modes is rejected rather than silently enabling the discarded experiment.
The existing device correction and slot mapping remain authoritative. No early
page recycling, new copy stream, or unproven pinned-buffer reuse is introduced.
"""

import torch
from torch.profiler import record_function


def stable_verification(runner, schedule, counts):
    ids = tuple(runner.input_batch.req_ids)
    previous = runner.input_batch.prev_req_id_to_index
    return (
        1 <= len(ids) <= 4
        and len(counts) == len(ids)
        and all(int(n) == 6 for n in counts)
        and previous == {rid: i for i, rid in enumerate(ids)}
        and not schedule.scheduled_new_reqs
        and set(schedule.num_scheduled_tokens) == set(ids)
        and all(
            len(schedule.scheduled_spec_decode_tokens.get(rid, ())) == 5 for rid in ids
        )
        and runner.valid_sampled_token_count_gpu is not None
        and runner._draft_token_ids is not None
        and all(
            runner.input_batch.num_computed_tokens_cpu[i]
            >= runner.input_batch.num_prompt_tokens[i]
            for i in range(len(ids))
        )
    )


class CrossStepBounds:
    def __init__(self, worker, all_modes=False):
        if all_modes:
            raise ValueError("All-mode N+2 is not part of the kept patch bundle")
        self.all_modes = False
        r = self.runner = worker.model_runner
        assert r.use_compress and r.use_async_spec_decode and not r.use_dcp
        assert r.vllm_config.model_config.hf_config.model_type == "deepseek_v4"
        assert r.num_spec_tokens == 5 and not r.need_accepted_tokens
        assert not r.supports_mm_inputs and not r.enable_prompt_embeds
        assert all(
            type(group.get_metadata_builder()).__name__ == "AscendDSACPMetadataBuilder"
            for groups in r.attn_groups
            for group in groups
        )
        self.update = r._update_states
        self.forward = r._model_forward
        self.pending = None
        self.late_commits = 0
        self.full_forwards = 0
        self.prepare = r._prepare_inputs
        self.correct = r._correct_optimistic_seq_lens_cpu
        self.build = r._build_attention_metadata
        self.enabled = True
        self.admitted = False
        self.skipped = False
        self.build_args = None
        self.rows = []
        self.calls = self.bypassed = self.reference_checks = 0
        r._update_states = self.update_states
        r._model_forward = self.model_forward
        r._prepare_inputs = self.prepare_inputs
        r._correct_optimistic_seq_lens_cpu = self.correct_bounds
        r._build_attention_metadata = self.build_metadata

    def authorized(self, schedule, counts):
        return stable_verification(self.runner, schedule, counts)

    def update_states(self, schedule):
        assert self.pending is None, "Unretired CPU bookkeeping callback"
        callback = self.update(schedule)
        r = self.runner
        counts = [
            schedule.num_scheduled_tokens.get(rid, 0) for rid in r.input_batch.req_ids
        ]
        if (
            callback is None
            or not self.enabled
            or not self.authorized(schedule, counts)
        ):
            return callback

        def defer():
            assert self.pending is None
            self.pending = callback

        return defer

    def model_forward(self, *args, **kwargs):
        with record_function("strengthen::enqueue_forward"):
            result = self.forward(*args, **kwargs)
        if self.pending is not None:
            # Enqueue the current model BEFORE waiting for the previous receipt.
            # CPU state tensors may still be H2D sources: retire input-prep DMA
            # before allowing the callback to mutate them. Never substitute a
            # device wait for ownership of pinned host memory.
            event = self.runner.prepare_inputs_event
            assert event is not None
            with record_function("strengthen::retire_input_dma"):
                event.synchronize()
            callback, self.pending = self.pending, None
            with record_function("strengthen::late_receipt"):
                callback()
            self.late_commits += 1
        return result

    def prepare_inputs(self, schedule, counts):
        self.calls += 1
        self.skipped = False
        self.admitted = self.enabled and self.authorized(schedule, counts)
        return self.prepare(schedule, counts)

    def correct_bounds(self, n):
        r = self.runner
        upper = r.optimistic_seq_lens_cpu[:n]
        if not self.admitted or not bool(
            ((upper > 0) & (upper <= r.max_model_len)).all()
        ):
            return self.correct(n)
        # This is ONLY a CPU tiling bound. Positions, slot mapping, compression
        # and attention lengths continue to use corrected device tensors.
        self.skipped = True
        self.bypassed += 1
        if len(self.rows) < 256:
            self.rows.append(dict(requests=n, upper=upper.tolist()))

    def build_metadata(self, *args, **kwargs):
        self.build_args = (args, kwargs)
        return self.build(*args, **kwargs)

    def reference_begin(self, context):
        """Shadow only: rebuild with the original synchronized exact CPU view."""
        if not self.skipped:
            return None
        r = self.runner
        n = r.input_batch.num_reqs
        upper = r.optimistic_seq_lens_cpu.clone()
        self.correct(n)
        exact = r.optimistic_seq_lens_cpu[:n]
        torch.testing.assert_close(exact, r.seq_lens[:n].cpu(), rtol=0, atol=0)
        assert bool(
            (upper[:n] >= exact).all()
        ), "CPU authorization underestimates device progress"
        args, kwargs = self.build_args
        context.attn_metadata = self.build(*args, **kwargs)[0]
        self.reference_checks += 1
        if self.rows:
            self.rows[-1]["exact"] = exact.tolist()
        return upper

    def reference_end(self, context, upper):
        if upper is None:
            return
        self.runner.optimistic_seq_lens_cpu.copy_(upper)
        args, kwargs = self.build_args
        context.attn_metadata = self.build(*args, **kwargs)[0]

    def receipt(self):
        result = dict(
            enabled=self.enabled,
            all_modes=self.all_modes,
            calls=self.calls,
            bypassed=self.bypassed,
            full_forwards=self.full_forwards,
            exact_metadata_shadow_checks=self.reference_checks,
            late_commits=self.late_commits,
            rows=self.rows,
        )
        return result


def install(worker, enabled=True, all_modes=False):
    torch.npu.synchronize()  # configuration transition, never a serving wave
    r = worker.model_runner
    state = getattr(r, "_cross_step_bounds", None)
    if state is None:
        if not enabled:
            return dict(enabled=False)
        state = r._cross_step_bounds = CrossStepBounds(worker, all_modes=all_modes)
    assert state.all_modes == all_modes
    state.enabled = enabled
    return state.receipt()
