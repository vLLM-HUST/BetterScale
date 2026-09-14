"""Experiment-only consumer of EngineCore's agreed upcoming budget."""

import os
import torch
from vllm.config import CUDAGraphMode
from strengthen_dsv4.worker import Worker
from early_executor import receipt
from early_protocol import Consumer


class EarlyBudgetMixin:
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        r = self.model_runner
        assert r.dp_size > 1 and not r.use_dcp
        self.budget_consumer = Consumer(r.dp_rank)
        native = r._sync_metadata_across_dp

        def sync(
            num_tokens,
            is_draft_model=False,
            cudagraph_mode=CUDAGraphMode.NONE,
            allow_dp_padding=False,
        ):
            resolved = self.budget_consumer.resolve(
                num_tokens, cudagraph_mode.value, is_draft_model, allow_dp_padding
            )
            if resolved is None:
                return native(
                    num_tokens, is_draft_model, cudagraph_mode, allow_dp_padding
                )
            maximum, counts, mode = resolved
            if os.environ.get("EARLY_BUDGET_ORACLE") == "1":
                # Diagnostic gate only: compare the early decision with the
                # unchanged late native exchange BEFORE the same model forward.
                # Both ranks enter this oracle; never time it as an optimization.
                expected = native(
                    num_tokens, is_draft_model, cudagraph_mode, allow_dp_padding
                )
                assert maximum == expected[0]
                assert list(counts) == expected[1].tolist()
                assert mode == expected[2].value
                receipt(
                    "worker",
                    r.dp_rank,
                    event="native_budget_check",
                    sequence=self.budget_consumer.current.sequence,
                )

            receipt(
                "worker",
                r.dp_rank,
                event="consume",
                sequence=self.budget_consumer.current.sequence,
                tokens=num_tokens,
            )
            return maximum, torch.tensor(counts, dtype=torch.int32), CUDAGraphMode(mode)

        r._sync_metadata_across_dp = sync
        forward = r._model_forward

        def observed_forward(*args, **kwargs):
            plan = self.budget_consumer.current
            if plan is not None:
                receipt(
                    "worker",
                    r.dp_rank,
                    event="forward_enter",
                    sequence=plan.sequence,
                    admitted=plan.admitted,
                )
            result = forward(*args, **kwargs)
            if plan is not None:
                receipt(
                    "worker",
                    r.dp_rank,
                    event="forward_return",
                    sequence=plan.sequence,
                    admitted=plan.admitted,
                )
            return result

        r._model_forward = observed_forward
        return result

    def early_budget_capabilities(self):
        r = self.model_runner
        result = {}
        spec = self.vllm_config.speculative_config
        query_tokens = 1 if spec is None else 6
        for tokens in range(
            query_tokens, (r.max_num_reqs + 1) * query_tokens, query_tokens
        ):
            mode, desc = r.cudagraph_dispatcher.dispatch(
                num_tokens=tokens,
                has_lora=False,
                uniform_decode=True,
                num_active_loras=0,
            )
            if mode == CUDAGraphMode.FULL and desc.num_tokens == tokens:
                result[tokens] = (tokens, mode.value)
        return result

    def execute_model(self, scheduler_output, early_budget=None):
        if not scheduler_output.total_num_scheduled_tokens:
            assert early_budget is None
            return super().execute_model(scheduler_output)
        assert early_budget is not None
        self.budget_consumer.begin(early_budget)
        receipt(
            "worker",
            self.model_runner.dp_rank,
            event="enter",
            sequence=early_budget.sequence,
            admitted=early_budget.admitted,
        )
        result = super().execute_model(scheduler_output)
        self.budget_consumer.end()
        return result

    def execute_dummy_batch(self, early_budget=None):
        assert early_budget is not None and not early_budget.admitted
        self.budget_consumer.begin(early_budget)
        result = super().execute_dummy_batch()
        self.budget_consumer.end()
        return result


class EarlyWorker(EarlyBudgetMixin, Worker):
    """Original bounded DeepSeek candidate; package admission remains unchanged."""
