"""Experimental native MP executor: agree in EngineCore, before worker RPC.

MP is intentional: a worker still issuing the previous eager draft must not
prevent its existing independent EngineCore from scheduling the next packet.
Compare against the same native MP executor, NOT a UniProc baseline.
"""

import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import vllm.envs as envs
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.distributed.utils import (
    stateless_destroy_torch_distributed_process_group,
    stateless_init_torch_distributed_process_group,
)
from early_protocol import Proposal, agree, propose
from early_rpc import execute_with_budget


def receipt(role, rank, **values):
    folder = os.environ.get("EARLY_BUDGET_OUTPUT")
    if folder:
        path = Path(folder) / f"{role}-rank{rank}.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps(dict(time_ns=time.time_ns(), **values)) + "\n")


class EarlyExecutor(MultiprocExecutor):
    def _init_executor(self):
        p = self.parallel_config
        assert p.tensor_parallel_size == p.pipeline_parallel_size == 1
        assert p.data_parallel_size > 1 and p.enable_expert_parallel
        assert self.scheduler_config.async_scheduling
        assert not self.vllm_config.lora_config
        assert not self.vllm_config.kv_transfer_config
        assert p.data_parallel_size_local == p.data_parallel_size
        # Local experiment endpoint, separate from native worker/EngineCore PGs.
        # Do not consume or mutate donor's initialization-port sequence.
        self.budget_group = stateless_init_torch_distributed_process_group(
            "127.0.0.1",
            int(os.environ["EARLY_BUDGET_PORT"]),
            p.data_parallel_rank,
            p.data_parallel_size,
            backend="gloo",
        )
        spec = self.vllm_config.speculative_config
        if spec is not None:
            assert spec.method == "dspark" and spec.num_speculative_tokens == 5
        self.query_tokens = 1 if spec is None else 6
        self.max_requests = self.scheduler_config.max_num_seqs
        self.budget_sequence = 0
        self.previous_ids = ()
        self.capacities = {}
        super()._init_executor()

    def initialize_from_config(self, configs):
        super().initialize_from_config(configs)
        (self.capacities,) = self.collective_rpc("early_budget_capabilities")

    def _agree(self, schedule):
        rank = self.parallel_config.data_parallel_rank
        proposal = propose(
            self.budget_sequence,
            schedule,
            self.previous_ids,
            self.capacities,
            query_tokens=self.query_tokens,
            max_requests=self.max_requests,
        )
        started = time.time_ns()
        size = self.parallel_config.data_parallel_size
        packet = torch.zeros((4, size), dtype=torch.int64)
        packet[:, rank] = torch.tensor(
            (
                proposal.sequence + 1,
                proposal.tokens,
                proposal.mode,
                int(proposal.eligible),
            )
        )
        # This wait is in the existing EngineCore, not the busy NPU worker.
        # Its effectiveness must be measured against preceding device work.
        dist.all_reduce(packet, group=self.budget_group)
        budget = agree(
            [
                Proposal(
                    int(packet[0, r]) - 1,
                    int(packet[1, r]),
                    int(packet[2, r]),
                    bool(packet[3, r]),
                )
                for r in range(size)
            ],
            allowed_tokens=tuple({shape[0] for shape in self.capacities.values()}),
        )
        receipt(
            "parent",
            rank,
            sequence=budget.sequence,
            event="agreed",
            started_ns=started,
            admitted=budget.admitted,
            tokens=budget.tokens,
        )
        self.budget_sequence += 1
        self.previous_ids = tuple(schedule.num_scheduled_tokens) if schedule else ()
        return budget

    def execute_model(self, scheduler_output, non_block=False):
        if not scheduler_output.total_num_scheduled_tokens:
            # Native non-external-launcher zero-token execute returns without
            # a target. EngineCore emits a separate dummy forward if needed.
            self.previous_ids = ()
            return super().execute_model(scheduler_output, non_block=non_block)
        budget = self._agree(scheduler_output)
        return self.collective_rpc(
            execute_with_budget,
            args=(scheduler_output,),
            kwargs={"early_budget": budget},
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
        )

    def execute_dummy_batch(self):
        budget = self._agree(None)
        self.collective_rpc(
            "execute_dummy_batch",
            kwargs={"early_budget": budget},
            unique_reply_rank=self.output_rank,
        )

    def shutdown(self):
        try:
            super().shutdown()
        finally:
            group = getattr(self, "budget_group", None)
            if group is not None:
                self.budget_group = None
                stateless_destroy_torch_distributed_process_group(group)
