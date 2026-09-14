"""Exercise the actual executor agreement over two local Gloo processes, no NPU."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS
import multiprocessing as mp
import os


def rank_main(rank, port, output):
    import torch.distributed as dist
    from vllm.distributed.utils import (
        stateless_init_torch_distributed_process_group,
        stateless_destroy_torch_distributed_process_group,
    )
    from early_executor import EarlyExecutor
    from early_protocol import Consumer
    from test_protocol import schedule

    os.environ["EARLY_BUDGET_OUTPUT"] = output
    ex = EarlyExecutor.__new__(EarlyExecutor)
    ex.parallel_config = NS(data_parallel_rank=rank, data_parallel_size=2)
    ex.budget_group = stateless_init_torch_distributed_process_group(
        "127.0.0.1", port, rank, 2, backend="gloo"
    )
    ex.budget_sequence = 0
    ex.previous_ids = ("a",) if rank == 0 else ("b", "c")
    ex.capacities = {6: (6, 2), 12: (12, 2)}
    consumer = Consumer(rank)
    rows = []
    try:
        for i in range(5):
            s = schedule(("a",) if rank == 0 else ("b", "c"))
            if i == 1 and rank == 1:
                s = None  # actual dummy, not a zero-token non-forward RPC
            if i == 3 and rank == 0:
                s.finished_req_ids = {"old"}
            budget = ex._agree(s)
            consumer.begin(budget)
            value = consumer.resolve(6 if rank == 0 else 12, 2, False, True)
            consumer.end()
            expected = i in (0, 4)
            assert budget.admitted == expected, (i, budget)
            assert (value is not None) == expected
            rows.append(
                dict(sequence=i, admitted=budget.admitted, counts=budget.tokens)
            )
        # Validate real execute_model RPC handoff without launching a worker.
        ex.output_rank = 0
        ex.kv_output_aggregator = None
        calls = []
        ex.collective_rpc = lambda method, **kw: calls.append((method, kw))
        ex._agree = lambda s: budget
        ex.execute_model(schedule(("a",)), non_block=True)
        assert calls[-1][1]["kwargs"]["early_budget"] == budget
        assert calls[-1][1]["non_block"] is True
        zero = schedule(())
        ex.execute_model(zero, non_block=True)
        assert "early_budget" not in calls[-1][1].get("kwargs", {})
        ex.execute_dummy_batch()
        assert calls[-1][0] == "execute_dummy_batch"
        Path(output, f"cpu-result-rank{rank}.json").write_text(
            json.dumps(rows, indent=2)
        )
    finally:
        group, ex.budget_group = ex.budget_group, None
        stateless_destroy_torch_distributed_process_group(group)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    Path(a.output).mkdir(parents=True, exist_ok=False)
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=rank_main, args=(r, a.port, a.output)) for r in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(100)
            assert process.exitcode == 0, process.exitcode
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(5)
    print("TWO_RANK_GLOO_PROTOCOL_PASS")
