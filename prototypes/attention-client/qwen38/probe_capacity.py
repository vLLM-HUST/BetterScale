"""Real-weight memory-fit gate, not a long-context numerical/quality test.

Physically touch all activated State, then exercise long-context prefill and
FULL MTP decode over synthetic zero history. This separates memory feasibility
from the hours needed to generate each retained prefix through the network.
"""

import json
import time
import torch
from model_transport import install_transport
from trace_client import TraceEngine


def run_capacity(root, cfg, args, rank, stage):
    if args.mtp_tokens != 1 or args.trace_plan:
        raise ValueError("capacity qualification uses K1 and no workload replay")
    install_transport(cfg, args, rank)
    torch.distributed.barrier()
    torch.npu.synchronize()
    width = min(512, cfg.scheduler_config.max_num_batched_tokens // args.batch_size)
    available_per_seat = root.history_domain.capacity // args.batch_size * 64
    prefix = min(available_per_seat, args.trace_max_context) - width - 64
    if prefix < 64:
        raise ValueError("State cannot fit even one pressure page per seat")
    # range only supplies len(); no million-element Python token lists.
    sessions = [
        dict(turns=[dict(ids=range(prefix + width), output=8)])
        for _ in range(args.batch_size)
    ]
    samples = []

    def sample(phase):
        torch.npu.synchronize()
        free, total = torch.npu.mem_get_info()
        row = dict(
            phase=phase,
            allocated=torch.npu.memory_allocated(),
            reserved=torch.npu.memory_reserved(),
            peak_allocated=torch.npu.max_memory_allocated(),
            peak_reserved=torch.npu.max_memory_reserved(),
            driver_free=free,
            driver_total=total,
        )
        samples.append(row)
        stage("capacity-memory", **row)

    sample("transport-loaded")
    engine = TraceEngine(root, cfg, args, sessions)
    try:
        engine.warm()
        sample("warm-graph")
        # Never clone the State: that would double the very capacity under test.
        touched = 0
        for _, state in root.named_states():
            state.tensor.zero_()
            touched += state.tensor.numel() * state.tensor.element_size()
        root.request_state.accepted_tokens.tensor.fill_(1)
        engine.multi.zero_()
        sample("all-state-touched")
        stage(
            "capacity-pressure-start",
            prefix_per_request=prefix,
            query_width=width,
            requests_per_source=args.batch_size,
            history_tokens_per_group=root.history_domain.capacity * 64,
        )
        begin = time.monotonic()
        tokens = engine.prefill(
            [[9707] * width for _ in range(args.batch_size)], [prefix] * args.batch_size
        )
        if not bool(torch.isfinite(engine.multi).all().cpu()):
            raise RuntimeError("nonfinite long-context prefill output")
        sample("long-prefill")
        prefill_seconds = time.monotonic() - begin
        engine.positions.fill_(prefix + width)
        engine.pending.copy_(
            torch.tensor(tokens, dtype=torch.int64, device="npu")[:, None]
        )
        engine.remaining.fill_(8)
        engine.active.fill_(True)
        engine.after_prefill.fill_(True)
        decode_seconds = []
        for step in range(3):
            begin = time.monotonic()
            engine.graph.replay()
            count, pending, observed, committed = engine.outputs
            if not bool(observed[: args.batch_size * 2].all().cpu()):
                raise RuntimeError("PLE long-context decode failed")
            if not bool(torch.isfinite(engine.multi).all().cpu()):
                raise RuntimeError("nonfinite long-context decode output")
            sample(f"long-decode-{step}")
            decode_seconds.append(time.monotonic() - begin)
        # Driver-free samples and allocator peaks occur at different moments.
        # Do not subtract a historical peak reserve from later free memory:
        # capture can release pools while communicator allocations grow.
        sampled_free = min(s["driver_free"] for s in samples)
        result = dict(
            status="PASS",
            scope="real48+MTP weights, all State touched, synthetic zero-history memory-pressure gate; NOT quality or genuine long-prefix service",
            topology="colocated" if args.colocated else "separated",
            source=args.source,
            state_budget_gib=args.state_gib,
            state_tensor_bytes=touched,
            requests_per_source=args.batch_size,
            sources=args.sources,
            history_tokens_per_group=root.history_domain.capacity * 64,
            allocated_history_tokens_whole_machine=args.sources
            * root.history_domain.capacity
            * 64,
            pressure_prefix_per_request=prefix,
            pressure_prefill_tokens_per_source=width * args.batch_size,
            exercised_history_tokens_whole_machine=args.sources
            * args.batch_size
            * prefix,
            model_max_context=args.trace_max_context,
            minimum_sampled_driver_free_bytes=sampled_free,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            samples=samples,
        )
        (
            args.directory / f"attention{args.source * args.tp_size + rank}.json"
        ).write_text(json.dumps(result, indent=2))
    finally:
        if engine.graph is not None:
            engine.graph.reset()
        engine.serving.close()
        if rank == 0 or args.colocated:
            cfg.remote_expert_transport.close()
