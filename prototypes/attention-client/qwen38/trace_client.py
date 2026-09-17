"""SWE-derived multi-turn runner over persistent request State.

Native EP uses a global phase vote, not independently ordered collectives.
This initial scheduler gives pending prefill priority; no claim of an optimized
vLLM scheduler. Sessions retain KV, GDN and PLE across turns. No trace tool runs.
"""

import json
import time

import torch
from livemodule.llm.forward_context import ForwardContext
from livemodule.llm.qwen35.batch import Qwen35DeviceBatchTopology as Topology
from trace_ple import TraceSession
from trace_commit import RetainedCommit
from livemodule.serve.qwen38.wave import qwen38_decode_cabin_topologies
from model_transport import install_transport
from trace_plan import Seat, load_sessions


class TraceEngine:
    def __init__(self, root, cfg, args, sessions):
        self.root, self.cfg, self.args = root, cfg, args
        self.batch = args.batch_size
        self.k = args.mtp_tokens
        self.lanes = cfg.scheduler_config.max_num_batched_tokens
        self.serving = TraceSession(root)
        self.serving.target_cabin.commit = RetainedCommit(root)
        self.mailbox = self.serving.start_ple(
            token_lanes=self.lanes, max_polls=10000000
        )[root.contract.ple_layer_indices[0]]
        self.generation = torch.zeros((), dtype=torch.int64, device="npu")
        self.response = torch.zeros(
            self.mailbox.codec.response_bytes, dtype=torch.uint8, device="npu"
        )
        self.status = torch.zeros((), dtype=torch.int32, device="npu")
        self.identities = torch.ones(self.lanes, dtype=torch.int64, device="npu")
        horizons = [
            sum(len(t["ids"]) + t["output"] for t in s["turns"]) for s in sessions
        ]
        if max(horizons) + self.k > args.trace_max_context:
            raise ValueError("trace horizon exceeds configured model context")
        pages = [(max(h, 64) + self.k + 63) // 64 for h in horizons]
        if sum(pages) > root.history_domain.capacity:
            raise ValueError(
                f"request page reservation {sum(pages)} exceeds State capacity {root.history_domain.capacity}"
            )
        # Reserve actual complete horizons, not max_context for every seat.
        # Padded columns are unreachable under checked per-session lengths.
        self.blocks = torch.zeros(
            self.batch, max(pages), dtype=torch.int32, device="npu"
        )
        offset = 0
        for row, count in enumerate(pages):
            self.blocks[row, :count] = torch.arange(
                offset, offset + count, dtype=torch.int32, device="npu"
            )
            offset += count
        self.positions = torch.zeros(self.batch, dtype=torch.int64, device="npu")
        self.active = torch.zeros(self.batch, dtype=torch.bool, device="npu")
        self.remaining = torch.zeros(self.batch, dtype=torch.int32, device="npu")
        self.pending = torch.zeros(self.batch, 1, dtype=torch.int64, device="npu")
        self.multi = torch.zeros(
            self.batch,
            1,
            root.contract.hc_count * root.contract.hidden_size,
            dtype=torch.bfloat16,
            device="npu",
        )
        self.after_prefill = torch.zeros(self.batch, dtype=torch.bool, device="npu")
        self.eos = torch.full((self.batch,), -1, dtype=torch.int64, device="npu")
        self.graph = None

    def prefill(self, chunks, cursors):
        # One qualified bucket; tails use validity masks rather than triggering
        # dozens of first-use Triton compilations inside the measured session.
        width = min(512, self.lanes // self.batch)
        if max(map(len, chunks)) > width:
            raise ValueError("prefill exceeds warmed bucket")
        b = self.batch
        lengths = torch.tensor(list(map(len, chunks)), dtype=torch.int64, device="npu")
        active = lengths.gt(0)
        before = torch.tensor(cursors, dtype=torch.int64, device="npu")
        fresh = before.eq(0) & active
        ids = torch.zeros(b, width, dtype=torch.int64, device="npu")
        for row, values in enumerate(chunks):
            if values:
                ids[row, -len(values) :] = torch.tensor(
                    values, dtype=torch.int64, device="npu"
                )
        pos = (
            before[:, None]
            + torch.arange(width, device="npu")[None]
            - width
            + lengths[:, None]
        ).clamp_min(0)
        is_fresh = all(c == 0 for c in cursors)
        topology = Topology(
            before + lengths,
            lengths,
            active,
            self.blocks,
            width,
            is_fresh,
            continuation_prefill=not is_fresh,
            query_padding=True,
        )
        # Continuation prefill reads canonical GDN row0; speculative decode
        # selects a candidate and a convolution slice. Materialize only those
        # endpoints before switching phase. PLE consumes its own selector later.
        accepted = self.root.request_state.accepted_tokens.tensor[:b].clamp(
            1, self.k + 1
        )
        for layer in self.root.model.language_model.layers:
            gdn = getattr(layer, "linear_attn", None)
            if gdn is None:
                continue
            rows = (
                torch.arange(b, device="npu") * (self.k + 1)
                + gdn.conv_state.leading_physical_blocks
            )
            indices = rows[:, None] + torch.arange(self.k + 1, device="npu")[None]
            _, conv, recurrent = gdn._backend._accepted_state(gdn, indices, accepted)
            old_conv = gdn.conv_state.tensor.index_select(0, rows)
            canonical = torch.zeros_like(old_conv)
            canonical[..., : gdn.conv_kernel_size - 1] = conv
            gdn.conv_state.tensor.index_copy_(
                0, rows, torch.where(active[:, None, None], canonical, old_conv)
            )
            old_rec = gdn.recurrent_state.tensor.index_select(0, rows)
            gdn.recurrent_state.tensor.index_copy_(
                0, rows, torch.where(active[:, None, None, None], recurrent, old_rec)
            )
        context = ForwardContext({}, {}, {})
        context.batch_topology = topology
        self.cfg.remote_expert_priority = 1
        self.generation.add_(1)
        self.status.zero_()
        with context.activate():
            hidden, multi, _, valid = self.root.forward_request_owned_continuous_ple(
                ids,
                positions=pos,
                mailbox=self.mailbox,
                generation=self.generation,
                response_payload=self.response,
                status=self.status,
                slot_ids=torch.arange(self.lanes, device="npu") // width,
                request_generations=self.identities,
                destination_generations=self.identities,
            )
        valid_mask = topology.query_valid.reshape(-1)
        if not bool((valid[: b * width] | ~valid_mask).all().cpu()):
            raise RuntimeError(
                f"PLE prefill publication failed: status={self.status.cpu().tolist()}, valid={int(valid.sum().cpu())}, generation={self.generation.cpu().tolist()}, worker_errors={[str(w._error) for w in self.serving._workers.values()]}"
            )
        # Pair each real input token with its immediately preceding target row.
        # Right padding is NOT a preceding row: replace that boundary explicitly.
        previous = torch.cat((self.multi, multi[:, :-1]), dim=1)
        rowids = torch.arange(b, device="npu")
        previous[rowids, (width - lengths).clamp_max(width - 1)] = self.multi[:, 0]
        mtp_lengths = (lengths - fresh.to(lengths.dtype)).clamp_min(0)
        mtp_topology = Topology(
            (before + lengths - 1).clamp_min(0),
            mtp_lengths,
            mtp_lengths.gt(0),
            self.blocks,
            width,
            is_fresh,
            continuation_prefill=not is_fresh,
            query_padding=True,
        )
        self.serving.draft_cabin._run_mtp(
            ids, (pos - 1).clamp_min(0), previous, mtp_topology
        )
        self.multi.copy_(torch.where(active[:, None, None], multi[:, -1:], self.multi))
        return self.root.compute_top_tokens(hidden[:, -1:])[:, 0].cpu().tolist()

    def wave(self):
        # Initial scheme-A prompt pairing ends at position p-2. Its first pending
        # token is processed at p-1, exactly as initialize_after_prefill_chunk.
        draft_pos = self.positions - self.after_prefill.to(self.positions.dtype)
        drafts, _ = qwen38_decode_cabin_topologies(
            draft_pos, self.active, self.blocks, self.k
        )
        _, target = qwen38_decode_cabin_topologies(
            self.positions, self.active, self.blocks, self.k
        )
        proposals, draft_multi = self.serving.draft_cabin(
            self.pending, draft_pos[:, None], self.multi, drafts
        )
        self.generation.add_(1)
        self.status.zero_()
        self.cfg.remote_expert_priority = 0
        result = self.serving.target_cabin(
            torch.cat((self.pending, proposals), dim=1),
            self.positions[:, None] + torch.arange(self.k + 1, device="npu")[None],
            proposals,
            target,
            self.active[:, None].expand(self.batch, self.k + 1),
            mailbox=self.mailbox,
            generation=self.generation,
            response_payload=self.response,
            status=self.status,
            slot_ids=torch.arange(self.lanes, device="npu") // (self.k + 1),
            request_generations=self.identities,
            destination_generations=self.identities,
            active=self.active,
            remaining=self.remaining,
            eos_token_ids=self.eos,
        )
        accepted, committed, _, count, _, _, pending, multi, observed = result
        live = self.active & accepted.eq(self.k)
        topology = Topology(
            self.positions + self.k + 1,
            live.to(self.positions.dtype),
            live,
            self.blocks,
            1,
            False,
        )
        self.serving.draft_cabin.reconcile_all_accepted(
            proposals[:, -1:], self.positions[:, None] + self.k, draft_multi, topology
        )
        self.pending.copy_(
            torch.where(self.active[:, None], pending[:, None], self.pending)
        )
        self.multi.copy_(
            torch.where(self.active[:, None, None], multi[:, None], self.multi)
        )
        self.positions.add_(count.to(self.positions.dtype))
        self.remaining.sub_(count.to(self.remaining.dtype))
        self.after_prefill.zero_()
        return count, self.pending, observed, committed

    def sync_seats(self, seats):
        self.positions.copy_(torch.tensor([s.cursor for s in seats], device="npu"))
        self.active.copy_(
            torch.tensor(
                [not s.done and not s.todo and s.remaining > 0 for s in seats],
                device="npu",
            )
        )
        self.remaining.copy_(
            torch.tensor([s.remaining for s in seats], dtype=torch.int32, device="npu")
        )
        self.pending.copy_(
            torch.tensor([[s.pending or 0] for s in seats], device="npu")
        )

    def warm(self):
        self.prefill([[9707, 11, 1879] for _ in range(self.batch)], [0] * self.batch)
        self.prefill([[9707, 11, 1879] for _ in range(self.batch)], [3] * self.batch)
        self.positions.fill_(6)
        self.pending.fill_(11)
        self.remaining.fill_(32)
        self.active.fill_(True)
        self.after_prefill.fill_(True)
        self.wave()
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            with torch.npu.graph(self.graph):
                self.outputs = self.wave()
        stream.synchronize()
        self.graph.replay()
        torch.npu.synchronize()
        self.positions.zero_()
        self.multi.zero_()
        self.after_prefill.zero_()
        self.root.request_state.accepted_tokens.tensor.fill_(1)


def run_trace(root, cfg, args, rank, stage):
    if args.mtp_tokens != 1:
        raise ValueError("first retained-trace qualification uses K1")
    sessions = load_sessions(
        args.trace_plan, args.trace_count, args.trace_turns, args.trace_output_cap
    )
    assigned = sessions[args.source :: args.sources]
    if len(assigned) != args.batch_size:
        raise ValueError("fixed-resident pilot requires count == sources * batch")
    seats = [Seat(s) for s in assigned]
    install_transport(cfg, args, rank)
    # No source may initialize a new data-plane communicator while its peer
    # is still assembling the real expert catalog on CPU.
    torch.distributed.barrier()
    torch.npu.synchronize()
    stage(
        "trace-transport-ready",
        allocated=torch.npu.memory_allocated(),
        reserved=torch.npu.memory_reserved(),
    )
    engine = TraceEngine(root, cfg, args, assigned)
    engine.warm()
    torch.npu.synchronize()
    (args.directory / f"trace-ready-{2 * args.source + rank}").touch()
    deadline = time.monotonic() + 180
    while not all(
        (args.directory / f"trace-ready-{i}").exists() for i in range(2 * args.sources)
    ):
        if time.monotonic() > deadline:
            raise TimeoutError("trace warmup rendezvous")
        time.sleep(0.01)
    stage(
        "trace-ready",
        state_history_tokens=root.history_domain.capacity * 64,
        allocated=torch.npu.memory_allocated(),
        reserved=torch.npu.memory_reserved(),
        free=torch.npu.mem_get_info()[0],
    )
    started = time.monotonic()
    events = []
    prefixes = []
    outputs_by_seat = [[] for _ in seats]
    arrivals = [started] * len(seats)
    last_output = [None] * len(seats)
    intervals = []
    step = 0
    try:
        while True:
            # Co-located EP cannot enter different layers/phases across sources.
            intent = torch.tensor(
                [
                    int(any(s.todo for s in seats if not s.done)),
                    int(any(not s.done for s in seats)),
                ],
                dtype=torch.int32,
                device="npu",
            )
            if args.colocated:
                torch.distributed.all_reduce(intent, op=torch.distributed.ReduceOp.MAX)
            prefill, alive = intent.cpu().tolist()
            if not alive:
                break
            begin = time.monotonic()
            if prefill:
                width = min(512, engine.lanes // engine.batch)
                chunks = [s.take(width) if not s.done and s.todo else [] for s in seats]
                tokens = engine.prefill(chunks, [s.cursor for s in seats])
                for i, (seat, chunk, token) in enumerate(zip(seats, chunks, tokens)):
                    if chunk:
                        if seat.finish_prefill(len(chunk), token):
                            engine.after_prefill[i] = True
                            outputs_by_seat[i].append(int(token))
                            last_output[i] = time.monotonic()
                            seat.events.append(
                                dict(
                                    turn=seat.turn,
                                    first_token_seconds=time.monotonic() - started,
                                    ttft_seconds=time.monotonic() - arrivals[i],
                                    cumulative_reused_tokens=seat.reused,
                                )
                            )
                counts = [int(bool(c) and not s.todo) for c, s in zip(chunks, seats)]
            else:
                engine.sync_seats(seats)
                engine.cfg.remote_expert_priority = 0
                engine.graph.replay()
                count, pending, observed, committed = engine.outputs
                if not bool(
                    (
                        observed[: engine.batch * 2].reshape(engine.batch, 2)
                        | ~engine.active[:, None]
                    )
                    .all()
                    .cpu()
                ):
                    raise RuntimeError("PLE decode publication failed")
                counts = count.cpu().tolist()
                pending = pending[:, 0].cpu().tolist()
                committed = committed.cpu().tolist()
                for i, (seat, n, token, row) in enumerate(
                    zip(seats, counts, pending, committed)
                ):
                    if n:
                        now = time.monotonic()
                        if last_output[i] is not None:
                            intervals.append(now - last_output[i])
                        last_output[i] = now
                        outputs_by_seat[i].extend(row[:n])
                    seat.finish_decode(n, token)
            for i, seat in enumerate(seats):
                if not seat.done and not seat.todo and not seat.remaining:
                    # First old history page must survive all following turns.
                    if seat.turn == 0 and seat.cursor >= 64:
                        for _, state in root.named_states():
                            if state.domain is root.history_domain:
                                page = (
                                    int(engine.blocks[i, 0].cpu())
                                    + state.leading_physical_blocks
                                )
                                value = state.tensor[page]
                                prefixes.append((value, value.clone()))
                    seat.next_turn()
                    arrivals[i] = time.monotonic()
                    last_output[i] = None
            elapsed = time.monotonic() - begin
            events.append(
                dict(
                    step=step,
                    phase="prefill" if prefill else "decode",
                    seconds=elapsed,
                    output_tokens=sum(counts),
                    contexts=[s.cursor for s in seats],
                )
            )
            if step % 20 == 0 or prefill:
                stage("trace-wave", **events[-1])
            step += 1
        torch.npu.synchronize()
        elapsed = time.monotonic() - started
        prefix_exact = all(torch.equal(a, b) for a, b in prefixes)
        if not prefix_exact:
            raise RuntimeError("a retained prefix history page was overwritten")
        from livemodule.llm.distributed import get_tp_group

        flat = torch.tensor(
            [t for row in outputs_by_seat for t in row], dtype=torch.int64, device="npu"
        )
        sizes = [torch.empty((), dtype=torch.int64, device="npu") for _ in range(2)]
        torch.distributed.all_gather(
            sizes,
            torch.tensor(flat.numel(), dtype=torch.int64, device="npu"),
            group=get_tp_group().device_group,
        )
        if not torch.equal(sizes[0], sizes[1]):
            raise RuntimeError("TP output counts disagree")
        agreed = [torch.empty_like(flat) for _ in range(2)]
        torch.distributed.all_gather(agreed, flat, group=get_tp_group().device_group)
        if not torch.equal(*agreed):
            raise RuntimeError("TP output token IDs disagree")
        receipt = dict(
            prefix_first_page_exact=prefix_exact,
            tp_output_exact=True,
            output_intervals_seconds=intervals,
            status="PASS",
            scope="SWE-derived retained State workload; native EP8 same-model control, not unmodified vLLM",
            topology="colocated" if args.colocated else "separated",
            source=args.source,
            seconds=elapsed,
            truncated_gate=bool(args.trace_turns or args.trace_output_cap),
            mtp_tokens=args.mtp_tokens,
            sessions=[
                dict(
                    trace_id=s.session["trace_id"],
                    completed_turns=s.turn,
                    output_tokens=s.generated,
                    prefill_tokens=s.prefilled,
                    prefix_reused_tokens=s.reused,
                    final_encoded_context=s.cursor,
                    events=s.events,
                )
                for s in seats
            ],
            waves=events,
            memory=dict(
                allocated=torch.npu.memory_allocated(),
                reserved=torch.npu.memory_reserved(),
                peak_allocated=torch.npu.max_memory_allocated(),
            ),
        )
        (args.directory / f"attention{2 * args.source + rank}.json").write_text(
            json.dumps(receipt, indent=2)
        )
    finally:
        engine.graph.reset()
        engine.serving.close()
        if rank == 0 or args.colocated:
            cfg.remote_expert_transport.close()
