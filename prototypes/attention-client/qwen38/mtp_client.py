"""Real greedy MTP closure over the existing persistent target/BF16 expert pool.

State endpoint selection and the all-accepted MTP cache reconciliation are
borrowed directly from the owned LiveInfer Qwen38 serving programs. This gate
keeps request membership fixed; it does not introduce a new serving scheduler.
"""

import gc
import json
import time

import torch

from client import Session
from livemodule.llm.distributed import get_tp_group
from livemodule.llm.forward_context import ForwardContext
from livemodule.llm.qwen35.batch import Qwen35DeviceBatchTopology
from livemodule.serve.qwen38 import Qwen38ServingSession
from livemodule.serve.qwen38.wave import qwen38_decode_cabin_topologies


def run_mtp(root, cfg, args, rank, stage):
    batch, k = args.batch_size, args.mtp_tokens
    lanes = cfg.scheduler_config.max_num_batched_tokens
    from model_transport import install_transport

    install_transport(cfg, args, rank)
    torch.distributed.barrier()
    serving = Qwen38ServingSession(root)
    mailbox = serving.start_ple(token_lanes=lanes, max_polls=10000000)[
        root.contract.ple_layer_indices[0]
    ]
    generation = torch.zeros((), dtype=torch.int64, device="npu")
    response = torch.zeros(
        mailbox.codec.response_bytes, dtype=torch.uint8, device="npu"
    )
    status = torch.zeros((), dtype=torch.int32, device="npu")
    identities = torch.ones(lanes, dtype=torch.int64, device="npu")
    # Bound the entire run, including warm-up, by disjoint request-owned pages.
    pages = (args.prompt_width + (args.decode_steps + 8) * (k + 1) + 63) // 64
    blocks = torch.arange(batch * pages, dtype=torch.int32, device="npu").view(
        batch, pages
    )
    active = torch.ones(batch, dtype=torch.bool, device="npu")
    positions = torch.full((batch,), args.prompt_width, dtype=torch.int64, device="npu")
    remaining = torch.full((batch,), 100000, dtype=torch.int32, device="npu")
    eos = torch.full((batch,), -1, dtype=torch.int64, device="npu")
    prompt = torch.tensor(
        [
            (
                [
                    9707
                    + 17
                    * (args.source * batch + row if args.distinct_prompts else row),
                    11,
                    1879,
                ]
                * ((args.prompt_width + 2) // 3)
            )[: args.prompt_width]
            for row in range(batch)
        ],
        device="npu",
    )
    prompt_positions = (
        torch.arange(args.prompt_width, device="npu")[None]
        .expand(batch, -1)
        .contiguous()
    )
    slots = torch.arange(lanes, device="npu") // args.prompt_width
    topology = Qwen35DeviceBatchTopology(
        sequence_lengths=positions.to(torch.int32),
        query_lengths=positions.to(torch.int32),
        active=active,
        block_table=blocks,
        query_length=args.prompt_width,
        fresh_prefill=True,
    )
    context = ForwardContext({}, {}, {})
    context.batch_topology = topology
    cfg.remote_expert_priority = 1
    generation.add_(1)
    with context.activate():
        hidden, multi, _, valid = root.forward_request_owned_continuous_ple(
            prompt,
            positions=prompt_positions,
            mailbox=mailbox,
            generation=generation,
            response_payload=response,
            status=status,
            slot_ids=slots,
            request_generations=identities,
            destination_generations=identities,
        )
    assert bool(valid[: prompt.numel()].all().cpu())
    pending = root.compute_top_tokens(hidden[:, -1:]).clone()
    all_generated = pending.cpu().tolist()
    reference = None
    reference_layers = {}

    def layer_hooks(destination):
        def hook(index):
            def save(module, inputs, output):
                value = (
                    output.mixed_input
                    if hasattr(output, "mixed_input")
                    else (output[0] if isinstance(output, tuple) else output)
                )
                if isinstance(value, torch.Tensor):
                    destination[index] = value.reshape(batch, -1, value.shape[-1])[
                        :, :1
                    ].clone()

            return save

        handles = [
            layer.register_forward_hook(hook(index))
            for index, layer in enumerate(root.model.language_model.layers)
        ]
        for name, module in root.model.language_model.layers[2].named_modules():
            if name:
                handles.append(module.register_forward_hook(hook("layer2/" + name)))
        return handles

    if args.reference_tokens:
        # Independent width-one target continuation from identical post-prefill
        # State. Speculation has no permission to change the greedy policy.
        reference = [row.copy() for row in all_generated]
        reference_ids = pending.clone()
        state_pairs = [
            (state.tensor, state.tensor.clone()) for _, state in root.named_states()
        ]
        hooks = layer_hooks(reference_layers)
        for offset in range(args.reference_tokens - 1):
            ref_positions = positions[:, None] + offset
            ref_context = ForwardContext({}, {}, {})
            ref_context.batch_topology = Qwen35DeviceBatchTopology(
                sequence_lengths=(ref_positions[:, 0] + 1).to(torch.int32),
                query_lengths=torch.ones(batch, dtype=torch.int32, device="npu"),
                active=active,
                block_table=blocks,
                query_length=1,
                fresh_prefill=False,
            )
            generation.add_(1)
            status.zero_()
            with ref_context.activate():
                ref_hidden, _, _, ref_valid = root.forward_request_owned_continuous_ple(
                    reference_ids,
                    positions=ref_positions,
                    mailbox=mailbox,
                    generation=generation,
                    response_payload=response,
                    status=status,
                    slot_ids=torch.arange(lanes, device="npu"),
                    request_generations=identities,
                    destination_generations=identities,
                )
            if offset == 0:
                for handle in hooks:
                    handle.remove()
            assert bool(ref_valid[:batch].all().cpu())
            reference_ids = root.compute_top_tokens(ref_hidden)
            for row, token in zip(reference, reference_ids[:, 0].cpu().tolist()):
                row.append(token)
        for value, backup in state_pairs:
            value.copy_(backup)
        del state_pairs, backup, value, ref_hidden
        stage("target-reference", tokens=args.reference_tokens)

    def record_outputs(outputs):
        for row, count, cabin in zip(
            all_generated, outputs[2].cpu().tolist(), outputs[1].cpu().tolist()
        ):
            row.extend(cabin[:count])

    # Scheme-A initialization pairs shifted prompt/pending tokens with the
    # preceding target multi stream, rather than inventing zero draft history.
    initial_topology = Qwen35DeviceBatchTopology(
        sequence_lengths=positions.to(torch.int32),
        query_lengths=positions.to(torch.int32),
        active=active,
        block_table=blocks,
        query_length=args.prompt_width,
        fresh_prefill=True,
    )
    continuation, _ = qwen38_decode_cabin_topologies(positions, active, blocks, k)
    proposals, draft_multi = serving.draft_cabin.initialize_after_prefill(
        torch.cat((prompt, pending), dim=1),
        prompt_positions,
        multi,
        initial_topology,
        continuation[: k - 1],
    )
    next_multi = multi[:, -1:].clone()
    verify_slots = torch.arange(lanes, device="npu") // (k + 1)
    cfg.remote_expert_priority = 0

    def verify(proposals, draft_multi, target_topology):
        generation.add_(1)
        status.zero_()
        token_positions = positions[:, None] + torch.arange(k + 1, device="npu")[None]
        verification_ids = torch.cat((pending, proposals), dim=1)
        result = serving.target_cabin(
            verification_ids,
            token_positions,
            proposals,
            target_topology,
            active[:, None].expand(batch, k + 1),
            mailbox=mailbox,
            generation=generation,
            response_payload=response,
            status=status,
            slot_ids=verify_slots,
            request_generations=identities,
            destination_generations=identities,
            active=active,
            remaining=remaining,
            eos_token_ids=eos,
        )
        accepted, committed, _, count, _, _, next_pending, target_multi, observed = (
            result
        )
        reconcile_active = active & accepted.eq(k)
        reconcile_topology = Qwen35DeviceBatchTopology(
            sequence_lengths=positions + k + 1,
            query_lengths=reconcile_active.to(positions.dtype),
            active=reconcile_active,
            block_table=blocks,
            query_length=1,
            fresh_prefill=False,
        )
        serving.draft_cabin.reconcile_all_accepted(
            proposals[:, -1:], positions[:, None] + k, draft_multi, reconcile_topology
        )
        pending.copy_(next_pending[:, None])
        next_multi.copy_(target_multi[:, None])
        positions.add_(count.to(positions.dtype))
        remaining.sub_(count.to(remaining.dtype))
        return accepted, committed, count, observed

    _, target_topology = qwen38_decode_cabin_topologies(positions, active, blocks, k)
    initial_layers = {}
    hooks = layer_hooks(initial_layers) if reference is not None else []
    first = verify(proposals, draft_multi, target_topology)
    for handle in hooks:
        handle.remove()
    if reference is not None:
        layer_errors = [
            float(
                (initial_layers[i].float() - reference_layers[i].float()).norm()
                / reference_layers[i].float().norm().clamp_min(1e-9)
            )
            for i in range(48)
        ]
        (
            args.directory
            / f"initial-layer-shadow-{args.tp_size * args.source + rank}.json"
        ).write_text(json.dumps(layer_errors))
        stage("initial-layer-shadow", first_layers=layer_errors[:8])
        fine_errors = {
            name: float(
                (initial_layers[name].float() - ref.float()).norm()
                / ref.float().norm().clamp_min(1e-9)
            )
            for name, ref in reference_layers.items()
            if isinstance(name, str) and name in initial_layers
        }
        (
            args.directory
            / f"initial-layer2-fine-{args.tp_size * args.source + rank}.json"
        ).write_text(json.dumps(fine_errors, indent=2))
        del initial_layers, reference_layers
    assert bool(first[3][: batch * (k + 1)].all().cpu())
    record_outputs(first)
    stage("mtp-initialized", accepted=first[0].cpu().tolist())

    def wave():
        drafts, target = qwen38_decode_cabin_topologies(positions, active, blocks, k)
        proposals, draft_multi = serving.draft_cabin(
            pending, positions[:, None], next_multi, drafts
        )
        return verify(proposals, draft_multi, target)

    graph = None
    shadow = None
    # The snapshot includes model State and all advancing scheduler operands,
    # but NOT transport/PLE generations: those must remain monotonically fresh.
    if args.decode_graph:
        operands = [state.tensor for _, state in root.named_states()] + [
            pending,
            next_multi,
            positions,
            remaining,
        ]
        saved = [x.clone() for x in operands]
        expected = [x.clone() for x in wave()[:3]]
        torch.npu.synchronize()
        for x, backup in zip(operands, saved):
            x.copy_(backup)
        capture_stream = torch.npu.Stream()
        capture_stream.wait_stream(torch.npu.current_stream())
        graph = torch.npu.NPUGraph()
        with torch.npu.stream(capture_stream):
            with torch.npu.graph(graph):
                outputs = wave()
        capture_stream.synchronize()
        for x, backup in zip(operands, saved):
            x.copy_(backup)
        graph.replay()
        torch.npu.synchronize()
        shadow = [torch.equal(x, ref) for x, ref in zip(outputs[:3], expected)]
        assert all(shadow), shadow
        del saved, expected, operands
        stage("mtp-full-graph-shadow", equal=shadow)
    else:
        outputs = wave()
        torch.npu.synchronize()

    record_outputs(outputs)
    if args.align_steady_start:
        if args.defer_steady_gc:
            gc.collect()
        (args.directory / f"steady-ready-{args.tp_size * args.source + rank}").touch()
        deadline = time.monotonic() + 180
        while not all(
            (args.directory / f"steady-ready-{i}").exists()
            for i in range(args.tp_size * args.sources)
        ):
            if time.monotonic() > deadline:
                raise TimeoutError("MTP source warm-up rendezvous")
            time.sleep(0.001)
    gc_enabled = gc.isenabled()
    if args.defer_steady_gc:
        gc.disable()
    times, starts, counts, acceptances = [], [], [], []
    generated = [[] for _ in range(batch)]
    try:
        for step in range(args.decode_steps):
            begin = time.monotonic()
            if graph is not None:
                graph.replay()
            else:
                outputs = wave()
            accepted, committed, count, observed = outputs
            assert bool(observed[: batch * (k + 1)].all().cpu())
            payload = torch.cat((accepted[:, None], count[:, None], committed), dim=1)
            agreed = [torch.empty_like(payload) for _ in range(args.tp_size)]
            torch.distributed.all_gather(
                agreed, payload, group=get_tp_group().device_group
            )
            assert all(torch.equal(agreed[0], other) for other in agreed[1:])
            rows = payload.cpu().tolist()
            elapsed = time.monotonic() - begin
            starts.append(begin)
            times.append(elapsed)
            counts.append(sum(row[1] for row in rows))
            acceptances.append([row[0] for row in rows])
            for output, row in zip(generated, rows):
                output.extend(row[2 : 2 + row[1]])
            for row, values in zip(all_generated, rows):
                row.extend(values[2 : 2 + values[1]])
            stage(
                "mtp-wave",
                step=step,
                seconds=elapsed,
                output_tokens=counts[-1],
                accepted=acceptances[-1],
            )
    finally:
        if args.defer_steady_gc:
            if gc_enabled:
                gc.enable()
            gc.collect()
        if graph is not None:
            graph.reset()
        serving.close()
    torch.distributed.barrier()
    calls = cfg.remote_expert_transport.close() if rank == 0 else None
    reference_matches = (
        None
        if reference is None
        else [
            actual[: len(ref)] == ref for actual, ref in zip(all_generated, reference)
        ]
    )
    receipt = dict(
        target_reference_matches=reference_matches,
        target_reference_ids=reference,
        all_output_ids_by_request=all_generated,
        status=(
            "PASS" if reference_matches is None or all(reference_matches) else "FAIL"
        ),
        scope="full48 target plus BF16 MTP greedy speculative closure; not quality",
        topology=(
            "TP2xDP4_EP8_colocated" if args.colocated else "TP2_sources_E4_separated"
        ),
        source=args.source,
        batch_size=batch,
        mtp_tokens=k,
        state_gib=args.state_gib,
        prompt_width=args.prompt_width,
        calls=calls,
        decode_graph=bool(args.decode_graph),
        graph_shadow_exact=shadow,
        aligned_steady_start=args.align_steady_start,
        deferred_steady_gc=args.defer_steady_gc,
        wave_seconds=times,
        wave_started_seconds=starts,
        wave_output_tokens=counts,
        accepted_drafts=acceptances,
        output_ids_by_request=generated,
        output_tokens_per_second=sum(counts) / sum(times),
        timing_scope="post-initialization and post-shadow closed-loop MTP+target+reconciliation+host readback; all measured steps included",
        repaired_ple_metadata=root.repaired_ple_metadata,
    )
    (args.directory / f"attention{args.tp_size * args.source + rank}.json").write_text(
        json.dumps(receipt, indent=2)
    )

    if reference_matches is not None:
        assert all(reference_matches), {
            "reference_matches": reference_matches,
            "details": "attention receipt and initial-layer-shadow JSON",
        }
