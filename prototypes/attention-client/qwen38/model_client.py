"""TP2 target-only full-root gate with real PLE and external routed experts.

Correctness/continuation first. This is not a throughput benchmark or a quality
suite. Decode graph qualification follows the eager all-layer gate.
"""

import argparse
import ctypes as C
import faulthandler
import gc
import threading
import json
import os
import time
from datetime import timedelta
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--directory", type=Path, required=True)
p.add_argument("--build", type=Path, required=True)
p.add_argument("--construct-only", action="store_true")
p.add_argument("--decode-graph", action="store_true")
p.add_argument("--source", type=int, choices=(0, 1), default=0)
p.add_argument("--sources", type=int, choices=(1, 2), default=1)
p.add_argument("--decode-steps", type=int, default=3)
p.add_argument("--align-steady-start", action="store_true")
p.add_argument("--observe-pauses", action="store_true")
p.add_argument("--defer-steady-gc", action="store_true")
a = p.parse_args()
assert 3 <= a.decode_steps <= 96
assert not a.align_steady_start or a.decode_graph
assert not a.defer_steady_gc or a.align_steady_start
rank = int(os.environ["RANK"])
faulthandler.dump_traceback_later(240, repeat=False)


def stage(name, **extra):
    print(json.dumps(dict(source=a.source, rank=rank, stage=name, **extra)), flush=True)


from livemodule.arch.ascend._native.package import activate_native_package

activate_native_package(load_extension=False)
import torch
import torch_npu

torch.set_num_threads(2)
torch.npu.set_device(0)
torch_npu.npu.config.allow_internal_format = True
activate_native_package(load_extension=True)
acl = C.CDLL("/usr/local/Ascend/cann-9.0.1/lib64/libascendcl.so")
acl.aclrtSetOpExecuteTimeOut.argtypes = [C.c_uint32]
assert acl.aclrtSetOpExecuteTimeOut(1200) == 0
from livemodule.arch.ascend.process_context import install_ascend_process_context
from livemodule.arch.ascend.vllm.moe_runtime.platform import AscendDeviceType

install_ascend_process_context(device_type=AscendDeviceType.A2, local_rank=0)
torch.distributed.init_process_group(
    "hccl",
    init_method="env://",
    rank=rank,
    world_size=2,
    timeout=timedelta(seconds=600),
)
probe = torch.ones(1, device="npu")
torch.distributed.all_reduce(probe)
assert float(probe.cpu()[0]) == 2
torch.distributed.broadcast(probe, src=0)
torch.npu.synchronize()
stage("hccl-ready")
from livemodule import LiveModule, live_runtime
from livemodule.llm.configuration import set_current_vllm_config
from livemodule.llm.runtime_support import set_default_torch_dtype
from livemodule.llm.loading.checkpoint import CheckpointLoader
from livemodule.llm.loading.weights import process_weights_after_loading
from livemodule.llm.forward_context import ForwardContext
from livemodule.llm.qwen35.batch import Qwen35DeviceBatchTopology
from livemodule.serve.qwen38 import Qwen38ServingSession
from attention import AttentionRoot
from model_setup import configure
from client import Session

cfg, runtime = configure(rank, stage)
with (
    live_runtime(runtime),
    set_current_vllm_config(cfg),
    set_default_torch_dtype(torch.bfloat16),
    torch.inference_mode(),
    cfg.device_config.device,
):
    stage("construct")
    root = AttentionRoot(vllm_config=cfg)
    assert not any(".experts." in n for n, _ in root.named_parameters())
    started = time.monotonic()
    CheckpointLoader(cfg.load_config).load_weights(root, cfg.model_config)
    process_weights_after_loading(root)
    for module in root.modules():
        if isinstance(module, LiveModule):
            module._process_live_weights_after_loading(torch.bfloat16)
    root.eval()
    root.activate()
    root.request_state.accepted_tokens.tensor.fill_(1)
    stage(
        "loaded",
        seconds=time.monotonic() - started,
        allocated=torch.npu.memory_allocated(),
        reserved=torch.npu.memory_reserved(),
    )
    if not a.construct_only:
        if rank == 0:
            cfg.remote_expert_transport = Session(a.directory, a.build, source=a.source)
        torch.distributed.barrier()
        stage("transport-ready")
        ple = Qwen38ServingSession(root)
        mailbox = ple.start_ple(token_lanes=32, max_polls=10000000)[
            root.contract.ple_layer_indices[0]
        ]
        stage("ple-ready")
        hooks = [
            layer.register_forward_pre_hook(
                lambda module, inputs, i=i: stage("layer", layer=i)
            )
            for i, layer in enumerate(root.model.language_model.layers)
        ]
        codec = mailbox.codec
        generation = torch.zeros((), dtype=torch.int64, device="npu")
        response = torch.zeros(codec.response_bytes, dtype=torch.uint8, device="npu")
        status = torch.zeros((), dtype=torch.int32, device="npu")
        slots = torch.zeros(codec.lanes, dtype=torch.int64, device="npu")
        identities = torch.ones_like(slots)
        blocks = torch.arange(2, dtype=torch.int32, device="npu")[None]
        # Identical short token sequence on both attention ranks; tokenizer
        # prompt/quality sampling is a later gate after all-layer transport.
        ids = [9707, 11, 1879]
        generated = []
        wave_seconds = []
        wave_started_seconds = []
        pause_events = []
        gc_was_enabled = gc.isenabled()
        wave_phases = []

        def observe_gc(phase, info):
            pause_events.append(
                dict(
                    kind="gc",
                    phase=phase,
                    time=time.monotonic(),
                    thread=threading.get_ident(),
                    main_thread=threading.get_ident() == threading.main_thread().ident,
                    **info,
                )
            )

        if a.observe_pauses:
            gc.callbacks.append(observe_gc)
        decode_graph = None
        decode_inputs = None
        graph_shadow_error = None
        for wave in range(1 + a.decode_steps):
            if a.observe_pauses:
                phases = {"wave": wave, "begin_prepare": time.monotonic()}
                wave_phases.append(phases)
            count = len(ids)
            start = 0 if wave == 0 else 3 + wave - 1
            tokens = torch.tensor([ids], dtype=torch.long, device="npu")
            positions = torch.arange(start, start + count, device="npu")[None]
            context = ForwardContext({}, {}, {})
            context.batch_topology = Qwen35DeviceBatchTopology(
                sequence_lengths=torch.tensor(
                    [start + count], dtype=torch.int32, device="npu"
                ),
                query_lengths=torch.tensor([count], dtype=torch.int32, device="npu"),
                active=torch.ones(1, dtype=torch.bool, device="npu"),
                block_table=blocks,
                query_length=count,
                fresh_prefill=wave == 0,
            )
            cfg.remote_expert_priority = int(wave == 0)

            def forward():
                generation.add_(1)
                status.zero_()
                with context.activate():
                    return root.forward_request_owned_continuous_ple(
                        tokens,
                        positions=positions,
                        mailbox=mailbox,
                        generation=generation,
                        response_payload=response,
                        status=status,
                        slot_ids=slots,
                        request_generations=identities,
                        destination_generations=identities,
                    )

            if a.align_steady_start and wave == 2:
                # A single measurement-boundary rendezvous removes cold loading /
                # capture skew. No per-layer or per-step batching barrier follows.
                torch.npu.synchronize()
                if a.defer_steady_gc:
                    # Causal timing control only: bounded <=96-step window.
                    # Refcounts remain active; cyclic GC is restored at teardown.
                    gc.collect()
                    gc.disable()
                (a.directory / f"steady-ready-{2 * a.source + rank}").touch()
                deadline = time.monotonic() + 180
                while not all(
                    (a.directory / f"steady-ready-{i}").exists()
                    for i in range(2 * a.sources)
                ):
                    if time.monotonic() > deadline:
                        raise TimeoutError("other attention source did not warm up")
                    time.sleep(0.001)
            wave_started = time.monotonic()
            wave_started_seconds.append(wave_started)
            if a.decode_graph and wave == 1:
                # Preserve the exact post-prefill State for eager-vs-replay.
                # Full copies are a correctness-fixture cost, never serving work.
                states = [
                    (state.tensor, state.tensor.clone())
                    for _, state in root.named_states()
                ]
                eager_hidden, _, _, eager_valid = forward()
                expected = eager_hidden.clone()
                assert bool(eager_valid[:count].all().cpu())
                for state, saved in states:
                    state.copy_(saved)
                capture_stream = torch.npu.Stream()
                capture_stream.wait_stream(torch.npu.current_stream())
                decode_graph = torch.npu.NPUGraph()
                with torch.npu.stream(capture_stream):
                    with torch.npu.graph(decode_graph):
                        graph_output = forward()
                capture_stream.synchronize()
                for state, saved in states:
                    state.copy_(saved)
                decode_inputs = (
                    tokens,
                    positions,
                    context,  # Own every external metadata tensor through graph.reset().
                )
                decode_graph.replay()
                hidden, _, _, valid = graph_output
                torch.npu.synchronize()
                graph_shadow_error = float(
                    (hidden.float() - expected.float()).norm()
                    / expected.float().norm().clamp_min(1e-9)
                )
                assert graph_shadow_error < 0.001, graph_shadow_error
                del states, expected
            elif a.decode_graph and wave > 1:
                decode_inputs[0].copy_(tokens)
                decode_inputs[1].copy_(positions)
                decode_inputs[2].batch_topology.sequence_lengths.copy_(
                    context.batch_topology.sequence_lengths
                )
                decode_graph.replay()
                hidden, _, _, valid = graph_output
            else:
                hidden, _, _, valid = forward()
            if a.observe_pauses:
                phases["submitted"] = time.monotonic()
            assert bool(valid[:count].all().cpu())
            if a.observe_pauses:
                phases["valid_readback"] = time.monotonic()
            logits = root.compute_logits(hidden[:, -1:])
            assert not bool(torch.isnan(logits).any().cpu())
            if a.observe_pauses:
                phases["logits_readback"] = time.monotonic()
            token = logits.argmax(-1).reshape(1)
            agreed = [torch.empty_like(token) for _ in range(2)]
            torch.distributed.all_gather(agreed, token)
            assert torch.equal(agreed[0], agreed[1])
            ids = [int(token.cpu()[0])]
            if a.observe_pauses:
                phases["token_agreement"] = time.monotonic()
            generated.extend(ids)
            wave_seconds.append(time.monotonic() - wave_started)
            stage("wave", wave=wave, token=ids[0], seconds=wave_seconds[-1])
            if wave == 0:
                for hook in hooks:
                    hook.remove()
        if a.defer_steady_gc:
            if gc_was_enabled:
                gc.enable()
            gc.collect()
        if a.observe_pauses:
            gc.callbacks.remove(observe_gc)
        if decode_graph is not None:
            decode_graph.reset()
        ple.close()
        torch.distributed.barrier()
        count = cfg.remote_expert_transport.close() if rank == 0 else None
        (a.directory / f"attention{2 * a.source + rank}.json").write_text(
            json.dumps(
                dict(
                    status="PASS",
                    scope="full48 target, real PLE; not quality",
                    output_ids=generated,
                    source=a.source,
                    wave_seconds=wave_seconds,
                    wave_started_seconds=wave_started_seconds,
                    decode_steps=a.decode_steps,
                    aligned_steady_start=a.align_steady_start,
                    pause_events=pause_events,
                    deferred_steady_gc=a.defer_steady_gc,
                    wave_phases=wave_phases if a.observe_pauses else [],
                    timing_scope="host wall time; wave1 includes capture and shadow when enabled",
                    calls=count,
                    decode_graph=bool(a.decode_graph),
                    graph_shadow_relative_l2=graph_shadow_error,
                    repaired_ple_metadata=root.repaired_ple_metadata,
                ),
                indent=2,
            )
        )
    root.close() if hasattr(root, "close") else None
torch.distributed.destroy_process_group()
faulthandler.cancel_dump_traceback_later()
stage("done")
