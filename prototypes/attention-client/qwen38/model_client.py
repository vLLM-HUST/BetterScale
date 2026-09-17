"""TP2 target-only full-root gate with real PLE and external routed experts.

Correctness/continuation first. This is not a throughput benchmark or a quality
suite. Decode graph qualification follows the eager all-layer gate.
"""

import argparse
import ctypes as C
import faulthandler
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
a = p.parse_args()
rank = int(os.environ["RANK"])
faulthandler.dump_traceback_later(240, repeat=False)


def stage(name, **extra):
    print(json.dumps(dict(rank=rank, stage=name, **extra)), flush=True)


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
            cfg.remote_expert_transport = Session(a.directory, a.build)
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
        decode_graph = None
        decode_inputs = None
        graph_shadow_error = None
        for wave in range(4):
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
            assert bool(valid[:count].all().cpu())
            logits = root.compute_logits(hidden[:, -1:])
            assert not bool(torch.isnan(logits).any().cpu())
            token = logits.argmax(-1).reshape(1)
            agreed = [torch.empty_like(token) for _ in range(2)]
            torch.distributed.all_gather(agreed, token)
            assert torch.equal(agreed[0], agreed[1])
            ids = [int(token.cpu()[0])]
            generated.extend(ids)
            stage("wave", wave=wave, token=ids[0])
            if wave == 0:
                for hook in hooks:
                    hook.remove()
        if decode_graph is not None:
            decode_graph.reset()
        ple.close()
        torch.distributed.barrier()
        count = cfg.remote_expert_transport.close() if rank == 0 else None
        (a.directory / f"attention{rank}.json").write_text(
            json.dumps(
                dict(
                    status="PASS",
                    scope="full48 target, real PLE, four waves; not quality",
                    output_ids=generated,
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
