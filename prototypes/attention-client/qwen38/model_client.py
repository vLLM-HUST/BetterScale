"""TP2 target-only full-root gate with real PLE and external routed experts.

Correctness/continuation first. This is not a throughput benchmark or a quality
suite. Decode graph qualification follows the eager all-layer gate.
"""

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--directory", type=Path, required=True)
p.add_argument("--build", type=Path, required=True)
p.add_argument("--construct-only", action="store_true")
a = p.parse_args()
rank = int(os.environ["RANK"])


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
        ple = Qwen38ServingSession(root)
        mailbox = ple.start_ple(token_lanes=32, max_polls=10000000)[
            root.contract.ple_layer_indices[0]
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
            generation.add_(1)
            status.zero_()
            with context.activate():
                hidden, _, _, valid = root.forward_request_owned_continuous_ple(
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
        ple.close()
        torch.distributed.barrier()
        count = cfg.remote_expert_transport.close() if rank == 0 else None
        (a.directory / f"attention{rank}.json").write_text(
            json.dumps(
                dict(
                    status="PASS",
                    scope="eager full48 target, real PLE, four waves; not quality",
                    output_ids=generated,
                    calls=count,
                ),
                indent=2,
            )
        )
    root.close() if hasattr(root, "close") else None
torch.distributed.destroy_process_group()
stage("done")
