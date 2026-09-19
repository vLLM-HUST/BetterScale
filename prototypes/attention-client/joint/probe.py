"""Four physical cards, isolated native TP1 attention engines, IPC EP2."""

import argparse
import json
import multiprocessing as mp
import os
import queue
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]


def child(rank, device, links, out, ready_queue, start_event):
    # Set visibility before importing any accelerator modules. Every child owns
    # logical device0; rank here denotes the service role, not HCCL world rank.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Each independent TP1 engine allocates additional rendezvous ports.
    # Adjacent base ports can collide even though the devices are disjoint.
    port_base = int(os.environ.get("ATTENTION_JOINT_PORT_BASE", "41500")) + 100 * rank
    os.environ["MASTER_PORT"] = str(port_base + 40)
    os.environ["VLLM_PORT"] = str(port_base + 50)
    import torch
    import torch_npu

    torch.set_num_threads(2)
    torch.npu.set_device(0)
    if rank >= 2:
        import importlib

        serve = importlib.import_module(
            os.environ.get("ATTENTION_JOINT_SERVER_MODULE", "server")
        ).serve

        serve(rank - 2, links, str(Path(out, f"expert{rank-2}.json")))
    else:
        import common

        common.LINKS = links
        common.OUTPUT = str(Path(out, f"attention{rank}.json"))
        from vllm import LLM, SamplingParams

        llm = LLM(
            model="/data/shared_models/Qwen3-30B-A3B",
            hf_overrides=dict(num_hidden_layers=2),
            load_format="dummy",
            dtype="bfloat16",
            tensor_parallel_size=1,
            distributed_executor_backend="uni",
            worker_cls=os.environ.get(
                "ATTENTION_JOINT_WORKER", "joint_worker.JointWorker"
            ),
            enforce_eager=True,
            max_model_len=256,
            max_num_batched_tokens=32,
            max_num_seqs=1,
            kv_cache_memory_bytes=128 * 1024**2,
            enable_prefix_caching=False,
            skip_tokenizer_init=True,
            async_scheduling=False,
            additional_config=dict(enable_cpu_binding=False),
        )
        llm.collective_rpc("attach")
        profiler = None
        if os.environ.get("DEVICE_SERVICE_PROFILE"):
            from profile_capture import start, stop

            profiler = start(f"attention{rank}")
        output_tokens = int(os.environ.get("ATTENTION_JOINT_OUTPUT_TOKENS", "3"))
        assert 1 <= output_tokens <= 4
        if os.environ.get("ATTENTION_JOINT_SERVER_MODULE") == "device_joint":
            assert (
                int(os.environ.get("DEVICE_SERVICE_TASKS", "24")) == 8 * output_tokens
            )
        params = SamplingParams(
            temperature=0, max_tokens=output_tokens, ignore_eos=True, detokenize=False
        )
        for size in (16, 32):
            llm.generate([dict(prompt_token_ids=[17 + rank] * size)], params)
        banks = llm.collective_rpc("seal")
        # Episode-start gate only: do not let the earlier-loading engine finish
        # every measured request before the second client finishes warmup.
        ready_queue.put(rank)
        if not start_event.wait(900):
            raise TimeoutError("joint episode startup gate")
        generations = []
        for size in ((32, 16) if rank == 0 else (16, 32)):
            begin = time.monotonic()
            results = llm.generate([dict(prompt_token_ids=[19 + rank] * size)], params)
            generations.append(
                dict(
                    input_tokens=size,
                    token_ids=results[0].outputs[0].token_ids,
                    wall_seconds=time.monotonic() - begin,
                )
            )
            assert len(results[0].outputs[0].token_ids) == output_tokens
        llm.collective_rpc("detach")
        if profiler is not None:
            stop(profiler)
        Path(out, f"attention{rank}-complete.json").write_text(
            json.dumps(
                dict(
                    banks=banks,
                    complete=True,
                    generations=generations,
                    shadow=os.environ.get("ATTENTION_JOINT_SHADOW", "1") == "1",
                )
            )
        )
    torch.npu.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    devices = [int(x) for x in args.devices.split(",")]
    assert len(devices) == 4 and len(set(devices)) == 4
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    links = [[], [], [], []]
    for client in range(2):
        for server in range(2):
            a, b = ctx.Pipe()
            links[client].append(a)
            links[2 + server].append(b)
    processes = []
    ready_queue = ctx.Queue()
    start_event = ctx.Event()
    ready_clients = set()
    try:
        for rank, device in enumerate(devices):
            p = ctx.Process(
                target=child,
                args=(rank, device, links[rank], args.out, ready_queue, start_event),
                name=f"neural-role-{rank}",
            )
            p.start()
            processes.append(p)
        for group in links:
            for pipe in group:
                pipe.close()
        deadline = time.monotonic() + 1200
        while any(p.is_alive() for p in processes):
            try:
                while True:
                    ready_clients.add(ready_queue.get_nowait())
            except queue.Empty:
                pass
            if ready_clients == {0, 1}:
                start_event.set()
            for p in processes:
                if p.exitcode not in (None, 0):
                    raise RuntimeError(f"{p.name} exited {p.exitcode}")
            if time.monotonic() > deadline:
                raise TimeoutError("four-card neural closure exceeded1200s")
            time.sleep(0.2)
        assert all(p.exitcode == 0 for p in processes)
        Path(args.out, "result.json").write_text(
            json.dumps(
                dict(
                    status="pass",
                    attention=2,
                    experts=2,
                    model="Qwen3-30B-A3B",
                    layers=2,
                    full_layer_dimensions=True,
                    dummy_weights=True,
                    host_control=not bool(
                        os.environ.get("ATTENTION_JOINT_SERVER_MODULE")
                    ),
                    performance_claim=False,
                ),
                indent=2,
            )
        )
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(10)


if __name__ == "__main__":
    main()
