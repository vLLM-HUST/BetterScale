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
    os.environ["MASTER_PORT"] = str(31540 + rank)
    os.environ["VLLM_PORT"] = str(31550 + rank)
    import torch
    import torch_npu

    torch.set_num_threads(2)
    torch.npu.set_device(0)
    if rank >= 2:
        from server import serve

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
            worker_cls="joint_worker.JointWorker",
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
        params = SamplingParams(
            temperature=0, max_tokens=3, ignore_eos=True, detokenize=False
        )
        for size in (16, 32):
            llm.generate([dict(prompt_token_ids=[17 + rank] * size)], params)
        banks = llm.collective_rpc("seal")
        # Episode-start gate only: do not let the earlier-loading engine finish
        # every measured request before the second client finishes warmup.
        ready_queue.put(rank)
        if not start_event.wait(900):
            raise TimeoutError("joint episode startup gate")
        for size in ((32, 16) if rank == 0 else (16, 32)):
            results = llm.generate([dict(prompt_token_ids=[19 + rank] * size)], params)
            assert len(results[0].outputs[0].token_ids) == 3
        llm.collective_rpc("detach")
        Path(out, f"attention{rank}-complete.json").write_text(
            json.dumps(dict(banks=banks, complete=True))
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
                    host_control=True,
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
