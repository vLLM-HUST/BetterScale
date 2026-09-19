"""Two native offline clients keep EP peers alive; fixed balanced prefills."""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path


def rank_main(rank, root, barrier):
    os.environ.update(
        VLLM_DP_RANK=str(rank),
        VLLM_DP_RANK_LOCAL=str(rank),
        VLLM_DP_SIZE="2",
        VLLM_DP_MASTER_IP="127.0.0.1",
        VLLM_DP_MASTER_PORT="31641",
        VLLM_ASCEND_ENABLE_FLASHCOMM1="0",
        SHARED_EP_OUTPUT=str(root),
    )
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="/data/shared_models/Qwen3-30B-A3B",
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        dtype="bfloat16",
        distributed_executor_backend="mp",
        worker_cls="worker.ObserveWorker",
        max_model_len=8192,
        max_num_batched_tokens=4096,
        max_num_seqs=1,
        kv_cache_memory_bytes=2 * 1024**3,
        enable_prefix_caching=False,
        async_scheduling=False,
        skip_tokenizer_init=True,
        seed=123,
        compilation_config=dict(
            cudagraph_mode="FULL",
            cudagraph_capture_sizes=[1, 16, 256, 1024, 4096],
            max_cudagraph_capture_size=4096,
        ),
        additional_config=dict(
            ascend_compilation_config=dict(
                enable_npugraph_ex=True, enable_static_kernel=False
            ),
            enable_cpu_binding=False,
        ),
    )
    import time
    engine = llm.llm_engine
    assert engine.dp_group is not None, "Fixture requires synchronous offline DP"
    barrier.wait(timeout=600)
    for label, size, profile in [("decode-control",0,False), ("p256",256,False),
                                  ("p1024",1024,False), ("p4096",4096,False),
                                  ("p4096-profile",4096,True)]:
        barrier.wait(timeout=600)
        llm.collective_rpc("start_window", args=(label,profile))
        barrier.wait(timeout=600)
        def add(rid, tokens, count):
            engine.add_request(rid,dict(prompt_token_ids=tokens),
                SamplingParams(temperature=0,max_tokens=count,ignore_eos=True,detokenize=False))
        if rank == 0:
            add(label+"-decode",[17]*16,24)
        else:
            add(label+"-seed",[18]*16,2)
        step = 0
        events = []
        while True:
            barrier.wait(timeout=120)
            if step == 4 and rank == 1 and size:
                add(label+"-prefill",[19]*size,2)
            if not engine.has_unfinished_requests():
                break
            outputs = engine.step()
            for out in outputs:
                events.append(dict(step=step,time_ns=time.perf_counter_ns(),request_id=out.request_id,
                                   finished=out.finished, tokens=list(out.outputs[0].token_ids)))
            step += 1
            assert step <= 128, "unexpected drain"
        barrier.wait(timeout=600)
        llm.collective_rpc("stop_window")
        dest=root/label
        dest.mkdir(exist_ok=True)
        (dest/f"outputs-rank{rank}.json").write_text(json.dumps(events,indent=2))
        finished=[e for e in events if e['finished']]
        assert len(finished) == (1 if rank == 0 or not size else 2), finished
        assert all(len(e['tokens']) == (24 if rank == 0 else 2) for e in finished)
        print("WINDOW_DONE",rank,label,step,flush=True)
    barrier.wait(timeout=600)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    processes = [
        ctx.Process(
            target=rank_main, args=(i, a.out, barrier), name=f"qwen-dma-rank{i}"
        )
        for i in range(2)
    ]
    for process in processes:
        process.start()
    while any(p.is_alive() for p in processes):
        if any(p.exitcode not in (None, 0) for p in processes):
            barrier.abort()
            raise RuntimeError(f"DP client failed: {[p.exitcode for p in processes]}")
        for process in processes:
            process.join(timeout=0.5)
    assert all(p.exitcode == 0 for p in processes)
    (a.out / "complete.json").write_text(
        json.dumps(dict(exitcodes=[p.exitcode for p in processes]))
    )


if __name__ == "__main__":
    main()
