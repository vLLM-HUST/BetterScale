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
        VLLM_DP_MASTER_PORT="31541",
        VLLM_ASCEND_ENABLE_FLASHCOMM1="0",
        DMA_PROBE_OUTPUT=str(root),
    )
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="/data/shared_models/Qwen3-30B-A3B",
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        dtype="bfloat16",
        distributed_executor_backend="mp",
        worker_cls="worker.DMAWorker",
        max_model_len=8192,
        max_num_batched_tokens=4096,
        max_num_seqs=1,
        kv_cache_memory_bytes=2 * 1024**3,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        seed=123,
        compilation_config=dict(
            cudagraph_mode="FULL",
            cudagraph_capture_sizes=[1, 4096],
            max_cudagraph_capture_size=4096,
        ),
        additional_config=dict(
            ascend_compilation_config=dict(
                enable_npugraph_ex=True, enable_static_kernel=False
            ),
            enable_cpu_binding=False,
        ),
    )
    barrier.wait(timeout=600)
    llm.collective_rpc("arm_dma_probe")
    barrier.wait(timeout=120)
    outputs = llm.generate(
        [dict(prompt_token_ids=[17 + rank] * 4096)],
        SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, detokenize=False),
    )
    (root / f"completion-rank{rank}.json").write_text(
        json.dumps([o.outputs[0].token_ids for o in outputs])
    )
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
