"""Balanced native DP clients: every EP partner enters the same six-wave oracle."""
import json
import multiprocessing as mp
import os
from pathlib import Path


def client(rank, barrier):
    dp, tp = int(os.environ['QWEN_DP']), int(os.environ['QWEN_TP'])
    os.environ.update(VLLM_DP_RANK=str(rank), VLLM_DP_RANK_LOCAL=str(rank),
                      VLLM_DP_SIZE=str(dp), VLLM_DP_MASTER_IP='127.0.0.1',
                      VLLM_DP_MASTER_PORT='31861')
    from vllm import LLM, SamplingParams
    root = Path(os.environ['QWEN_JOINT_OUTPUT'])
    config = dict(model='/data/shared_models/Qwen3-30B-A3B', tensor_parallel_size=tp,
                  enable_expert_parallel=True, dtype='bfloat16',
                  distributed_executor_backend='mp', worker_cls='worker.JointWorker',
                  max_model_len=256, max_num_batched_tokens=64, max_num_seqs=1,
                  kv_cache_memory_bytes=256 * 1024**2, enable_prefix_caching=False,
                  async_scheduling=False, skip_tokenizer_init=True, seed=123,
                  compilation_config=dict(cudagraph_mode='FULL_DECODE_ONLY',
                                          cudagraph_capture_sizes=[1], max_cudagraph_capture_size=1),
                  additional_config=dict(enable_cpu_binding=False, pa_shape_list=[1],
                      ascend_compilation_config=dict(enable_npugraph_ex=True, enable_static_kernel=False)))
    if os.environ.get('QWEN_DUMMY') == '1':
        config.update(load_format='dummy', hf_overrides=dict(num_hidden_layers=2))
    (root/f'config-dp{rank}.json').write_text(json.dumps(config, indent=2))
    llm = LLM(**config)
    barrier.wait(timeout=600)
    llm.collective_rpc('enable_joint_oracle')
    barrier.wait(timeout=600)
    # Local tokenizer-verified IDs; optional varied-token control for the
    # combined TP/DP matrix, without loading a tokenizer in each engine.
    prompts = [
        ('Write the numbers from one to ten in English, separated by commas. Answer: one,',
         [7985,279,5109,504,825,311,5779,304,6364,11,18663,553,76602,13,21806,25,825,11]),
        ('Complete the sequence of days of the week in English, separated by commas. Answer: Monday,',
         [12548,279,8500,315,2849,315,279,2003,304,6364,11,18663,553,76602,13,21806,25,7014,11]),
    ]
    if os.environ.get('QWEN_PROMPTS') == 'language':
        prompt_text, prompt_ids = prompts[rank % len(prompts)]
    else:
        prompt_text, prompt_ids = None, [17+rank]*32
    (root/f'prompt-dp{rank}.json').write_text(json.dumps(dict(text=prompt_text, ids=prompt_ids)))
    result = llm.generate([dict(prompt_token_ids=prompt_ids)],
                          SamplingParams(temperature=0, max_tokens=12, ignore_eos=True, detokenize=False))
    barrier.wait(timeout=600)
    receipts = llm.collective_rpc('joint_receipt')
    (root/f'result-dp{rank}.json').write_text(json.dumps(dict(status='PASS', receipts=receipts,
        outputs=[list(x.outputs[0].token_ids) for x in result]), indent=2))
    barrier.wait(timeout=600)


if __name__ == '__main__':
    dp = int(os.environ['QWEN_DP'])
    ctx = mp.get_context('spawn')
    barrier = ctx.Barrier(dp)
    jobs = [ctx.Process(target=client, args=(rank, barrier), name=f'qwen-joint-dp{rank}')
            for rank in range(dp)]
    for job in jobs:
        job.start()
    while any(job.is_alive() for job in jobs):
        if any(job.exitcode not in (None, 0) for job in jobs):
            barrier.abort()
            raise RuntimeError(f'DP client failed: {[j.exitcode for j in jobs]}')
        for job in jobs:
            job.join(timeout=0.5)
    assert all(job.exitcode == 0 for job in jobs)
    root = Path(os.environ['QWEN_JOINT_OUTPUT'])
    (root/'complete.json').write_text(json.dumps(dict(status='PASS', exitcodes=[j.exitcode for j in jobs])))
