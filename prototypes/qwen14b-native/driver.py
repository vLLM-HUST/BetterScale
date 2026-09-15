"""Native single-card BF16 offline cohorts; no BetterScale execution patches."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from vllm import LLM, SamplingParams

    config = dict(
        model="/data/shared_models/Qwen2.5-14B-Instruct",
        tensor_parallel_size=1,
        dtype="bfloat16",
        distributed_executor_backend="mp",
        worker_extension_cls="observe.Observer",
        max_model_len=8192,
        max_num_batched_tokens=4096,
        max_num_seqs=8,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        async_scheduling=True,
        seed=123,
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    llm = LLM(**config)
    results = []
    for label, lengths in [("c1-4k", [4096]), ("c8-512", [512] * 8)]:
        prompts = [dict(prompt_token_ids=[100 + i] * n) for i, n in enumerate(lengths)]
        # Warm actual execution before collecting; APC is off so requests remain cold.
        llm.generate(prompts, SamplingParams(temperature=0, max_tokens=8,
                     ignore_eos=True, detokenize=False), use_tqdm=False)
        for repeat in range(2):
            start = time.monotonic()
            outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=32,
                                   ignore_eos=True, detokenize=False), use_tqdm=False)
            elapsed = time.monotonic() - start
            assert all(len(o.outputs[0].token_ids) == 32 for o in outputs)
            results.append(dict(label=label, repeat=repeat, elapsed_s=elapsed,
                                output_tokens=32 * len(outputs)))
            (args.output / "unprofiled.json").write_text(json.dumps(results, indent=2))
        window = args.output / label
        window.mkdir()
        manifest = llm.collective_rpc("start_observation", args=(str(window / "profile"),))
        (window / "manifest.json").write_text(json.dumps(manifest, indent=2))
        outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=8,
                               ignore_eos=True, detokenize=False), use_tqdm=False)
        llm.collective_rpc("stop_observation")
        assert all(len(o.outputs[0].token_ids) == 8 for o in outputs)
        (window / "complete.json").write_text(json.dumps(dict(requests=len(outputs),
                                            input_lengths=lengths, output_each=8)))
    (args.output / "complete.json").write_text(json.dumps(dict(status="PASS")))


if __name__ == "__main__":
    main()
