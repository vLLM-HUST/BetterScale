"""Independent native runner reference; must use the frozen baseline environment."""

import json
import os
from pathlib import Path


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model = "/workspace/models/Qwen3.5-35B-A3B"
    llm = LLM(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        worker_cls="native_worker.Worker",
        max_model_len=4096,
        max_num_seqs=2,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=2 << 30,
        enforce_eager=True,
        enable_prefix_caching=False,
        seed=17,
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={"enable_cpu_binding": False},
        speculative_config={"method": "mtp", "num_speculative_tokens": 2},
    )

    def run(tokens, count, ignore_eos):
        result = llm.generate(
            [{"prompt_token_ids": tokens}],
            SamplingParams(
                temperature=0, max_tokens=count, ignore_eos=ignore_eos, logprobs=5
            ),
            use_tqdm=False,
        )[0].outputs[0]
        return {
            "prompt": tokens,
            "token_ids": list(result.token_ids),
            "text": result.text,
            "logprobs": [
                {str(k): v.logprob for k, v in row.items()} for row in result.logprobs
            ],
        }

    tokens = [9707, 11, 358, 1079, 264, 1786, 13]
    base = run(tokens, 12, True)
    continuation = run(tokens + base["token_ids"] + [271], 8, True)
    Path(os.environ["CAPSULE"], "raw-reference.json").write_text(
        json.dumps({"base": base, "continuation": continuation}, indent=2) + "\n"
    )
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    chats = []
    for question in [
        "What is 2 + 2? Answer with only the number.",
        "Repeat exactly this code and nothing else: ORCHID-7319",
    ]:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        chats.append({"question": question, **run(prompt, 16, False)})
    receipt = {
        "base": base,
        "continuation": continuation,
        "chats": chats,
        "scope": "independent pinned native TP2 BF16 MTP2 eager runner, raw fixtures ignore EOS; chat honors EOS",
    }
    Path(os.environ["CAPSULE"], "reference.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in receipt.items() if k != "chats"}), flush=True)
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
