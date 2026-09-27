"""Native AsyncScheduler/EngineCore/Worker TP2 gate; lease required before imports."""

import asyncio
import json
import os
from pathlib import Path

from vllm.platforms import current_platform

# Match native CLI bootstrap before importing model/Worker classes.
current_platform.pre_register_and_update()
from betterscale import qwen35_worker  # qualified capture policy before config
from betterscale.models.qwen35 import CAPTURE_SIZES, STATE_SCHEDULER
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

if os.environ.get("CACHE_BYTE_AUDIT") == "1":
    from native_audit import install

    install()


async def main():
    run = Path(os.environ["CAPSULE"])
    model = "/workspace/my-ascend-workspace/runs/qwen35-moe-mtp-256k/model"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

    def prompt(text):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            enable_thinking=False,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )

    seed = prompt(
        "Remember the access code amber-7319. "
        + "An ordinary archive record. " * 24
        + " Reply with exactly amber-7319."
    )
    args = AsyncEngineArgs(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        worker_cls="betterscale.qwen35_worker.Worker",
        max_model_len=8192,
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        kv_cache_memory_bytes=6 << 30,
        seed=17,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        async_scheduling=True,
        scheduler_cls=STATE_SCHEDULER,
        additional_config={
            "enable_cpu_binding": False,
            "using_live_runtime": True,
            "state_cache_host_bytes": 512 << 20,
        },
        speculative_config={"method": "mtp", "num_speculative_tokens": 2},
        compilation_config={
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": list(CAPTURE_SIZES),
            "max_cudagraph_capture_size": 4096,
        },
        limit_mm_per_prompt={"image": 0, "video": 0},
        generation_config="vllm",
    )
    engine = AsyncLLM.from_engine_args(args)
    results = {"passed": False, "route": "native35B TP2 MTP2 AsyncScheduler"}

    async def control(**command):
        return await asyncio.wait_for(
            engine.engine_core.call_utility_async("state_cache", command), 90
        )

    async def generate(name, tokens, count, salt):
        output = None
        async for output in engine.generate(
            {"prompt_token_ids": tokens, "cache_salt": salt},
            SamplingParams(temperature=0, max_tokens=count, ignore_eos=True),
            request_id=name,
        ):
            pass
        return dict(ids=output.outputs[0].token_ids, cached=output.num_cached_tokens)

    async def settled(salt=None):
        for _ in range(100):
            state = await control(kind="snapshot")
            matches = [
                s for s in state["seats"] if s["cache_salt"] == salt and s["cursor"]
            ]
            if matches or (salt is None and all(s["cursor"] for s in state["seats"])):
                return state, matches[0] if matches else None
            await asyncio.sleep(0.02)
        raise AssertionError(("resident did not reach its final-writer fence", state))

    try:
        initial = await control(kind="snapshot")
        assert initial["native_async"] and initial["batch_queue_size"] > 1
        assert initial["step_fn"] == "step_with_batch_queue"
        # Fill the resident arena so the stored seat is the ONLY empty seat;
        # its physical GDN rows must then be reused by an unrelated request.
        await asyncio.gather(
            *(generate("fill-" + str(i), seed, 4, "fill-" + str(i)) for i in range(16))
        )
        await asyncio.gather(
            *(
                generate("fill-" + str(i), seed, 4, "fill-" + str(i))
                for i in range(16, 20)
            )
        )
        await settled()
        twin = await generate("A-twin", seed, 32, "A-twin")
        a = await generate("A", seed, 32, "A")
        assert twin["ids"] == a["ids"], (twin, a)
        _, source = await settled("A")
        b = asyncio.create_task(
            generate("B", prompt("Count from one to one hundred."), 128, "B")
        )
        await asyncio.sleep(0.01)
        operation = await control(kind="store", seat=source["seat"], key="A-v1")
        store = await control(kind="wait", operation=operation)
        assert store["ranks"] == [0, 1]
        results["store_finished_before_other_request"] = not b.done()
        await b
        await generate(
            "C", prompt("The other access code is violet-8921. Repeat it."), 8, "C"
        )
        _, c = await settled("C")
        assert c["seat"] == source["seat"]
        before = await control(kind="snapshot")
        destination = (source["seat"] + 2) % 20
        operation = await control(kind="load", key="A-v1", seat=destination)
        load = await control(kind="wait", operation=operation)
        after = await control(kind="snapshot")
        assert after["model_steps"] == before["model_steps"]
        restored = after["seats"][destination]
        assert (
            restored["cursor"] == source["cursor"]
            and restored["blocks"] != source["blocks"]
        )
        continuation = (
            seed
            + a["ids"]
            + tokenizer.encode(
                "\n"
                + "Keep the archive in mind. " * 80
                + " What is the original access code?"
            )
        )
        hot = await generate("A-unmoved", continuation, 32, "A-twin")
        warm = await generate("A-restored", continuation, 32, "A")
        cold = await generate("A-cold", continuation, 32, "independent-cold")
        assert warm["cached"] == len(seed) + len(a["ids"]) - 1, (source, warm)
        results.update(
            source=source,
            restored=restored,
            store=store,
            load=load,
            hot=hot,
            warm=warm,
            cold=cold,
            initial=initial,
        )
        assert hot["cached"] == warm["cached"]
        assert hot["ids"] == warm["ids"], ("transport", hot, warm)
        assert cold["cached"] == 0 and warm["ids"] == cold["ids"], (
            "cold parity",
            hot,
            warm,
            cold,
        )
        operation = await control(kind="drop", key="A-v1")
        await control(kind="wait", operation=operation)
        final = await control(kind="snapshot")
        assert not final["host"] and not final["pending"]
        results.update(
            passed=True,
            initial=initial,
            source=source,
            overwritten_by=c,
            restored=restored,
            store=store,
            load=load,
            warm=warm,
            cold=cold,
            maintenance_only_restore=True,
            final=final,
        )
        print("NATIVE_ASYNC_TP2_CACHE_PASS", json.dumps(results), flush=True)
    finally:
        engine.shutdown()
        (run / "result.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
