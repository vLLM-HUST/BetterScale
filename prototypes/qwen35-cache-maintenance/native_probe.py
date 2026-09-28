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


if os.environ.get("CACHE_PROFILE") == "1":
    from native_profile import install, stamp

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

    incremental = os.environ.get("CACHE_INCREMENTAL") == "1"
    legacy = os.environ.get("CACHE_LEGACY_FIXTURE") == "1"
    code = "cobalt-seven-42-alpha-nine-17-zulu-eight-63-bravo-five-29-delta-six-84"
    seed = prompt(
        "Remember the access code amber-7319. "
        + "An ordinary archive record. " * 24
        + " Reply with exactly amber-7319."
        if legacy
        else "Read these records. Find the access code.\nBEGIN_RECORDS\n"
        + "The archive contains ordinary historical records. "
        * (600 if incremental else 128)
        + "\nThe access code is "
        + code
        + ".\nEND_RECORDS\n"
        + "Reply with only the access code, without explanation."
    )
    first_budget = 32 if legacy else 8
    automatic = os.environ.get("CACHE_AUTO_POLICY") == "1"
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
            "state_cache_host_bytes": (4 << 30) if automatic else (512 << 20),
            "state_cache_policy": automatic,
            "state_cache_incremental": incremental,
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
    profiling = os.environ.get("CACHE_PROFILE") == "1"
    if profiling:
        from vllm.config import ProfilerConfig

        args.profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(run / "profiles"),
            ignore_frontend=True,
            torch_profiler_with_stack=False,
        )
    engine = AsyncLLM.from_engine_args(args)
    results = {
        "passed": False,
        "route": "native35B TP2 MTP2 AsyncScheduler",
        "byte_audit": os.environ.get("CACHE_BYTE_AUDIT") == "1",
    }

    async def control(**command):
        if profiling:
            stamp("control_begin", command=command)
        value = await asyncio.wait_for(
            engine.engine_core.call_utility_async("state_cache", command), 90
        )
        if profiling:
            stamp("control_end", command=command)
        return value

    async def generate(name, tokens, count, salt, *, force=True):
        output = None
        async for output in engine.generate(
            {"prompt_token_ids": tokens, "cache_salt": salt},
            SamplingParams(temperature=0, max_tokens=count, ignore_eos=force),
            request_id=name,
        ):
            pass
        return dict(
            ids=output.outputs[0].token_ids,
            cached=output.num_cached_tokens,
            text=output.outputs[0].text,
        )

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
        if automatic:
            from native_policy_probe import exercise

            results.update(
                await exercise(control, generate, prompt, seed, tokenizer, code)
            )
            print("NATIVE_AUTO_POLICY_PASS", json.dumps(results), flush=True)
            return
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
        twin = await generate("A-twin", seed, first_budget, "A-twin")
        a = await generate("A", seed, first_budget, "A")
        assert twin["ids"] == a["ids"], (twin, a)
        _, source = await settled("A")
        if profiling:
            stamp("profile_start_begin")
            await engine.start_profile()
            stamp("profile_start_end")
        b = asyncio.create_task(
            generate(
                "B",
                prompt("Count from one to one hundred."),
                256 if profiling else 1024,
                "B",
            )
        )
        for _ in range(100):
            busy = await control(kind="snapshot")
            if any(
                s["owner"] and s["owner"].split("-", 1)[0] == "B" for s in busy["seats"]
            ):
                break
            assert not b.done(), "B finished before store was issued"
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("B did not enter native scheduling")
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
        if incremental:
            assert 0 < load["restored_pages"] < len(source["blocks"]), load
            assert all(
                value < next(iter(store["transfer_bytes_per_rank"].values()))
                for value in load["transfer_bytes_per_rank"].values()
            ), (store, load)
        if legacy:
            delta = tokenizer.encode(
                "\n"
                + "Keep the archive in mind. " * 80
                + " What is the original access code?"
            )
        else:
            template = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": "dummy"},
                    {"role": "assistant", "content": "DELTA_MARKER"},
                    {
                        "role": "user",
                        "content": "Please retain the earlier access record. " * 80
                        + "Repeat the original access code exactly. Reply only with that code, no explanation.",
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            delta = tokenizer.encode(
                template.split("DELTA_MARKER")[1], add_special_tokens=False
            )
        continuation = seed + a["ids"] + delta
        limit = 32 if legacy else 128
        hot = await generate("A-unmoved", continuation, limit, "A-twin", force=legacy)
        warm = await generate("A-restored", continuation, limit, "A", force=legacy)
        if profiling:
            stamp("profile_stop_begin")
            await engine.stop_profile()
            stamp("profile_stop_end")
        cold = await generate(
            "A-cold", continuation, limit, "independent-cold", force=legacy
        )
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
            fixture="forced-post-eos" if legacy else "chat-role-closure",
        )
        assert hot["cached"] == warm["cached"]
        assert hot["ids"] == warm["ids"], ("transport", hot, warm)
        assert cold["cached"] == 0 and warm["ids"] == cold["ids"], (
            "cold parity",
            hot,
            warm,
            cold,
        )
        if not legacy:
            assert warm["text"].strip() == code, warm
        if incremental:
            # A has advanced; shared sealed pages must not be copied/charged again.
            _, advanced = await settled("A")
            before_backup = await control(kind="snapshot")
            operation = await control(kind="store", seat=advanced["seat"], key="A-v2")
            incremental_store = await control(kind="wait", operation=operation)
            after_backup = await control(kind="snapshot")
            assert all(
                value < next(iter(store["transfer_bytes_per_rank"].values()))
                for value in incremental_store["transfer_bytes_per_rank"].values()
            )
            delta_bytes = (
                after_backup["allocated_host_bytes"]
                - before_backup["allocated_host_bytes"]
            )
            assert delta_bytes == next(
                iter(incremental_store["transfer_bytes_per_rank"].values())
            )
            results.update(
                incremental=True,
                incremental_store=incremental_store,
                incremental_host_bytes=delta_bytes,
            )
            operation = await control(kind="drop", key="A-v2")
            await control(kind="wait", operation=operation)
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
