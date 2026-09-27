"""Automatic policy gate, driven only by workload; no manual store/load/drop."""

import asyncio


async def exercise(control, generate, prompt, seed, tokenizer, code):
    async def drain():
        for _ in range(1000):
            state = await control(kind="snapshot")
            if not state["pending"]:
                await asyncio.sleep(0.05)
                other = await control(kind="snapshot")
                if not other["pending"] and other["completed"] == state["completed"]:
                    return other
            await asyncio.sleep(0.02)
        raise AssertionError(("automatic maintenance did not settle", state))

    async def churn(start, count):
        for i in range(start, start + count):
            await generate(
                "churn-" + str(i),
                prompt("Remember marker " + str(i) + ". Repeat it."),
                4,
                "churn-" + str(i),
            )
        return await drain()

    # Cross70% and let automatic backup fill the host while seats stay hot.
    filled = await churn(0, 20)
    assert filled["host"] and sum(bool(s["cursor"]) for s in filled["seats"]) == 20
    a = await generate("A", seed, 8, "A")
    twin = await generate("twin", seed, 8, "twin")
    assert a["ids"] == twin["ids"]
    stored = await drain()
    source = next(s for s in stored["seats"] if s["cache_salt"] == "A")
    checkpoint = next(e for e in stored["host_entries"] if e["cache_salt"] == "A")
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
    continuation = (
        seed
        + a["ids"]
        + tokenizer.encode(template.split("DELTA_MARKER")[1], add_special_tokens=False)
    )
    hot = await generate("unmoved", continuation, 128, "twin", force=False)
    evicted = await churn(20, 20)
    assert not any(s["cache_salt"] == "A" for s in evicted["seats"])
    assert checkpoint["key"] in evicted["host"]
    warm = await generate("restored", continuation, 128, "A", force=False)
    restored = await drain()
    cold = await generate("cold", continuation, 128, "independent-cold", force=False)
    assert hot["cached"] == warm["cached"] == len(seed) + len(a["ids"]) - 1
    assert cold["cached"] == 0 and hot["ids"] == warm["ids"] == cold["ids"]
    assert warm["text"].strip() == code
    assert any(
        x["kind"] == "load" and x["ranks"] == [0, 1] for x in restored["completed"]
    )
    # Churn both LRUs; no mandatory backup on device eviction and no over-budget host.
    final = await churn(40, 40)
    assert checkpoint["key"] not in final["host"]
    assert final["allocated_host_bytes"] <= 4 << 30
    assert any(x["kind"] == "drop" and x["ranks"] == [0, 1] for x in final["completed"])
    recomputed = await generate(
        "after-host-eviction", continuation, 128, "A", force=False
    )
    assert recomputed["cached"] == 0 and recomputed["ids"] == cold["ids"]
    return dict(
        passed=True,
        automatic=True,
        filled=filled,
        source=source,
        checkpoint=checkpoint,
        evicted=evicted,
        restored=restored,
        hot=hot,
        warm=warm,
        cold=cold,
        recomputed=recomputed,
        final=final,
    )
