"""Ingress rejects unsupported features before any distributed/model call."""

import pytest

from betterscale.live.llm.qwen35.http import request_tokens


class Tokenizer:
    def encode(self, prompt, **kwargs):
        return [1, 2]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["return_dict"] is False
        return [3, 4]


def test_request_admission_leaves_two_mtp_lookahead_tokens():
    assert request_tokens(
        {"prompt": [1, 2], "max_tokens": 4},
        Tokenizer(),
        chat=False,
        context_tokens=8,
        vocab_size=10,
    ) == ([1, 2], 4)
    with pytest.raises(ValueError, match="lookahead"):
        request_tokens(
            {"prompt": [1, 2], "max_tokens": 5},
            Tokenizer(),
            chat=False,
            context_tokens=8,
            vocab_size=10,
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"temperature": 1},
        {"stream": True},
        {"n": 2},
        {"top_p": 0.9},
        {"presence_penalty": 0.1},
        {"prompt": [-1]},
        {"max_tokens": 0},
        {"prompt": [True]},
        {"ignore_eos": "false"},
    ],
)
def test_unsupported_inputs_do_not_reach_execution(extra):
    with pytest.raises(ValueError):
        request_tokens(
            {"prompt": "hello", **extra},
            Tokenizer(),
            chat=False,
            context_tokens=128,
            vocab_size=10,
        )


def test_chat_is_plain_text_and_uses_native_non_thinking_template():
    tokens, _ = request_tokens(
        {"messages": [{"role": "user", "content": "hello"}]},
        Tokenizer(),
        chat=True,
        context_tokens=128,
        vocab_size=10,
    )
    assert tokens == [3, 4]
    with pytest.raises(ValueError, match="plain"):
        request_tokens(
            {"messages": [{"role": "user", "content": []}]},
            Tokenizer(),
            chat=True,
            context_tokens=128,
            vocab_size=10,
        )


def test_http_rejects_before_dispatch_and_returns_owned_runtime_receipt():
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from betterscale.live.llm.qwen35.http import create_app

    calls = []

    class Root:
        context_tokens = 128
        capacity = SimpleNamespace(resident_seats=20)
        target_model = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(eos_token_id=9), vocab_size=10)
        )

        def named_graphs(self):
            return [(name, None) for name in ("target1", "target3", "draft1", "draft2")]

    class TextTokenizer(Tokenizer):
        def decode(self, ids, **kwargs):
            return "answer"

    def execute(tokens, count, stops):
        calls.append((tokens, count, stops))
        return {
            "token_ids": [3, 9],
            "cached_tokens": 1,
            "seat": 0,
            "proposed": 2,
            "accepted_drafts": 1,
        }

    client = TestClient(create_app(Root(), TextTokenizer(), "qwen", execute))
    assert client.get("/health").json()["runtime"] == "live"
    assert (
        client.post(
            "/v1/completions", json={"prompt": "hi", "temperature": 1}
        ).status_code
        == 400
    )
    assert calls == []
    result = client.post(
        "/v1/completions", json={"model": "qwen", "prompt": "hi", "max_tokens": 4}
    ).json()
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 1
    assert calls == [([1, 2], 4, (9,))]


def test_async_http_batches_requests_and_reports_failure_as_unhealthy():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import patch

    import httpx
    import torch
    from test_live_qwen_scheduler import BatchRoot

    from betterscale.live.llm.qwen35.http import create_app
    from betterscale.live.llm.qwen35.ingress import Ingress

    class TextTokenizer(Tokenizer):
        def decode(self, ids, **kwargs):
            return str(ids)

    async def run():
        root = BatchRoot()
        root.capacity = root.residents_table.capacity
        root.target_model = SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(eos_token_id=63), vocab_size=64
            )
        )
        root.named_graphs = list
        ingress = Ingress(root, lambda commands: None)
        app = create_app(root, TextTokenizer(), "qwen", ingress.execute)
        async with (
            ingress.lifespan(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/v1/completions", json={"prompt": [i, i + 1], "max_tokens": 4}
                    )
                    for i in (1, 20)
                ]
            )
            assert all(r.status_code == 200 for r in responses)
            assert [r.json()["betterscale"]["token_ids"] for r in responses] == [
                [3, 4, 5, 6],
                [22, 23, 24, 25],
            ]
            assert (await client.get("/health")).json()["scheduler"]["max_batch"] == 2
            root.serving_error = "probe failure"
            assert (await client.get("/health")).status_code == 503

    with patch.object(
        torch, "npu", SimpleNamespace(synchronize=lambda _: None), create=True
    ):
        asyncio.run(run())


def test_detected_disconnect_cancels_execution_without_an_asgi_failure():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    import httpx
    from fastapi import Request

    from betterscale.live.llm.qwen35.http import create_app

    cancelled = []

    async def execute(tokens, count, stops):
        try:
            await asyncio.Future()
        finally:
            cancelled.append(True)

    async def run():
        root = SimpleNamespace(
            context_tokens=128,
            target_model=SimpleNamespace(
                model=SimpleNamespace(
                    config=SimpleNamespace(eos_token_id=9), vocab_size=10
                )
            ),
        )
        app = create_app(root, Tokenizer(), "qwen", execute)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            with patch.object(Request, "is_disconnected", AsyncMock(return_value=True)):
                response = await client.post(
                    "/v1/completions", json={"prompt": [1, 2], "max_tokens": 4}
                )
        assert response.status_code == 499
        assert cancelled == [True]

    asyncio.run(run())
