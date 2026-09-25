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
