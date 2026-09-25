"""Narrow loopback HTTP ingress; one synchronous root transaction at a time.

Only greedy text completion/chat is admitted. Unsupported serving features are
rejected, not silently approximated or delegated to a native engine.
"""

import json
import time
import uuid
from pathlib import Path


def request_tokens(payload, tokenizer, *, chat, context_tokens, vocab_size):
    allowed = {
        "model",
        "max_tokens",
        "temperature",
        "top_p",
        "n",
        "stream",
        "ignore_eos",
    }
    allowed.add("messages" if chat else "prompt")
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError("unsupported live parameters: " + ", ".join(sorted(unknown)))
    if (
        payload.get("temperature", 0) != 0
        or payload.get("top_p", 1) != 1
        or payload.get("n", 1) != 1
        or payload.get("stream", False)
    ):
        raise ValueError(
            "live entry currently supports greedy, n=1, non-streaming only"
        )
    count = payload.get("max_tokens", 16)
    if type(count) is not int or count <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if chat:
        messages = payload.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(m, dict)
                or set(m) != {"role", "content"}
                or m["role"] not in ("user", "assistant", "system")
                or not isinstance(m["content"], str)
                for m in messages
            )
        ):
            raise ValueError("messages must contain plain role/content text pairs")
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
    else:
        prompt = payload.get("prompt")
        tokens = (
            tokenizer.encode(prompt, add_special_tokens=False)
            if isinstance(prompt, str)
            else prompt
        )
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(t) is not int or not 0 <= t < vocab_size for t in tokens)
    ):
        raise ValueError("prompt must be nonempty text or valid token IDs")
    # Two speculative lookahead writes must fit even when output stops earlier.
    if len(tokens) + count + 2 > context_tokens:
        raise ValueError(
            "prompt, output and MTP lookahead exceed the live context envelope"
        )
    if type(payload.get("ignore_eos", False)) is not bool:
        raise ValueError("ignore_eos must be boolean")
    return tokens, count


def create_app(root, tokenizer, model_name, execute, *, eos_token_ids=None):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="BetterScale experimental live runtime")
    eos = (
        root.target_model.model.config.eos_token_id
        if eos_token_ids is None
        else eos_token_ids
    )
    eos = (eos,) if isinstance(eos, int) else tuple(eos or ())

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "runtime": "live",
            "root": type(root).__name__,
            "graphs": [name for name, _ in root.named_graphs()],
            "context_tokens": root.context_tokens,
            "resident_seats": root.capacity.resident_seats,
        }

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "betterscale"}],
        }

    def complete(payload, chat):
        if payload.get("model", model_name) != model_name:
            raise HTTPException(400, "model does not match this live deployment")
        try:
            tokens, count = request_tokens(
                payload,
                tokenizer,
                chat=chat,
                context_tokens=root.context_tokens,
                vocab_size=root.target_model.model.vocab_size,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        stops = () if payload.get("ignore_eos", False) else eos
        result = execute(tokens, count, stops)
        output = result["token_ids"]
        text = tokenizer.decode(output, skip_special_tokens=True)
        choice = {
            "index": 0,
            "finish_reason": "stop" if output[-1] in stops else "length",
        }
        choice.update(
            {"message": {"role": "assistant", "content": text}}
            if chat
            else {"text": text}
        )
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion" if chat else "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(output),
                "total_tokens": len(tokens) + len(output),
                "prompt_tokens_details": {"cached_tokens": result["cached_tokens"]},
            },
            "betterscale": {**result, "runtime": "live"},
        }

    @app.post("/v1/completions")
    async def completions(payload: dict):
        return complete(payload, False)

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict):
        return complete(payload, True)

    return app


def serve(root, model_path, *, port):
    import torch
    import uvicorn
    from transformers import AutoTokenizer
    from vllm.distributed import get_world_group

    group = get_world_group().cpu_group
    rank = torch.distributed.get_rank()
    if rank:
        while True:
            message = [None]
            torch.distributed.broadcast_object_list(message, src=0, group=group)
            if message[0] is None:
                return
            tokens, count, stops = message[0]
            root.generate(tokens, count, eos_token_ids=stops)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        def execute(tokens, count, stops):
            torch.distributed.broadcast_object_list(
                [(tokens, count, stops)], src=0, group=group
            )
            return root.generate(tokens, count, eos_token_ids=stops)

        generation_path = Path(model_path) / "generation_config.json"
        eos = (
            json.loads(generation_path.read_text()).get("eos_token_id")
            if generation_path.exists()
            else None
        )
        app = create_app(
            root, tokenizer, Path(model_path).name, execute, eos_token_ids=eos
        )
        try:
            uvicorn.run(app, host="127.0.0.1", port=port)
        finally:
            torch.distributed.broadcast_object_list([None], src=0, group=group)
